"""Reject stale, incompatible or failed native plugin inspection results."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'work')]
import eq_vst3_probe as probe


def response_fixture():
    request = dict(source='source.flp', source_sha256='source-hash',
                   plugin_path=str(probe.INSTALLED_PLUGIN_PATH), plugin_sha256=probe.PINNED_PLUGIN_SHA256,
                   rows=[dict(insert=2, slot=0, wrapper_sha256='wrapper-hash', component_state='')])
    names = {index: f'Band {band} Gain' for band, index in probe.GAIN_IDS.items()}
    parameters = [dict(index=i, id=i, title=names.get(i, f'Parameter {i}'), steps=0) for i in range(519)]
    values = [dict(index=i, id=i, title=parameters[i]['title'], normalized=.5, plain=0., result=0)
              for i in range(91)]
    result = {k: request[k] for k in ('source_sha256', 'plugin_path', 'plugin_sha256')}
    result.update(controller_class=probe.CONTROLLER_CLASS, count=519, parameters=parameters,
                  gain_probes=[dict(band=band, normalized_to_plain=[dict(normalized=n, plain=n * 60 - 30)
                                for n in (0., .25, .45, .5, .55, .75, 1.)]) for band in probe.GAIN_IDS],
                  rows=[dict(insert=2, slot=0, wrapper_sha256='wrapper-hash', state_result=0, parameters=values)])
    return request, result


class ResponseValidationTests(unittest.TestCase):
    def test_plugin_confirmed_schema_is_accepted(self):
        request, result = response_fixture()
        self.assertIs(probe._validate_result(result, request), result)

    def test_other_binary_source_or_controller_is_rejected(self):
        for key in ('source_sha256', 'plugin_sha256', 'plugin_path', 'controller_class'):
            with self.subTest(key=key):
                request, result = response_fixture()
                result[key] = 'different'
                with self.assertRaises(ValueError):
                    probe._validate_result(result, request)

    def test_reordered_or_relabelled_gain_address_is_rejected(self):
        for key, value in [('id', 30), ('title', 'Band 3 Frequency'), ('steps', 1)]:
            with self.subTest(key=key):
                request, result = response_fixture()
                result['parameters'][29][key] = value
                with self.assertRaises(ValueError):
                    probe._validate_result(result, request)

    def test_wrong_gain_scale_and_nonfinite_conversion_are_rejected(self):
        for value in (-1.627907, float('nan'), float('inf')):
            with self.subTest(value=value):
                request, result = response_fixture()
                result['gain_probes'][0]['normalized_to_plain'][2]['plain'] = value
                with self.assertRaises(ValueError):
                    probe._validate_result(result, request)

    def test_missing_conversion_band_is_rejected(self):
        request, result = response_fixture()
        result['gain_probes'].pop()
        with self.assertRaises(ValueError):
            probe._validate_result(result, request)

    def test_stale_or_failed_saved_state_is_rejected(self):
        for key, value in [('wrapper_sha256', 'stale'), ('state_result', 1), ('insert', 3)]:
            with self.subTest(key=key):
                request, result = response_fixture()
                result['rows'][0][key] = value
                with self.assertRaises(ValueError):
                    probe._validate_result(result, request)

    def test_invalid_saved_values_are_rejected(self):
        for key, value in [('normalized', 1.1), ('plain', float('nan')), ('id', 99), ('result', 1)]:
            with self.subTest(key=key):
                request, result = response_fixture()
                result['rows'][0]['parameters'][29][key] = value
                with self.assertRaises(ValueError):
                    probe._validate_result(result, request)

    def test_missing_or_duplicate_saved_band_readouts_are_rejected(self):
        request, result = response_fixture()
        result['rows'][0]['parameters'][29] = deepcopy(result['rows'][0]['parameters'][30])
        with self.assertRaises(ValueError):
            probe._validate_result(result, request)


class ProcessIsolationTests(unittest.TestCase):
    def setUp(self):
        self.request, self.result = response_fixture()
        self.enterContext(patch.object(probe, '_extract_request', return_value=self.request))
        self.enterContext(patch.object(probe.os, 'name', 'nt'))
        self.enterContext(patch.object(probe, '_sha', side_effect=lambda path:
                          'source-hash' if path == 'source.flp' else probe.PINNED_PLUGIN_SHA256))

    def child_response(self, *, payload=None, returncode=0):
        """Model the worker's file protocol without loading a plugin DLL."""
        if payload is None:
            payload = json.dumps({'result': self.result})
        def run(command, **kwargs):
            self.request_path = Path(command[command.index('--request') + 1])
            self.response_path = Path(command[command.index('--response') + 1])
            self.assertEqual(json.loads(self.request_path.read_text(encoding='utf-8')), self.request)
            self.response_path.write_text(payload, encoding='utf-8')
            return SimpleNamespace(returncode=returncode, stdout=b'', stderr=b'')
        return run

    def assert_ipc_cleaned(self):
        self.assertFalse(self.request_path.exists())
        self.assertFalse(self.response_path.exists())

    def test_timeout_is_bounded_and_descriptive(self):
        with patch.object(probe.subprocess, 'run', side_effect=subprocess.TimeoutExpired('probe', 3)) as run:
            with self.assertRaisesRegex(ValueError, 'timed out'):
                probe.probe_project('source.flp', timeout=3)
            self.assertEqual(run.call_args.kwargs['timeout'], 3)
            self.assertNotIn('shell', run.call_args.kwargs)
            self.assertIn('--probe-child', run.call_args.args[0])
            self.assertEqual(run.call_args.kwargs['creationflags'], getattr(subprocess, 'CREATE_NO_WINDOW', 0))

    def test_crash_and_invalid_json_cannot_become_calibration(self):
        cases = [(1, ''), (0, 'not-json'), (1, json.dumps({'error': 'plugin failed'})),
                 (1, json.dumps({'result': self.result}))]
        for returncode, payload in cases:
            with self.subTest(returncode=returncode, payload=payload), \
                 patch.object(probe.subprocess, 'run', side_effect=self.child_response(payload=payload, returncode=returncode)):
                with self.assertRaises(ValueError):
                    probe.probe_project('source.flp')
                self.assert_ipc_cleaned()

    def test_changed_source_after_child_response_is_rejected(self):
        with patch.object(probe.subprocess, 'run', side_effect=self.child_response()), patch.object(probe, '_sha', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'changed during'):
                probe.probe_project('source.flp')
        self.assert_ipc_cleaned()

    def test_valid_child_response_is_returned_without_loading_plugin_in_parent(self):
        with patch.object(probe.subprocess, 'run', side_effect=self.child_response()), patch.object(probe, '_probe_child', side_effect=AssertionError('No parent DLL load')):
            self.assertEqual(probe.probe_project('source.flp'), self.result)
        self.assert_ipc_cleaned()


class WrapperBoundariesTests(unittest.TestCase):
    def test_valid_chunks_preserve_opaque_payload(self):
        payload = b'opaque\x00payload'
        wrapped = struct.pack('<I', 12) + struct.pack('<IQ', 53, len(payload)) + payload
        self.assertEqual(probe._wrapper_chunks(wrapped), {53: payload})

    def test_bad_version_truncation_and_duplicate_chunks_rejected(self):
        chunk = struct.pack('<IQ', 53, 2) + b'ab'
        cases = [b'', struct.pack('<I', 13), struct.pack('<I', 12) + chunk[:-1],
                 struct.pack('<I', 12) + chunk + chunk]
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                probe._wrapper_chunks(data)

    def test_project_cannot_select_an_unverified_plugin_binary(self):
        def chunk(kind, value):
            return struct.pack('<IQ', kind, len(value)) + value
        state = (struct.pack('<I', 12) + chunk(52, bytes.fromhex(probe.COMPONENT_CLASS))
                 + chunk(55, b'C:\\unexpected\\FabFilter Pro-Q 3.vst3\0'))
        record = {'plugin_label': 'FabFilter Pro-Q 3', 'fields': {213: state}}
        with patch('flp_song_analysis.read_project', return_value={'sha256': 'hash'}), \
             patch('plugin_templates.plugin_records', return_value={(1, 0): record}), \
             patch.object(Path, 'is_file', return_value=True), \
             patch.object(probe, '_sha', return_value='unverified-binary-hash') as digest, \
             patch.object(probe.subprocess, 'run') as child:
            with self.assertRaisesRegex(ValueError, 'binary has not been verified'):
                probe._extract_request('source.flp')
            digest.assert_called_once_with(Path(r'C:\unexpected\FabFilter Pro-Q 3.vst3'))
            child.assert_not_called()

    def test_project_component_mismatch_stops_before_plugin_load(self):
        wrong_id = bytes(16)
        state = struct.pack('<I', 12) + struct.pack('<IQ', 52, len(wrong_id)) + wrong_id
        record = {'plugin_label': 'FabFilter Pro-Q 3', 'fields': {213: state}}
        with patch('flp_song_analysis.read_project', return_value={'sha256': 'hash'}), \
             patch('plugin_templates.plugin_records', return_value={(1, 0): record}), \
             patch.object(probe.subprocess, 'run') as child:
            with self.assertRaisesRegex(ValueError, 'identity'):
                probe._extract_request('source.flp')
            child.assert_not_called()


if __name__ == '__main__':
    unittest.main()
