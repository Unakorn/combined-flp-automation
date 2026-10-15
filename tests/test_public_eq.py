"""Portable EQ invariants; all fixtures are synthetic, with no song data."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'work')]
import combined_eq_setup as setup
import eq_vst3_probe as probe


def neutral_row():
    params = [dict(index=i, id=i, title=f'Parameter {i}', normalized=0., plain=0.,
                   display='0', result=0) for i in range(setup.FULL_PARAMETER_COUNT)]
    for band in range(1, 25):
        first = (band - 1) * 13
        params[first]['title'] = f'Band {band} Used'
        if band in setup.BANDS:
            params[first]['normalized'] = 1.
            params[first + 1]['normalized'] = 1.
            params[first + 2].update(plain=100., display='100 Hz')
            params[first + 3].update(normalized=.5, plain=0.)
            params[first + 7].update(plain=1., display='1.000')
            params[first + 8]['display'] = 'Bell'
            params[first + 10]['display'] = 'Stereo'
    required = {
        312: ('Processing Mode', 0.), 314: ('Gain Scale', .5),
        315: ('Output Level', .5), 316: ('Output Pan', .5),
        317: ('Output Pan Mode', 0.), 318: ('Bypass', 0.),
        319: ('Output Invert Phase', 0.), 320: ('Auto Gain', 0.),
        332: ('Receive Midi', 0.), 333: ('Solo Gain', .5), 382: ('Host Bypass', 0.)}
    for index, (title, normalized) in required.items():
        params[index].update(title=title, normalized=normalized)
    return {'parameters': params}


def synthetic_wrapper(gain=2.):
    values = [0.] * 358
    values[0], values[13] = 1., 1.
    for index in (29, 42, 55, 68, 81):
        values[index] = gain
    component = b'FFBS' + struct.pack('<II358f', 1, 358, *values) + b'FFpr\1\0\0\0\0\0\0\0'
    def chunk(kind, value):
        return struct.pack('<IQ', kind, len(value)) + value
    return struct.pack('<I', 12) + chunk(52, bytes.fromhex(probe.COMPONENT_CLASS)) + chunk(53, chunk(3, component))


class NeutralityTests(unittest.TestCase):
    def test_all_five_flat_bands_accepted(self):
        self.assertEqual([b['baseline_db'] for b in setup.validate_neutral_row(neutral_row())], [0.] * 5)

    def test_processing_changes_rejected(self):
        for index, value in ((0, 1.), (91, 1.), (29, .51), (315, .6), (316, .6),
                             (319, 1.), (320, 1.), (38, 1.), (334, 1.), (382, 1.)):
            with self.subTest(index=index):
                row = neutral_row()
                row['parameters'][index]['normalized'] = value
                with self.assertRaises(ValueError):
                    setup.validate_neutral_row(row)

    def test_dynamic_bands_rejected(self):
        row = neutral_row()
        row['parameters'][30]['plain'] = 2.
        with self.assertRaisesRegex(ValueError, 'dynamic EQ'):
            setup.validate_neutral_row(row)

    def test_incomplete_global_readout_rejected(self):
        row = neutral_row()
        row['parameters'] = row['parameters'][:91]
        with self.assertRaisesRegex(ValueError, 'every band and global'):
            setup.validate_neutral_row(row)

    def test_unrelated_controller_change_rejected(self):
        before, after = neutral_row(), neutral_row()
        before['parameters'][324]['normalized'] = .6
        with self.assertRaisesRegex(ValueError, 'unrelated parameter'):
            setup._verify_intended_changes(before, after)

    def test_different_user_states_use_only_seven_allowed_fields(self):
        for gain in (1., 2., -3.):
            old = synthetic_wrapper(gain)
            new, component = setup._neutral_candidate(old)
            at, end = setup._component_bounds(old)
            allowed = {at + 12 + 4 * index + byte for index in setup.NEUTRAL_FIELDS for byte in range(4)}
            self.assertLessEqual({i for i, (a, b) in enumerate(zip(old, new)) if a != b}, allowed)
            self.assertEqual(component, new[at:end])
            self.assertEqual(len(old), len(new))
            for index in setup.NEUTRAL_FIELDS:
                self.assertEqual(struct.unpack_from('<f', new, at + 12 + 4 * index)[0], 0.)

    def test_malformed_component_rejected(self):
        wrapper = bytearray(synthetic_wrapper())
        wrapper[-1] ^= 1
        with self.assertRaisesRegex(ValueError, 'trailer'):
            setup._neutral_candidate(bytes(wrapper))


class CalibrationSelectionTests(unittest.TestCase):
    def test_native_anchor_selects_any_insert_and_slot(self):
        record = {'plugin_label': 'FabFilter Pro-Q 3'}
        anchor = {'bands': {7: {'insert': 12, 'slot': 4}}}
        self.assertEqual(setup._select_donor(anchor, {(12, 4): record}), ((12, 4), record))

    def test_multiple_instances_require_unambiguous_selection(self):
        anchor = {'bands': {3: {'insert': 1, 'slot': 0}, 7: {'insert': 2, 'slot': 0}}}
        with self.assertRaisesRegex(ValueError, 'one Pro-Q instance'):
            setup._select_donor(anchor, {})

    def test_explicit_instance_requires_native_anchor(self):
        anchor = {'bands': {7: {'insert': 1, 'slot': 0}}}
        with self.assertRaisesRegex(ValueError, 'no native gain'):
            setup._select_donor(anchor, {}, 2, 0)


class WorkerTests(unittest.TestCase):
    def fake_worker(self, command, **kwargs):
        self.command = command
        self.request = Path(command[command.index('--request') + 1])
        self.response = Path(command[command.index('--response') + 1])
        self.assertEqual(json.loads(self.request.read_text()), {'input': 42})
        self.response.write_text(json.dumps({'result': {'accepted': True}}))
        return type('Completed', (), {'returncode': 0})()

    def test_frozen_worker_uses_app_dispatch_and_cleans_ipc(self):
        with patch.object(sys, 'frozen', True, create=True), patch.object(probe.subprocess, 'run', self.fake_worker):
            result = probe._isolated_probe({'input': 42}, '--probe-child', 'unused.py')
        self.assertEqual(result, {'accepted': True})
        self.assertEqual(self.command[:2], [sys.executable, '--probe-child'])
        self.assertNotIn('unused.py', self.command)
        self.assertFalse(self.request.exists())
        self.assertFalse(self.response.exists())

    def test_source_worker_preserves_module_and_extra_library_arg(self):
        with patch.object(sys, 'frozen', False, create=True), patch.object(probe.subprocess, 'run', self.fake_worker):
            probe._isolated_probe({'input': 42}, '--probe-full-child', 'setup.py', extra_args=('library',))
        self.assertEqual(self.command[:5], [sys.executable, '-B', 'setup.py', '--probe-full-child', 'library'])

    def test_worker_error_is_reported_and_files_cleaned(self):
        def fail(command, **kwargs):
            completed = self.fake_worker(command, **kwargs)
            self.response.write_text(json.dumps({'error': 'unsupported plugin'}))
            completed.returncode = 1
            return completed
        with patch.object(probe.subprocess, 'run', fail):
            with self.assertRaisesRegex(ValueError, 'unsupported plugin'):
                probe._isolated_probe({'input': 42}, '--probe-child', 'worker.py')
        self.assertFalse(self.request.exists())
        self.assertFalse(self.response.exists())

    def test_binary_validation_is_independent_of_install_directory(self):
        with patch.object(Path, 'is_file', return_value=True), patch.object(probe, '_sha', return_value=probe.PINNED_PLUGIN_SHA256):
            path = Path('D:/Audio/Plugins/FabFilter Pro-Q 3.vst3')
            self.assertEqual(probe._validate_plugin_binary(path), path.resolve())

    def test_different_binary_is_rejected(self):
        with patch.object(Path, 'is_file', return_value=True), patch.object(probe, '_sha', return_value='unverified'):
            with self.assertRaisesRegex(ValueError, '3.29'):
                probe._validate_plugin_binary('D:/Audio/Plugins/FabFilter Pro-Q 3.vst3')


if __name__ == '__main__':
    unittest.main()
