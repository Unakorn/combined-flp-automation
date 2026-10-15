"""Synthetic native FL containers exercise ownership and preservation, not sound."""
from copy import deepcopy
from pathlib import Path
import hashlib
import struct
import sys
import tempfile
import shutil
from uuid import uuid4
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'work'))
from eq_project import (inspect_eq_stems, read_eq_calibration, preview_eq_automation,
                        export_eq_automation, normalized_gain)
from flp_raw import read_bytes, encode_fl
from drum_template import _mixer
from eq_audio import infer_role
from eq_vst3_probe import GAIN_IDS
from flp_song_analysis import read_project
from plugin_templates import plugin_records
from stem_native import build_channel, build_link, build_playlist_clip, parse_points
from test_stem_project import audio_fixture, PPQ, END, text


def fixture(path, bands=(3,), different_second=False, master=False):
    audio_fixture(path)
    header, events = read_bytes(path.read_bytes())
    count = round(END / PPQ * 60 / 170 * 8000)
    for name in ('1_808 Snare', '1_808 Tom'):
        sf.write(str(path.parent / (name + '.wav')), np.zeros(count), 8000, subtype='PCM_16')
    def state(different=False):
        label = b'FabFilter Pro-Q 3\0'
        ffbs = b'FFBS' + struct.pack('<II4f', 1, 4, 1, 2, 3, 5 if different else 4)
        return struct.pack('<I', 12) + struct.pack('<IQ', 54, len(label)) + label + struct.pack('<IQ', 53, len(ffbs)) + ffbs
    at = next(i for i, (k, b) in enumerate(events) if k == 213 and b == b'original third-party plugin state')
    events[at] = 213, state()
    # Add the second Pro-Q before insert2's first empty slot.
    begin = next(i for i, (k, _) in enumerate(events) if k == 103) + 1
    boundaries = [i for i in range(begin, len(events)) if events[i][0] == 147]
    slot_at = next(i for i in range(boundaries[1] + 1, boundaries[2]) if events[i][0] == 98)
    wrapper = bytearray(52)
    struct.pack_into('<II', wrapper, 0, 2, 0)
    events[slot_at:slot_at] = [(201, text('Fruity Wrapper')), (212, bytes(wrapper)), (213, state(different_second))]
    mixer_at = next(i for i, (k, _) in enumerate(events) if k == 225)
    events[mixer_at] = 225, events[mixer_at][1] + struct.pack('<III', 0, 0x70801f00, 1) + struct.pack('<III', 0, 0x70801f01, 12800)
    arrangement = next(i for i, (k, _) in enumerate(events) if k == 99)
    channels, links, init = [], [], []
    for index, band in enumerate(bands):
        cid = index + 2
        target = 0x70408000 + GAIN_IDS[band]
        channels.extend(build_channel(cid, f'Pro-Q 3 - Band {band} Gain', [(0, .51), (96, .43), (192, .61), (384, .51)], PPQ))
        links.append(build_link(cid, target))
        init.append(struct.pack('<III', 0, target, struct.unpack('<I', struct.pack('<f', .51))[0]))
    events[arrangement:arrangement] = channels
    first = next(i for i, (k, _) in enumerate(events) if k == 64)
    events[first:first] = ([(216, b''.join(init))] if init else []) + links
    pi = next(i for i, (k, _) in enumerate(events) if k == 233)
    events[pi] = 233, events[pi][1] + b''.join(build_playlist_clip(index + 2, 0, 384, index + 2, index + 3) for index, _ in enumerate(bands))
    for index, (key, payload) in enumerate(events):
        if key == 238:
            data = bytearray(payload)
            data[12] = 1
            events[index] = key, bytes(data)
    if master:
        at = next(i for i, (k, _) in enumerate(events) if k == 104)
        events[at] = 104, b'\0\0'
    header = bytearray(header)
    struct.pack_into('<H', header, 2, 2 + len(bands))
    path.write_bytes(encode_fl(bytes(header), events))
    return path.read_bytes()


def fake_plan(stems, *, bpm, ppq, end_tick, centers_hz, **kwargs):
    return dict(stems=[dict(id=s['id'], bands=[dict(band_index=i, center_hz=frequency, active=True,
                    points=[dict(seconds=0., db=0.), dict(seconds=1.337, db=-1.5),
                            dict(seconds=end_tick / ppq * 60 / bpm, db=0.)])
                    for i, frequency in enumerate(s.get('centers_hz', centers_hz))]) for s in stems], warnings=[])


def fake_probe(source):
    """Mock only the isolated DLL; container ownership and gain math stay real."""
    project = read_project(source)
    rows = []
    for (insert, slot), record in plugin_records(project).items():
        params = [dict(index=i, id=i, title='Unused', normalized=0., plain=0., display='0', result=0) for i in range(91)]
        for band, center in zip(range(3, 8), (76.34, 244.55, 806.13, 2485.7, 7307.3)):
            index = GAIN_IDS[band]
            params[index - 3].update(normalized=1.)
            params[index - 2].update(normalized=1.)
            params[index - 1].update(display=f'{center} Hz')
            params[index].update(normalized=.51, plain=.6)
            params[index + 4].update(display='1.000')
            params[index + 5].update(display='Bell')
            params[index + 7].update(display='Stereo')
        rows.append(dict(insert=insert, slot=slot, parameters=params,
                         wrapper_sha256=hashlib.sha256(record['fields'][213]).hexdigest()))
    return dict(source_sha256=project['sha256'], plugin_path=str(Path(__file__).resolve()),
                plugin_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=rows)


class EQProjectTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.gettempdir()) / ('eq-project-test-' + uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        self.source = self.root / 'stems.flp'
        self.output = self.root / 'result.flp'
        self.planner = patch('eq_audio.plan_eq_automation', side_effect=fake_plan)
        self.planner.start()
        self.addCleanup(self.planner.stop)
        self.probe = patch('eq_project.probe_project', side_effect=fake_probe)
        self.probe.start()
        self.addCleanup(self.probe.stop)

    def label_first_stem(self, channel_name, mixer_name):
        header, events = read_bytes(self.source.read_bytes())
        events = [(key, text(channel_name) if key == 203 and payload == text('1_808 Snare') else payload)
                  for key, payload in events]
        blocks, _ = _mixer(events)
        events.insert(blocks[1][0], (204, text(mixer_name)))
        self.source.write_bytes(encode_fl(header, events))

    def test_recognizable_mixer_name_supplies_unknown_channel_role_and_display_name(self):
        fixture(self.source, bands=())
        self.label_first_stem('ORIGINAL_2_Insert 7', 'Vocal1')
        original = self.source.read_bytes()
        stem = inspect_eq_stems(self.source)['stems'][0]
        self.assertEqual((stem['name'], stem['role'], stem['role_source']), ('Vocal1', 'vocals', 'mixer'))
        self.assertEqual(stem['channel_name'], 'ORIGINAL_2_Insert 7')
        self.assertEqual(stem['mixer_name'], 'Vocal1')
        self.assertEqual(stem['mixer_insert'], 1)
        preview = preview_eq_automation(self.source)
        self.assertEqual(preview['lanes'][0]['stem_name'], 'Vocal1')
        self.assertEqual(self.source.read_bytes(), original)

    def test_conflicting_channel_and_mixer_roles_stay_explicitly_unknown(self):
        fixture(self.source, bands=())
        self.label_first_stem('ORIGINAL_2_snare', 'Vocal2')
        info = inspect_eq_stems(self.source)
        stem = info['stems'][0]
        self.assertEqual((stem['name'], stem['role_source']), ('ORIGINAL_2_snare', 'conflict'))
        self.assertEqual(stem['role'], 'unknown')
        self.assertIsNone(infer_role(stem['name'], stem['role']))
        self.assertTrue(any('conflicts' in warning and 'Vocal2' in warning for warning in info['warnings']))

    def test_generic_mixer_name_preserves_recognized_channel_identity(self):
        fixture(self.source, bands=())
        self.label_first_stem('ORIGINAL_2_snare', 'Insert 1')
        stem = inspect_eq_stems(self.source)['stems'][0]
        self.assertEqual((stem['name'], stem['role'], stem['role_source']), ('ORIGINAL_2_snare', 'snare', 'channel'))
        self.assertEqual(stem['mixer_name'], 'Insert 1')

    def test_missing_calibration_still_analyzes_and_export_is_blocked(self):
        fixture(self.source, bands=())
        preview = preview_eq_automation(self.source)
        self.assertEqual(len(preview['lanes']), 10)
        self.assertFalse(preview['export_ready'])
        self.assertIn('Missing native address anchor', preview['blocked'][0])
        with self.assertRaisesRegex(ValueError, 'calibration'):
            export_eq_automation(preview, self.output)
        self.assertFalse(self.output.exists())

    def test_one_native_anchor_binds_all_verified_bands_with_saved_gain_baseline(self):
        fixture(self.source)
        preview = preview_eq_automation(self.source)
        exported = [x for x in preview['lanes'] if x['exportable']]
        self.assertEqual([x['band'] for x in exported], [3, 4, 5, 6, 7] * 2)
        self.assertFalse(preview['blocked'])
        self.assertTrue(preview['export_ready'])
        self.assertAlmostEqual(exported[0]['points'][1][1], .485)
        self.assertAlmostEqual(normalized_gain(1.5, exported[0]['calibration']), .535)
        self.assertAlmostEqual(exported[0]['baseline_db'], .6)
        self.assertEqual(exported[0]['frequency_hz'], 76.34)
        self.assertEqual(exported[0]['target'], 0x7040801d)
        self.assertEqual(exported[5]['target'], 0x7080801d)

    def test_export_replaces_setup_and_adds_other_instance_preserving_every_plugin(self):
        original = fixture(self.source)
        preview = preview_eq_automation(self.source)
        report = export_eq_automation(preview, self.output)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(report['new_channels'], 9)
        self.assertEqual(report['replaced_setup_channels'], 1)
        self.assertFalse(report['validation']['native_fl_playback_tested'])
        before, after = read_project(self.source), read_project(self.output)
        self.assertEqual([p for k, p in before['events'] if k in (213, 225, 235, 236)],
                         [p for k, p in after['events'] if k in (213, 225, 235, 236)])
        self.assertEqual({k: r['events'] for k, r in plugin_records(before).items()},
                         {k: r['events'] for k, r in plugin_records(after).items()})
        lanes = [x for x in preview['lanes'] if x['exportable']]
        for channel, lane in zip([c for c in after['channels'] if c['type'] == 5], lanes):
            parsed = parse_points(dict(after['events'][channel['start']:channel['stop']])[234], PPQ)
            self.assertAlmostEqual(parsed[1][0], round(1.337 * 170 * PPQ / 60))
            self.assertAlmostEqual(parsed[1][1], .485)
            clips = [c for c in after['clips'] if c.get('channel_id') == channel['id']]
            self.assertEqual([(c['position'], c['duration']) for c in clips], [(0, END)])
        self.assertEqual(len([c for c in after['channels'] if c['type'] == 5]), 10)

    def test_different_opaque_states_are_supported_when_each_instance_is_verified(self):
        fixture(self.source, different_second=True)
        preview = preview_eq_automation(self.source)
        self.assertEqual(sum(x['exportable'] for x in preview['lanes']), 10)

    def test_unavailable_controller_never_uses_guessed_four_point_db_mapping(self):
        fixture(self.source)
        with patch('eq_project.probe_project', side_effect=ValueError('Controller unavailable')):
            preview = preview_eq_automation(self.source)
        self.assertEqual(len(preview['lanes']), 10)
        self.assertFalse(preview['export_ready'])
        self.assertEqual(preview['provider']['status'], 'blocked')
        self.assertFalse(any(lane['points'] for lane in preview['lanes']))
        with self.assertRaisesRegex(ValueError, 'not been verified'):
            normalized_gain(-1., dict(baseline=.51, anchors=[(-3., .43), (0., .51), (3., .61)]))

    def test_saved_gain_and_song_initializer_conflict_blocks_export(self):
        fixture(self.source)
        def differing_gain(source):
            result = fake_probe(source)
            for row in result['rows']:
                row['parameters'][GAIN_IDS[3]].update(normalized=.5, plain=0.)
            return result
        with patch('eq_project.probe_project', side_effect=differing_gain):
            preview = preview_eq_automation(self.source)
        self.assertFalse(preview['export_ready'])
        self.assertTrue(any('initializer' in warning for warning in preview['blocked']))

    def test_unsupported_saved_band_is_skipped_and_each_instance_uses_its_actual_center(self):
        fixture(self.source)
        def different_settings(source):
            result = fake_probe(source)
            result['rows'][1]['parameters'][GAIN_IDS[3] - 1]['display'] = '93.27 Hz'
            result['rows'][1]['parameters'][GAIN_IDS[4] + 4]['display'] = '2.000'
            return result
        with patch('eq_project.probe_project', side_effect=different_settings):
            preview = preview_eq_automation(self.source)
        self.assertTrue(preview['export_ready'])
        self.assertEqual(preview['lanes'][5]['frequency_hz'], 93.27)
        self.assertFalse(preview['lanes'][6]['exportable'])
        self.assertTrue(any('Q differs' in warning for warning in preview['skipped']))

    def test_native_anchor_parameter_index_must_match_the_installed_gain_control(self):
        fixture(self.source)
        header, events = read_bytes(self.source.read_bytes())
        events = [(k, struct.pack('<5I', 2 << 16, 0, 0x7040801c, 8, 469) if k == 227 else p)
                  for k, p in events]
        self.source.write_bytes(encode_fl(header, events))
        with self.assertRaisesRegex(ValueError, 'address disagrees'):
            read_eq_calibration(self.source)

    def test_master_stem_remains_context_but_never_receives_eq(self):
        fixture(self.source, bands=(), master=True)
        info = inspect_eq_stems(self.source)
        self.assertEqual(info['stems'][0]['mixer_insert'], 0)
        self.assertTrue(any('context only' in x for x in info['warnings']))
        self.assertEqual(len(preview_eq_automation(self.source)['stems']), 2)

    def test_calibration_frequency_label_not_reinterpreted_as_gain(self):
        fixture(self.source)
        h, events = read_bytes(self.source.read_bytes())
        events = [(k, text('Pro-Q 3 - Band 3 Frequency') if k == 203 and p == text('Pro-Q 3 - Band 3 Gain') else p) for k, p in events]
        self.source.write_bytes(encode_fl(h, events))
        with self.assertRaisesRegex(ValueError, 'Missing native address anchor'):
            read_eq_calibration(self.source)

    def test_calibration_shared_fanout_is_rejected(self):
        fixture(self.source)
        h, events = read_bytes(self.source.read_bytes())
        first = next(i for i, (k, _) in enumerate(events) if k == 64)
        events.insert(first, build_link(2, 0x7080801d))
        self.source.write_bytes(encode_fl(h, events))
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            read_eq_calibration(self.source)

    def test_existing_parameter_owner_is_preserved(self):
        fixture(self.source)
        h, events = read_bytes(self.source.read_bytes())
        first = next(i for i, (k, _) in enumerate(events) if k == 64)
        events.insert(first, (226, struct.pack('<5I', 253, 0, 0x7080801d, 4, 469)))
        self.source.write_bytes(encode_fl(h, events))
        preview = preview_eq_automation(self.source)
        self.assertEqual(sum(x['exportable'] for x in preview['lanes']), 9)
        self.assertTrue(any('existing automation owns' in x for x in preview['skipped']))

    def test_changed_source_and_existing_output_are_not_overwritten(self):
        fixture(self.source)
        preview = preview_eq_automation(self.source)
        self.output.write_bytes(b'preserve')
        with self.assertRaises(FileExistsError):
            export_eq_automation(preview, self.output)
        self.assertEqual(self.output.read_bytes(), b'preserve')
        self.source.write_bytes(self.source.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            export_eq_automation(preview, self.root / 'new.flp')

    def test_changed_normalized_points_or_boosts_are_rejected(self):
        fixture(self.source)
        preview = preview_eq_automation(self.source)
        copy = deepcopy(preview)
        copy['lanes'][0]['points'][1] = (copy['lanes'][0]['points'][1][0], .99)
        with self.assertRaisesRegex(ValueError, 'normalized points'):
            export_eq_automation(copy, self.output)
        preview['lanes'][0]['points_db'][1] = (1.337, 1.)
        with self.assertRaisesRegex(ValueError, 'cut-only'):
            export_eq_automation(preview, self.output)

    def test_duration_mismatch_is_rejected_before_analysis(self):
        fixture(self.source)
        sf.write(str(self.root / '1_808 Snare.wav'), np.zeros(8000), 8000)
        with self.assertRaisesRegex(ValueError, 'duration differs'):
            inspect_eq_stems(self.source)

    def test_inactive_other_instance_does_not_receive_a_flat_lane(self):
        fixture(self.source)
        def flat(*args, **kwargs):
            result = fake_plan(*args, **kwargs)
            for stem in result['stems']:
                for band in stem['bands']:
                    band['active'] = False
                    for point in band['points']:
                        point['db'] = 0.
            return result
        with patch('eq_audio.plan_eq_automation', side_effect=flat):
            preview = preview_eq_automation(self.source)
        self.assertEqual(sum(x['exportable'] for x in preview['lanes']), 1)
        report = export_eq_automation(preview, self.output)
        self.assertEqual(report['new_channels'], 0)
        self.assertEqual(report['replaced_setup_channels'], 1)

    def test_failed_final_link_leaves_no_partial_output(self):
        fixture(self.source)
        preview = preview_eq_automation(self.source)
        with patch('eq_project.os.link', side_effect=OSError('disk unavailable')):
            with self.assertRaisesRegex(OSError, 'disk unavailable'):
                export_eq_automation(preview, self.output)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('.eq-export-*')))

    def test_relative_audio_paths_cannot_be_silently_relocated(self):
        fixture(self.source)
        header, events = read_bytes(self.source.read_bytes())
        events = [(key, text(Path(payload.decode('utf-16-le').rstrip('\0')).name) if key == 196 else payload)
                  for key, payload in events]
        self.source.write_bytes(encode_fl(header, events))
        preview = preview_eq_automation(self.source)
        with self.assertRaisesRegex(ValueError, 'relative audio paths'):
            export_eq_automation(preview, self.root / 'elsewhere' / 'result.flp')

    def test_automated_mixer_level_is_not_analyzed_as_static(self):
        fixture(self.source, bands=())
        header, events = read_bytes(self.source.read_bytes())
        first = next(i for i, (key, _) in enumerate(events) if key == 64)
        events.insert(first, (226, struct.pack('<5I', 253, 0, 0x70401fc0, 4, 469)))
        self.source.write_bytes(encode_fl(header, events))
        with self.assertRaisesRegex(ValueError, 'volume automation'):
            inspect_eq_stems(self.source)

    def test_full_five_band_setup_roundtrips_ten_independent_targets(self):
        original = fixture(self.source, bands=(3, 4, 5, 6, 7))
        preview = preview_eq_automation(self.source)
        report = export_eq_automation(preview, self.output)
        self.assertEqual(report['automation_channels'], 10)
        self.assertEqual(report['new_channels'], 5)
        self.assertEqual(report['replaced_setup_channels'], 5)
        self.assertEqual(self.source.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
