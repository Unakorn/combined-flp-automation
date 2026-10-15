"""Synthetic native-container fixtures; these are NOT real VST/plugin presets."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import unittest
import tempfile
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'work'))
from flp_raw import encode_fl, read_bytes
from flp_song_analysis import read_project
from plugin_templates import (FAMILIES, build_plugin_record, import_template,
                              load_template, normalized_depth, plugin_records)
from stem_native import build_channel, build_link, build_playlist_clip
from test_stem_project import audio_fixture, PPQ, END, text


def fake_spec(family='width'):
    """Opaque test bytes exercise serialization, never claim a playable plugin."""
    label = dict(filter='F6', delay='H-Delay', reverb='RVerb', width='Auto Shape Stereo')[family]
    encoded = label.encode()+b'\0'
    state = struct.pack('<I', 12)+struct.pack('<IQ', 54, len(encoded))+encoded
    opaque = b'synthetic fixture only; not a loadable VST state'
    state += struct.pack('<IQ', 53, len(opaque))+opaque
    wrapper = bytearray(52)
    struct.pack_into('<II', wrapper, 0, 1, 1)
    struct.pack_into('<I', wrapper, 16, 0x141)
    base = dict(filter=.9, delay=.0, reverb=.0, width=1/3)[family]
    low = .1 if family == 'filter' else 0.0
    high = .9 if family == 'filter' else .6
    rows = [(201, text('Fruity Wrapper')), (212, bytes(wrapper)), (213, state)]
    return dict(family=family, label=label, plugin='Fruity Wrapper', plugin_label=label,
                events=[(key, value.hex()) for key, value in rows],
                state_sha256=hashlib.sha256(state).hexdigest(), target_offset=0x8005,
                minimum=low, baseline=base, maximum=high,
                initial_parameter_raw=struct.unpack('<I', struct.pack('<f', base))[0],
                initial_mix_raw=12800, initial_enabled_raw=1, native_playback_verified=False)


def calibration_fixture(path, family='width', mutate=None):
    audio_fixture(path)
    header, events = read_bytes(path.read_bytes())
    spec = fake_spec(family)
    target = 0x70000000 | 1 << 22 | 1 << 16 | spec['target_offset']
    arrangement = next(i for i, (key, _) in enumerate(events) if key == 99)
    rows = build_channel(2, FAMILIES[family]['clip'],
                         [(0, spec['baseline']), (96, spec['minimum']),
                          (192, spec['maximum']), (384, spec['baseline'])], PPQ)
    events[arrangement:arrangement] = rows
    first = next(i for i, (key, _) in enumerate(events) if key == 64)
    events.insert(first, build_link(2, target))
    # Locate insert1 via the existing opaque test effect; its next slot is1.
    original = next(i for i, (key, value) in enumerate(events)
                    if key == 213 and value == b'original third-party plugin state')
    slot1 = next(i for i in range(original, len(events))
                 if events[i] == (98, struct.pack('<H', 1)))
    events[slot1:slot1] = [(key, bytes.fromhex(value)) for key, value in spec['events']]
    pi = next(i for i, (key, _) in enumerate(events) if key == 233)
    events[pi] = (233, events[pi][1]+build_playlist_clip(2, 0, 384, 2, 3))
    ii = next(i for i, (key, _) in enumerate(events) if key == 225)
    prefix = target & 0xffff0000
    values = [(0, target, spec['initial_parameter_raw']), (0, prefix | 0x1f00, 1),
              (0, prefix | 0x1f01, 12800)]
    events[ii] = (225, events[ii][1]+b''.join(struct.pack('<III', *row) for row in values))
    if mutate:
        mutate(events)
    header = bytearray(header)
    struct.pack_into('<H', header, 2, 3)
    data = encode_fl(bytes(header), events)
    path.write_bytes(data)
    return spec, target


def move_knob_initializer_to_native_event(events, retain_legacy=False):
    """Use the event216 placement observed in FL26, without real plugin data."""
    at = next(i for i, (key, _) in enumerate(events) if key == 225)
    rows = list(struct.iter_unpack('<III', events[at][1]))
    knobs = [row for row in rows if row[1] & 0xffff >= 0x8000]
    assert len(knobs) == 1
    if not retain_legacy:
        events[at] = 225, b''.join(struct.pack('<III', *row) for row in rows if row not in knobs)
    global_at = next(i for i, (key, _) in enumerate(events) if key in (227, 64))
    events.insert(global_at, (216, b''.join(struct.pack('<III', *row) for row in knobs)))


class PluginTemplateTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.gettempdir()) / ('plugin-template-test-'+uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        self.source = self.root / 'synthetic-calibration.flp'
        self.output = self.root / 'local-template.json'

    def test_import_reads_actual_native_link_and_opaque_record(self):
        expected, target = calibration_fixture(self.source)
        original = self.source.read_bytes()
        status = import_template(self.source, self.output)
        self.assertTrue(status['families']['width']['ready'])
        actual = load_template(self.output)['families']['width']
        self.assertEqual(actual['target_offset'], target & 0xffff)
        self.assertEqual(actual['events'], [list(row) for row in expected['events']])
        self.assertEqual(actual['initial_parameter_raw'], expected['initial_parameter_raw'])
        self.assertFalse(actual['native_playback_verified'])
        self.assertEqual(self.source.read_bytes(), original)

    def test_native_song_initializer_and_stock_disabled_lfo_header(self):
        def native(events):
            move_knob_initializer_to_native_event(events)
            at = next(i for i, (key, _) in enumerate(events) if key == 234)
            payload = bytearray(events[at][1])
            # The FL26 user sample saves amount64 and byte8zero with LFO off.
            # Width baseline1/3 also exercises its observed float32 initializer.
            struct.pack_into('<i', payload, 4, 64)
            events[at] = 234, bytes(payload)
        expected, _ = calibration_fixture(self.source, mutate=native)
        original = self.source.read_bytes()
        import_template(self.source, self.output)
        actual = load_template(self.output)['families']['width']
        self.assertEqual(actual['initial_parameter_raw'], expected['initial_parameter_raw'])
        self.assertEqual(actual['initial_parameter_raw'], 1051372203)
        self.assertEqual(self.source.read_bytes(), original)

    def test_matching_initializer_in_native_and_legacy_events_is_supported(self):
        calibration_fixture(self.source, mutate=lambda events: move_knob_initializer_to_native_event(events, True))
        self.assertTrue(import_template(self.source, self.output)['families']['width']['ready'])

    def test_conflicting_or_duplicate_initializers_do_not_replace_template(self):
        existing = json.dumps(dict(version=1, families={})).encode()
        for case in ('conflict', 'native_duplicate', 'legacy_duplicate'):
            def corrupt(events):
                if case != 'legacy_duplicate':
                    move_knob_initializer_to_native_event(events, retain_legacy=case == 'conflict')
                key = 225 if case == 'legacy_duplicate' else 216
                at = next(i for i, (kind, _) in enumerate(events) if kind == key)
                rows = list(struct.iter_unpack('<III', events[at][1]))
                knob = next(row for row in rows if row[1] & 0xffff >= 0x8000)
                if case == 'conflict':
                    rows = [(knob[0], knob[1], knob[2]+1)]
                else:
                    rows.append(knob)
                events[at] = key, b''.join(struct.pack('<III', *row) for row in rows)
            calibration_fixture(self.source, mutate=corrupt)
            self.output.write_bytes(existing)
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, 'conflicting or duplicate'):
                import_template(self.source, self.output)
            self.assertEqual(self.output.read_bytes(), existing)

    def test_native_and_legacy_initializers_must_match_clip_baseline(self):
        for event_id in (216, 225):
            def corrupt(events):
                if event_id == 216:
                    move_knob_initializer_to_native_event(events)
                at = next(i for i, (key, _) in enumerate(events) if key == event_id)
                rows = list(struct.iter_unpack('<III', events[at][1]))
                wrong = struct.unpack('<I', struct.pack('<f', .5))[0]
                rows = [(tag, target, wrong if target & 0xffff >= 0x8000 else value)
                        for tag, target, value in rows]
                events[at] = event_id, b''.join(struct.pack('<III', *row) for row in rows)
            calibration_fixture(self.source, mutate=corrupt)
            with self.subTest(event_id=event_id), self.assertRaisesRegex(ValueError, 'calibration baseline'):
                import_template(self.source, self.output)
            self.assertFalse(self.output.exists())

    def test_baseline_comparison_allows_only_float32_rounding_tolerance(self):
        for raw_delta in (-1, 0, 1, 3):
            spec = fake_spec()
            spec['initial_parameter_raw'] += raw_delta
            self.output.write_text(json.dumps(dict(version=1, families={'width': spec})))
            with self.subTest(raw_delta=raw_delta):
                if raw_delta == 3:
                    with self.assertRaisesRegex(ValueError, 'calibration baseline'):
                        load_template(self.output)
                else:
                    self.assertIn('width', load_template(self.output)['families'])

    def test_dry_baseline_and_normalized_initializer_are_checked(self):
        for value in (1e-8, -.1, 1.1, float('nan'), float('inf')):
            spec = fake_spec('delay')
            spec['initial_parameter_raw'] = struct.unpack('<I', struct.pack('<f', value))[0]
            self.output.write_text(json.dumps(dict(version=1, families={'delay': spec})))
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_template(self.output)

    def test_malformed_native_initializer_is_rejected(self):
        def corrupt(events):
            move_knob_initializer_to_native_event(events)
            at = next(i for i, (key, _) in enumerate(events) if key == 216)
            events[at] = 216, events[at][1]+b'\0'
        calibration_fixture(self.source, mutate=corrupt)
        with self.assertRaisesRegex(ValueError, 'initialization layout'):
            import_template(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_plugin_clone_only_changes_wrapper_destination_and_editor_flag(self):
        spec = fake_spec()
        original = deepcopy(spec)
        before = {key: bytes.fromhex(value) for key, value in spec['events']}
        after = dict(build_plugin_record(spec, 23, 7))
        wrapper = bytearray(before[212])
        struct.pack_into('<II', wrapper, 0, 23, 7)
        struct.pack_into('<I', wrapper, 16, 0x140)
        self.assertEqual(after[212], wrapper)
        self.assertEqual(after[213], before[213])
        self.assertEqual(spec, original)

    def test_depth_mapping_respects_nonlinear_plugin_calibration_bounds(self):
        spec = fake_spec('filter')
        self.assertAlmostEqual(normalized_depth(spec, 0), .9)
        self.assertAlmostEqual(normalized_depth(spec, -.5), .5)
        self.assertAlmostEqual(normalized_depth(spec, -1), .1)
        width = fake_spec('width')
        self.assertAlmostEqual(normalized_depth(width, 0), 1/3)
        self.assertAlmostEqual(normalized_depth(width, 1), .6)
        self.assertAlmostEqual(normalized_depth(width, -1), 0)
        for bad in (float('nan'), float('inf'), True, 1.01):
            with self.assertRaises(ValueError):
                normalized_depth(spec, bad)

    def test_mixer_blend_or_transformed_controller_is_not_internal_parameter(self):
        def corrupt(events):
            index = next(i for i, (key, _) in enumerate(events) if key == 227)
            payload = bytearray(events[index][1])
            struct.pack_into('<I', payload, 8, 0x70411f01)
            events[index] = 227, bytes(payload)
            ii = next(i for i, (key, _) in enumerate(events) if key == 225)
            # Preserve one initializer at the altered address.
            rows = list(struct.iter_unpack('<III', events[ii][1]))
            rows = [row for row in rows if row[1] != 0x70418005]
            events[ii] = 225, b''.join(struct.pack('<III', *row) for row in rows)
        calibration_fixture(self.source, mutate=corrupt)
        with self.assertRaisesRegex(ValueError, 'internal plugin control'):
            import_template(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_wrong_plugin_identity_rejected_despite_friendly_clip_name(self):
        def corrupt(events):
            for i, (key, payload) in enumerate(events):
                if key == 213 and b'Auto Shape Stereo' in payload:
                    events[i] = key, payload.replace(b'Auto Shape Stereo', b'Wrong VST Product')
        calibration_fixture(self.source, mutate=corrupt)
        with self.assertRaisesRegex(ValueError, 'identity'):
            import_template(self.source, self.output)

    def test_duplicate_native_links_are_rejected(self):
        def corrupt(events):
            events.insert(5, next(row for row in events if row[0] == 227))
        calibration_fixture(self.source, mutate=corrupt)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            import_template(self.source, self.output)

    def test_calibration_minmax_and_lfo_must_not_transform_endpoint_values(self):
        for field in ('minimum', 'maximum', 'lfo_amount', 'lfo_enabled'):
            def corrupt(events):
                index = next(i for i, (key, value) in enumerate(events)
                             if key == 64 and int.from_bytes(value, 'little') == 2)
                event_id = 219 if field in ('minimum', 'maximum') else 234
                at = next(i for i in range(index, len(events)) if events[i][0] == event_id)
                payload = bytearray(events[at][1])
                if field in ('minimum', 'maximum'):
                    struct.pack_into('<I', payload, 0 if field == 'minimum' else 4, 6400)
                elif field == 'lfo_amount':
                    struct.pack_into('<i', payload, 4, 1)
                else:
                    payload[8] = 1
                events[at] = event_id, bytes(payload)
            calibration_fixture(self.source, mutate=corrupt)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Min/Max|LFO'):
                import_template(self.source, self.output)
            self.assertFalse(self.output.exists())

    def test_unknown_stock_lfo_headers_are_rejected(self):
        for amount, byte8 in ((64, 1), (64, 255), (-1, 0), (65, 0), (128, 0)):
            def corrupt(events):
                at = next(i for i, (key, _) in enumerate(events) if key == 234)
                payload = bytearray(events[at][1])
                struct.pack_into('<i', payload, 4, amount)
                payload[8] = byte8
                events[at] = 234, bytes(payload)
            calibration_fixture(self.source, mutate=corrupt)
            with self.subTest(amount=amount, byte8=byte8), self.assertRaisesRegex(ValueError, 'LFO'):
                import_template(self.source, self.output)
            self.assertFalse(self.output.exists())

    def test_nonidentity_native_link_is_not_reinterpreted(self):
        for offset, value in ((0, (2 << 16) | 0x8001), (4, 1), (12, 4), (16, 470)):
            def corrupt(events):
                index = next(i for i, (key, _) in enumerate(events) if key == 227)
                payload = bytearray(events[index][1])
                struct.pack_into('<I', payload, offset, value)
                events[index] = 227, bytes(payload)
            calibration_fixture(self.source, mutate=corrupt)
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, 'exactly one'):
                import_template(self.source, self.output)

    def test_changed_source_during_import_cannot_replace_local_template(self):
        calibration_fixture(self.source)
        before = self.source.read_bytes()
        existing = json.dumps(dict(version=1, families={})).encode()
        original_read = Path.read_bytes
        source_identity = self.source.resolve()
        # Windows temp directories may use an 8.3 alias while read_project
        # resolves the full path. Exercise a lexical alias on every machine.
        alias = self.root / '..' / self.root.name / self.source.name
        for input_path in (self.source, alias):
            with self.subTest(input_path=input_path):
                self.output.write_bytes(existing)
                calls = 0
                def changed(path):
                    nonlocal calls
                    value = original_read(path)
                    if path.resolve() == source_identity:
                        calls += 1
                        if calls >= 2:
                            return value+b' changed while reading'
                    return value
                with patch.object(Path, 'read_bytes', changed), self.assertRaisesRegex(ValueError, 'changed while importing'):
                    import_template(input_path, self.output)
                self.assertGreaterEqual(calls, 2, 'The test must intercept the preservation re-read.')
                self.assertEqual(self.source.read_bytes(), before)
                self.assertEqual(self.output.read_bytes(), existing)

    def test_integrity_and_real_plugin_offset_required_on_reload(self):
        for field, bad in [('target_offset', 0x1f01), ('state_sha256', '0'*64)]:
            spec = fake_spec()
            spec[field] = bad
            self.output.write_text(json.dumps(dict(version=1, families={'width': spec})))
            with self.assertRaises(ValueError):
                load_template(self.output)


if __name__ == '__main__':
    unittest.main()
