"""Portable checks for exact factory states and dedicated-slot allocation."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'work'))
from stem_effects import CATALOG, build_effect_record, get_effect_spec
from stem_project import _free_effect_slot


class StockEffectRecordTests(unittest.TestCase):
    def test_only_explicit_native_controls_are_supported(self):
        for key, plugin, parameter in (
            ('filter_lowpass', 'Fruity Filter', 'lowpass_blend_percent'),
            ('filter_highpass', 'Fruity Filter', 'highpass_blend_percent'),
            ('delay', 'Fruity Delay 3', 'effect_blend_percent'),
            ('reverb', 'Fruity Reeverb 2', 'effect_blend_percent'),
        ):
            with self.subTest(key=key):
                spec = get_effect_spec(plugin, parameter)
                self.assertEqual(spec, get_effect_spec(key))
                self.assertEqual(spec['initial_mix_raw'], 0)
                self.assertEqual(spec['initial_enabled_raw'], 1)
                self.assertEqual(spec['raw_mix_range'], [0, 12800])
                self.assertEqual(spec['normalized_mix_range'], [0, 1])
                self.assertEqual(spec['control'], 'mixer_slot_mix')
                self.assertEqual(spec['target_offset'], 0x1f01)
                self.assertEqual(spec['baseline_normalized'], 0)
        for plugin, parameter in (('Mixer', 'volume_db'), ('Fruity Filter', 'lowpass_hz'),
                                  ('Fruity Filter', 'highpass_hz'), ('Fruity Delay 3', 'wet_percent'),
                                  ('Fruity Reeverb 2', 'wet_percent'), ('Fruity Wrapper', 'effect_blend_percent')):
            self.assertIsNone(get_effect_spec(plugin, parameter))

    def test_exact_factory_states_are_preserved_and_inputs_are_independent(self):
        for key, size in (('filter_lowpass', 29), ('filter_highpass', 29), ('delay', 108), ('reverb', 66),
                          ('filter_lowpass_cutoff', 29)):
            spec = get_effect_spec(key)
            original = deepcopy(spec)
            record = build_effect_record(spec, 11, 7)
            self.assertEqual(spec, original)
            before = {k: bytes.fromhex(v) for k, v in spec['events']}
            after = dict(record)
            self.assertEqual(len(after[213]), size)
            self.assertEqual(after[213], before[213])
            self.assertEqual(hashlib.sha256(after[213]).hexdigest(), spec['state_sha256'])
            self.assertFalse(spec['provenance']['state_modified'])
            self.assertEqual(spec['provenance']['vendor'], 'Image-Line')
            self.assertEqual(struct.unpack_from('<II', after[212]), (11, 7))
            self.assertEqual(struct.unpack_from('<I', after[212], 16)[0] & 1, 0)
            expected = bytearray(before[212])
            struct.pack_into('<II', expected, 0, 11, 7)
            struct.pack_into('<I', expected, 16, struct.unpack_from('<I', expected, 16)[0] & ~1)
            self.assertEqual(after[212], expected)
            self.assertEqual([(k, v) for k, v in record if k != 212],
                             [(k, bytes.fromhex(v)) for k, v in spec['events'] if k != 212])
            # A caller's edits must not mutate the next lookup.
            spec['events'][0][1] = ''
            self.assertEqual(get_effect_spec(key), original)

    def test_real_lowpass_cutoff_has_open_baseline_and_separate_native_target(self):
        spec = get_effect_spec('Fruity Filter', 'lowpass_cutoff_percent')
        self.assertEqual(spec, get_effect_spec('filter_lowpass_cutoff'))
        self.assertEqual(spec['control'], 'plugin_parameter')
        self.assertEqual(spec['target_offset'], 0x8000)
        self.assertEqual(spec['baseline_normalized'], 1)
        self.assertEqual(spec['initial_parameter_raw'], 1024)
        self.assertEqual(spec['initial_mix_raw'], 12800)
        self.assertEqual(spec['raw_parameter_range'], [0, 1024])
        self.assertEqual(spec['normalized_parameter_range'], [0, 1])
        fields = dict(build_effect_record(spec, 6, 1))
        self.assertEqual(struct.unpack('<7IB', fields[213]), (2, 1024, 0, 1024, 0, 0, 0, 1))
        evidence = spec['provenance']['native_parameter_evidence']
        prefix = 0x70000000 | 6 << 22 | 1 << 16
        self.assertEqual(evidence['native_link_words'], [40 << 16, 0, prefix | 0x8000, 8, 469])
        self.assertIn('Cutoff frequency', evidence['source_channel_name'])
        self.assertEqual(evidence['saved_slot_mix'], 12800)
        self.assertIn('not Hz', spec['description'])

    def test_cutoff_baseline_state_and_parameter_address_cannot_drift(self):
        for field, value in (('target_offset', 0x1f01), ('baseline_normalized', 0),
                             ('initial_parameter_raw', 12800), ('initial_mix_raw', 0)):
            spec = get_effect_spec('filter_lowpass_cutoff')
            spec[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_effect_record(spec, 1, 1)
        # Even a recomputed state hash cannot silently substitute a resonant
        # factory blend preset for the specific observed open low-pass state.
        spec = get_effect_spec('filter_lowpass_cutoff')
        alternate = dict(build_effect_record(get_effect_spec('filter_lowpass'), 1, 1))[213]
        next(row for row in spec['events'] if row[0] == 213)[1] = alternate.hex()
        spec['state_sha256'] = hashlib.sha256(alternate).hexdigest()
        with self.assertRaises(ValueError):
            build_effect_record(spec, 1, 1)

    def test_unproven_native_state_edits_and_wrapper_plugin_are_rejected(self):
        for corrupt in ('state', 'wrapper', 'name', 'neutral', 'duplicate'):
            spec = get_effect_spec('delay')
            if corrupt == 'state':
                next(row for row in spec['events'] if row[0] == 213)[1] = '00' * 108
            elif corrupt == 'wrapper':
                next(row for row in spec['events'] if row[0] == 212)[1] = '00' * 51
            elif corrupt == 'name':
                spec['plugin'] = 'Fruity Wrapper'
            elif corrupt == 'neutral':
                spec['initial_mix_raw'] = 12800
            else:
                spec['events'].append(deepcopy(spec['events'][-1]))
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                build_effect_record(spec, 1, 1)

    def test_master_current_and_invalid_slot_are_never_accepted(self):
        for insert, slot in ((0, 0), (501, 0), (64, 0), (True, 0), (1, -1), (1, 10), (1, True)):
            with self.subTest(insert=insert, slot=slot), self.assertRaises(ValueError):
                build_effect_record(get_effect_spec('reverb'), insert, slot)

    def test_factory_native_links_support_zero_mix_initialization(self):
        catalog = json.loads(CATALOG.read_text(encoding='utf-8'))
        self.assertGreaterEqual(len(catalog['native_mix_examples']), 2)
        for example in catalog['native_mix_examples']:
            self.assertEqual(example['saved_mix_values'], [0])
            target = int(example['target_hex'], 16)
            self.assertEqual(target & 0xffff, 0x1f01)
            self.assertTrue(example['native_link_words'])
            for source, zero, address, flags, smoothing in example['native_link_words']:
                self.assertEqual(source & 0xffff, 0)
                self.assertEqual((zero, address, flags, smoothing), (0, target, 8, 469))


class DedicatedSlotAllocationTests(unittest.TestCase):
    def setUp(self):
        slots = [(98, struct.pack('<H', slot)) for slot in range(10)]
        self.blocks = [(0, 10, list(slots)) for _ in range(4)]

    def allocate(self, occupied=(), reserved=(), owned=(), counts=None):
        return _free_effect_slot(self.blocks, [dict(insert=1, slot=s) for s in occupied],
                                 1, set(reserved), set(owned), Counter(counts or {}))

    def test_prefers_after_chain_and_reserves_unique_new_slots(self):
        reserved = set()
        for expected in (4, 5, 6):
            actual = self.allocate((0, 1, 3), reserved)
            self.assertEqual(actual, expected)
            reserved.add(actual)

    def test_falls_back_to_empty_gap_without_replacing_last_effect(self):
        self.assertEqual(self.allocate((0, 1, 2, 3, 4, 9)), 5)
        with self.assertRaises(ValueError):
            self.allocate(range(10))

    def test_skips_automated_duplicate_and_stale_internal_parameter_slots(self):
        prefix = 0x70400000
        owned = {prefix | 1 << 16 | 0x1f01}
        counts = {prefix | 2 << 16 | 0x1f01: 2,
                  prefix | 3 << 16 | 0x8000: 1}
        self.assertEqual(self.allocate((0,), owned=owned, counts=counts), 4)


if __name__ == '__main__':
    unittest.main()
