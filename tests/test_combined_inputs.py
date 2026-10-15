from copy import deepcopy
from pathlib import Path
import sys
import unittest

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
sys.path.insert(0, str(APP / 'work'))

from combined_inputs import build_input_config, compare_exported_notes
from stem_audio import detect_role, role_family
from stem_balance import plan_balance


class MappingTests(unittest.TestCase):
    def test_shortened_terminal_note_disclosed_without_mutation(self):
        source = [dict(tick=0, note=60, duration=96), dict(tick=192, note=62, duration=96)]
        exported = deepcopy(source)
        exported[-1]['duration'] -= 1
        before = deepcopy(exported)
        result = compare_exported_notes(source, exported)
        self.assertFalse(result['durations_exact'])
        self.assertEqual(result['terminal_duration_deltas'][0]['delta_ticks'], -1)
        self.assertEqual(exported, before)

    def test_changed_earlier_duration_or_note_is_rejected(self):
        source = [dict(tick=0, note=60, duration=96), dict(tick=192, note=62, duration=96)]
        for changed_key, index in [('duration', 0), ('note', 1), ('tick', 1)]:
            exported = deepcopy(source)
            exported[index][changed_key] -= 1
            with self.subTest(changed_key=changed_key):
                with self.assertRaises(ValueError):
                    compare_exported_notes(source, exported)



class MixedDrumsTests(unittest.TestCase):
    def source(self, notes=False):
        levels = [.1] * 20 + [.8] * 6 + [.1] * 54
        windows = [dict(start_seconds=i/10, end_seconds=(i+1)/10, rms=level, peak=level*1.2)
                   for i, level in enumerate(levels)]
        source = dict(id=6, name='DRUMS', role='drums', mixer_insert=5,
                      analysis={'bpm': 120, 'level_windows': windows}, timing_provenance='shared-drum-guide')
        if notes:
            source['note_events'] = [dict(tick=400, note=60, duration=100, velocity=120)]
        return source

    def test_role_does_not_mislabel_mixed_drums_as_percussion(self):
        self.assertEqual(detect_role('DRUMS'), 'drums')
        self.assertEqual(role_family('drums'), 'Drums')
        self.assertEqual(detect_role('kick drum'), 'kick')

    def test_mixed_bus_has_no_rms_or_note_driven_gain(self):
        rows = [dict(type='verse', start_tick=0, end_tick=1600, energy=.7)]
        settings = {'section_depth_db': 0, 'max_level_correction_db': 6, 'amount_scale': 1.5}
        result = plan_balance([self.source(notes=True)], rows, 100, settings=settings)
        self.assertEqual(result['moves'], [])
        self.assertEqual(result['balance_details'][0]['maximum_correction_db'], 0)
        self.assertEqual(result['balance_details'][0]['guide_note_count'], 0)

    def test_mixed_bus_section_only_is_slow_and_capped(self):
        rows = [dict(type='verse', start_tick=0, end_tick=1600, energy=.7)]
        settings = {'section_depth_db': 6, 'max_level_correction_db': 6, 'amount_scale': 1.5}
        first = plan_balance([self.source()], rows, 100, settings=settings)
        second = plan_balance([self.source(notes=True)], rows, 100, settings=settings)
        move = first['moves'][0]
        self.assertEqual(move['points'], second['moves'][0]['points'])
        self.assertEqual((move['parameter'], move['plugin']), ('volume_db', 'Mixer'))
        self.assertTrue(all(-1.5 <= gain <= 0 for _, gain in move['points']))
        for (a, x), (b, y) in zip(move['points'], move['points'][1:]):
            self.assertLessEqual(abs(y-x) / ((b-a)/200), .2501)


if __name__ == '__main__':
    unittest.main()
