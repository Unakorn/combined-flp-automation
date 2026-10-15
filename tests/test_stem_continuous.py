"""Section development, silence, timing and bounds of continuous native curves."""
from copy import deepcopy
from pathlib import Path
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'work'))
from stem_continuous import DEFAULT_SETTINGS, _value, plan_continuous, validate_settings

PPQ = 96
BAR = PPQ * 4


def stem(identifier=1, role='keys', **analysis):
    return {'id': identifier, 'name': f'{role} {identifier}', 'role': role,
            'mixer_insert': identifier, 'analysis': {'bpm': 170, **analysis}}


def sections():
    return [dict(type=kind, start_tick=i * BAR * 8, end_tick=(i + 1) * BAR * 8, energy=energy)
            for i, (kind, energy) in enumerate([('intro', .6), ('verse', .75),
                    ('pre-chorus', .8), ('chorus', 1), ('bridge', .6), ('outro', .5)])]


def lane(plan, parameter, stem_id=1):
    return next(m for m in plan['moves'] if m['parameter'] == parameter and m['stem_id'] == stem_id)


class ContinuousTests(unittest.TestCase):
    def test_settings_are_editable_and_reject_invalid_fields_ranges(self):
        self.assertEqual(validate_settings(), DEFAULT_SETTINGS)
        self.assertEqual(validate_settings({'pan_maximum_percent': 12})['pan_maximum_percent'], 12)
        for settings in ({'unknown': 1}, {'amount_scale': float('nan')}, {'rhythm_release_ms': 99},
                         {'volume_minimum_db': 1}, {'phrase_bars': 7}, {'filter_minimum_percent': 90, 'filter_maximum_percent': 80}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_settings(settings)

    def test_one_full_song_curve_per_target_valid_explicit_points(self):
        plan = plan_continuous([stem(), stem(2, 'bass', sustained=.8), stem(3, 'brass', sustained=.7)], sections(), PPQ)
        targets = [(m['stem_id'], m['plugin'], m['parameter']) for m in plan['moves']]
        self.assertEqual(len(targets), len(set(targets)))
        self.assertLessEqual(len({m['control_family'] for m in plan['moves']}), 6)
        for move in plan['moves']:
            self.assertEqual(move['shape'], 'explicit')
            self.assertEqual((move['start_tick'], move['end_tick']), (0, BAR * 48))
            self.assertEqual((move['points'][0][0], move['points'][-1][0]), (0, BAR * 48))
            ticks = [t for t, _ in move['points']]
            self.assertEqual(ticks, sorted(set(ticks)))
            self.assertTrue(all(math.isfinite(v) for _, v in move['points']))
            self.assertLess(len(move['points']), 1500)

    def test_every_section_has_changing_active_controls_and_large_filter_range(self):
        plan = plan_continuous([stem()], sections(), PPQ)
        self.assertEqual(plan['movement_coverage']['sections_with_movement'], 6)
        move = lane(plan, 'lowpass_cutoff_percent')
        self.assertGreater(move['maximum_value'] - move['minimum_value'], 60)
        self.assertGreater(move['coverage']['changing_active_fraction'], .8)
        for section in plan['movement_coverage']['section_activity']:
            self.assertGreater(section['moving_targets'], 0)
        # A build finishes open; the ending gets progressively darker.
        self.assertEqual(_value(move['points'], BAR * 24), 100)
        self.assertLess(_value(move['points'], BAR * 48), _value(move['points'], BAR * 44))

    def test_known_silent_stems_have_no_lanes(self):
        result = plan_continuous([stem(activity_segments=[])], sections(), PPQ)
        self.assertEqual(result['moves'], [])
        self.assertEqual(result['movement_coverage']['sections_with_movement'], 0)

    def test_long_known_rest_is_flat_neutral_and_not_counted_as_active(self):
        source = stem()
        source['note_spans'] = [[0, BAR * 8], [BAR * 24, BAR * 32]]
        plan = plan_continuous([source], sections(), PPQ)
        for move in plan['moves']:
            self.assertEqual(_value(move['points'], BAR * 9), move['neutral_value'])
            self.assertEqual(_value(move['points'], BAR * 20), move['neutral_value'])
            self.assertEqual(move['coverage']['active_ticks'], BAR * 16)
        self.assertEqual(plan['movement_coverage']['section_activity'][1]['moving_targets'], 0)
        self.assertEqual(plan['movement_coverage']['section_activity'][2]['moving_targets'], 0)

    def test_audio_activity_uses_playlist_offset_once_and_retains_tail(self):
        source = stem(activity_segments=[{'start_seconds': 0, 'end_seconds': 60 / 170 * 16}])
        source.update(position=BAR * 8, offset_tick=BAR * 8, note_spans=[[BAR * 8, BAR * 9]])
        plan = plan_continuous([source], sections(), PPQ)
        move = lane(plan, 'lowpass_cutoff_percent')
        self.assertEqual(move['coverage']['active_ticks'], BAR * 4)
        self.assertEqual(_value(move['points'], BAR * 4), 100)
        self.assertLess(_value(move['points'], BAR * 11), 100)
        self.assertEqual(_value(move['points'], BAR * 13), 100)

    def test_no_added_gain_bass_or_kick_pan_and_bounded_effects(self):
        sources = [stem(i + 1, role, sustained=.7) for i, role in enumerate(['kick', 'bass', 'keys', 'hat', 'snare', 'brass'])]
        plan = plan_continuous(sources, sections(), PPQ, intensity_override='High', settings={'amount_scale': 1.5})
        self.assertFalse(any(m['role'] == 'kick' for m in plan['moves']))
        self.assertFalse(any(m['role'] == 'bass' and m['parameter'] != 'volume_db' for m in plan['moves']))
        for move in plan['moves']:
            values = [v for _, v in move['points']]
            if move['parameter'] == 'volume_db':
                self.assertLessEqual(max(values), 0)
                self.assertGreaterEqual(min(values), -9)
            elif move['parameter'] == 'lowpass_cutoff_percent':
                self.assertGreaterEqual(min(values), 0)
                self.assertLessEqual(max(values), 100)
            elif move['parameter'] == 'effect_blend_percent':
                self.assertGreaterEqual(min(values), 0)
                self.assertLessEqual(max(values), 50)

    def test_rhythm_only_main_sections_for_bass_or_sustained_melodic(self):
        sources = [stem(1, 'bass'), stem(2, 'guitar', sustained=.3), stem(3, 'brass', sustained=.7)]
        plan = plan_continuous(sources, sections(), PPQ)
        self.assertGreater(lane(plan, 'volume_db', 1)['rhythm_cycles'], 20)
        self.assertEqual(lane(plan, 'volume_db', 2)['rhythm_cycles'], 0)
        self.assertGreater(lane(plan, 'volume_db', 3)['rhythm_cycles'], 20)
        points = lane(plan, 'volume_db')['points']
        fast_falls = [(a, b) for a, b in zip(points, points[1:]) if b[1] - a[1] < -2 and b[0] - a[0] < PPQ]
        self.assertTrue(fast_falls)
        self.assertTrue(all(BAR * 24 <= a[0] < BAR * 32 for a, b in fast_falls))
        recoveries = [(a, b) for a, b in zip(points, points[1:]) if b[1] - a[1] > 2 and a[0] >= BAR * 24 and b[0] <= BAR * 32]
        self.assertTrue(all((b[0] - a[0]) * 60000 / (170 * PPQ) >= 100 for a, b in recoveries))

    def test_kick_timing_guides_volume_and_no_kicks_means_no_repeated_reduction(self):
        kick = stem(2, 'kick')
        kick['note_spans'] = [[BAR * 24 + 48, BAR * 24 + 72], [BAR * 26 + 24, BAR * 26 + 48]]
        plan = plan_continuous([stem(role='bass'), kick], sections(), PPQ)
        move = lane(plan, 'volume_db')
        self.assertEqual(move['rhythm_cycles'], 2)
        self.assertEqual(move['rhythm_timing'], 'Kick note timing')
        points = move['points']
        for start, _ in kick['note_spans']:
            self.assertGreater(_value(points, start) - _value(points, start + 5), 4)
        kick['note_spans'] = []
        self.assertEqual(lane(plan_continuous([stem(role='bass'), kick], sections(), PPQ), 'volume_db')['rhythm_cycles'], 0)

    def test_half_time_changes_rhythm_spacing_and_long_melodic_disables_it(self):
        normal = plan_continuous([stem(role='bass')], sections(), PPQ)
        half = plan_continuous([stem(role='bass')], sections(), PPQ, half_time=True)
        long = plan_continuous([stem(role='bass')], sections(), PPQ, flavor='long_melodic')
        self.assertEqual(lane(normal, 'volume_db')['rhythm_cycles'], lane(half, 'volume_db')['rhythm_cycles'] * 2)
        self.assertEqual(lane(long, 'volume_db')['rhythm_cycles'], 0)
        self.assertEqual(lane(half, 'volume_db')['end_tick'], BAR * 48)

    def test_silence_in_dynamic_range_does_not_make_active_curves_tiny(self):
        ordinary = lane(plan_continuous([stem(dynamic_range_db=12)], sections(), PPQ), 'lowpass_cutoff_percent')
        sparse = lane(plan_continuous([stem(dynamic_range_db=90)], sections(), PPQ), 'lowpass_cutoff_percent')
        self.assertEqual(ordinary['points'], sparse['points'])
        busy = lane(plan_continuous([stem(onset_density=12)], sections(), PPQ), 'lowpass_cutoff_percent')
        self.assertGreater(busy['maximum_value'] - busy['minimum_value'], 50)

    def test_lower_intensity_reduces_amount_without_removing_section_development(self):
        high = plan_continuous([stem()], sections(), PPQ, intensity_override='High')
        low = plan_continuous([stem()], sections(), PPQ, intensity_override='Low')
        self.assertGreater(lane(low, 'lowpass_cutoff_percent')['minimum_value'], lane(high, 'lowpass_cutoff_percent')['minimum_value'])
        self.assertEqual(low['movement_coverage']['sections_with_movement'], 6)

    def test_main_section_marker_uses_chorus_profiles_for_every_control(self):
        chorus = sections()
        main = deepcopy(chorus)
        main[3]['type'] = 'drop'
        a = plan_continuous([stem(sustained=.7)], chorus, PPQ)
        b = plan_continuous([stem(sustained=.7)], main, PPQ)
        self.assertEqual([m['points'] for m in a['moves']], [m['points'] for m in b['moves']])
        self.assertEqual(b['sections'][3]['type'], 'drop')

    def test_preset_baseline_changes_auto_but_not_explicit_intensity(self):
        high = plan_continuous([stem()], sections(), PPQ, baseline_intensity='High')
        low = plan_continuous([stem()], sections(), PPQ, baseline_intensity='Low')
        self.assertGreater(lane(low, 'lowpass_cutoff_percent')['minimum_value'], lane(high, 'lowpass_cutoff_percent')['minimum_value'])
        self.assertTrue(all(s['intensity'] == 'Low' for s in low['sections']))
        explicit_a = plan_continuous([stem()], sections(), PPQ, baseline_intensity='Low', intensity_override='High')
        explicit_b = plan_continuous([stem()], sections(), PPQ, baseline_intensity=.8, intensity_override='High')
        self.assertEqual([m['points'] for m in explicit_a['moves']], [m['points'] for m in explicit_b['moves']])
        for baseline in ('Extreme', float('nan'), 0, 2, True):
            with self.subTest(baseline=baseline), self.assertRaises(ValueError):
                plan_continuous([stem()], sections(), PPQ, baseline_intensity=baseline)

    def test_unknown_roles_sections_and_nonfinite_energy_require_correction(self):
        with self.assertRaisesRegex(ValueError, 'Confirm a role'):
            plan_continuous([stem(role='unknown')], sections(), PPQ)
        for kind in ('unknown', 'chorous'):
            rows = sections()
            rows[0]['type'] = kind
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'Confirm the section type'):
                plan_continuous([stem()], rows, PPQ)
        for energy in (float('nan'), float('inf'), 'loud', True):
            rows = sections()
            rows[0]['energy'] = energy
            with self.subTest(energy=energy), self.assertRaisesRegex(ValueError, 'finite number'):
                plan_continuous([stem()], rows, PPQ)

    def test_inputs_unchanged_and_empty_sections_supported(self):
        sources, rows = [stem()], sections()
        before = deepcopy((sources, rows))
        plan_continuous(sources, rows, PPQ)
        self.assertEqual((sources, rows), before)
        self.assertEqual(plan_continuous(sources, [], PPQ)['moves'], [])
        with self.assertRaises(ValueError):
            plan_continuous(sources, rows, 0)
        rows[1]['start_tick'] = 0
        with self.assertRaises(ValueError):
            plan_continuous(sources, rows, PPQ)


if __name__ == '__main__':
    unittest.main()
