"""Audio-derived reductions, preserved intent, optional timing, and native bounds."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'work'))
import stem_balance as balance
from stem_native import build_curve, parse_points

PPQ = 100
RATIO = 200  # 120BPM,100ticks per beat
LEVEL_ONLY = {'section_depth_db': 0}


def stem(levels, role='keys', identifier=1, **extra):
    windows = [dict(start_seconds=i / 10, end_seconds=(i + 1) / 10,
                    rms=level, peak=level * 1.2) for i, level in enumerate(levels)]
    return dict(id=identifier, name=f'{role} {identifier}', role=role, mixer_insert=identifier,
                analysis={'bpm': 120, 'level_windows': windows}, **extra)


def sections(end, kind='verse'):
    return [dict(type=kind, start_tick=0, end_tick=end, energy=.7)]


def plan(source, *, settings=None, rows=None, end=None):
    sources = source if isinstance(source, list) else [source]
    end = end or len(sources[0]['analysis']['level_windows']) * 20
    return balance.plan_balance(sources, rows or sections(end), PPQ, settings=settings)


def curve(result, identifier=1):
    return next(move['points'] for move in result['moves'] if move['stem_id'] == identifier)


class BalanceTests(unittest.TestCase):
    def test_loud_excursions_reduce_without_lifting_quiet_passages(self):
        source = stem([.1] * 20 + [.8] * 4 + [.1] * 40)
        result = plan(source, settings=LEVEL_ONLY)
        points = curve(result)
        self.assertEqual(balance._value(points, 200), 0)
        self.assertLess(balance._value(points, 480), -2)
        self.assertEqual(balance._value(points, 900), 0)
        self.assertTrue(all(-3 <= value <= 0 for _, value in points))
        self.assertAlmostEqual(result['balance_details'][0]['maximum_correction_db'], 3)

    def test_sustained_section_level_difference_is_preserved(self):
        source = stem([.04] * 40 + [.5] * 40)
        rows = [dict(type='verse', start_tick=0, end_tick=800, energy=.4),
                dict(type='chorus', start_tick=800, end_tick=1600, energy=1)]
        result = plan(source, settings=LEVEL_ONLY, rows=rows)
        self.assertEqual(result['moves'], [])
        detail = result['balance_details'][0]
        self.assertEqual(detail['status'], 'unchanged')
        self.assertGreater(detail['reference_by_section'][1]['reference_db'] - detail['reference_by_section'][0]['reference_db'], 20)

    def test_parts_with_different_original_levels_are_not_equalized(self):
        result = plan([stem([.01] * 50), stem([.7] * 50, 'brass', 2)], settings=LEVEL_ONLY)
        self.assertFalse(result['moves'])
        a, b = result['balance_details']
        self.assertLess(a['measured_maximum_db'], b['measured_minimum_db'] - 30)
        self.assertEqual([a['status'], b['status']], ['unchanged', 'unchanged'])

    def test_quiet_decay_holds_attenuation_until_measured_silence(self):
        source = stem([.1] * 20 + [.8] * 5 + [.0005, .0004, .0002, .0001, .00005] + [0] * 10)
        source.update(note_events=[dict(tick=400, duration=40, note=60, velocity=100)],
                      note_guide_source='Named MIDI')
        result = plan(source, settings=LEVEL_ONLY)
        points = curve(result)
        tail = [balance._value(points, tick) for tick in range(505, 600)]
        self.assertLess(tail[0], -1)
        self.assertTrue(all(abs(value - tail[0]) < 1e-6 for value in tail))
        self.assertEqual(balance._value(points, 610), 0)
        self.assertEqual(balance._value(points, 790), 0)
        self.assertTrue(all(value <= 0 for _, value in points))

    def test_midi_protects_entire_attack_but_audio_still_drives_gain(self):
        source = stem([.1] + [.9] * 3 + [.1] * 40, 'vocals')
        audio_only = plan(source, settings=LEVEL_ONLY)
        guided = deepcopy(source)
        guided.update(note_events=[dict(tick=20, duration=100, note=60, velocity=100)],
                      note_guide_source='MIDI: Vocals')
        result = plan(guided, settings=LEVEL_ONLY)
        points = curve(result)
        self.assertLess(balance._value(curve(audio_only), 30), 0)
        for tick in range(20, 37):
            self.assertEqual(balance._value(points, tick), 0)
        self.assertLess(balance._value(points, 60), 0)
        self.assertEqual(result['balance_details'][0]['guide_used'], 'MIDI: Vocals')
        # Velocity is already baked into audio; changing positive MIDI values
        # must not introduce a second velocity-to-volume conversion.
        guided['note_events'][0]['velocity'] = 20
        self.assertEqual(points, curve(plan(guided, settings=LEVEL_ONLY)))
        flat = stem([.1] * 44, 'vocals', note_events=guided['note_events'])
        self.assertFalse(plan(flat, settings=LEVEL_ONLY)['moves'])

    def test_dense_notes_leave_time_for_measured_level_correction(self):
        source = stem([.1] * 20 + [.8] * 10 + [.1] * 30)
        source.update(note_events=[dict(tick=tick, duration=15, note=60, velocity=100)
                                  for tick in range(0, 1200, 25)], note_guide_source='Matched arpeggio')
        result = plan(source, settings=LEVEL_ONLY)
        points = curve(result)
        self.assertGreater(result['balance_details'][0]['maximum_correction_db'], 2)
        self.assertEqual(result['balance_details'][0]['guide_note_count'], 48)
        # Guards shrink to40% of each125ms interval; reduction can progress
        # between attacks, but may not start a fresh dip inside an attack.
        for onset in range(400, 600, 25):
            self.assertAlmostEqual(balance._value(points, onset), balance._value(points, onset + 10), places=5)

    def test_phrase_end_protection_keeps_audio_release_and_has_exact_edges(self):
        source = stem([.1] * 20 + [.8] * 10 + [.1] * 30, 'vocals')
        source.update(note_spans=[[100, 400]], phrase_ends=[400], note_guide_source='Named source FLP')
        result = plan(source, settings=LEVEL_ONLY)
        points = curve(result)
        self.assertEqual(balance._value(points, 460), 0)
        self.assertEqual(balance._value(points, 470), 0)
        self.assertLess(balance._value(points, 500), 0)
        self.assertGreater(result['moves'][0]['coverage']['active_ticks'], 400)

    def test_regular_drum_attacks_are_not_compared_to_decay_median(self):
        decay = [.3, .12, .04, .01, .002, .0005, .0001, 0, 0, 0]
        for role in ('kick', 'snare', 'hat'):
            with self.subTest(role=role):
                result = plan(stem(decay * 8, role), settings=LEVEL_ONLY)
                self.assertFalse(result['moves'])
                self.assertEqual(result['balance_details'][0]['maximum_correction_db'], 0)

    def test_eight_drums_have_bounded_independent_volume_only_curves(self):
        sources = [stem([.1] * 15 + [.7] * 8 + [.1] * 57, role, i + 1)
                   for i, role in enumerate(sorted(balance._DRUMS))]
        result = plan(sources, settings={'section_depth_db': 6, 'max_level_correction_db': 6, 'amount_scale': 1.5})
        self.assertEqual(len(result['moves']), 8)
        for move in result['moves']:
            self.assertEqual((move['plugin'], move['parameter']), ('Mixer', 'volume_db'))
            self.assertGreaterEqual(move['minimum_value'], -1.5 if move['role'] == 'kick' else -3)
            self.assertLessEqual(move['maximum_value'], 0)

    def test_offset_duration_and_absolute_guide_are_applied_once(self):
        source = stem([.1] * 3 + [.7] * 3 + [.1] * 20,
                      offset_tick=400, position=400, duration=200,
                      note_events=[dict(tick=460, duration=100, note=60, velocity=100)],
                      note_guide_source='Named MIDI')
        result = plan(source, end=1200)
        points = curve(result)
        self.assertEqual(result['balance_details'][0]['guide_note_count'], 1)
        for tick in (0, 200, 399, 400, 650, 1199):
            self.assertEqual(balance._value(points, tick), 0)
        self.assertLess(balance._value(points, 500), 0)

    def test_no_guide_and_invalid_guide_keep_audio_analysis_available(self):
        source = stem([.1] * 20 + [.8] * 4 + [.1] * 40)
        a = plan(source, settings=LEVEL_ONLY)
        source['note_events'] = [dict(tick=float('nan'), duration=100)]
        b = plan(source, settings=LEVEL_ONLY)
        self.assertEqual(curve(a), curve(b))
        self.assertEqual(b['balance_details'][0]['guide_used'], 'Audio only')
        self.assertTrue(any('invalid note timing' in warning for warning in b['warnings']))

    def test_missing_audio_silence_and_disabled_amount_have_honest_status(self):
        missing = stem([.1] * 10)
        del missing['analysis']['level_windows']
        result = plan(missing, end=200)
        self.assertEqual(result['balance_details'][0]['status'], 'no_audio')
        self.assertFalse(result['moves'])
        silent = plan(stem([0] * 10))
        self.assertEqual(silent['balance_details'][0]['status'], 'unchanged')
        off = plan(stem([.1] * 10 + [.8] * 4 + [.1] * 30), settings={'amount_scale': 0})
        self.assertFalse(off['moves'])

    def test_finite_validation_and_tempo_are_required_for_actual_measurements(self):
        source = stem([.1] * 20)
        for key in ('rms', 'peak', 'end_seconds'):
            bad = deepcopy(source)
            bad['analysis']['level_windows'][0][key] = float('nan')
            with self.subTest(key=key), self.assertRaises(ValueError):
                plan(bad)
        bad = deepcopy(source)
        bad['analysis'].pop('bpm')
        with self.assertRaisesRegex(ValueError, 'tempo'):
            plan(bad)
        for settings in ({'attack_ms': 1}, {'release_ms': 0}, {'amount_scale': True}, {'unknown': 1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                plan(source, settings=settings)

    def test_full_native_curve_limits_no_source_mutation_and_section_contract(self):
        source = stem([.1] * 20 + [.8] * 4 + [.1] * 40)
        original = deepcopy(source)
        result = plan(source)
        self.assertEqual(source, original)
        self.assertEqual(result['movement_mode'], 'volume_balance')
        self.assertIn(result['sections'][0]['intensity'], balance._INTENSITY)
        points = curve(result)
        self.assertEqual((points[0][0], points[-1][0]), (0, 1280))
        self.assertEqual([t for t, _ in points], sorted({t for t, _ in points}))
        native = [(tick, (value + 6) / 6) for tick, value in points]
        decoded = parse_points(build_curve(native, PPQ), PPQ)
        self.assertEqual(len(native), len(decoded))
        for (tick, value), (expected_tick, expected_value) in zip(decoded, native):
            self.assertAlmostEqual(tick, expected_tick, places=8)
            self.assertAlmostEqual(value, expected_value, places=8)
        with patch.object(balance, 'MAX_POINTS', 2):
            limited = plan(source)
        self.assertEqual(limited['balance_details'][0]['status'], 'skipped')
        self.assertFalse(limited['moves'])


if __name__ == '__main__':
    unittest.main()
