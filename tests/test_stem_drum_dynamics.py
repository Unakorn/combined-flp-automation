"""Matched-hit contrast, preserved attacks, native curve limits, and coverage."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'work'))
from stem_continuous import _covered, _value, plan_continuous, validate_settings
from stem_drum_dynamics import DRUM_ROLES, MAX_POINTS, apply_drum_dynamics

PPQ = 96
END = PPQ * 32
SECTIONS = [dict(type='verse', start_tick=0, end_tick=END // 2, energy=.7),
            dict(type='chorus', start_tick=END // 2, end_tick=END, energy=1)]


def source(role='hat', identifier=1, velocities=None):
    ticks = list(range(0, END, PPQ // 4))
    velocities = velocities or [100] * len(ticks)
    return dict(id=identifier, name=f'{role} {identifier}', role=role, mixer_insert=identifier,
                duration=END, analysis={'bpm': 120, 'activity_segments': [dict(start_seconds=0, end_seconds=16)]},
                note_events=[dict(tick=tick, duration=12, note=60, velocity=velocities[i % len(velocities)])
                             for i, tick in enumerate(ticks)],
                note_velocity_max=128, note_guide_source='Matched source FLP',
                note_spans=[[tick, tick + 12] for tick in ticks])


def volume(plan, identifier=1):
    return next(m for m in plan['moves'] if m['stem_id'] == identifier and m['parameter'] == 'volume_db')


class DrumDynamicsTests(unittest.TestCase):
    def test_all_eight_roles_get_one_full_song_volume_curve(self):
        stems = [source(role, i + 1) for i, role in enumerate(sorted(DRUM_ROLES))]
        result = plan_continuous(stems, SECTIONS, PPQ, intensity_override='High')
        self.assertEqual(len(result['drum_dynamics']), 8)
        self.assertTrue(all(r['status'] == 'planned' for r in result['drum_dynamics']))
        for stem in stems:
            moves = [m for m in result['moves'] if m['stem_id'] == stem['id'] and m['parameter'] == 'volume_db']
            self.assertEqual(len(moves), 1)
            move = moves[0]
            self.assertTrue(move['drum_dynamics'])
            self.assertEqual((move['start_tick'], move['end_tick']), (0, END))
            self.assertEqual(move['shape'], 'explicit')
            ticks = [t for t, _ in move['points']]
            self.assertEqual(ticks, sorted(set(ticks)))
            self.assertLess(len(ticks), MAX_POINTS)
            self.assertGreater(move['maximum_value'] - move['minimum_value'], .1)
            self.assertGreaterEqual(move['minimum_value'], -9)
            self.assertLessEqual(move['maximum_value'], 0)
            if stem['role'] == 'kick':
                self.assertGreaterEqual(move['minimum_value'], -1.5)
                self.assertEqual(len([m for m in result['moves'] if m['stem_id'] == stem['id']]), 1)

    def test_written_velocity_order_is_not_replaced_by_meter_accents(self):
        stem = source('tom', velocities=[48, 100])
        result = plan_continuous([stem], SECTIONS, PPQ, intensity_override='High')
        points = volume(result)['points']
        for event in stem['note_events']:
            gain = _value(points, event['tick'])
            self.assertAlmostEqual(gain, 0 if event['velocity'] == 100 else -2.6, places=5)
        self.assertEqual(result['drum_dynamics'][0]['accent_source'], 'Written note velocity order')

    def test_gain_is_ready_at_each_fast_hit_and_attack_does_not_dip(self):
        stem = source('snare', velocities=[100, 50, 90, 45])
        # A roll every24ticks at120BPM =125ms, shorter than old180ms release.
        result = plan_continuous([stem], SECTIONS, PPQ, intensity_override='High')
        points = volume(result)['points']
        self.assertEqual(result['drum_dynamics'][0]['hit_count'], len(stem['note_events']))
        for event in stem['note_events'][:-1]:
            tick = event['tick']
            expected = -4 * (100 - event['velocity']) / 100
            if tick:
                self.assertAlmostEqual(_value(points, tick - 1), expected, places=5)
            self.assertAlmostEqual(_value(points, tick), expected, places=5)
            self.assertAlmostEqual(_value(points, tick + 2), expected, places=5)
        self.assertEqual(result, plan_continuous([stem], SECTIONS, PPQ, intensity_override='High'))

    def test_zero_length_notes_still_generate_hits(self):
        stem = source('kick')
        for event in stem['note_events']:
            event['duration'] = 0
        result = plan_continuous([stem], SECTIONS, PPQ)
        self.assertEqual(result['drum_dynamics'][0]['hit_count'], len(stem['note_events']))
        self.assertTrue(volume(result)['drum_dynamics'])

    def test_one_and_two_tick_retriggers_are_not_dropped_or_retimed(self):
        for spacing in (1, 2):
            with self.subTest(spacing=spacing):
                stem = source('tom')
                stem['note_events'] = [dict(tick=i * spacing, duration=0, note=60,
                                           velocity=100 if i % 2 else 48) for i in range(50)]
                result = plan_continuous([stem], SECTIONS, PPQ, intensity_override='High')
                self.assertEqual(result['drum_dynamics'][0]['hit_count'], 50)
                points = volume(result)['points']
                for event in stem['note_events']:
                    self.assertAlmostEqual(_value(points, event['tick']),
                                           0 if event['velocity'] == 100 else -2.6, places=5)

    def test_isolated_hits_hold_attack_then_restore_smoothly(self):
        stem = source('tom')
        stem['note_events'] = [dict(tick=PPQ, duration=0, note=60, velocity=48),
                               dict(tick=PPQ * 4, duration=0, note=60, velocity=100)]
        result = plan_continuous([stem], SECTIONS, PPQ, intensity_override='High')
        points = volume(result)['points']
        self.assertAlmostEqual(_value(points, PPQ - 1), -2.6)
        self.assertAlmostEqual(_value(points, PPQ + 15), -2.6)
        tail = [_value(points, tick) for tick in range(PPQ + 15, PPQ + 45)]
        self.assertEqual(tail, sorted(tail))
        self.assertEqual(tail[-1], 0)
        self.assertEqual(_value(points, PPQ * 4), 0)

    def test_absolute_notes_apply_playlist_offset_once_and_reject_silent_notes(self):
        stem = source('tom', velocities=[100, 48])
        stem.update(position=PPQ * 8, offset_tick=PPQ * 8)
        stem['analysis']['activity_segments'] = [dict(start_seconds=0, end_seconds=4)]
        result = plan_continuous([stem], SECTIONS, PPQ)
        report = result['drum_dynamics'][0]
        self.assertEqual(report['hit_count'], 32)
        self.assertEqual(report['ignored_note_count'], 96)
        points = volume(result)['points']
        self.assertEqual(_value(points, PPQ * 4), 0)
        self.assertLess(_value(points, PPQ * 8 + 24), 0)
        self.assertEqual(_value(points, PPQ * 20), 0)

    def test_missing_guides_keep_original_section_curves_and_kick_unmodified(self):
        stems = [source(role, i + 1) for i, role in enumerate(('hat', 'snare', 'kick'))]
        for stem in stems:
            stem.pop('note_events')
        result = plan_continuous(stems, SECTIONS, PPQ)
        disabled = plan_continuous(stems, SECTIONS, PPQ, settings={'drum_dynamics_amount': 0})
        self.assertEqual(result['moves'], disabled['moves'])
        self.assertTrue(all(r['status'] == 'unmatched' for r in result['drum_dynamics']))
        self.assertEqual(len(result['warnings']), 3)
        self.assertFalse(any(m['role'] == 'kick' for m in result['moves']))

    def test_known_silence_is_reported_without_any_curves(self):
        stem = source('kick')
        stem['analysis']['activity_segments'] = []
        result = plan_continuous([stem], SECTIONS, PPQ)
        self.assertFalse(result['moves'])
        self.assertEqual(result['drum_dynamics'][0]['status'], 'silent')

    def test_excessive_note_count_skips_dynamics_and_retains_base_curve(self):
        stem = source('hat')
        stem['note_events'] = [dict(tick=i, duration=0, note=60, velocity=50 + i % 2 * 50)
                               for i in range(13000)]
        base = [(0, -.3), (15000, -.6)]
        rows = [dict(start_tick=0, end_tick=15000, amount=1)]
        points, report = apply_drum_dynamics(stem, base, rows, ppq=PPQ, bpm=120,
            song_end=15000, activity=[(0, 15000)])
        self.assertIs(points, base)
        self.assertEqual(report['status'], 'skipped')
        self.assertIn('64,000', report['reason'])

    def test_duplicate_onsets_use_strongest_note_once(self):
        stem = source('clap', velocities=[100, 50])
        stem['note_events'] += [dict(event, velocity=20) for event in stem['note_events']]
        result = plan_continuous([stem], SECTIONS, PPQ, intensity_override='High')
        self.assertEqual(result['drum_dynamics'][0]['hit_count'], len(stem['note_spans']))
        self.assertEqual(_value(volume(result)['points'], 0), 0)

    def test_amount_and_volume_bounds_are_editable_without_altering_other_roles(self):
        for value in (-.1, 1.1, True, float('nan')):
            with self.assertRaises(ValueError):
                validate_settings({'drum_dynamics_amount': value})
        melodic = source('keys')
        a = plan_continuous([melodic], SECTIONS, PPQ)
        b = plan_continuous([melodic], SECTIONS, PPQ, settings={'drum_dynamics_amount': 0})
        self.assertEqual(a['moves'], b['moves'])
        self.assertEqual(a['movement_coverage'], b['movement_coverage'])
        kick = source('kick')
        result = plan_continuous([kick], SECTIONS, PPQ, settings={'volume_minimum_db': -.5})
        self.assertGreaterEqual(volume(result)['minimum_value'], -.5)


class CoverageTests(unittest.TestCase):
    def test_coverage_does_not_scan_every_point_for_every_audio_segment(self):
        class CountingList(list):
            scans = 0
            def __iter__(self):
                self.scans += 1
                return super().__iter__()
        points = CountingList((tick, -float(tick % 2)) for tick in range(20001))
        activity = [(tick, tick + 1) for tick in range(0, 20000, 2)]
        coverage = _covered(points, activity, 0, 20000, .02)
        self.assertEqual(coverage['active_ticks'], 10000)
        self.assertEqual(coverage['moving_ticks'], 10000)
        self.assertLessEqual(points.scans, 1)


if __name__ == '__main__':
    unittest.main()
