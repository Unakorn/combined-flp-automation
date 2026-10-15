"""Behavioral synthetic-audio tests; never relies on recorded user material."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'work'))
from eq_audio import DEFAULT_CENTERS, extract_features_array, plan_from_features, plan_eq_automation


RATE = 24000


def tone(frequency, duration=5, amplitude=.1):
    return amplitude * np.sin(2 * np.pi * frequency * np.arange(round(RATE * duration)) / RATE)


def row(identifier, role, audio):
    return {'id': identifier, 'name': role or 'Insert 7', 'role': role,
            'features': extract_features_array(audio, RATE)}


def band_curve(plan, stem_id, band):
    return next(r for r in plan['stems'] if r['id'] == stem_id)['bands'][band]['points']


def value_at(points, time):
    return float(np.interp(time, [p['seconds'] for p in points], [p['db'] for p in points]))


class EqAudioTests(unittest.TestCase):
    def test_silence_has_no_movements_or_nonfinite_values(self):
        plan = plan_from_features([row(1, 'pad', np.zeros(RATE * 2))])
        for band in plan['stems'][0]['bands']:
            self.assertFalse(band['active'])
            self.assertEqual(band['points'], [{'seconds': 0., 'db': 0.}, {'seconds': 2., 'db': 0.}])

    def test_antiphase_stereo_retains_same_band_energy_as_mono(self):
        mono = tone(DEFAULT_CENTERS[0], duration=1)
        a = extract_features_array(mono, RATE)
        b = extract_features_array(np.stack((mono, -mono), axis=1), RATE)
        np.testing.assert_allclose(a['rms'], b['rms'], atol=1e-12)
        np.testing.assert_allclose(a['band_power'], b['band_power'], atol=1e-12)
        self.assertGreater(float(b['band_power'][:, 0].max()), .004)

    def test_constant_pitched_stem_is_not_forced_toward_flat_spectrum(self):
        plan = plan_from_features([row(1, 'bass', tone(DEFAULT_CENTERS[0]))])
        self.assertFalse(any(b['active'] for b in plan['stems'][0]['bands']))

    def test_overlap_selectively_cuts_pad_low_band_when_bass_enters(self):
        pad = tone(DEFAULT_CENTERS[0]) + tone(DEFAULT_CENTERS[3])
        bass = tone(DEFAULT_CENTERS[0])
        bass[:2 * RATE] = 0
        bass[4 * RATE:] = 0
        plan = plan_from_features([row(1, 'pad', pad), row(2, 'bass', bass)])
        low = band_curve(plan, 1, 0)
        self.assertAlmostEqual(value_at(low, 1), 0, places=6)
        self.assertLess(value_at(low, 3.5), -.6)
        self.assertEqual(value_at(low, 5), 0)
        self.assertFalse(plan['stems'][0]['bands'][3]['active'])
        self.assertFalse(any(b['active'] for b in plan['stems'][1]['bands']))

    def test_nonoverlapping_stems_do_not_create_cuts(self):
        plan = plan_from_features([row(1, 'pad', tone(DEFAULT_CENTERS[3])),
                                   row(2, 'bass', tone(DEFAULT_CENTERS[0]))])
        self.assertFalse(any(b['active'] for s in plan['stems'] for b in s['bands']))

    def test_silent_lead_in_does_not_get_automation_from_later_activity(self):
        pad = tone(DEFAULT_CENTERS[0])
        pad[:2 * RATE] = 0
        plan = plan_from_features([row(1, 'pad', pad), row(2, 'bass', tone(DEFAULT_CENTERS[0]))])
        low = band_curve(plan, 1, 0)
        self.assertEqual(value_at(low, 1.8), 0)
        self.assertLess(value_at(low, 3.5), -.5)

    def test_unknown_role_is_explicitly_left_flat(self):
        plan = plan_from_features([row(1, None, tone(DEFAULT_CENTERS[0])),
                                   row(2, 'bass', tone(DEFAULT_CENTERS[0]))])
        self.assertTrue(plan['warnings'])
        self.assertIsNone(plan['stems'][0]['role'])
        self.assertFalse(any(b['active'] for b in plan['stems'][0]['bands']))

    def test_level_change_alone_does_not_ride_all_bands(self):
        audio = sum(tone(f) for f in DEFAULT_CENTERS)
        audio[2 * RATE:] *= 3
        plan = plan_from_features([row(1, 'pad', audio)])
        self.assertFalse(any(b['active'] for b in plan['stems'][0]['bands']))

    def test_sustained_new_prominence_is_selective_and_short_attack_is_ignored(self):
        baseline = tone(DEFAULT_CENTERS[0]) + tone(DEFAULT_CENTERS[2], amplitude=.008)
        extra = tone(DEFAULT_CENTERS[2], amplitude=.18)
        sustained = baseline.copy()
        sustained[2 * RATE:4 * RATE] += extra[2 * RATE:4 * RATE]
        short = baseline.copy()
        short[2 * RATE:round(2.08 * RATE)] += extra[2 * RATE:round(2.08 * RATE)]
        plan = plan_from_features([row(1, 'pad', sustained)])
        self.assertLess(value_at(band_curve(plan, 1, 2), 3.7), -.3)
        self.assertFalse(plan['stems'][0]['bands'][0]['active'])
        transient_plan = plan_from_features([row(1, 'pad', short)])
        self.assertFalse(any(b['active'] for b in transient_plan['stems'][0]['bands']))

    def test_muted_competitor_cannot_drive_cuts(self):
        pad = row(1, 'pad', tone(DEFAULT_CENTERS[0]))
        bass = row(2, 'bass', tone(DEFAULT_CENTERS[0]))
        bass['mixer_gain'] = 0
        plan = plan_from_features([pad, bass])
        self.assertFalse(any(b['active'] for s in plan['stems'] for b in s['bands']))

    def test_activity_to_silence_releases_without_jumps(self):
        pad = tone(DEFAULT_CENTERS[0], duration=7)
        pad[3 * RATE:] = 0
        plan = plan_from_features([row(1, 'pad', pad),
                                   row(2, 'bass', tone(DEFAULT_CENTERS[0], duration=7))])
        low = band_curve(plan, 1, 0)
        self.assertLess(value_at(low, 2.5), -.6)
        self.assertLess(value_at(low, 3.2), -.1)
        self.assertAlmostEqual(value_at(low, 6.8), 0, delta=.02)
        for a, b in zip(low, low[1:]):
            self.assertLessEqual(abs(b['db'] - a['db']), 1.5 * (b['seconds'] - a['seconds']) + 2e-6)

    def test_intermittent_pulses_have_rate_limited_continuous_curves(self):
        pad = sum(tone(f, duration=8) for f in DEFAULT_CENTERS)
        for start in np.arange(.6, 8, 1.2):
            pad[round(start * RATE):min(len(pad), round((start + .6) * RATE))] = 0
        full = sum(tone(f, duration=8) for f in DEFAULT_CENTERS)
        plan = plan_from_features([row(1, 'pad', pad), row(2, 'lead', full), row(3, 'bass', full)])
        for stem in plan['stems']:
            for band in stem['bands']:
                for a, b in zip(band['points'], band['points'][1:]):
                    self.assertLessEqual(abs(b['db'] - a['db']), 1.5 * (b['seconds'] - a['seconds']) + 2e-6)

    def test_finite_bounded_curves_with_exact_baseline_endpoints(self):
        audio = sum(tone(f) for f in DEFAULT_CENTERS)
        plan = plan_from_features([row(1, 'pad', audio), row(2, 'lead', audio),
                                   row(3, 'bass', audio)], max_cut_db=1.5)
        times = np.arange(0, 5.001, .1)
        for stem in plan['stems']:
            values = []
            for band in stem['bands']:
                points = band['points']
                self.assertEqual((points[0]['seconds'], points[-1]['seconds']), (0, 5))
                self.assertEqual((points[0]['db'], points[-1]['db']), (0, 0))
                self.assertTrue(all(np.isfinite(p['db']) and -1.5 <= p['db'] <= 0 for p in points))
                self.assertEqual([p['seconds'] for p in points], sorted({p['seconds'] for p in points}))
                values.append(np.interp(times, [p['seconds'] for p in points], [p['db'] for p in points]))
            self.assertLessEqual(float(-np.sum(values, axis=0).min()), 2.30)

    def test_nonfinite_pcm_is_rejected(self):
        audio = tone(77, duration=1)
        audio[50] = np.nan
        with self.assertRaises(ValueError):
            extract_features_array(audio, RATE)

    def test_differently_tuned_bands_do_not_create_false_ordinal_overlap(self):
        shifted = (240., 500., 900., 2500., 7400.)
        queries = sorted(set(DEFAULT_CENTERS) | set(shifted))
        pad = {'id': 1, 'name': 'pad', 'centers_hz': DEFAULT_CENTERS,
               'features': extract_features_array(tone(77), RATE, comparison_centers_hz=queries)}
        bass = {'id': 2, 'name': 'bass', 'centers_hz': shifted,
                'features': extract_features_array(tone(240), RATE, centers_hz=shifted,
                                                   comparison_centers_hz=queries)}
        plan = plan_from_features([pad, bass])
        self.assertFalse(any(b['active'] for s in plan['stems'] for b in s['bands']))
        self.assertEqual(plan['stems'][1]['bands'][0]['center_hz'], 240)

    def test_competitor_is_measured_at_destination_frequency_not_its_own_band(self):
        shifted = (240., 500., 900., 2500., 7400.)
        queries = sorted(set(DEFAULT_CENTERS) | set(shifted))
        pad = {'id': 1, 'name': 'pad', 'centers_hz': DEFAULT_CENTERS,
               'features': extract_features_array(tone(77), RATE, comparison_centers_hz=queries)}
        bass = {'id': 2, 'name': 'bass', 'centers_hz': shifted,
                'features': extract_features_array(tone(77), RATE, centers_hz=shifted,
                                                   comparison_centers_hz=queries)}
        # The bass's own first EQ band misses its 77 Hz audio; the pad's band
        # still competes with that actual audio and should receive a cut.
        self.assertLess(float(bass['features']['band_power'][:, 0].max()), 1e-5)
        plan = plan_from_features([pad, bass])
        self.assertLess(value_at(band_curve(plan, 1, 0), 3), -.6)
        self.assertFalse(plan['warnings'])

    def test_feature_frequencies_cannot_be_relabelled_after_analysis(self):
        pad = row(1, 'pad', tone(77))
        pad['centers_hz'] = (90., 300., 900., 2500., 7400.)
        with self.assertRaisesRegex(ValueError, 'differ from the analyzed'):
            plan_from_features([pad])

    def test_file_api_passes_actual_centers_and_returns_delta_gain(self):
        centers = (220., 500., 900., 2500., 7400.)
        calls = []
        def synthetic_read(path, *, centers_hz, hop_seconds, comparison_centers_hz):
            calls.append((path, tuple(centers_hz), tuple(comparison_centers_hz)))
            return extract_features_array(tone(220), RATE, centers_hz=centers_hz,
                                          comparison_centers_hz=comparison_centers_hz,
                                          hop_seconds=hop_seconds)
        stems = [{'id': 1, 'name': 'pad', 'sample': 'pad.wav', 'centers_hz': centers,
                  'saved_gain_db': [2., 0., 0., 0., 0.]},
                 {'id': 2, 'name': 'bass', 'sample': 'bass.wav'}]
        with patch('eq_audio.analyze_stem', side_effect=synthetic_read):
            plan = plan_eq_automation(stems, bpm=130, ppq=96)
        self.assertEqual(calls[0][1], centers)
        self.assertEqual(calls[1][1], DEFAULT_CENTERS)
        self.assertEqual(set(calls[0][2]), set(centers) | set(DEFAULT_CENTERS))
        self.assertEqual(plan['stems'][0]['bands'][0]['center_hz'], 220.)
        self.assertEqual(plan['stems'][0]['bands'][0]['evidence']['center_hz'], 220.)
        self.assertEqual(plan['gain_semantics'], 'delta_db_relative_to_saved_band_gain')
        points = band_curve(plan, 1, 0)
        self.assertEqual(points[0]['db'], 0.)
        self.assertLess(min(p['db'] for p in points), -.5)


if __name__ == '__main__':
    unittest.main()
