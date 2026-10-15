"""Signal-based checks for bounded audio analysis and conservative role labels."""
from pathlib import Path
import json
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'work'))
from stem_audio import AudioAnalysisError, ROLES, analyze_audio, detect_role, detect_sections, role_family

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    np = sf = None


class RoleTests(unittest.TestCase):
    def test_user_examples_and_code_precedence(self):
        expected = {'1_808 OH': 'hat', '1_808 CH': 'hat', '1_808 Kick': 'kick',
                    '1_808 Snare': 'snare', '1_808 Tom': 'tom', '1_Alum Snare 2': 'snare',
                    '1_BAS Rocker': 'bass', '1_BRS Horns Sustain': 'brass',
                    '1_GTR Acoustic Padded': 'guitar', '1_GTR Acoustic': 'guitar',
                    '1_KEY Electric Piano Bright': 'keys', '12_02_BAS Kick layer': 'bass',
                    '04_VOC Alto': 'vocals', '1_PAD CH texture': 'pads',
                    '2_SYN Saw': 'synth', '2_LD Square': 'synth', '001_FX Boom': 'effects'}
        for name, role in expected.items():
            with self.subTest(name=name):
                self.assertEqual(detect_role(name), role)

    def test_808_requires_explicit_bass_or_measured_sustain(self):
        for name in ('808', '1_808', '808 experimental', '12_808 mystery'):
            self.assertIsNone(detect_role(name))
        for name in ('808 Bass', '808 Sub', '808 Low', '1_808 Low'):
            self.assertEqual(detect_role(name), 'bass')
        self.assertEqual(detect_role('808', {'sub_ratio': .95, 'sustained': .9}), 'bass')
        self.assertIsNone(detect_role('808', {'sub_ratio': .95, 'sustained': .1}))
        self.assertIsNone(detect_role('808', {'sub_ratio': .10, 'sustained': .95}))
        self.assertEqual(detect_role('808 Kick', {'sub_ratio': .99, 'sustained': .99}), 'kick')

    def test_unknowns_do_not_get_substring_or_default_roles(self):
        for name in ('01_Mystery', 'Thatcher', 'Snareless', '8080', '', 'Rhodesish Texture'):
            self.assertIsNone(detect_role(name), name)
        self.assertIsNone(detect_role('Mystery', {'sub_ratio': .99, 'sustained': .99}))
        self.assertEqual(role_family('hat'), 'Drums')
        self.assertEqual(role_family('brass'), 'Melodic')
        self.assertEqual(role_family('bass'), 'Bass')
        self.assertEqual(len(set(ROLES)), len(ROLES))
        with self.assertRaises(ValueError):
            role_family(None)


@unittest.skipIf(sf is None, 'Install numpy and soundfile to run signal-analysis tests.')
class AudioTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.rate = 8000

    def tone(self, name, frequency, duration=4, amplitude=.2, stereo=False):
        t = np.arange(round(duration * self.rate)) / self.rate
        wave = amplitude * np.sin(2 * np.pi * frequency * t)
        if stereo:
            wave = np.column_stack((wave, -wave))
        path = self.root / name
        sf.write(path, wave, self.rate, subtype='FLOAT')
        return path

    def test_low_and_bright_tones_have_distinct_measured_spectra(self):
        low = analyze_audio(self.tone('808.wav', 55), bpm=120)
        bright = analyze_audio(self.tone('bright.wav', 1800), bpm=120)
        self.assertAlmostEqual(low['chunks'][0]['rms'], .2 / math.sqrt(2), places=5)
        self.assertLess(abs(low['centroid'] - 55), 5)
        self.assertGreater(low['sub_ratio'], .98)
        self.assertLess(bright['sub_ratio'], .01)
        self.assertGreater(bright['centroid'], 1700)
        self.assertGreater(low['sustained'], .95)
        self.assertLess(low['dynamic_range_db'], 1)
        self.assertEqual(detect_role('808', low), 'bass')
        self.assertIsNone(detect_role('808', bright))
        json.dumps(low, allow_nan=False)

    def test_anti_phase_stereo_preserves_energy_and_centroid(self):
        audio = analyze_audio(self.tone('stereo.wav', 700, stereo=True), bpm=120)
        self.assertEqual(audio['channels'], 2)
        self.assertAlmostEqual(audio['chunks'][0]['rms'], .2 / math.sqrt(2), places=5)
        self.assertLess(abs(audio['centroid'] - 700), 5)

    def test_silent_and_short_audio_are_finite(self):
        path = self.root / 'silent.wav'
        sf.write(path, np.zeros(13), self.rate, subtype='FLOAT')
        audio = analyze_audio(path, bpm=120)
        self.assertEqual(audio['chunks'][0]['rms'], 0)
        self.assertEqual(audio['onset_density'], 0)
        self.assertEqual(audio['sustained'], 0)
        self.assertEqual(audio['dynamic_range_db'], 0)
        self.assertEqual(audio['duration_seconds'], 13 / self.rate)
        json.dumps(audio, allow_nan=False)

    def test_impulses_have_more_onsets_and_less_sustain_than_pad(self):
        duration = 8
        audio = np.zeros(self.rate * duration)
        rng = np.random.default_rng(7)
        for start in range(self.rate // 2, len(audio), self.rate // 2):
            length = 240
            audio[start:start + length] = .3 * rng.normal(size=length) * np.linspace(1, 0, length)
        path = self.root / 'drums.wav'
        sf.write(path, audio, self.rate, subtype='FLOAT')
        drums = analyze_audio(path, bpm=120)
        pad = analyze_audio(self.tone('pad.wav', 220, duration=duration), bpm=120)
        self.assertGreater(drums['onset_density'], 1.5)
        self.assertLess(pad['onset_density'], .3)
        self.assertLess(drums['sustained'], .15)
        self.assertGreater(drums['dynamic_range_db'], 40)

    def test_eight_bar_chunks_follow_tempo_and_stream_bounded_reads(self):
        path = self.tone('long.wav', 110, duration=35)
        reads = []
        original = sf.SoundFile

        class Reader:
            def __init__(self, path):
                self.file = original(path)
            def __getattr__(self, name):
                return getattr(self.file, name)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.file.close()
            def read(self, frames=-1, **kwargs):
                reads.append(frames)
                if not 0 < frames <= 65536:
                    raise AssertionError('A whole-file read violates the streaming budget.')
                return self.file.read(frames, **kwargs)

        with mock.patch.object(sf, 'SoundFile', Reader):
            result = analyze_audio(path, bpm=120)
        self.assertGreater(len(reads), 4)
        self.assertEqual([(c['start_seconds'], c['end_seconds']) for c in result['chunks']],
                         [(0, 16), (16, 32), (32, 35)])
        result = analyze_audio(path, bpm=60)
        self.assertEqual([(c['start_seconds'], c['end_seconds']) for c in result['chunks']], [(0, 32), (32, 35)])

    def test_cache_hits_and_pcm_changes_invalidate_with_same_timestamp(self):
        path = self.tone('cache.wav', 55)
        cache = self.root / 'cache'
        first = analyze_audio(path, bpm=120, cache_dir=cache)
        with mock.patch('stem_audio._audio_modules', side_effect=AssertionError('Should use cache')):
            second = analyze_audio(path, bpm=120, cache_dir=cache)
        self.assertFalse(first['cache_hit'])
        self.assertTrue(second['cache_hit'])
        original = path.stat()
        self.tone('cache.wav', 1700)
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
        third = analyze_audio(path, bpm=120, cache_dir=cache)
        self.assertFalse(third['cache_hit'])
        self.assertNotEqual(first['sha256'], third['sha256'])
        self.assertGreater(third['centroid'], first['centroid'] + 1000)
        fourth = analyze_audio(path, bpm=125, cache_dir=cache)
        self.assertFalse(fourth['cache_hit'])
        self.assertNotEqual(third['cache_key'], fourth['cache_key'])

    def test_corrupt_cache_is_recomputed(self):
        path = self.tone('cache.wav', 110)
        cache = self.root / 'cache'
        first = analyze_audio(path, bpm=120, cache_dir=cache)
        cached = cache / (first['cache_key'] + '.json')
        cached.write_text('{bad json', encoding='utf-8')
        self.assertFalse(analyze_audio(path, bpm=120, cache_dir=cache)['cache_hit'])
        saved = json.loads(cached.read_text(encoding='utf-8'))
        saved['chunks'][0]['rms'] = -10
        cached.write_text(json.dumps(saved), encoding='utf-8')
        self.assertFalse(analyze_audio(path, bpm=120, cache_dir=cache)['cache_hit'])

    def test_invalid_settings_and_nonfinite_pcm_fail(self):
        path = self.tone('test.wav', 100)
        for value in (0, -1, float('nan'), float('inf')):
            with self.subTest(bpm=value), self.assertRaises(AudioAnalysisError):
                analyze_audio(path, bpm=value)
        sf.write(path, np.array([0, np.nan, .1]), self.rate, subtype='FLOAT')
        with self.assertRaisesRegex(AudioAnalysisError, 'non-finite'):
            analyze_audio(path, bpm=120)

    def test_phrase_ends_follow_envelope_gaps_not_fixed_bar_boundaries(self):
        t = np.arange(6 * self.rate) / self.rate
        audio = .2 * np.sin(2 * np.pi * 440 * t)
        audio[(t < .5) | ((t >= 1.5) & (t < 2.0)) | (t >= 4.0)] = 0
        # A short consonant-like gap stays inside the second phrase.
        audio[(t >= 2.5) & (t < 2.57)] = 0
        path = self.root / 'vocal.wav'
        sf.write(path, audio, self.rate, subtype='FLOAT')
        result = analyze_audio(path, bpm=120)
        self.assertEqual(len(result['activity_segments']), 2)
        for measured, expected in zip(result['phrase_ends'], (1.5, 4.0)):
            self.assertLess(abs(measured - expected), .07)
        self.assertLess(abs(result['activity_segments'][1]['start_seconds'] - 2.0), .07)


class SectionTests(unittest.TestCase):
    def analysis(self, values, seconds=16):
        return dict(chunks=[dict(start_seconds=i * seconds, end_seconds=(i + 1) * seconds,
                                rms=v, onset_density=1, centroid=500, sub_ratio=.1, sustained=.5)
                           for i, v in enumerate(values)])

    def test_energy_changes_snap_to_bars_and_preserve_complete_coverage(self):
        ppq = 96
        sections = detect_sections([self.analysis([.1, .1, .7, .7, .15, .15, .7, .7, .05])],
                                   bpm=120, ppq=ppq, end_tick=72 * 4 * ppq)
        self.assertEqual([s['bar'] for s in sections], [1, 17, 33, 49, 65])
        self.assertEqual([s['type'] for s in sections], ['intro', 'chorus', 'verse', 'chorus', 'outro'])
        self.assertEqual(sections[1]['energy'], 1)
        self.assertGreater(sections[1]['energy_jump'], .7)
        self.assertEqual(sections[0]['start_tick'], 0)
        self.assertEqual(sections[-1]['end_tick'], 72 * 4 * ppq)
        for a, b in zip(sections, sections[1:]):
            self.assertEqual(a['end_tick'], b['start_tick'])
            self.assertEqual(b['start_tick'] % (ppq * 4), 0)

    def test_all_stems_affect_aggregate_and_offsets_are_respected(self):
        constant = self.analysis([.2] * 4)
        later = self.analysis([.8] * 2)
        sections = detect_sections([constant, dict(analysis=later, offset_tick=16 * 384)],
                                   bpm=120, ppq=96, end_tick=32 * 384)
        self.assertEqual([s['bar'] for s in sections], [1, 17])
        self.assertAlmostEqual(sections[0]['energy'], .2 / math.sqrt(.2 ** 2 + .8 ** 2))
        self.assertAlmostEqual(sections[1]['energy'], 1)
        same = detect_sections([constant, dict(analysis=later, offset_seconds=32)],
                               bpm=120, ppq=96, end_tick=32 * 384)
        self.assertEqual(sections, same)

    def test_markers_override_heuristic_and_halftime_keeps_real_bar_labels(self):
        markers = [dict(tick=0, name='Intro'), dict(tick=8 * 384 + 20, name='Pre-Chorus'),
                   dict(tick=16 * 384 + 10, name='Drop'), dict(tick=24 * 384, name='Outro')]
        sections = detect_sections([self.analysis([.5] * 4)], bpm=180, ppq=96,
                                   end_tick=32 * 384, markers=markers, half_time=True)
        self.assertEqual([s['type'] for s in sections], ['intro', 'pre-chorus', 'drop', 'outro'])
        self.assertEqual([s['bar'] for s in sections], [1, 9, 17, 25])
        self.assertTrue(all(s['start_tick'] % 768 == 0 for s in sections))
        self.assertTrue(all(s['source'] == 'project marker' for s in sections))

    def test_constant_or_silent_stems_do_not_invent_choruses(self):
        for value in (0, .5):
            result = detect_sections([self.analysis([value] * 4)], bpm=120, ppq=96, end_tick=32 * 384)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['type'], 'verse')
            self.assertEqual(result[0]['energy_jump'], 0)

    def test_post_chorus_preserves_marker_name_without_triggering_a_chorus(self):
        sections = detect_sections([self.analysis([.5] * 4)], bpm=120, ppq=96,
                                   end_tick=32 * 384, markers=[dict(tick=8 * 384, name='Post-Chorus')])
        self.assertEqual(sections[1]['type'], 'verse')
        self.assertEqual(sections[1]['marker_name'], 'Post-Chorus')


if __name__ == '__main__':
    unittest.main()
