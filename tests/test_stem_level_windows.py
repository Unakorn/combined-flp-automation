"""Measured short-window RMS/sample peaks, streaming boundaries and cache QA."""
from copy import deepcopy
from pathlib import Path
import json
import math
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'work'))
import stem_audio

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    np = sf = None


@unittest.skipIf(sf is None, 'Install numpy and soundfile to run level-window tests.')
class LevelWindowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.rate = 8000

    def write(self, samples, name='levels.wav', rate=None):
        path = self.root / name
        sf.write(path, np.asarray(samples, dtype=np.float32), rate or self.rate, subtype='FLOAT')
        return path

    def analyze(self, samples, **kwargs):
        return stem_audio.analyze_audio(self.write(samples), bpm=120, **kwargs)

    def test_timed_amplitude_steps_have_measured_rms_and_sample_peak(self):
        # Forty complete sine cycles per bin have exact RMS A/sqrt(2).
        carrier = np.sin(2 * np.pi * 400 * np.arange(800) / self.rate)
        amplitudes = (.05, .2, .8, 0, .1)
        result = self.analyze(np.concatenate([amplitude * carrier for amplitude in amplitudes]))
        self.assertEqual(len(result['level_windows']), 5)
        for index, (window, amplitude) in enumerate(zip(result['level_windows'], amplitudes)):
            self.assertEqual(window['start_seconds'], index / 10)
            self.assertEqual(window['end_seconds'], (index + 1) / 10)
            self.assertAlmostEqual(window['rms'], amplitude / math.sqrt(2), places=7)
            self.assertAlmostEqual(window['peak'], amplitude, places=7)
        # Existing long-window RMS still represents the complete signal.
        self.assertAlmostEqual(result['chunks'][0]['rms'],
                               math.sqrt(sum(a*a/2 for a in amplitudes)/len(amplitudes)), places=7)

    def test_antiphase_stereo_keeps_same_levels_as_mono(self):
        wave = .4 * np.sin(2 * np.pi * 500 * np.arange(2563) / self.rate)
        mono = stem_audio.analyze_audio(self.write(wave, 'mono.wav'), bpm=120)
        stereo = stem_audio.analyze_audio(self.write(np.column_stack((wave, -wave)), 'stereo.wav'), bpm=120)
        for left, right in zip(mono['level_windows'], stereo['level_windows']):
            self.assertAlmostEqual(left['rms'], right['rms'], places=12)
            self.assertEqual(left['peak'], right['peak'])
        self.assertEqual(stereo['level_windows'][-1]['end_seconds'], len(wave)/self.rate)

    def test_rms_averages_channel_power_and_peak_uses_loudest_channel(self):
        samples = np.tile(np.array([.25, -.5, 0, .75], dtype=np.float32), (800, 1))
        window = self.analyze(samples)['level_windows'][0]
        self.assertAlmostEqual(window['rms'], math.sqrt((.25**2+.5**2+.75**2)/4), places=12)
        self.assertEqual(window['peak'], .75)

    def test_silence_and_short_partial_window_are_not_omitted_or_padded(self):
        samples = np.zeros((1613, 2), dtype=np.float32)
        samples[-13:, 0] = .5
        result = self.analyze(samples)
        windows = result['level_windows']
        self.assertEqual([(w['start_seconds'], w['end_seconds']) for w in windows],
                         [(0, .1), (.1, .2), (.2, 1613/self.rate)])
        self.assertEqual([(w['rms'], w['peak']) for w in windows[:2]], [(0, 0), (0, 0)])
        self.assertAlmostEqual(windows[-1]['rms'], .5/math.sqrt(2), places=12)
        self.assertEqual(windows[-1]['peak'], .5)
        tiny = self.analyze(np.zeros(13))['level_windows']
        self.assertEqual(tiny, [dict(start_seconds=0, end_seconds=13/self.rate, rms=0, peak=0)])

    def test_quiet_decay_tail_remains_measured_below_activity_gate(self):
        times = np.arange(17864) / self.rate
        samples = np.zeros(len(times), dtype=np.float32)
        active = times >= .2
        samples[active] = .5 * np.exp(-8*(times[active]-.2))
        result = self.analyze(samples)
        windows = result['level_windows']
        self.assertEqual(len(windows), 23)
        self.assertEqual(windows[-1]['end_seconds'], len(samples)/self.rate)
        self.assertGreater(windows[-1]['rms'], 0)
        self.assertLess(windows[-1]['rms'], 1e-5)
        self.assertGreater(windows[-1]['start_seconds'], result['activity_segments'][-1]['end_seconds'])
        for before, after in zip(windows[2:-1], windows[3:]):
            self.assertGreater(before['rms'], after['rms'])
            self.assertGreater(before['peak'], after['peak'])
        tail = samples[17600:].astype(np.float64)
        self.assertAlmostEqual(windows[-1]['rms'], math.sqrt(float(np.mean(tail**2))), places=14)

    def test_fixed_sample_bins_cover_odd_sample_rates_without_accumulating_gaps(self):
        path = self.write(np.full(2205, .25), rate=11025)
        result = stem_audio.analyze_audio(path, bpm=120)
        self.assertEqual([(round(w['start_seconds']*11025), round(w['end_seconds']*11025))
                          for w in result['level_windows']], [(0, 1102), (1102, 2204), (2204, 2205)])
        self.assertEqual(result['duration_seconds'], .2)
        self.assertTrue(all(w['rms'] == w['peak'] == .25 for w in result['level_windows']))

    def test_float_audio_above_full_scale_is_measured_without_clipping(self):
        result = self.analyze(np.tile([1.5, -2.0], 400))
        window = result['level_windows'][0]
        self.assertEqual(window['peak'], 2)
        self.assertAlmostEqual(window['rms'], math.sqrt((1.5**2+2**2)/2), places=12)
        self.assertTrue(stem_audio._cache_valid(result, result['cache_key']))

    def test_one_streaming_decode_preserves_bins_across_arbitrary_read_boundaries(self):
        samples = np.random.default_rng(117).uniform(-.7, .7, size=(18769, 2)).astype(np.float32)
        path = self.write(samples)
        reference = stem_audio.analyze_audio(path, bpm=120)
        original = sf.SoundFile
        calls = []
        opened = []

        class Reader:
            def __init__(self, name):
                opened.append(name)
                self.file = original(name)
            def __getattr__(self, name):
                return getattr(self.file, name)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.file.close()
            def read(self, frames=-1, **kwargs):
                calls.append(frames)
                if frames != 257:
                    raise AssertionError('Expected bounded reads from one decoder.')
                return self.file.read(frames, **kwargs)

        with mock.patch.object(sf, 'SoundFile', Reader), mock.patch.object(stem_audio, '_BLOCK_FRAMES', 257):
            actual = stem_audio.analyze_audio(path, bpm=120)
        self.assertEqual(len(opened), 1)
        self.assertEqual(len(calls), math.ceil(len(samples)/257)+1)
        for left, right in zip(actual['level_windows'], reference['level_windows']):
            for key in ('start_seconds', 'end_seconds', 'rms', 'peak'):
                self.assertAlmostEqual(left[key], right[key], places=12)
        self.assertEqual(len(actual['level_windows']), len(reference['level_windows']))
        for left, right in zip(actual['chunks'], reference['chunks']):
            for key in left:
                self.assertAlmostEqual(left[key], right[key], places=12)

    def test_cache_reuse_returns_complete_windows_without_decoding_again(self):
        path = self.write(np.tile([.25, -.25], 1213))
        cache = self.root / 'cache'
        first = stem_audio.analyze_audio(path, bpm=120, cache_dir=cache)
        with mock.patch.object(stem_audio, '_audio_modules', side_effect=AssertionError('Cache must bypass decoding')):
            second = stem_audio.analyze_audio(path, bpm=120, cache_dir=cache)
        self.assertTrue(second['cache_hit'])
        self.assertEqual(first['level_windows'], second['level_windows'])
        json.dumps(second, allow_nan=False)

    def test_old_or_corrupt_level_caches_are_recomputed(self):
        path = self.write(np.tile([.25, -.25], 1013))
        cache = self.root / 'cache'
        original = stem_audio.analyze_audio(path, bpm=120, cache_dir=cache)
        saved = cache / (original['cache_key']+'.json')
        changes = {
            'old_version': lambda d: d.update(analysis_version=2),
            'missing_levels': lambda d: d.pop('level_windows'),
            'empty_levels': lambda d: d.update(level_windows=[]),
            'missing_tail': lambda d: d['level_windows'].pop(),
            'missing_peak': lambda d: d['level_windows'][0].pop('peak'),
            'nonfinite_peak': lambda d: d['level_windows'][0].update(peak=float('nan')),
            'nonfinite_rms': lambda d: d['level_windows'][0].update(rms=float('inf')),
            'negative_rms': lambda d: d['level_windows'][0].update(rms=-.1),
            'negative_peak': lambda d: d['level_windows'][0].update(peak=-.1),
            'boolean_peak': lambda d: d['level_windows'][0].update(peak=True),
            'oversized_integer_peak': lambda d: d['level_windows'][0].update(peak=10**400),
            'peak_below_rms': lambda d: d['level_windows'][0].update(peak=.1),
            'impossible_pcm_peak': lambda d: d['level_windows'][0].update(peak=1e100),
            'gap': lambda d: d['level_windows'][1].update(start_seconds=.11),
            'overlap': lambda d: d['level_windows'][1].update(start_seconds=.09),
            'nonzero_origin': lambda d: d['level_windows'][0].update(start_seconds=.01),
            'wrong_tail': lambda d: d['level_windows'][-1].update(end_seconds=.3),
            'zero_length': lambda d: d['level_windows'][1].update(end_seconds=.1),
            'shifted_boundary': lambda d: (d['level_windows'][0].update(end_seconds=.11),
                                           d['level_windows'][1].update(start_seconds=.11)),
        }
        for name, change in changes.items():
            with self.subTest(corruption=name):
                damaged = deepcopy(original)
                change(damaged)
                saved.write_text(json.dumps(damaged), encoding='utf-8')
                result = stem_audio.analyze_audio(path, bpm=120, cache_dir=cache)
                self.assertFalse(result['cache_hit'])
                self.assertEqual(result['level_windows'], original['level_windows'])


if __name__ == '__main__':
    unittest.main()
