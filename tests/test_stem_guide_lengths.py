"""Guide endings may differ; actual instrument notes still need alignment."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'work'))
from flp_song_analysis import analyze_flp
from stem_midi import read_source_timing
from test_stem_layer import layer_fixture, NAMES, PPQ, END


class SourceGuideLengthTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'eight drum guide.flp'
        self.original = layer_fixture(self.path)
        self.stems = [dict(id=30+i, name='1_'+name) for i, name in enumerate(NAMES)]
        self.model = analyze_flp(self.path, max_instruments=128, allow_single_pattern=True)

    def read(self, *, end=END, ppq=PPQ, model=None, bpm=170, stems=None):
        arguments = dict(ppq=ppq, bpm=bpm, end_tick=end)
        if model is None:
            result = read_source_timing(self.path, stems or self.stems, **arguments)
        else:
            before = deepcopy(model)
            with patch('flp_song_analysis.analyze_flp', return_value=model):
                result = read_source_timing(self.path, stems or self.stems, **arguments)
            self.assertEqual(model, before, 'Reading a guide must not edit the resolved source notes.')
        self.assertEqual(self.path.read_bytes(), self.original)
        return result

    def assert_length_warning(self, guide):
        text = ' '.join(guide['warnings']).lower()
        self.assertIn('source', text)
        self.assertTrue(any(word in text for word in ('end', 'length', 'duration')), text)
        self.assertTrue(any(word in text for word in ('stem', 'audio')), text)
        self.assertTrue(any(word in text for word in ('unchanged', 'preserv', 'retain', 'not shift')), text)

    def test_longer_rendered_stem_tail_does_not_shift_or_stretch_layer_notes(self):
        normal = self.read()
        longer_stems = self.read(end=END+1540)
        self.assertEqual(longer_stems['matches'], normal['matches'])
        self.assertEqual(len(longer_stems['matches']), 8)
        self.assert_length_warning(longer_stems)

    def test_longer_source_playlist_padding_does_not_block_shorter_stems(self):
        # The real fixture ends at eight bars, but its last hit ends before four.
        # Its extra empty pattern duration is not extra musical content.
        normal = self.read()
        shorter_stems = self.read(end=END-1540)
        for identifier in normal['matches']:
            for field in ('note_events', 'note_spans', 'phrase_ends', 'note_velocity_max'):
                self.assertEqual(shorter_stems['matches'][identifier][field], normal['matches'][identifier][field])
        self.assert_length_warning(shorter_stems)

    def test_unrelated_source_audio_tail_does_not_define_note_alignment(self):
        normal = self.read()
        model = deepcopy(self.model)
        model['end_tick'] = END+384*64
        model['audio_clips'] = [dict(name='Vocal delay tail', position=END-96,
                                     duration=384*64+96, channel_id=40)]
        guide = self.read(model=model)
        self.assertEqual(guide['matches'], normal['matches'])
        self.assert_length_warning(guide)

    def test_longer_source_duration_scales_ppq_without_normalizing_notes_to_fit(self):
        guide = self.read(ppq=PPQ*2, end=(END-1540)*2)
        for index, stem in enumerate(self.stems):
            match = guide['matches'][stem['id']]
            self.assertEqual(match['note_events'], [
                dict(tick=index*PPQ*2, duration=48, note=60+index, velocity=100),
                dict(tick=(8+index)*PPQ*2, duration=48, note=60+index, velocity=100),
            ])
            self.assertEqual(match['note_velocity_max'], 128)
        self.assert_length_warning(guide)

    def test_exact_one_project_beat_tolerance_retains_complete_note(self):
        model = deepcopy(self.model)
        note = dict(tick=END-24, duration=PPQ+24, note=60, velocity=73)
        model['voices'][0]['notes'].append(note)
        model['end_tick'] = END+PPQ+1000
        guide = self.read(model=model, ppq=PPQ*2, end=END*2)
        expected = dict(tick=(END-24)*2, duration=(PPQ+24)*2, note=60, velocity=73)
        match = guide['matches'][30]
        self.assertIn(expected, match['note_events'])
        self.assertIn((expected['tick'], (END+PPQ)*2), match['note_spans'])
        self.assertTrue(all(tick < END*2 for tick in match['phrase_ends']))
        self.assert_length_warning(guide)

    def test_note_ending_one_tick_beyond_allowed_beat_is_rejected(self):
        model = deepcopy(self.model)
        model['voices'][0]['notes'].append(dict(tick=END-24, duration=PPQ+25, note=60, velocity=100))
        model['end_tick'] = END+PPQ+1000
        with self.assertRaisesRegex(ValueError, '(?i)note|musical|instrument'):
            self.read(model=model)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_zero_duration_trigger_after_allowed_ending_is_rejected(self):
        model = deepcopy(self.model)
        model['voices'][0]['notes'].append(dict(tick=END+PPQ+1, duration=0, note=60, velocity=100))
        model['end_tick'] = END+PPQ+1000
        with self.assertRaisesRegex(ValueError, '(?i)note|musical|instrument'):
            self.read(model=model)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_unmatched_late_instrument_is_not_hidden_by_successful_drum_matches(self):
        model = deepcopy(self.model)
        model['voices'].append(dict(channel_id=70, name='Unmatched lead',
            notes=[dict(tick=END+PPQ+1, duration=48, note=72, velocity=95)]))
        model['end_tick'] = END+PPQ+1000
        with self.assertRaisesRegex(ValueError, '(?i)note|musical|instrument'):
            self.read(model=model)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_markers_are_scaled_then_limited_to_the_audio_arrangement(self):
        model = deepcopy(self.model)
        model['end_tick'] = END+1540
        model['markers'] = [dict(tick=tick, name=name, source='playlist') for tick, name in (
            (-1, 'Before start'), (0, 'Intro'), (END-1, 'Last in-range marker'),
            (END, 'Audio end'), (END+100, 'Only in source tail'))]
        guide = self.read(model=model, ppq=PPQ*2, end=END*2)
        self.assertEqual(guide['markers'], [
            dict(tick=0, name='Intro', source='playlist'),
            dict(tick=(END-1)*2, name='Last in-range marker', source='playlist'),
        ])

    def test_equal_arrangement_with_fitting_notes_has_no_length_warning(self):
        guide = self.read()
        self.assertEqual(guide['warnings'], [])
        self.assertEqual(len(guide['matches']), 8)

    def test_different_tempo_is_still_rejected_despite_only_extra_tail(self):
        with self.assertRaisesRegex(ValueError, '(?i)tempo|meter'):
            self.read(end=END+1540, bpm=160)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_different_meter_is_still_rejected_despite_only_extra_tail(self):
        model = deepcopy(self.model)
        model['time_signature'] = [3, 4]
        model['end_tick'] = END+1540
        with self.assertRaisesRegex(ValueError, '(?i)tempo|meter'):
            self.read(model=model)
        self.assertEqual(self.path.read_bytes(), self.original)


if __name__ == '__main__':
    unittest.main()
