"""Note-level guides retain performance data without changing source files."""
from pathlib import Path
import sys
import tempfile
import unittest

from mido import Message, MetaMessage, MidiFile, MidiTrack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'work'))
from flp_raw import encode_fl, read_bytes
from stem_midi import read_source_timing, read_timing
from test_stem_layer import layer_fixture, NAMES, PPQ, END


class NoteGuideTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.stem = dict(id=17, name='1_808 Kick', role='kick')

    def midi(self, tracks, ppq=96):
        """Write tracks supplied as absolute-tick messages, preserving tie order."""
        path = self.folder / 'performance.mid'
        midi = MidiFile(type=1, ticks_per_beat=ppq)
        for name, events in tracks:
            track = MidiTrack([MetaMessage('track_name', name=name)])
            previous = 0
            for tick, message in sorted(events, key=lambda item: item[0]):
                track.append(message.copy(time=tick-previous))
                previous = tick
            midi.tracks.append(track)
        midi.save(path)
        return path

    def read(self, path, stems=None, ppq=96, end_tick=END, **kwargs):
        original = path.read_bytes()
        result = read_timing(path, stems or [self.stem], ppq=ppq,
                             bpm=170, end_tick=end_tick, **kwargs)
        self.assertEqual(path.read_bytes(), original)
        return result

    def test_midi_note_on_velocity_and_scaled_gate_survive_note_off(self):
        path = self.midi([('808 Kick', [
            (120, Message('note_on', note=36, velocity=43)),
            (360, Message('note_off', note=36, velocity=99)),
        ])], ppq=480)
        guide = self.read(path)['matches'][17]
        self.assertEqual(guide['note_events'], [dict(tick=24, duration=48, note=36, velocity=43)])
        self.assertEqual(guide['note_spans'], [[24, 72]])
        self.assertEqual(guide['phrase_ends'], [72])
        self.assertEqual(guide['note_velocity_max'], 127)
        self.assertEqual(guide['note_guide_source'], 'MIDI: performance.mid — 808 Kick')

    def test_simultaneous_pitches_channels_and_retriggers_pair_independently(self):
        path = self.midi([('808 Kick', [
            (0, Message('note_on', channel=0, note=36, velocity=40)),
            (0, Message('note_on', channel=1, note=36, velocity=80)),
            (0, Message('note_on', channel=0, note=38, velocity=70)),
            (5, Message('note_on', channel=0, note=36, velocity=110)),
            (8, Message('note_off', channel=0, note=38)),
            (10, Message('note_off', channel=0, note=36)),
            (15, Message('note_on', channel=1, note=36, velocity=0)),
            (20, Message('note_off', channel=0, note=36)),
        ])])
        guide = self.read(path)['matches'][17]
        self.assertCountEqual(guide['note_events'], [
            dict(tick=0, duration=10, note=36, velocity=40),
            dict(tick=0, duration=15, note=36, velocity=80),
            dict(tick=0, duration=8, note=38, velocity=70),
            dict(tick=5, duration=15, note=36, velocity=110),
        ])
        self.assertEqual(guide['phrase_ends'], [20])

    def test_zero_length_hit_matches_without_inventing_a_gate(self):
        path = self.midi([('808 Kick', [
            (12, Message('note_on', note=36, velocity=127)),
            (12, Message('note_off', note=36)),
        ])])
        guide = self.read(path)['matches'][17]
        self.assertEqual(guide['note_events'], [dict(tick=12, duration=0, note=36, velocity=127)])
        self.assertEqual(guide['note_spans'], [])
        self.assertEqual(guide['phrase_ends'], [])

    def test_sub_tick_hit_retains_velocity_when_rounding_collapses_gate(self):
        path = self.midi([('808 Kick', [
            (2, Message('note_on', note=36, velocity=61)),
            (3, Message('note_off', note=36)),
        ])], ppq=960)
        guide = self.read(path)['matches'][17]
        self.assertEqual(guide['note_events'], [dict(tick=0, duration=0, note=36, velocity=61)])
        self.assertEqual(guide['note_spans'], [])

    def test_zero_length_hit_past_arrangement_still_fails_alignment(self):
        path = self.midi([('808 Kick', [
            (END+PPQ+1, Message('note_on', note=36, velocity=100)),
            (END+PPQ+1, Message('note_off', note=36)),
        ])])
        with self.assertRaisesRegex(ValueError, 'extend beyond'):
            self.read(path)

    def test_unterminated_note_remains_an_error(self):
        path = self.midi([('808 Kick', [(0, Message('note_on', note=36, velocity=100))])])
        with self.assertRaisesRegex(ValueError, 'unterminated'):
            self.read(path)

    def test_combined_drums_are_not_guessed_from_midi_channel_or_pitch(self):
        path = self.midi([('DRUMS', [
            (0, Message('note_on', channel=9, note=36, velocity=100)),
            (24, Message('note_off', channel=9, note=36)),
            (96, Message('note_on', channel=9, note=38, velocity=70)),
            (120, Message('note_off', channel=9, note=38)),
        ])])
        stems = [self.stem, dict(id=18, name='1_808 Snare', role='snare')]
        guide = self.read(path, stems)
        self.assertEqual(guide['matches'], {})
        self.assertEqual(len(guide['warnings']), 2)
        for warning in guide['warnings']:
            self.assertIn('source instrument FLP', warning)
            self.assertIn('separately named MIDI track', warning)
            self.assertIn('custom drum slots', warning)

    def test_duplicate_normalized_track_names_remain_ambiguous(self):
        hits = [(0, Message('note_on', note=36, velocity=100)),
                (24, Message('note_off', note=36))]
        path = self.midi([('808 Kick', hits), ('1_808 Kick', hits)])
        guide = self.read(path)
        self.assertEqual(guide['matches'], {})
        self.assertIn('ambiguous', guide['warnings'][0])

    def test_explicit_named_track_mapping_still_works(self):
        path = self.midi([('My custom kick', [
            (0, Message('note_on', note=60, velocity=52)),
            (24, Message('note_off', note=60)),
        ])])
        guide = self.read(path, mapping={'17': 'My custom kick'})['matches'][17]
        self.assertEqual(guide['note_events'][0]['velocity'], 52)
        self.assertEqual(guide['midi_track'], 'My custom kick')

    def source(self):
        path = self.folder / 'eight drums.flp'
        header, events = read_bytes(layer_fixture(path))
        changed = []
        for kind, payload in events:
            if kind == 224:
                payload = bytearray(payload)
                for index, offset in enumerate(range(0, len(payload), 24)):
                    # Both accents and quiet hits differ for each Layer child.
                    payload[offset+21] = 32+index//2 if index % 2 == 0 else 128-index//2
                payload = bytes(payload)
            changed.append((kind, payload))
        path.write_bytes(encode_fl(header, changed))
        return path

    def test_eight_layer_children_keep_their_own_native_velocities_and_scaled_notes(self):
        path = self.source()
        original = path.read_bytes()
        stems = [dict(id=30+i, name='1_'+name) for i, name in enumerate(NAMES)]
        guides = read_source_timing(path, stems, ppq=PPQ*2, bpm=170, end_tick=END*2)
        self.assertEqual(len(guides['matches']), 8)
        self.assertEqual(guides['warnings'], [])
        for index, stem in enumerate(stems):
            guide = guides['matches'][stem['id']]
            self.assertEqual(guide['note_events'], [
                dict(tick=index*PPQ*2, duration=48, note=60+index, velocity=32+index),
                dict(tick=(8+index)*PPQ*2, duration=48, note=60+index, velocity=128-index),
            ])
            self.assertEqual(guide['note_velocity_max'], 128)
            self.assertEqual(guide['note_guide_source'], f'Source FLP: eight drums.flp — {NAMES[index]}')
            self.assertEqual(guide['source_track'], NAMES[index])
            self.assertEqual([list(span) for span in guide['note_spans']],
                             [[n['tick'], n['tick']+n['duration']] for n in guide['note_events']])
        self.assertEqual(path.read_bytes(), original)

    def test_source_override_replaces_all_note_performance_fields_together(self):
        midi = self.midi([('808 Kick', [
            (3, Message('note_on', note=36, velocity=90)),
            (7, Message('note_off', note=36)),
        ])])
        result = self.read(midi)['matches'][17]
        source = self.source()
        replacement = read_source_timing(source, [self.stem], ppq=PPQ, bpm=170, end_tick=END)['matches'][17]
        result.update(replacement)
        for field in ('note_events', 'note_velocity_max', 'note_guide_source', 'note_spans', 'phrase_ends'):
            self.assertEqual(result[field], replacement[field])
        self.assertEqual([n['velocity'] for n in result['note_events']], [32, 128])
        self.assertEqual({n['note'] for n in result['note_events']}, {60})

    def test_unmatched_source_drum_gets_actionable_warning_without_a_guide(self):
        source = self.source()
        guide = read_source_timing(source, [dict(id=10, name='Other snare', role='snare')],
                                   ppq=PPQ, bpm=170, end_tick=END)
        self.assertEqual(guide['matches'], {})
        self.assertIn('source instrument FLP', guide['warnings'][0])


if __name__ == '__main__':
    unittest.main()
