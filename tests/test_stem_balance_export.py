"""Measured audio + optional MIDI -> separate native volume clips, losslessly."""
from copy import deepcopy
import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import mido
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'work'))
sys.path.insert(0, str(ROOT / 'tests'))
from test_stem_project import audio_fixture, PPQ, END
from flp_raw import read_bytes, encode_fl
from flp_song_analysis import read_project
from stem_native import parse_points
from stem_planner import load_presets, plan_moves
from stem_project import inspect_stems, analyze_stems, preview_automation, export_automation


class BalanceExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.source = self.folder / 'stems.flp'
        self.original = audio_fixture(self.source)
        duration = END / PPQ * 60 / 170
        rate = 8000
        t = np.arange(round(duration * rate)) / rate
        # Abrupt recorded excesses mixed with intentional sections and pauses.
        amplitude = np.where((t % 4 > 1) & (t % 4 < 2), .7, .16)
        amplitude[(t > 8) & (t < 10)] = 0
        for index, name in enumerate(('1_808 Snare', '1_808 Tom')):
            audio = amplitude * np.sin(2 * np.pi * (200 + index * 70) * t)
            sf.write(self.folder / (name + '.wav'), audio, rate, subtype='FLOAT')
        self.midi = self.folder / 'song.mid'
        midi = mido.MidiFile(ticks_per_beat=PPQ)
        for name in ('808 Snare', '808 Tom'):
            track = mido.MidiTrack()
            track.extend([mido.MetaMessage('track_name', name=name),
                          mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(170))])
            for index in range(END // PPQ):
                track.append(mido.Message('note_on', note=60, velocity=100 if index % 2 == 0 else 70, time=0))
                track.append(mido.Message('note_off', note=60, time=PPQ))
            midi.tracks.append(track)
        midi.save(self.midi)
        self.midi_before = self.midi.read_bytes()
        self.audio_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.folder.glob('*.wav')}

    def analyze(self, use_midi=False):
        return analyze_stems(inspect_stems(self.source),
                             midi_path=self.midi if use_midi else None,
                             cache_dir=self.folder / 'cache')

    def preview(self, analysis):
        return preview_automation(analysis, 'Hip-Hop', 'Drill', movement_mode='volume_balance')

    def test_audio_only_and_matching_midi_both_make_volume_only_native_projects(self):
        for use_midi in (False, True):
            with self.subTest(midi=use_midi):
                analysis = self.analyze(use_midi)
                snapshot = deepcopy(analysis)
                self.assertTrue(all(s['analysis']['level_windows'] for s in analysis['stems']))
                preview = self.preview(analysis)
                self.assertEqual(analysis, snapshot)
                self.assertEqual(preview['movement_mode'], 'volume_balance')
                self.assertEqual(preview['effects'], [])
                self.assertEqual(len(preview['lanes']), 2)
                self.assertEqual(preview['balance_coverage']['stems_ready'], 2)
                self.assertTrue(all(len(l['targets']) == 1 and l['parameter'] == 'volume_db' for l in preview['lanes']))
                self.assertTrue(preview['output'].endswith('_volume_balance_preview.flp'))
                if use_midi:
                    self.assertEqual(len(analysis['midi_guide']['matches']), 2)
                    self.assertTrue(any(d.get('guide_used') for d in preview['balance_details']))
                    self.assertIn(str(self.midi.resolve()), [i['path'] for i in preview['input_files']])
                output = self.folder / f'balanced-{use_midi}.flp'
                export_automation(preview, output)
                before, after = read_project(self.source), read_project(output)
                for key in (213, 223, 224, 225, 235):
                    self.assertEqual([p for k,p in before['events'] if k == key],
                                     [p for k,p in after['events'] if k == key])
                for channel in before['channels']:
                    new = next(c for c in after['channels'] if c['id'] == channel['id'])
                    self.assertEqual(before['events'][channel['start']:channel['stop']],
                                     after['events'][new['start']:new['stop']])
                old_playlist = next(p for k,p in before['events'] if k == 233)
                self.assertTrue(next(p for k,p in after['events'] if k == 233).startswith(old_playlist))
                channels = [c for c in after['channels'] if c['type'] == 5]
                self.assertEqual(len(channels), 2)
                for channel, lane in zip(channels, preview['lanes']):
                    payload = dict(after['events'][channel['start']:channel['stop']])[234]
                    points = parse_points(payload, PPQ)
                    self.assertEqual(len(points), len(lane['points']))
                    for actual, expected in zip(points, lane['points']):
                        self.assertAlmostEqual(actual[0], expected[0], places=6)
                        self.assertAlmostEqual(actual[1], expected[1], places=9)
                    clips = [c for c in after['clips'] if c.get('channel_id') == channel['id']]
                    self.assertEqual([(c['position'], c['duration']) for c in clips], [(0, END)])
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(self.midi.read_bytes(), self.midi_before)
        self.assertEqual(self.audio_hashes, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.audio_hashes})

    def test_genre_selection_does_not_change_audio_balance(self):
        analysis = self.analyze()
        a = self.preview(analysis)
        b = preview_automation(analysis, 'Rock', 'Alt Rock', movement_mode='volume_balance')
        self.assertEqual(a['lanes'], b['lanes'])
        self.assertEqual(a['balance_details'], b['balance_details'])

    def test_unsafe_native_target_is_skipped_and_not_reported_ready(self):
        header, events = read_bytes(self.original)
        # Existing native ownership must win; a separate existing link record
        # suffices to exercise the writer's owner guard without any audio edit.
        at = next(i for i,(k,_) in enumerate(events) if k == 64)
        events.insert(at, (227, struct.pack('<5I', 0, 0, 0x70401fc0, 8, 469)))
        self.source.write_bytes(encode_fl(header, events))
        preview = self.preview(self.analyze())
        statuses = {d['stem_id']: d['status'] for d in preview['balance_details']}
        self.assertEqual(statuses, {0: 'skipped', 1: 'ready'})
        self.assertEqual(preview['balance_coverage']['stems_skipped'], 1)
        self.assertEqual(preview['balance_coverage']['stems_ready'], 1)
        self.assertTrue(any('already has automation' in s['reason'] for s in preview['skipped']))

    def test_missing_level_measurements_are_explicit_not_template_fallback(self):
        analysis = self.analyze()
        for stem in analysis['stems']:
            stem['analysis'].pop('level_windows')
        preview = self.preview(analysis)
        self.assertFalse(preview['lanes'])
        self.assertEqual(preview['balance_coverage']['stems_without_audio'], 2)
        with self.assertRaisesRegex(ValueError, 'No supported native automation'):
            export_automation(preview, self.folder / 'must-not-exist.flp')


if __name__ == '__main__':
    unittest.main()
