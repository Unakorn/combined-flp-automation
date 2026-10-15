"""Eight real audio stems and named MIDI guides become native mixer curves."""
from collections import Counter
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
from flp_raw import encode_fl
from flp_song_analysis import read_project
from stem_native import build_channel, build_link, build_playlist_clip, parse_points
from stem_project import analyze_stems, export_automation, inspect_stems, preview_automation, relative_volume

PPQ = 96
BPM = 120
END = PPQ * 4 * 8
ROLES = ('kick', 'snare', 'clap', 'tom', 'hat', 'perc', 'ride', 'crash')
NAMES = ('808 Kick', '808 Snare', '808 Clap', '808 Tom', '808 CH', 'Perc Rim', '808 Ride', '808 Crash')
HITS = tuple(PPQ + bar * PPQ * 4 for bar in range(8))
VELOCITIES = (127, 48, 110, 64, 100, 52, 120, 80)
SAVED_FADERS = tuple(10000 + index * 300 for index in range(8))


def _text(value):
    return (value + '\0').encode('utf-16-le')


def _volume_target(index):
    return 0x70000000 | ((index + 1) << 22) | 0x1fc0


def eight_stem_fixture(path, *, owned_index=None):
    """Minimal FL26 audio arrangement with eight independent saved faders."""
    events = [(199, b'26.1.7.5653\0'), (156, struct.pack('<I', BPM * 1000)),
              (17, b'\4'), (18, b'\4'), (231, _text('All'))]
    samples = []
    rate = 8000
    duration = END / PPQ * 60 / BPM
    for index, name in enumerate(NAMES):
        sample = path.parent / (name + '.wav')
        audio = np.zeros(round(rate * duration), dtype=np.float32)
        tail = np.arange(round(rate * 0.3)) / rate
        pulse = (0.3 * np.sin(2 * np.pi * (90 + index * 180) * tail) * np.exp(-tail * 12)).astype(np.float32)
        for tick, velocity in zip(HITS, VELOCITIES):
            start = round(tick / PPQ * 60 / BPM * rate)
            audio[start:start + len(pulse)] += pulse * velocity / 127
        sf.write(sample, audio, rate, subtype='FLOAT')
        samples.append(sample)
        events += [(64, struct.pack('<H', index)), (21, b'\4'), (0, b'\1'),
                   (203, _text(name)), (104, struct.pack('<H', index + 1)),
                   (196, _text(str(sample))), (213, b'original synthetic audio state')]
    placements = [build_playlist_clip(cid, 0, END, cid, cid + 1) for cid in range(8)]
    channel_count = 8
    if owned_index is not None:
        events.extend(build_channel(8, 'Saved volume automation',
                                    [(0, SAVED_FADERS[owned_index] / 16000),
                                     (END, SAVED_FADERS[owned_index] / 16000)], PPQ))
        first = next(index for index, (key, _) in enumerate(events) if key == 64)
        events.insert(first, build_link(8, _volume_target(owned_index)))
        placements.append(build_playlist_clip(8, 0, END, 8, 9, offset_tick=0, ppq=PPQ))
        channel_count += 1
    events += [(99, b'\0\0'), (233, b''.join(placements))]
    for number in range(1, 65):
        track = bytearray(66)
        struct.pack_into('<I', track, 0, number)
        track[12] = int(number <= channel_count)
        events.append((238, bytes(track)))
    events.append((103, struct.pack('<H', 10)))
    initial = []
    for insert in range(10):
        prefix = 0x70000000 | (insert << 22)
        events += [(235, b'\1'), (154, b'\xff' * 4), (42, b'\0'),
                   (236, bytes.fromhex('000000004c00000000000000'))]
        saved = SAVED_FADERS[insert - 1] if 1 <= insert <= 8 else 12800
        initial += [(0, prefix | 0x1fc0, saved), (0, prefix | 0x1fc1, 0)]
        for slot in range(10):
            events.append((98, struct.pack('<H', slot)))
        events.append((147, b'\xff' * 4))
    events.append((225, b''.join(struct.pack('<III', *row) for row in initial)))
    original = encode_fl(struct.pack('<HHH', 0, channel_count, PPQ), events)
    path.write_bytes(original)
    return original, samples


def named_midi_fixture(path, *, empty_index=None, combined=False):
    midi = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(BPM)))
    conductor.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))
    midi.tracks.append(conductor)
    for index, name in enumerate(('DRUMS',) if combined else NAMES):
        track = mido.MidiTrack()
        track.append(mido.MetaMessage('track_name', name=name))
        previous = 0
        if index != empty_index:
            for tick, velocity in zip(HITS, VELOCITIES):
                track.append(mido.Message('note_on', note=36 + index, velocity=velocity,
                                          channel=9, time=tick - previous))
                track.append(mido.Message('note_off', note=36 + index, velocity=0, channel=9, time=24))
                previous = tick + 24
        else:
            # A velocity-zero note-on is a release, never a musical hit.
            track.append(mido.Message('note_on', note=36 + index, velocity=0, channel=9, time=PPQ))
        midi.tracks.append(track)
    midi.save(path)
    return path.read_bytes()


class EightDrumNativeExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.source = self.folder / 'eight drums.flp'
        self.midi = self.folder / 'named drum guide.mid'
        self.output = self.folder / 'eight drums automated.flp'

    def analyze(self, *, empty_index=None, combined=False, owned_index=None):
        self.original, self.samples = eight_stem_fixture(self.source, owned_index=owned_index)
        self.midi_original = named_midi_fixture(self.midi, empty_index=empty_index, combined=combined)
        self.audio_original = {path: path.read_bytes() for path in self.samples}
        return analyze_stems(inspect_stems(self.source), roles=dict(enumerate(ROLES)),
                             midi_path=self.midi, cache_dir=self.folder / 'cache')

    def preview(self, analysis):
        return preview_automation(analysis, 'Hip-Hop', 'Drill', movement_mode='full_song')

    def assert_inputs_preserved(self):
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(self.midi.read_bytes(), self.midi_original)
        for path, original in self.audio_original.items():
            self.assertEqual(path.read_bytes(), original)

    @staticmethod
    def volume_targets(preview):
        return {target['stem_id']: (lane, target) for lane in preview['lanes']
                for target in lane['targets'] if target['parameter'] == 'volume_db'}

    def assert_drum_coverage(self, preview, *, ready, hits, unmatched=0, skipped=0):
        self.assertEqual(preview['drum_coverage'], dict(
            drums_total=8, drums_ready=ready, hits_guided=hits,
            drums_unmatched=unmatched, drums_skipped=skipped,
            drums_silent=0, drums_disabled=0))

    def test_all_eight_named_guides_export_native_volume_without_note_changes(self):
        analysis = self.analyze()
        for index, stem in enumerate(analysis['stems']):
            self.assertEqual(stem['note_velocity_max'], 127)
            self.assertEqual([note['velocity'] for note in stem['note_events']], list(VELOCITIES))
            self.assertEqual([note['tick'] for note in stem['note_events']], list(HITS))
            self.assertIn(NAMES[index], stem['note_guide_source'])
        preview = self.preview(analysis)
        volumes = self.volume_targets(preview)
        self.assertEqual(set(volumes), set(range(8)))
        self.assertFalse(any('overlapping' in item['reason'] for item in preview['skipped']))
        dynamics = {row['stem_id']: row for row in preview['drum_dynamics']}
        self.assertEqual(set(dynamics), set(range(8)))
        self.assert_drum_coverage(preview, ready=8, hits=64)
        for index, (lane, target) in volumes.items():
            self.assertEqual(dynamics[index]['status'], 'ready')
            self.assertEqual(dynamics[index]['hit_count'], len(HITS))
            self.assertEqual(target['target'], _volume_target(index))
            self.assertEqual(target['baseline'], SAVED_FADERS[index] / 16000)
            self.assertEqual(lane['points'][0][0], 0)
            self.assertEqual(lane['points'][-1], (END, SAVED_FADERS[index] / 16000))
            self.assertTrue(all(0 <= value <= target['baseline'] for _, value in lane['points']))
            self.assertGreater(max(v for _, v in lane['points']) - min(v for _, v in lane['points']), 1e-4)
            self.assertAlmostEqual(min(value for _, value in lane['points']),
                                   relative_volume(SAVED_FADERS[index], dynamics[index]['minimum_db']), places=8)
            moves = [move for move in lane['moves'] if move['stem_id'] == index]
            self.assertEqual(len(moves), 1)
            self.assertTrue(moves[0]['drum_dynamics'])
        kick = volumes[0][0]
        self.assertGreaterEqual(min(value for _, value in kick['points']), relative_volume(SAVED_FADERS[0], -1.5) - 1e-9)
        self.assertGreaterEqual(dynamics[0]['minimum_db'], -1.5)
        self.assertLessEqual(dynamics[0]['maximum_db'], 0)

        report = export_automation(preview, self.output)
        before, after = read_project(self.source), read_project(self.output)
        channels = {channel['id']: channel for channel in after['channels'] if channel['type'] == 5}
        links = [struct.unpack('<5I', payload) for key, payload in after['events'] if key == 227]
        owners = Counter(row[2] for row in links)
        self.assertTrue(all(count == 1 for count in owners.values()))
        for index, (lane, target) in volumes.items():
            matching = [row for row in links if row[2] == _volume_target(index)]
            self.assertEqual(len(matching), 1)
            cid = matching[0][0] >> 16
            self.assertEqual(matching[0][1:], (0, target['target'], 8, 469))
            channel = channels[cid]
            fields = dict(after['events'][channel['start']:channel['stop']])
            decoded = parse_points(fields[234], PPQ)
            self.assertEqual(len(decoded), len(lane['points']))
            for (actual_tick, actual_value), (tick, value) in zip(decoded, lane['points']):
                # Native point offsets use floating-point beats; accumulated
                # fractional-beat offsets need only sub-tick roundoff tolerance.
                self.assertAlmostEqual(actual_tick, tick, places=7)
                self.assertAlmostEqual(actual_value, value, places=12)
            clips = [clip for clip in after['clips'] if clip['kind'] == 'channel' and clip['channel_id'] == cid]
            self.assertEqual([(clip['position'], clip['duration']) for clip in clips], [(0, END)])
        initial = {target: value for key, payload in after['events'] if key == 225
                   for _, target, value in struct.iter_unpack('<III', payload)}
        for index, saved in enumerate(SAVED_FADERS):
            self.assertEqual(initial[_volume_target(index)], saved)
        self.assertEqual([event for event in after['events'] if event[0] in (223, 224)],
                         [event for event in before['events'] if event[0] in (223, 224)])
        self.assertFalse(any(channel['plugin'] == 'Fruity Keyboard Controller' for channel in after['channels']))
        self.assertEqual(report['automation_links'], len(links))
        self.assertEqual(report['automation_channels'], len(channels))
        self.assertEqual(report['drum_dynamics'], preview['drum_dynamics'])
        self.assertEqual(report['drum_coverage'], preview['drum_coverage'])
        self.assertFalse(report['validation']['native_fl_playback_tested'])
        self.assert_inputs_preserved()

    def test_empty_named_track_is_zero_hits_and_never_claimed_ready(self):
        analysis = self.analyze(empty_index=7)
        preview = self.preview(analysis)
        rows = {row['stem_id']: row for row in preview['drum_dynamics']}
        self.assertEqual(rows[7]['hit_count'], 0)
        self.assertNotEqual(rows[7]['status'], 'ready')
        self.assertTrue(rows[7]['reason'])
        self.assertEqual(sum(row['status'] == 'ready' for row in rows.values()), 7)
        self.assertEqual(sum(row['hit_count'] for row in rows.values()), 7 * len(HITS))
        self.assert_drum_coverage(preview, ready=7, hits=56, unmatched=1)
        report = export_automation(preview, self.output)
        self.assertEqual(report['drum_dynamics'], preview['drum_dynamics'])
        self.assert_inputs_preserved()

    def test_combined_drums_guide_does_not_claim_eight_individual_matches(self):
        analysis = self.analyze(combined=True)
        preview = self.preview(analysis)
        self.assertEqual(len(preview['drum_dynamics']), 8)
        self.assertTrue(all(row['status'] != 'ready' and row['hit_count'] == 0
                            for row in preview['drum_dynamics']))
        self.assertTrue(all(row['reason'] for row in preview['drum_dynamics']))
        self.assert_drum_coverage(preview, ready=0, hits=0, unmatched=8)
        self.assertTrue(any('drum' in warning.lower() and ('match' in warning.lower() or 'guide' in warning.lower())
                            for warning in preview['warnings']))
        self.assert_inputs_preserved()

    def test_existing_native_volume_owner_is_preserved_and_excluded_from_ready_count(self):
        analysis = self.analyze(owned_index=1)
        preview = self.preview(analysis)
        rows = {row['stem_id']: row for row in preview['drum_dynamics']}
        self.assertNotEqual(rows[1]['status'], 'ready')
        self.assertEqual(rows[1]['hit_count'], len(HITS))
        self.assertTrue(rows[1]['reason'])
        self.assertEqual(sum(row['status'] == 'ready' for row in rows.values()), 7)
        self.assertEqual(rows[1]['status'], 'skipped')
        self.assert_drum_coverage(preview, ready=7, hits=56, skipped=1)
        self.assertNotIn(1, self.volume_targets(preview))
        self.assertTrue(any(target['stem_id'] == 1 and target['parameter'] != 'volume_db'
                            for lane in preview['lanes'] for target in lane['targets']))
        report = export_automation(preview, self.output)
        before, after = read_project(self.source), read_project(self.output)
        old_links = [payload for key, payload in before['events'] if key == 227]
        new_links = [payload for key, payload in after['events'] if key == 227]
        self.assertTrue(all(link in new_links for link in old_links))
        self.assertEqual(sum(struct.unpack_from('<I', payload, 8)[0] == _volume_target(1)
                             for payload in new_links), 1)
        self.assertEqual(report['automation_links'], len(new_links) - len(old_links))
        self.assert_inputs_preserved()


if __name__ == '__main__':
    unittest.main()
