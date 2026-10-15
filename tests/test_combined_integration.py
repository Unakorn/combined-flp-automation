"""Full public runner checks using generated tones, MIDI and synthetic FLPs."""
from pathlib import Path
import hashlib
import json
import shutil
import struct
import sys
import tempfile
import unittest
import uuid

import numpy as np
import soundfile as sf
from mido import Message, MetaMessage, MidiFile, MidiTrack

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'work')]
from combined_inputs import build_input_config
from combined_runner import run_combined
from flp_raw import encode_fl
from flp_song_analysis import read_project
from plugin_templates import plugin_records
from test_stem_project import audio_fixture, PPQ, END, text


def synthetic_stems(source, names=('PAD', 'ARP')):
    audio_fixture(source)
    project = read_project(source)
    events = list(project['events'])
    for index, (key, payload) in enumerate(events):
        if key == 156:
            events[index] = (key, struct.pack('<I', 120000))
    for channel, name in zip(project['channels'], names):
        for index in range(channel['start'], channel['stop']):
            if events[index][0] == 203:
                events[index] = (203, text(name))
    source.write_bytes(encode_fl(project['header'], events))
    sample_rate = 8000
    seconds = END / PPQ / 2
    timeline = np.arange(round(seconds * sample_rate)) / sample_rate
    # Three clearly different sections, with continuous supporting parts.
    amplitude = np.where(timeline < 8, .045, np.where(timeline < 16, .13,
                        np.where(timeline < 24, .36, .07)))
    for index, channel in enumerate(project['channels']):
        signal = amplitude * np.sin(2 * np.pi * (220 + 110 * index) * timeline)
        sf.write(channel['sample'], signal, sample_rate, subtype='PCM_16')


def synthetic_midi(path, names=('PAD', 'ARP')):
    midi = MidiFile(type=1, ticks_per_beat=PPQ)
    conductor = MidiTrack([MetaMessage('set_tempo', tempo=500000),
                           MetaMessage('time_signature', numerator=4, denominator=4)])
    last = 0
    for bar, label in ((0, 'Intro'), (4, 'Build'), (8, 'Drop'), (12, 'Break'), (16, 'Outro')):
        tick = bar * 4 * PPQ
        conductor.append(MetaMessage('marker', text=label, time=tick-last))
        last = tick
    midi.tracks.append(conductor)
    for index, name in enumerate(names):
        track = MidiTrack([MetaMessage('track_name', name=name)])
        rows = []
        for bar in range(17):
            hits = 4 if 8 <= bar < 12 else 1
            step = PPQ * 4 // hits
            for hit in range(hits):
                tick = bar * 4 * PPQ + hit * step
                duration = max(1, step - 6)
                rows += [(tick, Message('note_on', note=60 + index * 7, velocity=90)),
                         (tick + duration, Message('note_off', note=60 + index * 7))]
        last = 0
        for tick, message in sorted(rows, key=lambda row: row[0]):
            track.append(message.copy(time=tick-last))
            last = tick
        midi.tracks.append(track)
    midi.save(path)


class CombinedRunnerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.gettempdir()) / ('oa-combined-integration-' + uuid.uuid4().hex)
        self.folder.mkdir()
        self.addCleanup(shutil.rmtree, self.folder)
        self.source = self.folder / 'stems.flp'
        synthetic_stems(self.source)
        self.original = self.source.read_bytes()
        self.config = dict(source=str(self.source), output=str(self.folder / 'combined.flp'),
                           work_dir=str(self.folder / 'runs'), modules=dict(volume=True, section=False, eq=False))

    def verify_output(self, result):
        self.assertEqual(self.source.read_bytes(), self.original)
        before, after = read_project(self.source), read_project(result['output'])
        self.assertEqual(plugin_records(before), plugin_records(after))
        report = json.loads(Path(result['report_path']).read_text(encoding='utf-8'))
        self.assertEqual(report['output_sha256'], hashlib.sha256(Path(result['output']).read_bytes()).hexdigest())
        self.assertTrue(report['original_inputs_unchanged'])
        self.assertTrue(report['validation']['unique_new_parameter_owners'])
        self.assertTrue(report['validation']['original_notes_preserved'])
        return report, after

    def test_volume_only_runs_without_midi_or_effect_setup(self):
        result = run_combined(self.config)
        report, project = self.verify_output(result)
        self.assertEqual([stage['stage'] for stage in report['stages']], ['Volume'])
        self.assertEqual(report['validation']['automation_channels'], 2)
        targets = [struct.unpack_from('<I', payload, 8)[0] for key, payload in project['events'] if key == 227]
        self.assertEqual({target & 0xffff for target in targets}, {0x1fc0})
        self.assertFalse(report['input_matching']['section_matching_required'])
        with self.assertRaisesRegex(ValueError, 'new .flp'):
            run_combined(self.config)

    def test_section_pan_and_volume_run_together_without_plugin_setup(self):
        midi = self.folder / 'song.mid'
        synthetic_midi(midi)
        self.config.update(midi=str(midi), section_template='',
                           section_enabled=dict(filter=False, delay=False, width=False, pan=True, reverb=False))
        self.config['modules']['section'] = True
        result = run_combined(self.config)
        report, project = self.verify_output(result)
        self.assertEqual([stage['stage'] for stage in report['stages']], ['Section', 'Volume'])
        self.assertGreater(report['stages'][0]['channels'], 0)
        self.assertEqual(report['stages'][0]['effects'], [])
        self.assertEqual(report['stages'][1]['channels'], 2)
        targets = [struct.unpack_from('<I', payload, 8)[0] for key, payload in project['events'] if key == 227]
        self.assertEqual({target & 0xffff for target in targets}, {0x1fc0, 0x1fc1})
        self.assertEqual(len(targets), len(set(targets)))

    def test_shared_drum_mapping_preserves_whole_kit_ownership(self):
        synthetic_stems(self.source, names=('DRUMS', 'KICK'))
        midi = self.folder / 'kit.mid'
        synthetic_midi(midi, names=('DRUMS',))
        config = build_input_config(self.source, midi)
        self.assertEqual(config['midi_mapping'], {})
        self.assertEqual(config['drum_midi_guide'], {'track_index': 1})
        self.assertEqual(config['verification']['shared_drum_stem_ids'], [0, 1])
        self.assertEqual(config['protected_stems'], [0])
        self.assertEqual(config['roles'], {0: 'drums', 1: 'kick'})


if __name__ == '__main__':
    unittest.main()
