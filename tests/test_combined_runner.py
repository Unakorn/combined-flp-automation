"""Independent preservation and ownership checks for composed FLP exports."""
from pathlib import Path
import hashlib
import struct
import shutil
import sys
import tempfile
import unittest
import uuid

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP / 'work'), str(APP / 'tests')]
from combined_runner import stage_source, validate_final, publish_new
from flp_song_analysis import read_project
from flp_raw import encode_fl
from stem_project import inspect_stems, preview_automation, export_automation
from test_stem_project import audio_fixture, PPQ, END


class CombinedExportChecks(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.gettempdir()) / ('combined-test-' + uuid.uuid4().hex)
        self.folder.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.folder)
        self.source = self.folder / 'source.flp'
        audio_fixture(self.source)
        self.before = self.source.read_bytes()

    def make_output(self):
        preview = dict(source=str(self.source), source_sha256=hashlib.sha256(self.before).hexdigest(),
                       ppq=PPQ, end_tick=END, bpm=170, warnings=[],
                       lanes=[dict(name='Drums volume', family='Drums', parameter='volume_db',
                                   points=[(0,.6875),(PPQ*32,.5),(PPQ*64,.6875),(END,.6875)],
                                   targets=[dict(target=0x70401fc0),dict(target=0x70801fc0)])])
        output = self.folder / 'output.flp'
        export_automation(preview, output)
        return output

    def test_new_export_keeps_existing_audio_and_has_unique_targets(self):
        output = self.make_output()
        result = validate_final(self.source, self.source, output)
        self.assertGreater(result['automation_channels'], 0)
        self.assertTrue(result['original_channels_preserved'])
        self.assertFalse(result['native_fl_playback_tested'])
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_duplicate_target_owner_is_detected(self):
        output = self.make_output()
        project = read_project(output)
        events = list(project['events'])
        link = next(row for row in events if row[0] == 227)
        events.insert(project['layout']['first_channel'], link)
        bad = self.folder / 'duplicate.flp'
        bad.write_bytes(encode_fl(project['header'], events))
        with self.assertRaisesRegex(ValueError, 'competing parameter owner'):
            validate_final(self.source, self.source, bad)

    def test_existing_output_never_overwritten(self):
        with self.assertRaises(FileExistsError):
            publish_new(b'changed', self.source)
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_changed_original_fader_is_detected(self):
        output = self.make_output()
        project = read_project(output)
        events = []
        for key, payload in project['events']:
            if key == 225:
                payload = b''.join(struct.pack('<III', prefix, target, 10000 if target == 0x70401fc0 else value)
                                   for prefix, target, value in struct.iter_unpack('<III', payload))
            events.append((key, payload))
        bad = self.folder / 'changed-fader.flp'
        bad.write_bytes(encode_fl(project['header'], events))
        with self.assertRaisesRegex(ValueError, 'saved mixer'):
            validate_final(self.source, self.source, bad)


if __name__ == '__main__':
    unittest.main()
