"""Movement selection is independent of genre; native output stays preserved."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'work'))
sys.path.insert(0, str(ROOT / 'tests'))
from flp_song_analysis import read_project
from stem_planner import load_presets, plan_moves
from stem_project import export_automation, inspect_stems, preview_automation
from test_stem_project import audio_fixture, PPQ, END

BAR = PPQ * 4


def sections(end=BAR * 32):
    return [dict(type=kind, start_tick=start, end_tick=stop, energy=energy)
            for kind, start, stop, energy in (
                ('intro', 0, end // 4, .5), ('verse', end // 4, end // 2, .65),
                ('chorus', end // 2, end * 3 // 4, .95), ('outro', end * 3 // 4, end, .4))]


def stems():
    return [dict(id=i, name=name, role=role, mixer_insert=i + 1,
                 analysis=dict(bpm=170, sustained=.7, dynamic_range_db=12))
            for i, (name, role) in enumerate((('Keys', 'keys'), ('Bass', 'bass'), ('Guitar', 'guitar')))]


class MovementModeTests(unittest.TestCase):
    def setUp(self):
        self.presets = load_presets()

    def plan(self, genre='Hip-Hop', style='Drill', movement_mode=None, **kwargs):
        return plan_moves(stems(), sections(), self.presets[genre][style], ppq=PPQ,
                          movement_mode=movement_mode, **kwargs)

    def test_drill_full_song_has_real_cutoff_and_continuous_active_section_coverage(self):
        plan = self.plan(movement_mode='full_song')
        self.assertEqual(plan['movement_mode'], 'full_song')
        self.assertEqual(plan['planning_mode'], 'continuous')
        cutoff = [m for m in plan['moves'] if m['parameter'] == 'lowpass_cutoff_percent']
        self.assertEqual({m['role'] for m in cutoff}, {'keys', 'guitar'})
        self.assertGreater(min(m['maximum_value'] - m['minimum_value'] for m in cutoff), 50)
        self.assertTrue(all(m['coverage']['changing_active_fraction'] > .75 for m in cutoff))
        self.assertTrue(all(m['shape'] == 'explicit' and m['start_tick'] == 0 and m['end_tick'] == BAR * 32
                            for m in plan['moves']))
        self.assertEqual(plan['movement_coverage']['sections_with_movement'], 4)

    def test_changing_genre_keeps_shared_high_baseline_and_the_same_full_song_curves(self):
        drill = self.plan(movement_mode='full_song')
        low = self.plan(style='Lo-fi', movement_mode='full_song')
        rock = self.plan('Rock', 'Alt Rock', movement_mode='full_song')
        self.assertEqual(self.presets['Hip-Hop']['Lo-fi']['baseline_intensity'], 'Low')
        for plan in (drill, low, rock):
            self.assertEqual(plan['baseline_intensity'], 'High')
            self.assertEqual(plan['moves'], drill['moves'])
            self.assertEqual(plan['sections'], drill['sections'])

    def test_full_song_reads_shared_editable_settings_instead_of_genre_recipes(self):
        edited = deepcopy(self.presets)
        edited['EDM']['Full song movement']['settings']['pan_maximum_percent'] = 7
        with patch('stem_planner.load_presets', return_value=edited):
            plan = self.plan(movement_mode='full_song')
        self.assertEqual(plan['settings']['pan_maximum_percent'], 7)
        self.assertTrue(all(abs(v) <= 7 for m in plan['moves'] if m['parameter'] == 'pan_percent' for _, v in m['points']))

    def test_brief_mode_retains_drill_recipes_and_omitted_mode_is_backward_compatible(self):
        explicit = self.plan(movement_mode='brief', intensity_override='High')
        old = self.plan(intensity_override='High')
        self.assertEqual(explicit, old)
        self.assertEqual(explicit['movement_mode'], 'brief')
        self.assertTrue(explicit['moves'])
        self.assertFalse(any(m['shape'] == 'explicit' for m in explicit['moves']))
        self.assertFalse(any(m['parameter'] == 'lowpass_cutoff_percent' for m in explicit['moves']))
        self.assertTrue(all(m['end_tick'] - m['start_tick'] <= BAR for m in explicit['moves']))
        alias = self.plan('EDM', 'Full song movement')
        self.assertEqual(alias['movement_mode'], 'full_song')
        self.assertEqual(alias['planning_mode'], 'continuous')

    def test_brief_alias_and_unknown_modes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Choose another style'):
            self.plan('EDM', 'Full song movement', movement_mode='brief')
        for mode in ('continuous', 'FULL_SONG', '', True, []):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'Movement mode'):
                self.plan(movement_mode=mode)

    def test_missing_or_wrong_shared_preset_fails_clearly_instead_of_falling_back_to_brief(self):
        edited = deepcopy(self.presets)
        del edited['EDM']['Full song movement']
        with patch('stem_planner.load_presets', return_value=edited), self.assertRaisesRegex(ValueError, 'Shared full-song settings are missing'):
            self.plan(movement_mode='full_song')
        edited['EDM']['Full song movement'] = edited['Hip-Hop']['Drill']
        with patch('stem_planner.load_presets', return_value=edited), self.assertRaisesRegex(ValueError, 'continuous engine'):
            self.plan(movement_mode='full_song')

    def test_selected_preset_and_analysis_inputs_are_unchanged(self):
        source, rows, preset = stems(), sections(), self.presets['Hip-Hop']['Drill']
        before = deepcopy((source, rows, preset))
        plan_moves(source, rows, preset, ppq=PPQ, movement_mode='full_song')
        self.assertEqual((source, rows, preset), before)

    def test_full_song_handles_six_or_eight_drums_without_requiring_extra_parts(self):
        roles = ('kick', 'snare', 'clap', 'tom', 'hat', 'perc', 'ride', 'crash')
        for count in (6, 8):
            source = [dict(id=i, name=role, role=role, mixer_insert=i + 1,
                           analysis=dict(bpm=170)) for i, role in enumerate(roles[:count])]
            with self.subTest(drums=count):
                plan = plan_moves(source, sections(), self.presets['Hip-Hop']['Drill'],
                                  ppq=PPQ, movement_mode='full_song')
                self.assertEqual({m['role'] for m in plan['moves']}, set(roles[1:count]))
                self.assertEqual(plan['movement_coverage']['sections_with_movement'], 4)
                self.assertTrue(all(m['shape'] == 'explicit' for m in plan['moves']))


class MovementPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.source = self.folder / 'original.flp'
        self.original = audio_fixture(self.source)
        self.analysis = dict(inspect_stems(self.source), half_time=False, sections=sections(END))
        # Fixture channels are arbitrary audio clips; roles are explicitly
        # confirmed here to exercise real melodic cutoff and bass controls.
        for record, role in zip(self.analysis['stems'], ('keys', 'bass')):
            record['role'] = role
            record['analysis'] = dict(bpm=170, sustained=.7, dynamic_range_db=12)

    def test_drill_preview_propagates_mode_and_exports_native_full_song_cutoff(self):
        before = deepcopy(self.analysis)
        preview = preview_automation(self.analysis, 'Hip-Hop', 'Drill', movement_mode='full_song')
        self.assertEqual(preview['movement_mode'], 'full_song')
        self.assertEqual(preview['planning_mode'], 'continuous')
        self.assertGreater(preview['capabilities']['filter_targets'], 0)
        self.assertEqual(preview['movement_coverage']['sections_with_movement'], 4)
        self.assertFalse(preview['skipped'])
        output = self.folder / 'full_song.flp'
        export_automation(preview, output)
        project = read_project(output)
        channels = [c for c in project['channels'] if c['type'] == 5]
        self.assertEqual(len(channels), len(preview['lanes']))
        for channel in channels:
            clips = [c for c in project['clips'] if c.get('channel_id') == channel['id']]
            self.assertEqual(len(clips), 1)
            self.assertEqual((clips[0]['position'], clips[0]['duration']), (0, END))
        self.assertFalse(any(k in (223, 224) for k, _ in project['events']))
        self.assertFalse(any(c['plugin'] == 'Fruity Keyboard Controller' for c in project['channels']))
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(self.analysis, before)

    def test_explicit_brief_preview_keeps_fixed_blends_and_reports_brief_mode(self):
        preview = preview_automation(self.analysis, 'Hip-Hop', 'Drill', movement_mode='brief')
        self.assertEqual(preview['movement_mode'], 'brief')
        self.assertEqual(preview['planning_mode'], 'selected_changes')
        self.assertEqual(preview['capabilities']['filter_targets'], 0)
        self.assertTrue(any(t['parameter'] == 'lowpass_blend_percent' for lane in preview['lanes'] for t in lane['targets']))
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_preview_rejects_invalid_mode_and_brief_continuous_alias(self):
        with self.assertRaisesRegex(ValueError, 'Movement mode'):
            preview_automation(self.analysis, 'Hip-Hop', 'Drill', movement_mode='mystery')
        with self.assertRaisesRegex(ValueError, 'Choose another style'):
            preview_automation(self.analysis, 'EDM', 'Full song movement', movement_mode='brief')
        self.assertEqual(self.source.read_bytes(), self.original)


if __name__ == '__main__':
    unittest.main()
