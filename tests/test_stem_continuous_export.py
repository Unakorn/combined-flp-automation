"""Native full-song curves, exact cutoff targeting, and source preservation."""
from copy import deepcopy
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'work'))
sys.path.insert(0, str(ROOT / 'tests'))
from test_stem_project import audio_fixture, PPQ, END
from drum_template import OPEN_FILTER, PLUGIN_KEYS, _mixer
from flp_connector import _effects
from flp_song_analysis import read_project
from stem_native import parse_points
from stem_project import export_automation, inspect_stems, preview_automation


class ContinuousNativeExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.source = self.folder / 'original.flp'
        self.output = self.folder / 'continuous.flp'
        self.original = audio_fixture(self.source)
        self.knob_points = [(0, 100), (2 * PPQ, 38), (12 * PPQ, 78),
                            (25 * PPQ, 24), (42 * PPQ, 92), (END, 100)]
        self.section = dict(type='bridge', start_tick=0, end_tick=END,
                            bar=1, energy=.6, intensity='High')

    def cutoff_move(self):
        return dict(name='Low-pass cutoff', stem_id=0, stem_name='1_808 Snare',
                    mixer_insert=1, plugin='Fruity Filter', parameter='lowpass_cutoff_percent',
                    family='Drums', shape='explicit', start_tick=0, end_tick=END,
                    start_value=100, end_value=100, section_start_tick=0,
                    points=deepcopy(self.knob_points))

    def preview(self, moves, sections=None, **plan_metadata):
        sections = sections or [self.section]
        analysis = dict(inspect_stems(self.source), half_time=False, sections=sections)
        plan = dict(moves=moves, sections=sections, warnings=[],
                    planning_mode='continuous', section_activity=[], movement_coverage={})
        plan.update(plan_metadata)
        with patch('stem_project.plan_moves', return_value=plan):
            return preview_automation(analysis, 'Rock', 'Alt Rock')

    @staticmethod
    def initial_values(project):
        data = next(p for k, p in project['events'] if k == 225)
        return {target: value for _, target, value in struct.iter_unpack('<III', data)}

    @staticmethod
    def effect_record(project, insert, slot):
        events = project['events']
        blocks, _ = _mixer(events)
        first, stop, _ = blocks[insert]
        for i in range(first, stop):
            if events[i][0] != 201:
                continue
            j = i + 1
            while j < stop and events[j][0] in PLUGIN_KEYS:
                j += 1
            record = events[i:j]
            if struct.unpack_from('<II', dict(record)[212]) == (insert, slot):
                return record
        raise AssertionError(f'Missing effect at {insert}:{slot}')

    def test_cutoff_export_targets_native_parameter_with_exact_open_state(self):
        preview = self.preview([self.cutoff_move()])
        self.assertFalse(preview['skipped'])
        self.assertEqual(len(preview['effects']), 1)
        self.assertEqual(preview['effects'][0]['slot'], 1)
        self.assertEqual(preview['effects'][0]['target_offset'], 0x8000)
        target = 0x70418000  # Insert 1, dedicated zero-based slot 1, native cutoff.
        lane = preview['lanes'][0]
        self.assertEqual(lane['targets'][0]['target'], target)
        self.assertEqual(lane['targets'][0]['baseline'], 1)
        expected = [(tick, value / 100) for tick, value in self.knob_points]
        self.assertEqual(lane['points'], expected)

        report = export_automation(preview, self.output)
        after = read_project(self.output)
        initial = self.initial_values(after)
        self.assertEqual(initial[target], 1024)
        self.assertEqual(initial[0x70411f01], 12800)
        self.assertEqual(initial[0x70411f00], 1)
        record = self.effect_record(after, 1, 1)
        self.assertEqual(dict(record)[213], OPEN_FILTER)
        channels = [c for c in after['channels'] if c['type'] == 5]
        self.assertEqual(len(channels), 1)
        channel = channels[0]
        fields = dict(after['events'][channel['start']:channel['stop']])
        self.assertEqual(parse_points(fields[234], PPQ), expected)
        links = [struct.unpack('<5I', p) for k, p in after['events'] if k == 227]
        self.assertEqual(links, [(channel['id'] << 16, 0, target, 8, 469)])
        self.assertFalse(any(k in (223, 224) for k, _ in after['events']))
        self.assertFalse(any(c['plugin'] == 'Fruity Keyboard Controller' for c in after['channels']))
        self.assertEqual(report['validation']['delivery_status'], 'development_preview')
        self.assertFalse(report['validation']['native_fl_playback_tested'])

    def test_full_song_curves_have_one_complete_playlist_placement_per_control(self):
        delay = dict(self.cutoff_move(), name='Delay amount', plugin='Fruity Delay 3',
                     parameter='effect_blend_percent', start_value=0, end_value=0,
                     points=[(0, 0), (PPQ * 8, 30), (PPQ * 16, 8),
                             (PPQ * 44, 48), (END, 0)])
        preview = self.preview([self.cutoff_move(), delay])
        self.assertEqual(len(preview['lanes']), 2)
        report = export_automation(preview, self.output)
        after = read_project(self.output)
        channels = [c for c in after['channels'] if c['type'] == 5]
        self.assertEqual(len(channels), 2)
        for channel in channels:
            placements = [clip for clip in after['clips']
                          if clip['kind'] == 'channel' and clip['channel_id'] == channel['id']]
            self.assertEqual(len(placements), 1)
            self.assertEqual((placements[0]['position'], placements[0]['duration']), (0, END))
            raw = after['events'][placements[0]['event_index']][1]
            offset = placements[0]['record_offset']
            self.assertEqual(struct.unpack_from('<ff', raw, offset + 24), (0., END / PPQ))
        self.assertEqual(report['automation_channels'], 2)
        self.assertEqual(report['automation_links'], 2)

    def test_original_effect_record_fader_and_audio_placement_are_preserved(self):
        before = read_project(self.source)
        saved_fx = self.effect_record(before, 1, 0)
        saved_values = self.initial_values(before)
        export_automation(self.preview([self.cutoff_move()]), self.output)
        after = read_project(self.output)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(self.effect_record(after, 1, 0), saved_fx)
        current_values = self.initial_values(after)
        for target, value in saved_values.items():
            self.assertEqual(current_values[target], value)
        self.assertEqual(current_values[0x70401fc0], 11000)
        for channel in before['channels']:
            new_channel = next(c for c in after['channels'] if c['id'] == channel['id'])
            self.assertEqual(before['events'][channel['start']:channel['stop']],
                             after['events'][new_channel['start']:new_channel['stop']])
        old_playlist = next(p for k, p in before['events'] if k == 233)
        self.assertTrue(next(p for k, p in after['events'] if k == 233).startswith(old_playlist))
        blocks, _ = _mixer(after['events'])
        self.assertEqual([(e['slot'], e['plugin']) for e in _effects(after['events'], blocks)],
                         [(0, 'Fruity Wrapper'), (1, 'Fruity Filter')])

    def test_malformed_explicit_curves_are_reported_without_allocating_effects(self):
        bad_curves = [None, [], [(0, 100)], [(1, 100), (END, 70)],
                      [(0, 100), (END - 1, 70)], [(0, 100), (PPQ, 40), (PPQ, 60), (END, 100)],
                      [(0, 100), (END, float('nan'))], [(0, 100), (END, float('inf'))],
                      [(0, 100), (PPQ, 'quiet'), (END, 100)],
                      [(0, 100), (PPQ, True), (END, 100)],
                      [(0, 100), None, (END, 100)], [(0, 100), (PPQ, 40, 20), (END, 100)]]
        for curve in bad_curves:
            with self.subTest(curve=curve):
                move = self.cutoff_move()
                move['points'] = curve
                preview = self.preview([move])
                self.assertEqual(preview['lanes'], [])
                self.assertEqual(preview['effects'], [])
                self.assertEqual(len(preview['skipped']), 1)
                self.assertTrue(preview['skipped'][0]['reason'])
                self.assertTrue(any('skipped' in warning.lower() for warning in preview['warnings']))
                with self.assertRaisesRegex(ValueError, 'No supported native automation'):
                    export_automation(preview, self.output)
                self.assertFalse(self.output.exists())
                self.assertEqual(self.source.read_bytes(), self.original)

    def test_coverage_counts_only_exportable_normalized_movement_in_each_section(self):
        a, b = 16 * PPQ, 32 * PPQ
        sections = [dict(self.section, type=kind, start_tick=start, end_tick=end, bar=bar)
                    for kind, start, end, bar in (('intro', 0, a, 1), ('chorus', a, b, 5),
                                                   ('outro', b, END, 9))]
        moving = dict(self.cutoff_move(), points=[(0, 100), (a, 100),
                                                  ((a + b) // 2, 35), (b, 100), (END, 100)])
        unsupported = dict(self.cutoff_move(), name='Unverified frequency map', parameter='lowpass_hz')
        # The planner's values vary, but clamping the real control at its upper
        # limit produces no change. This stem must not count as moving.
        clipped_flat = dict(self.cutoff_move(), name='Already open', stem_id=1,
                            stem_name='1_808 Tom', mixer_insert=2,
                            points=[(0, 100), (PPQ, 120), (END, 100)])
        preview = self.preview([moving, unsupported, clipped_flat], sections,
                               section_activity=[dict(stem_count=99, moving_targets=99)],
                               movement_coverage=dict(sections_with_movement=99))
        self.assertEqual(len(preview['lanes']), 1)
        self.assertEqual(len(preview['effects']), 1)
        self.assertEqual(len(preview['skipped']), 2)
        coverage = preview['movement_coverage']
        self.assertEqual(coverage['sections_with_movement'], 1)
        self.assertEqual(coverage['total_sections'], 3)
        activity = preview['section_activity']
        self.assertEqual([s['moving_targets'] for s in activity], [0, 1, 0])
        self.assertEqual([s['stem_count'] for s in activity], [0, 1, 0])
        target = activity[1]['targets'][0]
        self.assertEqual((target['stem_id'], target['parameter']), (0, 'lowpass_cutoff_percent'))
        self.assertEqual(target['moving_ticks'], b - a)
        self.assertEqual(target['changing_active_fraction'], 1)
        self.assertIn('no supported movement', preview['summary'][0])
        self.assertIn('1 moving control', preview['summary'][1])
        self.assertIn('1 active stem', preview['summary'][1])
        self.assertIn('no supported movement', preview['summary'][2])


if __name__ == '__main__':
    unittest.main()
