"""Musical planning contracts; native target/FLP tests live in the writer suite."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'work'))
from stem_planner import BLEND_PARAMETERS, _stem_scale, group_moves, load_presets, plan_moves, validate_presets


PPQ = 96
BAR = PPQ * 4


def stem(identifier=1, name='Keys', role='keys', **analysis):
    return {'id': identifier, 'name': name, 'role': role, 'mixer_insert': identifier,
            'analysis': analysis}


def sections():
    return [dict(type='intro', start_tick=0, end_tick=BAR * 4, energy=.15),
            dict(type='verse', start_tick=BAR * 4, end_tick=BAR * 12, energy=.25),
            dict(type='pre-chorus', start_tick=BAR * 12, end_tick=BAR * 16, energy=.45),
            dict(type='chorus', start_tick=BAR * 16, end_tick=BAR * 24, energy=1),
            dict(type='bridge', start_tick=BAR * 24, end_tick=BAR * 28, energy=.3),
            dict(type='outro', start_tick=BAR * 28, end_tick=BAR * 32, energy=.1)]


class PresetTests(unittest.TestCase):
    def test_all_requested_subgenres_have_three_to_six_characteristic_moves(self):
        data = load_presets()
        expected = {'Hip-Hop': {'Trap', 'Boom Bap', 'Lo-fi', 'Drill'},
                    'EDM': {'Full song movement', 'House', 'Dubstep', 'Future Bass', 'Techno'},
                    'Drum & Bass': {'Liquid', 'Neurofunk', 'Jungle'},
                    'Funk/Soul': {'Modern Funk', 'Soul'},
                    'Pop': {'Mainstream Pop', 'Synth-pop'},
                    'R&B': {'Modern R&B', 'Neo-Soul'},
                    'Rock': {'Alt Rock', 'Pop Punk'}}
        self.assertEqual({g: set(s) for g, s in data.items()}, expected)
        for choices in data.values():
            for preset in choices.values():
                if preset.get('engine')=='continuous':
                    self.assertEqual(preset['baseline_intensity'],'High')
                    self.assertIn('filter_minimum_percent',preset['settings'])
                    continue
                names = {m['name'] for moves in preset['roles'].values() for m in moves}
                self.assertTrue(3 <= len(names) <= 6)
                self.assertTrue(any(m['parameter'] == 'volume_db'
                                    for moves in preset['roles'].values() for m in moves))

    def test_untrusted_preset_cannot_request_third_party_or_nonfinite_values(self):
        data = load_presets()
        move = data['Hip-Hop']['Trap']['roles']['bass'][0]
        move['plugin'] = 'Third-party synth'
        with self.assertRaisesRegex(ValueError, 'stock plugin'):
            validate_presets(data)
        move['plugin'] = 'Mixer'
        move['end_value'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            validate_presets(data)


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.presets = load_presets()

    def test_last_bar_ends_exactly_at_chorus_and_volume_is_relative(self):
        result = plan_moves([stem(role='bass')], sections(), self.presets['Hip-Hop']['Trap'], ppq=PPQ)
        move = result['moves'][0]
        self.assertEqual(move['start_tick'], 16 * BAR - PPQ)
        self.assertEqual(move['end_tick'], 16 * BAR)
        self.assertEqual(move['section_start_tick'], 16 * BAR)
        self.assertTrue(move['relative'])
        self.assertEqual(move['neutral_value'], 0)
        self.assertLess(move['end_value'], 0)

    def test_half_time_changes_move_length_but_not_project_bar_labels(self):
        data = [stem(role='bass')]
        preset = self.presets['Hip-Hop']['Trap']
        normal = plan_moves(data, sections(), preset, ppq=PPQ)
        half = plan_moves(data, sections(), preset, ppq=PPQ, half_time=True)
        a, b = normal['moves'][0], half['moves'][0]
        self.assertEqual(a['end_tick'], b['end_tick'])
        self.assertEqual(b['end_tick'] - b['start_tick'], 2 * (a['end_tick'] - a['start_tick']))
        self.assertEqual(normal['sections'][3]['bar'], half['sections'][3]['bar'])

    def test_section_energy_and_transition_triggers(self):
        lofi = plan_moves([stem()], sections(), self.presets['Hip-Hop']['Lo-fi'], ppq=PPQ)
        filters = [m for m in lofi['moves'] if m['parameter'] == 'lowpass_blend_percent']
        self.assertEqual({m['section_type'] for m in filters}, {'intro', 'verse', 'pre-chorus', 'bridge', 'outro'})
        self.assertTrue(any(m['start_tick'] == BAR * 28 and m['parameter'] == 'volume_db' for m in lofi['moves']))
        techno = plan_moves([stem(role='bass')], sections(), self.presets['EDM']['Techno'], ppq=PPQ)
        self.assertEqual(len(techno['moves']), 1)
        self.assertEqual(techno['moves'][0]['end_tick'], 16 * BAR)

    def test_auto_increases_for_energy_jump_and_reduces_busy_dynamic_stems(self):
        data = [stem(1, 'Sparse', 'bass', onset_density=.1, dynamic_range_db=3),
                stem(2, 'Busy', 'bass', onset_density=9, dynamic_range_db=25)]
        result = plan_moves(data, sections(), self.presets['Hip-Hop']['Trap'], ppq=PPQ)
        self.assertGreater(result['sections'][3]['intensity_value'], result['sections'][1]['intensity_value'])
        sparse, busy = result['moves']
        self.assertGreater(abs(sparse['end_value']), abs(busy['end_value']))
        self.assertLess(busy['intensity_value'], sparse['intensity_value'])

    def test_override_remains_capped_for_volume_and_drum_blend(self):
        preset = deepcopy(self.presets['Hip-Hop']['Trap'])
        bass = preset['roles']['bass'][0]
        bass.update(end_value=-80, max_delta=80)
        hat = preset['roles']['hat'][0]
        hat.update(end_value=1000, max_delta=50000)
        result = plan_moves([stem(1, role='bass'), stem(2, role='hat')], sections(), preset,
                            ppq=PPQ, intensity_override='High')
        volume = next(m for m in result['moves'] if m['parameter'] == 'volume_db')
        cutoff = next(m for m in result['moves'] if m['parameter'] == 'lowpass_blend_percent')
        self.assertEqual(volume['end_value'], -12)
        self.assertLessEqual(cutoff['end_value'], 40)
        self.assertTrue(all(s['intensity'] == 'High' for s in result['sections']))

    def test_short_sections_clamp_and_first_section_has_no_pickup(self):
        short = [dict(type='chorus', start_tick=0, end_tick=PPQ, energy=.8),
                 dict(type='chorus', start_tick=PPQ, end_tick=PPQ * 2, energy=1)]
        result = plan_moves([stem(role='brass')], short, self.presets['Funk/Soul']['Modern Funk'], ppq=PPQ)
        move = result['moves'][0]
        self.assertEqual((move['start_tick'], move['end_tick']), (0, PPQ))
        self.assertEqual(len(result['moves']), 1)

    def test_house_pump_repeats_on_beat_without_overlap(self):
        result = plan_moves([stem(role='bass')], sections(), self.presets['EDM']['House'], ppq=PPQ)
        moves = result['moves']
        self.assertEqual(len(moves), 8 * 4)
        self.assertEqual(moves[0]['start_tick'], BAR * 16)
        self.assertEqual(moves[-1]['end_tick'], BAR * 24)
        self.assertTrue(all(a['end_tick'] == b['start_tick'] for a, b in zip(moves, moves[1:])))
        self.assertTrue(all(m['end_value'] == 0 and m['start_value'] < 0 for m in moves))

    def test_unknown_role_and_overlapping_sections_fail_before_plan(self):
        with self.assertRaisesRegex(ValueError, 'Confirm a role'):
            plan_moves([stem(role='unknown')], sections(), self.presets['Hip-Hop']['Trap'], ppq=PPQ)
        broken = sections()
        broken[1]['start_tick'] = 0
        with self.assertRaisesRegex(ValueError, 'non-overlapping'):
            plan_moves([stem()], broken, self.presets['Hip-Hop']['Trap'], ppq=PPQ)

    def test_phrase_throws_use_only_matched_phrase_endings_and_missing_guidance_warns(self):
        vocal = stem(role='vocals')
        preset = self.presets['Hip-Hop']['Trap']
        missing = plan_moves([vocal], sections(), preset, ppq=PPQ)
        self.assertFalse(missing['moves'])
        self.assertTrue(any('no matched MIDI phrase endings' in warning for warning in missing['warnings']))
        vocal['phrase_ends'] = [6 * BAR, 8 * BAR, 18 * BAR]
        result = plan_moves([vocal], sections(), preset, ppq=PPQ)
        self.assertEqual([move['end_tick'] for move in result['moves']], vocal['phrase_ends'])
        self.assertTrue(all(move['end_tick'] - move['start_tick'] == PPQ for move in result['moves']))

    def test_close_phrase_endings_never_overlap_parameter_moves(self):
        vocal = stem(role='vocals')
        vocal['phrase_ends'] = [6 * BAR, 6 * BAR + 24]
        preset = deepcopy(self.presets['Hip-Hop']['Trap'])
        preset['roles']['vocals'][0]['trigger'].pop('min_spacing_bars')
        preset['roles']['vocals'][0]['trigger'].pop('max_per_section')
        result = plan_moves([vocal], sections(), preset, ppq=PPQ)
        first, second = result['moves']
        self.assertEqual(first['end_tick'], second['start_tick'])
        self.assertEqual(first['release_ticks'], 0)

    def test_sparse_stem_silence_does_not_quarter_its_intensity(self):
        sparse = stem(role='guitar', dynamic_range_db=88, onset_density=.4,
                      chunks=[{'rms': 0}, {'rms': .1}, {'rms': .08}, {'rms': 0}])
        busy = stem(role='guitar', active_dynamic_range_db=25, onset_density=9)
        self.assertGreaterEqual(_stem_scale(sparse), .95)
        self.assertLess(_stem_scale(busy), _stem_scale(sparse))

    def test_presets_use_explicit_supported_effect_blends(self):
        supported = {'volume_db', 'pan_percent'} | BLEND_PARAMETERS
        for genre in self.presets.values():
            for preset in genre.values():
                if preset.get('engine')=='continuous':continue
                self.assertNotIn('all', preset['roles'])
                for moves in preset['roles'].values():
                    self.assertTrue(all(m['parameter'] in supported for m in moves))

    def test_full_song_preset_dispatches_continuous_curves_and_respects_baseline(self):
        preset=deepcopy(self.presets['EDM']['Full song movement'])
        part=stem(bpm=170)
        part.update(position=0,end_tick=32*BAR)
        high=plan_moves([part],sections(),preset,ppq=PPQ)
        self.assertEqual(high['planning_mode'],'continuous')
        self.assertEqual(high['movement_coverage']['sections_with_movement'],len(sections()))
        cutoff=next(m for m in high['moves'] if m['parameter']=='lowpass_cutoff_percent')
        self.assertEqual(cutoff['shape'],'explicit')
        self.assertEqual((cutoff['start_tick'],cutoff['end_tick']),(0,32*BAR))
        self.assertLess(cutoff['minimum_value'],60)
        preset['baseline_intensity']='Low'
        low=plan_moves([part],sections(),preset,ppq=PPQ)
        self.assertGreater(next(m for m in low['moves'] if m['parameter']=='lowpass_cutoff_percent')['minimum_value'],cutoff['minimum_value'])

    def test_trap_gives_actual_band_material_section_contrast_and_effect_releases(self):
        roles = ['hat', 'hat', 'snare', 'snare', 'tom', 'kick', 'bass', 'brass', 'guitar', 'guitar', 'keys']
        stems = [stem(i + 1, role=role) for i, role in enumerate(roles)]
        for row in stems:
            row['phrase_ends'] = [6 * BAR, 8 * BAR, 18 * BAR]
        result = plan_moves(stems, sections(), self.presets['Hip-Hop']['Trap'], ppq=PPQ)
        self.assertEqual({m['stem_id'] for m in result['moves']}, set(range(1, 12)) - {6})
        pockets = [m for m in result['moves'] if m['name'] == 'Lower instrument volume during verse']
        self.assertTrue(pockets)
        self.assertTrue(all(m['end_tick'] - m['start_tick'] == BAR * 8 for m in pockets))
        fx = [m for m in result['moves'] if m['parameter'] in BLEND_PARAMETERS]
        self.assertTrue(all(m['neutral_value'] == 0 and m['release_end_tick'] == m['end_tick'] + m['release_ticks'] for m in fx))
        self.assertTrue(any(m['release_ticks'] > 0 for m in fx))

    def test_known_silent_intervals_are_skipped_but_later_repeats_are_kept(self):
        bass = stem(role='bass')
        bass['note_spans'] = [(BAR * 16 + PPQ, BAR * 16 + PPQ * 3)]
        result = plan_moves([bass], sections(), self.presets['EDM']['House'], ppq=PPQ)
        self.assertEqual([m['start_tick'] for m in result['moves']], [BAR * 16 + PPQ, BAR * 16 + PPQ * 2])

    def test_chorus_continuation_does_not_repeat_a_new_drop_gesture(self):
        rows = sections()
        rows[4]['type'] = 'chorus'
        result = plan_moves([stem(role='bass')], rows, self.presets['Hip-Hop']['Trap'], ppq=PPQ)
        self.assertEqual(len(result['moves']), 1)

    def test_audio_activity_uses_nonzero_playlist_placement_once(self):
        bass = stem(role='bass', bpm=120, activity_segments=[{'start_seconds': 0, 'end_seconds': 1}])
        bass['position'] = bass['offset_tick'] = BAR * 16
        result = plan_moves([bass], sections(), self.presets['EDM']['House'], ppq=PPQ)
        self.assertEqual([m['start_tick'] for m in result['moves']], [BAR * 16, BAR * 16 + PPQ])

    def test_all_subgenres_emit_restrained_moves_for_matching_roles(self):
        roles = ['kick', 'snare', 'clap', 'tom', 'hat', 'perc', 'ride', 'crash', 'bass',
                 'brass', 'guitar', 'keys', 'vocals', 'pads', 'synth', 'effects']
        stems = [stem(i + 1, role=role) for i, role in enumerate(roles)]
        for choices in self.presets.values():
            for preset in choices.values():
                result = plan_moves(stems, sections(), preset, ppq=PPQ)
                self.assertTrue(result['moves'])
                for move in result['moves']:
                    self.assertLess(move['start_tick'], move['end_tick'])
                    self.assertGreaterEqual(move['start_tick'], 0)
                    self.assertLessEqual(move['end_tick'], BAR * 32)
                    if move['parameter'] == 'volume_db':
                        self.assertTrue(-12 <= move['start_value'] <= 3)
                        self.assertTrue(-12 <= move['end_value'] <= 3)


class GroupingTests(unittest.TestCase):
    def moves(self):
        preset = load_presets()['Pop']['Mainstream Pop']
        result = plan_moves([stem(1, 'Guitar A', 'guitar'), stem(2, 'Guitar B', 'guitar')],
                            sections(), preset, ppq=PPQ)
        return result['moves']

    def test_same_family_exact_curves_merge_and_preserve_every_target(self):
        groups = group_moves(self.moves(), PPQ)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['clip_name'], 'Melodic - Restore volume before chorus')
        self.assertEqual({m['stem_id'] for m in groups[0]['moves']}, {1, 2})
        self.assertEqual(len(groups[0]['targets']), 2)

    def test_different_ranges_stay_separate_without_verified_mapping(self):
        moves = self.moves()
        moves[1]['start_value'] *= .5
        self.assertEqual(len(group_moves(moves, PPQ)), 2)
        self.assertEqual(len(group_moves(moves, PPQ, per_link_ranges=True)), 1)

    def test_timing_within_one_beat_is_kept_when_merging_would_change_it(self):
        moves = self.moves()
        moves[1]['start_tick'] += 1
        self.assertEqual(len(group_moves(moves, PPQ)), 2)
        self.assertEqual(len(group_moves(moves, PPQ, per_link_ranges=True)), 2)

    def test_shape_section_family_and_actual_points_are_required(self):
        for field, value in [('shape', 'hold'), ('section_start_tick', 0), ('family', 'Drums'),
                             ('points', [(0, .5), (96, .8)])]:
            moves = self.moves()
            moves[1][field] = value
            self.assertEqual(len(group_moves(moves, PPQ)), 2, field)


if __name__ == '__main__':
    unittest.main()
