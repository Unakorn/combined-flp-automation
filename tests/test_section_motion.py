"""Musical constraints and reproducibility of sparse section automation."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'work'))
from section_motion import (FAMILIES, default_settings, plan_section_motion,
                            resolve_intensities, validate_settings)


PPQ = 96
BAR = PPQ * 4
END = BAR * 32
BPM = 120


def stem(identifier=1, role='pads', *, midi=True, channels=2, spans=None):
    spans = [[0, END]] if spans is None else spans
    ticks_per_second = BPM * PPQ / 60
    result = {'id': identifier, 'name': f'{role} {identifier}', 'role': role,
              'mixer_insert': identifier, 'midi_verified': midi,
              'timing_provenance': 'midi' if midi else 'audio-exempt',
              'analysis': {'channels': channels,
                           'activity_segments': [dict(start_seconds=a / ticks_per_second,
                                                      end_seconds=b / ticks_per_second)
                                                 for a, b in spans],
                           'level_windows': [dict(start_seconds=0, end_seconds=END / ticks_per_second,
                                                  rms=.1 if spans else 0, peak=.2 if spans else 0)]}}
    if midi:
        result['note_events'] = [dict(tick=a, duration=b-a, note=60, velocity=100) for a, b in spans]
    return result


def analysis(stems=None):
    return {'ppq': PPQ, 'bpm': BPM, 'end_tick': END, 'time_signature': [4, 4],
            'stems': [stem()] if stems is None else stems,
            'sections': [dict(name=f'Custom part {i}', start_tick=i * BAR * 8,
                              end_tick=(i + 1) * BAR * 8, energy=energy)
                         for i, energy in enumerate((.3, .9, .4, 1.))]}


def lane(plan, family, stem_id=1):
    return next(c for c in plan['curves'] if c['family'] == family and c['stem_id'] == stem_id)


def value_at(points, tick):
    for (a, first), (b, last) in zip(points, points[1:]):
        if a <= tick <= b:
            return first + (last - first) * (tick - a) / (b - a)
    raise AssertionError(f'Tick {tick} is outside the envelope.')


def shared_guide(song, identifiers):
    for source in song['stems']:
        if source['id'] in identifiers:
            source.update(midi_verified=False, midi_guide_verified=True,
                          timing_provenance='shared-drum-guide', midi_file_track_index=2)
            source.pop('note_events', None)
    song['shared_drum_guide'] = dict(midi_track='DRUMS', midi_file_track_index=2,
        stem_ids=list(identifiers), verified=True, timing_provenance='shared-drum-guide',
        note_events=[dict(tick=0, duration=END, note=60)])
    return song


class SectionSettingsTests(unittest.TestCase):
    def test_defaults_are_independent_and_do_not_share_mutable_state(self):
        first = default_settings(5)
        self.assertEqual(first['reverb_mode'], 'baked')
        self.assertEqual(first['intensities'], dict.fromkeys(FAMILIES, 'auto'))
        first['enabled']['filter'] = False
        self.assertTrue(default_settings(5)['enabled']['filter'])
        amounts = resolve_intensities({'seed': 5, 'reverb_mode': 'add'})
        self.assertGreater(len(set(amounts.values())), 1)
        self.assertTrue(all(0 <= amount <= 100 for amount in amounts.values()))

    def test_seeded_defaults_use_audible_calibrated_depth_ranges(self):
        bounds = {'filter': (55, 80), 'delay': (40, 65),
                  'width': (45, 75), 'reverb': (35, 60),
                  'volume': (55, 75), 'pan': (30, 50)}
        for seed in range(20):
            for family, value in resolve_intensities({'seed': seed, 'reverb_mode': 'add'}).items():
                low, high = bounds[family]
                self.assertTrue(low <= value <= high, (seed, family, value))

    def test_numeric_override_is_fixed_across_seeds(self):
        settings = dict(seed=1, intensities={'filter': 73.5})
        one = resolve_intensities(settings)
        settings['seed'] = 9182
        two = resolve_intensities(settings)
        self.assertEqual(one['filter'], 73.5)
        self.assertEqual(two['filter'], 73.5)
        self.assertNotEqual((one['delay'], one['width']), (two['delay'], two['width']))

    def test_baked_reverb_overrides_explicit_amount(self):
        settings = {'intensities': {'reverb': 100}, 'reverb_mode': 'baked'}
        self.assertEqual(resolve_intensities(settings)['reverb'], 0)
        settings['reverb_mode'] = 'add'
        self.assertEqual(resolve_intensities(settings)['reverb'], 100)
        settings['enabled'] = {'reverb': False}
        self.assertEqual(resolve_intensities(settings)['reverb'], 0)

    def test_invalid_settings_fail(self):
        for settings in ({'seed': True}, {'seed': -1}, {'seed': 2.5},
                         {'intensities': {'delay': float('nan')}},
                         {'intensities': {'filter': 101}},
                         {'intensities': {'width': -1}},
                         {'intensities': {'width': True}},
                         {'intensities': {'phaser': 50}},
                         {'enabled': {'filter': 1}}, {'reverb_mode': 'unknown'},
                         {'genre': 'anything'}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_settings(settings)


class SectionMotionTests(unittest.TestCase):
    def test_sparse_linear_fullsong_curves_and_signed_bounds(self):
        song = analysis([stem(), stem(2, 'vocals', midi=False)])
        plan = plan_section_motion(song, {'reverb_mode': 'add'})
        self.assertTrue(plan['curves'])
        for curve in plan['curves']:
            points = curve['points']
            self.assertEqual(points[0], (0, 0.))
            self.assertEqual(points[-1], (END, 0.))
            self.assertTrue(all(a[0] < b[0] for a, b in zip(points, points[1:])))
            low, high = {'volume': (-3, 0), 'pan': (-8, 8)}.get(curve['family'], (-1, 1))
            self.assertTrue(all(low <= value <= high for _, value in points))
            self.assertLessEqual(len(points), 2 * len(song['sections']) + 3)
            if curve['family'] == 'filter':
                self.assertTrue(all(value <= 0 for _, value in points))
            if curve['family'] in ('delay', 'reverb'):
                self.assertTrue(all(value >= 0 for _, value in points))
        for source in song['stems']:
            own = [c for c in plan['curves'] if c['stem_id'] == source['id']]
            self.assertLessEqual(len(own), 4)
            self.assertLessEqual(sum(c['family'] not in ('volume', 'pan') for c in own), 2)

    def test_seed_only_changes_values_not_sections_families_or_point_times(self):
        song = analysis([stem(), stem(2, 'vocals', midi=False,
                                    spans=[[0, 7 * BAR], [9 * BAR, 15 * BAR], [17 * BAR, END]])])
        first = plan_section_motion(song, {'seed': 5, 'reverb_mode': 'add'})
        second = plan_section_motion(song, {'seed': 9, 'reverb_mode': 'add'})
        self.assertEqual(first['sections'], second['sections'])
        first_timing = [(c['stem_id'], c['family'], [t for t, _ in c['points']]) for c in first['curves']]
        second_timing = [(c['stem_id'], c['family'], [t for t, _ in c['points']]) for c in second['curves']]
        self.assertEqual(first_timing, second_timing)
        self.assertNotEqual(first['curves'], second['curves'])

    def test_input_order_cannot_change_output_or_seed_values(self):
        song = analysis([stem(2, 'guitar'), stem(1, 'pads')])
        first = plan_section_motion(song, {'seed': 984})
        song['stems'].reverse()
        second = plan_section_motion(song, {'seed': 984})
        self.assertEqual(first, second)

    def test_numeric_intensity_scales_same_geometry(self):
        high = plan_section_motion(analysis(), {'intensities': {'filter': 100}})
        low = plan_section_motion(analysis(), {'intensities': {'filter': 20}})
        for (a, v), (b, w) in zip(lane(high, 'filter')['points'], lane(low, 'filter')['points']):
            self.assertEqual(a, b)
            self.assertAlmostEqual(w, v * .2)

    def test_changes_are_sustained_section_ramps_while_audio_plays(self):
        plan = plan_section_motion(analysis(), {'intensities': dict.fromkeys(FAMILIES, 100)})
        for family in ('filter', 'width'):
            points = lane(plan, family)['points']
            slopes = [(a, b, first, last) for (a, first), (b, last)
                      in zip(points, points[1:]) if abs(last - first) > .1]
            self.assertTrue(any(b - a >= 7 * BAR for a, b, _, _ in slopes), family)
            # The movement spans the middle of a playing section, not merely
            # its last bar or a silent transition after it.
            self.assertNotAlmostEqual(value_at(points, 2 * BAR), value_at(points, 6 * BAR))
            self.assertTrue(any(a <= 2 * BAR and b >= 6 * BAR for a, b, _, _ in slopes))

    def test_full_intensity_reaches_calibrated_family_depths(self):
        sources = [stem(), stem(2, 'vocals', midi=False,
                               spans=[[0, 7 * BAR], [9 * BAR, END]])]
        settings = {'intensities': dict.fromkeys(FAMILIES, 100), 'reverb_mode': 'add'}
        plan = plan_section_motion(analysis(sources), settings)
        self.assertEqual(min(v for _, v in lane(plan, 'filter')['points']), -1.)
        width = [v for _, v in lane(plan, 'width')['points']]
        self.assertEqual(min(width), -1.)
        self.assertEqual(max(width), 1.)
        self.assertEqual(max(v for _, v in lane(plan, 'delay', 2)['points']), 1.)
        self.assertEqual(max(v for _, v in lane(plan, 'reverb', 2)['points']), 1.)

    def test_equal_section_targets_are_holds_between_ramps(self):
        song = analysis()
        song['sections'][1]['energy'] = song['sections'][2]['energy'] = .6
        curve = lane(plan_section_motion(song, {'intensities': {'filter': 100}}), 'filter')
        self.assertEqual(value_at(curve['points'], 10 * BAR),
                         value_at(curve['points'], 14 * BAR))
        self.assertNotEqual(value_at(curve['points'], 18 * BAR),
                            value_at(curve['points'], 22 * BAR))

    def test_long_silence_is_baseline_and_entry_exit_ramps_are_audible(self):
        source = stem(spans=[[0, 7 * BAR], [18 * BAR, END]])
        plan = plan_section_motion(analysis([source]), {'intensities': {'filter': 100, 'width': 100}})
        for curve in plan['curves']:
            points = curve['points']
            self.assertEqual(value_at(points, 7 * BAR), 0.)
            self.assertEqual(value_at(points, 12 * BAR), 0.)
            self.assertEqual(value_at(points, 18 * BAR), 0.)
            self.assertNotEqual(value_at(points, 2 * BAR), value_at(points, 5 * BAR))
            # No changing line falls entirely inside the long silent gap.
            self.assertFalse(any(7 * BAR <= a and b <= 18 * BAR and v != w
                                 for (a, v), (b, w) in zip(points, points[1:])))

    def test_short_phrase_gaps_do_not_create_repeated_baseline_dips(self):
        source = stem(spans=[[0, 3 * BAR], [4 * BAR, END]])
        plan = plan_section_motion(analysis([source]), {'intensities': {'filter': 100}})
        points = lane(plan, 'filter')['points']
        self.assertNotEqual(value_at(points, 3 * BAR), 0.)
        self.assertNotEqual(value_at(points, 4 * BAR), 0.)
        self.assertFalse(any(t in (3 * BAR, 4 * BAR) for t, _ in points))

    def test_silent_section_extremes_do_not_absorb_audible_depth_range(self):
        song = analysis([stem(spans=[[8 * BAR, END]])])
        for row, energy in zip(song['sections'], (0., .4, .7, 1.)):
            row['energy'] = energy
        plan = plan_section_motion(song, {'intensities': {'filter': 100, 'width': 100}})
        points = lane(plan, 'filter')['points']
        self.assertEqual(value_at(points, 4 * BAR), 0.)
        self.assertIn((8 * BAR + PPQ, -1.), points)
        self.assertNotEqual(value_at(points, 10 * BAR), value_at(points, 14 * BAR))

    def test_disabled_or_zero_families_generate_nothing(self):
        zero = plan_section_motion(analysis(), {'intensities': dict.fromkeys(FAMILIES, 0)})
        disabled = plan_section_motion(analysis(), {'enabled': dict.fromkeys(FAMILIES, False)})
        self.assertEqual(zero['curves'], [])
        self.assertEqual(disabled['curves'], [])

    def test_every_audible_instrument_and_drum_requires_matching_midi(self):
        song = analysis([stem(1, 'violin', midi=False), stem(2, 'hat', midi=False),
                         stem(3, 'kick', midi=False), stem(4, 'vocals', midi=False)])
        with self.assertRaisesRegex(ValueError, 'hat 2.*kick 3.*violin 1'):
            plan_section_motion(song)

    def test_verified_shared_drums_need_no_fake_individual_note_ownership(self):
        song = shared_guide(analysis([stem(1, 'hat', midi=False), stem(2, 'kick', midi=False)]), [1, 2])
        plan = plan_section_motion(song, {'intensities': {'filter': 100}})
        self.assertEqual([(curve['stem_id'], curve['family']) for curve in plan['curves']],
                         [(1, 'filter'), (1, 'volume'), (1, 'pan')])
        self.assertTrue(any(row['stem_id'] == 2 and 'stable' in row['reason'] for row in plan['skipped']))
        self.assertTrue(all('note_events' not in source for source in song['stems']))

    def test_shared_guide_does_not_move_a_drum_during_its_own_long_silence(self):
        song = shared_guide(analysis([stem(1, 'hat', midi=False,
                                         spans=[[0, 8 * BAR], [16 * BAR, END]])]), [1])
        plan = plan_section_motion(song, {'intensities': {'filter': 100}})
        points = lane(plan, 'filter')['points']
        self.assertEqual(value_at(points, 12 * BAR), 0.)
        self.assertNotEqual(value_at(points, 18 * BAR), value_at(points, 22 * BAR))

    def test_shared_flag_alone_or_nonmember_never_bypasses_midi_requirement(self):
        song = analysis([stem(1, 'hat', midi=False), stem(2, 'ride', midi=False)])
        song['stems'][0]['midi_guide_verified'] = True
        with self.assertRaisesRegex(ValueError, 'Matching exported MIDI'):
            plan_section_motion(song)
        shared_guide(song, [1])
        with self.assertRaisesRegex(ValueError, 'ride 2'):
            plan_section_motion(song)

    def test_shared_drum_guide_never_exempts_bass_or_melodic_stems(self):
        for role in ('bass', 'synth', 'pads'):
            song = shared_guide(analysis([stem(1, role, midi=False)]), [1])
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, 'supported drum stems'):
                plan_section_motion(song)

    def test_verified_silent_drum_needs_no_shared_membership_and_gets_no_curve(self):
        song = shared_guide(analysis([stem(1, 'hat', midi=False, spans=[]),
                                     stem(2, 'ride', midi=False)]), [2])
        plan = plan_section_motion(song)
        self.assertFalse(any(curve['stem_id'] == 1 for curve in plan['curves']))
        self.assertTrue(any(row['stem_id'] == 1 and 'silent' in row['reason'] for row in plan['skipped']))

    def test_source_flp_only_is_not_midi_and_forged_notes_are_not_a_match(self):
        source = stem()
        source.update(midi_verified=False, timing_provenance='source-flp',
                      note_guide_source='Source FLP: Original.flp — Pads')
        with self.assertRaisesRegex(ValueError, 'Matching exported MIDI'):
            plan_section_motion(analysis([source]))

    def test_real_midi_match_survives_legacy_source_flp_notes(self):
        source = stem()
        source.update(midi_verified=False, timing_provenance='source-flp', note_events=[])
        song = analysis([source])
        song['midi_guide'] = {'matches': {'1': {'note_events': [dict(tick=0, duration=END)]}}}
        self.assertTrue(plan_section_motion(song)['curves'])

    def test_verified_silent_stem_is_exempt_but_empty_activity_is_not_proof(self):
        silent = stem(midi=False, spans=[])
        plan = plan_section_motion(analysis([silent]))
        self.assertEqual(plan['curves'], [])
        self.assertIn('silent', plan['skipped'][0]['reason'])
        silent['analysis']['level_windows'][0]['rms'] = .1
        silent['analysis']['level_windows'][0]['peak'] = .2
        with self.assertRaisesRegex(ValueError, 'Matching exported MIDI'):
            plan_section_motion(analysis([silent]))

    def test_partial_zero_windows_cannot_certify_silent_audio(self):
        source = stem(midi=False, spans=[])
        source['analysis']['level_windows'][0]['end_seconds'] = .1
        with self.assertRaisesRegex(ValueError, 'Matching exported MIDI'):
            plan_section_motion(analysis([source]))

    def test_low_level_audio_cannot_become_a_silence_exemption(self):
        source = stem(midi=False, spans=[])
        source['analysis']['level_windows'][0].update(rms=1e-9, peak=1e-8)
        with self.assertRaisesRegex(ValueError, 'Matching exported MIDI'):
            plan_section_motion(analysis([source]))

    def test_zero_duration_drum_note_counts_as_midi_proof(self):
        source = stem(role='kick')
        source['note_events'] = [dict(tick=10, duration=0)]
        self.assertEqual(plan_section_motion(analysis([source]))['curves'], [])

    def test_vocals_and_effects_can_use_audio_only(self):
        plan = plan_section_motion(analysis([stem(1, 'vocals', midi=False),
                                            stem(2, 'effects', midi=False)]), {'reverb_mode': 'add'})
        self.assertTrue(plan['curves'])
        self.assertNotIn('width', [c['family'] for c in plan['curves'] if c['stem_id'] == 1])

    def test_protected_fx_preserves_processing_without_changing_other_curves(self):
        source = stem(2, 'effects', midi=False)
        song = analysis([stem(1, 'pads'), source])
        before = plan_section_motion(song, {'seed': 73})
        self.assertEqual({c['family'] for c in before['curves'] if c['stem_id'] == 2},
                         {'filter', 'width', 'volume', 'pan'})
        source['automation_protected'] = True
        protected = plan_section_motion(song, {'seed': 73})
        self.assertFalse(any(c['stem_id'] == 2 for c in protected['curves']))
        self.assertEqual([c for c in before['curves'] if c['stem_id'] == 1],
                         protected['curves'])
        self.assertEqual(before['sections'], protected['sections'])
        self.assertEqual(before['effective_intensities'], protected['effective_intensities'])
        self.assertIn({'stem_id': 2, 'stem_name': 'effects 2',
                       'reason': 'Preserve this protected stem and its existing processing.'},
                      protected['skipped'])
        source['automation_protected'] = False
        self.assertEqual(before, plan_section_motion(song, {'seed': 73}))

    def test_protection_never_bypasses_required_midi_matching(self):
        source = stem(role='guitar', midi=False)
        source['automation_protected'] = True
        with self.assertRaisesRegex(ValueError, 'Matching exported MIDI.*guitar 1'):
            plan_section_motion(analysis([source]))

    def test_protection_flag_requires_a_boolean(self):
        for flag in (0, 1, 'true', None, [], {}):
            source = stem(role='effects', midi=False)
            source['automation_protected'] = flag
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'automation_protected'):
                plan_section_motion(analysis([source]))

    def test_kick_sub_and_primary_drum_hits_remain_stable(self):
        roles = ('kick', 'bass', 'sub', 'snare', 'clap', 'tom', 'crash')
        song = analysis([stem(i + 1, role) for i, role in enumerate(roles)])
        self.assertEqual(plan_section_motion(song, {'reverb_mode': 'add'})['curves'], [])

    def test_percussion_only_gets_filter_fx_at_evidenced_rise(self):
        plan = plan_section_motion(analysis([stem(role='hat')]), {'intensities': {'filter': 100}})
        self.assertEqual([c['family'] for c in plan['curves'] if c['family'] not in ('volume', 'pan')],
                         ['filter'])
        points = plan['curves'][0]['points']
        self.assertEqual(min(v for _, v in points), -1.)
        self.assertTrue(any(b - a >= 7 * BAR and v < w
                            for (a, v), (b, w) in zip(points, points[1:])))
        song = analysis([stem(role='hat')])
        for row in song['sections']:
            row['energy'] = .8
        self.assertEqual(plan_section_motion(song)['curves'], [])

    def test_inaudible_percussion_rise_does_not_reduce_the_user_range(self):
        song = analysis([stem(role='hat', spans=[[0, 16 * BAR]])])
        for section, energy in zip(song['sections'], (.2, .45, 0., 1.)):
            section['energy'] = energy
        points = lane(plan_section_motion(song, {'intensities': {'filter': 100}}), 'filter')['points']
        self.assertEqual(min(v for _, v in points), -1.)
        self.assertEqual(value_at(points, 20 * BAR), 0.)

    def test_no_width_for_mono_or_unknown_channel_layout(self):
        for channels in (1, None):
            plan = plan_section_motion(analysis([stem(channels=channels)]))
            self.assertNotIn('width', [c['family'] for c in plan['curves']])

    def test_no_section_or_constant_energy_means_no_invented_swells(self):
        song = analysis()
        for row in song['sections']:
            row['energy'] = .7
        self.assertEqual(plan_section_motion(song)['curves'], [])
        song['sections'] = []
        self.assertEqual(plan_section_motion(song)['curves'], [])

    def test_section_names_do_not_change_rules(self):
        first = analysis()
        second = deepcopy(first)
        for row in second['sections']:
            row.update(name='chorus drop bridge', type='drop')
        self.assertEqual(plan_section_motion(first)['curves'], plan_section_motion(second)['curves'])

    def test_dense_notes_do_not_generate_per_note_motion(self):
        source = stem(role='synth')
        source['note_events'] = [dict(tick=t, duration=12) for t in range(0, END, 24)]
        plan = plan_section_motion(analysis([source]))
        self.assertNotIn('delay', [c['family'] for c in plan['curves']])
        self.assertTrue(all(len(c['points']) < 14 for c in plan['curves']))

    def test_delay_requires_real_gap_near_existing_boundary(self):
        source = stem(role='guitar', spans=[[0, 7 * BAR], [9 * BAR, END]])
        plan = plan_section_motion(analysis([source]))
        delay = lane(plan, 'delay')
        nonzero = [(t, v) for t, v in delay['points'] if v]
        self.assertTrue(nonzero)
        self.assertTrue(all(6 * BAR <= t <= 8 * BAR for t, _ in nonzero))
        source['note_events'] = [dict(tick=0, duration=END)]
        self.assertNotIn('delay', [c['family'] for c in plan_section_motion(analysis([source]))['curves']])

    def test_silent_section_target_is_baseline(self):
        source = stem(spans=[[0, 8 * BAR], [16 * BAR, END]])
        plan = plan_section_motion(analysis([source]))
        for curve in plan['curves']:
            self.assertEqual(value_at(curve['points'], 8 * BAR), 0.)
            self.assertEqual(value_at(curve['points'], 12 * BAR), 0.)

    def test_inputs_are_not_mutated(self):
        song, settings = analysis(), default_settings(9)
        original_song, original_settings = deepcopy(song), deepcopy(settings)
        plan_section_motion(song, settings)
        self.assertEqual(song, original_song)
        self.assertEqual(settings, original_settings)

    def test_invalid_timing_and_audio_are_rejected(self):
        changes = [('ppq', 0), ('end_tick', 0), ('bpm', float('nan')),
                   ('time_signature', [3, 4]), ('tempo_changes', [120, 130])]
        for key, value in changes:
            song = analysis()
            song[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                plan_section_motion(song)
        song = analysis()
        song['sections'][1]['start_tick'] += 1
        with self.assertRaises(ValueError):
            plan_section_motion(song)
        song = analysis()
        song['stems'][0]['analysis']['level_windows'][0]['rms'] = float('nan')
        with self.assertRaises(ValueError):
            plan_section_motion(song)


class NativeMixMotionTests(unittest.TestCase):
    @staticmethod
    def settings(**changes):
        result = {'enabled': {family: family in ('volume', 'pan') for family in FAMILIES},
                  'intensities': {'volume': 100, 'pan': 100}}
        result.update(changes)
        return result

    @staticmethod
    def levels(source, values):
        scale = BPM * PPQ / 60
        source['analysis']['level_windows'] = [
            dict(start_seconds=index * 8 * BAR / scale,
                 end_seconds=(index + 1) * 8 * BAR / scale, rms=value, peak=value * 2)
            for index, value in enumerate(values)]
        return source

    def test_existing_effect_curves_match_pre_mix_feature_snapshots(self):
        fixtures = [analysis(), analysis([
            stem(1, 'vocals', midi=False, spans=[[0, 7 * BAR], [9 * BAR, END]]),
            stem(2, 'hat'), stem(3, 'synth')]), analysis([
            stem(1, 'effects', midi=False, spans=[[0, 7 * BAR], [18 * BAR, END]]),
            stem(2, 'guitar')])]
        expected = ['50011f20471941e8c3007ab75f4516b0a6aaaa3935febd1e82766f6a1c6f1860',
                    '9914c091e0b8b34d191ffd09a4ffee1862219c68597960a164ed02d9576ae334',
                    'fc21012bebed33a3351613d95b258b506dc75de6a6604e5f080de909eec4ee1b']
        for song, seed, checksum in zip(fixtures, (0, 19, 817), expected):
            with self.subTest(seed=seed):
                curves = [curve for curve in plan_section_motion(song, {
                    'seed': seed, 'reverb_mode': 'add'})['curves']
                          if curve['family'] not in ('volume', 'pan')]
                actual = hashlib.sha256(json.dumps(curves, sort_keys=True).encode()).hexdigest()
                self.assertEqual(actual, checksum)

    def test_volume_is_reduction_only_with_role_specific_caps(self):
        for role, cap in (('pads', 3.), ('effects', 3.), ('vocals', 1.5), ('hat', 1.5)):
            source = self.levels(stem(role=role), (.025, .4, .025, .4))
            with self.subTest(role=role):
                curve = lane(plan_section_motion(analysis([source]), self.settings()), 'volume')
                self.assertEqual(curve['parameter'], 'volume_db')
                self.assertEqual(min(value for _, value in curve['points']), -cap)
                self.assertTrue(all(-cap <= value <= 0 for _, value in curve['points']))
                self.assertEqual(curve['points'][0], (0, 0.))
                self.assertEqual(curve['points'][-1], (END, 0.))

    def test_volume_uses_each_stems_own_active_rms(self):
        sources = [self.levels(stem(1), (.05, .3, .05, .3)),
                   self.levels(stem(2), (.3, .3, .3, .3))]
        song = analysis(sources)
        for section in song['sections']:
            section['energy'] = .7
        plan = plan_section_motion(song, self.settings())
        self.assertEqual([(c['stem_id'], c['family']) for c in plan['curves']], [(1, 'volume')])
        self.assertEqual(value_at(lane(plan, 'volume')['points'], 8 * BAR), -3.)

    def test_silence_is_excluded_from_section_rms(self):
        source = stem(spans=[[0, 2 * BAR], [8 * BAR, END]])
        scale = BPM * PPQ / 60
        source['analysis']['level_windows'] = [
            dict(start_seconds=a / scale, end_seconds=b / scale, rms=rms, peak=rms * 2)
            for a, b, rms in ((0, 2 * BAR, .2), (2 * BAR, 8 * BAR, 0), (8 * BAR, END, .2))]
        song = analysis([source])
        for section in song['sections']:
            section['energy'] = .6
        # First section contains six silent bars, yet its playing level matches
        # the others. No false level ride is justified by that missing audio.
        self.assertEqual(plan_section_motion(song, self.settings())['curves'], [])

    def test_volume_never_invents_level_evidence_when_only_activity_exists(self):
        source = stem()
        source['analysis'].pop('level_windows')
        plan = plan_section_motion(analysis([source]), self.settings())
        self.assertNotIn('volume', [curve['family'] for curve in plan['curves']])
        self.assertIn('pan', [curve['family'] for curve in plan['curves']])

    def test_native_lines_are_slow_sparse_and_baseline_during_long_silence(self):
        source = self.levels(stem(spans=[[0, 8 * BAR], [18 * BAR, END]]), (.1, .2, .3, .2))
        song = analysis([source])
        plan = plan_section_motion(song, self.settings())
        self.assertEqual({curve['family'] for curve in plan['curves']}, {'volume', 'pan'})
        for curve in plan['curves']:
            with self.subTest(family=curve['family']):
                points = curve['points']
                self.assertLessEqual(len(points), len(song['sections']) + 10)
                self.assertTrue(all(math.isfinite(value) for _, value in points))
                self.assertTrue(all(a < b for (a, _), (b, _) in zip(points, points[1:])))
                self.assertEqual(value_at(points, 8 * BAR), 0.)
                self.assertEqual(value_at(points, 13 * BAR), 0.)
                self.assertEqual(value_at(points, 18 * BAR), 0.)
                for (a, first), (b, last) in zip(points, points[1:]):
                    if first != last:
                        self.assertGreaterEqual(b - a, BAR)
                        self.assertFalse(8 * BAR <= a < b <= 18 * BAR)

    def test_brief_isolated_sounds_do_not_get_native_pulses(self):
        source = stem(role='effects', midi=False,
                      spans=[[BAR, BAR + PPQ], [12 * BAR, 12 * BAR + PPQ]])
        self.assertEqual(plan_section_motion(analysis([source]), self.settings())['curves'], [])

    def test_mix_amounts_scale_same_geometry_and_switch_independently(self):
        high = plan_section_motion(analysis(), self.settings())
        low = plan_section_motion(analysis(), self.settings(intensities={'volume': 20, 'pan': 20}))
        for family in ('volume', 'pan'):
            for (tick, value), (other_tick, other_value) in zip(
                    lane(high, family)['points'], lane(low, family)['points']):
                self.assertEqual(tick, other_tick)
                self.assertAlmostEqual(other_value, value * .2)
            settings = self.settings()
            settings['enabled'][family] = False
            plan = plan_section_motion(analysis(), settings)
            self.assertNotIn(family, [curve['family'] for curve in plan['curves']])
            self.assertEqual(len(plan['curves']), 1)
            settings['enabled'][family] = True
            settings['intensities'][family] = 0
            self.assertEqual(plan, plan_section_motion(analysis(), settings))

    def test_pan_is_light_identity_based_and_seed_cannot_change_side(self):
        song = analysis([stem(index) for index in range(1, 9)])
        settings = self.settings()
        first = plan_section_motion(song, settings)
        second = plan_section_motion(song, self.settings(seed=438))
        pans = [curve for curve in first['curves'] if curve['family'] == 'pan']
        self.assertEqual(first['curves'], second['curves'])
        directions = set()
        for curve in pans:
            self.assertEqual(curve['parameter'], 'pan_percent')
            values = [value for _, value in curve['points'] if value]
            direction = math.copysign(1, values[0])
            directions.add(direction)
            self.assertTrue(all(math.copysign(1, value) == direction for value in values))
            self.assertEqual(max(abs(value) for value in values), 8.)
        self.assertEqual(directions, {-1., 1.})

    def test_pan_leaves_vocals_foundations_and_named_leads_at_saved_position(self):
        roles = ('vocals', 'kick', 'bass', 'sub', 'snare', 'clap', 'tom', 'crash')
        sources = [stem(index, role) for index, role in enumerate(roles, 1)]
        for index, name in enumerate(('Lead synth', 'main keys', 'Guitar solo'), 20):
            source = stem(index, 'synth')
            source['name'] = name
            sources.append(source)
        plan = plan_section_motion(analysis(sources), self.settings())
        self.assertNotIn('pan', [curve['family'] for curve in plan['curves']])
        self.assertFalse(any(curve['stem_id'] in range(2, 9) for curve in plan['curves']))
        self.assertTrue(any(curve['stem_id'] == 20 and curve['family'] == 'volume'
                            for curve in plan['curves']))


if __name__ == '__main__':
    unittest.main()
