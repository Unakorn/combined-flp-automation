"""Headless arrangement evidence and boundary tests; no FL or desktop control."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'work'))
from section_structure import detect_section_structure

PPQ, BPM = 96, 120
BAR = PPQ * 4
SCALE = BPM * PPQ / 60


def stem(identifier, notes=(), *, bars=24, midi=True, offset=0, rms=.2):
    end = round(bars * BAR)
    return dict(id=identifier, name=f'Part {identifier}', midi_verified=midi,
                position=offset, note_events=[dict(tick=a, duration=b, note=pitch)
                                             for a, b, pitch in notes],
                analysis=dict(channels=2, chunks=[dict(start_seconds=0,
                    end_seconds=end / SCALE, rms=rms)]))


def pattern(start, stop, hits=4, *, length=12):
    return [(bar * BAR + hit * BAR // hits, length, 60)
            for bar in range(start, stop) for hit in range(hits)]


def sections(stems, bars=24, markers=None, shared_drum_guide=None):
    return detect_section_structure(stems, bpm=BPM, ppq=PPQ,
                                    end_tick=round(bars * BAR), markers=markers,
                                    shared_drum_guide=shared_drum_guide)


def shared_guide(stems, notes, track=2):
    for source in stems:
        source.update(role='hat', midi_verified=False, midi_guide_verified=True,
                      timing_provenance='shared-drum-guide', midi_file_track_index=track)
        source.pop('note_events', None)
    return dict(midi_track='DRUMS', midi_file_track_index=track,
                verified=True, timing_provenance='shared-drum-guide',
                stem_ids=[source['id'] for source in stems],
                note_events=[dict(tick=a, duration=b, note=pitch) for a, b, pitch in notes])


class SectionStructureTests(unittest.TestCase):
    def test_equal_rms_midi_entries_and_exits_create_non_eight_bar_boundaries(self):
        result = sections([stem(1, [(0, 24 * BAR, 60)]),
                           stem(2, [(6 * BAR, 6 * BAR, 64)])])
        self.assertEqual([row['start_tick'] for row in result], [0, 6 * BAR, 12 * BAR])
        self.assertGreater(result[1]['energy'], result[0]['energy'] + .2)
        self.assertGreater(result[1]['energy'], result[2]['energy'] + .2)
        self.assertEqual(result[1]['source'], 'MIDI arrangement')

    def test_persistent_note_density_change_creates_boundary_at_equal_audio_level(self):
        result = sections([stem(1, pattern(0, 12, 1) + pattern(12, 24, 4))])
        self.assertEqual([row['start_tick'] for row in result], [0, 12 * BAR])
        self.assertGreater(result[1]['energy'], result[0]['energy'] + .3)
        self.assertGreater(result[1]['evidence']['midi_density'], result[0]['evidence']['midi_density'])

    def test_two_bar_break_preserves_both_edges_only_for_multiple_restored_parts(self):
        notes = pattern(0, 6) + pattern(8, 24)
        result = sections([stem(1, notes), stem(2, notes), stem(3, pattern(0, 24))])
        self.assertEqual([row['start_tick'] for row in result], [0, 6 * BAR, 8 * BAR])
        self.assertEqual([row['source'] for row in result[1:]], ['MIDI break start', 'MIDI break return'])
        one_part = sections([stem(1, notes), stem(2, pattern(0, 24))])
        self.assertFalse(any(row['source'].startswith('MIDI break') for row in one_part))
        density_only = pattern(0, 6, 4) + pattern(6, 8, 1) + pattern(8, 24, 4)
        filled = sections([stem(1, density_only), stem(2, density_only)])
        self.assertFalse(any(row['source'].startswith('MIDI break') for row in filled))

    def test_repeated_pattern_and_one_bar_fill_do_not_create_sections(self):
        repeated = pattern(0, 24)
        self.assertEqual(len(sections([stem(1, repeated)])), 1)
        fill = pattern(0, 10) + pattern(10, 11, 16) + pattern(11, 24)
        self.assertEqual(len(sections([stem(1, fill)])), 1)

    def test_chord_voicing_and_pitch_changes_do_not_count_as_density_changes(self):
        notes = pattern(0, 24, 1)
        notes += [(bar * BAR, 12, pitch) for bar in range(12, 24) for pitch in (64, 67, 72)]
        result = sections([stem(1, notes)])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['evidence']['midi_density'], 1.)

    def test_sustained_note_remains_active_without_repeated_attacks(self):
        result = sections([stem(1, [(0, 30 * BAR, 60)])])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['evidence']['midi_activity'], 1.)
        self.assertEqual(result[0]['end_tick'], 24 * BAR)

    def test_zero_duration_drums_still_supply_density_and_activity(self):
        result = sections([stem(1, pattern(0, 12, 1, length=0) + pattern(12, 24, 8, length=0))])
        self.assertEqual([row['start_tick'] for row in result], [0, 12 * BAR])
        self.assertTrue(all(row['evidence']['midi_activity'] == 1. for row in result))

    def test_dense_drum_lane_cannot_outvote_other_parts_by_raw_note_count(self):
        pad = stem(2, [(8 * BAR, 16 * BAR, 64)])
        slow = sections([stem(1, pattern(0, 24, 4)), pad])
        dense = sections([stem(1, pattern(0, 24, 128)), pad])
        self.assertEqual(slow, dense)
        self.assertEqual([row['start_tick'] for row in dense], [0, 8 * BAR])

    def test_arbitrary_named_markers_survive_and_can_be_closer_than_four_bars(self):
        marks = [dict(tick=5 * BAR + 20, name='Part A'), dict(tick=7 * BAR, name='Air'),
                 dict(tick=9 * BAR, name='Loop endpoint', kind=8)]
        result = sections([stem(1, pattern(0, 24))], markers=marks)
        self.assertEqual([row['start_tick'] for row in result], [0, 5 * BAR, 7 * BAR])
        self.assertEqual([row['marker_name'] for row in result[1:]], ['Part A', 'Air'])
        self.assertTrue(all(row['source'] == 'project marker' for row in result[1:]))

    def test_audio_offset_and_partial_ending_are_clipped_without_shifting_midi(self):
        source = stem(1, [(4 * BAR, 20 * BAR, 60)], bars=20, offset=4 * BAR)
        result = sections([source], bars=12.5)
        self.assertEqual([row['start_tick'] for row in result], [0, 4 * BAR])
        self.assertEqual(result[-1]['end_tick'], int(12.5 * BAR))
        for left, right in zip(result, result[1:]):
            self.assertEqual(left['end_tick'], right['start_tick'])
        self.assertEqual(result[0]['evidence']['midi_activity'], 0.)
        self.assertEqual(result[-1]['evidence']['midi_activity'], 1.)

    def test_input_order_is_irrelevant_and_nothing_is_mutated(self):
        stems = [stem(1, pattern(0, 24, 4)), stem(2, [(8 * BAR, 8 * BAR, 60)]),
                 stem(3, pattern(0, 24, 1), midi=False)]
        markers = [dict(tick=4 * BAR, name='B'), dict(tick=4 * BAR, name='A')]
        original = deepcopy((stems, markers))
        first = sections(stems, markers=markers)
        self.assertEqual(first, sections(list(reversed(stems)), markers=list(reversed(markers))))
        self.assertEqual((stems, markers), original)

    def test_audio_only_fallback_and_unverified_midi_do_not_forge_note_evidence(self):
        source = stem(1, pattern(0, 12, 1) + pattern(12, 24, 8), midi=False)
        self.assertEqual(len(sections([source])), 1)
        source['analysis']['chunks'] = [dict(start_seconds=0, end_seconds=8 * BAR / SCALE, rms=.1),
            dict(start_seconds=8 * BAR / SCALE, end_seconds=24 * BAR / SCALE, rms=.5)]
        result = sections([source])
        self.assertEqual([row['start_tick'] for row in result], [0, 8 * BAR])
        self.assertTrue(all(row['evidence']['verified_midi_stems'] == 0 for row in result))
        self.assertEqual(result[1]['source'], 'audio energy')
        self.assertAlmostEqual(result[0]['energy'], .2)
        self.assertAlmostEqual(result[1]['energy'], 1.)

    def test_shared_group_is_counted_once_regardless_of_member_count(self):
        notes = pattern(0, 12, 1) + pattern(12, 24, 8)
        members = [stem(i, midi=False, rms=0) for i in range(1, 6)]
        group = shared_guide(members, notes)
        many = sections(members, shared_drum_guide=group)
        one_group = deepcopy(group)
        one_group['stem_ids'] = [1]
        one = sections(members[:1], shared_drum_guide=one_group)
        self.assertEqual(many, one)
        self.assertEqual([row['start_tick'] for row in many], [0, 12 * BAR])
        self.assertTrue(all(row['evidence']['shared_drum_guides'] == 1 for row in many))
        self.assertTrue(all(row['evidence']['verified_midi_stems'] == 0 for row in many))
        self.assertTrue(all(row['evidence']['midi_parts'] == 1 for row in many))

    def test_exact_subset_and_whole_group_do_not_double_count_drum_arrangement(self):
        notes = pattern(0, 6) + pattern(8, 24)
        member = stem(1, midi=False, rms=0)
        group = shared_guide([member], notes)
        exact = stem(2, notes, rms=0)
        exact.update(role='snare', midi_file_track_index=2)
        grouped = sections([member, exact], shared_drum_guide=group)
        single = sections([member], shared_drum_guide=group)
        self.assertEqual(grouped, single)
        self.assertFalse(any(row['source'].startswith('MIDI break') for row in grouped))

    def test_several_shared_members_do_not_invent_a_multi_part_break(self):
        members = [stem(i, midi=False) for i in (1, 2, 3)]
        group = shared_guide(members, pattern(0, 6) + pattern(8, 24))
        result = sections(members, shared_drum_guide=group)
        self.assertFalse(any(row['source'].startswith('MIDI break') for row in result))
        self.assertTrue(all(row['evidence']['midi_parts'] == 1 for row in result))

    def test_shared_guide_and_an_independent_part_can_prove_real_break(self):
        notes = pattern(0, 6) + pattern(8, 24)
        member = stem(1, midi=False)
        group = shared_guide([member], notes)
        melodic = stem(2, notes)
        melodic.update(role='synth', midi_file_track_index=3)
        result = sections([member, melodic], shared_drum_guide=group)
        self.assertEqual([row['source'] for row in result[1:]], ['MIDI break start', 'MIDI break return'])
        self.assertTrue(all(row['evidence']['midi_parts'] == 2 for row in result))

    def test_shared_guide_member_identity_and_timeline_are_validated(self):
        member = stem(1, midi=False)
        group = shared_guide([member], pattern(0, 24))
        original = deepcopy((member, group))
        sections([member], shared_drum_guide=group)
        self.assertEqual((member, group), original)
        for change in ({'stem_ids': [99]}, {'verified': False}, {'stem_ids': [1, 1]},
                       {'note_events': []}, {'note_events': [dict(tick=24 * BAR, duration=1)]},
                       {'note_events': [dict(tick=0, duration=1, note=128)]},
                       {'note_events': [dict(tick=0, duration=-1, note=60)]},
                       {'note_events': [dict(tick=True, duration=1, note=60)]},
                       {'note_events': [dict(tick=0, duration=1, note=60, midi_channel=16)]}):
            bad = dict(group, **change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'Shared drum section guide'):
                sections([member], shared_drum_guide=bad)


if __name__ == '__main__':
    unittest.main()
