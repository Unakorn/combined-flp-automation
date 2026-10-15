"""Strict identity, exported-MIDI precedence and drum-map boundaries."""
from copy import deepcopy
from pathlib import Path
import sys
import shutil
import unittest
import tempfile
from unittest.mock import patch
from uuid import uuid4

from mido import Message, MetaMessage, MidiFile, MidiTrack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'work'))
import section_inputs as inputs


class SectionInputsTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.gettempdir()) / ('section-input-test-' + uuid4().hex)
        self.folder.mkdir()
        self.addCleanup(shutil.rmtree, self.folder)

    def stems(self, *pairs, silent=False, position=0, duration=1536):
        audio = dict(duration_seconds=8, level_windows=[
            dict(start_seconds=0, end_seconds=8, peak=0 if silent else .3)],
            chunks=[dict(start_seconds=0, end_seconds=8, rms=0 if silent else .1)])
        stems = [dict(id=i, name=name, role=role, position=position, duration=duration,
                      analysis=deepcopy(audio), phrase_ends=[100], timing_source='audio activity / silence')
                 for i, (name, role) in enumerate(pairs)]
        return dict(stems=stems, warnings=[], markers=[], ppq=96, bpm=120,
                    end_tick=position+duration, sections=[], path='unused.flp', sha256='unused')

    def midi(self, *tracks, ppq=96, tempo=500000):
        path = self.folder / 'full-song.mid'
        midi = MidiFile(type=1, ticks_per_beat=ppq)
        midi.tracks.append(MidiTrack([MetaMessage('set_tempo', tempo=tempo)]))
        for name, notes in tracks:
            track = MidiTrack([MetaMessage('track_name', name=name)])
            events = []
            for start, duration, pitch, channel in notes:
                events.extend([(start, Message('note_on', channel=channel, note=pitch, velocity=73)),
                               (start+duration, Message('note_off', channel=channel, note=pitch))])
            previous = 0
            for tick, message in sorted(events, key=lambda item: item[0]):
                track.append(message.copy(time=tick-previous))
                previous = tick
            midi.tracks.append(track)
        midi.save(path)
        return path

    def analyze(self, base, **kwargs):
        # Exercise actual timing readers and matching. Audio DSP and FLP
        # structural validation have their own integration tests.
        with patch.object(inputs, 'analyze_stems', return_value=deepcopy(base)), \
             patch.object(inputs, 'detect_sections', return_value=[]):
            return inputs.analyze_section_inputs(base, **kwargs)

    def source(self, notes, stem_id=0):
        path = self.folder / 'source.flp'
        path.write_bytes(b'source fixture, read_source_timing mocked')
        result = dict(matches={stem_id: dict(source_track='lead', note_events=notes)},
                      warnings=[], markers=[], path=str(path))
        return path, result

    def test_midi_primary_when_original_contains_older_notes(self):
        base = self.stems(('lead', 'synth'))
        midi = self.midi(('lead', [(96, 48, 64, 2)]))
        source, guide = self.source([dict(tick=0, duration=12, note=40, velocity=100)])
        with patch.object(inputs, 'read_source_timing', return_value=guide):
            result = self.analyze(base, midi_path=midi, source_flp=source)
        self.assertFalse(result['match_errors'])
        stem = result['stems'][0]
        self.assertTrue(stem['midi_verified'])
        self.assertEqual(stem['note_events'], [dict(tick=96, duration=48, note=64,
                                                  velocity=73, midi_channel=2)])
        self.assertEqual(stem['timing_provenance'], 'midi')
        self.assertEqual(len(result['timing_files']), 2)

    def test_source_only_never_satisfies_required_exported_midi(self):
        base = self.stems(('lead', 'synth'))
        source, guide = self.source([dict(tick=0, duration=24, note=60)])
        with patch.object(inputs, 'read_source_timing', return_value=guide):
            result = self.analyze(base, source_flp=source)
        self.assertIn('exported MIDI is required', result['match_errors'][0])
        self.assertNotIn('note_events', result['stems'][0])

    def test_source_cannot_fill_missing_or_changed_midi_part(self):
        base = self.stems(('lead', 'synth'))
        midi = self.midi(('other', [(96, 24, 62, 0)]))
        source, guide = self.source([dict(tick=0, duration=24, note=60)])
        with patch.object(inputs, 'read_source_timing', return_value=guide):
            result = self.analyze(base, midi_path=midi, source_flp=source)
        self.assertTrue(result['match_errors'])
        self.assertEqual(result['midi_guide']['matches'], {})

    def test_fullsong_ticks_scale_precisely_without_adding_stem_offset(self):
        base = self.stems(('lead', 'synth'), position=24)
        midi = self.midi(('lead', [(120, 240, 60, 0)]), ppq=480)
        stem = self.analyze(base, midi_path=midi)['stems'][0]
        self.assertTrue(stem['midi_verified'])
        self.assertEqual(stem['note_events'][0]['tick'], 24)
        self.assertEqual(stem['note_events'][0]['duration'], 48)

    def test_verified_midi_and_both_marker_sources_reach_section_detector(self):
        bar, end = 384, 24 * 384
        base = self.stems(('melody', 'synth'), duration=end)
        base['markers'] = [dict(tick=8 * bar, name='Part B')]
        base['stems'][0]['analysis'] = dict(chunks=[dict(start_seconds=0, end_seconds=end / 192, rms=.2)])
        notes = [(index * bar + hit * bar // hits, 12, 60, 0)
                 for index in range(24) for hits in (1 if index < 12 else 4,)
                 for hit in range(hits)]
        midi_path = self.midi(('melody', notes))
        midi = MidiFile(midi_path)
        midi.tracks[0].insert(1, MetaMessage('marker', text='Air', time=4 * bar))
        midi.save(midi_path)
        with patch.object(inputs, 'analyze_stems', return_value=deepcopy(base)):
            result = inputs.analyze_section_inputs(base, midi_path=midi_path)
        self.assertFalse(result['match_errors'])
        self.assertEqual([part['start_tick'] for part in result['sections']], [0, 4 * bar, 8 * bar, 12 * bar])
        self.assertEqual([part['marker_name'] for part in result['sections'][1:3]], ['Air', 'Part B'])
        self.assertEqual(result['sections'][-1]['source'], 'MIDI arrangement')
        self.assertGreater(result['sections'][-1]['energy'], result['sections'][0]['energy'])

    def test_wrong_relative_export_is_rejected_instead_of_shifted(self):
        base = self.stems(('lead', 'synth'), position=384)
        midi = self.midi(('lead', [(0, 48, 60, 0)]))
        result = self.analyze(base, midi_path=midi)
        self.assertIn('same song start', result['match_errors'][0])
        self.assertFalse(result['stems'][0]['midi_verified'])

    def test_vocals_effects_exempt_but_consolidated_name_not_vocal(self):
        result = self.analyze(self.stems(('voice', 'vocals'), ('riser', 'effects'),
                                        ('Insert 16 (consolidated)', 'synth')))
        self.assertEqual(len(result['match_errors']), 1)
        self.assertIn('Insert 16', result['match_errors'][0])
        self.assertFalse(result['stems'][0]['midi_required'])

    def test_silent_exemption_requires_zero_peaks_and_complete_coverage(self):
        base = self.stems(('lead', 'synth'), silent=True)
        self.assertFalse(self.analyze(base)['match_errors'])
        base['stems'][0]['analysis']['level_windows'][0]['end_seconds'] = 4
        self.assertTrue(self.analyze(base)['match_errors'])
        base['stems'][0]['analysis']['level_windows'][0]['end_seconds'] = 8
        base['stems'][0]['analysis']['level_windows'][0]['peak'] = 1e-12
        self.assertTrue(self.analyze(base)['match_errors'])

    def test_custom_combined_drums_share_structure_without_claiming_gm_hits(self):
        base = self.stems(('kick', 'kick'), ('snare', 'snare'))
        midi = self.midi(('DRUMS', [(0, 12, 60, 0), (96, 12, 61, 0)]))
        result = self.analyze(base, midi_path=midi)
        self.assertEqual(result['match_errors'], [])
        self.assertFalse(any(s['midi_verified'] for s in result['stems']))
        self.assertTrue(all(s['midi_guide_verified'] for s in result['stems']))
        self.assertTrue(all(s['timing_provenance'] == 'shared-drum-guide' for s in result['stems']))
        self.assertTrue(all('note_events' not in s for s in result['stems']))
        self.assertEqual(result['shared_drum_guide']['stem_ids'], [0, 1])
        self.assertEqual(len(result['shared_drum_guide']['note_events']), 2)
        self.assertEqual(result['midi_guide']['matches'], {})
        mapped = self.analyze(base, midi_path=midi, midi_mapping={0: 'DRUMS'})
        self.assertIn('explicit MIDI notes/channels', mapped['match_errors'][0])
        self.assertEqual(mapped['shared_drum_guide']['stem_ids'], [1])

    def test_custom_named_combined_guide_is_selected_once_by_file_index(self):
        base = self.stems(('hat A', 'hat'), ('perc A', 'perc'), ('kick A', 'kick'))
        midi = self.midi(('My custom sequencer', [(0, 12, 60, 0), (96, 12, 61, 0)]))
        missing = self.analyze(base, midi_path=midi)
        self.assertEqual(len(missing['match_errors']), 3)
        self.assertTrue(all('Choose Shared drum guide once' in e for e in missing['match_errors']))
        for selector in ({'track_index': 1}, {'track': 'My custom sequencer'}):
            result = self.analyze(base, midi_path=midi, drum_midi_guide=selector)
            self.assertEqual(result['match_errors'], [])
            self.assertEqual(result['shared_drum_guide']['stem_ids'], [0, 1, 2])
            self.assertEqual(result['shared_drum_guide']['midi_file_track_index'], 1)

    def test_automatic_shared_selection_requires_one_combined_track(self):
        base = self.stems(('hat', 'hat'))
        for name in ('Unison Drum Monkey', 'drumkit', 'Drum Kit'):
            midi = self.midi((name, [(0, 12, 60, 0)]))
            self.assertFalse(self.analyze(base, midi_path=midi)['match_errors'])
        midi = self.midi(('DRUMS', [(0, 12, 60, 0)]), ('Drum Kit', [(96, 12, 61, 0)]))
        self.assertTrue(self.analyze(base, midi_path=midi)['match_errors'])
        result = self.analyze(base, midi_path=midi, drum_midi_guide={'track_index': 2})
        self.assertFalse(result['match_errors'])
        self.assertEqual(result['shared_drum_guide']['midi_track'], 'Drum Kit')

    def test_shared_guide_never_satisfies_a_melodic_stem(self):
        base = self.stems(('hat', 'hat'), ('mystery', 'synth'), ('sub', 'bass'), ('voice', 'vocals'))
        midi = self.midi(('DRUMS', [(0, 12, 60, 0)]))
        result = self.analyze(base, midi_path=midi)
        self.assertEqual(result['shared_drum_guide']['stem_ids'], [0])
        self.assertEqual(len(result['match_errors']), 2)
        self.assertTrue(any('mystery' in e for e in result['match_errors']))
        self.assertTrue(any('sub' in e for e in result['match_errors']))
        self.assertFalse(result['stems'][3]['midi_guide_verified'])

    def test_invalid_explicit_shared_guide_is_never_silently_automatic(self):
        base = self.stems(('hat', 'hat'))
        midi = self.midi(('DRUMS', [(0, 12, 60, 0)]))
        for selector in ({}, 'DRUMS', {'track_index': True}, {'track_index': -1},
                         {'track_index': 99}, {'track': 'missing'}, {'track': ''},
                         {'track': 'DRUMS', 'notes': [60]}, {'track': 'DRUMS', 'track_index': 1}):
            with self.subTest(selector=selector), self.assertRaisesRegex(ValueError, '[Ss]hared drum guide'):
                self.analyze(base, midi_path=midi, drum_midi_guide=selector)
        with self.assertRaisesRegex(ValueError, 'full-song exported MIDI'):
            self.analyze(base, drum_midi_guide={'track_index': 1})

    def test_exact_match_and_filtered_mapping_take_precedence_over_shared_fallback(self):
        base = self.stems(('kick', 'kick'), ('snare', 'snare'), ('hat', 'hat'))
        midi = self.midi(('kick', [(0, 12, 36, 0)]),
                         ('DRUMS', [(0, 12, 60, 0), (96, 12, 61, 0)]))
        result = self.analyze(base, midi_path=midi,
                              midi_mapping={1: {'track': 'DRUMS', 'notes': [60]}})
        self.assertEqual(result['match_errors'], [])
        self.assertEqual(set(result['midi_guide']['matches']), {0, 1})
        self.assertEqual(result['shared_drum_guide']['stem_ids'], [2])
        self.assertEqual(len(result['shared_drum_guide']['note_events']), 2)
        self.assertTrue(result['stems'][0]['midi_verified'])
        self.assertTrue(result['stems'][1]['midi_verified'])
        self.assertFalse(result['stems'][2]['midi_verified'])

    def test_explicit_note_ownership_conflicts_still_block_with_shared_guide(self):
        base = self.stems(('hat', 'hat'), ('ride', 'ride'), ('perc', 'perc'))
        midi = self.midi(('DRUMS', [(0, 12, 60, 0), (96, 12, 61, 0)]))
        result = self.analyze(base, midi_path=midi, midi_mapping={
            0: {'track': 'DRUMS', 'notes': [60]}, 1: {'track': 'DRUMS', 'notes': [60]}})
        self.assertEqual(len(result['match_errors']), 2)
        self.assertTrue(all('same MIDI notes' in e for e in result['match_errors']))
        self.assertEqual(result['shared_drum_guide']['stem_ids'], [2])

    def test_silent_drums_are_not_added_to_shared_members(self):
        base = self.stems(('hat', 'hat'), ('ride', 'ride'))
        base['stems'][0]['analysis']['level_windows'][0]['peak'] = 0
        midi = self.midi(('DRUMS', [(0, 12, 60, 0)]))
        result = self.analyze(base, midi_path=midi)
        self.assertEqual(result['match_errors'], [])
        self.assertEqual(result['shared_drum_guide']['stem_ids'], [1])
        self.assertFalse(result['stems'][0]['midi_guide_verified'])

    def test_explicit_custom_pitch_and_channel_mapping_keeps_only_owned_hits(self):
        base = self.stems(('kick', 'kick'), ('snare', 'snare'))
        midi = self.midi(('DRUMS', [(0, 12, 60, 0), (96, 12, 60, 1)]))
        result = self.analyze(base, midi_path=midi, midi_mapping={
            0: dict(track='DRUMS', notes=[60], channels=[0]),
            1: dict(track='DRUMS', notes=[60], channels=[1])})
        self.assertFalse(result['match_errors'])
        self.assertEqual([s['note_events'][0]['tick'] for s in result['stems']], [0, 96])

    def test_source_resolved_layer_can_verify_combined_midi_lane(self):
        base = self.stems(('kick', 'kick'))
        midi = self.midi(('DRUMS', [(0, 12, 60, 0), (96, 12, 61, 0)]))
        source, guide = self.source([dict(tick=0, duration=12, note=60, velocity=128)])
        with patch.object(inputs, 'read_source_timing', return_value=guide):
            result = self.analyze(base, midi_path=midi, source_flp=source)
        self.assertFalse(result['match_errors'])
        self.assertEqual(len(result['stems'][0]['note_events']), 1)
        self.assertEqual(result['stems'][0]['note_events'][0]['velocity'], 73)

    def test_actual_eight_child_layer_proves_custom_combined_export(self):
        from test_stem_layer import layer_fixture, NAMES, END
        source = self.folder / 'eight-child-layer.flp'
        original = layer_fixture(source)
        roles = ('kick', 'snare', 'snare', 'hat', 'hat', 'tom', 'ride', 'clap')
        base = self.stems(*zip(NAMES, roles), duration=END)
        base['bpm'] = 170
        notes = [(tick, 24, 60+i, 0) for i in range(8)
                 for tick in (i*96, (8+i)*96)]
        midi = self.midi(('DRUMS', notes), tempo=round(60_000_000/170))
        result = self.analyze(base, midi_path=midi, source_flp=source)
        self.assertFalse(result['match_errors'])
        for i, stem in enumerate(result['stems']):
            self.assertEqual([n['tick'] for n in stem['note_events']], [i*96, (8+i)*96])
            self.assertEqual({n['note'] for n in stem['note_events']}, {60+i})
            self.assertTrue(stem['midi_verified'])
        self.assertEqual(source.read_bytes(), original)

    def test_inexact_source_falls_back_to_shared_structure_without_claiming_hits(self):
        base = self.stems(('kick', 'kick'))
        midi = self.midi(('DRUMS', [(0, 12, 60, 0), (96, 12, 60, 0)]))
        source, guide = self.source([dict(tick=0, duration=12, note=60)])
        with patch.object(inputs, 'read_source_timing', return_value=guide):
            result = self.analyze(base, midi_path=midi, source_flp=source)
        self.assertFalse(result['match_errors'])
        self.assertFalse(result['stems'][0]['midi_verified'])
        self.assertNotIn('note_events', result['stems'][0])
        self.assertEqual(result['shared_drum_guide']['stem_ids'], [0])

    def test_duplicate_names_and_reused_explicit_tracks_block(self):
        base = self.stems(('lead', 'synth'))
        midi = self.midi(('lead', [(0, 12, 60, 0)]), ('lead', [(0, 12, 62, 0)]))
        self.assertTrue(self.analyze(base, midi_path=midi)['match_errors'])
        result = self.analyze(base, midi_path=midi, midi_mapping={0: dict(track_index=2)})
        self.assertFalse(result['match_errors'])
        base = self.stems(('lead A', 'synth'), ('lead B', 'synth'))
        result = self.analyze(base, midi_path=midi,
                              midi_mapping={0: dict(track_index=1), 1: dict(track_index=1)})
        self.assertEqual(len(result['match_errors']), 2)

    def test_meter_tempo_changes_rejected_and_zero_duration_hits_kept(self):
        base = self.stems(('kick', 'kick'))
        midi = self.midi(('kick', [(96, 0, 60, 9)]))
        result = self.analyze(base, midi_path=midi)
        self.assertFalse(result['match_errors'])
        self.assertEqual(result['stems'][0]['note_events'][0]['duration'], 0)
        changed = MidiFile(midi)
        changed.tracks[0].append(MetaMessage('set_tempo', tempo=400000, time=192))
        changed.save(midi)
        with self.assertRaisesRegex(ValueError, 'tempo'):
            self.analyze(base, midi_path=midi)


if __name__ == '__main__':
    unittest.main()
