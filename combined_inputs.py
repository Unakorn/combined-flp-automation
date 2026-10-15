"""Verify combined automation inputs without assigning kit notes to drum stems.

This module reads projects and MIDI only. A PAD -> CHORDS alias is accepted only
when the original instrument and rendered stem use the same dedicated insert.
It is routing/name evidence, not an acoustic transcription claim.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import hashlib
import sys

_WORK = Path(__file__).resolve().parent / 'work'
if str(_WORK) not in sys.path:
    sys.path.insert(0, str(_WORK))

from flp_song_analysis import analyze_flp, read_project
from section_inputs import _inventory
from section_structure import DRUM_ROLES
from stem_audio import ROLES
from stem_midi import match_name, read_timing
from stem_project import inspect_stems


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_exported_notes(source_notes, exported_notes):
    """Require every onset/pitch and duration, except recorded terminal -1 tick.

    FL's MIDI export can shorten a note exactly at its export boundary by one
    tick. Do not alter either source or export; disclose each accepted delta.
    Velocities are not compared because FL velocity 128 is not MIDI-representable.
    """
    source = defaultdict(list)
    exported = defaultdict(list)
    for note in source_notes:
        source[note['tick'], note['note']].append(note['duration'])
    for note in exported_notes:
        exported[note['tick'], note['note']].append(note['duration'])
    if source.keys() != exported.keys() or any(len(source[k]) != len(exported[k]) for k in source):
        raise ValueError('MIDI onset, pitch, or note count differs from the original arrangement.')
    terminal_tick = max((n['tick'] + n['duration'] for n in source_notes), default=0)
    deltas = []
    for (tick, pitch), durations in source.items():
        for expected, actual in zip(sorted(durations), sorted(exported[tick, pitch])):
            if expected == actual:
                continue
            if actual == expected - 1 and tick + expected == terminal_tick:
                deltas.append(dict(tick=tick, note=pitch, source_duration=expected,
                                   midi_duration=actual, delta_ticks=-1))
            else:
                raise ValueError('MIDI note duration differs from the original arrangement away from its final tick.')
    return dict(note_count=sum(map(len, source.values())), all_onsets_and_pitches_match=True,
                durations_exact=not deltas, terminal_duration_deltas=deltas,
                source_last_note_end_tick=terminal_tick,
                midi_last_note_end_tick=max((n['tick'] + n['duration'] for n in exported_notes), default=0))


def _unique_named(items, name, label):
    candidates = [item for item in items if match_name(item['name']) == match_name(name)]
    if len(candidates) != 1:
        raise ValueError(f'{name}: {label} name is missing or ambiguous.')
    return candidates[0]


def build_input_config(stem_path, midi_path, original_path=None):
    """Return explicit input choices and reviewable identity/timing evidence."""
    paths = [Path(stem_path).resolve(), Path(midi_path).resolve()]
    if original_path:
        paths.append(Path(original_path).resolve())
    before = {str(path): _sha(path) for path in paths}
    info = inspect_stems(stem_path)
    # Validate common PPQ conversion, constant tempo, meter, note pairs and extent.
    read_timing(midi_path, [], ppq=info['ppq'], bpm=info['bpm'], end_tick=info['end_tick'])
    tracks = _inventory(midi_path, info['ppq'])
    roles = {stem['id']: stem['role'] for stem in info['stems']}
    for stem in info['stems']:
        if stem['role'] not in ROLES:
            raise ValueError(f'{stem["name"]}: name this stem with a supported role before combining.')
    voices, comparisons = [], []
    if original_path:
        original = read_project(original_path)
        if original['time_signature'] != (4, 4) or abs(original['bpm'] - info['bpm']) > .01:
            raise ValueError('The original and stem projects must use matching constant-tempo 4/4 timing.')
        model = analyze_flp(original_path, max_instruments=128, allow_single_pattern=True)
        voices = model['voices']
        # Convert original musical ticks when projects use different PPQ values.
        ratio = info['ppq'] / original['ppq']
        for voice in voices:
            track = _unique_named(tracks, voice['name'], 'exported MIDI track')
            notes = [dict(n, tick=round(n['tick'] * ratio), duration=round(n['duration'] * ratio))
                     for n in voice['notes']]
            comparison = compare_exported_notes(notes, track['notes'])
            comparisons.append(dict(source_name=voice['name'], source_channel_id=voice['id'],
                                    source_mixer_insert=voice['mixer_insert'],
                                    midi_file_track_index=track['file_index'], **comparison))
    midi_mapping, identity = {}, []
    combined = [track for track in tracks if match_name(track['name']) in
                {'drums', 'drum kit', 'drumkit', 'drum monkey', 'unison drum monkey', 'mixed drums', 'drum bus'}]
    shared_ids = []
    for stem in info['stems']:
        if stem['role'] in {'vocals', 'effects'}:
            identity.append(dict(stem_id=stem['id'], stem_name=stem['name'], basis='Measured audio; MIDI exempt.'))
            continue
        if stem['role'] in DRUM_ROLES:
            individual = [t for t in tracks if match_name(t['name']) == match_name(stem['name']) and t not in combined]
            if stem['role'] != 'drums' and len(individual) == 1:
                track = individual[0]
                midi_mapping[stem['id']] = {'track_index': track['file_index']}
                identity.append(dict(stem_id=stem['id'], stem_name=stem['name'], midi_track=track['name'],
                                     midi_file_track_index=track['file_index'], basis='Exact unique track name.'))
            elif len(combined) == 1:
                shared_ids.append(stem['id'])
                identity.append(dict(stem_id=stem['id'], stem_name=stem['name'], midi_track=combined[0]['name'],
                                     midi_file_track_index=combined[0]['file_index'],
                                     basis='Shared whole-kit section guide with this stem\'s own audio activity; no individual hit ownership.'))
            else:
                raise ValueError(f'{stem["name"]}: no unique separate drum track or shared full-kit MIDI guide.')
            continue
        candidates = [track for track in tracks if match_name(track['name']) == match_name(stem['name'])]
        basis = 'Exact unique track name.'
        source_name = None
        if not candidates and match_name(stem['name']) in {'pad', 'pads'}:
            source = [voice for voice in voices if voice['mixer_insert'] == stem['mixer_insert']
                      and match_name(voice['name']) == 'chords']
            if len(source) == 1:
                candidates = [track for track in tracks if match_name(track['name']) == 'chords']
                source_name = source[0]['name']
                basis = f'PAD/CHORDS alias: original CHORDS and rendered PAD both use dedicated mixer insert {stem["mixer_insert"]}; exported source notes verified.'
        if len(candidates) != 1:
            raise ValueError(f'{stem["name"]}: exported MIDI name or source-backed alias is missing or ambiguous.')
        track = candidates[0]
        midi_mapping[stem['id']] = {'track_index': track['file_index']}
        identity.append(dict(stem_id=stem['id'], stem_name=stem['name'], source_name=source_name,
                             mixer_insert=stem['mixer_insert'], midi_track=track['name'],
                             midi_file_track_index=track['file_index'], basis=basis))
    selected = [spec['track_index'] for spec in midi_mapping.values()]
    if len(selected) != len(set(selected)):
        raise ValueError('One MIDI track was assigned to multiple stems; an explicit verified Layer mapping is required.')
    if any(_sha(path) != before[str(path)] for path in paths):
        raise ValueError('An input changed while verifying the mapping. Run again.')
    return dict(roles=roles, midi_mapping=midi_mapping,
                drum_midi_guide={'track_index': combined[0]['file_index']} if shared_ids else None,
                protected_stems=[stem['id'] for stem in info['stems'] if stem['role'] == 'drums'],
                verification=dict(input_sha256=before, bpm=info['bpm'], ppq=info['ppq'],
                                  song_end_tick=info['end_tick'], shared_drum_stem_ids=shared_ids,
                                  source_note_comparisons=comparisons, mappings=identity,
                                  midi_tracks=[dict(name=t['name'], file_index=t['file_index'], note_count=len(t['notes']),
                                                    first_tick=min(n['tick'] for n in t['notes']),
                                                    last_note_end_tick=max(n['tick'] + n['duration'] for n in t['notes']))
                                               for t in tracks]))
