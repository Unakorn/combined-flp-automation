"""Strict exported-MIDI identity and timing for the section-automation launcher.

The original instrument FLP can identify a part in a combined export. It never
replaces missing exported MIDI. Names establish identity, not a claim that MIDI
has been acoustically transcribed from a stem. All note positions remain on the
full-song timeline; the stem's Playlist offset is never added to MIDI positions.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from fractions import Fraction
from pathlib import Path
import math
import re

from mido import MidiFile

from section_structure import DRUM_ROLES, detect_section_structure as detect_sections
from stem_midi import match_name, read_timing, read_source_timing
from stem_names import match_stem_names
from stem_project import analyze_stems, _file_digest

_EXEMPT = {'vocals', 'effects'}
_DRUMS = DRUM_ROLES
_COMBINED = re.compile(r'\b(drums|drumkit|drum kit|drum monkey|percussion kit)\b', re.I)


def _inventory(path, ppq):
    """Retain MIDI channel identity, which the legacy timing reader omits."""
    midi = MidiFile(path)
    tracks = []
    for file_index, track in enumerate(midi.tracks):
        absolute, name, active, notes = 0, '', defaultdict(deque), []
        for message in track:
            absolute += message.time
            tick = round(Fraction(absolute * ppq, midi.ticks_per_beat))
            if message.type == 'track_name':
                name = message.name
            elif message.type == 'note_on' and message.velocity:
                active[message.channel, message.note].append((tick, message.velocity))
            elif message.type in {'note_on', 'note_off'}:
                queue = active[message.channel, message.note]
                if queue:
                    start, velocity = queue.popleft()
                    notes.append(dict(tick=start, duration=tick-start,
                                      note=message.note, velocity=velocity,
                                      midi_channel=message.channel))
        # read_timing validates unmatched note-ons, tempo and meter first.
        if notes:
            notes.sort(key=lambda n: (n['tick'], n['note'], n['duration'],
                                     n['midi_channel'], n['velocity']))
            tracks.append(dict(id=len(tracks), file_index=file_index, name=name, notes=notes))
    return tracks


def _verified_silence(audio):
    """Only exact measured zero samples, with complete level-window coverage."""
    duration = audio.get('duration_seconds')
    windows = audio.get('level_windows')
    if (not isinstance(duration, (int, float)) or not math.isfinite(duration)
            or duration <= 0 or not isinstance(windows, list) or not windows):
        return False
    previous = 0.0
    for window in windows:
        try:
            start, end, peak = (window[k] for k in ('start_seconds', 'end_seconds', 'peak'))
            if (not all(isinstance(v, (int, float)) and math.isfinite(v)
                        for v in (start, end, peak)) or peak != 0
                    or abs(start-previous) > 1e-6 or end <= start):
                return False
        except (KeyError, TypeError):
            return False
        previous = end
    return abs(previous-duration) <= 1e-6


def _signature(notes):
    # FL velocity 128 cannot be represented by MIDI. Velocity is deliberately
    # excluded from identity, but the exported MIDI's velocity is kept intact.
    return Counter((n['tick'], n['duration'], n['note']) for n in notes)


def _source_selection(source, tracks):
    """Find one exact full-song lane group proved by source-resolved notes.

    read_source_timing uses the existing FL Layer resolver. For a combined MIDI
    track, a selected (channel, pitch) lane must match in its entirety, including
    every repeat. Coincident individual hits are never enough to infer a map.
    """
    expected = _signature(source.get('note_events', []))
    if not expected:
        return None
    choices = []
    for track in tracks:
        lanes = defaultdict(list)
        for index, note in enumerate(track['notes']):
            lanes[note['midi_channel'], note['note']].append((index, note))
        selected = []
        for lane in lanes.values():
            signature = _signature([n for _, n in lane])
            if signature and not (signature - expected):
                selected.extend(lane)
        if _signature([n for _, n in selected]) == expected:
            choices.append((track, sorted(selected)))
    return choices[0] if len(choices) == 1 else None


def _selection_from_mapping(value, tracks, stem):
    """Explicit drum maps use MIDI pitches 0–127 and zero-based channels 0–15."""
    spec = {'track': value} if isinstance(value, str) else value
    if not isinstance(spec, dict) or set(spec) - {'track', 'track_index', 'notes', 'channels'}:
        raise ValueError('Use a MIDI track name or a track/notes/channels mapping.')
    if ('track' in spec) == ('track_index' in spec):
        raise ValueError('Choose exactly one track name or zero-based MIDI file track_index.')
    if 'track_index' in spec:
        if type(spec['track_index']) is not int:
            raise ValueError('MIDI track_index must be an integer.')
        candidates = [t for t in tracks if t['file_index'] == spec['track_index']]
    else:
        if not isinstance(spec['track'], str) or not match_name(spec['track']):
            raise ValueError('A MIDI track name cannot be empty.')
        candidates = [t for t in tracks if match_name(t['name']) == match_name(spec['track'])]
    if len(candidates) != 1:
        raise ValueError('The explicitly selected MIDI track is missing or ambiguous.')
    filters = {}
    for field, limit in (('notes', 127), ('channels', 15)):
        if field in spec:
            items = spec[field]
            if (not isinstance(items, (list, tuple)) or not items
                    or any(type(x) is not int or not 0 <= x <= limit for x in items)):
                raise ValueError(f'{field} must be a nonempty list of integers from 0 to {limit}.')
            filters[field] = set(items)
    track = candidates[0]
    if (stem['role'] in _DRUMS and _COMBINED.search(match_name(track['name']))
            and not filters):
        raise ValueError('A combined drum track needs a verified source mapping or explicit MIDI notes/channels.')
    selected = [(i, n) for i, n in enumerate(track['notes'])
                if ('notes' not in filters or n['note'] in filters['notes'])
                and ('channels' not in filters or n['midi_channel'] in filters['channels'])]
    if not selected:
        raise ValueError('The explicit MIDI mapping contains no matching notes.')
    return track, selected


def _shared_drum_selection(value, tracks):
    """Select the group once; a custom kit's pitches are never inferred."""
    if value is None:
        candidates = [track for track in tracks if _COMBINED.search(match_name(track['name']))]
        return candidates[0] if len(candidates) == 1 else None
    if (not isinstance(value, dict) or len(value) != 1
            or not set(value) <= {'track', 'track_index'}):
        raise ValueError('Shared drum guide must select one whole MIDI track by track or track_index, without note/channel filters.')
    if 'track_index' in value:
        if type(value['track_index']) is not int or value['track_index'] < 0:
            raise ValueError('Shared drum guide track_index must be a nonnegative integer.')
        candidates = [track for track in tracks if track['file_index'] == value['track_index']]
    else:
        if not isinstance(value['track'], str) or not match_name(value['track']):
            raise ValueError('Shared drum guide track name cannot be empty.')
        candidates = [track for track in tracks if match_name(track['name']) == match_name(value['track'])]
    if len(candidates) != 1:
        raise ValueError('The selected shared drum guide is missing or ambiguous; choose its MIDI file track once.')
    return candidates[0]


def _timing_fields(path, track, selected, ppq, end_tick, reason):
    notes = [dict(note) for _, note in selected]
    spans = sorted((n['tick'], n['tick']+n['duration']) for n in notes if n['duration'] > 0)
    phrases = []
    for start, stop in spans:
        if phrases and start <= phrases[-1][1]+ppq:
            phrases[-1][1] = max(phrases[-1][1], stop)
        else:
            phrases.append([start, stop])
    return dict(midi_track=track['name'], midi_file_track_index=track['file_index'],
                note_events=notes, note_spans=[list(span) for span in spans],
                phrase_ends=[stop for _, stop in phrases if 0 < stop < end_tick],
                note_velocity_max=127, midi_verified=True, timing_provenance='midi',
                match_reason=reason,
                note_guide_source=f'MIDI: {Path(path).name} — {track["name"]}')


def analyze_section_inputs(info, *, roles=None, midi_path=None, source_flp=None,
                           midi_mapping=None, drum_midi_guide=None, cache_dir=None, progress=None):
    """Analyze stems and attach only verified, full-song exported MIDI notes.

    Returns the legacy analyzer shape plus ``match_errors``. Missing or
    ambiguous musical matches block this launcher's planner; audio analysis is
    still available for a useful preview. Bad guide meter/tempo/files raise
    ValueError. ``midi_mapping`` maps stem IDs to track names or explicit specs:
    ``{id: {'track': 'DRUMS', 'notes': [36], 'channels': [9]}}``.
    Unmatched drums may instead share one whole-track section guide. ``None``
    auto-selects a uniquely named combined drum track; an explicit selector is
    ``{'track_index': 2}`` or ``{'track': 'My kit'}``. Its notes are stored once,
    never assigned to each drum as claimed individual hit timing.
    """
    result = analyze_stems(info, roles=roles, cache_dir=cache_dir, progress=progress)
    stems = result['stems']
    required = []
    for stem in stems:
        stem['silent'] = _verified_silence(stem['analysis'])
        stem['midi_verified'] = False
        stem['midi_guide_verified'] = False
        stem['timing_provenance'] = ('audio-exempt' if stem['silent'] or stem['role'] in _EXEMPT
                                     else 'unmatched')
        stem['midi_required'] = stem['timing_provenance'] == 'unmatched'
        if stem['midi_required']:
            required.append(stem)
    errors, timing_files = [], []
    source = None
    if source_flp:
        before = _file_digest(source_flp)
        source = read_source_timing(source_flp, required, ppq=info['ppq'],
                                    bpm=info['bpm'], end_tick=info['end_tick'])
        if _file_digest(source_flp) != before:
            raise ValueError('Source instrument FLP changed during analysis; run again.')
        timing_files.append(dict(path=str(Path(source_flp).resolve()), sha256=before))
        # Preserve source identity evidence for review without giving the
        # planner fallback note timing or the legacy "audio retained" message.
    guide, shared = None, None
    if not midi_path:
        if drum_midi_guide is not None:
            raise ValueError('A shared drum guide requires a full-song exported MIDI file.')
        errors.extend(f'{s["name"]}: matching full-song exported MIDI is required.' for s in required)
    else:
        before = _file_digest(midi_path)
        guide = read_timing(midi_path, required, ppq=info['ppq'], bpm=info['bpm'],
                            end_tick=info['end_tick'])
        tracks = _inventory(midi_path, info['ppq'])
        shared_track = _shared_drum_selection(drum_midi_guide, tracks)
        if _file_digest(midi_path) != before:
            raise ValueError('MIDI changed during analysis; run again.')
        timing_files.insert(0, dict(path=str(Path(midi_path).resolve()), sha256=before))
        automatic = match_stem_names(required, tracks)['matches']
        by_id = {t['id']: t for t in tracks}
        pending, shared_stems = {}, []
        for stem in required:
            stem_id = stem['id']
            mapping = (midi_mapping or {}).get(stem_id, (midi_mapping or {}).get(str(stem_id)))
            chosen, reason = None, ''
            if mapping is not None:
                try:
                    chosen = _selection_from_mapping(mapping, tracks, stem)
                    reason = 'Explicit exported-MIDI mapping.'
                except ValueError as error:
                    errors.append(f'{stem["name"]}: {error}')
                    continue
            elif stem_id in automatic:
                match = automatic[stem_id]
                track = by_id[match['track_id']]
                if not (stem['role'] in _DRUMS and _COMBINED.search(match_name(track['name']))):
                    chosen = (track, list(enumerate(track['notes'])))
                    reason = match['reason']
            if chosen is None and source and stem_id in source['matches']:
                chosen = _source_selection(source['matches'][stem_id], tracks)
                reason = 'Exact full-song exported notes verified against the source-FLP part.'
            if chosen is None:
                if stem['role'] in _DRUMS and shared_track is not None:
                    shared_stems.append(stem)
                    continue
                extra = (' Choose Shared drum guide once for the combined drum MIDI, or provide an exact pitch/channel mapping.'
                         if stem['role'] in _DRUMS else ' Name its MIDI track to match the stem or select an explicit pairing.')
                errors.append(f'{stem["name"]}: exported MIDI match is missing or ambiguous.{extra}')
                continue
            track, selected = chosen
            start, stop = stem['position'], stem['position']+stem['duration']
            if any(n['tick'] < start or n['tick']+n['duration'] > stop for _, n in selected):
                errors.append(f'{stem["name"]}: MIDI notes lie outside this stem\'s Playlist interval; export both from the same song start.')
                continue
            pending[stem_id] = (track, selected, reason)
        owners = defaultdict(list)
        for sid, (track, selected, reason) in pending.items():
            for index, _ in selected:
                owners[track['file_index'], index].append(sid)
        conflicts = set()
        for ids in owners.values():
            if len(ids) > 1 and not all(pending[sid][2].startswith('Exact full-song') for sid in ids):
                conflicts.update(ids)
        verified_matches = {}
        for stem in required:
            sid = stem['id']
            if sid in conflicts:
                errors.append(f'{stem["name"]}: the same MIDI notes were assigned to multiple stems; supply distinct tracks or a source-verified Layer mapping.')
            elif sid in pending:
                track, selected, reason = pending[sid]
                fields = _timing_fields(midi_path, track, selected, info['ppq'], info['end_tick'], reason)
                stem.update(fields)
                stem['timing_source'] = fields['note_guide_source']
                verified_matches[sid] = fields
        guide = dict(guide, matches=verified_matches, warnings=[])
        if shared_stems:
            label = shared_track['name'] or f'Track {shared_track["file_index"] + 1}'
            text = f'Shared drum guide + audio: {Path(midi_path).name} — {label}'
            shared = dict(midi_track=shared_track['name'],
                          midi_file_track_index=shared_track['file_index'],
                          note_events=[dict(note) for note in shared_track['notes']],
                          stem_ids=[stem['id'] for stem in shared_stems],
                          verified=True, timing_provenance='shared-drum-guide',
                          note_guide_source=text)
            for stem in shared_stems:
                stem.pop('note_events', None)
                stem.pop('note_spans', None)
                stem.update(midi_verified=False, midi_guide_verified=True,
                            midi_track=shared['midi_track'],
                            midi_file_track_index=shared['midi_file_track_index'],
                            timing_provenance='shared-drum-guide',
                            match_reason='Shared drum section timing; this stem retains its own audio activity, not individual MIDI hit ownership.',
                            note_guide_source=text, timing_source=text)
    # Keep all named boundaries; FL markers take precedence at the same tick.
    markers_by_tick = {item.get('tick'): dict(item) for item in (guide or {}).get('markers', [])}
    markers_by_tick.update({item.get('tick'): dict(item) for item in info.get('markers', [])})
    markers = list(markers_by_tick.values())
    result.update(midi_guide=guide, shared_drum_guide=shared, source_guide=source, timing_files=timing_files,
                  match_errors=errors, midi_required=bool(required), half_time=False,
                  sections=detect_sections(stems, bpm=info['bpm'], ppq=info['ppq'],
                                           end_tick=info['end_tick'],
                                           markers=markers, shared_drum_guide=shared))
    return result
