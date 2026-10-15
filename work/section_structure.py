"""Deterministic section evidence from verified MIDI arrangement and stem audio.

The eight-bar analysis cache is storage, not a mandatory musical grid. This
detector measures real bars, preserves named boundaries, and requires sustained
arrangement changes before adding its own. Energy is relative arrangement
activity with audio support, not a claim about the rendered mix's loudness.
"""
from __future__ import annotations

import math
from collections.abc import Mapping


DRUM_ROLES = {'kick', 'snare', 'clap', 'tom', 'hat', 'perc', 'ride', 'crash', 'drums'}


def shared_drum_members(guide, stems, end_tick):
    """Validate one section-level drum guide; it proves no individual hits."""
    if guide is None:
        return set()
    if (not isinstance(guide, Mapping) or guide.get('verified') is not True
            or guide.get('timing_provenance') != 'shared-drum-guide'
            or type(guide.get('midi_file_track_index')) is not int
            or guide['midi_file_track_index'] < 0
            or not isinstance(guide.get('midi_track'), str)):
        raise ValueError('Shared drum section guide is not a verified MIDI track.')
    ids = guide.get('stem_ids')
    if (not isinstance(ids, list) or not ids
            or any(type(sid) not in (str, int) for sid in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('Shared drum section guide needs distinct member stem IDs.')
    by_id = {stem.get('id'): stem for stem in stems}
    aliases = {'hihat': 'hat', 'hi-hat': 'hat', 'percussion': 'perc', 'drum_bus': 'drums'}
    for sid in ids:
        stem = by_id.get(sid, {})
        role = str(stem.get('role', '')).strip().lower()
        if (aliases.get(role, role) not in DRUM_ROLES
                or stem.get('midi_guide_verified') is not True
                or stem.get('midi_verified') is not False
                or stem.get('timing_provenance') != 'shared-drum-guide'
                or type(stem.get('midi_file_track_index')) is not int
                or stem.get('midi_file_track_index') != guide['midi_file_track_index']):
            raise ValueError('Shared drum section guide members must be supported drum stems with matching guide identity.')
    notes = guide.get('note_events')
    if not isinstance(notes, list) or not notes:
        raise ValueError('Shared drum section guide needs exported MIDI notes.')
    for note in notes:
        if (not isinstance(note, Mapping) or type(note.get('tick')) is not int
                or type(note.get('duration')) is not int or note['tick'] < 0
                or note['duration'] < 0 or note['tick'] >= end_tick
                or note['tick'] + note['duration'] > end_tick
                or type(note.get('note')) is not int or not 0 <= note['note'] <= 127
                or any(field in note and (type(note[field]) is not int or not 0 <= note[field] <= limit)
                       for field, limit in (('velocity', 127), ('midi_channel', 15)))):
            raise ValueError('Shared drum section guide notes must fit the full song timeline.')
    return set(ids)


def _finite(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f'{name} must be a finite number >= {minimum}.')
    return value


def _union(spans):
    result = []
    for start, stop in sorted(spans):
        if stop <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(stop, result[-1][1])
        else:
            result.append([start, stop])
    return result


def _average(values, start, stop, bar, end):
    total = 0.
    for index in range(max(0, start // bar), min(len(values), math.ceil(stop / bar))):
        overlap = max(0, min(stop, end, (index + 1) * bar) - max(start, index * bar))
        total += values[index] * overlap
    return total / max(1, stop - start)


def _sustained_change(values, index):
    """Both new bars must differ from both old bars; single fills cannot split."""
    before, after = values[index - 2:index], values[index:index + 2]
    if len(before) != 2 or len(after) != 2:
        return 0.
    return max(0., min(after) - max(before), min(before) - max(after))


def _midi_bars(stem, count, bar, end):
    if stem.get('midi_verified') is not True:
        return None
    starts = [set() for _ in range(count)]
    spans = []
    for note in stem.get('note_events', []):
        start = _finite(note.get('tick'), 'MIDI note start')
        duration = _finite(note.get('duration'), 'MIDI note duration')
        if start >= end:
            continue
        starts[int(start // bar)].add(start)
        spans.append((start, min(end, start + duration)))
    if not any(starts):
        return None
    occupancy = [0.] * count
    for start, stop in _union(spans):
        for index in range(int(start // bar), min(count, math.ceil(stop / bar))):
            width = min(end, (index + 1) * bar) - index * bar
            occupancy[index] += max(0, min(stop, (index + 1) * bar) - max(start, index * bar)) / width
    # A short or zero-gate drum hit still means its part is present. Sustained
    # notes remain present across later bars even without another note-on.
    active = [max(min(1., value), float(bool(onsets))) for value, onsets in zip(occupancy, starts)]
    rates = [len(onsets) * bar / (min(end, (index + 1) * bar) - index * bar)
             for index, onsets in enumerate(starts)]
    maximum = max(rates, default=1.) or 1.
    density = [value / maximum for value in rates]
    return active, density


def detect_section_structure(stems, *, bpm, ppq, end_tick, markers=None, shared_drum_guide=None):
    """Return contiguous sections on a fixed 4/4 timeline, independent of seed."""
    _finite(bpm, 'Tempo', 1)
    for value, name in ((ppq, 'PPQ'), (end_tick, 'Song end')):
        _finite(value, name, 1)
        if type(value) is not int:
            raise ValueError(f'{name} must be an integer.')
    bar, end = ppq * 4, end_tick
    count = math.ceil(end / bar)
    scale = bpm * ppq / 60
    power = [0.] * count
    midi = []
    shared_ids = shared_drum_members(shared_drum_guide, stems, end)
    shared_track = shared_drum_guide['midi_file_track_index'] if shared_ids else None
    if shared_ids:
        midi.append(_midi_bars(dict(midi_verified=True,
                                   note_events=shared_drum_guide['note_events']), count, bar, end))
    # Stable sum order ensures input permutation cannot change float thresholds.
    ordered = sorted(stems, key=lambda item: (str(type(item.get('id')).__name__), str(item.get('id')), str(item.get('name', ''))))
    for stem in ordered:
        # Exact subsets remain valid per-stem timing, but their source track
        # already contributes once as the whole shared group. Counting both
        # would overweight drums and invent multi-part arrangement breaks.
        measured = (None if shared_ids and stem.get('midi_file_track_index') == shared_track
                    else _midi_bars(stem, count, bar, end))
        if measured:
            midi.append(measured)
        audio = stem.get('analysis', {})
        offset = stem.get('offset_tick', stem.get('position', 0))
        _finite(offset, 'Stem offset')
        rows = audio.get('level_windows') or []
        if not rows or any('rms' not in row for row in rows):
            rows = audio.get('chunks', [])
        for row in rows:
            start = offset + _finite(row.get('start_seconds'), 'Audio start') * scale
            stop = offset + _finite(row.get('end_seconds'), 'Audio end') * scale
            rms = _finite(row.get('rms'), 'Audio RMS')
            if stop <= start:
                raise ValueError('Audio windows must have positive duration.')
            start, stop = max(0., start), min(end, stop)
            if stop <= start:
                continue
            for index in range(int(start // bar), min(count, math.ceil(stop / bar))):
                width = min(end, (index + 1) * bar) - index * bar
                overlap = max(0., min(stop, (index + 1) * bar) - max(start, index * bar))
                power[index] += rms * rms * overlap / width
    audio = [math.sqrt(value) for value in power]
    audio_peak = max(audio, default=0.)
    if audio_peak:
        audio = [value / audio_peak for value in audio]
    activity = [sum(part[0][i] for part in midi) / len(midi) if midi else 0. for i in range(count)]
    density = [sum(part[1][i] for part in midi) / len(midi) if midi else 0. for i in range(count)]
    structure = [.7 * present + .3 * busy for present, busy in zip(activity, density)]
    low, high = min(structure, default=0.), max(structure, default=0.)
    # Changes are interpreted relative to this song, independent of the count
    # or absolute note rate of its drum lanes. Constant MIDI contributes a
    # constant level; it cannot fabricate a rise from repeated patterns.
    if high - low > 1e-9:
        structure = [(value - low) / (high - low) for value in structure]
    elif high:
        structure = [1.] * count
    energy = [.7 * notation + .3 * sound for notation, sound in zip(structure, audio)] if midi else audio

    named = {}
    for marker in markers or []:
        tick = marker.get('tick')
        name = str(marker.get('name', marker.get('text', ''))).strip()
        if (not name or marker.get('kind') == 8 or isinstance(tick, bool)
                or not isinstance(tick, (int, float)) or not math.isfinite(tick) or not 0 <= tick < end):
            continue
        snapped = int(math.floor(tick / bar + .5)) * bar
        if snapped < end:
            # Duplicate marker order cannot change the result. Caller may
            # resolve FL/MIDI precedence first; otherwise choose lexically.
            named[snapped] = min(name, named.get(snapped, name))
    candidates = []
    for index in range(4, count - 1):
        tick = index * bar
        if end - tick < 4 * bar or any(abs(tick - marker) < 4 * bar for marker in named):
            continue
        parts = [max(_sustained_change(present, index), .7 * _sustained_change(busy, index))
                 for present, busy in midi]
        clear = [value for value in parts if value >= .25]
        midi_change = min(1., max(clear, default=0.) + .15 * (sum(clear) - max(clear, default=0.)))
        audio_change = _sustained_change(audio, index)
        if midi_change or audio_change >= .12:
            candidates.append((max(midi_change, .75 * audio_change), index,
                               'MIDI + audio' if midi_change and audio_change >= .12 else
                               'MIDI arrangement' if midi_change else 'audio energy'))
    selected = {}
    # A brief arrangement break can be musically real even though its two
    # edges are closer than the usual spacing. Require the same two or more
    # verified parts to disappear for two full bars and return for two full
    # bars. Density-only fills and one-part gaps cannot take this exception.
    breaks = []
    for index in range(4, count - 3):
        entry, release = index * bar, (index + 2) * bar
        if (index + 4) * bar > end:
            continue
        restored = sum(min(present[index - 2:index]) >= .75
                       and max(present[index:index + 2]) <= .25
                       and min(present[index + 2:index + 4]) >= .75
                       for present, _ in midi)
        if restored < 2:
            continue
        if any(marker not in (entry, release)
               and min(abs(marker - entry), abs(marker - release)) < 4 * bar for marker in named):
            continue
        breaks.append((restored, entry, release))
    for restored, entry, release in sorted(breaks, key=lambda item: (-item[0], item[1])):
        if all(abs(edge - chosen) >= 4 * bar for edge in (entry, release) for chosen in selected):
            selected[entry] = (1., 'MIDI break start')
            selected[release] = (1., 'MIDI break return')
    for score, index, source in sorted(candidates, key=lambda item: (-item[0], item[1])):
        if all(abs(index * bar - chosen) >= 4 * bar for chosen in selected):
            selected[index * bar] = (score, source)
    boundaries = sorted({0, end, *named, *selected})
    sections = []
    for start, stop in zip(boundaries, boundaries[1:]):
        score, source = selected.get(start, (0., 'song start'))
        value = _average(energy, start, stop, bar, end)
        sections.append(dict(type='section', start_tick=start, end_tick=stop,
            bar=start // bar + 1, energy=value,
            energy_jump=value - sections[-1]['energy'] if sections else 0.,
            marker_name=named.get(start), source='project marker' if start in named else source,
            evidence=dict(verified_midi_stems=len(midi) - bool(shared_ids),
                shared_drum_guides=int(bool(shared_ids)), midi_parts=len(midi),
                midi_activity=_average(activity, start, stop, bar, end),
                midi_density=_average(density, start, stop, bar, end),
                audio_energy=_average(audio, start, stop, bar, end),
                boundary_change=score)))
    return sections
