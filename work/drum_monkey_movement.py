"""Shared-bus automation for the other drums in one unchanged Drum Monkey.

The caller supplies only non-kick MIDI notes for musical analysis and routes
other drums to their shared bus, kick to its separate bus. This module writes
no audio routing, plugin state or kick controls.
Delay/reverb affect every sound routed to the shared bus, even when a snare or
tom fill supplies the timing cue. They are not per-pad effect automation.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
import hashlib
import math
from pathlib import Path
import random
from statistics import median

from mido import Message, MetaMessage, MidiFile, MidiTrack
from drum_automation import _clamp, _drops, _ease, _hit_envelopes, _number
from energy_curve import sample_energy_curve
from velocity_automation_midi import (
    MOVEMENTS, MovementPlan, NoteEvent, build_grid, intensity_factors, song_end_tick, source_meta_track,
)


BUS_LANES = frozenset({'volume', 'cutoff', 'delay', 'reverb'})
ROLES = frozenset({'kick', 'snare', 'clap', 'off_snare', 'closed_hat', 'open_hat',
                   'tom', 'percussion'})
BASELINES = {'volume': 102, 'cutoff': 127, 'delay': 1, 'reverb': 1}


def _roles(drum_parts):
    result = {}
    for part in drum_parts or []:
        row = part.get('drum', part)
        pitch, role = row.get('source_note'), row.get('role', 'percussion')
        if isinstance(pitch, bool) or not isinstance(pitch, int) or not 0 <= pitch <= 127 or role not in ROLES:
            raise ValueError('Drum bus metadata needs MIDI note numbers and supported drum types.')
        if pitch in result and result[pitch] != role:
            raise ValueError('A drum MIDI note has conflicting drum types.')
        result[pitch] = role
    return result


def _hits(notes):
    grouped = {}
    for note in notes:
        grouped[note.tick] = max(grouped.get(note.tick, 0), int(_clamp(note.velocity, 1, 127)))
    return sorted(grouped.items())


def _effect_envelopes(notes, roles, boundaries, ppq, end, lane, style, seed):
    # Trigger on musical fill voices when identified. Hats and kick must not
    # turn a regular busy beat into a stream of artificial snare throws.
    fill = [note for note in notes if roles.get(note.note) in {'snare', 'clap', 'off_snare', 'tom'}]
    if not fill:
        fill = [note for note in notes if roles.get(note.note, 'percussion') == 'percussion']
    if not fill:
        fill = notes
    envelopes = _hit_envelopes(_hits(fill), boundaries, ppq, end, lane, 'snare', style, seed)
    toms = [note for note in notes if roles.get(note.note) == 'tom']
    if toms:
        envelopes += _hit_envelopes(_hits(toms), boundaries, ppq, end, lane, 'tom', style, seed + 19)
    # A layered hit can be selected by more than one musical clue, but it still
    # produces one shared effect throw. Retain the longer chosen tail.
    by_peak = {}
    for envelope in envelopes:
        previous = by_peak.get(envelope[1])
        if previous is None or envelope[3] > previous[3]:
            by_peak[envelope[1]] = envelope
    return sorted(by_peak.values())


def _cutoff_context(notes, roles, anchors, ppq, end):
    """Find sparse accents and continuous activity without guessing pad sounds.

    Unknown percussion playing sustained dense subdivisions is a rhythm cue,
    not a reason to reopen the whole bus on every hit. Verified snare/tom roles
    remain accents. A locally isolated hit on a busy key is still protected.
    """
    by_pitch = {}
    for note in notes:
        by_pitch.setdefault(note.note, set()).add(note.tick)
    by_pitch = {pitch: sorted(ticks) for pitch, ticks in by_pitch.items()}
    busy = {pitch for pitch, ticks in by_pitch.items()
            if roles.get(pitch, 'percussion') == 'percussion' and len(ticks) >= 8
            and ticks[-1] - ticks[0] >= 8 * ppq
            and median(b - a for a, b in zip(ticks, ticks[1:])) <= .75 * ppq}
    accents = set(anchors)
    for note in notes:
        role = roles.get(note.note, 'percussion')
        if note.velocity < 80 or role not in {'snare', 'clap', 'off_snare', 'tom', 'percussion'}:
            continue
        if note.note in busy:
            ticks = by_pitch[note.note]
            index = bisect_left(ticks, note.tick)
            close_before = index > 0 and note.tick - ticks[index - 1] <= .75 * ppq
            close_after = index + 1 < len(ticks) and ticks[index + 1] - note.tick <= .75 * ppq
            if close_before or close_after:
                continue
        accents.add(note.tick)
    spans = []
    for note in notes:
        start = note.tick
        stop = min(end, max(start + note.duration, start + max(1, round(.5 * ppq))))
        if spans and start <= spans[-1][1] + .25 * ppq:
            spans[-1] = (spans[-1][0], max(spans[-1][1], stop))
        else:
            spans.append((start, stop))
    return sorted(accents), spans


def _cutoff_activity(tick, spans, starts, ppq):
    index = bisect_right(starts, tick) - 1
    amount = 0.0
    # A release can overlap the next span's gentle onset. The minimum span
    # length makes two candidates sufficient for this short return ramp.
    for cursor in (index - 1, index):
        if cursor < 0:
            continue
        start, stop = spans[cursor]
        if tick < start:
            continue
        gate = _ease((tick - start) / max(1, ppq * .125))
        if tick > stop:
            gate *= 1 - _ease((tick - stop) / max(1, ppq * .75))
        amount = max(amount, gate)
    return amount


def _accent_protection(tick, accents, before, after):
    amount = 0.0
    first = bisect_left(accents, tick - after)
    last = bisect_right(accents, tick + before)
    for hit in accents[first:last]:
        weight = (_ease((tick - hit + before) / before) if tick < hit
                  else 1 - _ease((tick - hit) / after))
        amount = max(amount, weight)
    return amount


def make_drum_monkey_ctrl_midi(source_midi: MidiFile, notes: list[NoteEvent],
                              drum_parts: list[dict], lane: str, output_path: Path,
                              options: dict, plan: MovementPlan, energy_curve=None,
                              kick_notes: list[NoteEvent] | None = None) -> dict:
    """Write one note-60 controller MIDI for the intended shared non-kick bus.

    Notes and drum_parts describe the non-kick signal; optional kick_notes are
    read only to locate the first actual kick near a planned drop. All source
    notes, pitches, velocities and timing remain unchanged. No sample/preset
    choice or audio routing is performed. This function never writes a kick
    lane. Volume never exceeds 102; cutoff is modest and opens before strong
    hits; wet lanes rest at 1. Strength zero produces constant baselines.
    Optional intensity (0..2, default 1) extends throws and increases contrast.
    Above one, cutoff builds get longer and deeper, with smooth protection for
    sparse accents instead of reopening on every unknown subdivision. Volume
    never boosts. Intensity at or below one retains the original cutoff shape.
    """
    if lane not in BUS_LANES:
        raise ValueError('Choose volume, cutoff, delay or reverb for the drum bus.')
    strength = _number(options, 'strength', .8, 0, 1)
    intensity = _number(options, 'intensity', 1.0, 0, 2)
    depth, hold_scale, release_scale = intensity_factors(intensity)
    division = _number(options, 'division', 16, 1, 64, True)
    seed = _number(options, 'seed', 1, 0, 2147483647, True)
    gate = _number(options, 'gate', .9, .01, 1)
    movement = options.get('movement', 'drop')
    policy = options.get('drop_fx', 'reduced')
    style = options.get('fx_style', 'full_range')
    if movement not in MOVEMENTS or policy not in {'dry', 'reduced', 'keep'}:
        raise ValueError('Choose a supported drum-bus movement and drop FX policy.')
    if style not in {'original', 'fuller', 'full_range'}:
        raise ValueError('Choose a supported drum-bus FX style.')
    ppq, end = source_midi.ticks_per_beat, song_end_tick(source_midi)
    if ppq <= 0 or end <= 0 or plan.end_tick != end:
        raise ValueError('Drum-bus automation needs a matching musical-time song and plan.')
    roles = _roles(drum_parts)
    notes = sorted((note for note in notes if note.duration > 0 and note.velocity > 0
                    and 0 <= note.tick < end), key=lambda note: (note.tick, note.note))
    if any(roles.get(note.note) == 'kick' for note in notes):
        raise ValueError('Supply non-kick notes to the other-drums bus; pass kicks separately as kick_notes.')
    valid_kicks = [note for note in kick_notes or [] if note.duration > 0
                   and note.velocity > 0 and 0 <= note.tick < end]
    kick_pitches = {kick.note for kick in valid_kicks}
    if any(note.note in kick_pitches for note in notes):
        raise ValueError('Kick MIDI keys must not be included in the other-drums analysis notes.')
    kicks = sorted({note.tick for note in valid_kicks})
    hits = _hits(notes)
    hit_ticks = [tick for tick, _ in hits]
    boundaries = sorted({0, end, *(int(tick) for tick in plan.phrase_bounds if 0 < tick < end)})
    drops = _drops(plan, end)
    drop_ticks = [start for start, _ in drops]
    anchors = []
    for start, stop in drops:
        index = bisect_left(kicks, start)
        actual = kicks[index] if index < len(kicks) else None
        anchors.append(actual if actual is not None and actual < min(stop, start + 2 * ppq) else start)
    envelopes = (_effect_envelopes(notes, roles, boundaries, ppq, end, lane, style, seed)
                 if strength and intensity and lane in {'delay', 'reverb'} else [])
    if intensity > 1:
        # The same fill cues get longer, more audible holds and tails. Do not
        # manufacture extra hits or turn every hat into a shared-bus throw.
        envelopes = [(attack, peak, min(end, peak + round((hold - peak) * hold_scale)),
                      min(end, peak + round((hold - peak) * hold_scale)
                          + round((finish - hold) * release_scale)), velocity)
                     for attack, peak, hold, finish, velocity in envelopes]
    # Restore brightness/level before strong non-kick attacks. With unknown
    # roles, protect strong hits rather than infer a snare from its MIDI key.
    strong = {note.tick for note in notes if note.velocity >= 80
              and roles.get(note.note, 'percussion') in {'snare', 'clap', 'off_snare', 'tom', 'percussion'}}
    strong.update(anchors)
    strong_ticks = sorted(strong)
    before_hit = max(1, round(ppq * .125))
    after_hit = max(1, round(ppq * .25))
    active_length = max(1, round(ppq * .5))
    stronger_cutoff = lane == 'cutoff' and intensity > 1
    extra = max(0.0, intensity - 1)
    build_length = 8 * ppq * (1 + extra)
    accents, activity_spans = (_cutoff_context(notes, roles, anchors, ppq, end)
                               if stronger_cutoff else ([], []))
    activity_starts = [start for start, _ in activity_spans]
    ticks, _ = build_grid(notes, ppq, division, end)
    positions = {0, end - 1, *ticks, *boundaries, *hit_ticks}
    positions.update(tick + active_length for tick in hit_ticks)
    for hit in strong_ticks:
        positions.update((hit - before_hit, hit, hit + after_hit))
    for start, peak, hold, stop, _ in envelopes:
        positions.update((start, peak, hold, stop, (start + peak) // 2, (hold + stop) // 2))
    for (start, stop), anchor in zip(drops, anchors):
        positions.update((start - 8 * ppq, start - before_hit, start, stop,
                          anchor, anchor + ppq // 2, anchor + 2 * ppq))
        if stronger_cutoff:
            positions.update((round(start - build_length), round(start - ppq * .5)))
    if stronger_cutoff:
        for start, stop in activity_spans:
            positions.update((start, min(end, start + max(1, round(ppq * .125))),
                              stop, min(end, stop + max(1, round(ppq * .75)))))
    if energy_curve is not None:
        positions.update(int(tick) for tick, _ in energy_curve['points'])
    ticks = sorted(tick for tick in positions if 0 <= tick < end)
    painting = sample_energy_curve(energy_curve, ticks) if energy_curve is not None else [.5] * len(ticks)
    salt = int.from_bytes(hashlib.sha256(('other drums bus:' + lane).encode()).digest()[:8], 'little')
    rng = random.Random(seed + salt)
    variation = [rng.uniform(-1, 1) for _ in range(end // (8 * ppq) + 2)]
    baseline = BASELINES[lane]
    velocities = []
    fx_index, active_fx = 0, []
    for tick, painted in zip(ticks, painting):
        hit_index = bisect_right(hit_ticks, tick) - 1
        active = hit_index >= 0 and tick - hit_ticks[hit_index] < active_length
        strong_index = bisect_right(strong_ticks, tick + before_hit) - 1
        protected_hit = strong_index >= 0 and tick < strong_ticks[strong_index] + after_hit
        phrase_index = max(0, bisect_right(boundaries, tick) - 1)
        begin, stop = boundaries[phrase_index:phrase_index + 2]
        ramp = min(8 * ppq, max(1, (stop - begin) // 2))
        lift = _ease((tick - stop + ramp) / ramp)
        drop_index = bisect_right(drop_ticks, tick) - 1
        in_drop = drop_index >= 0 and tick < drops[drop_index][1]
        next_drop = drop_index + 1
        tension = (_ease(1 - (drop_ticks[next_drop] - tick) / (8 * ppq))
                   if next_drop < len(drop_ticks) else 0)
        phase, remainder = divmod(tick, 8 * ppq)
        blend = _ease(remainder / (8 * ppq))
        drift = variation[phase] * (1 - blend) + variation[phase + 1] * blend
        value = float(baseline)
        if strength and active and lane == 'volume':
            # Keep the bus level steady. Optional small dips create contrast;
            # MIDI note velocities remain responsible for individual accents.
            reduction = max(0, .5 - painted) * 8
            if movement in {'groove', 'full'}:
                reduction += 1.5 * (1 + math.sin(math.tau * tick / (2 * ppq))) / 2
            if movement in {'transitions', 'full'}:
                reduction += 2.5 * lift ** 3
            if movement == 'drop':
                reduction += 5 * tension ** 4
            if not protected_hit:
                value -= strength * reduction * (min(depth, 1.5) if intensity != 1 else 1)
        elif strength and lane == 'cutoff' and (active or stronger_cutoff):
            if stronger_cutoff and next_drop < len(drop_ticks):
                tension = _ease(1 - (drop_ticks[next_drop] - tick) / build_length)
            reduction = max(0, .5 - painted) * 10
            if movement in {'transitions', 'full'}:
                reduction += 10 * lift
            if movement == 'drop':
                reduction += 12 * tension
            if movement in {'groove', 'full'}:
                reduction += 2 * (1 + math.sin(math.tau * tick / (2 * ppq))) / 2
            if movement != 'classic':
                reduction += max(0, drift) * 1.5
            reduction *= 1 - max(0, painted - .5)
            if stronger_cutoff and not (movement == 'drop' and in_drop):
                reduction *= depth
                if movement == 'drop':
                    reduction += 27 * extra * tension
                elif movement in {'transitions', 'full'}:
                    reduction += 33 * extra * lift
                # Crossfade the legacy hard guard into sparse, smooth accents.
                # Dense unknown percussion cannot pin the bus fully open at 200%.
                protection = ((1 - extra) * float(protected_hit)
                              + extra * _accent_protection(tick, accents, before_hit, after_hit))
                activity = ((1 - extra) * float(active)
                            + extra * _cutoff_activity(tick, activity_spans, activity_starts, ppq))
                restore = 1.0
                if movement == 'drop' and next_drop < len(drop_ticks):
                    edge = drop_ticks[next_drop]
                    restore = 1 - _ease((tick - edge + ppq * .5) / max(1, ppq * .5 - before_hit))
                value = max(110 - 46 * extra,
                            baseline - strength * reduction * activity * (1 - protection) * restore)
            elif not stronger_cutoff and not protected_hit and not (movement == 'drop' and in_drop):
                value = max(110, baseline - strength * reduction * depth)
        elif strength and lane in {'delay', 'reverb'}:
            while fx_index < len(envelopes) and envelopes[fx_index][0] <= tick:
                active_fx.append(envelopes[fx_index])
                fx_index += 1
            active_fx = [row for row in active_fx if tick < row[3]]
            amount = 0.0
            for attack, peak, hold, finish, velocity in active_fx:
                if tick < peak:
                    envelope = _ease((tick - attack) / max(1, peak - attack))
                elif tick <= hold:
                    envelope = 1.0
                else:
                    envelope = 1 - _ease((tick - hold) / max(1, finish - hold))
                if style == 'full_range':
                    level, ceiling = min(1.0, strength / .65), 127
                else:
                    level = strength * (.78 + .22 * velocity / 127)
                    ceiling = (48 if lane == 'delay' else 58) if style == 'original' else (88 if lane == 'delay' else 96)
                level *= _clamp(.4 + 1.2 * painted, 0, 1.6)
                if intensity != 1:
                    level *= depth
                amount = max(amount, (ceiling - 1) * min(1.0, level) * envelope)
            value = 1 + amount
            if movement == 'drop' and policy != 'keep':
                protected_drop = drop_index if in_drop else None
                if protected_drop is None and next_drop < len(drops) and 0 < drop_ticks[next_drop] - tick <= before_hit:
                    protected_drop = next_drop
                if protected_drop is not None:
                    if policy == 'dry':
                        value = 1
                    else:
                        recovery = _ease((tick - anchors[protected_drop] - ppq / 2) / (1.5 * ppq))
                        # A cap on an existing tail, not an added all-song wet
                        # floor. The first actual drop kick gets a clear hit.
                        value = min(value, 1 + 126 * .3 * recovery)
        if tick in (0, end - 1):
            value = baseline
        velocities.append(int(math.floor(_clamp(value, 1, 102 if lane == 'volume' else 127) + .5)))

    output = MidiFile(type=1, ticks_per_beat=ppq)
    output.tracks.append(source_meta_track(source_midi))
    track = MidiTrack([MetaMessage('track_name', name=f'Drum Monkey other drums CTRL {lane}', time=0)])
    output.tracks.append(track)
    last_tick = 0
    for index, (tick, velocity) in enumerate(zip(ticks, velocities)):
        stop = ticks[index + 1] if index + 1 < len(ticks) else end
        duration = min(stop - tick, max(1, round((stop - tick) * gate)))
        track.append(Message('note_on', note=60, velocity=velocity, channel=0, time=tick - last_tick))
        track.append(Message('note_off', note=60, velocity=0, channel=0, time=duration))
        last_tick = tick + duration
    track.append(MetaMessage('end_of_track', time=end - last_tick))
    output.save(Path(output_path))
    return {'lane': lane, 'baseline_velocity': baseline, 'peak_velocity': max(velocities),
            'minimum_velocity': min(velocities), 'controller_notes': len(velocities),
            'analyzed_non_kick_notes': len(notes), 'effect_throws': len(envelopes),
            'drop_anchor_ticks': anchors, 'kick_controls_created': 0,
            'effect_scope': 'Shared other-drums bus; caller supplies routing, this curve module does not change it.'}
