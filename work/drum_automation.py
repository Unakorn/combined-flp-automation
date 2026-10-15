"""Musical, role-specific drum controller MIDI without changing source notes."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
import hashlib
import math
from pathlib import Path
import random
from statistics import median

from mido import Message, MetaMessage, MidiFile, MidiTrack
from energy_curve import sample_energy_curve
from velocity_automation_midi import (
    MOVEMENTS, MovementPlan, NoteEvent, build_grid, song_end_tick, source_meta_track,
)


ROLE_LANES = {
    'kick': ('volume',),
    'snare': ('volume', 'delay', 'reverb'),
    'clap': ('volume', 'delay', 'reverb'),
    'off_snare': ('volume', 'reverb'),
    'closed_hat': ('volume', 'cutoff'),
    'open_hat': ('volume', 'cutoff'),
    'tom': ('volume', 'delay', 'reverb'),
    'percussion': ('volume', 'cutoff'),
}
VOLUME_BASELINE = 102  # Native Balance unity is approximately 102/128 of its range.


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _ease(value):
    value = _clamp(value)
    return value * value * (3 - 2 * value)


def _number(options, name, default, low, high, integer=False):
    value = options.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a number.')
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high or (integer and value != int(value)):
        raise ValueError(f'{name} must be between {low} and {high}.')
    return int(value) if integer else value


def _drops(plan: MovementPlan, end: int):
    result = []
    for start, stop in sorted(plan.drops):
        start, stop = max(0, int(start)), min(end, int(stop))
        if stop <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(stop, result[-1][1]))
        else:
            result.append((start, stop))
    return result


def _hit_envelopes(hits, boundaries, ppq, end, lane, role, style, seed):
    """Choose a last fill/phrase hit, not every normal backbeat."""
    result = []
    starts = [tick for tick, _ in hits]
    usual_gap = median([b - a for a, b in zip(starts, starts[1:])]) if len(starts) > 1 else ppq
    rest_threshold = max(3 * ppq, usual_gap * 2.5)
    salt = int.from_bytes(hashlib.sha256((role + ':' + lane + ':throws').encode()).digest()[:8], 'little')
    rng = random.Random(seed + salt)
    cluster_start = 0
    for index, (tick, velocity) in enumerate(hits):
        if index and tick - hits[index - 1][0] > ppq * .5:
            cluster_start = index
        following = starts[index + 1] if index + 1 < len(starts) else end + 4 * ppq
        phrase_index = min(len(boundaries) - 1, bisect_right(boundaries, tick))
        phrase_end = boundaries[phrase_index]
        near_end = 0 < phrase_end - tick <= ppq * (2 if role == 'tom' else 1.5)
        phrase_last = near_end and following >= phrase_end
        fill_last = index - cluster_start + 1 >= (2 if role == 'tom' else 3) and following - tick > ppq * .5
        long_rest = following - tick >= rest_threshold or index + 1 == len(hits)
        # A single opening hit is not an automatic effect throw simply because
        # it is followed by space. Require phrase/fill context or prior hits.
        rest_last = long_rest and index > 0
        if not (phrase_last or fill_last or rest_last):
            continue
        if style == 'original':
            hold, release = (.125, .5) if lane == 'delay' else (.25, 1)
        else:
            hold, release = (.5, 2) if lane == 'delay' else (.75, 3)
        scale = .65 if role == 'off_snare' else .85 if role == 'tom' else 1.0
        attack_start = max(0, tick - max(1, round(ppq * .125)))
        hold_end = min(end - 1, tick + max(1, round(hold * scale * ppq * rng.uniform(.9, 1.1))))
        release_end = min(end - 1, hold_end + max(1, round(release * scale * ppq * rng.uniform(.92, 1.08))))
        result.append((attack_start, tick, hold_end, release_end, velocity))
    return result


def make_drum_ctrl_midi(source_midi: MidiFile, notes: list[NoteEvent], role: str,
                        lane: str, output_path: Path, options: dict,
                        plan: MovementPlan, energy_curve=None) -> dict:
    """Write one fixed-note (60) controller part using original MIDI timing.

    Volume rests at 102 and never boosts above it. Cutoff rests at 127 and uses
    a small role-specific range. Effects rest at 1 and appear only at selected
    phrase/fill endings. Strength zero writes constant baseline controls.
    Existing global min/max, invert and controller-note settings intentionally
    do not move these role-specific unity/dry/open baselines.
    """
    if role not in ROLE_LANES or lane not in ROLE_LANES[role]:
        raise ValueError('This drum role does not use the requested automation lane.')
    strength = _number(options, 'strength', .8, 0, 1)
    division = _number(options, 'division', 16, 1, 64, True)
    seed = _number(options, 'seed', 1, 0, 2147483647, True)
    gate = _number(options, 'gate', .9, .01, 1)
    movement = options.get('movement', 'drop')
    policy = options.get('drop_fx', 'reduced')
    style = options.get('fx_style', 'full_range')
    if movement not in MOVEMENTS or policy not in {'dry', 'reduced', 'keep'}:
        raise ValueError('Choose a supported drum movement and drop FX setting.')
    if style not in {'original', 'fuller', 'full_range'}:
        raise ValueError('Choose a supported drum FX style.')
    ppq, end = source_midi.ticks_per_beat, song_end_tick(source_midi)
    if ppq <= 0 or end <= 0 or plan.end_tick != end:
        raise ValueError('Drum automation needs a matching musical-time song and movement plan.')
    baseline = VOLUME_BASELINE if lane == 'volume' else 127 if lane == 'cutoff' else 1
    note_list = sorted((note for note in notes if 0 <= note.tick < end
                        and note.duration > 0 and note.velocity > 0), key=lambda note: note.tick)
    # Simultaneous notes are one rhythmic onset; a layer or flam must not
    # manufacture a dense fill simply by placing several pitches together.
    grouped = {}
    for note in note_list:
        grouped[note.tick] = max(grouped.get(note.tick, 0), int(_clamp(note.velocity, 1, 127)))
    hits = sorted(grouped.items())
    hit_ticks = [tick for tick, _ in hits]
    boundaries = sorted({0, end, *(int(tick) for tick in plan.phrase_bounds if 0 < tick < end)})
    drops = _drops(plan, end)
    drop_ticks = [tick for tick, _ in drops]
    envelopes = (_hit_envelopes(hits, boundaries, ppq, end, lane, role, style, seed)
                 if strength and lane in {'delay', 'reverb'} else [])
    gaps = []
    if role == 'kick' and movement == 'drop' and strength:
        for begin, _ in drops:
            previous = bisect_left(hit_ticks, begin) - 1
            # Only a recent kick can have a tail to interrupt. Do not insert
            # fake kick motion into silence or around a first downbeat.
            if previous >= 0 and begin - hits[previous][0] <= ppq * .5:
                gaps.append((max(hits[previous][0], begin - max(1, round(ppq * .125))), begin))

    ticks, _ = build_grid(note_list, ppq, division, end)
    positions = {0, end - 1, *ticks, *hit_ticks, *boundaries}
    active_length = max(1, round(ppq * {'kick': .5, 'snare': .5, 'clap': .5,
        'off_snare': .35, 'closed_hat': .25, 'open_hat': .5, 'tom': .6, 'percussion': .4}[role]))
    positions.update(tick + active_length for tick in hit_ticks)
    for start, peak, hold, stop, _ in envelopes:
        positions.update((start, peak, hold, stop, (start + peak) // 2, (hold + stop) // 2))
    for start, stop in gaps:
        positions.update((start, stop, start - 1))
    for start, stop in drops:
        positions.update((start - 8 * ppq, start - max(1, ppq // 8), start,
                          start + ppq // 2, start + 2 * ppq, stop))
    if energy_curve is not None:
        positions.update(int(tick) for tick, _ in energy_curve['points'])
    ticks = sorted(tick for tick in positions if 0 <= tick < end)
    energy = sample_energy_curve(energy_curve, ticks) if energy_curve is not None else [.5] * len(ticks)
    digest = int.from_bytes(hashlib.sha256((role + ':' + lane).encode()).digest()[:8], 'little')
    rng = random.Random(seed + digest)
    bar_variation = [rng.uniform(-1, 1) for _ in range(end // (8 * ppq) + 2)]
    velocities = []
    active_fx = []
    fx_index = 0
    gap_index = 0
    for tick, painted in zip(ticks, energy):
        value = float(baseline)
        hit_index = bisect_right(hit_ticks, tick) - 1
        active = hit_index >= 0 and tick - hits[hit_index][0] < active_length
        amplitude = hits[hit_index][1] / 127 if hit_index >= 0 else 0
        phrase_index = max(0, bisect_right(boundaries, tick) - 1)
        phrase_start, phrase_end = boundaries[phrase_index:phrase_index + 2]
        ramp = min(8 * ppq, max(1, (phrase_end - phrase_start) // 2))
        lift = _ease((tick - phrase_end + ramp) / ramp)
        drop_index = bisect_right(drop_ticks, tick) - 1
        in_drop = drop_index >= 0 and tick < drops[drop_index][1]
        next_drop = drop_index + 1
        tension = (_ease(1 - (drops[next_drop][0] - tick) / (8 * ppq))
                   if next_drop < len(drops) else 0)
        phase, remainder = divmod(tick, 8 * ppq)
        blend = _ease(remainder / (8 * ppq))
        variation = bar_variation[phase] * (1 - blend) + bar_variation[phase + 1] * blend
        if strength and lane == 'volume':
            if role == 'kick':
                while gap_index < len(gaps) and gaps[gap_index][1] <= tick:
                    gap_index += 1
                if gap_index < len(gaps) and gaps[gap_index][0] <= tick < gaps[gap_index][1]:
                    value = baseline - (baseline - 1) * strength
            elif active:
                # Source note velocities already carry the groove. Add only a
                # few controller steps of contrast, with no gain increase.
                reduction = 2 + (1 - amplitude) * 7
                if movement in {'groove', 'full'}:
                    beat_fraction = (hits[hit_index][0] / ppq) % 1
                    reduction += 3 * math.sin(math.pi * beat_fraction) ** 2
                reduction += max(0, .5 - painted) * 12 - max(0, painted - .5) * 6
                if movement in {'transitions', 'full'}:
                    reduction *= 1 - .6 * lift
                if movement == 'drop':
                    reduction *= 1 - .65 * tension
                    if in_drop and tick - drops[drop_index][0] < ppq:
                        reduction = 0
                reduction += .65 * variation
                value -= strength * max(0, reduction)
        elif strength and lane == 'cutoff' and active:
            floor = {'closed_hat': 94, 'open_hat': 102, 'percussion': 98}[role]
            opening = .45 + .28 * amplitude + .28 * (painted - .5)
            if movement in {'transitions', 'full'}:
                opening += .35 * lift
            if movement == 'drop':
                opening += .35 * tension
                if in_drop:
                    opening = 1
            if movement in {'groove', 'full'}:
                opening += .07 * math.sin(math.tau * tick / (2 * ppq))
            opening += .04 * variation
            value = baseline - strength * (baseline - floor) * (1 - _clamp(opening))
        elif strength and lane in {'delay', 'reverb'}:
            while fx_index < len(envelopes) and envelopes[fx_index][0] <= tick:
                active_fx.append(envelopes[fx_index])
                fx_index += 1
            active_fx = [envelope for envelope in active_fx if tick < envelope[3]]
            amount = 0.0
            for start, peak, hold, stop, hit_velocity in active_fx:
                if tick < peak:
                    envelope = _ease((tick - start) / max(1, peak - start))
                elif tick <= hold:
                    envelope = 1.0
                else:
                    envelope = 1 - _ease((tick - hold) / max(1, stop - hold))
                # Full-range is deliberate: useful throws can reach 127 at the
                # normal strength, while gentle amounts still scale down.
                if style == 'full_range':
                    level = min(1.0, strength / .65)
                    peak_velocity = 127
                else:
                    level = strength * (.78 + .22 * hit_velocity / 127)
                    peak_velocity = (52 if lane == 'delay' else 64) if style == 'original' else (92 if lane == 'delay' else 100)
                level *= _clamp(.4 + 1.2 * painted, 0, 1.6)
                amount = max(amount, (peak_velocity - 1) * min(1.0, level) * envelope)
            value = 1 + amount
            if movement == 'drop' and policy != 'keep':
                protected = in_drop
                drop_start = drops[drop_index][0] if in_drop else None
                if not protected and next_drop < len(drops) and 0 < drops[next_drop][0] - tick <= max(1, ppq // 8):
                    protected, drop_start = True, drops[next_drop][0]
                if protected:
                    if policy == 'dry':
                        value = 1
                    else:
                        recovery = _ease((tick - drop_start - ppq / 2) / (1.5 * ppq))
                        # Keep the hit clear, then allow a little existing tail;
                        # do not add a wet floor when no throw exists.
                        ceiling = 1 + 126 * .3 * recovery
                        value = min(value, ceiling)
        if tick in (0, end - 1):
            value = baseline
        velocities.append(int(math.floor(_clamp(value, 1, baseline if lane == 'volume' else 127) + .5)))

    output = MidiFile(type=1, ticks_per_beat=ppq)
    output.tracks.append(source_meta_track(source_midi))
    track = MidiTrack([MetaMessage('track_name', name=f'{role} CTRL {lane} drums', time=0)])
    output.tracks.append(track)
    last_tick = 0
    for index, (tick, velocity) in enumerate(zip(ticks, velocities)):
        stop = ticks[index + 1] if index + 1 < len(ticks) else end
        length = min(stop - tick, max(1, round((stop - tick) * gate)))
        track.append(Message('note_on', note=60, velocity=velocity, channel=0, time=tick - last_tick))
        track.append(Message('note_off', note=60, velocity=0, channel=0, time=length))
        last_tick = tick + length
    track.append(MetaMessage('end_of_track', time=end - last_tick))
    output.save(Path(output_path))
    return {'role': role, 'lane': lane, 'baseline_velocity': baseline,
            'peak_velocity': max(velocities), 'minimum_velocity': min(velocities),
            'controller_notes': len(velocities), 'source_notes': len(note_list),
            'effect_throws': len(envelopes), 'kick_gaps': len(gaps)}
