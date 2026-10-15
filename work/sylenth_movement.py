"""Preset-relative, note-aware curves for continuous Sylenth parameters.

Parameter discovery and safe ranges belong to the preset inspector. This module
never chooses a parameter, edits a patch, or interprets a VST parameter index.
It returns absolute MIDI ticks and normalized parameter values for the exporter.
"""
from __future__ import annotations

from bisect import bisect_right
import hashlib
import math
import random

from energy_curve import sample_energy_curve
from velocity_automation_midi import (
    MOVEMENTS, MovementPlan, NoteEvent, intensity_factors, merged_fx_spans, song_end_tick,
)


CHARACTERS = frozenset({'brightness', 'drive', 'decay', 'release', 'width',
                        'modulation', 'effect'})


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _ease(value: float) -> float:
    value = _clamp(value)
    return value * value * (3.0 - 2.0 * value)


def _drop_spans(plan: MovementPlan, end: int) -> list[tuple[int, int]]:
    result = []
    for begin, stop in sorted(plan.drops):
        begin, stop = max(0, int(begin)), min(end, int(stop))
        if stop <= begin:
            continue
        if result and begin <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], stop))
        else:
            result.append((begin, stop))
    return result


def _activity(notes: list[NoteEvent], ticks: list[int], ppq: int):
    """Sweep note edges once; polyphony must not make this quadratic."""
    changes = []
    for note in notes:
        velocity = _clamp(note.velocity / 127.0)
        changes.append((note.tick, 1, velocity))
        changes.append((note.tick + note.duration, -1, -velocity))
    changes.sort(key=lambda item: item[0])
    cursor, count = 0, 0
    velocity_sum = 0.0
    recent_onset = -100 * ppq
    onset_velocity = last_activity = 0.0
    result = []
    for tick in ticks:
        while cursor < len(changes) and changes[cursor][0] <= tick:
            position, change, velocity = changes[cursor]
            count += change
            velocity_sum += velocity
            if change > 0:
                if position != recent_onset:
                    onset_velocity = velocity
                else:
                    onset_velocity = max(onset_velocity, velocity)
                recent_onset = position
            cursor += 1
        if count > 0:
            average = _clamp(velocity_sum / count)
            last_activity = .82 * average + .18 * min(1.0, count / 4.0)
        # Hold the musical response through short articulation gaps and tails;
        # the separately calculated phrase gate brings long rests to baseline.
        accent = onset_velocity * math.exp(-(tick - recent_onset) / max(1, ppq * .35))
        result.append((last_activity, accent))
    return result


def generate_points(control: dict, notes: list[NoteEvent], midi, options: dict,
                    plan: MovementPlan, energy_curve=None) -> list[tuple[int, float]]:
    """Generate a bounded, reproducible curve around the saved preset value.

    Required control fields: parameter_id, baseline, minimum, maximum,
    character. Optional role is bass/chords/arp/lead/default. All three values
    are normalized and must satisfy 0 <= minimum <= baseline <= maximum <= 1.
    Optional channel_id (or synth_channel) separates otherwise identical parts.

    The first and final points, and rests beyond the short return ramp, equal
    baseline exactly. Strength zero returns just those two baseline points.
    Optional intensity (0..2, default 1) scales depth around that saved value;
    two gives up to three times the movement, still inside the supplied bounds.
    Input note durations remain untouched. Curves have a 16th/32nd-note grid
    plus exact note, phrase, drop, painting and return-to-baseline boundaries.
    """
    minimum, baseline, maximum = (float(control[key]) for key in
                                  ('minimum', 'baseline', 'maximum'))
    if (not all(math.isfinite(value) for value in (minimum, baseline, maximum))
            or not 0 <= minimum <= baseline <= maximum <= 1):
        raise ValueError('Sylenth automation needs safe bounds containing the saved preset value.')
    character = control['character']
    if character not in CHARACTERS:
        raise ValueError('Unsupported Sylenth automation character.')
    role = control.get('role', 'default')
    if role not in {'bass', 'chords', 'arp', 'lead', 'default'}:
        role = 'default'
    strength = float(options.get('strength', .65))
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Sylenth movement strength must be between zero and one.')
    intensity = float(options.get('intensity', 1.0))
    depth, _, release_scale = intensity_factors(intensity)
    movement = options.get('movement', 'drop')
    if movement not in MOVEMENTS:
        raise ValueError('Unsupported Sylenth movement style.')
    policy = options.get('drop_fx', 'reduced')
    if policy not in {'dry', 'reduced', 'keep'}:
        raise ValueError('Unsupported drop effect policy.')
    end = song_end_tick(midi)
    ppq = midi.ticks_per_beat
    if ppq <= 0 or end <= 0:
        raise ValueError('Sylenth automation needs a song with musical beat timing and duration.')
    last = end - 1
    resting = [(0, baseline)] if not last else [(0, baseline), (last, baseline)]
    if not strength or not intensity or minimum == maximum or not last:
        return resting
    valid_notes = sorted((note for note in notes if note.duration > 0
                          and note.velocity > 0 and 0 <= note.tick < end),
                         key=lambda note: note.tick)
    if not valid_notes:
        return resting

    # Related notes remain a phrase, instead of resetting a knob on every
    # sixteenth-note pluck. Onset and envelope controls retain their patch shape.
    spans = merged_fx_spans(valid_notes, ppq, role)
    attack = max(1, round(ppq * (.125 if role == 'arp' else .25)))
    if character in {'width', 'modulation'} and role == 'chords':
        attack = max(1, round(ppq * .5))
    tail = max(1, round(ppq * (2 if character == 'effect' else 1)))
    if character == 'effect' and intensity > 1:
        tail = max(1, round(tail * release_scale))
    envelopes = []
    for index, (start, stop, _) in enumerate(spans):
        following = spans[index + 1][0] if index + 1 < len(spans) else end
        stop = min(stop, end)
        # Return early enough to make the middle of a real rest exactly the
        # saved patch. Do not run a fade into the next phrase's first note.
        release_end = min(last, stop + tail, following)
        release_start = min(stop, release_end)
        if release_start == release_end:
            release_start = max(start, release_end - attack)
        attack_end = min(start + attack, max(start + 1, release_start))
        envelopes.append((start, attack_end, release_start, release_end))

    division = 32 if int(options.get('division', 16)) >= 32 else 16
    count = math.ceil(end * division / (ppq * 4))
    positions = {index * ppq * 4 // division for index in range(count)}
    positions.update((0, last))
    for note in valid_notes:
        positions.update((note.tick, note.tick + note.duration))
    for envelope in envelopes:
        positions.update(envelope)
        # Short attacks must have an intermediate point, even at low PPQ.
        positions.add((envelope[0] + envelope[1]) // 2)
        positions.add((envelope[2] + envelope[3]) // 2)
    boundaries = sorted({0, end, *(int(tick) for tick in plan.phrase_bounds
                                    if 0 < tick < end)})
    positions.update(boundaries)
    drops = _drop_spans(plan, end)
    drop_starts = [start for start, _ in drops]
    for start, stop in drops:
        positions.update((start - 8 * ppq, start - ppq // 4, start,
                          start + ppq // 2, start + 2 * ppq, stop))
    if energy_curve is not None:
        positions.update(int(tick) for tick, _ in energy_curve['points'])
    ticks = sorted(tick for tick in positions if 0 <= tick <= last)
    painted = (sample_energy_curve(energy_curve, ticks) if energy_curve is not None
               else [.5] * len(ticks))
    activity = _activity(valid_notes, ticks, ppq)

    identity = '{}:{}:{}:{}'.format(control['parameter_id'], role, character,
                                     control.get('channel_id', control.get('synth_channel', '')))
    digest = int.from_bytes(hashlib.sha256(identity.encode('utf-8')).digest()[:8], 'little')
    rng = random.Random(int(options.get('seed', 0)) + digest)
    anchor_length = 8 * ppq  # Coherent two-bar gestures, not per-grid noise.
    anchors = [rng.uniform(-1, 1) for _ in range(end // anchor_length + 2)]
    phase = rng.uniform(0, math.tau)
    span_index = 0
    current = 0.0
    previous_tick = 0
    result = []
    for tick, energy, (activity_value, accent) in zip(ticks, painted, activity):
        while span_index + 1 < len(envelopes) and envelopes[span_index + 1][0] <= tick:
            span_index += 1
        start, attack_end, release_start, release_end = envelopes[span_index]
        if tick <= start or tick >= release_end:
            gate = 0.0
        else:
            gate = _ease((tick - start) / max(1, attack_end - start))
            if tick > release_start:
                gate *= 1 - _ease((tick - release_start) / max(1, release_end - release_start))
        phrase_index = max(0, bisect_right(boundaries, tick) - 1)
        phrase_start, phrase_end = boundaries[phrase_index:phrase_index + 2]
        build_length = min(8 * ppq, max(1, (phrase_end - phrase_start) / 2))
        lift = _ease((tick - (phrase_end - build_length)) / build_length)
        index, remainder = divmod(tick, anchor_length)
        blend = _ease(remainder / anchor_length)
        drift = anchors[index] * (1 - blend) + anchors[index + 1] * blend
        response = _clamp((activity_value - .55) * 2, -1, 1)
        paint = (energy - .5) * 2
        # The groove term is deliberately small: articulation comes from the
        # notes and patch envelopes, not an imposed full-depth synth wobble.
        groove = math.sin(math.tau * tick / (2 * ppq) + phase)
        target = .22 * response + .10 * drift + .42 * paint
        if character in {'brightness', 'drive', 'modulation'}:
            target += .13 * accent
        if movement in {'transitions', 'full'}:
            target += .32 * lift
        if movement in {'groove', 'full'}:
            target += (.08 if character in {'decay', 'release'} else .16) * groove
        if movement == 'drop' and drops:
            current_drop = bisect_right(drop_starts, tick) - 1
            inside = current_drop >= 0 and tick < drops[current_drop][1]
            coming = current_drop + 1
            tension = 0.0
            if coming < len(drops):
                tension = _ease(1 - (drops[coming][0] - tick) / (8 * ppq))
            if not inside and tension:
                target += (-.52 if character in {'brightness', 'width'} else .35) * tension
            if inside:
                age = tick - drops[current_drop][0]
                if character in {'brightness', 'drive', 'modulation'}:
                    target += .42
                elif character == 'width':
                    target += -.3 if role == 'bass' else .22
                elif character in {'decay', 'release'}:
                    target -= .12  # Small tightening, not a new envelope.
                elif character == 'effect' and policy != 'keep':
                    recovery = _ease((age - ppq / 2) / (1.5 * ppq))
                    target = -1.0 if policy == 'dry' else -1.0 + .8 * recovery
            elif character == 'effect' and policy != 'keep' and coming < len(drops):
                # Begin clearing the internal wet control just before the hit.
                distance = drops[coming][0] - tick
                if distance <= ppq / 4:
                    target = -1.0

        # Saved short pluck envelopes should stay short. The inspector's hard
        # bounds apply too; these per-character amounts are a second restraint.
        restraint = {'brightness': 1.0, 'drive': .70, 'decay': .48, 'release': .38,
                     'width': .75, 'modulation': .75, 'effect': 1.0}[character]
        if role == 'arp' and character in {'decay', 'release'}:
            restraint *= .55
        target = _clamp(target * restraint, -1.0, 1.0)
        smoothing = ppq * (.5 if character in {'width', 'modulation'} else .16)
        coefficient = 1 - math.exp(-(tick - previous_tick) / max(1.0, smoothing))
        if gate == 0:
            current = 0.0
        else:
            current += (target - current) * coefficient
        previous_tick = tick
        amount = current * strength * gate
        if intensity != 1:
            amount = _clamp(amount * depth, -1.0, 1.0)
        extent = maximum - baseline if amount >= 0 else baseline - minimum
        value = baseline + amount * extent
        if tick == 0 or tick == last or gate == 0:
            value = baseline
        result.append((tick, _clamp(value, minimum, maximum)))
    return result
