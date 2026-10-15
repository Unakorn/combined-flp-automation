from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
from io import BytesIO
import json
import math
import random
import re
import zlib
from dataclasses import dataclass
from pathlib import Path

from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo
from energy_curve import CurveError, load_curve, sample_energy_curve, save_curve


LANES = {
    "cutoff",
    "reverb",
    "delay",
    "distortion",
    "decay",
    "width",
    "volume",
    "formant",
    "pitch",
}


DEFAULT_PROFILES = {
    "cutoff": {"low": 58, "high": 122, "smooth": 0.18, "accent": 0.22, "gamma": 0.72},
    "reverb": {"low": 4, "high": 58, "smooth": 0.10, "accent": 0.78, "gamma": 1.65},
    "delay": {"low": 3, "high": 64, "smooth": 0.08, "accent": 0.82, "gamma": 1.7},
    "distortion": {"low": 8, "high": 62, "smooth": 0.14, "accent": 0.30, "gamma": 0.9},
    "decay": {"low": 28, "high": 78, "smooth": 0.16, "accent": 0.18, "gamma": 1.0},
    "width": {"low": 42, "high": 96, "smooth": 0.12, "accent": 0.22, "gamma": 0.9},
    "volume": {"low": 74, "high": 108, "smooth": 0.22, "accent": 0.46, "gamma": 1.25},
    "formant": {"low": 20, "high": 95, "smooth": 0.12, "accent": 0.38, "gamma": 1.0},
    "pitch": {"low": 24, "high": 102, "smooth": 0.10, "accent": 0.48, "gamma": 1.0},
}


ROLE_PROFILES = {
    "drums": {
        "cutoff": {"low": 90, "high": 127, "smooth": 0.20, "accent": 0.14, "gamma": 0.65},
        "reverb": {"low": 1, "high": 48, "smooth": 0.08, "accent": 0.88, "gamma": 2.0},
        "delay": {"low": 1, "high": 22, "smooth": 0.06, "accent": 0.92, "gamma": 2.2},
        "distortion": {"low": 10, "high": 54, "smooth": 0.14, "accent": 0.28, "gamma": 0.95},
        "decay": {"low": 42, "high": 86, "smooth": 0.14, "accent": 0.18, "gamma": 1.0},
        "width": {"low": 38, "high": 76, "smooth": 0.10, "accent": 0.18, "gamma": 1.0},
        "volume": {"low": 88, "high": 114, "smooth": 0.22, "accent": 0.42, "gamma": 1.25},
    },
    "bass": {
        "cutoff": {"low": 82, "high": 124, "smooth": 0.20, "accent": 0.16, "gamma": 0.70},
        "reverb": {"low": 1, "high": 12, "smooth": 0.08, "accent": 0.88, "gamma": 2.4},
        "delay": {"low": 1, "high": 10, "smooth": 0.06, "accent": 0.90, "gamma": 2.4},
        "distortion": {"low": 22, "high": 84, "smooth": 0.14, "accent": 0.24, "gamma": 0.82},
        "decay": {"low": 34, "high": 72, "smooth": 0.18, "accent": 0.16, "gamma": 1.0},
        "width": {"low": 1, "high": 18, "smooth": 0.10, "accent": 0.12, "gamma": 1.2},
        "volume": {"low": 86, "high": 110, "smooth": 0.24, "accent": 0.38, "gamma": 1.3},
    },
    "chords": {
        "cutoff": {"low": 68, "high": 124, "smooth": 0.18, "accent": 0.22, "gamma": 0.72},
        "reverb": {"low": 8, "high": 68, "smooth": 0.10, "accent": 0.72, "gamma": 1.6},
        "delay": {"low": 3, "high": 42, "smooth": 0.08, "accent": 0.78, "gamma": 1.8},
        "distortion": {"low": 4, "high": 42, "smooth": 0.12, "accent": 0.22, "gamma": 1.1},
        "decay": {"low": 34, "high": 84, "smooth": 0.16, "accent": 0.16, "gamma": 1.0},
        "width": {"low": 64, "high": 116, "smooth": 0.12, "accent": 0.20, "gamma": 0.9},
        "volume": {"low": 74, "high": 108, "smooth": 0.24, "accent": 0.48, "gamma": 1.25},
    },
    "arp": {
        "cutoff": {"low": 58, "high": 124, "smooth": 0.15, "accent": 0.26, "gamma": 0.75},
        "reverb": {"low": 5, "high": 48, "smooth": 0.08, "accent": 0.74, "gamma": 1.7},
        "delay": {"low": 8, "high": 78, "smooth": 0.07, "accent": 0.70, "gamma": 1.35},
        "distortion": {"low": 4, "high": 46, "smooth": 0.12, "accent": 0.24, "gamma": 1.0},
        "decay": {"low": 24, "high": 66, "smooth": 0.14, "accent": 0.18, "gamma": 1.0},
        "width": {"low": 50, "high": 98, "smooth": 0.10, "accent": 0.20, "gamma": 0.95},
        "volume": {"low": 70, "high": 104, "smooth": 0.20, "accent": 0.48, "gamma": 1.3},
    },
    "lead": {
        "cutoff": {"low": 70, "high": 126, "smooth": 0.17, "accent": 0.24, "gamma": 0.72},
        "reverb": {"low": 7, "high": 72, "smooth": 0.08, "accent": 0.78, "gamma": 1.55},
        "delay": {"low": 8, "high": 84, "smooth": 0.07, "accent": 0.78, "gamma": 1.35},
        "distortion": {"low": 8, "high": 56, "smooth": 0.12, "accent": 0.26, "gamma": 0.95},
        "decay": {"low": 26, "high": 70, "smooth": 0.14, "accent": 0.20, "gamma": 1.0},
        "width": {"low": 46, "high": 106, "smooth": 0.11, "accent": 0.20, "gamma": 0.95},
        "volume": {"low": 72, "high": 110, "smooth": 0.22, "accent": 0.50, "gamma": 1.25},
    },
}


IMPACT_LANES_BY_ROLE = {
    "drums": ("volume", "decay", "reverb"),
    "bass": ("volume", "cutoff", "distortion"),
    "chords": ("volume", "cutoff", "width"),
    "arp": ("volume", "cutoff", "delay"),
    "lead": ("volume", "cutoff", "delay"),
    "default": ("volume", "cutoff", "delay"),
}

EXTRA_LANE_BY_ROLE = {"drums": "delay", "chords": "reverb", "arp": "width", "lead": "reverb", "default": "reverb"}
MOVEMENTS = ("classic", "transitions", "groove", "full", "drop")


def song_end_tick(mid: MidiFile) -> int:
    return max((sum(message.time for message in track) for track in mid.tracks), default=0)


def timed_meta(mid: MidiFile) -> list[tuple[int, object]]:
    result = []
    for track in mid.tracks:
        tick = 0
        for message in track:
            tick += message.time
            if message.is_meta:
                result.append((tick, message))
    return sorted(result, key=lambda item: item[0])


@dataclass
class MovementPlan:
    end_tick: int
    phrase_bounds: list[int]
    drops: list[tuple[int, int]]
    drop_source: str


def movement_plan(mid: MidiFile, phrase_bars: int, drop_bars: str, drop_length: int) -> MovementPlan:
    end = song_end_tick(mid)
    bar = mid.ticks_per_beat * 4
    markers = [(tick, msg.text) for tick, msg in timed_meta(mid)
               if msg.type in {"marker", "cue_marker"} and 0 <= tick < end]
    boundaries = {0, end}
    section_edges = sorted({0, end, *(tick for tick, _ in markers)})
    for start, stop in zip(section_edges, section_edges[1:]):
        boundaries.update(range(start, stop, phrase_bars * bar))
    if drop_bars.strip():
        if not re.fullmatch(r"\s*\d+(?:\s*,\s*\d+)*\s*", drop_bars):
            raise ValueError("Drop bars must be whole bar numbers separated by commas, such as 17,49.")
        positions = sorted({(int(value.strip()) - 1) * bar for value in drop_bars.split(',')})
        if any(tick < 0 or tick >= end for tick in positions):
            raise ValueError("Each drop bar must be inside the source song (bar 1 is the start).")
        origin = "your drop bars"
    else:
        positions = sorted({tick for tick, name in markers
                            if re.search(r"\bdrop\b", name.replace('_', ' '), re.I)
                            and not re.search(r"\b(pre|before|build|end|after)\b", name.replace('_', ' '), re.I)})
        origin = "MIDI Drop markers"
        if not positions:
            positions = list(range(0, end, phrase_bars * bar))
            origin = f"{phrase_bars}-bar grid (no Drop markers; set Drop bars for exact positions)"
    return MovementPlan(end, sorted(boundaries),
                        [(tick, min(end, tick + drop_length * bar)) for tick in positions], origin)


@dataclass
class NoteEvent:
    tick: int
    duration: int
    note: int
    velocity: int
    channel: int


def slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return value or "track"


def detect_role(track_name: str) -> str:
    name = slug(track_name)
    if any(word in name for word in ("drum", "kick", "snare", "clap", "hat", "perc", "tom", "ride", "crash")):
        return "drums"
    if any(word in name for word in ("bass", "sub", "808")):
        return "bass"
    if any(word in name for word in ("chord", "pad", "stab")):
        return "chords"
    if any(word in name for word in ("arp", "pluck", "seq")):
        return "arp"
    if any(word in name for word in ("lead", "melody", "hook", "top")):
        return "lead"
    return "default"


def lane_profile(lane: str, role: str) -> dict[str, float]:
    profile = DEFAULT_PROFILES[lane].copy()
    profile.update(ROLE_PROFILES.get(role, {}).get(lane, {}))
    return profile


def read_track_notes(mid: MidiFile) -> list[tuple[str, int, list[NoteEvent]]]:
    tracks: list[tuple[str, int, list[NoteEvent]]] = []
    end_tick = song_end_tick(mid)
    for track_index, track in enumerate(mid.tracks):
        abs_tick = 0
        name = f"track_{track_index + 1}"
        open_notes: dict[tuple[int, int], list[tuple[int, int]]] = {}
        notes: list[NoteEvent] = []

        for msg in track:
            abs_tick += msg.time
            if msg.is_meta:
                if msg.type == "track_name" and msg.name.strip():
                    name = msg.name.strip()
                continue

            if msg.type == "note_on" and msg.velocity > 0:
                open_notes.setdefault((msg.channel, msg.note), []).append((abs_tick, msg.velocity))
            elif msg.type in {"note_off", "note_on"}:
                key = (msg.channel, msg.note)
                if key not in open_notes or not open_notes[key]:
                    continue
                start_tick, velocity = open_notes[key].pop(0)
                notes.append(
                    NoteEvent(
                        tick=start_tick,
                        duration=max(1, abs_tick - start_tick),
                        note=msg.note,
                        velocity=velocity,
                        channel=msg.channel,
                    )
                )

        # Hold a missing note-off only to the known song end, never discard it.
        for (channel, note), starts in open_notes.items():
            for start_tick, velocity in starts:
                if start_tick < end_tick:
                    notes.append(NoteEvent(start_tick, end_tick - start_tick, note, velocity, channel))
        if notes:
            tracks.append((name, track_index, sorted(notes, key=lambda n: n.tick)))

    return tracks


def source_meta_track(mid: MidiFile) -> MidiTrack:
    meta = MidiTrack()
    events = [(tick, msg) for tick, msg in timed_meta(mid)
              if msg.type in {"set_tempo", "time_signature", "key_signature", "marker", "cue_marker"}]
    # MIDI defaults apply until the first explicit event, even if it occurs later.
    if not any(tick == 0 and msg.type == 'set_tempo' for tick, msg in events):
        events.insert(0, (0, MetaMessage('set_tempo', tempo=bpm2tempo(120))))
    if not any(tick == 0 and msg.type == 'time_signature' for tick, msg in events):
        events.insert(0, (0, MetaMessage('time_signature', numerator=4, denominator=4)))
    last = 0
    for tick, message in sorted(events, key=lambda item: item[0]):
        meta.append(message.copy(time=tick - last))
        last = tick
    meta.append(MetaMessage("end_of_track", time=song_end_tick(mid) - last))
    return meta


def build_grid(notes: list[NoteEvent], ticks_per_beat: int, division: int, end_tick: int | None = None) -> tuple[list[int], int]:
    grid = max(1, ticks_per_beat * 4 // division)
    end_tick = end_tick if end_tick is not None else max(n.tick + n.duration for n in notes)
    # Derive each tick from its absolute subdivision; odd PPQ must not drift.
    count = math.ceil(end_tick * division / (ticks_per_beat * 4))
    ticks = sorted({index * ticks_per_beat * 4 // division for index in range(count)})
    return ticks, grid


def window_intensity(notes: list[NoteEvent], ticks: list[int], grid: int) -> list[float]:
    values: list[float] = []
    note_index = 0
    active: list[NoteEvent] = []

    for tick in ticks:
        while note_index < len(notes) and notes[note_index].tick < tick + grid:
            active.append(notes[note_index])
            note_index += 1

        active = [n for n in active if n.tick + n.duration > tick]
        touching = [n for n in active if n.tick < tick + grid and n.tick + n.duration > tick]

        if not touching:
            values.append(0.0)
            continue

        avg_vel = sum(n.velocity for n in touching) / (127 * len(touching))
        density = min(1.0, len(touching) / 4)
        high_note = max(n.note for n in touching) / 127
        values.append(min(1.0, avg_vel * 0.72 + density * 0.18 + high_note * 0.10))

    return values


def accent_curve(values: list[float], phrase_steps: int) -> list[float]:
    accents: list[float] = []
    for i, value in enumerate(values):
        local = i % phrase_steps
        phrase_position = local / max(1, phrase_steps - 1)
        end_lift = max(0.0, (phrase_position - 0.72) / 0.28)
        transient = max(0.0, value - (values[i - 1] if i else value))
        accents.append(min(1.0, end_lift * 0.65 + transient * 0.85))
    return accents


def smooth(values: list[float], amount: float) -> list[float]:
    if not values:
        return []

    forward: list[float] = []
    current = values[0]
    for value in values:
        current += (value - current) * amount
        forward.append(current)

    backward: list[float] = [0.0] * len(values)
    current = forward[-1]
    for i in range(len(values) - 1, -1, -1):
        current += (forward[i] - current) * amount
        backward[i] = current

    return backward


def lane_velocities(
    values: list[float],
    lane: str,
    role: str,
    phrase_steps: int,
    invert: bool = False,
    min_velocity: int = 1,
    max_velocity: int = 127,
) -> list[int]:
    settings = lane_profile(lane, role)
    accents = accent_curve(values, phrase_steps)
    energy = smooth(values, settings["smooth"])

    result: list[int] = []
    for energy_value, accent in zip(energy, accents):
        accent_mix = settings["accent"]
        value = energy_value * (1 - accent_mix) + accent * accent_mix
        value = max(0.0, min(1.0, value)) ** settings["gamma"]

        if invert:
            value = 1.0 - value

        velocity = round(settings["low"] + value * (settings["high"] - settings["low"]))
        velocity = max(min_velocity, min(max_velocity, velocity))
        result.append(velocity)

    return result


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def special_velocities(
    base: list[int], values: list[float], ticks: list[int], notes: list[NoteEvent],
    ticks_per_beat: int, grid: int, lane: str, role: str, track_name: str,
    plan: MovementPlan, movement: str, strength: float, seed: int,
    minimum: int, maximum: int, invert: bool, drop_fx: str = 'dry',
) -> list[int]:
    if movement == 'classic' or strength == 0:
        return base
    profile = lane_profile(lane, role)
    low = clamp(profile['low'], minimum, maximum)
    high = clamp(profile['high'], minimum, maximum)
    span = max(1, high - low)
    # Stable per-track variation: Python's randomized hash must not change takes.
    rng = random.Random(seed + zlib.crc32(track_name.encode('utf-8')))
    phase_offset = rng.random() * math.tau
    pulse_rate = rng.choice((1, 1, 2))
    variation = rng.uniform(0.82, 1.0)
    next_start = sorted({n.tick for n in notes})
    throws = []
    for event in notes:
        end = event.tick + event.duration
        following = bisect_right(next_start, event.tick)
        gap_after = following == len(next_start) or next_start[following] >= end + ticks_per_beat // 2
        phrase_index = max(0, bisect_right(plan.phrase_bounds, event.tick) - 1)
        phrase_end = plan.phrase_bounds[min(phrase_index + 1, len(plan.phrase_bounds) - 1)]
        at_end = phrase_end - 4 * ticks_per_beat <= event.tick < phrase_end
        if gap_after or at_end:
            throws.append((max(event.tick, end - ticks_per_beat), end))
    throws.sort()
    throw_index = 0
    active_throws = []
    drops = plan.drops
    result = []
    for tick, original, energy in zip(ticks, base, values):
        value = float(original)
        active = energy > 0
        phrase_index = max(0, bisect_right(plan.phrase_bounds, tick) - 1)
        start = plan.phrase_bounds[phrase_index]
        stop = plan.phrase_bounds[min(phrase_index + 1, len(plan.phrase_bounds) - 1)]
        ramp_length = max(1, min(8 * ticks_per_beat, (stop - start) // 2))
        lift = clamp((tick - (stop - ramp_length)) / ramp_length)
        lift = lift * lift * (3 - 2 * lift)
        beat_phase = ((tick / ticks_per_beat) * pulse_rate) % 1.0
        pump = math.sqrt(beat_phase)
        wobble = (1 + math.sin(math.tau * tick / (ticks_per_beat * 2) + phase_offset)) / 2
        drift = (1 + math.sin(math.tau * tick / (ticks_per_beat * 16) + phase_offset)) / 2
        while throw_index < len(throws) and throws[throw_index][0] <= tick:
            active_throws.append(throws[throw_index])
            throw_index += 1
        active_throws = [(begin, end) for begin, end in active_throws if tick < end]
        throw = max((math.sin(math.pi * clamp((tick + grid / 2 - begin) / max(grid, end - begin))) ** 2
                     for begin, end in active_throws), default=0.0)
        target = value
        if movement in {'transitions', 'full'} and active:
            if lane == 'cutoff':
                target = value + (high - value) * lift
            elif lane in {'reverb', 'delay'}:
                target = low + span * max(throw * variation, lift * (0.65 if lane == 'reverb' else 0.35))
            elif lane in {'distortion', 'decay'}:
                target = value + span * lift * 0.4
            elif lane == 'width':
                target = value + (high - value) * lift * 0.6
        if movement in {'groove', 'full'} and active:
            if lane == 'volume':
                # Drums retain their transient; other instruments breathe around it.
                target = high - (1 - pump) * span * (0.18 if role == 'drums' else 0.8)
            elif lane in {'cutoff', 'formant'}:
                rhythmic = low + span * (0.25 + 0.75 * wobble)
                target = rhythmic if movement == 'groove' else target * 0.6 + rhythmic * 0.4
            elif lane == 'width':
                target = low + span * (0.25 + drift * 0.75)
            elif lane == 'decay':
                target = low + span * (0.4 + 0.6 * pump)
            elif lane == 'pitch':
                target = 64 + 8 * math.sin(math.tau * tick / (ticks_per_beat * 4) + phase_offset)
            elif lane in {'reverb', 'delay'}:
                target = low + span * throw * variation
        if movement == 'drop':
            coming = next((begin for begin, _ in drops if tick < begin <= tick + 8 * ticks_per_beat), None)
            inside = next(((begin, end) for begin, end in drops if begin <= tick < end), None)
            if coming is not None and active:
                tension = clamp(1 - (coming - tick) / (8 * ticks_per_beat))
                tension = tension * tension * (3 - 2 * tension)
                if lane == 'volume':
                    target = high - span * (0.35 + 0.65 * tension)
                elif lane == 'cutoff':
                    target = high - span * (0.30 + 0.70 * tension)
                elif lane in {'reverb', 'delay'}:
                    target = low + span * tension * variation
                    # Stop feeding/wetting FX one beat before the hit.
                    if tick >= coming - ticks_per_beat:
                        target = minimum
                elif lane == 'width':
                    target = high - span * tension * 0.65
                elif lane == 'distortion':
                    target = low + span * (0.5 + 0.5 * tension)
            elif inside is not None and active:
                if lane in {'volume', 'cutoff', 'decay', 'distortion'}:
                    target = high
                elif lane == 'width':
                    target = high if role != 'bass' else low
            # FX limits are independent of contrast amount and invert.
            # 1 is a real MIDI note; velocity 0 is note-off.
            if lane in {'delay', 'reverb'} and drop_fx != 'keep':
                ceiling = drop_fx_ceiling(tick, lane, role, plan, ticks_per_beat, minimum, maximum, drop_fx)
                if ceiling is not None:
                    result.append(min(original, ceiling))
                    continue
        if not active:
            if lane in {'reverb', 'delay'}:
                result.append(minimum)
                continue
            result.append(original)
            continue
        delta = (target - value) * strength
        if invert:
            delta = -delta
        result.append(round(clamp(value + delta, minimum, maximum)))
    return result


def drop_fx_ceiling(tick: int, lane: str, role: str, plan: MovementPlan,
                    ticks_per_beat: int, minimum: int, maximum: int, policy: str,
                    fx_style: str = 'original') -> int | None:
    if lane not in {'reverb', 'delay'} or policy == 'keep':
        return None
    if fx_style == 'full_range' and policy == 'reduced':
        window = next(((start, end) for start, end in plan.drops
                       if start - ticks_per_beat / 4 <= tick < end), None)
        if window is None:
            return None
        recovery = clamp((tick - window[0] - ticks_per_beat / 2) / (ticks_per_beat * 1.5))
        recovery = recovery * recovery * (3 - 2 * recovery)
        return round(clamp(minimum + (maximum - minimum) * recovery, minimum, maximum))
    if fx_style == 'fuller' and policy == 'reduced' and role != 'bass':
        # A brief clear hit, then a smooth recovery of the existing wet tail.
        # These are ceilings, not forced wet levels, so genuine silence stays
        # silent and the incoming phrase can still be naturally restrained.
        window = next(((start, end) for start, end in plan.drops
                       if start - ticks_per_beat / 4 <= tick < end), None)
        if window is None:
            return None
        recovery = clamp((tick - window[0] - ticks_per_beat / 2) / (ticks_per_beat * 1.5))
        recovery = recovery * recovery * (3 - 2 * recovery)
        amount = (.25 if role == 'drums' else .70) * recovery
        profile = lane_profile(lane, role)
        return round(clamp(minimum + (min(maximum, profile['high']) - minimum) * amount, minimum, maximum))
    window = next(((start, end) for start, end in plan.drops
                   if start - ticks_per_beat <= tick < end), None)
    if window is None:
        return None
    if policy == 'dry' or role == 'bass':
        return minimum
    # Balanced keeps melodic space while keeping the first hit and low end clear.
    role_amount = {'drums': .16, 'chords': .40, 'arp': .35, 'lead': .45, 'default': .35}[role]
    if tick < window[0] + ticks_per_beat:
        role_amount *= .25
    profile = lane_profile(lane, role)
    return round(clamp(minimum + (min(maximum, profile['high']) - minimum) * role_amount, minimum, maximum))


def merged_fx_spans(notes: list[NoteEvent], ticks_per_beat: int, role: str) -> list[tuple[int, int, int]]:
    """Sounding spans, including short articulation gaps, with a peak velocity.

    Detect endings of the complete part rather than each voice in a chord. A
    short note ending underneath a held pad therefore cannot create a throw.
    """
    bridge = ticks_per_beat * {'bass': .125, 'drums': .25}.get(role, .5)
    spans: list[tuple[int, int, int]] = []
    for event in sorted(notes, key=lambda n: (n.tick, n.tick + n.duration)):
        if event.duration <= 0 or event.velocity <= 0:
            continue
        start, end = event.tick, event.tick + event.duration
        if spans and start <= spans[-1][1] + bridge:
            begin, previous_end, velocity = spans[-1]
            spans[-1] = (begin, max(previous_end, end), max(velocity, event.velocity))
        else:
            spans.append((start, end, event.velocity))
    return spans


def intensity_factors(intensity: float = 1.0) -> tuple[float, float, float]:
    """Independent movement depth, hold and release; one is the legacy sound.

    Depth expands around a known neutral/saved value, never an audio gain or
    feedback limit. Keep the old arithmetic path unchanged at exactly one.
    """
    intensity = float(intensity)
    if not math.isfinite(intensity) or not 0 <= intensity <= 2:
        raise ValueError('Movement intensity must be between zero and two.')
    extra = max(0.0, intensity - 1.0)
    return (intensity if intensity <= 1 else 1 + 2 * extra,
            1 + 1.5 * extra, 1 + extra)


def fuller_fx_times(lane: str, role: str, ticks_per_beat: int,
                    intensity: float = 1.0) -> tuple[int, int]:
    """Hold and release times in musical ticks; bass/drum ambience is shorter."""
    hold, release = (1, 4) if lane == 'delay' else (2, 8)
    scale = {'bass': .25, 'drums': .5}.get(role, 1)
    if intensity != 1:
        _, hold_scale, release_scale = intensity_factors(intensity)
        hold *= hold_scale
        release *= release_scale
    return max(1, round(hold * scale * ticks_per_beat)), max(1, round(release * scale * ticks_per_beat))


def fuller_fx_velocities(values: list[float], ticks: list[int], spans: list[tuple[int, int, int]],
                         ticks_per_beat: int, lane: str, role: str, track_name: str,
                         plan: MovementPlan, movement: str, strength: float, seed: int,
                         minimum: int, maximum: int, invert: bool,
                         full_range: bool = False,
                         intensity: float = 1.0) -> tuple[list[int], list[float]]:
    """Sustained wet presence plus phrase throws that continue after note-off.

    This controls a wet output/slot mix, so a MIDI rest does not mean that the
    effect itself is silent. Tail activity is also returned for energy painting.
    """
    profile = lane_profile(lane, role)
    high = clamp(profile['high'], minimum, maximum)
    span = high - minimum
    output_span = maximum - minimum if full_range else span
    floor_amount = {'bass': .10, 'drums': .14, 'chords': .38, 'arp': .30,
                    'lead': .34, 'default': .32}.get(role, .32)
    depth, _, _ = intensity_factors(intensity)
    hold, release = fuller_fx_times(lane, role, ticks_per_beat, intensity)
    catch_scale = 1 + .75 * max(0.0, intensity - 1)
    rng = random.Random(seed + zlib.crc32((track_name + ':' + lane).encode('utf-8')))
    prepared = []
    for start, end, velocity in spans:
        if full_range:
            # Explicit full-range mode guarantees the requested endpoint on
            # phrase throws; role/strength still shape the underlying body.
            peak_amount = 1.0
        else:
            sensitivity = .70 + .30 * clamp(velocity / 127)
            peak_amount = (.76 + .18 * strength) * sensitivity * rng.uniform(.94, 1)
            if role == 'bass':
                peak_amount *= .45
            elif role == 'drums':
                peak_amount *= .65
        prepared.append((start, end, peak_amount))
    result, activity = [], []
    next_span = 0
    active = []
    for tick, energy in zip(ticks, values):
        while next_span < len(prepared) and prepared[next_span][0] <= tick:
            active.append(prepared[next_span])
            next_span += 1
        active = [item for item in active if tick < item[1] + hold + release]
        amount, envelope = 0.0, 0.0
        for start, end, peak_amount in active:
            if tick < end:
                # Dynamics affect wetness gently; invert changes this response
                # but never turns a true silent passage into an FX opening.
                response = clamp(energy) ** .65
                if invert:
                    response = 1 - response
                body = floor_amount + (.24 + .12 * strength) * response
                if role in {'bass', 'drums'}:
                    body = floor_amount + (.10 + .08 * strength) * response
                if full_range:
                    # Keep the same restrained body, but leave a real plateau
                    # at the maximum instead of approaching it asymptotically.
                    body *= span / output_span if output_span else 0
                    plateau = max(start, end - ticks_per_beat / 2 * catch_scale)
                    catch_start = max(start, plateau - ticks_per_beat * catch_scale)
                    catch = 1.0 if plateau == catch_start else clamp((tick - catch_start) / (plateau - catch_start))
                else:
                    catch_start = max(start, end - ticks_per_beat * catch_scale)
                    catch = clamp((tick - catch_start) / max(1, end - catch_start))
                if intensity > 1:
                    # More audible space throughout the played phrase, while
                    # reserving full wet for throws rather than a flat wash.
                    body_limit = ({'bass': .20, 'drums': .32}.get(role, .78)
                                  if full_range else .88)
                    body = max(body, min(body_limit, body * depth))
                catch = catch * catch * (3 - 2 * catch)
                wet = max(body, peak_amount * catch)
                weight = 1.0
                if movement == 'drop' and strength > 0:
                    coming = next((begin for begin, _ in plan.drops
                                   if tick < begin <= tick + 8 * ticks_per_beat), None)
                    if coming is not None:
                        tension = clamp(1 - (coming - tick) / (8 * ticks_per_beat))
                        tension = tension * tension * (3 - 2 * tension)
                        wet = max(wet, peak_amount * tension * strength)
            else:
                progress = clamp((tick - end - hold) / release)
                weight = 1 - progress * progress * (3 - 2 * progress)
                wet = peak_amount * weight
            amount = max(amount, wet)
            envelope = max(envelope, weight)
        result.append(round(clamp(minimum + output_span * clamp(amount), minimum, maximum)))
        activity.append(envelope)
    return result, activity


def intensity_velocities(base: list[int], lane: str, role: str, intensity: float,
                         minimum: int, maximum: int) -> list[int]:
    """Scale contrast without increasing gain or widening bass beyond its range.

    These neutral values describe newly owned external effect controls. Saved
    synth values are handled by sylenth_movement instead. A caller using an
    existing unknown control should preserve it by omitting that new lane.
    """
    depth, _, _ = intensity_factors(intensity)
    if intensity == 1:
        return base
    profile = lane_profile(lane, role)
    neutral = {'cutoff': maximum, 'delay': minimum, 'reverb': minimum,
               'volume': clamp(102, minimum, maximum), 'pitch': clamp(64, minimum, maximum),
               'distortion': minimum}.get(lane, clamp((profile['low'] + profile['high']) / 2,
                                                     minimum, maximum))
    if intensity <= 1:
        return [round(clamp(neutral + (value - neutral) * depth, minimum, maximum))
                for value in base]
    if lane == 'pitch':
        return base  # An intensity control is not permission for larger pitch shifts.
    result = []
    for value in base:
        if lane in {'cutoff', 'volume'}:
            changed = value - max(0, neutral - value) * (depth - 1)
        else:
            changed = neutral + (value - neutral) * depth
        # The original role limits keep bass width and envelope lengths safe.
        if lane in {'width', 'decay', 'formant', 'distortion'}:
            changed = clamp(changed, min(value, profile['low']), max(value, profile['high']))
        result.append(round(clamp(changed, minimum, maximum)))
    return result


def painted_velocities(base: list[int], values: list[float], ticks: list[int], curve: dict | None,
                       lane: str, role: str, minimum: int, maximum: int,
                       fx_high: int | None = None) -> list[int]:
    if curve is None:
        return base
    profile = lane_profile(lane, role)
    low, high = clamp(profile['low'], minimum, maximum), clamp(profile['high'], minimum, maximum)
    if fx_high is not None and lane in {'delay', 'reverb'}:
        high = clamp(fx_high, minimum, maximum)
    result = []
    for original, activity, energy in zip(base, values, sample_energy_curve(curve, ticks)):
        direction = energy * 2 - 1
        if direction == 0:
            result.append(original)
            continue
        if activity <= 0:
            result.append(minimum if lane in {'reverb', 'delay'} else original)
            continue
        if lane == 'cutoff':
            floor = {'drums': 64, 'bass': 48}.get(role, 24)
            target = high if direction > 0 else clamp(min(low, floor), minimum, maximum)
        elif lane == 'volume':
            target = high if direction > 0 else low
        elif lane in {'delay', 'reverb'}:
            # Energy is intensity, not wetness. Preserve the existing FX rhythm,
            # allowing only a modest increase; quiet passages can be drier.
            target = min(high, original + (high - low) * .18) if direction > 0 else minimum
        elif lane in {'pitch', 'formant'}:
            centre = clamp(64, minimum, maximum)
            target = centre + (original - centre) * 1.35 if direction > 0 else centre
        elif lane == 'width':
            target = high if direction > 0 else min(low, max(minimum, 32))
        else:
            target = high if direction > 0 else low
        result.append(round(clamp(original + (target - original) * abs(direction), minimum, maximum)))
    return result


def make_ctrl_midi(
    source: MidiFile,
    notes: list[NoteEvent],
    track_name: str,
    lane: str,
    out_path: Path,
    division: int,
    note: int,
    gate: float,
    channel: int,
    invert: bool,
    min_velocity: int,
    max_velocity: int,
    movement: str = 'classic',
    strength: float = 0.65,
    phrase_bars: int = 16,
    seed: int = 1,
    plan: MovementPlan | None = None,
    energy_curve: dict | None = None,
    drop_fx: str = 'dry',
    fx_style: str = 'original',
    intensity: float = 1.0,
) -> None:
    if fx_style not in {'original', 'fuller', 'full_range'}:
        raise ValueError('FX style must be original, fuller or full_range.')
    intensity_factors(intensity)
    intensity = float(intensity)
    role = detect_role(track_name)
    plan = plan or movement_plan(source, phrase_bars, '', 8)
    ticks, grid = build_grid(notes, source.ticks_per_beat, division, plan.end_tick)
    boundaries = {plan.end_tick - 1}
    fuller_fx = fx_style in {'fuller', 'full_range'} and lane in {'delay', 'reverb'}
    full_range = fx_style == 'full_range' and fuller_fx
    fx_maximum = 127 if full_range else max_velocity
    extended_original = intensity > 1 and lane in {'delay', 'reverb'} and not fuller_fx
    fx_spans = merged_fx_spans(notes, source.ticks_per_beat, role) if fuller_fx or extended_original else []
    if fuller_fx or extended_original:
        hold, release = fuller_fx_times(lane, role, source.ticks_per_beat, intensity)
        for start, end, _ in fx_spans:
            boundaries.update((start, max(start, end - source.ticks_per_beat), end - 1,
                               end, end + hold, end + hold + release))
            if full_range:
                catch_scale = 1 + .75 * max(0.0, intensity - 1)
                plateau = max(start, end - source.ticks_per_beat / 2 * catch_scale)
                ramp_start = max(start, plateau - source.ticks_per_beat * catch_scale)
                # ceil selects the first representable tick on the plateau.
                boundaries.update((math.ceil(plateau), math.ceil(ramp_start)))
                if end >= plan.end_tick and start < plan.end_tick - 1:
                    boundaries.add(max(start, plan.end_tick - 2))
        if movement == 'drop' and strength > 0:
            for start, _ in plan.drops:
                boundaries.update((round(start - source.ticks_per_beat / 4),
                                   round(start + source.ticks_per_beat / 2),
                                   start + 2 * source.ticks_per_beat))
    if energy_curve is not None and any(value != .5 for _, value in energy_curve['points']):
        boundaries.update(tick for tick, _ in energy_curve['points'])
    if movement != 'classic' and strength > 0:
        boundaries.update(plan.phrase_bounds)
        for start, end in plan.drops:
            boundaries.update((start, end, start - source.ticks_per_beat,
                               start - 8 * source.ticks_per_beat, start + source.ticks_per_beat))
    ticks = sorted({*ticks, *(tick for tick in boundaries if 0 <= tick < plan.end_tick)})
    values = window_intensity(notes, ticks, grid)
    phrase_steps = max(1, division * 4)  # Preserve the original base shape.
    if fuller_fx:
        velocities, tail_activity = fuller_fx_velocities(values, ticks, fx_spans,
            source.ticks_per_beat, lane, role, track_name, plan, movement, strength,
            seed, min_velocity, fx_maximum, invert, full_range=full_range,
            intensity=intensity)
        painted = painted_velocities(velocities, tail_activity, ticks, energy_curve,
                                     lane, role, min_velocity, fx_maximum,
                                     fx_high=fx_maximum if full_range else None)
        # Fade the painting influence with the tail as well. Otherwise high
        # paint could add a fixed wet floor right up to an abrupt release end.
        velocities = [round(clamp(base + (value - base) * activity, min_velocity, fx_maximum))
                      for base, value, activity in zip(velocities, painted, tail_activity)]
    else:
        velocities = lane_velocities(values, lane, role, phrase_steps, invert, min_velocity, max_velocity)
        # Original mode remains byte-for-byte compatible with prior exports.
        if lane in {'reverb', 'delay'}:
            velocities = [value if energy > 0 else min_velocity for value, energy in zip(velocities, values)]
        velocities = special_velocities(velocities, values, ticks, notes, source.ticks_per_beat,
                                       grid, lane, role, track_name, plan, movement, strength,
                                       seed, min_velocity, max_velocity, invert, drop_fx)
        velocities = painted_velocities(velocities, values, ticks, energy_curve, lane, role,
                                        min_velocity, max_velocity)
        if extended_original:
            # Grow the short original gestures continuously into longer throws
            # as intensity rises. At one this branch does not exist at all.
            sustained, tail_activity = fuller_fx_velocities(values, ticks, fx_spans,
                source.ticks_per_beat, lane, role, track_name, plan, movement, strength,
                seed, min_velocity, fx_maximum, invert, intensity=intensity)
            painted = painted_velocities(sustained, tail_activity, ticks, energy_curve,
                                          lane, role, min_velocity, max_velocity)
            sustained = [round(base + (value - base) * activity)
                         for base, value, activity in zip(sustained, painted, tail_activity)]
            blend = intensity - 1
            velocities = [round(base + max(0, target - base) * blend)
                          for base, target in zip(velocities, sustained)]
    if intensity != 1 and (lane not in {'delay', 'reverb'} or intensity < 1):
        velocities = intensity_velocities(velocities, lane, role, intensity,
                                           min_velocity, fx_maximum)
    if movement == 'drop' and strength > 0:
        for index, tick in enumerate(ticks):
            ceiling = drop_fx_ceiling(tick, lane, role, plan, source.ticks_per_beat,
                                      min_velocity, fx_maximum, drop_fx, fx_style)
            if ceiling is not None:
                velocities[index] = min(velocities[index], ceiling)
    if lane in {'reverb', 'delay'}:
        velocities[-1] = min_velocity

    out = MidiFile(type=1, ticks_per_beat=source.ticks_per_beat)
    out.tracks.append(source_meta_track(source))

    track = MidiTrack()
    track.append(MetaMessage("track_name", name=f"{track_name} CTRL {lane} {role}", time=0))

    last_tick = 0
    for index, (tick, velocity) in enumerate(zip(ticks, velocities)):
        next_tick = ticks[index + 1] if index + 1 < len(ticks) else plan.end_tick
        note_length = min(next_tick - tick, max(1, round((next_tick - tick) * gate)))
        track.append(Message("note_on", note=note, velocity=velocity, channel=channel, time=tick - last_tick))
        track.append(Message("note_off", note=note, velocity=0, channel=channel, time=note_length))
        last_tick = tick + note_length

    track.append(MetaMessage("end_of_track", time=plan.end_tick - last_tick))
    out.tracks.append(track)
    out.save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Fruity Keyboard Controller MIDI patterns with smoothed velocity automation."
    )
    parser.add_argument("midi", type=Path, help="Source song MIDI file.")
    parser.add_argument("--out", type=Path, default=Path("outputs/ctrl_midi"), help="Output folder.")
    parser.add_argument(
        "--preset",
        choices=("edm-impact", "custom"),
        default="edm-impact",
        help="edm-impact makes only the simplest high-impact lanes per instrument. custom uses --lanes.",
    )
    parser.add_argument(
        "--lanes",
        default="cutoff,reverb,delay,distortion,decay,width",
        help=f"Comma list for --preset custom. Available: {', '.join(sorted(LANES))}",
    )
    parser.add_argument("--division", type=int, default=16, help="Control-note grid per bar. 16 = 1/16 notes.")
    parser.add_argument("--note", type=int, default=60, help="Fixed controller note. 60 = C5/C4 depending display.")
    parser.add_argument("--gate", type=float, default=0.85, help="Control note length as a fraction of the grid.")
    parser.add_argument("--channel", type=int, default=0, help="MIDI channel for CTRL notes, 0-15.")
    parser.add_argument("--invert", action="store_true", help="Invert generated velocity curves.")
    parser.add_argument("--min-velocity", type=int, default=1, help="Lowest generated velocity.")
    parser.add_argument("--max-velocity", type=int, default=127, help="Highest generated velocity.")
    parser.add_argument('--movement', choices=MOVEMENTS, default='classic', help='Choose classic curves, transitions, groove, full movement or cleaner, harder drops.')
    parser.add_argument('--strength', type=float, default=0.65, help='Special movement amount from 0 to 1. Zero preserves classic curves.')
    parser.add_argument('--phrase-bars', type=int, choices=(4, 8, 16), default=16, help='Phrase length and fallback drop interval when there are no Drop markers.')
    parser.add_argument('--seed', type=int, default=1, help='Repeatable variation number.')
    parser.add_argument('--extra-lanes', action='store_true', help='Add one useful extra role lane in EDM Impact, where appropriate.')
    parser.add_argument('--drop-bars', default='', help='Optional 1-based drop bar numbers, e.g. 17,49. Overrides MIDI Drop markers.')
    parser.add_argument('--drop-length', type=int, choices=(4, 8, 16), default=8, help='Number of bars for the selected drop treatment.')
    parser.add_argument('--drop-fx', choices=('dry', 'reduced', 'keep'), default='dry', help='Drop FX: dry, balanced/reduced, or keep the movement.')
    parser.add_argument('--energy-curve', type=Path, help='Saved painted-energy JSON for this exact source MIDI.')
    args = parser.parse_args()
    if args.division not in {8, 16, 32}:
        parser.error('Density must be 8, 16 or 32.')
    if not 0 <= args.note <= 127 or not 0 <= args.channel <= 15:
        parser.error('Controller note must be 0–127 and channel 0–15.')
    if not 1 <= args.min_velocity <= args.max_velocity <= 127:
        parser.error('Velocity limits must satisfy 1 <= minimum <= maximum <= 127.')
    if not math.isfinite(args.strength) or not 0 <= args.strength <= 1:
        parser.error('Strength must be between 0 and 1.')
    if not math.isfinite(args.gate) or not 0 < args.gate <= 1:
        parser.error('Gate must be greater than 0 and at most 1.')
    if args.extra_lanes and args.preset != 'edm-impact':
        parser.error('Extra role lanes are available in EDM Impact mode. In custom mode select the lanes yourself.')
    return args


def main() -> None:
    args = parse_args()
    lanes = list(dict.fromkeys(lane.strip().lower() for lane in args.lanes.split(",") if lane.strip()))
    unknown = [lane for lane in lanes if lane not in LANES]
    if unknown:
        raise SystemExit(f"Unknown lane(s): {', '.join(unknown)}")
    if args.preset == 'custom' and not lanes:
        raise SystemExit('Select at least one automation lane.')

    source_bytes = args.midi.read_bytes()
    mid = MidiFile(file=BytesIO(source_bytes))
    if mid.type == 2 or mid.ticks_per_beat <= 0:
        raise SystemExit('Use a synchronous MIDI type 0 or 1 file with musical beat timing.')
    if song_end_tick(mid) <= 0:
        raise SystemExit('This MIDI has no duration. Export a song with notes and a positive length first.')
    if (args.movement != 'classic' and args.strength > 0) or args.energy_curve is not None:
        if any(msg.type == 'time_signature' and (msg.numerator, msg.denominator) != (4, 4)
               for _, msg in timed_meta(mid)):
            raise SystemExit('Special movement currently uses 4/4. Choose Classic for this MIDI.')
    tracks = read_track_notes(mid)
    if not tracks:
        raise SystemExit("No note tracks found in the MIDI file.")
    try:
        plan = movement_plan(mid, args.phrase_bars, args.drop_bars, args.drop_length)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    energy = None
    if args.energy_curve is not None:
        identity = {'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
                    'ticks_per_beat': mid.ticks_per_beat, 'end_tick': plan.end_tick}
        try:
            energy = load_curve(args.energy_curve, identity)
        except CurveError as exc:
            raise SystemExit(str(exc)) from None

    args.out.mkdir(parents=True, exist_ok=True)
    made = 0
    for track_name, track_index, notes in tracks:
        clean_name = slug(track_name)
        role = detect_role(track_name)
        track_lanes = list(IMPACT_LANES_BY_ROLE[role]) if args.preset == "edm-impact" else lanes
        if args.preset == 'edm-impact':
            if args.movement == 'drop' and args.strength > 0:
                track_lanes = list(dict.fromkeys([*track_lanes, 'delay', 'reverb']))
            if args.extra_lanes and role in EXTRA_LANE_BY_ROLE:
                track_lanes = list(dict.fromkeys([*track_lanes, EXTRA_LANE_BY_ROLE[role]]))
        for lane in track_lanes:
            out_path = args.out / f"{track_index + 1:02d}_{clean_name}_CTRL_{lane}.mid"
            if out_path.resolve() == args.midi.resolve():
                raise SystemExit('Choose a different output folder so the source MIDI is preserved.')
            make_ctrl_midi(
                source=mid,
                notes=notes,
                track_name=track_name,
                lane=lane,
                out_path=out_path,
                division=args.division,
                note=args.note,
                gate=args.gate,
                channel=args.channel,
                invert=args.invert,
                min_velocity=args.min_velocity,
                max_velocity=args.max_velocity,
                movement=args.movement,
                strength=args.strength,
                phrase_bars=args.phrase_bars,
                seed=args.seed,
                plan=plan,
                energy_curve=energy,
                drop_fx=args.drop_fx,
            )
            made += 1

    print(f"Created {made} CTRL MIDI file(s) in {args.out}")
    print(f'Movement: {args.movement}; amount: {args.strength:.0%}; variation: {args.seed}')
    if args.movement == 'drop':
        print(f'Drop positions: {plan.drop_source}. Length: {args.drop_length} bars. FX: {args.drop_fx}.')
        print('Drop bars: ' + ', '.join(f'{start / (mid.ticks_per_beat * 4) + 1:g}' for start, _ in plan.drops))
        if args.drop_fx == 'dry':
            print('Delay/reverb reach minimum before the hit. Map minimum velocity to 0% wet or FX return for a dry drop.')
        elif args.drop_fx == 'reduced':
            print('Balanced FX: dry low end, restrained drums, controlled space for melodic parts.')
        print('Lowering a send alone does not instantly mute an existing effect tail.')
    elif args.preset == "edm-impact":
        print('Preset: EDM Impact. Volume plus 2 role lanes' + (' and one extra where appropriate.' if args.extra_lanes else '.'))
    report = {
        'source_midi': str(args.midi.resolve()), 'movement': args.movement, 'strength': args.strength,
        'drop_fx': args.drop_fx, 'energy_curve': energy,
        'seed': args.seed, 'phrase_bars': args.phrase_bars, 'drop_length': args.drop_length,
        'drop_source': plan.drop_source, 'drop_ticks': [start for start, _ in plan.drops],
        'end_tick': plan.end_tick, 'ticks_per_beat': mid.ticks_per_beat, 'files_created': made,
        'note': args.note, 'channel': args.channel,
        'min_velocity': args.min_velocity, 'max_velocity': args.max_velocity,
    }
    (args.out / 'Movement settings.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if energy is not None:
        save_curve(args.out / 'Energy curve.json', energy)
        print('Applied your painted energy. 50% keeps the current feel; high energy does not mean full wet FX.')
    (args.out / 'How to use these moves.txt').write_text(
        'Place every CTRL MIDI at the same song start as the original MIDI.\n'
        'Each instrument/target has ONE combined controller file. Replace its old CTRL pattern; do not layer two controllers on the same knob.\n'
        'Link to Fruity Keyboard Controller Velocity using your usual setup.\n'
        'Balanced drop FX keeps bass effects at minimum, restrains drums and allows controlled melodic space. Roles come from MIDI track names.\n'
        'For dry drops, map the lowest velocity to 0% on delay/reverb wet level or FX return volume.\n'
        'A send level controls new input and may leave old tails audible; wet/return level controls the audible effect.\n'
        'Keep attack/release and link smoothing short for sharp drop changes.\n'
        'Use your normal small volume range; the build pulls back then restores intensity.\n'
        'Keep bass width narrow. Drop bars are 1-based and editable in the generator.\n'
        'Painted energy: 50% preserves the current movement; lower is restrained, higher is more forward. It does not mean full wet FX.\n'
        'When painting is enabled, Energy curve.json saves this run\'s drawing for the exact source MIDI.\n'
        'See Movement settings.json for the exact drop positions and variation used.\n', encoding='utf-8')


if __name__ == "__main__":
    main()
