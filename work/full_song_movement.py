"""Shared, deterministic arrangement gestures for the FLP connector.

The planner uses sounding voices as well as note attacks, so a busy hi-hat
cannot turn a sparse melodic break into a drop. Ticks are global throughout;
the exporter may split their storage without restarting these curves.
"""
from __future__ import annotations

from bisect import bisect_right
import hashlib
import math
import re


FLOOR = 1 / 128


def _clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def _ease(x):
    x = _clamp(x)
    return x * x * (3 - 2 * x)


def _value(note, key):
    return note[key] if isinstance(note, dict) else getattr(note, key)


def role_for(name, plugin=''):
    text = str(name).lower()
    if 'drum monkey' in (text + str(plugin).lower()) or re.search(r'\b(drum|drums|kick|snare|clap|hat|perc|tom)\b', text):
        return 'drums'
    if re.search(r'bass|sub|808', text):
        return 'bass'
    if re.search(r'pad|chord|texture|keys|piano', text):
        return 'chords'
    if re.search(r'arp|pluck|seq', text):
        return 'arp'
    if re.search(r'lead|melody|hook', text):
        return 'lead'
    return 'default'


def _notes(notes, end):
    result = []
    for note in notes:
        tick, duration = int(_value(note, 'tick')), int(_value(note, 'duration'))
        velocity = int(_value(note, 'velocity'))
        if duration > 0 and velocity > 0 and 0 <= tick < end:
            result.append((tick, min(end, tick + duration), min(128, velocity)))
    return sorted(result)


def _spans(rows, bridge=0):
    result = []
    for begin, end, _ in rows:
        if result and begin <= result[-1][1] + bridge:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((begin, end))
    return result


def _stable(seed, *parts):
    text = ':'.join(map(str, (seed,) + parts)).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:4], 'little') / 0xffffffff


def build_plan(model, options):
    """Return JSON-serializable bar evidence, regions and scheduled gestures.

    Required model fields: ppq, end_tick, voices[{channel_id,name,notes}].
    Explicit one-based drop bars override semantic markers; otherwise only
    persistent changes in actual voice activity nominate drops. Meter markers
    and regular phrase boundaries never nominate drops.
    """
    ppq, end = int(model['ppq']), int(model['end_tick'])
    if ppq <= 0 or end <= 0:
        raise ValueError('Full-song movement needs positive musical timing.')
    bar = ppq * 4
    count = math.ceil(end / bar)
    seed = int(options.get('seed', 1))
    phrase = max(8, min(32, int(options.get('phrase_bars', 16))))
    voices, roles, names, weights = {}, {}, {}, {}
    for voice in sorted(model['voices'], key=lambda v: int(v['channel_id'])):
        cid = str(voice['channel_id'])
        roles[cid] = role_for(voice.get('name', ''), voice.get('plugin_name', voice.get('plugin', '')))
        names[cid] = voice.get('name', f'Instrument {cid}')
        voices[cid] = _notes(voice.get('notes', []), end)
        weights[cid] = {'bass': 1.5, 'lead': 1.3, 'chords': 1.1, 'arp': .9, 'drums': .35, 'default': 1.0}[roles[cid]]
    features = {}
    for cid, rows in voices.items():
        occupancy = [0.0] * count
        attacks = [0] * count
        velocity = [0.0] * count
        for a, b in _spans(rows):
            for index in range(a // bar, min(count, (b - 1) // bar + 1)):
                occupancy[index] += max(0, min(b, (index + 1) * bar) - max(a, index * bar)) / bar
        for a, _, vel in rows:
            attacks[a // bar] += 1
            velocity[a // bar] += vel / 128
        peak_attacks = max(1, max(attacks, default=1))
        features[cid] = [(.65 * _clamp(occupancy[i]) + .20 * math.sqrt(attacks[i] / peak_attacks)
                          + .15 * (velocity[i] / attacks[i] if attacks[i] else occupancy[i] * .65))
                         for i in range(count)]
    total_weight = sum(weights.values()) or 1
    bars = []
    for i in range(count):
        bars.append(dict(start=i * bar, end=min(end, (i + 1) * bar),
                         energy=sum(features[c][i] * weights[c] for c in voices) / total_weight,
                         active_ids=[int(c) for c in voices if features[c][i] > .06]))
    energies = [b['energy'] for b in bars]
    # Energy uses a fixed song-wide scale; local normalization would erase a
    # bass/lead disappearance merely because its hats became busier.
    def average(a, b):
        rows = energies[max(0, a):min(count, b)]
        return sum(rows) / max(1, len(rows))
    candidates = []
    for i in range(4, count - 2):
        before, after = average(i - 4, i), average(i, i + 4)
        changed = sum(weights[c] for c in voices if roles[c] != 'drums'
                      and (sum(features[c][max(0, i - 4):i]) / 4 > .10)
                      != (sum(features[c][i:i + 4]) / min(4, count - i) > .10)) / total_weight
        if abs(after - before) >= .105 or changed >= .24:
            edge = sum(weights[c] for c in voices if roles[c] != 'drums'
                       and (features[c][i - 1] > .06) != (features[c][i] > .06)) / total_weight
            candidates.append((abs(after - before) + changed * .35 + edge * .5, i, before, after))
    selected = []
    for score, i, before, after in sorted(candidates, reverse=True):
        if all(abs(i - other[1]) >= 8 for other in selected):
            selected.append((score, i, before, after))
    selected.sort(key=lambda x: x[1])
    markers = []
    for marker in model.get('markers', []):
        name = re.sub(r'[_-]+', ' ', str(marker.get('name', marker.get('text', '')))).strip()
        tick = int(marker.get('tick', 0))
        if marker.get('kind') == 8 or re.fullmatch(r'\d+\s*/\s*\d+', name) or not name or not 0 <= tick < end:
            continue
        markers.append((tick, name))
    explicit = str(options.get('drop_bars', '')).strip()
    if explicit:
        try:
            drop_positions = sorted(set((int(s.strip()) - 1) * bar for s in explicit.split(',') if s.strip()))
        except ValueError as exc:
            raise ValueError('Drop bars must be one-based bar numbers separated by commas.') from exc
        if not drop_positions or any(not 0 <= p < end for p in drop_positions):
            raise ValueError('A chosen drop bar is outside this song.')
        drop_source = 'chosen bars'
    else:
        drop_positions = sorted({p for p, n in markers if re.search(r'\bdrop\b', n, re.I)
                                 and not re.search(r'\b(?:pre|before|build|end|after)\b', n, re.I)})
        drop_source = 'song markers' if drop_positions else 'arrangement changes'
        if not drop_positions:
            drop_positions = [i * bar for _, i, before, after in selected
                              if after - before >= .085 or after > before and any(
                                  roles[c] == 'bass' and sum(features[c][i - 4:i]) < .12
                                  and sum(features[c][i:i + 4]) > .5 for c in voices)]
    drops = [dict(start=p, end=min(end, p + max(1, int(options.get('drop_length', 8))) * bar,
                                  min((m for m, name in markers if m > p and re.search(r'break|outro|verse|build|intro|chorus', name, re.I)), default=end)), source=drop_source)
             for p in drop_positions]
    bounds = sorted({0, end, *(i * bar for _, i, _, _ in selected), *(p for p, _ in markers), *drop_positions})
    regions = []
    mean = sum(energies) / max(1, count)
    for a, b in zip(bounds, bounds[1:]):
        energy = average(a // bar, math.ceil(b / bar))
        kind = 'open section' if energy < mean * .88 else 'full section'
        if a == 0 and energy < mean * .90:
            kind = 'intro'
        if b == end and energy < mean * .85:
            kind = 'outro'
        regions.append(dict(start=a, end=b, kind=kind, energy=energy))
    builds = []
    for drop in drops:
        p = drop['start']
        previous = max((b for b in bounds if b < p), default=0)
        named_builds = [m for m, name in markers if m < p and p - m <= 16 * bar and re.search(r'build|riser', name, re.I)]
        length = p - max(named_builds) if named_builds else min(8 * bar, p - previous)
        if length >= 4 * bar:
            builds.append(dict(start=p - length, end=p, drop_tick=p))
    # Long-song phrase accents are separate from drops. Only one selected
    # voice owns a wet throw, rather than every bus peaking at each boundary.
    opportunities = sorted({*range(phrase * bar, end, phrase * bar), *bounds[1:]})
    gestures = []
    previous_owner = None
    eligible_roles = {'lead', 'arp', 'chords', 'default'}
    for index, boundary in enumerate(opportunities):
        if boundary < 4 * bar:
            continue
        eligible = []
        for cid, rows in voices.items():
            if roles[cid] not in eligible_roles:
                continue
            sounding = [(a, min(b, boundary)) for a, b, _ in rows if a < boundary and b > boundary - 2 * ppq]
            if sounding:
                catch = max(b for _, b in sounding)
                onset = min(a for a, b in sounding if b == catch)
                if catch - onset > 1:
                    eligible.append((cid, catch, onset))
        if not eligible:
            continue
        eligible.sort(key=lambda item: (_stable(seed, index, item[0]), int(item[0])))
        choice = next((row for row in eligible if row[0] != previous_owner), eligible[0])
        cid, catch, onset = choice
        previous_owner = cid
        lane = 'delay' if index % 2 == 0 else 'reverb'
        # A drop throw ends before the hit and resumes only through its tail;
        # the final policy clamp still clears that tail on the downbeat.
        peak = min(end - 2, max(onset + 1, catch - max(1, ppq // 2)))
        hold = ppq if lane == 'delay' else 2 * ppq
        release = 4 * ppq if lane == 'delay' else 8 * ppq
        if end <= 2:
            continue
        gestures.append(dict(channel_id=int(cid), name=names[cid], lane=lane,
                             start=max(onset, peak - ppq), peak=peak,
                             hold_end=min(end - 1, catch + hold), end=min(end - 1, catch + hold + release),
                             peak_value=1.0 if options.get('fx_style', 'full_range') == 'full_range' else (.64 if lane == 'delay' else .56)))
    return dict(version=1, ppq=ppq, end_tick=end, phrase_bars=phrase, seed=seed,
                bars=bars, regions=regions, builds=builds, drops=drops,
                drop_source=drop_source if drops else 'none detected', gestures=gestures,
                voice_roles=roles, voice_names=names,
                explanation='Movement follows sounding instruments and real arrangement changes. Phrase accents are not assumed drops.')


def sample_macro(plan, tick, channel_id=None):
    """Shared energy, multi-bar tension, first-hit impact and phrase contrast."""
    ppq, end = plan['ppq'], plan['end_tick']
    bar = ppq * 4
    bars = plan['bars']
    index = min(len(bars) - 1, max(0, int(tick) // bar))
    following = min(len(bars) - 1, index + 1)
    blend = _ease((tick % bar) / bar)
    energy = bars[index]['energy'] * (1 - blend) + bars[following]['energy'] * blend
    tension = max((_ease((tick - b['start']) / max(1, b['end'] - b['start']))
                   for b in plan['builds'] if b['start'] <= tick < b['end']), default=0.0)
    impact = max((1 - _ease((tick - d['start']) / (2 * bar))
                  for d in plan['drops'] if d['start'] <= tick < d['start'] + 2 * bar), default=0.0)
    length = plan['phrase_bars'] * bar
    phrase_index, remainder = divmod(max(0, int(tick)), length)
    a = _stable(plan['seed'], 'arc', channel_id, phrase_index) * 1.5 - .75
    b = _stable(plan['seed'], 'arc', channel_id, phrase_index + 1) * 1.5 - .75
    arc = a + (b - a) * _ease(remainder / length)
    # The full opening and closing spans evolve slowly, not a repeating loop.
    opening = _ease(tick / max(1, min(8 * bar, end / 4)))
    closing = _ease((end - 1 - tick) / max(1, min(8 * bar, end / 4)))
    arc = _clamp(arc + .30 * (energy - .5) - .45 * (1 - min(opening, closing)), -1, 1)
    return dict(energy=energy, tension=tension, impact=impact, arc=arc)


def point_ticks(plan, notes, options):
    ppq, end = plan['ppq'], plan['end_tick']
    division = 32 if int(options.get('division', 16)) >= 32 else 16
    count = math.ceil(end * division / (4 * ppq))
    ticks = {i * 4 * ppq // division for i in range(count)}
    ticks.update((0, end - 1))
    for a, b, _ in _notes(notes, end):
        ticks.update((a, a + max(1, ppq // 4), b, b + ppq, b + 2 * ppq, b + 6 * ppq, b + 10 * ppq))
    for gesture in plan['gestures']:
        ticks.update(gesture[k] for k in ('start', 'peak', 'hold_end', 'end'))
    for drop in plan['drops']:
        a, b = drop['start'], drop['end']
        ticks.update((a - max(1, ppq // 4), a, a + max(1, ppq // 2), a + 2 * ppq, b))
    for build in plan['builds']:
        ticks.update((build['start'], build['end']))
    return sorted(t for t in ticks if 0 <= t < end)


def _gate(spans, tick, ppq, tail=0):
    # The spans are few merged phrases, not thousands of note scans.
    index = bisect_right(spans, (tick, math.inf)) - 1
    if index < 0:
        return 0.0
    a, b = spans[index]
    if tick <= b:
        return _ease((tick - a) / max(1, ppq / 4))
    return 1 - _ease((tick - b) / max(1, tail)) if tail else 0.0


def _gesture_value(gesture, tick):
    if tick < gesture['start'] or tick >= gesture['end']:
        return 0.0
    if tick < gesture['peak']:
        return _ease((tick - gesture['start']) / max(1, gesture['peak'] - gesture['start']))
    if tick <= gesture['hold_end']:
        return 1.0
    return 1 - _ease((tick - gesture['hold_end']) / max(1, gesture['end'] - gesture['hold_end']))


def generate_fx_points(plan, notes, role, lane, options, voice_ids=None):
    """Sample global mixer curves; kick is excluded by the connector.

    Values are normalized native controller values (1/128..1). Only selected
    wet throws reach full range. Bass remains almost dry and its low end is
    not subjected to periodic deep sweeps.
    """
    if lane not in {'cutoff', 'delay', 'reverb'}:
        raise ValueError('Unsupported full-song effect lane.')
    ppq, end = plan['ppq'], plan['end_tick']
    rows = _notes(notes, end)
    spans = _spans(rows, ppq // 2)
    ticks = point_ticks(plan, notes, options)
    ids = {int(cid) for cid in (voice_ids or [])}
    owners = [g for g in plan['gestures'] if g['lane'] == lane and g['channel_id'] in ids] if role != 'bass' else []
    strength = _clamp(float(options.get('strength', 1)))
    intensity = _clamp(float(options.get('intensity', 1.6)), 0, 2)
    amount = strength * min(1.15, intensity / 1.6)
    baseline = 1.0 if lane == 'cutoff' else FLOOR
    result = []
    for tick in ticks:
        macro = sample_macro(plan, tick, min(ids) if ids else None)
        if lane == 'cutoff':
            gate = _gate(spans, tick, ppq, ppq)
            depth = {'bass': .13, 'drums': .22, 'chords': .36, 'arp': .38, 'lead': .30, 'default': .30}.get(role, .30)
            # Activity changes set the broad scene. A build closes gradually,
            # then the genuine arrival opens the filter; no free-running LFO.
            shade = .26 * (1 - macro['energy']) + .40 * max(0, -macro['arc']) + .75 * macro['tension']
            shade *= 1 - macro['impact']
            value = 1 - amount * depth * _clamp(shade) * gate
        else:
            tail = (5 if lane == 'delay' else 10) * ppq
            gate = _gate(spans, tick, ppq, tail)
            bed = {'bass': .012, 'drums': .025, 'chords': .065, 'arp': .045, 'lead': .055, 'default': .05}.get(role, .05)
            if lane == 'reverb' and role == 'chords':
                bed *= 1.3
            if lane == 'delay' and role == 'chords':
                bed *= .5
            value = FLOOR + amount * bed * (.55 + .45 * (1 - macro['energy'])) * gate
            for gesture in owners:
                # No pre-roll into an empty part: the throw must still catch
                # its own phrase or a genuine continuation of that phrase.
                if spans and _gate(spans, gesture['peak'], ppq, tail) > 0:
                    peak = gesture['peak_value']
                    peak = FLOOR + (peak - FLOOR) * min(1, amount)
                    value = max(value, FLOOR + (peak - FLOOR) * _gesture_value(gesture, tick))
            if options.get('drop_fx', 'reduced') != 'keep':
                for drop in plan['drops']:
                    a, b = drop['start'], drop['end']
                    if a - ppq / 4 <= tick < b:
                        if options.get('drop_fx') == 'dry':
                            value = FLOOR
                        else:
                            recovery = _ease((tick - a - ppq / 2) / (1.5 * ppq))
                            # Clear the first hit, then retain bounded space.
                            value = min(value, FLOOR + (.32 if lane == 'delay' else .24) * recovery)
        if tick == 0 or tick == end - 1 or not amount:
            value = baseline
        result.append((tick, _clamp(value, FLOOR, 1)))
    return result
