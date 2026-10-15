"""Straight control ramps and holds derived from each part's played notes.

All musical decisions are explicit anchors. Painting, Amount and drop policy
are resolved at those anchors, then interpolated once. No easing, randomness,
oscillator, smoothing filter or per-grid multiplication is used here.
"""
from __future__ import annotations

from bisect import bisect_right
import math

FLOOR = 1 / 128


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _field(note, key):
    return note[key] if isinstance(note, dict) else getattr(note, key)


def _rows(notes, end):
    return sorted((int(_field(n, 'tick')), min(end, int(_field(n, 'tick')) + int(_field(n, 'duration'))),
                   min(128, int(_field(n, 'velocity')))) for n in notes
                  if 0 <= int(_field(n, 'tick')) < end and int(_field(n, 'duration')) > 0
                  and int(_field(n, 'velocity')) > 0)


def _spans(rows, gap):
    spans = []
    for a, b, _ in rows:
        if spans and a <= spans[-1][1] + gap:
            spans[-1] = (spans[-1][0], max(spans[-1][1], b))
        else:
            spans.append((a, b))
    return spans


def _performance(rows, start, stop, ppq):
    """Four-bar summaries retain dynamics, attacks and actual held lengths."""
    sounding = [(max(start, a), min(stop, b), v) for a, b, v in rows if a < stop and b > start]
    if not sounding:
        return 0.0
    duration = max(1, stop - start)
    played = sum(b - a for a, b in _spans(sounding, 0)) / duration
    average_velocity = sum(v for _, _, v in sounding) / (128 * len(sounding))
    attacks = sum(start <= a < stop for a, _, _ in rows)
    density = min(1.0, attacks * ppq / (4 * duration))
    held = min(1.0, sum(b - a for a, b, _ in sounding) / (len(sounding) * 4 * ppq))
    return _clamp(.45 * average_velocity + .25 * played + .20 * density + .10 * held)


def _at(anchors, tick):
    index = max(0, bisect_right([t for t, _ in anchors], tick) - 1)
    if index == len(anchors) - 1:
        return anchors[index][1]
    a, av = anchors[index]
    b, bv = anchors[index + 1]
    return av + (bv - av) * (tick - a) / (b - a)


def sample_linear(anchors, ppq, end_tick, division=16):
    """Sample straight segments on a musical grid plus every exact anchor."""
    if not isinstance(ppq, int) or ppq <= 0 or not isinstance(end_tick, int) or end_tick <= 0:
        raise ValueError('Linear controls need positive integer timing.')
    rows = list(anchors)
    if not rows or rows[0][0] != 0 or rows[-1][0] != end_tick - 1:
        raise ValueError('Linear anchors must cover tick0 through end_tick-1.')
    previous = -1
    for tick, value in rows:
        if (type(tick) is not int or tick <= previous or not 0 <= tick < end_tick
                or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('Linear anchors must have unique increasing ticks and normalized values.')
        previous = tick
    division = int(division)
    if not 1 <= division <= 64:
        raise ValueError('Choose a controller division between1 and64.')
    ticks = {i * ppq * 4 // division for i in range(math.ceil(end_tick * division / (ppq * 4)))}
    ticks.update(t for t, _ in rows)
    return [(t, _at(rows, t)) for t in sorted(ticks)]


def _settings(plan, options):
    ppq, end = plan['ppq'], plan['end_tick']
    if type(ppq) is not int or ppq <= 0 or type(end) is not int or end <= 0:
        raise ValueError('Linear controls need positive integer timing.')
    strength, intensity = float(options.get('strength', 1)), float(options.get('intensity', 1.6))
    if not math.isfinite(strength) or not 0 <= strength <= 1 or not math.isfinite(intensity) or not 0 <= intensity <= 2:
        raise ValueError('Linear control Amount must be0–200% and strength0–100%.')
    return ppq, end, strength * intensity / 1.6


def _part_anchors(plan, notes, target, baseline, release):
    ppq, end = plan['ppq'], plan['end_tick']
    last, bar = end - 1, ppq * 4
    rows = _rows(notes, end)
    spans = _spans(rows, ppq)  # Staccato articulation stays one played phrase.
    points = {0: baseline, last: baseline}
    for index, (a, b) in enumerate(spans):
        following = spans[index + 1][0] if index + 1 < len(spans) else end
        stop = min(last, b)
        first_stop = min(b, a + 4 * bar)
        first_value = target(_performance(rows, a, first_stop, ppq))
        attack_end = min(stop, a + max(1, min(bar, (b - a) // 4)))
        points[a] = baseline
        if attack_end > a:
            points[attack_end] = first_value
        # Source changes become broad straight ramps, not individual-note
        # ripples. Steady four-bar summaries produce true flat holds.
        previous_value = first_value
        for position in range(a + 4 * bar, b, 4 * bar):
            value = target(_performance(rows, position, min(b, position + 4 * bar), ppq))
            if abs(value - previous_value) < 1 / 256:
                value = previous_value
            points[position] = value
            previous_value = value
        if stop > a:
            points[stop] = previous_value
        finish = min(last, following, b + release)
        if finish > stop:
            points[finish] = baseline
    points[0] = points[last] = baseline
    return sorted(points.items()), rows, spans


def _combine_envelopes(left, right, highest=True):
    """Join two straight envelopes, including their integer-tick crossings."""
    def reader(rows):
        positions = [t for t, _ in rows]
        def read(tick):
            index = max(0, bisect_right(positions, tick) - 1)
            if index + 1 == len(rows):
                return rows[index][1]
            a, av = rows[index]
            b, bv = rows[index + 1]
            return av + (bv - av) * (tick - a) / (b - a)
        return positions, read
    left_ticks, left_at = reader(left)
    right_ticks, right_at = reader(right)
    ticks = sorted(set(left_ticks) | set(right_ticks))
    crossings = set()
    for a, b in zip(ticks, ticks[1:]):
        da, db = left_at(a) - right_at(a), left_at(b) - right_at(b)
        if da * db < 0:
            crossing = a + (b - a) * da / (da - db)
            crossings.update((math.floor(crossing), math.ceil(crossing)))
    choose = max if highest else min
    return [(t, choose(left_at(t), right_at(t))) for t in sorted(set(ticks) | crossings)]


def _end_tail(anchors, last, release, baseline, minimum, maximum):
    # Limit the remaining excursion by the time left for a gentle fade.
    # Short songs reserve up to half their length; late notes cannot produce
    # a full-depth throw followed by a one-tick reset. Existing quieter motion
    # is retained until it meets this straight closing envelope.
    start = last - min(release, max(1, last // 2))
    upper = sorted({0: maximum, start: maximum, last: baseline}.items())
    lower = sorted({0: minimum, start: minimum, last: baseline}.items())
    return _combine_envelopes(_combine_envelopes(anchors, upper, False), lower)


def _finalize(anchors, plan, options, baseline, minimum, maximum, wet=False, wet_release=0):
    """Compose all level decisions at anchor positions; interpolate only later."""
    last, ppq = plan['end_tick'] - 1, plan['ppq']
    curve = options.get('energy_curve')
    paint = [] if curve is None else [(min(last, int(t)), float(v)) for t, v in curve['points']]
    # A curve ending at exact song end maps its final point to the final
    # positive-duration control note. The reset has final precedence below.
    paint = sorted(dict(paint).items())
    ticks = {t for t, _ in anchors} | {t for t, _ in paint}
    policy = options.get('drop_fx', 'reduced')
    if policy not in {'dry', 'reduced', 'keep'}:
        raise ValueError('Choose a valid drop effect policy.')
    drops = plan.get('drops', []) if wet and policy != 'keep' else []
    for drop in drops:
        a, b = drop['start'], drop['end']
        if policy == 'dry':
            ticks.update((max(0, a - ppq // 4), a, min(last, b - 1), min(last, b)))
        else:
            ticks.update((max(0, a - 4 * ppq), a, min(last, a + 8 * ppq)))
    result = []
    for tick in sorted(ticks):
        if not 0 <= tick <= last:
            continue
        value = _at(anchors, tick)
        if paint:
            # Add an explicit level offset at the union, not the product of
            # two sampled ramps (which would silently make a curved line).
            value += (_at(paint, tick) - .5) * (maximum - minimum)
        if policy == 'reduced' and drops:
            # A light relative dip: one bar into the arrival, two bars out.
            # Balanced never turns an existing wet signal abruptly off.
            duck = 0.0
            for drop in drops:
                a = drop['start']
                if a - 4 * ppq <= tick <= a:
                    duck = max(duck, (tick - a + 4 * ppq) / (4 * ppq))
                elif a < tick < a + 8 * ppq:
                    duck = max(duck, 1 - (tick - a) / (8 * ppq))
            duck_depth = options['fx_controls']['duck'] if options.get('fx_controls') is not None else .25
            value = minimum + (value - minimum) * (1 - duck_depth * duck)
        if tick in (0, last):
            value = baseline
        result.append((tick, _clamp(value, minimum, maximum)))
    if wet_release:
        result = _end_tail(result, last, wet_release, baseline, minimum, maximum)
    if policy == 'dry' and drops:
        # Explicit Dry keeps its intentional full-drop mute, including when
        # a drop overlaps the reserved outro fade. The final reset still wins.
        result = [(t, minimum if t not in (0, last) and any(
            d['start'] - ppq // 4 <= t < d['end'] for d in drops) else v) for t, v in result]
    return result


def _configured_wet_part_anchors(plan, notes, role, lane, options, baseline, maximum, internal=False):
    """Explicit levels and musical timing, independent of nonzero Amount.

    Each new phrase starts under the old tail. Throws catch this part's last
    played note or a held note near the chosen boundary; empty periods cannot
    produce throws. All envelopes stay straight and are joined by their max.
    """
    from fx_controls import validate_fx_controls
    settings = validate_fx_controls(options['fx_controls'])[lane]
    ppq, end, _ = _settings(plan, options)
    last, bar = end - 1, ppq * 4
    release = settings['fade_bars'] * bar
    rise = max(1, round(settings['rise_beats'] * ppq))
    rows = _rows(notes, end)
    spans = _spans(rows, ppq)
    combined = [(0, baseline), (last, baseline)]

    def level(value):
        # External values are absolute wet fractions. Internal settings are
        # bounded excursions above the saved patch's existing wet amount.
        return (baseline + (maximum - baseline) * value if internal
                else _clamp(value, baseline, maximum))

    for a, b in spans:
        stop = min(last, b)
        if stop <= a:
            continue
        first = level(settings['amount'] * (.65 + .35 * _performance(rows, a, min(b, a + 4 * bar), ppq)))
        attack = min(stop, a + rise)
        points = {0: baseline, a: baseline, attack: first}
        value = first
        for tick in range(a + 4 * bar, b, 4 * bar):
            value = level(settings['amount'] * (.65 + .35 * _performance(rows, tick, min(b, tick + 4 * bar), ppq)))
            if tick > attack:
                points[tick] = value
        points[stop] = value
        points[min(last, b + release)] = baseline
        points[last] = baseline
        combined = _combine_envelopes(combined, sorted(points.items()))

    throws = settings['throws']
    candidates = []
    if throws == 'phrases':
        candidates = [(a, b) for i, (a, b) in enumerate(spans)
                      if b - a >= 4 * bar
                      and (spans[i + 1][0] if i + 1 < len(spans) else end) - b >= bar]
    elif throws in {'every8', 'every4'}:
        stride = (8 if throws == 'every8' else 4) * bar
        # Include the full-song endpoint if it falls exactly on the grid; the
        # final reserved fade still has precedence over that last throw.
        for boundary in range(stride, end + 1, stride):
            sounding = [(a, min(b, boundary)) for a, b, _ in rows
                        if a < boundary and b > boundary - bar]
            if not sounding:
                continue
            catch_end = max(b for _, b in sounding)
            note_start = min(a for a, b in sounding if b == catch_end)
            span_start = max((a for a, b in spans if a <= note_start < b), default=note_start)
            candidates.append((span_start, catch_end))
    for a, b in sorted(set(candidates)):
        peak_tick = max(a + 1, min(last - 1, b - max(1, ppq // 2)))
        if not a < peak_tick < b or peak_tick >= last:
            continue
        start = max(a, peak_tick - rise)
        # A late single pickup must not receive wet level before it has
        # started. The merged phrase may contain short articulation gaps.
        if not any(na <= peak_tick < nb for na, nb, _ in rows):
            eligible = [(na, nb) for na, nb, _ in rows if na < b and nb >= b]
            if not eligible:
                continue
            onset = min(na for na, _ in eligible)
            peak_tick = max(peak_tick, onset + 1)
            # The earlier notes of this same staccato phrase provide the
            # chosen rise time. Only an isolated pickup clamps to its onset.
            start = max(a, peak_tick - rise)
        if peak_tick >= min(b, last):
            continue
        hold_end = min(last, b)
        points = {t: v for t, v in combined if t < start}
        points.update({0: baseline, start: _at(combined, start), peak_tick: level(settings['peak']),
                       hold_end: level(settings['peak']), min(last, b + release): baseline, last: baseline})
        combined = _combine_envelopes(combined, sorted(points.items()))
    return combined, release


def _wet_part_anchors(plan, notes, role, lane, options, baseline, maximum, internal=False):
    """A steady wet bed and overlapping, slow tails instead of repeated pumps."""
    if options.get('fx_controls') is not None:
        return _configured_wet_part_anchors(plan, notes, role, lane, options, baseline, maximum, internal)
    ppq, end, amount = _settings(plan, options)
    last, bar = end - 1, ppq * 4
    release = (4 if lane == 'delay' else 8) * bar
    rows = _rows(notes, end)
    spans = _spans(rows, ppq)
    combined = [(0, baseline), (last, baseline)]
    ceiling = {'bass': .035, 'drums': .12, 'chords': .20,
               'arp': .16, 'lead': .18, 'default': .16}.get(role, .16)
    if lane == 'delay' and role == 'chords':
        ceiling *= .65
    if internal:
        ceiling = .22
    extent = maximum - baseline
    for index, (a, b) in enumerate(spans):
        stop = min(last, b)
        if stop <= a:
            continue
        def target(start, finish):
            return baseline + extent * min(.45, amount * ceiling * _performance(rows, start, finish, ppq))
        first = target(a, min(b, a + 4 * bar))
        attack = min(stop, a + max(1, min(bar, (b - a) // 4)))
        points = {0: baseline, a: baseline, attack: first}
        value = first
        for tick in range(a + 4 * bar, b, 4 * bar):
            value = target(tick, min(b, tick + 4 * bar))
            points[tick] = value
        points[stop] = value
        following = spans[index + 1][0] if index + 1 < len(spans) else end
        # Only a substantial phrase followed by a real space gets a throw.
        # Full-range remains available at 200%; it is no longer every ending.
        if (role != 'bass' and not internal and b - a >= 4 * bar
                and following - b >= bar and b + release <= last):
            normal_peak = .50 if lane == 'delay' else .60
            peak = normal_peak * min(1, amount)
            peak += (1 - normal_peak) * _clamp((amount - 1) / .25)
            if options.get('fx_style', 'full_range') != 'full_range':
                peak *= .70
            catch = max(a + 1, b - ppq // 2)
            rise = max(a, catch - (2 if lane == 'delay' else 4) * ppq)
            old = sorted(points.items())
            points = {t: v for t, v in points.items() if not rise < t < catch}
            points[rise] = _at(old, rise)
            points[catch] = points[stop] = baseline + extent * peak
        # Unlike 2.4.0, the NEXT phrase does not truncate this release.
        points[min(last, b + release)] = baseline
        points[last] = baseline
        combined = _combine_envelopes(combined, sorted(points.items()))
    return combined, release


def fx_anchors(plan, notes, role, lane, options, voice_ids=None):
    ppq, end, amount = _settings(plan, options)
    if lane not in {'cutoff', 'delay', 'reverb'}:
        raise ValueError('Unsupported linear effect lane.')
    baseline = 1.0 if lane == 'cutoff' else FLOOR
    if not amount or end == 1:
        return [(0, baseline)] if end == 1 else [(0, baseline), (end - 1, baseline)]
    if lane == 'cutoff':
        minimum = .55 if role == 'bass' else .12 if role != 'drums' else .32
        def target(score):
            return _clamp(1 - amount * (1 - minimum) * (1 - score), minimum, 1)
        anchors, _, _ = _part_anchors(plan, notes, target, baseline, ppq)
        return _finalize(anchors, plan, options, baseline, minimum, 1)
    if options.get('fx_controls') is not None:
        from fx_controls import validate_fx_controls
        settings = validate_fx_controls(options['fx_controls'])[lane]
        if settings['amount'] == settings['peak'] == 0:
            return [(0, baseline), (end - 1, baseline)]
    allow_bass = options.get('fx_controls') is not None and options['fx_controls']['bass_full_range']
    maximum = .10 if role == 'bass' and not allow_bass else 1
    anchors, release = _wet_part_anchors(plan, notes, role, lane, options, baseline, maximum)
    return _finalize(anchors, plan, options, baseline, FLOOR, maximum, wet=True, wet_release=release)


def synth_anchors(plan, control, notes, options):
    ppq, end, amount = _settings(plan, options)
    low, baseline, high = (float(control[k]) for k in ('minimum', 'baseline', 'maximum'))
    if not all(math.isfinite(v) for v in (low, baseline, high)) or not 0 <= low <= baseline <= high <= 1:
        raise ValueError('Linear synth movement needs safe saved-preset bounds.')
    cid = str(control.get('channel_id', ''))
    role = plan.get('voice_roles', {}).get(cid, control.get('role', 'default'))
    character = control['character']
    if not amount or end == 1 or high == low or character == 'width' and role == 'bass':
        return [(0, baseline)] if end == 1 else [(0, baseline), (end - 1, baseline)]
    if character == 'effect' and control.get('parameter_id') in {25, 143}:
        lane = 'delay' if control['parameter_id'] == 25 else 'reverb'
        if options.get('fx_controls') is not None:
            from fx_controls import validate_fx_controls
            settings = validate_fx_controls(options['fx_controls'])[lane]
            if settings['amount'] == settings['peak'] == 0:
                return [(0, baseline), (end - 1, baseline)]
        anchors, release = _wet_part_anchors(plan, notes, role, lane, options, baseline, high, internal=True)
        return _finalize(anchors, plan, options, baseline, low, high, wet=True, wet_release=release)
    scale = 1 if control.get('song_primary') else .75
    if character in {'decay', 'release'}:
        scale *= .55
    if role == 'bass':
        scale *= .65
    def target(score):
        direction = (score * 2 - 1) * amount * 1.5 * scale
        if character in {'decay', 'release'}:
            direction *= -1
        extent = high - baseline if direction >= 0 else baseline - low
        return _clamp(baseline + direction * extent, low, high)
    anchors, _, _ = _part_anchors(plan, notes, target, baseline, ppq)
    time_effect = character == 'effect' and control.get('parameter_id') in {25, 143}
    return _finalize(anchors, plan, options, baseline, low, high, wet=time_effect)


def generate_fx_points(plan, notes, role, lane, options, voice_ids=None):
    return sample_linear(fx_anchors(plan, notes, role, lane, options, voice_ids), plan['ppq'], plan['end_tick'], options.get('division', 16))


def generate_synth_points(plan, control, notes, options):
    return sample_linear(synth_anchors(plan, control, notes, options), plan['ppq'], plan['end_tick'], options.get('division', 16))
