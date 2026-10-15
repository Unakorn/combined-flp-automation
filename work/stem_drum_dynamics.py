"""MIDI-guided extra volume contrast on rendered drum stems, never MIDI edits."""
from bisect import bisect_right
import math

DRUM_ROLES = {'kick', 'snare', 'clap', 'tom', 'hat', 'perc', 'ride', 'crash'}
MAX_POINTS = 64000  # Leave headroom below the native 65,535-anchor limit.
_DEPTH = {'kick': 1.5, 'snare': 4, 'clap': 4, 'tom': 5,
          'hat': 5, 'perc': 5, 'ride': 4, 'crash': 3}
_HOLD_MS = {'kick': 100, 'snare': 120, 'clap': 120, 'tom': 100,
            'hat': 40, 'perc': 60, 'ride': 150, 'crash': 300}


def _value(points, tick):
    index = max(0, bisect_right(points, (tick, math.inf)) - 1)
    left = points[index]
    if index + 1 == len(points):
        return left[1]
    right = points[index + 1]
    return left[1] + (right[1] - left[1]) * (tick - left[0]) / (right[0] - left[0])


def _compact(points):
    result = []
    for point in sorted({int(t): round(v, 5) for t, v in points}.items()):
        while len(result) > 1:
            a, b = result[-2:]
            expected = a[1] + (point[1] - a[1]) * (b[0] - a[0]) / (point[0] - a[0])
            if abs(expected - b[1]) > 1e-5:
                break
            result.pop()
        result.append(point)
    return result


def _meter_weight(tick, ppq):
    # Repeated, predictable accents; no timing changes or random humanization.
    beat = tick / ppq
    nearest = round(beat)
    if abs(beat - nearest) <= .06:
        return (0, .35, .15, .4)[nearest % 4]
    if abs(beat * 2 - round(beat * 2)) <= .12:
        return .65
    return .9


def apply_drum_dynamics(stem, base_points, sections, *, ppq, bpm, song_end,
                        activity, amount=1.0, minimum_db=-9.0):
    """Return one merged volume curve and an honest per-stem planning report.

    Guide velocities already affect the rendered audio. Their relative order
    supplies only additional attenuation, never an absolute amplitude mapping.
    Curves reach their hit level just before the onset and hold the attack; they do not
    perform the post-onset reduction used on sustained melodic instruments.
    """
    role = stem['role']
    if role not in DRUM_ROLES:
        raise ValueError('Drum dynamics require a confirmed drum role.')
    report = dict(stem_id=stem['id'], stem_name=stem['name'], role=role,
                  status='unmatched', hit_count=0, guide=stem.get('note_guide_source', ''),
                  minimum_db=0.0, maximum_db=0.0, reason='')

    def unchanged(status, reason):
        report.update(status=status, reason=reason)
        return base_points, report

    if not math.isfinite(amount) or not 0 <= amount <= 1:
        raise ValueError('Drum dynamics amount must be from 0 to 1.')
    if not amount:
        return unchanged('disabled', 'Drum dynamics amount is zero.')
    if stem.get('missing'):
        return unchanged('skipped', 'The drum audio is unavailable.')
    if not activity:
        return unchanged('silent', 'No audible drum activity was detected.')
    events = stem.get('note_events')
    if not events:
        return unchanged('unmatched', 'No matched notes; existing section movement is retained.')
    maximum_velocity = stem.get('note_velocity_max')
    if maximum_velocity not in (127, 128):
        return unchanged('skipped', 'The note guide has no supported velocity range.')
    starts = [a for a, _ in activity]
    section_starts = [s['start_tick'] for s in sections]
    tolerance = max(1, round(.035 * bpm * ppq / 60))
    hits = {}
    ignored = 0
    for event in events:
        if not isinstance(event, dict) or any(type(event.get(k)) is not int for k in ('tick', 'duration', 'note', 'velocity')):
            return unchanged('skipped', 'The note guide contains an invalid hit record.')
        tick, velocity = event['tick'], event['velocity']
        if event['duration'] < 0 or not 0 <= event['note'] <= 127 or not 0 <= velocity <= maximum_velocity:
            return unchanged('skipped', 'The note guide contains an invalid note or velocity.')
        index = bisect_right(starts, tick + tolerance) - 1
        section_index = bisect_right(section_starts, tick) - 1
        if (not velocity or not 0 <= tick < song_end or index < 0 or tick >= activity[index][1]
                or section_index < 0 or tick >= sections[section_index]['end_tick']):
            ignored += 1
            continue
        # Chords/layers on one drum stem share one fader, so one onset wins.
        if tick not in hits or velocity > hits[tick]['velocity']:
            hits[tick] = dict(event, activity_end=activity[index][1], section_index=section_index)
    hits = [hits[tick] for tick in sorted(hits)]
    report.update(hit_count=len(hits), ignored_note_count=ignored)
    if not hits:
        return unchanged('unmatched', 'No matched notes fall within audible drum activity.')
    if len(hits) * 5 + len(base_points) + 2 > MAX_POINTS:
        return unchanged('skipped', f'Hit dynamics would exceed the {MAX_POINTS:,}-point planning limit; existing section movement is retained.')
    highest = max(h['velocity'] for h in hits)
    uniform = len({h['velocity'] for h in hits}) == 1
    weights = [_meter_weight(h['tick'], ppq) for h in hits] if uniform else [(highest - h['velocity']) / highest for h in hits]
    if uniform and len(hits) > 1 and max(weights) - min(weights) < 1e-6:
        weights = [(0, .35, .15, .5)[i % 4] for i in range(len(hits))]
    # A uniform guide can begin off the beat; its strongest hit still keeps
    # its current section level, rather than attenuating every hit.
    if uniform:
        lowest = min(weights)
        weights = [w - lowest for w in weights]
    floor = max(minimum_db, -1.5) if role == 'kick' else minimum_db
    depth = min(abs(floor), _DEPTH[role]) * amount
    gains = [-depth * weight * min(1, max(0, sections[h['section_index']].get('amount', 1)))
             for h, weight in zip(hits, weights)]
    if not gains or min(gains) >= -1e-6:
        return unchanged('skipped', 'The matched hits have no distinct quieter accent; existing section movement is retained.')
    report['accent_source'] = 'Repeated meter accents for uniform notes' if uniform else 'Written note velocity order'
    ticks_per_ms = bpm * ppq / 60000
    ramp = max(1, round(8 * ticks_per_ms))
    hold = max(1, round(_HOLD_MS[role] * ticks_per_ms))
    release = max(1, round(60 * ticks_per_ms))
    nearby = max(hold + release + ramp, round(500 * ticks_per_ms))
    envelope = [(0, 0.0)]
    previous_tick, previous_gain = -1, 0.0
    for index, (hit, gain) in enumerate(zip(hits, gains)):
        tick = hit['tick']
        # Set the fader one project tick early when resolution permits it.
        # Adjacent one-tick hits necessarily reach their value at the onset.
        ready = max(0, tick - 1, hits[index - 1]['tick'] + 1 if index else 0)
        pre = max(0, ready - ramp, previous_tick + 1)
        if pre < ready:
            envelope.append((pre, previous_gain))
        envelope.append((ready, gain))
        envelope.append((tick, gain))
        next_tick = hits[index + 1]['tick'] if index + 1 < len(hits) else song_end
        next_pre = max(tick, next_tick - 1 - ramp)
        if index + 1 < len(hits) and next_tick - tick <= nearby:
            # Keep the attack and body intact, changing just before the next hit.
            previous_tick, previous_gain = tick, gain
            continue
        end = min(song_end, hit['activity_end'], next_pre)
        hold_end = min(end, tick + max(hold, min(hit['duration'], nearby)))
        restore_end = min(song_end, next_pre, hold_end + release)
        if hold_end > tick:
            envelope.append((hold_end, gain))
        if restore_end > hold_end:
            envelope.append((restore_end, 0.0))
            previous_tick, previous_gain = restore_end, 0.0
        else:
            previous_tick, previous_gain = max(tick, end), gain
    envelope.append((song_end, 0.0))
    envelope = _compact(envelope)
    ticks = sorted({t for t, _ in base_points} | {t for t, _ in envelope})
    points = _compact((tick, max(floor, min(0, _value(base_points, tick) + _value(envelope, tick)))) for tick in ticks)
    if len(points) > MAX_POINTS:
        return unchanged('skipped', f'The merged curve exceeds {MAX_POINTS:,} points; existing section movement is retained.')
    if all(abs(_value(points, tick) - _value(base_points, tick)) < 1e-5 for tick in ticks):
        return unchanged('skipped', 'The section volume is already at its lower limit; no extra hit contrast is possible.')
    report.update(status='planned', minimum_db=min(v for _, v in points),
                  maximum_db=max(v for _, v in points),
                  reason='Additional volume contrast follows matched hits; source notes and audio files stay unchanged.')
    return points, report
