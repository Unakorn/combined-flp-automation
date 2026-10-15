"""Audio-measured, reduction-only stem balancing with gentle section movement.

RMS references belong to each part and section, never to the whole mix. These
are sample measurements, not LUFS, perceived loudness, or a mastering target.
Optional note timing protects existing attacks; note velocity never sets gain.
"""
from bisect import bisect_left, bisect_right
from copy import deepcopy
import math

DEFAULT_SETTINGS = dict(amount_scale=1.0, level_threshold_db=1.5,
    max_level_correction_db=3.0, section_depth_db=2.0,
    attack_ms=250.0, release_ms=800.0, active_floor_db=-60.0)
_BOUNDS = dict(amount_scale=(0, 1.5), level_threshold_db=(0, 12),
    max_level_correction_db=(0, 6), section_depth_db=(0, 6),
    attack_ms=(100, 2000), release_ms=(300, 5000), active_floor_db=(-90, -30))
_INTENSITY = {'Low': .4, 'Medium-Low': .55, 'Medium': .7, 'Medium-High': .85, 'High': 1.0}
_DRUMS = {'kick', 'snare', 'clap', 'tom', 'hat', 'perc', 'ride', 'crash'}
_MELODIC = {'brass', 'guitar', 'keys', 'pads', 'synth'}
_ROLES = _DRUMS | _MELODIC | {'bass', 'vocals', 'effects', 'drums'}
_PROFILES = {'intro': (.85, .4, .1), 'verse': (.3, .15, .35),
    'pre-chorus': (.65, .3, 0), 'chorus': (0, .09, 0), 'drop': (0, .09, 0),
    'bridge': (.4, .6, .25), 'outro': (.15, .5, 1)}
MAX_POINTS = 64000
_ZERO = 1e-8


def validate_settings(settings=None):
    if settings is not None and not isinstance(settings, dict):
        raise ValueError('Volume balance settings must be an object.')
    result = dict(DEFAULT_SETTINGS)
    for key, value in (settings or {}).items():
        if key not in _BOUNDS:
            raise ValueError(f'Unknown volume balance setting: {key}.')
        lower, upper = _BOUNDS[key]
        if not _finite(value) or not lower <= value <= upper:
            raise ValueError(f'{key} must be a finite number from {lower} to {upper}.')
        result[key] = value
    return result


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _db(value):
    return 20 * math.log10(max(value, 1e-12))


def _quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    left = int(position)
    return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (position - left)


def _value(points, tick):
    index = max(0, bisect_right(points, (tick, math.inf)) - 1)
    if index + 1 == len(points):
        return points[index][1]
    (a, x), (b, y) = points[index:index + 2]
    return x + (y - x) * (tick - a) / (b - a)


def _compact(points):
    result = []
    for point in sorted({round(t): round(v, 6) for t, v in points}.items()):
        while len(result) > 1:
            a, b = result[-2:]
            expected = a[1] + (point[1] - a[1]) * (b[0] - a[0]) / (point[0] - a[0])
            if abs(expected - b[1]) > 1e-5:
                break
            result.pop()
        result.append(point)
    return result


def _merge(spans):
    result = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def _section_rows(sections, ppq, amount):
    rows, previous = [], 0
    for section in sorted(deepcopy(sections), key=lambda row: row.get('start_tick', 0)):
        start, end = section.get('start_tick'), section.get('end_tick')
        if type(start) is not int or type(end) is not int or start < previous or end <= start:
            raise ValueError('Sections need ordered, nonoverlapping positive tick ranges.')
        kind = str(section.get('type', '')).lower()
        kind = {'prechorus': 'pre-chorus', 'pre_chorus': 'pre-chorus', 'build': 'pre-chorus', 'chorus/drop': 'chorus'}.get(kind, kind)
        if kind not in _PROFILES:
            raise ValueError(f'Confirm the section type {kind!r} before volume balancing.')
        energy = section.get('energy', .65)
        if not _finite(energy):
            raise ValueError('Section energy must be finite.')
        rows.append(dict(section, type=kind, bar=start // (4 * ppq) + 1,
                         energy=max(0, min(1, energy)), amount=amount, intensity_value=amount))
        previous = end
    return rows


def _section_shape(rows):
    points, previous_value, previous_end = [], None, 0
    for row in rows:
        start, end = row['start_tick'], row['end_tick']
        first, middle, last = _PROFILES[row['type']]
        if previous_value is None:
            previous_value = first
            points.append((0, first))
        if start > previous_end:
            points.append((start, previous_value))
        width = end - start
        points.extend([(start, previous_value), (start + round(width * .2), first),
                       (start + round(width * .6), middle), (end, last)])
        previous_value, previous_end = last, end
    return _compact(points)


def _windows(stem, ppq, bpm, song_end):
    raw = (stem.get('analysis') or {}).get('level_windows')
    if raw is None or raw == []:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{stem['name']}: level measurements must be a list.")
    offset = stem.get('offset_tick', stem.get('position', 0))
    if not _finite(offset) or offset < 0:
        raise ValueError(f"{stem['name']}: invalid stem placement.")
    duration = stem.get('duration', song_end - offset)
    if not _finite(offset) or not _finite(duration) or offset < 0 or duration <= 0:
        raise ValueError(f"{stem['name']}: invalid stem placement or duration.")
    ratio, stop = bpm * ppq / 60, min(song_end, offset + duration)
    result, previous_end = [], 0.0
    for window in raw:
        if not isinstance(window, dict) or any(not _finite(window.get(key)) for key in ('start_seconds', 'end_seconds', 'rms', 'peak')):
            raise ValueError(f"{stem['name']}: level measurements must contain finite times, RMS and sample peaks.")
        start, end, rms, peak = (window[key] for key in ('start_seconds', 'end_seconds', 'rms', 'peak'))
        if (start < 0 or end <= start or abs(start - previous_end) > 1e-5
                or rms < 0 or peak < 0 or rms > peak + max(1e-9, peak * 1e-6)):
            raise ValueError(f"{stem['name']}: level measurements overlap, omit audio, or contain invalid levels.")
        previous_end = end
        a, b = max(0, round(offset + start * ratio)), min(round(stop), round(offset + end * ratio))
        if b > a:
            result.append(dict(start_tick=a, end_tick=b, tick=(a + b) // 2, rms=rms, peak=peak))
    return result


def _references(windows, rows, floor, role):
    starts = [row['start_tick'] for row in rows]
    groups = [[] for _ in rows]
    for window in windows:
        index = bisect_right(starts, window['tick']) - 1
        index = index if index >= 0 and window['tick'] < rows[index]['end_tick'] else None
        window['section_index'] = index
        if index is not None and window['rms'] > _ZERO:
            groups[index].append(window['rms'])
    references, measured = [], []
    for row, levels in zip(rows, groups):
        gate = max(10 ** (floor / 20), _quantile(levels, .8) * 10 ** (-24 / 20)) if levels else 10 ** (floor / 20)
        active = [level for level in levels if level >= gate]
        # Drum bins contain the whole decay, not one comparable value per hit.
        # Use the upper active range so normal attacks are not leveled toward
        # the median of their own tails.
        reference = _db(_quantile(active, .95 if role in _DRUMS else .65)) if active else None
        measured.extend(_db(level) for level in active)
        references.append(dict(start_tick=row['start_tick'], end_tick=row['end_tick'],
                               reference_db=reference, active_window_count=len(active), gate=gate))
    return references, measured


def _guide(stem, windows, ratio, song_end, warnings):
    """Timing only: velocities may identify silent notes but never set gain."""
    if stem.get('role') == 'drums' or stem.get('timing_provenance') == 'shared-drum-guide':
        # A full kit guide proves section structure, not ownership of kick,
        # snare, or bus transients. Never consume its notes as individual hits.
        return [], [], 'Audio only', 0
    events = stem.get('note_events')
    spans = stem.get('note_spans') if events is None else None
    if events is None and spans is None:
        return [], [], 'Audio only', 0
    attacks = []
    try:
        if events is not None:
            if not isinstance(events, list):
                raise ValueError()
            for event in events:
                if not isinstance(event, dict) or not _finite(event.get('tick')) or not _finite(event.get('duration')) or event['duration'] < 0:
                    raise ValueError()
                if 'velocity' in event and (not _finite(event['velocity']) or not 0 <= event['velocity'] <= 128):
                    raise ValueError()
                if event.get('velocity', 1) > 0:
                    attacks.append(event['tick'])
        else:
            if not isinstance(spans, list):
                raise ValueError()
            for pair in spans:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2 or not all(_finite(v) for v in pair) or pair[1] < pair[0]:
                    raise ValueError()
                attacks.append(pair[0])
        phrase_ends = stem.get('phrase_ends', [])
        if not isinstance(phrase_ends, list) or not all(_finite(value) for value in phrase_ends):
            raise ValueError()
    except (ValueError, TypeError):
        warnings.append(f"{stem['name']}: invalid note timing was ignored; audio measurements still drive volume balance.")
        return [], [], 'Audio only', 0
    starts = [w['start_tick'] for w in windows]
    matched = []
    for tick in sorted(attacks):
        index = bisect_right(starts, tick) - 1
        if (0 <= tick < song_end and index >= 0 and tick < windows[index]['end_tick']
                and windows[index]['rms'] > _ZERO):
            matched.append(round(tick))
    if not matched:
        return [], [], 'Audio only', 0
    onsets = sorted(set(matched))
    attack_spans = []
    for index, tick in enumerate(onsets):
        guard = max(1, round(.08 * ratio))
        if index + 1 < len(onsets):
            guard = min(guard, max(1, int((onsets[index + 1] - tick) * .4)))
        attack_spans.append((tick, min(song_end, tick + guard)))
    attack_spans = _merge(attack_spans)
    tails = _merge((round(tick), min(song_end, round(tick + .35 * ratio)))
                   for tick in phrase_ends if 0 <= tick < song_end)
    label = stem.get('note_guide_source') or 'Matched note timing'
    return attack_spans, tails, str(label), len(matched)


def _protected(spans, starts, tick):
    index = bisect_right(starts, tick) - 1
    return index >= 0 and tick < spans[index][1]


def _slew(current, target, seconds, increase_rate, decrease_rate):
    return current + max(-decrease_rate * seconds, min(increase_rate * seconds, target - current))


def _coverage(points, spans, start, end):
    active = moving = affected = 0
    for a, b in spans:
        a, b = max(a, start), min(b, end)
        if b <= a:
            continue
        active += b - a
        first, last = bisect_right(points, (a, math.inf)), bisect_left(points, (b, -math.inf))
        cuts = [a] + [t for t, _ in points[first:last]] + [b]
        for left, right in zip(cuts, cuts[1:]):
            x, y = _value(points, left), _value(points, right)
            moving += (right - left) if abs(x - y) > 1e-6 else 0
            affected += (right - left) if max(abs(x), abs(y)) > .02 else 0
    return dict(active_ticks=active, moving_ticks=moving,
                changing_active_fraction=round(moving / active, 4) if active else 0,
                affected_active_fraction=round(affected / active, 4) if active else 0)


def plan_balance(stems, sections, ppq, half_time=False, intensity_override='Auto', baseline_intensity='High', settings=None):
    """Plan at most one full-song native mixer-volume curve per measured stem."""
    settings = validate_settings(settings)
    if type(ppq) is not int or ppq <= 0:
        raise ValueError('Volume balance needs a positive integer project PPQ.')
    baseline = _INTENSITY.get(baseline_intensity) if isinstance(baseline_intensity, str) else baseline_intensity
    if not _finite(baseline) or not 0 < baseline <= 1:
        raise ValueError('Baseline intensity must be a named intensity or a number greater than0 and at most1.')
    if intensity_override not in {'Auto', *_INTENSITY}:
        raise ValueError('Unknown volume balance intensity.')
    amount = settings['amount_scale'] * (baseline if intensity_override == 'Auto' else _INTENSITY[intensity_override])
    rows = _section_rows(sections, ppq, amount)
    intensity_label = intensity_override if intensity_override != 'Auto' else min(_INTENSITY, key=lambda label: abs(_INTENSITY[label] - amount))
    for row in rows:
        row['intensity'] = intensity_label
    warnings, moves, details, activities = [], [], [], {}
    for stem in stems:
        if stem.get('role') not in _ROLES:
            raise ValueError(f"Confirm a role for {stem.get('name', 'Unnamed stem')} before volume balancing.")
    tempos = [(stem.get('analysis') or {}).get('bpm') for stem in stems]
    if any(value is not None and (not _finite(value) or value <= 0) for value in tempos):
        raise ValueError('Stem analysis tempo must be positive and finite.')
    tempos = [value for value in tempos if value is not None]
    has_measurements = any((stem.get('analysis') or {}).get('level_windows') for stem in stems)
    if not tempos and has_measurements:
        raise ValueError('Project tempo is missing from audio analysis; reanalyze the stems before converting measured seconds to ticks.')
    bpm = tempos[0] if tempos else 120
    if tempos and max(tempos) - min(tempos) > .01:
        raise ValueError('Stem analysis tempos differ; reanalyze these stems with the project tempo.')
    song_end = rows[-1]['end_tick'] if rows else 0
    section_shape = _section_shape(rows) if rows else [(0, 0)]
    ratio = bpm * ppq / 60
    for stem in stems:
        role = stem['role']
        detail = dict(stem_id=stem['id'], stem_name=stem['name'], role=role, status='no_audio',
            minimum_db=0.0, maximum_db=0.0, maximum_correction_db=0.0,
            measured_minimum_db=None, measured_maximum_db=None, guide_used='Audio only', explanation='')
        details.append(detail)
        windows = _windows(stem, ppq, bpm, song_end) if rows and not stem.get('missing') else []
        if not windows:
            detail['explanation'] = 'No usable audio-level windows; reanalyze this stem before balancing.'
            warnings.append(f"{stem['name']}: {detail['explanation']}")
            continue
        references, measured = _references(windows, rows, settings['active_floor_db'], role)
        detail['reference_by_section'] = [{k: v for k, v in row.items() if k != 'gate'} for row in references]
        if measured:
            detail.update(measured_minimum_db=min(measured), measured_maximum_db=max(measured))
        if not measured:
            detail.update(status='unchanged', explanation='No active audio above the configured measurement floor; quiet audio is preserved.')
            continue
        attack_spans, tail_spans, guide, matched_count = _guide(stem, windows, ratio, song_end, warnings)
        attack_starts, tail_starts = [a for a, _ in attack_spans], [a for a, _ in tail_spans]
        guide_edges = sorted({tick for span in attack_spans + tail_spans for tick in span})
        detail.update(guide_used=guide, guide_note_count=matched_count)
        cap = 1.5 if role in {'kick', 'drums'} else 3.0 if role in _DRUMS | {'bass'} else 6.0
        # Mixed drums retain their internal dynamics. Only a slow section
        # envelope is permitted; no RMS leveling or inferred per-hit gain.
        level_factor = 0.0 if role == 'drums' else .25 if role == 'kick' else .35 if role in _DRUMS else 1.0
        section_factor = .15 if role in {'kick', 'drums'} else .35 if role in _DRUMS else .45 if role == 'bass' else 1.0
        level_limit = min(cap, settings['max_level_correction_db'] * amount * level_factor)
        section_depth = min(cap, settings['section_depth_db'] * amount * section_factor)
        up_rate = max(level_limit, .001) * 1000 / settings['attack_ms']
        down_rate = max(level_limit, .001) * 1000 / settings['release_ms']
        points = [(0, 0.0), (windows[0]['start_tick'], 0.0)]
        correction = section_level = maximum_correction = 0.0
        previous_gain = 0.0
        previous_rms = previous_peak = 0.0
        active_spans = []
        for window in windows:
            start, end, tick = window['start_tick'], window['end_tick'], window['tick']
            index = window['section_index']
            reference = references[index] if index is not None else None
            if window['rms'] <= _ZERO:
                # Reset only *inside* measured silence, never at a note-off
                # or at the reference gate while a quiet release still plays.
                points.extend([(start, previous_gain), (min(end, start + 1), 0.0), (end, 0.0)])
                correction = section_level = previous_gain = 0.0
                previous_rms = previous_peak = 0.0
                continue
            active_spans.append((start, end))
            is_active = reference is not None and window['rms'] >= reference['gate']
            target_correction = correction
            if is_active:
                excess = max(0.0, _db(window['rms']) - reference['reference_db'] - settings['level_threshold_db']) if reference['reference_db'] is not None else 0.0
                target_correction = min(level_limit, excess * amount * level_factor)
                # A high sample crest is evidence of a transient, not a reason
                # to turn down a whole phrase as though its RMS were constant.
                if _db(window['peak'] / max(window['rms'], 1e-12)) > 12:
                    target_correction *= .5
            transient = (role in _DRUMS and window['rms'] > max(_ZERO, previous_rms) * 2
                         and window['peak'] > max(_ZERO, previous_peak) * 1.4)
            first, last = bisect_right(guide_edges, start), bisect_left(guide_edges, end)
            edges = [start] + guide_edges[first:last] + [end]
            points.append((start, previous_gain))
            for left, right in zip(edges, edges[1:]):
                if is_active:
                    wanted = target_correction
                    if transient or _protected(attack_spans, attack_starts, left):
                        wanted = min(wanted, correction)
                    if _protected(tail_spans, tail_starts, left):
                        wanted = correction
                    seconds = (right - left) / ratio
                    correction = _slew(correction, wanted, seconds, up_rate, down_rate)
                    section_target = section_depth * _value(section_shape, right)
                    section_rate = .25 if role == 'drums' else 1.0
                    section_level = _slew(section_level, section_target, seconds, section_rate, section_rate)
                # Quiet releases retain their gain. Guide-edge anchors keep
                # protection exact while leaving measured correction time
                # between fast notes, rather than freezing entire100ms bins.
                gain = -min(cap, max(0.0, section_level + correction))
                maximum_correction = max(maximum_correction, max(0.0, -gain - section_level))
                points.append((right, gain))
                previous_gain = gain
            previous_rms, previous_peak = window['rms'], window['peak']
        last_end = windows[-1]['end_tick']
        points.append((last_end, previous_gain))
        if last_end < song_end:
            points.extend([(min(song_end, last_end + 1), 0.0), (song_end, 0.0)])
        else:
            points.append((song_end, previous_gain))
        points = _compact(points)
        detail.update(minimum_db=min(v for _, v in points), maximum_db=max(v for _, v in points),
                      maximum_correction_db=round(maximum_correction, 6), point_count=len(points))
        if len(points) > MAX_POINTS:
            detail.update(status='skipped', explanation=f'The curve exceeds {MAX_POINTS:,} native points; no balance automation is proposed.')
            warnings.append(f"{stem['name']}: {detail['explanation']}")
            continue
        if all(abs(v) < 1e-5 for _, v in points):
            detail.update(status='unchanged', explanation='Audio measurements and protection rules produced no volume change at these settings.')
            continue
        activities[stem['id']] = _merge(active_spans)
        detail.update(status='planned', explanation='Reductions use this part’s own section RMS references; saved volume remains the ceiling. Quiet releases retain their gain.' +
                      (' Matched note attacks and phrase releases protect existing accents; velocities are not applied again.' if guide != 'Audio only' else ' Audio measurements supply all timing.'))
        if role == 'drums':
            detail['explanation'] = ('Mixed drums use only a slow section envelope capped at 1.5 dB reduction. '
                                     'Internal hit dynamics stay intact; no per-hit note ownership or RMS leveling is inferred.')
        moves.append(dict(stem_id=stem['id'], stem_name=stem['name'], role=role,
            family='Drums' if role in _DRUMS | {'drums'} else 'Melodic' if role in _MELODIC else role.capitalize(),
            mixer_insert=stem.get('mixer_insert'), plugin='Mixer', parameter='volume_db', units='dB',
            name='Volume balance and section movement', control_description=detail['explanation'],
            shape='explicit', start_tick=0, end_tick=song_end, points=points,
            start_value=points[0][1], end_value=points[-1][1], neutral_value=0, relative=True,
            section_start_tick=0, section_type='full-song', section_bar=1,
            intensity=intensity_label,
            intensity_value=amount, control_family='volume', balance=True,
            minimum_value=detail['minimum_db'], maximum_value=detail['maximum_db'],
            coverage=_coverage(points, activities[stem['id']], 0, song_end), activity_source='Measured audio levels'))
    section_activity = []
    for section in rows:
        targets, active_ids = [], set()
        for move in moves:
            coverage = _coverage(move['points'], activities[move['stem_id']], section['start_tick'], section['end_tick'])
            if coverage['active_ticks']:
                active_ids.add(move['stem_id'])
            if coverage['moving_ticks']:
                targets.append(dict(stem_id=move['stem_id'], plugin='Mixer', parameter='volume_db', **coverage))
        section['summary'] = f"{section['type'].capitalize()} at bar {section['bar']}: audio-based volume movement on {len(active_ids)} active stems."
        section_activity.append(dict(start_tick=section['start_tick'], end_tick=section['end_tick'],
            stem_count=len(active_ids), moving_targets=len(targets), targets=targets))
    if not rows:
        warnings.append('No song sections were available.')
    return dict(moves=moves, sections=rows, summary=[s['summary'] for s in rows], warnings=warnings,
        ppq=ppq, beats_per_bar=4, bpm=bpm, half_time=bool(half_time), settings=settings,
        intensity_override=intensity_override, baseline_intensity=baseline_intensity,
        planning_mode='continuous', movement_mode='volume_balance', engine='volume_balance',
        development_preview=True, balance_details=details, section_activity=section_activity,
        movement_coverage=dict(song_end_tick=song_end, section_activity=section_activity,
            sections_with_movement=sum(s['moving_targets'] > 0 for s in section_activity),
            total_sections=len(rows), note='Coverage measures volume controls during measured audio; listening review is still required.'))
