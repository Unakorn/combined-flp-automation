"""Section-led, full-song native automation curves for the development preview.

Values are control units, not normalized FLP values. Filter percentages describe
verified knob positions, not frequencies. Mixer values are offsets from the saved
mix. Effect percentages describe the blend of a dedicated effect slot.
"""
from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
import math
from stem_drum_dynamics import apply_drum_dynamics


DEFAULT_SETTINGS = {
    'filter_minimum_percent': 28.0,
    'filter_maximum_percent': 100.0,
    'volume_minimum_db': -9.0,
    'amount_scale': 1.0,
    'pan_maximum_percent': 24.0,
    'delay_maximum_percent': 32.0,
    'reverb_maximum_percent': 30.0,
    'rhythm_depth_db': 6.0,
    'rhythm_release_ms': 180.0,
    'phrase_bars': 8,
    'drum_dynamics_amount': 1.0,
}
_BOUNDS = {
    'filter_minimum_percent': (0, 99), 'filter_maximum_percent': (1, 100),
    'volume_minimum_db': (-12, 0), 'amount_scale': (.1, 1.5),
    'pan_maximum_percent': (0, 40), 'delay_maximum_percent': (0, 50),
    'reverb_maximum_percent': (0, 50), 'rhythm_depth_db': (0, 12),
    'rhythm_release_ms': (100, 1000), 'phrase_bars': (4, 16),
    'drum_dynamics_amount': (0, 1),
}
_INTENSITY = {'Low': .4, 'Medium-Low': .55, 'Medium': .7,
              'Medium-High': .85, 'High': 1.0}
_DRUMS = {'kick', 'snare', 'clap', 'tom', 'hat', 'perc', 'ride', 'crash'}
_MELODIC = {'brass', 'guitar', 'keys', 'pads', 'synth'}
_LABELS = {'intro': 'Opening', 'verse': 'Verse', 'pre-chorus': 'Before chorus',
           'chorus': 'Chorus', 'drop': 'Main section', 'bridge': 'Middle section',
           'outro': 'Ending'}
_PROFILES = {
    'filter': {
        'intro': (0, .48, .92), 'verse': (.72, .35, .82),
        'pre-chorus': (.26, .55, 1), 'chorus': (1, .70, 1),
        'bridge': (.58, .08, .58), 'outro': (.72, .38, 0),
    },
    'volume': {
        'intro': (.7, .4, .1), 'verse': (.32, .08, .35),
        'pre-chorus': (.45, .28, 0), 'chorus': (0, .09, 0),
        'bridge': (.42, .65, .24), 'outro': (.18, .5, 1),
    },
    'pan': {
        'intro': (-.15, .45, .1), 'verse': (.65, -.65, .35),
        'pre-chorus': (-.25, .25, 0), 'chorus': (.3, -.3, .15),
        'bridge': (.8, -.8, .3), 'outro': (.5, -.3, 0),
    },
    'delay': {
        'intro': (.1, .45, .15), 'verse': (.12, .45, .2),
        'pre-chorus': (.2, .8, .1), 'chorus': (.16, .42, .16),
        'bridge': (.35, .8, .25), 'outro': (.25, .7, 0),
    },
    'reverb': {
        'intro': (.3, .75, .3), 'verse': (.2, .5, .2),
        'pre-chorus': (.3, .85, .12), 'chorus': (.15, .4, .15),
        'bridge': (.65, 1, .4), 'outro': (.5, .85, 0),
    },
}


def validate_settings(settings=None):
    """Merge editable settings with defaults; reject typos and unsafe values."""
    if settings is not None and not isinstance(settings, dict):
        raise ValueError('Continuous movement settings must be an object.')
    result = dict(DEFAULT_SETTINGS)
    for key, value in (settings or {}).items():
        if key not in _BOUNDS:
            raise ValueError(f'Unknown continuous movement setting: {key}.')
        low, high = _BOUNDS[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key} must be a finite number from {low} to {high}.')
        result[key] = value
    if result['filter_minimum_percent'] >= result['filter_maximum_percent']:
        raise ValueError('Filter minimum must be below its maximum.')
    if result['phrase_bars'] not in (4, 8, 16):
        raise ValueError('phrase_bars must be 4, 8, or 16.')
    return result


def _finite(value, fallback=0):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else fallback


def _baseline(value):
    if isinstance(value, str) and value in _INTENSITY:
        return _INTENSITY[value]
    if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and 0 < value <= 1:
        return float(value)
    raise ValueError('Baseline intensity must be a named intensity or a finite number greater than 0 and at most 1.')


def _intensity_label(amount):
    return min(_INTENSITY, key=lambda label: abs(_INTENSITY[label] - amount))


def _merge(spans, gap=0):
    merged = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + gap:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _activity(stem, song_end, ppq, bpm):
    """Prefer audible intervals, including instrument tails, over note gates."""
    analysis = stem.get('analysis') or {}
    offset = _finite(stem.get('offset_tick', stem.get('position', 0)))
    if 'activity_segments' in analysis:
        ratio = _finite(analysis.get('bpm'), bpm) * ppq / 60
        spans = [(round(offset + _finite(s.get('start_seconds')) * ratio),
                  round(offset + _finite(s.get('end_seconds')) * ratio))
                 for s in analysis['activity_segments']]
        source = 'Audio activity'
    elif 'note_spans' in stem:
        spans = [(round(s[0]), round(s[1])) for s in stem['note_spans'] if len(s) == 2]
        source = 'Note timing'
    else:
        start = max(0, round(offset))
        duration = _finite(stem.get('duration'), song_end - start)
        spans = [(start, min(song_end, round(start + duration)))]
        source = 'Stem placement; detailed activity unavailable'
    return _merge([(max(0, a), min(song_end, b)) for a, b in spans]), source


def _value(points, tick):
    index = bisect_right(points, (tick, float('inf'))) - 1
    if index < 0:
        return points[0][1]
    if index >= len(points) - 1:
        return points[-1][1]
    left, right = points[index], points[index + 1]
    return left[1] + (right[1] - left[1]) * (tick - left[0]) / (right[0] - left[0])


def _simplify(points, end):
    """Keep exact linear curves; remove duplicate ticks and collinear points."""
    ordered = sorted({max(0, min(end, int(round(t)))): round(float(v), 5)
                      for t, v in points}.items())
    result = []
    for point in ordered:
        while len(result) > 1:
            a, b = result[-2:]
            expected = a[1] + (point[1] - a[1]) * (b[0] - a[0]) / (point[0] - a[0])
            if abs(expected - b[1]) > 1e-5:
                break
            result.pop()
        result.append(point)
    return result


def _section_curves(sections, kind, mapper, phrase_ticks, song_end):
    points = []
    previous_end = 0
    previous_value = None
    for section in sections:
        start, end = section['start_tick'], section['end_tick']
        profile = _PROFILES[kind]['chorus' if section['type'] == 'drop' else section['type']]
        if previous_value is None:
            previous_value = mapper(profile[0], section)
            points.append((0, previous_value))
        if start > previous_end:
            points.append((start, previous_value))
        # Opening, build, and ending have one long direction. Other sections
        # develop over an editable 4/8/16-bar phrase, rather than holding still.
        span = end - start if section['type'] in {'intro', 'pre-chorus', 'outro'} else phrase_ticks
        for phrase_start in range(start, end, max(1, span)):
            phrase_end = min(end, phrase_start + span)
            width = phrase_end - phrase_start
            points.extend([(phrase_start, previous_value),
                           (phrase_start + round(width * .20), mapper(profile[0], section)),
                           (phrase_start + round(width * .60), mapper(profile[1], section)),
                           (phrase_end, mapper(profile[2], section))])
            previous_value = mapper(profile[2], section)
        previous_end = end
    points.append((song_end, previous_value))
    return _simplify(points, song_end)


def _mask_curve(points, spans, neutral, song_end, edge_ticks):
    """Keep short rests within a phrase; flatten longer rests to neutral.

    The caller has merged sub-bar rests. The entry and exit ramps are inside
    each phrase, so an absent stem does not receive invented movement.
    """
    result = [(0, neutral), (song_end, neutral)]
    for start, end in spans:
        width = end - start
        edge = min(edge_ticks, max(1, width // 4))
        left = start + edge if start > 0 else start
        right = end - edge if end < song_end else end
        result.extend([(start, neutral if start > 0 else _value(points, start)),
                       (left, _value(points, left))])
        result.extend((t, v) for t, v in points if left < t < right)
        result.extend([(right, _value(points, right)),
                       (end, neutral if end < song_end else _value(points, end))])
    return _simplify(result, song_end)


def _rhythm(points, sections, role, sustained, flavor, settings, ppq, bpm, half_time, song_end, kick_onsets):
    if flavor == 'long_melodic' or not (role == 'bass' or role in _MELODIC and sustained >= .55):
        return points, 0
    release = max(1, math.ceil(settings['rhythm_release_ms'] * bpm * ppq / 60000))
    attack = max(1, math.ceil(.015 * bpm * ppq / 60))
    cycle = ppq * (2 if half_time else 1)
    while cycle < release + attack + ppq * .2:
        cycle *= 2
    result = list(points)
    count = 0
    for section in sections:
        if section['type'] not in {'chorus', 'drop'} or section['energy'] < .45:
            continue
        depth = settings['rhythm_depth_db'] * section['amount'] * (1 if role == 'bass' else .65)
        start, end = section['start_tick'], section['end_tick']
        # Half-time and explicit phrase structure alter the spacing, not the
        # project timebase. Only the main sections receive repeated reductions.
        anchors = list(range(start, end, cycle)) if kick_onsets is None else [t for t in kick_onsets if start <= t < end]
        if not anchors:
            continue
        result = [(t, v) for t, v in result if not start < t < end]
        previous_release = start - 1
        for tick in anchors:
            if tick < previous_release:
                continue
            if tick + attack + release >= end:
                break
            base = _value(points, tick)
            result.extend([(tick, base),
                           (tick + attack, max(settings['volume_minimum_db'], base - depth)),
                           (tick + attack + release, _value(points, tick + attack + release))])
            count += 1
            previous_release = tick + attack + release
        result.append((end, _value(points, end)))
    return _simplify(result, song_end), count


def _covered(points, spans, start, end, threshold):
    active = moving = affected = 0
    neutral = 100 if threshold == .25 else 0
    for left, right in spans:
        left, right = max(start, left), min(end, right)
        if right <= left:
            continue
        active += right - left
        first = bisect_right(points, (left, float('inf')))
        last = bisect_right(points, (right, -float('inf')))
        cuts = [left] + [t for t, _ in points[first:last]] + [right]
        for a, b in zip(cuts, cuts[1:]):
            va, vb = _value(points, a), _value(points, b)
            if abs(va - vb) > 1e-6:
                moving += b - a
            if max(abs(va - neutral), abs(vb - neutral)) > threshold:
                affected += b - a
    return {'active_ticks': active, 'moving_ticks': moving,
            'changing_active_fraction': round(moving / active, 4) if active else 0,
            'affected_active_fraction': round(affected / active, 4) if active else 0}


def plan_continuous(stems, sections, ppq, half_time=False, intensity_override='Auto', flavor='combined', settings=None, baseline_intensity='High'):
    """Plan one full-song curve per target; no velocity or gain above saved mix.

    Coverage reports describe mathematical movement during detected audio. They
    do not certify audibility, balance, sound quality, or the rendered result.
    """
    settings = validate_settings(settings)
    baseline = _baseline(baseline_intensity)
    if isinstance(ppq, bool) or not isinstance(ppq, int) or ppq <= 0:
        raise ValueError('Continuous movement needs a positive integer project PPQ.')
    if intensity_override not in {'Auto', *_INTENSITY}:
        raise ValueError(f'Unknown intensity: {intensity_override}.')
    flavor = {'long-melodic': 'long_melodic', 'melodic': 'long_melodic'}.get(flavor, flavor)
    if flavor not in {'combined', 'long_melodic', 'rhythmic'}:
        raise ValueError('Movement flavor must be combined, long_melodic, or rhythmic.')
    for stem in stems:
        if stem.get('role') not in _DRUMS | _MELODIC | {'bass', 'vocals', 'effects'}:
            raise ValueError(f'Confirm a role for {stem.get("name", "Unnamed stem")} before planning.')
    warnings = []
    rows = []
    previous_end = 0
    for section in sorted(deepcopy(sections), key=lambda s: s.get('start_tick', 0)):
        start, end = section.get('start_tick'), section.get('end_tick')
        if type(start) is not int or type(end) is not int or start < previous_end or end <= start:
            raise ValueError('Sections need ordered, nonoverlapping positive tick ranges.')
        section_type = str(section.get('type', 'verse')).lower()
        section_type = {'prechorus': 'pre-chorus', 'pre_chorus': 'pre-chorus', 'build': 'pre-chorus', 'chorus/drop': 'chorus'}.get(section_type, section_type)
        if section_type not in _LABELS:
            raise ValueError(f'Confirm the section type {section_type!r} before planning.')
        energy = section.get('energy', .65)
        if isinstance(energy, bool) or not isinstance(energy, (int, float)) or not math.isfinite(energy):
            raise ValueError('Section energy must be a finite number.')
        energy = max(0, min(1, energy))
        amount = settings['amount_scale'] * (baseline * (.85 + energy * .15) if intensity_override == 'Auto' else _INTENSITY[intensity_override])
        rows.append({**section, 'type': section_type, 'energy': energy, 'amount': amount,
                     'bar': start // (ppq * 4) + 1, 'intensity': intensity_override if intensity_override != 'Auto' else _intensity_label(amount),
                     'intensity_value': round(amount, 4)})
        previous_end = end
    if not rows:
        return {'moves': [], 'sections': [], 'summary': [], 'warnings': ['No song sections were available.'], 'movement_coverage': {}, 'drum_dynamics': []}
    song_end = rows[-1]['end_tick']
    if song_end > ppq * 4 * 10000:
        raise ValueError('Continuous movement is limited to 10,000 bars per song.')
    bpms = [_finite((s.get('analysis') or {}).get('bpm')) for s in stems]
    bpms = [b for b in bpms if b > 0]
    bpm = bpms[0] if bpms else 120.0
    if not bpms:
        warnings.append('Tempo was unavailable in stem analysis; 120 BPM is used only for release timing.')
    elif max(bpms) - min(bpms) > .01:
        warnings.append('Stem analysis tempos differ; the first analyzed tempo is used for release timing.')
    moves = []
    drum_dynamics = []
    activity_by_stem = {}
    phrase_ticks = round(settings['phrase_bars'] * 4 * ppq * (2 if half_time else 1))
    edge_ticks = max(1, math.ceil(settings['rhythm_release_ms'] * bpm * ppq / 60000))
    kick_guides = [s for s in stems if s.get('role') == 'kick' and 'note_spans' in s and not s.get('missing')]
    kick_onsets = sorted({round(pair[0]) for s in kick_guides for pair in s['note_spans'] if len(pair) == 2}) if kick_guides else None
    for stem_index, stem in enumerate(stems):
        role = stem.get('role')
        if stem.get('missing'):
            if role in _DRUMS:
                _, report = apply_drum_dynamics(stem, [(0, 0), (song_end, 0)], rows,
                    ppq=ppq, bpm=bpm, song_end=song_end, activity=[],
                    amount=settings['drum_dynamics_amount'], minimum_db=settings['volume_minimum_db'])
                drum_dynamics.append(report)
                warnings.append(f"{stem['name']}: {report['reason']}")
            continue
        activity, activity_source = _activity(stem, song_end, ppq, bpm)
        activity_by_stem[stem['id']] = activity
        if not activity:
            if role in _DRUMS:
                _, report = apply_drum_dynamics(stem, [(0, 0), (song_end, 0)], rows,
                    ppq=ppq, bpm=bpm, song_end=song_end, activity=activity,
                    amount=settings['drum_dynamics_amount'], minimum_db=settings['volume_minimum_db'])
                drum_dynamics.append(report)
            continue
        phrases = _merge(activity, gap=ppq * 4)
        analysis = stem.get('analysis') or {}
        sustained = max(0, min(1, _finite(analysis.get('sustained'))))
        density = max(0, _finite(analysis.get('onset_density')))
        melodic = role in _MELODIC
        kinds = []
        if melodic or role in {'vocals', 'hat', 'ride', 'perc', 'effects'}:
            kinds.append('filter')
        if melodic or role in _DRUMS | {'bass', 'effects'}:
            kinds.append('volume')
        if role in {'guitar', 'keys', 'brass', 'pads', 'synth', 'effects'}:
            kinds.append('pan')
        if role in {'guitar', 'keys', 'vocals', 'effects'}:
            kinds.append('delay')
        elif role in {'brass', 'pads', 'synth', 'snare', 'clap', 'tom', 'crash'}:
            kinds.append('reverb')
        for kind in kinds:
            neutral = 100 if kind == 'filter' else 0
            if kind == 'filter':
                # Busy acoustic parts keep more upper frequencies, while still
                # receiving a substantial range. Silence-driven dynamic range
                # never reduces this movement to a barely visible curve.
                strength = {'guitar': .85, 'keys': .94, 'brass': 1, 'vocals': .35,
                            'hat': .42, 'ride': .35, 'perc': .35}.get(role, 1)
                floor = settings['filter_minimum_percent'] + min(8, max(0, density - 3) * 1.5)
                def mapper(value, section, floor=floor, strength=strength):
                    maximum = settings['filter_maximum_percent']
                    return max(floor, min(100, maximum - (maximum - floor) * (1 - value) * strength * section['amount']))
                plugin, parameter, name = 'Fruity Filter', 'lowpass_cutoff_percent', 'High frequencies through each section'
                description = 'Moves the low-pass cutoff knob; percentages are control positions, not Hz.'
            elif kind == 'volume':
                depth = min(abs(settings['volume_minimum_db']), 3.5 if melodic else 2.5 if role in _DRUMS else 4.5)
                def mapper(value, section, depth=depth):
                    return max(settings['volume_minimum_db'], -depth * value * section['amount'])
                plugin, parameter, name = 'Mixer', 'volume_db', 'Volume through each section'
                description = 'Reductions from the saved mixer volume; the curve never raises the saved level.'
            elif kind == 'pan':
                direction = -1 if stem_index % 2 else 1
                def mapper(value, section, direction=direction):
                    return max(-40, min(40, value * settings['pan_maximum_percent'] * direction * section['amount']))
                plugin, parameter, name = 'Mixer', 'pan_percent', 'Slow left and right movement'
                description = 'Slow pan offsets from the saved pan position; this does not change stereo width.'
            else:
                maximum = settings['delay_maximum_percent' if kind == 'delay' else 'reverb_maximum_percent']
                maximum *= .5 if role in _DRUMS else 1
                def mapper(value, section, maximum=maximum):
                    return min(50, value * maximum * section['amount'])
                plugin = 'Fruity Delay 3' if kind == 'delay' else 'Fruity Reeverb 2'
                parameter = 'effect_blend_percent'
                name = 'Delay through each section' if kind == 'delay' else 'Reverberation through each section'
                description = 'Blend of the dedicated effect slot; 0% leaves the original sound. Internal effect settings stay fixed.'
            points = ([(0, 0), (song_end, 0)] if kind == 'volume' and role in {'kick', 'snare', 'clap', 'tom', 'crash'}
                      else _section_curves(rows, kind, mapper, phrase_ticks, song_end))
            cycles = 0
            if kind == 'volume':
                points, cycles = _rhythm(points, rows, role, sustained, flavor, settings, ppq, bpm, half_time, song_end, kick_onsets)
            points = _mask_curve(points, phrases, neutral, song_end, edge_ticks)
            drum_report = None
            if kind == 'volume' and role in _DRUMS:
                points, drum_report = apply_drum_dynamics(stem, points, rows,
                    ppq=ppq, bpm=bpm, song_end=song_end, activity=activity,
                    amount=settings['drum_dynamics_amount'], minimum_db=settings['volume_minimum_db'])
                drum_dynamics.append(drum_report)
                if drum_report['status'] in {'unmatched', 'skipped'}:
                    warnings.append(f"{stem['name']}: {drum_report['reason']}")
                if drum_report['status'] == 'planned':
                    name = 'Drum hit volume through the song'
                    description = 'Additional volume contrast follows matched note timing and accents, combined with section volume. The rendered audio already contains the original note dynamics.'
            if max(v for _, v in points) - min(v for _, v in points) < 1e-4:
                continue
            threshold = .25 if kind == 'filter' else .02 if kind == 'volume' else .1
            coverage = _covered(points, activity, 0, song_end, threshold)
            family = 'Drums' if role in _DRUMS else 'Melodic' if melodic else role.capitalize()
            move = {'stem_id': stem['id'], 'stem_name': stem['name'], 'role': role,
                    'family': family, 'mixer_insert': stem.get('mixer_insert'),
                    'plugin': plugin, 'parameter': parameter, 'units': 'dB' if kind == 'volume' else '%',
                    'name': name, 'control_description': description, 'shape': 'explicit',
                    'start_tick': 0, 'end_tick': song_end, 'points': points,
                    'start_value': points[0][1], 'end_value': points[-1][1],
                    'neutral_value': neutral, 'relative': kind in {'volume', 'pan'},
                    'section_start_tick': 0, 'section_type': 'full-song', 'section_bar': 1,
                    'intensity': intensity_override if intensity_override != 'Auto' else _intensity_label(max(s['amount'] for s in rows)),
                    'intensity_value': round(max(s['amount'] for s in rows), 4),
                    'control_family': kind, 'rhythm_cycles': cycles,
                    'rhythm_timing': 'Kick note timing' if kick_onsets is not None else 'Beat grid',
                    'release_minimum_ms': settings['rhythm_release_ms'],
                    'activity_source': activity_source, 'coverage': coverage,
                    'minimum_value': min(v for _, v in points), 'maximum_value': max(v for _, v in points)}
            moves.append(move)
            if drum_report is not None and drum_report['status'] == 'planned':
                move['drum_dynamics'] = True
                move['drum_hit_count'] = drum_report['hit_count']
                move['rhythm_timing'] = drum_report['guide']
    section_activity = []
    for section in rows:
        moving = []
        active_stems = set()
        for move in moves:
            coverage = _covered(move['points'], activity_by_stem[move['stem_id']], section['start_tick'], section['end_tick'], .25 if move['control_family'] == 'filter' else .02 if move['control_family'] == 'volume' else .1)
            if coverage['active_ticks']:
                active_stems.add(move['stem_id'])
            if coverage['moving_ticks']:
                moving.append({'stem_id': move['stem_id'], 'parameter': move['parameter'],
                               'plugin': move['plugin'], **coverage})
        section['summary'] = f'{_LABELS.get(section["type"], section["type"].capitalize())} at bar {section["bar"]}: section-length movement on {len(active_stems)} active stems.'
        section_activity.append({'start_tick': section['start_tick'], 'end_tick': section['end_tick'],
                                 'stem_count': len(active_stems), 'moving_targets': len(moving), 'targets': moving})
    if not moves:
        warnings.append('No active, supported stems were available for continuous movement.')
    return {'moves': moves, 'sections': rows, 'summary': [s['summary'] for s in rows],
            'warnings': warnings, 'half_time': bool(half_time), 'intensity_override': intensity_override,
            'ppq': ppq, 'beats_per_bar': 4, 'bpm': bpm, 'settings': settings, 'flavor': flavor,
            'baseline_intensity': baseline_intensity,
            'engine': 'continuous', 'planning_mode': 'continuous', 'development_preview': True,
            'drum_dynamics': drum_dynamics,
            'section_activity': section_activity,
            'movement_coverage': {'song_end_tick': song_end, 'section_activity': section_activity,
                                  'sections_with_movement': sum(s['moving_targets'] > 0 for s in section_activity),
                                  'total_sections': len(rows),
                                  'note': 'Coverage measures changing controls during detected activity; listening review is still required.'}}
