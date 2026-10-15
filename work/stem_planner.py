"""Editable style recipes and bounded, native-automation planning for audio stems.

This module is deliberately independent of FLP serialization. Its values are in
real units; mixer volume and pan values are offsets from the existing controls.
The writer must check its supported capabilities before presenting the final plan.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path


from app_paths import resource_path

PRESET_PATH = resource_path('stem-presets.json')
INTENSITIES = {'Low': .3, 'Medium-Low': .45, 'Medium': .6, 'Medium-High': .8, 'High': 1.0}
DRUMS = {'kick', 'snare', 'clap', 'tom', 'hat', 'perc', 'ride', 'crash', 'drums'}
MELODIC = {'brass', 'guitar', 'keys', 'pads', 'synth'}
ROLES = DRUMS | MELODIC | {'bass', 'vocals', 'effects'}
PARAMETERS = {
    'volume_db': ('dB', 0), 'pan_percent': ('%', 0),
    'lowpass_hz': ('Hz', 20000), 'highpass_hz': ('Hz', 20),
    'wet_percent': ('%', 0), 'band_gain_db': ('dB', 0),
    'width_percent': ('%', 100), 'duck_db': ('dB', 0),
    'lowpass_blend_percent': ('%', 0), 'highpass_blend_percent': ('%', 0),
    'effect_blend_percent': ('%', 0),
    'lowpass_cutoff_percent': ('%', 100),
}
BLEND_PARAMETERS = {'lowpass_blend_percent', 'highpass_blend_percent', 'effect_blend_percent'}
PLUGIN_PARAMETERS = {
    'Mixer': {'volume_db', 'pan_percent'},
    'Fruity Filter': {'lowpass_hz', 'highpass_hz', 'lowpass_blend_percent', 'highpass_blend_percent', 'lowpass_cutoff_percent'},
    'Fruity Love Philter': {'lowpass_hz', 'highpass_hz'},
    'Fruity Reeverb 2': {'wet_percent', 'effect_blend_percent'},
    'Fruity Delay 3': {'wet_percent', 'effect_blend_percent'},
    'Fruity Parametric EQ 2': {'band_gain_db'},
    'Fruity Stereo Shaper': {'width_percent'},
    'Fruity Stereo Enhancer': {'width_percent'},
    'Fruity Limiter': {'duck_db'},
}
SHAPES = {'ramp_up', 'ramp_down', 'hold', 'step', 'swell'}
SECTION_TYPES = {'intro', 'verse', 'pre-chorus', 'chorus', 'drop', 'bridge', 'outro'}
SECTION_LABELS = {'intro': 'Opening', 'verse': 'Verse', 'pre-chorus': 'Before chorus',
                  'chorus': 'Chorus', 'drop': 'Main section', 'bridge': 'Middle section',
                  'outro': 'Ending'}


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number.')
    return float(value)


def _baseline(value):
    if isinstance(value, str):
        for label, amount in INTENSITIES.items():
            if label.casefold() == value.casefold():
                return amount
        raise ValueError(f'Unknown intensity: {value}.')
    value = _number(value, 'baseline_intensity')
    if not 0 < value <= 1:
        raise ValueError('baseline_intensity must be greater than zero and at most 1.')
    return value


def validate_presets(document):
    """Reject unsafe or malformed edits before any project is changed."""
    if not isinstance(document, dict) or not document:
        raise ValueError('Presets must be a genre-to-subgenre object.')
    for genre, subgenres in document.items():
        if not isinstance(subgenres, dict) or not subgenres:
            raise ValueError(f'{genre}: provide at least one subgenre.')
        for subgenre, preset in subgenres.items():
            where = f'{genre} / {subgenre}'
            if not isinstance(preset, dict):
                raise ValueError(f'{where}: preset must be an object.')
            _baseline(preset.get('baseline_intensity'))
            if preset.get('engine'):
                if preset['engine']!='continuous':raise ValueError(f'{where}: unknown planning engine.')
                from stem_continuous import validate_settings
                validate_settings(preset.get('settings'))
                continue
            roles = preset.get('roles')
            if not isinstance(roles, dict) or not roles:
                raise ValueError(f'{where}: provide roles and move lists.')
            distinct = set()
            for role, moves in roles.items():
                if role not in ROLES | {'all'} or not isinstance(moves, list):
                    raise ValueError(f'{where}: invalid role or move list: {role}.')
                for move in moves:
                    if not isinstance(move, dict) or not isinstance(move.get('name'), str) or not move['name'].strip():
                        raise ValueError(f'{where}: every move needs a plain-English name.')
                    distinct.add(move['name'])
                    parameter, plugin = move.get('parameter'), move.get('plugin')
                    if plugin not in PLUGIN_PARAMETERS or parameter not in PLUGIN_PARAMETERS[plugin]:
                        raise ValueError(f'{where}: unsupported stock plugin/parameter recipe {plugin}/{parameter}.')
                    if move.get('units') != PARAMETERS[parameter][0] or move.get('shape') not in SHAPES:
                        raise ValueError(f'{where}: invalid real units or shape for {move["name"]}.')
                    for key in ('start_value', 'end_value', 'max_delta'):
                        _number(move.get(key), f'{where}: {key}')
                    if move['max_delta'] <= 0:
                        raise ValueError(f'{where}: max_delta must be positive.')
                    length = move.get('length', {})
                    if not isinstance(length, dict) or length.get('unit') not in {'beats', 'bars'} or _number(length.get('value'), 'length') <= 0:
                        raise ValueError(f'{where}: length requires positive value and beats or bars.')
                    repeat = move.get('repeat_every')
                    if repeat is not None and (not isinstance(repeat, dict) or repeat.get('unit') not in {'beats', 'bars'} or _number(repeat.get('value'), 'repeat_every') <= 0):
                        raise ValueError(f'{where}: repeat_every requires positive value and beats or bars.')
                    release = move.get('release')
                    if release is not None and (not isinstance(release, dict) or release.get('unit') not in {'beats', 'bars'} or _number(release.get('value'), 'release') < 0):
                        raise ValueError(f'{where}: release requires a nonnegative value and beats or bars.')
                    trigger = move.get('trigger', {})
                    if not isinstance(trigger, dict) or trigger.get('type') not in {'section', 'transition', 'last_bar', 'energy', 'phrase_end'}:
                        raise ValueError(f'{where}: invalid trigger for {move["name"]}.')
                    for key in ('sections', 'from', 'to'):
                        if key in trigger and (not isinstance(trigger[key], list) or not trigger[key] or any(t not in SECTION_TYPES for t in trigger[key])):
                            raise ValueError(f'{where}: trigger {key} must list section types.')
                    if trigger['type'] in {'section', 'last_bar'} and 'sections' not in trigger:
                        raise ValueError(f'{where}: {trigger["type"]} requires sections.')
                    if trigger['type'] == 'transition' and not ({'from', 'to'} & set(trigger)):
                        raise ValueError(f'{where}: transition requires from or to sections.')
                    if repeat is not None and trigger['type'] not in {'section', 'energy'}:
                        raise ValueError(f'{where}: only section or energy moves can repeat.')
                    if 'min_spacing_bars' in trigger and _number(trigger['min_spacing_bars'], 'min_spacing_bars') < 0:
                        raise ValueError(f'{where}: phrase spacing must not be negative.')
                    if 'max_per_section' in trigger and (type(trigger['max_per_section']) is not int or trigger['max_per_section'] <= 0):
                        raise ValueError(f'{where}: max_per_section must be a positive integer.')
                    if trigger['type'] == 'energy':
                        if trigger.get('comparison') not in {'above', 'below'} or not 0 <= _number(trigger.get('threshold'), 'energy threshold') <= 1:
                            raise ValueError(f'{where}: energy needs above/below and a threshold in 0..1.')
                    limits = move.get('limits', {})
                    if not isinstance(limits, dict):
                        raise ValueError(f'{where}: limits must be an object.')
                    for key, value in limits.items():
                        if key not in {'min', 'max', 'drum_min_hz', 'drum_max_hz', 'drum_max_percent'}:
                            raise ValueError(f'{where}: unknown limit {key}.')
                        _number(value, f'{where}: {key}')
                    if 'min' in limits and 'max' in limits and limits['min'] > limits['max']:
                        raise ValueError(f'{where}: minimum limit exceeds maximum.')
            if not 3 <= len(distinct) <= 6:
                raise ValueError(f'{where}: use 3 to 6 characteristic move names, found {len(distinct)}.')
    return document


def load_presets(path=None):
    """Load and validate the one user-editable genre recipe file."""
    with Path(path or PRESET_PATH).open(encoding='utf-8') as source:
        return validate_presets(json.load(source))


def role_family(role):
    return 'Drums' if role in DRUMS else 'Melodic' if role in MELODIC else {
        'bass': 'Bass', 'vocals': 'Vocals', 'effects': 'Effects',
    }.get(role, 'All Stems')


def _label(value):
    return min(INTENSITIES, key=lambda key: abs(INTENSITIES[key] - value))


def _section_type(section):
    value = str(section.get('type', section.get('name', 'verse'))).lower()
    return {'prechorus': 'pre-chorus', 'pre_chorus': 'pre-chorus', 'build': 'pre-chorus', 'chorus/drop': 'chorus'}.get(value, value)


def _matches(trigger, section, previous):
    kind, section_type = trigger['type'], _section_type(section)
    if kind in {'section', 'last_bar'}:
        return section_type in trigger['sections']
    if kind == 'phrase_end':
        return 'sections' not in trigger or section_type in trigger['sections']
    if kind == 'transition':
        return previous is not None and (
            'from' not in trigger or _section_type(previous) in trigger['from']
        ) and ('to' not in trigger or section_type in trigger['to'])
    energy = section.get('energy', 0)
    return energy > trigger['threshold'] if trigger['comparison'] == 'above' else energy < trigger['threshold']


def _stem_scale(stem):
    analysis = stem.get('analysis') or {}
    chunks = analysis.get('chunks') or []
    density = analysis.get('onset_density')
    if density is None:
        density = sum(c.get('onset_density', 0) for c in chunks) / max(1, len(chunks))
    dynamic = analysis.get('active_dynamic_range_db')
    if dynamic is None:
        dynamic = max(0, float(analysis.get('dynamic_range_db', 0)))
        if dynamic > 40 and chunks:
            # Whole-song silence is arrangement sparsity, not extreme playing
            # dynamics. Older cached analysis includes it in the 10th percentile.
            rms = [float(c.get('rms', 0)) for c in chunks]
            peak = max(rms, default=0)
            active_db = sorted(20 * math.log10(v) for v in rms if v > max(1e-5, peak * .015))
            if active_db:
                lo, hi = int((len(active_db) - 1) * .1), int((len(active_db) - 1) * .9)
                dynamic = active_db[hi] - active_db[lo]
    dynamic = max(0, float(dynamic))
    factor = 1 / (1 + max(0, dynamic - 12) / 45 + max(0, float(density) - 2) / 12)
    sustained = float(analysis.get('sustained', 0))
    if stem['role'] in {'pads', 'brass', 'keys'} or sustained >= .65:
        factor *= 1.08
    return max(.5, min(1.1, factor))


def _active_during(stem, start, end, ppq):
    """Avoid gestures wholly inside known rests; unknown activity is allowed."""
    if 'note_spans' in stem:
        return any(a < end and b > start for a, b in stem['note_spans'])
    analysis = stem.get('analysis') or {}
    spans, bpm = analysis.get('activity_segments'), analysis.get('bpm')
    if spans is None or not bpm:
        return True
    # inspect_stems exposes offset_tick as the Playlist placement, not a trim.
    offset = stem.get('offset_tick', stem.get('position', 0))
    factor = float(bpm) * ppq / 60
    return any(span['start_seconds'] * factor + offset < end
               and span['end_seconds'] * factor + offset > start for span in spans)


def _bounded_value(value, move, factor, role):
    parameter = move['parameter']
    neutral = PARAMETERS[parameter][1]
    if parameter.endswith('_hz'):
        value = neutral * (max(1, value) / neutral) ** factor
    else:
        value = neutral + (value - neutral) * factor
    cap = move['max_delta']
    if parameter == 'volume_db':
        cap = min(12, cap)
    value = max(neutral - cap, min(neutral + cap, value))
    limits = move.get('limits', {})
    lower = limits.get('min', -math.inf)
    upper = limits.get('max', math.inf)
    if parameter.endswith('_hz'):
        lower, upper = max(20, lower), min(20000, upper)
        if role in DRUMS:
            if parameter == 'lowpass_hz':
                lower = max(lower, limits.get('drum_min_hz', 1800))
            else:
                upper = min(upper, limits.get('drum_max_hz', 60 if role == 'kick' else 350))
    elif parameter == 'volume_db':
        lower, upper = max(-12, lower), min(3, upper)
    elif parameter in BLEND_PARAMETERS:
        lower, upper = max(0, lower), min(100, upper)
        if role in DRUMS:
            default_cap = 25 if parameter == 'highpass_blend_percent' else 45
            upper = min(upper, limits.get('drum_max_percent', default_cap), default_cap)
    elif parameter in {'wet_percent', 'width_percent'}:
        lower, upper = max(0, lower), min(100 if parameter == 'wet_percent' else 200, upper)
    elif parameter == 'pan_percent':
        lower, upper = max(-100, lower), min(100, upper)
    if lower > upper:
        raise ValueError(f'{move["name"]}: configured limits contradict the safety limits for {role}.')
    return round(max(lower, min(upper, value)), 6)


def plan_moves(stems, sections, preset, *, ppq, beats_per_bar=4, half_time=False, intensity_override='Auto', movement_mode=None):
    """Build a previewable plan; no files, audio, or plugin instances are modified.

    ``sections`` use project ticks and normalized 0..1 energy. ``stems`` carry
    confirmed roles and optional analysis summaries. Half-time doubles musical
    move durations while section labels remain in the project's actual bars.
    Explicit full_song movement uses the shared editable continuous settings,
    independently of genre. None retains the selected preset's original engine.
    """
    if ppq <= 0 or beats_per_bar <= 0:
        raise ValueError('Positive project PPQ and beats per bar are required.')
    if movement_mode not in (None, 'full_song', 'brief', 'volume_balance'):
        raise ValueError('Movement mode must be volume_balance, full_song or brief, or omitted to use the selected preset.')
    if movement_mode == 'volume_balance':
        # The user's existing mix and measured stem audio supply the reference.
        # Genre effect recipes do not define volume-balance behavior.
        from stem_balance import plan_balance
        return plan_balance(stems, sections, ppq, half_time=half_time,
                            intensity_override=intensity_override)
    validate_presets({'Selected genre': {'Selected subgenre': preset}})
    if movement_mode == 'full_song':
        try:
            preset = load_presets()['EDM']['Full song movement']
        except KeyError:
            raise ValueError('Shared full-song settings are missing. Restore EDM / Full song movement in resources/stem-presets.json.') from None
        if preset.get('engine') != 'continuous':
            raise ValueError('Shared EDM / Full song movement settings must use the continuous engine.')
    elif movement_mode == 'brief' and preset.get('engine') == 'continuous':
        raise ValueError('Brief section changes require a style with brief recipes. Choose another style instead of Full song movement.')
    if preset.get('engine')=='continuous':
        from stem_continuous import plan_continuous
        plan = plan_continuous(stems,sections,ppq,half_time=half_time,
                              intensity_override=intensity_override,settings=preset.get('settings'),
                              baseline_intensity=preset['baseline_intensity'])
        plan['movement_mode'] = 'full_song'
        return plan
    base = _baseline(preset['baseline_intensity'])
    auto = str(intensity_override).casefold() == 'auto'
    override = None if auto else _baseline(intensity_override)
    for stem in stems:
        if stem.get('role') not in ROLES:
            raise ValueError(f'Confirm a role for {stem.get("name", "Unnamed stem")} before planning.')
    section_rows = []
    prior_end = 0
    for section in sorted(sections, key=lambda s: s.get('start_tick', s.get('start', 0))):
        section = dict(section)
        section['start_tick'] = int(section.get('start_tick', section.get('start', 0)))
        section['end_tick'] = int(section.get('end_tick', section.get('end', 0)))
        if section['start_tick'] < prior_end or section['end_tick'] <= section['start_tick']:
            raise ValueError('Sections must have non-overlapping, increasing positive durations.')
        section['energy'] = max(0, min(1, _number(section.get('energy', 0), 'section energy')))
        prior_end = section['end_tick']
        section_rows.append(section)
    beat_ticks = ppq * (2 if half_time else 1)
    bar_ticks = beat_ticks * beats_per_bar
    moves, summaries, warnings = [], [], []
    for index, section in enumerate(section_rows):
        previous = section_rows[index - 1] if index else None
        jump = abs(section['energy'] - previous['energy']) if previous else 0
        section_scale = override if override is not None else min(1, base * (.8 + .5 * jump + .1 * section['energy']))
        section_type = _section_type(section)
        bar = 1 + section['start_tick'] // (ppq * beats_per_bar)
        summary = {**section, 'type': section_type, 'bar': bar,
                   'intensity': _label(section_scale), 'intensity_value': round(section_scale, 4)}
        summaries.append(summary)
        for stem in stems:
            role = stem['role']
            recipes = preset['roles'].get('all', []) + preset['roles'].get(role, [])
            for recipe in recipes:
                trigger = recipe['trigger']
                if not _matches(trigger, section, previous):
                    continue
                length = recipe['length']
                duration = max(1, round(length['value'] * (bar_ticks if length['unit'] == 'bars' else beat_ticks)))
                if trigger['type'] == 'phrase_end':
                    endings = stem.get('phrase_ends')
                    if not endings:
                        warning = f'{stem["name"]}: skipped {recipe["name"]}; no matched MIDI phrase endings are available.'
                        if warning not in warnings:
                            warnings.append(warning)
                        continue
                    windows = [(max(section['start_tick'], int(tick) - duration), int(tick))
                               for tick in sorted(set(endings))
                               if section['start_tick'] < tick <= section['end_tick']]
                    if not windows:
                        continue
                    spacing = round(trigger.get('min_spacing_bars', 0) * bar_ticks)
                    if spacing or trigger.get('max_per_section'):
                        selected = []
                        for window in reversed(windows):
                            if not selected or selected[-1][1] - window[1] >= spacing:
                                selected.append(window)
                            if len(selected) >= trigger.get('max_per_section', len(windows)):
                                break
                        windows = list(reversed(selected))
                    # Nearby phrases cannot drive two conflicting instances of
                    # the same parameter at the same moment.
                    windows = [(max(start, windows[i - 1][1] if i else start), end)
                               for i, (start, end) in enumerate(windows)]
                    start, end = windows[0]
                elif trigger['type'] in {'transition', 'last_bar'}:
                    # Never reach before the preceding section or the song.
                    if previous is None:
                        continue
                    end = section['start_tick']
                    start = max(previous['start_tick'], end - duration)
                    windows = [(start, end)]
                else:
                    start = section['start_tick']
                    end = min(section['end_tick'], start + duration)
                    windows = [(start, end)]
                repeat = recipe.get('repeat_every')
                if repeat:
                    stride = max(1, round(repeat['value'] * (bar_ticks if repeat['unit'] == 'bars' else beat_ticks)))
                    if stride < duration:
                        raise ValueError(f'{recipe["name"]}: repeats must not overlap their own duration.')
                    if (section['end_tick'] - start) // stride > 20000:
                        raise ValueError('This preset would create more than 20,000 moves; increase repeat spacing.')
                    windows = [(position, min(section['end_tick'], position + duration))
                               for position in range(start, section['end_tick'], stride)]
                windows = [(a, b) for a, b in windows if b > a and _active_during(stem, a, b, ppq)]
                if not windows:
                    continue
                start, end = windows[0]
                factor = min(1, section_scale * _stem_scale(stem))
                a = _bounded_value(recipe['start_value'], recipe, factor, role)
                b = _bounded_value(recipe['end_value'], recipe, factor, role)
                neutral = PARAMETERS[recipe['parameter']][1]
                if a == b == neutral:
                    continue
                move = deepcopy(recipe)
                move.update(stem_id=stem['id'], stem_name=stem['name'], role=role,
                            family=role_family(role), mixer_insert=stem.get('mixer_insert'),
                            start_tick=start, end_tick=end, start_value=a, end_value=b,
                            neutral_value=neutral, section_start_tick=section['start_tick'],
                            section_type=section_type, section_bar=bar,
                            intensity=_label(factor), intensity_value=round(factor, 4),
                            relative=recipe['parameter'] in {'volume_db', 'pan_percent'})
                moves.append(move)
                for occurrence_start, occurrence_end in windows[1:]:
                    moves.append({**deepcopy(move), 'start_tick': occurrence_start,
                                  'end_tick': occurrence_end})
                if len(moves) > 20000:
                    raise ValueError('This preset would create more than 20,000 moves; increase repeat spacing.')
    # Explicit effect-slot blends fade back to dry after their main gesture.
    # Clamp releases before the next gesture targeting that same native control.
    by_target = {}
    for move in moves:
        by_target.setdefault((move['stem_id'], move['plugin'], move['parameter']), []).append(move)
    song_end = section_rows[-1]['end_tick'] if section_rows else 0
    for target_moves in by_target.values():
        target_moves.sort(key=lambda move: (move['start_tick'], move['end_tick']))
        for index, move in enumerate(target_moves):
            if move['parameter'] not in BLEND_PARAMETERS:
                continue
            release = move.get('release', {'value': .25 if move['plugin'] == 'Fruity Filter' else 1, 'unit': 'beats'})
            requested = round(release['value'] * ppq * (beats_per_bar if release['unit'] == 'bars' else 1))
            available_end = target_moves[index + 1]['start_tick'] if index + 1 < len(target_moves) else song_end
            move['release_ticks'] = max(0, min(requested, song_end - move['end_tick'], available_end - move['end_tick']))
            move['release_end_tick'] = move['end_tick'] + move['release_ticks']
            move['control_description'] = 'Amount of the added effect heard; 0% leaves the original sound. The effect settings remain fixed.'
    if not section_rows:
        warnings.append('No audio sections were available; no automation was planned.')
    elif not moves:
        warnings.append('No characteristic moves match the confirmed roles and detected sections.')
    for section in summaries:
        applicable = [m for m in moves if m['section_start_tick'] == section['start_tick']]
        names = list(dict.fromkeys(m['name'] for m in applicable))
        section['summary'] = f'{SECTION_LABELS.get(section["type"],section["type"].capitalize())} at bar {section["bar"]}: {section["intensity"]}' + (': ' + '; '.join(names) if names else ': no matching changes')
    return {'sections': summaries, 'moves': moves, 'summary': [s['summary'] for s in summaries],
            'warnings': warnings, 'half_time': bool(half_time), 'intensity_override': intensity_override,
            'ppq': ppq, 'beats_per_bar': beats_per_bar, 'movement_mode': 'brief'}


def _same_envelope(a, b, per_link_ranges):
    if a['start_tick'] != b['start_tick'] or a['end_tick'] != b['end_tick'] or a.get('release_ticks', 0) != b.get('release_ticks', 0):
        # A one-beat tolerance is necessary but not sufficient: rounding timing
        # changes the intended move, even if links can remap the value range.
        return False
    if a.get('points') != b.get('points'):
        return False
    if per_link_ranges:
        return (a['end_value'] >= a['start_value']) == (b['end_value'] >= b['start_value'])
    return all(a.get(key) == b.get(key) for key in ('start_value', 'end_value', 'neutral_value', 'relative'))


def group_moves(moves, ppq, per_link_ranges=False):
    """Return clips containing target move records, preserving intended curves.

    The default requires exact real-unit envelopes. A writer must also compare
    target-normalized curves (especially differing mixer faders) before linking.
    Only a writer with verified per-target mappings may enable range grouping.
    """
    if ppq <= 0:
        raise ValueError('Grouping needs positive project PPQ.')
    groups, buckets = [], {}
    for move in sorted(moves, key=lambda m: (m['start_tick'], m.get('family', ''), str(m['stem_id']))):
        key = tuple(move.get(k) for k in ('family', 'parameter', 'units', 'shape', 'section_start_tick', 'start_tick', 'end_tick'))
        bucket = buckets.setdefault(key, [])
        candidate = None
        for group in bucket:
            first = group['moves'][0]
            if any(first.get(k) != move.get(k) for k in ('family', 'parameter', 'units', 'shape', 'section_start_tick')):
                continue
            if abs(first['start_tick'] - move['start_tick']) > ppq or abs(first['end_tick'] - move['end_tick']) > ppq:
                continue
            if not _same_envelope(first, move, per_link_ranges):
                continue
            if any(m['stem_id'] == move['stem_id'] and m['plugin'] == move['plugin'] for m in group['moves']):
                continue
            candidate = group
            break
        if candidate is None:
            candidate = {**deepcopy(move), 'moves': [], 'targets': []}
            groups.append(candidate)
            bucket.append(candidate)
        candidate['moves'].append(deepcopy(move))
        candidate['targets'].append({'stem_id': move['stem_id'], 'mixer_insert': move.get('mixer_insert'),
                                     'plugin': move['plugin'], 'parameter': move['parameter'],
                                     'start_value': move['start_value'], 'end_value': move['end_value']})
    for index, group in enumerate(groups):
        first = group['moves'][0]
        prefix = first.get('family', 'All Stems') if len(group['moves']) > 1 else first['stem_name']
        group['id'] = index
        group['clip_name'] = f'{prefix} - {first["name"]}'
        group['grouped'] = len(group['moves']) > 1
        group['per_link_ranges'] = bool(per_link_ranges and group['grouped'])
    return groups
