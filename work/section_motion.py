"""Deterministic, sparse section automation in explicit relative units.

This module deliberately knows nothing about plugin parameter IDs. The exporter
maps effect depths to calibrated plugin bounds and native mix movements to the
saved mixer levels. Volume is reduction-only dB; pan is a signed percentage of
one full side. No per-note ducking, LFOs, or fixed genre templates are generated.
All temporal decisions precede the application of seeded global amounts.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
import re

from section_structure import shared_drum_members


FAMILIES = ('filter', 'delay', 'reverb', 'width', 'volume', 'pan')
_AUTO_RANGES = {'filter': (55, 80), 'delay': (40, 65),
                'reverb': (35, 60), 'width': (45, 75),
                'volume': (55, 75), 'pan': (30, 50)}
_AUDIO_ONLY = {'vocals', 'effects'}
_STABLE = {'kick', 'bass', 'sub', 'snare', 'clap', 'crash', 'tom'}
_PERCUSSION = {'hat', 'perc', 'ride', 'drums'}
_PAN_ROLES = _PERCUSSION | {'pads', 'chords', 'keys', 'guitar', 'brass', 'synth', 'effects'}


def default_settings(seed=0):
    """Return independent automatic family amounts; numeric overrides stay fixed."""
    return validate_settings({'seed': seed})


def _number(value, name, low=None, high=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (low is not None and value < low)
            or (high is not None and value > high)):
        raise ValueError(f'{name} must be a finite number'
                         + (f' from {low} to {high}.' if high is not None else '.'))
    return value


def validate_settings(settings=None):
    """Validate and copy settings. ``auto`` is resolved only after validation.

    ``reverb_mode='baked'`` suppresses additional reverb regardless of its slider.
    ``add`` opts into a new reverb. Changing seed affects only auto amounts.
    """
    settings = {} if settings is None else settings
    if not isinstance(settings, Mapping):
        raise ValueError('Section settings must be a mapping.')
    unknown = set(settings) - {'seed', 'intensities', 'enabled', 'reverb_mode'}
    if unknown:
        raise ValueError('Unknown section settings: ' + ', '.join(sorted(unknown)))
    seed = settings.get('seed', 0)
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError('Seed must be an integer from 0 to 9223372036854775807.')
    result = {'seed': seed, 'intensities': {}, 'enabled': {},
              'reverb_mode': settings.get('reverb_mode', 'baked')}
    if result['reverb_mode'] not in ('baked', 'add'):
        raise ValueError('Reverb mode must be baked or add.')
    for key in ('intensities', 'enabled'):
        supplied = settings.get(key, {})
        if not isinstance(supplied, Mapping) or set(supplied) - set(FAMILIES):
            raise ValueError(f"{key} must contain only {', '.join(FAMILIES)}.")
        for family in FAMILIES:
            value = supplied.get(family, 'auto' if key == 'intensities' else True)
            if key == 'enabled':
                if type(value) is not bool:
                    raise ValueError(f'{family} enabled must be true or false.')
            elif value != 'auto':
                _number(value, f'{family} intensity', 0, 100)
            result[key][family] = value
    return result


def resolve_intensities(settings=None):
    """Return the effective 0..100 amounts, suitable for a GUI preview."""
    settings = validate_settings(settings)
    resolved = {}
    for family in FAMILIES:
        value = settings['intensities'][family]
        if value == 'auto':
            digest = hashlib.sha256(f"section-motion-v1:{settings['seed']}:{family}".encode()).digest()
            low, high = _AUTO_RANGES[family]
            value = low + int.from_bytes(digest[:8], 'big') % (high - low + 1)
        if (not settings['enabled'][family]
                or (family == 'reverb' and settings['reverb_mode'] == 'baked')):
            value = 0
        resolved[family] = value
    return resolved


def _tick(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    return value


def _role(stem):
    raw = str(stem.get('role') or '').strip().lower()
    aliases = {'vocal': 'vocals', 'voice': 'vocals', 'fx': 'effects',
               'effect': 'effects', 'pad': 'pads', 'hihat': 'hat',
               'hi-hat': 'hat', 'percussion': 'perc', '808': 'sub',
               'sub bass': 'sub', 'sub_bass': 'sub', 'drum_bus': 'drums'}
    if raw:
        return aliases.get(raw, raw)
    # Missing labels are only used for obvious protection/exemption; all other
    # instruments remain generic and require a verified MIDI guide.
    words = set(re.findall(r'[a-z0-9]+', str(stem.get('name', '')).lower()))
    for words_to_match, role in (({'kick'}, 'kick'), ({'808', 'sub', 'bass'}, 'sub'),
                                  ({'vocal', 'vocals', 'voice'}, 'vocals'),
                                  ({'fx', 'effects'}, 'effects')):
        if words & words_to_match:
            return role
    return 'instrument'


def _merge(spans, gap=0):
    result = []
    for start, stop in sorted(spans):
        if stop <= start:
            continue
        if result and start <= result[-1][1] + gap:
            result[-1][1] = max(stop, result[-1][1])
        else:
            result.append([start, stop])
    return result


def _audio_evidence(stem, ppq, bpm, end):
    data = stem.get('analysis')
    if not isinstance(data, Mapping):
        raise ValueError(f"{stem['name']}: analyzed audio is required.")
    offset = _tick(stem.get('offset_tick', stem.get('position', 0)), 'Stem offset')
    scale = bpm * ppq / 60
    levels = data.get('level_windows', data.get('chunks', []))
    if not isinstance(levels, (list, tuple)):
        raise ValueError(f"{stem['name']}: audio windows must be a list.")
    measured = []
    for row in levels:
        start = _number(row.get('start_seconds'), 'Audio window start', 0)
        stop = _number(row.get('end_seconds'), 'Audio window end', 0)
        rms = _number(row.get('rms'), 'Audio window RMS', 0)
        peak = _number(row.get('peak', rms), 'Audio window peak', 0)
        if stop <= start:
            raise ValueError(f"{stem['name']}: audio window has no duration.")
        measured.append((max(0, round(offset + start * scale)),
                         min(end, round(offset + stop * scale)), rms, peak))
    # Empty activity is insufficient to certify silence on its own. The input
    # adapter can explicitly certify it after inspecting all source samples.
    duration = stem.get('duration')
    if duration is None:
        duration = (round(_number(data['duration_seconds'], 'Audio duration', 0) * scale)
                    if 'duration_seconds' in data else end - offset)
    duration = _tick(duration, 'Stem duration')
    expected_stop = min(end, offset + duration)
    covered = _merge([(a, b) for a, b, _, _ in measured])
    measured_silence = (expected_stop > offset
                        and _overlap(covered, offset, expected_stop) == expected_stop - offset
                        and all(rms == 0 and peak == 0 for _, _, rms, peak in measured))
    silent = bool(stem.get('silent') is True or data.get('silent') is True or measured_silence)
    spans = []
    if 'activity_segments' in data:
        for row in data['activity_segments']:
            start = _number(row.get('start_seconds'), 'Audio activity start', 0)
            stop = _number(row.get('end_seconds'), 'Audio activity end', 0)
            if stop <= start:
                raise ValueError(f"{stem['name']}: audio activity has no duration.")
            spans.append((max(0, round(offset + start * scale)), min(end, round(offset + stop * scale))))
    elif measured:
        threshold = max(1e-7, max(row[2] for row in measured) * .015)
        spans = [(a, b) for a, b, rms, peak in measured if max(rms, peak) > threshold]
    elif not silent:
        raise ValueError(f"{stem['name']}: audio activity or level windows are required.")
    return _merge(spans), measured, silent


def _notes(stem, analysis, end):
    guide = analysis.get('midi_guide') or {}
    matches = guide.get('matches', {}) if isinstance(guide, Mapping) else {}
    matched = matches.get(stem['id'], matches.get(str(stem['id'])))
    if isinstance(matched, Mapping):
        source = matched
        verified = True
    else:
        source = stem
        verified = (stem.get('midi_verified') is True
                    or ('midi_verified' not in stem
                        and (stem.get('timing_provenance') == 'midi'
                             or (bool(stem.get('midi_track'))
                                 and str(stem.get('note_guide_source', '')).startswith('MIDI:')))))
    notes = []
    for row in source.get('note_events', []):
        start = _tick(row.get('tick'), f"{stem['name']} note start")
        length = _tick(row.get('duration'), f"{stem['name']} note duration")
        if start >= end or start + length > end:
            raise ValueError(f"{stem['name']}: MIDI extends beyond the analyzed song.")
        notes.append((start, start + length))
    return sorted(notes), bool(verified and notes)


def _overlap(spans, start, stop):
    return sum(max(0, min(b, stop) - max(a, start)) for a, b in spans)


def _sections(analysis, end, stems):
    result = []
    previous = 0
    for index, source in enumerate(analysis.get('sections', [])):
        row = dict(source)
        start = _tick(row.get('start_tick'), 'Section start')
        stop = _tick(row.get('end_tick'), 'Section end', 1)
        if start != previous or stop <= start or stop > end:
            raise ValueError('Sections must be ordered, contiguous, and inside the song.')
        previous = stop
        if 'energy' in row:
            row['energy'] = float(_number(row['energy'], 'Section energy', 0, 1))
        else:
            # A section may be an arbitrary user-named marker. Measure its
            # average audio energy; its name never imposes a genre convention.
            row['energy'] = math.sqrt(sum(
                rms * rms * max(0, min(b, stop) - max(a, start)) / (stop - start)
                for stem in stems for a, b, rms, _ in stem['_levels']))
        row['active_stem_ids'] = [stem['id'] for stem in stems
                                  if _overlap(stem['_activity'], start, stop)]
        result.append(row)
    if result and previous != end:
        raise ValueError('Sections must cover the complete analyzed song.')
    if result and not all('energy' in s for s in analysis['sections']):
        highest = max(s['energy'] for s in result)
        if highest:
            for row in result:
                row['energy'] /= highest
    return result


def _simplify(points):
    """Remove redundant geometry before scaling, so seed cannot change timing."""
    clean = []
    for tick, value in sorted(dict(points).items()):
        clean.append((tick, float(value)))
        while len(clean) >= 3:
            a, b, c = clean[-3:]
            if abs((b[1] - a[1]) * (c[0] - b[0])
                   - (c[1] - b[1]) * (b[0] - a[0])) > 1e-9:
                break
            clean.pop(-2)
    return clean


def _value_at(points, tick):
    """Evaluate a straight-line envelope before adding its activity guards."""
    if tick <= points[0][0]:
        return points[0][1]
    for (a, first), (b, last) in zip(points, points[1:]):
        if tick <= b:
            return first + (last - first) * (tick - a) / (b - a)
    return points[-1][1]


def _section_curve(targets, sections, bar, end, activity):
    """Ramp between section targets over the section, with audible entry/exit.

    Equal section targets remain holds. Short phrase gaps are bridged instead
    of creating repeated dips, while longer silences stay at baseline. The
    guards lie *inside* each audible region, so a ramp does not arrive only
    after the sound that should carry it has stopped.
    """
    if not any(targets):
        return []
    core = [(section['start_tick'], target)
            for section, target in zip(sections, targets)]
    core.append((end, targets[-1]))
    points = [(0, 0.), (end, 0.)]
    for start, stop in _merge(activity, gap=bar):
        start, stop = max(0, start), min(end, stop)
        if stop - start < 2:
            continue
        guard = max(1, min(bar // 4, (stop - start) // 4))
        attack, release = start + guard, stop - guard
        points.append((start, 0.))
        if attack >= release:
            points.append((attack, _value_at(core, attack)))
        else:
            # Preserve the intended endpoint depths while moving the
            # baseline transitions into the audible phrase by one beat.
            points.append((attack, _value_at(core, start)))
            points.extend((tick, value) for tick, value in core
                          if attack < tick < release)
            points.append((release, _value_at(core, stop)))
        points.append((stop, 0.))
    return _simplify(points)


def _delay_curve(stem, sections, ppq, end):
    bar = ppq * 4
    spans = stem['_notes'] if stem['_verified'] else stem['_activity']
    # Group nearby notes/words; never turn each note into a throw.
    phrases = _merge(spans, gap=ppq)
    points = [(0, 0.), (end, 0.)]
    previous_release = 0
    for section in sections[:-1]:
        start, boundary = section['start_tick'], section['end_tick']
        if boundary - start < 2 * bar:
            continue
        candidates = []
        for index, (a, b) in enumerate(phrases):
            following = phrases[index + 1][0] if index + 1 < len(phrases) else end
            if (max(start, boundary - 2 * bar) < b <= boundary
                    and following - b >= 2 * ppq and b - a >= ppq
                    and _overlap(stem['_activity'], max(a, b - bar), b) > 0):
                candidates.append((a, b, following))
        if not candidates:
            continue
        a, b, following = candidates[-1]
        attack = max(start, a, b - bar, previous_release)
        peak = max(attack + 1, b - ppq // 2)
        hold = min(following, b + ppq)
        release = min(end, following, b + bar)
        if not attack < peak <= hold < release:
            continue
        points.extend([(attack, 0.), (peak, 1.), (hold, 1.), (release, 0.)])
        previous_release = release
    return _simplify(points) if len(points) > 2 else []


def _active_section_rms(stem, section):
    """Measure the stem while it plays; gaps must not masquerade as quiet audio."""
    start, stop = section['start_tick'], section['end_tick']
    weighted = []
    for a, b, rms, _ in stem['_levels']:
        if b <= start or a >= stop or rms <= 0:
            continue
        overlap = _overlap(stem['_activity'], max(a, start), min(b, stop))
        if overlap > 0:
            weighted.append((rms, overlap))
    if not weighted:
        return None
    # Scale before squaring so any accepted finite RMS remains finite.
    scale = max(rms for rms, _ in weighted)
    return scale * math.sqrt(sum((rms / scale) ** 2 * duration for rms, duration in weighted)
                             / sum(duration for _, duration in weighted))


def _mix_curve(targets, sections, bar, end, activity):
    """Slow section lines with sparse, audible returns to the saved mix.

    Native faders need gentler entry/exit than effect throws. A brief isolated
    sound does not get its own fader or pan swell; dense hits and phrases are
    first joined across gaps of at most one bar.
    """
    if not any(targets):
        return []
    core = [(section['start_tick'], target) for section, target in zip(sections, targets)]
    core.append((end, targets[-1]))
    points = [(0, 0.), (end, 0.)]
    for start, stop in activity:
        guard = max(1, min(2 * bar, (stop - start) // 4))
        attack, release = start + guard, stop - guard
        points.extend([(start, 0.), (attack, _value_at(core, start))])
        points.extend((tick, value) for tick, value in core if attack < tick < release)
        points.extend([(release, _value_at(core, stop)), (stop, 0.)])
    return _simplify(points)


def _mix_candidates(stem, sections, bar, end):
    activity = [span for span in _merge(stem['_activity'], gap=bar)
                if span[1] - span[0] >= 2 * bar]
    if not activity:
        return {}
    active = [_overlap(activity, s['start_tick'], s['end_tick']) > 0 for s in sections]
    energies = [s['energy'] for s in sections]
    audible = [energy for energy, on in zip(energies, active) if on]
    low, high = min(audible), max(audible)
    contrast = high - low
    relative = [(energy - low) / contrast if contrast >= .12 else 1. for energy in energies]
    candidates = {}
    rms = [_active_section_rms(stem, section) if on else None
           for section, on in zip(sections, active)]
    measured = sorted((20 * math.log10(level), _overlap(stem['_activity'],
                       section['start_tick'], section['end_tick']))
                      for section, level in zip(sections, rms) if level is not None)
    if measured:
        # A duration-weighted quiet reference resists a tiny quiet fragment
        # pulling the whole song down. Half of an excess above 0.75 dB is ridden
        # down, while a smaller section-energy term preserves arrangement shape.
        threshold = sum(duration for _, duration in measured) * .25
        elapsed = 0
        quiet_db = measured[-1][0]
        for level_db, duration in measured:
            elapsed += duration
            if elapsed >= threshold:
                quiet_db = level_db
                break
        gentle = stem['_role'] == 'vocals' or stem['_role'] in _PERCUSSION
        limit = 1.5 if gentle else 3.
        arrangement_depth = .2 if stem['_role'] == 'vocals' else .35
        targets = []
        for level, energy, on in zip(rms, relative, active):
            if not on or level is None:
                targets.append(0.)
                continue
            excess_db = max(0., 20 * math.log10(level) - quiet_db - .75)
            shape = limit * arrangement_depth * max(0., min(1., 1 - energy))
            targets.append(-min(limit, .5 * excess_db + shape))
        candidates['volume'] = (
            _mix_curve(targets, sections, bar, end, activity),
            'Ride louder active sections down gently and ease sparse passages below the saved level.',
            'volume_db')
    name_words = set(re.findall(r'[a-z0-9]+', stem['name'].lower()))
    if (stem['_role'] in _PAN_ROLES and not name_words & {'lead', 'main', 'solo'}
            and contrast >= .12):
        # Identity controls the side; seed only changes the global amount. No
        # periodic sweep or random direction changes are imposed on the music.
        identity = f"section-pan-v1:{type(stem['id']).__name__}:{stem['id']}:{stem['name']}"
        digest = hashlib.sha256(identity.encode()).digest()
        direction = 1 if digest[0] & 1 else -1
        targets = [direction * 8 * (.25 + .75 * max(0., min(1., energy))) if on else 0.
                   for energy, on in zip(relative, active)]
        candidates['pan'] = (
            _mix_curve(targets, sections, bar, end, activity),
            'Gently place this supporting part to one side, with slow section-sized movement.',
            'pan_percent')
    return candidates


def plan_section_motion(analysis, settings=None):
    """Plan <=2 effect curves plus independent volume and pan lines per stem.

    Musical and drum stems require verified exported MIDI, even when a protected
    low-end role will receive no movement. Vocals/effects permit audio timing.
    Validation errors list every missing MIDI match before generating curves.
    """
    settings = validate_settings(settings)
    if not isinstance(analysis, Mapping):
        raise ValueError('Analysis must be a mapping.')
    ppq = _tick(analysis.get('ppq'), 'PPQ', 1)
    end = _tick(analysis.get('end_tick'), 'Song end tick', 2)
    bpm = _number(analysis.get('bpm', analysis.get('tempo')), 'Tempo', 1, 999)
    if tuple(analysis.get('time_signature', (4, 4))) != (4, 4) or analysis.get('tempo_changes'):
        raise ValueError('Section motion requires constant-tempo 4/4 audio and MIDI.')
    stems = []
    missing = []
    shared_members = shared_drum_members(analysis.get('shared_drum_guide'), analysis.get('stems', []), end)
    seen_ids, seen_inserts = set(), set()
    for source in analysis.get('stems', []):
        stem = dict(source)
        stem['name'] = str(stem.get('name') or stem.get('id', 'Unnamed stem'))
        if 'automation_protected' in stem and type(stem['automation_protected']) is not bool:
            raise ValueError(f"{stem['name']}: automation_protected must be true or false.")
        identifier = stem.get('id')
        if type(identifier) not in (str, int) or identifier in seen_ids:
            raise ValueError('Each stem needs a unique string or integer ID.')
        insert = _tick(stem.get('mixer_insert'), f"{stem['name']} mixer insert", 1)
        if insert in seen_inserts:
            raise ValueError('Each stem needs a dedicated mixer insert.')
        seen_ids.add(identifier)
        seen_inserts.add(insert)
        activity, levels, silent = _audio_evidence(stem, ppq, bpm, end)
        notes, verified = ([], False) if silent else _notes(stem, analysis, end)
        role = _role(stem)
        if not silent and role not in _AUDIO_ONLY and not verified and identifier not in shared_members:
            missing.append(stem['name'])
        stem.update(_role=role, _activity=activity, _levels=levels, _silent=silent,
                    _notes=notes, _verified=verified)
        stems.append(stem)
    if missing:
        raise ValueError('Matching exported MIDI is required for these audible instrument/drum stems (drums may use Shared drum guide): '
                         + ', '.join(sorted(missing)) + '. Source-FLP notes alone do not satisfy this requirement.')
    if not stems:
        raise ValueError('At least one analyzed audio stem is required.')
    # Stable ordering makes seed and output reproducible across input order.
    stems.sort(key=lambda s: (str(type(s['id']).__name__), str(s['id'])))
    sections = _sections(analysis, end, stems)
    effective = resolve_intensities(settings)
    warnings = []
    if not sections:
        warnings.append('No section boundaries were supplied; no automation was generated.')
    if settings['reverb_mode'] == 'baked':
        warnings.append('Baked reverb mode: added reverb is disabled.')
    curves, skipped = [], []
    bar = ppq * 4
    for stem in stems:
        role = stem['_role']
        reason = None
        if stem.get('automation_protected') is True:
            reason = 'Preserve this protected stem and its existing processing.'
        elif stem['_silent']:
            reason = 'Verified silent audio; no automation or MIDI guide needed.'
        elif role in _STABLE:
            reason = 'Keep foundational bass/kick and primary drum hits stable.'
        elif not stem['_activity']:
            reason = 'No reliable audible activity; no movement generated.'
        if reason or not sections:
            skipped.append({'stem_id': stem['id'], 'stem_name': stem['name'],
                            'reason': reason or 'No sections available.'})
            continue
        active = [_overlap(stem['_activity'], s['start_tick'], s['end_tick']) > 0 for s in sections]
        energies = [s['energy'] for s in sections]
        # An empty intro/outro must not consume this stem's strongest target.
        # Contrast is meaningful only among sections carrying audible material.
        audible_energies = [energy for energy, on in zip(energies, active) if on]
        high, low = max(audible_energies), min(audible_energies)
        contrast = high - low
        relative = [max(0., min(1., (e - low) / contrast)) if contrast >= .12 else 1.
                    for e in energies]
        candidates = {}
        if role in _PERCUSSION:
            # An opening into an evidenced rise, never rhythmic filtering of
            # drum hits. The strongest rise can use the user's full range.
            targets = [0.] * len(sections)
            rises = [energies[i + 1] - energies[i] for i in range(len(sections) - 1)]
            eligible = [i for i in range(len(sections) - 1)
                        if (active[i] and active[i + 1]
                            and sections[i]['end_tick'] - sections[i]['start_tick'] >= 4 * bar
                            and rises[i] >= .2)]
            strongest = max((rises[i] for i in eligible), default=0.)
            for index in eligible:
                targets[index] = -rises[index] / strongest
            candidates['filter'] = (_section_curve(targets, sections, bar, end, stem['_activity']),
                                    'Open percussion across the section into a measured energy rise.')
            order = ('filter',)
        else:
            if role != 'vocals':
                targets = [-(1 - e) if on and contrast >= .12 else 0.
                           for e, on in zip(relative, active)]
                candidates['filter'] = (_section_curve(targets, sections, bar, end, stem['_activity']),
                                        'Ramp brightness across sections, opening toward fuller passages.')
            candidates['delay'] = (_delay_curve(stem, sections, ppq, end),
                                   'A phrase-end echo ramp near a section boundary with a verified gap.')
            targets = [.2 + .8 * (1 - e) if on and contrast >= .12 else 0.
                       for e, on in zip(relative, active)]
            candidates['reverb'] = (_section_curve(targets, sections, bar, end, stem['_activity']),
                                    'Ramp space across sections, leaving denser passages clearer.')
            if role != 'vocals' and stem['analysis'].get('channels') == 2:
                targets = [2 * e - 1 if on and contrast >= .12 else 0.
                           for e, on in zip(relative, active)]
                candidates['width'] = (_section_curve(targets, sections, bar, end, stem['_activity']),
                                       'Ramp stereo width between sparse and fuller sections.')
            if role == 'vocals':
                order = ('delay', 'reverb')
            elif role in ('pads', 'chords', 'effects'):
                order = ('filter', 'width', 'reverb', 'delay')
            else:
                order = ('filter', 'delay', 'width', 'reverb')
        added = 0
        for family in order:
            if not effective[family] or family not in candidates:
                continue
            points, explanation = candidates[family]
            if not points or not any(value for _, value in points):
                continue
            factor = effective[family] / 100
            curves.append({'stem_id': stem['id'], 'stem_name': stem['name'],
                           'mixer_insert': stem['mixer_insert'], 'family': family,
                           'points': [(tick, value * factor) for tick, value in points],
                           'reason': explanation})
            added += 1
            if added == 2:
                break
        mix = _mix_candidates(stem, sections, bar, end) if effective['volume'] or effective['pan'] else {}
        for family, (points, explanation, parameter) in mix.items():
            if not effective[family] or not points or not any(value for _, value in points):
                continue
            factor = effective[family] / 100
            curves.append({'stem_id': stem['id'], 'stem_name': stem['name'],
                           'mixer_insert': stem['mixer_insert'], 'family': family,
                           'parameter': parameter,
                           'points': [(tick, value * factor) for tick, value in points],
                           'reason': explanation})
            added += 1
        if not added:
            skipped.append({'stem_id': stem['id'], 'stem_name': stem['name'],
                            'reason': 'No enabled movement has sufficient section/activity evidence.'})
    return {'curves': curves, 'sections': sections, 'warnings': warnings,
            'skipped': skipped, 'effective_intensities': effective,
            'seed': settings['seed'], 'reverb_mode': settings['reverb_mode']}
