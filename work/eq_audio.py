"""Conservative, offline audio evidence for five fixed Pro-Q gain lanes.

This proposes editable cuts, not a judgement that an instrument's spectrum is
wrong. Shared energy alone is insufficient: an overlapping, known musical role
must have higher priority, or a band must grow unusually prominent relative to
that SAME stem over time. It never aims for a flat/equal-band spectrum.

Only feature arrays are retained for a whole song. WAV data is read in bounded
batches; stereo powers are averaged AFTER FFT, so anti-phase audio is retained.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re

import numpy as np


DEFAULT_CENTERS = (77.0, 266.48, 853.64, 2533.6, 7377.3)
_EPS = 1e-20
# These are conservative mix priorities, not inferred instrument identifications.
_PRIORITIES = {
    'kick': (6, 3, 1, 4, 1), 'bass': (5, 5, 2, 1, 1),
    'snare': (1, 4, 3, 5, 3), 'clap': (1, 2, 3, 5, 3),
    'hat': (1, 1, 1, 3, 5), 'perc': (1, 2, 2, 3, 4),
    'vocals': (1, 3, 5, 6, 4), 'lead': (1, 3, 5, 5, 3),
    'pad': (1, 2, 2, 2, 2), 'pluck': (1, 2, 3, 4, 3),
    'keys': (1, 3, 3, 3, 2), 'guitar': (1, 3, 3, 3, 2),
    'effects': (1, 1, 1, 1, 1),
}
_ALIASES = {'pads': 'pad', 'hats': 'hat', 'percussion': 'perc', 'vocal': 'vocals',
            'ride': 'hat', 'crash': 'hat', 'tom': 'perc'}


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite.')
    if positive and value <= 0:
        raise ValueError(f'{name} must be positive.')
    return float(value)


def _centers(values):
    values = tuple(_number(v, 'band center', positive=True) for v in values)
    if len(values) != 5 or any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError('Exactly five increasing band centers are required.')
    return values


def infer_role(name, explicit=None):
    """Use unambiguous name tokens only; unnamed Inserts stay unknown."""
    if explicit is not None:
        role = _ALIASES.get(str(explicit).lower(), str(explicit).lower())
        return role if role in _PRIORITIES else None
    tokens = set(re.findall(r'[a-z]+', str(name).lower()))
    found = {_ALIASES.get(t, t) for t in tokens} & _PRIORITIES.keys()
    return next(iter(found)) if len(found) == 1 else None


def _geometry(sample_rate, centers, hop_seconds):
    _number(sample_rate, 'sample_rate', positive=True)
    _number(hop_seconds, 'hop_seconds', positive=True)
    if not .025 <= hop_seconds <= .5:
        raise ValueError('hop_seconds must be between .025 and .5 seconds.')
    # >=120 ms supports sub/bass discrimination and overlaps at 100 ms cadence.
    fft_size = 2 ** math.ceil(math.log2(max(256, sample_rate * .12)))
    if fft_size > 65536:
        raise ValueError('Sample rate is too high for this bounded FFT analyzer.')
    window = np.hanning(fft_size)
    freqs = np.fft.rfftfreq(fft_size, 1 / sample_rate)
    band_masks, flank_masks = [], []
    for center in centers:
        distance = np.abs(np.log2(np.maximum(freqs, 1e-12) / center))
        band_masks.append((distance <= .5) & (freqs > 0))
        flank_masks.append((distance >= .7) & (distance <= 1.25) & (freqs > 0))
    return fft_size, window, np.asarray(band_masks), np.asarray(flank_masks)


def _extract(read, frame_count, sample_rate, channels, centers, hop_seconds, comparison_centers=None):
    """read(start,count) returns PCM channels; at most ~7 seconds per batch."""
    fft_size, window, masks, flanks = _geometry(sample_rate, centers, hop_seconds)
    duration = frame_count / sample_rate
    times = np.arange(max(1, math.ceil(duration / hop_seconds))) * hop_seconds
    rms = np.zeros(len(times))
    total_power = np.zeros(len(times))
    powers = np.zeros((len(times), 5))
    prominence = np.zeros_like(powers)
    # A competitor is measured at the DESTINATION band's actual frequency.
    # Shared queries prevent two differently tuned ordinal bands being treated
    # as if they covered the same spectrum. Prefix sums make queries cheap.
    extra_centers = () if comparison_centers is None else comparison_centers
    extra_centers = tuple(_number(v, 'comparison band center', positive=True) for v in extra_centers)
    comparison_centers = tuple(sorted(set(centers) | set(extra_centers)))
    comparison_power = np.zeros((len(times), len(comparison_centers)))
    frequencies = np.fft.rfftfreq(fft_size, 1 / sample_rate)
    lower = np.searchsorted(frequencies, np.asarray(comparison_centers) / math.sqrt(2), side='left')
    upper = np.searchsorted(frequencies, np.asarray(comparison_centers) * math.sqrt(2), side='right')
    norm = fft_size * float(np.dot(window, window))
    half = fft_size // 2
    # Keep multichannel/high-rate batches bounded too (about 8 MiB of PCM,
    # plus one FFT frame); normal stereo takes the faster 64-frame batches.
    batch_count = max(1, min(64, int(1048576 / max(1, hop_seconds * sample_rate * channels))))
    for first in range(0, len(times), batch_count):
        last = min(len(times), first + batch_count)
        start = max(0, round(times[first] * sample_rate) - half)
        stop = min(frame_count, round(times[last - 1] * sample_rate) + half)
        block = np.asarray(read(start, max(0, stop - start)), dtype=np.float64)
        if block.ndim == 1:
            block = block[:, None]
        if block.shape != (max(0, stop - start), channels) or not np.isfinite(block).all():
            raise ValueError('Audio contains invalid samples or could not be read completely.')
        for index in range(first, last):
            center_sample = round(times[index] * sample_rate)
            lo, hi = center_sample - half, center_sample + half
            frame = np.zeros((fft_size, channels), dtype=np.float64)
            left, right = max(0, lo), min(frame_count, hi)
            if right > left:
                frame[left - lo:right - lo] = block[left - start:right - start]
            # Gate using one hop, independently of the wider spectral window.
            gate_half = max(1, round(sample_rate * hop_seconds / 2))
            gate = frame[max(0, half - gate_half):min(fft_size, half + gate_half)]
            rms[index] = math.sqrt(float(np.mean(gate * gate)))
            spectrum = np.mean(np.abs(np.fft.rfft(frame * window[:, None], axis=0)) ** 2, axis=1) / norm
            spectrum[1:-1] *= 2
            total_power[index] = spectrum.sum()
            cumulative = np.concatenate(([0.], np.cumsum(spectrum)))
            comparison_power[index] = np.maximum(0, cumulative[upper] - cumulative[lower])
            powers[index] = masks @ spectrum
            local = powers[index] / np.maximum(1, masks.sum(axis=1))
            sides = (flanks @ spectrum) / np.maximum(1, flanks.sum(axis=1))
            prominence[index] = np.clip(10 * np.log10((local + _EPS) / (sides + _EPS)), -100, 100)
    return {'times': times, 'rms': rms, 'total_power': total_power, 'band_power': powers,
            'prominence_db': prominence, 'duration_seconds': duration,
            'sample_rate': sample_rate, 'channels': channels, 'centers_hz': list(centers),
            'comparison_centers_hz': list(comparison_centers), 'comparison_power': comparison_power,
            'fft_size': fft_size, 'spectral_window_seconds': fft_size / sample_rate}


def extract_features_array(audio, sample_rate, *, centers_hz=DEFAULT_CENTERS, hop_seconds=.1,
                           comparison_centers_hz=None):
    """Pure-array feature API, accepting mono or samples-by-channels PCM."""
    audio = np.asarray(audio)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2 or audio.shape[0] == 0 or not 1 <= audio.shape[1] <= 64:
        raise ValueError('Audio must be a nonempty samples-by-channels array.')
    return _extract(lambda start, count: audio[start:start + count], len(audio),
                    sample_rate, audio.shape[1], _centers(centers_hz), hop_seconds, comparison_centers_hz)


def _identity(path):
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def analyze_stem(path, *, centers_hz=DEFAULT_CENTERS, hop_seconds=.1, comparison_centers_hz=None):
    """Read only; hashes the source and rejects concurrent source replacement."""
    import soundfile as sf
    path = Path(path)
    before = _identity(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    with sf.SoundFile(str(path)) as handle:
        if len(handle) == 0 or not 1 <= handle.channels <= 64:
            raise ValueError(f'Invalid or empty audio: {path.name}')
        def read(start, count):
            handle.seek(start)
            return handle.read(count, dtype='float64', always_2d=True)
        result = _extract(read, len(handle), handle.samplerate, handle.channels,
                          _centers(centers_hz), hop_seconds, comparison_centers_hz)
    if _identity(path) != before:
        raise ValueError(f'Source changed during analysis: {path.name}')
    result['source_sha256'] = digest.hexdigest()
    result['source_path'] = str(path.resolve())
    return result


def _simplify(times, values, tolerance=.01):
    """Piecewise-linear compression bounded by dB error, preserving endpoints."""
    keep = {0, len(times) - 1}
    # Preserve both edges of exact baseline runs, so compression cannot start
    # a curve in silence merely because its deviation would be under .01 dB.
    transitions = np.flatnonzero((values[:-1] == 0) != (values[1:] == 0))
    for transition in transitions:
        keep.update((int(transition), int(transition + 1)))
    anchors = sorted(keep)
    pending = list(zip(anchors, anchors[1:]))
    while pending:
        a, b = pending.pop()
        if b - a < 2:
            continue
        line = values[a] + (values[b] - values[a]) * (times[a + 1:b] - times[a]) / (times[b] - times[a])
        error = np.abs(values[a + 1:b] - line)
        offset = int(np.argmax(error))
        if error[offset] > tolerance:
            middle = a + 1 + offset
            keep.add(middle)
            pending.extend(((a, middle), (middle, b)))
    return [{'seconds': round(float(times[i]), 9), 'db': round(float(values[i]), 6)} for i in sorted(keep)]


def _resample(features, times):
    source_times = np.asarray(features['times'], dtype=float)
    if source_times.ndim != 1 or len(source_times) == 0 or not np.isfinite(source_times).all():
        raise ValueError('Features need finite increasing times.')
    if np.any(np.diff(source_times) <= 0) or source_times[0] < 0:
        raise ValueError('Feature times must increase from a nonnegative origin.')
    result = {}
    for key, shape in [('rms', (len(source_times),)), ('total_power', (len(source_times),)),
                       ('band_power', (len(source_times), 5)), ('prominence_db', (len(source_times), 5))]:
        value = np.asarray(features.get(key, np.asarray(features['rms']) ** 2) if key == 'total_power' else features[key], dtype=float)
        if value.shape != shape or not np.isfinite(value).all() or (key != 'prominence_db' and np.any(value < 0)):
            raise ValueError(f'Invalid feature array: {key}')
        if value.ndim == 1:
            result[key] = np.interp(times, source_times, value, left=0, right=0)
        else:
            result[key] = np.stack([np.interp(times, source_times, value[:, b], left=0, right=0) for b in range(5)], axis=1)
    comparison_centers = features.get('comparison_centers_hz')
    if comparison_centers is not None:
        comparison_centers = tuple(_number(v, 'comparison band center', positive=True) for v in comparison_centers)
        if not comparison_centers or any(a >= b for a, b in zip(comparison_centers, comparison_centers[1:])):
            raise ValueError('Comparison centers must be unique and increasing.')
        values = np.asarray(features.get('comparison_power'), dtype=float)
        if values.shape != (len(source_times), len(comparison_centers)) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Invalid comparison power array.')
        result['comparison_centers_hz'] = comparison_centers
        result['comparison_power'] = np.stack([np.interp(times, source_times, values[:, b], left=0, right=0)
                                              for b in range(len(comparison_centers))], axis=1)
    return result


def _comparison_power(data, own_centers, destination_centers):
    """Return measured powers and availability at the requested exact centers."""
    measured = data.get('comparison_centers_hz', own_centers)
    source = data.get('comparison_power', data['band_power'])
    result = np.zeros((len(source), 5))
    available = np.zeros(5, dtype=bool)
    for band, center in enumerate(destination_centers):
        matches = [i for i, frequency in enumerate(measured) if abs(frequency - center) <= 1e-7 * max(1, center)]
        if len(matches) == 1:
            result[:, band] = source[:, matches[0]]
            available[band] = True
    return result, available


def _priority_at(role, centers):
    return np.interp(np.log2(centers), np.log2(DEFAULT_CENTERS), _PRIORITIES[role])


def plan_from_features(feature_rows, times=None, *, centers_hz=DEFAULT_CENTERS,
                       end_seconds=None, max_cut_db=1.5, hop_seconds=.1,
                       attack_seconds=.35, release_seconds=.8, rate_db_per_second=1.5):
    """Pure planner. Rows contain id, name, role?, mixer_gain?, centers_hz? and features.

    mixer_gain is a nonnegative LINEAR amplitude multiplier (not dB). Unknown
    roles are reported and left flat; they are never assigned a guessed role.
    Up to two strongest bands receive new cut targets at any frame. Release
    tails may overlap; their total reduction is capped at 1.5 * max_cut_db.
    """
    centers = _centers(centers_hz)
    _number(max_cut_db, 'max_cut_db', positive=True)
    if max_cut_db > 3:
        raise ValueError('max_cut_db must be at most 3 dB.')
    for name, value in [('hop_seconds', hop_seconds), ('attack_seconds', attack_seconds),
                        ('release_seconds', release_seconds), ('rate_db_per_second', rate_db_per_second)]:
        _number(value, name, positive=True)
    rows = list(feature_rows)
    if not rows:
        raise ValueError('At least one stem is required.')
    if len({str(row['id']) for row in rows}) != len(rows):
        raise ValueError('Stem IDs must be unique.')
    stem_centers = [_centers(row.get('centers_hz', row['features'].get('centers_hz', centers))) for row in rows]
    for row, actual_centers in zip(rows, stem_centers):
        measured_centers = row['features'].get('centers_hz')
        if measured_centers is not None and not np.allclose(_centers(measured_centers), actual_centers, rtol=1e-9, atol=1e-7):
            raise ValueError(f"{row.get('name', row['id'])}: requested frequencies differ from the analyzed frequencies.")
    duration = end_seconds if end_seconds is not None else max(float(r['features']['duration_seconds']) for r in rows)
    duration = _number(duration, 'end_seconds', positive=True)
    if times is None:
        times = np.arange(math.ceil(duration / hop_seconds)) * hop_seconds
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError('Planning times must start at zero and strictly increase.')
    times = times[times < duration]
    times = np.append(times, duration)
    data = [_resample(row['features'], times) for row in rows]
    roles = [infer_role(row.get('name', ''), row.get('role')) for row in rows]
    gains = [_number(row.get('mixer_gain', 1.0), 'mixer_gain') for row in rows]
    if min(gains) < 0:
        raise ValueError('mixer_gain must be nonnegative linear amplitude.')
    rms = np.asarray([d['rms'] for d in data]) * np.asarray(gains)[:, None]
    power = np.asarray([d['band_power'] for d in data]) * np.asarray(gains)[:, None, None] ** 2
    total_power = np.asarray([d['total_power'] for d in data]) * np.asarray(gains)[:, None] ** 2
    prominence = np.asarray([d['prominence_db'] for d in data])
    # Peak-relative gate with an absolute -72 dBFS floor. Spectral shares are
    # independent of volume, so overall level changes cannot cut every band.
    gate = np.maximum(10 ** (-72 / 20), rms.max(axis=1) * 10 ** (-42 / 20))
    active = rms > gate[:, None]
    shares = np.clip(power / np.maximum(_EPS, total_power[:, :, None]), 0, 1)
    presence = np.clip((shares - .02) / .12, 0, 1) * active[:, :, None]
    result, warnings = [], []
    for index, row in enumerate(rows):
        role = roles[index]
        excess = np.zeros_like(power[index])
        resonance = np.zeros_like(excess)
        overlap = np.zeros_like(excess)
        competitors = [set() for _ in range(5)]
        if role is not None and active[index].sum() >= 5:
            baseline = np.percentile(shares[index, active[index]], 35, axis=0)
            excess = 10 * np.log10((shares[index] + _EPS) / (baseline[None, :] + _EPS))
            # A band's share can rise solely because a different band quiets.
            # Require actual power growth too; otherwise a stable foundation
            # would get cut when an unrelated instrument harmonic disappears.
            power_baseline = np.percentile(power[index, active[index]], 35, axis=0)
            power_excess = 10 * np.log10((power[index] + _EPS) / (power_baseline[None, :] + _EPS))
            excess = np.minimum(excess, power_excess)
            resonance = np.clip((excess - 3) / 9, 0, 1) * np.clip((prominence[index] - 6) / 12, 0, 1) * presence[index]
            # A pitched note alone is not a defect; require sustained change.
            held = np.zeros(5)
            for ti in range(len(times)):
                dt = times[ti] - times[ti - 1] if ti else 0
                held = np.where(resonance[ti] > .05, held + dt, 0)
                resonance[ti] *= np.clip((held - .25) / .5, 0, 1)
            for other, other_role in enumerate(roles):
                if other == index or other_role is None:
                    continue
                other_power, available = _comparison_power(data[other], stem_centers[other], stem_centers[index])
                other_power *= gains[other] ** 2
                if not available.all():
                    warning = 'Some cross-stem comparisons were skipped because features lack the destination band frequencies; use a shared comparison_centers_hz set.'
                    if warning not in warnings:
                        warnings.append(warning)
                other_shares = np.clip(other_power / np.maximum(_EPS, total_power[other, :, None]), 0, 1)
                other_presence = np.clip((other_shares - .02) / .12, 0, 1) * active[other, :, None]
                priority = np.maximum(0, _priority_at(other_role, stem_centers[index]) - _priority_at(role, stem_centers[index])) / 4
                priority = np.minimum(1, priority)
                relative_db = 10 * np.log10((other_power + _EPS) / (power[index] + _EPS))
                balance = np.clip(1 - np.abs(relative_db) / 24, 0, 1)
                balance *= (relative_db > -12) & (relative_db < 18)
                evidence = presence[index] * other_presence * balance * priority[None, :] * available[None, :]
                overlap = np.maximum(overlap, evidence)
                for band in range(5):
                    if np.max(evidence[:, band]) > .1:
                        competitors[band].add(str(rows[other]['id']))
        elif role is None:
            warnings.append(f"{row.get('name', row['id'])}: unknown role; EQ left flat pending identification.")
        target = max_cut_db * np.maximum(overlap, resonance * .6)
        target[target < .1] = 0
        # Suppress general tonal reshaping; use the two clearest issues only.
        for ti in range(len(times)):
            target[ti, np.argsort(target[ti])[:-2]] = 0
        reduction = np.zeros_like(target)
        for ti in range(1, len(times)):
            dt = times[ti] - times[ti - 1]
            tau = np.where(target[ti] > reduction[ti - 1], attack_seconds, release_seconds)
            next_value = reduction[ti - 1] + (target[ti] - reduction[ti - 1]) * (1 - np.exp(-dt / tau))
            next_value = np.clip(next_value, reduction[ti - 1] - rate_db_per_second * dt,
                                 reduction[ti - 1] + rate_db_per_second * dt)
            # Silence supplies a zero TARGET through the activity gate. Let
            # the existing cut release smoothly so recurring hits cannot make
            # the automation jump between a cut and zero every 100 ms.
            negligible = (next_value <= .01) & (reduction[ti - 1] <= rate_db_per_second * dt)
            next_value[negligible] = 0
            if next_value.sum() > max_cut_db * 1.5:
                # Only scale positive changes; shrinking the whole vector
                # could violate the rate limit on already-releasing bands.
                delta = next_value - reduction[ti - 1]
                falling = reduction[ti - 1] + np.minimum(delta, 0)
                rising = np.maximum(delta, 0)
                room = max(0, max_cut_db * 1.5 - float(falling.sum()))
                next_value = falling + rising * min(1, room / max(_EPS, float(rising.sum())))
            # Return naturally to the exact project-end baseline.
            reduction[ti] = np.minimum(next_value, rate_db_per_second * max(0, duration - times[ti]))
        reduction[0] = reduction[-1] = 0
        bands = []
        for band, center in enumerate(stem_centers[index]):
            moved = float(reduction[:, band].max()) >= .1
            if not moved:
                reduction[:, band] = 0
            reason_parts = []
            if competitors[band] and moved:
                reason_parts.append('Concurrent band energy overlaps a higher-priority known role.')
            if resonance[:, band].max() > .1 and moved:
                reason_parts.append('Sustained local prominence increases relative to this stem’s own spectral balance.')
            if not reason_parts:
                reason_parts.append('Insufficient evidence for an EQ change; retained at the saved gain (0 dB change).')
            bands.append({'band_index': band, 'center_hz': center, 'active': moved,
                          'points': _simplify(times, -reduction[:, band]),
                          'reason': ' '.join(reason_parts),
                          'evidence': {'center_hz': center,
                                       'analysis_region_hz': [center / math.sqrt(2), center * math.sqrt(2)],
                                       'maximum_cut_db': round(float(reduction[:, band].max()), 6),
                                       'maximum_overlap_score': round(float(overlap[:, band].max()), 6),
                                       'maximum_temporal_excess_db': round(float(np.maximum(0, excess[active[index], band]).max()) if active[index].any() else 0, 6),
                                       'competing_stem_ids': sorted(competitors[band]),
                                       'changed_fraction': round(float(np.mean(reduction[:, band] >= .05)), 6)}})
        result.append({'id': row['id'], 'name': row.get('name', str(row['id'])), 'role': role,
                       'mixer_gain': gains[index], 'bands': bands, 'centers_hz': list(stem_centers[index]),
                       'center_source': 'per_stem' if 'centers_hz' in row else 'feature_metadata_or_default',
                       'source_sha256': row['features'].get('source_sha256'),
                       'source_path': row['features'].get('source_path'),
                       'active_fraction': round(float(active[index].mean()), 6)})
    return {'version': 1, 'method': 'five-band conservative spectral overlap and temporal prominence',
            'end_seconds': duration, 'centers_hz': list(centers), 'max_cut_db': max_cut_db,
            'boosts_enabled': False, 'gain_semantics': 'delta_db_relative_to_saved_band_gain',
            'stems': result, 'warnings': warnings,
            'analysis': {'hop_seconds': hop_seconds, 'maximum_new_targets_per_frame': 2,
                         'maximum_total_reduction_db': max_cut_db * 1.5,
                         'curve_tolerance_db': .01, 'absolute_activity_floor_dbfs': -72,
                         'minimum_lane_peak_cut_db': .1,
                         'attack_seconds': attack_seconds, 'release_seconds': release_seconds,
                         'rate_db_per_second': rate_db_per_second,
                         'role_priorities': {k: list(v) for k, v in _PRIORITIES.items()},
                         'priority_interpolation': 'log frequency at each destination band center',
                         'cross_stem_frequency_matching': 'both sources measured at each destination band center',
                         'limitations': ['Spectral overlap is a masking candidate, not proof of audible masking.',
                                         'Fixed-frequency gain automation only; supplied band shapes are preserved.',
                                         'Unknown roles remain flat. No boosts or flat-spectrum target.',
                                         'Analysis assumes aligned stems and supplied linear mixer gains.',
                                         'Source-WAV spectral evidence does not simulate the saved EQ response.',
                                         'Final judgement requires level-matched listening in the project.']}}


def plan_eq_automation(stems, *, bpm, ppq, end_tick=None, end_seconds=None,
                       centers_hz=DEFAULT_CENTERS, max_cut_db=1.5,
                       hop_seconds=.1, progress=None):
    """File API. progress(message) is called before and after each source read."""
    bpm = _number(bpm, 'bpm', positive=True)
    ppq = _number(ppq, 'ppq', positive=True)
    if end_tick is not None:
        tick_duration = _number(end_tick, 'end_tick', positive=True) * 60 / bpm / ppq
        if end_seconds is not None and abs(end_seconds - tick_duration) > .001:
            raise ValueError('end_seconds and end_tick disagree.')
        end_seconds = tick_duration
    rows = []
    stems = list(stems)
    default_centers = _centers(centers_hz)
    actual_centers = [_centers(stem.get('centers_hz', default_centers)) for stem in stems]
    comparison_centers = tuple(sorted({value for group in actual_centers for value in group}))
    for index, stem in enumerate(stems):
        if progress:
            progress(f"Analyzing EQ audio {index + 1}/{len(stems)}: {stem.get('name', stem['id'])}")
        features = analyze_stem(stem['sample'], centers_hz=actual_centers[index], hop_seconds=hop_seconds,
                                comparison_centers_hz=comparison_centers)
        rows.append({**stem, 'features': features})
        if progress:
            progress(f"Analyzed {stem.get('name', stem['id'])}: {features['duration_seconds']:.2f} seconds")
    plan = plan_from_features(rows, centers_hz=centers_hz, end_seconds=end_seconds,
                              max_cut_db=max_cut_db, hop_seconds=hop_seconds)
    plan.update({'bpm': bpm, 'ppq': int(ppq), 'end_tick': end_tick})
    return plan
