"""Bounded-memory audio evidence and conservative stem-role recognition.

Audio uses the connector's eight-bar analysis cadence. Spectral features are
computed from short overlapping FFT frames; no whole-song waveform is retained.
Section names inferred from energy are heuristics. Explicit Playlist markers
take precedence and the caller should include sections in its review preview.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


ROLES = ('kick', 'snare', 'clap', 'tom', 'hat', 'perc', 'ride', 'crash',
         'bass', 'brass', 'guitar', 'keys', 'vocals', 'pads', 'synth', 'effects', 'drums')
_DRUMS = set(ROLES[:8]) | {'drums'}
_MELODIC = {'brass', 'guitar', 'keys', 'pads', 'synth'}
_CACHE_VERSION = 3
_BLOCK_FRAMES = 65536
_LEVEL_WINDOWS_PER_SECOND = 10


class AudioAnalysisError(ValueError):
    pass


def role_family(role):
    """Return a display/grouping family, rejecting unconfirmed unknown roles."""
    if role not in ROLES:
        raise ValueError(f'Unrecognized stem role: {role!r}')
    if role in _DRUMS:
        return 'Drums'
    if role in _MELODIC:
        return 'Melodic'
    return {'bass': 'Bass', 'vocals': 'Vocals', 'effects': 'Effects'}[role]


def _spectral_average(spectral, key):
    if not isinstance(spectral, dict):
        return 0.0
    if isinstance(spectral.get(key), (int, float)):
        return float(spectral[key])
    rows = spectral.get('chunks', [])
    # Silent regions cannot supply evidence that an 808 is sustained sub-bass.
    rows = [r for r in rows if float(r.get('rms', 0)) > 1e-5]
    weights = [max(0.0, r.get('end_seconds', 0) - r.get('start_seconds', 0)) for r in rows]
    return sum(float(r.get(key, 0)) * w for r, w in zip(rows, weights)) / (sum(weights) or 1)


def detect_role(name, spectral=None):
    """Return a canonical role or None; never give an arbitrary file a default.

    Numeric prefixes are removed, then exact category-code tokens win over
    drum words. Bare ``808`` requires both sub-band dominance and sustain.
    Unknown names remain unresolved even when their general timbre is similar
    to another instrument; the UI must ask the user to confirm those roles.
    """
    from stem_names import classify_name, normalized_name
    lexical = classify_name(name)
    if lexical['role'] is not None:
        return lexical['role']
    # Ambiguous explicit categories/drum names require user confirmation;
    # the 808 spectral fallback must never resolve those unrelated conflicts.
    if '808' in normalized_name(name).split() and lexical['candidates'] == ['bass']:
        if (_spectral_average(spectral, 'sub_ratio') >= .75
                and _spectral_average(spectral, 'sustained') >= .70):
            return 'bass'
    return None


def _audio_modules():
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise AudioAnalysisError(
            'Audio-stem analysis needs numpy and soundfile. Install this repo\'s requirements.txt first.'
        ) from exc
    return np, sf


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise AudioAnalysisError(f'{name} must be a positive finite number.')
    return value


def _stat_identity(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _content_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def _cache_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False  # JSON integers can exceed the floating-point range.


def _cache_valid(data, key):
    if not isinstance(data, dict) or data.get('cache_key') != key or data.get('analysis_version') != _CACHE_VERSION:
        return False
    if not isinstance(data.get('chunks'), list) or not data['chunks']:
        return False
    for field in ('duration_seconds', 'sample_rate', 'dynamic_range_db', 'onset_density', 'centroid', 'sub_ratio', 'sustained'):
        if not _cache_number(data.get(field)) or data[field] < 0:
            return False
    previous = 0.0
    for chunk in data['chunks']:
        if not isinstance(chunk, dict):
            return False
        for field in ('start_seconds', 'end_seconds', 'rms', 'onset_density', 'centroid', 'sub_ratio', 'sustained'):
            if not _cache_number(chunk.get(field)) or chunk[field] < 0:
                return False
        if abs(chunk['start_seconds'] - previous) > 1e-6 or chunk['end_seconds'] <= previous:
            return False
        if chunk['sub_ratio'] > 1 or chunk['sustained'] > 1:
            return False
        previous = chunk['end_seconds']
    if abs(previous - data['duration_seconds']) >= 1e-6:
        return False
    # Level bins cover the exact source sample timeline, including silence and
    # the final partial bin. A valid-looking list with gaps or moved boundaries
    # must not silently drive volume automation at the wrong time.
    rate = data['sample_rate']
    duration = data['duration_seconds']
    if type(rate) is not int or rate <= 0 or duration <= 0:
        return False
    if type(data.get('channels')) is not int or not 1 <= data['channels'] <= 64:
        return False
    sample_count = duration * rate
    if not _cache_number(sample_count):
        return False
    frames = round(sample_count)
    if frames <= 0 or abs(frames / rate - duration) > 1e-9:
        return False
    window_frames = max(1, round(rate / _LEVEL_WINDOWS_PER_SECOND))
    levels = data.get('level_windows')
    if not isinstance(levels, list) or len(levels) != (frames + window_frames - 1) // window_frames:
        return False
    previous = 0.0
    for index, level in enumerate(levels):
        if not isinstance(level, dict):
            return False
        if any(not _cache_number(level.get(field)) or level[field] < 0
               for field in ('start_seconds', 'end_seconds', 'rms', 'peak')):
            return False
        end = min(frames, (index + 1) * window_frames) / rate
        if (abs(level['start_seconds'] - previous) > 1e-9
                or abs(level['end_seconds'] - end) > 1e-9
                or level['end_seconds'] <= level['start_seconds']):
            return False
        # Floating-point PCM may legitimately exceed full scale. The bound is
        # the finite float32 decoder range, rather than an amplitude ceiling of 1.
        if (level['peak'] > 3.4028234663852886e38
                or level['rms'] > level['peak'] + 1e-12 + 1e-7 * level['peak']):
            return False
        previous = end
    spans = data.get('activity_segments')
    ends = data.get('phrase_ends')
    if not isinstance(spans, list) or not isinstance(ends, list) or len(spans) != len(ends):
        return False
    previous = 0.0
    for span, end in zip(spans, ends):
        if not isinstance(span, dict):
            return False
        a, b = span.get('start_seconds'), span.get('end_seconds')
        if not all(_cache_number(v) for v in (a, b, end)):
            return False
        if a < previous or b <= a or b > data['duration_seconds'] or end != b:
            return False
        previous = b
    return True


def _write_cache(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(result, handle, allow_nan=False, separators=(',', ':'))
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def analyze_audio(path, *, bpm, beats_per_bar=4, chunk_bars=8, cache_dir=None):
    """Measure audio in eight-bar chunks, independent of song duration in RAM.

    RMS uses all source channels, so anti-phase stereo does not disappear.
    Onset density is attacks/second, centroid is Hz, sub_ratio is spectral power
    below 120 Hz, and sustained is 0..1 occupancy adjusted by envelope variation.
    Dynamic range is the 95th minus 10th percentile of short-frame RMS in dB.
    ``level_windows`` contains nonoverlapping, approximately 100 ms bins with
    start_seconds, end_seconds, rms and peak, including silence and the true
    final partial bin. Peak is the largest absolute decoded sample across all
    channels; these are linear RMS/sample-peak levels, not LUFS or true peak.
    Floating-point audio above full scale remains above 1 in these values.
    Activity segments merge silence gaps under 150 ms using a streaming gate
    36 dB below the observed envelope peak (absolute floor -100 dB). Their
    endpoints are phrase-end candidates, accurate to one analysis frame.
    ``cache_dir=None`` disables disk caching. A provided directory caches by
    canonical file path, stat identity, complete streaming content digest, and
    analysis settings; unchanged timestamps cannot conceal changed PCM samples.
    """
    from section_chunks import chunk_length
    bpm = _positive(bpm, 'Tempo')
    _positive(beats_per_bar, 'Beats per bar')
    _positive(chunk_bars, 'Chunk bars')
    if int(beats_per_bar) != beats_per_bar or int(chunk_bars) != chunk_bars:
        raise AudioAnalysisError('Beats per bar and chunk bars must be whole numbers.')
    chunk_seconds = chunk_length(1, beats_per_bar=int(beats_per_bar), bars=int(chunk_bars)) * 60 / bpm
    path = Path(path).expanduser().resolve()
    before = _stat_identity(path)
    digest = _content_hash(path)
    signature = dict(version=_CACHE_VERSION, path=os.path.normcase(str(path)), stat=before,
                     sha256=digest, bpm=bpm, beats_per_bar=beats_per_bar, chunk_bars=chunk_bars)
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache_path = Path(cache_dir) / (key + '.json') if cache_dir is not None else None
    if cache_path is not None and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding='utf-8'))
            if _cache_valid(cached, key) and before == _stat_identity(path):
                return dict(cached, cache_hit=True)
        except (ValueError, OSError):
            pass  # A damaged cache is recomputed from the audio, never trusted.
    np, sf = _audio_modules()
    try:
        handle = sf.SoundFile(str(path))
    except (RuntimeError, OSError) as exc:
        raise AudioAnalysisError(f'Cannot decode stem audio {path.name}: {exc}') from exc
    with handle:
        rate, frames, channels = handle.samplerate, handle.frames, handle.channels
        if frames <= 0 or rate <= 0 or not 1 <= channels <= 64:
            raise AudioAnalysisError('Stem audio must contain samples in 1–64 channels at a valid sample rate.')
        duration = frames / rate
        count = math.ceil(duration / chunk_seconds)
        # Only compact statistics grow with song length; PCM stays bounded.
        accum = [dict(square=0.0, samples=0, onset=0, spectral=0, centroid=0.0,
                      sub=0.0, frame_rms=0.0, frame_square=0.0, active=0) for _ in range(count)]
        level_window_frames = max(1, round(rate / _LEVEL_WINDOWS_PER_SECOND))
        level_windows = []
        level_start = level_frames = 0
        level_square = level_peak = 0.0
        fft_size = max(512, min(8192, 2 ** round(math.log2(rate * .04644))))
        hop = fft_size // 2
        window = np.hanning(fft_size)
        frequencies = np.fft.rfftfreq(fft_size, 1 / rate)
        sub_mask = (frequencies >= 20) & (frequencies < 120)
        histogram = np.zeros(529, dtype=np.int64)  # -120..12 dB in .25 dB steps.
        flux_history = deque(maxlen=32)
        prior_magnitude = None
        last_onset = -math.inf
        envelope_peak = 0.0
        activity_start = None
        activity_end = None
        activity_segments = []
        pending = np.empty((0, channels), dtype=np.float32)
        pending_start = 0

        def frame_features(frame, start, valid):
            nonlocal prior_magnitude, last_onset, envelope_peak, activity_start, activity_end
            # Power is averaged across channels instead of summing waveforms.
            rms = float(np.sqrt(np.mean(np.square(frame[:valid], dtype=np.float64))))
            envelope_peak = max(envelope_peak, rms)
            now = start / rate
            if rms > max(1e-5, envelope_peak * .015):
                if activity_start is None:
                    activity_start = now
                activity_end = min(duration, (start + valid) / rate)
            elif activity_start is not None and now - activity_end >= .15:
                if activity_end - activity_start >= .08:
                    activity_segments.append(dict(start_seconds=activity_start, end_seconds=activity_end))
                activity_start = activity_end = None
            db = 20 * math.log10(max(rms, 1e-6))
            histogram[max(0, min(528, int(round((db + 120) * 4))))] += 1
            power = np.mean(np.abs(np.fft.rfft(frame * window[:, None], axis=0)) ** 2, axis=1)
            magnitude = np.sqrt(power)
            magnitude_sum = float(np.sum(magnitude))
            power_sum = float(np.sum(power))
            centroid = float(np.sum(magnitude * frequencies) / magnitude_sum) if magnitude_sum > 1e-12 else 0.0
            sub = float(np.sum(power[sub_mask]) / power_sum) if power_sum > 1e-12 else 0.0
            row = accum[min(count - 1, int((start / rate) / chunk_seconds))]
            row['spectral'] += 1
            row['centroid'] += centroid
            row['sub'] += sub
            row['frame_rms'] += rms
            row['frame_square'] += rms * rms
            row['active'] += int(rms > 1e-5)
            flux = (float(np.sum(np.maximum(magnitude - prior_magnitude, 0))) /
                    max(magnitude_sum, 1e-12)) if prior_magnitude is not None else 1.0
            threshold = max(.20, float(np.median(flux_history)) * 2.5) if flux_history else .20
            if rms > 1e-5 and flux > threshold and now - last_onset >= .10:
                row['onset'] += 1
                last_onset = now
            flux_history.append(flux)
            prior_magnitude = magnitude

        position = 0
        while True:
            block = handle.read(_BLOCK_FRAMES, dtype='float32', always_2d=True)
            if not len(block):
                break
            if not np.isfinite(block).all():
                raise AudioAnalysisError(f'{path.name} contains non-finite audio samples.')
            # Accumulate level bins from this same decode block. Keeping the
            # unfinished bin as scalars avoids buffering a second waveform.
            level_cursor = 0
            while level_cursor < len(block):
                length = min(len(block) - level_cursor, level_window_frames - level_frames)
                part = block[level_cursor:level_cursor + length]
                level_square += float(np.sum(np.square(part, dtype=np.float64)))
                level_peak = max(level_peak, float(np.max(np.abs(part))))
                level_frames += length
                level_cursor += length
                if level_frames == level_window_frames or position + level_cursor == frames:
                    end_frame = level_start + level_frames
                    level_windows.append(dict(start_seconds=level_start / rate,
                        end_seconds=end_frame / rate,
                        rms=math.sqrt(level_square / (level_frames * channels)), peak=level_peak))
                    level_start = end_frame
                    level_frames = 0
                    level_square = level_peak = 0.0
            cursor = 0
            while cursor < len(block):
                index = min(count - 1, int(((position + cursor) / rate) / chunk_seconds))
                boundary = min(frames, math.ceil((index + 1) * chunk_seconds * rate))
                length = min(len(block) - cursor, max(1, boundary - position - cursor))
                part = block[cursor:cursor + length]
                accum[index]['square'] += float(np.sum(np.square(part, dtype=np.float64)))
                accum[index]['samples'] += part.size
                cursor += length
            pending = np.concatenate((pending, block), axis=0)
            consumed = 0
            while len(pending) - consumed >= fft_size:
                frame_features(pending[consumed:consumed + fft_size], pending_start + consumed, fft_size)
                consumed += hop
            pending = pending[consumed:].copy()
            pending_start += consumed
            position += len(block)
        # The overlap remainder was already analyzed. Pad only when samples
        # beyond that overlap remain, or the entire source is shorter than FFT.
        if len(pending) > hop or not int(histogram.sum()):
            valid = len(pending)
            padded = np.zeros((fft_size, channels), dtype=np.float32)
            padded[:valid] = pending
            frame_features(padded, pending_start, valid)
        if activity_start is not None and activity_end - activity_start >= .08:
            activity_segments.append(dict(start_seconds=activity_start, end_seconds=activity_end))
    if position != frames or before != _stat_identity(path):
        raise AudioAnalysisError('The stem audio changed during analysis. Save it and run again.')
    chunks = []
    for index, row in enumerate(accum):
        start, end = index * chunk_seconds, min(duration, (index + 1) * chunk_seconds)
        spectral = max(1, row['spectral'])
        mean_rms = row['frame_rms'] / spectral
        variance = max(0.0, row['frame_square'] / spectral - mean_rms ** 2)
        variation = math.sqrt(variance) / max(mean_rms, 1e-12)
        sustained = row['active'] / spectral * math.exp(-max(0.0, variation - .10))
        chunks.append(dict(start_seconds=start, end_seconds=end,
                           rms=math.sqrt(row['square'] / max(1, row['samples'])),
                           onset_density=row['onset'] / (end - start),
                           centroid=row['centroid'] / spectral,
                           sub_ratio=max(0.0, min(1.0, row['sub'] / spectral)),
                           sustained=max(0.0, min(1.0, sustained))))
    cumulative = np.cumsum(histogram)
    low = int(np.searchsorted(cumulative, cumulative[-1] * .10))
    high = int(np.searchsorted(cumulative, cumulative[-1] * .95))
    result = dict(path=str(path), sha256=digest, sample_rate=rate, channels=channels,
                  duration_seconds=duration, dynamic_range_db=(high - low) / 4,
                  chunks=chunks, cache_hit=False, cache_key=key, analysis_version=_CACHE_VERSION,
                  level_windows=level_windows,
                  activity_segments=activity_segments,
                  phrase_ends=[span['end_seconds'] for span in activity_segments],
                  chunk_bars=int(chunk_bars), beats_per_bar=int(beats_per_bar), bpm=float(bpm))
    for feature in ('onset_density', 'centroid', 'sub_ratio', 'sustained'):
        result[feature] = sum(c[feature] * (c['end_seconds'] - c['start_seconds']) for c in chunks) / duration
    if cache_path is not None:
        _write_cache(cache_path, result)
    return result


def _marker_type(name):
    name = re.sub(r'[_-]+', ' ', str(name).casefold())
    # A post-chorus is not automatically a high-energy chorus or a drop.
    if re.search(r'\b(post\s*chorus|after\s+drop)\b', name):
        return 'verse'
    if re.search(r'\b(before|pre)\s+drop\b', name):
        return 'pre-chorus'
    if re.search(r'\b(pre\s*chorus|build|buildup)\b', name):
        return 'pre-chorus'
    for role, pattern in (('outro', r'\boutro\b'), ('intro', r'\bintro\b'),
                          ('bridge', r'\b(bridge|breakdown|break)\b'),
                          ('drop', r'\bdrop\b'), ('chorus', r'\b(chorus|hook)\b'),
                          ('verse', r'\bverse\b')):
        if re.search(pattern, name):
            return role
    return None


def detect_sections(stem_analyses, *, bpm, ppq, end_tick, beats_per_bar=4, half_time=False, markers=None):
    """Infer tempo-snapped regions from aggregate stem energy, honoring markers.

    Each row is either an analysis result, or ``{'analysis': result,
    'offset_tick': N}`` (``position`` and ``offset_seconds`` are also accepted).
    Sum-of-powers energy is a full-stem aggregate, not a phase-accurate rendered
    mixer sum. Energy/energy_jump use one shared song-wide scale in 0..1/-1..1.
    Half-time doubles the boundary grid; ``bar`` always refers to FL's real bar.
    """
    _positive(bpm, 'Tempo')
    for value, name in ((ppq, 'PPQ'), (end_tick, 'Song duration'), (beats_per_bar, 'Beats per bar')):
        _positive(value, name)
        if int(value) != value:
            raise AudioAnalysisError(f'{name} must be a whole number.')
    real_bar = int(ppq * beats_per_bar)
    bar = real_bar * (2 if half_time else 1)
    count = math.ceil(end_tick / bar)
    energy = [0.0] * count
    seconds_to_ticks = bpm * ppq / 60
    for stem in stem_analyses:
        analysis = stem.get('analysis', stem)
        offset = stem.get('offset_tick', stem.get('position', 0))
        if 'offset_seconds' in stem:
            offset = stem['offset_seconds'] * seconds_to_ticks
        if not isinstance(offset, (int, float)) or not math.isfinite(offset):
            raise AudioAnalysisError('Stem Playlist offsets must be finite ticks or seconds.')
        for chunk in analysis.get('chunks', []):
            a = offset + chunk['start_seconds'] * seconds_to_ticks
            b = offset + chunk['end_seconds'] * seconds_to_ticks
            rms = chunk['rms']
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (a, b, rms)) or rms < 0 or b <= a:
                raise AudioAnalysisError('Audio chunk timing and RMS must be finite and nonnegative.')
            a, b = max(0, a), min(end_tick, b)
            if b <= a:
                continue
            for index in range(int(a // bar), min(count, math.ceil(b / bar))):
                span = min(end_tick, (index + 1) * bar) - index * bar
                overlap = max(0.0, min(b, (index + 1) * bar) - max(a, index * bar))
                energy[index] += rms * rms * overlap / span
    energy = [math.sqrt(e) for e in energy]
    peak = max(energy, default=0.0)
    if peak:
        energy = [e / peak for e in energy]

    def average(a, b):
        values = energy[max(0, a):min(count, b)]
        return sum(values) / max(1, len(values))

    named, marker_names = {}, {}
    for marker in markers or []:
        kind = _marker_type(marker.get('name', marker.get('text', '')))
        tick = marker.get('tick', 0)
        if kind is None or marker.get('kind') == 8 or not isinstance(tick, (int, float)) or not math.isfinite(tick):
            continue
        if not 0 <= tick < end_tick:
            continue
        snapped = int(math.floor(tick / bar + .5)) * bar
        if snapped < end_tick:
            named[snapped] = kind
            marker_names[snapped] = str(marker.get('name', marker.get('text', '')))
    candidates = []
    for index in range(1, count):
        jump = energy[index] - energy[index - 1]
        persistent = average(index, index + 2) - average(index - 2, index)
        if abs(jump) >= .10 and abs(persistent) >= .10:
            candidates.append((abs(persistent) + abs(jump), index))
    selected = []
    for _, index in sorted(candidates, key=lambda item: (-item[0], item[1])):
        if all(abs(index - chosen) >= 4 for chosen in selected) and all(abs(index * bar - tick) >= 2 * bar for tick in named):
            selected.append(index)
    boundaries = sorted({0, int(end_tick), *named, *(index * bar for index in selected)})
    sections = []
    varied = max(energy, default=0) - min(energy, default=0) >= .12
    for a, b in zip(boundaries, boundaries[1:]):
        value = average(a // bar, math.ceil(b / bar))
        sections.append(dict(type=named.get(a, 'verse'), start_tick=a, end_tick=b,
                             bar=a // real_bar + 1, energy=value,
                             energy_jump=value - sections[-1]['energy'] if sections else 0.0,
                             marker_name=marker_names.get(a),
                             source='project marker' if a in named else 'energy heuristic'))
    for index, section in enumerate(sections):
        if section['source'] == 'project marker' or not varied:
            continue
        value = section['energy']
        previous = sections[index - 1]['energy'] if index else value
        following = sections[index + 1]['energy'] if index + 1 < len(sections) else value
        if index == 0 and value < .72:
            kind = 'intro'
        elif index == len(sections) - 1 and value < .72 and value < previous - .12:
            kind = 'outro'
        elif value >= .72:
            kind = 'chorus'
        elif following >= .72 and value > previous + .10:
            kind = 'pre-chorus'
        elif index > 0 and section['start_tick'] > end_tick * .5 and previous >= .72 and following >= .72:
            kind = 'bridge'
        else:
            kind = 'verse'
        section['type'] = kind
    return sections
