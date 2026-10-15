"""Portable painted energy envelopes and a MIDI-only timeline overview."""
from __future__ import annotations

from collections import defaultdict, deque
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re

from mido import MidiFile

MAX_POINTS = 8192
MAX_FILE_BYTES = 1_000_000
FORMAT = 'velocity-energy-curve'


class CurveError(ValueError):
    pass


def source_identity(path: Path, mid: MidiFile | None = None) -> dict:
    """Tie a painting to the exact source bytes, including its note arrangement."""
    try:
        data = Path(path).read_bytes()
        mid = mid or MidiFile(file=BytesIO(data))
    except (OSError, ValueError, EOFError) as exc:
        raise CurveError('Could not read this song MIDI. Choose a valid MIDI file.') from exc
    if mid.type == 2 or mid.ticks_per_beat <= 0:
        raise CurveError('Use a type 0 or 1 MIDI with musical beat timing.')
    end = max((sum(message.time for message in track) for track in mid.tracks), default=0)
    if end <= 0:
        raise CurveError('This MIDI has no duration to paint.')
    return {'source_sha256': hashlib.sha256(data).hexdigest(),
            'ticks_per_beat': mid.ticks_per_beat, 'end_tick': end}


def inspect_song(path: Path) -> dict:
    path = Path(path)
    try:
        data = path.read_bytes()
        mid = MidiFile(file=BytesIO(data))
    except (OSError, ValueError, EOFError) as exc:
        raise CurveError('Could not read this song MIDI. Choose a valid MIDI file.') from exc
    identity = source_identity(path, mid)
    # Detect a source edited during loading rather than combining two versions.
    if hashlib.sha256(data).hexdigest() != identity['source_sha256']:
        raise CurveError('The MIDI changed while loading. Load it again.')
    end, ppq = identity['end_tick'], identity['ticks_per_beat']
    stride = ppq * max(1, math.ceil(end / (ppq * 4096)))
    ticks = list(range(0, end, stride)) + [end]
    markers = []
    changes = []
    paintable = True
    for track in mid.tracks:
        tick = 0
        pending = defaultdict(deque)
        for message in track:
            tick += message.time
            if message.is_meta:
                if message.type in {'marker', 'cue_marker'} and 0 <= tick <= end:
                    markers.append({'tick': tick, 'text': message.text[:100]})
                if message.type == 'time_signature' and (message.numerator, message.denominator) != (4, 4):
                    paintable = False
            elif message.type == 'note_on' and message.velocity > 0:
                pending[(message.channel, message.note)].append((tick, message.velocity / 127))
            elif message.type in {'note_on', 'note_off'}:
                events = pending[(message.channel, message.note)]
                if events:
                    start, weight = events.popleft()
                    if tick > start:
                        changes.extend(((start, weight), (tick, -weight)))
        for events in pending.values():
            for start, weight in events:
                if end > start:
                    changes.extend(((start, weight), (end, -weight)))
    changes.sort(key=lambda item: item[0])
    # Integrate simultaneous-note activity over each displayed beat. This is a
    # visual note-density guide, not an audio loudness estimate or desired curve.
    activity = []
    cursor = 0
    weight = 0.0
    for start, stop in zip(ticks, ticks[1:]):
        position = start
        total = 0.0
        while cursor < len(changes) and changes[cursor][0] < stop:
            tick, change = changes[cursor]
            total += max(0.0, weight) * max(0, tick - position)
            position = max(position, tick)
            weight += change
            cursor += 1
        total += max(0.0, weight) * (stop - position)
        activity.append(total / max(1, stop - start))
    peak = max(activity, default=0.0)
    activity = [round(value / peak, 4) if peak else 0.0 for value in activity] + [0.0]
    return {**identity, 'title': path.stem, 'bar_ticks': ppq * 4, 'bars': end / (ppq * 4),
            'duration_seconds': mid.length, 'paintable': paintable, 'ticks': ticks,
            'activity': activity, 'markers': sorted(markers, key=lambda item: item['tick'])}


def make_curve(overview: dict, points=None) -> dict:
    if not overview.get('paintable', True):
        raise CurveError('Energy painting currently uses 4/4. You can still generate Classic without painting.')
    return validate_curve({
        'format': FORMAT, 'version': 1,
        **{key: overview[key] for key in ('source_sha256', 'ticks_per_beat', 'end_tick')},
        'points': points if points is not None else [[tick, 0.5] for tick in overview['ticks']],
    }, overview)


def validate_curve(document: dict, overview: dict | None = None) -> dict:
    fields = {'format', 'version', 'source_sha256', 'ticks_per_beat', 'end_tick', 'points'}
    if not isinstance(document, dict) or set(document) != fields:
        raise CurveError('Choose an Energy curve JSON file saved by this writer.')
    if document['format'] != FORMAT or type(document['version']) is not int or document['version'] != 1:
        raise CurveError('This energy curve version is unsupported.')
    fingerprint = document['source_sha256']
    if not isinstance(fingerprint, str) or not re.fullmatch('[0-9a-f]{64}', fingerprint):
        raise CurveError('The energy curve has no valid source identity.')
    for field in ('ticks_per_beat', 'end_tick'):
        if type(document[field]) is not int or document[field] <= 0:
            raise CurveError('The energy curve has invalid MIDI timing.')
    if document['ticks_per_beat'] > 32767:
        raise CurveError('The energy curve has unsupported beat timing.')
    if overview is not None:
        if overview.get('paintable') is False:
            raise CurveError('Energy painting currently uses 4/4.')
        if any(document[field] != overview[field] for field in ('source_sha256', 'ticks_per_beat', 'end_tick')):
            raise CurveError('This painting belongs to a different MIDI. Load its matching song or paint this song separately.')
    points = document['points']
    if not isinstance(points, list) or not 2 <= len(points) <= MAX_POINTS:
        raise CurveError(f'An energy curve needs 2 to {MAX_POINTS} points.')
    clean = []
    previous = -1
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise CurveError('An energy point needs a MIDI position and energy value.')
        tick, energy = point
        if type(tick) is not int or not previous < tick <= document['end_tick']:
            raise CurveError('Energy points must be in increasing order inside the song, without duplicate positions.')
        if type(energy) not in (int, float) or not math.isfinite(energy) or not 0 <= energy <= 1:
            raise CurveError('Painted energy must stay between 0% and 100%.')
        clean.append([tick, float(energy)])
        previous = tick
    if clean[0][0] != 0 or clean[-1][0] != document['end_tick']:
        raise CurveError('The energy curve must cover the song from its start to its exact end.')
    return {**document, 'points': clean}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CurveError('The energy file contains duplicate fields.')
        result[key] = value
    return result


def load_curve(path: Path, overview: dict | None = None) -> dict:
    try:
        with Path(path).open('rb') as stream:
            raw = stream.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            raise CurveError('This energy file is too large.')
        document = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CurveError('Could not read this energy file. Choose a saved Energy curve JSON.') from exc
    return validate_curve(document, overview)


def save_curve(path: Path, document: dict) -> None:
    document = validate_curve(document)
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        temporary.write_text(json.dumps(document, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(path)
    except OSError as exc:
        raise CurveError('Could not save the painting in this location.') from exc


def sample_energy_curve(curve: dict, ticks: list[int]) -> list[float]:
    """Linear interpolation at sorted controller ticks, with no time-grid drift."""
    points = curve['points']
    index = 0
    result = []
    previous = -1
    for tick in ticks:
        if tick < previous or not 0 <= tick <= curve['end_tick']:
            raise CurveError('Controller positions must be ordered inside the energy curve.')
        previous = tick
        while index + 1 < len(points) - 1 and points[index + 1][0] <= tick:
            index += 1
        start, value = points[index]
        end, next_value = points[index + 1]
        position = (tick - start) / (end - start)
        result.append(value + (next_value - value) * position)
    return result
