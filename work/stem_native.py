"""Native FL Automation Clip serialization, using observed FL Studio records.

This is deliberately separate from the velocity-controller writer.  Type 5
channels hold sparse event-234 points; event-227 sources have no Keyboard
Controller flag.  Identity links can fan out.  Different target ranges must
use separate channels because per-link formula encoding is not verified.

The 80-byte native clip and neutral channel fields come from Image-Line's
installed SFX transitions template. The source FL26 channel clip supplies the
88-byte layout. See resources/stem-native.json for provenance and limitations.
Binary verification is not a claim of native FL Studio playback validation.
"""
from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
import struct


from app_paths import resource_path

RESOURCE = resource_path('stem-native.json')
POINT = struct.Struct('<ddfI')
LINEAR = 0x02000000  # Observed zero-tension straight segment in native template.
DEFAULT_COLOR = 0x007EA7DB


@lru_cache(maxsize=1)
def _resource():
    data = json.loads(RESOURCE.read_text(encoding='utf-8'))
    if data.get('version') != 1:
        raise ValueError('Unsupported native Automation Clip definition.')
    return data


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer from {low} to {high}.')
    return value


def parse_points(payload, ppq):
    """Return native event-234 anchors as (local ticks, normalized value).

    This inspection helper also validates the specific supported envelope
    layout, including the 112-byte opaque native tail that we preserve.
    """
    _integer(ppq, 'PPQ', 24, 9600)
    if len(payload) < 21:
        raise ValueError('Truncated native Automation Clip curve.')
    count = struct.unpack_from('<I', payload, 17)[0]
    if count < 2 or len(payload) != 21 + count * POINT.size + 112:
        raise ValueError('Unsupported native Automation Clip curve layout.')
    position = 0.0
    result = []
    for index in range(count):
        delta, value, tension, flags = POINT.unpack_from(payload, 21 + index * POINT.size)
        if not all(math.isfinite(v) for v in (delta, value, tension)) or delta < 0 or not 0 <= value <= 1:
            raise ValueError('Invalid native Automation Clip point.')
        if index and delta <= 0:
            raise ValueError('Native Automation Clip points must be strictly ordered.')
        position += delta
        result.append((position * ppq, value))
    return result


def build_curve(points, ppq):
    """Encode sparse straight anchors, with local tick positions and 0–1 values."""
    _integer(ppq, 'PPQ', 24, 9600)
    points = list(points)
    if not 2 <= len(points) <= 65535:
        raise ValueError('A native curve needs 2–65535 sparse anchors.')
    previous = -1
    normalized = []
    for tick, value in points:
        _integer(tick, 'Point time', 0, 0xffffffff)
        if tick <= previous:
            raise ValueError('Native curve point times must increase strictly.')
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Native curve values must be finite and between 0 and 1.')
        normalized.append((tick, float(value)))
        previous = tick
    if normalized[0][0] != 0:
        raise ValueError('Native curves must start at local time zero.')
    source = next(bytes.fromhex(p) for k, p in _resource()['channel'] if k == 234)
    count = struct.unpack_from('<I', source, 17)[0]
    head = bytearray(source[:17])
    struct.pack_into('<i', head, 4, 0)  # No LFO amount; the captured flag at8 is off.
    head[8] = 0
    tail = source[21 + count * POINT.size:]
    encoded = bytearray(head + struct.pack('<I', len(normalized)))
    previous = 0
    for index, (tick, value) in enumerate(normalized):
        encoded.extend(POINT.pack((tick - previous) / ppq, value, 0.0, 0 if index == 0 else LINEAR))
        previous = tick
    encoded.extend(tail)
    parse_points(encoded, ppq)
    return bytes(encoded)


def build_channel(cid, name, points, ppq, group=0, color=DEFAULT_COLOR):
    """Build a native type-5 channel without note or plugin-state events."""
    _integer(cid, 'Channel ID', 0, 4094)
    _integer(group, 'Channel group', 0, 0x7fffffff)
    _integer(color, 'Color', 0, 0xffffffff)
    if not isinstance(name, str) or not name.strip() or '\0' in name:
        raise ValueError('An Automation Clip needs a nonempty name without NUL characters.')
    curve = build_curve(points, ppq)
    replacements = {
        64: struct.pack('<H', cid),
        203: (name[:190] + '\0').encode('utf-16-le'),
        128: struct.pack('<I', color),
        145: struct.pack('<I', group),
        234: curve,
    }
    result = [(k, replacements.get(k, bytes.fromhex(p))) for k, p in _resource()['channel']]
    fields = dict(result)
    if fields[21] != b'\x05' or fields[219][:8] != struct.pack('<II', 0, 12800):
        raise ValueError('The native definition must have a 0–1 Automation Clip range.')
    if any(k in (213, 223, 224) for k, _ in result):
        raise ValueError('The native definition unexpectedly contains plugin or note data.')
    return result


def build_link(cid, target):
    """Native identity link; several targets may use the same native channel."""
    _integer(cid, 'Channel ID', 0, 4094)
    _integer(target, 'Parameter address', 0, 0xffffffff)
    return 227, struct.pack('<5I', cid << 16, 0, target, 8, 469)


def build_playlist_clip(cid, start_tick, duration, track, serial, record_size=88,
                        offset_tick=None, ppq=96):
    """Place native channel on a zero-based Playlist track, in project ticks.

    Version 24/25 uses 80 bytes in the captured template; the supplied FL26
    project uses 88. Unknown layouts fail explicitly rather than fabricating
    their version-specific fields.
    """
    _integer(cid, 'Channel ID', 0, 4094)
    _integer(start_tick, 'Clip position', 0, 0xffffffff)
    _integer(duration, 'Clip duration', 1, 0xffffffff)
    _integer(track, 'Playlist track', 0, 499)
    _integer(serial, 'Playlist serial', 1, 0xffffffff)
    if start_tick + duration > 0xffffffff:
        raise ValueError('Native clip extends beyond the supported project timeline.')
    template = _resource()['playlist_templates'].get(str(record_size))
    if template is None:
        raise ValueError('Native automation supports observed 80- and 88-byte Playlist records only.')
    clip = bytearray.fromhex(template)
    struct.pack_into('<IHHIHH', clip, 0, start_tick, 0x5000, cid, duration, 499 - track, 0)
    if offset_tick is None:
        struct.pack_into('<ff', clip, 24, -1.0, -1.0)  # No left/right trim.
    else:
        _integer(ppq, 'PPQ', 24, 9600)
        _integer(offset_tick, 'Curve offset', 0, 0xffffffff)
        # Native automation trim units are beats, confirmed by FL25 demo
        # P.Priime - Use Me: CID16 is split into (0,16) / (16,32) at PPQ96.
        struct.pack_into('<ff', clip, 24, offset_tick / ppq, (offset_tick + duration) / ppq)
    struct.pack_into('<I', clip, 32, serial)
    return bytes(clip)


def compatible_groups(moves):
    """Group only identical normalized curves; differing ranges stay separate.

    Input dictionaries require family, parameter, shape, start_tick,
    duration_ticks, points and target. This intentionally uses exact timing,
    an allowed subset of the user's one-beat tolerance, to avoid changing a
    stem's timing or shape when merging.
    """
    groups = {}
    targets = set()
    for move in moves:
        target = move['target']
        # A target may occur in successive sections, but never twice within
        # one proposed native clip span.
        ownership = (target, move['start_tick'], move['duration_ticks'])
        if ownership in targets:
            raise ValueError('The same native target has duplicate moves in one span.')
        targets.add(ownership)
        key = (move['family'], move['parameter'], move['shape'], move['start_tick'],
               move['duration_ticks'], tuple(tuple(p) for p in move['points']))
        groups.setdefault(key, []).append(move)
    return list(groups.values())
