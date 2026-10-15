"""Lossless FL event-container reader/writer, including FL 25+ extended events.

Opaque plugin states are never decoded or regenerated.
"""
from pathlib import Path
import struct


def read_bytes(data):
    if len(data) < 22 or data[:8] != b'FLhd\x06\0\0\0' or data[14:18] != b'FLdt':
        raise ValueError('This is not a supported FL Studio project container.')
    end = 22 + struct.unpack_from('<I', data, 18)[0]
    if end != len(data):
        raise ValueError('The FL project is incomplete or uses an unsupported container.')
    header, events, pos = data[8:14], [], 22
    while pos < end:
        key = data[pos]
        pos += 1
        start = pos
        kind = key
        if key == 172:
            if pos + 4 > end:
                raise ValueError('Truncated extended FL event.')
            kind = struct.unpack_from('<I', data, pos)[0] >> 24
            pos += 4
            if kind not in (0, 64, 128, 192):
                raise ValueError('Unsupported extended FL event type.')
        if kind < 192:
            length = (1, 2, 4)[kind // 64]
        else:
            length = shift = 0
            while True:
                if pos >= end or shift > 28:
                    raise ValueError('Invalid FL event length.')
                byte = data[pos]
                pos += 1
                length |= (byte & 127) << shift
                shift += 7
                if not byte & 128:
                    break
        if pos + length > end:
            raise ValueError('Truncated FL event.')
        events.append((key, data[start if key == 172 else pos:pos + length]))
        pos += length
    return header, events


def read_fl(path):
    return read_bytes(Path(path).read_bytes())


def encode_fl(header, events):
    if len(header) != 6:
        raise ValueError('Invalid FL header.')
    stream = bytearray()
    for key, payload in events:
        stream.append(key)
        if key == 172:
            if len(payload) < 5:
                raise ValueError('Invalid extended FL event.')
        elif key < 192:
            if len(payload) != (1, 2, 4)[key // 64]:
                raise ValueError('Invalid fixed FL event.')
        else:
            size = len(payload)
            while size >= 128:
                stream.append((size & 127) | 128)
                size >>= 7
            stream.append(size)
        stream.extend(payload)
    return b'FLhd' + struct.pack('<I', 6) + header + b'FLdt' + struct.pack('<I', len(stream)) + stream
