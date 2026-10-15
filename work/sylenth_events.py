"""Append independent Keyboard Controllers for Sylenth1 parameter movement.

Generator destinations and link records were checked against native FL saves.
A controller velocity link to Sylenth1 channel1 Filter A Cutoff42 uses
address0x0001802a. Pattern-event223 VST value encoding is NOT assumed.

Native piano-roll velocity is 0..128.  We use positive velocities1..128;
thus a normalized zero is approximated by1/128.  Musical notes, original
controllers, plugin states and templates are never edited here.
"""
from __future__ import annotations

import math
import struct
from collections import Counter


NOTE = struct.Struct('<IHHIHHHHBBBB')
LINK = struct.Struct('<5I')
GROUP_NAME = 'Sylenth movement'
FK_NAME = 'Fruity Keyboard Controller'
MAX_CHANNEL = 4094  # 0x0fff is the selected-channel MIDI target in native saves.
MAX_PARAMETER = 242  # Enumerated Sylenth1 continuous/discrete program parameters.


def _int(value, label, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f'{label} must be an integer from {minimum} to {maximum}.')
    return value


def _text(payload):
    return payload.decode('utf-16-le', errors='strict').rstrip('\0')


def _name(value):
    if not isinstance(value, str) or not value.strip() or '\0' in value or len(value) > 200:
        raise ValueError('Choose a nonempty controller name up to 200 characters without nulls.')
    return (value + '\0').encode('utf-16-le')


def target_address(instrument_id, parameter_id):
    """Native generator REC address using Sylenth's enumerated wrapper index."""
    cid = _int(instrument_id, 'Instrument channel', 0, MAX_CHANNEL)
    param = _int(parameter_id, 'Sylenth parameter index', 0, MAX_PARAMETER)
    return (cid << 16) | (0x8000 + param)


target_id = target_address


def velocity_link(source_id, instrument_id, parameter_id):
    """One native event227; final8/469 fields match the saved default link."""
    cid = _int(source_id, 'Controller channel', 0, MAX_CHANNEL)
    return (227, LINK.pack((cid << 16) | 0x8001, 0,
                           target_address(instrument_id, parameter_id), 8, 469))


def quantized_velocity(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Sylenth movement values must be finite normalized numbers from 0 to 1.')
    return max(1, min(128, int(math.floor(value * 128 + 0.5))))


def quantized_value(value):
    return quantized_velocity(value) / 128.0


def encode_controller_points(controller_id, points, end_tick):
    """Encode sorted (native FL tick, normalized value) points as held notes.

    Points must include tick0.  Every supplied point gets positive duration;
    the caller supplies its desired final baseline at end_tick-1.  No smoothing,
    tick rescaling, parameter shaping or original note mutation occurs here.
    """
    cid = _int(controller_id, 'Controller channel', 0, MAX_CHANNEL)
    end = _int(end_tick, 'Song end tick', 1, 0x07ffffff)
    rows = list(points)
    if not rows:
        raise ValueError('A Sylenth controller needs at least one point.')
    checked = []
    previous = -1
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError('Each Sylenth point must contain a tick and normalized value.')
        tick = _int(row[0], 'Point tick', 0, end - 1)
        if tick <= previous:
            raise ValueError('Sylenth point ticks must be unique and strictly increasing.')
        checked.append((tick, quantized_velocity(row[1])))
        previous = tick
    if checked[0][0] != 0:
        raise ValueError('A Sylenth controller must begin at tick 0 with its preset baseline.')
    result = bytearray()
    for index, (tick, velocity) in enumerate(checked):
        stop = checked[index + 1][0] if index + 1 < len(checked) else end
        result.extend(NOTE.pack(tick, 0x4000, cid, stop - tick, 60, 0,
                                120, 0, 64, velocity, 128, 128))
    return bytes(result)


def _channel_blocks(events):
    starts = [i for i, (key, _) in enumerate(events) if key == 64]
    arrangements = [i for i, (key, _) in enumerate(events) if key == 99]
    if not starts or len(arrangements) != 1 or starts[-1] >= arrangements[0]:
        raise ValueError('Use a prepared template with channel records and one Playlist arrangement.')
    blocks = [events[start:stop] for start, stop in zip(starts, starts[1:] + arrangements)]
    return starts, arrangements[0], blocks


def _exemplar(events):
    _, _, blocks = _channel_blocks(events)
    for block in blocks:
        fields = dict(block)
        if _text(fields.get(201, b'')) != FK_NAME:
            continue
        # Reject extra song/pattern data in an exemplar rather than clone it.
        if any(key in {65, 193, 223, 224, 227, 231, 234} for key, _ in block):
            continue
        required = {64: 2, 21: 1, 212: 52, 213: 554, 0: 1, 132: 4, 145: 4}
        if any(sum(key == wanted for key, _ in block) != 1 or len(fields.get(wanted, b'')) != size
               for wanted, size in required.items()):
            continue
        if fields[21] != b'\x02' or fields[212][:12] != struct.pack('<III', 0, 0, 2):
            continue
        if struct.unpack_from('<I', fields[213])[0] != 3:
            continue
        if sum(key == 203 for key, _ in block) != 1:
            continue
        return block
    raise ValueError('This template needs a compatible native Fruity Keyboard Controller exemplar.')


def clone_controller(events, new_id, name, group_id, cut_group):
    """Clone only a validated native generator block, with independent identity."""
    cid = _int(new_id, 'New controller channel', 0, MAX_CHANNEL)
    group = _int(group_id, 'Display group', 0, 0x7fffffff)
    cut = _int(cut_group, 'Independent cut group', 1, 65534)
    if any(key == 64 and int.from_bytes(payload, 'little') == cid for key, payload in events):
        raise ValueError('The new controller channel is already in use.')
    replacements = {64: struct.pack('<H', cid), 203: _name(name),
                    132: struct.pack('<HH', cut, cut), 145: struct.pack('<I', group), 0: b'\x01'}
    result = []
    for key, payload in _exemplar(events):
        if key == 212:
            # Generator wrapper identity words are0,0,2 in every native exemplar;
            # they are NOT the channel ID.  Only suppress the cloned editor popup.
            wrapper = bytearray(payload)
            flags = struct.unpack_from('<I', wrapper, 16)[0]
            struct.pack_into('<I', wrapper, 16, flags & ~1)
            payload = bytes(wrapper)
        result.append((key, replacements.get(key, payload)))
    return result


def prepare_controllers(header, events, requested):
    """Return (new_header, new_events, mappings), adding owned FK channels only.

    Each request has instrument_id, parameter_id and name.  Original events can
    be recovered exactly by deleting tagged additions, except existing links
    may move into native source order.  Header changes only its channel count.
    """
    original = list(events)
    requests = [dict(request) for request in requested]
    if len(header) != 6:
        raise ValueError('The FL header is not a six-byte project header.')
    if not requests:
        return bytes(header), original, []
    starts, arrangement, blocks = _channel_blocks(original)
    ids = [int.from_bytes(dict(block)[64], 'little') for block in blocks]
    if len(set(ids)) != len(ids) or struct.unpack_from('<H', header, 2)[0] != len(ids):
        raise ValueError('Template channel identity/count validation failed.')
    next_id = max(ids) + 1
    if next_id + len(requests) - 1 > MAX_CHANNEL:
        raise ValueError('Too many channels to append independent Sylenth controllers.')
    groups = [i for i, (key, _) in enumerate(original) if key == 231]
    if not groups or groups[-1] >= starts[0]:
        raise ValueError('The prepared template has an unsupported channel display-group layout.')
    group_id = len(groups)
    cut_groups = [v for key, payload in original if key == 132 and len(payload) == 4
                  for v in struct.unpack('<HH', payload)]
    next_cut = max(cut_groups, default=0) + 1
    if next_cut + len(requests) - 1 > 65534:
        raise ValueError('No independent cut groups remain for Sylenth movement.')
    link_rows = [(i, payload) for i, (key, payload) in enumerate(original) if key == 227]
    if any(len(payload) != LINK.size or i >= starts[0] for i, payload in link_rows):
        raise ValueError('The prepared template has an unsupported controller-link layout.')
    existing_targets = {LINK.unpack(payload)[2] for _, payload in link_rows}
    for key, payload in original:
        if key == 226:
            if len(payload) != LINK.size:
                raise ValueError('The prepared template has an unsupported MIDI-controller link layout.')
            existing_targets.add(LINK.unpack(payload)[2])
    channels = {cid: dict(block) for cid, block in zip(ids, blocks)}
    additions, new_links, mappings = [], [], []
    new_targets = set()
    for offset, request in enumerate(requests):
        instrument = request.get('instrument_id')
        parameter = request.get('parameter_id')
        target = target_address(instrument, parameter)
        fields = channels.get(instrument, {})
        if _text(fields.get(201, b'')) != 'Fruity Wrapper' or not (b'Sylenth' in fields.get(213, b'') or 'Sylenth'.encode('utf-16-le') in fields.get(213, b'')):
            raise ValueError('Each new movement destination must be an existing Sylenth1 generator.')
        if target in existing_targets or target in new_targets:
            raise ValueError('A Sylenth parameter already has a controller link; do not add a conflicting movement lane.')
        controller_id, cut_group = next_id + offset, next_cut + offset
        additions.extend(clone_controller(original, controller_id, request.get('name'), group_id, cut_group))
        new_links.append(velocity_link(controller_id, instrument, parameter))
        new_targets.add(target)
        mappings.append(dict(request, controller_id=controller_id, target=target,
                             cut_group=cut_group, group_id=group_id))
    merged_links = [((227, payload), False) for _, payload in link_rows] + [(row, True) for row in new_links]
    merged_links.sort(key=lambda item: LINK.unpack(item[0][1])[0])
    link_insert = link_rows[0][0] if link_rows else starts[0]
    tagged = []
    for index, event in enumerate(original):
        if index == groups[-1] + 1:
            tagged.append(((231, _name(GROUP_NAME)), True))
        if index == link_insert:
            tagged.extend(merged_links)
        if index == arrangement:
            tagged.extend((event, True) for event in additions)
        if event[0] != 227:
            tagged.append((event, False))
    # Deleting additions restores all original non-link events in exact order.
    # Existing link bytes are immutable, but native source sorting can move them.
    recovered = [event for event, added in tagged if not added]
    if ([event for event in recovered if event[0] != 227] !=
            [event for event in original if event[0] != 227] or
            Counter(event for event in recovered if event[0] == 227) !=
            Counter(event for event in original if event[0] == 227)):
        raise ValueError('Controller preparation would change an original event; export stopped.')
    result = [event for event, _ in tagged]
    new_header = bytearray(header)
    struct.pack_into('<H', new_header, 2, len(ids) + len(requests))
    if sum(key == 64 for key, _ in result) != len(ids) + len(requests):
        raise ValueError('Controller preparation produced an invalid channel count.')
    return bytes(new_header), result, mappings
