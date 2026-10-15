"""Append only the requested drum samplers and their verified controller effects.

The small local kit stores native sampler records from the user's saved projects.
Preset envelopes, tuning, key ranges and gain remain exact. New mixer effects use
the same verified plugin types and remote-link addresses as the prepared synth
template; original synths, effects, notes and controller links are preserved.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

from flp_raw import read_fl, read_bytes, encode_fl
from sylenth_events import clone_controller


from app_paths import bundle_root

APP = bundle_root()
KIT_FILE = APP / 'resources' / 'drum-kit.json'
ROLES = ('kick', 'snare', 'clap', 'off_snare', 'closed_hat', 'open_hat', 'tom', 'percussion')
FX = {
    'kick': ('volume',),
    'snare': ('volume', 'delay', 'reverb'),
    'clap': ('volume', 'delay', 'reverb'),
    'off_snare': ('volume', 'reverb'),
    'closed_hat': ('cutoff', 'volume'),
    'open_hat': ('cutoff', 'volume'),
    'tom': ('volume', 'delay', 'reverb'),
    'percussion': ('cutoff', 'volume'),
}
PLUGIN_KEYS = {201, 212, 203, 155, 128, 41, 213}
OPEN_FILTER = struct.pack('<7IB', 2, 1024, 0, 1024, 0, 0, 0, 1)
UNITY_BALANCE = struct.pack('<ii', 0, 256)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _text(data):
    return data.decode('utf-16-le').rstrip('\0')


def _name(value):
    return (value + '\0').encode('utf-16-le')


def _channels(events):
    starts = [i for i, (key, _) in enumerate(events) if key == 64]
    arrangements = [i for i, (key, _) in enumerate(events) if key == 99]
    if not starts or len(arrangements) != 1 or starts[-1] >= arrangements[0]:
        raise ValueError('Drum setup needs a prepared project with one Playlist arrangement.')
    return starts, arrangements[0], [events[a:b] for a, b in zip(starts, starts[1:] + arrangements)]


def _mixer(events):
    counts = [i for i, (key, _) in enumerate(events) if key == 103]
    initial = [i for i, (key, _) in enumerate(events) if key == 225]
    if len(counts) != 1 or len(initial) != 1 or counts[0] >= initial[0]:
        raise ValueError('The template has an unsupported mixer layout.')
    begin = counts[0] + 1
    blocks = []
    for index in range(begin, initial[0]):
        if events[index][0] == 147:
            blocks.append((begin, index + 1, events[begin:index + 1]))
            begin = index + 1
    if begin != initial[0] or len(blocks) != int.from_bytes(events[counts[0]][1], 'little'):
        raise ValueError('The template mixer blocks do not match their saved count.')
    return blocks, initial[0]


def _effects(events, blocks):
    result = []
    for insert, (start, stop, _) in enumerate(blocks):
        index = start
        while index < stop:
            if events[index][0] != 201:
                index += 1
                continue
            first = index
            while index < stop and events[index][0] in PLUGIN_KEYS:
                index += 1
            record = events[first:index]
            fields = dict(record)
            if (index == stop or events[index][0] != 98 or len(fields.get(212, b'')) != 52
                    or any(sum(k == x for k, _ in record) != 1 for x in (201, 212, 213, 155, 128, 41))):
                raise ValueError('A mixer plugin has an unsupported native record.')
            route, slot = struct.unpack_from('<II', fields[212])
            if route != insert or slot != int.from_bytes(events[index][1], 'little'):
                raise ValueError('A mixer plugin address differs from its native slot.')
            result.append({'insert': insert, 'slot': slot, 'plugin': _text(fields[201]),
                           'record': record, 'state': fields[213]})
    return result




def load_kit(kit_file=KIT_FILE):
    data = json.loads(Path(kit_file).read_text(encoding='utf-8'))
    if data.get('version') != 1 or not isinstance(data.get('samples'), dict):
        raise ValueError('The included drum kit metadata is unsupported.')
    for role, sample in data['samples'].items():
        relative = Path(sample['sample'])
        path = (APP / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(APP):
            raise ValueError('A drum sample path leaves the application folder.')
        if not path.is_file() or _sha(path.read_bytes()) != sample['sample_sha256']:
            raise ValueError(f'The included {role} drum sample is missing or changed.')
    return data


def prepare_drums(header, events, slots, kit_file=KIT_FILE):
    """Return (header, events, manifest) for 1..8 user-assigned drum slots.

    slots contains {'slot':1..8,'role':one of ROLES}, in any input order.
    Role order is unrestricted. Retune split input hits to playback_note, which
    can differ from root_note in a saved user sampler (the clap is one example).
    Only requested roles receive plugins; unused kit sounds are never inserted.
    """
    original = list(events)
    requested = [dict(item) for item in slots]
    if not requested:
        return bytes(header), original, {'slots': [], 'controllers': []}
    if len(header) != 6 or not 1 <= len(requested) <= 8:
        raise ValueError('Choose between one and eight independent drum slots.')
    if any(isinstance(item.get('slot'), bool) or not isinstance(item.get('slot'), int) for item in requested):
        raise ValueError('Each drum slot needs its own integer position.')
    requested.sort(key=lambda item: item['slot'])
    if [item['slot'] for item in requested] != list(range(1, len(requested) + 1)):
        raise ValueError('Drum slots must be consecutive and start at slot 1.')
    if any(item.get('role') not in ROLES for item in requested):
        raise ValueError('Choose a supported role for every drum slot.')
    kit = load_kit(kit_file)
    starts, arrangement, channels = _channels(original)
    ids = [int.from_bytes(dict(block)[64], 'little') for block in channels]
    if len(ids) != len(set(ids)) or struct.unpack_from('<H', header, 2)[0] != len(ids):
        raise ValueError('The prepared template has inconsistent channel identities.')
    fk_count = sum(len(FX[item['role']]) for item in requested)
    next_channel = max(ids) + 1
    if next_channel + len(requested) + fk_count > 4095:
        raise ValueError('The prepared template has too many channels for this drum setup.')
    groups = [i for i, (key, _) in enumerate(original) if key == 231]
    if not groups or groups[-1] >= starts[0]:
        raise ValueError('The template display groups use an unsupported layout.')
    group_id = len(groups)
    links = [(i, payload) for i, (key, payload) in enumerate(original) if key == 227]
    if any(len(payload) != 20 or i >= starts[0] for i, payload in links):
        raise ValueError('The template controller links use an unsupported layout.')
    linked_routes = set()
    for key, payload in original:
        if key in {226, 227}:
            if len(payload) != 20:
                raise ValueError('The template has an unsupported existing controller link.')
            target = struct.unpack_from('<I', payload, 8)[0]
            if target >> 28 == 7:
                linked_routes.add((target & 0x0fc00000) >> 22)
    blocks, init_index = _mixer(original)
    effects = _effects(original, blocks)
    occupied_routes = {int.from_bytes(dict(block).get(104, b'\0\0'), 'little') for block in channels}
    occupied_routes |= {effect['insert'] for effect in effects} | linked_routes
    free = []
    for insert in range(9, min(64, len(blocks) - 1)):
        if insert in occupied_routes:
            continue
        fields = dict(blocks[insert][2])
        if _text(fields.get(204, b'')) or fields.get(235) != b'\x01' or fields.get(154) != b'\xff' * 4:
            continue
        if [int.from_bytes(p, 'little') for k, p in blocks[insert][2] if k == 98] != list(range(10)):
            continue
        if any(len(dict(other).get(235, b'')) > insert and dict(other)[235][insert] for _, _, other in blocks):
            continue
        free.append(insert)
    if len(free) < len(requested):
        raise ValueError('The prepared template needs more unused mixer inserts for the requested drum slots.')
    references = {}
    for lane, plugin in [('volume', 'Fruity Balance'), ('cutoff', 'Fruity Filter'), ('delay', 'Fruity Delay 3')]:
        candidates = [effect for effect in effects if effect['plugin'] == plugin]
        if candidates:
            references[lane] = candidates[0]
    reverbs = [effect for effect in effects if effect['plugin'] == 'Fruity Wrapper' and b'ValhallaFutureVerb' in effect['state']]
    if reverbs:
        references['reverb'] = reverbs[0]
    required_lanes = {lane for item in requested for lane in FX[item['role']]}
    if required_lanes - references.keys():
        raise ValueError('The prepared template is missing one of the verified drum effect types.')
    cut_values = [v for key, payload in original if key == 132 for v in struct.unpack('<HH', payload)]
    next_cut = max(cut_values, default=0) + 1
    if next_cut + fk_count > 65535:
        raise ValueError('No independent controller cut groups remain.')
    additions_at = {}
    new_channels, new_controllers, new_links, controller_rows, slot_rows = [], [], [], [], []
    required_initial = {}
    next_fk = next_channel + len(requested)
    for offset, request in enumerate(requested):
        role, slot_number = request['role'], request['slot']
        source_role = kit.get('role_aliases', {}).get(role, role)
        sample = kit['samples'][source_role]
        native_block = [(int(key), bytes.fromhex(payload)) for key, payload in sample['events']]
        fields = dict(native_block)
        if fields.get(21) != b'\0' or any(key in {65, 94, 223, 224, 227, 234} for key, _ in native_block):
            raise ValueError('The included drum sampler contains unsupported song relationships.')
        channel_id, insert = next_channel + offset, free[offset]
        name = f'Drum {slot_number}'
        path = (APP / sample['sample']).resolve()
        replacements = {64: struct.pack('<H', channel_id), 203: _name(name),
                        104: struct.pack('<H', insert), 145: struct.pack('<I', group_id),
                        196: _name(str(path))}
        if any(sum(key == wanted for key, _ in native_block) != 1 for wanted in replacements):
            raise ValueError('A source sampler is missing an expected native setting.')
        copied = [(key, replacements.get(key, payload)) for key, payload in native_block]
        if [(k, p) for k, p in copied if k not in replacements] != [(k, p) for k, p in native_block if k not in replacements]:
            raise ValueError('Copying a drum would change its sound settings.')
        new_channels.extend(copied)
        block_start, _, mixer_block = blocks[insert]
        label_at = next(block_start + i for i, (key, _) in enumerate(mixer_block) if key == 236)
        additions_at.setdefault(label_at, []).extend([(204, _name(name + ' - ' + role.replace('_', ' '))), (149, bytes.fromhex('bd896600'))])
        positions = {int.from_bytes(payload, 'little'): block_start + i for i, (key, payload) in enumerate(mixer_block) if key == 98}
        own_controls = []
        for effect_slot, lane in enumerate(FX[role]):
            reference = references[lane]
            record = []
            for key, payload in reference['record']:
                if key == 212:
                    value = bytearray(payload)
                    struct.pack_into('<II', value, 0, insert, effect_slot)
                    struct.pack_into('<I', value, 16, struct.unpack_from('<I', value, 16)[0] & ~1)
                    payload = bytes(value)
                elif key == 213 and lane == 'volume':
                    if len(payload) != 8:
                        raise ValueError('The Fruity Balance state layout is unsupported.')
                    payload = UNITY_BALANCE
                elif key == 213 and lane == 'cutoff':
                    if len(payload) != 29:
                        raise ValueError('The Fruity Filter state layout is unsupported.')
                    payload = OPEN_FILTER
                record.append((key, payload))
            additions_at.setdefault(positions[effect_slot], []).extend(record)
            prefix = 0x70000000 | (insert << 22) | (effect_slot << 16)
            parameter = 0x8001 if lane == 'volume' else 0x8000 if lane == 'cutoff' else 0x1f01
            target = prefix | parameter
            required_initial[prefix | 0x1f00] = 1
            required_initial[prefix | 0x1f01] = 0 if lane in {'delay', 'reverb'} else 12800
            controller_id = next_fk
            next_fk += 1
            controller_name = f'{name} CTRL {lane} drums'
            new_controllers.extend(clone_controller(original, controller_id, controller_name, group_id + 1, next_cut))
            next_cut += 1
            new_links.append((227, struct.pack('<5I', (controller_id << 16) | 0x8001, 0, target, 8, 469)))
            item = {'controller_id': controller_id, 'instrument_id': channel_id, 'name': controller_name,
                    'lane': lane, 'mixer_insert': insert, 'effect_slot': effect_slot,
                    'plugin': reference['plugin'], 'target': f'{target:08x}'}
            own_controls.append(item)
            controller_rows.append(item)
        slot_rows.append({'slot': slot_number, 'role': role, 'name': name, 'channel_id': channel_id,
                         'playback_note': sample['playback_note'], 'root_note': sample['root_note'],
                         'key_range': sample['key_range'], 'mixer_insert': insert,
                         'sample': str(path), 'sample_sha256': sample['sample_sha256'],
                         'sample_source_role': source_role, 'starter_alias': role != source_role,
                         'channel_gain': sample['channel_gain'], 'controllers': own_controls})
    initial = original[init_index][1]
    if len(initial) % 12:
        raise ValueError('The template initialization values use an unsupported layout.')
    changed_initial, records, seen = [], [], set()
    used_inserts = set(free[:len(requested)])
    for position in range(0, len(initial), 12):
        record = initial[position:position + 12]
        target, value = struct.unpack_from('<II', record, 4)
        if target >> 28 == 7 and (target & 0x0fc00000) >> 22 in used_inserts and target & 0xffff >= 0x8000:
            raise ValueError('An unused drum insert retains old plugin automation initializers.')
        if target in required_initial:
            if target in seen:
                raise ValueError('A drum effect slot has duplicate initialization values.')
            seen.add(target)
            replacement = required_initial[target]
            if value != replacement:
                changed_initial.append({'target': f'{target:08x}', 'before': value, 'after': replacement})
                record = record[:8] + struct.pack('<I', replacement)
        records.append(record)
    for target in sorted(required_initial.keys() - seen):
        records.append(struct.pack('<III', 0, target, required_initial[target]))
        changed_initial.append({'target': f'{target:08x}', 'before': None, 'after': required_initial[target]})
    tagged_links = [((227, payload), False) for _, payload in links] + [(item, True) for item in new_links]
    tagged_links.sort(key=lambda item: struct.unpack_from('<I', item[0][1])[0])
    first_link = links[0][0] if links else starts[0]
    tagged = []
    for index, event in enumerate(original):
        if index == groups[-1] + 1:
            tagged.extend([((231, _name('Drums')), True), ((231, _name('Drum automation')), True)])
        if index == first_link:
            tagged.extend(tagged_links)
        if index == arrangement:
            tagged.extend((item, True) for item in new_channels)
            tagged.extend((item, True) for item in new_controllers)
        tagged.extend((item, True) for item in additions_at.get(index, []))
        if event[0] != 227:
            tagged.append(((225, b''.join(records)) if index == init_index else event, False))
    recovered = [event for event, added in tagged if not added]
    if ([item for item in recovered if item[0] not in {225, 227}] != [item for item in original if item[0] not in {225, 227}]
            or Counter(item for item in recovered if item[0] == 227) != Counter(item for item in original if item[0] == 227)):
        raise ValueError('Drum setup would modify an original instrument or mixer record.')
    result = [event for event, _ in tagged]
    new_header = bytearray(header)
    struct.pack_into('<H', new_header, 2, len(ids) + len(requested) + fk_count)
    if read_bytes(encode_fl(bytes(new_header), result)) != (bytes(new_header), result):
        raise ValueError('The prepared drum project failed its lossless container check.')
    if sum(key == 64 for key, _ in result) != len(ids) + len(requested) + fk_count:
        raise ValueError('The prepared drum project has an invalid channel count.')
    new_blocks, _ = _mixer(result)
    after_effects = _effects(result, new_blocks)
    after_lookup = {(effect['insert'], effect['slot']): effect for effect in after_effects}
    if any(after_lookup[(effect['insert'], effect['slot'])]['record'] != effect['record'] for effect in effects):
        raise ValueError('An original mixer plugin changed during drum preparation.')
    if len(after_effects) != len(effects) + fk_count:
        raise ValueError('The prepared drum effect count does not match the controller lanes.')
    return bytes(new_header), result, {'slots': slot_rows, 'controllers': controller_rows,
            'channel_count_before': len(ids), 'channel_count_after': len(ids) + len(requested) + fk_count,
            'samplers_added': len(requested), 'effects_added': fk_count, 'links_added': fk_count,
            'initial_state_changes': changed_initial, 'existing_instruments_and_effects_byte_identical': True,
            'existing_link_payloads_byte_identical': True, 'source_sampler_sound_settings_byte_identical': True,
            'native_FL_playback_verified': False, 'kit_notice': kit['notice']}
