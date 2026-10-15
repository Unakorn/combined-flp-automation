"""Add owned automation to a saved FLP; retain its music and plugin states.

The input FLP supplies every musical instrument and the entire arrangement.
Only effect/controller definitions are packaged as assets. This module never
loads the legacy template, imports a sound preset, or rebuilds musical notes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct

from mido import MidiFile, MidiTrack, Message, MetaMessage
from flp_raw import read_bytes, encode_fl
from flp_export_files import create_export_stage as _export_stage
from flp_automation import ExportError, _options, _number, _note_bytes, _clip_bytes
from drum_template import _mixer, PLUGIN_KEYS, OPEN_FILTER
import velocity_automation_midi as movement
from sylenth_presets import inspect_instruments
from sylenth_events import encode_controller_points, quantized_velocity

APP_VERSION = '3.0.0'
from app_paths import bundle_root

APP = bundle_root()
CATALOG = APP / 'resources' / 'automation-effects.json'
MAX_SIZE = 256 * 1024 * 1024


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _text(data):
    return data.decode('utf-16-le', errors='strict').rstrip('\0')


def _name(value):
    return (str(value)[:190] + '\0').encode('utf-16-le')


def _coalesce_control_notes(payload):
    """Join redundant touching holds in an owned velocity-controller lane.

    Keep all value changes, gaps and other note properties. This must never
    process musical notes: retriggers matter there. The smaller controller
    payload avoids storing redundant repeated control updates.
    """
    if len(payload) % 24:
        raise ExportError('A generated control lane has an incomplete note record.')
    records = []
    for offset in range(0, len(payload), 24):
        record = payload[offset:offset + 24]
        tick, = struct.unpack_from('<I', record)
        if records:
            previous = records[-1]
            previous_tick, = struct.unpack_from('<I', previous)
            previous_length, = struct.unpack_from('<I', previous, 8)
            if (previous_tick + previous_length == tick
                    and previous[4:8] == record[4:8]
                    and previous[12:] == record[12:]):
                length, = struct.unpack_from('<I', record, 8)
                merged = bytearray(previous)
                struct.pack_into('<I', merged, 8, previous_length + length)
                records[-1] = bytes(merged)
                continue
        records.append(record)
    return b''.join(records)


def _section_control_notes(payloads, ppq, end_tick):
    """Place owned controller notes into eight-bar, pattern-local sections.

    Touching sections split held notes without changing their value, other
    attributes, or gaps. Empty sections are omitted. Musical notes never pass
    through this helper. The returned positions are global song ticks.
    """
    if (not isinstance(ppq, int) or isinstance(ppq, bool) or ppq <= 0
            or not isinstance(end_tick, int) or isinstance(end_tick, bool)
            or not 0 < end_tick <= 0xffffffff):
        raise ExportError('Controller sections need a valid song time resolution and duration.')
    from section_chunks import chunk_length
    section_length = chunk_length(ppq)  # Shared with streamed stem analysis.
    if (end_tick + section_length - 1) // section_length > 999:
        raise ExportError('The song needs more automation sections than FL Studio pattern space allows.')
    sections = defaultdict(list)
    previous_ends = {}
    for payload in payloads:
        if len(payload) % 24:
            raise ExportError('A generated control lane has an incomplete note record.')
        for offset in range(0, len(payload), 24):
            record = payload[offset:offset + 24]
            tick = struct.unpack_from('<I', record)[0]
            duration = struct.unpack_from('<I', record, 8)[0]
            cid = struct.unpack_from('<H', record, 6)[0]
            finish = tick + duration
            if not duration or finish > end_tick or tick < previous_ends.get(cid, 0):
                raise ExportError('A generated control lane has overlapping, unordered or out-of-song notes.')
            previous_ends[cid] = finish
            for index in range(tick // section_length, (finish - 1) // section_length + 1):
                start = index * section_length
                a, b = max(tick, start), min(finish, start + section_length)
                row = bytearray(record)
                struct.pack_into('<I', row, 0, a - start)
                struct.pack_into('<I', row, 8, b - a)
                sections[index].append(bytes(row))
    result = []
    for index, rows in sorted(sections.items()):
        rows.sort(key=lambda row: (struct.unpack_from('<I', row)[0],
                                   -struct.unpack_from('<H', row, 6)[0]))
        position = index * section_length
        result.append(dict(position=position, duration=min(section_length, end_tick - position),
                           payload=b''.join(rows)))
    return result


def _is_kick(label):
    return bool(re.search(r'(?:^|[^a-z])(?:kick[a-z]?|kck)(?:$|[^a-z])', label, re.I))


def _synth_event_policy(events):
    """Return (block synth movement, found verified tempo events).

    Native event223 uses 12-byte tick/target/value records. The one exempt
    target is the verified global tempo address0x40000005, whose integer value
    is BPM*1000. Every unknown layout, address or value keeps the conservative
    block. This inspection never rewrites existing automation.
    """
    blocked = tempo_only = False
    for key, payload in events:
        if key != 223 or not payload:
            continue
        if len(payload) % 12:
            blocked = True
            continue
        for _, target, value in struct.iter_unpack('<III', payload):
            if target == 0x40000005 and 10000 <= value <= 522000:
                tempo_only = True
            else:
                blocked = True
    return blocked, tempo_only


def _saved_drum_split(channel):
    """Recognize the native saved MAIN0 / OUT2+1 map without rewriting it.

    Native Drum Monkey projects use eight 12-byte output entries in wrapper
    chunk32. The verified saved split has(0,1,0),(1,1,0), with the other outputs
    inactive. FL output offsets follow the channel's mixer assignment. Moving
    only event104 therefore preserves the existing kit and relative split.
    Unknown maps/pad buses are not guessed or patched.
    """
    from drum_detection import _tree, _preset_kit, drum_role
    state = channel['state']
    wrapper = channel['fields'].get(212, b'')
    if (channel['plugin'].casefold() != 'fruity wrapper' or len(wrapper) != 52
            or wrapper[:12] != struct.pack('<3I', 0, 0, 2)
            or state[:4] != struct.pack('<I', 12)):
        raise ExportError('The saved Drum Monkey wrapper layout is not supported for automatic routing.')
    chunks, at = {}, 4
    while at < len(state):
        if len(state) - at < 12:
            raise ExportError('The saved Drum Monkey wrapper chunks are incomplete.')
        key, size = struct.unpack_from('<IQ', state, at)
        at += 12
        if key in chunks or size > len(state) - at:
            raise ExportError('The saved Drum Monkey wrapper chunks are ambiguous.')
        chunks[key] = state[at:at + size]
        at += size
    if (chunks.get(54) != b'Unison Drum Monkey'
            or chunks.get(30) != struct.pack('<4I', 0, 8, 0, 0)
            or len(chunks.get(32, b'')) != 96):
        raise ExportError('The saved Drum Monkey output layout is not supported for automatic routing.')
    outputs = list(struct.iter_unpack('<iii', chunks[32]))
    if (outputs[:2] != [(0, 1, 0), (1, 1, 0)]
            or any(row not in {(i, 0, 0), (0, 0, 0)} for i, row in enumerate(outputs[2:], 2))):
        raise ExportError('Save Drum Monkey with MAIN at output offset0 and OUT2 at offset1; its current output map cannot be routed automatically yet.')
    plugin_state = chunks.get(53, b'')
    if plugin_state.count(b'DrumMonkey\0') != 1:
        raise ExportError('The saved Drum Monkey kit tree is ambiguous.')
    try:
        tree = _tree(plugin_state, plugin_state.index(b'DrumMonkey\0'))
        parameters = [n for n in tree['children'] if n['type'] == 'Parameters']
        engines = [n for n in tree['children'] if n['type'] == 'DrumSynthesiser']
        if len(parameters) != 1 or len(engines) != 1:
            raise ValueError('Ambiguous pad groups')
        pads = [n for n in parameters[0]['children'] if n['type'] == 'Pad']
        sounds = [n for n in engines[0]['children'] if n['type'] == 'Sound']
        if len(pads) != 8 or len(sounds) != 8:
            raise ValueError('Unsupported pad count')
        active, kicks = [], []
        for index, (pad, sound) in enumerate(zip(pads, sounds), 1):
            p, s = pad['properties'], sound['properties']
            if p.get('id') != f'Pad {index}' or s.get('name') != f'Pad {index}' or s.get('On') not in {'Yes', 'No'}:
                raise ValueError('Unverified pad identity')
            if s['On'] == 'No':
                continue
            role = drum_role(p.get('kitPiece', ''))
            if role == 'kick':
                if s.get('Bus') != 'OUT 2':
                    raise ValueError('Kick is not isolated on OUT2')
                kicks.append(index)
            elif s.get('Bus') != 'MAIN':
                raise ValueError('Another active pad is not on MAIN')
            active.append(index)
        # The current curve detector also uses the first/lowest drum lane as
        # kick. Refuse a differently ordered kit until that mapping is proven.
        if kicks != [1] or len(active) < 2:
            raise ValueError('The supported split needs one first-pad kick and other active drums')
        kit = _preset_kit(state, 'routing', channel['name'])
        if not kit['can_apply'] or kit['anchor_pad'] != 1:
            raise ValueError('The first kick must have the lowest unique readable pad note')
    except (ValueError, KeyError, UnicodeError) as exc:
        raise ExportError('Save the first Drum Monkey kick on OUT2 and the other active drums on MAIN; the current kit split cannot be routed automatically yet.') from exc
    return dict(status='auto_saved_split', drums_insert=29, kick_insert=30,
                original_drums_insert=channel['mixer_insert'], original_kick_insert=channel['mixer_insert'] + 1,
                main_output_offset=0, kick_output_offset=1, kick_pad=1,
                active_pads=active, plugin_state_unchanged=True, wrapper_output_map_unchanged=True,
                assignment_change_required=channel['mixer_insert'] != 29)


def _audio_output_routes(channel):
    """Conservatively identify saved outputs that could occupy drum inserts."""
    if channel['type'] in {0, 4}:
        return {channel['mixer_insert']}
    if channel['type'] != 2 or channel['plugin'].casefold() in {
            'fruity keyboard controller', 'fruity envelope controller', 'midi out', 'dashboard'}:
        return set()
    state = channel['state']
    if channel['plugin'].casefold() != 'fruity wrapper' or state[:4] != struct.pack('<I', 12):
        raise ExportError('Another instrument has an unverified output layout; drum routing was skipped to avoid an output collision.')
    at, chunks = 4, {}
    while at < len(state):
        if len(state) - at < 12:
            raise ExportError('Another instrument has an incomplete output layout; drum routing was skipped.')
        key, size = struct.unpack_from('<IQ', state, at)
        at += 12
        if key in chunks or size > len(state) - at:
            raise ExportError('Another instrument has an ambiguous output layout; drum routing was skipped.')
        chunks[key] = state[at:at + size]
        at += size
    io, output = chunks.get(30, b''), chunks.get(32, b'')
    if len(io) != 16 or not output or len(output) % 12 or len(output) // 12 != struct.unpack_from('<I', io, 4)[0]:
        raise ExportError('Another instrument has an unverified output map; drum routing was skipped.')
    routes = set()
    for offset, enabled, reserved in struct.iter_unpack('<iii', output):
        if enabled not in {0, 1} or reserved != 0 or not -500 <= offset <= 500:
            raise ExportError('Another instrument has an unverified output map; drum routing was skipped.')
        # Include even inactive output offsets: a wrapper can process inactive
        # plugin buses, so treating those destinations as free is unnecessary.
        routes.add(channel['mixer_insert'] + offset)
    return routes


def _channels(events):
    starts = [i for i, (k, _) in enumerate(events) if k == 64]
    arrangements = [i for i, (k, _) in enumerate(events) if k == 99]
    if not starts or len(arrangements) != 1 or starts[-1] >= arrangements[0]:
        raise ExportError('Save one Playlist arrangement in this FLP first.')
    rows = []
    for a, b in zip(starts, starts[1:] + arrangements):
        d = dict(events[a:b])
        rows.append(dict(id=int.from_bytes(d[64], 'little'), start=a, stop=b,
                         name=_text(d.get(203, b'')), plugin=_text(d.get(201, b'')),
                         type=int.from_bytes(d.get(21, b'\xff'), 'little'),
                         mixer_insert=int.from_bytes(d.get(104, b'\0\0'), 'little'),
                         state=d.get(213, b''), fields=d))
    return rows, arrangements[0]


def _effects(events, blocks):
    """Inspect slots without interpreting third-party plugin state.

    FL's final Current mixer block has address501, not its ordinal position.
    It is never eligible for effect insertion.
    """
    result = []
    for ordinal, (start, stop, block) in enumerate(blocks):
        index = start
        while index < stop:
            if events[index][0] != 201:
                index += 1
                continue
            first = index
            while index < stop and events[index][0] in PLUGIN_KEYS:
                index += 1
            fields = dict(events[first:index])
            if index == stop or events[index][0] != 98 or len(fields.get(212, b'')) != 52:
                raise ExportError('A saved mixer effect uses an unsupported slot layout.')
            route, slot = struct.unpack_from('<II', fields[212])
            if slot != int.from_bytes(events[index][1], 'little') or not (
                    route == ordinal or ordinal == len(blocks) - 1 and route == 501):
                raise ExportError('A saved mixer effect has an unsupported address.')
            result.append(dict(insert=route, slot=slot, plugin=_text(fields[201]), name=_text(fields.get(203, fields[201]))))
    return result


def _layout(events):
    pattern_ids = {int.from_bytes(p, 'little') for k, p in events if k == 65}
    playlists = [(i, p) for i, (k, p) in enumerate(events) if k == 233]
    if len(playlists) != 1:
        raise ExportError('This version needs one saved Playlist arrangement.')
    for size in (88, 60, 32):
        payload = playlists[0][1]
        if not payload or len(payload) % size:
            continue
        clips = [payload[j:j + size] for j in range(0, len(payload), size)]
        if all(struct.unpack_from('<H', r, 4)[0] == 0x5000 and
               (struct.unpack_from('<H', r, 6)[0] <= 0x5000 or
                struct.unpack_from('<H', r, 6)[0] - 0x5000 in pattern_ids) and
               struct.unpack_from('<H', r, 12)[0] <= 499 for r in clips):
            example = next((r for r in clips if struct.unpack_from('<H', r, 6)[0] > 0x5000), None)
            if example is not None:
                return dict(pattern_ids=pattern_ids, playlist_index=playlists[0][0],
                            clip=example, size=size, clips=clips,
                            next_row=max(499 - struct.unpack_from('<H', r, 12)[0] for r in clips) + 1,
                            serial=max((struct.unpack_from('<I', r, 32)[0] for r in clips), default=0) + 1 if size >= 60 else 0)
    raise ExportError('The Playlist clip format is not supported yet.')


def _load(path):
    from flp_song_analysis import analyze_flp, read_project
    path = Path(path).expanduser().resolve()
    if path.suffix.casefold() != '.flp' or not path.is_file() or path.stat().st_size > MAX_SIZE:
        raise ExportError('Choose a saved FLP project smaller than 256 MB.')
    data = path.read_bytes()
    header, events = read_bytes(data)
    if encode_fl(header, events) != data:
        raise ExportError('This FLP cannot be preserved exactly by the current reader.')
    project = read_project(path, data_override=data, allow_single_pattern=True)
    model = analyze_flp(path, max_instruments=20, data_override=data, allow_single_pattern=True)
    if path.read_bytes() != data or model['sha256'] != _sha(data):
        raise ExportError('The FLP changed during inspection. Save it, then load it again.')
    channels, arrangement = _channels(events)
    if len(channels) != struct.unpack_from('<H', header, 2)[0]:
        raise ExportError('The saved FLP channel count is inconsistent.')
    blocks, init_index = _mixer(events)
    effects = _effects(events, blocks)
    inferred_clip = project['layout'].get('inferred_clip') or b''
    layout_events = events
    if inferred_clip:
        layout_events = list(events)
        index = project['layout']['playlist_index']
        layout_events[index] = (233, inferred_clip)
    layout = _layout(layout_events)
    layout['inferred_clip'] = inferred_clip
    layout['inferred_playlist'] = project.get('inferred_playlist')
    return path, data, header, events, model, channels, arrangement, blocks, init_index, effects, layout


def _plan(loaded, effect_lanes=None):
    effect_lanes = tuple(('cutoff', 'delay', 'reverb') if effect_lanes is None else effect_lanes)
    if len(set(effect_lanes)) != len(effect_lanes) or any(x not in ('cutoff', 'delay', 'reverb') for x in effect_lanes):
        raise ExportError('Choose filter, delay and/or reverb without duplicate lanes.')
    _, _, _, events, model, channels, _, blocks, init_index, effects, layout = loaded
    by_id = {c['id']: c for c in channels}
    voices = model['voices']
    warnings = list(model.get('warnings', []))
    initial = events[init_index][1]
    if len(initial) % 12:
        raise ExportError('The saved mixer initialization format is unsupported.')
    initial_values = {struct.unpack_from('<I', initial, o + 4)[0]: struct.unpack_from('<I', initial, o + 8)[0]
                      for o in range(0, len(initial), 12)}
    occupied = defaultdict(set)
    for effect in effects:
        occupied[effect['insert']].add(effect['slot'])
    source_routes = {c['mixer_insert'] for c in channels if c['type'] in (0, 2, 4)}
    # A multi-output instrument can already use inserts beyond its main one.
    # Keep those buses free of automatically assigned synth effects, even if
    # a later drum-routing check cannot move the instrument to inserts29/30.
    output_routes = set(source_routes)
    unknown_outputs = False
    for channel in channels:
        try:
            output_routes.update(_audio_output_routes(channel))
        except ExportError:
            unknown_outputs = True
    linked = set()
    addressed_slots = defaultdict(set)
    for target in initial_values:
        if target >> 28 == 7 and target & 0xffff >= 0x8000:
            addressed_slots[(target >> 22) & 63].add((target >> 16) & 63)
    for k, p in events:
        if k in (226, 227):
            if len(p) != 20:
                raise ExportError('An existing controller link uses an unsupported format.')
            target = struct.unpack_from('<I', p, 8)[0]
            if target >> 28 == 7:
                linked.add((target >> 22) & 63)
                addressed_slots[(target >> 22) & 63].add((target >> 16) & 63)
        elif k == 223:
            # Conservatively reserve any native event word resembling a mixer
            # destination, even in a format we do not otherwise interpret.
            for offset in range(0, len(p) - 3, 4):
                target = struct.unpack_from('<I', p, offset)[0]
                if target >> 28 == 7:
                    addressed_slots[(target >> 22) & 63].add((target >> 16) & 63)
    free = []
    for insert in range(1, min(len(blocks) - 1, 64)):
        d = dict(blocks[insert][2])
        prefix = 0x70000000 | insert << 22
        incoming = any(len(dict(b).get(235, b'')) > insert and dict(b)[235][insert] for _, _, b in blocks)
        if (not unknown_outputs and insert not in output_routes | linked | {29, 30} and not occupied[insert] and not incoming
                and not _text(d.get(204, b'')) and d.get(235) == b'\x01'
                and d.get(154) == b'\xff' * 4
                and d.get(147) == b'\xff' * 4 and d.get(42) == b'\0'
                and d.get(236) == bytes.fromhex('000000004c00000000000000')
                and initial_values.get(prefix | 0x1fc0, 12800) == 12800
                and all(initial_values.get(prefix | parameter, 0) == 0
                        for parameter in (0x1fc1, 0x1fc2, 0x1fd0, 0x1fd1, 0x1fd2))):
            free.append(insert)
    profiles = inspect_instruments(events, channels)
    profile_by_id = {p['instrument_id']: p for p in profiles['instruments']}
    warnings.extend(profiles['warnings'])
    # Tempo is global, so its verified event address cannot compete with a
    # Sylenth knob. Every other native event keeps the conservative exclusion.
    native_events, tempo_events = _synth_event_policy(events)
    if native_events:
        warnings.append('Existing event automation is preserved; additional internal synth movement is disabled to avoid competing controls.')
    elif tempo_events:
        warnings.append('Existing tempo event automation is retained unchanged; tempo-only events do not block internal Sylenth movement.')
    if any(k == 231 and _text(p) == 'FLP Connector automation' for k, p in events):
        raise ExportError('This is already a connector output. Choose your original saved song to generate a different intensity without stacking the effects twice.')
    rows, groups, changes = [], defaultdict(list), {}

    def neutral_insert(insert):
        if not 1 <= insert < min(64, len(blocks) - 1) or occupied[insert] or insert in linked:
            return False
        d = dict(blocks[insert][2])
        prefix = 0x70000000 | insert << 22
        # Labels are display-only. All saved audible settings must be neutral
        # before moving sound away from its original insert or into a new one.
        return (d.get(235) == b'\x01' and d.get(154) == b'\xff' * 4
                and d.get(147) == b'\xff' * 4
                and d.get(236) == bytes.fromhex('000000004c00000000000000')
                and initial_values.get(prefix | 0x1fc0, 12800) == 12800
                and all(initial_values.get(prefix | parameter, 0) == 0
                        for parameter in (0x1fc1, 0x1fc2, 0x1fd0, 0x1fd1, 0x1fd2))
                and not addressed_slots[insert])

    for voice in voices:
        cid = voice['channel_id']
        channel = by_id[cid]
        insert = channel['mixer_insert']
        monkey = b'DrumMonkey' in channel['state'] or b'Drum Monkey' in channel['state'] or 'drum monkey' in channel['name'].lower()
        kick = insert == 30 or _is_kick(channel['name'])
        reason = ''
        drum_routing = None
        if monkey:
            try:
                drum_routing = _saved_drum_split(channel)
                if len(blocks) <= 31:
                    raise ExportError('This FLP has no saved ordinary inserts29 and30.')
                for target in (29, 30):
                    conflicting = [c for c in channels if c['id'] != cid and c['type'] in (0, 2, 4)
                                   and target in _audio_output_routes(c)]
                    incoming = any(len(dict(b).get(235, b'')) > target and dict(b)[235][target]
                                   for _, _, b in blocks)
                    if conflicting or incoming:
                        raise ExportError(f'Insert{target} already receives another source; drum routing and effects were skipped.')
                    if insert != 29 and not neutral_insert(target):
                        raise ExportError(f'Insert{target} has saved processing or mix settings; drum routing and effects were skipped.')
                if insert != 29:
                    # A new direct-to-Master bus still passes through the saved
                    # Master chain. Only the old separate kick bus is bypassed
                    # when the instrument's main output was already Master.
                    if (insert != 0 and not neutral_insert(insert)) or not neutral_insert(insert + 1):
                        raise ExportError('The old drum/kick inserts have processing or mix settings that cannot be bypassed automatically; drum effects were skipped.')
                    changes[cid] = 29
                insert = 29
                warnings.append(f"{channel['name']}: automatically assigned to insert29; its saved OUT2+1 kick output follows to insert30. Kit sounds and the saved output map stay unchanged.")
            except ExportError as exc:
                reason = str(exc)
                drum_routing = dict(status='unsupported', drums_insert=29, kick_insert=30,
                                    reason=reason, assignment_change_required=False)
                warnings.append(f"{channel['name']}: {reason}")
        elif kick:
            reason = 'Kick kept steady; no added automation.'
        elif insert == 0:
            # Only proven stereo instruments may move from the Master to an
            # unused direct-to-Master insert. Arbitrary wrappers may have
            # relative multi-output maps, so their routing is never guessed.
            stereo = channel['type'] == 0 or cid in profile_by_id
            if stereo and free:
                insert = free.pop(0)
                changes[cid] = insert
                warnings.append(f"{channel['name']}: assigned unused mixer insert {insert}, feeding the Master, for its own effects.")
            else:
                reason = 'Assign this instrument to its own mixer insert to add mixer effects.'
        elif not 1 <= insert < min(64, len(blocks) - 1):
            reason = 'Mixer effects require a saved ordinary insert between 1 and 63.'
        elif insert == 29:
            reason = 'Insert 29 is reserved for Drum Monkey; move this instrument to another insert.'
        profile = profile_by_id.get(cid)
        internal = [] if kick or monkey or native_events or not profile else profile['controls']
        row = dict(id=cid, name=channel['name'] or f'Instrument {cid + 1}', plugin=voice.get('plugin_name') or voice.get('plugin', channel['plugin']),
                   mixer_insert=insert, original_mixer_insert=channel['mixer_insert'],
                   automation_status=reason, lanes=[], internal_controls=internal,
                   sylenth_profile=profile, drum_monkey=monkey, kick=kick, drum_routing=drum_routing)
        rows.append(row)
        if not reason:
            groups[insert].append(row)
    fx = []
    for insert, group in groups.items():
        used = occupied[insert]
        tail = max(used, default=-1) + 1
        slots = [slot for slot in range(tail, 10) if slot not in addressed_slots[insert]]
        if len(slots) < len(effect_lanes):
            for row in group:
                row['automation_status'] = f'Needs {len(effect_lanes)} free effect slots after the existing chain; added effects skipped.'
                changes.pop(row['id'], None)
                if row['drum_routing']:
                    row['drum_routing'] = dict(row['drum_routing'], status='unsupported',
                                              reason=row['automation_status'], assignment_change_required=False)
            continue
        # Never apply a melodic shared-bus envelope to any kick on that bus.
        other_kicks = [c for c in channels if c['mixer_insert'] == insert and _is_kick(c['name'])]
        if other_kicks and not any(r['drum_monkey'] for r in group):
            for row in group:
                row['automation_status'] = 'This mixer insert also contains a kick; separate it to add mixer effects.'
            continue
        for row in group:
            row['lanes'] = list(effect_lanes)
            row['automation_status'] = 'Automatic drums29 / kick30 routing' if row['drum_monkey'] else ('Shared mixer effects' if len(group) > 1 else 'Independent mixer effects')
        if len(group) > 1:
            warnings.append(f"Insert {insert}: {', '.join(r['name'] for r in group)} share effects following their combined notes. Their saved routing is retained.")
        other_sources = [c for c in channels if c['type'] in (0, 2, 4) and c['mixer_insert'] == insert
                         and c['id'] not in {r['id'] for r in group}]
        if other_sources and not any(r['drum_monkey'] for r in group):
            warnings.append(f"Insert {insert} also receives {', '.join(c['name'] or 'an unnamed channel' for c in other_sources)}; its mixer effects apply to that entire insert.")
        for slot, lane in zip(slots, effect_lanes):
            fx.append(dict(insert=insert, slot=slot, lane=lane, instrument_ids=[r['id'] for r in group],
                           name=' + '.join(r['name'] for r in group), drum_monkey=any(r['drum_monkey'] for r in group)))
    for row in rows:
        if row['internal_controls']:
            row['lanes'] += ['sylenth: ' + c['name'] for c in row['internal_controls']]
            if not row['lanes'][:3] == ['cutoff', 'delay', 'reverb']:
                row['automation_status'] += ' Internal Sylenth movement is available.'
        if not row['lanes'] and not row['kick']:
            warnings.append(row['name'] + ': ' + row['automation_status'])
    return dict(instruments=rows, effects=fx, route_changes=changes, warnings=list(dict.fromkeys(warnings)))


def inspect_flp(path):
    try:
        loaded = _load(path)
        model = loaded[4]
        plan = _plan(loaded)
        from mixer_movement import candidates
        mixer_targets, mixer_warnings = candidates(loaded, plan['instruments'])
        available = bool(plan['effects'] or mixer_targets or any(r['internal_controls'] for r in plan['instruments']))
        return dict(supported=available, errors=[] if available else ['No safe automation destinations are available. Give instruments their own mixer inserts with three spare effect slots.'],
                    path=str(loaded[0]), sha256=_sha(loaded[1]), tempo=model['tempo'],
                    bars=math.ceil(model['end_tick'] / (model['ppq'] * 4)), instrument_count=len(plan['instruments']),
                    instruments=[{k: v for k, v in row.items() if k not in ('internal_controls', 'sylenth_profile')} for row in plan['instruments']],
                    warnings=plan['warnings'] + mixer_warnings, mixer_targets=mixer_targets, version=APP_VERSION,
                    inferred_playlist=loaded[-1].get('inferred_playlist'))
    except (OSError, ValueError, KeyError, struct.error) as exc:
        return dict(supported=False, errors=[str(exc)], warnings=[], instruments=[], path=str(path))


def _controller(record, cid, label, group, cut):
    fields = dict(record)
    if fields.get(21) != b'\2' or len(fields.get(213, b'')) != 554 or len(fields.get(212, b'')) != 52:
        raise ExportError('The bundled controller definition is unsupported.')
    replacements = {64: struct.pack('<H', cid), 203: _name(label), 145: struct.pack('<I', group),
                    132: struct.pack('<HH', cut, cut), 0: b'\1'}
    result = []
    for k, p in record:
        if k == 212:
            wrapper = bytearray(p)
            struct.pack_into('<I', wrapper, 16, struct.unpack_from('<I', wrapper, 16)[0] & ~1)
            p = bytes(wrapper)
        result.append((k, replacements.get(k, p)))
    return result


def _midi_notes(mid, filename, opts, plan, notes, label, lane, drum=False):
    if drum:
        from drum_monkey_movement import make_drum_monkey_ctrl_midi
        pitches = sorted({n.note for n in notes})
        kick = pitches[0] if pitches else None
        others = [n for n in notes if n.note != kick]
        parts = [dict(source_note=p, role='percussion') for p in pitches if p != kick]
        make_drum_monkey_ctrl_midi(mid, others, parts, lane, filename, opts, plan,
                                  kick_notes=[n for n in notes if n.note == kick])
    else:
        movement.make_ctrl_midi(mid, notes, label, lane, filename, opts['division'], 60, 1, 0, False, 1, 127,
                                movement=opts['movement'], strength=opts['strength'], phrase_bars=opts['phrase_bars'],
                                seed=opts['seed'], plan=plan, drop_fx=opts['drop_fx'], fx_style=opts['fx_style'],
                                intensity=opts['intensity'])
    return [n for _, _, notes in movement.read_track_notes(MidiFile(filename)) for n in notes]


def _song_midi_notes(mid, filename, points, end, label):
    """Save the same held curve used in the FLP, including its 100% endpoint.

    FL128 is MIDI127. FL127 is avoided because a MIDI companion cannot
    distinguish it from that endpoint; the adjacent value126 is retained.
    """
    output = MidiFile(type=1, ticks_per_beat=mid.ticks_per_beat)
    conductor = MidiTrack()
    output.tracks.append(conductor)
    conductor.append(MetaMessage('set_tempo', tempo=next((m.tempo for t in mid.tracks for m in t if m.type == 'set_tempo'), 500000), time=0))
    conductor.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    conductor.append(MetaMessage('end_of_track', time=end))
    track = MidiTrack()
    output.tracks.append(track)
    track.append(MetaMessage('track_name', name=label, time=0))
    generated = []
    for index, (tick, value) in enumerate(points):
        stop = points[index + 1][0] if index + 1 < len(points) else end
        native = quantized_velocity(value)
        velocity = 127 if native == 128 else min(126, native)
        pitch = round((velocity - 1) * 127 / 126)
        generated.append(movement.NoteEvent(tick, stop - tick, pitch, velocity, 0))
        track.append(Message('note_on', channel=0, note=pitch, velocity=velocity, time=tick if not index else 0))
        track.append(Message('note_off', channel=0, note=pitch, velocity=0, time=stop - tick))
    track.append(MetaMessage('end_of_track', time=0))
    output.save(filename)
    return generated


def _song_controls(controls, role):
    """Use two complementary preset gestures instead of moving every knob."""
    usable = [c for c in controls if c['minimum'] < c['maximum']]
    priority = {'brightness': 0, 'modulation': 1, 'drive': 2, 'width': 3, 'decay': 4, 'release': 5, 'effect': 6}
    usable.sort(key=lambda c: (priority.get(c['character'], 9), c['parameter_id']))
    primary = next((c for c in usable if c['character'] != 'effect'), None)
    if primary is None:
        return []
    result = [dict(primary, song_primary=True)]
    choices = ('modulation', 'drive', 'decay') if role == 'bass' else ('width', 'modulation', 'drive', 'decay')
    secondary = next((c for character in choices for c in usable
                      if c['character'] == character and c['parameter_id'] != primary['parameter_id']), None)
    if secondary:
        result.append(dict(secondary, song_secondary=True))
    # The two saved time-effect wet knobs may clear on real drops, but chorus
    # and distortion are part of the patch body and are never treated as tails.
    result.extend(dict(c, song_secondary=True) for c in usable if c['parameter_id'] in (25, 143)
                  and c['character'] == 'effect')
    return result


def export_flp(source_flp, output_flp, options=None):
    from flp_song_analysis import build_analysis_midi
    from sylenth_movement import generate_points
    loaded = _load(source_flp)
    source, source_bytes, header, events, model, channels, arrangement, blocks, init_index, _, layout = loaded
    options = dict(options or {})
    chosen_effects = options.get('effect_lanes', ('cutoff', 'delay', 'reverb'))
    if not isinstance(chosen_effects, (list, tuple)):
        raise ExportError('Selected added effects must be a list.')
    prepared = _plan(loaded, chosen_effects)
    from mixer_movement import validate, candidates, select, anchors as mixer_anchors
    mixer_settings = validate(options.get('mixer_movement'))
    if type(options.get('add_effects', True)) is not bool:
        raise ExportError('Add effects must be enabled or disabled.')
    if not options.get('add_effects', True) or not chosen_effects:
        prepared['effects'] = []
        prepared['route_changes'] = {}
        for row in prepared['instruments']:
            row['mixer_insert'] = row['original_mixer_insert']
    for row in prepared['instruments']:
        if row['id'] not in prepared['route_changes']:
            row['mixer_insert'] = row['original_mixer_insert']
    mixer_targets, mixer_warnings = candidates(loaded, prepared['instruments'])
    mixer_targets = select(mixer_targets, mixer_settings)
    prepared['warnings'].extend(mixer_warnings)
    explicit_fx = None
    if 'fx_controls' in options:
        from fx_controls import validate_fx_controls
        explicit_fx = validate_fx_controls(options['fx_controls'])
    if options.get('expected_source_sha256', _sha(source_bytes)) != _sha(source_bytes):
        raise ExportError('The FLP changed since the preview. Load it again before generating.')
    song_mode = options.get('movement', 'song') == 'song'
    if explicit_fx is not None and not song_mode:
        raise ExportError('The editable FX controls require Full song straight ramps. Remove fx_controls to use a legacy movement style.')
    opts = _options(dict(options, movement='full' if song_mode else options['movement'],
                         strength=options.get('strength', 1.0)))
    opts['intensity'] = _number(options, 'intensity', 1.6 if song_mode else 1.4, 0, 2)
    opts['mixer_movement'] = mixer_settings
    opts['add_effects'] = options.get('add_effects', True)
    opts['effect_lanes'] = list(chosen_effects)
    if type(options.get('expanded_synth', False)) is not bool:
        raise ExportError('Expanded synth movement must be enabled or disabled.')
    opts['expanded_synth'] = options.get('expanded_synth', False)
    if mixer_targets and not song_mode:
        raise ExportError('Additional mixer movement requires Full song straight ramps.')
    if explicit_fx is not None:
        opts['fx_controls'] = explicit_fx
        if explicit_fx['bass_full_range']:
            prepared['warnings'].append('Bass full range is enabled: added bass delay and reverb may reach the chosen peak, up to 100%.')
    if song_mode:
        opts['movement'] = 'song'
        if opts['energy_curve'] is not None:
            from energy_curve import validate_curve, load_curve
            identity = dict(source_sha256=model['sha256'], ticks_per_beat=model['ppq'], end_tick=model['end_tick'])
            opts['energy_curve'] = (validate_curve(opts['energy_curve'], identity)
                                    if isinstance(opts['energy_curve'], dict)
                                    else load_curve(Path(opts['energy_curve']), identity))
    output = Path(output_flp).expanduser().resolve()
    assets = output.with_name(output.stem + ' - Automation')
    if output == source or output.suffix.casefold() != '.flp' or output.exists() or assets.exists():
        raise ExportError('Choose a new FLP filename; originals and earlier exports are never overwritten.')
    if not prepared['effects'] and not mixer_targets and not any(r['internal_controls'] for r in prepared['instruments']) and opts['intensity']:
        raise ExportError('No safe automation destinations are available. Check the instrument coverage in the preview.')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = _export_stage(output.parent)
    moved = completed = False
    try:
        mid_dir = stage / 'CTRL MIDI'
        mid_dir.mkdir()
        catalog = json.loads(CATALOG.read_text(encoding='utf-8'))
        if catalog.get('version') != 1:
            raise ExportError('The packaged effect definitions are unsupported.')
        controller_record = [(int(k), bytes.fromhex(p)) for k, p in catalog['controller']]
        model_midi = build_analysis_midi(model)
        end = model['end_tick']
        if song_mode:
            from full_song_movement import build_plan
            from linear_control_midi import fx_anchors, synth_anchors, sample_linear
            song_plan = build_plan(model, opts)
            song_plan['curve_shape'] = 'piecewise_linear'
            song_plan['gestures'] = []
            song_plan['builds'] = []
            song_plan['explanation'] = ('Each control follows its own part: note timing, held lengths, velocity, density and rests. '
                                        'Explicit straight ramps and flat holds cover the whole song; there is no easing or random motion.')
        else:
            song_plan = movement.movement_plan(model_midi, opts['phrase_bars'], opts['drop_bars'], opts['drop_length'])
        notes = {v['channel_id']: [movement.NoteEvent(n['tick'], n['duration'], n['note'], min(127, n['velocity']), 0)
                                   for n in v['notes']] for v in model['voices']}
        groups = [i for i, (k, _) in enumerate(events) if k == 231]
        first_channel = channels[0]['start']
        if not groups or groups[-1] >= first_channel:
            raise ExportError('The saved channel grouping layout is unsupported.')
        links = [i for i, (k, _) in enumerate(events) if k == 227]
        if links and max(links) >= first_channel:
            raise ExportError('The saved controller-link ordering is unsupported.')
        next_cid = max(c['id'] for c in channels) + 1
        cut = max((v for k, p in events if k == 132 and len(p) == 4 for v in struct.unpack('<HH', p)), default=0) + 1
        pid = max(layout['pattern_ids'], default=0) + 1
        additions = defaultdict(list)
        changes = {}
        added_channels, note_events, metadata, new_links, clips, lanes = [], [], [], [], [], []
        controller_payloads = []
        automation_patterns = []
        initial_values = {}
        # Native pattern data precedes channels; append new IDs after all
        # original pattern data, before the native controller-link section.
        pattern_data = [i for i in range(first_channel) if events[i][0] in (65, 223, 224)]
        note_data_at = max(pattern_data, default=first_channel - 1) + 1
        # Metadata may include markers and native tail fields. Append after
        # the complete last original metadata block, not before its markers.
        existing_metadata = [i for i in range(first_channel + 1, arrangement) if events[i][0] == 65]
        metadata_context = max(existing_metadata, default=first_channel)
        metadata_at = next((i for i in range(metadata_context + 1, arrangement)
                            if events[i][0] == 64), arrangement)

        def add_control(label, target, payload_builder, info):
            nonlocal next_cid, cut
            if next_cid > 4094 or cut > 65534 or pid >= 1000 or layout['next_row'] >= 500:
                raise ExportError('There is not enough channel, pattern or Playlist space for these automations.')
            cid = next_cid
            payload = payload_builder(cid)
            raw_point_count = len(payload) // 24
            payload = _coalesce_control_notes(payload)
            if not payload:
                return
            added_channels.extend(_controller(controller_record, cid, label, len(groups), cut))
            new_links.append((227, struct.pack('<5I', cid << 16 | 0x8001, 0, target, 8, 469)))
            # Sections share all controller channels and keep one Playlist
            # lane; their independent native links and cut groups stay intact.
            controller_payloads.append(payload)
            lanes.append(dict(info, name=label, controller_id=cid, target=f'{target:08x}', pattern_id=pid,
                              point_count=len(payload) // 24, curve_point_count=len(payload) // 24,
                              raw_point_count=raw_point_count,
                              flp_min_velocity=min(payload[21::24]), flp_peak_velocity=max(payload[21::24])))
            next_cid += 1
            cut += 1

        active = opts['intensity'] > 0 and opts['strength'] > 0
        if active:
            for fx in prepared['effects']:
                insert, slot, lane = fx['insert'], fx['slot'], fx['lane']
                prefix = 0x70000000 | insert << 22 | slot << 16
                target = prefix | (0x8000 if lane == 'cutoff' else 0x1f01)
                record = []
                for k, value in catalog['effects'][lane]['events']:
                    p = bytes.fromhex(value)
                    if k == 212:
                        p = bytearray(p)
                        struct.pack_into('<II', p, 0, insert, slot)
                        struct.pack_into('<I', p, 16, struct.unpack_from('<I', p, 16)[0] & ~1)
                        p = bytes(p)
                    elif k == 213 and lane == 'cutoff':
                        p = OPEN_FILTER
                    elif k == 203:
                        p = _name('AUTO ' + lane.title())
                    record.append((k, p))
                slot_positions = [blocks[insert][0] + i for i, (k, p) in enumerate(blocks[insert][2])
                                  if k == 98 and int.from_bytes(p, 'little') == slot]
                if len(slot_positions) != 1:
                    raise ExportError('An effect slot cannot be located uniquely.')
                additions[slot_positions[0]].extend(record)
                initial_values[prefix | 0x1f00] = 1
                initial_values[prefix | 0x1f01] = 12800 if lane == 'cutoff' else 0
                combined = sorted((n for cid in fx['instrument_ids'] for n in notes[cid]), key=lambda n: (n.tick, n.note))
                label = 'AUTO ' + fx['name'][:105] + ' ' + lane.title()
                filename = f'{next_cid:03d}_{movement.slug(label)}.mid'
                curve_info = {}
                if song_mode:
                    roles = [song_plan['voice_roles'][str(cid)] for cid in fx['instrument_ids']]
                    role = 'bass' if 'bass' in roles else ('drums' if fx['drum_monkey'] else roles[0])
                    if fx['drum_monkey'] and combined:
                        # This route is only admitted when the saved first pad
                        # is the verified isolated, lowest-note kick.
                        kick_note = min(n.note for n in combined)
                        combined = [n for n in combined if n.note != kick_note]
                    anchors = fx_anchors(song_plan, combined, role, lane, opts, fx['instrument_ids'])
                    points = sample_linear(anchors, model['ppq'], end, opts['division'])
                    generated = _song_midi_notes(model_midi, mid_dir / filename, points, end, label)
                    curve_info = dict(curve_shape='piecewise_linear', anchors=anchors,
                                      pitch_rule='round((MIDI velocity - 1) * 127 / 126)')
                else:
                    generated = _midi_notes(model_midi, mid_dir / filename, opts, song_plan, combined,
                                            fx['name'], lane, fx['drum_monkey'])
                add_control(label, target, lambda cid, ns=generated: _note_bytes(ns, cid, model['ppq'], model['ppq'], end, True),
                            dict(lane=lane, instrument_ids=fx['instrument_ids'], mixer_insert=insert, effect_slot=slot,
                                 midi_file=str(assets / 'CTRL MIDI' / filename), **curve_info))
            for target_info in mixer_targets:
                combined = sorted((n for cid in target_info['instrument_ids'] for n in notes[cid]), key=lambda n: (n.tick, n.note))
                anchors = mixer_anchors(target_info, song_plan, combined, mixer_settings, opts['intensity'], opts.get('energy_curve'))
                points = sample_linear(anchors, model['ppq'], end, opts['division'])
                if len({quantized_velocity(v) for _, v in points}) <= 1:
                    continue
                detail = target_info.get('plugin', target_info['lane'].title())
                label = 'AUTO ' + target_info['name'][:70] + ' ' + detail[:35]
                filename = f'{next_cid:03d}_{movement.slug(label)}.mid'
                generated = _song_midi_notes(model_midi, mid_dir / filename, points, end, label)
                initial_values[target_info['target']] = target_info['saved_value']
                add_control(label, target_info['target'],
                            lambda cid, ns=generated: _note_bytes(ns, cid, model['ppq'], model['ppq'], end, True),
                            dict(target_info, curve_shape='piecewise_linear', anchors=anchors,
                                 midi_file=str(assets / 'CTRL MIDI' / filename)))
            if opts['sylenth_movement']:
                for row in prepared['instruments']:
                    controls = (_song_controls(row['internal_controls'], song_plan['voice_roles'][str(row['id'])])
                                if song_mode and not opts['expanded_synth'] else row['internal_controls'])
                    for control in controls:
                        if song_mode:
                            anchors = synth_anchors(song_plan, dict(control, channel_id=row['id']), notes[row['id']], opts)
                            points = sample_linear(anchors, model['ppq'], end, opts['division'])
                        else:
                            points = generate_points(dict(control, channel_id=row['id']), notes[row['id']], model_midi, opts, song_plan)
                        if len({quantized_velocity(v) for _, v in points}) <= 1:
                            continue
                        label = 'AUTO ' + row['name'][:105] + ' SYL ' + control['name']
                        target = row['id'] << 16 | 0x8000 + control['parameter_id']
                        if song_mode:
                            filename = f'{next_cid:03d}_{movement.slug(label)}.mid'
                            generated = _song_midi_notes(model_midi, mid_dir / filename, points, end, label)
                            add_control(label, target, lambda cid, ns=generated: _note_bytes(ns, cid, model['ppq'], model['ppq'], end, True),
                                        dict(lane='sylenth', parameter=control['name'], instrument_ids=[row['id']], baseline=control['baseline'],
                                             curve_shape='piecewise_linear', anchors=anchors,
                                             pitch_rule='round((MIDI velocity - 1) * 127 / 126)',
                                             midi_file=str(assets / 'CTRL MIDI' / filename)))
                        else:
                            add_control(label, target, lambda cid, ps=points: encode_controller_points(cid, ps, end),
                                        dict(lane='sylenth', parameter=control['name'], instrument_ids=[row['id']], baseline=control['baseline']))
            for cid, insert in prepared['route_changes'].items():
                c = next(c for c in channels if c['id'] == cid)
                positions = [i for i in range(c['start'], c['stop']) if events[i][0] == 104]
                if len(positions) != 1:
                    raise ExportError('The instrument mixer destination cannot be located uniquely.')
                changes[positions[0]] = struct.pack('<H', insert)
            if not lanes:
                raise ExportError('These settings produced no moving controls. Increase the intensity or choose a project with available effect slots.')
        if lanes:
            sections = _section_control_notes(controller_payloads, model['ppq'], end)
            if not sections or pid + len(sections) - 1 >= 1000:
                raise ExportError('There is not enough pattern space for these automation sections.')
            lane_patterns = defaultdict(list)
            lane_point_counts = Counter()
            for index, section in enumerate(sections):
                section_pid = pid + index
                label = f'AUTO All Automations {index + 1:02d}'
                payload = section['payload']
                counts = Counter(struct.unpack_from('<H', payload, offset + 6)[0]
                                 for offset in range(0, len(payload), 24))
                for cid, count in counts.items():
                    lane_patterns[cid].append(section_pid)
                    lane_point_counts[cid] += count
                note_events.extend([(65, struct.pack('<H', section_pid)), (224, payload)])
                metadata.extend([(65, struct.pack('<H', section_pid)), (193, _name(label)),
                                 (150, bytes.fromhex('7aaa6400')), (157, b'\xff' * 4), (158, b'\xff' * 4)])
                clip = bytearray(_clip_bytes(layout['clip'], section_pid, layout['next_row'],
                                            section['duration'], layout['serial'] + index))
                struct.pack_into('<I', clip, 0, section['position'])
                if len(clip) == 88:
                    struct.pack_into('<d', clip, 64, 1.0)
                clips.append(bytes(clip))
                automation_patterns.append(dict(id=section_pid, name=label,
                    position=section['position'], duration=section['duration'],
                    playlist_track=layout['next_row'] + 1, controller_count=len(counts),
                    note_count=len(payload) // 24))
            for lane in lanes:
                cid = lane['controller_id']
                lane['pattern_ids'] = lane_patterns[cid]
                lane['pattern_id'] = lane_patterns[cid][0]  # First section, for older report readers.
                # Stored note counts include held notes split at boundaries.
                lane['point_count'] = lane_point_counts[cid]
                lane['serialized_point_count'] = lane_point_counts[cid]
            additions[groups[-1] + 1].append((231, _name('FLP Connector automation')))
            additions[note_data_at].extend(note_events)
            additions[max(links) + 1 if links else first_channel].extend(new_links)
            additions[metadata_at].extend(metadata)
            additions[arrangement].extend(added_channels)
            changes[layout['playlist_index']] = (events[layout['playlist_index']][1]
                                                + layout.get('inferred_clip', b'') + b''.join(clips))
            initial = events[init_index][1]
            patched, seen = [], set()
            for o in range(0, len(initial), 12):
                row = initial[o:o + 12]
                address = struct.unpack_from('<I', row, 4)[0]
                if address in initial_values:
                    if address in seen:
                        raise ExportError('An effect slot has duplicate initialized controls.')
                    row = row[:8] + struct.pack('<I', initial_values[address])
                    seen.add(address)
                patched.append(row)
            patched += [struct.pack('<3I', 0, t, v) for t, v in sorted(initial_values.items()) if t not in seen]
            changes[init_index] = b''.join(patched)
            new_rows = {layout['next_row'] + 1}
            for i, (k, p) in enumerate(events):
                if k == 238 and len(p) >= 13 and struct.unpack_from('<I', p)[0] in new_rows and not p[12]:
                    p = bytearray(p)
                    p[12] = 1
                    changes[i] = bytes(p)
        result, original_rows = [], []
        for i, (k, p) in enumerate(events):
            result.extend(additions[i])
            original_rows.append(len(result))
            result.append((k, changes.get(i, p)))
        new_header = bytearray(header)
        struct.pack_into('<H', new_header, 2, len(channels) + len(lanes))
        binary = encode_fl(bytes(new_header), result)
        rh, revents = read_bytes(binary)
        if rh != bytes(new_header) or revents != result:
            raise ExportError('The generated FLP failed its binary round-trip check.')
        if any(result[original_rows[i]] != (k, changes.get(i, p)) for i, (k, p) in enumerate(events)):
            raise ExportError('A saved source event changed unexpectedly.')
        if any(events[i][0] not in (104, 225, 233, 238) for i in changes):
            raise ExportError('An unsupported source event was modified.')
        if sum(k == 64 for k, p in result) != len(channels) + len(lanes):
            raise ExportError('The generated FLP channel count is inconsistent.')
        if source.read_bytes() != source_bytes:
            raise ExportError('The source FLP changed during generation. Load it again.')
        report = dict(version=APP_VERSION, mode='preserve_flp', source_flp=str(source), source_sha256=_sha(source_bytes),
                      output_flp=str(output), output_sha256=_sha(binary), instrument_count=len(prepared['instruments']),
                      native_fl_playback_tested=False, intensity=opts['intensity'], options=opts, tempo=model['tempo'], bars=math.ceil(end / (model['ppq'] * 4)),
                      automation_tracks=lanes, warnings=prepared['warnings'], route_changes=prepared['route_changes'] if active else {},
                      automation_pattern=dict(id=pid, name='AUTO All Automations', playlist_track=layout['next_row'] + 1,
                                              controller_count=len(lanes), section_bars=8,
                                              pattern_ids=[p['id'] for p in automation_patterns],
                                              pattern_count=len(automation_patterns)) if lanes else None,
                      automation_patterns=automation_patterns,
                      instruments=inspect_flp(source)['instruments'],
                      inferred_playlist=layout.get('inferred_playlist') if lanes else None,
                      validation=dict(native_fl_playback_tested=False, input_unchanged=True, original_note_events_identical=True,
                                      original_plugin_states_identical=True, original_automation_identical=True,
                                      original_playlist_clips_identical=True, roundtrip_exact=True),
                      assets_folder=str(assets))
        if song_mode:
            report['song_movement'] = song_plan
            if explicit_fx is not None:
                report['fx_controls'] = explicit_fx
            bar_ticks = model['ppq'] * 4
            def bar_label(tick):
                return f'{1 + tick / bar_ticks:g}'
            def bar_range(region):
                return f"{region['start'] // bar_ticks + 1} to {math.ceil(region['end'] / bar_ticks)}"
            summary = ['Full-song movement', '', song_plan['explanation'], '',
                       'Drops: ' + song_plan['drop_source'], '', 'Sections:']
            summary += [f"Bars {bar_range(r)}: {r['kind']}"
                        for r in song_plan['regions']]
            summary += ['', 'Builds:'] + [f"Bars {bar_range(r)}: gradual tension into the arrival at bar {bar_label(r['end'])}"
                                        for r in song_plan['builds']]
            summary += ['', 'Controls:'] + [f"{lane['name']}: {len(lane.get('anchors', []))} straight-line anchors; full-song MIDI included"
                                            for lane in lanes]
            if explicit_fx is not None:
                summary += ['', 'Chosen effect settings:']
                for effect in ('delay', 'reverb'):
                    setting = explicit_fx[effect]
                    summary.append(f"{effect.title()}: normal {setting['amount']:.0%}, peak {setting['peak']:.0%}, rise {setting['rise_beats']:g} beats, fade {setting['fade_bars']} bars, throws {setting['throws']}.")
                summary.append(f"Balanced drop duck: {explicit_fx['duck']:.0%}. Bass full range: {'on' if explicit_fx['bass_full_range'] else 'off (10% ceiling)' }.")
            summary += ['', 'The automation follows the full timeline. Eight-bar clips are only its FLP storage.',
                        'Every exported control MIDI has touching notes from start to end. Height initially follows velocity on a 0Ã¢â‚¬â€œ127 note range.',
                        'Height and velocity are independent after export. Existing links continue to read velocity.',
                        'MIDI velocity127 maps to native FL velocity128. The nearby native127 value uses126 in MIDI to keep the endpoint unambiguous.',
                        'Original notes and preset states are preserved. Audition the result and adjust Intensity to taste.']
            (stage / 'Song movement.txt').write_text('\n'.join(summary) + '\n', encoding='utf-8')
        (stage / 'Export report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        (stage / 'Project.flp').write_bytes(binary)
        if source.read_bytes() != source_bytes:
            raise ExportError('The source FLP changed before publishing. Load it again.')
        if output.exists() or assets.exists():
            raise ExportError('This output name was just created by another process. Generate a fresh copy.')
        os.rename(stage, assets)
        moved = True
        os.rename(assets / 'Project.flp', output)
        completed = True
        return report
    finally:
        cleanup = assets if moved and not completed else stage if not moved else None
        if cleanup is not None and cleanup.exists():
            if cleanup.resolve().parent != output.parent.resolve():
                raise ExportError('The temporary export path failed its safety check.')
            shutil.rmtree(cleanup)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--intensity', type=float, default=1.6)
    args = parser.parse_args()
    print(json.dumps(export_flp(args.source, args.out, {'intensity': args.intensity}) if args.out else inspect_flp(args.source), indent=2))
