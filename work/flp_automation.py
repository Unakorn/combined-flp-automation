"""Build a new MIDI song and controller arrangement in a prepared FLP copy.

The template is never written. Plugins, mixer settings and controller links remain
opaque and byte-identical. Only musical records, project tempo and pattern/clip
metadata are replaced or added. Native FL playback is a separate validation step.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct

from mido import MidiFile, MidiTrack, Message, MetaMessage
from flp_raw import read_bytes, read_fl, encode_fl
from flp_export_files import create_export_stage
import velocity_automation_midi as automation
from energy_curve import load_curve, validate_curve


class ExportError(ValueError):
    pass


APP_VERSION = '1.5'
MAX_BYTES = 256 * 1024 * 1024
NOTE = struct.Struct('<IHHIHHHHBBBB')
CONTROLLERS = {'fruity keyboard controller', 'fruity envelope controller', 'midi out', 'dashboard', 'layer'}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _text(payload):
    return payload.decode('utf-16-le', errors='strict').rstrip('\0')


def _encode_text(value):
    return (value + '\0').encode('utf-16-le')


def _normal(name):
    return re.sub(r'[^a-z0-9]', '', name.casefold())


def _read(path, limit=MAX_BYTES):
    path = Path(path).expanduser().resolve()
    if path.stat().st_size > limit:
        raise ExportError(f'{path.name} exceeds the inspection size limit.')
    return path, path.read_bytes()


def _project(path, data_override=None):
    path, data = _read(path)
    if data_override is not None:
        data = data_override
    header, events = read_bytes(data)
    if encode_fl(header, events) != data:
        raise ExportError('This FLP cannot be rewritten without changing its event encoding.')
    if struct.unpack_from('<H', header)[0] != 0:
        raise ExportError('Choose a complete .flp project, not an instrument preset.')
    version = next((p.decode('ascii').rstrip('\0') for k, p in events if k == 199), '')
    try:
        major = int(version.split('.')[0])
    except ValueError:
        raise ExportError('The FL Studio file version cannot be identified.') from None
    if major < 21 or major > 26:
        raise ExportError('This first version supports prepared projects saved in FL Studio 21–26.')
    if any(k in {1, 68, 129, 141, 222} for k, _ in events):
        raise ExportError('Save this legacy musical-data project in current FL Studio first.')
    ppq = struct.unpack_from('<H', header, 4)[0]
    if not 24 <= ppq <= 9600:
        raise ExportError('The FLP uses an unsupported musical time resolution.')
    starts = [i for i, (k, _) in enumerate(events) if k == 64]
    if not starts or len(starts) != struct.unpack_from('<H', header, 2)[0]:
        raise ExportError('The FLP channel records do not match its header.')
    arrangement = next((i for i in range(starts[-1] + 1, len(events)) if events[i][0] == 99), None)
    if arrangement is None or sum(k == 99 for k, _ in events) != 1:
        raise ExportError('Use a prepared template with one Playlist arrangement.')
    if any(k in {148, 33, 34, 205} for k, _ in events[arrangement:]):
        raise ExportError('Remove existing Playlist time markers from the prepared template first; new song markers will be supplied by the MIDI.')
    if any(k == 224 and len(p) % NOTE.size for k, p in events):
        raise ExportError('The template contains an unsupported piano-roll note format.')
    channels = []
    for start, stop in zip(starts, starts[1:] + [arrangement]):
        # Pattern metadata can sit within the first channel block. The fields
        # below do not overlap that metadata.
        d = dict(events[start:stop])
        channels.append({'id': int.from_bytes(d[64], 'little'),
                         'name': _text(d.get(203, d.get(192, b''))),
                         'plugin': _text(d.get(201, b'')),
                         'type': int.from_bytes(d.get(21, b'\xff'), 'little'),
                         'mixer_insert': int.from_bytes(d.get(104, b'\0\0'), 'little'),
                         'sample': _text(d.get(196, b'')),
                         'start': start, 'stop': stop})
    if len({c['id'] for c in channels}) != len(channels):
        raise ExportError('The template contains duplicate channel IDs.')
    if any(c['type'] == 5 for c in channels):
        raise ExportError('Remove old Automation Clip channels from the prepared template first; their saved curves are not reusable song data.')
    instruments = [dict(c, aliases=[c['name']]) for c in channels if c['type'] in {0, 2}
                   and c['plugin'].casefold() not in CONTROLLERS]
    if not instruments:
        raise ExportError('The template contains no playable instrument channels.')
    if any(c['mixer_insert'] >= 64 for c in instruments):
        raise ExportError('This first version supports prepared instrument routing on mixer inserts 0–63. Higher insert link addresses still need validation.')
    links = defaultdict(list)
    unsupported_links = set()
    for key, payload in events:
        if key != 227:
            continue
        if len(payload) != 20:
            raise ExportError('The template contains an unsupported controller-link format.')
        cid = struct.unpack_from('<H', payload, 2)[0]
        if payload[:2] == b'\x01\x80':  # Keyboard Controller velocity output.
            links[cid].append(struct.unpack_from('<I', payload, 8)[0])
        else:
            unsupported_links.add(cid)
    destinations = [target for targets in links.values() for target in targets]
    if len(destinations) != len(set(destinations)):
        raise ExportError('Two velocity controllers address the same parameter. Remove the conflicting link in the template first.')
    controllers = []
    for channel in channels:
        if channel['plugin'].casefold() != 'fruity keyboard controller':
            continue
        match = re.fullmatch(r'\s*(.+?)\s+CTRL\s+(\w+)(?:\s+.*)?', channel['name'], re.I)
        label, lane = (match[1].strip(), match[2].casefold()) if match else ('', '')
        targets = links[channel['id']]
        # A known mixer destination identifies aliases such as sub #2 -> bass.
        inserts = {(dest & 0x0fc00000) >> 22 for dest in targets if dest >> 28 == 7}
        by_route = [c for c in instruments if c['mixer_insert'] in inserts] if len(inserts) == 1 else []
        by_name = [c for c in instruments if _normal(c['name']) == _normal(label)]
        candidates = by_route if len(by_route) == 1 else by_name
        instrument = candidates[0] if len(candidates) == 1 else None
        if instrument is not None and label and label not in instrument['aliases']:
            instrument['aliases'].append(label)
        controllers.append({'id': channel['id'], 'name': channel['name'], 'lane': lane,
                            'instrument_id': instrument['id'] if instrument else None,
                            'instrument_alias': label,
                            'linked': bool(targets) and channel['id'] not in unsupported_links,
                            'targets': [f'{target:08x}' for target in targets]})
    # Unlinked lanes can inherit the exact alias learned from linked siblings.
    for ctrl in controllers:
        if ctrl['instrument_id'] is None:
            matches = [c for c in instruments if any(_normal(a) == _normal(ctrl['instrument_alias']) for a in c['aliases'])]
            if len(matches) == 1:
                ctrl['instrument_id'] = matches[0]['id']
    playlist = [(i, p) for i, (k, p) in enumerate(events) if k == 233]
    if not playlist:
        raise ExportError('Save the template with at least one Playlist pattern clip first.')
    # FL 21 introduced 60-byte records; FL 24+ uses 88 bytes. Confirm a record
    # against real pattern IDs rather than accepting divisibility alone.
    pattern_ids = {int.from_bytes(p, 'little') for k, p in events if k == 65}
    record_size = None
    exemplar = None
    sizes = (88, 60, 32)
    for size in sizes:
        if not any(p for _, p in playlist) or any(len(p) % size for _, p in playlist):
            continue
        records = [p[j:j + size] for _, p in playlist for j in range(0, len(p), size)]
        if all(struct.unpack_from('<H', r, 4)[0] == 0x5000 and
               (struct.unpack_from('<H', r, 6)[0] <= 0x5000 or
                struct.unpack_from('<H', r, 6)[0] - 0x5000 in pattern_ids) and
               struct.unpack_from('<I', r, 12)[0] <= 499 for r in records):
            exemplar = next((r for r in records if struct.unpack_from('<H', r, 6)[0] > 0x5000), None)
            if exemplar:
                record_size = size
                break
    if record_size is None:
        raise ExportError('The template Playlist uses an unsupported clip layout; use the supplied prepared template.')
    tempo = next((int.from_bytes(p, 'little') / 1000 for k, p in events if k == 156), None)
    if tempo is None:
        raise ExportError('The FLP does not contain a supported project tempo.')
    warnings = []
    pending = [c for c in controllers if not c['linked'] or c['lane'] not in automation.LANES or c['instrument_id'] is None]
    if pending:
        warnings.append(f'{len(pending)} controller lanes are unlinked or unmapped and will not be generated.')
    if any(c['type'] == 4 for c in channels):
        warnings.append('Template audio clips will not be placed in the new song.')
    public = {'path': str(path), 'sha256': _sha(data), 'version': version, 'ppq': ppq, 'bpm': tempo,
              'instruments': [{k: v for k, v in c.items() if k not in {'start', 'stop'}} for c in instruments],
              'controllers': controllers, 'linked_count': len(controllers) - len(pending),
              'pending': pending, 'warnings': warnings, 'supported': True}
    from sylenth_presets import inspect_instruments
    public['sylenth'] = inspect_instruments(events, channels)
    clip_serial = max((struct.unpack_from('<I', p, j + 32)[0]
                       for _, p in playlist for j in range(0, len(p), record_size)), default=0) if record_size >= 60 else 0
    if clip_serial >= 0x7ffff000:
        raise ExportError('The template clip identity range cannot be extended safely.')
    return public, header, events, data, {'first_channel': starts[0], 'arrangement': arrangement,
                                        'record_size': record_size, 'clip': exemplar,
                                        'playlist_index': playlist[0][0], 'pattern_ids': pattern_ids,
                                        'next_clip_serial': clip_serial + 1}


def inspect_project(path):
    return _project(path)[0]


def _song(path, drum_roles=None):
    path, data = _read(path, 64 * 1024 * 1024)
    midi = MidiFile(file=BytesIO(data))
    if midi.type not in {0, 1} or midi.ticks_per_beat <= 0:
        raise ExportError('Use a type 0 or type 1 musical-time MIDI file.')
    parts = automation.read_track_notes(midi)
    end = automation.song_end_tick(midi)
    if not parts or end <= 0:
        raise ExportError('The song MIDI contains no playable notes.')
    timed_tempos = [(t, m.tempo) for t, m in automation.timed_meta(midi) if m.type == 'set_tempo']
    tempos = {tempo for _, tempo in timed_tempos}
    if not any(t == 0 for t, _ in timed_tempos):
        tempos.add(500000)  # MIDI starts at 120 BPM until a tempo event occurs.
    if len(tempos) > 1:
        raise ExportError('Tempo changes are not supported yet. Export this first version at one fixed BPM.')
    signatures = {(m.numerator, m.denominator) for _, m in automation.timed_meta(midi) if m.type == 'time_signature'}
    if signatures - {(4, 4)}:
        raise ExportError('This automation project builder currently supports 4/4 songs.')
    tempo = next(iter(tempos), 500000)
    bpm = 60000000 / tempo
    if not 10 <= bpm <= 522:
        raise ExportError('The MIDI tempo is outside FL Studio’s supported range.')
    # Do not silently lose note-changing performance data. MIDI setup/reset and
    # expression messages are explicitly summarized; the template owns patches.
    ignored = Counter()
    for track in midi.tracks:
        channels_with_notes = {m.channel for m in track if m.type == 'note_on' and m.velocity > 0}
        if len(channels_with_notes) > 1:
            raise ExportError('A MIDI track contains several instrument channels. Export a multitrack MIDI with one instrument per track before building the project.')
        for msg in track:
            if msg.type == 'pitchwheel' and msg.pitch != 0:
                raise ExportError('This MIDI contains pitch bends. This version imports notes and velocities only; flatten the bends or use a MIDI without them.')
            if msg.type == 'control_change' and msg.control in {64, 66} and msg.value >= 64:
                raise ExportError('This MIDI uses a held sustain/sostenuto pedal. Convert sustained notes to note lengths before export.')
            if not msg.is_meta and msg.type not in {'note_on', 'note_off'}:
                ignored[msg.type] += 1
    warnings = []
    if ignored:
        warnings.append('Imports notes and note velocities. Existing MIDI CC/expression, pressure, program changes and MIDI setup messages are not imported; the prepared sounds and newly generated movement are used.')
    from drum_detection import expand_drum_parts
    parts, drum_parts, drum_warnings = expand_drum_parts(parts, midi, source_path=path, role_overrides=drum_roles)
    warnings.extend(drum_warnings)
    drum_by_id = {row['id']: row for row in drum_parts}
    overview = {'path': str(path), 'sha256': _sha(data), 'ppq': midi.ticks_per_beat,
                'bpm': bpm, 'end_tick': end, 'bars': end / (4 * midi.ticks_per_beat),
                'parts': [{'id': str(i), 'name': n, 'note_count': len(notes), 'suggested_channel_id': None,
                           **({'drum': drum_by_id[str(i)]} if str(i) in drum_by_id else {})}
                          for n, i, notes in parts], 'warnings': warnings,
                'drum_parts': drum_parts,
                'ignored_midi_messages': dict(ignored)}
    return overview, midi, parts, data


def inspect_song_parts(path, template=None):
    result, _, _, _ = _song(path)
    if template is not None:
        project = template if isinstance(template, dict) else inspect_project(template)
        for part in result['parts']:
            if part.get('drum'):
                continue
            matches = [c for c in project['instruments'] if any(_normal(alias) == _normal(part['name']) for alias in c['aliases'])]
            if len(matches) == 1:
                part['suggested_channel_id'] = matches[0]['id']
    return result


def _number(options, key, default, low, high, integer=False):
    value = options.get(key, default)
    if isinstance(value, bool):
        raise ExportError(f'{key} must be a number.')
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise ExportError(f'{key} must be a number.') from None
    if not math.isfinite(number) or not low <= number <= high or (integer and number != int(number)):
        raise ExportError(f'{key} must be between {low} and {high}.')
    return int(number) if integer else number


def _options(options):
    options = dict(options or {})
    if options.get('mode', 'new') != 'new':
        raise ExportError('This version builds a new song from the prepared template. Editing an existing arrangement is not supported yet.')
    result = {'movement': options.get('movement', 'drop'), 'drop_fx': options.get('drop_fx', 'reduced'),
              'drop_bars': str(options.get('drop_bars', '')), 'invert': options.get('invert', False),
              'fx_style': options.get('fx_style', 'full_range'),
              'sylenth_movement': options.get('sylenth_movement', True)}
    result['drum_roles'] = options.get('drum_roles', {})
    if not isinstance(result['drum_roles'], dict):
        raise ExportError('Drum types must be selected by MIDI lane.')
    preset = options.get('drum_preset')
    if preset is not None:
        if not isinstance(preset, dict) or not isinstance(preset.get('path'), str) or not preset['path'].strip():
            raise ExportError('Choose a saved Drum Monkey preset or project.')
        if preset.get('kit_id') is not None and not isinstance(preset['kit_id'], str):
            raise ExportError('Choose the Drum Monkey instance saved in that preset.')
        if preset.get('sha256') is not None and (not isinstance(preset['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', preset['sha256'])):
            raise ExportError('The saved Drum Monkey preset fingerprint is invalid. Load the preset again.')
        preset = {key: preset[key] for key in ('path', 'kit_id', 'sha256') if key in preset}
    result['drum_preset'] = preset
    if result['movement'] not in automation.MOVEMENTS or result['drop_fx'] not in {'dry', 'reduced', 'keep'}:
        raise ExportError('Choose a valid movement and drop FX setting.')
    if result['fx_style'] not in {'full_range', 'fuller', 'original'}:
        raise ExportError('Choose Full-range FX (100%), Fuller tails or Original short throws for FX feel.')
    if not isinstance(result['invert'], bool):
        raise ExportError('invert must be true or false.')
    if not isinstance(result['sylenth_movement'], bool):
        raise ExportError('Sylenth movement must be on or off.')
    for key, default, low, high, integer in [
        ('strength', .8, 0, 1, False), ('phrase_bars', 16, 4, 16, True),
        ('drop_length', 8, 4, 16, True), ('seed', 1, 0, 2147483647, True),
        ('division', 16, 1, 64, True), ('note', 60, 0, 127, True), ('gate', .9, .01, 1, False),
        ('min_velocity', 1, 1, 127, True), ('max_velocity', 127, 1, 127, True)]:
        result[key] = _number(options, key, default, low, high, integer)
    if result['min_velocity'] > result['max_velocity']:
        raise ExportError('Minimum velocity must not exceed maximum velocity.')
    result['energy_curve'] = options.get('energy_curve')
    return result


def _tick(value, source_ppq, target_ppq):
    return (int(value) * target_ppq * 2 + source_ppq) // (source_ppq * 2)


def _note_bytes(notes, cid, source_ppq, target_ppq, end_tick, full_velocity_endpoint=False):
    result = bytearray()
    for note in sorted(notes, key=lambda n: (n.tick, n.note)):
        start = _tick(note.tick, source_ppq, target_ppq)
        stop = _tick(note.tick + note.duration, source_ppq, target_ppq)
        if start >= end_tick:
            raise ExportError('A MIDI note is too short to represent at the template’s PPQ; save a higher-PPQ template.')
        stop = min(end_tick, max(start + 1, stop))
        if not 0 <= note.note <= 127 or not 1 <= note.velocity <= 127:
            raise ExportError('Invalid source MIDI note.')
        # Native FL notes have a 0..128 velocity range; standard MIDI ends at
        # 127. Map only explicit full-range FX peaks to the native endpoint.
        # Preserve musical velocities and all legacy control lanes literally.
        velocity = 128 if full_velocity_endpoint and note.velocity == 127 else note.velocity
        result.extend(NOTE.pack(start, 0x4000, cid, stop - start, note.note, 0, 120, 0,
                                64, velocity, 128, 128))
    return bytes(result)


def _clip_bytes(example, pid, row, length, serial):
    record = bytearray(example)
    struct.pack_into('<IHHII', record, 0, 0, 0x5000, 0x5000 + pid, length, 499 - row)
    # Full pattern from its beginning, normal selection, not a trimmed/looped
    # copy of the exemplar. The remaining version-specific bytes are preserved.
    struct.pack_into('<HH', record, 16, 120, 0x40)
    record[20:24] = bytes((64, 100, 128, 128))
    record[24:32] = b'\xff' * 8
    if len(record) >= 60:
        # Different FL-saved instances of the same pattern have distinct values
        # here (e.g. 3/4, or 26). Keep generated clip identities unique.
        struct.pack_into('<I', record, 32, serial)
        record[36:60] = b'\0' * 24
    return bytes(record)


def export_project(source_flp, song_midi, output_flp, options=None, mapping=None):
    opts = _options(options)
    project, header, events, source_bytes, layout = _project(source_flp)
    song, midi, parts, song_bytes = _song(song_midi, opts['drum_roles'])
    for key, actual, label in [('expected_template_sha256', project['sha256'], 'prepared template'),
                               ('expected_midi_sha256', song['sha256'], 'song MIDI')]:
        expected = (options or {}).get(key)
        if expected is not None and expected != actual:
            raise ExportError(f'The {label} changed since the preview. Reload it and review the part mapping before generating.')
    source_path, midi_path = Path(project['path']), Path(song['path'])
    output = Path(output_flp).expanduser().resolve()
    if output.suffix.casefold() != '.flp':
        raise ExportError('Choose an output filename ending in .flp.')
    assets = output.with_name(output.stem + ' - automation')
    report_path = assets / 'Export report.json'
    if output in {source_path, midi_path} or output.exists() or assets.exists():
        raise ExportError('Choose a new output filename. Originals and previous exports are never overwritten.')
    drum_setup, drum_targets, drum_instruments = {}, {}, {}
    drum_notes, kick_notes, other_drum_notes = [], [], []
    preset_bytes = None
    if song['drum_parts']:
        from drum_monkey_template import prepare_drum_monkey
        preset = opts['drum_preset']
        if preset is None:
            raise ExportError('This song contains drums. Choose its Drum Monkey .fst preset or .flp so the actual instrument and sounds can be included.')
        preset_path, preset_bytes = _read(preset['path'], 64 * 1024 * 1024)
        original_sha = project['sha256']
        header, events, drum_setup = prepare_drum_monkey(header, events, preset_path,
            kit_id=preset.get('kit_id'), expected_sha256=preset.get('sha256'))
        project, header, events, _, layout = _project(source_path, encode_fl(header, events))
        project['sha256'] = original_sha
        drum_cid = drum_setup['channel_id']
        drum_by_id = {row['id']: row for row in song['drum_parts']}
        for row in song['drum_parts']:
            drum_targets[row['id']] = drum_cid
        for name, index, notes in parts:
            if str(index) in drum_by_id:
                drum_notes.extend(notes)
                (kick_notes if drum_by_id[str(index)]['role'] == 'kick' else other_drum_notes).extend(notes)
        for notes in (drum_notes, kick_notes, other_drum_notes):
            notes.sort(key=lambda n: (n.tick, n.note, n.duration, n.velocity))
        drum_instruments[drum_cid] = {'role': 'drum_bus', 'parts': song['drum_parts'],
                                     'drums_insert': 29, 'kick_insert': 30}
    instruments = {c['id']: c for c in project['instruments']}
    provided = {str(k): v for k, v in (mapping or {}).items()}
    known_parts = {str(i) for _, i, _ in parts}
    if set(provided) - known_parts:
        raise ExportError('The saved part mapping contains tracks that are not in this MIDI. Review the mapping again.')
    resolved = {}
    for name, index, notes in parts:
        key = str(index)
        if key in drum_targets:
            cid = drum_targets[key]
            if key in provided and provided[key] != cid:
                raise ExportError('All drum notes play the selected Drum Monkey instrument. Choose the saved drum preset instead of assigning drum lanes to a synth.')
        elif key in provided:
            cid = provided[key]
            if isinstance(cid, bool) or not isinstance(cid, int) or cid not in instruments:
                raise ExportError(f'Choose a valid template instrument for {name}.')
        else:
            matches = [c['id'] for c in instruments.values() if any(_normal(a) == _normal(name) for a in c['aliases'])]
            if len(matches) != 1:
                raise ExportError(f'Cannot uniquely match MIDI part “{name}”. Choose its template instrument before generating.')
            cid = matches[0]
        resolved[key] = cid
    # All drum keys stay exactly as authored. They share one Drum Monkey
    # generator and one music pattern; roles only guide the bus automation.
    end = _tick(song['end_tick'], midi.ticks_per_beat, project['ppq'])
    if not 1 <= end < 0x08000000:
        raise ExportError('The song duration cannot be represented safely in this FLP.')
    curve = opts.pop('energy_curve')
    if curve is not None:
        overview = {'source_sha256': song['sha256'], 'ticks_per_beat': midi.ticks_per_beat,
                    'end_tick': song['end_tick'], 'paintable': True}
        curve = validate_curve(curve, overview) if isinstance(curve, dict) else load_curve(curve, overview)
    plan = automation.movement_plan(midi, opts['phrase_bars'], opts['drop_bars'], opts['drop_length'])
    pending_ids = {c['id'] for c in project['pending']}
    active = [c for c in project['controllers'] if c['id'] not in pending_ids and c['instrument_id'] in set(resolved.values())]
    synth_profiles = [p for p in project['sylenth']['instruments']
                      if opts['sylenth_movement'] and p['instrument_id'] in set(resolved.values())]
    synth_controls = [(p, c) for p in synth_profiles for c in p['controls']]
    merged = defaultdict(list)
    for name, index, notes in parts:
        merged[resolved[str(index)]].extend(notes)
    synth_points, synth_skipped = {}, []
    if synth_controls:
        from sylenth_movement import generate_points
        from sylenth_events import quantized_velocity
        moving = []
        for profile, control in synth_controls:
            cid = profile['instrument_id']
            points = generate_points(dict(control, channel_id=cid), sorted(merged[cid], key=lambda n: n.tick), midi, opts, plan, curve)
            collapsed = {min(end - 1, _tick(t, midi.ticks_per_beat, project['ppq'])): v for t, v in points}
            collapsed[0] = collapsed[end - 1] = control['baseline']
            native_points = sorted(collapsed.items())
            if len({quantized_velocity(v) for _, v in native_points}) <= 1:
                synth_skipped.append({'instrument_id': cid, 'preset': profile['preset'],
                                      'parameter_id': control['parameter_id'], 'name': control['name'],
                                      'reason': 'No movement at controller velocity resolution with these notes and amount.'})
            else:
                moving.append((profile, control))
                synth_points[(cid, control['parameter_id'])] = native_points
        synth_controls = moving
    synth_connections = []
    if synth_controls:
        from sylenth_events import prepare_controllers
        requests = [dict(instrument_id=p['instrument_id'], parameter_id=c['parameter_id'],
                         name=f"{instruments[p['instrument_id']]['name']} SYL {c['name']}")
                    for p, c in synth_controls]
        header, events, synth_connections = prepare_controllers(header, events, requests)
        # Inspect the newly prepared copy to validate its channels and recover
        # shifted event positions. Source paths/hashes remain those of the input.
        _, header, events, _, layout = _project(source_path, encode_fl(header, events))
    if not active and not synth_controls:
        raise ExportError('No connected velocity-controller lanes match these song parts. Prepare the template links first.')
    music_pattern_count = len(parts) - len(drum_targets) + bool(drum_targets)
    if music_pattern_count + len(active) + len(synth_controls) > 500:
        raise ExportError('The generated arrangement exceeds FL Studio’s 500 Playlist tracks.')
    next_pid = max(layout['pattern_ids'], default=0) + 1
    if next_pid + music_pattern_count + len(active) + len(synth_controls) >= 1000:
        raise ExportError('This template has too many existing patterns; save a clean prepared template first.')
    music_rows, lanes, additions, metadata, clips = [], [], [], [], []
    synth_lanes = []
    marker_groups = defaultdict(list)
    for tick, msg in automation.timed_meta(midi):
        if msg.type in {'marker', 'cue_marker'} and tick <= song['end_tick']:
            label = ''.join(c for c in msg.text if ord(c) >= 32).strip()[:128]
            if label and label not in marker_groups[_tick(tick, midi.ticks_per_beat, project['ppq'])]:
                marker_groups[_tick(tick, midi.ticks_per_beat, project['ppq'])].append(label)
    marker_events = []
    for tick, labels in sorted(marker_groups.items()):
        marker_events.extend([(148, struct.pack('<I', tick)), (205, _encode_text(' / '.join(labels)[:200]))])
    note_counts = {}

    def add_pattern(name, notes, cid, kind, source_ppq, full_velocity_endpoint=False, raw_note_payload=None):
        nonlocal next_pid
        pid = next_pid
        next_pid += 1
        payload = _note_bytes(notes, cid, source_ppq, project['ppq'], end, full_velocity_endpoint) if raw_note_payload is None else raw_note_payload
        additions.extend([(65, struct.pack('<H', pid)), (224, payload)])
        metadata.extend([(65, struct.pack('<H', pid)), (193, _encode_text(name)),
                         (150, bytes.fromhex('7aaa6400' if kind == 'automation' else 'b59f7100')),
                         (157, b'\xff' * 4), (158, b'\xff' * 4)])
        clips.append(_clip_bytes(layout['clip'], pid, len(clips), end,
                                 layout['next_clip_serial'] + len(clips)))
        note_counts[pid] = len(notes)
        return pid

    drums_written = False
    for name, index, notes in parts:
        cid = resolved[str(index)]
        if str(index) in drum_targets:
            if drums_written:
                continue
            drums_written = True
            name, notes = 'DRUMS - Drum Monkey', drum_notes
        pid = add_pattern(name, notes, cid, 'music', midi.ticks_per_beat)
        music_rows.append({'midi_track': str(index), 'name': name, 'channel_id': cid,
                           'instrument': instruments[cid]['name'], 'pattern_id': pid, 'note_count': len(notes),
                           **({'drum': drum_instruments[cid], 'midi_tracks': list(drum_targets)} if cid in drum_instruments else {})})
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = create_export_stage(output.parent, prefix='.automation-build-')
    moved_assets = False
    completed = False
    try:
        mid_dir = stage / 'CTRL MIDI'
        mid_dir.mkdir()
        if drum_setup:
            drum_details = ['DRUM LANES', '',
                'One Drum Monkey instrument keeps the selected saved kit and all original drum notes together.',
                'Drum automation: mixer insert 29. Kick: mixer insert 30, with no added automation.',
                'Route Drum Monkey’s drum output to 29 and its kick output to 30 in FL Studio.',
                'The saved plugin output settings are preserved. Separate-output routing is not set automatically.',
                'Effects apply to every sound sent to insert 29. The original keys, timing, lengths and velocities are retained.',
                f"Saved kit: {opts['drum_preset']['path']}", '']
            for row in song['drum_parts']:
                drum_details.extend([f"Drum {row['slot']} | source MIDI key {row['source_note']} | {row['role'].replace('_', ' ')}",
                    f"Analysis type source: {row['role_source']}", ''])
            (stage / 'Drum lanes.txt').write_text('\n'.join(drum_details), encoding='utf-8')
        for ctrl in active:
            cid = ctrl['instrument_id']
            instrument = instruments[cid]
            filename = f"{ctrl['id']:03d}_{automation.slug(instrument['name'])}_CTRL_{ctrl['lane']}.mid"
            ctrl_path = mid_dir / filename
            if cid in drum_instruments:
                from drum_monkey_movement import make_drum_monkey_ctrl_midi
                make_drum_monkey_ctrl_midi(midi, other_drum_notes,
                    [row for row in song['drum_parts'] if row['role'] != 'kick'],
                    ctrl['lane'], ctrl_path, opts, plan, curve, kick_notes=kick_notes)
            else:
                automation.make_ctrl_midi(midi, sorted(merged[cid], key=lambda n: n.tick),
                    ctrl['name'].split(' CTRL ')[0] + ' ' + automation.detect_role(ctrl['name']),
                    ctrl['lane'], ctrl_path, opts['division'], opts['note'], opts['gate'], 0,
                    opts['invert'], opts['min_velocity'], opts['max_velocity'],
                    movement=opts['movement'], strength=opts['strength'], phrase_bars=opts['phrase_bars'],
                    seed=opts['seed'], plan=plan, energy_curve=curve, drop_fx=opts['drop_fx'],
                    fx_style=opts['fx_style'])
            generated = MidiFile(ctrl_path)
            ctrl_notes = [n for _, _, notes in automation.read_track_notes(generated) for n in notes]
            # Coarser FL resolutions can merge adjacent samples. The final
            # sample at each tick wins, and controller notes never overlap.
            collapsed = {}
            for n in ctrl_notes:
                tick = min(end - 1, _tick(n.tick, midi.ticks_per_beat, project['ppq']))
                stop = min(end, max(tick + 1, _tick(n.tick + n.duration, midi.ticks_per_beat, project['ppq'])))
                collapsed[tick] = automation.NoteEvent(tick, stop - tick, n.note, n.velocity, 0)
            ctrl_notes = [collapsed[t] for t in sorted(collapsed)]
            for i in range(len(ctrl_notes) - 1):
                n = ctrl_notes[i]
                n.duration = min(n.duration, ctrl_notes[i + 1].tick - n.tick)
            full_endpoint = (opts['fx_style'] == 'full_range' and ctrl['lane'] in {'delay', 'reverb'}) or (cid in drum_instruments and ctrl['lane'] == 'cutoff')
            pid = add_pattern(ctrl['name'], ctrl_notes, ctrl['id'], 'automation', project['ppq'], full_endpoint)
            midi_peak = max((n.velocity for n in ctrl_notes), default=0)
            lanes.append({'controller_id': ctrl['id'], 'instrument_id': cid, 'name': ctrl['name'],
                          'lane': ctrl['lane'], 'targets': ctrl['targets'], 'pattern_id': pid,
                          'note_count': len(ctrl_notes), 'midi_file': str(assets / 'CTRL MIDI' / filename),
                          'midi_peak_velocity': midi_peak,
                          'flp_peak_velocity': 128 if full_endpoint and midi_peak == 127 else midi_peak})
            if cid in drum_instruments:
                lanes[-1].update(drum_role='drum_bus', mixer_insert=29)
        if synth_controls:
            from sylenth_events import encode_controller_points, target_id
            details = ['SYLENTH PRESET MOVEMENT', '',
                       'Each named Sylenth pattern drives a knob through its own linked Keyboard Controller. Edit note velocities in its Piano roll.',
                       'Saved presets and existing controller links remain unchanged. Playback control values use 128 velocity steps; baseline restoration is approximate.',
                       'Values below are normalized knob positions. Companion MIDI uses the standard MIDI 1–127 range.', '']
            for (profile, control), connection in zip(synth_controls, synth_connections):
                cid = profile['instrument_id']
                native_points = synth_points[(cid, control['parameter_id'])]
                controller_id = connection['controller_id']
                payload = encode_controller_points(controller_id, native_points, end)
                name = f"{instruments[cid]['name']} SYL {control['name']}"
                # The helper supplies verified native controller notes (event
                # 224), including FL's full-scale velocity128 endpoint.
                pid = add_pattern(name, [], controller_id, 'automation', project['ppq'], raw_note_payload=payload)
                ctrl_midi = MidiFile(type=1, ticks_per_beat=project['ppq'])
                conductor, track = MidiTrack(), MidiTrack()
                ctrl_midi.tracks.extend([conductor, track])
                conductor.extend([MetaMessage('set_tempo', tempo=round(60000000 / song['bpm']), time=0),
                                  MetaMessage('time_signature', numerator=4, denominator=4, time=0)])
                track.append(MetaMessage('track_name', name=name, time=0))
                last = 0
                for offset in range(0, len(payload), NOTE.size):
                    fields = NOTE.unpack_from(payload, offset)
                    tick, duration, key, velocity = fields[0], fields[3], fields[4], fields[9]
                    track.append(Message('note_on', note=key, velocity=min(127, velocity), time=tick - last))
                    track.append(Message('note_off', note=key, velocity=0, time=duration))
                    last = tick + duration
                track.append(MetaMessage('end_of_track', time=max(0, end - last)))
                filename = f"{controller_id:03d}_{automation.slug(name)}.mid"
                ctrl_midi.save(mid_dir / filename)
                entry = dict(control, instrument_id=cid, instrument=instruments[cid]['name'], preset=profile['preset'],
                             name=name, pattern_id=pid, controller_id=controller_id,
                             target=f"{target_id(cid, control['parameter_id']):08x}",
                             midi_file=str(assets / 'CTRL MIDI' / filename),
                             playback_baseline=payload[21] / 128,
                             playback_minimum=min(payload[21::NOTE.size]) / 128,
                             playback_maximum=max(payload[21::NOTE.size]) / 128,
                             event_count=len(payload) // NOTE.size, observed_minimum=min(v for _, v in native_points),
                             observed_maximum=max(v for _, v in native_points))
                synth_lanes.append(entry)
                details.append(f"{name} | {profile['preset']} | saved {control['baseline']:.1%} | allowed {control['minimum']:.1%}–{control['maximum']:.1%}")
            (stage / 'Sylenth movement.txt').write_text('\n'.join(details) + '\n', encoding='utf-8')
        first_channel = layout['first_channel']
        meta_at = next((i for i in range(first_channel + 1, layout['arrangement']) if events[i][0] == 65), layout['arrangement'])
        result, changed, inserted = [], [], []
        for index, (key, payload) in enumerate(events):
            if index == first_channel:
                inserted.extend(range(len(result), len(result) + len(additions)))
                result.extend(additions)
            if index == meta_at:
                inserted.extend(range(len(result), len(result) + len(metadata)))
                result.extend(metadata)
            if index == layout['playlist_index']:
                inserted.extend(range(len(result), len(result) + len(marker_events)))
                result.extend(marker_events)
            replacement = payload
            if key in {223, 224}:
                replacement = b''
            elif key == 233:
                replacement = b''.join(clips) if index == layout['playlist_index'] else b''
            elif key == 156:
                replacement = struct.pack('<I', round(song['bpm'] * 1000))
            elif key == 67:
                replacement = struct.pack('<H', music_rows[0]['pattern_id'])
            elif key == 9:
                replacement = b'\0'  # Disable the template's selected loop range.
            elif key in {17, 18}:
                replacement = b'\x04'
            elif key == 217:
                if len(payload) != 8:
                    raise ExportError('Unsupported Playlist selection record.')
                replacement = b'\xff' * 8
            elif key == 152:
                replacement = b'\0' * 4
            elif key == 238 and len(payload) >= 13 and 1 <= struct.unpack_from('<I', payload)[0] <= len(clips):
                track_data = bytearray(payload)
                track_data[12] = 1
                replacement = bytes(track_data)
            elif key == 225:
                if len(payload) % 12:
                    raise ExportError('Unsupported initialized-control data.')
                init = bytearray(payload)
                for offset in range(0, len(init), 12):
                    if struct.unpack_from('<I', init, offset + 4)[0] == 0x40000005:
                        struct.pack_into('<I', init, offset + 8, round(song['bpm'] * 1000))
                replacement = bytes(init)
            if replacement != payload:
                changed.append({'source_event': index, 'event_id': key, 'old_bytes': len(payload), 'new_bytes': len(replacement)})
            result.append((key, replacement))
        binary = encode_fl(header, result)
        rh, revents = read_bytes(binary)
        if rh != header or revents != result:
            raise ExportError('Internal FLP round-trip validation failed.')
        # Exhaustive source-index proof: remove inserted events, then each
        # remaining event must either be exactly original or an allowed edit.
        original_positions = [event for i, event in enumerate(result) if i not in set(inserted)]
        if len(original_positions) != len(events):
            raise ExportError('Internal event-preservation validation failed.')
        changed_indices = {row['source_event'] for row in changed}
        if any(original_positions[i] != event for i, event in enumerate(events) if i not in changed_indices):
            raise ExportError('An unrelated source event changed unexpectedly.')
        if any(row['event_id'] not in {223, 224, 233, 156, 67, 225, 9, 17, 18, 217, 152, 238} for row in changed):
            raise ExportError('An unsupported source event was modified.')
        if [(k, p) for k, p in result if k in {212, 213, 227, 235, 236, 239}] != [(k, p) for k, p in events if k in {212, 213, 227, 235, 236, 239}]:
            raise ExportError('Plugin, link, or mixer-state preservation check failed.')
        if source_path.read_bytes() != source_bytes or midi_path.read_bytes() != song_bytes:
            raise ExportError('An input file changed during generation. Reload it and try again.')
        if preset_bytes is not None and preset_path.read_bytes() != preset_bytes:
            raise ExportError('The Drum Monkey preset changed during generation. Load it again and retry.')
        warnings = list(project['warnings']) + list(song['warnings'])
        if opts['sylenth_movement']:
            warnings.extend(project['sylenth']['warnings'])
        warnings.extend(drum_setup.get('warnings', []))
        unused = [c['name'] for c in instruments.values() if c['id'] not in set(resolved.values())]
        if unused:
            warnings.append('Unused prepared instruments: ' + ', '.join(unused) + '.')
        report = {'version': APP_VERSION, 'output': str(output), 'report_path': str(report_path),
                  'template': project['path'], 'template_sha256': project['sha256'],
                  'song_midi': song['path'], 'song_midi_sha256': song['sha256'],
                  'output_sha256': _sha(binary), 'mode': 'new', 'bpm': round(song['bpm'] * 1000) / 1000,
                  'midi_bpm': song['bpm'],
                  'ppq': project['ppq'], 'song_end_tick': end, 'music_tracks': music_rows,
                  'generated_lanes': lanes, 'pending_lanes': project['pending'], 'warnings': warnings,
                  'sylenth_lanes': synth_lanes,
                  'sylenth_skipped_lanes': synth_skipped,
                  'drum_setup': drum_setup, 'drum_parts': song['drum_parts'],
                  'options': opts, 'energy_curve': curve, 'changed_events': changed,
                  'validation': {'roundtrip_exact': True, 'inputs_unchanged': True,
                                 'unrelated_events_byte_identical': True, 'plugin_and_link_state_byte_identical': True,
                                 'new_music_notes': sum(r['note_count'] for r in music_rows),
                                 'new_control_notes': sum(r['note_count'] for r in lanes),
                                 'new_sylenth_events': sum(r['event_count'] for r in synth_lanes),
                                 'new_sylenth_controllers': len(synth_connections),
                                 'new_drum_instruments': len(drum_instruments),
                                 'new_drum_samplers': 0,
                                 'patterns_added': len(note_counts), 'playlist_clips': len(clips),
                                 'playlist_markers': len(marker_groups), 'template_loop_range_disabled': True,
                                 'native_FL_playback_verified': False}}
        (stage / 'Export report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if curve:
            (stage / 'Energy curve.json').write_text(json.dumps(curve, indent=2), encoding='utf-8')
        staged_flp = stage / 'Project.flp'
        staged_flp.write_bytes(binary)
        # Windows rename fails when a destination exists; no replace() call is
        # used, so a racing export cannot overwrite previous work.
        if output.exists() or assets.exists():
            raise ExportError('The output name was created by another process. Choose a new name.')
        os.rename(stage, assets)
        moved_assets = True
        os.rename(assets / 'Project.flp', output)
        completed = True
        return report
    finally:
        cleanup = assets if moved_assets and not completed else stage if not moved_assets else None
        if cleanup is not None and cleanup.exists():
            # Both paths were created exclusively by this export. Resolve and
            # check the intended parent before any recursive cleanup.
            if cleanup.resolve().parent != output.parent.resolve():
                raise ExportError('Temporary export cleanup path failed its parent check.')
            shutil.rmtree(cleanup)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('template', type=Path)
    parser.add_argument('song', type=Path, nargs='?')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.out and args.song:
        print(json.dumps(export_project(args.template, args.song, args.out), indent=2))
    else:
        print(json.dumps(inspect_project(args.template), indent=2))
