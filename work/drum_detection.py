"""Conservative drum identity and kit maps; never infer a kit from pitches alone.

All MIDI notes remain untouched. Maps describe existing keys for musical
analysis; they do not select samples or change the instrument playing them.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from io import BytesIO
import hashlib
import json
import re


DRUM_ROLES = frozenset({'kick', 'snare', 'clap', 'off_snare', 'closed_hat',
                        'open_hat', 'percussion', 'tom'})
VOCAL_BUILDER_MAP = {60: 'kick', 61: 'snare', 62: 'off_snare',
                     63: 'closed_hat', 64: 'open_hat'}
GM_MAP = {35: 'kick', 36: 'kick', 37: 'off_snare', 38: 'snare', 39: 'clap',
          40: 'snare', 41: 'tom', 42: 'closed_hat', 43: 'tom', 44: 'closed_hat',
          45: 'tom', 46: 'open_hat', 47: 'tom', 48: 'tom', 49: 'percussion', 50: 'tom',
          51: 'percussion', 52: 'percussion', 53: 'percussion', 55: 'percussion',
          57: 'percussion', 59: 'percussion'}
GM_MAP.update({note: 'percussion' for note in range(35, 82) if note not in GM_MAP})


def _words(name):
    return ' '.join(re.findall(r'[a-z]+|[0-9]+', str(name).casefold()))


def drum_role(name):
    """Return a named single-drum role; full tokens avoid 'what'/'tomorrow'."""
    text = _words(name)
    if re.search(r'\b(?:steel drums?|steel pan|pan flute)\b', text):
        return None
    for pattern, role in (
        (r'\b(?:off snare|offsnare|ghost snare|ghostsnare|rimshot|rim shot|rim)\b', 'off_snare'),
        (r'\b(?:open (?:hi )?hats?|open hihats?|openhats?|ohh|oh)\b', 'open_hat'),
        (r'\b(?:closed (?:hi )?hats?|closed hihats?|closedhats?|chh|hihats?|hi hats?|hats?|hh)\b', 'closed_hat'),
        (r'\b(?:kicks?|bass drums?|bassdrum|bd)\b', 'kick'),
        (r'\bclaps?\b', 'clap'),
        (r'\b(?:snares?|sd)\b', 'snare'),
        (r'\b(?:toms?|tomtom)\b', 'tom'),
        (r'\b(?:crash|crashes|cymbals?|rides?)\b', 'percussion'),
        (r'\b(?:percussion|perc|shakers?|tambourines?|congas?|bongos?|cowbell)\b', 'percussion'),
    ):
        if re.search(pattern, text):
            return role
    return None


def _generic_drum_name(name):
    text = _words(name)
    return not re.search(r'\bsteel drums?\b', text) and bool(
        re.search(r'\b(?:drums?|drumkit|drum kit|drum monkey)\b', text))


def _json_file(path):
    try:
        if not path.is_file() or path.stat().st_size > 2_000_000:
            return None
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError, UnicodeError):
        return None


def _sidecar_maps(source_path, tracks):
    """Translate source-scoped 16 Bar Studio maps via its Song routing.json."""
    if source_path is None:
        return {}
    folder = Path(source_path).parent
    labels = _json_file(folder / 'Drum Note Map.json')
    routing = _json_file(folder / 'Song routing.json')
    if (not isinstance(labels, dict) or labels.get('format') != '16 Bar Studio drum note map'
            or labels.get('version') != 1 or not isinstance(labels.get('masters'), dict)
            or not isinstance(routing, list)):
        return {}
    result = {}
    for track in tracks:
        matches = [route for route in routing if isinstance(route, dict)
                   and route.get('track_name') == track['name']
                   and track['channels'] == [route.get('channel')]]
        if len(matches) != 1 or not isinstance(matches[0].get('sources'), dict):
            continue
        source_maps = []
        for master, source in matches[0]['sources'].items():
            if not isinstance(source, dict):
                source_maps = []
                break
            bank = labels['masters'].get(master, {})
            source_map = bank.get(str(source.get('track_index')), {}) if isinstance(bank, dict) else {}
            source_maps.append(source_map if isinstance(source_map, dict) else {})
        if not source_maps:
            continue
        mapped = {}
        for pitch in track['pitches']:
            roles = [drum_role(source_map.get(str(pitch), '')) for source_map in source_maps]
            if roles and roles[0] is not None and all(role == roles[0] for role in roles):
                mapped[pitch] = roles[0]
        # Even explicitly Unassigned custom keys establish a custom kit;
        # channel10 must not turn them into an unrelated GM note map.
        if any(source_maps):
            result[track['id']] = mapped
    return result


def _explicit_map(raw, pitches):
    if not isinstance(raw, dict):
        raise ValueError('A drum note map must contain MIDI keys and named drum roles.')
    mapped = {}
    for key, role in raw.items():
        if isinstance(key, bool) or not str(key).isdigit() or not 0 <= int(key) <= 127:
            raise ValueError('Drum-map keys must be MIDI note numbers from 0 to 127.')
        pitch = int(key)
        if pitch in mapped or not isinstance(role, str) or role not in DRUM_ROLES:
            raise ValueError('Drum-map entries must have unique keys and supported drum roles.')
        if pitch in pitches:
            mapped[pitch] = role
    return mapped


def _tree(data, offset):
    """Bounded reader for observed JUCE ValueTree strings in Drum Monkey."""
    visits = 0
    def number(pos):
        if pos >= len(data):
            raise ValueError('Truncated tree number')
        head = data[pos]
        size = head & 127
        if size > 4 or pos + 1 + size > len(data):
            raise ValueError('Unsupported tree number')
        value = int.from_bytes(data[pos + 1:pos + 1 + size], 'little')
        return (-value if head & 128 else value), pos + 1 + size
    def text(pos):
        end = data.find(b'\0', pos, min(len(data), pos + 4097))
        if end < 0:
            raise ValueError('Truncated tree string')
        return data[pos:end].decode('utf-8'), end + 1
    def node(pos, depth):
        nonlocal visits
        visits += 1
        if depth > 24 or visits > 4096:
            raise ValueError('Oversized tree')
        kind, pos = text(pos)
        count, pos = number(pos)
        if not 0 <= count <= 1024:
            raise ValueError('Unsupported tree properties')
        properties = {}
        for _ in range(count):
            key, pos = text(pos)
            size, pos = number(pos)
            if not 0 <= size <= 1_000_000 or pos + size > len(data):
                raise ValueError('Truncated tree value')
            raw, pos = data[pos:pos + size], pos + size
            if raw[:1] == b'\x05':
                properties[key] = raw[1:].rstrip(b'\0').decode('utf-8')
        count, pos = number(pos)
        if not 0 <= count <= 1024:
            raise ValueError('Unsupported child count')
        children = []
        for _ in range(count):
            child, pos = node(pos, depth + 1)
            children.append(child)
        return {'type': kind, 'properties': properties, 'children': children}, pos
    return node(offset, 0)[0]


def _kit_from_state(state):
    start = state.find(b'DrumMonkey\0')
    if start < 0:
        return None
    tree = _tree(state, start)
    parameters = [node for node in tree['children'] if node['type'] == 'Parameters']
    engines = [node for node in tree['children'] if node['type'] == 'DrumSynthesiser']
    if len(parameters) != 1 or len(engines) != 1:
        raise ValueError('Ambiguous Drum Monkey tree')
    pads = [node for node in parameters[0]['children'] if node['type'] == 'Pad']
    sounds = [node for node in engines[0]['children'] if node['type'] == 'Sound']
    if not 1 <= len(pads) <= 8 or len(pads) != len(sounds):
        raise ValueError('Unsupported pad count')
    result = []
    semitones = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    for index, (pad, sound) in enumerate(zip(pads, sounds), 1):
        if pad['properties'].get('id') != f'Pad {index}':
            raise ValueError('Unverified pad order')
        label = pad['properties'].get('kitPiece', '')
        role = drum_role(label)
        match = re.fullmatch(r'([A-G])([#b]?)(-?\d+)', sound['properties'].get('Note', ''))
        if role is None or not match:
            raise ValueError('Unknown pad role or relative note')
        relative = 12 * int(match[3]) + semitones[match[1]] + {'': 0, '#': 1, 'b': -1}[match[2]]
        result.append((relative, role, label))
    # The user authorizes the lowest source note as kick. Use interval offsets,
    # never an assumed MIDI octave number for the plugin's displayed C3.
    anchor = result[0][0]
    if result[0][1] != 'kick' or anchor != min(row[0] for row in result):
        raise ValueError('The source kit does not match first-note kick')
    if len({row[0] for row in result}) != len(result):
        raise ValueError('Overlapping pad notes')
    return {note - anchor: role for note, role, _ in result}


def _preset_kit(state, identifier, channel_name=''):
    start = state.find(b'DrumMonkey\0')
    if start < 0:
        raise ValueError('This saved plugin state is not a readable Drum Monkey kit.')
    tree = _tree(state, start)
    parameters = [node for node in tree['children'] if node['type'] == 'Parameters']
    engines = [node for node in tree['children'] if node['type'] == 'DrumSynthesiser']
    if len(parameters) != 1 or len(engines) != 1:
        raise ValueError('The saved Drum Monkey pad layout is ambiguous.')
    pads = [node for node in parameters[0]['children'] if node['type'] == 'Pad']
    sounds = [node for node in engines[0]['children'] if node['type'] == 'Sound']
    if not 1 <= len(pads) <= 8 or len(pads) != len(sounds):
        raise ValueError('The saved Drum Monkey kit must contain one to eight matching pads and sounds.')
    result, warnings, coordinates = [], [], []
    semitones = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    for slot, (pad, sound) in enumerate(zip(pads, sounds), 1):
        if pad['properties'].get('id') != f'Pad {slot}':
            raise ValueError('The saved pad order could not be matched to its sound settings.')
        properties = pad['properties']
        label = properties.get('kitPiece', '').strip()
        category = properties.get('midiLineCategory', '').strip()
        sample_name = properties.get('sample', '').strip()
        if not sample_name:
            sample_name = properties.get('sampleFile', '').replace('\\', '/').rsplit('/', 1)[-1]
        role = drum_role(label or category)
        note_label = sound['properties'].get('Note', '').strip()
        match = re.fullmatch(r'([A-G])([#b]?)(-?\d{1,2})', note_label)
        coordinate = (12 * int(match[3]) + semitones[match[1]] + {'': 0, '#': 1, 'b': -1}[match[2]]
                      if match else None)
        coordinates.append(coordinate)
        result.append({'slot': slot, 'label': label or category or f'Pad {slot}',
                       'category': category, 'sample_name': sample_name,
                       'role': role, 'note_label': note_label,
                       'note_offset': None})
        if role is None:
            description = label or category or 'unnamed sound'
            if sample_name:
                description += f', sample {sample_name}'
            warnings.append(f'Pad {slot} ({description}) has an unknown sound type; choose its drum role manually.')
        if coordinate is None:
            warnings.append(f'Pad {slot} has an unreadable note label; its MIDI position cannot be inferred.')
    valid = [note for note in coordinates if note is not None]
    anchor = min(valid) if valid else None
    if anchor is not None:
        for pad, coordinate in zip(result, coordinates):
            pad['note_offset'] = coordinate - anchor if coordinate is not None else None
    duplicate_notes = len(set(valid)) != len(valid)
    unsupported_span = bool(valid) and max(valid) - min(valid) > 127
    if duplicate_notes:
        warnings.append('Two pads use the same note. Choose their assignments manually; a MIDI key cannot select them separately.')
    if unsupported_span:
        warnings.append('The saved pad notes span more than the MIDI key range; choose their assignments manually.')
    anchor_slot = coordinates.index(anchor) + 1 if anchor is not None else None
    preset = tree['properties'].get('presetName', '').strip() or 'Drum Monkey kit'
    return {'id': identifier, 'name': channel_name or preset, 'preset': preset,
            'pads': result, 'anchor_pad': anchor_slot,
            'anchor_note_label': result[anchor_slot - 1]['note_label'] if anchor_slot else None,
            'anchor_role': result[anchor_slot - 1]['role'] if anchor_slot else None,
            'note_order': [pad['slot'] for pad in sorted(result, key=lambda row: (row['note_offset'] is None,
                                                                                        row['note_offset'] or 0))],
            'can_apply': len(valid) == len(result) and not duplicate_notes and not unsupported_span,
            'offset_basis': 'Semitones above the lowest saved pad note; no absolute MIDI octave is assumed.',
            'warnings': warnings}


def inspect_drum_preset(path):
    """Read an explicitly chosen FL Studio .fst preset or .flp project.

    Returns every readable Drum Monkey kit; the caller must choose when several
    kits are present. ``pads[].note_offset`` preserves actual semitone gaps,
    anchored to the kit's lowest saved note. To apply to a song, the UI must
    explicitly anchor that lowest note to the song's lowest drum key. Unknown
    sound labels remain ``role=None``. No plugin is loaded and no file changes.

    Drum Monkey's local .dmp files are opaque in the observed version and are
    intentionally rejected instead of decoded speculatively.
    """
    from flp_raw import read_bytes
    import struct
    source = Path(path)
    try:
        size = source.stat().st_size
        if not 22 <= size <= 64_000_000:
            raise ValueError('Choose a Drum Monkey .fst preset or a project smaller than 64 MB.')
        data = source.read_bytes()
    except OSError as exc:
        raise ValueError('The selected drum preset could not be read.') from exc
    if data[:4] != b'FLhd':
        if source.suffix.casefold() == '.dmp':
            raise ValueError('Native Drum Monkey .dmp files are not readable yet. Load the kit in FL Studio and save its wrapper preset as .fst, then choose that file.')
        raise ValueError('Choose an FL Studio Drum Monkey .fst preset or an .flp project containing the kit.')
    try:
        _, events = read_bytes(data)
    except (ValueError, struct.error, IndexError) as exc:
        raise ValueError('The selected FL Studio preset or project is incomplete or unsupported.') from exc
    is_project = any(key == 64 for key, _ in events)
    channel, name, kits, warnings = None, '', [], []
    for index, (key, payload) in enumerate(events):
        if key == 64:
            channel, name = int.from_bytes(payload, 'little'), ''
        elif key == 99:
            channel, name = None, ''
        elif key == 203 and (channel is not None or not is_project):
            name = payload.decode('utf-16-le', errors='replace').rstrip('\0')
        elif key == 213 and b'DrumMonkey\0' in payload:
            identifier = f'channel:{channel}:state:{index}' if channel is not None else f'state:{index}'
            try:
                kit = _preset_kit(payload, identifier, name)
            except (ValueError, UnicodeError, IndexError, struct.error) as exc:
                warnings.append(f'{name or "Drum Monkey"}: {exc}')
                continue
            kit['channel_id'] = channel
            kits.append(kit)
    if not kits:
        detail = ' ' + warnings[0] if warnings else ''
        raise ValueError('No readable Drum Monkey kit was found in that file.' + detail)
    if len(kits) > 1:
        warnings.append('This file contains several Drum Monkey kits. Choose the one used for the song.')
    return {'path': str(source.resolve()), 'format': 'FL Studio project' if is_project else 'FL Studio preset',
            'preset': kits[0]['preset'] if len(kits) == 1 else source.stem,
            'sha256': hashlib.sha256(data).hexdigest(), 'kits': kits, 'warnings': warnings}


def _folder_kit_consensus(folder, source_pitches):
    """Inspect at most32 local FLPs/120MB; disagreement disables suggestions."""
    from flp_raw import read_bytes
    import struct
    files = sorted(folder.glob('*.flp'))
    if not files or len(files) > 32 or sum(path.stat().st_size for path in files) > 120_000_000:
        return None
    consensus, evidence = None, []
    for path in files:
        if path.stat().st_size > 16_000_000:
            return None
        try:
            _, events = read_bytes(path.read_bytes())
            channel, candidates, keys = None, [], {}
            for key, payload in events:
                if key == 64:
                    channel = int.from_bytes(payload, 'little')
                elif key == 213 and b'DrumMonkey\0' in payload:
                    candidates.append((channel, _kit_from_state(payload)))
                elif key == 224:
                    if len(payload) % 24:
                        raise ValueError('Unknown note format')
                    for pos in range(0, len(payload), 24):
                        cid, pitch = struct.unpack_from('<H', payload, pos + 6)[0], struct.unpack_from('<H', payload, pos + 12)[0]
                        keys.setdefault(cid, set()).add(pitch)
            for cid, offsets in candidates:
                # A source kit must actually have played the same source keys.
                if not offsets or not source_pitches <= keys.get(cid, set()):
                    continue
                anchor = min(source_pitches)
                mapped = {anchor + offset: role for offset, role in offsets.items()}
                if not source_pitches <= set(mapped):
                    return None
                if consensus is not None and consensus != mapped:
                    return None
                consensus = mapped
                evidence.append(str(path))
        except (OSError, ValueError, UnicodeError, IndexError, struct.error):
            return None
    return (consensus, evidence) if consensus else None


def _source_kit_maps(source_path, tracks):
    """Return reviewable suggestions from hash-verified source-project kits."""
    if source_path is None:
        return {}
    import mido
    folder = Path(source_path).parent
    project, routing = _json_file(folder / 'Project.json'), _json_file(folder / 'Song routing.json')
    if (not isinstance(project, dict) or project.get('format') != '16 Bar Studio dual master'
            or not isinstance(project.get('source_files'), dict) or not isinstance(routing, list)):
        return {}
    result = {}
    for track in tracks:
        if not (_generic_drum_name(track['name']) or drum_role(track['name'])):
            continue
        matches = [row for row in routing if isinstance(row, dict) and row.get('track_name') == track['name']
                   and track['channels'] == [row.get('channel')]]
        if len(matches) != 1 or not isinstance(matches[0].get('sources'), dict):
            continue
        maps, evidence, cache = [], [], {}
        try:
            for master, route in matches[0]['sources'].items():
                descriptor = project['source_files'][master]
                source = Path(descriptor['path'])
                if source.suffix.lower() not in {'.mid', '.midi'} or source.stat().st_size > 20_000_000:
                    raise ValueError('Unsupported source file')
                data = source.read_bytes()
                if hashlib.sha256(data).hexdigest() != descriptor['sha256']:
                    raise ValueError('Source MIDI changed')
                original = mido.MidiFile(file=BytesIO(data))
                source_track = original.tracks[int(route['track_index'])]
                notes = [message for message in source_track if message.type == 'note_on' and message.velocity > 0]
                if not notes or {message.channel for message in notes} != {route['channel']}:
                    raise ValueError('Source MIDI channel differs')
                pitches = {message.note for message in notes}
                if not set(track['pitches']) <= pitches:
                    raise ValueError('Song contains keys absent from its source')
                identity = str(source.parent), tuple(sorted(pitches))
                if identity not in cache:
                    cache[identity] = _folder_kit_consensus(source.parent, pitches)
                found = cache[identity]
                if not found:
                    raise ValueError('No unique matching source kit')
                maps.append({key: found[0][key] for key in track['pitches']})
                evidence.extend(found[1])
            if maps and all(mapped == maps[0] for mapped in maps):
                result[track['id']] = {'note_map': maps[0], 'evidence': sorted(set(evidence))}
        except (KeyError, OSError, ValueError, TypeError, IndexError, EOFError):
            continue
    return result


def inspect_midi_parts(midi, source_path=None, note_maps=None):
    """Return drum identities and evidence-backed key maps for note tracks.

    ``midi`` is an already-loaded mido MidiFile. ``note_maps`` optionally maps
    raw MIDI track index (integer or decimal string) to {MIDI key: drum role}.
    Identity is independent of instrument assignment: a stale drums->bass
    assignment must not override a positively identified drum part.

    Unknown custom keys are left in ``unmapped_keys``; do not silently discard
    them or assign them a synth. No inferred map changes source pitches.
    """
    if note_maps is not None and not isinstance(note_maps, dict):
        raise ValueError('Drum note maps must be keyed by MIDI track index.')
    tracks, warnings = [], []
    metadata_names = {message.name for track in midi.tracks for message in track
                      if message.is_meta and message.type == 'track_name'}
    for index, midi_track in enumerate(midi.tracks):
        attacks = [message for message in midi_track
                   if message.type == 'note_on' and message.velocity > 0]
        if not attacks:
            continue
        name = next((m.name.strip() for m in midi_track if m.type == 'track_name' and m.name.strip()), '')
        if not name:
            name = next((m.name.strip() for m in midi_track if m.type == 'instrument_name' and m.name.strip()), f'track_{index + 1}')
        tracks.append({'id': str(index), 'track_index': index, 'name': name,
                       'channels': sorted({m.channel for m in attacks}), 'note_count': len(attacks),
                       'pitches': dict(sorted(Counter(m.note for m in attacks).items()))})
    sidecar_maps = _sidecar_maps(source_path, tracks)
    source_kits = _source_kit_maps(source_path, tracks)
    project = _json_file(Path(source_path).parent / 'Project.json') if source_path is not None else None
    custom_document = (bool(sidecar_maps) or 'Vocal Beat Builder' in metadata_names
                       or isinstance(project, dict) and project.get('format') == '16 Bar Studio dual master')
    explicit = {str(key): value for key, value in (note_maps or {}).items()}
    for track in tracks:
        role = drum_role(track['name'])
        named = bool(role or _generic_drum_name(track['name']))
        single_channel = len(track['channels']) == 1
        melodic_name = bool(re.search(r'\b(?:bass|sub|808|lead|pad|keys|piano|guitar|arp|pluck|melody|chords?|texture|steel drums?)\b', _words(track['name'])))
        explicit_gm = bool(re.search(r'\b(?:gm|general midi)\b', _words(track['name'])))
        gm = (track['channels'] == [9] and not custom_document and not (melodic_name and not named)
              and (not _generic_drum_name(track['name']) or explicit_gm))
        is_vocal_builder = ('Vocal Beat Builder' in metadata_names
                            and track['name'] == 'Combined drums C5-E5')
        mapping, origin = {}, None
        if track['id'] in explicit:
            mapping = _explicit_map(explicit[track['id']], track['pitches'])
            origin = 'Explicit drum note map'
        elif sidecar_maps.get(track['id']):
            mapping, origin = sidecar_maps[track['id']], '16 Bar Studio source drum labels'
        elif track['id'] in source_kits:
            mapping, origin = source_kits[track['id']]['note_map'], 'Source kit FLP suggestion'
        elif track['id'] in sidecar_maps:
            origin = '16 Bar Studio custom kit with unassigned labels'
        elif is_vocal_builder:
            mapping = {key: VOCAL_BUILDER_MAP[key] for key in track['pitches'] if key in VOCAL_BUILDER_MAP}
            origin = 'Vocal Beat Builder C5-E5 map'
        elif role:
            mapping = {key: role for key in track['pitches']}
            origin = 'Named single-drum track'
        elif gm:
            mapping = {key: GM_MAP[key] for key in track['pitches'] if key in GM_MAP}
            origin = 'General MIDI percussion channel10'
        is_drums = bool(named or gm or origin)
        if not single_channel and is_drums:
            warnings.append(f"{track['name']}: mixed MIDI channels need separating before drum routing.")
            mapping = {}
        track.update(is_drums=is_drums, note_map=mapping, mapping_source=origin,
                     unmapped_keys=sorted(set(track['pitches']) - set(mapping)) if is_drums else [])
        if origin == 'Source kit FLP suggestion':
            track['mapping_evidence'] = source_kits[track['id']]['evidence']
            track['mapping_requires_review'] = True
        if (is_drums and (re.search(r'\b(?:crash|crashes|cymbals?|rides?)\b', _words(track['name']))
                           or origin == 'General MIDI percussion channel10'
                           and set(track['pitches']) & {49,51,52,53,55,57,59})):
            warnings.append(f"{track['name']}: cymbal/ride lines use the generic percussion role.")
        if is_drums and track['unmapped_keys']:
            warnings.append(f"{track['name']}: drum keys {', '.join(map(str, track['unmapped_keys']))} have no verified sound map; preserve them and choose their drum sounds.")
    unknown_overrides = set(explicit) - {track['id'] for track in tracks}
    if unknown_overrides:
        raise ValueError('A drum map refers to a missing or empty MIDI track.')
    return {'tracks': tracks, 'warnings': warnings}


def expand_drum_parts(parts, midi, source_path=None, role_overrides=None):
    """Split identified kits into at most eight editable, pitch-scoped parts.

    ``parts`` is the existing (name, raw_track_index, NoteEvent-list) sequence.
    Original NoteEvent objects/timing are retained. Unknown roles use generic
    percussion, except the first drum slot, which the user reserves for kick.
    ``role_overrides`` keys are stable virtual IDs like '5:drum:60'.
    """
    if role_overrides is not None and not isinstance(role_overrides, dict):
        raise ValueError('Drum role choices must be keyed by drum lane ID.')
    detected = inspect_midi_parts(midi, source_path)
    by_id = {track['id']: track for track in detected['tracks']}
    warnings = [warning for warning in detected['warnings']
                if 'have no verified sound map' not in warning]
    overrides = {str(key): value for key, value in (role_overrides or {}).items()}
    expanded, metadata = [], []
    for name, source_index, notes in parts:
        track = by_id.get(str(source_index))
        if not track or not track['is_drums']:
            expanded.append((name, source_index, notes))
            continue
        if len(track['channels']) != 1:
            raise ValueError(f'{name}: separate mixed MIDI channels before generating drum parts.')
        pitches = sorted({note.note for note in notes})
        for pitch in pitches:
            slot = len(metadata) + 1
            if slot > 8:
                raise ValueError('This builder supports up to eight drum lanes. Combine or remove extra drum keys before generating.')
            identifier = f'{source_index}:drum:{pitch}'
            mapped = track['note_map'].get(pitch)
            needs_role = False
            if identifier in overrides:
                role = overrides[identifier]
                if not isinstance(role, str) or role not in DRUM_ROLES:
                    raise ValueError(f'Drum {slot}: choose a supported drum role.')
                origin = 'Your drum role selection'
            elif mapped:
                role, origin = mapped, track['mapping_source']
                needs_role = bool(track.get('mapping_requires_review')) and not (slot == 1 and role == 'kick')
            elif slot == 1:
                role, origin = 'kick', 'Your first-drum-is-kick convention'
            else:
                role, origin, needs_role = 'percussion', 'Unassigned custom drum key', True
            expanded.append((f'Drum {slot}', identifier, [note for note in notes if note.note == pitch]))
            metadata.append({'id': identifier, 'source_track': source_index, 'source_name': name,
                             'source_note': pitch, 'slot': slot, 'role': role,
                             'role_source': origin, 'needs_role': needs_role})
            if track.get('mapping_evidence'):
                metadata[-1]['role_evidence'] = track['mapping_evidence']
    if set(overrides) - {row['id'] for row in metadata}:
        raise ValueError('A saved drum role refers to a drum key that is not in this MIDI.')
    pending = [row for row in metadata if row['needs_role']]
    if pending:
        warnings.append('Review roles for ' + ', '.join(f"Drum {row['slot']}" for row in pending)
                        + '. Suggestions remain editable; unidentified keys are labelled Percussion for analysis only.')
    return expanded, metadata, warnings
