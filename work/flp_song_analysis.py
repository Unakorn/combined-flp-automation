"""Read a saved FL Playlist for beat-domain automation analysis, without editing it.

The analysis MIDI is an internal rhythmic sketch, not a replacement/export of
the source music. Native plugin states, note expression, audio and automation
remain in the original FLP. Pattern trims use signed integer ticks; channel-clip
offsets are a different format and are deliberately not interpreted as notes.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
from pathlib import Path
import struct

from mido import Message, MetaMessage, MidiFile, MidiTrack
from flp_raw import read_bytes, encode_fl


class AnalysisError(ValueError):
    pass


MAX_BYTES = 256 * 1024 * 1024
MAX_NOTES = 2_000_000
CONTROLLERS = {'fruity keyboard controller', 'fruity envelope controller',
               'midi out', 'dashboard', 'control surface', 'fruity formula controller'}


def _u(payload, default=0):
    return int.from_bytes(payload, 'little') if payload else default


def _text(payload):
    try:
        return payload.decode('utf-16-le').rstrip('\0')
    except UnicodeError as exc:
        raise AnalysisError('An FL project label has an unsupported text encoding.') from exc


def _plugin_label(wrapper, state):
    """Read a wrapper display-name chunk when the known envelope is present."""
    if wrapper.casefold() != 'fruity wrapper' or len(state) < 16:
        return wrapper
    if struct.unpack_from('<I', state)[0] != 12:
        return wrapper
    offset = 4
    while offset + 12 <= len(state):
        kind, size = struct.unpack_from('<IQ', state, offset)
        offset += 12
        if size > len(state) - offset:
            return wrapper
        payload = state[offset:offset + size]
        if kind == 54:
            return payload.split(b'\0', 1)[0].decode('utf-8', errors='replace') or wrapper
        offset += size
    return wrapper


def _single_pattern_clip(pattern, duration):
    """A neutral FL 26 pattern placement, observed in a native saved project.

    The seed is the untrimmed, unmuted, normal-speed pattern clip from the
    FL 26.1.5 source used for the host-confirmed native-note-order fixture.
    Preserve its version-specific neutral fields; change only known placement
    and identity fields. This is only used with an otherwise empty Playlist.
    """
    pid = pattern['id']
    if not 1 <= pid <= 0xafff or not 0 < duration <= 0xffffffff:
        raise AnalysisError('The single pattern is outside the supported Playlist range.')
    clip = bytearray.fromhex(
        '000000000050045000f00000f30100007800400040648080ffffffffffffffff'
        '92000000010000000000000000000000000000000000803f0000000000000000'
        '000000000000f03f00000000ffffffff0000000000000000')
    struct.pack_into('<IHHIHH', clip, 0, 0, 0x5000, 0x5000 + pid, duration, 499, 0)
    struct.pack_into('<I', clip, 32, 1)
    return bytes(clip)


def read_project(path, *, data_override=None, allow_single_pattern=False):
    """Return private, lossless inspection primitives for the preservation writer.

    `header`, `events` and `data` are bytes; do not serialize this dictionary as
    the public preview. `analyze_flp` returns the JSON-friendly public model.
    Opting into `allow_single_pattern` can infer one placement for an empty
    FL 26 Playlist. It never changes these source bytes: `inferred_playlist`
    and `layout.inferred_clip` tell the exporter what to add to its new copy.
    """
    path = Path(path).expanduser().resolve()
    if data_override is None:
        if path.stat().st_size > MAX_BYTES:
            raise AnalysisError('This FLP exceeds the 256 MB inspection limit.')
        data = path.read_bytes()
    else:
        data = bytes(data_override)
    if len(data) > MAX_BYTES:
        raise AnalysisError('This FLP exceeds the 256 MB inspection limit.')
    header, events = read_bytes(data)
    if encode_fl(header, events) != data:
        raise AnalysisError('The FLP event encoding cannot be preserved exactly.')
    if _u(header[:2]) != 0:
        raise AnalysisError('Choose a complete saved FLP project, not an instrument preset.')
    version = next((p.decode('ascii').rstrip('\0') for k, p in events if k == 199), '')
    try:
        major = int(version.split('.')[0])
    except ValueError:
        raise AnalysisError('The saved FL Studio version could not be identified.') from None
    if not 21 <= major <= 26:
        raise AnalysisError('Save a copy in FL Studio 21–26 before analysing this project.')
    ppq = _u(header[4:6])
    if not 24 <= ppq <= 9600:
        raise AnalysisError('The FLP musical time resolution is unsupported.')
    if any(k in {1, 68, 129, 141, 222} for k, _ in events):
        raise AnalysisError('Legacy pattern data needs to be resaved in current FL Studio first.')
    starts = [i for i, (k, _) in enumerate(events) if k == 64]
    arrangements = [i for i, (k, _) in enumerate(events) if k == 99]
    if len(arrangements) != 1:
        raise AnalysisError('This version needs one Playlist arrangement. Save the desired arrangement as a separate FLP copy.')
    arrangement = arrangements[0]
    if not starts or len(starts) != _u(header[2:4]) or starts[-1] >= arrangement:
        raise AnalysisError('The project channel layout does not match its header.')
    channels = []
    for start, stop in zip(starts, starts[1:] + [arrangement]):
        block = events[start:stop]
        d = dict(block)
        misc = d.get(215, b'')
        wrapper = _text(d.get(201, b''))
        channels.append({
            'id': _u(d[64]), 'name': _text(d.get(203, d.get(192, b''))),
            'plugin': wrapper, 'plugin_name': _plugin_label(wrapper, d.get(213, b'')),
            'type': _u(d.get(21), 255), 'enabled': bool(_u(d.get(0), 1)),
            'mixer_insert': _u(d.get(104)), 'sample': _text(d.get(196, b'')),
            'root_note': _u(d.get(135), 60),
            'key_low': struct.unpack_from('<I', misc, 68)[0] if len(misc) >= 76 else 0,
            'key_high': struct.unpack_from('<I', misc, 72)[0] if len(misc) >= 76 else 131,
            'children': [_u(p) for k, p in block if k == 94],
            'layer_flags': _u(d.get(144)), 'start': start, 'stop': stop,
            'plugin_state_sha256': hashlib.sha256(d.get(213, b'')).hexdigest(),
        })
    if len({c['id'] for c in channels}) != len(channels):
        raise AnalysisError('Duplicate native channel IDs make the project ambiguous.')
    patterns = {}
    current = None
    marker = None
    loop_channel = None
    playlist_markers = []
    in_arrangement = False
    for index, (key, payload) in enumerate(events):
        if key == 99:
            in_arrangement, current, marker = True, None, None
        elif key == 65 and not in_arrangement:
            pid = _u(payload)
            current = patterns.setdefault(pid, {'id': pid, 'name': f'Pattern {pid}', 'notes': [],
                'markers': [], 'automation': [], 'looped': False, 'loop_channels': {}, 'length': 0})
            marker, loop_channel = None, None
        elif key == 224:
            if current is None or len(payload) % 24:
                raise AnalysisError('An unsupported piano-roll note block was found.')
            for offset in range(0, len(payload), 24):
                tick, flags, cid, duration, pitch = struct.unpack_from('<IHHIH', payload, offset)
                current['notes'].append({'tick': tick, 'duration': duration, 'note': pitch,
                    'velocity': payload[offset + 21], 'channel_id': cid,
                    'flags': flags, 'color': payload[offset + 19]})
        elif key == 223 and current is not None:
            if len(payload) % 12:
                raise AnalysisError('A pattern automation block has an unsupported layout.')
            current['automation'].extend(struct.unpack_from('<III', payload, i) for i in range(0, len(payload), 12))
        elif key == 193 and current is not None:
            current['name'] = _text(payload)
        elif key == 26 and current is not None:
            current['looped'] = bool(_u(payload))
        elif key == 164 and current is not None:
            current['length'] = _u(payload)
        elif key == 160 and current is not None:
            loop_channel = _u(payload)
        elif key == 161 and current is not None and loop_channel is not None:
            current['loop_channels'][loop_channel] = int.from_bytes(payload, 'little', signed=True)
        elif key == 148:
            packed = _u(payload)
            marker = {'tick': packed & 0x00ffffff, 'kind': packed >> 24,
                      'name': '', 'numerator': 4, 'denominator': 4}
            if in_arrangement:
                playlist_markers.append(marker)
            elif current is not None:
                current['markers'].append(marker)
        elif marker is not None and key in {33, 34, 205}:
            marker[{33: 'numerator', 34: 'denominator', 205: 'name'}[key]] = _text(payload) if key == 205 else _u(payload)
    tracks = {}
    for key, payload in events[arrangement:]:
        if key == 238:
            if len(payload) < 13:
                raise AnalysisError('A Playlist track record is incomplete.')
            number = struct.unpack_from('<I', payload)[0]
            tracks[number] = {'number': number, 'enabled': bool(payload[12]),
                              'grouped': bool(payload[46]) if len(payload) > 46 else False}
    playlists = [(i, p) for i, (k, p) in enumerate(events) if k == 233]
    inferred_playlist = None
    inferred_clip = None
    if not any(p for _, p in playlists):
        if not allow_single_pattern:
            raise AnalysisError('The Playlist contains no arranged clips. Place the song patterns in the Playlist and save it first.')
        if major != 26 or len(playlists) != 1:
            raise AnalysisError('Automatic single-pattern placement needs an FL Studio 26 project with one empty Playlist.')
        candidates = [p for p in patterns.values() if p['notes']]
        if len(candidates) != 1:
            raise AnalysisError('The empty Playlist needs exactly one nonempty song pattern. Arrange the desired patterns in FL Studio and save a copy first.')
        pattern = candidates[0]
        if any(p['automation'] for p in patterns.values() if p['id'] != pattern['id']):
            raise AnalysisError('Another pattern contains event automation. Arrange the desired patterns in FL Studio before generating automation.')
        if not tracks.get(1, {}).get('enabled', True):
            raise AnalysisError('Playlist track 1 is muted. Enable it in a saved copy before placing the single song pattern.')
        signature = (next((_u(p) for k, p in events if k == 17), 4),
                     next((_u(p) for k, p in events if k == 18), 4))
        if signature != (4, 4):
            raise AnalysisError('Automatic single-pattern placement currently needs a 4/4 song.')
        step_gate = max(1, ppq // 4)
        last_tick = max(n['tick'] + (n['duration'] or step_gate) for n in pattern['notes'])
        last_tick = max(last_tick, max((tick + 1 for tick, _, _ in pattern['automation']), default=0))
        bar = ppq * 4
        duration = ((last_tick + bar - 1) // bar) * bar
        inferred_clip = _single_pattern_clip(pattern, duration)
        inferred_playlist = dict(pattern_id=pattern['id'], name=pattern['name'],
                                 position=0, duration=duration, track=1)
        # A virtual placement supplies note analysis and a verified clip seed.
        # The source events, raw bytes, and their hash remain completely intact.
        playlists = [(playlists[0][0], inferred_clip)]
    by_id = {c['id']: c for c in channels}
    size = None
    for candidate in (88, 60, 32):
        if any(len(p) % candidate for _, p in playlists):
            continue
        records = [p[j:j + candidate] for _, p in playlists for j in range(0, len(p), candidate)]
        if all(struct.unpack_from('<H', r, 4)[0] == 0x5000 and
               ((struct.unpack_from('<H', r, 6)[0] - 0x5000 in patterns) if struct.unpack_from('<H', r, 6)[0] > 0x5000
                else struct.unpack_from('<H', r, 6)[0] in by_id) and
               struct.unpack_from('<H', r, 12)[0] <= 499 for r in records):
            size = candidate
            break
    if size is None:
        raise AnalysisError('This Playlist uses an unsupported clip record layout.')
    clips = []
    for event_index, payload in playlists:
        for offset in range(0, len(payload), size):
            r = payload[offset:offset + size]
            pos, base, item, length, reverse_row, group = struct.unpack_from('<IHHIHH', r)
            flags = struct.unpack_from('<H', r, 18)[0]
            row = 500 - reverse_row
            clip = {'position': pos, 'duration': length, 'track': row, 'flags': flags,
                    'group': group, 'muted': bool(flags & 0x2000),
                    'track_enabled': tracks.get(row, {}).get('enabled', True),
                    'event_index': event_index, 'record_offset': offset,
                    'serial': struct.unpack_from('<I', r, 32)[0] if size >= 60 else None}
            if item > base:
                a, b = struct.unpack_from('<ii', r, 24)
                clip.update(kind='pattern', pattern_id=item - base,
                            start_offset=0 if a == -1 else a, end_offset=None if b == -1 else b,
                            scale=struct.unpack_from('<d', r, 64)[0] if size == 88 else 1.0)
            else:
                clip.update(kind='channel', channel_id=item)
            clips.append(clip)
    tempo = next((_u(p) / 1000 for k, p in events if k == 156), None)
    signature = (next((_u(p) for k, p in events if k == 17), 4),
                 next((_u(p) for k, p in events if k == 18), 4))
    if tempo is None or not 10 <= tempo <= 522:
        raise AnalysisError('The saved project tempo is unsupported.')
    first_pattern = next((i for i, (k, _) in enumerate(events) if k == 65), starts[0])
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(),
            'version': version, 'ppq': ppq, 'bpm': tempo, 'time_signature': signature,
            'header': header, 'events': events, 'data': data, 'channels': channels,
            'patterns': patterns, 'clips': clips, 'tracks': tracks, 'markers': playlist_markers,
            'inferred_playlist': inferred_playlist,
            'play_truncated_notes': bool(next((_u(p) for k, p in events[:first_pattern] if k == 30), 0)),
            'layout': {'first_channel': starts[0], 'arrangement': arrangement,
                       'record_size': size, 'pattern_ids': set(patterns),
                       'playlist_index': playlists[0][0],
                       'inferred_clip': inferred_clip,
                       'next_clip_serial': max((c['serial'] or 0 for c in clips), default=0) + 1,
                       'clip': next((p[o:o + size] for i, p in playlists for o in range(0, len(p), size)
                                     if struct.unpack_from('<H', p, o + 6)[0] > 0x5000), None)}}


def analyze_flp(path, *, max_instruments=20, data_override=None, allow_single_pattern=False):
    """Expand the audible arranged piano-roll notes into actual instrument parts.

    Original audio clips and control data are inventoried and preserved by the
    caller; they are not fabricated into MIDI notes. Unsupported musical loops,
    crossfading/random Layers and ambiguous overlapping clips fail explicitly.
    `allow_single_pattern` is an explicit opt-in to previewing one full-pattern
    placement when a supported project has no Playlist clips yet.
    """
    project = read_project(path, data_override=data_override, allow_single_pattern=allow_single_pattern)
    if not isinstance(max_instruments, int) or not 1 <= max_instruments <= 128:
        raise AnalysisError('The instrument limit must be between 1 and 128.')
    if project['time_signature'] != (4, 4):
        raise AnalysisError('This automation engine currently needs a 4/4 song; the FLP was not changed.')
    channels = {c['id']: c for c in project['channels']}
    warnings = set()
    if project['inferred_playlist'] is not None:
        placement = project['inferred_playlist']
        warnings.add(f"The Playlist is empty. Its single song pattern '{placement['name']}' will be placed at bar 1 on Playlist track 1 in the generated copy; the original FLP stays unchanged.")
    audible_clips = [c for c in project['clips'] if not c['muted'] and c['track_enabled'] and c['duration'] > 0]
    duration = max((c['position'] + c['duration'] for c in audible_clips), default=0)
    pattern_clips = [c for c in audible_clips if c['kind'] == 'pattern']
    # Two clips on one row can interrupt held notes. Refuse that ambiguity.
    by_track = defaultdict(list)
    for clip in pattern_clips:
        by_track[clip['track']].append(clip)
    for row, clips in by_track.items():
        last_end = -1
        for c in sorted(clips, key=lambda x: x['position']):
            if c['position'] < last_end:
                raise AnalysisError(f'Playlist track {row} contains overlapping pattern clips. Move them to separate tracks in a saved copy before analysis.')
            last_end = c['position'] + c['duration']
    audio_clips, automation_clips = [], []
    for clip in audible_clips:
        if clip['kind'] != 'channel':
            continue
        channel = channels[clip['channel_id']]
        if not channel['enabled']:
            continue
        if channel['type'] == 4:
            audio_clips.append(dict(clip, name=channel['name']))
        elif channel['type'] == 5:
            automation_clips.append(dict(clip, name=channel['name']))
        else:
            raise AnalysisError(f"Playlist channel clip '{channel['name']}' has an unsupported type.")
    if audio_clips:
        warnings.add(f'{len(audio_clips)} audio clips stay in the FLP unchanged; note-driven movement is generated only for instrument patterns.')
    if automation_clips:
        warnings.add(f'{len(automation_clips)} existing Automation Clips stay unchanged. Analysis uses musical ticks, including when existing tempo automation changes playback speed.')

    def destinations(cid, pitch, chain=()):
        if cid not in channels:
            raise AnalysisError(f'A placed pattern references missing channel {cid}.')
        c = channels[cid]
        if not c['enabled']:
            return []
        if c['type'] == 3:
            if cid in chain:
                raise AnalysisError('A Layer contains a circular child relationship.')
            if c['layer_flags'] & 3:
                raise AnalysisError(f"Layer '{c['name']}' uses random or crossfade mode. Burn that Layer to ordinary child notes in a copy first.")
            if c['root_note'] != 60:
                raise AnalysisError(f"Layer '{c['name']}' has transposed root-note routing, which needs to be flattened before analysis.")
            if not c['children']:
                warnings.add(f"Layer '{c['name']}' has no children, so its notes have no playable destination.")
            result = []
            for child in c['children']:
                if child not in channels:
                    raise AnalysisError(f"Layer '{c['name']}' references a missing child channel.")
                cc = channels[child]
                # Native unrestricted channels commonly store upper bound 256.
                if cc['key_low'] <= pitch <= cc['key_high']:
                    result.extend(destinations(child, pitch, chain + (cid,)))
            return result
        if c['plugin'].casefold() in CONTROLLERS or c['type'] == 5:
            return []
        if c['type'] not in {0, 2}:
            raise AnalysisError(f"Notes address unsupported channel '{c['name']}' (type {c['type']}).")
        return [cid]

    played = defaultdict(list)
    source_patterns = defaultdict(set)
    source_layers = defaultdict(set)
    markers = []
    tempo_events = []
    expanded_count = 0
    for clip in pattern_clips:
        pattern = project['patterns'][clip['pattern_id']]
        if pattern['length']:
            raise AnalysisError(f"'{pattern['name']}' has an explicit pattern-loop length. Burn its looping channels in a saved copy before analysis.")
        if any(v != 0 for v in pattern['loop_channels'].values()) and pattern['looped']:
            raise AnalysisError(f"'{pattern['name']}' has per-channel looping. Use FL Studio's Burn all looping channels in a copy before analysis.")
        if any(m['kind'] == 11 and m['tick'] > 0 for m in pattern['markers']) and pattern['looped']:
            raise AnalysisError(f"'{pattern['name']}' has a custom loop marker. Burn its looping channels in a saved copy first.")
        if not math.isclose(clip['scale'], 1.0, rel_tol=0, abs_tol=1e-12):
            raise AnalysisError(f"'{pattern['name']}' has a stretched pattern clip. Consolidate its note timing in a saved copy first.")
        start = clip['start_offset']
        finish = start + clip['duration']
        if start < 0 or clip['end_offset'] is not None and clip['end_offset'] < start:
            raise AnalysisError(f"'{pattern['name']}' has unsupported reversed or negative clip trims.")
        if clip['end_offset'] is not None:
            finish = min(finish, clip['end_offset'])
        for marker in pattern['markers']:
            if marker['kind'] == 8 and (marker['numerator'], marker['denominator']) != (4, 4):
                raise AnalysisError('A placed pattern changes time signature; only 4/4 analysis is supported yet.')
            if start <= marker['tick'] < finish and marker['kind'] in {0, 8}:
                markers.append(dict(marker, tick=clip['position'] + marker['tick'] - start,
                                    source='pattern', pattern_id=pattern['id']))
        for tick, target, value in pattern['automation']:
            if target == 0x40000005 and start <= tick < finish:
                tempo_events.append((clip['position'] + tick - start, value / 1000))
        for note in pattern['notes']:
            if not note['velocity']:
                continue
            if note['velocity'] > 128:
                raise AnalysisError('A native note velocity lies outside the supported 0–128 range.')
            # A zero-length step-sequencer hit triggers a one-shot. A sixteenth
            # note is a rhythmic analysis proxy; its actual sample tail stays
            # governed by the unchanged source instrument.
            note_duration = note['duration']
            if not note_duration:
                note_duration = max(1, project['ppq'] // 4)
                warnings.add('Step-sequencer one-shots use a sixteenth-note analysis gate; the original sample envelopes and tails stay unchanged.')
            a, b = note['tick'], note['tick'] + note_duration
            if b <= start or a >= finish:
                continue
            if a < start and not project['play_truncated_notes']:
                continue
            a, b = max(a, start), min(b, finish)
            if b <= a:
                continue
            targets = destinations(note['channel_id'], note['note'])
            if not targets:
                continue
            if not 0 <= note['note'] <= 127:
                raise AnalysisError('An arranged instrument note is above MIDI note 127; that range is not supported for analysis yet.')
            if note['flags'] & 8:
                warnings.add('Slide-note starts inform rhythm analysis; native slide/pitch behavior stays untouched in the FLP.')
            for cid in targets:
                played[cid].append({'tick': clip['position'] + a - start, 'duration': b - a,
                                    'note': note['note'], 'velocity': note['velocity'],
                                    'color': note['color']})
                source_patterns[cid].add(pattern['id'])
                if note['channel_id'] != cid:
                    source_layers[cid].add(note['channel_id'])
                expanded_count += 1
                if expanded_count > MAX_NOTES:
                    raise AnalysisError('The expanded Playlist exceeds the two-million-note analysis limit.')
    for marker in project['markers']:
        if marker['kind'] == 8 and (marker['numerator'], marker['denominator']) != (4, 4):
            raise AnalysisError('The Playlist changes time signature; only 4/4 analysis is supported yet.')
        if marker['kind'] in {0, 8}:
            markers.append(dict(marker, source='playlist'))
    if not played:
        raise AnalysisError('There are no enabled, arranged instrument notes to automate. Audio-only songs are preserved but cannot supply instrument note activity.')
    if len(played) > max_instruments:
        raise AnalysisError(f'The Playlist plays {len(played)} instruments; this version supports at most {max_instruments}.')
    if len({bpm for _, bpm in tempo_events} | {project['bpm']}) > 1:
        warnings.add('Existing pattern tempo changes are preserved. Automation is calculated in musical ticks; the internal analysis MIDI uses the saved project tempo.')
    voices = []
    for index, cid in enumerate(sorted(played), 1):
        channel = channels[cid]
        voices.append({k: v for k, v in channel.items() if k not in {'start', 'stop'}} | {
            'channel_id': cid, 'analysis_track_index': index,
            'notes': sorted(played[cid], key=lambda n: (n['tick'], n['note'], n['duration'], n['velocity'])),
            'source_pattern_ids': sorted(source_patterns[cid]), 'source_layer_ids': sorted(source_layers[cid]),
            'note_count': len(played[cid])})
    markers = sorted({(m['tick'], m['kind'], m['name']): m for m in markers}.values(), key=lambda m: (m['tick'], m['kind'], m['name']))
    return {k: project[k] for k in ('path', 'sha256', 'version', 'ppq', 'bpm')} | {
        'time_signature': [4, 4], 'duration': duration, 'duration_ticks': duration,
        'tempo': project['bpm'], 'end_tick': duration,
        'inferred_playlist': project['inferred_playlist'],
        'voices': voices, 'instruments': [{k: v for k, v in c.items() if k not in {'start', 'stop', 'notes'}} for c in voices],
        'channels': [{k: v for k, v in c.items() if k not in {'start', 'stop'}} for c in project['channels']],
        'markers': markers, 'audio_clips': audio_clips, 'automation_clips': automation_clips,
        'pattern_tempo_events': [{'tick': t, 'bpm': b} for t, b in sorted(set(tempo_events))],
        'note_count': expanded_count, 'instrument_count': len(voices), 'warnings': sorted(warnings),
        'analysis_only': True, 'timing_domain': 'musical_ticks', 'supported': True}


def build_analysis_midi(model):
    """Make a nominal-tempo rhythmic sketch; never use this to replace FL notes.

    Overlapping identical pitches use distinct MIDI channels, keeping note-off
    pairing exact even for nested note lengths. It is still one track per voice.
    Native velocity128 is clamped to MIDI127 in this analysis-only transport.
    """
    midi = MidiFile(type=1, ticks_per_beat=model['ppq'], charset='utf-8')
    conductor = MidiTrack()
    midi.tracks.append(conductor)
    events = [(0, MetaMessage('track_name', name='FLP analysis - source music remains unchanged')),
              (0, MetaMessage('set_tempo', tempo=round(60_000_000 / model['bpm']))),
              (0, MetaMessage('time_signature', numerator=4, denominator=4))]
    for marker in model['markers']:
        if marker['kind'] == 0 and marker['name'] and 0 <= marker['tick'] <= model['duration']:
            events.append((marker['tick'], MetaMessage('marker', text=marker['name'])))
    last = 0
    for tick, event in sorted(events, key=lambda e: e[0]):
        conductor.append(event.copy(time=tick - last))
        last = tick
    conductor.append(MetaMessage('end_of_track', time=max(0, model['duration'] - last)))
    for voice in model['voices']:
        track = MidiTrack([MetaMessage('track_name', name=voice['name'] or f"Instrument {voice['channel_id']}")])
        midi.tracks.append(track)
        timed = []
        busy = defaultdict(lambda: [0] * 16)
        for note in voice['notes']:
            start, end, pitch = note['tick'], note['tick'] + note['duration'], note['note']
            free = next((i for i, until in enumerate(busy[pitch]) if until <= start), None)
            if free is None:
                raise AnalysisError('More than 16 simultaneous notes of the same pitch need direct note-model analysis instead of MIDI transport.')
            busy[pitch][free] = end
            timed.append((start, 1, Message('note_on', channel=free, note=pitch, velocity=min(127, note['velocity']))))
            timed.append((end, 0, Message('note_off', channel=free, note=pitch, velocity=0)))
        last = 0
        for tick, _, message in sorted(timed, key=lambda e: (e[0], e[1])):
            track.append(message.copy(time=tick - last))
            last = tick
        track.append(MetaMessage('end_of_track', time=max(0, model['duration'] - last)))
    return midi
