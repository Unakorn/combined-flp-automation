"""Review audio-driven EQ proposals and serialize plugin-verified native gains.

Installed Pro-Q controller readouts supply actual centers and saved gain values.
An FL-saved link anchors native parameter addresses. Gain moves are relative to
saved settings; opaque EQ state is never rewritten.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import os
from pathlib import Path
import re
import struct
import tempfile

import soundfile as sf

from drum_template import _mixer
from flp_raw import encode_fl, read_bytes
from flp_song_analysis import read_project
from plugin_templates import plugin_records, _initializer_rows, _validate_parameter_baseline
from eq_audio import infer_role
from eq_vst3_probe import probe_project, GAIN_IDS
from stem_native import build_channel, build_curve, build_link, build_playlist_clip, parse_points
from stem_project import _file_digest, _interpolate, FADER_DB


CENTERS_HZ = (76.0, 254.0, 806.0, 2460.0, 7519.0)
BANDS = (3, 4, 5, 6, 7)


def inspect_eq_stems(path):
    """Same strict continuous-stem checks, but mixer 0 remains analysis context."""
    project = read_project(path)
    if project['time_signature'] != (4, 4):
        raise ValueError('EQ analysis requires a constant-tempo 4/4 stem project.')
    if sum(k == 233 for k, _ in project['events']) != 1:
        raise ValueError('EQ analysis requires one Playlist arrangement.')
    for key, payload in project['events']:
        if key in (226, 227):
            if len(payload) != 20:
                raise ValueError('An existing automation link has an unknown layout.')
            if struct.unpack_from('<I', payload, 8)[0] == 0x40000005:
                raise ValueError('Render constant-tempo stems before analyzing tempo automation.')
        if key == 223:
            if len(payload) % 12:
                raise ValueError('Unsupported existing event automation.')
            if any(t == 0x40000005 for _, t, _ in struct.iter_unpack('<III', payload)):
                raise ValueError('Render constant-tempo stems before analyzing tempo automation.')
    by_id = {c['id']: c for c in project['channels']}
    mixer_blocks, _ = _mixer(project['events'])
    mixer_names = {}
    for insert, (_, _, block) in enumerate(mixer_blocks):
        labels = [payload for key, payload in block if key == 204]
        if len(labels) > 1:
            raise ValueError(f'Mixer insert {insert} has ambiguous saved names.')
        mixer_names[insert] = labels[0].decode('utf-16-le').rstrip('\0') if labels else ''
    placements = defaultdict(list)
    for clip in project['clips']:
        if clip['kind'] == 'pattern' and not clip['muted'] and clip['track_enabled']:
            if project['patterns'][clip['pattern_id']]['notes']:
                raise ValueError('Use a stem-only arrangement; instrument patterns are not audio evidence.')
        elif clip['kind'] == 'channel' and by_id[clip['channel_id']]['type'] != 5:
            placements[clip['channel_id']].append(clip)
    stems, warnings = [], []
    for cid, clips in placements.items():
        channel = by_id[cid]
        if len(clips) != 1:
            raise ValueError(f"{channel['name']}: consolidate to one continuous audio clip.")
        clip = clips[0]
        if clip['muted'] or not clip['track_enabled'] or not channel['enabled']:
            warnings.append(f"{channel['name']}: muted stem excluded.")
            continue
        if clip['position'] != 0:
            raise ValueError(f"{channel['name']}: render every stem from song start, including leading silence.")
        if not channel['sample']:
            raise ValueError(f"{channel['name']}: no audio file is referenced.")
        raw = project['events'][clip['event_index']][1]
        raw = raw[clip['record_offset']:clip['record_offset'] + project['layout']['record_size']]
        if any(x not in (-1., 0.) for x in struct.unpack_from('<ff', raw, 24)):
            raise ValueError(f"{channel['name']}: consolidate trimmed audio before analysis.")
        fields = dict(project['events'][channel['start']:channel['stop']])
        if int.from_bytes(fields.get(70, b'\0\0'), 'little') & 2:
            raise ValueError(f"{channel['name']}: consolidate reversed audio before analysis.")
        parameters = fields.get(215, b'')
        if any(len(parameters) >= offset + 4 and parameters[offset:offset + 4] != b'\0' * 4
               for offset in (96, 100, 104, 148)):
            raise ValueError(f"{channel['name']}: consolidate stretched, pitched or offset audio first.")
        sample = Path(os.path.expandvars(channel['sample']))
        if not sample.is_absolute():
            sample = Path(project['path']).parent / sample
        if not sample.is_file():
            matches = [p for p in (Path(project['path']).parent / sample.name,
                                  Path(project['path']).parent / 'stems' / sample.name) if p.is_file()]
            if len(matches) == 1:
                sample = matches[0]
        if not sample.is_file():
            raise ValueError(f'Audio file is missing: {sample}')
        audio = sf.info(str(sample))
        expected = clip['duration'] / project['ppq'] * 60 / project['bpm']
        if abs(audio.duration - expected) > max(.15, expected * .005):
            raise ValueError(f"{channel['name']}: WAV duration differs from its Playlist duration; consolidate first.")
        channel_name = channel['name']
        mixer_name = mixer_names.get(channel['mixer_insert'], '')
        channel_role, mixer_role = infer_role(channel_name), infer_role(mixer_name)
        display_name, role, role_source = channel_name or sample.stem, channel_role, 'channel'
        if channel_role and mixer_role and channel_role != mixer_role:
            # An explicit unknown prevents the planner from inferring the old
            # channel role again from display_name after detecting a conflict.
            role, role_source = 'unknown', 'conflict'
            warnings.append(f"{display_name}: channel role {channel_role} conflicts with mixer '{mixer_name}' role {mixer_role}; EQ role left unknown.")
        elif channel_role is None and mixer_role:
            display_name, role, role_source = mixer_name, mixer_role, 'mixer'
        elif channel_role is None:
            role_source = 'unknown'
        stems.append(dict(id=cid, name=display_name, channel_name=channel_name,
                          mixer_name=mixer_name, role=role, role_source=role_source,
                          mixer_insert=channel['mixer_insert'],
                          sample=str(sample.resolve()), position=0, offset_tick=0,
                          duration=clip['duration'], duration_seconds=audio.duration, track=clip['track']))
    if not stems:
        raise ValueError('No active audio stems found.')
    assignments = Counter(s['mixer_insert'] for s in stems)
    mixer = _initializer_rows(project['events'], 225)
    owned = set()
    for key, payload in project['events']:
        if key in (226, 227):
            owned.add(struct.unpack_from('<I', payload, 8)[0])
        elif key == 223:
            owned.update(t for _, t, _ in struct.iter_unpack('<III', payload))
    for stem in stems:
        insert = stem['mixer_insert']
        if not 0 <= insert <= 63:
            raise ValueError(f"{stem['name']}: unsupported mixer insert; use 1–63 for EQ.")
        if insert and assignments[insert] != 1:
            raise ValueError('Each independently automated stem requires its own mixer insert.')
        if insert == 0:
            warnings.append(f"{stem['name']}: analyzed as context only; Master routing is excluded from stem EQ.")
        volume_target = 0x70001fc0 | insert << 22
        if volume_target in owned:
            raise ValueError(f"{stem['name']}: render existing mixer volume automation into the stems before EQ analysis.")
        saved = [v for _, t, v in mixer if t == volume_target]
        if len(saved) != 1 or not 0 <= saved[0] <= 16000:
            raise ValueError(f"{stem['name']}: unsupported saved mixer gain.")
        # Master-fader gain affects the complete mix and must not be applied a
        # second time only to a stem routed directly there.
        value = 12800 if insert == 0 else saved[0]
        if value == 0:
            gain = 0.
        elif value / 16000 < FADER_DB[0][1]:
            raise ValueError(f"{stem['name']}: mixer fader is below the calibrated analysis range.")
        else:
            gain = 10 ** (_interpolate(value / 16000, [(v, db) for db, v in FADER_DB]) / 20)
        stem['mixer_gain'] = gain
    warnings.append('Analysis reads the WAVs with static mixer-fader weighting. Existing plugin processing and channel/clip gain are not rendered; export stems with the intended channel gain baked in.')
    return dict(path=project['path'], sha256=project['sha256'], bpm=project['bpm'], ppq=project['ppq'],
                end_tick=max(s['duration'] for s in stems), stems=sorted(stems, key=lambda s: s['track']),
                warnings=warnings)


def _gain_band(name):
    if not re.search(r'\bgain\b', name, re.I):
        return None
    match = re.search(r'\b(?:band\s*|b\s*)([3-7])\b', name, re.I)
    return int(match[1]) if match else None


def _display_number(value, *, frequency=False):
    match = re.fullmatch(r'\s*([+\-]?[0-9]+(?:\.[0-9]+)?)\s*(k?Hz)?\s*', value, re.I)
    if not match or (frequency and not match[2]):
        raise ValueError('The plugin returned an unsupported physical parameter display.')
    result = float(match[1]) * (1000 if match[2] and match[2].casefold() == 'khz' else 1)
    if not math.isfinite(result) or result <= 0:
        raise ValueError('The plugin returned an invalid physical parameter value.')
    return result


def _runtime_band(row, band):
    """Use installed-plugin readouts, never decode the opaque FFBS values."""
    index = GAIN_IDS[band]
    params = {p['index']: p for p in row['parameters']}
    gain = params[index]
    if params[index - 3]['normalized'] != 1 or params[index - 2]['normalized'] != 1:
        raise ValueError('band must be used and enabled')
    if params[index + 5]['display'] != 'Bell':
        raise ValueError('only existing Bell bands are supported')
    q = _display_number(params[index + 4]['display'])
    if abs(q - 1.) > 1e-6:
        raise ValueError('band Q differs from the verified 1.000 analysis setup')
    if abs(params[index + 1]['plain']) > 1e-6:
        raise ValueError('existing dynamic EQ range must be zero for gain-lane analysis')
    if params[index + 7]['display'] != 'Stereo' or params[index + 9]['normalized'] != 0:
        raise ValueError('band must use Stereo placement with solo disabled')
    frequency = _display_number(params[index - 1]['display'], frequency=True)
    baseline = gain['normalized']
    if abs(gain['plain'] - (baseline * 60 - 30)) > 1e-4:
        raise ValueError('saved gain readout disagrees with the verified plugin conversion')
    return dict(band=band, parameter_id=index, offset=0x8000 + index,
                baseline=baseline, baseline_db=gain['plain'], frequency_hz=frequency,
                q=q, mapping='installed-vst3-linear-gain', db_per_normalized=60.)


def read_eq_calibration(path, runtime=None):
    """Read native FL address/ownership anchors; their dB shapes are not assumed.

    Image-Line's installed Shared/Python/Lib/midi.py defines REC_PluginBase
    0x8000 and REC_Plug_Plugin_First. A real user-saved Band 7 Gain link anchors
    0x8051 = 0x8000 + installed-plugin parameter index 81 on this host.
    """
    project = read_project(path)
    runtime = probe_project(path) if runtime is None else runtime
    if runtime['source_sha256'] != project['sha256']:
        raise ValueError('The native address anchor changed during plugin inspection.')
    runtime_rows = {(r['insert'], r['slot']): r for r in runtime['rows']}
    records = plugin_records(project)
    links = defaultdict(list)
    for key, payload in project['events']:
        if key in (226, 227):
            if len(payload) != 20:
                raise ValueError('Unknown automation link layout in EQ setup.')
            row = struct.unpack('<5I', payload)
            links[row[0] >> 16].append((key, row))
    bands = {}
    mixer = _initializer_rows(project['events'], 225)
    initial = _initializer_rows(project['events'], 216)
    for channel in project['channels']:
        band = _gain_band(channel['name'])
        if band is None or channel['type'] != 5:
            continue
        if band in bands:
            raise ValueError(f'Duplicate Band {band} Gain setup clips; select one native address anchor per band.')
        rows = links[channel['id']]
        if len(rows) != 1 or rows[0][0] != 227:
            raise ValueError(f"{channel['name']}: setup clip must own exactly one native parameter link.")
        source, extra, target, kind, flags = rows[0][1]
        if source != channel['id'] << 16 or extra != 0 or kind != 8 or flags != 469:
            raise ValueError(f"{channel['name']}: use an identity link without a mapping formula.")
        if target >> 28 != 7 or target & 65535 != 0x8000 + GAIN_IDS[band]:
            raise ValueError(f"{channel['name']}: saved FL address disagrees with the plugin's actual Band {band} Gain index.")
        insert, slot = (target >> 22) & 63, (target >> 16) & 63
        record, verified = records.get((insert, slot)), runtime_rows.get((insert, slot))
        if not record or not verified or not 1 <= insert <= 63:
            raise ValueError('The native setup link must address a verified dedicated Pro-Q instance.')
        if hashlib.sha256(record['fields'][213]).hexdigest() != verified['wrapper_sha256']:
            raise ValueError('Saved Pro-Q state changed during address inspection.')
        actual = _runtime_band(verified, band)
        fields = dict(project['events'][channel['start']:channel['stop']])
        if fields.get(219, b'')[:8] != struct.pack('<II', 0, 12800):
            raise ValueError('Reset EQ Automation Clip Min/Max to the full range.')
        curve = fields.get(234, b'')
        if len(curve) < 9 or struct.unpack_from('<i', curve, 4)[0] not in (0, 64) or curve[8] != 0:
            raise ValueError('Disable EQ Automation Clip LFO before using the native setup link.')
        parse_points(curve, project['ppq'])
        placements = [c for c in project['clips'] if c.get('channel_id') == channel['id']]
        if len(placements) != 1 or placements[0]['muted'] or not placements[0]['track_enabled'] or not channel['enabled']:
            raise ValueError('Place each enabled setup clip exactly once in the Playlist.')
        native = [v for _, t, v in initial if t == target]
        legacy = [v for _, t, v in mixer if t == target]
        if len(native) > 1 or len(legacy) > 1 or native and legacy and native != legacy:
            raise ValueError('Conflicting EQ parameter initializers.')
        if native or legacy:
            _validate_parameter_baseline((native or legacy)[0], actual['baseline'])
        prefix = target & 0xffff0000
        for address, expected in ((prefix | 0x1f00, 1), (prefix | 0x1f01, 12800)):
            if [v for _, t, v in mixer if t == address] != [expected]:
                raise ValueError('Enable the setup Pro-Q slot at 100% host mix.')
        bands[band] = dict(actual, channel_id=channel['id'], target=target, insert=insert, slot=slot)
    if not bands:
        raise ValueError('Missing native address anchor: save one Automation Clip linked to a Pro-Q Band 3–7 Gain control. Its shape can stay flat.')
    return dict(path=project['path'], sha256=project['sha256'], bands=bands,
                plugin_sha256=runtime['plugin_sha256'], mapping='installed-vst3',
                address_basis='Image-Line REC_PluginBase 0x8000 plus installed plugin index; verified against saved FL gain link')


def normalized_gain(db, calibration):
    """Apply a dB DELTA around the exact saved normalized gain baseline."""
    if not math.isfinite(db) or not -3 <= db <= 3:
        raise ValueError('EQ gain delta exceeds the supported -3 to +3 dB range.')
    if calibration.get('mapping') != 'installed-vst3-linear-gain' or calibration.get('db_per_normalized') != 60.:
        raise ValueError('EQ gain conversion has not been verified by the installed plugin.')
    result = calibration['baseline'] + db / 60.
    if not 0 <= result <= 1:
        raise ValueError('EQ gain delta exceeds the plugin range around the saved baseline.')
    return result


def _bindings(project, stems, calibration, runtime):
    if runtime['source_sha256'] != project['sha256'] or runtime['plugin_sha256'] != calibration['plugin_sha256']:
        raise ValueError('Installed Pro-Q mapping does not match the source/address anchor.')
    runtime_rows = {(r['insert'], r['slot']): r for r in runtime['rows']}
    records = plugin_records(project)
    blocks, _ = _mixer(project['events'])
    mixer = _initializer_rows(project['events'], 225)
    initial = _initializer_rows(project['events'], 216)
    owned = defaultdict(list)
    for key, payload in project['events']:
        if key in (226, 227):
            row = struct.unpack('<5I', payload)
            owned[row[2]].append((key, row))
        elif key == 223:
            for row in struct.iter_unpack('<III', payload):
                owned[row[1]].append((223, row))
    same = project['path'] == calibration['path'] and project['sha256'] == calibration['sha256']
    result, skipped = {}, []
    for stem in stems:
        insert = stem['mixer_insert']
        reason = None
        matches = [(slot, record) for (route, slot), record in records.items()
                   if route == insert and record['plugin_label'].casefold() == 'fabfilter pro-q 3']
        if insert == 0:
            reason = 'Master-routed stem is analysis context only'
        elif len(matches) != 1:
            reason = 'requires exactly one existing Pro-Q 3 instance on its dedicated insert'
        elif any(len(dict(block).get(235, b'')) > insert and dict(block)[235][insert]
                 for _, _, block in blocks):
            reason = 'another mixer insert feeds this insert'
        if reason:
            skipped.append(f"{stem['name']}: {reason}.")
            continue
        slot, record = matches[0]
        verified = runtime_rows.get((insert, slot))
        if not verified or hashlib.sha256(record['fields'][213]).hexdigest() != verified['wrapper_sha256']:
            raise ValueError('A saved Pro-Q instance differs from its runtime inspection.')
        prefix = 0x70000000 | insert << 22 | slot << 16
        if any([v for _, t, v in mixer if t == prefix | offset] != [value]
               for offset, value in ((0x1f00, 1), (0x1f01, 12800))):
            skipped.append(f"{stem['name']}: Pro-Q slot must be enabled at 100% mix.")
            continue
        for band in BANDS:
            try:
                actual = _runtime_band(verified, band)
            except ValueError as exc:
                skipped.append(f"{stem['name']} Band {band}: {exc}.")
                continue
            target = prefix | actual['offset']
            anchor = calibration['bands'].get(band)
            replacement = anchor['channel_id'] if same and anchor and target == anchor['target'] else None
            allowed_owner = [(227, (replacement << 16, 0, target, 8, 469))] if replacement is not None else []
            if owned.get(target, []) != allowed_owner:
                skipped.append(f"{stem['name']} Band {band}: existing automation owns this control.")
                continue
            existing_init = [v for _, t, v in initial + mixer if t == target]
            if existing_init and replacement is None:
                skipped.append(f"{stem['name']} Band {band}: an existing initialized control is preserved.")
                continue
            if replacement is not None:
                for value in existing_init:
                    _validate_parameter_baseline(value, actual['baseline'])
            result[stem['id'], band] = dict(calibration=actual, target=target,
                                            replace_channel_id=replacement, slot=slot,
                                            baseline_db=actual['baseline_db'])
    return result, skipped


def _normalized_points(points_db, bpm, ppq, end_tick, calibrated):
    by_tick = {}
    previous = -1.
    for seconds, db in points_db:
        if not all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in (seconds, db)):
            raise ValueError('EQ points must contain finite seconds and dB.')
        if seconds < previous or seconds < 0 or seconds > end_tick / ppq * 60 / bpm + .001:
            raise ValueError('EQ point times are outside the song or out of order.')
        previous = seconds
        tick = min(end_tick, round(seconds * bpm * ppq / 60))
        by_tick[tick] = normalized_gain(db, calibrated)
    if not points_db or points_db[0][0] != 0 or points_db[0][1] != 0 or points_db[-1][1] != 0:
        raise ValueError('EQ lanes must start and finish at zero change from the saved gain baseline.')
    by_tick[0] = calibrated['baseline']
    by_tick[end_tick] = calibrated['baseline']
    return sorted(by_tick.items())


def preview_eq_automation(source, max_cut_db=1.5, progress=None, calibration_path=None):
    from eq_audio import plan_eq_automation
    if isinstance(max_cut_db, bool) or not isinstance(max_cut_db, (float, int)) or not math.isfinite(max_cut_db) or not 0 < max_cut_db <= 3:
        raise ValueError('Maximum EQ cut must be greater than 0 and at most 3 dB.')
    info = inspect_eq_stems(source)
    project = read_project(source)
    if project['sha256'] != info['sha256']:
        raise ValueError('Source changed during inspection; analyze again.')
    inputs = [dict(path=info['path'], sha256=info['sha256'])]
    for stem in info['stems']:
        if progress:
            progress(f"Reading {stem['name']}...")
        inputs.append(dict(path=stem['sample'], sha256=_file_digest(stem['sample'])))
    blocked, warnings, calibration, bindings, skipped = [], list(info['warnings']), None, {}, []
    runtime = None
    provider = dict(kind='installed-vst3', status='blocked', message='Installed Pro-Q mapping has not been verified.')
    try:
        if progress:
            progress('Reading the installed Pro-Q controller and saved band settings...')
        runtime = probe_project(source)
        inputs.append(dict(path=runtime['plugin_path'], sha256=runtime['plugin_sha256']))
        provider.update(status='verified', message='Installed Pro-Q 3 verifies saved band settings and the gain conversion.',
                        plugin_path=runtime['plugin_path'], plugin_sha256=runtime['plugin_sha256'])
        for stem in info['stems']:
            matches = [r for r in runtime['rows'] if r['insert'] == stem['mixer_insert']]
            if stem['mixer_insert'] and len(matches) == 1:
                params = {p['index']: p for p in matches[0]['parameters']}
                stem['centers_hz'] = [_display_number(params[GAIN_IDS[b] - 1]['display'], frequency=True) for b in BANDS]
                stem['centers_source'] = 'installed-plugin-saved-state'
            else:
                stem['centers_hz'] = list(CENTERS_HZ)
                stem['centers_source'] = 'analysis-context-reference'
        anchor_path = Path(calibration_path or source).expanduser().resolve()
        anchor_runtime = runtime if str(anchor_path) == info['path'] else probe_project(anchor_path)
        calibration = read_eq_calibration(anchor_path, anchor_runtime)
        inputs.append(dict(path=calibration['path'], sha256=calibration['sha256']))
        bindings, skipped = _bindings(project, info['stems'], calibration, runtime)
        warnings.append('Gain curves are cuts relative to each saved gain value, using the installed plugin conversion. Native FL playback and audible results still need review.')
    except (ValueError, OSError) as exc:
        blocked.append(str(exc))
    if runtime is None:
        warnings.append('Installed Pro-Q settings could not be verified. Review assumes reference centers of 76, 254, 806, 2460 and 7519 Hz; export is disabled.')
        provider['message'] = blocked[-1]
    plan = plan_eq_automation(info['stems'], bpm=info['bpm'], ppq=info['ppq'], end_tick=info['end_tick'],
                              centers_hz=CENTERS_HZ, max_cut_db=max_cut_db, progress=progress)
    warnings.extend(plan.get('warnings', []))
    by_id = {s['id']: s for s in info['stems']}
    lanes = []
    for analyzed in plan['stems']:
        stem = by_id[analyzed['id']]
        for band in analyzed['bands']:
            actual = BANDS[band['band_index']]
            points_db = [(p['seconds'], p['db']) for p in band['points']]
            if any(not -max_cut_db - 1e-8 <= db <= 1e-8 for _, db in points_db):
                raise ValueError('The audio planner exceeded the requested cut-only range.')
            lane = dict(name=f"{stem['name']} EQ B{actual} Gain", stem_name=stem['name'], stem_id=stem['id'],
                        band=actual, frequency_hz=band['center_hz'], max_cut_db=max_cut_db,
                        mixer_insert=stem['mixer_insert'], points_db=points_db, points=[],
                        active=bool(band.get('active')), reason=band.get('reason', ''), evidence=band.get('evidence', {}),
                        exportable=False)
            binding = bindings.get((stem['id'], actual))
            if binding:
                lane.update(binding)
                if lane['active'] or binding['replace_channel_id'] is not None:
                    lane['points'] = _normalized_points(points_db, info['bpm'], info['ppq'], info['end_tick'], binding['calibration'])
                    lane['exportable'] = True
            lanes.append(lane)
    if calibration and not any(lane['exportable'] for lane in lanes):
        blocked.append('No matching, unowned Pro-Q gain controls can be exported from this project.')
    for item in inputs:
        if _file_digest(item['path']) != item['sha256']:
            raise ValueError('An input changed during analysis; analyze again.')
    return dict(format='audio-eq-preview', version=1, source=info['path'], source_sha256=info['sha256'],
                ppq=info['ppq'], bpm=info['bpm'], end_tick=info['end_tick'], stems=info['stems'],
                lanes=lanes, blocked=blocked, warnings=warnings, skipped=skipped, input_files=inputs,
                max_cut_db=max_cut_db, calibration=calibration, provider=provider,
                output=str(Path(source).with_name(Path(source).stem + '_EQ_automation.flp')),
                export_ready=bool(calibration and not blocked and any(x['exportable'] for x in lanes)),
                analysis_assumptions=dict(reference_centers_hz=list(CENTERS_HZ), physical_centers_verified=runtime is not None,
                                          gain_semantics='delta_from_saved_gain',
                                          gain_mapping='installed-plugin normalized/plain conversion' if runtime else 'unverified'))


def export_eq_automation(preview, output=None):
    """Write a new project only; replace designated setup clips or append gains."""
    output = Path(output or preview['output']).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    if not preview.get('export_ready') or preview.get('blocked'):
        raise ValueError('EQ export requires a valid calibration and exportable gain lanes.')
    if output.suffix.casefold() != '.flp':
        raise ValueError('Choose a new .flp output file.')
    for item in preview['input_files']:
        if _file_digest(item['path']) != item['sha256']:
            raise ValueError('An input changed after preview; analyze again.')
    project = read_project(preview['source'])
    if output.parent != Path(project['path']).parent and any(c['sample'] and not Path(os.path.expandvars(c['sample'])).is_absolute() for c in project['channels']):
        raise ValueError('This project has relative audio paths. Save its EQ copy beside the source FLP.')
    info = inspect_eq_stems(preview['source'])
    if project['sha256'] != preview['source_sha256'] or any(preview[k] != info[k] for k in ('ppq', 'bpm', 'end_tick')):
        raise ValueError('Source or timing changed after preview.')
    runtime = probe_project(preview['source'])
    if preview.get('provider', {}).get('plugin_sha256') != runtime['plugin_sha256']:
        raise ValueError('Installed Pro-Q provider changed after preview.')
    anchor_path = preview['calibration']['path']
    anchor_runtime = runtime if anchor_path == project['path'] else probe_project(anchor_path)
    calibrated = read_eq_calibration(anchor_path, anchor_runtime)
    if calibrated != preview['calibration']:
        raise ValueError('EQ calibration changed after preview.')
    bindings, _ = _bindings(project, info['stems'], calibrated, runtime)
    lanes = [lane for lane in preview['lanes'] if lane.get('exportable')]
    if not lanes or len({(lane['stem_id'], lane['band']) for lane in lanes}) != len(lanes):
        raise ValueError('EQ preview has missing or duplicate gain lanes.')
    for lane in lanes:
        expected = bindings.get((lane['stem_id'], lane['band']))
        if expected is None or any(lane.get(key) != value for key, value in expected.items()):
            raise ValueError('An EQ destination or calibration changed after preview.')
        if any(not -preview['max_cut_db'] - 1e-8 <= db <= 0 for _, db in lane['points_db']):
            raise ValueError('EQ preview exceeds its cut-only limit.')
        normalized = _normalized_points(lane['points_db'], preview['bpm'], preview['ppq'], preview['end_tick'], expected['calibration'])
        if normalized != [tuple(p) for p in lane['points']]:
            raise ValueError('EQ normalized points no longer match their calibrated dB values.')
    events = project['events']
    replacements, before = {}, defaultdict(list)
    new_channels, new_links, initial_additions, generated = [], [], [], []
    channel_by_id = {c['id']: c for c in project['channels']}
    next_id = max(channel_by_id) + 1
    next_track = max(c['track'] for c in project['clips'])
    serial = project['layout']['next_clip_serial']
    size = project['layout']['record_size']
    playlist_at = project['layout']['playlist_index']
    playlist = events[playlist_at][1]
    replaced_cids = {lane['replace_channel_id'] for lane in lanes if lane['replace_channel_id'] is not None}
    kept = [playlist[i:i + size] for i in range(0, len(playlist), size)
            if struct.unpack_from('<H', playlist, i + 6)[0] not in replaced_cids]
    initialized = {t: v for _, t, v in _initializer_rows(events, 216) + _initializer_rows(events, 225)}
    track_rows = set()
    for lane in lanes:
        cid = lane['replace_channel_id']
        if cid is not None:
            channel = channel_by_id[cid]
            at = [i for i in range(channel['start'], channel['stop']) if events[i][0] == 234]
            if len(at) != 1:
                raise ValueError('Calibration channel has an unsupported curve layout.')
            replacements[at[0]] = (234, build_curve(lane['points'], preview['ppq']))
            placement = next(c for c in project['clips'] if c.get('channel_id') == cid)
            track = placement['track'] - 1
        else:
            cid, next_id = next_id, next_id + 1
            track, next_track = next_track, next_track + 1
            if cid > 4094 or track > 499:
                raise ValueError('No room for additional EQ channels or Playlist tracks.')
            new_channels.extend(build_channel(cid, lane['name'], lane['points'], preview['ppq']))
            new_links.append(build_link(cid, lane['target']))
        track_rows.add(track + 1)
        kept.append(build_playlist_clip(cid, 0, preview['end_tick'], track, serial, size,
                                        offset_tick=0, ppq=preview['ppq']))
        serial += 1
        if lane['target'] not in initialized:
            value = struct.unpack('<I', struct.pack('<f', lane['calibration']['baseline']))[0]
            initial_additions.append(struct.pack('<III', 0, lane['target'], value))
        generated.append(dict(name=lane['name'], channel_id=cid, target=lane['target'], points=lane['points']))
    replacements[playlist_at] = (233, b''.join(kept))
    for i, (key, payload) in enumerate(events):
        if key == 238 and len(payload) >= 13 and struct.unpack_from('<I', payload)[0] in track_rows and not payload[12]:
            data = bytearray(payload)
            data[12] = 1
            replacements[i] = (238, bytes(data))
    first = project['layout']['first_channel']
    global_at = next((i for i, (k, _) in enumerate(events[:first]) if k in (226, 227)), first)
    existing_init = [i for i, (key, _) in enumerate(events) if key == 216]
    if len(existing_init) > 1 or existing_init and existing_init[0] >= first:
        raise ValueError('Unsupported project initializer layout.')
    if initial_additions:
        if existing_init:
            at = existing_init[0]
            replacements[at] = (216, events[at][1] + b''.join(initial_additions))
        else:
            before[global_at].append((216, b''.join(initial_additions)))
    before[first].extend(new_links)
    before[project['layout']['arrangement']].extend(new_channels)
    result, source_positions = [], []
    for index, event in enumerate(events):
        result.extend(before[index])
        source_positions.append(len(result))
        result.append(replacements.get(index, event))
    header = bytearray(project['header'])
    new_count = sum(k == 64 for k, _ in new_channels)
    struct.pack_into('<H', header, 2, len(project['channels']) + new_count)
    binary = encode_fl(bytes(header), result)
    checked_header, checked_events = read_bytes(binary)
    if checked_header != bytes(header) or checked_events != result:
        raise ValueError('EQ output failed native-container roundtrip validation.')
    if any(result[position] != events[i] for i, position in enumerate(source_positions) if i not in replacements):
        raise ValueError('An unrelated project event changed unexpectedly.')
    if any(events[i][0] not in (216, 233, 234, 238) for i in replacements):
        raise ValueError('EQ output changed an unsupported source field.')
    for key in (213, 225, 235, 236, 223, 224):
        if [b for k, b in events if k == key] != [b for k, b in result if k == key]:
            raise ValueError('EQ output changed saved audio, plugin or mixer state.')
    checked = read_project(project['path'], data_override=binary)
    if {key: row['events'] for key, row in plugin_records(project).items()} != {key: row['events'] for key, row in plugin_records(checked).items()}:
        raise ValueError('EQ output changed a saved plugin record.')
    for row in generated:
        channel = next(c for c in checked['channels'] if c['id'] == row['channel_id'])
        fields = dict(checked['events'][channel['start']:channel['stop']])
        parsed = parse_points(fields[234], preview['ppq'])
        if len(parsed) != len(row['points']) or any(abs(a - t) > 1e-6 or b != v for (a, b), (t, v) in zip(parsed, row['points'])):
            raise ValueError('EQ output curves failed read-back validation.')
    for item in preview['input_files']:
        if _file_digest(item['path']) != item['sha256']:
            raise ValueError('An input changed during export; analyze again.')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.eq-export-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(binary)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return dict(output=str(output), source=project['path'], output_sha256=hashlib.sha256(binary).hexdigest(),
                automation_channels=len(lanes), new_channels=new_count, replaced_setup_channels=len(replaced_cids),
                warnings=preview['warnings'], skipped=preview['skipped'],
                validation=dict(roundtrip_exact=True, unrelated_events_byte_identical=True,
                                plugin_and_mixer_state_byte_identical=True, inputs_unchanged=True,
                                native_fl_playback_tested=False))
