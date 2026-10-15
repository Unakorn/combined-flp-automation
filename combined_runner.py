"""One local workflow: prepare EQ, add section effects, then add stem volume.

All analysis uses the original rendered WAVs. This is not a mix renderer, and
the EQ recommendations do not simulate the subsequently automated mix.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
import uuid

APP = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(APP / 'work'))

from flp_raw import encode_fl, read_bytes
from flp_song_analysis import read_project
from plugin_templates import DEFAULT_TEMPLATE, plugin_records
from stem_project import inspect_stems, analyze_stems, preview_automation, export_automation
from stem_planner import load_presets
from stem_native import parse_points
from section_inputs import analyze_section_inputs
from section_motion import validate_settings
from section_project import preview_section_automation, export_section_automation
from eq_project import preview_eq_automation, export_eq_automation


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def publish_new(data, path):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.combined-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def stage_source(source, destination):
    """Copy source, resolving only relative sample references before relocation."""
    project = read_project(source)
    resolved_stems = {s['id']: Path(s['sample']) for s in inspect_stems(source)['stems'] if not s['missing']}
    events = list(project['events'])
    replacements = []
    for channel in project['channels']:
        sample = channel.get('sample')
        if not sample:
            continue
        original = Path(os.path.expandvars(sample))
        resolved = resolved_stems.get(channel['id'])
        if resolved is None:
            resolved = original if original.is_absolute() else (Path(source).parent / original).resolve()
        if original.is_absolute() and original == resolved:
            continue
        if not resolved.is_file():
            raise ValueError(f'Cannot relocate an unresolved sample reference: {sample}')
        positions = [i for i in range(channel['start'], channel['stop']) if events[i][0] == 196]
        if len(positions) != 1:
            raise ValueError('Ambiguous sample path event.')
        events[positions[0]] = (196, (str(resolved) + '\0').encode('utf-16-le'))
        replacements.append(dict(channel_id=channel['id'], original=sample, resolved=str(resolved)))
    data = encode_fl(project['header'], events) if replacements else project['data']
    publish_new(data, destination)
    return replacements


def validate_final(original_path, staged_path, final_path):
    """Validate native links and preservation independently of stage exporters."""
    original = read_project(original_path)
    before = read_project(staged_path)
    after = read_project(final_path)
    if read_bytes(after['data']) != (after['header'], after['events']):
        raise ValueError('Combined FLP raw round-trip failed.')
    for key in ('bpm', 'ppq', 'time_signature', 'patterns'):
        if before[key] != after[key]:
            raise ValueError(f'Combined export changed original {key}.')
    channels = {c['id']: c for c in after['channels']}
    for old in before['channels']:
        new = channels.get(old['id'])
        if new is None:
            raise ValueError('An original channel disappeared.')
        for key in ('name', 'type', 'mixer_insert', 'sample', 'enabled'):
            if old[key] != new[key]:
                raise ValueError(f'An original channel changed: {old["name"]}, {key}.')
        old_events = before['events'][old['start']:old['stop']]
        new_events = after['events'][new['start']:new['stop']]
        if old_events != new_events:
            raise ValueError(f'Original channel state changed: {old["name"]}.')
    old_playlist = before['events'][before['layout']['playlist_index']][1]
    new_playlist = after['events'][after['layout']['playlist_index']][1]
    if not new_playlist.startswith(old_playlist):
        raise ValueError('Original Playlist audio placements changed.')
    old_plugins, new_plugins = plugin_records(before), plugin_records(after)
    for position, record in old_plugins.items():
        if position not in new_plugins or record['fields'] != new_plugins[position]['fields']:
            raise ValueError(f'Original plugin state changed on insert/slot {position}.')
    prior_links = [(k, p) for k, p in before['events'] if k in (226, 227)]
    all_links = [(k, p) for k, p in after['events'] if k in (226, 227)]
    if Counter(prior_links) - Counter(all_links):
        raise ValueError('An original controller link was changed.')
    original_ids = {c['id'] for c in original['channels']}
    owners, counts, lane_details = {}, Counter(), []
    for key, payload in all_links:
        if len(payload) != 20:
            raise ValueError('Unknown automation link layout in combined output.')
        src, extra, target, kind, flags = struct.unpack('<5I', payload)
        cid = src >> 16
        counts[target] += 1
        owners.setdefault(cid, []).append(target)
    for key, payload in after['events']:
        if key == 223:
            if len(payload) % 12:
                raise ValueError('Unknown event automation layout.')
            for _, target, _ in struct.iter_unpack('<III', payload):
                counts[target] += 1
    for channel in after['channels']:
        if channel['id'] in original_ids:
            continue
        if channel['type'] != 5:
            raise ValueError('Combined output added an unexpected musical channel.')
        targets = owners.get(channel['id'], [])
        if not targets or any(counts[target] != 1 for target in targets):
            raise ValueError('New automation has a missing or competing parameter owner.')
        fields = dict(after['events'][channel['start']:channel['stop']])
        points = parse_points(fields[234], after['ppq'])
        if any(not math.isfinite(v) or not 0 <= v <= 1 for _, v in points):
            raise ValueError('Automation has invalid normalized values.')
        placements = [c for c in after['clips'] if c.get('channel_id') == channel['id']]
        if (len(placements) != 1 or placements[0]['position'] != 0 or placements[0]['muted']
                or not placements[0]['track_enabled'] or not channel['enabled']):
            raise ValueError('Generated automation is not a single active full-song clip.')
        if points[0][0] != 0 or abs(points[-1][0] - placements[0]['duration']) > 1:
            raise ValueError('Generated automation does not span its Playlist placement.')
        lane_details.append(dict(name=channel['name'], channel_id=channel['id'], targets=targets, points=len(points)))
    old_initial = {}
    for key, payload in before['events']:
        if key in (216, 225):
            for prefix, target, value in struct.iter_unpack('<III', payload):
                old_initial.setdefault((key, target), []).append((prefix, value))
    new_initial = {}
    for key, payload in after['events']:
        if key in (216, 225):
            for prefix, target, value in struct.iter_unpack('<III', payload):
                new_initial.setdefault((key, target), []).append((prefix, value))
    new_targets = {t for lane in lane_details for t in lane['targets']}
    if any(len(new_initial.get((216, t), [])) > 1 for t in new_targets):
        raise ValueError('Duplicate native parameter initialization in combined output.')
    added_slots = set(new_plugins) - set(old_plugins)
    allowed_slot_values = {0x70000000 | insert << 22 | slot << 16 | offset
                          for insert, slot in added_slots for offset in (0x1f00, 0x1f01)}
    for (kind, target), values in old_initial.items():
        if kind == 225 and target in allowed_slot_values:
            continue
        if values != new_initial.get((kind, target), []):
            raise ValueError('An original saved mixer or initialized control value changed.')
    return dict(raw_roundtrip=True, original_channels_preserved=True, original_playlist_preserved=True,
                original_plugins_preserved=True, original_notes_preserved=True,
                unique_new_parameter_owners=True, valid_full_song_clips=True,
                native_fl_playback_tested=False, automation_channels=len(lane_details), lanes=lane_details)


def run_combined(config, progress=None):
    progress = progress or (lambda message: None)
    source = Path(config['source']).expanduser().resolve()
    output = Path(config['output']).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != '.flp':
        raise ValueError('Choose an existing stem FLP.')
    if output.suffix.lower() != '.flp' or output == source or output.exists():
        raise ValueError('Choose a new .flp output filename; existing files are preserved.')
    report_path = output.with_name(output.stem + ' - automation report.json')
    if report_path.exists():
        raise FileExistsError(f'The report already exists: {report_path}')
    modules = dict(volume=True, section=False, eq=False)
    modules.update(config.get('modules', {}))
    if set(modules) != {'volume', 'section', 'eq'} or any(type(v) is not bool for v in modules.values()) or not any(modules.values()):
        raise ValueError('Select at least one of volume, section, and EQ.')
    midi = str(Path(config['midi']).expanduser().resolve()) if config.get('midi') else ''
    original = str(Path(config['original']).expanduser().resolve()) if config.get('original') else ''
    if modules['section'] and (not midi or not Path(midi).is_file()):
        raise ValueError('Section effects require a matching full-song MIDI export.')
    root = Path(config.get('work_dir') or Path(tempfile.gettempdir()) / 'OpenAutomationCombined').resolve()
    root.mkdir(parents=True, exist_ok=True)
    work = root / ('run-' + uuid.uuid4().hex)
    work.mkdir()
    cache = work.parent / 'audio-cache'
    originals = {str(source): digest(source)}
    for path in (midi, original):
        if path:
            originals[path] = digest(path)
    initial_info = inspect_stems(source)
    for stem in initial_info['stems']:
        if stem['missing']:
            raise ValueError(f'Missing audio: {stem["sample"]}')
        originals[stem['sample']] = digest(stem['sample'])
    if modules['section']:
        from combined_inputs import build_input_config
        progress('Checking stem names and full-song MIDI timing…')
        inputs = build_input_config(str(source), midi, original)
    else:
        from stem_audio import ROLES
        roles = {s['id']: s['role'] for s in initial_info['stems']}
        if modules['volume'] and any(role not in ROLES for role in roles.values()):
            raise ValueError('Give each stem a supported instrument/drum name before volume balancing.')
        inputs = dict(roles=roles, midi_mapping={}, drum_midi_guide=None,
                      protected_stems=[], verification={'section_matching_required': False})
    staged = work / '00_source.flp'
    relocated = stage_source(source, staged)
    current = staged
    stages, warnings = [], []
    report = dict(version=1, source=str(source), source_sha256=originals[str(source)], output=str(output),
                  modules=modules, input_matching=inputs.get('verification', {}), relative_paths_resolved=relocated,
                  ownership={'volume': 'Stem volume balance', 'effects_and_pan': 'Section Automation', 'EQ': 'Pro-Q 3 band gain'},
                  analysis_assumptions=['EQ uses original rendered audio and saved static faders; later automated processing is not rendered into its analysis.',
                                        'Shared drum MIDI guides sections, not individual kick hits.'], stages=stages)
    if modules['eq']:
        calibration = config.get('eq_calibration')
        if not calibration or not Path(calibration).is_file():
            raise ValueError('Choose the verified Pro-Q native-link calibration FLP.')
        originals[str(Path(calibration).resolve())] = digest(calibration)
        records = plugin_records(read_project(current))
        counts = [sum(insert == s['mixer_insert'] and r['plugin_label'].casefold() == 'fabfilter pro-q 3'
                      for (insert, _), r in records.items()) for s in initial_info['stems']]
        if all(count == 0 for count in counts):
            progress('Preparing a verified flat Pro-Q 3 setup…')
            from combined_eq_setup import prepare_eq_copy
            prepared = work / '01_eq_setup.flp'
            setup = prepare_eq_copy(str(current), str(prepared), str(calibration))
            stages.append(dict(stage='EQ setup', result=setup))
        elif all(count == 1 for count in counts):
            prepared = current
            stages.append(dict(stage='EQ setup', existing_instances_preserved=True))
        else:
            raise ValueError('EQ requires either no Pro-Q instances (automatic flat setup), or exactly one existing Pro-Q 3 on every stem insert. Mixed/duplicate setups need adjustment first.')
        progress('Analyzing audio for Pro-Q band-gain automation…')
        eq = preview_eq_automation(prepared, max_cut_db=float(config.get('max_cut_db', 1.5)),
                                  calibration_path=calibration, progress=progress)
        if eq['blocked']:
            raise ValueError('EQ setup needs attention: ' + '; '.join(eq['blocked']))
        result = export_eq_automation(eq, output=work / '02_eq.flp')
        current = Path(result['output'])
        stages.append(dict(stage='EQ', channels=result['automation_channels'],
                           max_cut_db=eq['max_cut_db'], skipped=eq['skipped'], validation=result['validation']))
        warnings.extend(eq['warnings'])
    if modules['section']:
        progress('Analyzing MIDI sections and stem audio…')
        analysis = analyze_section_inputs(inspect_stems(current), roles=inputs['roles'], midi_path=midi,
                                         midi_mapping=inputs['midi_mapping'], drum_midi_guide=inputs['drum_midi_guide'],
                                         cache_dir=cache, progress=progress)
        if analysis['match_errors']:
            raise ValueError('MIDI matching needs attention: ' + '; '.join(analysis['match_errors']))
        protected = set(inputs.get('protected_stems', []))
        for stem in analysis['stems']:
            if stem['id'] in protected:
                stem['automation_protected'] = True
        enabled = dict(filter=False, delay=False, reverb=False, width=False, pan=True)
        enabled.update(config.get('section_enabled', {}))
        enabled['volume'] = False
        settings = validate_settings(dict(seed=int(config.get('seed', 150)),
                         enabled=enabled, reverb_mode=config.get('reverb_mode', 'baked'),
                         intensities=config.get('section_intensities', {})))
        section = preview_section_automation(analysis, settings, config.get('section_template') or DEFAULT_TEMPLATE)
        if section['blocked']:
            raise ValueError('Section export needs attention: ' + '; '.join(section['blocked']))
        progress('Adding section effects and pan; keeping volume with the stem engine…')
        result = export_section_automation(section, output=work / '03_sections.flp')
        current = Path(result['output'])
        stages.append(dict(stage='Section', channels=len(section['lanes']), settings=section['settings'],
                           effective_intensities=section['effective_intensities'],
                           effects=[{k:e.get(k) for k in ('family','insert','slot','stem_name')} for e in section['effects']],
                           sections=section['sections'], skipped=section['skipped']))
        warnings.extend(section['warnings'])
    if modules['volume']:
        progress('Planning stem volume balance, builds, and fades…')
        analysis = analyze_stems(inspect_stems(current), roles=inputs['roles'], cache_dir=cache, progress=progress)
        presets = load_presets()
        genre = next(iter(presets))
        subgenre = next(iter(presets[genre]))
        volume = preview_automation(analysis, genre, subgenre,
                     intensity_override=config.get('volume_intensity', 'Medium'), movement_mode='volume_balance')
        if not volume['lanes']:
            raise ValueError('No stem volume curves are available; review the saved faders and existing automation.')
        result = export_automation(volume, output=work / '04_volume.flp')
        current = Path(result['output'])
        stages.append(dict(stage='Volume', channels=result['automation_channels'],
                           intensity=config.get('volume_intensity','Medium'), skipped=volume.get('skipped',[]),
                           coverage=volume.get('coverage'), validation=result['validation']))
        warnings.extend(volume['warnings'])
    progress('Validating combined automation links and original project preservation…')
    validation = validate_final(source, staged, current)
    for path, expected in originals.items():
        if digest(path) != expected:
            raise ValueError(f'An input changed during generation; run again: {path}')
    report.update(validation=validation, input_files=[dict(path=p,sha256=h) for p,h in originals.items()],
                  warnings=list(dict.fromkeys(warnings)), original_inputs_unchanged=True,
                  output_sha256=digest(current), sound_quality_reviewed=False,
                  summary=f'{validation["automation_channels"]} editable automation clips; saved for listening review.')
    report_data = json.dumps(report, indent=2, ensure_ascii=False, default=str).encode('utf-8')
    publish_new(current.read_bytes(), output)
    publish_new(report_data, report_path)
    progress(report['summary'])
    return dict(output=str(output), report_path=str(report_path), summary=report['summary'], report=report)
