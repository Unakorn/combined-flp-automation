"""Preview and export sparse section effects, preserving the saved stem FLP."""
from __future__ import annotations
from collections import defaultdict, Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile

from drum_template import _mixer
from flp_raw import read_bytes, encode_fl
from flp_song_analysis import read_project
from stem_project import _targets, _free_effect_slot, _file_digest, relative_volume, _signed, inspect_stems
from stem_native import build_channel, build_link, build_playlist_clip, parse_points
from plugin_templates import (DEFAULT_TEMPLATE, FAMILIES, load_template, spec_digest,
                              build_plugin_record, normalized_depth, plugin_records)
from section_motion import plan_section_motion, validate_settings


_NATIVE = {
    'volume': dict(label='Volume', parameter='volume_db', offset=0x1fc0),
    'pan': dict(label='Light pan', parameter='pan_percent', offset=0x1fc1),
}


def _is_native_target(target):
    return target & 0xf0000000 == 0x70000000 and target & 0x003fffff in (0x1fc0, 0x1fc1)


def _native_control(curve, stem, blocks, values, counts, owned, project_init_rows, end):
    """Resolve saved mixer controls without assuming a missing fader baseline.

    A saved InitSong value is an independent owner even when it agrees with
    the current mixer value. Leave that existing automation choice untouched.
    """
    family=curve['family']; spec=_NATIVE[family]; insert=curve['mixer_insert']
    if curve['stem_id']!=stem['id'] or insert!=stem['mixer_insert']:
        raise ValueError('Stem mixer assignment differs from the planned control.')
    if stem.get('automation_protected') is True:
        raise ValueError('This stem is protected from automation.')
    role=stem.get('role')
    if role in {'kick','bass','sub','snare','clap','crash','tom'}:
        raise ValueError('This anchor stem keeps its saved volume and pan.')
    if family=='pan' and (role not in {'hat','perc','ride','drums','pads','chords','keys',
                                      'guitar','brass','synth','effects'}
                         or {'lead','main','solo'} & set(re.findall(r'[a-z0-9]+',stem['name'].lower()))):
        raise ValueError('This foreground stem keeps its saved pan position.')
    if not 1<=insert<len(blocks)-1:
        raise ValueError('Mixer insert is unavailable.')
    if any(len(dict(block).get(235,b''))>insert and dict(block)[235][insert]
           for _,_,block in blocks):
        raise ValueError('An incoming mixer connection needs checking before a mixer ride.')
    target=0x70000000|insert<<22|spec['offset']
    if target in owned:
        raise ValueError('This mixer control already has automation or a controller link.')
    if any(address==target for _,address,_ in project_init_rows):
        raise ValueError('This mixer control has an existing InitSong value; its automation is preserved.')
    if counts[target]!=1:
        raise ValueError('The saved mixer control value is missing or duplicated.')
    saved=values[target]
    if family=='volume':
        if not 0<=saved<=16000:
            raise ValueError('Unsupported saved fader value.')
        baseline=saved/16000
    else:
        pan=_signed(saved)
        if not -6400<=pan<=6400:
            raise ValueError('Unsupported saved pan position.')
        baseline=(pan+6400)/12800
    if curve.get('parameter',spec['parameter'])!=spec['parameter']:
        raise ValueError('Native control parameter differs from its family.')
    raw=curve.get('points')
    if (not isinstance(raw,(list,tuple)) or not 2<=len(raw)<=65535
            or any(not isinstance(point,(list,tuple)) or len(point)!=2 for point in raw)):
        raise ValueError('Native curves need two or more tick/value points.')
    minimum=-1.5 if family=='volume' and role in {'vocals','hat','perc','ride','drums'} else -3 if family=='volume' else -8
    maximum=0 if family=='volume' else 8
    if any(type(tick) is not int or not 0<=tick<=end or isinstance(value,bool)
           or not isinstance(value,(int,float)) or not math.isfinite(value)
           or not minimum<=value<=maximum for tick,value in raw):
        raise ValueError('Native curve points exceed the supported time or control range.')
    if list(raw[0])!=[0,0] or list(raw[-1])!=[end,0]:
        raise ValueError('Native curves must start and finish at the saved mixer baseline.')
    if any(a[0]>=b[0] for a,b in zip(raw,raw[1:])):
        raise ValueError('Native curve point times must increase strictly.')
    if family=='volume':
        # Measured fader interpolation may round close to an anchor. Zero delta
        # is exact and a ride must never boost beyond the saved fader value.
        points=[(tick,min(baseline,relative_volume(saved,value))) for tick,value in raw]
    else:
        points=[(tick,max(0.,min(1.,baseline+value/200))) for tick,value in raw]
    detail=dict(stem_id=stem['id'],stem_name=stem['name'],mixer_insert=insert,
                family=family,parameter=spec['parameter'],plugin='Mixer',effect_slot=None,
                target=target,baseline=baseline,saved_raw=saved,role=stem.get('role'),
                automation_protected=stem.get('automation_protected',False))
    return _compact(points),detail


def _compact(points):
    result=[]
    for tick,value in points:
        if result and tick <= result[-1][0]:
            raise ValueError('Automation point times must increase strictly.')
        while len(result)>=2:
            (a,x),(b,y)=result[-2:]
            if abs((y-x)*(tick-b)-(value-y)*(b-a)) > 1e-9:
                break
            result.pop()
        result.append((tick,value))
    return result


def _project_initializers(project):
    """Read FL's global initial values separately from mixer-state event 225.

    Native plugin knob initial values are raw float32 bits in event 216. The
    row prefix and order are opaque here and must survive a section export.
    """
    entries=[(i,p) for i,(k,p) in enumerate(project['events']) if k==216]
    if len(entries)>1:
        raise ValueError('Duplicate project initializer event 216.')
    if not entries:
        return None, []
    index,payload=entries[0]
    if index>=project['layout']['first_channel'] or len(payload)%12:
        raise ValueError('Unsupported project initializer event 216.')
    rows=list(struct.iter_unpack('<III',payload))
    targets=Counter(target for _,target,_ in rows)
    if any(count>1 and not _is_native_target(target) for target,count in targets.items()):
        raise ValueError('Duplicate project initializer target in event 216.')
    return index,rows


def _project_initializer_position(project):
    # FL 26 saves event 216 after global settings/group selection (146), before
    # remote/automation links and channels. Minimal projects may omit 146 and
    # all links; in that case the end of the global header is the same boundary.
    first=project['layout']['first_channel']
    return next((i for i,(k,_) in enumerate(project['events'][:first])
                 if k in (226,227)),first)


def preview_section_automation(analysis, settings, template_path=None):
    settings=validate_settings(settings)
    project=read_project(analysis['path'])
    if project['sha256']!=analysis['sha256']:
        raise ValueError('Stem FLP changed; load and analyze it again.')
    plan=plan_section_motion(analysis,settings)
    template_path=Path(template_path or DEFAULT_TEMPLATE)
    uses_plugins=any(curve['family'] not in _NATIVE for curve in plan['curves'])
    template=load_template(template_path) if uses_plugins else {'families':{}}
    blocks,values,counts,owned,saved_effects=_targets(project)
    _,project_init_rows=_project_initializers(project)
    reserved_targets=owned|{target for _,target,_ in project_init_rows}
    warnings=list(analysis.get('warnings',[]))+list(plan.get('warnings',[]))
    skipped=list(plan.get('skipped',[])); blocked=[]; lanes=[]; additions=[]
    reserved=defaultdict(set)
    role_family={'bass':'Bass','vocals':'Vocals','effects':'Effects'}
    drums={'kick','snare','clap','tom','hat','perc','ride','crash'}
    stems={s['id']:s for s in analysis['stems']}
    for curve in plan['curves']:
        family=curve['family']; insert=curve['mixer_insert']; stem=stems[curve['stem_id']]
        display_family='Drums' if stem['role'] in drums else role_family.get(stem['role'],'Melodic')
        if family in _NATIVE:
            try:
                points,detail=_native_control(curve,stem,blocks,values,counts,owned,
                                              project_init_rows,analysis['end_tick'])
                if all(abs(value-detail['baseline'])<1e-12 for _,value in points):
                    raise ValueError('The saved control is at its limit; this ride would be flat.')
            except ValueError as exc:
                reason=f'{_NATIVE[family]["label"]}: {exc}'
                skipped.append(dict(stem_id=stem['id'],stem_name=stem['name'],reason=reason))
                warnings.append(f'{stem["name"]}: {reason}')
                continue
            lanes.append(dict(name=f'{stem["name"]} - {_NATIVE[family]["label"]}',points=points,
                              targets=[detail],family=display_family,control_family=family,
                              parameter=_NATIVE[family]['parameter'],
                              reason=curve.get('reason','Section change'),moves=[curve]))
            continue
        spec=template['families'].get(family)
        if spec is None:
            message=f'{FAMILIES[family]["label"]}: one-time FL control template is missing.'
            if message not in blocked:blocked.append(message)
            continue
        reason=None
        if not 1<=insert<len(blocks)-1:
            reason='Mixer insert is unavailable.'
        # Connection flags include sidechain-only sends, so do not reinterpret
        # incoming buses as a clean independent stem without inspecting levels.
        elif any(len(dict(block).get(235,b''))>insert and dict(block)[235][insert]
                 for _,_,block in blocks):
            reason='An incoming mixer connection needs checking before adding a stem effect.'
        if reason:
            blocked.append(f'{stem["name"]}: {reason}');continue
        try:
            slot=_free_effect_slot(blocks,saved_effects,insert,reserved[insert],reserved_targets,counts)
        except ValueError as exc:
            blocked.append(f'{stem["name"]}: {exc}.');continue
        target=0x70000000|insert<<22|slot<<16|spec['target_offset']
        if target in owned or counts[target]>1:
            blocked.append(f'{stem["name"]}: target already has automation.');continue
        points=_compact([(tick,normalized_depth(spec,depth)) for tick,depth in curve['points']])
        if len(points)<2 or points[0][0]!=0 or points[-1][0]!=analysis['end_tick']:
            raise ValueError('A planned curve must cover the complete song.')
        if all(abs(value-spec['baseline'])<1e-12 for _,value in points):
            continue
        reserved[insert].add(slot)
        detail=dict(stem_id=stem['id'],stem_name=stem['name'],mixer_insert=insert,
                    family=family,parameter=family,plugin=spec['plugin_label'],
                    effect_slot=slot,target=target,baseline=spec['baseline'])
        lanes.append(dict(name=f'{stem["name"]} - {FAMILIES[family]["label"]}',points=points,
                          targets=[detail],family=display_family,control_family=family,
                          parameter=family,reason=curve.get('reason','Section change'),moves=[curve]))
        additions.append(dict(family=family,insert=insert,slot=slot,stem_name=stem['name'],
                              spec=spec,spec_sha256=spec_digest(spec)))
    files=[dict(path=s['sample'],sha256=s['analysis']['sha256']) for s in analysis['stems']
           if 'sha256' in s.get('analysis',{})]+analysis.get('timing_files',[])
    if uses_plugins and template_path.is_file():
        files.append(dict(path=str(template_path.resolve()),sha256=_file_digest(template_path)))
    if any(e['family'] in ('delay','reverb') for e in additions):
        warnings.append('Delay/reverb use internal wet controls with gradual releases. Wet fades also affect existing tails; this version does not select notes with an auxiliary send.')
    return dict(source=analysis['path'],source_sha256=analysis['sha256'],
                ppq=analysis['ppq'],end_tick=analysis['end_tick'],bpm=analysis['bpm'],
                lanes=lanes,planned_curves=plan['curves'],sections=plan.get('sections',analysis['sections']),
                settings=settings,effective_intensities=plan['effective_intensities'],effects=additions,
                warnings=list(dict.fromkeys(warnings)),skipped=skipped,blocked=list(dict.fromkeys(blocked)),
                input_files=files,template_path=str(template_path.resolve()),
                coverage={'stems_affected':len({t['stem_id'] for l in lanes for t in l['targets']}),
                          'stems_total':len(stems),'controls_ready':len(lanes),
                          'controls_planned':len(plan['curves']),'setup_issues':len(blocked)},
                output=str(Path(analysis['path']).with_name(Path(analysis['path']).stem+'_section_automation.flp')),
                native_fl_playback_tested=False,sound_quality_reviewed=False,
                delivery_status='development_preview')


def export_section_automation(preview, output=None):
    if preview.get('blocked'):
        raise ValueError('Complete effect setup or resolve routing before exporting:\n'+'\n'.join(preview['blocked']))
    source=Path(preview['source']).resolve()
    project=read_project(source)
    if project['sha256']!=preview['source_sha256']:
        raise ValueError('Source FLP changed; generate a fresh preview.')
    for item in preview.get('input_files',[]):
        if _file_digest(item['path'])!=item['sha256']:
            raise ValueError('An audio, MIDI, source-project or plugin-template input changed; analyze again.')
    output=Path(output or preview['output']).resolve()
    if output==source or output.exists():
        raise FileExistsError('Choose a new filename; existing projects are never overwritten.')
    if output.suffix.casefold()!='.flp':
        raise ValueError('Choose an FLP output filename.')
    # Retain relative audio references without changing their meaning.
    if output.parent!=source.parent and any(c.get('sample') and not Path(c['sample']).is_absolute()
        for c in project['channels']):
        raise ValueError('This project has relative audio paths. Save its automated copy beside the source FLP.')
    lanes=preview['lanes']
    events=project['events']; additions=defaultdict(list); changes={}
    blocks,values,counts,owned,saved_effects=_targets(project)
    _,init_index=_mixer(events)
    project_init_index,project_init_rows=_project_initializers(project)
    # Re-resolve native lanes from the saved source rather than trusting a
    # modified preview's target, baseline or normalized points.
    native_baselines={}
    native_lanes=[lane for lane in lanes if lane.get('control_family') in _NATIVE
                  or any(_is_native_target(t.get('target',0)) for t in lane.get('targets',[]))]
    if native_lanes:
        native_info=inspect_stems(source)
        if preview['ppq']!=native_info['ppq'] or preview['end_tick']!=native_info['end_tick']:
            raise ValueError('Native lane timeline differs from the saved source.')
        active={stem['id']:stem for stem in native_info['stems']}
        for lane in native_lanes:
            family=lane.get('control_family')
            if family not in _NATIVE or len(lane.get('targets',[]))!=1 or len(lane.get('moves',[]))!=1:
                raise ValueError('Native lane control metadata is invalid; generate a fresh preview.')
            detail=lane['targets'][0]; curve=lane['moves'][0]
            stem=active.get(detail.get('stem_id'))
            if stem is None or curve.get('family')!=family:
                raise ValueError('Native lane stem metadata is invalid; generate a fresh preview.')
            stem=dict(stem,role=detail.get('role'),
                      automation_protected=detail.get('automation_protected'))
            expected_points,expected_detail=_native_control(curve,stem,blocks,values,counts,owned,
                                                            project_init_rows,preview['end_tick'])
            if (detail!=expected_detail or lane.get('parameter')!=_NATIVE[family]['parameter']
                    or lane.get('points')!=expected_points):
                raise ValueError('Native lane points or saved baseline differ from the preview metadata.')
            native_baselines[detail['target']]=detail['saved_raw']
    reserved_targets=owned|{target for _,target,_ in project_init_rows}
    occupied={(e['insert'],e['slot']) for e in saved_effects}
    init_values={}; parameter_init_values={}
    for effect in preview.get('effects',[]):
        insert,slot,spec=effect['insert'],effect['slot'],effect['spec']
        if spec_digest(spec)!=effect['spec_sha256']:
            raise ValueError('Plugin definition changed; generate a fresh preview.')
        if not 1<=insert<len(blocks)-1 or not 0<=slot<10 or (insert,slot) in occupied:
            raise ValueError('Plugin destination is no longer available.')
        prefix=0x70000000|insert<<22|slot<<16
        if (any(t & 0xffff0000==prefix for t in reserved_targets)
                or any(t & 0xffff0000==prefix and (count>1 or t & 0xffff>=0x8000) for t,count in counts.items())):
            raise ValueError('Effect slot has an existing automation owner or initializer.')
        positions=[blocks[insert][0]+i for i,(k,p) in enumerate(blocks[insert][2])
                   if k==98 and int.from_bytes(p,'little')==slot]
        if len(positions)!=1:
            raise ValueError('Cannot identify an empty effect slot.')
        additions[positions[0]].extend(build_plugin_record(spec,insert,slot))
        occupied.add((insert,slot))
        init_values[prefix|0x1f00]=1
        init_values[prefix|0x1f01]=spec['initial_mix_raw']
        parameter_init_values[prefix|spec['target_offset']]=spec['initial_parameter_raw']
    if init_values:
        original=events[init_index][1]; rows=[]; seen=set()
        for first,target,value in struct.iter_unpack('<III',original):
            if target in init_values:
                value=init_values[target]; seen.add(target)
            rows.append(struct.pack('<III',first,target,value))
        rows.extend(struct.pack('<III',0,t,v) for t,v in sorted(init_values.items()) if t not in seen)
        changes[init_index]=b''.join(rows)
    new_project_init=None
    if parameter_init_values:
        original=events[project_init_index][1] if project_init_index is not None else b''
        new_project_init=original+b''.join(struct.pack('<III',0,t,v)
                                         for t,v in sorted(parameter_init_values.items()))
        if project_init_index is not None:
            changes[project_init_index]=new_project_init
            new_project_init=None
    next_id=max(c['id'] for c in project['channels'])+1
    next_track=max(c['track'] for c in project['clips'])+1
    if next_id+len(lanes)>4095 or next_track+len(lanes)>501:
        raise ValueError('Not enough channel or Playlist space for these clips.')
    first=project['layout']['first_channel']; arrangement=project['layout']['arrangement']
    serial=project['layout']['next_clip_serial']; clips=[]
    groups=sum(k==231 for k,_ in events)
    group_at=max((i for i,(k,_) in enumerate(events[:first]) if k==231),default=-1)
    if lanes and group_at<0:raise ValueError('Missing native channel group layout.')
    if lanes:additions[group_at+1].append((231,'Section automation\0'.encode('utf-16-le')))
    if new_project_init is not None:
        additions[_project_initializer_position(project)].append((216,new_project_init))
    colors={'filter':0xb79473,'delay':0xcba2db,'reverb':0x9bbca5,'width':0x6594ef,
            'volume':0x63b193,'pan':0xd6b968}
    all_targets=set()
    for index,lane in enumerate(lanes):
        cid,track=next_id+index,next_track+index
        if len(lane['targets'])!=1:raise ValueError('Each section lane must own one stem control.')
        target=lane['targets'][0]['target']
        if target in all_targets or target in owned:raise ValueError('Duplicate or already automated target.')
        all_targets.add(target)
        additions[arrangement].extend(build_channel(cid,lane['name'],lane['points'],preview['ppq'],
            group=groups,color=colors.get(lane['control_family'],0x94abca)))
        additions[first].append(build_link(cid,target))
        clips.append(build_playlist_clip(cid,0,preview['end_tick'],track-1,serial+index,
            record_size=project['layout']['record_size'],offset_tick=0,ppq=preview['ppq']))
    for i,(k,p) in enumerate(events):
        if k==238 and len(p)>=13 and next_track<=struct.unpack_from('<I',p)[0]<next_track+len(lanes):
            modified=bytearray(p);modified[12]=1;changes[i]=bytes(modified)
    if clips:
        pi=project['layout']['playlist_index'];changes[pi]=events[pi][1]+b''.join(clips)
    result=[];original_indices=[]
    for i,(k,p) in enumerate(events):
        result.extend(additions[i]);original_indices.append(len(result));result.append((k,changes.get(i,p)))
    header=bytearray(project['header']);struct.pack_into('<H',header,2,len(project['channels'])+len(lanes))
    binary=encode_fl(bytes(header),result)
    if read_bytes(binary)!=(bytes(header),result):raise ValueError('Export round-trip validation failed.')
    if any(result[j]!=event for i,(j,event) in enumerate(zip(original_indices,events)) if i not in changes):
        raise ValueError('Original project preservation failed.')
    checked=read_project(source,data_override=binary)
    if len(checked['channels'])!=len(project['channels'])+len(lanes):raise ValueError('Native channel count differs.')
    _,checked_rows=_project_initializers(checked)
    checked_values={target:value for _,target,value in checked_rows}
    if (checked_rows[:len(project_init_rows)]!=project_init_rows or
            any(checked_values.get(target)!=value for target,value in parameter_init_values.items())):
        raise ValueError('Exported project initializer values differ from the preview.')
    if native_baselines:
        _,checked_mixer_values,checked_counts,_,_=_targets(checked)
        if any(checked_counts[target]!=1 or checked_mixer_values.get(target)!=saved
               for target,saved in native_baselines.items()):
            raise ValueError('Exported native mixer baselines differ from the saved source.')
    actual=plugin_records(checked) if preview.get('effects') else {}
    for effect in preview.get('effects',[]):
        record=actual.get((effect['insert'],effect['slot']))
        if not record or hashlib.sha256(record['fields'][213]).hexdigest()!=effect['spec']['state_sha256']:
            raise ValueError('Inserted plugin state failed preservation validation.')
    for channel in checked['channels']:
        if channel['id']>=next_id:
            fields=dict(checked['events'][channel['start']:channel['stop']])
            parsed=parse_points(fields[234],preview['ppq'])
            expected=lanes[channel['id']-next_id]['points']
            if len(parsed)!=len(expected) or any(abs(a-c)>.00001 or abs(b-d)>1e-10 for (a,b),(c,d) in zip(parsed,expected)):
                raise ValueError('Exported automation points differ from the preview.')
    if source.read_bytes()!=project['data']:raise ValueError('Source changed during export.')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent,prefix='.section-export-',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name);stream.write(binary);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,output)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    return dict(output=str(output),output_sha256=hashlib.sha256(binary).hexdigest(),
        automation_channels=len(lanes),plugin_instances_added=len(preview.get('effects',[])),
        seed=preview['settings']['seed'],effective_intensities=preview['effective_intensities'],
        validation={'roundtrip_exact':True,'original_events_preserved':True,
                    'native_fl_playback_tested':False,'sound_quality_reviewed':False},
        delivery_status='development_preview')
