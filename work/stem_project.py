"""Audio-stem inspection, preview and preservation export using native clips."""
from __future__ import annotations
from collections import defaultdict, Counter
from pathlib import Path
import hashlib
import json
import math
import os
import struct
import tempfile

from flp_raw import read_bytes, encode_fl
from flp_song_analysis import read_project
from flp_connector import _effects
from drum_template import _mixer
from stem_audio import ROLES, detect_role, analyze_audio, detect_sections
from stem_planner import load_presets, plan_moves, SECTION_LABELS
from stem_midi import read_timing
from stem_effects import get_effect_spec, build_effect_record

from app_paths import bundle_root

APP = bundle_root()
LIMITATION = ('Supported effects use dedicated stock Filter, Delay 3 and Reeverb 2 instances, '
             'controlling how much of each added effect is heard. '
             'Unsupported controls are listed as skipped.')


def _signed(value):
    return value-2**32 if value>=2**31 else value


def _file_digest(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def inspect_stems(path):
    project=read_project(path)
    if project['time_signature'] != (4,4):
        raise ValueError('The stem mode currently supports constant-tempo 4/4 projects.')
    if len([1 for k,_ in project['events'] if k==233])!=1:
        raise ValueError('Save one Playlist arrangement before using stem mode.')
    # Constant musical ticks cannot be used as seconds when tempo is automated.
    for key,payload in project['events']:
        if key in (226,227) and len(payload)!=20:
            raise ValueError('An existing automation link has an unknown layout.')
        if key in (226,227) and struct.unpack_from('<I',payload,8)[0]==0x40000005:
            raise ValueError('Tempo automation needs a rendered constant-tempo stem project.')
        if key==223:
            if len(payload)%12: raise ValueError('Unsupported existing event automation.')
            if any(t==0x40000005 for _,t,_ in struct.iter_unpack('<III',payload)):
                raise ValueError('Tempo automation needs a rendered constant-tempo stem project.')
    channels={c['id']:c for c in project['channels']}
    clips=defaultdict(list)
    for clip in project['clips']:
        if clip['kind']=='pattern' and not clip['muted']:
            if project['patterns'][clip['pattern_id']]['notes']:
                raise ValueError('Use a stem-only arrangement; arranged instruments would be missing from audio analysis.')
        elif clip['kind']=='channel' and channels[clip['channel_id']]['type']!=5:
            clips[clip['channel_id']].append(clip)
    stems=[]; warnings=[]
    for cid,placements in clips.items():
        channel=channels[cid]
        if not channel['sample']:
            raise ValueError(f"{channel['name']}: this Playlist channel has no audio file.")
        if len(placements)!=1:
            raise ValueError(f"{channel['name']}: stem mode requires one continuous audio clip.")
        clip=placements[0]
        if clip['muted'] or not clip['track_enabled'] or not channel['enabled']:
            warnings.append(f"{channel['name']}: muted stem excluded."); continue
        raw=project['events'][clip['event_index']][1]
        raw=raw[clip['record_offset']:clip['record_offset']+project['layout']['record_size']]
        offsets=struct.unpack_from('<ff',raw,24)
        if any(x not in (-1.,0.) for x in offsets):
            raise ValueError(f"{channel['name']}: trimmed audio needs consolidation before analysis.")
        fields=dict(project['events'][channel['start']:channel['stop']])
        # Channel FX bit 1 is Reverse. Known channel parameter fields are
        # stretch time/pitch/multiplier (96/100/104) and sample start (148).
        # See PyFLP channel.ParametersEvent; do not analyze altered playback
        # as though its referenced waveform still has the same timing.
        if int.from_bytes(fields.get(70,b'\0\0'),'little')&2:
            raise ValueError(f"{channel['name']}: consolidate reversed audio before analysis.")
        parameters=fields.get(215,b'')
        if any(len(parameters)>=offset+4 and parameters[offset:offset+4]!=b'\0'*4
               for offset in (96,100,104,148)):
            raise ValueError(f"{channel['name']}: consolidate stretched, pitched or offset audio before analysis.")
        sample=Path(os.path.expandvars(channel['sample']))
        if not sample.is_absolute(): sample=Path(project['path']).parent/sample
        if not sample.is_file():
            alternatives=[Path(project['path']).parent/sample.name,
                          Path(project['path']).parent/'stems'/sample.name]
            found=[p for p in alternatives if p.is_file()]
            if len(found)==1: sample=found[0]
        stems.append(dict(id=cid,name=channel['name'] or sample.stem,role=detect_role(channel['name']),
                          mixer_insert=channel['mixer_insert'],sample=str(sample.resolve()),
                          missing=not sample.is_file(),position=clip['position'],
                          offset_tick=clip['position'],duration=clip['duration'],track=clip['track']))
    if not stems: raise ValueError('No active audio stems were found on the Playlist.')
    assignments=Counter(s['mixer_insert'] for s in stems)
    for stem in stems:
        if not 1<=stem['mixer_insert']<=63:
            raise ValueError(f"{stem['name']}: use a dedicated mixer insert 1–63.")
        if assignments[stem['mixer_insert']]!=1:
            raise ValueError('Each stem needs its own mixer insert for independent automation.')
    stems.sort(key=lambda stem:stem['track'])
    return dict(path=project['path'],sha256=project['sha256'],bpm=project['bpm'],
                ppq=project['ppq'],end_tick=max(s['position']+s['duration'] for s in stems),
                stems=stems,warnings=warnings,half_time_available=project['bpm']>150,
                markers=project['markers'])


def analyze_stems(info, *, roles=None, midi_path=None, midi_mapping=None, source_flp=None,
                  half_time=False, cache_dir=None, progress=None):
    if hashlib.sha256(Path(info['path']).read_bytes()).hexdigest()!=info['sha256']:
        raise ValueError('The FLP changed. Load it again before analysis.')
    roles=roles or {}; stems=[dict(s) for s in info['stems']]; warnings=list(info['warnings'])
    for stem in stems:
        stem['role']=roles.get(stem['id'],roles.get(str(stem['id']),stem['role']))
        if stem['role'] not in ROLES:
            raise ValueError(f"Choose a role for {stem['name']} before running analysis.")
        if stem['missing']: raise ValueError(f"Audio file is missing: {stem['sample']}")
        if progress: progress(f"Analyzing {stem['name']}...")
        stem['analysis']=analyze_audio(stem['sample'],bpm=info['bpm'],cache_dir=cache_dir)
        stem['phrase_ends']=[stem['position']+round(t*info['bpm']*info['ppq']/60)
                            for t in stem['analysis'].get('phrase_ends',[])
                            if 0<t<stem['analysis']['duration_seconds']]
        stem['timing_source']='audio activity / silence'
        seconds=stem['duration']/info['ppq']*60/info['bpm']
        if abs(stem['analysis']['duration_seconds']-seconds)>max(.15,seconds*.005):
            raise ValueError(f"{stem['name']}: audio duration differs from its Playlist duration; consolidate stretched/trimmed stems first.")
    guide=None; markers=info['markers']; timing_files=[]
    if midi_path:
        before=_file_digest(midi_path)
        guide=read_timing(midi_path,stems,ppq=info['ppq'],bpm=info['bpm'],end_tick=info['end_tick'],mapping=midi_mapping)
        if _file_digest(midi_path)!=before:raise ValueError('MIDI changed during analysis; run again.')
        timing_files.append(dict(path=str(Path(midi_path).resolve()),sha256=before))
        if not source_flp: warnings.extend(guide['warnings'])
        for stem in stems: stem.update(guide['matches'].get(stem['id'],{}))
        if not markers: markers=guide['markers']
    source_guide=None
    if source_flp:
        from stem_midi import read_source_timing
        before=_file_digest(source_flp)
        source_guide=read_source_timing(source_flp,stems,ppq=info['ppq'],bpm=info['bpm'],end_tick=info['end_tick'])
        if _file_digest(source_flp)!=before:raise ValueError('Source instrument FLP changed during analysis; run again.')
        timing_files.append(dict(path=str(Path(source_flp).resolve()),sha256=before))
        warnings.extend(source_guide['warnings'])
        for stem in stems: stem.update(source_guide['matches'].get(stem['id'],{}))
        if not markers: markers=source_guide['markers']
    sections=detect_sections(stems,bpm=info['bpm'],ppq=info['ppq'],end_tick=info['end_tick'],
                             half_time=half_time,markers=markers)
    return dict(info,stems=stems,sections=sections,warnings=warnings,
                half_time=bool(half_time),midi_guide=guide,source_guide=source_guide,timing_files=timing_files)


# Measured FL normalized fader/dB anchors. Interpolation is approximate (about
# 0.1 dB in the normal mixing range); zero-delta always preserves the exact raw
# saved value. Source: olyrhc/flstudio-volume-converter, measured table.
FADER_DB=[(-60,.0033),(-50,.0104),(-40,.03175),(-35,.0544),(-30,.092),
          (-25,.149),(-20,.232),(-18,.272),(-16,.316),(-14,.365),(-12,.418),
          (-10,.476),(-8,.536),(-6,.6),(-4,.664),(-2,.732),(0,.8),
          (1,.835),(2,.87),(3,.907),(4,.943),(5,.98),(5.6,1.)]


def _interpolate(value, pairs):
    if value<=pairs[0][0]: return pairs[0][1]
    if value>=pairs[-1][0]: return pairs[-1][1]
    for (a,x),(b,y) in zip(pairs,pairs[1:]):
        if a<=value<=b: return x+(value-a)/(b-a)*(y-x)
    raise ValueError('Invalid mapping.')


def relative_volume(saved, delta_db):
    baseline=saved/16000
    if not 0<=baseline<=1: raise ValueError('Unsupported saved fader value.')
    if delta_db==0 or baseline==0: return baseline
    # Below the calibrated range, leave the already quiet stem untouched.
    if baseline < FADER_DB[0][1]:
        raise ValueError('Fader below calibrated -60 dB range; volume moves skipped.')
    saved_db=_interpolate(baseline,[(v,db) for db,v in FADER_DB])
    desired=min(FADER_DB[-1][0],saved_db+max(-12,min(12,delta_db)))
    if desired<FADER_DB[0][0]:
        raise ValueError('Requested fader value below calibrated -60 dB range; move skipped.')
    return _interpolate(desired,FADER_DB)


def _targets(project):
    blocks,init_index=_mixer(project['events'])
    payload=project['events'][init_index][1]
    if len(payload)%12: raise ValueError('Unsupported mixer initialization.')
    rows=list(struct.iter_unpack('<III',payload)); counts=Counter(t for _,t,_ in rows)
    values={t:v for _,t,v in rows}; owned=set()
    for key,data in project['events']:
        if key in (226,227):
            if len(data)!=20: raise ValueError('Unrecognized existing controller link.')
            owned.add(struct.unpack_from('<I',data,8)[0])
        elif key==223:
            if len(data)%12: raise ValueError('Unrecognized existing event automation.')
            owned.update(t for _,t,_ in struct.iter_unpack('<III',data))
    return blocks,values,counts,owned,_effects(project['events'],blocks)


def _move_points(move, normalize, baseline):
    start,end=move['start_tick'],move['end_tick']
    if move['shape']=='explicit':
        raw=move.get('points',[])
        if (not isinstance(raw,(list,tuple)) or len(raw)<2 or
                any(not isinstance(point,(list,tuple)) or len(point)!=2 for point in raw)):
            raise ValueError('Continuous curve points must be tick/value pairs.')
        if any(type(t) is not int or isinstance(v,bool) or
               not isinstance(v,(int,float)) or not math.isfinite(v) for t,v in raw):
            raise ValueError('Continuous curve points need integer ticks and finite values.')
        if raw[0][0]!=start or raw[-1][0]!=end:
            raise ValueError('A continuous curve must cover its complete time range.')
        if any(a[0]>=b[0] for a,b in zip(raw,raw[1:])):
            raise ValueError('Continuous curve points must increase in time.')
        if len(raw)>65535:
            raise ValueError('This continuous curve exceeds the 65535-point native limit.')
        return [(tick,normalize(value)) for tick,value in raw]
    a,b=normalize(move['start_value']),normalize(move['end_value'])
    last=max(start,end-1)
    release_end=max(end,move.get('release_end_tick',end))
    if move['shape']=='swell':
        peak=start+(last-start)//2
        if release_end>end:
            value_at_end=b+(baseline-b)*(end-peak)/(release_end-peak)
            points=[(start,a),(peak,b),(end,value_at_end)]
        else:
            points=[(start,a),(peak,b),(last,a)]
    elif move['shape']=='hold': points=[(start,a),(last,a)]
    elif move['shape']=='step': points=[(start,b),(last,b)]
    else: points=[(start,a),(last,b)]
    if start: points.insert(0,(start-1,baseline))
    # Blend tails fade back to dry, rather than being cut off at a phrase edge.
    # For ordinary mixer moves the existing immediate return is retained.
    if release_end>end:
        points.append((end,points[-1][1]))
    points.append((release_end,baseline))
    return points


def _free_effect_slot(blocks, effects, insert, reserved, owned, counts):
    """Prefer a genuinely empty slot after the saved chain, then an empty gap."""
    occupied={e['slot'] for e in effects if e['insert']==insert}|reserved
    last=max(occupied,default=-1)
    candidates=list(range(last+1,10))+list(range(last+1))
    for slot in candidates:
        if slot in occupied:continue
        prefix=0x70000000|insert<<22|slot<<16
        # An apparently empty slot may still have a saved controller owner.
        if any(t & 0xffff0000 == prefix for t in owned):continue
        if any(t & 0xffff0000 == prefix and count>1 for t,count in counts.items()):continue
        if any(t & 0xffff0000 == prefix and t & 0xffff>=0x8000 for t in counts):continue
        positions=[i for i,(k,p) in enumerate(blocks[insert][2])
                   if k==98 and int.from_bytes(p,'little')==slot]
        if len(positions)==1:return slot
    raise ValueError('no unused effect slot without existing automation is available')


def preview_automation(analysis, genre, subgenre, *, intensity_override='Auto', sections=None, movement_mode=None):
    project=read_project(analysis['path'])
    if project['sha256']!=analysis['sha256']: raise ValueError('FLP changed; analyze again.')
    preset=load_presets()[genre][subgenre]
    plan=plan_moves(analysis['stems'],sections or analysis['sections'],preset,ppq=analysis['ppq'],
                    half_time=analysis['half_time'],intensity_override=intensity_override,
                    movement_mode=movement_mode)
    blocks,values,counts,owned,effects=_targets(project)
    warnings=list(analysis['warnings'])+plan['warnings']; skipped=[]
    by_target=defaultdict(list); baselines={}; target_details={}
    pending_effects={}; reserved=defaultdict(set)
    for move in plan['moves']:
        insert=move['mixer_insert']; parameter=move['parameter']; prefix=0x70000000|(insert<<22)
        reason=None; effect=None
        if not 1<=insert<len(blocks)-1: reason='mixer insert is unavailable'
        elif any(len(dict(block).get(235,b''))>insert and dict(block)[235][insert]
                 for _,_,block in blocks): reason='another mixer insert feeds this stem insert'
        spec=get_effect_spec(move['plugin'],parameter)
        if reason is None and move['plugin']=='Mixer' and parameter in ('volume_db','pan_percent'):
            target=prefix|(0x1fc0 if parameter=='volume_db' else 0x1fc1)
            saved=values.get(target,12800 if parameter=='volume_db' else 0)
            if parameter=='volume_db':
                baseline=saved/16000; normalize=lambda d,s=saved: relative_volume(s,d)
            else:
                pan=_signed(saved)
                if not -6400<=pan<=6400: raise ValueError('Unsupported saved pan position.')
                baseline=(pan+6400)/12800
                normalize=lambda d,b=baseline: max(0,min(1,b+d/200))
        elif reason is None and spec is not None:
            key=(insert,spec['key'])
            effect=pending_effects.get(key)
            if effect is None:
                try:slot=_free_effect_slot(blocks,effects,insert,reserved[insert],owned,counts)
                except ValueError as exc:reason=str(exc)
                else:
                    effect=dict(key=spec['key'],plugin=move['plugin'],parameter=parameter,
                                insert=insert,slot=slot,stem_name=move['stem_name'],
                                target_offset=spec.get('target_offset',0x1f01),
                                spec_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest())
            if effect is not None:
                target=prefix|effect['slot']<<16|effect['target_offset']
                baseline=spec.get('baseline_normalized',0.)
                normalize=lambda value: max(0.,min(1.,value/100))
        elif reason is None:
            reason=f"{move['plugin']} / {parameter}: no verified native parameter support"
        if reason is None and (target in owned or counts[target]>1): reason='parameter already has automation or duplicate initialization'
        if reason is not None:
            skipped.append(dict(stem_id=move['stem_id'],stem=move['stem_name'],move=move['name'],reason=reason)); continue
        try: points=_move_points(move,normalize,baseline)
        except ValueError as exc:
            skipped.append(dict(stem_id=move['stem_id'],stem=move['stem_name'],move=move['name'],reason=str(exc))); continue
        if all(abs(value-baseline)<1e-12 for _,value in points):
            skipped.append(dict(stem_id=move['stem_id'],stem=move['stem_name'],move=move['name'],reason='saved control is at its limit; this move would be flat'));continue
        if any(move['start_tick']<m['end_tick'] and move['end_tick']>m['start_tick'] for m in by_target[target]):
            skipped.append(dict(stem_id=move['stem_id'],stem=move['stem_name'],move=move['name'],reason='overlapping move on the same parameter')); continue
        by_target[target].append(dict(move,points=points)); baselines[target]=baseline
        if effect is not None:
            pending_effects[(insert,effect['key'])]=effect
            reserved[insert].add(effect['slot'])
        target_details[target]=dict(stem_id=move['stem_id'],stem_name=move['stem_name'],
                                    mixer_insert=insert,parameter=parameter,family=move['family'],
                                    plugin=move['plugin'],effect_slot=effect['slot'] if effect else None)
    # One channel owns each parameter. Joining the sparse gestures avoids
    # competing links from different clips on the same parameter.
    lanes=[]; groups={}
    for target,moves in by_target.items():
        points={0:baselines[target],analysis['end_tick']:baselines[target]}
        ordered=sorted(moves,key=lambda m:m['start_tick'])
        for index,move in enumerate(ordered):
            # Leave one baseline tick before the next gesture. Release tails
            # may be shortened but must never overwrite the following move.
            stop=(ordered[index+1]['start_tick']-1 if index+1<len(ordered) else analysis['end_tick'])
            local=[(t,v) for t,v in move['points'] if t<=stop]
            if index and ordered[index-1]['end_tick']>=move['start_tick']:
                # Adjacent moves share a boundary, not the prior endpoint.
                # The next move's pre-roll baseline must not erase that peak.
                local=[(t,v) for t,v in local if t!=move['start_tick']-1]
            if move.get('release_end_tick',move['end_tick'])>stop and stop>=move['end_tick']:
                local.append((stop,baselines[target]))
            points.update(local)
        anchors=sorted(points.items())
        compact=[]
        for point in anchors:
            if len(compact)>=2:
                (a,x),(b,y)=compact[-2:]; c,z=point
                if abs((y-x)*(c-b)-(z-y)*(b-a))<1e-10: compact.pop()
            compact.append(point)
        details=target_details[target]
        key=(details['family'],details['plugin'],details['parameter'],tuple(compact))
        if plan.get('movement_mode')=='volume_balance':
            # Keep each stem independently editable even when its calculated
            # curve happens to match another stem's curve exactly.
            key+=(details['stem_id'],)
        if key not in groups:
            names=list(dict.fromkeys(m['name'] for m in moves))
            groups[key]=dict(points=compact,targets=[],moves=moves,names=names,
                             parameter=details['parameter'],family=details['family'])
            lanes.append(groups[key])
        else:
            groups[key]['moves'].extend(moves)
            groups[key]['names']=list(dict.fromkeys(groups[key]['names']+[m['name'] for m in moves]))
        groups[key]['targets'].append(dict(details,target=target,baseline=baselines[target]))
    for lane in lanes:
        prefix=lane['family'] if len(lane['targets'])>1 else lane['targets'][0]['stem_name']
        lane['name']=prefix+' - '+' / '.join(lane['names'])
    used_effects=[effect for effect in pending_effects.values()
                  if (0x70000000|effect['insert']<<22|effect['slot']<<16|effect['target_offset']) in by_target]
    if used_effects:
        if any(e['parameter']=='lowpass_cutoff_percent' for e in used_effects):
            warnings.append('Filter curves move the actual cutoff control. Percent values are its knob position, not a frequency in Hz. The original effects retain their settings.')
        else:
            warnings.append('The added effect amount returns to zero after each change. Filter settings stay fixed; the automation changes how much of the filtered sound is heard.')
    if skipped:
        warnings.append(LIMITATION)
        warnings.extend(f"{s['stem']}: {s['move']} skipped — {s['reason']}." for s in skipped)
    warnings=list(dict.fromkeys(warnings))
    summaries=[]; section_activity=[]; movement_coverage={}
    if plan.get('planning_mode')=='continuous':
        from stem_continuous import _activity, _covered
        activity={s['id']:_activity(s,analysis['end_tick'],analysis['ppq'],analysis['bpm'])[0]
                  for s in analysis['stems']}
    for section in plan['sections']:
        label=f"{SECTION_LABELS.get(section['type'],section['type'].title())} at bar {section['bar']}: {section['intensity']} — "
        if plan.get('planning_mode')=='continuous':
            moving=[]
            for lane in lanes:
                for target in lane['targets']:
                    measured=_covered(lane['points'],activity[target['stem_id']],
                                      section['start_tick'],section['end_tick'],.02)
                    # Measure the accepted, normalized curves, including clipping
                    # at saved control limits. Planned or skipped controls do not
                    # contribute to the export's section coverage.
                    if measured['moving_ticks']:
                        moving.append({k:target[k] for k in ('stem_id','plugin','parameter')} |
                                      {k:measured[k] for k in ('active_ticks','moving_ticks','changing_active_fraction')})
            stem_count=len({t['stem_id'] for t in moving})
            section_activity.append(dict(start_tick=section['start_tick'],end_tick=section['end_tick'],
                                         stem_count=stem_count,moving_targets=len(moving),targets=moving))
            summaries.append(label+(f'{len(moving)} moving controls across {stem_count} active stems'
                                    if moving else 'no supported movement during detected sound'))
        else:
            used=[m for lane in lanes for m in lane['moves'] if m['section_start_tick']==section['start_tick']]
            actions=list(dict.fromkeys(f"{m['name']} on {m['stem_name']}" for m in used))
            summaries.append(label+('; '.join(actions) or 'no supported changes'))
    if section_activity:
        movement_coverage=dict(song_end_tick=analysis['end_tick'],section_activity=section_activity,
                               sections_with_movement=sum(s['moving_targets']>0 for s in section_activity),
                               total_sections=len(section_activity),
                               note='Coverage measures exported control changes during detected activity; listening review is still required.')
    rendered=sum(len(l['moves']) for l in lanes)
    coverage=dict(stems_affected=len({t['stem_id'] for l in lanes for t in l['targets']}),
                  stems_total=len(analysis['stems']),gestures_rendered=rendered,
                  gestures_planned=len(plan['moves']),gestures_skipped=len(skipped))
    if skipped:
        first_skip=skipped[0]
        warnings.insert(0,f"Partial result: {len(skipped)} of {len(plan['moves'])} planned changes skipped. "
                         f"{first_skip['stem']}: {first_skip['reason']}.")
    capabilities=dict(mixer_targets=sum(t['effect_slot'] is None for l in lanes for t in l['targets']),
                      effect_blend_targets=sum(t['effect_slot'] is not None and t['parameter']!='lowpass_cutoff_percent' for l in lanes for t in l['targets']),
                      filter_targets=sum(t['parameter']=='lowpass_cutoff_percent' for l in lanes for t in l['targets']),
                      effects_added=len(used_effects))
    # A planned drum curve is ready only if the native volume target accepted
    # it. Existing owners/limits must never count as successfully covered drums.
    accepted_drums={m['stem_id'] for lane in lanes for m in lane['moves']
                    if m.get('drum_dynamics') and m['parameter']=='volume_db'}
    drum_dynamics=[]
    for planned in plan.get('drum_dynamics',[]):
        record=dict(planned)
        if record.get('status')=='planned':
            if record['stem_id'] in accepted_drums:
                record['status']='ready'
            else:
                reasons=[s['reason'] for s in skipped if s.get('stem_id')==record['stem_id']
                         and 'volume' in s['move'].lower()]
                record.update(status='skipped',reason='; '.join(dict.fromkeys(reasons)) or
                              'The native volume control could not accept this curve.')
        drum_dynamics.append(record)
    drum_coverage=dict(drums_total=len(drum_dynamics),
                       drums_ready=sum(d['status']=='ready' for d in drum_dynamics),
                       hits_guided=sum(d.get('hit_count',0) for d in drum_dynamics if d['status']=='ready'),
                       drums_unmatched=sum(d['status']=='unmatched' for d in drum_dynamics),
                       drums_skipped=sum(d['status']=='skipped' for d in drum_dynamics),
                       drums_silent=sum(d['status']=='silent' for d in drum_dynamics),
                       drums_disabled=sum(d['status']=='disabled' for d in drum_dynamics))
    accepted_balance={t['stem_id'] for lane in lanes for t in lane['targets']
                      if t['parameter']=='volume_db'}
    balance_details=[]
    for planned in plan.get('balance_details',[]):
        record=dict(planned)
        if record.get('status')=='planned':
            if record['stem_id'] in accepted_balance:
                record['status']='ready'
            else:
                reasons=[s['reason'] for s in skipped if s.get('stem_id')==record['stem_id']]
                record.update(status='skipped',reason='; '.join(dict.fromkeys(reasons)) or
                              'The native volume control could not accept this curve.')
        balance_details.append(record)
    balance_coverage=dict(stems_total=len(balance_details),
                          stems_ready=sum(d['status']=='ready' for d in balance_details),
                          stems_unchanged=sum(d['status']=='unchanged' for d in balance_details),
                          stems_skipped=sum(d['status']=='skipped' for d in balance_details),
                          stems_without_audio=sum(d['status']=='no_audio' for d in balance_details),
                          guides_used=sum(d.get('guide_note_count',0)>0 for d in balance_details if d['status']=='ready'))
    output_suffix=('_volume_balance_preview.flp' if plan.get('movement_mode')=='volume_balance'
                   else '_automation_preview.flp')
    return dict(source=analysis['path'],source_sha256=analysis['sha256'],genre=genre,subgenre=subgenre,
                ppq=analysis['ppq'],end_tick=analysis['end_tick'],bpm=analysis['bpm'],lanes=lanes,
                sections=plan['sections'],summary=summaries,warnings=warnings,skipped=skipped,
                effects=used_effects,coverage=coverage,capabilities=capabilities,
                section_activity=section_activity,
                movement_coverage=movement_coverage,
                drum_dynamics=drum_dynamics,drum_coverage=drum_coverage,
                balance_details=balance_details,balance_coverage=balance_coverage,
                planning_mode=plan.get('planning_mode','selected_changes'),
                movement_mode=plan.get('movement_mode','full_song' if plan.get('planning_mode')=='continuous' else 'brief'),
                half_time=analysis['half_time'],intensity_override=intensity_override,
                input_files=[dict(path=s['sample'],sha256=s['analysis']['sha256']) for s in analysis['stems']
                             if 'sha256' in s.get('analysis',{})]+analysis.get('timing_files',[]),
                output=str(Path(analysis['path']).with_name(Path(analysis['path']).stem+output_suffix)),
                native_fl_playback_tested=False,delivery_status='development_preview',sound_quality_reviewed=False)


def recalculate_section_energy(analysis):
    """Reintegrate streamed chunk powers after a user changes boundaries."""
    seconds_per_tick=60/(analysis['bpm']*analysis['ppq'])
    values=[]
    for section in analysis['sections']:
        start=section['start_tick']*seconds_per_tick; end=section['end_tick']*seconds_per_tick
        power=0.
        for stem in analysis['stems']:
            offset=stem['position']*seconds_per_tick
            for chunk in stem['analysis']['chunks']:
                overlap=max(0,min(end,offset+chunk['end_seconds'])-max(start,offset+chunk['start_seconds']))
                power+=chunk['rms']**2*overlap/(end-start)
        values.append(math.sqrt(power))
    peak=max(values,default=0) or 1.; previous=0.
    for index,(section,raw) in enumerate(zip(analysis['sections'],values)):
        energy=raw/peak; section.update(energy=energy,energy_jump=energy-previous if index else 0.)
        previous=energy


def export_automation(preview, output=None):
    from stem_native import build_channel, build_link, build_playlist_clip
    source=Path(preview['source']); project=read_project(source)
    if project['sha256']!=preview['source_sha256']: raise ValueError('The source FLP changed; create a new preview.')
    output=Path(output or preview['output']).resolve()
    if output==source.resolve() or output.exists(): raise FileExistsError('Choose a new output filename; existing files are never overwritten.')
    if output.suffix.casefold()!='.flp':raise ValueError('The output must be an FLP file.')
    for item in preview.get('input_files',[]):
        if _file_digest(item['path'])!=item['sha256']:raise ValueError('An audio/timing input changed; analyze and preview again.')
    lanes=preview['lanes']
    if not lanes: raise ValueError('No supported native automation moves are available for this selection.')
    events=project['events']; additions=defaultdict(list); changes={}
    effects_to_add=preview.get('effects',[])
    if effects_to_add:
        blocks,values,counts,owned,saved_effects=_targets(project)
        _,init_index=_mixer(events)
        occupied={(e['insert'],e['slot']) for e in saved_effects}
        initial_values={}
        for effect in effects_to_add:
            insert,slot=effect['insert'],effect['slot']
            if not 1<=insert<len(blocks)-1 or not 0<=slot<10 or (insert,slot) in occupied:
                raise ValueError('A requested effect slot is unavailable; create a new preview.')
            prefix=0x70000000|insert<<22|slot<<16
            if any(t & 0xffff0000 == prefix for t in owned):
                raise ValueError('The requested effect slot already has automation.')
            if any(t & 0xffff0000 == prefix and count>1 for t,count in counts.items()):
                raise ValueError('The requested effect slot has duplicate initialization.')
            if any(t & 0xffff0000 == prefix and t & 0xffff>=0x8000 for t in counts):
                raise ValueError('The requested empty effect slot retains internal parameter initialization.')
            spec=get_effect_spec(effect['plugin'],effect['parameter'])
            if spec is None or hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()!=effect['spec_sha256']:
                raise ValueError('Stock-effect definitions changed; create a new preview.')
            positions=[blocks[insert][0]+i for i,(k,p) in enumerate(blocks[insert][2])
                       if k==98 and int.from_bytes(p,'little')==slot]
            if len(positions)!=1:raise ValueError('Cannot locate a unique empty effect slot.')
            additions[positions[0]].extend(build_effect_record(spec,insert,slot))
            occupied.add((insert,slot))
            initial_values[prefix|0x1f00]=1
            initial_values[prefix|0x1f01]=spec['initial_mix_raw']
            if spec.get('control')=='plugin_parameter':
                initial_values[prefix|spec['target_offset']]=spec['initial_parameter_raw']
        initial=events[init_index][1]; patched=[]; seen=set()
        for offset in range(0,len(initial),12):
            row=initial[offset:offset+12]; target=struct.unpack_from('<I',row,4)[0]
            if target in initial_values:
                row=row[:8]+struct.pack('<I',initial_values[target]);seen.add(target)
            patched.append(row)
        patched.extend(struct.pack('<III',0,t,v) for t,v in sorted(initial_values.items()) if t not in seen)
        changes[init_index]=b''.join(patched)
    next_id=max(c['id'] for c in project['channels'])+1
    next_track=max(c['track'] for c in project['clips'])+1
    if next_id+len(lanes)>4095 or next_track+len(lanes)>501:
        raise ValueError('Insufficient FL channel or Playlist track space.')
    first=project['layout']['first_channel']; arrangement=project['layout']['arrangement']
    serial=project['layout']['next_clip_serial']; clips=[]
    group_count=sum(k==231 for k,_ in events)
    group_at=max((i for i,(k,_) in enumerate(events[:first]) if k==231),default=-1)
    if group_at<0: raise ValueError('Missing channel group layout.')
    additions[group_at+1].append((231,('Stem automation\0').encode('utf-16-le')))
    color={'Drums':0x6594ef,'Melodic':0xb79473,'Bass':0x63b193,'Vocals':0xcba2db,'Effects':0x9bbca5}
    for index,lane in enumerate(lanes):
        cid=next_id+index; track=next_track+index
        additions[arrangement].extend(build_channel(cid,lane['name'],lane['points'],preview['ppq'],
                                                    group=group_count,color=color.get(lane['family'],0x94abca)))
        for target in lane['targets']: additions[first].append(build_link(cid,target['target']))
        # Native envelopes already contain the complete musical timeline.
        # One continuous placement per lane needs no velocity-era segmentation.
        clips.append(build_playlist_clip(cid,0,preview['end_tick'],track-1,serial+len(clips),
                     record_size=project['layout']['record_size'],offset_tick=0,ppq=preview['ppq']))
    # Reuse existing unused tracks and enable only new automation placements.
    for i,(k,p) in enumerate(events):
        if k==238 and len(p)>=13 and next_track<=struct.unpack_from('<I',p)[0]<next_track+len(lanes):
            q=bytearray(p); q[12]=1; changes[i]=bytes(q)
    pi=project['layout']['playlist_index']; changes[pi]=events[pi][1]+b''.join(clips)
    result=[]; original_indices=[]
    for i,(k,p) in enumerate(events):
        result.extend(additions[i]); original_indices.append(len(result)); result.append((k,changes.get(i,p)))
    header=bytearray(project['header']); struct.pack_into('<H',header,2,len(project['channels'])+len(lanes))
    binary=encode_fl(bytes(header),result)
    if read_bytes(binary)!=(bytes(header),result): raise ValueError('FLP round-trip validation failed.')
    if any(result[j]!=event for i,(j,event) in enumerate(zip(original_indices,events)) if i not in changes):
        raise ValueError('Original event preservation failed.')
    checked=read_project(source,data_override=binary)
    if len(checked['channels'])!=len(project['channels'])+len(lanes): raise ValueError('Native channel validation failed.')
    if effects_to_add:
        check_blocks,_=_mixer(checked['events'])
        actual={(e['insert'],e['slot']):e['plugin'] for e in _effects(checked['events'],check_blocks)}
        if any(actual.get((e['insert'],e['slot']))!=e['plugin'] for e in effects_to_add):
            raise ValueError('Inserted stock-effect validation failed.')
    if source.read_bytes()!=project['data']: raise ValueError('The input changed during export.')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent,prefix='.stem-export-',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name);stream.write(binary);stream.flush();os.fsync(stream.fileno())
        # Hard-link publication is atomic and fails if a racing output exists,
        # on Windows and POSIX. The staging file is in the same directory.
        os.link(temporary,output)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    report=dict(preview,output=str(output),output_sha256=hashlib.sha256(binary).hexdigest(),
                validation=dict(roundtrip_exact=True,original_events_preserved=True,
                                original_playlist_preserved=True,native_fl_playback_tested=False,delivery_status='development_preview',sound_quality_reviewed=False),
                automation_channels=len(lanes),automation_links=sum(len(l['targets']) for l in lanes),
                stock_effects_added=len(effects_to_add))
    return report
