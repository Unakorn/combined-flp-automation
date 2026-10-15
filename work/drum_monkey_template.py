"""Import one saved Drum Monkey instance, with shared effects on insert29.

The generator's opaque VST state and wrapper output configuration stay exact.
Insert30 is reserved for the kick; output isolation is left to the user.
No musical notes, sampler channels, pad buses or plugin output maps are changed.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import struct

from flp_raw import read_bytes, encode_fl
from drum_detection import _preset_kit
from drum_template import _channels, _mixer, _effects, _name, _text, OPEN_FILTER, UNITY_BALANCE
from sylenth_events import clone_controller

DRUMS_INSERT = 29
KICK_INSERT = 30
LANES = ('cutoff', 'volume', 'delay', 'reverb')
WRAPPER_KEYS = (201, 212, 203, 155, 128, 41, 213)
PATTERN_KEYS = {65, 193, 150, 157, 158, 223, 224}

# Native generator shell containing channel defaults, not a sampler.
# Only used when .fst supplies plugin settings but no full generator channel.
# These675 bytes contain FL channel defaults/envelopes, not plugin/sample data.
# The shell was checked against native saves. Identity/routing/group fields below
# are replaced for the new project; a wrapper-only FST has no channel cut rule,
# so its Cut/CutBy is explicitly neutral0/0 instead of the donor's self-cut.
# The chosen plugin record is supplied by the user's .fst. No donor dependency
# remains at runtime.
_NATIVE_SUFFIX_HEX = ((0, '01'), (209, '0000000000190000000000000400000090000000'), (138, '80008000'), (139, '00000100'), (89, '0000'), (97, '8000'), (48, '00'), (69, '8000'), (86, '0001'), (71, '0004'), (83, '0000'), (74, '0000'), (75, '0000'), (76, '0000'), (85, '0008'), (131, '00008000'), (70, '0000'), (104, '0000'), (50, '01'), (219, '001900001027000000000000000100000000000000000000'), (229, '0000000000320000000000000000000000000000'), (221, '00000000f401000000'), (215, 'ffffffff0000000001000001ffffffff3c0000000000803f0000803f0000803f0000803f0000803f0000000001000000ffffffff000400003000000000000100a7050000000000000001000000000000000000000000000000000000010000000000000000000000000000000000000002000000feffffffffffffff000000000000000000000000000000000000f03f0000000000000000ffffffff01010000000000000000e03f'), (132, '09000900'), (144, '00000000'), (145, '01000000'), (32, '00'), (228, '64000000000000000000000000000000'), (228, '3c000000000000000000000000000000'), (218, '000000000000000064000000204e0000204e00003075000032000000204e00000000000064000000204e000000000000b680000000000000000000000000000000000000'), (218, '040000000000000064000000204e0000204e00003075000032000000204e00000000000064000000204e000000000000b68000000000000000000000000000009bffffff'), (218, '000000000000000064000000204e0000204e00003075000032000000204e00000000000064000000204e000000000000b680000000000000000000000000000000000000'), (218, '000000000000000064000000204e0000204e00003075000032000000204e00000000000064000000204e000000000000b680000000000000000000000000000000000000'), (218, '000000000000000064000000204e0000204e00003075000032000000204e00000000000064000000204e000000000000b680000000000000000000000000000000000000'), (143, '0a000000'), (20, '00'), (170, 'ffffffff'), (51, '00'))


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _wrapper(record):
    if tuple(key for key, _ in record) != WRAPPER_KEYS:
        raise ValueError('The selected Drum Monkey wrapper record uses an unsupported layout.')
    fields = dict(record)
    if (_text(fields[201]) != 'Fruity Wrapper' or len(fields[212]) != 52
            or fields[212][:12] != struct.pack('<III', 0, 0, 2)
            or any(len(fields[key]) != size for key, size in ((155,4),(128,4),(41,1)))):
        raise ValueError('Choose a native FL Studio Drum Monkey generator preset.')
    state = fields[213]
    if len(state) < 4 or struct.unpack_from('<I', state)[0] != 12:
        raise ValueError('This Drum Monkey wrapper state version has not been validated.')
    chunks, position = {}, 4
    while position < len(state):
        if len(state) - position < 12:
            raise ValueError('The saved Drum Monkey wrapper state is incomplete.')
        key, size, reserved = struct.unpack_from('<III', state, position)
        if key in chunks or reserved != 0 or size > len(state) - position - 12:
            raise ValueError('The saved Drum Monkey wrapper chunks are unsupported.')
        chunks[key] = state[position + 12:position + 12 + size]
        position += 12 + size
    if chunks.get(54) != b'Unison Drum Monkey' or 53 not in chunks or b'DrumMonkey\0' not in chunks[53]:
        raise ValueError('The selected wrapper is not the verified Drum Monkey instrument.')
    return fields


def _read_selected(path, kit_id, expected_sha256):
    source = Path(path).expanduser().resolve()
    if not source.is_file() or not 22 <= source.stat().st_size <= 64_000_000:
        raise ValueError('Choose a saved Drum Monkey .fst preset or FL Studio project smaller than 64 MB.')
    data = source.read_bytes()
    digest = _sha(data)
    if expected_sha256 is not None and expected_sha256 != digest:
        raise ValueError('The Drum Monkey preset changed since selection. Choose it again before generating.')
    try:
        header, events = read_bytes(data)
    except ValueError as exc:
        raise ValueError('Choose a Drum Monkey FL Studio .fst preset or .flp project.') from exc
    if encode_fl(header, events) != data:
        raise ValueError('The chosen preset has unsupported event encoding.')
    is_project = any(key == 64 for key, _ in events)
    blocks = _channels(events)[2] if is_project else [events]
    candidates, cursor = [], 0
    for block in blocks:
        # Locate original event positions for stable inspect_drum_preset kit IDs.
        start = next((i for i in range(cursor, len(events)) if events[i:i+len(block)] == block), None)
        if start is None:
            raise ValueError('The selected Drum Monkey channel cannot be identified.')
        cursor = start + len(block)
        states = [(i,payload) for i,(key,payload) in enumerate(block)
                  if key == 213 and b'DrumMonkey\0' in payload]
        if not states:
            continue
        if len(states) != 1:
            raise ValueError('The selected channel contains several Drum Monkey states.')
        offset, state = states[0]
        first = offset - len(WRAPPER_KEYS) + 1
        if first < 0:
            raise ValueError('The selected Drum Monkey wrapper is incomplete.')
        record = block[first:offset+1]
        fields = _wrapper(record)
        cid = int.from_bytes(dict(block)[64], 'little') if is_project else None
        identifier = f'channel:{cid}:state:{start+offset}' if is_project else f'state:{start+offset}'
        kit = _preset_kit(state, identifier, _text(fields[203]))
        kit['channel_id'] = cid
        if is_project:
            channel = [event for event in block if event[0] not in PATTERN_KEYS]
            keys = [key for key, _ in channel]
            expected_keys = [64,21,*WRAPPER_KEYS,*[key for key,_ in _NATIVE_SUFFIX_HEX]]
            # Save of a changed root note may add event135; preserve it exactly.
            if [key for key in keys if key != 135] != expected_keys or keys.count(135) > 1:
                raise ValueError('This saved Drum Monkey channel has unverified shared settings. Save its wrapper preset as .fst and choose that file.')
            if dict(channel)[21] != b'\x02':
                raise ValueError('The selected Drum Monkey instance is not a generator.')
        else:
            if any(key in {64,21,104,132,145,215,234} for key,_ in block):
                raise ValueError('This preset contains a partial channel; save a wrapper-only .fst preset.')
            channel = [(64,b'\0\0'),(21,b'\x02'),*record,
                       *[(key,bytes.fromhex(payload)) for key,payload in _NATIVE_SUFFIX_HEX]]
        candidates.append((kit,channel,fields))
    if not candidates:
        raise ValueError('The selected file contains no supported Drum Monkey instrument.')
    if kit_id is None:
        if len(candidates) != 1:
            raise ValueError('This project contains several Drum Monkey kits. Choose one before generating.')
        selected = candidates[0]
    else:
        matches = [item for item in candidates if item[0]['id'] == kit_id]
        if len(matches) != 1:
            raise ValueError('The chosen Drum Monkey kit is no longer in this file. Select it again.')
        selected = matches[0]
    kit, channel, fields = selected
    return source, data, digest, kit, channel, fields


def prepare_drum_monkey(header, events, preset_path, kit_id=None, expected_sha256=None,
                        drums_insert=DRUMS_INSERT, kick_insert=KICK_INSERT):
    """Return (new_header, new_events, manifest), without writing any input.

    One actual Drum Monkey generator and four Keyboard Controllers are added.
    Main channel/effects use29;30 is labeled Kick with no new effects or links.
    The user must finish/verify Drum Monkey's output routing; selected plugin
    state, pad settings and wrapper output maps are never rewritten here.
    """
    if (drums_insert,kick_insert) != (29,30):
        raise ValueError('This prepared setup reserves mixer insert 29 for drums and insert 30 for kick.')
    source, preset_bytes, preset_sha, kit, source_channel, plugin = _read_selected(preset_path,kit_id,expected_sha256)
    original = list(events)
    starts, arrangement, channels = _channels(original)
    ids = [int.from_bytes(dict(block)[64],'little') for block in channels]
    if len(header) != 6 or len(ids) != len(set(ids)) or struct.unpack_from('<H',header,2)[0] != len(ids):
        raise ValueError('The template has inconsistent channel identities.')
    if any(key==213 and b'DrumMonkey\0' in payload for block in channels for key,payload in block):
        raise ValueError('The template already contains Drum Monkey. Choose the included template to add one dedicated instance.')
    next_id = max(ids)+1
    if next_id + 5 > 4095:
        raise ValueError('The template has too many channels for this drum setup.')
    groups = [i for i,(key,_) in enumerate(original) if key==231]
    if not groups or groups[-1] >= starts[0]:
        raise ValueError('The template channel groups use an unsupported layout.')
    group_id = len(groups)
    links = [(i,payload) for i,(key,payload) in enumerate(original) if key==227]
    if any(len(payload)!=20 or i>=starts[0] for i,payload in links):
        raise ValueError('The template controller-link layout is unsupported.')
    linked_routes = set()
    for key,payload in original:
        if key in {226,227}:
            if len(payload)!=20:
                raise ValueError('The template contains an unsupported parameter link.')
            target=struct.unpack_from('<I',payload,8)[0]
            if target>>28==7:
                linked_routes.add((target & 0x0fc00000)>>22)
    blocks, init_index = _mixer(original)
    original_effects = _effects(original,blocks)
    occupied = {int.from_bytes(dict(block).get(104,b'\0\0'),'little') for block in channels}
    occupied |= linked_routes | {effect['insert'] for effect in original_effects}
    for insert in (29,30):
        if insert >= len(blocks)-1 or insert in occupied:
            raise ValueError(f'Mixer insert {insert} must be unused for the drum/kick setup.')
        fields=dict(blocks[insert][2])
        if (_text(fields.get(204,b'')) or fields.get(235)!=b'\x01' or fields.get(154)!=b'\xff'*4
                or [int.from_bytes(p,'little') for k,p in blocks[insert][2] if k==98]!=list(range(10))
                or any(len(dict(block).get(235,b''))>insert and dict(block)[235][insert] for _,_,block in blocks)):
            raise ValueError(f'Mixer insert {insert} has existing routing/settings; use the included empty prepared insert.')
    references={}
    for lane, name in [('cutoff','Fruity Filter'),('volume','Fruity Balance'),('delay','Fruity Delay 3')]:
        matches=[fx for fx in original_effects if fx['plugin']==name]
        if matches: references[lane]=matches[0]
    matches=[fx for fx in original_effects if fx['plugin']=='Fruity Wrapper' and b'ValhallaFutureVerb' in fx['state']]
    if matches: references['reverb']=matches[0]
    if set(LANES)-references.keys():
        raise ValueError('The prepared template is missing a required drum effect type.')
    cuts=[value for key,payload in original if key==132 for value in struct.unpack('<HH',payload)]
    next_cut=max(cuts,default=0)+1
    if next_cut+4>65535:
        raise ValueError('No independent channel cut groups remain.')
    name='Drum Monkey'
    saved_cut=struct.unpack('<HH',dict(source_channel)[132])
    if kit['channel_id'] is None or saved_cut==(0,0):
        generator_cut=(0,0)
    elif saved_cut[0]==saved_cut[1]:
        generator_cut=(next_cut,next_cut)
    else:
        raise ValueError('The saved Drum Monkey channel has a cross-channel cut relationship. Save its wrapper preset as .fst to import one independent instrument.')
    replacements={64:struct.pack('<H',next_id),203:_name(name),104:struct.pack('<H',29),
                  132:struct.pack('<HH',*generator_cut),145:struct.pack('<I',group_id)}
    if any(sum(key==wanted for key,_ in source_channel)!=1 for wanted in replacements):
        raise ValueError('The Drum Monkey generator has an incomplete channel setting.')
    new_channels=[(key,replacements.get(key,payload)) for key,payload in source_channel]
    if dict(new_channels)[213]!=plugin[213] or dict(new_channels)[212]!=plugin[212]:
        raise ValueError('The saved Drum Monkey state changed while importing it.')
    additions_at={}
    for insert,label in [(29,'Drums'),(30,'Kick')]:
        start,_,block=blocks[insert]
        at=next(start+i for i,(key,_) in enumerate(block) if key==236)
        additions_at[at]=[(204,_name(label)),(149,bytes.fromhex('bd896600'))]
    start,_,block=blocks[29]
    positions={int.from_bytes(payload,'little'):start+i for i,(key,payload) in enumerate(block) if key==98}
    controllers,new_links,required_initial=[],[],{}
    for slot,lane in enumerate(LANES):
        reference=references[lane]
        record=[]
        for key,payload in reference['record']:
            if key==212:
                settings=bytearray(payload)
                struct.pack_into('<II',settings,0,29,slot)
                struct.pack_into('<I',settings,16,struct.unpack_from('<I',settings,16)[0]&~1)
                payload=bytes(settings)
            elif key==213 and lane=='cutoff':
                if len(payload)!=29: raise ValueError('The filter state layout is unsupported.')
                payload=OPEN_FILTER
            elif key==213 and lane=='volume':
                if len(payload)!=8: raise ValueError('The volume state layout is unsupported.')
                payload=UNITY_BALANCE
            record.append((key,payload))
        additions_at.setdefault(positions[slot],[]).extend(record)
        prefix=0x70000000|(29<<22)|(slot<<16)
        parameter=0x8000 if lane=='cutoff' else 0x8001 if lane=='volume' else 0x1f01
        target=prefix|parameter
        required_initial[prefix|0x1f00]=1
        required_initial[prefix|0x1f01]=0 if lane in {'delay','reverb'} else 12800
        cid=next_id+slot+1
        label=f'{name} CTRL {lane} drums'
        new_channels.extend(clone_controller(original,cid,label,group_id+1,next_cut+slot+1))
        new_links.append((227,struct.pack('<5I',(cid<<16)|0x8001,0,target,8,469)))
        controllers.append({'controller_id':cid,'instrument_id':next_id,'name':label,'lane':lane,
                            'mixer_insert':29,'effect_slot':slot,'plugin':reference['plugin'],'target':f'{target:08x}'})
    initial=original[init_index][1]
    if len(initial)%12: raise ValueError('The template initialization data is unsupported.')
    records,changes,seen=[],[],set()
    for at in range(0,len(initial),12):
        record=initial[at:at+12]
        target,value=struct.unpack_from('<II',record,4)
        if target>>28==7 and (target&0x0fc00000)>>22 in {29,30} and target&0xffff>=0x8000:
            raise ValueError('The reserved drum/kick insert retains old plugin parameter initializers.')
        if target in required_initial:
            if target in seen: raise ValueError('A drum effect has duplicate saved initialization values.')
            seen.add(target)
            replacement=required_initial[target]
            if value!=replacement:
                changes.append({'target':f'{target:08x}','before':value,'after':replacement})
                record=record[:8]+struct.pack('<I',replacement)
        records.append(record)
    for target in sorted(required_initial.keys()-seen):
        records.append(struct.pack('<III',0,target,required_initial[target]))
        changes.append({'target':f'{target:08x}','before':None,'after':required_initial[target]})
    tagged_links=[((227,p),False) for _,p in links]+[(item,True) for item in new_links]
    tagged_links.sort(key=lambda row:struct.unpack_from('<I',row[0][1])[0])
    first_link=links[0][0] if links else starts[0]
    tagged=[]
    for index,event in enumerate(original):
        if index==groups[-1]+1:
            tagged.extend([((231,_name('Drum Monkey')),True),((231,_name('Drum automation')),True)])
        if index==first_link: tagged.extend(tagged_links)
        if index==arrangement: tagged.extend((row,True) for row in new_channels)
        tagged.extend((row,True) for row in additions_at.get(index,[]))
        if event[0]!=227:
            tagged.append(((225,b''.join(records)) if index==init_index else event,False))
    recovered=[event for event,added in tagged if not added]
    if ([row for row in recovered if row[0] not in {225,227}] != [row for row in original if row[0] not in {225,227}]
            or Counter(row for row in recovered if row[0]==227)!=Counter(row for row in original if row[0]==227)):
        raise ValueError('Drum Monkey import would change an original project record.')
    result=[event for event,_ in tagged]
    new_header=bytearray(header)
    struct.pack_into('<H',new_header,2,len(ids)+5)
    if read_bytes(encode_fl(bytes(new_header),result))!=(bytes(new_header),result):
        raise ValueError('The prepared Drum Monkey project failed its lossless container check.')
    after_blocks,_=_mixer(result)
    after_effects=_effects(result,after_blocks)
    lookup={(row['insert'],row['slot']):row for row in after_effects}
    if len(after_effects)!=len(original_effects)+4 or any(lookup[(row['insert'],row['slot'])]['record']!=row['record'] for row in original_effects):
        raise ValueError('An original mixer plugin changed during Drum Monkey preparation.')
    if source.read_bytes()!=preset_bytes:
        raise ValueError('The selected Drum Monkey preset changed during import. Select it again.')
    return bytes(new_header),result,{'channel_id':next_id,'name':name,'mixer_insert':29,'drums_insert':29,'kick_insert':30,
        'controllers':controllers,'preset_path':str(source),'preset_sha256':preset_sha,'kit_id':kit['id'],'kit':kit,
        'preset':kit['preset'],'plugin_state_sha256':_sha(plugin[213]),'wrapper_config_sha256':_sha(plugin[212]),
        'plugin_state_byte_identical':True,'wrapper_config_byte_identical':True,'manual_output_routing_required':True,
        'initial_state_changes':changes,'channel_count_before':len(ids),'channel_count_after':len(ids)+5,
        'instruments_added':1,'controllers_added':4,'effects_added':4,'links_added':4,'samples_added':0,
        'generator_cut_group':generator_cut[0],'generator_cut_by_group':generator_cut[1],
        'existing_instruments_and_effects_byte_identical':True,'existing_link_payloads_byte_identical':True,
        'native_FL_playback_verified':False,
        'warnings':['Drum Monkey uses its selected saved sounds and output settings. Route/verify its main drums to insert 29 and kick to insert 30 in FL Studio. Kick has no added effects or automation.']}
