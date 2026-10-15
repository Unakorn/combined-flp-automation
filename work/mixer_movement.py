"""Native mixer movement and saved effect-slot mix automation.

Parameter IDs/ranges: https://pyflp.readthedocs.io/en/latest/_modules/pyflp/mixer.html
Targets use the connector's native mixer event address convention. No plugin
parameter index is inferred from a third-party plugin's state.
"""
import math
import re
import struct
from collections import defaultdict

DEFAULTS = dict(volume=False, pan=False, stereo=False, existing_fx=False,
                depth=.15, rhythm='Phrase swells', inserts='', fx_slots='')
RHYTHMS = ('Phrase swells', 'Beat pump', 'Half-beat gate')

def validate(document=None):
    if document is not None and not isinstance(document, dict):
        raise ValueError('Mixer movement settings must be an object.')
    result = dict(DEFAULTS, **(document or {}))
    for key in ('volume', 'pan', 'stereo', 'existing_fx'):
        if type(result[key]) is not bool:
            raise ValueError(f'{key} must be enabled or disabled.')
    depth = result['depth']
    if isinstance(depth, bool) or not isinstance(depth, (int, float)) or not math.isfinite(depth) or not 0 <= depth <= .75:
        raise ValueError('Mixer movement depth must be between 0 and 75%.')
    if result['rhythm'] not in RHYTHMS:
        raise ValueError('Choose Phrase swells, Beat pump or Half-beat gate.')
    for key, pattern in (('inserts', r'[1-9]\d?'), ('fx_slots', r'[1-9]\d?:([1-9]|10)')):
        text = result[key]
        if not isinstance(text, str): raise ValueError(f'{key} must be text.')
        tokens = [t.strip() for t in text.split(',') if t.strip()]
        if any(not re.fullmatch(pattern, t) or int(t.split(':')[0]) > 63 for t in tokens):
            raise ValueError('Use inserts 1–63, or effect targets such as 4:2 (insert 4, slot 2).')
        result[key] = ','.join(dict.fromkeys(tokens))
    return {key: result[key] for key in DEFAULTS}

def candidates(loaded, rows):
    """Find destinations, retaining saved baselines and all existing ownership."""
    events, channels, blocks, init_index, effects = loaded[3], loaded[5], loaded[7], loaded[8], loaded[9]
    payload = events[init_index][1]
    if len(payload) % 12: raise ValueError('Unsupported mixer initialization.')
    values = {target: value for _, target, value in struct.iter_unpack('<III', payload)}
    counts = defaultdict(int)
    for _, target, _ in struct.iter_unpack('<III', payload): counts[target] += 1
    owned = set(); unknown = False
    for key, data in events:
        if key in (226, 227):
            if len(data) != 20: unknown = True
            else: owned.add(struct.unpack_from('<I', data, 8)[0])
        elif key == 223 and data:
            if len(data) % 12: unknown = True
            else: owned.update(target for _, target, _ in struct.iter_unpack('<III', data))
        elif key == 21 and data == b'\x05':
            # Native automation-clip parameter maps are not parsed here.
            unknown = True
    warnings=[]
    if unknown:
        return [], ['Additional mixer movement skipped: an existing automation format cannot be checked safely.']
    groups=defaultdict(list)
    for row in rows:
        if not row.get('kick') and not row.get('drum_monkey'):
            insert=row['mixer_insert']
            if 1 <= insert < min(64, len(blocks)-1): groups[insert].append(row)
    result=[]
    for insert, group in groups.items():
        # Do not modulate the master, reserved kick bus or a shared kick/sub bus.
        if insert in (29,30): continue
        sources=[c for c in channels if c['mixer_insert']==insert and c['type'] in (0,2,4)]
        if any(re.search(r'kick|\bkck\b', c['name'], re.I) for c in sources):
            warnings.append(f'Insert {insert}: additional mixer movement skipped because it contains a kick.')
            continue
        incoming=any(len(dict(b).get(235,b''))>insert and dict(b)[235][insert] for _,_,b in blocks)
        if incoming:
            warnings.append(f'Insert {insert}: additional movement skipped because another mixer insert feeds this bus.')
            continue
        bass=any(re.search(r'bass|sub|808', r['name'], re.I) for r in group)
        prefix=0x70000000 | insert<<22
        definitions=[('volume', prefix|0x1fc0, 12800, 0, 16000)]
        if not bass: definitions += [('pan',prefix|0x1fc1,0,-6400,6400), ('stereo',prefix|0x1fc2,0,-64,64)]
        for fx in effects:
            if fx['insert']!=insert: continue
            slot_prefix=prefix | fx['slot']<<16
            # Never wake a bypassed or unknown effect, and require its saved mix.
            target=slot_prefix|0x1f01
            if values.get(slot_prefix|0x1f00)!=1 or target not in values: continue
            definitions.append(('existing_fx',target,values[target],0,12800))
        for lane,target,default,low,high in definitions:
            raw=values.get(target,default)
            if low<0 and raw>=2**31: raw-=2**32
            if target in owned or counts[target]>1 or not low<=raw<=high:
                warnings.append(f'Insert {insert}: {lane} skipped because its destination is already automated or unsupported.')
                continue
            item=dict(lane=lane,target=target,mixer_insert=insert,
                      instrument_ids=[r['id'] for r in group], name=' + '.join(r['name'] for r in group),
                      baseline=(raw-low)/(high-low), saved_value=values.get(target,default), bass=bass)
            if lane=='existing_fx':
                item['effect_slot']=(target>>16)&63
                fx=next(e for e in effects if e['insert']==insert and e['slot']==item['effect_slot'])
                item['plugin']=fx.get('name') or fx['plugin']
            result.append(item)
    return result,warnings

def select(items, settings):
    inserts={int(t) for t in settings['inserts'].split(',') if t}
    slots={tuple(map(int,t.split(':'))) for t in settings['fx_slots'].split(',') if t}
    selected=[x for x in items if settings[x['lane']]
              and (not inserts or x['mixer_insert'] in inserts)
              and (x['lane']!='existing_fx' or not slots or (x['mixer_insert'],x['effect_slot']+1) in slots)]
    if slots and settings['existing_fx']:
        available={(x['mixer_insert'],x['effect_slot']+1) for x in selected if x['lane']=='existing_fx'}
        if slots-available: raise ValueError(f'Effect targets are unavailable or already automated: {sorted(slots-available)}')
    return selected

def anchors(item, plan, notes, settings, intensity=1, energy_curve=None):
    """Explicit straight ramps; no positive gain boosts; return to saved mix."""
    ppq,end=plan['ppq'],plan['end_tick']; bar=4*ppq
    baseline=item['baseline']; lane=item['lane']
    depth=min(.75,settings['depth']*min(2,max(0,intensity)))
    if item['bass']: depth=min(depth,.10)
    spans=[(n.tick,min(end,n.tick+n.duration)) for n in notes if n.duration>0]
    points={0:baseline,end-1:baseline}
    if depth==0:return sorted(points.items())
    period=bar*4 if settings['rhythm']=='Phrase swells' else (ppq if settings['rhythm']=='Beat pump' else max(2,ppq//2))
    def put(t,value):
        if 0<t<end-1:points[t]=max(1/128,min(1,value))
    base_depth = depth
    periods = list(range(0,end,period))
    if energy_curve:
        from energy_curve import sample_energy_curve
        energies = sample_energy_curve(energy_curve, periods)
    else:
        energies = [1.0] * len(periods)
    for start, energy in zip(periods, energies):
        depth = base_depth * energy
        stop=min(end-1,start+period)
        if not any(a<stop and b>start for a,b in spans): continue
        if lane=='pan':
            direction=1 if (start//period)%2==0 else -1
            put(start,baseline); put(start+period//2,baseline+direction*depth/2); put(stop,baseline)
        elif lane=='stereo':
            # Narrow at the start; reopen to the saved width, never exaggerate it.
            put(start,baseline+depth/2); put(start+period*3//4,baseline); put(stop,baseline)
        else:
            low=baseline*(1-depth)
            put(start,low); put(start+max(1,period//4),low if settings['rhythm']=='Half-beat gate' else baseline)
            put(start+period*3//4,baseline); put(stop,baseline)
    return sorted(points.items())
