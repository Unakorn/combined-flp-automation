"""Import host-saved plugin instances and real Automation Clip target links.

No VST parameter index is guessed. A local calibration FLP supplies a native
link and four points (saved baseline, lower bound, upper bound, baseline) for
each supported control. Plugin binaries, activation data and audio are never
bundled. These local resources must be verified in the installed FL host.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile

from drum_template import _mixer
from flp_song_analysis import read_project, _plugin_label
from stem_native import parse_points

from combined_settings import template_path

DEFAULT_TEMPLATE = template_path()
FAMILIES = {
    'filter': {'label': 'Waves F6 — LP frequency', 'tokens': ('f6',), 'clip': 'OA Filter'},
    'delay': {'label': 'Waves H-Delay — Dry/Wet', 'tokens': ('h-delay', 'hdelay'), 'clip': 'OA Delay'},
    'reverb': {'label': 'Waves RVerb — Wet/Dry', 'tokens': ('rverb', 'renaissance reverb'), 'clip': 'OA Reverb'},
    'width': {'label': 'Auto Shape Stereo — Width', 'tokens': ('auto shape stereo', 'autoshapestereo'), 'clip': 'OA Width'},
}
PLUGIN_KEYS = {201, 212, 203, 155, 128, 41, 213}


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def spec_digest(spec):
    return _sha(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode())


def _text(raw):
    return raw.decode('utf-16-le', errors='replace').rstrip('\0')


def plugin_records(project):
    """Return exact effect records, preserving opaque wrapper/plugin state."""
    events = project['events']
    blocks, _ = _mixer(events)
    records = {}
    for ordinal, (start, stop, _) in enumerate(blocks):
        at = start
        while at < stop:
            if events[at][0] != 201:
                at += 1
                continue
            first = at
            while at < stop and events[at][0] in PLUGIN_KEYS:
                at += 1
            rows = events[first:at]
            fields = dict(rows)
            if len(rows) != len(fields) or len(fields.get(212, b'')) != 52 or at >= stop or events[at][0] != 98:
                raise ValueError('An effect uses an unsupported wrapper record.')
            insert, slot = struct.unpack_from('<II', fields[212])
            if slot != int.from_bytes(events[at][1], 'little') or not 0 <= slot < 10:
                raise ValueError('An effect has inconsistent mixer addressing.')
            if insert != ordinal and not (ordinal == len(blocks)-1 and insert == 501):
                raise ValueError('An effect has an inconsistent insert address.')
            wrapper = _text(fields[201])
            label = _plugin_label(wrapper, fields.get(213, b''))
            records[(insert, slot)] = dict(events=rows, fields=fields, wrapper=wrapper,
                                           plugin_label=label, name=_text(fields.get(203, fields[201])))
    return records


def _initializer_rows(events, event_id):
    rows = []
    for key, payload in events:
        if key != event_id:
            continue
        if len(payload) % 12:
            raise ValueError('Unsupported parameter initialization layout.')
        rows.extend(struct.iter_unpack('<III', payload))
    return rows


def _parameter_initializer(target, native_rows, mixer_rows, name):
    # FL saves initialized plugin knobs in event216. Older fixtures/templates
    # used event225; accept that fallback only when it is unambiguous.
    native = [value for _, address, value in native_rows if address == target]
    legacy = [value for _, address, value in mixer_rows if address == target]
    if len(native) > 1 or len(legacy) > 1 or (native and legacy and native != legacy):
        raise ValueError(f'{name}: conflicting or duplicate parameter initializers.')
    values = native or legacy
    if not values:
        raise ValueError(f'{name}: initialize the knob at the baseline, save, then import again.')
    return values[0]


def _validate_parameter_baseline(raw, baseline):
    initialized = struct.unpack('<f', struct.pack('<I', raw))[0]
    if not math.isfinite(initialized) or not 0 <= initialized <= 1:
        raise ValueError('The saved parameter initializer must be a normalized finite value.')
    expected = struct.unpack('<I', struct.pack('<f', baseline))[0]
    # Envelope anchors are float64 but native knob initializers are float32.
    # Allow one float32 step around the rounded anchor, not a broad epsilon
    # that would accidentally accept a nonzero initializer at a dry baseline.
    if initialized != baseline and abs(raw - expected) > 1:
        raise ValueError('The saved parameter initializer does not match the calibration baseline.')


def _validate_spec(family, spec):
    if family not in FAMILIES or not isinstance(spec, dict) or spec.get('family') != family:
        raise ValueError('Unknown plugin template family.')
    try:
        rows = [(int(k), bytes.fromhex(p)) for k, p in spec['events']]
        fields = dict(rows)
        low, base, high = (float(spec[k]) for k in ('minimum', 'baseline', 'maximum'))
        address = spec['target_offset']
    except (KeyError, ValueError, TypeError):
        raise ValueError(f'{family}: malformed plugin template.') from None
    if (len(rows) != len(fields) or not rows or rows[0][0] != 201
            or any(k not in PLUGIN_KEYS for k, _ in rows)
            or len(fields.get(212, b'')) != 52 or not 0 < len(fields.get(213, b'')) < 8*1024*1024
            or fields.get(201) != 'Fruity Wrapper\0'.encode('utf-16-le')):
        raise ValueError(f'{family}: unsupported third-party wrapper format.')
    if _sha(fields[213]) != spec.get('state_sha256'):
        raise ValueError(f'{family}: plugin state integrity check failed.')
    if type(address) is not int or not 0x8000 <= address <= 0xffff:
        raise ValueError(f'{family}: template must link an internal plugin control, not a mixer blend.')
    if not all(math.isfinite(v) for v in (low, base, high)) or not 0 <= low <= base <= high <= 1 or low == high:
        raise ValueError(f'{family}: invalid calibrated control bounds.')
    if family == 'filter' and abs(base-high) > 1e-7:
        raise ValueError('Filter template baseline must be its upper (open) bound.')
    if family in ('delay', 'reverb') and abs(base-low) > 1e-7:
        raise ValueError(f'{family}: start the template at its lower (dry) bound.')
    if family == 'width' and not low < base < high:
        raise ValueError('Width needs a baseline between its lower and upper bounds.')
    if spec.get('initial_mix_raw') != 12800 or spec.get('initial_enabled_raw') != 1:
        raise ValueError('Templates need their effects enabled at full host slot mix.')
    if type(spec.get('initial_parameter_raw')) is not int or not 0 <= spec['initial_parameter_raw'] <= 0xffffffff:
        raise ValueError('Template needs the host-saved parameter initializer.')
    _validate_parameter_baseline(spec['initial_parameter_raw'], base)
    return rows


def load_template(path=None):
    path = Path(path or DEFAULT_TEMPLATE)
    if not path.is_file():
        return {'version': 1, 'families': {}}
    if path.stat().st_size > 40*1024*1024:
        raise ValueError('Plugin template exceeds the size limit.')
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('version') != 1 or not isinstance(data.get('families'), dict):
        raise ValueError('Unsupported plugin-template version.')
    for family, spec in data['families'].items():
        _validate_spec(family, spec)
    return data


def template_status(path=None):
    path = Path(path or DEFAULT_TEMPLATE)
    try:
        data = load_template(path)
        reason = ''
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        data, reason = {'families': {}}, str(exc)
    return {'path': str(path), 'families': {
        family: {'ready': family in data['families'], 'label': entry['label'],
                 'reason': reason or ('' if family in data['families'] else 'Import an FL-saved control template once.')}
        for family, entry in FAMILIES.items()}}


def import_template(path, out_path=None):
    project = read_project(path)
    records = plugin_records(project)
    events = project['events']
    native_initial_rows = _initializer_rows(events, 216)
    initial_rows = _initializer_rows(events, 225)
    links = {}
    for key, payload in events:
        if key != 227:
            continue
        if len(payload) != 20:
            raise ValueError('Unsupported native control link.')
        source, extra, target, kind, flags = struct.unpack('<5I', payload)
        if source & 0xffff or extra != 0 or kind != 8 or flags != 469:
            continue  # Never reinterpret non-native or transformed sources.
        links.setdefault(source >> 16, []).append(target)
    result = deepcopy(load_template(out_path))
    imported = []
    for channel in project['channels']:
        family = next((k for k, v in FAMILIES.items()
                       if channel['name'].strip().casefold() == v['clip'].casefold()), None)
        if family is None:
            continue
        if family in imported:
            raise ValueError(f'Template contains duplicate {FAMILIES[family]["clip"]} clips.')
        if channel['type'] != 5:
            raise ValueError(f'{channel["name"]} must be a native Automation Clip.')
        block = dict(events[channel['start']:channel['stop']])
        if block.get(219,b'')[:8] != struct.pack('<II',0,12800):
            raise ValueError(f'{channel["name"]}: reset the Automation Clip Min/Max to the full 0–1 range.')
        envelope = block.get(234,b'')
        # Only these stock disabled headers have been observed. Field4 is an
        # int32 amount; byte8 has no verified general flag interpretation.
        if (len(envelope) < 9 or struct.unpack_from('<i', envelope, 4)[0] not in (0, 64)
                or envelope[8] != 0):
            raise ValueError(f'{channel["name"]}: unsupported Automation Clip LFO header; disable LFO before calibration.')
        points = parse_points(envelope, project['ppq'])
        if len(points) != 4:
            raise ValueError(f'{channel["name"]}: use exactly four points: baseline, lower bound, upper bound, baseline.')
        base, low, high, final = [v for _, v in points]
        if abs(base-final) > 1e-7:
            raise ValueError(f'{channel["name"]}: last point must return to baseline.')
        targets = links.get(channel['id'], [])
        if len(targets) != 1:
            raise ValueError(f'{channel["name"]}: link exactly one plugin parameter.')
        target = targets[0]
        if target & 0xf0000000 != 0x70000000:
            raise ValueError('Template target is not a mixer effect.')
        insert, slot, offset = (target >> 22) & 0x3f, (target >> 16) & 0x3f, target & 0xffff
        record = records.get((insert, slot))
        if record is None or not 1 <= insert <= 63:
            raise ValueError('Template points to an unavailable dedicated effect.')
        fields = record['fields']
        # The host-saved component identity and state, not a displayed slot name,
        # must contain the expected product identity. Preserve WaveShell data.
        identity = (record['plugin_label']+' '+fields.get(213,b'').decode('utf-8',errors='ignore')
                    +' '+fields.get(213,b'').decode('utf-16-le',errors='ignore')).casefold()
        if not any(re.search(r'(?<![a-z0-9])'+re.escape(token)+r'(?![a-z0-9])',identity)
                   for token in FAMILIES[family]['tokens']):
            raise ValueError(f'{channel["name"]}: the saved plugin identity does not match {FAMILIES[family]["label"]}.')
        parameter_raw = _parameter_initializer(target, native_initial_rows, initial_rows, channel['name'])
        prefix = target & 0xffff0000
        mix = [v for _,t,v in initial_rows if t == prefix | 0x1f01]
        enabled = [v for _,t,v in initial_rows if t == prefix | 0x1f00]
        if mix and mix != [12800] or enabled and enabled != [1]:
            raise ValueError('Set each template effect enabled with host slot mix at 100%.')
        spec = dict(family=family,label=FAMILIES[family]['label'],plugin='Fruity Wrapper',
                    plugin_label=record['plugin_label'],events=[(k,p.hex()) for k,p in record['events']],
                    state_sha256=_sha(fields[213]),target_offset=offset,minimum=low,baseline=base,maximum=high,
                    initial_parameter_raw=parameter_raw,initial_mix_raw=12800,initial_enabled_raw=1,
                    source_fl_version=project['version'],source_sha256=project['sha256'],
                    binding='FL-saved native link; endpoint choices supplied by calibration project',
                    native_playback_verified=False)
        _validate_spec(family, spec)
        result['families'][family] = spec
        imported.append(family)
    if not imported:
        raise ValueError('No calibration clips found. Name them OA Filter, OA Delay, OA Reverb and/or OA Width.')
    if Path(project['path']).read_bytes() != project['data']:
        raise ValueError('Calibration project changed while importing; save and import it again.')
    output = Path(out_path or DEFAULT_TEMPLATE)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=output.parent, suffix='.tmp', delete=False) as stream:
        json.dump(result,stream,indent=2); stream.write('\n'); temporary = Path(stream.name)
    try:
        os.replace(temporary,output)
    finally:
        temporary.unlink(missing_ok=True)
    return template_status(output)


def build_plugin_record(spec, insert, slot):
    rows = _validate_spec(spec['family'], spec)
    if type(insert) is not int or not 1 <= insert <= 63 or type(slot) is not int or not 0 <= slot <= 9:
        raise ValueError('Plugin destination is outside the supported mixer range.')
    output = []
    for key,payload in rows:
        if key == 212:
            payload = bytearray(payload)
            struct.pack_into('<II',payload,0,insert,slot)
            flags = struct.unpack_from('<I',payload,16)[0]
            struct.pack_into('<I',payload,16,flags & ~1)
            payload = bytes(payload)
        output.append((key,payload))
    return output


def normalized_depth(spec, depth):
    if isinstance(depth,bool) or not isinstance(depth,(int,float)) or not math.isfinite(depth) or not -1 <= depth <= 1:
        raise ValueError('Automation depth must be finite and within -1..1.')
    base = spec['baseline']
    return base + (spec['maximum']-base)*depth if depth >= 0 else base+(base-spec['minimum'])*depth
