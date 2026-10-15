"""Prepare neutral Pro-Q 3 slots for a NEW combined-automation FLP.

A user's FL-saved Pro-Q record supplies its wrapper and five Bell-band settings.
Only the first two bands' Used flags and five gain fields can change in an
in-memory candidate. The installed, hash-pinned controller must confirm all
383 processing parameter readouts, neutrality and the permitted changes before
any new FLP is published. No calibration project or plugin binary is bundled.
Existing-plugin EQ automation never rewrites its opaque state.
"""
from __future__ import annotations

import base64
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile

if len(sys.argv) >= 3 and sys.argv[1] == '--probe-full-child' and not sys.argv[2].startswith('--'):
    sys.path.insert(0, sys.argv[2])

import eq_vst3_probe
from eq_vst3_probe import (COMPONENT_CLASS, CONTROLLER_CLASS, GAIN_IDS,
                           PINNED_PLUGIN_SHA256)
from eq_project import BANDS, _runtime_band, inspect_eq_stems, read_eq_calibration
from flp_raw import encode_fl, read_bytes
from flp_song_analysis import read_project
from plugin_templates import _initializer_rows, plugin_records
from drum_template import _mixer
from stem_project import _free_effect_slot, _targets

# The saved-state layout and binary remain pinned, while each user's own
# calibration is verified through the installed controller before copying.
FULL_PARAMETER_COUNT = 383  # 24 complete bands, globals and host bypass.
NEUTRAL_FIELDS = {0: 0.0, 13: 0.0, 29: 0.0, 42: 0.0,
                  55: 0.0, 68: 0.0, 81: 0.0}
EXPECTED_CHANGED_PARAMETERS = {0, 13, 29, 42, 55, 68, 81}


def _sha(value):
    return hashlib.sha256(Path(value).read_bytes()).hexdigest()


def _validate_full_result(result, request):
    short = dict(result, rows=[dict(row, parameters=row['parameters'][:91])
                               for row in result.get('rows', [])])
    eq_vst3_probe._validate_result(short, request)
    metadata = result['parameters']
    for row in result['rows']:
        values = row['parameters']
        if len(values) != FULL_PARAMETER_COUNT:
            raise ValueError('Full Pro-Q processing readout is incomplete.')
        for index, value in enumerate(values):
            if (value['index'] != index or value['id'] != metadata[index]['id']
                    or value['title'] != metadata[index]['title'] or value['result'] != 0
                    or not all(math.isfinite(value[k]) for k in ('normalized', 'plain'))
                    or not 0 <= value['normalized'] <= 1):
                raise ValueError('Full Pro-Q readout identity or value is invalid.')
    return result


def _probe_request(request, timeout=30):
    """Isolate the identical controller ABI, with a broader read-only readout."""
    if not 1 <= timeout <= 60:
        raise ValueError('Controller timeout must be 1–60 seconds.')
    result = eq_vst3_probe._isolated_probe(
        request, '--probe-full-child', Path(__file__).resolve(),
        extra_args=(Path(eq_vst3_probe.__file__).resolve().parent,),
        timeout=timeout, max_response_bytes=16 * 1024 * 1024)
    result = _validate_full_result(result, request)
    if (_sha(request['source']) != request['source_sha256']
            or _sha(request['plugin_path']) != request['plugin_sha256']):
        raise ValueError('The donor/project or plugin changed during inspection.')
    return result


def probe_full_project(source):
    return _probe_request(eq_vst3_probe._extract_request(source))


def validate_neutral_row(row):
    """Require five neutral Bell bands and no additional processing."""
    params = {p['index']: p for p in row['parameters']}
    if set(params) != set(range(FULL_PARAMETER_COUNT)):
        raise ValueError('Neutrality needs every band and global readout.')
    for band in range(1, 25):
        first = (band - 1) * 13
        if params[first]['title'] != f'Band {band} Used':
            raise ValueError('Band addressing is not the verified Pro-Q mapping.')
        if band in BANDS:
            actual = _runtime_band(row, band)
            if actual['baseline'] != 0.5 or actual['baseline_db'] != 0.0:
                raise ValueError('The prepared Bell bands must be exactly 0 dB.')
        elif params[first]['normalized'] != 0:
            raise ValueError('Additional Pro-Q bands must be unused.')
        if params[first + 12]['normalized'] != 0:
            raise ValueError('Every Pro-Q band must have Solo disabled.')
    required = {
        312: ('Processing Mode', 0.0),
        314: ('Gain Scale', 0.5),
        315: ('Output Level', 0.5),
        316: ('Output Pan', 0.5),
        317: ('Output Pan Mode', 0.0),
        318: ('Bypass', 0.0),
        319: ('Output Invert Phase', 0.0),
        320: ('Auto Gain', 0.0),
        332: ('Receive Midi', 0.0),
        333: ('Solo Gain', 0.5),
        382: ('Host Bypass', 0.0),
    }
    for index, (title, normalized) in required.items():
        if params[index]['title'] != title or params[index]['normalized'] != normalized:
            raise ValueError(f'Pro-Q global {title} is not neutral.')
    if params[315]['plain'] != 0 or params[316]['plain'] != 0:
        raise ValueError('Output gain and pan must read exactly zero.')
    for index in range(334, 382):
        if params[index]['normalized'] != 0:
            raise ValueError('External side-chain/speaker extensions must be disabled.')
    return [_runtime_band(row, band) for band in BANDS]


def _component_bounds(wrapper):
    # The existing extractor independently checks wrapper chunks, binary class,
    # size, FFBS count/version and its complete trailer before this is reached.
    matches = [m.start() for m in re.finditer(b'FFBS', wrapper)]
    if len(matches) != 1:
        raise ValueError('Ambiguous Pro-Q component state.')
    at = matches[0]
    if at < 12 or struct.unpack_from('<II', wrapper, at + 4) != (1, 358):
        raise ValueError('Unsupported Pro-Q component state format.')
    kind, length = struct.unpack_from('<IQ', wrapper, at - 12)
    if kind != 3 or length != 12 + 358 * 4 + 12:
        raise ValueError('Unsupported Pro-Q component-state bounds.')
    if wrapper[at + length - 12:at + length] != b'FFpr\x01\0\0\0\0\0\0\0':
        raise ValueError('Incomplete Pro-Q component-state trailer.')
    return at, at + length


def _neutral_candidate(wrapper):
    chunks = eq_vst3_probe._wrapper_chunks(wrapper)
    if chunks.get(52, b'').hex() != COMPONENT_CLASS:
        raise ValueError('The calibration is not the supported Pro-Q 3 VST3 component.')
    at, end = _component_bounds(wrapper)
    candidate = bytearray(wrapper)
    permitted = set()
    for index, value in NEUTRAL_FIELDS.items():
        offset = at + 12 + 4 * index
        struct.pack_into('<f', candidate, offset, value)
        permitted.update(range(offset, offset + 4))
    changed = {i for i, (a, b) in enumerate(zip(wrapper, candidate)) if a != b}
    if not changed <= permitted or len(wrapper) != len(candidate):
        raise ValueError('Neutral candidate altered bytes outside the seven fields.')
    return bytes(candidate), bytes(candidate[at:end])


def _verify_intended_changes(before, after):
    changed = set()
    for a, b in zip(before['parameters'], after['parameters']):
        if any(a[k] != b[k] for k in ('index', 'id', 'title')):
            raise ValueError('Pro-Q candidate parameter identity changed.')
        if a['normalized'] != b['normalized'] or a['plain'] != b['plain']:
            changed.add(a['index'])
    if not changed <= EXPECTED_CHANGED_PARAMETERS:
        raise ValueError('Pro-Q candidate changed an unrelated parameter.')
    for index in EXPECTED_CHANGED_PARAMETERS:
        expected = 0.0 if index in (0, 13) else 0.5
        if after['parameters'][index]['normalized'] != expected:
            raise ValueError('Pro-Q candidate did not set the intended parameter.')
    return sorted(changed)


def _clone_record(record, wrapper, insert, slot):
    cloned = []
    for key, payload in record['events']:
        if key == 212:
            payload = bytearray(payload)
            struct.pack_into('<II', payload, 0, insert, slot)
            flags = struct.unpack_from('<I', payload, 16)[0]
            struct.pack_into('<I', payload, 16, flags & ~1)
            payload = bytes(payload)
        elif key == 213:
            payload = wrapper
        cloned.append((key, payload))
    return cloned


def _select_donor(anchor, records, donor_insert=None, donor_slot=None):
    """Choose the calibrated instance from its real native FL gain link."""
    anchored = {(band['insert'], band['slot']) for band in anchor['bands'].values()}
    if donor_insert is None and donor_slot is None:
        if len(anchored) != 1:
            raise ValueError('Use a calibration FLP with gain automation clips linked to one Pro-Q instance.')
        selected = next(iter(anchored))
    elif donor_insert is None or donor_slot is None:
        raise ValueError('Select both calibration insert and slot, or leave both automatic.')
    else:
        selected = donor_insert, donor_slot
        if selected not in anchored:
            raise ValueError('The selected Pro-Q instance has no native gain automation anchor.')
    record = records.get(selected)
    if not record or record['plugin_label'].casefold() != 'fabfilter pro-q 3':
        raise ValueError('The calibration gain link must target a Pro-Q 3 instance.')
    return selected, record


def prepare_eq_copy(source, output, calibration_path, *, donor_insert=None, donor_slot=None):
    """Return prepared_path, calibration_path and evidence; never overwrite."""
    source, output, calibration_path = map(lambda p: Path(p).expanduser().resolve(),
                                          (source, output, calibration_path))
    if output.exists() or output in (source, calibration_path):
        raise FileExistsError(f'Choose a new EQ-preparation output: {output}')
    if output.suffix.lower() != '.flp':
        raise ValueError('EQ setup output must be a new .flp file.')
    project = read_project(source)
    stems = inspect_eq_stems(source)
    if project['sha256'] != stems['sha256']:
        raise ValueError('Stem project changed during setup.')
    donor = read_project(calibration_path)
    anchor = read_eq_calibration(calibration_path)
    if anchor['sha256'] != donor['sha256']:
        raise ValueError('Calibration project changed during setup.')
    records = plugin_records(donor)
    (donor_insert, donor_slot), record = _select_donor(anchor, records, donor_insert, donor_slot)
    request = eq_vst3_probe._extract_request(calibration_path)
    request['rows'] = [r for r in request['rows']
                       if (r['insert'], r['slot']) == (donor_insert, donor_slot)]
    if len(request['rows']) != 1:
        raise ValueError('The donor is not one verified Pro-Q instance.')
    original = _probe_request(request)['rows'][0]
    wrapper, component = _neutral_candidate(record['fields'][213])
    candidate_request = deepcopy(request)
    candidate_request['rows'][0].update(
        wrapper_sha256=hashlib.sha256(wrapper).hexdigest(),
        component_state=base64.b64encode(component).decode('ascii'))
    candidate = _probe_request(candidate_request)['rows'][0]
    changed_parameters = _verify_intended_changes(original, candidate)
    bands = validate_neutral_row(candidate)

    events = project['events']
    blocks, _, counts, owned, effects = _targets(project)
    _, mix_index = _mixer(events)
    initial = _initializer_rows(events, 216)
    reserved_targets = owned | {target for _, target, _ in initial}
    existing_records = plugin_records(project)
    assignments, additions, changes, init_values = [], defaultdict(list), {}, {}
    for stem in stems['stems']:
        insert = stem['mixer_insert']
        if not 1 <= insert <= 63:
            raise ValueError('Neutral EQ setup needs dedicated stem inserts 1–63.')
        if any(route == insert and r['plugin_label'].casefold() == 'fabfilter pro-q 3'
               for (route, _), r in existing_records.items()):
            raise ValueError('A stem already has Pro-Q; preserve it and use existing-instance EQ analysis.')
        if any(len(dict(block).get(235, b'')) > insert and dict(block)[235][insert]
               for _, _, block in blocks):
            raise ValueError('A stem insert receives another mixer insert; dedicated routing is required.')
        slot = _free_effect_slot(blocks, effects, insert, set(), reserved_targets, counts)
        positions = [blocks[insert][0] + i for i, (key, payload) in enumerate(blocks[insert][2])
                     if key == 98 and int.from_bytes(payload, 'little') == slot]
        if len(positions) != 1:
            raise ValueError('Cannot locate one empty effect slot.')
        cloned = _clone_record(record, wrapper, insert, slot)
        additions[positions[0]].extend(cloned)
        prefix = 0x70000000 | insert << 22 | slot << 16
        init_values[prefix | 0x1f00] = 1
        init_values[prefix | 0x1f01] = 12800
        assignments.append(dict(stem_id=stem['id'], stem_name=stem['name'],
                                insert=insert, slot=slot))
    rows, seen = [], set()
    for first, target, value in struct.iter_unpack('<III', events[mix_index][1]):
        if target in init_values:
            value = init_values[target]
            seen.add(target)
        rows.append(struct.pack('<III', first, target, value))
    rows.extend(struct.pack('<III', 0, target, value)
                for target, value in sorted(init_values.items()) if target not in seen)
    changes[mix_index] = b''.join(rows)
    # Copies live in outputs/work, so resolve existing audio references without
    # importing any notes, audio or channel records from the donor project.
    by_id = {s['id']: s for s in stems['stems']}
    for channel in project['channels']:
        if not channel['sample']:
            continue
        stem = by_id.get(channel['id'])
        sample = Path(stem['sample']) if stem else Path(os.path.expandvars(channel['sample']))
        if not sample.is_absolute():
            sample = source.parent / sample
        if not sample.is_file():
            raise ValueError(f'Cannot resolve source audio: {channel["sample"]}')
        positions = [i for i in range(channel['start'], channel['stop']) if events[i][0] == 196]
        if len(positions) != 1:
            raise ValueError('Ambiguous source audio reference.')
        changes[positions[0]] = (str(sample.resolve()) + '\0').encode('utf-16-le')
    new_events = []
    for index, (key, payload) in enumerate(events):
        new_events.extend(additions.get(index, []))
        new_events.append((key, changes.get(index, payload)))
    header, _ = read_bytes(project['data'])
    data = encode_fl(header, new_events)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.flp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        prepared = read_project(temporary)
        if prepared['events'] != new_events or len(prepared['channels']) != len(project['channels']):
            raise ValueError('Prepared EQ copy did not round-trip exactly.')
        final_runtime = probe_full_project(temporary)
        expected_keys = {(a['insert'], a['slot']) for a in assignments}
        if {(r['insert'], r['slot']) for r in final_runtime['rows']} != expected_keys:
            raise ValueError('Prepared Pro-Q assignments differ from the plan.')
        for row in final_runtime['rows']:
            validate_neutral_row(row)
            if row['wrapper_sha256'] != candidate['wrapper_sha256']:
                raise ValueError('Prepared Pro-Q state differs from verified neutral template.')
        if _sha(source) != project['sha256'] or _sha(calibration_path) != donor['sha256']:
            raise ValueError('An input changed before publication.')
        with output.open('xb') as stream:
            stream.write(data)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return dict(prepared_path=str(output), output=str(output), calibration_path=str(calibration_path),
                source=str(source), source_sha256=project['sha256'], output_sha256=hashlib.sha256(data).hexdigest(),
                calibration_sha256=donor['sha256'], donor_insert=donor_insert, donor_slot=donor_slot,
                donor_wrapper_sha256=hashlib.sha256(record['fields'][213]).hexdigest(),
                neutral_wrapper_sha256=candidate['wrapper_sha256'], plugin_sha256=PINNED_PLUGIN_SHA256,
                changed_parameters=changed_parameters, bands=bands, effects=assignments,
                verification='All 24 bands and processing globals verified by installed Pro-Q controller; host playback requires review.')


# Existing verified controller ABI; only the readout extent is broadened.
def _probe_child_full(request):
    import ctypes as c
    import uuid
    P, I, U, D = c.c_void_p, c.c_int32, c.c_uint32, c.c_double
    FUN = c.WINFUNCTYPE
    class ClassInfo(c.Structure):
        _fields_ = [('cid', c.c_ubyte * 16), ('cardinality', I), ('category', c.c_char * 32), ('name', c.c_char * 64)]
    class ParamInfo(c.Structure):
        _fields_ = [('id', U), ('title', c.c_wchar * 128), ('short', c.c_wchar * 128), ('units', c.c_wchar * 128),
                    ('steps', I), ('default', D), ('unitid', I), ('flags', I)]
    class Obj(c.Structure):
        _fields_ = [('vtable', c.POINTER(P))]
    def uid(text):
        return uuid.UUID(text).bytes_le
    def call(obj, index, restype, *types):
        table = c.cast(obj, c.POINTER(c.POINTER(P))).contents
        return FUN(restype, P, *types)(table[index])
    no_interface = -2147467262
    references = []
    def callback(restype, types, fn):
        function = FUN(restype, *types)(fn)
        references.append(function)
        return c.cast(function, P).value
    unknown = uid('00000000-0000-0000-c000-000000000046')
    def query_for(interface):
        def query(self, iid, out):
            if c.string_at(iid, 16) in (unknown, interface):
                out[0] = self
                return 0
            out[0] = None
            return no_interface
        return query
    def base(interface):
        return [callback(I, [P, P, c.POINTER(P)], query_for(interface)),
                callback(U, [P], lambda _: 1), callback(U, [P], lambda _: 1)]
    def name(self, out):
        data = 'Open Automation EQ inspection\0'.encode('utf-16-le')
        c.memset(out, 0, 256)
        c.memmove(out, data, len(data))
        return 0
    def create(self, cid, iid, out):
        out[0] = None
        return no_interface
    host_table = (P * 5)(*base(uid('58e595cc-db2d-4969-8b6a-af8c36a664e5')),
                        callback(I, [P, P], name), callback(I, [P, P, P, c.POINTER(P)], create))
    host = Obj(host_table)
    handler_table = (P * 7)(*base(uid('93a0bea3-0bd0-45db-8e89-0b0cc1e46ac6')),
                           callback(I, [P, U], lambda *_: 0), callback(I, [P, U, D], lambda *_: 0),
                           callback(I, [P, U], lambda *_: 0), callback(I, [P, I], lambda *_: 0))
    handler = Obj(handler_table)
    buffer, position = b'', 0
    def read(self, dest, amount, read_count):
        nonlocal position
        if amount < 0:
            return -2147024809
        data = buffer[position:position + amount]
        c.memmove(dest, data, len(data))
        position += len(data)
        if read_count:
            read_count[0] = len(data)
        return 0 if len(data) == amount else 1
    def write(self, source, amount, written):
        if written:
            written[0] = 0
        return -2147467259
    def seek(self, offset, mode, result):
        nonlocal position
        target = offset if mode == 0 else position + offset if mode == 1 else len(buffer) + offset if mode == 2 else -1
        if target < 0:
            return -2147024809
        position = target
        if result:
            result[0] = position
        return 0
    def tell(self, result):
        result[0] = position
        return 0
    stream_table = (P * 7)(*base(uid('c3bf6ea2-3099-4752-9b6b-f9901ee33e9b')),
                          callback(I, [P, P, I, c.POINTER(I)], read), callback(I, [P, P, I, c.POINTER(I)], write),
                          callback(I, [P, c.c_int64, I, c.POINTER(c.c_int64)], seek), callback(I, [P, c.POINTER(c.c_int64)], tell))
    stream = Obj(stream_table)
    eq_vst3_probe._validate_plugin_binary(request['plugin_path'], request['plugin_sha256'])
    library = c.WinDLL(request['plugin_path'])
    library.GetPluginFactory.restype = P
    library.GetPluginFactory.argtypes = []
    factory = library.GetPluginFactory()
    if not factory:
        raise ValueError('Pro-Q factory is unavailable.')
    controller, initialized = P(), False
    try:
        classes = {}
        count = call(factory, 4, I)(factory)
        if not 1 <= count <= 32:
            raise ValueError('Unsupported Pro-Q factory class count.')
        for index in range(count):
            info = ClassInfo()
            if call(factory, 5, I, I, c.POINTER(ClassInfo))(factory, index, c.byref(info)) != 0:
                raise ValueError('Cannot read Pro-Q factory classes.')
            classes[bytes(info.cid).hex()] = info.category.decode()
        if classes.get(COMPONENT_CLASS) != 'Audio Module Class' or classes.get(CONTROLLER_CLASS) != 'Component Controller Class':
            raise ValueError('Installed Pro-Q component/controller identity differs from the project.')
        status = call(factory, 6, I, P, P, c.POINTER(P))(factory, c.c_char_p(bytes.fromhex(CONTROLLER_CLASS)),
                    c.c_char_p(uid('dcd7bbe3-7742-448d-a874-aacc979c759e')), c.byref(controller))
        if status != 0 or not controller:
            raise ValueError('Cannot create Pro-Q parameter controller.')
        if call(controller, 3, I, P)(controller, c.byref(host)) != 0:
            raise ValueError('Cannot initialize Pro-Q parameter controller.')
        initialized = True
        if call(controller, 16, I, P)(controller, c.byref(handler)) != 0:
            raise ValueError('Cannot connect Pro-Q inspection host.')
        count = call(controller, 8, I)(controller)
        if count != 519:
            raise ValueError('Pro-Q exposes an unverified parameter count.')
        parameters = []
        for index in range(count):
            info = ParamInfo()
            if call(controller, 9, I, I, c.POINTER(ParamInfo))(controller, index, c.byref(info)) != 0:
                raise ValueError('Cannot read Pro-Q parameter metadata.')
            parameters.append(dict(index=index, id=info.id, title=info.title, units=info.units,
                                   steps=info.steps, default=info.default, flags=info.flags))
        gain_probes = []
        for band, parameter in GAIN_IDS.items():
            values = [dict(normalized=value, plain=call(controller, 12, D, U, D)(controller, parameter, value))
                      for value in (0., .25, .45, .5, .55, .75, 1.)]
            gain_probes.append(dict(band=band, parameter_id=parameter, normalized_to_plain=values))
        rows = []
        for row in request['rows']:
            buffer = base64.b64decode(row['component_state'], validate=True)
            position = 0
            status = call(controller, 5, I, P)(controller, c.byref(stream))
            if status != 0 or position != len(buffer):
                raise ValueError('Pro-Q could not load the exact saved component state.')
            values = []
            for parameter in parameters[:FULL_PARAMETER_COUNT]:
                index, tag = parameter['index'], parameter['id']
                normalized = call(controller, 14, D, U)(controller, tag)
                plain = call(controller, 12, D, U, D)(controller, tag, normalized)
                display = (c.c_wchar * 128)()
                result = call(controller, 10, I, U, D, P)(controller, tag, normalized, display)
                values.append(dict(index=index, id=tag, title=parameter['title'], normalized=normalized,
                                   plain=plain, display=display.value, result=result))
            rows.append(dict(insert=row['insert'], slot=row['slot'], wrapper_sha256=row['wrapper_sha256'],
                             state_result=status, parameters=values))
        return dict(source_sha256=request['source_sha256'], plugin_path=request['plugin_path'],
                    plugin_sha256=request['plugin_sha256'], controller_class=CONTROLLER_CLASS, count=count,
                    parameters=parameters, gain_probes=gain_probes, rows=rows)
    finally:
        if controller:
            if initialized:
                call(controller, 4, I)(controller)
            call(controller, 2, U)(controller)
        call(factory, 2, U)(factory)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "--probe-full-child":
        raise SystemExit("Import prepare_eq_copy from this module.")
    raise SystemExit(eq_vst3_probe.worker_main(_probe_child_full))
