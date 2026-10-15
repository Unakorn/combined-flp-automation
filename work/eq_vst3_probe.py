"""Isolated, pinned Pro-Q 3 parameter and saved-state inspection.

Only a standalone VST3 controller is created: no editor, audio processor,
project edits, plugin-file writes, or preference edits are requested. The
controller receives saved component state in memory and returns parameter
metadata/readouts. A child-process timeout isolates plugin faults.

ABI references: Steinberg's official vst3_pluginterfaces headers
base/{funknown,ipluginbase,ibstream}.h and
vst/{ivsthostapplication,ivsteditcontroller}.h at
https://github.com/steinbergmedia/vst3_pluginterfaces
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

PINNED_PLUGIN_SHA256 = 'bdb70fc219cc62d507d79d81b60e5ab986dd0afb356a86a978c1df9ca1827430'
SUPPORTED_PLUGIN_VERSION = 'FabFilter Pro-Q 3.29 (Windows x64 VST3, file version 3.2.9.62)'
INSTALLED_PLUGIN_PATH = Path(os.environ.get('CommonProgramFiles', r'C:\Program Files\Common Files')) / 'VST3' / 'FabFilter Pro-Q 3.vst3'
COMPONENT_CLASS = '71dbc4724d7a9a45b97e51745d84b39d'
CONTROLLER_CLASS = '1337e39076cd1e4ab41b37ef1b34e74e'
GAIN_IDS = {3: 29, 4: 42, 5: 55, 6: 68, 7: 81}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_plugin_binary(path, expected_sha256=PINNED_PLUGIN_SHA256):
    """Accept the verified binary in any user's local installation folder."""
    path = Path(path).expanduser()
    if not path.is_absolute() or not path.is_file() or path.suffix.casefold() != '.vst3':
        raise ValueError('The saved Pro-Q VST3 binary is unavailable. Save the FLP on this computer after locating Pro-Q in FL Studio.')
    if expected_sha256 != PINNED_PLUGIN_SHA256 or _sha(path) != PINNED_PLUGIN_SHA256:
        raise ValueError('EQ currently supports ' + SUPPORTED_PLUGIN_VERSION + '. This binary has not been verified; volume and section processing remain available.')
    return path.resolve()


def _isolated_probe(request, worker_flag, worker_script, *, extra_args=(), timeout=20,
                    max_response_bytes=4 * 1024 * 1024):
    """Use temporary JSON files so a windowed frozen EXE needs no console."""
    paths = []
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.json', delete=False) as stream:
            request_path = Path(stream.name)
            paths.append(request_path)
            json.dump(request, stream)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as stream:
            response_path = Path(stream.name)
            paths.append(response_path)
        if getattr(sys, 'frozen', False):
            command = [sys.executable, worker_flag]
        else:
            command = [sys.executable, '-B', str(worker_script), worker_flag, *map(str, extra_args)]
        command += ['--request', str(request_path), '--response', str(response_path)]
        try:
            completed = subprocess.run(command, capture_output=True, timeout=timeout,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except subprocess.TimeoutExpired as exc:
            raise ValueError('Pro-Q parameter inspection timed out; no project was changed.') from exc
        if response_path.stat().st_size > max_response_bytes:
            raise ValueError('Pro-Q inspection returned an oversized response.')
        try:
            response = json.loads(response_path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ValueError('Pro-Q inspection failed in its isolated process or returned an invalid response.') from exc
        if completed.returncode or not isinstance(response, dict) or 'result' not in response:
            detail = response.get('error', '') if isinstance(response, dict) else ''
            raise ValueError('Pro-Q parameter inspection failed in its isolated process. ' + str(detail)[-500:])
        return response['result']
    finally:
        for path in paths:
            path.unlink(missing_ok=True)


def worker_main(callback):
    """Worker entry shared by Python scripts and the frozen app dispatcher."""
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--request', required=True)
    parser.add_argument('--response', required=True)
    args, _ = parser.parse_known_args(sys.argv[1:])
    status = 0
    try:
        request_path = Path(args.request)
        if request_path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError('Pro-Q inspection request is too large.')
        request = json.loads(request_path.read_text(encoding='utf-8'))
        response = {'result': callback(request)}
    except Exception as exc:
        response, status = {'error': str(exc)}, 1
    Path(args.response).write_text(json.dumps(response), encoding='utf-8')
    return status


def _wrapper_chunks(state):
    if state[:4] != struct.pack('<I', 12):
        raise ValueError('Unsupported Pro-Q wrapper version.')
    chunks, at = {}, 4
    while at < len(state):
        if at + 12 > len(state):
            raise ValueError('Incomplete Pro-Q wrapper chunk.')
        kind, size = struct.unpack_from('<IQ', state, at)
        at += 12
        if size > len(state) - at or kind in chunks:
            raise ValueError('Invalid or duplicate Pro-Q wrapper chunks.')
        chunks[kind] = state[at:at + size]
        at += size
    return chunks


def _extract_request(source):
    from flp_song_analysis import read_project
    from plugin_templates import plugin_records
    project = read_project(source)
    rows, paths = [], set()
    for (insert, slot), record in plugin_records(project).items():
        if record['plugin_label'].casefold() != 'fabfilter pro-q 3':
            continue
        state = record['fields'][213]
        chunks = _wrapper_chunks(state)
        if chunks.get(52, b'').hex() != COMPONENT_CLASS:
            raise ValueError('This Pro-Q component identity has not been verified.')
        raw_path = chunks.get(55, b'').rstrip(b'\0').decode('utf-8')
        path = Path(raw_path).expanduser()
        paths.add(str(_validate_plugin_binary(path)))
        opaque = chunks.get(53, b'')
        offsets = [m.start() for m in re.finditer(b'FFBS', opaque)]
        if len(offsets) != 1:
            raise ValueError('Unsupported saved Pro-Q component state.')
        at = offsets[0]
        if at < 12 or at + 12 > len(opaque):
            raise ValueError('Incomplete saved Pro-Q component state.')
        chunk_kind, length = struct.unpack_from('<IQ', opaque, at - 12)
        version, count = struct.unpack_from('<II', opaque, at + 4)
        # This exact FL26/VST3 layout was accepted by this installed plugin.
        # Fail closed on new formats instead of scanning past uncertain bounds.
        if chunk_kind != 3 or version != 1 or count != 358 or length != 12 + count * 4 + 12:
            raise ValueError('Saved Pro-Q state layout differs from the verified version.')
        end = at + length
        if end > len(opaque) or opaque[end - 12:end] != b'FFpr\x01\0\0\0\0\0\0\0':
            raise ValueError('Saved Pro-Q component state trailer is incomplete.')
        rows.append(dict(insert=insert, slot=slot, wrapper_sha256=hashlib.sha256(state).hexdigest(),
                         component_state=base64.b64encode(opaque[at:end]).decode('ascii')))
    if not rows or len(paths) != 1:
        raise ValueError('EQ inspection requires existing instances of one verified Pro-Q 3 VST3 binary.')
    plugin = paths.pop()
    return dict(source=str(Path(source).resolve()), source_sha256=project['sha256'],
                plugin_path=plugin, plugin_sha256=PINNED_PLUGIN_SHA256, rows=rows)


def _validate_result(result, request):
    if not isinstance(result, dict) or any(result.get(k) != request[k] for k in ('source_sha256', 'plugin_path', 'plugin_sha256')):
        raise ValueError('Pro-Q inspection response does not match its source and binary.')
    if result.get('controller_class') != CONTROLLER_CLASS or result.get('count') != 519:
        raise ValueError('Pro-Q exposes an unverified controller or parameter count.')
    params = result.get('parameters', [])
    if len(params) != 519 or len({p['index'] for p in params}) != 519:
        raise ValueError('Incomplete Pro-Q parameter metadata.')
    by_index = {p['index']: p for p in params}
    for band, index in GAIN_IDS.items():
        info = by_index.get(index, {})
        if info.get('id') != index or info.get('title') != f'Band {band} Gain' or info.get('steps') != 0:
            raise ValueError('Pro-Q gain parameter addressing differs from the verified mapping.')
    probes = result.get('gain_probes', [])
    if {p.get('band') for p in probes} != set(GAIN_IDS):
        raise ValueError('Pro-Q gain conversions were not verified for all five bands.')
    for probe in probes:
        values = probe.get('normalized_to_plain', [])
        if len(values) != 7 or [v['normalized'] for v in values] != [0., .25, .45, .5, .55, .75, 1.]:
            raise ValueError('Pro-Q gain conversion samples are incomplete.')
        if any(not math.isfinite(v['plain']) or abs(v['plain'] - (v['normalized'] * 60 - 30)) > 1e-4 for v in values):
            raise ValueError('Pro-Q gain normalization is not the verified -30 to +30 dB mapping.')
    expected = {(r['insert'], r['slot']): r['wrapper_sha256'] for r in request['rows']}
    rows = result.get('rows', [])
    if len(rows) != len(expected) or {(r['insert'], r['slot']): r.get('wrapper_sha256') for r in rows} != expected:
        raise ValueError('Saved Pro-Q state inspection is incomplete or stale.')
    for row in rows:
        if row.get('state_result') != 0 or len(row.get('parameters', [])) != 91:
            raise ValueError('The installed Pro-Q controller could not read a saved state.')
        if {p['index'] for p in row['parameters']} != set(range(91)):
            raise ValueError('Saved Pro-Q band readouts are incomplete.')
        for p in row['parameters']:
            if p['result'] != 0 or p['id'] != p['index'] or p['title'] != by_index[p['index']]['title']:
                raise ValueError('Saved Pro-Q band readout identity differs from metadata.')
            if not all(isinstance(p[k], (float, int)) and math.isfinite(p[k]) for k in ('plain', 'normalized')) or not 0 <= p['normalized'] <= 1:
                raise ValueError('Saved Pro-Q band values are invalid.')
    return result


def probe_project(source, timeout=20):
    """Return exact plugin-supplied metadata/readouts or a descriptive error."""
    if os.name != 'nt':
        raise ValueError('Installed Pro-Q inspection is available on this Windows setup only.')
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= timeout <= 60:
        raise ValueError('Pro-Q inspection timeout must be 1–60 seconds.')
    request = _extract_request(source)
    result = _isolated_probe(request, '--probe-child', Path(__file__).resolve(), timeout=timeout)
    _validate_result(result, request)
    if _sha(request['source']) != request['source_sha256'] or _sha(request['plugin_path']) != request['plugin_sha256']:
        raise ValueError('The project or installed Pro-Q binary changed during inspection.')
    return result


def _probe_child(request):
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
    _validate_plugin_binary(request['plugin_path'], request['plugin_sha256'])
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
            for parameter in parameters[:91]:
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


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] != '--probe-child':
        raise SystemExit('This module is an isolated EQ inspection worker.')
    raise SystemExit(worker_main(_probe_child))
