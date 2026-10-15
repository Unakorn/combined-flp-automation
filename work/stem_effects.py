"""Dedicated stock effects controlled by native FL Studio parameters.

The catalog contains small, exact Image-Line factory preset states. Blends use
the host's effect-slot Mix control. A separate low-pass sweep uses the observed
Fruity Filter cutoff parameter in normalized knob position, never guessed Hz.
Callers must allocate an empty slot and initialize controls from the spec.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct


from app_paths import resource_path

CATALOG = resource_path('stem-effects.json')
_SUPPORTED = {
    ('Fruity Filter', 'lowpass_cutoff_percent'): 'filter_lowpass_cutoff',
    ('Fruity Filter', 'lowpass_blend_percent'): 'filter_lowpass',
    ('Fruity Filter', 'highpass_blend_percent'): 'filter_highpass',
    ('Fruity Delay 3', 'effect_blend_percent'): 'delay',
    ('Fruity Reeverb 2', 'effect_blend_percent'): 'reverb',
}
_PLUGIN_KEYS = {201, 212, 203, 155, 128, 41, 213}
# Exact state observed on insert 6 / slot 1 in Image-Line's factory demo
# Edlan & Ella Noel - Song For You.flp; its native link targets cutoff 0x8000.
# This also matches drum_template.OPEN_FILTER byte for byte. State layout and
# address provenance are recorded in stem-effects.json; no Hz map is implied.
_OPEN_LOWPASS_STATE = struct.pack('<7IB', 2, 1024, 0, 1024, 0, 0, 0, 1)


def _validate_spec(spec):
    expected = next(((plugin, parameter) for (plugin, parameter), key in _SUPPORTED.items()
                     if key == spec.get('key')), None)
    if expected is None or (spec.get('plugin'), spec.get('parameter')) != expected:
        raise ValueError('Unknown dedicated stock-effect definition.')
    rows = spec.get('events')
    if not isinstance(rows, list) or not 3 <= len(rows) <= len(_PLUGIN_KEYS):
        raise ValueError('Unsupported stock-effect event count.')
    try:
        events = [(key, bytes.fromhex(payload)) for key, payload in rows]
    except (TypeError, ValueError):
        raise ValueError('Malformed stock-effect event data.') from None
    if len({key for key, _ in events}) != len(events) or any(key not in _PLUGIN_KEYS for key, _ in events):
        raise ValueError('Unsupported stock-effect event layout.')
    fields = dict(events)
    if (events[0][0] != 201 or fields.get(201) != (expected[0] + '\0').encode('utf-16-le')
            or len(fields.get(212, b'')) != 52):
        raise ValueError('Stock effect must have a native name and a known wrapper layout.')
    state = fields.get(213, b'')
    if (not 1 <= len(state) <= 256 or len(state) != spec.get('state_size')
            or hashlib.sha256(state).hexdigest() != spec.get('state_sha256')):
        raise ValueError('Stock-effect state does not match its recorded provenance.')
    for key, size in ((155, 4), (128, 4), (41, 1)):
        if key in fields and len(fields[key]) != size:
            raise ValueError('Stock effect has an unsupported fixed-size event.')
    if 203 in fields and (len(fields[203]) > 400 or len(fields[203]) % 2):
        raise ValueError('Unsupported stock-effect display name.')
    if spec['key'] == 'filter_lowpass_cutoff':
        if (state != _OPEN_LOWPASS_STATE or spec.get('control') != 'plugin_parameter'
                or spec.get('target_offset') != 0x8000
                or spec.get('baseline_normalized') != 1
                or spec.get('initial_parameter_raw') != 1024
                or spec.get('initial_mix_raw') != 12800):
            raise ValueError('A cutoff sweep must use the verified open low-pass state and native cutoff address.')
    elif (spec.get('initial_mix_raw') != 0 or spec.get('control') != 'mixer_slot_mix'
          or spec.get('target_offset') != 0x1f01 or spec.get('baseline_normalized') != 0):
        raise ValueError('Added blend effects must start with their effect amount at zero.')
    if spec.get('initial_enabled_raw') != 1:
        raise ValueError('Added stock effects must have a known enabled state.')
    return events


def get_effect_spec(plugin, parameter=None):
    """Return an independent JSON-safe spec, or None for an unsupported control.

    A catalog key such as ``filter_lowpass_cutoff`` is also accepted without a
    parameter. Requests for internal ``wet_percent`` or cutoff Hz stay gated.
    """
    key = plugin if parameter is None and plugin in _SUPPORTED.values() else _SUPPORTED.get((plugin, parameter))
    if key is None:
        return None
    with CATALOG.open(encoding='utf-8') as source:
        catalog = json.load(source)
    if catalog.get('version') != 1:
        raise ValueError('Unsupported dedicated stock-effect catalog version.')
    spec = deepcopy(catalog['effects'][key])
    _validate_spec(spec)
    return spec


def build_effect_record(spec, insert, slot):
    """Clone a stock record for a caller-verified empty, zero-based mixer slot.

    Only native wrapper routing and the editor-visible bit are changed. This
    does not add the slot's event 98, overwrite user effects, or initialize Mix.
    The caller initializes slot Mix (0x1f01) from ``initial_mix_raw`` and enabled
    (0x1f00) to one. The cutoff spec additionally initializes target_offset
    0x8000 to initial_parameter_raw 1024. Native Automation Clip values for both
    controls are their requested percent / 100. Baseline is zero for blends
    and one for cutoff. Only cutoff's plugin parameter is directly controlled.
    """
    if type(insert) is not int or not 1 <= insert <= 63 or type(slot) is not int or not 0 <= slot <= 9:
        raise ValueError('Stock effects need insert 1-63 and slot 0-9.')
    events = _validate_spec(spec)
    result = []
    for key, payload in events:
        if key == 212:
            wrapper = bytearray(payload)
            struct.pack_into('<II', wrapper, 0, insert, slot)
            flags = struct.unpack_from('<I', wrapper, 16)[0]
            struct.pack_into('<I', wrapper, 16, flags & ~1)
            payload = bytes(wrapper)
        result.append((key, payload))
    return result
