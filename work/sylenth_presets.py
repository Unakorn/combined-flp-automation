"""Read-only Sylenth1 preset inspection and conservative movement selection.

Verified format: Sylenth1 3.073's 3045 bank state, 244 normalized float32
parameters. Indices follow the native controller's parameter enumeration, also
verified by native import/export of these banks. They are FL wrapper indices,
not VST3 hashed IDs. Parameter 42 is independently confirmed by native FL links.

No plugin state, oscillator setup, switch, tuning or routing is written here.
Returned baselines are the exact saved float values. The FK transport used by
the exporter has 1/128 steps; it cannot restore arbitrary floats exactly live.
"""
from __future__ import annotations

import math
import re
import struct
import zlib


class UnsupportedPreset(ValueError):
    """The state cannot be interpreted safely with the verified parameter map."""


def decode_sylenth_state(state: bytes) -> dict:
    """Decode the selected patch from an FL wrapper's state, without mutation."""
    marker = state.find(b'CcnK')
    if marker < 0 or marker + 160 > len(state):
        raise UnsupportedPreset('The saved Sylenth bank is missing or incomplete.')
    size = struct.unpack_from('>I', state, marker + 4)[0] + 8
    if not 160 <= size <= 8_000_000 or marker + size > len(state):
        raise UnsupportedPreset('The saved Sylenth bank length is unsupported.')
    bank = state[marker:marker + size]
    if bank[8:12] != b'FBCh' or bank[16:20] != b'syl1':
        raise UnsupportedPreset('This is not the verified Sylenth bank format.')
    chunk_size = struct.unpack_from('>I', bank, 156)[0]
    if chunk_size < 20 or 160 + chunk_size > len(bank):
        raise UnsupportedPreset('The Sylenth bank chunk is incomplete.')
    chunk = bank[160:160 + chunk_size]
    if chunk[:4] != b'1lys':
        raise UnsupportedPreset('The Sylenth bank signature is unknown.')
    if chunk[8:10] in (b'x\x01', b'x\x5e', b'x\x9c', b'x\xda'):
        try:
            decoder = zlib.decompressobj()
            decoded = decoder.decompress(chunk[8:], 8_000_001)
            if len(decoded) > 8_000_000 or not decoder.eof or decoder.unconsumed_tail:
                raise UnsupportedPreset('The Sylenth bank is oversized or incomplete.')
        except zlib.error as exc:
            raise UnsupportedPreset('The Sylenth bank could not be decompressed.') from exc
    else:
        decoded = chunk
    if len(decoded) < 20 or decoded[:4] != b'1lys':
        raise UnsupportedPreset('The decoded Sylenth header is incomplete.')
    _, version, count, parameter_count, selected = struct.unpack_from('<5I', decoded)
    if version != 3045 or parameter_count != 244 or not 1 <= count <= 512:
        raise UnsupportedPreset('This Sylenth state version has not been verified.')
    stride = parameter_count * 4 + 36
    if selected >= count or len(decoded) < 20 + count * stride:
        raise UnsupportedPreset('The selected Sylenth patch is incomplete.')
    offset = 20 + selected * stride
    values = struct.unpack_from('<244f', decoded, offset)
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise UnsupportedPreset('The saved patch contains invalid normalized values.')
    name = decoded[offset + 976:offset + stride].split(b'\0', 1)[0].decode('utf-8', errors='replace')
    return {'preset': name or 'Unnamed Sylenth patch', 'values': values,
            'version': version, 'selected': selected}


def _enum(value: float, choices: int) -> int:
    # Native controller text changes at 1/N, 2/N, ...; saved states normally
    # use endpoint-inclusive enum values such as 0, 1/3, 2/3, 1.
    return min(choices - 1, int(value * choices))


def _role(preset: str, channel_name: str) -> str:
    for label in (preset, channel_name):
        words = set(re.findall(r'[a-z]+', label.casefold()))
        if words & {'sub', 'bass'}:
            return 'bass'
        if words & {'pad', 'texture', 'keys', 'chord', 'chords', 'piano'}:
            return 'chords'
        if words & {'arp', 'pluck'}:
            return 'arp'
        if words & {'lead'}:
            return 'lead'
    return 'default'


def select_controls(values, preset: str = '', channel_name: str = '') -> list[dict]:
    """Select only sounding paths and existing continuous amounts.

    This is intentionally a small supported set. Unknown modulation routes,
    inactive effects, mono-voice detuning, and zero amounts are left alone.
    """
    if len(values) != 244 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise UnsupportedPreset('Expected 244 finite normalized Sylenth parameters.')
    if values[147] >= .5:
        raise UnsupportedPreset('Part Solo is enabled; its sounding routing is not supported for automatic movement.')
    if values[68] <= 0:
        return []
    role = _role(preset, channel_name)
    controls = []

    def add(index, name, character, below, above=None, *, lower=.0078125, upper=1.0):
        baseline = float(values[index])
        if baseline <= 0:
            return
        above = below if above is None else above
        minimum = min(baseline, max(lower, baseline - below))
        maximum = max(baseline, min(upper, baseline + above))
        if maximum - minimum < .004:
            return
        controls.append({'parameter_id': index, 'name': name, 'baseline': baseline,
                         'minimum': minimum, 'maximum': maximum,
                         'character': character, 'role': role})

    # Osc offsets start at detune; voices and volume are +9 and +10.
    oscillators = {'A': (81, 93), 'B': (105, 117)}
    sounding = {part: any(_enum(values[o + 9], 9) > 0 and values[o + 10] > 0
                           for o in offsets) for part, offsets in oscillators.items()}
    audible_sources = set()
    active_filters = set()
    for part, cutoff, mix in (('A', 42, 69), ('B', 48, 70)):
        if values[mix] <= 0:
            continue
        routing = _enum(values[cutoff + 2], 3)
        sources = {part} if routing == 0 else {'A', 'B'} if routing == 1 else set()
        sources = {source for source in sources if sounding[source]}
        audible_sources.update(sources)
        if sources and _enum(values[cutoff + 4], 4) != 0:
            active_filters.add(part)
            add(cutoff, f'Filter {part} cutoff', 'brightness', .085 if role == 'bass' else .12)
            add(cutoff + 1, f'Filter {part} drive', 'drive', min(.035, values[cutoff + 1] * .3), .025)
            add(cutoff + 3, f'Filter {part} resonance', 'modulation', .015, .015,
                upper=max(values[cutoff + 3], .3))

    if not audible_sources:
        return []
    for part, envelope in (('A', 0), ('B', 4)):
        if part not in audible_sources:
            continue
        if values[envelope + 3] < .99:
            add(envelope + 1, f'Amp {part} decay', 'decay', values[envelope + 1] * .22,
                values[envelope + 1] * .18)
        add(envelope + 2, f'Amp {part} release', 'release', values[envelope + 2] * .2,
            values[envelope + 2] * .15)
        for number, osc in enumerate(oscillators[part], 1):
            voices = _enum(values[osc + 9], 9)
            if voices < 2 or values[osc + 10] <= 0:
                continue
            # Noise detune is not useful; stereo spread is meaningful.
            if _enum(values[osc + 11], 8) != 7:
                add(osc, f'Osc {part}{number} detune', 'width', min(.025, values[osc] * .25),
                    min(.025, values[osc] * .2))
            add(osc + 8, f'Osc {part}{number} stereo', 'width', .08, .06)

    def filter_destination(index):
        destination = _enum(values[index], 33)
        # Verified native destination enumeration: Cutoff A/B/AB, Reso A/B/AB.
        parts = {13: {'A'}, 14: {'B'}, 15: {'A', 'B'},
                 16: {'A'}, 17: {'B'}, 18: {'A', 'B'}}.get(destination, set())
        return bool(parts) and parts <= active_filters

    def depth(index, name):
        distance = abs(values[index] - .5)
        if distance < .008:
            return False
        delta = min(.022, distance * .22)
        add(index, name, 'modulation', delta, lower=.5 if values[index] > .5 else .0078125,
            upper=1 if values[index] > .5 else .5)
        return True

    for number, envelope, amount, destination in ((1, 71, 197, 225), (2, 75, 199, 227)):
        used = False
        for slot in (0, 1):
            if filter_destination(destination + slot):
                used |= depth(amount + slot, f'Mod envelope {number} depth {slot + 1}')
        if used and values[envelope + 3] < .99:
            add(envelope + 1, f'Mod envelope {number} decay', 'decay', values[envelope + 1] * .2,
                values[envelope + 1] * .15)
    for number, gain, amount, destination in ((1, 59, 201, 229), (2, 64, 203, 231)):
        if values[gain] <= 0:
            continue
        for slot in (0, 1):
            if filter_destination(destination + slot):
                depth(amount + slot, f'LFO {number} filter depth {slot + 1}')

    # Only velocity modulation is driven by the song notes reliably. A saved
    # ModWheel route alone does not establish that a moving source exists.
    for row_name, source, amount, destination in (('1A', 207, 205, 233), ('1B', 210, 208, 235),
                                                  ('2A', 213, 211, 237), ('2B', 216, 214, 239)):
        if _enum(values[source], 12) == 1:  # Velocity, verified source enumeration.
            for slot in (0, 1):
                if filter_destination(destination + slot):
                    depth(amount + slot, f'Velocity filter depth {row_name}-{slot + 1}')

    if values[217] >= .5:
        add(8, 'Arp gate', 'decay', .06, .05)
    if values[220] >= .5:
        add(25, 'Delay wet', 'effect', .025, .02)
    if values[221] >= .5:
        add(35, 'Distortion amount', 'drive', .025, .02)
        add(36, 'Distortion wet', 'effect', .025, .02)
    if values[224] >= .5:
        add(143, 'Reverb wet', 'effect', .03, .025)
    if values[218] >= .5:
        add(16, 'Chorus wet', 'effect', .025, .02)
    # Feedback, synced rates, predelay, reverb size, EQ gains and compressor
    # settings deliberately stay as designed; external sends already move.
    return controls


def inspect_instruments(events, channels) -> dict:
    """Return recognized instrument profiles, excluding already linked targets."""
    linked_targets = set()
    unknown_links = False
    for key, payload in events:
        if key in (226, 227):
            if len(payload) != 20:
                unknown_links = True
            else:
                # Both observed FL remote-link records store destination at8.
                # 226's exact-match exclusion is conservative; no226 is changed.
                linked_targets.add(struct.unpack_from('<I', payload, 8)[0])
    profiles, warnings = [], []
    for channel in channels:
        if channel.get('type', 2) not in (0, 2):
            continue
        states = [payload for key, payload in events[channel['start']:channel['stop']] if key == 213]
        is_sylenth = ('sylenth' in channel.get('plugin', '').casefold()
                      or any(b'syl1' in state or b'Sylenth1' in state or b'S\0y\0l\0e\0n\0t\0h\0' in state
                             for state in states))
        if not is_sylenth:
            continue
        profile = {'instrument_id': channel['id'], 'preset': '', 'controls': []}
        try:
            if len(states) != 1:
                raise UnsupportedPreset('The instrument has an ambiguous saved plugin state.')
            decoded = decode_sylenth_state(states[0])
            profile['preset'] = decoded['preset']
            if unknown_links:
                raise UnsupportedPreset('An unknown existing control-link format prevents safe movement selection.')
            controls = select_controls(decoded['values'], decoded['preset'], channel.get('name', ''))
            for control in controls:
                target = channel['id'] << 16 | (0x8000 + control['parameter_id'])
                if target in linked_targets:
                    warnings.append(f"{channel.get('name', 'Sylenth')}: {control['name']} already has a control link; skipped.")
                else:
                    profile['controls'].append(control)
            if not profile['controls']:
                profile['reason'] = 'No unlinked supported continuous controls are active in this patch.'
        except (UnsupportedPreset, struct.error) as exc:
            profile['reason'] = str(exc)
            warnings.append(f"{channel.get('name', 'Sylenth')}: {exc}")
        profiles.append(profile)
    return {'instruments': profiles, 'warnings': warnings}
