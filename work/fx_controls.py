"""Shared, explicit wet-effect settings for the connector and its interface."""
from __future__ import annotations

from copy import deepcopy
import math


RISE_BEATS = (.5, 1, 2, 4, 8)
FADE_BARS = (1, 2, 4, 8, 16)
THROW_TIMINGS = ('off', 'phrases', 'every8', 'every4')


def _preset(delay, reverb, duck):
    keys = ('amount', 'peak', 'rise_beats', 'fade_bars', 'throws')
    return dict(delay=dict(zip(keys, delay)), reverb=dict(zip(keys, reverb)),
                duck=duck, bass_full_range=False)


FX_PRESETS = {
    'Gentle': _preset((.12, .50, 2, 4, 'phrases'), (.16, .60, 4, 8, 'phrases'), .25),
    'Big & smooth': _preset((.25, .85, 4, 4, 'every8'), (.30, .90, 4, 8, 'every8'), .15),
    'Huge throws': _preset((.30, 1, 4, 8, 'every4'), (.35, 1, 8, 16, 'every4'), .10),
}
DEFAULT_FX_CONTROLS = deepcopy(FX_PRESETS['Big & smooth'])


def _number(value, label, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{label} must be a finite number from {low:g} to {high:g}.')
    return float(value)


def validate_fx_controls(document):
    """Validate the complete schema and return an independent clean copy."""
    if not isinstance(document, dict) or set(document) != {'delay', 'reverb', 'duck', 'bass_full_range'}:
        raise ValueError('FX controls need delay, reverb, duck and bass_full_range settings, with no extra fields.')
    result = {}
    fields = {'amount', 'peak', 'rise_beats', 'fade_bars', 'throws'}
    for lane in ('delay', 'reverb'):
        row = document[lane]
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError(f'{lane.title()} needs amount, peak, rise, fade and throw timing settings, with no extra fields.')
        amount = _number(row['amount'], f'{lane.title()} amount', 0, 1)
        peak = _number(row['peak'], f'{lane.title()} peak', 0, 1)
        if peak < amount:
            raise ValueError(f'{lane.title()} peak must be at least its normal amount.')
        rise = _number(row['rise_beats'], f'{lane.title()} rise', .5, 8)
        fade = _number(row['fade_bars'], f'{lane.title()} fade', 1, 16)
        if rise not in RISE_BEATS or fade not in FADE_BARS:
            raise ValueError(f'Choose one of the supported {lane} rise and fade times.')
        if not isinstance(row['throws'], str) or row['throws'] not in THROW_TIMINGS:
            raise ValueError(f'Choose a supported {lane} throw timing.')
        result[lane] = dict(amount=amount, peak=peak, rise_beats=rise,
                            fade_bars=int(fade), throws=row['throws'])
    result['duck'] = _number(document['duck'], 'Drop duck depth', 0, 1)
    if type(document['bass_full_range']) is not bool:
        raise ValueError('Bass full range must be on or off.')
    result['bass_full_range'] = document['bass_full_range']
    return result
