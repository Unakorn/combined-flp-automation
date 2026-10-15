"""Shared eight-bar scheduling for MIDI storage and streamed audio analysis."""

def chunk_length(ppq, beats_per_bar=4, bars=8):
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or x <= 0
           for x in (ppq, beats_per_bar, bars)):
        raise ValueError('Section resolution, meter and length must be positive.')
    value = round(ppq * beats_per_bar * bars)
    if value < 1:
        raise ValueError('A section must contain at least one tick.')
    return value


def section_windows(ppq, end_tick, *, beats_per_bar=4, bars=8):
    length = chunk_length(ppq, beats_per_bar, bars)
    for index, position in enumerate(range(0, end_tick, length)):
        yield dict(index=index, position=position, duration=min(length, end_tick-position))
