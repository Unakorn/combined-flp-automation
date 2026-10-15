"""Optional named note guides for existing stems. Never changes musical notes.

``note_events`` retains note-on velocity and project-tick timing. Its duration
describes a note gate, not the audible tail of a rendered drum sample.
"""
from collections import defaultdict, deque
from pathlib import Path
from mido import MidiFile


_DRUM_ROLES = {'kick', 'snare', 'clap', 'tom', 'hat', 'perc', 'ride', 'crash'}


def match_name(name):
    from stem_names import normalized_name
    return normalized_name(name)


def _drum_guidance(stem):
    from stem_names import detect_name_role
    if (stem.get('role') or detect_name_role(stem['name'])) in _DRUM_ROLES:
        return (' Use the source instrument FLP or a separately named MIDI track for each drum; '
                'a combined DRUMS track cannot identify custom drum slots by pitch alone.')
    return ''


def _event_order(note):
    return note['tick'], note['note'], note['duration'], note['velocity']


def read_timing(path, stems, *, ppq, bpm, end_tick, mapping=None):
    midi = MidiFile(path)
    if midi.type == 2 or midi.ticks_per_beat <= 0:
        raise ValueError('The MIDI guide must use one common beat timeline.')
    tracks = []
    tempos = set()
    markers = []
    for track in midi.tracks:
        tick, name, active, spans, notes = 0, '', defaultdict(deque), [], []
        for msg in track:
            tick += msg.time
            at = round(tick * ppq / midi.ticks_per_beat)
            if msg.type == 'track_name':
                name = msg.name
            elif msg.type == 'set_tempo':
                tempos.add(msg.tempo)
            elif msg.type == 'time_signature' and (msg.numerator,msg.denominator)!=(4,4):
                raise ValueError('MIDI guide must have the same 4/4 meter as the FLP.')
            elif msg.type in ('marker','cue_marker'):
                markers.append(dict(tick=at,name=msg.text))
            elif msg.type == 'note_on' and msg.velocity:
                active[msg.channel,msg.note].append((at, msg.velocity))
            elif msg.type in ('note_on','note_off'):
                stack = active[msg.channel,msg.note]
                if stack:
                    start, velocity=stack.popleft()
                    notes.append(dict(tick=start, duration=at-start, note=msg.note, velocity=velocity))
                    if at>start: spans.append((start,at))
        if any(active.values()):
            raise ValueError(f'MIDI track {name!r} has unterminated notes.')
        if notes:
            tracks.append(dict(id=len(tracks),name=name,spans=sorted(spans),
                               notes=sorted(notes,key=_event_order)))
    if len(tempos)>1 or any(abs(60_000_000/t-bpm)>.1 for t in tempos):
        raise ValueError('MIDI tempo differs from the FLP. Export a matching constant-tempo guide.')
    if not tracks:
        raise ValueError('The MIDI guide has no musical notes.')
    if max(n['tick']+n['duration'] for t in tracks for n in t['notes']) > end_tick + ppq:
        raise ValueError('MIDI notes extend beyond the stem arrangement; check alignment.')
    by_name=defaultdict(list)
    for track in tracks: by_name[match_name(track['name'])].append(track)
    from stem_names import match_stem_names
    automatic=match_stem_names(stems,tracks)['matches']
    warnings, matched = [], {}
    stem_names=defaultdict(list)
    for stem in stems: stem_names[match_name(stem['name'])].append(stem)
    for stem in stems:
        explicit=(mapping or {}).get(stem['id'],(mapping or {}).get(str(stem['id'])))
        key=match_name(explicit or stem['name']); candidates=by_name[key]
        if not explicit and stem['id'] in automatic:
            candidates=[tracks[automatic[stem['id']]['track_id']]]
        if len(candidates)!=1 or (not explicit and stem['id'] not in automatic):
            warnings.append(f"{stem['name']}: MIDI name match {'ambiguous' if candidates else 'missing'}; audio timing retained."
                            + _drum_guidance(stem))
            continue
        track=candidates[0]; phrases=[]
        for start,stop in track['spans']:
            if phrases and start <= phrases[-1][1] + ppq:
                phrases[-1][1]=max(phrases[-1][1],stop)
            else: phrases.append([start,stop])
        matched[stem['id']] = dict(midi_track=track['name'],
            phrase_ends=[stop for _,stop in phrases if 0<stop<end_tick],
            note_spans=[list(pair) for pair in track['spans']],
            note_events=[dict(note) for note in track['notes']], note_velocity_max=127,
            note_guide_source=f"MIDI: {Path(path).name} — {track['name']}")
    return dict(matches=matched,warnings=warnings,markers=markers,
                path=str(Path(path).resolve()))


def read_source_timing(path, stems, *, ppq, bpm, end_tick):
    """Reuse the existing arrangement and Layer resolver for named FLP notes."""
    from flp_song_analysis import analyze_flp
    from stem_names import match_stem_names
    model=analyze_flp(path,max_instruments=128,allow_single_pattern=True)
    if abs(model['tempo']-bpm)>.1 or model['time_signature']!=[4,4]:
        raise ValueError('The source FLP has a different tempo or meter.')
    tracks=[dict(id=v['channel_id'],name=v['name'],notes=v['notes']) for v in model['voices']]
    # Rendered stems can include effect tails, and source arrangements can include
    # empty space or unrelated clips after the notes. Compare the musical notes
    # with the destination timeline instead of requiring equal project lengths.
    note_end=max((round((n['tick']+n['duration'])*ppq/model['ppq'])
                  for t in tracks for n in t['notes']),default=0)
    if note_end>end_tick+ppq:
        raise ValueError('The source FLP notes extend beyond the audio stems; check the guide alignment.')
    named=match_stem_names(stems,tracks)
    matches={}; warnings=[]
    duration=round(model['end_tick']*ppq/model['ppq'])
    if abs(duration-end_tick)>ppq:
        warnings.append('The source FLP and audio stems have different ending times. '
                        'The source notes fit the stem timeline; the full audio length '
                        'and original note timing are retained.')
    by_id={t['id']:t for t in tracks}
    for stem in stems:
        match=named['matches'].get(stem['id'])
        if not match:
            warnings.append(f"{stem['name']}: source-FLP name match unresolved; audio timing retained."
                            + _drum_guidance(stem))
            continue
        track=by_id[match['track_id']]; phrases=[]; spans=[]; notes=[]
        for note in track['notes']:
            start=round(note['tick']*ppq/model['ppq'])
            stop=round((note['tick']+note['duration'])*ppq/model['ppq'])
            spans.append((start,stop))
            notes.append(dict(tick=start,duration=stop-start,note=note['note'],velocity=note['velocity']))
        for start,stop in sorted(spans):
            if phrases and start<=phrases[-1][1]+ppq: phrases[-1][1]=max(phrases[-1][1],stop)
            else: phrases.append([start,stop])
        matches[stem['id']]=dict(source_track=track['name'],match_reason=match['reason'],
                                phrase_ends=[b for _,b in phrases if 0<b<end_tick],note_spans=spans,
                                note_events=sorted(notes,key=_event_order),note_velocity_max=128,
                                note_guide_source=f"Source FLP: {Path(path).name} — {track['name']}")
    markers=[dict(m,tick=round(m['tick']*ppq/model['ppq'])) for m in model['markers']]
    return dict(path=str(Path(path).resolve()),matches=matches,warnings=warnings,
                markers=[m for m in markers if 0<=m['tick']<end_tick])
