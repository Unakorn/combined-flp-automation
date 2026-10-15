"""Conservative sample-pack vocabulary and one-to-one stem/MIDI name matching.

Edit ``resources/stem-names.json`` to add vocabulary. Role identification and
track identity are separate: a shared role never proves that two tracks match.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata


from app_paths import resource_path

NAME_RULES_PATH = resource_path('stem-names.json')
ROLES = {'kick', 'snare', 'clap', 'tom', 'hat', 'perc', 'ride', 'crash',
         'bass', 'brass', 'guitar', 'keys', 'vocals', 'pads', 'synth', 'effects', 'drums'}


@lru_cache(maxsize=4)
def _read_rules(path, modified_ns, size):
    with Path(path).open(encoding='utf-8') as source:
        rules = json.load(source)
    if rules.get('version') != 1:
        raise ValueError('Unsupported stem naming vocabulary version.')
    if not isinstance(rules.get('normalization'), dict):
        raise ValueError('Stem naming rules need normalization settings.')
    for section in ('category_codes', 'drum_aliases', 'instrument_aliases', 'role_suggestions', 'matching_aliases'):
        if not isinstance(rules.get(section), dict):
            raise ValueError(f'Stem naming rules need a {section} object.')
    for code, role in rules['category_codes'].items():
        if not code or role not in ROLES:
            raise ValueError(f'Invalid naming category code: {code}.')
    for section in ('drum_aliases', 'instrument_aliases'):
        for role, aliases in rules[section].items():
            if role not in ROLES or not isinstance(aliases, list) or not all(isinstance(a, str) and a.strip() for a in aliases):
                raise ValueError(f'Invalid {section} entry for {role}.')
    for alias, candidates in rules['role_suggestions'].items():
        if not isinstance(alias, str) or not isinstance(candidates, list) or any(role not in ROLES for role in candidates):
            raise ValueError('Invalid role suggestion.')
    if not all(isinstance(a, str) and isinstance(b, str) and a.strip() and b.strip()
               for a, b in rules['matching_aliases'].items()):
        raise ValueError('Matching aliases must map non-empty phrases to non-empty phrases.')
    return rules


def load_name_rules(path=None):
    """Reload editable vocabulary when its file changes; no process restart needed."""
    path = Path(path or NAME_RULES_PATH).resolve()
    stat = path.stat()
    return _read_rules(str(path), stat.st_mtime_ns, stat.st_size)


def _words(text):
    return re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', str(text)).casefold())


def _normalized_name(name, rules):
    text = unicodedata.normalize('NFKC', str(name)).strip()
    # These can be Windows file paths even when the tests run on Linux.
    text = re.split(r'[/\\]', text)[-1]
    extensions = rules['normalization']['file_extensions']
    if '.' in text and text.rsplit('.', 1)[1].casefold() in extensions:
        text = text.rsplit('.', 1)[0]
    settings = rules['normalization']
    protected = set(settings['preserve_numeric_prefixes'])
    labels = '|'.join(re.escape(value) for value in settings['leading_export_labels'])
    # Export labels require a numeric index: an instrument actually named
    # "Stem" or "Track" is not discarded by a global stopword list.
    text = re.sub(rf'^\s*(?:{labels})\s*\d+\s*[_\-:.]+\s*', '', text, flags=re.I)
    while True:
        match = re.match(r'^\s*(\d+)[\s_\-]+', text)
        if not match or match.group(1) in protected:
            break
        text = text[match.end():]
    suffixes = '|'.join(re.escape(value) for value in settings['trailing_export_labels'])
    # Only recognizable exporter tags at the end are removed. Sound variants
    # such as "Padded", "2", "Bright", "Dry", and "Stereo" remain meaningful.
    suffix_pattern = rf'(?:\s*\((?:{suffixes})\)|\s*\[(?:{suffixes})\]|[ _\-]+(?:{suffixes}))(?:[ _\-]+\d+)?\s*$'
    while True:
        clean = re.sub(suffix_pattern, '', text, flags=re.I)
        if clean == text:
            break
        text = clean
    return ' '.join(_words(text))


def normalized_name(name):
    """Normalize exporter notation without removing instrument descriptions."""
    return _normalized_name(name, load_name_rules())


def _alias_hits(tokens, role_aliases):
    hits = []
    for role, aliases in role_aliases.items():
        for alias in aliases:
            phrase = _words(alias)
            for start in range(len(tokens) - len(phrase) + 1):
                if tokens[start:start + len(phrase)] == phrase:
                    hits.append((start, start + len(phrase), role, alias))
    # A precise instrument phrase wins over words contained inside it:
    # "lead guitar" is guitar, "synth bass" is bass, "string pad" is pads.
    return [hit for hit in hits if not any(other[0] <= hit[0] and other[1] >= hit[1]
            and other[1] - other[0] > hit[1] - hit[0] for other in hits)]


def _classification(name, rules):
    normalized = _normalized_name(name, rules)
    if normalized in {'drums', 'mixed drums', 'drum bus', 'drum kit', 'drumkit',
                       'drum monkey', 'unison drum monkey'}:
        return dict(role='drums', confidence=1.0,
                    reason='Explicit mixed drum bus name; individual hits are not inferred.',
                    candidates=['drums'])
    tokens = normalized.split()
    categories = {code.casefold(): role for code, role in rules['category_codes'].items()}
    codes = [(token, categories[token]) for token in tokens if token in categories]
    if codes:
        candidates = list(dict.fromkeys(role for _, role in codes))
        if len(candidates) == 1:
            return dict(role=candidates[0], confidence=1.0,
                        reason='Category code: ' + ', '.join(code.upper() for code, _ in codes), candidates=candidates)
        return dict(role=None, confidence=0.0, reason='Conflicting category codes require confirmation.', candidates=candidates)
    drum_hits = _alias_hits(tokens, rules['drum_aliases'])
    if drum_hits:
        candidates = list(dict.fromkeys(hit[2] for hit in drum_hits))
        return dict(role=candidates[0] if len(candidates) == 1 else None,
                    confidence=.98 if len(candidates) == 1 else 0.0,
                    reason='Drum name: ' + ', '.join(dict.fromkeys(hit[3] for hit in drum_hits)), candidates=candidates)
    if '808' in tokens:
        if {'bass', 'sub', 'low'} & set(tokens):
            return dict(role='bass', confidence=.98, reason='808 with an explicit bass, sub, or low qualifier.', candidates=['bass'])
        return dict(role=None, confidence=0.0,
                    reason='808 alone does not identify a role; confirm it or inspect sustained sub-bass evidence.', candidates=['bass'])
    hits = _alias_hits(tokens, rules['instrument_aliases'])
    candidates = list(dict.fromkeys(hit[2] for hit in hits))
    if candidates:
        return dict(role=candidates[0] if len(candidates) == 1 else None,
                    confidence=.92 if len(candidates) == 1 else 0.0,
                    reason='Instrument name: ' + ', '.join(dict.fromkeys(hit[3] for hit in hits)), candidates=candidates)
    suggestions = []
    for token in tokens:
        for candidate in rules['role_suggestions'].get(token, []):
            if candidate not in suggestions:
                suggestions.append(candidate)
    return dict(role=None, confidence=0.0, reason='No unambiguous instrument name; confirm the role.', candidates=suggestions)


def classify_name(name):
    """Explain lexical role evidence, keeping ambiguity visible to the user."""
    return _classification(name, load_name_rules())


def detect_name_role(name):
    return classify_name(name)['role']


def _canonical_key(name, rules):
    tokens = _normalized_name(name, rules).split()
    category = {code.casefold(): role for code, role in rules['category_codes'].items()}
    # Pack prefixes may redundantly precede a full instrument name. Remove one
    # only when the remaining words independently name the same instrument role.
    if len(tokens) > 1 and tokens[0] in category:
        remainder = _classification(' '.join(tokens[1:]), rules)
        if remainder['role'] == category[tokens[0]]:
            tokens = tokens[1:]
    aliases = sorted(((_words(a), _words(b)) for a, b in rules['matching_aliases'].items()),
                     key=lambda pair: -len(pair[0]))
    canonical, index = [], 0
    while index < len(tokens):
        for phrase, replacement in aliases:
            if tokens[index:index + len(phrase)] == phrase:
                canonical.extend(replacement)
                index += len(phrase)
                break
        else:
            canonical.append(tokens[index])
            index += 1
    # Word order changes from "GTR Acoustic" to "Acoustic Guitar" are benign;
    # every descriptive token and repeated token still has to be present.
    return tuple(sorted(canonical))


def _has_identity(key, rules):
    generic = set(ROLES) | {'percussion', 'guitar', 'brass', 'keys', 'vocals', 'pads', 'synth', 'effects',
                            'piano', 'electric', 'lead', 'horn', 'hat', 'bass', 'drum', 'drums'}
    return bool(set(key) - generic)


def match_stem_names(stems, tracks):
    """Match exact names, then conservative unique aliases; retain review cases.

    Both input lists contain ``id`` and ``name``. ``matches`` is keyed by the
    original stem ID. ``unmatched`` and ``ambiguous`` contain stem records with
    candidate track records and reasons. Input records are never mutated.
    """
    stems, tracks = list(stems), list(tracks)
    for records, label in ((stems, 'stem'), (tracks, 'MIDI track')):
        if len({record['id'] for record in records}) != len(records):
            raise ValueError(f'Every {label} needs a unique ID.')
    rules = load_name_rules()
    stem_info = {s['id']: (_classification(s['name'], rules)['role'], _canonical_key(s['name'], rules)) for s in stems}
    track_info = {t['id']: (_classification(t['name'], rules)['role'], _canonical_key(t['name'], rules)) for t in tracks}
    exact_stems, exact_tracks = defaultdict(list), defaultdict(list)
    for stem in stems:
        exact_stems[_normalized_name(stem['name'], rules)].append(stem)
    for track in tracks:
        exact_tracks[_normalized_name(track['name'], rules)].append(track)
    result = dict(matches={}, unmatched=[], ambiguous=[])
    used, resolved = set(), set()

    def candidate(track, confidence, reason):
        return dict(track_id=track['id'], track_name=track['name'], confidence=confidence, reason=reason)

    def unresolved(stem, candidates, reason, ambiguous=False):
        result['ambiguous' if ambiguous else 'unmatched'].append(
            dict(stem_id=stem['id'], stem_name=stem['name'], reason=reason, candidates=candidates))
        resolved.add(stem['id'])

    for stem in stems:
        key = _normalized_name(stem['name'], rules)
        choices = exact_tracks.get(key, []) if key else []
        if not choices:
            continue
        if len(choices) == len(exact_stems[key]) == 1:
            result['matches'][stem['id']] = candidate(choices[0], 1.0, 'Unique exact name after exporter normalization.')
            used.add(choices[0]['id'])
            resolved.add(stem['id'])
        else:
            unresolved(stem, [candidate(t, 1.0, 'Exact normalized name.') for t in choices],
                       'Duplicate normalized names require an explicit pairing.', True)
            used.update(track['id'] for track in choices)

    possible, demand = {}, Counter()
    for stem in stems:
        if stem['id'] in resolved:
            continue
        role, key = stem_info[stem['id']]
        choices = [track for track in tracks if track['id'] not in used
                   and role is not None and (role, key) == track_info[track['id']]
                   and _has_identity(key, rules)]
        possible[stem['id']] = choices
        demand.update(track['id'] for track in choices)
    for stem in stems:
        if stem['id'] in resolved:
            continue
        choices = possible[stem['id']]
        if len(choices) == 1 and demand[choices[0]['id']] == 1:
            result['matches'][stem['id']] = candidate(choices[0], .96, 'Unique instrument alias with all descriptive words preserved.')
            used.add(choices[0]['id'])
        elif choices:
            unresolved(stem, [candidate(t, .96, 'Same instrument alias and description.') for t in choices],
                       'Multiple possible pairings require confirmation.', True)
        else:
            role = stem_info[stem['id']][0]
            suggestions = [candidate(t, .35, 'Shared role only; not automatically matched.') for t in tracks
                           if t['id'] not in used and role is not None and role == track_info[t['id']][0]]
            unresolved(stem, suggestions, 'No unique descriptive name match; confirm a track or retain audio timing.')
    return result
