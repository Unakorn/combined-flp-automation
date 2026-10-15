"""Vocabulary, conservative identity matching, and visible ambiguity."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'work'))
from stem_names import classify_name, detect_name_role, load_name_rules, match_stem_names, normalized_name


class NameRoleTests(unittest.TestCase):
    def test_user_sample_pack_names(self):
        examples = {'1_808 OH': 'hat', '1_808 CH': 'hat', '1_808 Kick': 'kick',
                    '1_808 Snare': 'snare', '1_808 Tom': 'tom', '1_Alum Snare 2': 'snare',
                    '1_BAS Rocker': 'bass', '1_BRS Horns Sustain': 'brass',
                    '1_GTR Acoustic Padded': 'guitar', '1_GTR Acoustic': 'guitar',
                    '1_KEY Electric Piano Bright': 'keys'}
        for name, expected in examples.items():
            self.assertEqual(detect_name_role(name), expected, name)

    def test_codes_precede_drum_words_and_matching_uses_whole_tokens(self):
        self.assertEqual(detect_name_role('1_GTR Kick Padded'), 'guitar')
        self.assertEqual(detect_name_role('VOX Snare Adlib'), 'vocals')
        self.assertIsNone(detect_name_role('Keynote speaker'))
        self.assertIsNone(detect_name_role('Guitarist padded hammer'))
        self.assertIsNone(detect_name_role('Something strange'))

    def test_808_alone_never_means_bass_and_drum_words_win(self):
        for name in ['808', '1_808', '808 Deep', '808 Long']:
            detail = classify_name(name)
            self.assertIsNone(detail['role'], name)
            self.assertEqual(detail['confidence'], 0)
        for name in ['808 Bass', '1_808 Sub', '808 Low']:
            self.assertEqual(detect_name_role(name), 'bass', name)
        self.assertEqual(detect_name_role('808 Bass Drum'), 'kick')

    def test_common_instrument_and_drum_aliases(self):
        examples = {'Rhodes Bright': 'keys', 'Wurlitzer Soft': 'keys', 'PNO Grand': 'keys',
                    'Electric Bass': 'bass', 'Upright Bass': 'bass', 'Synth Bass': 'bass',
                    'Lead Guitar': 'guitar', 'String Pad': 'pads', 'Snare Rimshot': 'snare',
                    'Open Hi-Hat': 'hat', 'Tambourine Loop': 'perc', 'Conga loop 02': 'perc',
                    'Trumpet Section': 'brass', 'Saxophone solo': 'brass', 'BGV Soft': 'vocals',
                    'A Cappella': 'vocals', 'Pluck Bright': 'synth', 'White Noise': 'effects',
                    'Bass Drum': 'kick', 'BD Punch': 'kick'}
        for name, expected in examples.items():
            self.assertEqual(detect_name_role(name), expected, name)

    def test_conflicts_and_generic_descriptions_are_reviewable(self):
        detail = classify_name('GTR SYN Hybrid')
        self.assertIsNone(detail['role'])
        self.assertEqual(set(detail['candidates']), {'guitar', 'synth'})
        detail = classify_name('Snare Clap Layer')
        self.assertIsNone(detail['role'])
        self.assertEqual(set(detail['candidates']), {'snare', 'clap'})
        detail = classify_name('Orchestra Wide')
        self.assertIsNone(detail['role'])
        self.assertTrue(detail['candidates'])

    def test_vocabulary_is_an_editable_validated_file(self):
        rules = deepcopy(load_name_rules())
        rules['category_codes']['NEW'] = 'brass'
        path = Path(tempfile.gettempdir()) / f'stem-name-rules-{uuid4().hex}.json'
        try:
            path.write_text(json.dumps(rules), encoding='utf-8')
            self.assertEqual(load_name_rules(path)['category_codes']['NEW'], 'brass')
            rules['category_codes']['NEW'] = 'invented-role'
            path.write_text(json.dumps(rules), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'category code'):
                load_name_rules(path)
        finally:
            path.unlink(missing_ok=True)


class NormalizationTests(unittest.TestCase):
    def test_exporter_prefixes_suffixes_case_and_paths(self):
        for name in [r'C:\stems\1_GTR Acoustic (consolidated).wav',
                     'Track 01 - GTR Acoustic [Rendered].flac',
                     'Stem 03 - GTR Acoustic_bounced_2.aiff',
                     '/tmp/01_02_GTR Acoustic - exported.wav']:
            self.assertEqual(normalized_name(name), 'gtr acoustic', name)

    def test_model_numbers_and_meaningful_sound_variants_are_preserved(self):
        self.assertEqual(normalized_name('1_808 OH.wav'), '808 oh')
        self.assertEqual(normalized_name('909 Kick'), '909 kick')
        self.assertEqual(normalized_name('1_Alum Snare 2.wav'), 'alum snare 2')
        self.assertNotEqual(normalized_name('1_GTR Acoustic Padded'), normalized_name('GTR Acoustic'))
        self.assertNotEqual(normalized_name('Keys Dry'), normalized_name('Keys Wet'))
        self.assertEqual(normalized_name('Bass v2.1'), 'bass v2 1')


class NameMatchingTests(unittest.TestCase):
    def test_exact_project_names_match_without_numeric_export_prefix(self):
        stems = [dict(id=1, name='1_GTR Acoustic Padded'), dict(id=2, name='1_GTR Acoustic'),
                 dict(id=3, name='1_KEY Electric Piano Bright')]
        tracks = [dict(id=10, name='GTR Acoustic'), dict(id=11, name='KEY Electric Piano Bright'),
                  dict(id=12, name='GTR Acoustic Padded')]
        original = deepcopy((stems, tracks))
        result = match_stem_names(stems, tracks)
        self.assertEqual({key: value['track_id'] for key, value in result['matches'].items()}, {1: 12, 2: 10, 3: 11})
        self.assertFalse(result['unmatched'] or result['ambiguous'])
        self.assertEqual((stems, tracks), original)

    def test_aliases_match_only_when_description_and_instrument_agree(self):
        stems = [dict(id=1, name='GTR Acoustic Padded'), dict(id=2, name='KEY Electric Piano Bright'),
                 dict(id=3, name='808 OH')]
        tracks = [dict(id=10, name='Acoustic Guitar Padded'), dict(id=11, name='EP Bright'),
                  dict(id=12, name='808 Open Hat')]
        result = match_stem_names(stems, tracks)
        self.assertEqual({key: value['track_id'] for key, value in result['matches'].items()}, {1: 10, 2: 11, 3: 12})
        self.assertTrue(all(value['confidence'] >= .9 for value in result['matches'].values()))

    def test_same_role_is_a_suggestion_never_an_automatic_pair(self):
        result = match_stem_names([dict(id=1, name='GTR Acoustic Padded')], [dict(id=2, name='GTR Acoustic')])
        self.assertFalse(result['matches'])
        self.assertEqual(result['unmatched'][0]['candidates'][0]['track_id'], 2)
        self.assertIn('role only', result['unmatched'][0]['candidates'][0]['reason'])
        result = match_stem_names([dict(id=1, name='GTR')], [dict(id=2, name='Guitar')])
        self.assertFalse(result['matches'])

    def test_duplicate_names_do_not_pair_by_arbitrary_order(self):
        result = match_stem_names([dict(id=1, name='1_GTR Acoustic'), dict(id=2, name='2_GTR Acoustic')],
                                 [dict(id=10, name='GTR Acoustic')])
        self.assertFalse(result['matches'])
        self.assertEqual(len(result['ambiguous']), 2)
        result = match_stem_names([dict(id=1, name='GTR Acoustic')],
                                 [dict(id=10, name='GTR Acoustic'), dict(id=11, name='GTR Acoustic')])
        self.assertFalse(result['matches'])
        self.assertEqual(len(result['ambiguous'][0]['candidates']), 2)

    def test_alias_competition_and_mismatched_model_number_stay_unmatched(self):
        result = match_stem_names([dict(id=1, name='GTR Acoustic'), dict(id=2, name='Guit Acoustic')],
                                 [dict(id=10, name='Acoustic Guitar')])
        self.assertFalse(result['matches'])
        self.assertEqual(len(result['ambiguous']), 2)
        result = match_stem_names([dict(id=1, name='808 Kick')], [dict(id=2, name='909 Kick')])
        self.assertFalse(result['matches'])

    def test_unique_unknown_exact_name_can_match_without_inventing_role(self):
        result = match_stem_names([dict(id='stem-a', name='1_Mystery Bright')], [dict(id='track-a', name='Mystery Bright')])
        self.assertEqual(result['matches']['stem-a']['track_id'], 'track-a')
        self.assertIsNone(detect_name_role('Mystery Bright'))

    def test_duplicate_ids_and_empty_names_cannot_create_false_pairings(self):
        with self.assertRaisesRegex(ValueError, 'unique ID'):
            match_stem_names([dict(id=1, name='Keys'), dict(id=1, name='Guitar')], [])
        result = match_stem_names([dict(id=1, name='')], [dict(id=2, name='')])
        self.assertFalse(result['matches'])


if __name__ == '__main__':
    unittest.main()
