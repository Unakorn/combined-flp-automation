"""Structural end-to-end FL export with explicitly synthetic plugin states.

This proves preservation, native ownership and curve serialization. It makes
no assertion that a real Waves/VST instance has loaded or sounded correct.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import unittest
import tempfile
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'work'))
from flp_raw import encode_fl, read_bytes
from flp_song_analysis import read_project
from plugin_templates import plugin_records
from section_project import preview_section_automation, export_section_automation
from stem_native import parse_points, POINT, LINEAR
from stem_project import inspect_stems, relative_volume
from test_stem_project import audio_fixture, PPQ, END
from test_plugin_templates import fake_spec


class SectionProjectTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.gettempdir()) / ('section-project-test-'+uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        self.source = self.root / 'stems.flp'
        self.original = audio_fixture(self.source)
        self.template = self.root / 'synthetic-template.json'
        self.template.write_text(json.dumps(dict(version=1, families={
            family: fake_spec(family) for family in ('filter', 'width')})))
        info = inspect_stems(self.source)
        self.analysis = dict(info, sections=[], timing_files=[], match_errors=[])
        for stem in self.analysis['stems']:
            stem['analysis'] = dict(sha256=hashlib.sha256(Path(stem['sample']).read_bytes()).hexdigest())
        self.plan = dict(curves=[
            dict(stem_id=0, mixer_insert=1, family='filter',
                 points=[(0, 0), (PPQ*4, -.5), (PPQ*8, 0), (END, 0)]),
            dict(stem_id=0, mixer_insert=1, family='width',
                 points=[(0, 0), (PPQ*8, .5), (END, 0)])],
            effective_intensities=dict(filter=40, width=30, delay=0, reverb=0),
            sections=[], skipped=[], warnings=[])

    def preview(self):
        with patch('section_project.plan_section_motion', return_value=deepcopy(self.plan)):
            return preview_section_automation(self.analysis, {}, self.template)

    def test_end_to_end_preserves_audio_mixer_and_existing_plugin_bytes(self):
        preview = self.preview()
        self.assertFalse(preview['blocked'])
        self.assertEqual([e['slot'] for e in preview['effects']], [1, 2])
        report = export_section_automation(preview)
        after = read_project(report['output'])
        before = read_project(self.source)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertFalse(report['validation']['native_fl_playback_tested'])
        self.assertEqual(report['automation_channels'], 2)
        self.assertEqual(report['plugin_instances_added'], 2)
        for key in (235, 236, 147, 154, 224, 223):
            self.assertEqual([p for k, p in before['events'] if k == key],
                             [p for k, p in after['events'] if k == key])
        for old_channel in before['channels']:
            new_channel = next(c for c in after['channels'] if c['id'] == old_channel['id'])
            self.assertEqual(before['events'][old_channel['start']:old_channel['stop']],
                             after['events'][new_channel['start']:new_channel['stop']])
        old_plugin = plugin_records(before)[1, 0]['fields'][213]
        records = plugin_records(after)
        self.assertEqual(records[1, 0]['fields'][213], old_plugin)
        for effect in preview['effects']:
            payload = records[effect['insert'], effect['slot']]['fields'][213]
            self.assertEqual(hashlib.sha256(payload).hexdigest(), effect['spec']['state_sha256'])
        old_playlist = next(p for k, p in before['events'] if k == 233)
        self.assertTrue(next(p for k, p in after['events'] if k == 233).startswith(old_playlist))

    def test_each_lane_is_one_fullsong_linear_clip_with_real_internal_target(self):
        preview = self.preview()
        after = read_project(export_section_automation(preview)['output'])
        channels = [c for c in after['channels'] if c['type'] == 5]
        links = [struct.unpack('<5I', payload) for key, payload in after['events'] if key == 227]
        for channel, lane in zip(channels, preview['lanes']):
            target = lane['targets'][0]['target']
            self.assertGreaterEqual(target & 0xffff, 0x8000)
            self.assertIn((channel['id'] << 16, 0, target, 8, 469), links)
            fields = dict(after['events'][channel['start']:channel['stop']])
            payload = fields[234]
            self.assertEqual(parse_points(payload, PPQ), lane['points'])
            count = struct.unpack_from('<I', payload, 17)[0]
            for index in range(1, count):
                _, _, tension, flags = POINT.unpack_from(payload, 21+index*POINT.size)
                self.assertEqual((tension, flags), (0.0, LINEAR))
            clips = [c for c in after['clips'] if c.get('channel_id') == channel['id']]
            self.assertEqual([(c['position'], c['duration']) for c in clips], [(0, END)])
        values = {t: v for _, t, v in struct.iter_unpack('<III',
                  next(p for k, p in after['events'] if k == 216))}
        mixer_values = {t: v for _, t, v in struct.iter_unpack('<III',
                        next(p for k, p in after['events'] if k == 225))}
        for lane, effect in zip(preview['lanes'], preview['effects']):
            target = lane['targets'][0]['target']
            self.assertEqual(values[target], effect['spec']['initial_parameter_raw'])
            self.assertNotIn(target, mixer_values)
            prefix = target & 0xffff0000
            self.assertEqual(mixer_values[prefix | 0x1f00], 1)
            self.assertEqual(mixer_values[prefix | 0x1f01], 12800)
        initializer_index = next(i for i, (key, _) in enumerate(after['events']) if key == 216)
        first_link = next(i for i, (key, _) in enumerate(after['events']) if key == 227)
        self.assertLess(initializer_index, first_link)
        self.assertLess(first_link, after['layout']['first_channel'])

    def replace_source_events(self, header, events):
        changed = encode_fl(header, events)
        self.source.write_bytes(changed)
        self.analysis['sha256'] = hashlib.sha256(changed).hexdigest()
        return changed

    def test_existing_native_initializers_preserved_byte_for_byte_in_order(self):
        header, events = read_bytes(self.original)
        first = next(i for i, (key, _) in enumerate(events) if key == 64)
        # Unrelated global and plugin state retain their original order/prefix.
        rows = [(0, 0x70898007, 1056964608), (7, 0x00010000, 12345)]
        original_payload = b''.join(struct.pack('<III', *row) for row in rows)
        events[first:first] = [(146, struct.pack('<I', 0)), (216, original_payload)]
        self.replace_source_events(header, events)
        preview = self.preview()
        after = read_project(export_section_automation(preview)['output'])
        payloads = [payload for key, payload in after['events'] if key == 216]
        self.assertEqual(len(payloads), 1)
        self.assertTrue(payloads[0].startswith(original_payload))
        self.assertEqual(len(payloads[0]), len(original_payload) + 12*len(preview['effects']))
        at = next(i for i, (key, _) in enumerate(after['events']) if key == 216)
        self.assertEqual(after['events'][at-1], (146, struct.pack('<I', 0)))

    def test_new_native_initializer_precedes_saved_links_and_follows_global_settings(self):
        header, events = read_bytes(self.original)
        first = next(i for i, (key, _) in enumerate(events) if key == 64)
        remote = (226, struct.pack('<5I', 253, 0, 0x70878004, 4, 469))
        events[first:first] = [(146, struct.pack('<I', 0)), remote]
        self.replace_source_events(header, events)
        after = read_project(export_section_automation(self.preview())['output'])
        at = next(i for i, (key, _) in enumerate(after['events']) if key == 216)
        self.assertEqual(after['events'][at-1], (146, struct.pack('<I', 0)))
        self.assertEqual(after['events'][at+1], remote)

    def test_stale_native_plugin_initializers_reserve_empty_slots(self):
        header, events = read_bytes(self.original)
        first = next(i for i, (key, _) in enumerate(events) if key == 64)
        stale_rows = [(0, 0x70418005, 1056964608), (0, 0x70428006, 0)]
        payload = b''.join(struct.pack('<III', *row) for row in stale_rows)
        events.insert(first, (216, payload))
        self.replace_source_events(header, events)
        preview = self.preview()
        self.assertEqual([effect['slot'] for effect in preview['effects']], [3, 4])
        after = read_project(export_section_automation(preview)['output'])
        self.assertTrue(next(p for k, p in after['events'] if k == 216).startswith(payload))

    def test_export_rechecks_native_initializer_destination_ownership(self):
        preview = self.preview()
        header, events = read_bytes(self.original)
        first = next(i for i, (key, _) in enumerate(events) if key == 64)
        events.insert(first, (216, struct.pack('<III', 0, 0x70418005, 1056964608)))
        changed = self.replace_source_events(header, events)
        # The exporter must check ownership itself, not trust a supplied plan.
        preview['source_sha256'] = hashlib.sha256(changed).hexdigest()
        with self.assertRaisesRegex(ValueError, 'existing automation owner or initializer'):
            export_section_automation(preview)
        self.assertFalse(Path(preview['output']).exists())

    def test_malformed_or_ambiguous_native_initializers_rejected_before_writing(self):
        row = struct.pack('<III', 0, 0x70818005, 1056964608)
        for case in ('truncated', 'duplicate_event', 'duplicate_target', 'after_channels'):
            header, events = read_bytes(self.original)
            first = next(i for i, (key, _) in enumerate(events) if key == 64)
            if case == 'truncated':
                events.insert(first, (216, row[:-1]))
            elif case == 'duplicate_event':
                events[first:first] = [(216, row), (216, b'')]
            elif case == 'duplicate_target':
                events.insert(first, (216, row+row))
            else:
                events.append((216, row))
            # Start with a valid preview to exercise the independent save guard.
            self.source.write_bytes(self.original)
            self.analysis['sha256'] = hashlib.sha256(self.original).hexdigest()
            preview = self.preview()
            changed = self.replace_source_events(header, events)
            preview['source_sha256'] = hashlib.sha256(changed).hexdigest()
            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, 'initializer.*216'):
                    self.preview()
                with self.assertRaisesRegex(ValueError, 'initializer.*216'):
                    export_section_automation(preview)
                self.assertFalse(Path(preview['output']).exists())

    def test_missing_plugin_calibration_blocks_export_instead_of_volume_fallback(self):
        self.template.write_text(json.dumps(dict(version=1, families={})))
        preview = self.preview()
        self.assertEqual(preview['lanes'], [])
        self.assertEqual(len(preview['blocked']), 2)
        with self.assertRaisesRegex(ValueError, 'effect setup'):
            export_section_automation(preview)
        self.assertFalse(Path(preview['output']).exists())

    def test_changed_template_or_audio_aborts_before_output_creation(self):
        for target in (self.template, Path(self.analysis['stems'][0]['sample'])):
            with self.subTest(target=target):
                before = target.read_bytes()
                preview = self.preview()
                target.write_bytes(before+b' changed')
                with self.assertRaisesRegex(ValueError, 'input changed'):
                    export_section_automation(preview)
                self.assertFalse(Path(preview['output']).exists())
                target.write_bytes(before)

    def test_source_and_existing_output_never_overwritten(self):
        preview = self.preview()
        with self.assertRaises(FileExistsError):
            export_section_automation(preview, self.source)
        destination = self.root / 'keep.flp'
        destination.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            export_section_automation(preview, destination)
        self.assertEqual(destination.read_bytes(), b'keep')
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_plugin_definition_mutation_after_preview_rejected(self):
        preview = self.preview()
        preview['effects'][0]['spec']['target_offset'] += 1
        with self.assertRaisesRegex(ValueError, 'definition changed'):
            export_section_automation(preview)

    def test_routing_with_an_incoming_connection_requires_review(self):
        header, events = read_bytes(self.original)
        # Every synthetic insert currently only points at master. Give any
        # existing insert an incoming connection flag to stem insert1.
        index = next(i for i, (key, _) in enumerate(events) if key == 235)
        events[index] = 235, b'\1\1'
        changed = encode_fl(header, events)
        self.source.write_bytes(changed)
        self.analysis['sha256'] = hashlib.sha256(changed).hexdigest()
        preview = self.preview()
        self.assertTrue(any('incoming' in reason for reason in preview['blocked']))
        self.assertEqual(preview['lanes'], [])

    def test_plain_and_project_macro_audio_paths_keep_same_project_directory(self):
        for reference in ('stems\\snare.wav', '%PROJECTDIR%\\stems\\snare.wav'):
            header, events = read_bytes(self.original)
            index = next(i for i, (key, _) in enumerate(events) if key == 196)
            events[index] = 196, (reference+'\0').encode('utf-16-le')
            changed = encode_fl(header, events)
            self.source.write_bytes(changed)
            self.analysis['sha256'] = hashlib.sha256(changed).hexdigest()
            preview = self.preview()
            destination = self.root / 'elsewhere' / 'moved.flp'
            with self.subTest(reference=reference), self.assertRaisesRegex(ValueError, 'relative audio paths'):
                export_section_automation(preview, destination)
            self.assertFalse(destination.exists())

    def test_slot_ownership_and_stale_plugin_initializers_are_never_reused(self):
        header, events = read_bytes(self.original)
        first = next(i for i, (key, _) in enumerate(events) if key == 64)
        # An empty slot can retain a remote controller owner even without an
        # Automation Clip channel. Its exact link bytes must survive export.
        remote = struct.pack('<5I', 253, 0, 0x70418004, 4, 469)
        events.insert(first, (226, remote))
        init = next(i for i, (key, _) in enumerate(events) if key == 225)
        rows = [(0, 0x70428001, 1234),
                (0, 0x70431f01, 12800), (0, 0x70431f01, 12800)]
        events[init] = 225, events[init][1]+b''.join(struct.pack('<III', *row) for row in rows)
        changed = encode_fl(header, events)
        self.source.write_bytes(changed)
        self.analysis['sha256'] = hashlib.sha256(changed).hexdigest()
        preview = self.preview()
        self.assertEqual([e['slot'] for e in preview['effects']], [4, 5])
        exported = read_project(export_section_automation(preview)['output'])
        self.assertIn((226, remote), exported['events'])
        values = list(struct.iter_unpack('<III', next(p for k, p in exported['events'] if k == 225)))
        for row in rows:
            self.assertEqual(values.count(row), rows.count(row))

    def native_plan(self, *, with_effects=False):
        self.analysis['stems'][0]['role']='keys'
        curves=[dict(stem_id=0,mixer_insert=1,family='volume',parameter='volume_db',
                     points=[(0,0),(PPQ*16,-3),(PPQ*32,-1.5),(END,0)]),
                dict(stem_id=0,mixer_insert=1,family='pan',parameter='pan_percent',
                     points=[(0,0),(PPQ*16,-8),(PPQ*32,8),(END,0)])]
        self.plan['curves']=(self.plan['curves'] if with_effects else [])+curves

    def change_mixer_value(self,target,value):
        header,events=read_bytes(self.source.read_bytes())
        index=next(i for i,(key,_) in enumerate(events) if key==225)
        rows=[(first,address,value if address==target else old)
              for first,address,old in struct.iter_unpack('<III',events[index][1])]
        events[index]=(225,b''.join(struct.pack('<III',*row) for row in rows))
        self.replace_source_events(header,events)

    def test_native_roundtrip_uses_saved_nonunity_fader_and_preexisting_pan(self):
        self.native_plan()
        self.change_mixer_value(0x70401fc1,(-1920)&0xffffffff)
        original=self.source.read_bytes()
        preview=self.preview()
        self.assertEqual(preview['effects'],[])
        self.assertEqual(preview['blocked'],[])
        volume,pan=preview['lanes']
        self.assertEqual(volume['targets'][0]['target'],0x70401fc0)
        self.assertEqual(pan['targets'][0]['target'],0x70401fc1)
        self.assertEqual(volume['points'][0],(0,11000/16000))
        self.assertEqual(volume['points'][-1],(END,11000/16000))
        self.assertAlmostEqual(volume['points'][1][1],relative_volume(11000,-3))
        self.assertTrue(all(value<=11000/16000 for _,value in volume['points']))
        self.assertEqual(pan['points'][0],(0,.35))
        self.assertAlmostEqual(pan['points'][1][1],.31)
        self.assertAlmostEqual(pan['points'][2][1],.39)
        report=export_section_automation(preview)
        self.assertEqual(report['plugin_instances_added'],0)
        before=read_project(self.source);after=read_project(report['output'])
        self.assertEqual(self.source.read_bytes(),original)
        self.assertEqual([row for row in before['events'] if row[0] in (225,216,213,235)],
                         [row for row in after['events'] if row[0] in (225,216,213,235)])
        links=[struct.unpack('<5I',data)[2] for key,data in after['events'] if key==227]
        self.assertEqual(links,[0x70401fc0,0x70401fc1])
        for channel,lane in zip([c for c in after['channels'] if c['type']==5],preview['lanes']):
            fields=dict(after['events'][channel['start']:channel['stop']])
            self.assertEqual(parse_points(fields[234],PPQ),lane['points'])
            count=struct.unpack_from('<I',fields[234],17)[0]
            self.assertTrue(all(POINT.unpack_from(fields[234],21+i*POINT.size)[2:]==(0.,LINEAR)
                                for i in range(1,count)))

    def test_native_only_needs_no_plugin_template_or_free_effect_slots(self):
        self.native_plan()
        self.template.unlink()
        # Slot allocation and template parsing must never be consulted for
        # native controls, even when an unrelated plugin setup is unavailable.
        with (patch('section_project._free_effect_slot',side_effect=AssertionError('No free slots')),
              patch('section_project.load_template',side_effect=AssertionError('No calibration'))):
            preview=self.preview()
            report=export_section_automation(preview)
        self.assertEqual(report['automation_channels'],2)
        self.assertEqual(report['plugin_instances_added'],0)
        self.assertNotIn(str(self.template.resolve()),[item['path'] for item in preview['input_files']])

    def test_native_and_effects_export_together_without_consuming_more_slots(self):
        self.native_plan(with_effects=True)
        preview=self.preview()
        self.assertEqual([effect['slot'] for effect in preview['effects']],[1,2])
        self.assertEqual([lane['control_family'] for lane in preview['lanes']],
                         ['filter','width','volume','pan'])
        report=export_section_automation(preview)
        self.assertEqual(report['plugin_instances_added'],2)
        self.assertEqual(report['automation_channels'],4)

    def test_owned_native_control_skips_only_that_lane_and_retains_effects(self):
        for kind in (226,227,223,216):
            with self.subTest(kind=kind):
                self.source.write_bytes(self.original)
                header,events=read_bytes(self.original)
                first=next(i for i,(key,_) in enumerate(events) if key==64)
                data=(struct.pack('<5I',253,0,0x70401fc0,4,469) if kind in (226,227)
                      else struct.pack('<III',0,0x70401fc0,11000))
                events.insert(first,(kind,data))
                self.replace_source_events(header,events)
                self.plan['curves']=self.plan['curves'][:2]
                self.native_plan(with_effects=True)
                preview=self.preview()
                self.assertEqual(preview['blocked'],[])
                self.assertEqual([lane['control_family'] for lane in preview['lanes']],['filter','width','pan'])
                self.assertTrue(any('Volume:' in item['reason'] for item in preview['skipped']))
                output=self.root/f'owned-{kind}.flp'
                after=read_project(export_section_automation(preview,output)['output'])
                self.assertIn((kind,data),after['events']) if kind!=216 else self.assertTrue(
                    next(p for k,p in after['events'] if k==216).startswith(data))

    def test_duplicate_native_initsong_values_preserved_and_native_control_skipped(self):
        self.native_plan(with_effects=True)
        header,events=read_bytes(self.original)
        first=next(i for i,(key,_) in enumerate(events) if key==64)
        payload=struct.pack('<III',0,0x70401fc0,11000)*2
        events.insert(first,(216,payload))
        self.replace_source_events(header,events)
        preview=self.preview()
        self.assertEqual(preview['blocked'],[])
        self.assertNotIn('volume',[lane['control_family'] for lane in preview['lanes']])
        after=read_project(export_section_automation(preview)['output'])
        self.assertTrue(next(p for k,p in after['events'] if k==216).startswith(payload))

    def test_absent_duplicate_and_invalid_saved_values_skip_only_affected_control(self):
        self.native_plan()
        for case in ('absent','duplicate','invalid_volume','invalid_pan'):
            with self.subTest(case=case):
                header,events=read_bytes(self.original)
                index=next(i for i,(key,_) in enumerate(events) if key==225)
                rows=list(struct.iter_unpack('<III',events[index][1]))
                if case=='absent':rows=[row for row in rows if row[1]!=0x70401fc0]
                elif case=='duplicate':rows.append((0,0x70401fc0,11000))
                else:
                    target=0x70401fc0 if case=='invalid_volume' else 0x70401fc1
                    rows=[(first,address,16001 if case=='invalid_volume' else 6401)
                          if address==target else (first,address,value) for first,address,value in rows]
                events[index]=(225,b''.join(struct.pack('<III',*row) for row in rows))
                self.replace_source_events(header,events)
                preview=self.preview()
                self.assertEqual(preview['blocked'],[])
                self.assertEqual([lane['control_family'] for lane in preview['lanes']],
                                 ['volume'] if case=='invalid_pan' else ['pan'])
                self.assertEqual(len(preview['skipped']),1)

    def test_quiet_or_zero_saved_fader_never_becomes_louder(self):
        self.native_plan()
        for saved in (0,1,53):
            with self.subTest(saved=saved):
                self.change_mixer_value(0x70401fc0,saved)
                preview=self.preview()
                self.assertEqual([lane['control_family'] for lane in preview['lanes']],['pan'])
                self.assertEqual(preview['blocked'],[])

    def test_native_pan_limits_clamp_relative_motion_and_preserve_saved_position(self):
        self.native_plan()
        for saved in (-6400,6400):
            with self.subTest(saved=saved):
                self.change_mixer_value(0x70401fc1,saved&0xffffffff)
                lane=next(lane for lane in self.preview()['lanes'] if lane['control_family']=='pan')
                baseline=(saved+6400)/12800
                self.assertEqual(lane['points'][0],(0,baseline))
                self.assertEqual(lane['points'][-1],(END,baseline))
                self.assertTrue(all(0<=value<=1 for _,value in lane['points']))
                self.assertLessEqual(max(abs(value-baseline) for _,value in lane['points']),.040000001)

    def test_native_export_revalidates_existing_owner_and_saved_baseline(self):
        self.native_plan()
        for case in ('owner','initsong','fader','pan'):
            with self.subTest(case=case):
                self.source.write_bytes(self.original)
                self.analysis['sha256']=hashlib.sha256(self.original).hexdigest()
                preview=self.preview()
                if case in ('owner','initsong'):
                    header,events=read_bytes(self.original)
                    first=next(i for i,(key,_) in enumerate(events) if key==64)
                    row=(227,struct.pack('<5I',253,0,0x70401fc0,4,469)) if case=='owner' else (
                        216,struct.pack('<III',0,0x70401fc0,11000))
                    events.insert(first,row)
                    self.replace_source_events(header,events)
                else:self.change_mixer_value(0x70401fc0 if case=='fader' else 0x70401fc1,2000)
                preview['source_sha256']=self.analysis['sha256']
                with self.assertRaises(ValueError):export_section_automation(preview)
                self.assertFalse(Path(preview['output']).exists())

    def test_native_export_rejects_modified_target_points_or_timeline(self):
        self.native_plan()
        for case in ('target','points','baseline','parameter','family','mixer','timeline','boost','end'):
            with self.subTest(case=case):
                preview=self.preview();lane=preview['lanes'][0];target=lane['targets'][0]
                if case=='target':target['target']=0x70801fc0
                elif case=='points':lane['points'][1]=(PPQ*16,.1)
                elif case=='baseline':target['baseline']=.8
                elif case=='parameter':lane['parameter']='pan_percent'
                elif case=='family':lane['control_family']='filter'
                elif case=='mixer':target['mixer_insert']=2
                elif case=='timeline':preview['end_tick']+=PPQ
                elif case=='boost':lane['moves'][0]['points'][1]=(PPQ*16,1)
                else:lane['moves'][0]['points'][-1]=(END,-1)
                with self.assertRaises(ValueError):export_section_automation(preview)
                self.assertFalse(Path(preview['output']).exists())

    def test_exported_native_controls_are_not_automated_twice(self):
        self.native_plan()
        after=Path(export_section_automation(self.preview())['output'])
        info=inspect_stems(after)
        self.analysis.update(path=info['path'],sha256=info['sha256'])
        preview=self.preview()
        self.assertEqual(preview['lanes'],[])
        self.assertEqual(preview['blocked'],[])
        self.assertEqual(len(preview['skipped']),2)

    def test_native_protection_and_vocal_limits_preserve_foreground_controls(self):
        self.native_plan()
        for role,protected in (('kick',False),('keys',True),('vocals',False),('hat',False)):
            with self.subTest(role=role,protected=protected):
                self.analysis['stems'][0].update(role=role,automation_protected=protected)
                preview=self.preview()
                expected=['pan'] if role=='hat' else []
                self.assertEqual([lane['control_family'] for lane in preview['lanes']],expected)
                self.assertEqual(preview['blocked'],[])

    def test_native_invalid_points_are_skipped_without_harming_existing_effects(self):
        self.native_plan(with_effects=True)
        for invalid in ([(0,0),(END,float('nan'))],[(0,0),(PPQ,-3.1),(END,0)],
                        [(0,0),(PPQ,0.1),(END,0)],[(0,0),(PPQ,-1),(PPQ,-2),(END,0)]):
            with self.subTest(invalid=invalid):
                self.plan['curves'][2]['points']=invalid
                preview=self.preview()
                self.assertEqual([lane['control_family'] for lane in preview['lanes']],['filter','width','pan'])
                self.assertEqual(preview['blocked'],[])

    def test_native_pan_protects_underscore_separated_foreground_names(self):
        self.native_plan()
        for name in ('1_lead','main_keys','synth_solo_2'):
            with self.subTest(name=name):
                self.analysis['stems'][0]['name']=name
                preview=self.preview()
                self.assertEqual([lane['control_family'] for lane in preview['lanes']],['volume'])
                self.assertTrue(any('foreground' in item['reason'] for item in preview['skipped']))
                self.assertEqual(preview['blocked'],[])


if __name__ == '__main__':
    unittest.main()
