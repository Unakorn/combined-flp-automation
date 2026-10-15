"""Portable native-stem FLP export integration; native DAW playback is separate."""
from pathlib import Path
import hashlib
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work'))
from flp_raw import encode_fl,read_bytes
from flp_song_analysis import read_project
from stem_native import build_playlist_clip,parse_points
from stem_project import export_automation,inspect_stems,preview_automation,relative_volume
from flp_connector import _effects
from drum_template import _mixer

PPQ=96
END=PPQ*4*17


def text(value):return (value+'\0').encode('utf-16-le')


def audio_fixture(path):
    events=[(199,b'26.1.7.5653\0'),(156,struct.pack('<I',170000)),
            (17,b'\4'),(18,b'\4'),(231,text('All'))]
    for cid,name in enumerate(('1_808 Snare','1_808 Tom')):
        audio=path.parent/(name+'.wav');audio.write_bytes(b'not decoded in structural export tests')
        events += [(64,struct.pack('<H',cid)),(21,b'\4'),(0,b'\1'),(203,text(name)),
                   (104,struct.pack('<H',cid+1)),(196,text(str(audio))),
                   (213,b'original opaque audio-channel state')]
    events += [(99,b'\0\0'),(233,b''.join(build_playlist_clip(cid,0,END,cid,cid+1) for cid in (0,1)))]
    for number in range(1,8):
        track=bytearray(66);struct.pack_into('<I',track,0,number);track[12]=int(number<=2)
        events.append((238,bytes(track)))
    events.append((103,b'\4\0'))
    initial=[]
    for insert in range(4):
        prefix=0x70000000|insert<<22
        events += [(235,b'\1'),(154,b'\xff'*4),(42,b'\0'),(236,bytes.fromhex('000000004c00000000000000'))]
        initial += [(0,prefix|0x1fc0,11000),(0,prefix|0x1fc1,0)]
        for slot in range(10):
            if insert==1 and slot==0:
                wrapper=bytearray(52);struct.pack_into('<II',wrapper,0,insert,slot)
                events += [(201,text('Fruity Wrapper')),(212,bytes(wrapper)),(213,b'original third-party plugin state')]
                initial += [(0,prefix|0x1f00,1),(0,prefix|0x1f01,12800)]
            events.append((98,struct.pack('<H',slot)))
        events.append((147,b'\xff'*4))
    events.append((225,b''.join(struct.pack('<III',*row) for row in initial)))
    data=encode_fl(struct.pack('<HHH',0,2,PPQ),events);path.write_bytes(data);return data


class NativeStemExportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'source.flp';self.output=self.root/'source_automated.flp'
        self.original=audio_fixture(self.source)
        self.targets=[0x70401fc0,0x70801fc0]
        self.preview=dict(source=str(self.source),source_sha256=hashlib.sha256(self.original).hexdigest(),
            ppq=PPQ,end_tick=END,bpm=170,output=str(self.output),warnings=[],
            lanes=[dict(name='Drums - Shared section ride',family='Drums',parameter='volume_db',
                        points=[(0,.6875),(PPQ*32,.5),(PPQ*64,.6875),(END,.6875)],
                        targets=[dict(target=t) for t in self.targets])])

    def test_audio_only_without_pattern_metadata_loads(self):
        project=read_project(self.source)
        self.assertFalse(project['patterns'])
        info=inspect_stems(self.source)
        self.assertEqual(len(info['stems']),2)
        self.assertEqual([s['role'] for s in info['stems']],['snare','tom'])
        self.assertEqual(info['end_tick'],END)

    def test_export_preserves_originals_and_creates_real_grouped_native_clips(self):
        report=export_automation(self.preview)
        before=read_project(self.source);after=read_project(self.output)
        self.assertEqual(self.source.read_bytes(),self.original)
        original_states=[p for k,p in before['events'] if k==213]
        self.assertEqual([p for k,p in after['events'] if k==213],original_states)
        self.assertFalse(any(k in (223,224) for k,p in after['events']))
        channels=[c for c in after['channels'] if c['type']==5]
        self.assertEqual(len(channels),1)
        cid=channels[0]['id']
        links=[struct.unpack('<5I',p) for k,p in after['events'] if k==227]
        self.assertEqual(links,[(cid<<16,0,t,8,469) for t in self.targets])
        fields=dict(after['events'][channels[0]['start']:channels[0]['stop']])
        self.assertEqual(parse_points(fields[234],PPQ),self.preview['lanes'][0]['points'])
        self.assertEqual(len(fields[234]),21+4*24+112)
        old_playlist=next(p for k,p in before['events'] if k==233)
        new_playlist=next(p for k,p in after['events'] if k==233)
        self.assertTrue(new_playlist.startswith(old_playlist))
        self.assertEqual(next(p for k,p in before['events'] if k==225),next(p for k,p in after['events'] if k==225))
        self.assertFalse(report['validation']['native_fl_playback_tested'])
        self.assertEqual(report['automation_channels'],1)
        self.assertEqual(report['automation_links'],2)

    def test_each_native_lane_has_one_continuous_full_song_placement(self):
        export_automation(self.preview)
        project=read_project(self.output)
        clips=[c for c in project['clips'] if c['channel_id']==2]
        self.assertEqual([(c['position'],c['duration'],c['track']) for c in clips],
                         [(0,END,3)])
        playlist=next(p for k,p in project['events'] if k==233)
        self.assertEqual([struct.unpack_from('<ff',playlist,c['record_offset']+24) for c in clips],
                         [(0.,68.)])
        self.assertTrue(project['tracks'][3]['enabled'])
        self.assertFalse(project['tracks'][4]['enabled'])

    def test_source_and_existing_outputs_cannot_be_overwritten(self):
        with self.assertRaises(FileExistsError):export_automation(self.preview,self.source)
        self.output.write_bytes(b'keep me')
        with self.assertRaises(FileExistsError):export_automation(self.preview)
        self.assertEqual(self.output.read_bytes(),b'keep me')
        self.assertEqual(self.source.read_bytes(),self.original)

    def test_changed_source_fails_before_writing(self):
        h,e=read_bytes(self.original);e.append((200,b'changed\0\0'))
        self.source.write_bytes(encode_fl(h,e))
        with self.assertRaisesRegex(ValueError,'changed'):export_automation(self.preview)
        self.assertFalse(self.output.exists())

    def planned_preview(self, moves):
        info=inspect_stems(self.source)
        section=dict(type='bridge',start_tick=0,end_tick=END,bar=1,energy=.6,intensity='Medium')
        analysis=dict(info,half_time=False,sections=[section])
        with patch('stem_project.plan_moves',return_value=dict(moves=moves,sections=[section],warnings=[])):
            return preview_automation(analysis,'Rock','Alt Rock')

    def move(self, cid=0, name='Bridge texture swell', start_value=0, end_value=1.2):
        return dict(name=name,stem_id=cid,stem_name=('1_808 Snare','1_808 Tom')[cid],
                    mixer_insert=cid+1,plugin='Mixer',parameter='volume_db',shape='swell',
                    start_tick=384,end_tick=768,start_value=start_value,end_value=end_value,
                    family='Drums',section_start_tick=0)

    def test_positive_db_moves_lift_relative_to_saved_fader_and_stay_bounded(self):
        self.assertEqual(relative_volume(11727,0),11727/16000)
        self.assertGreater(relative_volume(12800,1.2),12800/16000)
        self.assertLess(relative_volume(12800,-2),12800/16000)
        self.assertLessEqual(relative_volume(15900,3),1)
        self.assertEqual(relative_volume(16000,3),1)
        self.assertEqual(relative_volume(0,3),0)
        preview=self.planned_preview([self.move()])
        self.assertEqual(len(preview['lanes']),1)
        lane=preview['lanes'][0]
        self.assertGreater(max(v for _,v in lane['points']),11000/16000)
        self.assertEqual(lane['points'][0][1],11000/16000)
        self.assertEqual(lane['points'][-1][1],11000/16000)

    def test_positive_move_at_fader_ceiling_is_skipped_visibly(self):
        header,events=read_bytes(self.original)
        for index,(key,data) in enumerate(events):
            if key!=225:continue
            rows=[(tick,target,16000 if target==self.targets[0] else value)
                  for tick,target,value in struct.iter_unpack('<III',data)]
            events[index]=(225,b''.join(struct.pack('<III',*row) for row in rows))
        self.source.write_bytes(encode_fl(header,events))
        preview=self.planned_preview([self.move()])
        self.assertEqual(preview['lanes'],[])
        self.assertEqual(len(preview['skipped']),1)
        self.assertIn('flat',preview['skipped'][0]['reason'])

    def test_grouped_preview_summary_names_every_affected_stem(self):
        preview=self.planned_preview([self.move(0),self.move(1)])
        self.assertEqual(len(preview['lanes']),1)
        self.assertEqual(len(preview['lanes'][0]['targets']),2)
        self.assertEqual(len(preview['lanes'][0]['moves']),2)
        self.assertIn('1_808 Snare',preview['summary'][0])
        self.assertIn('1_808 Tom',preview['summary'][0])

    def test_distinct_saved_faders_do_not_share_an_incorrect_native_curve(self):
        header,events=read_bytes(self.original)
        for index,(key,data) in enumerate(events):
            if key!=225:continue
            rows=[(tick,target,12800 if target==self.targets[1] else value)
                  for tick,target,value in struct.iter_unpack('<III',data)]
            events[index]=(225,b''.join(struct.pack('<III',*row) for row in rows))
        self.source.write_bytes(encode_fl(header,events))
        preview=self.planned_preview([self.move(0),self.move(1)])
        self.assertEqual(len(preview['lanes']),2)
        self.assertTrue(all(len(lane['targets'])==1 for lane in preview['lanes']))

    def effect_move(self, cid=0):
        return dict(self.move(cid,name='Phrase room blend',start_value=0,end_value=25),
                    plugin='Fruity Reeverb 2',parameter='effect_blend_percent',
                    shape='ramp_up',release_ticks=96,release_end_tick=864)

    def test_stock_effect_uses_empty_slot_and_preserves_existing_chain(self):
        preview=self.planned_preview([self.effect_move()])
        self.assertEqual(len(preview['effects']),1)
        self.assertEqual(preview['effects'][0]['slot'],1)
        self.assertEqual(preview['coverage']['gestures_rendered'],1)
        self.assertEqual(preview['coverage']['gestures_skipped'],0)
        self.assertEqual(preview['coverage']['stems_affected'],1)
        target=0x70411f01
        lane=preview['lanes'][0]
        self.assertEqual(lane['targets'][0]['target'],target)
        self.assertEqual(lane['points'][0],(0,0.))
        self.assertIn((768,.25),lane['points'])
        self.assertIn((864,0.),lane['points'])
        report=export_automation(preview,self.output)
        after=read_project(self.output);before=read_project(self.source)
        blocks,_=_mixer(after['events'])
        fx=_effects(after['events'],blocks)
        self.assertEqual([(e['slot'],e['plugin']) for e in fx],[(0,'Fruity Wrapper'),(1,'Fruity Reeverb 2')])
        self.assertIn((213,b'original third-party plugin state'),after['events'])
        old_initial={t:v for _,t,v in struct.iter_unpack('<III',next(p for k,p in before['events'] if k==225))}
        new_initial={t:v for _,t,v in struct.iter_unpack('<III',next(p for k,p in after['events'] if k==225))}
        for t,v in old_initial.items():self.assertEqual(new_initial[t],v)
        self.assertEqual(new_initial[target],0)
        self.assertEqual(new_initial[target-1],1)
        self.assertFalse(any(k in (223,224) for k,p in after['events']))
        self.assertEqual(report['stock_effects_added'],1)

    def test_same_effect_is_reused_across_gestures_with_one_owner(self):
        first=self.effect_move()
        second=dict(first,start_tick=900,end_tick=1200,release_end_tick=1296)
        preview=self.planned_preview([first,second])
        self.assertEqual(len(preview['effects']),1)
        self.assertEqual(len(preview['lanes']),1)
        self.assertEqual(len(preview['lanes'][0]['moves']),2)
        self.assertEqual(len(preview['lanes'][0]['targets']),1)

    def test_existing_owner_on_empty_slot_is_never_reused(self):
        header,events=read_bytes(self.original)
        first=next(i for i,(k,_) in enumerate(events) if k==64)
        events.insert(first,(227,struct.pack('<5I',99<<16,0,0x70411f01,8,469)))
        self.source.write_bytes(encode_fl(header,events))
        preview=self.planned_preview([self.effect_move()])
        self.assertEqual(preview['effects'][0]['slot'],2)

    def test_unsupported_effect_never_gets_silently_substituted(self):
        move=dict(self.effect_move(),parameter='wet_percent')
        preview=self.planned_preview([move])
        self.assertFalse(preview['lanes'])
        self.assertFalse(preview['effects'])
        self.assertEqual(preview['coverage']['gestures_skipped'],1)
        self.assertIn('no verified native parameter',preview['skipped'][0]['reason'])

    def test_full_or_reserved_chain_keeps_mixer_moves(self):
        header,events=read_bytes(self.original)
        first=next(i for i,(k,_) in enumerate(events) if k==64)
        links=[(227,struct.pack('<5I',99<<16,0,0x70400000|slot<<16|0x1f01,8,469))
               for slot in range(1,10)]
        events[first:first]=links
        self.source.write_bytes(encode_fl(header,events))
        preview=self.planned_preview([self.effect_move(),self.move()])
        self.assertEqual(preview['effects'],[])
        self.assertEqual(len(preview['lanes']),1)
        self.assertEqual(preview['lanes'][0]['parameter'],'volume_db')
        self.assertIn('no unused effect slot',preview['skipped'][0]['reason'])

    def test_catalog_change_requires_new_preview(self):
        preview=self.planned_preview([self.effect_move()])
        with patch('stem_project.get_effect_spec',return_value=None):
            with self.assertRaisesRegex(ValueError,'definitions changed'):
                export_automation(preview,self.output)
        self.assertFalse(self.output.exists())

    def test_release_cannot_overwrite_the_next_blend_gesture(self):
        first=self.effect_move()
        second=dict(first,start_tick=800,end_tick=1100,release_end_tick=1196)
        preview=self.planned_preview([first,second])
        points=dict(preview['lanes'][0]['points'])
        self.assertIn(799,points)
        self.assertEqual(points[799],0)
        self.assertNotIn(864,points)

    def test_adjacent_gestures_preserve_each_peak(self):
        first=self.effect_move()
        second=dict(first,start_tick=768,end_tick=1152,release_end_tick=1248)
        preview=self.planned_preview([first,second])
        points=dict(preview['lanes'][0]['points'])
        self.assertEqual(points[767],.25)
        self.assertEqual(points[768],0.)
        self.assertEqual(points[1151],.25)

    def test_flat_effect_does_not_reserve_last_slot_needed_by_audible_effect(self):
        header,events=read_bytes(self.original)
        first=next(i for i,(k,_) in enumerate(events) if k==64)
        # Slot 0 contains the user's plugin, slots 2–9 have existing owners.
        # Slot 1 is the only empty and unowned slot available to this preview.
        events[first:first]=[(227,struct.pack('<5I',99<<16,0,0x70400000|slot<<16|0x1f01,8,469))
                             for slot in range(2,10)]
        self.source.write_bytes(encode_fl(header,events))
        flat=dict(self.effect_move(),plugin='Fruity Filter',parameter='lowpass_blend_percent',
                  name='Inaudible hold',shape='hold',start_value=0,end_value=25)
        preview=self.planned_preview([flat,self.effect_move()])
        self.assertEqual(len(preview['skipped']),1)
        self.assertIn('flat',preview['skipped'][0]['reason'])
        self.assertEqual([(effect['plugin'],effect['slot']) for effect in preview['effects']],
                         [('Fruity Reeverb 2',1)])
        self.assertEqual(len(preview['lanes']),1)
        self.assertEqual(preview['lanes'][0]['targets'][0]['target'],0x70411f01)
        self.assertEqual(preview['coverage']['gestures_rendered'],1)


if __name__=='__main__':unittest.main()
