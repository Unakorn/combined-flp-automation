"""Eight-child drum Layer timing guides, without private songs or DAW control."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'work'))
from flp_raw import encode_fl
from flp_song_analysis import analyze_flp, read_project, _single_pattern_clip

PPQ=96
END=PPQ*4*8
NAMES=('808 Kick','808 Snare','Alum Snare 2','808 CH','808 OH','808 Tom','808 Ride','808 Clap')


def text(value):
    return (value+'\0').encode('utf-16-le')


def layer_fixture(path, used=8, flags=0, enabled=True, direct=False):
    events=[(199,b'26.1.7.5653\0'),(156,struct.pack('<I',170000)),
            (17,b'\4'),(18,b'\4'),(231,text('All'))]
    notes=[]
    for index in range(used):
        for tick in (index*PPQ,(8+index)*PPQ):
            notes.append(struct.pack('<IHHIHHHHBBBB',tick,0x4000,0,24,60+index,0,120,0,64,100,128,128))
    if direct:
        notes.append(struct.pack('<IHHIHHHHBBBB',24*PPQ,0x4000,1,48,60,0,120,0,64,90,128,128))
    events += [(65,b'\1\0'),(224,b''.join(notes))]
    events += [(64,b'\0\0'),(21,b'\3'),(0,b'\1'),(203,text('Drum Layer')),
               (135,struct.pack('<I',60)),(144,struct.pack('<I',flags))]
    events += [(94,struct.pack('<H',cid)) for cid in range(1,9)]
    for index,name in enumerate(NAMES):
        key=60+index
        misc=bytearray(76);struct.pack_into('<II',misc,68,key,key)
        events += [(64,struct.pack('<H',index+1)),(21,b'\0'),(0,b'\1' if enabled or index!=7 else b'\0'),
                   (203,text(name)),(104,struct.pack('<H',index+1)),(135,struct.pack('<I',60)),
                   (215,bytes(misc)),(213,b'unchanged synthetic sampler settings')]
    events += [(65,b'\1\0'),(193,text('Eight drum parts')),(99,b'\0\0'),
               (233,_single_pattern_clip({'id':1},END))]
    track=bytearray(66);struct.pack_into('<I',track,0,1);track[12]=1
    events.append((238,bytes(track)))
    data=encode_fl(struct.pack('<HHH',0,9,PPQ),events)
    path.write_bytes(data)
    return data


class LayerStemTimingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.source=Path(self.temp.name)/'source.flp'

    def test_eight_children_get_only_their_own_layer_notes(self):
        original=layer_fixture(self.source)
        model=analyze_flp(self.source)
        self.assertEqual(model['instrument_count'],8)
        self.assertEqual(model['note_count'],16)
        self.assertEqual(model['end_tick'],END)
        self.assertEqual({v['name'] for v in model['voices']},set(NAMES))
        for index,voice in enumerate(model['voices']):
            self.assertEqual(voice['channel_id'],index+1)
            self.assertEqual(voice['source_layer_ids'],[0])
            self.assertEqual({n['note'] for n in voice['notes']},{60+index})
            self.assertEqual([n['tick'] for n in voice['notes']],[index*PPQ,(8+index)*PPQ])
        self.assertEqual(self.source.read_bytes(),original)

    def test_eight_layer_slots_allow_two_unused_drums(self):
        original=layer_fixture(self.source,used=6)
        project=read_project(self.source)
        layer=next(c for c in project['channels'] if c['type']==3)
        self.assertEqual(layer['children'],list(range(1,9)))
        model=analyze_flp(self.source)
        self.assertEqual([v['channel_id'] for v in model['voices']],list(range(1,7)))
        self.assertEqual(model['note_count'],12)
        self.assertEqual(self.source.read_bytes(),original)

    def test_direct_child_notes_are_combined_without_cross_triggering(self):
        layer_fixture(self.source,direct=True)
        model=analyze_flp(self.source)
        self.assertEqual(model['voices'][0]['note_count'],3)
        self.assertTrue(all(v['note_count']==2 for v in model['voices'][1:]))

    def test_disabled_child_is_not_in_timing_guide(self):
        layer_fixture(self.source,enabled=False)
        model=analyze_flp(self.source)
        self.assertEqual([v['channel_id'] for v in model['voices']],list(range(1,8)))

    def test_random_and_crossfade_layer_modes_fail_explicitly(self):
        for flags in (1,2,3):
            layer_fixture(self.source,flags=flags)
            with self.assertRaisesRegex(ValueError,'random or crossfade'):
                analyze_flp(self.source)

    def test_named_audio_stems_receive_eight_individual_source_guides(self):
        from stem_midi import read_source_timing
        original=layer_fixture(self.source)
        stems=[dict(id=30+i,name='1_'+name) for i,name in enumerate(NAMES)]
        guide=read_source_timing(self.source,stems,ppq=PPQ,bpm=170,end_tick=END)
        self.assertEqual(set(guide['matches']),set(range(30,38)))
        for index,stem in enumerate(stems):
            result=guide['matches'][stem['id']]
            self.assertEqual(result['source_track'],NAMES[index])
            self.assertEqual([list(p) for p in result['note_spans']],
                             [[index*PPQ,index*PPQ+24],[(8+index)*PPQ,(8+index)*PPQ+24]])
        self.assertEqual(self.source.read_bytes(),original)


if __name__=='__main__': unittest.main()
