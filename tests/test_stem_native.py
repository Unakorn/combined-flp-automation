"""Native Automation Clip structure and identity-link tests; no DAW claim."""
import json
from pathlib import Path
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'work'))
from flp_raw import encode_fl, read_bytes
from stem_native import (RESOURCE, build_channel, build_curve, build_link,
                         build_playlist_clip, compatible_groups, parse_points)


class NativeAutomationTests(unittest.TestCase):
    def test_sparse_native_channel_is_not_a_velocity_controller(self):
        points = [(0, .8), (384, .6), (768, .8)]
        events = build_channel(23, 'Drums - Filter dip', points, 96, 2)
        fields = dict(events)
        self.assertEqual(fields[21], b'\5')
        self.assertEqual(fields[219][:8], struct.pack('<II', 0, 12800))
        self.assertEqual(parse_points(fields[234], 96), points)
        self.assertNotIn(224, fields)
        self.assertNotIn(213, fields)
        self.assertNotIn(b'Keyboard', b''.join(v for k,v in events))
        self.assertEqual(len(fields[234]), 21 + 3*24 + 112)

    def test_native_link_matches_installed_fl_fixture_and_fans_out(self):
        resource = json.loads(RESOURCE.read_text(encoding='utf-8'))
        source, _, target, flags, unknown = resource['native_link']
        self.assertEqual(build_link(source >> 16, target), (227, struct.pack('<5I', *resource['native_link'])))
        fixture = resource['multi_target_link_source']
        links = [build_link(fixture['controller_id'], target) for target in fixture['targets']]
        self.assertEqual(len(links), 6)
        self.assertEqual({struct.unpack_from('<I', p)[0] for _, p in links}, {fixture['controller_id'] << 16})
        self.assertTrue(all(struct.unpack_from('<I',p)[0] & 0xffff == 0 for _,p in links))

    def test_native_points_use_beats_while_playlist_uses_project_ticks(self):
        data = json.loads(RESOURCE.read_text(encoding='utf-8'))
        original = bytes.fromhex(next(p for k,p in data['channel'] if k==234))
        points = parse_points(original, 96)
        clip = bytes.fromhex(data['playlist_templates']['80'])
        self.assertEqual(points[-1][0], struct.unpack_from('<I', clip, 8)[0])
        for ppq in (96, 480, 960):
            points = [(0, .4), (ppq * 16, .9)]
            self.assertEqual(parse_points(build_curve(points, ppq), ppq), points)

    def test_native_playlist_uses_channel_id_and_section_alignment(self):
        for size in (80, 88):
            clip = build_playlist_clip(27, 96*32, 96*8, 17, 44, size)
            self.assertEqual(len(clip), size)
            self.assertEqual(struct.unpack_from('<IHHIHH', clip), (3072, 0x5000, 27, 768, 482, 0))
            self.assertEqual(struct.unpack_from('<I', clip, 32)[0], 44)
            self.assertEqual(struct.unpack_from('<ff', clip, 24), (-1., -1.))
        with self.assertRaises(ValueError): build_playlist_clip(27, 0, 100, 17, 44, 60)

    def test_native_container_roundtrip_and_target_ids(self):
        points = [(0, .7), (384, .4), (768, .7)]
        targets = [0x70401fc0, 0x70801fc0]
        events = [(199,b'26.1.7.5653\0')] + [build_link(13,t) for t in targets]
        events += build_channel(13, 'Drums - Shared dip', points, 96)
        events += [(99,b'\0\0'),(233,build_playlist_clip(13,384,768,11,12))]
        encoded = encode_fl(struct.pack('<HHH',0,1,96), events)
        h, readback = read_bytes(encoded)
        self.assertEqual(readback, events)
        self.assertEqual([struct.unpack_from('<I',p,8)[0] for k,p in readback if k==227], targets)
        self.assertEqual(sum(k==234 for k,p in readback),1)
        self.assertEqual(sum(k==224 for k,p in readback),0)

    def test_native_slices_match_installed_split_clip_offsets(self):
        data = json.loads(RESOURCE.read_text(encoding='utf-8'))
        fixture = data['split_clip_evidence']
        for source in fixture['clips']:
            clip = build_playlist_clip(16, source['position'], source['duration'], 12, 7,
                                       80, source['offset_tick'], fixture['ppq'])
            self.assertEqual(list(struct.unpack_from('<ff',clip,24)),source['offset_beats'])
        # Many placements keep a single native channel and a single link.
        clips = [build_playlist_clip(16, t, 3072, 12, i+1, 88, t, 96)
                 for i,t in enumerate(range(0,3072*4,3072))]
        self.assertEqual({struct.unpack_from('<H',c,6)[0] for c in clips},{16})
        self.assertEqual([struct.unpack_from('<f',c,24)[0] for c in clips],[0,32,64,96])

    def test_distinct_ranges_or_timing_never_merge(self):
        base = dict(family='Drums',parameter='volume',shape='swell',start_tick=384,
                    duration_ticks=384,points=[(0,.8),(192,.6),(384,.8)],target=1)
        second = dict(base,target=2)
        self.assertEqual(len(compatible_groups([base,second])),1)
        self.assertEqual(len(compatible_groups([base,dict(second,points=[(0,.6),(192,.4),(384,.6)])])),2)
        self.assertEqual(len(compatible_groups([base,dict(second,start_tick=385)])),2)

    def test_curve_rejects_nonfinite_unordered_or_missing_zero(self):
        for points in ([(1,.5),(9,.6)],[(0,.5),(0,.6)],[(0,.5),(9,float('nan'))],[(0,.5),(9,1.1)]):
            with self.assertRaises(ValueError): build_curve(points,96)


if __name__=='__main__': unittest.main()
