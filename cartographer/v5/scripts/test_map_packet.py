"""Reject mismatched map arrays instead of attaching a newer ROS timestamp."""
import json
from types import SimpleNamespace as NS
import unittest
from map_packet import metadata_from_knowledge, PREFIX


def packet(meta=None, count=24, offset=0):
    if meta is None:
        meta = [-1., -2., -1., .1, 2, 3, 4, 10., 9.9]
    return NS(layout=NS(dim=[NS(label=PREFIX + json.dumps(meta))],
                        data_offset=offset), data=[-1] * count)


class MapPacketTests(unittest.TestCase):
    def test_metadata_belongs_to_the_voxel_packet(self):
        p = packet()
        self.assertEqual(metadata_from_knowledge(p)[7:], [10., 9.9])
        newer_independent_metadata = [11., 10.9]
        self.assertNotEqual(metadata_from_knowledge(p)[7:], newer_independent_metadata)

    def test_missing_label_and_mismatched_geometry_reject(self):
        p = packet()
        p.layout.dim = []
        with self.assertRaises(ValueError): metadata_from_knowledge(p)
        for p in [packet(count=23), packet(offset=1), packet([0, 0, 0, .1, 2.5, 3, 4, 1, 1])]:
            with self.assertRaises(ValueError): metadata_from_knowledge(p)

    def test_nonfinite_and_unbounded_metadata_reject(self):
        for meta in [[0, 0, 0, .1, 2, 3, 4, float('nan'), 1],
                     [0, 0, 0, .001, 2, 3, 4, 1, 1],
                     [0, 0, 0, .1, 2049, 1, 1, 1, 1]]:
            with self.assertRaises(ValueError): metadata_from_knowledge(packet(meta))


if __name__ == '__main__': unittest.main()
