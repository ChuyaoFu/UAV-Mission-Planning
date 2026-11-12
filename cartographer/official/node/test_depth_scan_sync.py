#!/usr/bin/env python3
"""Tests for node/depth_scan_sync.py (ROS-free): /team13/depth_scan paired with the same-stamp depth pose.

Run:  .venv/bin/python cartographer/official/node/test_depth_scan_sync.py   (also works with pytest)
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                      # cartographer/official
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'realdrone'))

from node.depth_scan_sync import DepthScanSync  # noqa: E402
from depth_scan import DepthScan                # noqa: E402  (the stereo node's own scan, numpy only)
from mission.occgrid import GridFrame, OccGrid  # noqa: E402

K = (160.0, 160.0, 160.0, 100.0)
SIZE = (320, 200)


def _pose(yaw_deg, cam=(1.0, 2.0, -1.0)):
    """Rectified camera looking horizontally along NED yaw: optical x right, y down, z forward."""
    y = math.radians(yaw_deg)
    fwd, right = np.array([math.cos(y), math.sin(y), 0.0]), np.array([-math.sin(y), math.cos(y), 0.0])
    return np.stack([right, np.array([0.0, 0.0, 1.0]), fwd], axis=1), np.asarray(cam, float)


def _scan(zwall=2.0):
    ds = DepthScan(K, SIZE, h_min=-0.25, h_max=0.35, range_max=5.0)
    depth = np.zeros((SIZE[1], SIZE[0]), np.float32)
    depth[:, 60:260] = zwall                                     # a wall facing the camera, zwall ahead
    depth[95:105, 280:] = 9.0                                    # in band, beyond range_max: free (+inf)
    r, _ = ds(depth)
    return ds, r


def test_pose_then_scan_and_scan_then_pose():
    ds, r = _scan()
    R_nc, cam = _pose(40.0)
    s = DepthScanSync()
    assert s.add_pose(10.0, R_nc, cam) == []
    sc = s.add_scan(10.0, ds.angle_min, ds.angle_increment, r, ds.range_max)
    assert sc is not None
    fwd = R_nc[:, 2][:2]
    hit = np.isfinite(sc.ranges)
    along = (sc.hit_ned[hit] - cam[:2]) @ fwd
    assert hit.sum() > 50 and np.allclose(along, 2.0, atol=0.03), along      # hits ON the wall plane, 2 m ahead
    assert (~sc.valid).any() and np.isinf(r).any()                           # NaN columns: no information
    s2 = DepthScanSync()                                                     # the other order
    assert s2.add_scan(11.0, ds.angle_min, ds.angle_increment, r, ds.range_max) is None
    out = s2.add_pose(11.002, R_nc, cam)                                     # 2 ms apart: same image
    assert len(out) == 1 and np.allclose(out[0].hit_ned[hit], sc.hit_ned[hit])
    print('  pose->scan and scan->pose give the same scan; %d hits on the wall plane (+-3 cm)' % int(hit.sum()))


def test_stamps_must_match_and_buffers_are_bounded():
    ds, r = _scan()
    R_nc, cam = _pose(0.0)
    s = DepthScanSync(keep=5)
    s.add_pose(1.0, R_nc, cam)
    assert s.add_scan(1.1, ds.angle_min, ds.angle_increment, r, ds.range_max) is None   # another image: wait
    for k in range(20):
        s.add_scan(2.0 + k, ds.angle_min, ds.angle_increment, r, ds.range_max)
        s.add_pose(100.0 + k, R_nc, cam)
    assert len(s.pending) <= 5 and len(s.poses) <= 5
    print('  a scan never pairs with another image\'s pose (> 5 ms); pending / pose buffers stay <= keep')


def test_into_grid():
    """The converted scan updates an OccGrid: wall cells hit, the free column carved, no-data columns untouched."""
    ds, r = _scan()
    R_nc, cam = _pose(0.0, cam=(0.0, 0.0, -1.0))
    s = DepthScanSync()
    s.add_pose(5.0, R_nc, cam)
    sc = s.add_scan(5.0, ds.angle_min, ds.angle_increment, r, ds.range_max)
    g = OccGrid(GridFrame((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), (-1.0, 4.0, -3.0, 3.0))
    g.update_scan(sc)
    c = g.cell_of_uv(np.array([2.0, 0.0]))
    assert g.L[c[0] - 1:c[0] + 2, c[1]].max() > 0                            # the wall straight ahead
    assert g.known_free()[g.cell_of_uv(np.array([1.0, 0.0]))[0], g.cell_of_uv(np.array([1.0, 0.0]))[1]]
    print('  into the grid: wall hit at 2 m, free in front of it')


TESTS = [test_pose_then_scan_and_scan_then_pose, test_stamps_must_match_and_buffers_are_bounded, test_into_grid]


def main():
    failed = 0
    for t in TESTS:
        try:
            t()
            print('PASS', t.__name__)
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print('FAIL', t.__name__, e)
    print('%d/%d passed (python %s)' % (len(TESTS) - failed, len(TESTS), sys.version.split()[0]))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
