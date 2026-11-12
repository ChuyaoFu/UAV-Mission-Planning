#!/usr/bin/env python3
"""Tests for the map-mode maze planner (occgrid.py, gridplan.py, mazenav.py, core.py nav_mode='map').

Unit level: walls are 2-D segments (NED xy); scans are ray-cast against them (optionally with stereo-like range
noise and phantom returns).  Closed loop: test_core.py's fake sim (kinematic / PX4-like vehicle, depth-like map)
on the v4 venue map and on random mazes of the same topology (walls +-0.5 m, lanes 0.9-1.3 m, gaps 0.8-1.2 m),
with the ORIGINAL layout as the prior.  No ROS, no clock.

Run:  .venv/bin/python cartographer/official/mission/test_mazenav.py   (also works with pytest)
"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # cartographer/official

from mission.occgrid import GridFrame, OccGrid, PointsScanSource, Scan, edt, scan_from_points  # noqa: E402
from mission.gridplan import (astar, choose_frontier, cost_map, dijkstra, frontiers, los_ok,  # noqa: E402
                              nearest_unblocked, plan_to_region, prune)
from mission.mazenav import infer_gates  # noqa: E402

PLAN_CLEAR = 0.38


# ------------------------------------------------------------------ synthetic world
def ray_cast(segs, cam_xy, dirs, max_range):
    """Nearest intersection range of rays (cam + t dir) with segments [(a, b), ...]; inf if none <= max_range."""
    cam = np.asarray(cam_xy, float)
    out = np.full(len(dirs), np.inf)
    for a, b in segs:
        a = np.asarray(a, float)
        e = np.asarray(b, float) - a
        w = a - cam
        den = dirs[:, 0] * (-e[1]) - dirs[:, 1] * (-e[0])
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (w[0] * (-e[1]) - w[1] * (-e[0])) / den
            s = (dirs[:, 0] * w[1] - dirs[:, 1] * w[0]) / den
        ok = (np.abs(den) > 1e-12) & (t > 0) & (s >= 0) & (s <= 1) & (t <= max_range)
        out = np.where(ok & (t < out), t, out)
    return out


def world_scan(segs, cam_ned, heading, n=200, hfov_deg=60.0, max_range=3.0, noise=0.0, rng=None, stamp=0.0,
               phantoms=()):
    """Node-style scan: n columns over the FOV, nearest wall return; ``phantoms``: [(xy, radius)] discs that also
    return (view-dependent noise is modelled by passing them only for some scans)."""
    b = np.radians(np.linspace(-0.5 * hfov_deg, 0.5 * hfov_deg, n))
    yaw = heading + b
    dirs = np.stack([np.cos(yaw), np.sin(yaw)], axis=1)
    cam = np.asarray(cam_ned, float)
    r = ray_cast(segs, cam[:2], dirs, max_range)
    for c, rad in phantoms:                              # disc phantom: ray-circle intersection
        w = cam[:2] - np.asarray(c, float)
        bq = dirs @ w
        disc = bq * bq - (w @ w - rad * rad)
        t = -bq - np.sqrt(np.maximum(disc, 0.0))
        r = np.where((disc >= 0) & (t > 0) & (t < r), t, r)
    if noise > 0 and rng is not None:
        rf = np.where(np.isfinite(r), r, 0.0)
        r = np.where(np.isfinite(r), r + rng.normal(0.0, 1.0, len(r)) * noise * rf * rf, r)
    return Scan(stamp=stamp, cam_ned=np.array([cam[0], cam[1], -1.0]), heading=heading, bearings=b, ranges=r,
                valid=np.ones(n, bool), max_range=max_range)


def box_segs(u0, u1, v0, v1):
    return [((u0, v0), (u1, v0)), ((u1, v0), (u1, v1)), ((u1, v1), (u0, v1)), ((u0, v1), (u0, v0))]


# serpentine maze (start frame = NED with yaw 0: u = north = x, v = left = -east): fence u -0.6..5.4, v -3..3
FENCE = (-0.6, 5.4, -3.0, 3.0)
W1 = ((1.42, -3.0), (1.42, 1.8))      # gap at the left end (v 1.8..3.0)
W2 = ((2.62, -1.8), (2.62, 3.0))      # gap at the right end
W3 = ((3.82, -3.0), (3.82, 1.8))      # gap at the left end
MAZE_UV = [W1, W2, W3]


def uv_to_ned_seg(frame, seg):
    return [tuple(frame.to_ned_xy(np.asarray(p, float))) for p in seg]


def new_grid(frame=None, box=FENCE, **kw):
    frame = frame or GridFrame.from_start((0.0, 0.0, 0.0), 0.0)
    return OccGrid(frame, box, **kw)


def paint_truth(grid, walls_uv, thick=0.05):
    """A fully mapped grid: wall cells confirmed occupied, every other inside cell known free."""
    ii, jj = np.mgrid[0:grid.nu, 0:grid.nv]
    U = grid.uv_of_cell(np.stack([ii, jj], axis=-1))
    occ = np.zeros((grid.nu, grid.nv), bool)
    for a, b in walls_uv:
        a, b = np.asarray(a, float), np.asarray(b, float)
        e = b - a
        t = np.clip(((U - a) @ e) / (e @ e), 0.0, 1.0)
        d = np.linalg.norm(U - (a + t[..., None] * e), axis=-1)
        occ |= d <= 0.5 * grid.res + 0.5 * thick
    occ &= ~grid.fence
    grid.L[:] = np.where(occ, grid.l_max, grid.l_min)
    grid.L[grid.fence] = 0.0
    grid.multi[:] = occ
    grid.seen[:] = ~grid.fence
    return occ


def seg_clear(grid, cm, uv_pts, step=0.02):
    """Min clearance (m, to confirmed structure) along a polyline in the grid frame."""
    m = math.inf
    for a, b in zip(uv_pts[:-1], uv_pts[1:]):
        n = max(2, int(math.ceil(np.linalg.norm(b - a) / step)) + 1)
        q = a[None, :] + np.linspace(0, 1, n)[:, None] * (b - a)[None, :]
        ij = grid.cell_of_uv(q)
        m = min(m, float(cm.clear[ij[:, 0], ij[:, 1]].min()))
    return m


# ------------------------------------------------------------------ OccGrid
def test_wall_confirmed_only_from_two_views():
    g = new_grid()
    wall = [((2.02, -1.0), (2.02, 1.0))]
    A, B = np.array([0.0, 0.0]), np.array([0.0, 0.6])
    for k in range(3):
        g.update_scan(world_scan(wall, A, 0.0, stamp=0.1 * k))
    g.update_scan(world_scan(wall, A, math.radians(15), stamp=0.3))     # yaw only: same camera position
    cells = g.to_cell(np.array([[2.02, v, 0.0] for v in np.linspace(-0.3, 0.3, 7)]))
    assert not g.occupied()[cells[:, 0], cells[:, 1]].any(), 'one viewpoint must not confirm'
    assert g.suspect()[cells[:, 0], cells[:, 1]].all()
    g.update_scan(world_scan(wall, B, math.atan2(-0.6, 2.0), stamp=0.4))
    occ = g.occupied()[cells[:, 0], cells[:, 1]]
    assert occ.all(), occ
    free = g.to_cell(np.array([[1.0, 0.0, 0.0], [1.5, 0.2, 0.0]]))
    assert g.known_free()[free[:, 0], free[:, 1]].all()
    print('  wall: suspect after 4 scans from one position (incl. a yaw change), confirmed after a 0.6 m baseline')


def test_phantom_carved_by_second_view():
    rng = np.random.default_rng(1)
    g = new_grid()
    wall = [((2.52, -2.0), (2.52, 2.0))]
    ph = ((1.5, 0.0), 0.08)
    for k in range(4):
        g.update_scan(world_scan(wall, (0.0, 0.0), 0.0, stamp=0.1 * k, phantoms=[ph], noise=0.01, rng=rng))
    pc = g.to_cell(np.array([[1.5 - 0.08, 0.0, 0.0]]))[0]
    near = (slice(pc[0] - 1, pc[0] + 2), slice(pc[1] - 1, pc[1] + 2))
    assert g.suspect()[near].any() and not g.occupied()[near].any()
    B = np.array([0.2, 0.9])
    hd = math.atan2(0.0 - B[1], 1.5 - B[0])
    for k in range(8):                                   # the phantom is view dependent: not seen from B
        g.update_scan(world_scan(wall, B, hd, stamp=1.0 + 0.1 * k, noise=0.01, rng=rng))
    assert not g.suspect()[near].any() and not g.occupied()[near].any(), g.L[near]
    wc = g.to_cell(np.array([[2.52, v, 0.0] for v in (-0.3, 0.0, 0.3)]))
    assert any(g.occupied()[i - 2:i + 3, j].any() for i, j in wc), 'the real wall, seen from both, is confirmed'
    print('  phantom: suspect from one view, carved by 8 scans through it from 0.9 m aside; the wall confirmed')


def test_blind_zone_never_updated():
    g = new_grid()
    cam = np.array([1.0, 0.5])
    s = world_scan([((1.3, -1.0), (1.3, 2.0))], cam, 0.0)               # wall 0.3 m ahead: inside the blind zone
    g.update_scan(s)
    ii, jj = np.mgrid[0:g.nu, 0:g.nv]
    U = g.uv_of_cell(np.stack([ii, jj], axis=-1))
    cu = g.frame.to_uv(np.r_[cam, 0.0])
    d = np.hypot(U[..., 0] - cu[0], U[..., 1] - cu[1])
    assert not g.seen[d < 0.4 - g.res].any()
    assert not g.seen.any(), 'a return inside the blind zone carries no information'
    g.update_scan(world_scan([((2.5, -1.0), (2.5, 2.0))], cam, 0.0))
    assert not g.seen[d < 0.4 - g.res].any() and g.seen.any()
    print('  blind zone: nothing within 0.4 m of the camera is ever updated; a 0.3 m return is ignored')


def test_fence_rotated_frame_and_pseudo_scan():
    home, yaw = np.array([2.0, -1.0, 0.0]), math.radians(40.0)
    fr = GridFrame.from_start(home, yaw)
    g = OccGrid(fr, FENCE)
    occ = g.occupied()
    assert occ[0, :].all() and occ[:, -1].all() and not occ[g.to_cell(home)[0], g.to_cell(home)[1]]
    # a wall 2 m ahead in the start frame, scanned from the take-off point looking forward
    seg = uv_to_ned_seg(fr, ((2.02, -1.0), (2.02, 1.0)))
    for k, c in enumerate((np.array([0.0, 0.0]), np.array([0.0, 0.5]))):
        cn = fr.to_ned_xy(c)
        hd = yaw
        for r in range(3):
            g.update_scan(world_scan([seg], cn, hd, stamp=k + 0.1 * r))
    cell = g.cell_of_uv(np.array([2.02, 0.2]))
    assert g.occupied()[cell[0], cell[1]], 'start-frame wall must land at u = 2 in the rotated grid'
    # pseudo-scan from 3-D points: band, FOV and range filters, nearest per bin
    cam = np.array([0.0, 0.0, -1.0])
    P = np.array([[1.5, 0.0, -1.0], [1.2, 0.0, -1.0],       # same bearing: nearest wins (1.2)
                  [2.0, 0.0, -0.1],                        # below the band (floor clutter)
                  [0.0, 2.0, -1.0],                        # 90 deg right: outside the FOV
                  [0.3, 0.0, -1.0],                        # inside the blind zone
                  [3.5, 0.5, -1.0]])                       # beyond max range
    s = scan_from_points(0.0, cam, 0.0, P, band_d=(-1.5, -0.3))
    fin = s.ranges[np.isfinite(s.ranges)]
    assert 1 <= len(fin) <= 5 and np.allclose(fin, 1.2), fin       # one voxel: the bins it covers (3.4 deg at 1.2 m)
    k = np.nonzero(np.isfinite(s.ranges))[0]
    assert np.all(np.abs(np.degrees(s.bearings[k])) <= 2.5), np.degrees(s.bearings[k])
    assert np.allclose(s.carve[max(0, k[0] - 3):k[0]], 1.2), 'no-return bins next to a return carve no farther'
    assert s.max_range == 2.0, 'no-return bins carve to noreturn_m only'

    class FakeMap:                                        # node ObstacleMap: vox[key] = [hits, last_stamp, x, y, z]
        vox = {(0,): [5, 10.0, 1.2, 0.0, -1.0], (1,): [5, 8.0, 1.0, 0.1, -1.0], (2,): [2, 10.0, 0.9, -0.1, -1.0]}
        points = np.zeros((0, 3))
    src = PointsScanSource()
    pts = src.points(FakeMap())
    assert len(pts) == 1 and abs(pts[0, 0] - 1.2) < 1e-9, pts      # stale (remembered) and < min_hits dropped
    s2 = src(10.0, FakeMap(), cam, 0.0, floor_d=0.0)
    f2 = s2.ranges[np.isfinite(s2.ranges)]
    assert len(f2) and np.allclose(f2, 1.2) and np.allclose(s2.hit_ned[np.isfinite(s2.ranges)], [1.2, 0.0]), f2
    print('  rotated start frame (40 deg): fence border occupied, a start-frame wall lands at u 2; pseudo-scan '
          'filters band / FOV / blind zone / range and remembered voxels')


def test_edt_matches_brute_force():
    rng = np.random.default_rng(3)
    for _ in range(5):
        m = rng.random((23, 31)) < 0.04
        d = edt(m)
        ii, jj = np.mgrid[0:23, 0:31]
        pi, pj = np.nonzero(m)
        bf = np.min(np.hypot(ii[..., None] - pi, jj[..., None] - pj), axis=-1)
        assert np.allclose(d, bf), float(np.abs(d - bf).max())
    assert np.isinf(edt(np.zeros((4, 5), bool))).all()
    print('  exact EDT equals brute force on 5 random grids')


# ------------------------------------------------------------------ planning
def test_astar_through_serpentine():
    g = new_grid()
    paint_truth(g, MAZE_UV)
    cm = cost_map(g)
    t0 = time.perf_counter()
    p = plan_to_region(g, cm, (0.0, 0.0), (4.7, 0.0), 0.4)
    dt = time.perf_counter() - t0
    assert p is not None
    P = np.asarray(p.cells)
    assert float(cm.clear[P[:, 0], P[:, 1]].min()) >= PLAN_CLEAR - 1e-9
    uv = np.vstack([[0.0, 0.0], p.uv])
    U = g.uv_of_cell(P)
    # through the gaps in order: W1 crossed at v > 1.8 + 0.38, W2 at v < -1.8 - 0.38, W3 at v > 1.8 + 0.38
    for wu, side in ((1.42, 1), (2.62, -1), (3.82, 1)):
        k = int(np.argmin(np.abs(U[:, 0] - wu)))
        assert side * U[k, 1] >= 1.8 + PLAN_CLEAR - 0.05, (wu, U[k])
    mc = seg_clear(g, cm, uv)
    assert mc >= PLAN_CLEAR - 0.03, mc
    assert 4 <= len(p.uv) <= 22, len(p.uv)
    assert all(np.linalg.norm(b - a) >= 0.1 for a, b in zip(uv[:-1], uv[1:])), np.round(uv, 2)
    assert all(np.linalg.norm(b - a) <= 1.5 + 1e-6 for a, b in zip(uv[:-1], uv[1:]))
    print('  serpentine (1.2 m lanes): %d waypoints, min clearance along the segments %.2f m, plan %.0f ms'
          % (len(p.uv), mc, 1e3 * dt))


def test_drifted_wall_never_blocks_a_lane_seen_free():
    """VIO drift (~3 % of the distance flown, real hand-carried loops 2026-11-08): the same wall seen again later appears
    0.3 m into the lane.  Rays through the old (drifted) cells carve them within a few scans (l_max 3.5), and a
    confirmed cell not hit again for stale_s only counts as suspect -- an old wall never blocks a lane seen free now."""
    g = new_grid(l_max=3.5, stale_s=20.0)
    old = [((2.02, -1.0), (2.02, 1.0))]                 # first pass: wall at u 2.02
    for k, c in enumerate(((0.0, 0.0), (0.0, 0.5), (0.0, -0.5), (0.0, 0.0))):
        for r in range(3):
            g.update_scan(world_scan(old, np.array(c), 0.0, stamp=0.1 * (3 * k + r)))
    oc = g.to_cell(np.array([[2.02, v, 0.0] for v in (-0.3, 0.0, 0.3)]))
    assert g.occupied()[oc[:, 0], oc[:, 1]].all()
    new = [((2.32, -1.0), (2.32, 1.0))]                 # later: the same wall, 0.3 m further (drift)
    n = 0
    while g.occupied()[oc[:, 0], oc[:, 1]].any() or g.suspect()[oc[:, 0], oc[:, 1]].any():
        n += 1
        g.update_scan(world_scan(new, np.array([0.0, 0.25 * math.sin(n)]), 0.0, stamp=5.0 + 0.2 * n))
        assert n < 12, (n, g.L[oc[:, 0], oc[:, 1]])
    nc = g.to_cell(np.array([[2.32, 0.0, 0.0]]))[0]
    assert g.occupied()[nc[0] - 1:nc[0] + 2, nc[1]].any()
    # stale: confirmed, then 25 s of scans elsewhere (not hitting it) -> suspect only
    g2 = new_grid(l_max=3.5, stale_s=20.0)
    for k, c in enumerate(((0.0, 0.0), (0.0, 0.5), (0.0, -0.5))):
        for r in range(3):
            g2.update_scan(world_scan(old, np.array(c), 0.0, stamp=0.1 * (3 * k + r)))
    assert g2.occupied()[oc[:, 0], oc[:, 1]].all()
    g2.update_scan(world_scan(old, np.array([0.0, 0.0]), math.pi, stamp=26.0))   # looking away
    assert not g2.occupied()[oc[:, 0], oc[:, 1]].any() and g2.suspect()[oc[:, 0], oc[:, 1]].all()
    print('  drifted wall (0.3 m): old cells carved in %d scans; unseen for 26 s: confirmed -> suspect (still avoided)'
          % n)


def test_depth_scan_laserscan_into_grid():
    """/team13/depth_scan (realdrone/depth_scan.py on a depth image) -> scan_from_laserscan -> grid.  Camera 1 m up
    heading 30 deg (NED); a wall 2 m ahead whose LEFT half is texture-less (no depth): the right half becomes hits at
    the wall, the texture-less half is never carved (NaN = no information), open columns (+inf) are carved."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'realdrone'))
    from depth_scan import DepthScan
    from mission.occgrid import scan_from_laserscan
    W_, H_, fx = 320, 200, 160.0
    K = (fx, fx, 160.0, 100.0)
    ds = DepthScan(K, (W_, H_), h_min=-0.25, h_max=0.35, range_max=5.0)
    u = np.arange(W_)
    depth = np.zeros((H_, W_), np.float32)
    zwall = 2.0
    depth[:, 200:280] = zwall                              # textured part of the wall (right of centre)
    depth[:, 280:] = 0.0                                    # beyond the wall's right end: nothing (NaN columns)
    depth[:, :160] = 0.0                                    # texture-less left half: no depth
    depth[20:60, 290:] = 9.0                                # far background above the band: +inf in the scan
    depth[95:105, 290:] = 9.0                               # ... in the band, beyond range_max: +inf
    ranges, _ = ds(depth)
    yaw = math.radians(30.0)
    c, s_ = math.cos(yaw), math.sin(yaw)
    fwd, right = np.array([c, s_, 0.0]), np.array([-s_, c, 0.0])
    R_nc = np.stack([right, np.array([0.0, 0.0, 1.0]), fwd], axis=1)     # optical x right, y down, z forward
    cam = np.array([0.0, 0.0, -1.0])
    sc = scan_from_laserscan(1.0, R_nc, cam, ds.angle_min, ds.angle_increment, ranges, ds.range_max)
    g = new_grid()
    g.update_scan(sc)
    hits = np.isfinite(sc.ranges)
    assert hits.any() and (~sc.valid).any() and np.isposinf(ranges).any()
    H = sc.hit_ned[hits]
    along = H @ fwd[:2]
    assert np.allclose(along, zwall, atol=0.03), along     # hits at the wall plane (2 m along the optical axis)
    cells = g.to_cell(np.c_[H, np.zeros(len(H))])
    assert (g.L[cells[:, 0], cells[:, 1]] > 0).all()
    left = cam[:2] + 1.5 * fwd[:2] - 0.6 * right[:2]          # behind where the texture-less half would be
    lc = g.to_cell(np.r_[left, 0.0])
    assert not g.seen[lc[0], lc[1]], 'no-depth columns must not be carved'
    print('  depth_scan LaserScan -> grid: %d hits on the wall plane (+-3 cm), %d no-data columns never carved, '
          '%d free columns carved' % (int(hits.sum()), int((~sc.valid).sum()), int(np.isposinf(ranges).sum())))


def test_confirmed_board_end_persists():
    """Gazebo W1-end hits: the node map forgets a thin board end 4 s after it leaves the camera view, then the
    vehicle cuts the U-turn into it.  The grid keeps a confirmed end (plan A: no decay) and the planner keeps 0.38 m
    from it however long it has been out of view."""
    g = new_grid()
    end = [((1.40, 0.5), (1.40, 0.52))]                    # a board end seen edge-on: a 2 cm sliver
    for k, c in enumerate(((0.0, 0.0), (0.0, 0.5), (0.3, 0.9))):
        for r in range(3):
            g.update_scan(world_scan(end, np.array(c), math.atan2(0.51 - c[1], 1.40 - c[0]), stamp=0.1 * (3 * k + r)))
    ec = g.to_cell(np.array([1.40, 0.51, 0.0]))
    near = (slice(ec[0] - 1, ec[0] + 2), slice(ec[1] - 1, ec[1] + 2))
    assert g.occupied()[near].any()
    for k in range(150):                                   # 30 s looking the other way (5 Hz)
        g.update_scan(world_scan([((-0.5, -2.0), (-0.5, 2.0))], np.array([0.5, 0.0]), math.pi, stamp=1.0 + 0.2 * k))
    assert g.occupied()[near].any(), 'a confirmed board end must persist'
    cm = cost_map(g)
    q = g.frame.to_uv(np.array([1.40, 0.51, 0.0]))
    ii, jj = np.mgrid[0:g.nu, 0:g.nv]
    U = g.uv_of_cell(np.stack([ii, jj], axis=-1))
    d = np.hypot(U[..., 0] - q[0], U[..., 1] - q[1])
    assert cm.blocked[d < PLAN_CLEAR - 0.05].all() and not cm.blocked[(d > 0.5) & (d < 0.6) & ~g.fence].all()
    print('  a confirmed board end (2 cm sliver) still blocks 0.38 m round it after 30 s out of view')


def test_unknown_costs_more_than_free():
    g = new_grid()
    paint_truth(g, [((2.02, -1.2), (2.02, 1.2))])
    ii, jj = np.mgrid[0:g.nu, 0:g.nv]
    U = g.uv_of_cell(np.stack([ii, jj], axis=-1))
    unk = (U[..., 1] < 0) & (U[..., 0] > 0.8) & (U[..., 0] < 3.2) & ~g.fence     # the right side never seen
    g.seen[unk] = False
    g.L[unk] = 0.0
    cm = cost_map(g)
    p = plan_to_region(g, cm, (0.5, 0.0), (3.5, 0.0), 0.3)
    side = g.uv_of_cell(np.asarray(p.cells))[:, 1]
    assert side.max() > 1.2 and side.min() > -0.5 and p.unknown_share == 0.0, (side.min(), side.max())
    # the only way on is through unknown space: taken (optimistic planning)
    g2 = new_grid()
    paint_truth(g2, [((2.02, -3.0), (2.02, 1.0))])
    unk2 = (U[..., 1] > 1.0) & (U[..., 0] > 1.5) & (U[..., 0] < 2.5) & ~g2.fence
    g2.seen[unk2] = False
    g2.L[unk2] = 0.0
    cm2 = cost_map(g2)
    p2 = plan_to_region(g2, cm2, (0.5, 0.0), (3.0, 0.0), 0.3)
    assert p2 is not None and p2.unknown_share > 0.2
    print('  equal routes: the known-free side (unknown share 0); only way on through unknown: taken (share %.2f)'
          % p2.unknown_share)


def test_suspect_gap_only_as_last_resort():
    """A 1.6 m gap with a single-view blob (r 0.1) in its middle leaves only suspect-near cells; a clean 1.2 m gap
    1.8 m further left is taken instead.  Far gap closed: the near gap is used, never within 0.28 m of the blob."""
    def build(far_open):
        g = new_grid()
        walls = [((2.02, -3.0), (2.02, -1.8)), ((2.02, -0.2), (2.02, 0.8))]
        walls.append(((2.02, 2.0), (2.02, 3.0)) if far_open else ((2.02, 0.8), (2.02, 3.0)))
        paint_truth(g, walls)
        ii, jj = np.mgrid[0:g.nu, 0:g.nv]
        U = g.uv_of_cell(np.stack([ii, jj], axis=-1))
        blob = np.hypot(U[..., 0] - 2.02, U[..., 1] + 1.0) <= 0.1
        g.L[blob] = 1.0                                       # occupancy evidence, one view: suspect
        return g
    g = build(True)
    cm = cost_map(g, hard_clear_m=0.30)                       # (the navigator's relaxed planning)
    p = plan_to_region(g, cm, (0.5, -1.0), (3.5, -1.0), 0.3)
    U = g.uv_of_cell(np.asarray(p.cells))
    k = int(np.argmin(np.abs(U[:, 0] - 2.02)))
    assert U[k, 1] > 0.8, ('the clean far gap', U[k])
    g2 = build(False)
    cm2 = cost_map(g2, hard_clear_m=0.30)
    p2 = plan_to_region(g2, cm2, (0.5, -1.0), (3.5, -1.0), 0.3)
    assert p2 is not None
    P2 = np.asarray(p2.cells)
    U2 = g2.uv_of_cell(P2)
    k = int(np.argmin(np.abs(U2[:, 0] - 2.02)))
    assert abs(U2[k, 1] + 1.0) < 0.5 and cm2.cls[P2[k, 0], P2[k, 1]] == 4, (U2[k], cm2.cls[P2[k, 0], P2[k, 1]])
    ms = float(cm2.clear_s[P2[:, 0], P2[:, 1]].min())
    assert ms >= 0.28 - 1e-9, ms
    print('  blob in the near gap: far clean gap used; far gap closed: near gap used, >= %.2f m from the blob' % ms)


def test_frontier_choice_points_toward_goal():
    g = new_grid()
    ii, jj = np.mgrid[0:g.nu, 0:g.nv]
    U = g.uv_of_cell(np.stack([ii, jj], axis=-1))
    known = (np.abs(U[..., 0]) < 1.0) & (np.abs(U[..., 1]) < 1.0) & ~g.fence
    g.seen[known] = True
    g.L[known] = g.l_min                                     # a known-free 2 x 2 m box round the start, rest unknown
    wall = (np.abs(U[..., 0] - 1.0) < 0.05) & (np.abs(U[..., 1]) < 1.0)   # its forward side is a confirmed wall
    g.L[wall] = g.l_max
    g.multi[wall] = True
    g.seen[wall] = True
    cm = cost_map(g)
    fr = frontiers(cm)
    assert len(fr) >= 2, len(fr)
    goal = g.cell_of_uv(np.array([3.0, 2.5]))
    best = choose_frontier(cm, g.cell_of_uv(np.array([0.0, 0.0])), goal, fr)
    bu = g.uv_of_cell(np.asarray(best['cell']))
    assert bu[1] > 0.5, bu                                   # the left side (goal at v 2.5), not right or behind
    print('  %d frontiers round a known box walled in front; chosen at (%.2f, %.2f) toward the goal (3, 2.5)'
          % (len(fr), bu[0], bu[1]))


def test_blocked_start_and_los():
    g = new_grid()
    paint_truth(g, [((1.02, -1.0), (1.02, 1.0))])
    cm = cost_map(g)
    s = g.cell_of_uv(np.array([0.75, 0.0]))                  # 0.25 m from the wall: inside C-space
    assert cm.blocked[s[0], s[1]]
    n = nearest_unblocked(cm, s)
    assert n is not None and cm.clear[n] >= cm.hard_clear_m
    p = plan_to_region(g, cm, (0.75, 0.0), (2.0, 0.0), 0.3)
    assert p is not None and p.start_moved
    assert not los_ok(cm, g.cell_of_uv(np.array([0.3, 0.0])), g.cell_of_uv(np.array([1.8, 0.0])))
    gm = np.zeros_like(cm.blocked)
    gm[g.cell_of_uv(np.array([2.0, 0.0]))[0], :] = True
    raw, _ = astar(cm, n, gm & ~cm.blocked)
    pr = prune(cm, raw)
    assert len(pr) < len(raw) and len(pr) <= 5
    print('  start 0.25 m from a wall: planned from the nearest free cell; LOS through the wall refused; '
          'string pulling %d -> %d cells' % (len(raw), len(pr)))


def test_map_then_plan_serpentine():
    """Scans from a short scripted flight in the first lane (noisy, with a phantom seen once) -> plan to the far
    side of W1: through its real gap, never within 0.38 m of confirmed cells; phantom never confirmed."""
    rng = np.random.default_rng(7)
    fr = GridFrame.from_start((0.0, 0.0, 0.0), 0.0)
    g = OccGrid(fr, FENCE)
    segs = [uv_to_ned_seg(fr, w) for w in MAZE_UV] + [uv_to_ned_seg(fr, s) for s in box_segs(*FENCE)]
    ph_uv = np.array([0.9, -0.6])
    ph = (tuple(fr.to_ned_xy(ph_uv)), 0.06)
    ph_yaw = math.atan2(ph[0][1], ph[0][0])
    t = 0.0
    route = [(0.0, 0.0), (0.4, 0.4), (0.6, 1.0), (0.6, 1.6)]
    for k, (u, v) in enumerate(route):
        for hd in np.radians(np.arange(-60, 121, 10)):
            g.update_scan(world_scan(segs, fr.to_ned_xy((u, v)), hd, noise=0.01, rng=rng, stamp=t,
                                     phantoms=[ph] if k == 0 and abs(hd - ph_yaw) < 0.2 else []))
            t += 0.1
    st = g.stats()
    pc = g.cell_of_uv(np.array([0.9, -0.6]))
    assert not g.occupied()[pc[0] - 2:pc[0] + 3, pc[1] - 2:pc[1] + 3].any(), 'the phantom must not be confirmed'
    w1 = g.cell_of_uv(np.array([[1.42, v] for v in np.linspace(-1.0, 1.4, 9)]))
    conf = np.array([g.occupied()[i - 1:i + 2, j].any() for i, j in w1])
    assert conf.mean() >= 0.7, conf
    cm = cost_map(g)
    t0 = time.perf_counter()
    p = plan_to_region(g, cm, (0.6, 1.6), (2.0, 1.0), 0.3)
    dt = time.perf_counter() - t0
    assert p is not None
    U = g.uv_of_cell(np.asarray(p.cells))
    k = int(np.argmin(np.abs(U[:, 0] - 1.42)))
    assert U[k, 1] >= 1.8 + PLAN_CLEAR - 0.05, U[k]
    mc = seg_clear(g, cm, np.vstack([[0.6, 1.6], p.uv]))
    assert mc >= PLAN_CLEAR - 0.03, mc
    print('  mapped from %d scans (%d confirmed / %d suspect cells), W1 confirmed %.0f %%; plan through its gap, '
          'clearance %.2f m, %.0f ms' % (st['scans'], st['occupied'], st['suspect'], 100 * conf.mean(), mc,
                                         1e3 * dt))


def test_plan_time_budget():
    """Replan time on the full 6 x 6 m fence grid (0.05 m): fully mapped serpentine and mostly unknown."""
    times = []
    g = new_grid()
    paint_truth(g, MAZE_UV)
    for _ in range(5):
        t0 = time.perf_counter()
        cm = cost_map(g)
        plan_to_region(g, cm, (0.0, 0.0), (4.7, 0.0), 0.4)
        times.append(time.perf_counter() - t0)
    g2 = new_grid()
    for _ in range(5):
        t0 = time.perf_counter()
        cm2 = cost_map(g2)
        plan_to_region(g2, cm2, (0.0, 0.0), (4.7, 2.0), 0.4)
        times.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    dijkstra(cm, g.cell_of_uv(np.array([0.0, 0.0])))
    td = time.perf_counter() - t0
    p95 = float(np.percentile(times, 95))
    assert p95 <= 0.1, times
    print('  cost map + A* + prune on %d x %d cells: p95 %.0f ms (max %.0f); full Dijkstra %.0f ms'
          % (g.nu, g.nv, 1e3 * p95, 1e3 * max(times), 1e3 * td))


# ------------------------------------------------------------------ closed loop (core.py nav_mode='map')
def _sim():
    sys.path.insert(0, HERE)
    import test_core as T                                  # the fake sim (kinematic / PX4-like vehicle, depth map)
    from mission.core import Params
    return T, Params


NORTH, BARRIER = 2.134, -0.457


def random_maze(seed):
    """v4 venue map (test_core._venue_world) with the three maze walls redrawn: same topology (W1 gap at the north
    net, W2 hanging from the net with its gap above the barrier, W3 gap = Gate A at the net), wall lines moved
    (start lane 0.9-1.3 m with the fence fixed, W1 within +-0.5 m of the map), lanes 0.9-1.3 m, gaps 0.8-1.2 m.  The
    prior stays the ORIGINAL map."""
    T, _ = _sim()
    rng = np.random.default_rng(seed)
    pr, scn = T._venue_world('gate_b')
    l1 = float(np.clip(rng.uniform(0.53 - 1.3, 0.53 - 0.9), -0.597 - 0.5, -0.597 + 0.5))
    l2 = l1 - rng.uniform(0.9, 1.3)
    l3 = l2 - rng.uniform(0.9, 1.3)
    g1, g2, g3 = rng.uniform(0.8, 1.2, 3)
    walls = []
    for w in scn.depth_walls:
        w = list(w)
        if w[6] == 'W1':
            w[0:4] = [-0.427, l1, NORTH - g1, l1]
        elif w[6] == 'W2':
            w[0:4] = [BARRIER + g2, l2, 2.088, l2]
        elif w[6] == 'W3':
            w[0:4] = [-0.427, l3, NORTH - g3, l3]
        walls.append(tuple(w))
    scn.depth_walls = tuple(walls)
    scn.px4_mpc = True
    return pr, scn, dict(lanes=[round(0.53 - l1, 2), round(l1 - l2, 2), round(l2 - l3, 2)],
                         gaps=[round(g1, 2), round(g2, 2), round(g3, 2)])


def _prior_variant(pr, kind):
    import json
    pr = json.loads(json.dumps(pr))
    if kind == 'blind':                                    # no wall lines: no gates can be inferred
        pr.pop('walls', None)
    elif kind == 'gate_a':                                 # explicit checkpoint, test search radius 1.5 m
        pr['checkpoints'] = [dict(name='gate_a', kind='gate', center=dict(forward=1.66, left=-2.822),
                                  through_deg=-90.0, search_m=1.5)]
    return pr


def _fly_map(scn, **over):
    T, Params = _sim()
    m, px4, world, log = T.run(scn, Params(stop_after='corridor', nav_mode='map', **over))
    cor = [r for r in log if r['phase'] == 'corridor']
    wd = min(world.clearance_xy(r['p'], skip=()) for r in cor)
    cps = [e['name'] for e in T.events(log, 'checkpoint_passed')]
    return m, px4, world, log, cor[-1]['t'] - cor[0]['t'], wd, cps


def test_map_mode_venue_maze():
    """v4 venue map, prior = the same map (old prior: gates inferred where the corridor goes round a wall end)."""
    T, _ = _sim()
    pr, scn = T._venue_world('gate_b')
    scn.px4_mpc = True
    m, px4, world, log, dt, wd, cps = _fly_map(scn)
    init = T.events(log, 'nav_init')[0]
    assert init['source'] == 'inferred' and [s['name'] for s in init['stages']] == \
        ['gate_W1', 'gate_W2', 'gate_W3', 'tunnel_standoff'], init['stages']
    assert m.sub == 'stopped_after_corridor' and cps == ['gate_W1', 'gate_W2', 'gate_W3'], (m.abort_reason, cps)
    T.assert_no_collision(world, log, skip=())
    assert wd >= 0.33 and dt < 90.0, (wd, dt)
    print('  venue map, map mode: gates W1 / W2 / W3 in order, corridor %.1f s (prior mode ~71 s; map mode yaws <= 0.4 rad/s and looks before unknown), nearest %.2f m'
          % (dt, wd))


def test_map_mode_random_mazes():
    """Random mazes (walls +-0.5 m, lanes 0.9-1.3 m, gaps 0.8-1.2 m), prior = the ORIGINAL map: coarse (inferred
    gates), blind (no wall lines) and an explicit Gate A checkpoint (search 1.5 m).  Default: 4 mazes with the
    narrowest lanes / gaps; MISSION_LONG=1: 20 mazes x 3 priors (fake sim: 60/60; prior mode 11/20)."""
    T, _ = _sim()
    long_ = bool(os.environ.get('MISSION_LONG'))
    runs = [(sd, k) for sd in range(20) for k in ('coarse', 'blind', 'gate_a')] if long_ else \
        [(0, 'coarse'), (4, 'coarse'), (10, 'gate_a'), (12, 'blind')]
    out = []
    for sd, kind in runs:
        pr, scn, info = random_maze(sd)
        scn.prior = _prior_variant(pr, kind)
        m, px4, world, log, dt, wd, cps = _fly_map(scn)
        want = {'coarse': ['gate_W1', 'gate_W2', 'gate_W3'], 'blind': [], 'gate_a': ['gate_a']}[kind]
        assert m.sub == 'stopped_after_corridor' and cps == want, (sd, kind, m.abort_reason, cps, info)
        T.assert_no_collision(world, log, skip=())
        assert wd >= 0.29, (sd, kind, wd, info)
        assert not T.events(log, 'nav_fallback'), (sd, kind)
        out.append((dt, wd))
    print('  random mazes (%d runs): all reached the tunnel standoff through the gates in order, %.0f-%.0f s, '
          'nearest %.3f m (reach 0.28)' % (len(out), min(o[0] for o in out), max(o[0] for o in out),
                                           min(o[1] for o in out)))


def test_map_mode_phantoms_in_lanes():
    """Stereo phantom clusters in the lanes, off the lane centres (carved once the camera moved 0.3 m, like
    test_core's depth phantoms): planned round or past (relaxed), the corridor completes without contact."""
    T, _ = _sim()
    pr, scn, info = random_maze(6)
    scn.depth_phantoms = (((1.0, 0.36, 0.9), 8, 0.06, 0.3), ((0.9, -1.35, 0.9), 8, 0.06, 0.3),
                          ((1.0, -2.35, 0.8), 8, 0.06, 0.3))      # net-side / 0.25-0.3 m off the lane centres (a
    # consistent single-view phantom that squeezes a ~1 m lane below ~0.7 m can still stall map mode: the plan then
    # grazes the wall inside the follower's 0.35 m probe; see mission-REPORT.md section 32 limits)
    m, px4, world, log, dt, wd, cps = _fly_map(scn)
    assert m.sub == 'stopped_after_corridor' and cps == ['gate_W1', 'gate_W2', 'gate_W3'], (m.abort_reason, cps)
    T.assert_no_collision(world, log, skip=())
    print('  3 phantom clusters in the lanes: corridor %.1f s, nearest %.2f m, %d phantoms carved in the map'
          % (dt, wd, len(world.dmap.ph_vanished)))


def test_checkpoints_parse_and_gate_inference():
    T, Params = _sim()
    from mission.core import CoursePrior
    pr, _ = T._venue_world('gate_b')
    P = CoursePrior.from_dict(dict(pr, checkpoints=[
        dict(name='b', center=dict(forward=1.0, left=-1.0), order=2),
        dict(name='a', kind='gate', center=[1.66, -2.82], through_deg=-90, search_m=1.5, order=1)]))
    assert [c['name'] for c in P.checkpoints] == ['a', 'b'] and P.checkpoints[0]['search_m'] == 1.5
    try:
        CoursePrior.from_dict(dict(pr, checkpoints=[dict(name='g', kind='gate', center=dict(forward=1, left=0))]))
        raise AssertionError('a gate without through_deg must be refused')
    except ValueError:
        pass
    P0 = CoursePrior.from_dict(pr)
    walls = [(n, (np.array(a), np.array(b))) for n, (a, b, _) in zip(P0.wall_names, P0.walls)]
    g = infer_gates([np.zeros(2)] + [np.asarray(c[:2]) for c in P0.corridor], walls)
    assert [s.name for s in g] == ['gate_W1', 'gate_W2', 'gate_W3'], [s.name for s in g]   # not W4 / C1 (south)
    assert np.allclose(g[0].through, [0.0, -1.0]) and abs(g[2].center[1] + 2.822) < 1e-6
    m = T.Mission(P0, Params())
    assert Params().nav_mode == 'prior' and m._grid is None and m._nav is None
    print('  checkpoints sorted by order, gate needs through_deg; inferred gates W1 / W2 / W3 (W4 / C1 beyond the '
          'barrier skipped); nav_mode defaults to prior')


def test_map_mode_offset_priors():
    """(a) Eyeball-level priors on the v4 layout: every wall line (and its corridor vertices) off by +-0.5 m
    (test_core._offset_prior).  The prior walls are painted as the SOFT layer; the real walls, once seen, win.
    Default seeds 0, 5, 7, 10; MISSION_LONG=1: 20 (fake sim 20/20, nearest >= 0.41 m)."""
    T, Params = _sim()
    seeds = range(20) if os.environ.get('MISSION_LONG') else (0, 5, 7, 10)
    out = []
    for sd in seeds:
        pr0, scn = T._venue_world('gate_b')
        scn.prior, scn.t_max, scn.px4_mpc = T._offset_prior(pr0, sd), 260, True
        m, px4, world, log, dt, wd, cps = _fly_map(scn)
        assert m.sub == 'stopped_after_corridor', (sd, m.abort_reason)
        T.assert_no_collision(world, log, skip=())
        assert wd >= 0.33 and not T.events(log, 'nav_fallback'), (sd, wd)
        out.append((dt, wd))
    print('  offset (+-0.5 m) priors: %d/%d through the maze, %.0f-%.0f s, nearest %.2f m'
          % (len(out), len(list(seeds)), min(o[0] for o in out), max(o[0] for o in out), min(o[1] for o in out)))


def test_map_mode_outdated_prior():
    """(b) The maze changed after the prior was made (real venue 2026-11-08): W2 removed, or W3 moved 0.45 m east
    (its old line now runs through lane 2, Gate A elsewhere).  The prior lines only cost; rays through the old W3
    line clear it; the corridor completes without contact and never falls back to the prior route."""
    T, Params = _sim()
    res = {}
    for kind in ('w2_removed', 'w3_moved'):
        pr, scn = T._venue_world('gate_b')
        scn.px4_mpc = True
        walls = []
        for w in scn.depth_walls:
            w = list(w)
            if kind == 'w2_removed' and w[6] == 'W2':
                continue
            if kind == 'w3_moved' and w[6] == 'W3':
                w[1] -= 0.45
                w[3] -= 0.45
            walls.append(tuple(w))
        scn.depth_walls = tuple(walls)
        m, px4, world, log, dt, wd, cps = _fly_map(scn)
        assert m.sub == 'stopped_after_corridor' and not T.events(log, 'nav_fallback'), (kind, m.abort_reason)
        T.assert_no_collision(world, log, skip=())
        assert wd >= 0.33, (kind, wd)
        g = m._grid
        a, b = np.array([-0.427, -2.822]), np.array([1.189, -2.822])        # the OLD W3 line (prior)
        cells = g.cell_of_uv(np.array([a + (b - a) * t for t in np.linspace(0, 1, 30)]))
        res[kind] = (dt, wd, int((~g.prior[cells[:, 0], cells[:, 1]]).sum()))
    assert res['w3_moved'][2] >= 5, res               # the old W3 line was seen through and cleared
    print('  outdated prior: W2 removed %.0f s (nearest %.2f m); W3 moved 0.45 m %.0f s (nearest %.2f m, %d/30 cells of '
          'its old line cleared)' % (res['w2_removed'][0], res['w2_removed'][1], res['w3_moved'][0], res['w3_moved'][1],
                                     res['w3_moved'][2]))


def test_map_mode_official_w1_end_no_contacts():
    """(c) Official lab (runs 002-008): the W1 end 0.73 m from the north wall, PX4 ~0.3 m braking overshoot, prop-strike
    physics, slow hold oscillation after fast yaws.  Gazebo map/eyeball runs hit the W1 end repeatedly once it left
    the 4 s node map; the grid keeps it (confirmed cells persist) and plans the 0.73 m gap along its centre."""
    T, Params = _sim()
    pr, scn, prm = T._official_world(dict(nav_mode='map'))
    m, px4, world, log = T.run(scn, prm)
    cor = [r for r in log if r['phase'] == 'corridor']
    wd = min(world.clearance_xy(r['p'], skip=()) for r in cor)
    assert m.sub == 'stopped_after_corridor', m.abort_reason
    assert len(px4.contacts) == 0, px4.contacts
    print('  official lab world, map mode: corridor %.0f s, 0 prop strikes, nearest %.2f m (W1-end gap 0.73 m)'
          % (cor[-1]['t'] - cor[0]['t'], wd))


# compat mp3A maze24 (EW2): start between the barrier (0.4 m behind) and a board 0.6 m ahead, first leg WEST (~90 deg left)
M24_PRIOR = {'frame': 'start', 'tolerance_m': 0.5, 'cruise_up_m': 0.8, 'fence': {'forward_min': -4.199, 'forward_max': 1.897, 'left_min': -3.455, 'left_max': 2.641, 'up_max': 2.9}, 'corridor': [{'forward': -0.104, 'left': 2.169, 'up': 0.8}, {'forward': 1.208, 'left': 2.14, 'up': 0.8}, {'forward': 1.066, 'left': -0.582, 'up': 0.8}, {'forward': 1.156, 'left': -0.857, 'up': 0.8}, {'forward': 1.631, 'left': -2.714, 'up': 0.8}], 'tunnel': {'entry': {'forward': 0.237, 'left': -2.733, 'up': 0.585}, 'axis_deg': 179.14, 'length_m': 1.524, 'tol_m': 0.5}, 'window': {'center': {'forward': -3.395, 'left': 0.339, 'up': 1.072}, 'through_deg': 69.16, 'tol_m': 0.5}, 'landing_target': {'center': {'forward': -1.716, 'left': 1.653}, 'top_up': -0.045, 'tol_m': 0.5}, 'wall': {'cross_from': {'forward': -1.032, 'left': 0.986}, 'cross_to': {'forward': -0.08, 'left': 0.18}, 'top_up': 1.581, 'tol_m': 0.3}, 'walls': [{'name': 'barrier', 'from': {'forward': -0.422, 'left': 2.561}, 'to': {'forward': -0.394, 'left': -2.072}}, {'name': 'W3', 'from': {'forward': -0.89, 'left': -0.842}, 'to': {'forward': 0.725, 'left': -0.876}}, {'name': 'C1', 'from': {'forward': -0.538, 'left': -1.894}, 'to': {'forward': -2.029, 'left': -1.37}}, {'name': 'C2_north', 'from': {'forward': -1.794, 'left': -0.731}, 'to': {'forward': -2.794, 'left': -0.343}}, {'name': 'W4', 'from': {'forward': -1.686, 'left': 0.786}, 'to': {'forward': -0.68, 'left': 0.744}}, {'name': 'D1', 'from': {'forward': 0.648, 'left': 1.437}, 'to': {'forward': 0.594, 'left': -0.492}}]}


def _prior_world(pr, skip=()):
    """Fake-sim world whose walls are the prior's wall lines (boards 0.35-1.73 m) and whose tunnel is the prior's."""
    T, Params = _sim()
    walls = tuple((w['from']['forward'], w['from']['left'], w['to']['forward'], w['to']['left'], 0.35, 1.73, w['name'],
                   0.01) for w in pr['walls'] if w['name'] not in skip)
    te = pr['tunnel']['entry']
    return T.Scenario(name='prior_world', prior=pr, default_course=False, depth_walls=walls, px4_mpc=True,
                      true_over=dict(tunnel_entry=(te['forward'], te['left'], te['up']),
                                     tunnel_axis_deg=pr['tunnel']['axis_deg'], tunnel_length=pr['tunnel']['length_m'],
                                     tunnel_ring_gap=0.762, tunnel_radius=0.55),
                      vehicle_r=Params().prop_radius_m, t_max=220)


def test_map_mode_side_first_leg():
    """The first leg leaves 70-90 deg to the SIDE of the start heading (compat mp3A: map mode never left the start
    there).  maze24 (EW2): a board 0.6 m ahead, the barrier 0.4 m behind, first leg west; the real maze snapshot
    (course_prior.real_maze_20261108.json): a board 1.14 m ahead, first leg 73 deg right.  Map mode turns to face the
    first leg (nav_turn), looks, replans and goes -- no fallback to the prior route, the start-rear clamp untouched."""
    import json
    T, Params = _sim()
    with open(os.path.join(HERE, 'course_prior.real_maze_20261108.json')) as fh:
        real = json.load(fh)
    out = []
    for label, pr in (('maze24', M24_PRIOR), ('real_maze', real)):
        scn = _prior_world(pr)
        m, px4, world, log, dt, wd, cps = _fly_map(scn)
        assert m.sub == 'stopped_after_corridor' and not T.events(log, 'nav_fallback'), (label, m.abort_reason)
        T.assert_no_collision(world, log, skip=())
        assert wd >= 0.33, (label, wd)
        back = min(world.flu(r['p'])[0] for r in log if r['phase'] == 'corridor')
        assert back >= -0.1, (label, back)                # never behind the take-off point
        out.append((label, dt, wd, len(T.events(log, 'nav_turn'))))
    assert out[0][3] >= 1, out
    print('  side first leg: ' + '; '.join('%s %.0f s, nearest %.2f m, %d turn(s)' % o for o in out))


TESTS = [test_wall_confirmed_only_from_two_views, test_phantom_carved_by_second_view, test_blind_zone_never_updated,
         test_fence_rotated_frame_and_pseudo_scan, test_edt_matches_brute_force, test_astar_through_serpentine,
         test_drifted_wall_never_blocks_a_lane_seen_free, test_depth_scan_laserscan_into_grid,
         test_confirmed_board_end_persists, test_unknown_costs_more_than_free,
         test_suspect_gap_only_as_last_resort,
         test_frontier_choice_points_toward_goal, test_blocked_start_and_los, test_map_then_plan_serpentine,
         test_plan_time_budget, test_checkpoints_parse_and_gate_inference, test_map_mode_venue_maze,
         test_map_mode_random_mazes, test_map_mode_phantoms_in_lanes, test_map_mode_offset_priors,
         test_map_mode_outdated_prior, test_map_mode_official_w1_end_no_contacts, test_map_mode_side_first_leg]


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
