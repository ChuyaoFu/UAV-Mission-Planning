#!/usr/bin/env python3
"""Tests for perception/window.py.

Synthetic frames: a ray-cast scene (orange floor, tan board(s) with a rectangular opening
and a black bottom strip, a far hall wall) seen by the official horizontal OAK-D left camera
(OFFICIAL_K, 640x400). Depth gets OAK-D-like noise (sigma = 0.0063 z^2), speckle dropout,
dropout blobs and 1-px foreground fattening; the grey image is 2x super-sampled.

Run:  .venv/bin/python cartographer/official/perception/test_window.py        (all tests)
      .venv/bin/python cartographer/official/perception/test_window.py --sweep (accuracy table)
Also pytest-compatible (test_* functions, no fixtures).
"""
import math
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')     # honest single-core timing; set before numpy loads
import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(OFFICIAL))
if OFFICIAL not in sys.path:
    sys.path.insert(0, OFFICIAL)

from common.geometry import (F_FLU_FRD, OFFICIAL_K, OFFICIAL_R_BASE_CAM,  # noqa: E402
                             OFFICIAL_SIZE, Frame)
from perception import window as win  # noqa: E402

OUT_DIR = os.path.join(REPO, 'outputs', 'claude-official')
PHOTO3 = ('/private/tmp/claude-501/-Users-chuyao-Documents-Codex-2026-10-31-cartographer-dronehacks/'
          '2198afae-c72b-4034-8353-abe0a35bdca0/images/3.webp')
DOWN = np.array([0.0, 0.0, 1.0])

# ----------------------------------------------------------------------------- renderer

_LAT = np.random.default_rng(1234).random((256, 256)).astype(np.float32)
_LAT = cv2.GaussianBlur(_LAT, (0, 0), 1.0)
_LAT = (_LAT - _LAT.mean()) / _LAT.std()


def _noise(a, b):
    """Smooth value noise ~N(0,1), periodic 256."""
    a = np.mod(np.asarray(a, np.float32), 256).reshape(-1)
    b = np.mod(np.asarray(b, np.float32), 256).reshape(-1)
    sh = np.shape(a)
    n = a.size
    m = max(1024, int(math.ceil(n / 1024.0)) * 1024)
    pa = np.zeros(m, np.float32)
    pb = np.zeros(m, np.float32)
    pa[:n], pb[:n] = a, b
    out = cv2.remap(_LAT, pa.reshape(-1, 1024), pb.reshape(-1, 1024), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_WRAP)
    return out.reshape(-1)[:n].reshape(sh)


def axes_for(n):
    n = np.asarray(n, float) / np.linalg.norm(n)
    e_v = DOWN - (DOWN @ n) * n
    e_v /= np.linalg.norm(e_v)
    return np.cross(n, e_v), e_v


class Board:
    """Vertical tan board. n points to the camera side. holes: list of (hc, vc, w, h) in
    board coordinates (h right as seen from the camera side, v down, metres, from centre)."""

    def __init__(self, center, n, width, height, holes=(), strip_h=0.18, depth_ok=True,
                 tex='tan', depth_holes=()):
        self.tex = tex                     # 'tan' board or 'mesh' (dark net-like wall)
        self.depth_holes = list(depth_holes)   # (hc, vc, w, h): surface present, no depth
        self.c = np.asarray(center, float)
        self.n = np.asarray(n, float) / np.linalg.norm(n)
        self.e_h, self.e_v = axes_for(self.n)
        self.hw, self.hh = width / 2.0, height / 2.0
        self.holes = list(holes)
        self.strip_h = strip_h
        self.depth_ok = depth_ok


class Scene:
    def __init__(self, boards, floor_z=0.0, back_wall=None, hole_depth='wall'):
        """back_wall: (point, normal) of the far hall wall; hole_depth 'wall' gives the far
        wall/floor valid depth, 'nan' returns no depth beyond nan_beyond_m."""
        self.boards = boards
        self.floor_z = floor_z
        self.back_wall = back_wall
        self.hole_depth = hole_depth
        self.nan_beyond_m = 4.5


def _ned_R_for(heading, pitch=0.0, roll=0.0):
    cy, sy = math.cos(heading), math.sin(heading)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R_nb = Rz @ Ry @ Rx
    return R_nb @ F_FLU_FRD @ OFFICIAL_R_BASE_CAM, R_nb


def _cast(scene, cam, R_nc, K, W, H, ss):
    fx, fy, cx, cy = K
    fx, fy = fx * ss, fy * ss
    cx, cy = (cx + 0.5) * ss - 0.5, (cy + 0.5) * ss - 0.5
    vv, uu = np.mgrid[0:H * ss, 0:W * ss].astype(np.float64)
    rc = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], -1).reshape(-1, 3)
    rn = rc @ R_nc.T
    N = len(rn)
    best_t = np.full(N, np.inf)
    inten = np.full(N, np.nan)
    dvalid = np.ones(N, bool)
    surf = np.zeros(N, np.int8)                     # 0 none, 1 board, 2 floor, 3 wall
    # floor
    with np.errstate(divide='ignore', invalid='ignore'):
        tf = (scene.floor_z - cam[2]) / rn[:, 2]
    okf = (rn[:, 2] > 1e-6) & (tf > 0)
    X = cam + tf[:, None] * rn
    m = okf & (tf < best_t)
    best_t[m] = tf[m]
    surf[m] = 2
    inten[m] = 118 + 22 * _noise(X[m, 0] * 6, X[m, 1] * 6) + 10 * _noise(X[m, 0] * 40 + 50, X[m, 1] * 40)
    # back wall
    if scene.back_wall is not None:
        pw, nw = scene.back_wall
        e_h, e_v = axes_for(nw)
        with np.errstate(divide='ignore', invalid='ignore'):
            tw = ((pw - cam) @ nw) / (rn @ nw)
        okw = (tw > 0) & np.isfinite(tw)
        m = okw & (tw < best_t)
        X = cam + tw[:, None] * rn - pw
        hh, vv2 = X @ e_h, X @ e_v
        best_t[m] = tw[m]
        surf[m] = 3
        # busy far hall: furniture-sized blocks, fine clutter
        blocks = (np.floor(hh[m] * 4.0) + 3 * np.floor(vv2[m] * 3.1)) % 5
        inten[m] = 50 + 16 * blocks + 18 * _noise(hh[m] * 14, vv2[m] * 14) + 12 * _noise(hh[m] * 45 + 9, vv2[m] * 45)
    # boards
    for bi, b in enumerate(scene.boards):
        with np.errstate(divide='ignore', invalid='ignore'):
            tb = ((b.c - cam) @ b.n) / (rn @ b.n)
        X = cam + tb[:, None] * rn - b.c
        hh, vv2 = X @ b.e_h, X @ b.e_v
        inside = (tb > 0) & (np.abs(hh) <= b.hw) & (np.abs(vv2) <= b.hh)
        for (hc, vc, w, h) in b.holes:
            inside &= ~((np.abs(hh - hc) < w / 2) & (np.abs(vv2 - vc) < h / 2))
        m = inside & (tb < best_t)
        best_t[m] = tb[m]
        surf[m] = 1
        strip = vv2[m] > b.hh - b.strip_h
        val = 172 + 3.0 * _noise(hh[m] * 7 + 17 * bi, vv2[m] * 7) + 12 * hh[m] / max(b.hw, 1e-3)
        val = np.where(strip, 30 + 2 * _noise(hh[m] * 9, vv2[m] * 9), val)
        if b.tex == 'mesh':                # dark net over a dark hall: same look everywhere
            gl = ((np.mod(hh[m], 0.06) < 0.006) | (np.mod(vv2[m], 0.06) < 0.006))
            val = 38 + 6 * _noise(hh[m] * 11, vv2[m] * 11) + 14 * gl
        inten[m] = val
        dv = np.full(int(m.sum()), bool(b.depth_ok))
        for (hc, vc, w, h) in b.depth_holes:
            dv &= ~((np.abs(hh[m] - hc) < w / 2) & (np.abs(vv2[m] - vc) < h / 2))
        dvalid[m] = dv
    none = ~np.isfinite(best_t)
    inten[none] = 48 + 8 * _noise(uu.reshape(-1)[none] * 0.05 / ss, vv.reshape(-1)[none] * 0.05 / ss)
    dvalid[none] = False
    if scene.hole_depth == 'nan':
        dvalid &= ~((surf >= 2) & (best_t > scene.nan_beyond_m))
    return (best_t.reshape(H * ss, W * ss), inten.reshape(H * ss, W * ss),
            dvalid.reshape(H * ss, W * ss), surf.reshape(H * ss, W * ss))


def render(scene, cam, R_nc, K=OFFICIAL_K, size=OFFICIAL_SIZE, noise=1.0, speckle=0.03,
           blobs=0.0, fatten=1, seed=0, img_noise=2.0, strip_depth=True):
    W, H = size
    rng = np.random.default_rng(seed)
    _, inten, _, _ = _cast(scene, cam, R_nc, K, W, H, 2)
    img = cv2.resize(inten.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    img = img + rng.normal(0, img_noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    t, inten1, dvalid, surf = _cast(scene, cam, R_nc, K, W, H, 1)
    d = t.astype(np.float32).copy()
    if not strip_depth:
        dvalid &= ~((surf == 1) & (inten1 < 60))
    d[~dvalid] = np.nan
    if fatten > 0:                                 # stereo foreground fattening (clean depth)
        big = np.where(np.isfinite(d), d, np.float32(1e6)).astype(np.float32)
        er = cv2.erode(big, np.ones((2 * fatten + 1, 2 * fatten + 1), np.uint8))
        d = np.where(er < 1e5, er, np.float32(np.nan)).astype(np.float32)
    if noise > 0:
        d = d + (rng.normal(0, 1, d.shape) * 0.0063 * noise * d * d).astype(np.float32)
    if blobs > 0:                                  # correlated dropout (weak texture)
        f = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 6)
        f = (f - f.mean()) / f.std()
        thr = np.quantile(f, 1 - blobs)
        d[(f > thr) & (surf == 1)] = np.nan
    if speckle > 0:
        d[rng.random(d.shape) < speckle] = np.nan
    return img, d


def make_case(dist=1.5, yaw_deg=0.0, win_w=0.70, win_h=0.55, board_w=1.30, board_h=1.30,
              win_off=(0.0, -0.10), hole_depth='wall', look_off=(0.0, 0.0), cam_dz=0.0,
              extra_boards=(), heading_offset_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
              board_depth_ok=True, sgm_fill=False, **rk):
    """Window centre at NED (5, 1, -1.05) (1.05 m above floor at z=0). Board normal points
    toward the camera; the camera sits at `dist` from the window centre, yaw_deg off the
    board normal, looking at the window centre shifted by look_off (lateral, vertical)."""
    wc = np.array([5.0, 1.0, -1.05])
    n = np.array([-1.0, 0.0, 0.0])                 # board faces south; camera south of it
    e_h, e_v = axes_for(n)
    bc = wc - win_off[0] * e_h - win_off[1] * e_v
    board = Board(bc, n, board_w, board_h, holes=[(win_off[0], win_off[1], win_w, win_h)],
                  depth_ok=board_depth_ok)
    a = math.radians(yaw_deg)
    cam = wc + dist * (math.cos(a) * n + math.sin(a) * e_h) + np.array([0, 0, cam_dz])
    tgt = wc + look_off[0] * e_h + look_off[1] * e_v
    dv = tgt - cam
    heading = math.atan2(dv[1], dv[0]) + math.radians(heading_offset_deg)
    R_nc, _ = _ned_R_for(heading, math.radians(pitch_deg), math.radians(roll_deg))
    boards = [board] + list(extra_boards)
    scene = Scene(boards, floor_z=0.0, back_wall=(wc - 3.0 * n, n), hole_depth=hole_depth)
    img, depth = render(scene, cam, R_nc, **rk)
    if sgm_fill:
        # stereo SGM smooths the board's disparity into the (dark, weakly textured) opening:
        # depth inside the window = the board plane (+ noise)
        fx, fy, cx, cy = OFFICIAL_K
        vv, uu = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
        rn = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones(uu.shape)], -1) @ R_nc.T
        t = ((bc - cam) @ n) / (rn @ n)
        X = cam + t[..., None] * rn - wc
        inside = (np.abs(X @ e_h) < win_w / 2) & (np.abs(X @ e_v) < win_h / 2) & (t > 0)
        rng = np.random.default_rng(rk.get('seed', 0) + 99)
        depth = depth.copy()
        depth[inside] = (t[inside] * (1 + 0.004 * rng.normal(size=int(inside.sum())))).astype(np.float32)
    fr = Frame(image=img, depth=depth, K=OFFICIAL_K, R_nc=R_nc, cam_ned=cam,
               body_ned=cam.copy(), heading=heading)
    truth = {'center': wc, 'normal': n, 'width': win_w, 'height': win_h, 'e_h': e_h, 'e_v': e_v,
             'board': board}
    return fr, truth


def errors(est, truth):
    ce = float(np.linalg.norm(est.center_ned - truth['center']))
    ne = math.degrees(math.acos(float(np.clip(est.normal_ned @ truth['normal'], -1, 1))))
    return ce, ne, est.width_m - truth['width'], est.height_m - truth['height']


HALF_SPAN, HALF_HEIGHT = 0.20, 0.10      # Sky Beta 7" incl. props (approx), body+props height/2


def run(fr, **kw):
    kw.setdefault('half_span_m', HALF_SPAN)
    kw.setdefault('half_height_m', HALF_HEIGHT)
    return win.detect(fr, **kw)


# ----------------------------------------------------------------------------- sweep

def sweep_cases():
    cases = []
    for hole in ('wall', 'nan'):
        for d in (0.8, 1.2, 1.5, 2.0, 2.5, 3.0):
            cases.append(('frontal', dict(dist=d, hole_depth=hole, seed=int(d * 10))))
        for yaw in (10, 20, 30, 40):
            for d in (1.2, 2.0, 3.0):
                cases.append(('oblique', dict(dist=d, yaw_deg=yaw if d != 2.0 else -yaw,
                                              hole_depth=hole, seed=yaw + int(d * 10))))
        for d, yaw in ((1.5, 0), (2.5, 25), (2.0, -35)):
            cases.append(('noisy', dict(dist=d, yaw_deg=yaw, hole_depth=hole, noise=2.0, speckle=0.10,
                                        blobs=0.25, fatten=2, seed=int(7 + d * 3))))
        for lo in ((0.35, 0.15), (-0.3, -0.12)):
            cases.append(('off-centre', dict(dist=1.6, yaw_deg=15, look_off=lo, hole_depth=hole, seed=3)))
        cases.append(('partial-board', dict(dist=1.0, board_w=2.6, board_h=2.2, win_off=(0.2, -0.2),
                                            hole_depth=hole, seed=11)))
        cases.append(('attitude', dict(dist=1.8, yaw_deg=20, pitch_deg=-6, roll_deg=4, cam_dz=0.25,
                                       hole_depth=hole, seed=12)))
    return cases


def run_sweep(verbose=True):
    rows = []
    for name, kw in sweep_cases():
        fr, truth = make_case(**kw)
        t0 = time.perf_counter()
        est = run(fr)
        dt = (time.perf_counter() - t0) * 1e3
        row = {'name': name, 'kw': kw, 'found': est.found, 'ms': dt, 'src': est.source,
               'reason': est.reason, 'conf': est.confidence, 'clear': est.clearance_ok}
        if est.found:
            ce, ne, dw, dh = errors(est, truth)
            row.update(ce=ce, ne=ne, dw=dw, dh=dh)
        rows.append(row)
        if verbose:
            short = {k: v for k, v in kw.items() if k in ('dist', 'yaw_deg', 'hole_depth', 'look_off')}
            if est.found:
                print('%-13s %-48s %-11s c=%.3f ctr=%5.1fcm n=%4.1fdeg dw=%+5.1f dh=%+5.1fcm %5.1fms' % (
                    name, short, est.source, est.confidence, row['ce'] * 100, row['ne'], row['dw'] * 100,
                    row['dh'] * 100, dt))
            else:
                print('%-13s %-48s NOT FOUND (%s) %5.1fms' % (name, short, est.reason, dt))
    return rows


# ----------------------------------------------------------------------------- tests

def _check(est, truth, ctr, nrm, size, what=''):
    assert est.found, 'not found %s (%s)' % (what, est.reason)
    ce, ne, dw, dh = errors(est, truth)
    assert ce <= ctr, '%s centre err %.3f > %.3f' % (what, ce, ctr)
    assert ne <= nrm, '%s normal err %.2f > %.2f deg' % (what, ne, nrm)
    assert abs(dw) <= size and abs(dh) <= size, '%s size err %.3f %.3f > %.3f' % (what, dw, dh, size)
    assert est.normal_ned @ (est.center_ned - np.zeros(3)) is not None
    return ce, ne, dw, dh


def test_frontal_distances():
    for hole in ('wall', 'nan'):
        for d in (0.8, 1.2, 1.5, 2.0, 2.5, 3.0):
            fr, truth = make_case(dist=d, hole_depth=hole, seed=int(d * 10))
            est = run(fr)
            _check(est, truth, 0.03 if d <= 2.0 else 0.05, 2.0, 0.04 if d <= 2.0 else 0.06,
                   'frontal d=%.1f %s' % (d, hole))
            # normal points to the camera side
            assert est.normal_ned @ (fr.cam_ned - est.center_ned) > 0
            assert est.clearance_ok, 'frontal d=%.1f %s should be clear' % (d, hole)


def test_oblique_true_size():
    for hole in ('wall', 'nan'):
        for yaw in (10, 20, 30, 40):
            for d in (1.2, 2.0, 3.0):
                y = yaw if d != 2.0 else -yaw
                fr, truth = make_case(dist=d, yaw_deg=y, hole_depth=hole, seed=yaw + int(d * 10))
                est = run(fr)
                _check(est, truth, 0.04 if d <= 2.0 else 0.06, 2.5, 0.04 if d <= 2.0 else 0.06,
                       'oblique yaw=%d d=%.1f %s' % (y, d, hole))
                assert est.normal_ned @ (fr.cam_ned - est.center_ned) > 0


def test_noise_dropout_fattening():
    for hole in ('wall', 'nan'):
        for d, yaw in ((1.5, 0), (2.5, 25), (2.0, -35)):
            fr, truth = make_case(dist=d, yaw_deg=yaw, hole_depth=hole, noise=2.0, speckle=0.10,
                                  blobs=0.25, fatten=2, seed=int(7 + d * 3))
            est = run(fr)
            _check(est, truth, 0.06, 3.0, 0.06, 'noisy d=%.1f yaw=%d %s' % (d, yaw, hole))


def test_off_centre_and_attitude():
    for hole in ('wall', 'nan'):
        for lo in ((0.35, 0.15), (-0.3, -0.12)):
            fr, truth = make_case(dist=1.6, yaw_deg=15, look_off=lo, hole_depth=hole, seed=3)
            _check(run(fr), truth, 0.04, 2.5, 0.04, 'off-centre %s %s' % (lo, hole))
        fr, truth = make_case(dist=1.8, yaw_deg=20, pitch_deg=-6, roll_deg=4, cam_dz=0.25,
                              hole_depth=hole, seed=12)
        _check(run(fr), truth, 0.04, 2.5, 0.04, 'pitch/roll %s' % hole)


def test_partial_board_view():
    # board much larger than the view: its outer edges are outside the FOV
    for hole in ('wall', 'nan'):
        fr, truth = make_case(dist=1.0, board_w=2.6, board_h=2.2, win_off=(0.2, -0.2),
                              hole_depth=hole, seed=11)
        _check(run(fr), truth, 0.03, 2.0, 0.04, 'partial board %s' % hole)


def test_window_cut_by_fov_is_rejected():
    # window half outside the image: must not report a (wrong-size) window
    for hole in ('wall', 'nan'):
        fr, truth = make_case(dist=0.8, look_off=(0.0, 0.0), heading_offset_deg=-38,
                              hole_depth=hole, seed=5)
        est = run(fr)
        if est.found:                      # acceptable only if it is actually right
            _check(est, truth, 0.05, 3.0, 0.05, 'cut-by-fov %s' % hole)
        else:
            assert est.reason in ('cut_by_fov', 'no_hole', 'not_enclosed', 'not_rectangular'), est.reason


def test_side_gap_and_open_space_are_not_windows():
    wc = np.array([5.0, 1.0, -1.05])
    n = np.array([-1.0, 0.0, 0.0])
    e_h, e_v = axes_for(n)
    # a second board to the left, separated by a 0.35 m full-height slit
    left = Board(wc + (-0.65 - 0.35 - 0.65) * e_h + 0.10 * e_v, n, 1.30, 1.30)
    for hole in ('wall', 'nan'):
        fr, truth = make_case(dist=2.6, look_off=(-0.6, 0.0), extra_boards=[left], hole_depth=hole,
                              seed=21)
        est = run(fr)
        _check(est, truth, 0.05, 2.5, 0.06, 'window next to slit %s' % hole)
    # no window at all: two boards with a slit, board edges and open space around them
    for hole in ('wall', 'nan'):
        fr, truth = make_case(dist=2.2, win_w=0.0, win_h=0.0, look_off=(-0.6, 0.0),
                              extra_boards=[left], hole_depth=hole, seed=22)
        est = run(fr)
        assert not est.found, 'slit/side gap taken as a window (%s): %s w=%.2f h=%.2f' % (
            hole, est.source, est.width_m, est.height_m)
    # single board on legs, its sides and the space under it visible, no opening
    for hole in ('wall', 'nan'):
        fr, truth = make_case(dist=2.8, win_w=0.0, win_h=0.0, hole_depth=hole, seed=23)
        est = run(fr)
        assert not est.found, 'plain board produced a window (%s)' % hole


def test_floor_gaps_and_thin_frames_are_not_windows():
    """Replica false positives (no prior): the 0.35 m gap under a board seen from close and
    above, and a thin rail frame with nothing behind. The floor crosses the board plane at
    its foot line and must not count as board; thin rails are not a board ring."""
    wc = np.array([5.0, 1.0, -1.05])
    n = np.array([-1.0, 0.0, 0.0])
    e_h, e_v = axes_for(n)
    for hole in ('wall', 'nan'):
        for dist, cam_dz, pitch in ((0.9, -0.15, -20.0), (1.4, -0.15, -14.0), (2.2, 0.35, -6.0)):
            # wide board, bottom 0.35 m above the floor, no opening; camera aimed at the gap
            fr, truth = make_case(dist=dist, win_w=0.0, win_h=0.0, board_w=1.5, board_h=1.35,
                                  win_off=(0.0, -0.32), look_off=(0.0, 0.55), pitch_deg=pitch,
                                  cam_dz=cam_dz, hole_depth=hole, seed=71)
            est = run(fr)
            assert not est.found, 'gap under board taken as window (%s d=%.1f): w=%.2f h=%.2f %s' % (
                hole, dist, est.width_m, est.height_m, est.source)
    # thin rectangular rail frame (3 cm bars) standing in the open: not a window
    rail = Board(wc + np.array([0.0, 0.0, 0.0]), n, 0.80, 0.62, holes=[(0.0, 0.0, 0.74, 0.56)], strip_h=0.0)
    for hole in ('wall', 'nan'):
        fr, truth = make_case(dist=1.6, win_w=0.0, win_h=0.0, board_w=0.01, board_h=0.01,
                              extra_boards=[rail], hole_depth=hole, seed=72)
        est = run(fr)
        assert not est.found, 'thin rail frame taken as window (%s): %s w=%.2f h=%.2f' % (
            hole, est.source, est.width_m, est.height_m)
    # the real window is still found with the floor height given (e.g. from the ToF)
    for hole in ('wall', 'nan'):
        fr, truth = make_case(dist=1.6, yaw_deg=20, hole_depth=hole, seed=73)
        _check(run(fr, floor_z_ned=0.0), truth, 0.03, 2.0, 0.04, 'floor_z given %s' % hole)
        _check(run(fr, floor_z_ned=0.02), truth, 0.03, 2.0, 0.04, 'floor_z +2cm %s' % hole)


def test_mesh_wall_and_sgm_filled_opening():
    """Replica stereo round: (1) the cage net returns depth like a wall; with stereo holes it
    looks like a board with an opening. The image shows the same mesh on both sides of the
    'edges', so without a prior it must be rejected (grey-level step rule). (2) SGM fills a
    real opening with the board's disparity: depth alone sees a solid board; with a prior and
    the measured size the known-size search on the rectified board recovers it."""
    wc = np.array([5.0, 1.0, -1.05])
    n = np.array([-1.0, 0.0, 0.0])
    e_h, e_v = axes_for(n)
    net = Board(wc + 0.10 * e_v, n, 2.4, 1.6, strip_h=0.0, tex='mesh',
                depth_holes=[(0.0, -0.10, 0.62, 0.48)])
    for hole in ('wall', 'nan'):
        fr, truth = make_case(dist=2.0, yaw_deg=10, win_w=0.0, win_h=0.0, board_w=0.01, board_h=0.01,
                              extra_boards=[net], hole_depth=hole, seed=81)
        for prm in (None, win.WindowParams(min_edges_no_prior=0)):
            est = run(fr, params=prm)
            assert not est.found, 'mesh wall depth hole taken as window (%s): %s w=%.2f h=%.2f' % (
                hole, est.source, est.width_m, est.height_m)
    for hole in ('wall', 'nan'):
        for d, yaw in ((1.8, 0.0), (2.6, 25.0)):
            fr, truth = make_case(dist=d, yaw_deg=yaw, hole_depth=hole, sgm_fill=True, seed=82)
            assert not run(fr).found or run(fr).source != 'depth'
            prior = truth['center'] + np.array([0.25, -0.2, 0.1])
            est = run(fr, prior_center_ned=prior, known_size_m=(truth['width'], truth['height']))
            _check(est, truth, 0.03, 2.5, 0.03, 'sgm-filled opening %s d=%.1f' % (hole, d))
            assert est.source == 'prior+size', est.source


def test_clearance_ok():
    # half_span 0.20, half_height 0.10, margin 0.06 -> needs width >= 0.52, height >= 0.32
    for (w, h, expect) in ((0.70, 0.55, True), (0.56, 0.36, True), (0.48, 0.55, False),
                           (0.70, 0.29, False), (0.40, 0.25, False)):
        for hole in ('wall', 'nan'):
            fr, truth = make_case(dist=1.6, yaw_deg=20, win_w=w, win_h=h, hole_depth=hole, seed=31)
            est = run(fr, margin_m=0.06)
            _check(est, truth, 0.04, 2.5, 0.04, 'clearance w=%.2f h=%.2f %s' % (w, h, hole))
            assert est.clearance_ok == expect, 'clearance w=%.2f h=%.2f %s: got %s (%s)' % (
                w, h, hole, est.clearance_ok, est.reason)


def test_prior_selects_window():
    wc = np.array([5.0, 1.0, -1.05])
    n = np.array([-1.0, 0.0, 0.0])
    e_h, e_v = axes_for(n)
    other_c = wc - 1.6 * e_h + 0.10 * e_v
    other = Board(other_c, n, 1.30, 1.30, holes=[(0.0, -0.10, 0.60, 0.45)])
    fr, truth = make_case(dist=2.8, look_off=(-0.8, 0.0), extra_boards=[other], seed=41)
    est = run(fr, prior_center_ned=wc + np.array([0.3, -0.3, 0.2]))
    _check(est, truth, 0.06, 2.5, 0.06, 'prior right')
    est2 = run(fr, prior_center_ned=other_c - 0.10 * e_v + np.array([-0.3, 0.2, 0.0]))
    assert est2.found and np.linalg.norm(est2.center_ned - (other_c - 0.10 * e_v)) < 0.06
    # prior far from everything -> nothing
    est3 = run(fr, prior_center_ned=wc + np.array([0.0, 4.0, 0.0]))
    assert not est3.found


def test_weak_board_depth_fallbacks():
    # board depth almost gone (textureless) -> image quad + plane or measured size (PnP)
    fr, truth = make_case(dist=1.8, yaw_deg=15, hole_depth='nan', blobs=0.6, speckle=0.2, seed=51)
    est = run(fr, known_size_m=(0.70, 0.55))
    _check(est, truth, 0.08, 6.0, 0.08, 'sparse board')
    fr, truth = make_case(dist=1.8, yaw_deg=15, hole_depth='nan', board_depth_ok=False, seed=52)
    est = run(fr)
    assert not est.found or est.source != 'depth'
    est = run(fr, known_size_m=(0.70, 0.55))
    assert est.found and est.source == 'image+size', est.reason
    ce, ne, dw, dh = errors(est, truth)
    assert ce < 0.15 and ne < 10.0, (ce, ne)
    assert est.confidence <= 0.45


def test_timing_640x400():
    frames = [make_case(dist=d, yaw_deg=y, hole_depth=h, seed=60 + i)[0]
              for i, (d, y, h) in enumerate(((1.0, 0, 'wall'), (2.0, 30, 'nan'), (3.0, -20, 'wall'),
                                             (1.5, 40, 'nan')))]
    ts = []
    for _ in range(3):
        for fr in frames:
            t0 = time.perf_counter()
            run(fr)
            ts.append((time.perf_counter() - t0) * 1e3)
    med, mx = float(np.median(ts)), float(np.max(ts))
    print('timing: median %.1f ms, max %.1f ms over %d calls' % (med, mx, len(ts)))
    assert med < 50.0, med


def test_photo3_qualitative():
    if not os.path.exists(PHOTO3):
        print('photo 3 not available, skipped')
        return
    bgr = cv2.imread(PHOTO3, cv2.IMREAD_COLOR)
    assert bgr is not None
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    quads = win.find_window_quads(gray)
    dt = (time.perf_counter() - t0) * 1e3
    assert quads, 'no rectangular opening found in photo 3'
    q = quads[0]['quad']
    c = q.mean(0) / np.array([W, H])
    # hand-labelled opening in photo 3 (fractions of the image): x 0.379..0.634, y 0.262..0.538
    q = q.astype(np.float64)
    box = np.array([q[:, 0].min() / W, q[:, 0].max() / W, q[:, 1].min() / H, q[:, 1].max() / H])
    assert np.all(np.abs(box - [0.379, 0.634, 0.262, 0.538]) < 0.02), box.round(3)
    vis = bgr.copy()
    for i, qd in enumerate(quads[:3]):
        col = (0, 255, 0) if i == 0 else (0, 200, 255)
        cv2.polylines(vis, [qd['quad'].astype(np.int32)], True, col, 6 if i == 0 else 3)
        cv2.putText(vis, '#%d s=%.3f' % (i, qd['score']), tuple(int(x) for x in qd['quad'][0] + [0, -12]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)
    cv2.putText(vis, 'image-only window quads (no depth), %.0f ms at 640px' % dt, (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'window-photo3.png')
    cv2.imwrite(out, cv2.resize(vis, (W // 2, H // 2), interpolation=cv2.INTER_AREA))
    print('photo3: %d quads, best box x %.3f-%.3f y %.3f-%.3f (label 0.379-0.634, 0.262-0.538), '
          'scores %s, %.0f ms -> %s' % (len(quads), box[0], box[1], box[2], box[3],
                                        [round(x['score'], 3) for x in quads[:3]], dt, out))


def monte_carlo(n=200, seed=2026, verbose=False, dist=(0.8, 3.0), blobs=(0.0, 0.3), yaw=40.0,
                speckle=(0.0, 0.12)):
    """Random scenes: distance 0.8-3 m, yaw +-40 deg, attitude +-5 deg, window 0.45-0.9 x
    0.3-0.7 m, noise x0.5-2, speckle 0-12 %, blobs 0-30 %, fattening 0-2 px, hole with or
    without depth. Expected: found iff the window and a 5 cm board ring are in view."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        ww, wh = rng.uniform(0.45, 0.9), rng.uniform(0.3, 0.7)
        kw = dict(dist=rng.uniform(*dist), yaw_deg=rng.uniform(-yaw, yaw), win_w=ww, win_h=wh,
                  board_w=ww + rng.uniform(0.4, 0.8), board_h=wh + rng.uniform(0.5, 0.9),
                  win_off=(rng.uniform(-0.1, 0.1), rng.uniform(-0.2, 0.0)),
                  hole_depth=str(rng.choice(['wall', 'nan'])),
                  look_off=(rng.uniform(-0.3, 0.3), rng.uniform(-0.15, 0.15)),
                  pitch_deg=rng.uniform(-5, 5), roll_deg=rng.uniform(-5, 5), cam_dz=rng.uniform(-0.2, 0.3),
                  noise=rng.uniform(0.5, 2.0), speckle=rng.uniform(*speckle), blobs=rng.uniform(*blobs),
                  fatten=int(rng.integers(0, 3)), seed=1000 + i)
        fr, truth = make_case(**kw)
        e_h, e_v = truth['e_h'], truth['e_v']
        ring = [truth['center'] + sx * (ww / 2 + 0.05) * e_h + sy * (wh / 2 + 0.05) * e_v
                for sx in (-1, 1) for sy in (-1, 1)]
        uv, z = fr.project(np.array(ring))
        visible = bool(np.all(z > 0) and np.all((uv[:, 0] > 1) & (uv[:, 0] < 638) & (uv[:, 1] > 1) & (uv[:, 1] < 398)))
        t0 = time.perf_counter()
        est = run(fr)
        dt = (time.perf_counter() - t0) * 1e3
        rng_true = float(np.linalg.norm(truth['center'] - fr.cam_ned))
        exp_clear = ((ww / 2 - HALF_SPAN >= 0.06) and (wh / 2 - HALF_HEIGHT >= 0.06)
                     and rng_true <= win.WindowParams().clear_max_range)
        r = dict(i=i, kw=kw, visible=visible, found=est.found, ms=dt, src=est.source, reason=est.reason,
                 conf=est.confidence, clear=est.clearance_ok, exp_clear=exp_clear)
        if est.found:
            ce, ne, dw, dh = errors(est, truth)
            r.update(ce=ce, ne=ne, dw=dw, dh=dh)
        rows.append(r)
        if verbose and ((visible != est.found) or (est.found and r['ce'] > 0.05) or dt > 40):
            print('case', i, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in kw.items()},
                  'visible', visible, 'found', est.found, est.source, est.reason,
                  'ce %.3f' % r.get('ce', float('nan')), '%.0f ms' % dt)
    return rows


def summarize(rows):
    vis = [r for r in rows if r['visible']]
    inv = [r for r in rows if not r['visible']]
    fv = [r for r in vis if r['found']]
    print('visible windows: found %d/%d; not fully visible: reported %d/%d' % (
        len(fv), len(vis), sum(r['found'] for r in inv), len(inv)))
    bad_inv = [r for r in inv if r['found']]
    if bad_inv:
        print('  not-fully-visible but reported, centre err (m):', [round(r['ce'], 3) for r in bad_inv])
    for key, lab in (('ce', 'centre err m'), ('ne', 'normal err deg'), ('dw', 'width err m'), ('dh', 'height err m')):
        a = np.abs([r[key] for r in fv])
        print('  %-15s median %.4f  p95 %.4f  max %.4f' % (lab, np.median(a), np.percentile(a, 95), a.max()))
    near = [r for r in fv if r['kw']['dist'] <= 2.0]
    far = [r for r in fv if r['kw']['dist'] > 2.0]
    for nm, grp in (('d<=2m', near), ('d>2m', far)):
        if grp:
            a = np.abs([r['ce'] for r in grp])
            print('  centre err %s: median %.4f p95 %.4f (n=%d)' % (nm, np.median(a), np.percentile(a, 95), len(grp)))
    srcs = {}
    for r in fv:
        srcs[r['src']] = srcs.get(r['src'], 0) + 1
    print('  sources', srcs)
    cl = [r for r in fv]
    wrong = [r for r in cl if r['clear'] != r['exp_clear']]
    print('  clearance_ok wrong: %d/%d (false-OK %d, conservative %d)' % (
        len(wrong), len(cl), sum(r['clear'] and not r['exp_clear'] for r in wrong),
        sum(r['exp_clear'] and not r['clear'] for r in wrong)))
    for r in wrong[:8]:
        print('    w=%.3f h=%.3f est %.3f %.3f conf %.2f clear=%s %s' % (r['kw']['win_w'], r['kw']['win_h'],
              r['kw']['win_w'] + r['dw'], r['kw']['win_h'] + r['dh'], r['conf'], r['clear'], r['reason']))
    ms = [r['ms'] for r in rows]
    print('  time ms: median %.1f p95 %.1f max %.1f' % (np.median(ms), np.percentile(ms, 95), max(ms)))
    miss = [r for r in vis if not r['found']]
    reasons = {}
    for r in miss:
        reasons[r['reason']] = reasons.get(r['reason'], 0) + 1
    print('  misses by reason', reasons)


def main():
    if '--mc' in sys.argv:
        n = int(sys.argv[sys.argv.index('--mc') + 1])
        summarize(monte_carlo(n, verbose=True))
        return 0
    if '--sweep' in sys.argv:
        rows = run_sweep()
        f = [r for r in rows if r['found']]
        print('found %d/%d' % (len(f), len(rows)))
        for key in ('ce', 'ne', 'dw', 'dh'):
            a = np.abs([r[key] for r in f])
            print('%s: median %.4f p95 %.4f max %.4f' % (key, np.median(a), np.percentile(a, 95), a.max()))
        ms = [r['ms'] for r in rows]
        print('ms: median %.1f p95 %.1f max %.1f' % (np.median(ms), np.percentile(ms, 95), max(ms)))
        return 0
    names = [k for k in sorted(globals()) if k.startswith('test_') and callable(globals()[k])]
    if len(sys.argv) > 1:
        names = [k for k in names if any(a in k for a in sys.argv[1:])]
    fails = 0
    for k in names:
        t0 = time.perf_counter()
        try:
            globals()[k]()
            print('PASS %-45s %.1fs' % (k, time.perf_counter() - t0))
        except AssertionError as e:
            fails += 1
            print('FAIL %-45s %s' % (k, e))
    print('%d/%d passed' % (len(names) - fails, len(names)))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
