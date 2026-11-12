#!/usr/bin/env python3
"""Synthetic tests for perception/floor.py.

Run:  .venv/bin/python cartographer/official/perception/test_floor.py   (or pytest)

A small ray caster renders the official horizontal OAK-D view (OFFICIAL_K,
640x400) of: the floor (orange print ~ grey 120, dark painted rocks, light
painted patches), real rock cones, wooden beams, panels/walls and the white
landing block. Depth gets stereo-like noise (sigma_z = 0.15 px * z^2 / (fx*B),
part spatially correlated), blob-shaped holes on the glossy floor, optional
mirror-image returns below the floor and optional flying pixels at depth edges.
"""
import math
import os
import sys
import time

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # cartographer/official
from common.geometry import (Frame, camera_pose_ned, OFFICIAL_K, OFFICIAL_SIZE,  # noqa: E402
                             OFFICIAL_R_BASE_CAM, OFFICIAL_T_BASE_CAM)
from perception import floor as F                    # noqa: E402

FX = OFFICIAL_K[0]
B = 0.075


# =========================================================================== scene + renderer
def quat_ypr(yaw, pitch, roll):
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def cone(x, y, r, h, grey=70):
    return dict(kind='cone', x=x, y=y, r=r, h=h, grey=grey)


def box(x, y, length, width, h0, h1, yaw=0.0, grey=200, tex=8.0, side=0.9):
    """Box standing between heights h0..h1 above the floor (length along yaw)."""
    return dict(kind='box', x=x, y=y, L=length, W=width, h0=h0, h1=h1, yaw=yaw, grey=grey, tex=tex, side=side)


def paint(x, y, a, b, ang=0.0, grey=55):
    """Ellipse painted ON the floor (no relief)."""
    return dict(kind='paint', x=x, y=y, a=a, b=b, ang=ang, grey=grey)


def pad(x, y, size=0.60, yaw=0.0, body=132, dark=92, white=165):
    """Flat landing pad printed on the floor (venue photos 17/18): body, dark H + 4 edge bars,
    thin dashed white circle, rounded corners. Defaults = luma of photo 18."""
    return dict(kind='pad', x=x, y=y, S=size, yaw=yaw, body=body, dark=dark, white=white)


def tape(x, y, length, width, yaw=0.0, line=0.05, grey=35):
    """Black tape rectangle outline on the floor (line width `line`)."""
    return dict(kind='tape', x=x, y=y, L=length, W=width, yaw=yaw, line=line, grey=grey)


def floor_grey(grey):
    return dict(kind='floorgrey', grey=grey)


FLAT = ('paint', 'pad', 'tape', 'floorgrey')


def _pad_grey(u, v, s):
    """Pad colour at pad-normalised coords (u, v in [-0.5, 0.5]); NaN outside the pad."""
    r_c = 0.06
    au, av = np.abs(u), np.abs(v)
    inside = (au <= 0.5) & (av <= 0.5) & ~((au > 0.5 - r_c) & (av > 0.5 - r_c) & (np.hypot(au - (0.5 - r_c), av - (0.5 - r_c)) > r_c))
    g = np.full(u.shape, float(s['body']))
    rad = np.hypot(u, v); ang = np.arctan2(v, u)
    circ = (np.abs(rad - 0.36) <= 0.006) & (np.cos(4 * ang) < 0.7)                 # dashed white circle
    g = np.where(circ, float(s['white']), g)
    hw, hh, sw = 0.12, 0.165, 0.033
    dark = ((np.abs(au - hw) <= sw) & (av <= hh)) | ((au <= hw) & (av <= sw))
    bl, bw, bd = 0.10, 0.0225, 0.39
    dark |= ((au <= bl) & (np.abs(av - bd) <= bw)) | ((av <= bl) & (np.abs(au - bd) <= bw))
    g = np.where(dark, float(s['dark']), g)
    return np.where(inside, g, np.nan)


def _hit_cone(o, r, s, zf):
    k = s['r'] / s['h']; za = zf - s['h']
    Dx = o[..., 0] - s['x']; Dy = o[..., 1] - s['y']; Dz = o[..., 2] - za
    rx, ry, rz = r[..., 0], r[..., 1], r[..., 2]
    A = rx * rx + ry * ry - k * k * rz * rz
    Bq = 2 * (Dx * rx + Dy * ry - k * k * Dz * rz)
    C = Dx * Dx + Dy * Dy - k * k * Dz * Dz
    with np.errstate(invalid='ignore', divide='ignore'):
        disc = Bq * Bq - 4 * A * C
        sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
        t = np.full(A.shape, np.inf)
        for root in ((-Bq - sq) / (2 * A), (-Bq + sq) / (2 * A)):
            z = o[..., 2] + root * rz
            ok = np.isfinite(root) & (root > 1e-6) & (z >= za) & (z <= zf)
            t = np.where(ok & (root < t), root, t)
    return t


def _hit_box(o, r, s, zf):
    c, si = math.cos(s['yaw']), math.sin(s['yaw'])
    dx = o[..., 0] - s['x']; dy = o[..., 1] - s['y']
    lo_ = [np.full(dx.shape, -s['L'] / 2), np.full(dx.shape, -s['W'] / 2), np.full(dx.shape, zf - s['h1'])]
    hi_ = [-lo_[0], -lo_[1], np.full(dx.shape, zf - s['h0'])]
    oo = [c * dx + si * dy, -si * dx + c * dy, o[..., 2] * np.ones(dx.shape)]
    rr = [c * r[..., 0] + si * r[..., 1], -si * r[..., 0] + c * r[..., 1], r[..., 2]]
    tn = np.full(dx.shape, -np.inf); tf = np.full(dx.shape, np.inf)
    with np.errstate(invalid='ignore', divide='ignore'):
        for a in range(3):
            t1 = (lo_[a] - oo[a]) / rr[a]; t2 = (hi_[a] - oo[a]) / rr[a]
            tn = np.fmax(tn, np.fmin(t1, t2)); tf = np.fmin(tf, np.fmax(t1, t2))
    ok = (tn <= tf) & (tn > 1e-6)
    return np.where(ok, tn, np.inf)


def _tex(x, y, amp):
    return amp * (np.sin(37.0 * x + 3.1 * np.sin(11.0 * y)) * np.cos(29.0 * y + 2.3 * np.sin(13.0 * x)))


@np.errstate(invalid='ignore', over='ignore')
def _cast(o, r, scene, floor=True, zf=0.0):
    """Nearest hit. Returns t (inf = none), id (-1 none, 0 floor, i+1 object i), grey."""
    t = np.full(r.shape[:2], np.inf); hid = np.full(r.shape[:2], -1); grey = np.zeros(r.shape[:2])
    if floor:
        with np.errstate(invalid='ignore', divide='ignore'):
            tfl = (zf - o[..., 2]) / r[..., 2]
        tfl = np.where((r[..., 2] > 1e-9) & (tfl > 1e-6), tfl, np.inf)
        p = o + tfl[..., None] * r
        base = next((float(s['grey']) for s in scene if s['kind'] == 'floorgrey'), 120.0)
        g = base + _tex(p[..., 0], p[..., 1], 6.0)
        for s in scene:
            if s['kind'] == 'paint':
                c, si = math.cos(s['ang']), math.sin(s['ang'])
                dx = p[..., 0] - s['x']; dy = p[..., 1] - s['y']
                u = (c * dx + si * dy) / s['a']; v = (-si * dx + c * dy) / s['b']
                g = np.where(u * u + v * v <= 1.0, s['grey'] + _tex(p[..., 0], p[..., 1], 10.0), g)
            elif s['kind'] == 'pad':
                c, si = math.cos(s['yaw']), math.sin(s['yaw'])
                dx = p[..., 0] - s['x']; dy = p[..., 1] - s['y']
                pg = _pad_grey((c * dx + si * dy) / s['S'], (-si * dx + c * dy) / s['S'], s)
                g = np.where(np.isfinite(pg), pg + _tex(p[..., 0], p[..., 1], 3.0), g)
            elif s['kind'] == 'tape':
                c, si = math.cos(s['yaw']), math.sin(s['yaw'])
                dx = p[..., 0] - s['x']; dy = p[..., 1] - s['y']
                au, av = np.abs(c * dx + si * dy), np.abs(-si * dx + c * dy)
                hl, hw_, w = s['L'] / 2, s['W'] / 2, s['line'] / 2
                on = (((np.abs(au - hl) <= w) & (av <= hw_ + w)) | ((np.abs(av - hw_) <= w) & (au <= hl + w)))
                g = np.where(on, float(s['grey']), g)
        hit = np.isfinite(tfl)
        t = np.where(hit, tfl, t); hid = np.where(hit, 0, hid); grey = np.where(hit, g, grey)
    for i, s in enumerate(scene):
        if s['kind'] in FLAT:
            continue
        ti = _hit_cone(o, r, s, zf) if s['kind'] == 'cone' else _hit_box(o, r, s, zf)
        closer = ti < t
        if not closer.any():
            continue
        p = o + np.where(closer, ti, 0)[..., None] * r
        if s['kind'] == 'cone':
            g = s['grey'] + _tex(p[..., 0], p[..., 1] + p[..., 2], 12.0)
        else:
            top = np.abs(p[..., 2] - (zf - s['h1'])) < 1e-6
            g = (s['grey'] + _tex(p[..., 0] + p[..., 2], p[..., 1] - p[..., 2], s['tex'])) * np.where(top, 1.0, s['side'])
        t = np.where(closer, ti, t); hid = np.where(closer, i + 1, hid); grey = np.where(closer, g, grey)
    return t, hid, grey


def render(scene, cam, R_nc, K=OFFICIAL_K, size=OFFICIAL_SIZE, mirror=False):
    """Returns grey image (float), optical depth (inf = no hit), hit id, mirror depth (floor pixels)."""
    W, H = size
    fx, fy, cx, cy = K
    uu, vv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    ro = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], -1)
    r = ro @ np.asarray(R_nc).T                 # t along r == optical depth
    o = np.broadcast_to(np.asarray(cam, float), r.shape)
    t, hid, grey = _cast(o, r, scene)
    vdepth = None
    if mirror:                                  # glossy floor: virtual image of the scene below the floor
        p = o + np.where(np.isfinite(t), t, 0)[..., None] * r
        r2 = r * np.array([1.0, 1.0, -1.0])
        t2, _, _ = _cast(p, r2, scene, floor=False)
        vdepth = np.where(hid == 0, t + t2, np.inf)
    return grey, t, hid, vdepth


def degrade(depth, hid, rng, noise_px=0.15, floor_dropout=0.0, iid_dropout=0.02, flying=False,
            mirror_depth=None, mirror_frac=0.0, max_depth=8.0):
    z = depth.astype(np.float64).copy()
    z[~np.isfinite(z)] = np.nan
    if flying:                                  # mixed pixels across horizontal depth edges
        zl = np.roll(z, 1, 1); zr = np.roll(z, -1, 1)
        jump = np.abs(zl - zr) > 0.15 * np.fmin(zl, zr)
        sel = jump & (rng.random(z.shape) < 0.8)
        z = np.where(sel, 0.5 * (zl + zr), z)
    sz = noise_px * z * z / (FX * B)
    corr = cv2.GaussianBlur(rng.standard_normal(z.shape).astype(np.float32), (0, 0), 3.0)
    corr /= corr.std() + 1e-9
    z = z + sz * (0.5 * rng.standard_normal(z.shape) + 0.85 * corr)

    def blobs(frac, sigma):
        f = cv2.GaussianBlur(rng.standard_normal(z.shape).astype(np.float32), (0, 0), sigma)
        fl = hid == 0
        if frac <= 0 or not fl.any():
            return np.zeros(z.shape, bool)
        return fl & (f < np.quantile(f[fl], frac))
    if mirror_depth is not None and mirror_frac > 0:
        m = blobs(mirror_frac, 6.0)
        z = np.where(m, mirror_depth, z)
    z[blobs(floor_dropout, 8.0)] = np.nan
    z[rng.random(z.shape) < iid_dropout] = np.nan
    z[~np.isfinite(z) | (z > max_depth) | (z < 0.2)] = np.nan
    return z.astype(np.float32)


def make_frame(scene, cam_h=0.6, yaw=0.0, pitch=0.0, roll=0.0, err=(0.0, 0.0), seed=1, bgr=None,
               xy=(0.0, 0.0), **deg):
    """Camera at height cam_h above the floor (NED z = -cam_h), body yaw/pitch/roll.
    err = (pitch, roll) error (rad) of the attitude given to the algorithm."""
    _, cam0 = camera_pose_ned(quat_ypr(yaw, pitch, roll), np.zeros(3), OFFICIAL_R_BASE_CAM, OFFICIAL_T_BASE_CAM)
    body = np.array([xy[0], xy[1], -cam_h - cam0[2]])
    R_true, cam = camera_pose_ned(quat_ypr(yaw, pitch, roll), body, OFFICIAL_R_BASE_CAM, OFFICIAL_T_BASE_CAM)
    R_rep, _ = camera_pose_ned(quat_ypr(yaw, pitch + err[0], roll + err[1]), body, OFFICIAL_R_BASE_CAM,
                               OFFICIAL_T_BASE_CAM)
    rng = np.random.default_rng(seed)
    mirror = deg.get('mirror_frac', 0.0) > 0
    grey, t, hid, vd = render(scene, cam, R_true, mirror=mirror)
    depth = degrade(t, hid, rng, mirror_depth=vd, **deg)
    img = np.clip(grey + rng.normal(0, 3.0, grey.shape), 0, 255).astype(np.uint8)
    if bgr is not None:                         # bgr(grey, hid) -> HxWx3 colour image
        img = bgr(img, hid)
    fr = Frame(image=img, depth=depth, K=OFFICIAL_K, R_nc=R_rep, cam_ned=cam, body_ned=body, heading=yaw)
    fr.hid = hid                                # ground truth for tests only
    return fr


# =========================================================================== scenes
PAINTED = [paint(2.0, -0.45, 0.30, 0.18, 0.4, 55), paint(3.0, 0.45, 0.35, 0.22, -0.3, 60),
           paint(1.35, 0.25, 0.18, 0.12, 0.0, 50), paint(3.8, -0.2, 0.45, 0.3, 0.9, 58)]
LIGHT_PATCH = paint(2.4, -1.0, 0.40, 0.30, 0.2, 225)
ROCKS = [cone(1.8, 0.65, 0.15, 0.25), cone(2.6, -0.25, 0.18, 0.35), cone(3.4, 0.95, 0.12, 0.20),
         cone(3.5, -1.15, 0.20, 0.40)]
BEAM = box(2.2, 1.45, 1.2, 0.05, 0.0, 0.07, grey=170)
PANEL = box(3.9, -1.7, 1.0, 0.03, 0.10, 1.90, yaw=0.3, grey=215, tex=4.0)
BASE_SCENE = PAINTED + [LIGHT_PATCH] + ROCKS + [BEAM, PANEL]

BLOCK = box(2.5, 0.2, 0.6, 0.5, 0.0, 0.35, grey=215, tex=10.0)
BLOCK_SCENE = PAINTED[:2] + [paint(1.7, -0.8, 0.45, 0.35, 0.0, 228), BLOCK, PANEL,
                             cone(1.6, 0.9, 0.15, 0.3), cone(3.3, -0.6, 0.18, 0.35)]


def _in_ellipse(p, s, grow=0.0):
    c, si = math.cos(s['ang']), math.sin(s['ang'])
    dx = p[:, 0] - s['x']; dy = p[:, 1] - s['y']
    u = (c * dx + si * dy) / (s['a'] + grow); v = (-si * dx + c * dy) / (s['b'] + grow)
    return u * u + v * v <= 1.0


def _near_object(p, s, margin):
    if s['kind'] == 'cone':
        return np.hypot(p[:, 0] - s['x'], p[:, 1] - s['y']) <= s['r'] + margin
    c, si = math.cos(s['yaw']), math.sin(s['yaw'])
    dx = p[:, 0] - s['x']; dy = p[:, 1] - s['y']
    return (np.abs(c * dx + si * dy) <= s['L'] / 2 + margin) & (np.abs(-si * dx + c * dy) <= s['W'] / 2 + margin)


# =========================================================================== floor tests
def test_floor_altitudes_level():
    for cam_h in (0.3, 0.6, 0.9, 1.2):
        fr = make_frame(BASE_SCENE, cam_h=cam_h, floor_dropout=0.3, seed=int(cam_h * 10))
        for prior in (None, 0.06, 0.02):
            f = F.fit_floor(fr, prior)
            assert f.found, (cam_h, prior, f.reason)
            assert abs(f.d_stereo) < 0.02, (cam_h, prior, f.d_stereo)       # the stereo plane itself
            assert abs(f.height_above(fr.cam_ned) - cam_h) < 0.02
            assert f.tilt_deg < 1.5, (cam_h, f.tilt_deg)
            if prior is None:
                assert f.source == 'depth' and f.d == f.d_stereo
            else:                                   # metric d: between the plane and the prior
                assert f.source == 'depth+prior' and min(0, prior) - 1e-9 <= f.d <= max(0.02, prior)
            if prior == 0.02:                       # ToF-like prior error -> metric d within 2 cm
                assert abs(f.d) < 0.02, (cam_h, f.d)


def test_floor_reflective_sparse():
    """Glossy floor: 90 % of floor depth missing in blobs, plus mirror images below the floor."""
    scene = BASE_SCENE + [box(4.6, 0.0, 0.05, 4.0, 0.0, 2.5, grey=150)]
    for cam_h in (0.5, 1.0):
        fr = make_frame(scene, cam_h=cam_h, floor_dropout=0.9, mirror_frac=0.3, seed=3)
        fl = fr.hid == 0
        dens = np.isfinite(fr.depth[fl]).mean()
        assert dens < 0.12, dens
        for prior in (None, 0.05):
            f = F.fit_floor(fr, prior)
            assert f.found, (cam_h, prior, f.reason)
            assert abs(f.d_stereo) < 0.03, (cam_h, prior, f.d_stereo)


def test_floor_mostly_wall():
    """Wall fills ~85 % of the image, floor is a thin sparse strip below it."""
    wall = box(2.6, 0.0, 0.05, 8.0, 0.0, 3.0, grey=160, tex=20.0)
    fr = make_frame([wall] + PAINTED[:1], cam_h=0.9, floor_dropout=0.7, seed=5)
    wall_frac = np.isfinite(fr.depth[fr.hid == 1]).sum() / np.isfinite(fr.depth).sum()
    assert wall_frac > 0.85, wall_frac
    f = F.fit_floor(fr)
    assert f.found and abs(f.d) < 0.03, (f.found, f.d, f.reason)
    assert f.inlier_frac < 0.15
    # corridor: two side walls along the view + far wall, low camera
    corr = [box(3.0, -0.9, 6.0, 0.05, 0.0, 2.0, grey=200), box(3.0, 0.9, 6.0, 0.05, 0.0, 2.0, grey=190),
            box(4.5, 0.0, 0.05, 1.8, 0.0, 2.0, grey=150)]
    fr = make_frame(corr + PAINTED, cam_h=0.5, floor_dropout=0.8, seed=6)
    f = F.fit_floor(fr)
    assert f.found and abs(f.d) < 0.03, (f.found, f.d, f.reason)


def test_floor_not_visible():
    wall = box(1.0, 0.0, 0.05, 6.0, 0.0, 3.0, grey=160, tex=20.0)
    fr = make_frame([wall], cam_h=0.9, seed=7)
    assert not (fr.hid == 0).any()
    f = F.fit_floor(fr)
    assert not f.found and not f.usable and f.source == 'none'
    f = F.fit_floor(fr, prior_floor_d=0.01)
    assert not f.found and f.usable and f.d == 0.01 and f.source == 'prior'
    # no depth at all
    fr.depth[:] = np.nan
    f = F.fit_floor(fr, 0.0)
    assert not f.found and f.reason == 'no_depth'


def test_floor_prior_preferred():
    """A big raised platform (top 0.3 m) fills the middle of the view, the floor is only a
    strip in front and at the sides: the prior decides which horizontal surface is wanted."""
    plat = box(3.0, 0.0, 2.4, 3.0, 0.0, 0.30, grey=110, tex=15.0)
    fr = make_frame([plat] + PAINTED[:1], cam_h=0.6, floor_dropout=0.5, seed=8)
    f = F.fit_floor(fr, prior_floor_d=0.06)
    assert f.found and abs(f.d_stereo) < 0.03, (f.found, f.d_stereo, f.reason)
    f = F.fit_floor(fr, prior_floor_d=-0.25)
    assert f.found and abs(f.d_stereo + 0.30) < 0.03, (f.found, f.d_stereo)
    f = F.fit_floor(fr, prior_floor_d=-1.5)                 # nothing near a wrong prior
    assert not f.found and f.d == -1.5 and f.source == 'prior'


def test_floor_stereo_range_bias_and_tof_prior():
    """Hypothetical uniform 7 % range error (e.g. a stereo calibration error; the venue replica
    showed no such global bias, +1.7 % median on the floor): the plane rises by 7 % of the camera
    height. Heights stay self-consistent (paint stays at h~0); the metric d and the landing
    top_d follow the ToF prior through the fusion."""
    for cam_h in (0.6, 1.2):
        fr = make_frame(BLOCK_SCENE + [BEAM], cam_h=cam_h, floor_dropout=0.3, seed=15)
        fr.depth *= 0.93                                         # every return 7 % too close
        f = F.fit_floor(fr, prior_floor_d=0.01)                  # ToF: 1 cm error
        assert f.found and abs(f.d_stereo + 0.07 * cam_h) < 0.02, (cam_h, f.d_stereo)
        assert abs(f.d) < 0.03, (cam_h, f.d)
        _check_no_paint(F.obstacles(fr, f), fr, BLOCK_SCENE)
        t = F.landing_block(fr, f)
        assert t.found and abs(t.top_d + BLOCK['h1']) < 0.04, (cam_h, t.found, t.top_d)
        g = F.fit_floor(fr)                                      # no prior: d is the biased plane
        assert abs(g.d - g.d_stereo) < 1e-12 and g.d < -0.03


def test_keep_xy_geofence():
    """Depth outside the arena box (hall seen through the net) is ignored by all three."""
    out_wall = box(4.5, 0.0, 0.05, 6.0, 0.0, 2.5, grey=230, tex=3.0)    # beyond the 'net' at x=2.8
    scene = [out_wall, box(3.4, 0.4, 0.6, 0.5, 0.0, 0.35, grey=215, tex=10.0)] + PAINTED[:2]
    fr = make_frame(scene, cam_h=0.9, floor_dropout=0.3, seed=16)
    box_xy = (-1.0, 2.8, -3.0, 3.0)
    f_all, f_in = F.fit_floor(fr, 0.0), F.fit_floor(fr, 0.0, keep_xy=box_xy)
    assert f_all.found and f_in.found and f_in.n_inliers < f_all.n_inliers
    ob = F.obstacles(fr, f_in, keep_xy=box_xy)
    assert len(ob.points_ned) == 0 or ob.points_ned[:, 0].max() <= 2.8 + 1e-6
    assert len(F.obstacles(fr, f_in).points_ned) > 100           # the wall is there without the box
    assert F.landing_block(fr, f_all).found                     # block outside the box ...
    assert not F.landing_block(fr, f_in, keep_xy=box_xy).found  # ... is ignored with it


def test_floor_attitude_error_absorbed():
    """1.5 deg pitch + 1 deg roll error in the reported attitude: the fitted normal absorbs it,
    painted rocks still never become obstacles out to 4 m."""
    fr = make_frame(BASE_SCENE, cam_h=0.6, err=(math.radians(1.5), math.radians(-1.0)), floor_dropout=0.3, seed=9)
    f = F.fit_floor(fr, 0.0)
    assert f.found and abs(f.d) < 0.03, (f.d, f.reason)
    assert 1.0 < f.tilt_deg < 2.6, f.tilt_deg
    ob = F.obstacles(fr, f)
    _check_no_paint(ob, fr, BASE_SCENE)
    # a level-floor assumption would be off by ~0.1 m at 4 m: confirm the test is meaningful
    lvl = F.FloorFit(True, 0.0, np.array([0., 0., 1.]), 1.0, anchor_ned=np.array([0., 0., 0.]), sigma0=0.004, sigma1=0.0)
    ob_lvl = F.obstacles(fr, lvl)
    assert _floor_hits(ob_lvl, fr).sum() > 20, _floor_hits(ob_lvl, fr).sum()


# =========================================================================== obstacle tests
def _floor_hits(ob, fr):
    """Obstacle points whose pixel truly sees the floor (ground-truth hit id 0)."""
    v, u = ob.pixels_vu[:, 0], ob.pixels_vu[:, 1]
    return fr.hid[v, u] == 0


def _paint_hits(ob, fr, scene):
    """...of which inside a painted rock / light patch (grown 10 cm for depth noise)."""
    fl = _floor_hits(ob, fr)
    inside = np.zeros(len(fl), bool)
    for s in scene:
        if s['kind'] == 'paint':
            inside |= _in_ellipse(ob.points_ned, s, grow=0.10)
    return int((fl & inside).sum())


def _visible_px(fr, s, max_range=4.0):
    """Ground-truth stride-2 pixels of object s with valid depth within range."""
    sid = 1 + [id(x) for x in BASE_SCENE].index(id(s)) if any(s is x for x in BASE_SCENE) else None
    hid = fr.hid[::2, ::2]; z = fr.depth[::2, ::2]
    return int(((hid == sid) & np.isfinite(z) & (z < max_range)).sum())


def _check_no_paint(ob, fr, scene, max_floor_fp=2):
    assert _paint_hits(ob, fr, scene) == 0, _paint_hits(ob, fr, scene)
    assert _floor_hits(ob, fr).sum() <= max_floor_fp, _floor_hits(ob, fr).sum()


def test_obstacles_rocks_not_paint():
    for cam_h in (0.4, 0.8, 1.2):
        fr = make_frame(BASE_SCENE, cam_h=cam_h, floor_dropout=0.4, seed=11)
        f = F.fit_floor(fr)
        ob = F.obstacles(fr, f)
        assert ob.floor_used and len(ob.points_ned) == len(ob.height_above_floor) == len(ob.pixels_vu)
        assert (ob.height_above_floor > 0.05).all()
        _check_no_paint(ob, fr, BASE_SCENE)
        for s in ROCKS:
            if _visible_px(fr, s) < 30:              # below the field of view at this height
                continue
            m = _near_object(ob.points_ned, s, 0.06)
            assert m.sum() >= 10, (cam_h, s, m.sum())
            hmax = ob.height_above_floor[m].max()
            assert abs(hmax - s['h']) < 0.07, (cam_h, s['h'], hmax)
        assert _near_object(ob.points_ned, BEAM, 0.05).sum() >= 5, cam_h
        assert (fr.hid[ob.pixels_vu[:, 0], ob.pixels_vu[:, 1]] == 1 + BASE_SCENE.index(PANEL)).sum() >= 50, cam_h


def test_obstacles_reflections_dropped():
    scene = BASE_SCENE + [box(4.6, 0.0, 0.05, 4.0, 0.0, 2.5, grey=150)]
    fr = make_frame(scene, cam_h=0.7, floor_dropout=0.6, mirror_frac=0.35, seed=12)
    f = F.fit_floor(fr, 0.0)
    ob = F.obstacles(fr, f)
    assert ob.n_below_floor > 100, ob.n_below_floor
    assert (ob.height_above_floor > 0).all()
    _check_no_paint(ob, fr, scene)


def test_obstacles_flying_pixels():
    """Mixed pixels between a rock and the far floor/wall form a phantom 'shadow'; filtered."""
    scene = [cone(1.6, 0.0, 0.2, 0.4), cone(2.2, 0.7, 0.15, 0.3), box(3.6, 0.0, 0.05, 4.0, 0.0, 2.5, grey=150)]
    fr = make_frame(scene, cam_h=0.7, flying=True, noise_px=0.05, seed=13)
    f = F.fit_floor(fr, 0.0)

    def phantoms(ob):
        bad = np.ones(len(ob.points_ned), bool)
        for s in scene:
            bad &= ~_near_object(ob.points_ned, s, 0.10)
        return int(bad.sum())
    with_filter = phantoms(F.obstacles(fr, f))
    orig = F._flying_pixels
    try:
        F._flying_pixels = lambda z, v, *a, **k: np.zeros_like(v)
        without = phantoms(F.obstacles(fr, f))
    finally:
        F._flying_pixels = orig
    assert without > 20 and with_filter <= 0.1 * without, (with_filter, without)


def test_obstacles_floor_unknown_or_prior_only():
    fr = make_frame(BASE_SCENE, cam_h=0.6, seed=14)
    ob = F.obstacles(fr, F.FloorFit(False, math.nan, np.array([0., 0., 1.]), 0.0))
    assert not ob.floor_used and ob.points_ned.shape == (0, 3)
    # depth fit impossible -> prior-only plane (wider default noise model) still separates rocks/paint
    f = F.FloorFit(False, 0.01, np.array([0., 0., 1.]), 0.0, anchor_ned=np.array([0., 0., 0.01]), source='prior')
    ob = F.obstacles(fr, f)
    assert ob.floor_used
    _check_no_paint(ob, fr, BASE_SCENE)
    for s in ROCKS:                                   # default sigma: 4 sigma ~ 0.17 m at 3.4 m
        if s['h'] >= 0.25 or s['x'] <= 3.0:
            assert _near_object(ob.points_ned, s, 0.06).sum() >= 5, s


# =========================================================================== landing target tests
def _block_ok(t, blk, pos_tol=0.08):
    assert t.found, t.reason
    assert np.hypot(t.center_ned[0] - blk['x'], t.center_ned[1] - blk['y']) < pos_tol, t.center_ned
    assert abs(t.top_d + blk['h1']) < 0.03, t.top_d
    assert abs(t.height_above_floor - blk['h1']) < 0.04, t.height_above_floor


def test_landing_block_found():
    for cam_h, dist in ((0.9, 2.5), (0.6, 2.0), (1.2, 3.0), (0.9, 1.75)):
        blk = dict(BLOCK, x=dist)
        scene = [blk if s is BLOCK else s for s in BLOCK_SCENE]
        fr = make_frame(scene, cam_h=cam_h, floor_dropout=0.4, seed=21)
        f = F.fit_floor(fr)
        t = F.landing_block(fr, f)
        _block_ok(t, blk)
        assert abs(t.extent_xy_m[0] - 0.6) < 0.15 and abs(t.extent_xy_m[1] - 0.5) < 0.15, t.extent_xy_m
        assert t.confidence >= 0.5 and not t.truncated, (cam_h, dist, t.confidence)
        assert abs(t.range_m - math.hypot(dist - fr.cam_ned[0], 0.2 - fr.cam_ned[1])) < 0.1, t.range_m


def test_landing_lighting_top_only_bright():
    """Overhead light: top bright, sides darker than the floor threshold -> still found."""
    blk = dict(BLOCK, side=0.6)
    fr = make_frame([blk] + PAINTED[:2], cam_h=0.9, floor_dropout=0.3, seed=22)
    _block_ok(F.landing_block(fr, F.fit_floor(fr)), blk)


def test_landing_low_altitude_dark_sides():
    """Camera only 0.15 m above the top: the dark front face is ~85 % of the blob, only the
    thin top strip is bright (venue-replica finding, 2026-11-06) -> still found."""
    blk = dict(BLOCK, x=2.0, h1=0.25, side=0.5)
    fr = make_frame([blk] + PAINTED[:2], cam_h=0.40, floor_dropout=0.3, seed=37)
    _block_ok(F.landing_block(fr, F.fit_floor(fr, 0.0)), blk, pos_tol=0.10)


def test_landing_overexposed_top_holes():
    """Over-exposed white top: the saturated part has no stereo texture -> no depth; the pixels
    that keep depth are the darker texture. Bright holes on the top count as bright top."""
    blk = dict(BLOCK, x=2.0, side=0.5)
    fr = make_frame([blk] + PAINTED[:2], cam_h=0.9, floor_dropout=0.3, seed=38)
    fx, fy, cx, cy = OFFICIAL_K
    vv, uu = np.mgrid[0:fr.depth.shape[0], 0:fr.depth.shape[1]]
    ray = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones(uu.shape)], -1) @ np.asarray(fr.R_nc).T
    z_ned = fr.cam_ned[2] + ray[..., 2] * fr.depth
    top = (fr.hid == 1) & np.isfinite(fr.depth) & (z_ned < -(blk['h1'] - 0.03))
    blob = cv2.GaussianBlur(np.random.default_rng(0).standard_normal(top.shape).astype(np.float32), (0, 0), 4.0)
    sat = top & (blob < np.quantile(blob[top], 0.65))      # saturated patches are contiguous
    fr.image[sat] = 255; fr.depth[sat] = np.nan            # saturated -> no stereo match
    fr.image[top & ~sat] = 135                              # what keeps depth is darker than the gate
    _block_ok(F.landing_block(fr, F.fit_floor(fr, 0.0)), blk, pos_tol=0.10)


def test_landing_confidence_stable_under_exposure_and_bottom_edge():
    """Official-compat sim flight (oakimg_ideal_full, 2026-11-07): bright exposure with gain jitter
    (floor ~180-230 grey, white top clipped at 255) and the block's near edge right on the image
    bottom made the confidence flicker 0.36-0.53 around the 0.5 latch. It must stay >= 0.6."""
    confs = []
    for k, gain in enumerate((1.5, 1.6, 1.7, 1.8, 1.9, 1.75)):
        blk = dict(BLOCK, x=1.55 + 0.01 * (k % 3))         # near edge at image rows ~395-400
        fr = make_frame([blk] + PAINTED[:2], cam_h=0.88, floor_dropout=0.1, seed=40 + k)
        fr.image = np.clip(fr.image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        f = F.fit_floor(fr, 0.0)
        t = F.landing_block(fr, f, prior_center_ned=(1.5, 0.2, 0.0), bright_mode='soft')
        assert t.found, (gain, t.reason)
        assert np.hypot(t.center_ned[0] - blk['x'], t.center_ned[1] - blk['y']) < 0.10, (gain, t.center_ned)
        confs.append(t.confidence)
    assert min(confs) >= 0.6, confs


def _top_mask(fr, blk, sid):
    fx, fy, cx, cy = OFFICIAL_K
    vv, uu = np.mgrid[0:fr.depth.shape[0], 0:fr.depth.shape[1]]
    ray = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones(uu.shape)], -1) @ np.asarray(fr.R_nc).T
    z_ned = fr.cam_ned[2] + ray[..., 2] * fr.depth
    return (fr.hid == sid) & np.isfinite(fr.depth) & (z_ned < -(blk['h1'] - 0.03))


def test_landing_hole_filled_top_with_single_rim():
    """Stereo (SGM) in the official-compat sim (stereo_s1-s3, 2026-11-07): hovering ~1.2 m away,
    the block's near edge and faces are below the image, and the white top is a depth hole except
    a thin rim along one long edge. Only the hole-fill path can find it."""
    for k, (cam_h, x, y) in enumerate(((0.75, 1.30, 0.20), (0.72, 1.22, 0.05), (0.80, 1.35, 0.30))):
        blk = dict(BLOCK, x=x, y=y, side=0.5)
        fr = make_frame([blk] + PAINTED[:2], cam_h=cam_h, floor_dropout=0.3, seed=50 + k)
        top = _top_mask(fr, blk, 1)
        assert top[-1].any()                              # top runs off the image bottom
        keep = np.zeros_like(top)
        for r in np.nonzero(top.any(1))[0]:               # keep ~5 px along the image-left long edge
            c = np.nonzero(top[r])[0]; keep[r, c[:5]] = True
        fr.depth[top & ~keep] = np.nan
        f = F.fit_floor(fr, 0.0)
        assert not F.landing_block(fr, f, prior_center_ned=(x, y, 0.0), bright_mode='soft', hole_fill=False).found
        t = F.landing_block(fr, f, prior_center_ned=(x, y, 0.0), bright_mode='soft')
        assert t.found and t.factors.get('kind') == 'hole_fill', (cam_h, t.reason, t.factors.get('kind'))
        assert np.hypot(t.center_ned[0] - x, t.center_ned[1] - y) < 0.10, (cam_h, t.center_ned)
        assert abs(t.top_d + blk['h1']) < 0.05 and t.confidence >= 0.35, (t.top_d, t.confidence)


def test_landing_hole_fill_rejects_bar_over_featureless_floor():
    """A frame bar at a fixed height over a bright featureless floor (a depth hole on both sides
    of the bar) must not become a hole-filled block top, even without a position prior."""
    bar = box(2.2, 0.0, 0.05, 1.6, 0.0, 0.22, grey=200, tex=12.0)
    rock = cone(2.6, -0.6, 0.15, 0.3)
    for k, cam_h in enumerate((0.6, 0.8)):
        fr = make_frame([bar, rock] + PAINTED[:1], cam_h=cam_h, floor_dropout=0.97, seed=60 + k)
        fr.image = np.clip(fr.image.astype(np.float32) * 1.8, 0, 255).astype(np.uint8)
        f = F.fit_floor(fr, 0.0)
        for mode in ('soft', 'gate'):
            t = F.landing_block(fr, f, bright_mode=mode, hole_fill=True)
            assert not t.found, (cam_h, mode, t.factors.get('kind'), t.center_ned)


def test_landing_rejects_painted_light_patches():
    scene = PAINTED + [paint(2.0, 0.0, 0.6, 0.45, 0.0, 235), paint(3.0, -0.8, 0.3, 0.3, 0.0, 240),
                       paint(1.5, 0.8, 0.5, 0.2, 0.5, 230)]
    for cam_h in (0.5, 1.0):
        fr = make_frame(scene, cam_h=cam_h, floor_dropout=0.2, seed=23)
        f = F.fit_floor(fr)
        t = F.landing_block(fr, f)
        assert not t.found, (cam_h, t)
        # even with a 1.5 deg attitude error
        fr = make_frame(scene, cam_h=cam_h, err=(math.radians(1.5), 0.0), seed=24)
        assert not F.landing_block(fr, F.fit_floor(fr)).found


def test_landing_rejects_bright_walls_and_panels():
    scene = [box(4.2, 0.0, 0.05, 6.0, 0.0, 3.0, grey=235, tex=3.0),            # far bright wall
             box(2.4, -0.7, 0.9, 0.03, 0.0, 0.6, yaw=0.2, grey=230, tex=3.0),   # short thin bright board
             PANEL, box(2.0, 1.0, 1.0, 0.03, 0.05, 1.9, yaw=1.2, grey=225)]     # panel seen edge-on-ish
    for cam_h in (0.6, 1.0):
        fr = make_frame(scene + PAINTED[:2], cam_h=cam_h, floor_dropout=0.3, seed=25)
        t = F.landing_block(fr, F.fit_floor(fr))
        assert not t.found, (cam_h, t)


def test_landing_rejects_dark_or_coloured_box():
    dark = box(2.4, 0.0, 0.6, 0.5, 0.0, 0.35, grey=75, tex=10.0)
    fr = make_frame([dark] + PAINTED[:2], cam_h=0.9, seed=26)
    assert not F.landing_block(fr, F.fit_floor(fr)).found
    # BGR input: equally bright but saturated orange box -> rejected; white -> accepted
    def colourise(orange_ids):
        def f(grey, hid):
            img = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR).astype(np.float32)
            fl = (hid == 0) | np.isin(hid, orange_ids)
            img[fl] = img[fl] * np.array([0.35, 0.75, 1.6])
            return np.clip(img, 0, 255).astype(np.uint8)
        return f
    blk = BLOCK
    fr = make_frame([blk] + PAINTED[:2], cam_h=0.9, seed=27, bgr=colourise([1]))
    assert not F.landing_block(fr, F.fit_floor(fr)).found
    fr = make_frame([blk] + PAINTED[:2], cam_h=0.9, seed=27, bgr=colourise([]))
    _block_ok(F.landing_block(fr, F.fit_floor(fr)), blk)


def test_landing_soft_brightness_mode():
    """Mono sensor where the stone is barely brighter than the orange floor (photo 7 red
    channel: block 166 vs floor 188): 'gate' rejects, 'soft' + position prior finds it,
    and 'soft' still rejects painted patches and walls (geometry)."""
    grey_blk = dict(BLOCK, grey=130, tex=6.0)
    fr = make_frame([grey_blk] + PAINTED[:2], cam_h=0.9, floor_dropout=0.3, seed=35)
    f = F.fit_floor(fr)
    assert not F.landing_block(fr, f).found
    t = F.landing_block(fr, f, bright_mode='soft', prior_center_ned=(2.4, 0.3, 0.0), prior_radius_m=0.6)
    _block_ok(t, grey_blk)
    assert t.confidence < 0.6, t.confidence
    scene = PAINTED + [paint(2.0, 0.0, 0.6, 0.45, 0.0, 235), PANEL, box(4.2, 0.0, 0.05, 6.0, 0.0, 3.0, grey=235)]
    fr = make_frame(scene, cam_h=0.9, floor_dropout=0.3, seed=36)
    assert not F.landing_block(fr, F.fit_floor(fr), bright_mode='soft').found


def test_landing_too_far():
    for dist in (5.5, 7.0):
        blk = dict(BLOCK, x=dist)
        fr = make_frame([blk] + PAINTED[:2], cam_h=0.9, seed=28)
        t = F.landing_block(fr, F.fit_floor(fr))
        assert not t.found, (dist, t)
    # confidence falls with range inside the working range
    confs = []
    for dist in (2.0, 3.0, 3.8):
        blk = dict(BLOCK, x=dist, L=0.8, W=0.8)
        fr = make_frame([blk] + PAINTED[:2], cam_h=1.0, seed=29)
        confs.append(F.landing_block(fr, F.fit_floor(fr)).confidence)
    assert confs[0] >= confs[1] >= confs[2], confs


def test_landing_half_visible():
    """Block half outside the left image edge: flagged truncated, lower confidence, centre on the block."""
    full = dict(BLOCK, x=2.0, y=0.0)
    fr = make_frame([full] + PAINTED[:2], cam_h=0.9, seed=30)
    c_full = F.landing_block(fr, F.fit_floor(fr)).confidence
    half = dict(BLOCK, x=2.0, y=-1.52)         # HFOV half-angle 37 deg -> edge at y = -1.52
    fr = make_frame([half] + PAINTED[:2], cam_h=0.9, seed=30)
    t = F.landing_block(fr, F.fit_floor(fr))
    if t.found:
        assert t.truncated and t.confidence < c_full, (t.truncated, t.confidence, c_full)
        assert _near_object(t.center_ned[None], half, 0.0)[0], t.center_ned
    # a sliver (only ~0.1 m visible) must not be reported
    sliver = dict(BLOCK, x=2.0, y=-1.78)
    fr = make_frame([sliver] + PAINTED[:2], cam_h=0.9, seed=31)
    assert not F.landing_block(fr, F.fit_floor(fr)).found


def test_landing_top_not_visible():
    blk = dict(BLOCK, h1=0.45)
    fr = make_frame([blk] + PAINTED[:2], cam_h=0.35, seed=32)
    t = F.landing_block(fr, F.fit_floor(fr))
    assert not t.found and t.reason in ('top_not_visible', 'no_flat_top', 'too_few_top_points'), t.reason


def test_landing_prior_gate_and_no_floor():
    fr = make_frame(BLOCK_SCENE, cam_h=0.9, seed=33)
    f = F.fit_floor(fr)
    assert F.landing_block(fr, f, prior_center_ned=(2.4, 0.3, 0.0), prior_radius_m=0.5).found
    t = F.landing_block(fr, f, prior_center_ned=(2.4, 2.0, 0.0), prior_radius_m=0.5)
    assert not t.found and t.reason == 'outside_prior'
    t = F.landing_block(fr, F.FloorFit(False, math.nan, np.array([0., 0., 1.]), 0.0))
    assert not t.found and t.reason == 'no_floor'


def test_landing_with_attitude_error_and_prior_floor():
    fr = make_frame(BLOCK_SCENE, cam_h=0.9, err=(math.radians(-1.5), math.radians(1.0)), floor_dropout=0.5, seed=34)
    _block_ok(F.landing_block(fr, F.fit_floor(fr, 0.0)), BLOCK, pos_tol=0.10)


# =========================================================================== flat landing pad
PAD_LUMA = dict(floor=98, body=132, dark=92, white=165)      # photo 18, visible-like mono response
PAD_RED = dict(floor=159, body=90, dark=103, white=150)      # photo 18 red channel: red/NIR-heavy worst case


def _cinder(x, y, yaw=0.3):
    return box(x, y, 0.39, 0.19, 0.0, 0.19, yaw=yaw, grey=170, tex=12.0)


def _pad_scene(pol, x, y, yaw, rng):
    pd = pad(x, y, yaw=yaw, body=pol['body'], dark=pol['dark'], white=pol['white'])
    clutter = [paint(x + rng.uniform(-0.6, 0.6), y + rng.choice([-1, 1]) * rng.uniform(0.45, 0.7), 0.25, 0.18, rng.uniform(0, 3), 55),
               tape(x, y, 2.4, 1.6, 0.3), _cinder(x + 0.7, y - 0.5)]
    return [floor_grey(pol['floor'])] + clutter + [pd], pd


def test_pad_found_both_polarities():
    """Blue pad on the orange floor from the observe pose, with a painted rock, a tape rectangle
    and a cinder block around it. Body lighter than the floor (visible-like) or darker (red/NIR-
    like); landing_target() defaults to the pad."""
    rng = np.random.default_rng(5)
    for k, cam_h in enumerate((0.6, 0.8, 1.0, 1.2)):
        for pol in (PAD_LUMA, PAD_RED):
            x = 2.12 * cam_h + 0.65 + 0.116; y = float(rng.uniform(-0.3, 0.3))
            scene, pd = _pad_scene(pol, x, y, float(rng.uniform(0, math.pi)), rng)
            fr = make_frame(scene, cam_h=cam_h, floor_dropout=0.3, seed=70 + k)
            f = F.fit_floor(fr, 0.0)
            for pri in (None, (x + 0.25, y - 0.2, 0.0)):
                t = F.landing_target(fr, f, prior_center_ned=pri, bright_mode='soft')
                assert t.found and t.kind == 'pad', (cam_h, pol['floor'], pri, t.reason, t.factors)
                assert np.hypot(t.center_ned[0] - x, t.center_ned[1] - y) <= 0.05, (cam_h, pol['floor'], t.center_ned)
                assert abs(t.top_d - 0.0) < 0.03 and t.height_above_floor == 0.0 and abs(t.extent_m - 0.6) < 1e-9


def test_pad_rejects_tape_paint_blocks():
    """No pad: black tape square / rectangle / crossing, large dark painted rocks, cinder blocks and
    rock cones, a bright painted blob - on both floor responses, with and without a prior."""
    scenes = [[tape(2.3, 0.0, 0.6, 0.6, 0.4)], [tape(2.2, -0.3, 2.0, 0.6, 0.2), tape(2.6, 0.6, 1.5, 0.05, 1.3)],
              [paint(2.2, 0.0, 0.32, 0.28, 0.3, 55), paint(2.8, -0.7, 0.4, 0.3, 1.0, 60)],
              [_cinder(2.1, 0.1), _cinder(2.6, -0.5, 1.2), cone(2.4, 0.6, 0.15, 0.3)],
              [paint(2.3, 0.1, 0.3, 0.3, 0.0, 230)]]
    for i, items in enumerate(scenes):
        for pol in (PAD_LUMA, PAD_RED):
            for cam_h in (0.6, 1.0):
                fr = make_frame([floor_grey(pol['floor'])] + items, cam_h=cam_h, floor_dropout=0.3, seed=80 + i)
                f = F.fit_floor(fr, 0.0)
                for pri in (None, (2.3, 0.0, 0.0)):
                    t = F.landing_target(fr, f, prior_center_ned=pri)
                    assert not t.found, (i, pol['floor'], cam_h, pri, t.center_ned, t.factors)


def test_pad_out_of_view_and_no_floor():
    """Too close for the level camera (below the image), outside the image, or no floor plane."""
    pol = PAD_LUMA
    fr = make_frame([floor_grey(pol['floor']), pad(1.3, 0.0, yaw=0.3)], cam_h=0.9, seed=90)     # needs >= ~2.3 m
    f = F.fit_floor(fr, 0.0)
    assert not F.landing_target(fr, f).found
    assert not F.landing_target(fr, f, prior_center_ned=(1.3, 0.0, 0.0)).found
    fr = make_frame([floor_grey(pol['floor']), pad(2.6, 3.5, yaw=0.3)], cam_h=0.9, seed=91)     # far to the side
    assert not F.landing_target(fr, F.fit_floor(fr, 0.0)).found
    assert F.landing_target(fr, None).reason == 'no_floor'


def test_pad_occluder_without_depth():
    """Stereo leaves a textureless board face empty (depth only on a 3 px rim). Its interior must
    still be masked as an occluder (bracketed by nearer pixels); floor holes must not - including
    a floor hole column between a cinder block below and a far wall (above the horizon) above."""
    pol = PAD_LUMA
    x = 2.12 * 0.8 + 0.65 + 0.116
    scene = [floor_grey(pol['floor']), pad(x, 0.0, yaw=0.35, body=pol['body'], dark=pol['dark'], white=pol['white']),
             box(1.9, 0.5, 0.04, 0.3, 0.0, 0.55, grey=200, tex=0.0),          # board facing the camera
             _cinder(1.5, -0.55, 0.0), box(6.5, 0.0, 0.2, 8.0, 0.0, 2.5, grey=120, tex=10.0)]
    fr = make_frame(scene, cam_h=0.8, floor_dropout=0.3, seed=96)
    hid, dep = fr.hid, fr.depth
    board = (hid == 3)
    face = cv2.erode(board.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    dep[face] = np.nan
    blk = np.argwhere(hid == 4)                                  # empty floor between block top and wall
    c0, c1 = int(np.percentile(blk[:, 1], 25)), int(np.percentile(blk[:, 1], 75))
    top = int(blk[:, 0].min())
    gap = np.zeros_like(board); gap[:top, c0:c1] = True; gap &= (hid == 0)
    dep[gap] = np.nan
    f = F.fit_floor(fr, 0.0)
    occ = F._occlusion_image(np.asarray(dep, np.float32), tuple(float(k) for k in fr.K), np.asarray(fr.R_nc, float),
                             np.asarray(fr.cam_ned, float), np.asarray(f.normal_ned, float), np.asarray(f.anchor_ned, float))
    assert occ[face].mean() > 0.8, occ[face].mean()        # the rows at its foot are barely nearer than the floor
    fl_hole = (hid == 0) & ~np.isfinite(dep)
    assert gap.sum() > 200 and occ[gap].mean() < 0.03, (gap.sum(), occ[gap].mean())
    assert occ[fl_hole].mean() < 0.03, occ[fl_hole].mean()
    for pri in (None, (x - 0.2, 0.15, 0.0)):
        t = F.landing_target(fr, f, prior_center_ned=pri)
        assert t.found and np.hypot(t.center_ned[0] - x, t.center_ned[1]) <= 0.05, (pri, t.reason, t.factors, t.center_ned)


def test_pad_partly_hidden_by_legs_and_rail():
    """Seen through a window frame (replica v4, north of the window: 84-92 % visible): two thin
    legs and a low foot rail in front hide part of the pad. Still found, with `visible` < 1 and
    a lower confidence than the clear view - partial visibility is a confidence, not a miss."""
    rng = np.random.default_rng(11)
    for k, (cam_h, pol) in enumerate(((0.45, PAD_LUMA), (0.7, PAD_LUMA), (0.7, PAD_RED))):
        x = 2.12 * cam_h + 0.65 + 0.116; y = 0.05; yaw = float(rng.uniform(0, math.pi / 2))
        pd = pad(x, y, yaw=yaw, body=pol['body'], dark=pol['dark'], white=pol['white'])
        xf = x - 0.75
        frame_ = [box(xf, y - 0.12, 0.06, 0.06, 0.0, 1.2, grey=150, tex=4.0),          # legs
                  box(xf, y + 0.28, 0.06, 0.06, 0.0, 1.2, grey=150, tex=4.0),
                  box(xf, y, 0.05, 1.4, 0.0, 0.07, grey=60, tex=4.0)]                   # foot rail
        clear = make_frame([floor_grey(pol['floor']), pd], cam_h=cam_h, floor_dropout=0.3, seed=120 + k)
        hidden = make_frame([floor_grey(pol['floor']), pd] + frame_, cam_h=cam_h, floor_dropout=0.3, seed=120 + k)
        nopad = make_frame([floor_grey(pol['floor'])], cam_h=cam_h, floor_dropout=0.3, seed=120 + k)
        pad_px = np.abs(clear.image.astype(int) - nopad.image.astype(int)) > 6       # pad pixels in the clear view
        share = 1.0 - (hidden.hid[pad_px] == 0).mean()                           # share hidden by the frame
        res = []
        for fr in (clear, hidden):
            t = F.landing_target(fr, F.fit_floor(fr, 0.0), prior_center_ned=(x + 0.2, y - 0.15, 0.0))
            assert t.found and np.hypot(t.center_ned[0] - x, t.center_ned[1] - y) <= 0.05, (cam_h, pol['floor'], t.reason, t.factors)
            res.append(t)
        assert 0.12 <= share <= 0.30, share
        assert res[1].factors['visible'] < 0.95 and res[1].confidence < res[0].confidence, (share, res[0].factors, res[1].factors)


def test_landing_target_dispatch():
    """target='pad' (default) / 'block' / 'any'; results carry kind."""
    fr = make_frame(BLOCK_SCENE, cam_h=0.9, floor_dropout=0.4, seed=21)
    f = F.fit_floor(fr)
    assert not F.landing_target(fr, f, bright_mode='soft').found                   # no pad here
    tb = F.landing_target(fr, f, target='block')
    assert tb.found and tb.kind == 'block'
    ta = F.landing_target(fr, f, target='any')
    assert ta.found and ta.kind == 'block'
    scene, pd = _pad_scene(PAD_LUMA, 2.4, 0.1, 0.4, np.random.default_rng(3))
    fr = make_frame(scene, cam_h=0.8, floor_dropout=0.3, seed=22)
    tp = F.landing_target(fr, F.fit_floor(fr, 0.0), target='any')
    assert tp.found and tp.kind == 'pad'


# =========================================================================== timing
def _timing(n=30):
    fr = make_frame(BLOCK_SCENE + ROCKS[:2] + [BEAM], cam_h=0.9, floor_dropout=0.4, seed=40)
    F.landing_block(fr, F.fit_floor(fr))      # warm caches
    ts = {'fit_floor': [], 'obstacles': [], 'landing_target': [], 'total': []}
    for _ in range(n):
        t0 = time.perf_counter(); f = F.fit_floor(fr, 0.0)
        t1 = time.perf_counter(); F.obstacles(fr, f)
        t2 = time.perf_counter(); F.landing_target(fr, f, prior_center_ned=(2.5, 0.2, 0.0))
        t3 = time.perf_counter()
        for k, v in zip(ts, (t1 - t0, t2 - t1, t3 - t2, t3 - t0)):
            ts[k].append(v * 1e3)
    return {k: (float(np.median(v)), float(np.max(v))) for k, v in ts.items()}


def test_timing_under_50ms():
    t = _timing()
    assert t['total'][0] < 50.0, t


# =========================================================================== runner
def main():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    fails = 0
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            print(f'PASS {name} ({time.perf_counter() - t0:.1f}s)')
        except Exception as e:  # noqa: BLE001
            fails += 1
            import traceback
            print(f'FAIL {name}: {type(e).__name__}: {e}')
            traceback.print_exc(limit=3)
    t = _timing()
    print('timing ms (median, max) at 640x400:', {k: (round(a, 1), round(b, 1)) for k, (a, b) in t.items()})
    print(f'{len(tests) - fails}/{len(tests)} passed')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
