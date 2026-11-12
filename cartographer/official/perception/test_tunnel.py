#!/usr/bin/env python3
"""Tests for perception/tunnel.py.

    .venv/bin/python cartographer/official/perception/test_tunnel.py            # unit tests
    .venv/bin/python cartographer/official/perception/test_tunnel.py --report   # + montage / metrics

Synthetic frames: horizontal OAK-D left camera (OFFICIAL_K, 640x400) looking at a row of coaxial
brass hoops (vertical circles standing on wire feet, bottom `gap` above the floor), black mesh lining
(no stereo depth), rock-print fabric over the top (with depth), cage net wall behind, a bright hall.
Distractors: printed circle on a banner panel, chandelier rings, random bright arcs, wooden frames, rocks.
The venue photo 2 is used for a qualitative 2-D check only (phone intrinsics are guessed).
"""
import json
import math
import os
import sys
import time
import unittest
from dataclasses import dataclass

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(OFFICIAL))
if OFFICIAL not in sys.path:
    sys.path.insert(0, OFFICIAL)

from common.geometry import (Frame, OFFICIAL_K, OFFICIAL_SIZE, OFFICIAL_R_BASE_CAM,  # noqa: E402
                             OFFICIAL_T_BASE_CAM, camera_pose_ned)
from perception import tunnel as tun  # noqa: E402

OUT_DIR = os.path.join(REPO, 'outputs', 'claude-official')
PHOTO2 = os.environ.get('TUNNEL_PHOTO2', '/private/tmp/claude-501/-Users-chuyao-Documents-Codex-2026-10-31-'
                        'cartographer-dronehacks/2198afae-c72b-4034-8353-abe0a35bdca0/images/2.webp')
RESULTS = []


# ----------------------------------------------------------------------------- scene
@dataclass
class Scene:
    name: str = 'scene'
    body: tuple = (0.0, 0.0, -0.70)       # vehicle position NED (floor at z = 0)
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    tunnel: bool = True
    entry_xy: tuple = (2.0, 0.0)          # entrance hoop centre (x, y)
    axis_deg: float = 0.0                 # tunnel axis yaw (entrance -> exit)
    radius: float = 0.50
    gap: float = 0.08                     # hoop bottom above the floor
    n_rings: int = 4
    spacing: float = 0.45
    distractors: bool = False
    floor_z: float = 0.0
    noise: float = 3.0
    brass: float = 215.0                  # grey level of the entrance hoop wire
    blur: float = 0.6                     # optics / motion blur sigma (px)
    depth: bool = True                    # False: no stereo depth at all
    ring_depth: bool = False              # True: the hoop wire returns depth (Gazebo / SGBM on brass)
    seed: int = 0


def quat_from_euler(yaw, pitch=0.0, roll=0.0):
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    return (cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy)


def _wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def _rock(s, t):
    return np.clip(108 + 32*np.sin(13*s + 5.1*t)*np.sin(7*s - 9*t) + 18*np.sin(31*s + 3*t + 1.0)
                   + 10*np.sin(57*s - 23*t), 40, 200)


def render(sc):
    """-> (Frame, truth dict)."""
    W, H = OFFICIAL_SIZE
    fx, fy, cx, cy = OFFICIAL_K
    rng = np.random.default_rng(sc.seed)
    q = quat_from_euler(math.radians(sc.yaw_deg), math.radians(sc.pitch_deg), math.radians(sc.roll_deg))
    R_nc, cam = camera_pose_ned(q, sc.body, OFFICIAL_R_BASE_CAM, OFFICIAL_T_BASE_CAM)
    R_cn = R_nc.T
    u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    dn = np.stack([(u - cx)/fx, (v - cy)/fy, np.ones_like(u)], -1) @ R_nc.T   # NED ray / optical depth
    I = np.empty((H, W))
    Z = np.full((H, W), 30.0)
    D = np.full((H, W), np.nan)
    fz = sc.floor_z
    ex, ey = sc.entry_xy
    Xw = (ex if sc.tunnel else sc.body[0] + 2.0) + 3.0
    # hall beyond the cage
    hor = np.linalg.norm(dn[..., :2], axis=-1)
    el = -dn[..., 2]/hor
    az = np.arctan2(dn[..., 1], dn[..., 0])
    I[:] = np.where(el > 0.02, 115 + 50*np.clip(el/0.4, 0, 1) + 10*np.sin(az*23 + 1.0), 55)
    I += 30*np.exp(-((el - 0.12)/0.02)**2)
    # floor
    with np.errstate(divide='ignore', invalid='ignore'):
        zf = (fz - cam[2])/dn[..., 2]
    m = (dn[..., 2] > 1e-6) & (zf > 0) & (zf < Z)
    Px, Py = cam[0] + zf*dn[..., 0], cam[1] + zf*dn[..., 1]
    tex = 112 + 10*np.sin(1.7*Px + 0.3)*np.sin(2.1*Py)
    blot = np.sin(3.3*Px + 1)*np.sin(2.9*Py + 2) + 0.5*np.sin(7.1*Px - 5.3*Py) > 1.05
    tex = np.where(blot, 72, tex)
    tex = np.where((np.abs(Py) < 3.05) & (Px < Xw), tex, 50)
    I[m] = tex[m]
    Z[m] = zf[m]
    D[m] = np.where(zf < 6, zf, np.nan)[m]
    # cage net walls (dark cords, no depth) and posts
    for ai, val in ((0, Xw), (1, 3.05), (1, -3.05)):
        with np.errstate(divide='ignore', invalid='ignore'):
            zw = (val - cam[ai])/dn[..., ai]
        a_ = cam[1 - ai] + zw*dn[..., 1 - ai] if ai == 0 else cam[0] + zw*dn[..., 0]
        b_ = cam[2] + zw*dn[..., 2]
        mm = (zw > 0) & (zw < Z) & (b_ < fz) & (b_ > fz - 3.0)
        a2 = a_ + 0.012*np.sin(b_*37)
        b2 = b_ + 0.012*np.sin(a_*41)
        cell, wd = 0.05, 0.004
        d = np.minimum(np.abs(((a2/cell + 0.5) % 1) - 0.5), np.abs(((b2/cell + 0.5) % 1) - 0.5))*cell
        fp = np.maximum(zw, 0.05)/fx
        c = np.clip((0.5*(wd + fp) - d)/fp, 0, 1)*np.minimum(1, wd/fp)
        post = (np.abs(((a_/1.5 + 0.5) % 1) - 0.5)*1.5 < 0.02) | (np.abs(b_ - (fz - 1.5)) < 0.02)
        c = np.where(post, 1.0, c)
        I = np.where(mm, I*(1 - 0.9*c), I)
    segs = []                                          # (P0, P1, value, width_m)
    truth = dict(R_nc=R_nc, cam=cam)
    if sc.tunnel:
        ax = math.radians(sc.axis_deg)
        a = np.array([math.cos(ax), math.sin(ax), 0.0])
        eh = np.array([-math.sin(ax), math.cos(ax), 0.0])
        ez = np.array([0.0, 0.0, 1.0])
        r = sc.radius
        E = np.array([ex, ey, fz - sc.gap - r])
        L = (sc.n_rings - 1)*sc.spacing
        truth.update(entry=E, axis=a[:2], radius=r)
        w0 = cam - E
        wa = w0 @ a
        wp = w0 - wa*a
        da = dn @ a
        dp = dn - da[..., None]*a
        A = np.sum(dp*dp, -1)
        B = 2*(dp @ wp)
        for name, R, s0, s1 in (('mesh', r, 0.0, L), ('fabric', r + 0.02, -0.03, L + 0.05)):
            disc = B*B - 4*A*(wp @ wp - R*R)
            ok = (disc > 0) & (A > 1e-9)
            sq = np.sqrt(np.maximum(disc, 0))
            for zz in ((-B + sq)/(2*A + 1e-12), (-B - sq)/(2*A + 1e-12)):   # far root, then near root
                s = wa + zz*da
                rho = wp + zz[..., None]*dp
                th = np.arctan2(rho @ ez, rho @ eh)
                top = np.abs(_wrap(th + np.pi/2)) < math.radians(65)
                pz = cam[2] + zz*dn[..., 2]
                val = ok & (zz > 0.05) & (s >= s0) & (s <= s1) & (pz < fz) & (zz < Z)
                if name == 'mesh':
                    val &= ~top
                    g = (np.abs(((s/0.012 + 0.5) % 1) - 0.5) < 0.15) | (np.abs(((th*R/0.012 + 0.5) % 1) - 0.5) < 0.15)
                    tx = 30 + 7*g
                    I[val] = tx[val]
                    D[val] = np.nan
                else:
                    val &= top
                    tx = _rock(s, th)
                    I[val] = tx[val]
                    D[val] = zz[val]
                Z[val] = zz[val]
        P0 = E - 0.03*a                               # front flap of the fabric over the entrance top
        with np.errstate(divide='ignore', invalid='ignore'):
            zfl = ((P0 - cam) @ a)/da
        rel = cam + zfl[..., None]*dn - P0
        ph, pz = rel @ eh, rel @ ez
        rad = np.hypot(ph, pz)
        th = np.arctan2(pz, ph)
        mm = (zfl > 0.05) & (zfl < Z) & (rad > r - 0.07) & (rad < r + 0.03) & (np.abs(_wrap(th + np.pi/2)) < math.radians(60))
        I[mm] = _rock(rad*3, th)[mm]
        Z[mm] = zfl[mm]
        D[mm] = zfl[mm]
        th = np.linspace(0, 2*np.pi, 721)
        for k in range(sc.n_rings):
            Ck = E + k*sc.spacing*a
            Rk = r - 0.004
            P = Ck + Rk*(np.cos(th)[:, None]*eh + np.sin(th)[:, None]*ez)
            val = sc.brass if k == 0 else sc.brass - 25 - 10*k
            segs += [(P[i], P[i + 1], val, 0.006) for i in range(720)]
            for sg in (-1, 1):                         # wire feet
                tl = np.pi/2 + sg*0.45
                Pl = Ck + Rk*(math.cos(tl)*eh + math.sin(tl)*ez)
                Pf = Pl.copy()
                Pf[2] = fz - 0.004
                segs.append((Pl, Pf, val, 0.005))
                cs = [Pf + 0.10*a + 0.04*eh, Pf - 0.10*a + 0.04*eh, Pf - 0.10*a - 0.04*eh, Pf + 0.10*a - 0.04*eh]
                segs += [(cs[i], cs[(i + 1) % 4], val, 0.005) for i in range(4)]
    if sc.distractors:
        # banner panel with a bright printed circle (depth valid everywhere on the panel)
        Xp, Yp = (ex + 0.3) if sc.tunnel else sc.body[0] + 2.3, (ey - 1.15) if sc.tunnel else sc.body[1] - 0.3
        with np.errstate(divide='ignore', invalid='ignore'):
            zp = (Xp - cam[0])/dn[..., 0]
        py, pz = cam[1] + zp*dn[..., 1], cam[2] + zp*dn[..., 2]
        mm = (zp > 0.05) & (zp < Z) & (np.abs(py - Yp) < 0.5) & (pz < fz - 0.2) & (pz > fz - 1.5)
        rr = np.hypot(py - Yp, pz - (fz - 0.95))
        fp = np.maximum(zp, 0.05)/fx
        cov = np.clip((0.012 + 0.5*fp - np.abs(rr - 0.22))/fp, 0, 1)
        dark = np.clip((0.006 + 0.5*fp - np.abs(np.hypot(py - Yp - 0.25, pz - (fz - 0.45)) - 0.12))/fp, 0, 1)
        tx = (150 + 8*np.sin(py*40))*(1 - 0.7*dark)
        tx = tx*(1 - cov) + 235*cov
        I[mm] = tx[mm]
        Z[mm] = zp[mm]
        D[mm] = zp[mm]
        # chandeliers (horizontal bright rings above the cage)
        tc = np.linspace(0, 2*np.pi, 181)
        for (xc, yc) in ((Xw + 1.5, 0.8), (Xw + 3.0, -1.6)):
            Cc = np.array([xc, yc, fz - 3.3])
            P = Cc + 0.45*np.stack([np.cos(tc), np.sin(tc), 0*tc], -1)
            segs += [(P[i], P[i + 1], 235, 0.02) for i in range(180)]
        # wooden A-frame (bright thick bars) on the right
        b0 = np.array([ex - 0.4, ey + 1.1, fz]) if sc.tunnel else np.array([sc.body[0] + 2, 1.0, fz])
        for dy in (0.0, 0.6):
            segs.append((b0 + [0, dy, 0], b0 + [0, dy, -1.2], 185, 0.035))
        segs.append((b0 + [0, 0, -1.1], b0 + [0, 0.6, -1.1], 185, 0.035))
        segs.append((b0 + [0, 0, -0.01], b0 + [-0.8, 0, -0.01], 185, 0.035))
        # rocks (cones with depth)
        for (xr, yr) in ((ex - 0.6, ey - 0.55), (ex - 1.0, ey + 0.7)) if sc.tunnel else ((2.0, 0.3),):
            base = np.array([xr, yr, fz])
            pts = np.array([base + [0, -0.15, 0], base + [0, 0.15, 0], base + [0, 0, -0.3]])
            uv, zr = _proj(pts, R_cn, cam)
            if (zr > 0.1).all():
                lay = np.zeros((H, W), np.uint8)
                cv2.fillConvexPoly(lay, np.round(uv*16).astype(np.int32), 255, cv2.LINE_AA, 4)
                mm = (lay > 128) & (Z > zr.mean())
                I[mm] = 128 + 20*np.sin(u[mm]*0.5)
                Z[mm] = zr.mean()
                D[mm] = zr.mean()
    # rasterise wire-like primitives with a z-test against the surfaces
    prims = []
    for P0, P1, val, wm in segs:
        uv, zz = _proj(np.array([P0, P1]), R_cn, cam)
        if (zz < 0.1).any():
            continue
        zm = float(zz.mean())
        wpx = fx*wm/zm
        prims.append((zm, uv, val, min(1.0, wpx), 1 if wpx < 1.8 else int(round(wpx))))
    prims.sort(key=lambda t: -t[0])
    LA = np.zeros((H, W), np.uint8)                    # coverage (anti-aliased)
    LV = np.zeros((H, W), np.float32)                  # brightness
    LZ = np.full((H, W), np.inf, np.float32)
    for zm, uv, val, cov, th_px in prims:
        p0 = tuple(int(x) for x in np.round(uv[0]*16))
        p1 = tuple(int(x) for x in np.round(uv[1]*16))
        if max(abs(p0[0]), abs(p0[1]), abs(p1[0]), abs(p1[1])) > 16*4000:
            continue
        cv2.line(LA, p0, p1, int(255*(0.35 + 0.65*cov)), th_px, cv2.LINE_AA, 4)
        cv2.line(LV, p0, p1, float(val), th_px + 1, cv2.LINE_8, 4)
        cv2.line(LZ, p0, p1, float(zm), th_px + 1, cv2.LINE_8, 4)
    mm = (LA > 0) & (LZ <= Z + 0.03)
    al = np.where(mm, LA/255.0, 0.0)
    I = I*(1 - al) + LV*al
    if sc.ring_depth:
        wm = mm & (LA > 128) & np.isfinite(LZ)
        D[wm] = LZ[wm]
    if sc.distractors:                                  # random bright arcs in the hall background
        lay = np.zeros((H, W), np.uint8)
        for _ in range(4):
            c = (int(rng.uniform(40, W - 40)), int(rng.uniform(10, 140)))
            axes = (int(rng.uniform(20, 90)), int(rng.uniform(15, 70)))
            a0 = rng.uniform(0, 360)
            cv2.ellipse(lay, c, axes, rng.uniform(0, 180), a0, a0 + rng.uniform(60, 200), 215, 1, cv2.LINE_AA)
        mm = (lay > 0) & (Z >= 29)
        I = np.where(mm, np.maximum(I, lay), I)
    I = cv2.GaussianBlur(I.astype(np.float32), (0, 0), sc.blur)
    I = I + rng.normal(0, sc.noise, I.shape)
    img = np.clip(np.round(I), 0, 255).astype(np.uint8)
    D = D*(1 + rng.normal(0, 0.01, D.shape))
    D[rng.random(D.shape) < 0.05] = np.nan
    if not sc.depth:
        D[:] = np.nan
    fr = Frame(image=img, depth=D.astype(np.float32), K=OFFICIAL_K, R_nc=R_nc, cam_ned=cam,
               body_ned=np.asarray(sc.body, float), heading=math.radians(sc.yaw_deg))
    return fr, truth


def _proj(P, R_cn, cam):
    fx, fy, cx, cy = OFFICIAL_K
    pc = (np.atleast_2d(P) - cam) @ R_cn.T
    z = pc[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.stack([fx*pc[:, 0]/z + cx, fy*pc[:, 1]/z + cy], -1), z


def errors(est, truth):
    ec = float(np.linalg.norm(est.entry_ned - truth['entry']))
    ea = math.degrees(math.acos(float(np.clip(est.axis_ned @ truth['axis'], -1, 1))))
    er = abs(est.radius_m - truth['radius'])/truth['radius']
    return ec, ea, er


def run(sc, prior=None, **kw):
    fr, truth = render(sc)
    t0 = time.perf_counter()
    est = tun.detect(fr, prior, **kw)
    ms = 1e3*(time.perf_counter() - t0)
    return est, truth, fr, ms


# ----------------------------------------------------------------------------- unit tests
class TestSynthetic(unittest.TestCase):
    def check(self, sc, tol_c, tol_a, tol_r, prior=None, **kw):
        est, truth, fr, ms = run(sc, prior, **kw)
        rec = dict(name=sc.name, found=bool(est.found), ms=round(ms, 1), conf=round(est.confidence, 2),
                   n_rings=len(est.rings), scale=est.scale_source, axis_src=est.axis_source)
        if est.found:
            ec, ea, er = errors(est, truth)
            rec.update(center_err_m=round(ec, 3), axis_err_deg=round(ea, 1), radius_err_pct=round(100*er, 1),
                       dist_m=round(float(np.linalg.norm(truth['entry'] - fr.cam_ned)), 2))
        RESULTS.append(rec)
        self.assertTrue(est.found, '%s: not found (%s)' % (sc.name, est.debug.get('reject')))
        self.assertLess(ec, tol_c, '%s centre err %.3f m' % (sc.name, ec))
        self.assertLess(ea, tol_a, '%s axis err %.1f deg' % (sc.name, ea))
        self.assertLess(er, tol_r, '%s radius err %.1f %%' % (sc.name, 100*er))
        return est

    def test_frontal_1m(self):
        self.check(Scene('frontal_1m', entry_xy=(1.1, 0.0)), 0.08, 8, 0.08)

    def test_frontal_2m(self):
        self.check(Scene('frontal_2m', entry_xy=(2.0, 0.0)), 0.06, 6, 0.06)

    def test_frontal_3m(self):
        self.check(Scene('frontal_3m', entry_xy=(3.0, 0.0)), 0.10, 8, 0.08)

    def test_oblique_plus30(self):
        self.check(Scene('oblique_+30', entry_xy=(2.0, 0.0), axis_deg=30), 0.08, 6, 0.08)

    def test_oblique_minus30(self):
        self.check(Scene('oblique_-30', entry_xy=(2.0, 0.3), axis_deg=-30, seed=3), 0.08, 6, 0.08)

    def test_partial_out_of_frame(self):
        # hoop centre 0.95 m to the right at 2 m: about a third of the hoop is outside the image
        self.check(Scene('partial_right_edge', entry_xy=(2.0, 0.95), seed=5), 0.10, 10, 0.10)

    def test_camera_yawed_partial(self):
        # vehicle yawed 25 deg away: hoop cut by the left image border
        self.check(Scene('yawed_25_partial', entry_xy=(1.8, 0.0), yaw_deg=25, seed=6), 0.10, 10, 0.10)

    def test_distractors(self):
        self.check(Scene('distractors_2m', entry_xy=(2.0, 0.0), distractors=True, seed=11), 0.06, 6, 0.06)

    def test_distractors_oblique(self):
        self.check(Scene('distractors_oblique', entry_xy=(2.4, -0.2), axis_deg=20, distractors=True, seed=12),
                   0.08, 8, 0.08)

    def test_pitch_roll(self):
        self.check(Scene('pitch5_roll3', entry_xy=(2.0, 0.1), pitch_deg=-5, roll_deg=3, seed=13), 0.08, 8, 0.08)

    def test_entrance_is_nearest_hoop(self):
        est = self.check(Scene('entrance_id_2.5m', entry_xy=(2.5, 0.0), seed=14), 0.10, 8, 0.08)
        if len(est.rings) >= 2:
            d = [float(np.linalg.norm(r.center_ned - est.rings[0].center_ned)) for r in est.rings[1:]]
            self.assertGreater(min(d), 0.25)

    def test_prior_near(self):
        sc = Scene('prior_near', entry_xy=(2.2, 0.2), distractors=True, seed=15)
        self.check(sc, 0.08, 8, 0.08, prior=np.array([2.6, -0.1, -0.6]))

    def test_prior_far_is_gated(self):
        sc = Scene('prior_far', entry_xy=(2.2, 0.2), seed=16)
        est, truth, fr, ms = run(sc, np.array([2.2, 3.0, -0.6]))
        RESULTS.append(dict(name=sc.name, found=bool(est.found), ms=round(ms, 1), expect='not found'))
        self.assertFalse(est.found)

    def test_no_tunnel(self):
        for seed in (21, 22, 23):
            sc = Scene('no_tunnel_%d' % seed, tunnel=False, distractors=True, seed=seed)
            est, truth, fr, ms = run(sc)
            RESULTS.append(dict(name=sc.name, found=bool(est.found), ms=round(ms, 1), expect='not found'))
            self.assertFalse(est.found, 'false positive in %s: %s' % (sc.name, est.entry_ned))

    def test_single_hoop_with_axis_prior(self):
        sc = Scene('single_hoop_+30', entry_xy=(2.0, 0.0), axis_deg=30, n_rings=1, seed=17)
        self.check(sc, 0.10, 10, 0.08, prior_axis_ned=np.array([math.cos(0.5), math.sin(0.5)]))

    def test_radius_prior_fixes_wrong_gap(self):
        # true gap 0.15 m but the detector assumes 0.08 m; a measured radius restores the scale
        sc = Scene('wrong_gap+radius_prior', entry_xy=(2.0, 0.0), gap=0.15, seed=18)
        self.check(sc, 0.06, 6, 0.04, params=tun.TunnelParams(radius_m=0.50))

    def test_close_range_back_inference(self):
        # 0.63 m from the entrance: hoop 1 is bigger than the image, only hoops 2.. are seen.
        # Without a prior the module reports hoop 2 (documented); with a prior + spacing it steps back.
        sc = Scene('close_0.6m_backinfer', entry_xy=(0.75, 0.0), seed=24)
        fr, truth = render(sc)
        est0 = tun.detect(fr)
        self.assertTrue(est0.found)
        self.assertAlmostEqual(float((est0.entry_ned - truth['entry'])[:2] @ truth['axis']), sc.spacing, delta=0.08)
        # the back-inference needs the prior's along-axis error < spacing / 2 (here 0.225 m)
        prior = truth['entry'] + np.array([0.12, -0.25, 0.05])
        p = tun.TunnelParams(radius_m=0.5, hoop_spacing_m=sc.spacing, n_hoops=sc.n_rings)
        est = self.check(sc, 0.08, 6, 0.06, prior=prior, params=p)
        self.assertEqual(est.debug.get('entry_inferred'), 1)

    def test_prior_error_within_tolerance_keeps_confidence(self):
        sc = Scene('prior_0.45m_conf', entry_xy=(2.2, 0.0), seed=25)
        fr, truth = render(sc)
        c0 = tun.detect(fr).confidence
        c1 = tun.detect(fr, truth['entry'] + np.array([0.0, 0.45, 0.0])).confidence
        RESULTS.append(dict(name=sc.name, conf_no_prior=round(c0, 2), conf_prior_0p45=round(c1, 2)))
        self.assertGreater(c1, 0.9*c0)

    def test_align_ok_flag(self):
        # >= 2 hoops -> fit for final alignment; a lone hoop seen 60 deg side-on -> coarse only
        est = tun.detect(render(Scene('align_ok_frontal', entry_xy=(2.0, 0.0), seed=26))[0])
        self.assertTrue(est.found and est.align_ok)
        sc = Scene('align_ok_single_60', entry_xy=(2.0, 0.0), axis_deg=60, n_rings=1, seed=27)
        est = tun.detect(render(sc)[0], prior_axis_ned=np.array([math.cos(math.radians(60)), math.sin(math.radians(60))]))
        RESULTS.append(dict(name=sc.name, found=bool(est.found), align_ok=bool(est.align_ok),
                            view=round(est.rings[0].view_angle_deg, 1) if est.rings else None))
        self.assertFalse(est.align_ok)

    def test_floor_bias_wire_depth(self):
        # ubuntu-compat v4B regression: VIO height drift of ~7 cm makes the floor input wrong; depth on the
        # wire (+ fabric) must outvote it (before: confidence ~0.53, 0.10-0.15 m error)
        sc = Scene('floor_bias_7cm_wire_depth', entry_xy=(1.5, 0.0), ring_depth=True, seed=28)
        fr, truth = render(sc)
        est = self.check(sc, 0.05, 5, 0.05, prior=truth['entry'] + np.array([0.1, -0.1, 0.0]),
                         floor_z_ned=sc.floor_z - 0.07, prior_axis_ned=truth['axis'])
        self.assertGreaterEqual(est.confidence, 0.8)
        self.assertIn('wire', est.scale_source)

    def test_depth_floor_estimate(self):
        # no floor_z passed: floor estimated from depth (floor at z = +0.05 here)
        sc = Scene('floor_from_depth', entry_xy=(2.0, 0.0), floor_z=0.05, body=(0, 0, -0.65), seed=19)
        est = self.check(sc, 0.06, 6, 0.06)
        self.assertEqual(est.debug.get('floor_src'), 'depth')

    def test_timing(self):
        scenes = [Scene('t%d' % i, entry_xy=(1.5 + 0.5*(i % 4), 0.1*(i % 3)), axis_deg=(i % 3 - 1)*20,
                        distractors=bool(i % 2), seed=30 + i) for i in range(8)]
        frames = [render(s)[0] for s in scenes]
        tun.detect(frames[0])                           # warm-up
        ts = []
        for fr in frames:
            t0 = time.perf_counter()
            tun.detect(fr)
            ts.append(1e3*(time.perf_counter() - t0))
        RESULTS.append(dict(name='timing', median_ms=round(float(np.median(ts)), 1),
                            max_ms=round(float(np.max(ts)), 1)))
        self.assertLess(float(np.median(ts)), 80.0)


class TestPhoto2(unittest.TestCase):
    """Qualitative: does the 2-D stage lock onto the brass hoops of the real tunnel (venue photo 2)?"""

    def test_photo2(self):
        if not os.path.exists(PHOTO2):
            self.skipTest('photo 2 not available')
        bgr = cv2.imread(PHOTO2)
        self.assertIsNotNone(bgr)
        bgr = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        est, cands = photo_detect(gray)
        os.makedirs(OUT_DIR, exist_ok=True)
        vis = tun.draw(bgr, est)
        for i, e in enumerate(cands):
            c = tuple(int(x) for x in e.center_px)
            cv2.putText(vis, str(i), c, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, '2-D check only: phone K / pose guessed, 3-D numbers meaningless', (8, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, 'green = chosen hoop, grey = other ellipse candidates', (8, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT_DIR, 'tunnel-photo2.png'), vis)
        RESULTS.append(dict(name='photo2', n_candidates=len(cands), found=bool(est.found),
                            candidates=[dict(c=[round(float(x)) for x in e.center_px],
                                             axes=[round(float(x)) for x in e.axes_px],
                                             support=e.support, cover=round(e.cover, 2)) for e in cands]))
        # hoop traced by hand in the 640x480 image: left side x~215, right side x~447,
        # bottom y~377, top hidden under the fabric -> centre ~(331, 245), half-width ~116
        hit = [e for e in cands if math.hypot(e.center_px[0] - 331, e.center_px[1] - 245) < 30
               and 85 < min(e.axes_px) < 145 and 100 < max(e.axes_px) < 190]
        self.assertTrue(hit, 'traced hoop not among the 2-D candidates')


def photo_detect(gray):
    """Run the detector on the phone photo with guessed intrinsics / pose (horizontal camera ~1.2 m)."""
    H, W = gray.shape
    f = 0.5*W/math.tan(math.radians(34.7))            # ~26 mm-equivalent phone lens, HFOV ~69 deg
    K = (f, f, W/2, H/2)
    R_nc, cam = camera_pose_ned(quat_from_euler(0.0), (0, 0, -1.1), OFFICIAL_R_BASE_CAM, OFFICIAL_T_BASE_CAM)
    fr = Frame(image=gray, depth=np.full(gray.shape, np.nan, np.float32), K=K, R_nc=R_nc, cam_ned=cam)
    p = tun.TunnelParams(radius_range_m=(0.2, 1.5), range_m=(0.4, 8.0))
    est = tun.detect(fr, floor_z_ned=0.0, params=p)
    return est, est.candidates


# ----------------------------------------------------------------------------- report helpers

def decompose(est, truth):
    """Entrance error split into along-axis, cross-axis (horizontal) and vertical parts."""
    d = est.entry_ned - truth['entry']
    a = np.r_[truth['axis'], 0.0]
    c = np.array([-a[1], a[0], 0.0])
    return float(d @ a), float(d @ c), float(d[2])


def sweep():
    """Accuracy / robustness sweep for the report (written to tunnel-metrics.json)."""
    rows = []

    def one(group, sc, **kw):
        fr, truth = render(sc)
        t0 = time.perf_counter()
        est = tun.detect(fr, **kw)
        ms = 1e3*(time.perf_counter() - t0)
        r = dict(group=group, name=sc.name, found=bool(est.found), ms=round(ms, 1), conf=round(est.confidence, 2))
        if est.found and 'entry' in truth:
            ec, ea, er = errors(est, truth)
            al, cr, vz = decompose(est, truth)
            r.update(center=round(ec, 3), along=round(al, 3), cross=round(cr, 3), vert=round(vz, 3),
                     axis=round(ea, 2), radius=round(100*er, 1), n_rings=len(est.rings))
        rows.append(r)

    for d in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0):
        for ax in (0, 30):
            for sd in (1, 2):
                one('distance', Scene('d%.1f_ax%d_s%d' % (d, ax, sd), entry_xy=(d + 0.1, 0.05*sd),
                                      axis_deg=ax*(1 if sd == 1 else -1), seed=100 + sd + int(10*d)))
    for ax in (0, 15, 30, 45, 60, 70):
        for sd in (1, 2):
            one('angle', Scene('ax%d_s%d' % (ax, sd), entry_xy=(2.1, 0.0), axis_deg=ax*(1 if sd == 1 else -1),
                               seed=200 + sd + ax))
    for lat in (0.6, 0.9, 1.1, 1.3):
        one('partial', Scene('lat%.1f' % lat, entry_xy=(2.1, lat), seed=300 + int(10*lat)))
    for sd in range(30):
        one('no_tunnel', Scene('nt%d' % sd, tunnel=False, distractors=True, seed=400 + sd,
                               yaw_deg=float(np.random.default_rng(sd).uniform(-40, 40))))
    base = dict(entry_xy=(2.6, 0.1), axis_deg=20, distractors=True)
    conds = [('nominal', {}, {}), ('noise8', dict(noise=8.0), {}), ('blur1.5', dict(blur=1.5), {}),
             ('brass170', dict(brass=170.0), {}), ('brass150', dict(brass=150.0), {}),
             ('no_depth', dict(depth=False), {}),
             ('gap_true0.15', dict(gap=0.15), {}), ('gap_true0.01', dict(gap=0.01), {}),
             ('gap_true0.15+radius', dict(gap=0.15), dict(params=tun.TunnelParams(radius_m=0.5))),
             ('gap_true0.15+radius+no_depth', dict(gap=0.15, depth=False), dict(params=tun.TunnelParams(radius_m=0.5))),
             ('prior_0.4m_off', {}, dict(prior_entry_ned=np.array([2.9, -0.2, -0.6])))]
    for name, sk, kw in conds:
        for sd in (1, 2, 3):
            one('cond:' + name, Scene('%s_s%d' % (name, sd), seed=500 + sd, **dict(base, **sk)), **kw)
    # summary per group
    summ = {}
    for g in sorted(set(r['group'] for r in rows)):
        rs = [r for r in rows if r['group'] == g]
        f = [r for r in rs if r['found'] and 'center' in r]
        s = dict(n=len(rs), found=sum(r['found'] for r in rs), median_ms=round(float(np.median([r['ms'] for r in rs])), 1))
        if f:
            s.update(center_med=round(float(np.median([r['center'] for r in f])), 3),
                     center_max=round(float(np.max([r['center'] for r in f])), 3),
                     cross_max=round(float(np.max(np.abs([r['cross'] for r in f]))), 3),
                     vert_max=round(float(np.max(np.abs([r['vert'] for r in f]))), 3),
                     axis_med=round(float(np.median([r['axis'] for r in f])), 2),
                     axis_max=round(float(np.max([r['axis'] for r in f])), 2),
                     radius_max=round(float(np.max([r['radius'] for r in f])), 1))
        summ[g] = s
    return rows, summ

def montage():
    cases = [Scene('frontal 1.1 m', entry_xy=(1.1, 0.0)), Scene('frontal 2 m', entry_xy=(2.0, 0.0)),
             Scene('frontal 3 m', entry_xy=(3.0, 0.0)), Scene('oblique +30', entry_xy=(2.0, 0.0), axis_deg=30),
             Scene('partial (right edge)', entry_xy=(2.0, 0.95), seed=5),
             Scene('distractors', entry_xy=(2.0, 0.0), distractors=True, seed=11)]
    tiles = []
    for sc in cases:
        est, truth, fr, ms = run(sc)
        vis = tun.draw(fr.image, est)
        if 'entry' in truth:
            uv, z = _proj(truth['entry'][None], truth['R_nc'].T, truth['cam'])
            cv2.drawMarker(vis, tuple(int(x) for x in uv[0]), (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
        if est.found:
            uv, z = _proj(est.entry_ned[None], truth['R_nc'].T, truth['cam'])
            cv2.drawMarker(vis, tuple(int(x) for x in uv[0]), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 12, 1)
            ec, ea, er = errors(est, truth)
            txt = '%s: c %.0fmm ax %.1fdeg r %.1f%% %.0fms' % (sc.name, 1e3*ec, ea, 100*er, ms)
        else:
            txt = '%s: not found' % sc.name
        cv2.rectangle(vis, (0, 380), (640, 400), (0, 0, 0), -1)
        cv2.putText(vis, txt, (6, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(vis)
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    img = np.vstack(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUT_DIR, 'tunnel-synthetic.png'), cv2.resize(img, None, fx=0.75, fy=0.75,
                                                                          interpolation=cv2.INTER_AREA))


if __name__ == '__main__':
    report = '--report' in sys.argv
    argv = [a for a in sys.argv if a != '--report']
    prog = unittest.main(argv=argv, exit=False, verbosity=2)
    for r in RESULTS:
        print(json.dumps(r))
    if report:
        montage()
        rows, summ = sweep()
        for g, v in summ.items():
            print(g, json.dumps(v))
        with open(os.path.join(OUT_DIR, 'tunnel-metrics.json'), 'w') as fh:
            json.dump(dict(python=sys.version.split()[0], numpy=np.__version__, opencv=cv2.__version__,
                           unit_tests=RESULTS, sweep_summary=summ, sweep=rows), fh, indent=1)
    sys.exit(0 if prog.result.wasSuccessful() else 1)
