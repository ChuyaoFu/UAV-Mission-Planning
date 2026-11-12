"""Tests for perception/tunnel_servo.py on synthetic frames (renderer from test_tunnel.py).

Run:  python3 -m unittest perception.test_tunnel_servo     (from cartographer/official)
"""
import math
import os
import sys
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL = os.path.dirname(HERE)
if OFFICIAL not in sys.path:
    sys.path.insert(0, OFFICIAL)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from test_tunnel import Scene, render  # noqa: E402
from perception.tunnel_servo import (TunnelServo, TunnelModel, ServoParams, servo_velocity,  # noqa: E402
                                     _solve, _inv)

AXIS_DEG = 20.0
R = 0.76
SP = 0.64
N = 3
E = np.array([2.0, 0.0, -R])          # hoops on the floor (gap 0), floor at z = 0


def frames_axes(axis_deg=AXIS_DEG):
    a = math.radians(axis_deg)
    ax = np.array([math.cos(a), math.sin(a), 0.0])
    L = np.array([math.sin(a), -math.cos(a), 0.0])
    U = np.array([0.0, 0.0, -1.0])
    return ax, L, U


def scene_at(s, l, u, yaw_err_deg, seed, **kw):
    ax, L, U = frames_axes()
    body = E + s * ax + l * L + u * U
    return Scene('servo', body=tuple(body), yaw_deg=AXIS_DEG + yaw_err_deg, entry_xy=(E[0], E[1]), axis_deg=AXIS_DEG,
                 radius=R, gap=0.0, n_rings=N, spacing=SP, seed=seed, **kw)


def model(latch_err=(0.0, 0.0, 0.0), psi_err_deg=0.0):
    ax, _, _ = frames_axes(AXIS_DEG + psi_err_deg)
    return TunnelModel(entry_ned=E + np.asarray(latch_err, float), axis_ned=ax[:2], radius_m=R,
                       hoop_s=tuple(k * SP for k in range(N)))


def fly(path, mdl=None, params=None, odom=True, dt=0.1, depth=True):
    """path: list of (s, l, u, yaw_err_deg). Returns [(truth, state)]."""
    sv = TunnelServo(mdl or model(), params or ServoParams())
    ax, L, U = frames_axes()
    out = []
    for k, (s, l, u, y) in enumerate(path):
        fr, _ = render(scene_at(s, l, u, y, seed=100 + k, depth=depth))
        fr.stamp = k * dt
        body = E + s * ax + l * L + u * U
        st = sv.update(fr, odom_ned=body if odom else None)
        out.append(((s, l, u, math.radians(y)), st))
    return out


class TestServo(unittest.TestCase):

    def test_straight_traverse_tracks_truth(self):
        """Approach from 1.3 m out to the exit at 0.25 m/s, with lateral/vertical/yaw offsets that change slowly."""
        path = []
        for k, s in enumerate(np.arange(-1.3, 0.25, 0.05)):
            path.append((s, 0.08 * math.cos(0.15 * k), -0.05 * math.cos(0.1 * k + 1.0), 3.0 * math.sin(0.1 * k)))
        out = fly(path)
        val = [(t, st) for t, st in out if st.valid]
        self.assertGreaterEqual(len(val), 0.85 * len(out))
        el = [abs(st.offset_l - t[1]) for t, st in val]
        eu = [abs(st.offset_u - t[2]) for t, st in val]
        ey = [abs(math.degrees(st.yaw_err - t[3])) for t, st in val]
        es = [abs(st.progress_m - t[0]) for t, st in val]
        self.assertLess(np.percentile(el, 90), 0.03, el)
        self.assertLess(np.percentile(eu, 90), 0.03, eu)
        self.assertLess(np.percentile(ey, 90), 1.5, ey)
        self.assertLess(np.percentile(es, 90), 0.06, es)

    def test_sign_conventions(self):
        """offset_l > 0 when the vehicle is LEFT of the centre line, offset_u > 0 when high, yaw_err > 0 when the nose
        points right of the axis; correction_ned points from the vehicle to the centre line."""
        path = [(-1.0 + 0.02 * k, 0.12, 0.10, 6.0) for k in range(6)]
        out = fly(path)
        st = out[-1][1]
        self.assertTrue(st.valid)
        self.assertAlmostEqual(st.offset_l, 0.12, delta=0.03)
        self.assertAlmostEqual(st.offset_u, 0.10, delta=0.03)
        self.assertAlmostEqual(math.degrees(st.yaw_err), 6.0, delta=1.5)
        ax, L, U = frames_axes()
        c = st.correction_ned
        self.assertLess(c @ L, -0.08)                      # move right
        self.assertGreater(c[2], 0.06)                     # move down (NED +z)
        self.assertAlmostEqual(float(st.axis_ned @ ax), 1.0, delta=0.002)
        v, yr, adv = servo_velocity(st)
        self.assertLess(yr, 0.0)                           # yaw left to cancel a nose-right error
        self.assertLess(v @ L, 0.0)

    def test_latch_error_converges(self):
        """Mission latch off by 8 cm / -6 cm / 4 cm and 3 deg: the servo converges within a few frames."""
        path = [(-1.2 + 0.03 * k, 0.03, 0.0, 0.0) for k in range(12)]
        out = fly(path, mdl=model(latch_err=(0.06, -0.08, 0.04), psi_err_deg=3.0))
        for t, st in out[4:]:
            self.assertTrue(st.valid)
            self.assertLess(abs(st.offset_l - t[1]), 0.035)
            self.assertLess(abs(st.offset_u - t[2]), 0.03)
            self.assertLess(abs(math.degrees(st.yaw_err - t[3])), 1.5)

    def test_no_tunnel_never_valid(self):
        sv = TunnelServo(model())
        ax, L, U = frames_axes()
        for k in range(6):
            body = E + (-1.2 + 0.03 * k) * ax
            sc = Scene('empty', body=tuple(body), yaw_deg=AXIS_DEG, tunnel=False, distractors=True, seed=k)
            fr, _ = render(sc)
            fr.stamp = 0.1 * k
            st = sv.update(fr, odom_ned=body)
            self.assertFalse(st.valid)
            self.assertFalse(st.ok_to_advance)

    def test_blind_end_coasts_on_vio_only(self):
        """Close to the exit the exit hoop leaves the FOV: no vision. With VIO deltas the servo coasts (bounded);
        without VIO it must stop advancing after max_advance_gap_s."""
        path = [(s, 0.02, 0.0, 0.0) for s in np.arange(0.2, 1.75, 0.05)]
        out = fly(path, odom=True)
        modes = [st.mode for _, st in out]
        self.assertIn('vision', modes)
        self.assertIn('coast', modes)
        last_vision = max(i for i, m in enumerate(modes) if m == 'vision')
        for t, st in out[last_vision + 1:]:
            self.assertFalse(st.valid)
            if st.dr_dist_m <= ServoParams().coast_max_m:
                self.assertTrue(st.ok_to_advance or st.ok_to_coast)      # coast starts after max_advance_gap_s
                self.assertLess(abs(st.offset_l - t[1]), 0.05)          # dead reckoning on (perfect) VIO deltas
        out2 = fly(path, odom=False)
        lv = max(i for i, (_, st) in enumerate(out2) if st.valid)
        for _, st in out2[lv + 1:]:
            if st.since_valid_s > ServoParams().max_advance_gap_s:
                self.assertFalse(st.ok_to_advance)
                self.assertFalse(st.ok_to_coast)
                v, _, adv = servo_velocity(st)
                self.assertFalse(adv)

    def test_vio_jump_is_not_followed(self):
        sv = TunnelServo(model())
        ax, L, U = frames_axes()
        sts = []
        for k in range(8):
            s = -1.0 + 0.025 * k
            body = E + s * ax
            fr, _ = render(scene_at(s, 0.0, 0.0, 0.0, seed=k))
            fr.stamp = 0.1 * k
            odom = body + (np.array([0.0, 1.5, 0.0]) if k >= 5 else 0.0)      # VIO jumps 1.5 m sideways
            sts.append(sv.update(fr, odom_ned=odom))
        for st in sts[5:]:
            self.assertTrue(st.valid)
            self.assertLess(abs(st.offset_l), 0.04)

    def test_from_estimate_anchors_on_exit(self):
        """Detector named the 2nd hoop the entry: with on-site hoop positions the model is anchored on the
        farthest ring (exit) instead."""
        ax, _, _ = frames_axes()

        class Ring:
            def __init__(self, c):
                self.center_ned = c

        class Est:
            entry_ned = E + SP * ax
            axis_ned = ax[:2]
            radius_m = 0.78
            rings = [Ring(E + SP * ax), Ring(E + 2 * SP * ax)]
        m = TunnelModel.from_estimate(Est(), hoop_s=(0.0, SP, 2 * SP), radius_m=R)
        self.assertLess(np.linalg.norm(m.entry_ned - E), 1e-9)
        self.assertEqual(m.hoop_s, (0.0, SP, 2 * SP))
        m2 = TunnelModel.from_estimate(Est(), n_hoops=3)
        self.assertEqual(len(m2.hoop_s), 3)
        self.assertAlmostEqual(m2.hoop_s[1], SP, places=6)

    def test_small_solver(self):
        rng = np.random.default_rng(3)
        A = rng.normal(size=(7, 7))
        A = A @ A.T + 7 * np.eye(7)
        b = rng.normal(size=7)
        self.assertLess(np.abs(_solve(A, b) - np.linalg.solve(A, b)).max(), 1e-9)
        self.assertLess(np.abs(_inv(A) @ A - np.eye(7)).max(), 1e-9)

    def test_timing(self):
        frs = []
        for k in range(10):
            fr, _ = render(scene_at(-0.8 + 0.03 * k, 0.05, 0.0, 1.0, seed=k))
            fr.stamp = 0.1 * k
            frs.append(fr)
        sv = TunnelServo(model())
        sv.update(frs[0])
        ms = []
        for fr in frs[1:]:
            t0 = time.perf_counter()
            sv.update(fr)
            ms.append((time.perf_counter() - t0) * 1e3)
        print('\n[tunnel_servo] per-frame ms: median %.1f max %.1f' % (np.median(ms), max(ms)))
        self.assertLess(np.median(ms), 40.0)


if __name__ == '__main__':
    unittest.main()
