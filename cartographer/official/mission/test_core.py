#!/usr/bin/env python3
"""Closed-loop tests for mission/core.py.

Kinematic fake PX4 (setpoint first-order tracking, arming / mode / land / land-detector
logic), a box ObstacleMap, a toy course with ground truth that differs from the prior by
~0.2-0.3 m, and fake perception outputs with the INTERFACES.md fields.

Run:  .venv/bin/python cartographer/official/mission/test_core.py   (also works with pytest)
"""
import math
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # cartographer/official

from mission.core import (CoursePrior, Descent, Follower, Latch, Mission, wrap, Params, Perception, StartFrame, Vehicle,  # noqa: E402
                          NAV_AUTO_LAND, NAV_OFFBOARD, NAV_POSCTL, PHASES, TERMINAL_PHASES, stage_start_pose)

DT = 0.05
HALF_SPAN, HALF_HEIGHT = 0.20, 0.08


# ------------------------------------------------- perception output stand-ins
@dataclass
class TunnelEstimate:
    found: bool
    entry_ned: Any
    axis_ned: Any
    radius_m: float
    confidence: float


@dataclass
class WindowEstimate:
    found: bool
    center_ned: Any
    normal_ned: Any
    width_m: float
    height_m: float
    clearance_ok: bool
    confidence: float
    reason: str = ''


@dataclass
class LandingTarget:
    found: bool
    center_ned: Any
    top_d: float
    extent_m: float
    confidence: float
    reason: str = ''
    yaw_ned: float = 0.0                  # (perception/floor.py fields pad_approach reads)
    range_m: float = math.nan


@dataclass
class FloorFit:
    found: bool
    d: float
    normal_ned: Any
    inlier_frac: float


# ------------------------------------------------------------- toy course
# Toy layout; stage SIZES follow the Gazebo venue replica (outputs/venue-replica/REPORT.md section 1):
# 3 brass hoops r=0.55 m, 0.6 m apart, centre 0.62 m high; window hole 0.75 x 0.60 m centred 1.21 m high in a
# 1.5 m board from 0.35 to 1.70 m; wall board 0.35-1.70 m; white rock 0.60 x 0.45 x 0.25 m.
FENCE = dict(forward_min=-0.6, forward_max=5.8, left_min=-5.0, left_max=0.6, up_max=2.7)
TRUE = dict(
    tunnel_entry=(3.4, 0.0, 0.62), tunnel_axis_deg=0.0, tunnel_length=1.2, tunnel_radius=0.55, tunnel_ring_gap=0.6,
    window_center=(3.6, -2.8, 1.21), window_through_deg=180.0, window_w=0.75, window_h=0.60,
    board_half_w=0.75, board_bottom=0.35, board_top=1.70,
    rock_center=(1.6, -3.8), rock_half=(0.30, 0.30), rock_top=0.0,   # landing target: flat blue H pad on the floor
    wall_f=(0.3, 2.6), wall_l=(-2.1, -1.9), wall_bottom=0.35, wall_top=1.7)


def prior_dict(**over):
    d = dict(
        frame='start', tolerance_m=0.5, cruise_up_m=1.0, fence=dict(FENCE),
        corridor=[dict(forward=1.2, left=0.0, up=1.0)],
        tunnel=dict(entry=dict(forward=3.2, left=0.2, up=0.6), axis_deg=5.0, length_m=1.2, tol_m=0.5),
        window=dict(center=dict(forward=3.8, left=-2.6, up=1.1), through_deg=175.0, tol_m=0.5),
        landing_target=dict(center=dict(forward=1.9, left=-3.6), top_up=0.0, tol_m=0.5),
        wall=dict(cross_from=dict(forward=1.2, left=-2.7), cross_to=dict(forward=1.2, left=-1.2), top_up=1.7,
                  tol_m=0.3))
    d.update(over)
    return d


FLIGHT_PHASES_T = ('corridor', 'tunnel_align', 'tunnel_traverse', 'chicane', 'gate_pass', 'window_align',
                   'window_traverse', 'landing_route', 'land_target', 'climb_over_wall', 'return_home')


def _seg_d(q, a, b):
    """(distance, closest point) from the 2-D point q to the segment a-b."""
    ab = b - a
    u = float(np.clip((q - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
    c = a + u * ab
    return float(np.linalg.norm(q - c)), c


def course_boxes(T=TRUE, extra=()):
    """Obstacle boxes in the START frame: (lo(f,l,u), hi(f,l,u), name)."""
    b = []
    f0, l0, u0 = T['tunnel_entry']
    rc = T['tunnel_radius'] + 0.01
    n_rings = int(round(T['tunnel_length'] / T['tunnel_ring_gap'])) + 1
    for i in range(n_rings):
        ring_f = f0 + i * T['tunnel_ring_gap']
        for k in range(24):
            a = 2 * math.pi * k / 24
            c = np.array([ring_f, l0 + rc * math.cos(a), u0 + rc * math.sin(a)])
            b.append((c - [0.02, 0.08, 0.08], c + [0.02, 0.08, 0.08], 'hoop'))
    b.append(((f0 - 0.1, l0 - 0.6, u0 + rc + 0.02), (f0 + T['tunnel_length'] + 0.1, l0 + 0.6, u0 + rc + 0.06), 'fabric'))
    fc, lc, uc = T['window_center']
    w, h = T['window_w'] / 2, T['window_h'] / 2
    bw, b0, b1 = T['board_half_w'], T['board_bottom'], T['board_top']
    for lo, hi in [((fc - .02, lc - bw, b0), (fc + .02, lc - w, b1)), ((fc - .02, lc + w, b0), (fc + .02, lc + bw, b1)),
                   ((fc - .02, lc - w, b0), (fc + .02, lc + w, uc - h)), ((fc - .02, lc - w, uc + h), (fc + .02, lc + w, b1))]:
        b.append((lo, hi, 'board'))
    rf, rl = T['rock_center']
    rh = T['rock_half']
    if T['rock_top'] > 0.05:                                   # a raised target is an obstacle; a flat pad is not
        b.append(((rf - rh[0], rl - rh[1], 0.0), (rf + rh[0], rl + rh[1], T['rock_top']), 'rock'))
    b.append(((T['wall_f'][0], T['wall_l'][0], T['wall_bottom']),
              (T['wall_f'][1], T['wall_l'][1], T['wall_top']), 'wall'))
    for cf, cl in [(2.6, -0.6), (0.6, -1.2)]:
        b.append(((cf - 0.1, cl - 0.1, 0.0), (cf + 0.1, cl + 0.1, 0.3), 'cone'))
    b += list(extra)
    return [(np.array(lo, float), np.array(hi, float), n) for lo, hi, n in b]


class BoxMap:
    """Fake ObstacleMap in the true START frame: boxes = real structure the camera has observed, points = stereo
    phantom clusters.  Same API as node/obstacle_map.py: clearance/blocked with optional `exclude` cylinders
    (center_ned, axis_ned, radius_m, half_length_m) that hide POINTS (the openings never contain real boxes), and
    count(p, r) ~ number of voxels near p (a box counts as many)."""

    def __init__(self, boxes, sf, points=None):
        self.sf = sf
        self.set_boxes(boxes)
        self.pts = np.zeros((0, 3)) if points is None else np.asarray(points, float).reshape(-1, 3)
        self.calls = 0

    @property
    def points(self):
        """Like node ObstacleMap.points (NED): box surfaces sampled at 5 cm plus the phantom points."""
        P = np.concatenate([self._box_pts, self.pts]) if len(self.pts) else self._box_pts
        return self.sf.o + P[:, :1] * self.sf.fwd + P[:, 1:2] * self.sf.left - P[:, 2:3] * np.array([0.0, 0.0, 1.0])

    def set_boxes(self, boxes):
        self.lo = np.array([x[0] for x in boxes]) if boxes else np.zeros((0, 3))
        self.hi = np.array([x[1] for x in boxes]) if boxes else np.zeros((0, 3))
        pts = []
        for lo, hi in zip(self.lo, self.hi):
            ax = [np.arange(lo[i], hi[i] + 1e-9, 0.05) if hi[i] - lo[i] > 0.05 else np.array([(lo[i] + hi[i]) / 2])
                  for i in range(3)]
            g = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
            pts.append(g)
        self._box_pts = np.concatenate(pts) if pts else np.zeros((0, 3))

    def _flu(self, p_ned):
        p = np.atleast_2d(np.asarray(p_ned, float)) - self.sf.o
        return np.stack([p @ self.sf.fwd, p @ self.sf.left, -p[:, 2]], -1)

    def _points(self, exclude):
        P = self.pts
        for c, ax, r, hl in exclude or ():
            if not len(P):
                break
            c = self._flu(c)[0]
            ax = np.asarray(ax, float)
            ax = np.array([ax @ self.sf.fwd, ax @ self.sf.left, -ax[2]])
            ax = ax / np.linalg.norm(ax)
            rel = P - c
            along = rel @ ax
            radial = np.linalg.norm(rel - along[:, None] * ax[None, :], axis=1)
            P = P[~((np.abs(along) <= hl) & (radial <= r))]
        return P

    def _dist(self, q, exclude=None):
        d = np.full(len(q), np.inf)
        if len(self.lo):
            e = np.maximum(np.maximum(self.lo[None] - q[:, None], 0.0), q[:, None] - self.hi[None])
            d = np.linalg.norm(e, axis=2).min(axis=1)
        P = self._points(exclude)
        if len(P):
            d = np.minimum(d, np.linalg.norm(q[:, None] - P[None], axis=2).min(axis=1))
        return d

    def clearance(self, p_from, p_to, radius_m, exclude=None):
        self.calls += 1
        a, b = self._flu(p_from)[0], self._flu(p_to)[0]
        L = float(np.linalg.norm(b - a))
        n = max(2, int(math.ceil(L / 0.04)) + 1)
        s = np.linspace(0.0, L, n)
        q = a + (b - a) * (s / max(L, 1e-9))[:, None]
        hit = np.nonzero(self._dist(q, exclude) < radius_m)[0]
        return math.inf if len(hit) == 0 else float(s[hit[0]])

    def blocked(self, p_ned, radius_m, exclude=None):
        return bool(self._dist(self._flu(p_ned), exclude)[0] < radius_m)

    def count(self, p_ned, radius_m, exclude=None):
        q = self._flu(p_ned)
        if len(self.lo):
            e = np.maximum(np.maximum(self.lo - q, 0.0), q - self.hi)
            if np.linalg.norm(e, axis=1).min() < radius_m:
                return 1000                                    # real structure: many voxels
        P = self._points(exclude)
        return int(np.count_nonzero(np.linalg.norm(P - q, axis=1) < radius_m)) if len(P) else 0


class DepthLikeMap:
    """Depth-like obstacle map: wall surfaces sampled every 5 cm, a point is known only after it was inside the camera
    view (+-37 deg horizontal, +-25 deg vertical, 0.4..4.0 m) and is forgotten 4 s after it was last seen (like
    node/obstacle_map.py's memory).  No occlusion.  API like node ObstacleMap (.points, clearance, blocked)."""

    def __init__(self, walls, sf, memory_s=4.0, phantoms=(), seed=0):
        pts = []
        for w in walls:
            f0, l0, f1, l1, b0, b1 = w[:6]
            ht = w[7] if len(w) > 7 else 0.0
            end_only = w[8] if len(w) > 8 else None                    # 'a' / 'b' / 'ab': textureless face, only
            n = max(2, int(math.hypot(f1 - f0, l1 - l0) / 0.05) + 1)  # that edge-on END returns depth: one
            ln = math.hypot(f1 - f0, l1 - l0)                         # column, a point every 0.1 m (round B:
            nf, nl = (-(l1 - l0) / ln, (f1 - f0) / ln) if ln > 1e-9 else (0.0, 0.0)   # 6-11 point clusters)
            if end_only:
                for k in ([0] if 'a' in end_only else []) + ([n - 1] if 'b' in end_only else []):
                    f = f0 + (f1 - f0) * k / (n - 1)
                    l = l0 + (l1 - l0) * k / (n - 1)
                    for u in np.arange(b0, b1 + 1e-9, 0.10):
                        pts.append(sf.to_ned((f, l, u)))
                continue
            for side in ((-1.0, 1.0) if ht > 0 else (0.0,)):          # both faces of a thick board
                for k in range(n):
                    f = f0 + (f1 - f0) * k / (n - 1) + side * ht * nf
                    l = l0 + (l1 - l0) * k / (n - 1) + side * ht * nl
                    for u in np.arange(b0, b1 + 1e-9, 0.05):
                        pts.append(sf.to_ned((f, l, u)))
        self.all = np.array(pts)
        self.seen = np.full(len(self.all), -1e9)
        self.memory = memory_s
        rng = np.random.default_rng(seed)
        self.ph_groups = []                                 # [points, vanish_h (None: never), camera at first view]
        for g in phantoms:
            c, n, r = g[:3]
            pts_g = np.array([sf.to_ned(np.asarray(c, float) + rng.uniform(-r, r, 3)) for _ in range(n)]).reshape(-1, 3)
            self.ph_groups.append([pts_g, g[3] if len(g) > 3 else None, None])
        self.phantom = np.vstack([g[0] for g in self.ph_groups]) if self.ph_groups else np.zeros((0, 3))
        self.ph_vanished = []                               # (stereo phantoms: in the map, never in true_ahead;
                                                            # vanish_h: carved once the camera moved that far sideways)
        self.t = 0.0
        self._act = np.zeros((0, 3))

    def observe(self, t, cam, yaw):
        d = self.all - cam
        hd = np.hypot(d[:, 0], d[:, 1])
        bear = np.abs((np.arctan2(d[:, 1], d[:, 0]) - yaw + np.pi) % (2 * np.pi) - np.pi)
        elev = np.abs(np.arctan2(-d[:, 2], np.maximum(hd, 1e-6)))
        vis = (hd > 0.4) & (hd < 4.0) & (bear < math.radians(37.0)) & (elev < math.radians(25.0))
        self.seen[vis] = t
        self.t = t
        gone = False
        for g in list(self.ph_groups):
            if g[1] is None:
                continue
            if g[2] is None:
                g[2] = np.asarray(cam, float).copy()
            elif math.hypot(*(np.asarray(cam, float) - g[2])[:2]) >= g[1]:
                self.ph_groups.remove(g)
                self.ph_vanished.append(t)
                gone = True
        if gone:
            self.phantom = np.vstack([g[0] for g in self.ph_groups]) if self.ph_groups else np.zeros((0, 3))
        self._act = self.all[self.t - self.seen <= self.memory]
        if len(self.phantom):
            self._act = np.vstack([self._act, self.phantom])

    @property
    def points(self):
        return self._act

    def true_ahead(self, cam, h, r=0.2):
        """Ground truth: distance along the unit ray h from cam to the nearest wall point within r of the ray."""
        d = self.all - cam
        along = d @ h
        lat = np.linalg.norm(d - along[:, None] * h[None, :], axis=1)
        sel = (along > 0) & (lat < r)
        return float(along[sel].min()) if sel.any() else math.inf

    def _keep(self, exclude):
        P = self._act
        for c, ax, r, hl in exclude or ():
            ax = np.asarray(ax, float) / np.linalg.norm(ax)
            rel = P - np.asarray(c, float)
            along = rel @ ax
            radial = np.linalg.norm(rel - along[:, None] * ax[None, :], axis=1)
            P = P[~((np.abs(along) <= hl) & (radial <= r))]
        return P

    def clearance(self, p_from, p_to, radius_m, exclude=None):
        P = self._keep(exclude)
        if not len(P):
            return math.inf
        a, b = np.asarray(p_from, float), np.asarray(p_to, float)
        L = float(np.linalg.norm(b - a))
        lo, hi = np.minimum(a, b) - radius_m, np.maximum(a, b) + radius_m
        P = P[np.all((P >= lo) & (P <= hi), axis=1)]           # bounding-box prefilter
        if not len(P):
            return math.inf
        n = max(2, int(math.ceil(L / 0.04)) + 1)
        s = np.linspace(0.0, L, n)
        q = a + (b - a) * (s / max(L, 1e-9))[:, None]
        dmin = np.sqrt(((q[:, None, :] - P[None, :, :]) ** 2).sum(-1)).min(axis=1)
        hit = np.nonzero(dmin < radius_m)[0]
        return math.inf if len(hit) == 0 else float(s[hit[0]])

    def blocked(self, p_ned, radius_m, exclude=None):
        P = self._keep(exclude)
        return bool(len(P) and np.linalg.norm(P - np.asarray(p_ned, float), axis=1).min() < radius_m)


class LegacyMap:
    """A map with only the original two methods (no exclude, no count): the mission must still work."""

    def __init__(self, m):
        self.m = m

    def clearance(self, p_from, p_to, radius_m):
        return self.m.clearance(p_from, p_to, radius_m)

    def blocked(self, p_ned, radius_m):
        return self.m.blocked(p_ned, radius_m)


# ------------------------------------------------------------ scenario + world
@dataclass
class Scenario:
    name: str = 'nominal'
    home_ned: Any = (0.0, 0.0, 0.0)
    home_yaw: float = 0.0
    prior: dict = field(default_factory=prior_dict)
    extra_boxes: tuple = ()
    tunnel_jitter_s: float = 0.0          # seconds of unstable tunnel detections after it first becomes visible
    window_clearance_ok: bool = True
    tof_bias_m: float = 0.0
    landed_delay_s: float = 0.0
    pose_bad: tuple = ()                  # ((t0, t1), ...)
    takeover_t: Optional[float] = None
    wind: Optional[tuple] = None          # (t0, t1, (vf, vl, vu)) start-frame velocity disturbance
    est_drift: Optional[tuple] = None     # (t0, t1, (df, dl, du)) VIO drift: estimate - truth ramps from 0 to this
                                          # (start frame) over t0..t1; PX4 holds the ESTIMATE on the setpoint
    wall_measured_top: Optional[float] = None
    no_map: bool = False
    true_over: dict = field(default_factory=dict)   # override ground-truth geometry (TRUE keys)
    hidden_boxes: tuple = ()                        # real boxes the map only gets once the camera has looked at them
    hidden_reveal_m: float = 3.5                    # ... from this close
    phantoms: tuple = ()                            # ((f, l, u), n_points, vanish[, appear_t]) stereo phantom clusters;
                                                    # vanish: distance (carved when seen closer), None (never) or
                                                    # 'parallax' (view-locked: carved once the camera moves >= 8 cm
                                                    # across the line of sight from where it first saw it)
    legacy_map: bool = False                        # map without exclude / count
    landing_conf: tuple = (0.8, 0.8)                # uniform range of the rock detector confidence
    tunnel_conf: tuple = (0.42, 0.75)               # uniform range of the tunnel detector confidence
    tunnel_detector_off: bool = False               # official sim: a tiled arch, no hoops -> never found
    tunnel_det_period_s: float = 0.0                # > 0: a tunnel frame every this many s (round B tunnel.py ~1.9 s)
    tunnel_det_latency_s: float = 0.0               # ... delivered this long after its image (stamp = image time)
    landing_miss_p: float = 0.0
    fence_over: dict = field(default_factory=dict)
    invisible_boxes: tuple = ()                     # real (collisions) but never measured: edge-on thin boards
    default_course: bool = True                     # False: none of the standard course boxes (custom worlds)
    depth_walls: tuple = ()                         # ((f0, l0, f1, l1, bottom, top, name[, half_thick]), ...)
    vehicle_r: float = HALF_SPAN                    # collision radius (props: 0.23 m)
    px4_mpc: bool = False                           # PX4-like tracking: MPC_XY_P 0.95, MPC_XY_VEL_P_ACC 1.8 (~0.7
    px4_acc_ff: bool = True                         # m/s^2 effective deceleration) instead of the 0.15 s kinematic lag
    contact_physics: bool = False                   # prop strike: push out + bounce + IMU spike (Vehicle.accel)
    px4_official: bool = False                      # official lab sim: px4_mpc + horizontal acceleration <= 0.6 m/s^2
                                                    # with a 0.3 s lag (~0.3 m overshoot stopping from 0.5 m/s)
    yaw_wall_osc: bool = False                      # official run_003: a fast (> 0.8 rad/s) yaw with a wall < 0.8 m in
                                                    # front of the camera -> slow under-damped hold oscillation
                                                    # (~+-0.3 m, 6 s period) toward that wall
    contact_reveal: bool = False                    # an invisible box enters the map once we hit it (depth catches up)
    depth_phantoms: tuple = ()                      # ((f, l, u), n, radius[, vanish_h]): stereo phantoms (DepthLikeMap);
                                                    # vanish_h: carved after the camera moved that far horizontally
    pad_whole_view: bool = False                    # official-compat rule: the WHOLE pad (near edge) must be inside the
                                                    # 25.3 deg lower half-FOV (1.46 m at 0.55 m camera height)
    pilot_mode: bool = False                        # skybeta contract: scripted pilot, PilotPX4
    pilot_arm_t: float = 1.0                        # pilot arms (Altitude) ...
    pilot_react_s: float = 1.5                      # ... takes over to land this long after 'pilot_land_requested'
    pilot_rearm_delay_s: float = 3.0                # ... re-arms this long after the mission starts waiting in rearm
    pilot_refuse_offboard_on_rearm: bool = False    # flight interface 'staying out of Offboard' after the re-arm
    pilot_takeover_t: Optional[float] = None        # stick move / mode switch mid-flight
    pilot_z_bias_m: float = 0.0                     # the vehicle settles this far ABOVE its pose setpoint (real
                                                    # skybeta-05: PX4 local vs cuVSLAM odom 0.46 m apart)
    lat_drift_per_speed: float = 0.0                # PX4 tracking error model: drift to the right = k * |v_horizontal|
    lndmc_offboard: float = 0.25                    # PX4 LNDMC_Z_VEL_MAX as seen by the offboard land detector
    pad_cam: bool = False                           # flat pad seen by the level camera (pad_approach): detected only
                                                    # with >= 60 % of it below the horizon cut-off (floor visible from
                                                    # h / tan 25.3 deg ahead), <= 3 m, within 30 deg of the axis
    spawn: Optional[tuple] = None                   # (f, l, yaw_deg[, up]) course start frame: where the vehicle stands
                                                    # at arming (per-stage harness); None = home_ned / home_yaw
    t_max: float = 160.0
    seed: int = 1


class World:
    def __init__(self, scn: Scenario):
        self.scn = scn
        self.sf = StartFrame(scn.home_ned, scn.home_yaw)
        self.T = dict(TRUE, **scn.true_over)
        self.boxes = course_boxes(self.T, scn.extra_boxes) if scn.default_course else \
            [(np.array(lo, float), np.array(hi, float), n) for lo, hi, n in scn.extra_boxes]
        self.dmap = None
        self.segs = []                                         # diagonal depth walls (a, b, bottom, top, half_t, name)
        if scn.depth_walls:                                    # walls as thin boxes (collisions) + a depth-like map
            for w in scn.depth_walls:
                f0, l0, f1, l1, b0, b1, name = w[:7]
                ht = w[7] if len(w) > 7 else 0.025
                if abs(f0 - f1) > 1e-6 and abs(l0 - l1) > 1e-6:      # diagonal board: segment collision
                    self.segs.append((np.array([f0, l0], float), np.array([f1, l1], float), b0, b1, ht, name))
                    continue
                lo = (min(f0, f1) - ht, min(l0, l1) - ht, b0)
                hi = (max(f0, f1) + ht, max(l0, l1) + ht, b1)
                self.boxes.append((np.array(lo, float), np.array(hi, float), name))
            self.dmap = DepthLikeMap(scn.depth_walls, self.sf, phantoms=scn.depth_phantoms, seed=scn.seed)
        self.map_boxes = [(lo, hi) for lo, hi, _ in self.boxes]
        hidden = [(np.array(lo, float), np.array(hi, float), n) for lo, hi, n in scn.hidden_boxes]
        self.boxes += [(np.array(lo, float), np.array(hi, float), n) for lo, hi, n in scn.invisible_boxes]
        self.boxes += hidden                                   # they exist (collisions) ...
        self.hidden = list(hidden)                             # ... but are not in the map yet
        self.revealed = []
        self.rng = np.random.default_rng(scn.seed)
        self.phantoms = []
        for ph in scn.phantoms:
            c, n, vanish = ph[:3]
            pts = np.asarray(c, float) + self.rng.normal(0, 0.04, (n, 3))
            self.phantoms.append(dict(c=np.asarray(c, float), pts=pts, vanish=vanish,
                                      appear=ph[3] if len(ph) > 3 else 0.0, cam0=None))
        self.vanished = []
        self.map = BoxMap(self.map_boxes, self.sf, self._phantom_pts(0.0))
        self._n_live = sum(1 for ph in self.phantoms if ph['appear'] <= 0.0)
        self.tunnel_seen_t = None
        self._tun_next, self._tun_pend, self._tun_out = -math.inf, [], None

    def flu(self, p):
        return self.sf.from_ned(p)

    def _phantom_pts(self, t=None):
        live = [ph['pts'] for ph in self.phantoms if t is None or t >= ph['appear']]
        return np.concatenate(live) if live else None

    def _update_map(self, t, cam, yaw):
        changed = False
        for lo, hi, name in list(self.hidden):                 # reveal hidden boxes once in view
            corners = [np.array([x, y, z]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
            corners.append((lo + hi) / 2)
            if any(self._visible(cam, yaw, self.sf.to_ned(c), max_range=self.scn.hidden_reveal_m) for c in corners):
                self.hidden = [h for h in self.hidden if h[2] != name]
                self.map_boxes.append((lo, hi))
                self.revealed.append((t, name))
                changed = True
        n_live = sum(1 for ph in self.phantoms if t >= ph['appear'])
        if n_live != self._n_live:
            self._n_live, changed = n_live, True
        for ph in list(self.phantoms):                         # carving
            c_ned = self.sf.to_ned(ph['c'])
            if t < ph['appear'] or ph['vanish'] is None or not self._visible(cam, yaw, c_ned):
                continue
            if ph['vanish'] == 'parallax':                     # view-locked: needs a new viewpoint across the ray
                if ph['cam0'] is None:
                    ph['cam0'] = cam.copy()
                ray = c_ned - ph['cam0']
                ray /= np.linalg.norm(ray)
                d = cam - ph['cam0']
                gone = np.linalg.norm(d - (d @ ray) * ray) >= 0.08
            else:                                              # seen through from close enough
                gone = math.hypot(*(c_ned - cam)[:2]) < ph['vanish']
            if gone:
                self.phantoms.remove(ph)
                self.vanished.append(t)
                changed = True
        if changed:
            self.map.set_boxes(self.map_boxes)
            P = self._phantom_pts(t)
            self.map.pts = np.zeros((0, 3)) if P is None else P
            self._n_live = sum(1 for ph in self.phantoms if t >= ph['appear'])

    def ground_up(self, p_ned):
        f, l, _ = self.flu(p_ned)
        rf, rl = self.T['rock_center']
        rh = self.T['rock_half']
        return self.T['rock_top'] if (abs(f - rf) <= rh[0] and abs(l - rl) <= rh[1]) else 0.0

    def ground_z(self, p_ned):
        return float(self.sf.o[2] - self.ground_up(p_ned))

    def push_out(self, p_ned, skip=('rock',)):
        """Colliding box -> (name, outward horizontal unit normal in NED, penetration m) or None."""
        q = self.flu(p_ned)
        for lo, hi, name in self.boxes:
            if name in skip or max(lo[2] - q[2], 0, q[2] - hi[2]) >= HALF_HEIGHT:
                continue
            c = np.minimum(np.maximum(q[:2], lo[:2]), hi[:2])
            d = q[:2] - c
            nd = float(np.linalg.norm(d))
            if nd >= self.scn.vehicle_r:
                continue
            if nd < 1e-6:                                     # centre inside: leave by the nearest face
                gaps = [q[0] - lo[0], hi[0] - q[0], q[1] - lo[1], hi[1] - q[1]]
                k = int(np.argmin(gaps))
                d = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]][k], float)
                nd, pen = 1.0, self.scn.vehicle_r + gaps[k]
            else:
                pen = self.scn.vehicle_r - nd
            u = d / nd
            n_ned = u[0] * self.sf.fwd + u[1] * self.sf.left
            return name, n_ned, pen
        for a, b, b0, b1, ht, name in self.segs:
            if name in skip or max(b0 - q[2], 0, q[2] - b1) >= HALF_HEIGHT:
                continue
            dd, cpt = _seg_d(q[:2], a, b)
            if dd - ht < self.scn.vehicle_r:
                u = (q[:2] - cpt) / max(dd, 1e-6)
                return name, u[0] * self.sf.fwd + u[1] * self.sf.left, self.scn.vehicle_r + ht - dd
        return None

    def reveal(self, name):
        for lo, hi, n in self.boxes:
            if n == name and not any(np.allclose(lo, a) and np.allclose(hi, b) for a, b in self.map_boxes):
                self.map_boxes.append((lo, hi))
                self.revealed.append((None, name))
                self.map = BoxMap(self.map_boxes, self.sf, self._phantom_pts(0.0))

    def clearance_xy(self, p_ned, skip=('rock',)):
        """Horizontal distance from the body centre to the nearest box at this height."""
        q = self.flu(p_ned)
        best = math.inf
        for lo, hi, name in self.boxes:
            if name in skip or max(lo[2] - q[2], 0, q[2] - hi[2]) >= HALF_HEIGHT:
                continue
            best = min(best, float(np.linalg.norm(np.maximum(np.maximum(lo[:2] - q[:2], 0), q[:2] - hi[:2]))))
        for a, b, b0, b1, ht, name in self.segs:
            if name not in skip and max(b0 - q[2], 0, q[2] - b1) < HALF_HEIGHT:
                best = min(best, _seg_d(q[:2], a, b)[0] - ht)
        return best

    def collision(self, p_ned, skip=('rock',)):
        """Vehicle = vertical cylinder (r=HALF_SPAN, half height HALF_HEIGHT)."""
        q = self.flu(p_ned)
        for lo, hi, name in self.boxes:
            if name in skip:
                continue
            dxy = np.linalg.norm(np.maximum(np.maximum(lo[:2] - q[:2], 0), q[:2] - hi[:2]))
            dz = max(lo[2] - q[2], 0, q[2] - hi[2])
            if dxy < self.scn.vehicle_r and dz < HALF_HEIGHT:
                return name
        for a, b, b0, b1, ht, name in self.segs:
            if name not in skip and _seg_d(q[:2], a, b)[0] - ht < self.scn.vehicle_r and \
                    max(b0 - q[2], 0, q[2] - b1) < HALF_HEIGHT:
                return name
        return None

    def _visible(self, cam, heading, target, max_range=3.5, fov_deg=35.0, vfov_deg=25.0):
        d = np.asarray(target) - cam
        hd = math.hypot(d[0], d[1])
        if hd < 0.3 or hd > max_range:
            return False
        if abs((math.atan2(d[1], d[0]) - heading + math.pi) % (2 * math.pi) - math.pi) > math.radians(fov_deg):
            return False
        return abs(math.atan2(d[2], hd)) < math.radians(vfov_deg)

    def perceive(self, t, px4):
        scn, sf, rng = self.scn, self.sf, self.rng
        yaw = px4.yaw
        cam = px4.p + 0.116 * np.array([math.cos(yaw), math.sin(yaw), 0.0])     # oak_left is 0.116 m ahead
        self._update_map(t, cam, yaw)
        om = None if scn.no_map else (LegacyMap(self.map) if scn.legacy_map else self.map)
        if self.dmap is not None:
            self.dmap.observe(t, cam, yaw)
            om = self.dmap
        per = Perception(obstacles=om, obstacles_t=t,
                         floor=FloorFit(True, float(sf.o[2]), np.array([0, 0, -1.0]), 0.7), floor_t=t)
        # tunnel (behaviour measured on the venue replica): the detector reports the first hoop that is >= ~1 m
        # ahead of the camera, i.e. closer than that it jumps to the NEXT hoop (+0.6 m along the axis);
        # recall 100 % <= 2 m, ~60 % at 2-3 m, nothing beyond; confidence ~0.4-0.75 with an off prior
        entry0 = sf.to_ned(self.T['tunnel_entry'])
        axis = sf.direction(self.T['tunnel_axis_deg'])
        ring = None
        for k in range(int(round(self.T['tunnel_length'] / self.T['tunnel_ring_gap'])) + 1):
            c = entry0 + axis * k * self.T['tunnel_ring_gap']
            if (c - cam) @ axis >= 1.0:
                ring = c
                break
        seen = ring is not None and self._visible(cam, yaw, ring)
        if seen:
            dist = math.hypot(*(ring - cam)[:2])
            seen = rng.random() < (1.0 if dist <= 2.0 else 0.6 if dist <= 3.0 else 0.0)
        if seen:
            if self.tunnel_seen_t is None:
                self.tunnel_seen_t = t
            jitter = t - self.tunnel_seen_t < scn.tunnel_jitter_s
            e = ring + rng.normal(0, 0.02, 3) + (rng.uniform(-0.4, 0.4, 3) * [1, 1, 0.5] if jitter else 0)
            a = math.atan2(axis[1], axis[0]) + rng.normal(0, math.radians(1.5)) + (rng.uniform(-0.4, 0.4) if jitter else 0)
            tun = TunnelEstimate(True, e, np.array([math.cos(a), math.sin(a)]),
                                 self.T['tunnel_radius'] + rng.normal(0, 0.01), float(rng.uniform(*scn.tunnel_conf)))
        else:
            tun = TunnelEstimate(False, np.zeros(3), np.array([1.0, 0.0]), 0.0, 0.0)
        if scn.tunnel_detector_off:
            tun = TunnelEstimate(False, np.zeros(3), np.array([1.0, 0.0]), 0.0, 0.0)
        if scn.tunnel_det_period_s > 0:                           # slow detector: frames every period, each
            if t >= self._tun_next:                                # delivered latency s after its image
                self._tun_next = t + scn.tunnel_det_period_s
                self._tun_pend.append((t + scn.tunnel_det_latency_s, t, tun))
            while self._tun_pend and self._tun_pend[0][0] <= t:
                self._tun_out = self._tun_pend.pop(0)[1:]
            per.tunnel, per.tunnel_t = (None, None) if self._tun_out is None else (self._tun_out[1], self._tun_out[0])
        else:
            per.tunnel, per.tunnel_t = tun, t
        # window
        wc = sf.to_ned(self.T['window_center'])
        wd = sf.direction(self.T['window_through_deg'])
        why = 'no_board_plane'
        if (wc - cam) @ wd > 0.3 and self._visible(cam, yaw, wc, max_range=3.2):
            lat = np.array([-wd[1], wd[0], 0.0])                 # replica-like: the whole board must be in view
            edges = [wc + lat * self.T['board_half_w'], wc - lat * self.T['board_half_w']]
            ray = (wc - cam)[:2] / np.linalg.norm((wc - cam)[:2])
            oblique = math.degrees(math.acos(min(1.0, float(ray @ wd[:2]))))
            why = ('cut_by_fov' if not all(self._visible(cam, yaw, e, max_range=4.5) for e in edges)
                   else 'not_enclosed' if oblique > 13.0 else '')
        if why == '':
            n = wd * (1 if rng.random() < 0.5 else -1) + np.r_[rng.normal(0, 0.02, 2), 0]
            per.window = WindowEstimate(True, wc + rng.normal(0, 0.02, 3), n, self.T['window_w'], self.T['window_h'],
                                        scn.window_clearance_ok, 0.8)
        else:
            per.window = WindowEstimate(False, np.zeros(3), np.zeros(3), 0, 0, False, 0.0, reason=why)
        per.window_t = t
        # landing target (white rock): seen from the side only (horizontal camera), and only when the camera is
        # >= 0.35 m above the top (replica eval: lower, the low-texture top is a hole in stereo depth)
        rf, rl = self.T['rock_center']
        top = sf.to_ned((rf, rl, self.T['rock_top']))
        hd = math.hypot(*(top - cam)[:2])
        above = float(top[2] - cam[2])                          # camera height above the rock top
        bearing = abs((math.atan2(*(top - cam)[1::-1]) - yaw + math.pi) % (2 * math.pi) - math.pi)
        if scn.pad_cam:
            vis = min(1.0, max(0.0, (hd + 0.30 - above / math.tan(math.radians(25.3))) / 0.60))
            if hd > 3.0 or bearing > math.radians(30.0) or vis < 0.6 or rng.random() < scn.landing_miss_p:
                per.landing = LandingTarget(False, np.zeros(3), 0.0, 0.0, 0.0, reason='outside_view')
            else:
                c = top + np.r_[rng.normal(0, 0.01 + 0.012 * hd, 2), 0]
                per.landing = LandingTarget(True, c, float(top[2]), 0.6, float(rng.uniform(*scn.landing_conf)),
                                            yaw_ned=float(rng.normal(0, 0.03)), range_m=hd)
            per.landing_t = t
            if scn.wall_measured_top is not None:
                per.wall_top_d, per.wall_t = float(sf.o[2] - scn.wall_measured_top), t
            return per
        if hd > 3.5 or bearing > math.radians(35.0):
            why = 'nothing_raised'
        elif above < 0.35:
            why = 'no_flat_top'                                  # grazing view of the low-texture top
        elif scn.pad_whole_view and (hd - 0.30 <= 0.05 or math.degrees(math.atan2(above, hd - 0.30)) > 25.3):
            why = 'outside_view'                                 # pad near edge below the image
        elif hd <= 0.8 or math.degrees(math.atan2(above, hd)) > 22.0:
            why = 'top_not_visible'                              # below the vertical FOV
        elif rng.random() < scn.landing_miss_p:
            why = 'low_confidence'
        else:
            why = ''
        if why == '':
            c = top + np.r_[rng.normal(0, 0.03, 2), 0]
            per.landing = LandingTarget(True, c, float(top[2] + rng.normal(0, 0.02)), 0.7,
                                        float(rng.uniform(*scn.landing_conf)))
        else:
            per.landing = LandingTarget(False, np.zeros(3), 0.0, 0.0, 0.0, reason=why)
        per.landing_t = t
        if scn.wall_measured_top is not None:
            per.wall_top_d, per.wall_t = float(sf.o[2] - scn.wall_measured_top), t
        return per


class FakePX4:
    """Kinematic multicopter + the PX4 behaviours the mission relies on."""

    def __init__(self, world: World):
        self.w, self.scn = world, world.scn
        self.p = np.array(world.scn.home_ned, float)
        self.v = np.zeros(3)
        self.yaw = world.scn.home_yaw
        if world.scn.spawn is not None:
            sp = tuple(world.scn.spawn)
            self.p = world.sf.to_ned((sp[0], sp[1], sp[3] if len(sp) > 3 else 0.0))
            self.yaw = wrap(world.scn.home_yaw - math.radians(sp[2]))
        self.armed, self.nav, self.landed = False, NAV_POSCTL, True
        self.sp = None
        self.stream_t0 = None
        self.last_sp_t = -1.0
        self.contact_s = 0.0
        self.landed_s = 0.0
        self.vt = np.zeros(3)
        self.cmds = []                        # (t, Command, armed, landed, tof)
        self.contact_t = None
        self.contacts = []                    # (t, box) prop strikes (contact_physics)
        self.acc_f = np.zeros(2)              # px4_official: lagged horizontal acceleration
        self.osc_events = []                  # yaw_wall_osc: trigger times

    def tof(self):
        return float(self.p[2] - self.w.ground_z(self.p)) * -1.0 + self.scn.tof_bias_m

    def est_off(self, t):
        """NED estimate error (estimate - truth) of Scenario.est_drift at time t."""
        ed = self.scn.est_drift
        if not ed:
            return np.zeros(3)
        t0, t1, (df, dl, du) = ed
        k = float(np.clip((t - t0) / max(t1 - t0, 1e-6), 0.0, 1.0))
        sf = self.w.sf
        return k * (df * sf.fwd + dl * sf.left + np.array([0.0, 0.0, -du]))

    def vehicle(self, t):
        ok = not any(a <= t < b for a, b in self.scn.pose_bad)
        spike = self.contact_t is not None and 0.0 <= t - self.contact_t < 0.06
        acc = np.array([0.0, 0.0, -9.81]) + (np.array([60.0, 0.0, 0.0]) if spike else 0.0)
        return Vehicle(pos_ned=self.p.copy() + self.est_off(t), heading=self.yaw, vel_ned=self.v.copy(), armed=self.armed,
                       nav_state=self.nav, landed=self.landed, pose_ok=ok, tof_m=max(0.0, self.tof()), tof_t=t,
                       accel=acc if self.scn.contact_physics else None)

    def apply(self, out, t):
        if out.pos_ned is not None:
            if self.stream_t0 is None:
                self.stream_t0 = t
            self.sp = (np.array(out.pos_ned) - self.est_off(t), np.array(out.vel_ned), float(out.yaw))
            self.sp_acc = None if getattr(out, 'acc_ned', None) is None else np.array(out.acc_ned, float)
            self.last_sp_t = t
        else:
            self.stream_t0 = None
        for c in out.commands:
            self.cmds.append((t, c, self.armed, self.landed, self.tof()))
            if c.name == 'offboard' and self.stream_t0 is not None and t - self.stream_t0 >= 0.5:
                self.nav = NAV_OFFBOARD
            elif c.name == 'arm' and self.nav == NAV_OFFBOARD and not self.armed and self.landed:
                self.armed = True
            elif c.name == 'disarm' and self.armed and self.landed:     # normal disarm: only when landed
                self.armed = False
            elif c.name == 'land' and self.armed:
                self.nav = NAV_AUTO_LAND

    def integrate(self, t, dt):
        scn = self.scn
        if scn.takeover_t is not None and t >= scn.takeover_t and self.nav == NAV_OFFBOARD:
            self.nav = NAV_POSCTL
        vt = np.zeros(3)
        if self.armed:
            if self.nav == NAV_OFFBOARD and self.sp is not None and t - self.last_sp_t < 0.5:
                sp_p, sp_v, sp_yaw = self.sp
                vt = sp_v + (0.95 if scn.px4_mpc else 1.5) * (sp_p - self.p)
                h = np.linalg.norm(vt[:2])
                if h > 3.5:
                    vt[:2] *= 3.5 / h
                vt[2] = np.clip(vt[2], -2.0, 1.5)
                dy = (sp_yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
                self.yaw += float(np.clip(dy, -2.5 * dt, 2.5 * dt))
            elif self.nav == NAV_AUTO_LAND:
                vt = np.array([0.0, 0.0, 0.7])
            if scn.px4_mpc and self.nav == NAV_OFFBOARD:   # PX4 velocity loop: acc = ff + 1.8 (v_sp - v), <= 3 m/s^2
                acc = 1.8 * (vt - self.v)
                af = getattr(self, 'sp_acc', None)
                if scn.px4_acc_ff and af is not None and np.all(np.isfinite(af)):
                    acc = acc + af
                acc[2] = (vt[2] - self.v[2]) / 0.15          # vertical: keep the fast kinematic model
                h = np.linalg.norm(acc[:2])
                amax = 0.6 if scn.px4_official else 3.0
                if h > amax:
                    acc[:2] *= amax / h
                if scn.px4_official:                         # (attitude lag: the tilt follows in ~0.3 s)
                    self.acc_f = self.acc_f + (acc[:2] - self.acc_f) * min(1.0, dt / 0.3)
                    acc[:2] = self.acc_f
                dv = acc * dt
            else:
                dv = (vt - self.v) * min(1.0, dt / 0.15)
            n = np.linalg.norm(dv)
            if n > 6.0 * dt:
                dv *= 6.0 * dt / n
            self.v = self.v + dv
            wind = np.zeros(3)
            if scn.wind is not None and scn.wind[0] <= t < scn.wind[1]:
                f, l, u = scn.wind[2]
                wind = f * self.w.sf.fwd + l * self.w.sf.left + np.array([0, 0, -u])
            if scn.yaw_wall_osc:
                wind = wind + self._yaw_osc(t, dt)
            sp_h = float(np.hypot(self.v[0], self.v[1]))
            if scn.lat_drift_per_speed and sp_h > 0.05:
                wind = wind + scn.lat_drift_per_speed * sp_h * np.array([-self.v[1], self.v[0], 0.0]) / sp_h
            self.p = self.p + (self.v + wind) * dt
            if scn.contact_physics:                     # prop strike: pushed out, bounce, IMU spike
                hit = self.w.push_out(self.p)
                if hit is not None:
                    name, n, pen = hit
                    self.p = self.p + n * (pen + 1e-3)
                    vn = float(self.v @ n)
                    if vn < 0:
                        self.v = self.v - 1.3 * vn * n
                    if self.contact_t is None or t - self.contact_t > 0.2:
                        self.contacts.append((t, name))
                    self.contact_t = t
                    if scn.contact_reveal:
                        self.w.reveal(name)
        else:
            self.v = np.zeros(3)
        self.vt = vt
        gz = self.w.ground_z(self.p)
        if self.p[2] >= gz:
            self.p[2] = gz
            self.v[2] = min(self.v[2], 0.0)
        on_ground = self.p[2] >= gz - 0.02
        # PX4 1.17 MulticopterLandDetector, simplified: ground contact only while the COMMANDED descent
        # (trajectory_setpoint.velocity[2] in Offboard, the mode's own descent in AUTO_LAND) is >= 1.1 *
        # LNDMC_Z_VEL_MAX and the vehicle does not move; landed after LNDMC_TRIG_TIME (1 s) of contact;
        # stays landed until a climb is commanded.  The position error alone does NOT count.
        cmd_vz, climb, lndmc = 0.0, False, 0.25
        if self.armed and self.nav == NAV_OFFBOARD and self.sp is not None:
            sp_p, sp_v, _ = self.sp
            cmd_vz, lndmc = float(sp_v[2]), scn.lndmc_offboard
            climb = sp_v[2] < -0.1 or sp_p[2] < self.p[2] - 0.1
        elif self.armed and self.nav == NAV_AUTO_LAND:
            cmd_vz = 0.7
        contact = (self.armed and on_ground and cmd_vz >= 1.1 * lndmc and abs(self.v[2]) < 0.1
                   and np.linalg.norm(self.v[:2]) < 0.15)
        self.contact_s = self.contact_s + dt if contact else 0.0
        if not self.armed:
            self.landed = on_ground
        elif not on_ground or climb:
            self.landed = False
        elif not self.landed:
            self.landed = self.contact_s >= 1.0 + scn.landed_delay_s
        if self.armed and self.nav == NAV_AUTO_LAND and self.landed:
            self.landed_s += dt
            if self.landed_s >= 1.0:
                self.armed = False
        else:
            self.landed_s = 0.0


def _yaw_osc(self, t, dt):
    """Official-sim hold oscillation: triggered by a yaw faster than 0.8 rad/s (over 0.5 s) while a wall is within
    0.8 m straight ahead of the camera; then a velocity disturbance 0.45 sin(2 pi t / 6) exp(-t / 10) toward it."""
    yh = getattr(self, '_yh', [])
    yh.append((t, self.yaw))
    self._yh = yh = [x for x in yh if t - x[0] <= 0.5]
    rate = abs(((yh[-1][1] - yh[0][1] + math.pi) % (2 * math.pi) - math.pi) / max(1e-3, yh[-1][0] - yh[0][0])) \
        if len(yh) > 2 and yh[-1][0] - yh[0][0] > 0.3 else 0.0
    osc = getattr(self, '_osc', None)
    if osc is None and rate > 0.8 and self.w.dmap is not None:
        hd = np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])
        cam = self.p + 0.116 * hd
        if self.w.dmap.true_ahead(cam, hd, r=0.2) < 0.8:
            self._osc = osc = (t, hd.copy())
            self.osc_events.append(t)
    if osc is None:
        return np.zeros(3)
    tr = t - osc[0]
    if tr > 15.0:
        self._osc = None
        return np.zeros(3)
    return 0.45 * math.sin(2 * math.pi * tr / 6.0) * math.exp(-tr / 10.0) * osc[1]   # (+-0.3 m against the loop)


FakePX4._yaw_osc = _yaw_osc


class PilotPX4(FakePX4):
    """skybeta 0.3.2 as described by the real-drone-adapter session: no command channel; the pilot arms and switches to
    Position, the flight interface then enters Offboard ONCE per arming (needs a fresh setpoint, which the adapter
    always streams); pose setpoints only (velocity/acceleration feed-forward dropped), clamped to 0.3..3.0 m above the
    cuVSLAM origin; PX4 tracks the pose at ~gain x distance; position-only Offboard never reports landed (the land
    detector needs a commanded descent velocity), so the pilot lands; PX4 auto-disarms 2 s after landing."""
    GAIN = 0.95                                  # fleet MPC_XY_P
    MAX_LEAD = 1.5                               # adapter clips any pose to 1.5 m from the vehicle (sky_lead_clipped)

    def __init__(self, world):
        super().__init__(world)
        self.origin_d = float(world.scn.home_ned[2])
        self.script = [(world.scn.pilot_arm_t, 'arm'), (world.scn.pilot_arm_t + 1.0, 'position')]
        self.n_arm = 0
        self.landing = False
        self.rearm_scheduled = False
        self.landed_hold = 0.0
        self.offboard_starts = []                # times Offboard became active
        self.flew, self.armed_t = False, 0.0
        self.lead_clipped = 0
        self.clip_t = []

    def vehicle(self, t):
        v = super().vehicle(t)
        v.offboard_active = self.nav == NAV_OFFBOARD
        v.clamp_origin_d = self.origin_d
        return v

    def _clamp(self, q):
        q = np.array(q, float)
        q[2] = min(max(q[2], self.origin_d - 3.0), self.origin_d - 0.3)
        return q

    def apply(self, out, t):
        assert not out.commands, 'pilot_mode must never send commands: %s' % out.commands
        scn = self.scn
        for e in out.events:
            if e['kind'] == 'pilot_land_requested' and not self.landing:
                self.script.append((t + scn.pilot_react_s, 'pilot_land'))
        if out.phase == 'rearm' and not self.armed and not self.rearm_scheduled:
            self.rearm_scheduled = True
            self.script += [(t + scn.pilot_rearm_delay_s, 'arm'), (t + scn.pilot_rearm_delay_s + 1.0, 'position')]
        # the adapter: our pose while Offboard, otherwise the current pose (never a stale target); clamped
        if self.nav == NAV_OFFBOARD and out.pos_ned is not None:
            q = self._clamp(out.pos_ned)
            d = q - self.p
            n = float(np.linalg.norm(d))
            if n > self.MAX_LEAD + 1e-6:
                self.lead_clipped += 1
                self.clip_t.append(t)
                q = self.p + d * (self.MAX_LEAD / n)
            self.sp = (q, np.zeros(3), float(out.yaw))
        else:
            self.sp = (self._clamp(self.p), np.zeros(3), self.yaw)
        self.last_sp_t = t
        due = [a for a in self.script if a[0] <= t]
        self.script = [a for a in self.script if a[0] > t]
        for _, act in sorted(due):
            if act == 'arm' and not self.armed and self.landed:
                self.armed, self.n_arm, self.flew, self.armed_t = True, self.n_arm + 1, False, t
            elif act == 'position' and self.armed:
                refuse = scn.pilot_refuse_offboard_on_rearm and self.n_arm > 1
                self.nav = NAV_POSCTL if refuse else NAV_OFFBOARD
                if not refuse:
                    self.offboard_starts.append(t)
            elif act == 'pilot_land' and self.armed:
                self.nav, self.landing = NAV_POSCTL, True

    def integrate(self, t, dt):
        scn = self.scn
        if scn.pilot_takeover_t is not None and t >= scn.pilot_takeover_t and self.nav == NAV_OFFBOARD and self.n_arm == 1:
            self.nav = NAV_POSCTL
        vt = np.zeros(3)
        if self.armed:
            if self.nav == NAV_OFFBOARD:
                tgt = self.sp[0] - np.array([0.0, 0.0, scn.pilot_z_bias_m])   # (height bias: sits above it)
                vt = self.GAIN * (tgt - self.p)                         # position only, no feed-forward
                h = np.linalg.norm(vt[:2])
                if h > 5.0:
                    vt[:2] *= 5.0 / h
                vt[2] = np.clip(vt[2], -2.0, 1.5)
                dy = (self.sp[2] - self.yaw + math.pi) % (2 * math.pi) - math.pi
                self.yaw += float(np.clip(dy, -2.5 * dt, 2.5 * dt))
            elif self.landing:
                vt = np.array([0.0, 0.0, 0.5])                           # the pilot lands in place
            dv = (vt - self.v) * min(1.0, dt / 0.15)
            n = np.linalg.norm(dv)
            if n > 6.0 * dt:
                dv *= 6.0 * dt / n
            self.v = self.v + dv
            wind = np.zeros(3)
            if scn.wind is not None and scn.wind[0] <= t < scn.wind[1]:
                f, l, u = scn.wind[2]
                wind = f * self.w.sf.fwd + l * self.w.sf.left + np.array([0, 0, -u])
            self.p = self.p + (self.v + wind) * dt
        else:
            self.v = np.zeros(3)
        gz = self.w.ground_z(self.p)
        if self.p[2] >= gz:
            self.p[2] = gz
            self.v[2] = min(self.v[2], 0.0)
        on_ground = self.p[2] >= gz - 0.02
        if self.armed and not on_ground:
            self.flew = True
        if self.armed and not self.flew and t - self.armed_t >= 10.0:   # COM_DISARM_PRFLT: no takeoff in 10 s
            self.armed, self.nav = False, NAV_POSCTL
            self.rearm_scheduled = False
        if not self.armed:
            self.landed = on_ground
        elif self.nav == NAV_OFFBOARD or not on_ground:
            self.landed = False                                          # position-only Offboard: never 'landed'
            self.landed_hold = 0.0
        else:
            self.landed_hold += dt
            self.landed = self.landed_hold >= 1.0
        if self.armed and self.landed and self.flew:
            self.landed_s += dt
            if self.landed_s >= 2.0:                                     # COM_DISARM_LAND
                self.armed, self.landing, self.nav = False, False, NAV_POSCTL
                self.rearm_scheduled = False
        else:
            self.landed_s = 0.0


def run(scn: Scenario, params: Optional[Params] = None):
    world = World(scn)
    px4 = PilotPX4(world) if scn.pilot_mode else FakePX4(world)
    if params is None:
        params = Params(pilot_mode=True) if scn.pilot_mode else Params()
    m = Mission(CoursePrior.from_dict(scn.prior), params)
    log = []
    per = None
    t, end_t = 0.0, None
    while t < scn.t_max:
        if per is None or int(round(t / DT)) % 2 == 0:       # perception at 10 Hz
            per = world.perceive(t, px4)
        per.obstacles_t = t
        veh = px4.vehicle(t)
        out = m.step(t, veh, per)
        log.append(dict(t=t, phase=out.phase, sub=out.sub, p=px4.p.copy(), v=px4.v.copy(), yaw=px4.yaw, armed=veh.armed,
                        nav=veh.nav_state, landed=veh.landed, tof=veh.tof_m, out=out))   # what the mission saw
        px4.apply(out, t)
        px4.integrate(t, DT)
        if m.phase in TERMINAL_PHASES and not px4.armed:
            end_t = end_t if end_t is not None else t
            if t - end_t > 1.0:
                break
        t = round(t + DT, 6)
    return m, px4, world, log


# ------------------------------------------------------------------ helpers
def phase_seq(log):
    seq = []
    for r in log:
        if not seq or seq[-1] != r['phase']:
            seq.append(r['phase'])
    return seq


def events(log, kind=None):
    ev = [e for r in log for e in r['out'].events]
    return [e for e in ev if kind is None or e['kind'] == kind]


def first(log, pred):
    for r in log:
        if pred(r):
            return r
    return None


def assert_no_collision(world, log, skip=('rock',)):
    for r in log:
        hit = world.collision(r['p'], skip)
        assert hit is None, 't=%.2f phase=%s: vehicle hit %s at %s' % (r['t'], r['phase'], hit, world.flu(r['p']))


def assert_setpoints_inside_fence(world, log):
    for r in log:
        o = r['out']
        if o.pos_ned is None or r['phase'] in ('wait', 'arm'):
            continue
        f, l, u = world.flu(o.pos_ned)
        assert FENCE['forward_min'] <= f <= FENCE['forward_max'] and FENCE['left_min'] <= l <= FENCE['left_max'] \
            and u <= FENCE['up_max'], 'setpoint outside fence at t=%.2f: %s' % (r['t'], (f, l, u))


def assert_released_forever(log, t0):
    for r in log:
        if r['t'] > t0:
            assert r['out'].pos_ned is None and not r['out'].commands, 'output after release at t=%.2f' % r['t']


def crossing(world, log, f_plane, sign):
    """First logged position where the start-frame forward coordinate crosses f_plane."""
    prev = None
    for r in log:
        f = world.flu(r['p'])[0]
        if prev is not None and (prev - f_plane) * sign < 0 <= (f - f_plane) * sign:
            return world.flu(r['p']), r
        prev = f
    return None, None


# ------------------------------------------------------------------- tests
def _check_full_run(scn):
    m, px4, world, log = run(scn)
    seq = phase_seq(log)
    assert seq == list(PHASES), 'phase sequence %s' % seq
    assert not px4.armed and px4.landed
    home = np.array(scn.home_ned, float)
    assert np.linalg.norm(px4.p[:2] - home[:2]) < 0.15, 'landed %.3f m from home' % np.linalg.norm(px4.p[:2] - home[:2])
    assert_no_collision(world, log)
    assert_setpoints_inside_fence(world, log)
    # through the tunnel mouth near its axis
    q, _ = crossing(world, log, TRUE['tunnel_entry'][0], +1)
    assert q is not None and abs(q[1] - TRUE['tunnel_entry'][1]) < 0.12 and abs(q[2] - TRUE['tunnel_entry'][2]) < 0.12, q
    # through the window opening (travel is backwards in f)
    q, _ = crossing(world, log, TRUE['window_center'][0], -1)
    assert q is not None and abs(q[1] - TRUE['window_center'][1]) < TRUE['window_w'] / 2 - HALF_SPAN
    assert abs(q[2] - TRUE['window_center'][2]) < TRUE['window_h'] / 2 - HALF_HEIGHT
    # disarm on the rock: only with ToF ~0 and PX4 landed, never forced
    disarms = [c for c in px4.cmds if c[1].name == 'disarm']
    assert disarms and all(c[1].param2 == 0.0 for c in disarms)
    t_d, _, armed, landed, tof = disarms[0]
    assert armed and landed and tof <= Params().tof_touch_m, disarms[0]
    r = first(log, lambda r: r['t'] >= t_d)
    f, l, u = world.flu(r['p'])
    assert abs(f - TRUE['rock_center'][0]) < 0.15 and abs(l - TRUE['rock_center'][1]) < 0.12, (f, l)
    assert abs(u - TRUE['rock_top']) < 0.02
    # over the wall with margin
    over = [world.flu(r['p'])[2] for r in log if r['phase'] == 'climb_over_wall'
            and TRUE['wall_l'][0] - 0.3 <= world.flu(r['p'])[1] <= TRUE['wall_l'][1] + 0.3]
    assert over and min(over) > TRUE['wall_top'] + 0.3, over[:3]
    # nothing published after done
    t_done = first(log, lambda r: r['phase'] == 'done')['t']
    assert_released_forever(log, t_done)
    flight = m.history[-1][0] - m.t_arm
    assert flight < 90.0, 'mission took %.1f s' % flight
    return m, log, flight


def test_full_run():
    m, log, flight = _check_full_run(Scenario())
    print('  full run: %.1f s from arming to done' % flight)


def test_full_run_rotated_start():
    m, log, flight = _check_full_run(Scenario(name='rotated', home_ned=(0.7, -0.4, -0.05), home_yaw=0.7, seed=7))
    print('  rotated start (yaw 40 deg): %.1f s' % flight)


def test_start_phase_per_stage():
    """Per-stage harness (compat): Params.start_phase + start_pose.  The course start frame sits somewhere in NED
    (home_ned / home_yaw); the vehicle arms at stage_start_pose(prior, phase), facing the stage.  The mission rebuilds
    the start frame from that pose, climbs straight up at the spawn and starts IN the stage (climb_over_wall: from the
    ground, as after the re-arm on the pad); optional stages the prior lacks are skipped like in a full run; no
    start-zone rules away from the corridor."""
    home, hy = (0.9, -1.3, -0.02), 0.6
    P = CoursePrior.from_dict(prior_dict())
    cases = [('tunnel_align', 'tunnel_traverse', 'tunnel_align'),
             ('chicane', 'window_traverse', 'window_align'),          # the default prior has no chicane
             ('window_align', '', 'window_align'),
             ('landing_route', 'land_target', 'land_target'),          # ... and no landing route
             ('climb_over_wall', '', 'climb_over_wall'),
             ('return_home', '', 'return_home')]
    out = []
    for phase, stop, first_ph in cases:
        sp = stage_start_pose(P, phase)
        scn = Scenario(name='stage_' + phase, home_ned=home, home_yaw=hy, spawn=sp, t_max=150.0)
        m, px4, world, log = run(scn, Params(start_phase=phase, start_pose=sp, stop_after=stop))
        seq = phase_seq(log)
        lead = ['wait', 'arm'] + ([] if phase == 'climb_over_wall' else ['takeoff'])
        assert seq[:len(lead) + 1] == lead + [first_ph], (phase, seq)
        assert m.phase == 'done' and m.abort_reason is None, (phase, m.phase, m.sub, m.abort_reason, seq)
        assert np.linalg.norm(m.sf.o[:2] - world.sf.o[:2]) < 0.01 and abs(wrap(m.sf.yaw - world.sf.yaw)) < 1e-3, \
            (phase, m.sf.o, world.sf.o, m.sf.yaw, world.sf.yaw)
        assert abs(m.home[2] - world.sf.o[2]) < 0.01, (phase, m.home, world.sf.o)
        tk = [r for r in log if r['phase'] == 'takeoff']
        spn = world.sf.to_ned((sp[0], sp[1], 0.0))
        assert all(np.linalg.norm(r['p'][:2] - spn[:2]) < 0.10 for r in tk), phase      # straight up at the spawn
        assert not events(log, 'start_rear_clip') and not events(log, 'start_rear_free'), phase
        ev = events(log, 'start_phase')
        assert ev and ev[0]['first'] == first_ph and np.allclose(ev[0]['spawn_start_frame'][:2], sp[:2], atol=0.01), ev
        assert_no_collision(world, log)
        assert_setpoints_inside_fence(world, log)
        if stop == '':                                                   # ran on to the end: landed at home
            assert np.linalg.norm(px4.p[:2] - world.sf.o[:2]) < 0.15, (phase, world.flu(px4.p))
        out.append('%s %s %.0f s' % (phase, '->'.join(seq[len(lead):-1]), log[-1]['t']))
    try:
        Mission(P, Params(start_phase='gate_pass'))
        raise AssertionError('gate_pass without gate_b accepted')
    except ValueError:
        pass
    try:
        Mission(P, Params(start_phase='tunnel_traverse'))
        raise AssertionError('tunnel_traverse accepted')
    except ValueError:
        pass
    print('  start_phase (course frame at yaw 34 deg, spawn per stage_start_pose): ' + '; '.join(out))


def test_pad_approach_land_target():
    """Params.pad_approach (perception/pad_approach.py): the level camera sees the flat pad only from low and close.
    The prior pad is 0.36 m off the truth.  Observe pose 1.5 m out -> 0.40 m camera height -> lock -> creep ->
    the last < 1 m on VIO -> the ToF-gated descent on the PAD (not the prior).  Full course and a land_target start
    (pilot mode too).  Pad never detected: yaw scan, then abort-land (pilot mode: the pilot lands)."""
    out = []
    pad = np.array(TRUE['rock_center'], float)
    P = CoursePrior.from_dict(prior_dict())
    sp = stage_start_pose(P, 'land_target')
    for name, scn, prm in (
            ('full', Scenario(name='pad_full', pad_cam=True, seed=3, t_max=200.0), Params(pad_approach=True)),
            ('stage', Scenario(name='pad_stage', pad_cam=True, spawn=sp, seed=5),
             Params(pad_approach=True, start_phase='land_target', start_pose=sp, stop_after='land_target')),
            ('pilot', Scenario(name='pad_pilot', pad_cam=True, spawn=sp, pilot_mode=True, seed=6),
             Params(pilot_mode=True, pad_approach=True, start_phase='land_target', start_pose=sp,
                    stop_after='land_target'))):
        m, px4, world, log = run(scn, prm)
        assert m.abort_reason is None, (name, m.abort_reason, phase_seq(log))
        kinds = [e['kind'] for e in events(log)]
        assert 'pad_locked' in kinds and 'pad_approach_start' in kinds, (name, kinds)
        lat = [e for e in events(log, 'latched') if e.get('source') == 'pad_approach']
        assert lat and np.linalg.norm(world.flu(np.array(lat[0]['center']))[:2] - pad) < 0.08, (name, lat)
        land = [r for r in log if r['phase'] == 'land_target' and r['landed']] or \
            [r for r in log if r['phase'] in ('disarm', 'rearm', 'staged_stop') and r['landed']]
        assert land, name
        err = float(np.linalg.norm(world.flu(land[0]['p'])[:2] - pad))
        assert err < 0.10, (name, err)
        low = min(world.flu(r['p'])[2] for r in log if r['phase'] == 'land_target' and not r['landed'] and
                  float(np.linalg.norm(world.flu(r['p'])[:2] - pad)) > 1.0)
        assert low < 0.55, (name, low)                       # it did go low before closing in
        assert_no_collision(world, log)
        if name == 'full':
            assert m.phase == 'done' and phase_seq(log)[-1] == 'done', phase_seq(log)
        if name == 'pilot':
            assert events(log, 'pilot_land_requested'), name
        out.append('%s %.3f m from the pad centre (prior 0.36 m off)' % (name, err))
    for pilot in (False, True):                              # never detected
        scn = Scenario(name='pad_none', pad_cam=True, spawn=sp, landing_miss_p=1.0, pilot_mode=pilot, seed=7)
        m, px4, world, log = run(scn, Params(pilot_mode=pilot, pad_approach=True, start_phase='land_target',
                                             start_pose=sp, stop_after='land_target'))
        assert [e for e in events(log, 'pad_phase') if e.get('pad_phase') == 'scan'], pilot
        assert events(log, 'pad_handoff'), pilot
        if pilot:
            assert m.abort_reason is None and events(log, 'pilot_land_requested'), (m.abort_reason, phase_seq(log))
        else:
            assert m.abort_reason == 'pad_not_found', m.abort_reason
        assert_no_collision(world, log)
    out.append('not found: scan -> abort-land / pilot landing')
    print('  pad_approach: ' + '; '.join(out))


def test_corridor_obstacle_side_step():
    pillar = ((0.85, -0.15, 0.0), (1.05, 0.15, 2.2), 'pillar')
    m, px4, world, log = run(Scenario(name='pillar', extra_boxes=(pillar,)))
    assert phase_seq(log) == list(PHASES), phase_seq(log)
    av = events(log, 'avoid')
    assert av and av[0]['phase'] == 'corridor', av
    lat = [abs(world.flu(r['p'])[1]) for r in log if r['phase'] == 'corridor']
    assert max(lat) > 0.45, max(lat)
    assert_no_collision(world, log)
    # never closer than the vehicle half span to the pillar
    lo, hi = np.array(pillar[0]), np.array(pillar[1])
    dmin = min(np.linalg.norm(np.maximum(np.maximum(lo - world.flu(r['p']), 0), world.flu(r['p']) - hi)) for r in log
               if r['phase'] == 'corridor')
    assert dmin > HALF_SPAN + 0.05, dmin
    print('  pillar: side step %.2f m, min distance %.2f m' % (max(lat), dmin))


def test_vertex_moves_only_into_seen_space():
    """Gazebo v4J eyeball s2 (7d841286): at the corridor start, with W1 not yet in the map, wall-relative centring had
    only the west net to go by and pushed the first corner 0.6 m away from it -- past W1's real face; the first leg
    then ran diagonally into W1 (kept W1 as a 'cluster', sidestepped toward it, 10 contacts).  A vertex now moves only
    to points seen free (camera view within seen_free_s / own track).  Checked on the centring itself: no map points,
    the first corner 0.5 m from the north net; target region unseen -> stays; seen -> may move."""
    def follower(seen):
        f = Follower(np.array([0.0, 0.0, -0.8]), [np.array([1.635, 0.446, -0.8]), np.array([1.635, 1.61, -0.8]),
                                                   np.array([0.3, 1.61, -0.8])], Params(), 0.5, avoid='lateral',
                     gap_snap=True)
        f.snap = True
        f.boards_xy = lambda lo, hi, z: np.zeros((0, 2))
        f.fence_dist = lambda q: float(min(2.134 - q[0], 0.53 + q[1], q[0] + 3.962, 5.566 - q[1]))
        f.valid_pt = lambda q: f.fence_dist(q) >= Params().fence_margin_m
        f.seen_free = lambda q: seen(q)
        p0 = f.P[1].copy()
        f._snap_vertices(10.0, 0.0)
        return p0, f.P[1].copy()
    p0, p1 = follower(lambda q: False)
    assert np.allclose(p0, p1), (p0, p1)                         # unseen: the prior point stays
    p0, p2 = follower(lambda q: True)
    moved = float(np.linalg.norm((p2 - p0)[:2]))
    assert moved > 0.05, (p0, p2)                                 # (old behaviour: pushed off the net blind)
    p0, p3 = follower(lambda q: float(q[1]) < 0.6)                # only the near half was seen
    assert float(p3[1]) < 0.6 + 1e-6, p3
    print('  centring with no map points: unseen -> the corner stays; all seen -> moved %.2f m (to %s); half seen -> '
          'stays within it' % (moved, np.round(p2[:2], 2).tolist()))


def test_vio_jump_no_chase():
    """v4J eyeball s3 / v4K map s1 / the real drone (a 38 m VIO jump after an impact, then the mission resumed and
    chased it): an estimate jump the velocity does not explain -> hover with the setpoint ON the new estimate (the
    airframe stays put), follower frozen; 0.4 m: recovered after >= 1.5 s, held vio_quarantine_s, then on; 1.0 m (>
    vio_jump_abort_m): land in place.  (Old: the carrot stays on the path -> PX4 flies the sideways jump back.)"""
    def fly(over, jump):
        scn = Scenario(name='vio_jump', est_drift=(6.0, 6.05, (0.0, -jump, 0.0)), t_max=60)  # (sideways; the fake
        m, px4, world, log = run(scn, Params(stop_after='corridor', **over))                  # detectors: truth)
        at = [r for r in log if 5.9 <= r['t'] <= 6.0][-1]
        p0 = world.flu(at['p'])[:2]
        win = [r for r in log if 6.0 <= r['t'] <= 7.5 and r['armed']]
        moved = max(float(np.linalg.norm(world.flu(r['p'])[:2] - p0)) for r in win)
        return m, log, world, p0, moved
    m, log, world, p0, moved = fly({}, 0.4)
    ev = [e['kind'] for e in events(log)]
    assert 'vio_fault' in ev and 'vio_recovered' in ev and events(log, 'vio_fault')[0]['why'] == 'jump', ev
    assert moved < 0.15, moved
    assert_no_collision(world, log)
    _, _, _, _, moved0 = fly(dict(vio_guard=False), 0.4)
    assert moved0 > 0.3, moved0
    m2, log2, world2, p02, moved2 = fly({}, 1.0)
    assert m2.abort_reason == 'vio_fault_jump', (m2.abort_reason, phase_seq(log2))
    td = world2.flu(log2[-1]['p'])[:2]
    assert float(np.linalg.norm(td - p02)) < 0.25 and moved2 < 0.15, (td, p02, moved2)
    assert_no_collision(world2, log2)
    print('  VIO jump 0.4 m: hovered on the estimate (airframe moved %.2f m), recovered, went on; old: chased %.2f m; '
          '1.0 m: landed in place (%.2f m from where it jumped)' % (moved, moved0, float(np.linalg.norm(td - p02))))


def test_abort_landing_clears_wall_feet():
    """Gazebo v4J map s3 (7d841286): corridor_blocked at the U-turn round W1's north end, then the abort landing came
    straight down 0.26 m from the cinder block that weighs W1's A-frame there (blocks / A-frame feet stand on the
    floor beside the walls, too low to tell from the floor in the map).  A non-emergency abort (blocked / timeout)
    first slides back along the flown path (<= 1 m, <= 0.2 m/s) until land_clear_m (0.45) from prior wall lines and
    measured boards, then lands.  Here: blocked by a wall across the corridor beside a short prior wall line."""
    wall = ((1.9, -5.0, 0.0), (2.1, 0.6, 2.7), 'barrier')
    seg = (np.array([1.2, -0.25]), np.array([1.6, -0.25]))
    out = {}
    for label, over in (('new', {}), ('old', dict(land_clear_m=0.0))):
        pr = prior_dict(walls=[dict(**{'from': dict(forward=1.2, left=-0.25), 'to': dict(forward=1.6, left=-0.25)})],
                        corridor=[dict(forward=2.6, left=0.0, up=1.0)])
        m, px4, world, log = run(Scenario(name='blocked_feet', prior=pr, extra_boxes=(wall,), t_max=70),
                                 Params(**over))
        td = [r for r in log if r['phase'] == 'abort_land']
        q = world.flu(td[-1]['p'])[:2]
        d = _seg_d(q, *seg)[0]
        out[label] = (m, log, world, d)
    m, log, world, d = out['new']
    assert m.abort_reason == 'corridor_blocked' and events(log, 'abort_land_moved'), [e['kind'] for e in events(log)]
    assert d >= Params().land_clear_m - 0.03, d
    assert_no_collision(world, log)
    d0 = out['old'][3]
    assert d0 < 0.4 and not events(out['old'][1], 'abort_land_moved'), d0
    print('  blocked beside a prior wall line: abort landing slid back, touchdown %.2f m from it (old %.2f m)' % (d, d0))


def test_corridor_fully_blocked_hovers_then_lands():
    wall = ((0.9, -5.0, 0.0), (1.1, 0.6, 2.7), 'barrier')
    m, px4, world, log = run(Scenario(name='barrier', extra_boxes=(wall,), t_max=60))
    assert m.abort_reason == 'corridor_blocked', m.abort_reason
    assert phase_seq(log)[-2:] == ['abort_land', 'aborted']
    blk = events(log, 'blocked')
    t_abort = events(log, 'abort')[0]['t']
    assert blk and t_abort - blk[0]['t'] >= Params().blocked_timeout_s
    # hovering while waiting: slow in the last 3 s before the abort
    sp = [np.linalg.norm(r['v']) for r in log if t_abort - 3.0 <= r['t'] < t_abort]
    assert max(sp) < 0.15, max(sp)
    lo = np.array(wall[0])
    fmax = max(world.flu(r['p'])[0] for r in log)
    assert fmax < lo[0] - HALF_SPAN - 0.05, fmax
    land = [c for c in px4.cmds if c[1].name == 'land']
    assert land and land[0][0] >= t_abort
    after = [r for r in log if r['nav'] == NAV_AUTO_LAND]
    assert after and all(r['out'].pos_ned is None for r in after)
    print('  barrier: hovered %.1f s at f=%.2f, then LAND' % (t_abort - blk[0]['t'], fmax))


def test_tunnel_unstable_detection_waits():
    scn = Scenario(name='tunnel_jitter', tunnel_jitter_s=14.0, seed=3)   # (slow approach: still unstable on arrival)
    m, px4, world, log = run(scn)
    assert phase_seq(log) == list(PHASES), phase_seq(log)
    t_seen = world.tunnel_seen_t
    rows = [r for r in log if r['phase'] == 'tunnel_align']
    t_latch = [e for e in events(log, 'latched') if e['detector'] == 'tunnel'][0]['t']
    assert t_latch >= t_seen + scn.tunnel_jitter_s, (t_latch, t_seen)
    hover = [r for r in rows if r['sub'] == 'detect']
    assert hover and hover[-1]['t'] - hover[0]['t'] > 2.0
    sps = np.array([r['out'].pos_ned for r in hover])
    assert np.ptp(sps, axis=0).max() < 1e-9, 'hover setpoint must not move while detections are unstable'
    stand = world.flu(sps[0])
    drift = max(np.linalg.norm(world.flu(r['p']) - stand) for r in hover if r['t'] >= hover[0]['t'] + 1.5)
    assert drift < 0.10, drift
    assert max(world.flu(r['p'])[0] for r in hover) < stand[0] + 0.15
    t_trav = first(log, lambda r: r['phase'] == 'tunnel_traverse')['t']
    assert t_trav > t_latch
    print('  tunnel jitter: hovered %.1f s (drift %.2f m), latched %.1f s after first sight'
          % (hover[-1]['t'] - hover[0]['t'], drift, t_latch - t_seen))


def test_tunnel_prior_error_and_hoop_jump():
    # prior entry 0.5 m short / long along the axis, with the near-range hoop jump and 2-3 m partial recall
    for df in (-0.5, 0.5):
        pr = prior_dict()
        pr['tunnel'] = dict(pr['tunnel'], entry=dict(forward=TRUE['tunnel_entry'][0] + df, left=0.2, up=0.6))
        m, px4, world, log = run(Scenario(name='tunnel_prior_%+.1f' % df, prior=pr, seed=11))
        assert phase_seq(log) == list(PHASES), (df, phase_seq(log))
        lat = [e for e in events(log, 'latched') if e['detector'] == 'tunnel'][0]
        f_lat = world.flu(np.array(lat['center']))[0]
        assert abs(f_lat - TRUE['tunnel_entry'][0]) < 0.1, 'latched hoop %.2f, first hoop %.2f' % (f_lat, TRUE['tunnel_entry'][0])
        assert_no_collision(world, log)
        hover = [r for r in log if r['phase'] == 'tunnel_align' and r['sub'] in ('approach', 'detect')]
        cam_min = min(TRUE['tunnel_entry'][0] - world.flu(r['p'])[0] - 0.116 for r in hover if np.linalg.norm(r['v']) < 0.35)
        print('  tunnel prior %+.1f m: latched first hoop (err %.3f m) after %.1f s in tunnel_align, closest slow camera %.2f m'
              % (df, f_lat - TRUE['tunnel_entry'][0], lat['t'] - first(log, lambda r: r['phase'] == 'tunnel_align')['t'], cam_min))


def test_window_clearance_refused_lands():
    m, px4, world, log = run(Scenario(name='window_blocked', window_clearance_ok=False, t_max=90))
    assert m.abort_reason == 'window_clearance_refused', m.abort_reason
    assert 'window_traverse' not in phase_seq(log)
    fmin = min(world.flu(r['p'])[0] for r in log if r['t'] > 10 and r['phase'] in ('window_align', 'abort_land', 'aborted'))
    assert fmin > TRUE['window_center'][0] + 0.5, fmin
    assert any(c[1].name == 'land' for c in px4.cmds)
    assert phase_seq(log)[-1] == 'aborted' and not px4.armed
    assert all(r['out'].pos_ned is None for r in log if r['nav'] == NAV_AUTO_LAND)
    print('  window clearance_ok=False: refused, closest approach f=%.2f (window at %.2f)'
          % (fmin, TRUE['window_center'][0]))


def test_tof_gates_disarm():
    # (a) ToF never reads ~0 on the rock although PX4 says landed -> no disarm, timeout -> LAND
    m, px4, world, log = run(Scenario(name='tof_bias', tof_bias_m=0.3, t_max=120))
    assert m.abort_reason == 'no_touchdown_confirmation_on_target', m.abort_reason
    landed_rows = [r for r in log if r['phase'] == 'land_target' and r['sub'] == 'descend' and r['landed']]
    assert len(landed_rows) * DT > 3.0
    assert phase_seq(log)[-1] == 'aborted' and not px4.armed
    mission_disarms = [c for c in px4.cmds if c[1].name == 'disarm']
    assert not mission_disarms, mission_disarms
    assert any(c[1].name == 'land' for c in px4.cmds)
    # (b) ToF ~0 immediately but PX4's land detector is 2 s late -> disarm waits for it
    m2, px4b, world2, log2 = run(Scenario(name='landed_late', landed_delay_s=2.0))
    assert phase_seq(log2) == list(PHASES), phase_seq(log2)
    touch = first(log2, lambda r: r['phase'] == 'land_target' and r['sub'] == 'descend' and r['tof'] <= 0.02)
    t_d, _, armed, landed, tof = [c for c in px4b.cmds if c[1].name == 'disarm'][0]
    assert t_d - touch['t'] >= 2.0 and landed and tof <= Params().tof_touch_m, (t_d, touch['t'])
    print('  ToF bias 0.3 m: no disarm, landed %.1f s then LAND; late land detector: disarm %.1f s after contact'
          % (len(landed_rows) * DT, t_d - touch['t']))


def test_pose_unhealthy():
    # a 0.6 s glitch is ridden out (hold), a 2 s outage lands 1 s after it starts
    m0, _, _, log0 = run(Scenario(name='nominal_ref'))
    t_c = first(log0, lambda r: r['phase'] == 'corridor')['t'] + 0.5
    _, _, _, log1 = run(Scenario(name='glitch_ref', pose_bad=((t_c, t_c + 0.6),), t_max=120))   # (the glitch adds the
    t_w = first(log1, lambda r: r['phase'] == 'window_align')['t'] + 1.0                       # VIO-fault hold)
    scn = Scenario(name='pose_bad', pose_bad=((t_c, t_c + 0.6), (t_w, t_w + 3.0)), t_max=120)
    m, px4, world, log = run(scn)
    ab = events(log, 'abort')
    assert len(ab) == 1 and ab[0]['reason'] == 'pose_unhealthy', ab
    assert t_w + 1.0 <= ab[0]['t'] <= t_w + 1.0 + 2 * DT, (ab[0]['t'], t_w)
    land = [c for c in px4.cmds if c[1].name == 'land']
    assert land and land[0][0] == ab[0]['t'], 'LAND must go out immediately (no brake) when the pose is bad'
    glitch = [r for r in log if t_c <= r['t'] < t_c + 0.6]
    sps = np.array([r['out'].pos_ned for r in glitch])
    assert np.ptp(sps, axis=0).max() < 1e-9, 'setpoint must freeze during a pose glitch'
    assert 'tunnel_traverse' in phase_seq(log)          # survived the glitch
    assert phase_seq(log)[-1] == 'aborted'
    assert all(r['out'].pos_ned is None for r in log if r['nav'] == NAV_AUTO_LAND)
    print('  pose: 0.6 s glitch held, outage at %.1f s -> LAND at %.2f s' % (t_w, ab[0]['t']))


def test_px4_takeover_releases():
    m0, _, _, log0 = run(Scenario(name='nominal_ref'))
    t_k = first(log0, lambda r: r['phase'] == 'corridor')['t'] + 1.0
    m, px4, world, log = run(Scenario(name='takeover', takeover_t=t_k, t_max=t_k + 10))
    assert m.phase == 'released' and m.release_reason.startswith('left_offboard'), (m.phase, m.release_reason)
    t_rel = events(log, 'release')[0]['t']
    assert t_rel <= t_k + 2 * DT
    assert_released_forever(log, t_rel)
    print('  takeover at %.2f s -> released at %.2f s, nothing published after' % (t_k, t_rel))


def test_px4_auto_land_stops_setpoints():
    # PX4 failsafe to AUTO_LAND on its own: treat like leaving Offboard
    m0, _, _, log0 = run(Scenario(name='nominal_ref'))
    t_k = first(log0, lambda r: r['phase'] == 'tunnel_align')['t'] + 0.5

    class AutoLandPX4(FakePX4):
        def integrate(self, t, dt):
            if t >= t_k and self.nav == NAV_OFFBOARD:
                self.nav = NAV_AUTO_LAND
            super().integrate(t, dt)

    world = World(Scenario(t_max=t_k + 8))
    px4 = AutoLandPX4(world)
    m = Mission(CoursePrior.from_dict(world.scn.prior))
    t = 0.0
    seen = []
    while t < world.scn.t_max:
        out = m.step(t, px4.vehicle(t), world.perceive(t, px4))
        px4.apply(out, t)
        seen.append((t, out))
        px4.integrate(t, DT)
        t = round(t + DT, 6)
    assert m.phase == 'released'
    assert all(o.pos_ned is None and not o.commands for tt, o in seen if tt > t_k + DT)


def test_geofence_stop_and_land():
    m0, _, _, log0 = run(Scenario(name='nominal_ref'))
    t_g = first(log0, lambda r: r['phase'] == 'corridor')['t'] + 0.3
    scn = Scenario(name='fence', wind=(t_g, t_g + 2.5, (0.0, 1.6, 0.0)), t_max=60)   # gust toward +left
    m, px4, world, log = run(scn)
    assert m.abort_reason == 'geofence', m.abort_reason
    ab = events(log, 'abort')[0]
    out_rows = [r for r in log if world.flu(r['p'])[1] > FENCE['left_max'] - Params().fence_inset_m]
    assert out_rows and ab['t'] <= out_rows[0]['t'] + DT, (ab['t'], out_rows[0]['t'])
    # stop first (hold setpoint), then LAND
    brake = [r for r in log if r['phase'] == 'abort_land' and r['sub'] == 'brake']
    assert brake and np.ptp(np.array([r['out'].pos_ned for r in brake]), axis=0).max() < 1e-9
    land = [c for c in px4.cmds if c[1].name == 'land']
    assert land and land[0][0] - ab['t'] >= Params().brake_s - DT
    assert phase_seq(log)[-1] == 'aborted'
    assert all(r['out'].pos_ned is None for r in log if r['nav'] == NAV_AUTO_LAND)
    print('  fence: tripped at l=%.2f (fence %.2f), brake %.2f s then LAND'
          % (world.flu(out_rows[0]['p'])[1], FENCE['left_max'], land[0][0] - ab['t']))


def test_wall_measurement_raises_climb():
    pr = prior_dict()
    pr['wall'] = dict(pr['wall'], top_up=1.2)               # prior says 1.2 m, the wall is 1.7 m
    m, px4, world, log = run(Scenario(name='wall_measured', prior=pr, wall_measured_top=1.75))
    assert phase_seq(log) == list(PHASES)
    assert_no_collision(world, log)
    over = [world.flu(r['p'])[2] for r in log if r['phase'] == 'climb_over_wall'
            and TRUE['wall_l'][0] - 0.3 <= world.flu(r['p'])[1] <= TRUE['wall_l'][1] + 0.3]
    assert min(over) > 1.75 + 0.3, min(over)


def test_mission_time_limit():
    m, px4, world, log = run(Scenario(name='time_limit', t_max=40), Params(max_mission_s=8.0))
    assert m.abort_reason == 'mission_time_limit', m.abort_reason
    ab = events(log, 'abort')[0]
    assert abs(ab['t'] - m.t_arm - 8.0) <= 2 * DT
    assert phase_seq(log)[-1] == 'aborted' and not px4.armed


def test_missing_obstacle_map_caps_speed():
    m, px4, world, log = run(Scenario(name='no_map', no_map=True, t_max=60))
    assert events(log, 'obstacle_map_missing_or_stale')
    v = [np.linalg.norm(r['v'][:2]) for r in log if r['phase'] == 'corridor']
    assert max(v) < Params().v_no_map + 0.1, max(v)


def test_px4_land_detector_needs_commanded_descent():
    """Bug B (Ubuntu official-compat sim, PX4 1.17): ground contact needs velocity[2] >= 1.1*LNDMC_Z_VEL_MAX."""
    p = Params()
    assert p.v_touch >= 1.1 * p.px4_lndmc_z_vel_max + 0.05
    d = Descent((0.0, 0.0), -0.5, surface_z=0.0)
    for k in range(60):                                # vehicle sits on the surface (z = 0) the whole time
        sp, _ = d.step(0.05, k * 0.05, np.array([0.0, 0.0, 0.0]), False, 0.0, p)
        assert sp[1][2] >= 1.1 * p.px4_lndmc_z_vel_max, 'feed-forward dropped to %.2f at step %d' % (sp[1][2], k)
    assert abs(sp[0][2] - p.push_contact_m) < 1e-9     # setpoint capped below the vehicle, ff still on
    try:
        Mission(CoursePrior.from_dict(prior_dict()), Params(v_touch=0.2))
    except ValueError:
        pass
    else:
        raise AssertionError('v_touch below the PX4 land-detector threshold accepted')
    # the old behaviour, reproduced: commanded descent below PX4's threshold and no hand-off -> PX4 never says
    # landed -> no_touchdown_confirmation (what the compat sim saw with v_touch 0.2)
    old = Params(v_touch=0.2, px4_lndmc_z_vel_max=0.1, contact_handoff_s=None)
    m, px4, world, log = run(Scenario(name='old_descent', t_max=80), old)
    assert m.abort_reason == 'no_touchdown_confirmation_on_target', m.abort_reason
    assert not any(r['landed'] for r in log if r['phase'] == 'land_target' and r['sub'] == 'descend')
    # defaults on the same PX4 model: both landings detected in Offboard, no hand-off needed
    m, px4, world, log = run(Scenario(name='new_descent'))
    assert phase_seq(log) == list(PHASES) and not events(log, 'touchdown_handoff_to_px4')
    print('  PX4 land detector: old descent never detected (abort), defaults land in Offboard')


def test_touchdown_handoff_to_px4_land():
    """PX4 whose Offboard land detection never fires: after ToF contact the mission hands touchdown to LAND."""
    m, px4, world, log = run(Scenario(name='handoff', lndmc_offboard=10.0))
    assert phase_seq(log) == list(PHASES), phase_seq(log)
    ho = events(log, 'touchdown_handoff_to_px4')
    assert [e['phase'] for e in ho] == ['land_target', 'land_home'], ho
    lands = [c for c in px4.cmds if c[1].name == 'land']
    assert lands and all(tof <= Params().tof_touch_m and armed for _, _, armed, _, tof in lands), lands
    assert all(e['contact_s'] >= Params().contact_handoff_s for e in ho)
    # armed in AUTO_LAND (PX4 finishing the touchdown): silent.  Once PX4 has disarmed itself the mode stays
    # AUTO_LAND and the mission streams again so it can switch back to Offboard to re-arm.
    silent = [r for r in log if r['nav'] == NAV_AUTO_LAND and r['armed']]
    assert silent and all(r['out'].pos_ned is None and not r['out'].commands for r in silent)
    assert not events(log, 'abort') and not events(log, 'release')
    r = first(log, lambda r: r['phase'] == 'disarm')
    f, l, u = world.flu(r['p'])
    assert abs(u - TRUE['rock_top']) < 0.02 and abs(f - TRUE['rock_center'][0]) < 0.15
    assert not px4.armed and np.linalg.norm(px4.p[:2]) < 0.15
    print('  hand-off: PX4 never lands in Offboard -> LAND after %.1f s of ToF contact, both landings OK'
          % ho[0]['contact_s'])


def test_tunnel_near_net_standoff_shrinks():
    """Bug C (replica): tunnel entry 1.95 m from the net behind it; 1.8 m standoff would sit in the fence margin."""
    entry = (FENCE['forward_min'] + 1.95, 0.0, 0.62)
    pr = prior_dict(corridor=[])
    pr['tunnel'] = dict(pr['tunnel'], entry=dict(forward=entry[0], left=0.0, up=0.6))
    m, px4, world, log = run(Scenario(name='tunnel_near_net', prior=pr, true_over={'tunnel_entry': entry}))
    sh = events(log, 'standoff_shrunk')
    assert sh and sh[0]['stage'] == 'tunnel' and Params().standoff_min_m <= sh[0]['standoff_m'] < 1.8, sh
    assert not events(log, 'standoff_clipped')
    assert phase_seq(log) == list(PHASES), phase_seq(log)
    lat = [e for e in events(log, 'latched') if e['detector'] == 'tunnel'][0]
    assert abs(world.flu(np.array(lat['center']))[0] - entry[0]) < 0.1
    t_align = first(log, lambda r: r['phase'] == 'tunnel_align')['t']
    assert lat['t'] - t_align < 3.0, 'tunnel took %.1f s to latch' % (lat['t'] - t_align)
    assert_no_collision(world, log)
    assert_setpoints_inside_fence(world, log)
    # unit: the window standoff shrinks too, and is clipped (with an event) when even standoff_min_m does not fit
    for fmax, kind in ((5.5, 'standoff_shrunk'), (5.2, 'standoff_clipped')):
        pw = prior_dict(fence=dict(FENCE, forward_max=fmax))
        mw = Mission(CoursePrior.from_dict(pw))
        mw.now, mw._events = 0.0, []
        mw.veh = Vehicle(pos_ned=(0.0, 0.0, 0.0), heading=0.0, armed=True, nav_state=NAV_OFFBOARD)
        mw.pos = np.zeros(3)
        mw._record_home()
        q = mw._standoff('window')
        ev = [e for e in mw._events if e['kind'] == kind]
        assert ev and ev[0]['stage'] == 'window', (fmax, mw._events)
        assert mw._valid(q), (fmax, q)
        assert np.allclose(mw._standoff('window'), q)        # cached: one reference for flight and zone
    print('  tunnel 1.95 m from the net: standoff shrunk to %.2f m, latched %.1f s after tunnel_align'
          % (sh[0]['standoff_m'], lat['t'] - t_align))


def test_minor_phantom_cluster_slows_not_side_steps():
    """Floating phantoms (vertical support rule, user-approved): a 5-point cluster on the path that does not span
    0.3 m vertically and stands on nothing is a phantom candidate -- it only slows us down (no side step) and is
    carved away as we come closer; small clusters WITH vertical support (board ends) stay real (separate test)."""
    ph = ((0.95, 0.0, 1.0), 5, 0.6)
    m, px4, world, log = run(Scenario(name='minor_phantom', phantoms=(ph,)))
    assert phase_seq(log) == list(PHASES), phase_seq(log)
    kinds = [e['kind'] for e in events(log) if e['phase'] == 'corridor']
    assert 'floating_cluster' in kinds and 'small_cluster_kept' not in kinds and world.vanished, kinds
    assert not [e for e in events(log, 'avoid') if e['phase'] == 'corridor' and e['offset_m'] != 0.0], kinds
    v_near = max(np.linalg.norm(r['v'][:2]) for r in log if r['phase'] == 'corridor' and r['t'] < world.vanished[0])
    assert v_near < Params().v_minor + 0.15, v_near
    assert_no_collision(world, log)
    print('  floating 5-point phantom on the path: no vertical support -> slowed to %.2f m/s, no side step, carved'
          % v_near)


def test_persistent_small_cluster_never_side_steps():
    """Round 5b seed 7 (3-4 phantom voxels 0.65 m ahead -> a blind 1.2 m side step into a panel): a tiny FLOATING
    cluster that survives the nudge is ignored and passed slowly -- no blocked / look / side step."""
    out = []
    for n_pts in (4, 6):
        ph = ((0.65 + 0.116, 0.0, 1.0), n_pts, None)          # never carved
        m, px4, world, log = run(Scenario(name='persistent_small', phantoms=(ph,)))
        assert phase_seq(log) == list(PHASES), (phase_seq(log), m.abort_reason)
        cor = [e for e in events(log) if e['phase'] == 'corridor']
        kinds = [e['kind'] for e in cor]
        assert 'floating_cluster' in kinds and 'minor_ignored' in kinds, kinds
        assert not [k for k in kinds if k in ('blocked', 'look_before_side_step')], kinds
        assert not [e for e in cor if e['kind'] == 'avoid' and e['offset_m'] != 0.0], cor
        lat = max(abs(world.flu(r['p'])[1]) for r in log if r['phase'] == 'corridor')
        assert lat <= Params().nudge_lat_m + 0.05, lat
        out.append('%d pts: ignored after the nudge, max lateral %.2f m' % (n_pts, lat))
    print('  persistent floating tiny cluster: ' + '; '.join(out))


def test_view_locked_phantom_carved_by_nudge():
    """A phantom that only goes away with parallax (same mismatch every frame while hovering): real until then; the
    parallax nudge clears it and we go on without a side step."""
    ph = ((0.65 + 0.116, 0.0, 1.0), 4, 'parallax', 2.4)      # appears as the corridor starts (as in seed 7)
    m, px4, world, log = run(Scenario(name='view_locked', phantoms=(ph,)))
    assert phase_seq(log) == list(PHASES), (phase_seq(log), m.abort_reason)
    cor = [e for e in events(log) if e['phase'] == 'corridor']
    kinds = [e['kind'] for e in cor]
    assert 'parallax_nudge' in kinds and world.vanished and 'minor_ignored' not in kinds, kinds
    assert 'blocked' not in kinds, kinds
    assert not [e for e in cor if e['kind'] == 'avoid' and e['offset_m'] != 0.0], cor
    print('  view-locked phantom: carved %.1f s after the nudge started'
          % (world.vanished[0] - [e for e in events(log) if e['kind'] == 'parallax_nudge'][0]['t']))


def test_side_step_into_unseen_space_looks_first():
    """Stereo crash: the side step went into a panel the horizontal camera had never seen.  Here a pillar blocks
    the path right after takeoff and a wall stands, unseen, on the preferred (left) side."""
    pillar = ((0.85, -0.15, 0.0), (1.05, 0.15, 2.2), 'pillar')
    hidden_wall = ((-0.3, 0.65, 0.0), (0.9, 0.75, 2.0), 'hidden_wall')
    pr = prior_dict(fence=dict(FENCE, left_max=1.6))          # left side steps are allowed by the fence
    m, px4, world, log = run(Scenario(name='unseen_side', prior=pr, extra_boxes=(pillar,), hidden_boxes=(hidden_wall,)))
    assert_no_collision(world, log)
    looks = events(log, 'look_before_side_step')
    assert looks, 'no look before the side step'
    assert world.revealed and world.revealed[0][1] == 'hidden_wall'   # (seen at the latest later in the course)
    av = [e for e in events(log, 'avoid') if e['phase'] == 'corridor' and e['offset_m'] != 0.0]
    assert av and av[0]['offset_m'] > 0.3, av            # ends up on the right (+), away from the wall
    assert all(e['offset_m'] >= 0.0 for e in av) or av[0]['t'] >= world.revealed[0][0], av   # never left unseen
    assert phase_seq(log) == list(PHASES), phase_seq(log)
    lmax = max(world.flu(r['p'])[1] for r in log if r['phase'] == 'corridor')
    print('  unseen side: looked %s, wall revealed at %.1f s, side step %+.2f m, max left %.2f m (wall at 0.65)'
          % ([e['offset_m'] for e in looks], world.revealed[0][0], av[0]['offset_m'], lmax))


def test_tunnel_phantoms_excluded_after_latch():
    """Stereo runs: phantoms seen through the tunnel mesh blocked the align probe (2 tunnel_align_blocked)."""
    f0, l0, u0 = TRUE['tunnel_entry']
    ph = ((f0 - 0.45, l0, u0), 30, None)                      # 30 points (not minor), on the align path
    m, px4, world, log = run(Scenario(name='tunnel_phantom', phantoms=(ph,)))
    assert phase_seq(log) == list(PHASES), (phase_seq(log), m.abort_reason)
    assert_no_collision(world, log)
    # same phantom with a map that cannot exclude: blocked (shows the exclusion is what helps) ...
    m2, _, _, log2 = run(Scenario(name='tunnel_phantom_legacy', phantoms=(ph,), legacy_map=True, t_max=60))
    assert m2.abort_reason == 'tunnel_align_blocked', m2.abort_reason
    # ... and a legacy map without phantoms still flies the whole course (duck-typed fallback)
    m3, _, world3, log3 = run(Scenario(name='legacy_map', legacy_map=True))
    assert phase_seq(log3) == list(PHASES), phase_seq(log3)
    print('  tunnel phantom: excluded after latch -> done; legacy map (no exclude) -> %s; legacy map clean -> done'
          % m2.abort_reason)


def test_landing_latch_noisy_confidence():
    """Stereo-image run: rock confidence 0.36-0.53 never gave 5 consecutive frames >= 0.5 (observe timeout)."""
    scn = Scenario(name='noisy_rock', landing_conf=(0.36, 0.53), landing_miss_p=0.25, seed=5)
    m, px4, world, log = run(scn)
    assert phase_seq(log) == list(PHASES), (phase_seq(log), m.abort_reason)
    lat = [e for e in events(log, 'latched') if e['detector'] == 'landing'][0]
    t0 = first(log, lambda r: r['phase'] == 'land_target')['t']
    m2, _, _, log2 = run(scn, Params(landing_min_conf=0.5, landing_window=5))   # the old rule
    assert m2.abort_reason and m2.abort_reason.startswith('timeout_land_target'), m2.abort_reason   # never latches
    print('  noisy rock confidence: latched %.1f s into land_target (old rule: %s)' % (lat['t'] - t0, m2.abort_reason))


def test_staged_stops():
    """Real-drone staged tests: Params.stop_after = <phase> -> hover stop_hover_s, land in place, disarm, done."""
    import json as _json
    p = Params()
    for k, v in _json.loads('{"stop_after": "tunnel_align", "stop_hover_s": 3}').items():   # mission_node.py rule
        setattr(p, k, type(getattr(p, k))(v))
    assert p.stop_after == 'tunnel_align' and p.stop_hover_s == 3.0
    try:
        Mission(CoursePrior.from_dict(prior_dict()), Params(stop_after='tunnel'))
    except ValueError:
        pass
    else:
        raise AssertionError('unknown stop_after accepted')
    f_entry, f_window = TRUE['tunnel_entry'][0], TRUE['window_center'][0]
    rows = []
    for stop in ('takeoff', 'corridor', 'tunnel_align', 'tunnel_traverse', 'window_align', 'window_traverse',
                 'land_target'):
        m, px4, world, log = run(Scenario(name='stop_' + stop, t_max=90), Params(stop_after=stop))
        seq = phase_seq(log)
        want = list(PHASES[:PHASES.index(stop) + 1]) + ['staged_stop', 'done']
        assert seq == want, (stop, seq, m.abort_reason)
        assert log[-1]['out'].sub == 'stopped_after_' + stop and m.sub == 'stopped_after_' + stop
        ev = events(log, 'staged_stop_complete')
        assert len(ev) == 1 and ev[0]['after'] == stop
        assert not px4.armed and px4.landed
        assert_no_collision(world, log)
        assert_setpoints_inside_fence(world, log)
        assert_released_forever(log, first(log, lambda r: r['phase'] == 'done')['t'])
        st = [r for r in log if r['phase'] == 'staged_stop']
        hover = [r for r in st if r['sub'] == 'hover']
        if stop == 'land_target':
            assert not hover                                   # already on the rock: disarm only
            f, l, u = world.flu(px4.p)
            assert abs(u - TRUE['rock_top']) < 0.02 and abs(f - TRUE['rock_center'][0]) < 0.15
        else:
            t_land = [r['t'] for r in st if r['sub'] != 'hover'][0]
            assert t_land - hover[0]['t'] >= Params().stop_hover_s - 1e-6, (stop, t_land - hover[0]['t'])
            sps = np.array([r['out'].pos_ned for r in hover[1:]])   # [0] = hand-off step (old phase's setpoint)
            assert np.ptp(sps, axis=0).max() < 1e-9             # then holds one point
        for t_c, c, armed, landed, tof in px4.cmds:
            if c.name == 'disarm':
                assert landed and tof <= Params().tof_touch_m and c.param2 == 0.0
        f_end = world.flu(px4.p)[0]
        if stop == 'tunnel_align':
            assert max(world.flu(r['p'])[0] for r in log) < f_entry - 0.3   # aligned, never entered
        if stop == 'window_align':
            assert min(world.flu(r['p'])[0] for r in log if r['phase'] in ('window_align', 'staged_stop')) > f_window + 0.3
        rows.append('%s %.1fs f=%.2f' % (stop, m.history[-1][0] - m.t_arm, f_end))
    # PX4 LAND variant of the hover-and-land test
    m, px4, world, log = run(Scenario(name='stop_takeoff_px4', t_max=60), Params(stop_after='takeoff', stop_land_px4=True))
    assert phase_seq(log) == ['wait', 'arm', 'takeoff', 'staged_stop', 'done'] and m.sub == 'stopped_after_takeoff'
    assert any(c[1].name == 'land' for c in px4.cmds) and not any(c[1].name == 'disarm' for c in px4.cmds)
    assert all(r['out'].pos_ned is None for r in log if r['nav'] == NAV_AUTO_LAND and r['armed'])
    print('  staged stops: ' + '; '.join(rows) + '; takeoff + PX4 LAND ok')


def _mission_at_home(prior, params=None):
    """Mission with home recorded at the origin, for geometry unit checks."""
    m = Mission(CoursePrior.from_dict(prior), params)
    m.now, m._events = 0.0, []
    m.veh = Vehicle(pos_ned=(0.0, 0.0, 0.0), heading=0.0, armed=True, nav_state=NAV_OFFBOARD)
    m.pos = np.zeros(3)
    m._record_home()
    return m


def _fence_clearance(m, q):
    f, l, u = m.sf.from_ned(q)
    F = m.prior.fence
    return min(f - F['forward_min'], F['forward_max'] - f, l - F['left_min'], F['left_max'] - l)


def test_traverse_commit_gate_and_axis_tracking():
    """Official-compat round 4: window lateral offsets 0.6-16.3 cm at 1.2 m/s although the latch was 1.4-7.8 cm.
    PX4 tracking error is modelled as a drift to the right proportional to speed."""
    OLD = dict(v_window=1.2, commit_lat_tol_m=1.0, commit_vert_tol_m=1.0, commit_lat_vel=10.0, commit_hold_s=0.0,
               traverse_lat_gain=0.0)
    for seed in (0, 1):
        errs = {}
        for label, prm in (('new', Params()), ('old', Params(**OLD))):
            m, px4, world, log = run(Scenario(name='drift', seed=seed, lat_drift_per_speed=0.15, t_max=90), prm)
            assert phase_seq(log) == list(PHASES), (label, phase_seq(log))
            qw, _ = crossing(world, log, TRUE['window_center'][0], -1)
            qt, _ = crossing(world, log, TRUE['tunnel_entry'][0], +1)
            errs[label] = (abs(qw[1] - TRUE['window_center'][1]), abs(qt[1] - TRUE['tunnel_entry'][1]))
            if label == 'new':
                cm = events(log, 'commit')
                assert [e['stage'] for e in cm] == ['tunnel', 'window'], cm
                p = Params()
                for e in cm:
                    k = p.commit_relax_factor if e['relaxed'] else 1.0
                    assert e['lat_m'] < p.commit_lat_tol_m * k and e['vperp'] < p.commit_lat_vel * k, e
                assert len(events(log, 'plane_crossing')) == 2
        assert errs['new'][0] < 0.07 and errs['new'][0] < 0.75 * errs['old'][0], errs
        assert errs['new'][1] < 0.75 * errs['old'][1], errs
    # a lateral gust while aligning at the window: the commit waits until the vehicle is back on the axis and still
    m0, _, _, log0 = run(Scenario(name='nominal_ref'))
    t_w = [e for e in events(log0, 'latched') if e['detector'] == 'window'][0]['t']
    gust = (t_w - 0.3, t_w + 2.5, (0.0, 0.25, 0.0))
    m, px4, world, log = run(Scenario(name='window_gust', wind=gust, t_max=90))
    assert phase_seq(log) == list(PHASES), phase_seq(log)
    cw = [e for e in events(log, 'commit') if e['stage'] == 'window'][0]
    assert cw['t'] >= gust[1] and cw['lat_m'] < Params().commit_lat_tol_m, cw
    pc = [e for e in events(log, 'plane_crossing') if e['stage'] == 'window'][0]
    assert pc['lat_m'] < 0.05, pc
    assert_no_collision(world, log)
    print('  traverse: drift 0.15*v -> window lateral %.3f m (old settings %.3f), tunnel %.3f (old %.3f); '
          'gust: window commit %.1f s after the gust, plane lateral %.3f m'
          % (errs['new'][0], errs['old'][0], errs['new'][1], errs['old'][1], cw['t'] - gust[1], pc['lat_m']))


def test_standoff_keeps_fence_clearance():
    """Round 4 seed 4: a perturbed prior put the tunnel standoff 0.24 m from the west net; the cage frame blocked it."""
    fence = dict(forward_min=-0.75, forward_max=5.35, left_min=-5.2, left_max=0.9, up_max=2.9)
    clear = Params().standoff_fence_clear_m
    # seed-4 geometry: tunnel axis -95 deg, the standoff falls toward the west (left) net -> shrink along the axis
    pr = prior_dict(fence=dict(fence), corridor=[dict(forward=0.845, left=-0.167, up=0.779),
                                                  dict(forward=3.581, left=0.514, up=0.853)])
    pr['tunnel'] = dict(entry=dict(forward=4.583, left=-1.007, up=0.525), axis_deg=-95.24, length_m=1.2, tol_m=0.5)
    m = _mission_at_home(pr)
    q = m._standoff('tunnel')
    ev = [e for e in m._events if e['kind'] == 'standoff_shrunk']
    assert ev and Params().standoff_min_m <= ev[0]['standoff_m'] < 1.8, m._events
    assert _fence_clearance(m, q) >= clear - 1e-6, _fence_clearance(m, q)
    c, d = m.g['tunnel_entry'], m.g['tunnel_axis']
    rel = (q - c)[:2]
    assert abs(rel[0] * d[1] - rel[1] * d[0]) < 1e-6          # still on the axis (slid along it, not sideways)
    # the corridor goal that sat inside the fence margin is pulled in
    pulled = [e for e in m._events if e['kind'] == 'corridor_goal_pulled_in']
    assert pulled and pulled[0]['index'] == 1 and _fence_clearance(m, m.g['corridor'][1]) >= clear - 1e-6
    # axis parallel to the net: shrinking cannot help -> slide sideways into the clear box, with an event
    pr2 = prior_dict(fence=dict(fence))
    pr2['tunnel'] = dict(entry=dict(forward=3.0, left=0.6, up=0.6), axis_deg=0.0, length_m=1.2, tol_m=0.5)
    m2 = _mission_at_home(pr2)
    q2 = m2._standoff('tunnel')
    assert [e for e in m2._events if e['kind'] == 'standoff_clipped'] and _fence_clearance(m2, q2) >= clear - 1e-6
    # closed loop: the replica-like tunnel 1.95 m in front of the net hovers >= clear from it
    entry = (FENCE['forward_min'] + 1.95, 0.0, 0.62)
    pr3 = prior_dict(corridor=[])
    pr3['tunnel'] = dict(pr3['tunnel'], entry=dict(forward=entry[0], left=0.0, up=0.6))
    m3, px4, world, log = run(Scenario(name='tunnel_near_net', prior=pr3, true_over={'tunnel_entry': entry}))
    hover = [r for r in log if r['phase'] == 'tunnel_align' and r['sub'] == 'detect']
    gap = min(world.flu(r['out'].pos_ned)[0] - FENCE['forward_min'] for r in log if r['phase'] == 'tunnel_align')
    assert gap >= clear - 1e-6 and phase_seq(log) == list(PHASES), (gap, phase_seq(log))
    print('  standoff: seed-4 geometry shrunk to %.2f m (fence gap %.2f), parallel-to-net case slid in (gap %.2f), '
          'closed loop hover %.2f m from the net' % (ev[0]['standoff_m'], _fence_clearance(m, q), _fence_clearance(m2, q2), gap))


def test_tunnel_align_phantom_before_entry():
    """Round 5b seed 6: 13-24 view-locked phantom voxels (tiles seen THROUGH the tunnel) 0.5 m ahead of the hovering
    drone, ~1.1 m before the entry, blocked the align after an accurate latch."""
    ph = ((1.95, 0.02, 0.75), 20, None, 4.0)                  # appears while we hover at the standoff (f ~1.41)
    m, px4, world, log = run(Scenario(name='seed6_like', phantoms=(ph,)))
    assert phase_seq(log) == list(PHASES), (phase_seq(log), m.abort_reason)
    assert not [e for e in events(log, 'blocked') if e['phase'] == 'tunnel_align']
    ex = m._exclude['tunnel'][0]
    assert ex[3] * 2 > TRUE['tunnel_length'] + 1.5                # reaches back to the hover point
    ph30 = ((1.95, 0.02, 0.75), 30, None, 4.0)               # clearly not minor (20 sits at the 12-point boundary
    m2, _, _, log2 = run(Scenario(name='seed6_like_old', phantoms=(ph30,), t_max=60),       # when seen obliquely)
                         Params(exclude_to_standoff=False))
    assert m2.abort_reason == 'tunnel_align_blocked', m2.abort_reason
    assert [e for e in events(log2, 'parallax_nudge') if e['phase'] == 'tunnel_align']   # nudged before blocking
    ph_v = ((1.95, 0.02, 0.75), 20, 'parallax', 4.0)          # view-locked: the nudge alone clears it
    m3, _, world3, log3 = run(Scenario(name='seed6_like_view_locked', phantoms=(ph_v,)), Params(exclude_to_standoff=False))
    assert phase_seq(log3) == list(PHASES) and world3.vanished, (phase_seq(log3), m3.abort_reason)
    print('  seed-6-like tunnel phantom: excluded to the hover point -> done; old exclusion -> %s after a nudge; '
          'view-locked + old exclusion -> nudge carved it -> done' % m2.abort_reason)


def test_side_step_return_and_corridor_cap():
    """Round 5b seed 7: after a 1.2 m detour the return to 0.3 m slid sideways into an unmeasured panel."""
    pr = Params()
    path = Follower(np.array([0.0, 0.0, -1.0]), [np.array([5.0, 0.0, -1.0])], pr, 1.0, avoid='lateral')
    sf = StartFrame((0.0, 0.0, 0.0), 0.0)
    panel = BoxMap([(np.array([2.45, -0.60, 0.5]), np.array([2.50, -0.55, 1.5]))], sf)   # FLU: y east = -left

    def probe(chain, strict=False, extra=None):
        r = pr.radius + (pr.return_margin_m if strict else 0.0)
        free = 0.0
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            seg = float(np.linalg.norm(b - a))
            if seg < 1e-4:
                continue
            c = panel.clearance(a, b, r if (strict or i > 0) else pr.hard_radius)
            if c < seg:
                return free + c
            free += seg
        return math.inf
    pos = np.array([2.0, 1.2, -1.0])
    path.s, path.off = 2.6, 1.2
    path.off_tgt = 1.2
    old_chain = path._chain(pos, path.project(pos), min(path.S, 2.6 + 1.2), 0.0)
    assert probe(old_chain, strict=True) == math.inf, 'the old check (forward chain only) would allow 0.0'
    assert probe([pos, path.ref(2.6, 0.0)], strict=True) < math.inf, 'the diagonal to the new carrot hits the panel'
    back, _ = path._try_return(pos, 1.2, probe, None, None)
    for o in [x for x in np.arange(0.0, 1.2, 0.15) if back is None or x <= back + 1e-9]:
        if back is not None and abs(o - back) < 1e-9:
            assert probe([pos, path.ref(2.6, back)], strict=True) == math.inf
    assert back is None or probe([pos, path.ref(2.6, back)], strict=True) == math.inf
    # unseen sweep -> look first, seen -> return
    clear = lambda ch, strict=False, extra=None: math.inf       # noqa: E731
    b2, uns = path._try_return(pos, 1.2, clear, None, lambda q: False)
    assert b2 is None and uns is not None and abs(uns[0]) <= 1e-9
    b3, _ = path._try_return(pos, 1.2, clear, None, lambda q: True)
    assert b3 == 0.0
    # corridor cap: a wide obstacle that can only be passed through the next bay -> no detour beyond 0.6 m
    wide = ((0.9, -0.55, 0.0), (1.1, 0.6, 2.2), 'wide')        # fence-wide on the left, open only > 0.7 m right
    m, px4, world, log = run(Scenario(name='wide', extra_boxes=(wide,), t_max=40))
    lat = max(-world.flu(r['p'])[1] for r in log if r['phase'] in ('corridor', 'abort_land'))
    assert lat <= pr.corridor_side_max_m + pr.nudge_lat_m + 0.05, lat
    assert m.abort_reason == 'corridor_blocked', m.abort_reason
    assert_no_collision(world, log)
    print('  return sweep: diagonal into the panel rejected (got %s), unseen sweep -> look; corridor cap: max right '
          '%.2f m, then hover + LAND' % (back, lat))


def test_return_home_high_then_descend():
    """Round 6 seed 7: return_home descended while flying west and clipped an edge-on board stereo could not see."""
    board = ((0.58, -0.9, 0.45), (0.62, -0.3, 1.75), 'edge_on_board')   # on the old descending diagonal
    m, px4, world, log = run(Scenario(name='return_high', invisible_boxes=(board,)))
    assert phase_seq(log) == list(PHASES), (phase_seq(log), m.abort_reason)
    assert_no_collision(world, log)
    z_over_up = float(world.sf.o[2] - m._z_over)
    rh = [world.flu(r['p']) for r in log if r['phase'] == 'return_home']
    assert min(q[2] for q in rh[5:]) > z_over_up - 0.15, min(q[2] for q in rh[5:])     # no descent on the way
    for r in log:                                            # descend only over home
        if r['phase'] == 'land_home':
            q = world.flu(r['p'])
            if q[2] < z_over_up - 0.2:
                assert math.hypot(q[0], q[1]) < 0.3, q
    m2, _, world2, log2 = run(Scenario(name='return_low', invisible_boxes=(board,), t_max=80), Params(return_high=False))
    hits = [world2.collision(r['p']) for r in log2 if r['phase'] == 'return_home']
    assert 'edge_on_board' in hits, 'the old descending return should hit the edge-on board'
    # net trip line never inside the props
    p = Params(fence_inset_m=0.1)
    mm = Mission(CoursePrior.from_dict(prior_dict()), p)
    assert mm._trip_inset() >= p.half_span_m + p.fence_trip_margin_m - 1e-9
    try:
        Mission(CoursePrior.from_dict(prior_dict()), Params(fence_margin_m=0.3))
    except ValueError:
        pass
    else:
        raise AssertionError('fence_margin below the trip line + 0.1 accepted')
    print('  return home: stayed >= %.2f m up, descended over home; old descending return hit the edge-on board; '
          'trip line %.2f m' % (z_over_up - 0.15, Params().fence_inset_m))


def _window_reacquire_case(name, center_flu, through_deg):
    pr = prior_dict()
    pr['window'] = dict(center=dict(forward=center_flu[0], left=center_flu[1], up=center_flu[2]),
                        through_deg=through_deg, tol_m=0.5)
    m, px4, world, log = run(Scenario(name=name, prior=pr, t_max=140))
    m0, _, _, log0 = run(Scenario(name=name + '_old', prior=pr, t_max=90), Params(reacquire_max=0))
    return m, px4, world, log, m0, log0


def test_window_reacquire_after_cut_by_fov():
    """Round 6 seed 13: eyeball prior put the standoff 1.08 m from the board at -19 deg -> cut_by_fov until timeout."""
    m, px4, world, log, m0, log0 = _window_reacquire_case('seed13_like', (3.2, -3.15, 1.1), 168.3)
    miss0 = [e for e in events(log0) if e['kind'] == 'reacquire']
    assert m0.abort_reason and m0.abort_reason.startswith('timeout_window_align'), m0.abort_reason
    assert not miss0
    rq = [e for e in events(log, 'reacquire') if e['stage'] == 'window']
    assert rq and rq[0]['source'] == 'map' and 'cut_by_fov' in rq[0]['reasons'], rq
    assert phase_seq(log) == list(PHASES), (phase_seq(log), m.abort_reason)
    q, _ = crossing(world, log, TRUE['window_center'][0], -1)
    assert abs(q[1] - TRUE['window_center'][1]) < TRUE['window_w'] / 2 - HALF_SPAN
    assert_no_collision(world, log)
    print('  window seed-13-like: reacquired from the map board estimate (reasons %s), standoff %.2f m -> done; '
          'without: %s' % (rq[0]['reasons'], rq[0]['standoff_m'], m0.abort_reason))


def test_window_reacquire_after_oblique_view():
    """Round 6 seed 11: eyeball prior 0.72 m / +11.6 deg -> hover 0.8 m off the window axis, oblique -> not_enclosed."""
    m, px4, world, log, m0, log0 = _window_reacquire_case('seed11_like', (3.7, -2.1, 1.1), 191.6)
    assert m0.abort_reason and m0.abort_reason.startswith('timeout_window_align'), m0.abort_reason
    rq = [e for e in events(log, 'reacquire') if e['stage'] == 'window']
    assert rq and rq[0]['source'] == 'map', rq
    assert phase_seq(log) == list(PHASES), (phase_seq(log), m.abort_reason)
    q, _ = crossing(world, log, TRUE['window_center'][0], -1)
    assert abs(q[1] - TRUE['window_center'][1]) < TRUE['window_w'] / 2 - HALF_SPAN
    assert_no_collision(world, log)
    print('  window seed-11-like: reasons %s -> reacquired, standoff %.2f m -> done; without: %s'
          % (rq[0]['reasons'], rq[0]['standoff_m'], m0.abort_reason))


def test_rock_observe_distance_and_reacquire():
    """Round 6 seed 12: eyeball rock prior 0.72 m off -> observe point 1.06 m from the rock, top 36 deg below the
    horizon (outside the +-25 deg vertical FOV) -> no_flat_top until timeout."""
    u = np.array([1.6 - 3.0, -3.8 + 2.8])
    u /= np.linalg.norm(u)                                     # approach direction window exit -> rock
    far = np.array(TRUE['rock_center']) + 0.72 * u
    pr = prior_dict(landing_target=dict(center=dict(forward=float(far[0]), left=float(far[1])), top_up=0.0, tol_m=0.5))
    m, px4, world, log = run(Scenario(name='seed12_like', prior=pr, t_max=140))
    assert phase_seq(log) == list(PHASES), (phase_seq(log), m.abort_reason)
    lat = [e for e in events(log, 'latched') if e['detector'] == 'landing'][0]
    assert np.linalg.norm(world.flu(np.array(lat['center']))[:2] - np.array(TRUE['rock_center'])) < 0.1
    old = Params(observe_max_depression_deg=60.0, reacquire_max=0)     # 1.6 m from the prior, no back-off
    m0, _, _, log0 = run(Scenario(name='seed12_like_old', prior=pr, t_max=110), old)
    assert m0.abort_reason and m0.abort_reason.startswith('timeout_land_target'), m0.abort_reason
    rq = [e for e in events(log, 'reacquire') if e['stage'] == 'landing']
    # landing-only prior gate
    mg = _mission_at_home(prior_dict())
    mg.per, mg.vel, mg.dt, mg.phase, mg.sub = Perception(), np.zeros(3), 0.05, 'land_target', ''
    mg._hints, mg._cmds = {}, []
    mg._ph_land_target()
    assert abs(mg._latch.gate - Params().landing_gate_m) < 1e-9 and Params().landing_gate_m >= 1.2
    print('  rock seed-12-like: observe distance %.2f m, reacquire %s -> latched %.2f m from truth -> done; '
          'old 1.6 m rule: %s; landing gate %.1f m'
          % (m._observe_dist(), [(e['attempt'], e['observe_dist_m'], e['reasons']) for e in rq],
             np.linalg.norm(world.flu(np.array(lat['center']))[:2] - np.array(TRUE['rock_center'])),
             m0.abort_reason, mg._latch.gate))


PILOT_PHASES = [ph for ph in PHASES if ph != 'arm']


def test_pilot_mode_full_course():
    """skybeta contract: no commands, pose-only tracking, pilot arms / lands / re-arms, 0.3..3.0 m pose clamp."""
    m, px4, world, log = run(Scenario(name='pilot_full', pilot_mode=True, t_max=200))
    assert phase_seq(log) == PILOT_PHASES, (phase_seq(log), m.abort_reason, m.release_reason)
    assert not px4.armed and px4.landed
    plr = events(log, 'pilot_land_requested')
    assert [e['phase'] for e in plr] == ['land_target', 'land_home'], plr
    # take off on the first Offboard step (10 s rule)
    t_off = px4.offboard_starts[0]
    t_to = first(log, lambda r: r['phase'] == 'takeoff')['t']
    assert t_to - t_off <= 2 * DT, (t_off, t_to)
    # the second Offboard session (after the pilot landed us on the rock and re-armed) continues the course
    assert len(px4.offboard_starts) == 2
    t_climb = first(log, lambda r: r['phase'] == 'climb_over_wall')['t']
    assert t_climb - px4.offboard_starts[1] <= 2 * DT
    # every published pose inside the skybeta clamp; the rock disarm happened on the rock
    for r in log:
        o = r['out']
        if o.pos_ned is not None and r['phase'] not in ('wait',):
            up = px4.origin_d - o.pos_ned[2]
            assert 0.3 - 1e-9 <= up <= 3.0 + 1e-9, (r['t'], r['phase'], up)
    r = first(log, lambda r: r['phase'] == 'disarm')
    f, l, u = world.flu(r['p'])
    assert abs(u - TRUE['rock_top']) < 0.02 and abs(f - TRUE['rock_center'][0]) < 0.15, (f, l, u)
    assert np.linalg.norm(px4.p[:2] - np.array(Scenario().home_ned)[:2]) < 0.15
    assert_no_collision(world, log)
    # speed comes from the carrot lead (no feed-forward): speed ~ gain x lead, never a far waypoint
    tr = [r for r in log if r['out'].pos_ned is not None
          and r['phase'] in ('corridor', 'window_align', 'land_target', 'climb_over_wall', 'return_home')]
    vmax = max(np.linalg.norm(r['v']) for r in tr)
    lead = max(np.linalg.norm(r['out'].pos_ned - r['p']) for r in tr)
    p = Params(pilot_mode=True)
    assert vmax > 1.0, vmax
    assert lead <= p.pilot_max_lead_m + 0.05, lead
    assert px4.lead_clipped == 0, 'the adapter had to clip %d poses' % px4.lead_clipped
    # pose-only descent: the setpoint leads by rate / gain, so the home descent from the wall height is quick
    lh = [r for r in log if r['phase'] == 'land_home' and r['sub'] == 'descend']
    assert lh and lh[-1]['t'] - lh[0]['t'] < 8.0, lh[-1]['t'] - lh[0]['t']
    # gates still hold through the window with pose-only tracking
    q, _ = crossing(world, log, TRUE['window_center'][0], -1)
    assert abs(q[1] - TRUE['window_center'][1]) < TRUE['window_w'] / 2 - HALF_SPAN
    print('  pilot_mode: done %.1f s after the first Offboard (incl. pilot landing + re-arm), no commands, 2 pilot '
          'landings, max speed %.2f m/s, max carrot lead %.2f m' % (m.history[-1][0] - t_off, vmax, lead))


def test_pilot_mode_height_bias_never_climbs():
    """Real skybeta-05 hover (2026-11-08, core 0c2180b7): PX4 local and cuVSLAM odom 0.46 m apart, so the vehicle sat
    0.25-0.4 m ABOVE every setpoint.  The descent stalled above the clamp -> timeout -> pilot_land_requested; then
    pilot_hold's z = min(floor, pos + push) ratcheted UP every cycle (sp -0.5 -> -2.38, vehicle to 2.63 m) until the
    pilot took over.  Now: hover -> descend -> pilot_hold with a +0.4 m bias: the setpoint only ever goes down (also
    from the descent into the hold), the vehicle never climbs, and the pilot is still asked to land."""
    scn = Scenario(name='pilot_bias', pilot_mode=True, pilot_z_bias_m=0.4, pilot_react_s=20.0, t_max=120)
    m, px4, world, log = run(scn, Params(pilot_mode=True, stop_after='takeoff'))
    plr = events(log, 'pilot_land_requested')
    assert plr and plr[0]['why'] in ('timeout', 'at_clamp', 'contact'), plr
    land = [r for r in log if r['phase'] == 'staged_stop' and r['sub'] in ('descend', 'pilot_hold')
            and r['out'].pos_ned is not None]
    z = np.array([r['out'].pos_ned[2] for r in land])
    assert len(z) > 50 and np.all(np.diff(z) >= -1e-9), float(np.min(np.diff(z)))        # NED z never decreases
    hold = [r for r in log if r['phase'] == 'staged_stop' and r['sub'] == 'pilot_hold' and r['out'].pos_ned is not None]
    up0 = float(world.sf.o[2] - hold[0]['p'][2])
    up_max = max(float(world.sf.o[2] - r['p'][2]) for r in hold)
    assert up_max <= up0 + 0.02, (up0, up_max)              # the vehicle never climbs while we hold
    assert not events(log, 'climb_guard'), events(log, 'climb_guard')
    print('  pilot_mode +0.4 m height bias: pilot_land_requested (%s), hold setpoint only down for %.1f s, vehicle '
          '%.2f -> max %.2f m (never climbs)' % (plr[0]['why'], hold[-1]['t'] - hold[0]['t'], up0, up_max))


def test_pilot_mode_climb_guard():
    """pilot_mode global guard: > climb_guard_m above the highest altitude the prior / phase intends and still
    rising -> stop publishing ('climb_guard', released): the pilot gets control.  Forced by a 1 m/s updraft while
    hovering at the cruise height."""
    scn = Scenario(name='pilot_updraft', pilot_mode=True, t_max=60, wind=(6.0, 14.0, (0.0, 0.0, 1.0)))
    m, px4, world, log = run(scn, Params(pilot_mode=True, stop_after='takeoff', stop_hover_s=10.0))
    cg = events(log, 'climb_guard')
    assert cg and m.phase == 'released' and 'climb_guard' in (m.release_reason or ''), (cg, m.phase, m.release_reason)
    t_cg = cg[0]['t']
    assert all(r['out'].pos_ned is None for r in log if r['t'] > t_cg + DT), 'setpoints after the climb guard'
    assert cg[0]['up_m'] <= cg[0]['limit_m'] + 0.15, cg
    print('  pilot_mode climb guard: released at %.2f m (limit %.2f m, rising %.2f m/s), no setpoints after'
          % (cg[0]['up_m'], cg[0]['limit_m'], cg[0]['v_up']))


def test_pilot_mode_takeover_rearm_and_abort():
    # stick move / mode switch mid-corridor -> released, nothing published after
    m, px4, world, log = run(Scenario(name='pilot_takeover', pilot_mode=True, pilot_takeover_t=6.0, t_max=20))
    assert m.phase == 'released' and m.release_reason.startswith('left_offboard'), (m.phase, m.release_reason)
    assert_released_forever(log, events(log, 'release')[0]['t'])
    # re-arm but the flight interface stays out of Offboard -> pilot owns it after 3 s: released, not aborted
    m, px4, world, log = run(Scenario(name='pilot_refused', pilot_mode=True, pilot_refuse_offboard_on_rearm=True,
                                      t_max=120))
    assert m.phase == 'released' and m.release_reason == 'armed_without_offboard_after_rearm', (m.phase, m.release_reason)
    assert m.abort_reason is None
    # a slow re-arm (pilot needs 40 s) just waits on the ground, then continues
    m, px4, world, log = run(Scenario(name='pilot_slow', pilot_mode=True, pilot_rearm_delay_s=40.0, t_max=240))
    assert phase_seq(log) == PILOT_PHASES, (phase_seq(log), m.abort_reason)
    t_re = first(log, lambda r: r['phase'] == 'rearm')['t']
    assert first(log, lambda r: r['phase'] == 'climb_over_wall')['t'] - t_re > 40.0
    # abort (geofence gust) without a LAND command: brake, descend to the clamp, ask the pilot, pilot lands
    m0, _, _, log0 = run(Scenario(name='pilot_ref', pilot_mode=True, t_max=200))
    t_g = first(log0, lambda r: r['phase'] == 'corridor')['t'] + 0.3
    m, px4, world, log = run(Scenario(name='pilot_fence', pilot_mode=True, wind=(t_g, t_g + 2.0, (0.0, 1.6, 0.0)),
                                      t_max=80))
    assert m.abort_reason == 'geofence' and phase_seq(log)[-2:] == ['abort_land', 'aborted'], (m.abort_reason, phase_seq(log))
    assert [e['phase'] for e in events(log, 'pilot_land_requested')] == ['abort_land']
    assert not px4.armed
    # the only clipped poses are the pull-back toward the abort point while the gust blew us away
    assert all(t_g <= tc <= t_g + 2.0 + 1.5 for tc in px4.clip_t), px4.clip_t
    # staged stop: hover, descend to the clamp, the pilot lands -> done / stopped_after_takeoff
    m, px4, world, log = run(Scenario(name='pilot_stop', pilot_mode=True, t_max=60),
                             Params(pilot_mode=True, stop_after='takeoff'))
    assert phase_seq(log) == ['wait', 'takeoff', 'staged_stop', 'done'] and m.sub == 'stopped_after_takeoff', phase_seq(log)
    print('  pilot_mode: takeover -> released; refused Offboard on re-arm -> released (not aborted); slow re-arm '
          'waits; geofence -> pilot_land_requested, no LAND; stop_after takeoff -> done')


def test_corner_yaw_ahead_and_speed_cap():
    """Official lab runs 002-005: before a stop corner turn to the next leg ON THE WAY IN (last corner_yaw_ahead_m),
    slowly (yaw_rate_near 0.4 rad/s) and arrive no sooner than the yaw is done -- never a 90 deg yaw in place facing
    the wall; <= v_near_waypoint (0.3 m/s) within 1 m of the corner and of the end."""
    pr = Params()
    f = Follower(np.array([0.0, 0.0, -1.0]), [np.array([2.5, 0.0, -1.0]), np.array([2.5, 2.0, -1.0])], pr,
                 pr.v_cruise, avoid='lateral')
    corner, end = 2.5, 4.5
    next_leg = math.atan2(1.0, 0.0)
    vmax_far, t, h = 0.0, 0.0, 0.0
    t_in, yaw_at_corner = None, None
    while f.s < f.S - 1e-6 and t < 40:
        sp = f.step(0.05, t, f.ref(f.s), h)                   # perfect tracking; the heading slews at 0.4 rad/s
        h = wrap(h + float(np.clip(wrap(sp.sp[3] - h), -pr.yaw_rate_near * 0.05, pr.yaw_rate_near * 0.05)))
        s = f.s
        d = corner - s
        if 0.05 < d <= pr.corner_yaw_ahead_m:
            if t_in is None:
                t_in = t
            assert abs(wrap(sp.sp[3] - next_leg)) < math.radians(2.0), (s, math.degrees(sp.sp[3]))
        if yaw_at_corner is None and d <= 0.02:
            yaw_at_corner = abs(math.degrees(wrap(h - next_leg)))
        if 0.0 <= corner - s <= pr.near_waypoint_m or 0.0 <= end - s <= pr.near_waypoint_m:
            assert f.v <= pr.v_near_waypoint + 1e-6, (s, f.v)
        if corner - s > 1.3:
            vmax_far = max(vmax_far, f.v)
        t += 0.05
    assert vmax_far > 0.9, vmax_far                          # (2.5 m leg: accel 1.5, planned decel 0.5 m/s^2)
    assert yaw_at_corner is not None and yaw_at_corner <= pr.corner_yaw_tol_deg + 1.0, yaw_at_corner
    print('  corner: turns to the next leg during the last %.1f m (%.1f rad/s), %.0f deg left at the corner, '
          '<= %.1f m/s within %.1f m of the corner and the end, %.2f m/s earlier'
          % (pr.corner_yaw_ahead_m, pr.yaw_rate_near, yaw_at_corner, pr.v_near_waypoint, pr.near_waypoint_m, vmax_far))


def test_vio_wall_standoff_and_waypoint_clip():
    """Official lab run_003: the corridor goal sat right in front of the end wall; flying at it starved cuVSLAM."""
    wall = ((1.75, -0.6, 0.0), (1.85, 0.6, 2.5), 'end_wall')    # 0.34 m beyond the corridor end (tunnel standoff)
    scn = Scenario(name='end_wall', extra_boxes=(wall,), t_max=60)
    m, px4, world, log = run(scn, Params(stop_after='corridor'))
    assert phase_seq(log)[-2:] == ['staged_stop', 'done'] and m.sub == 'stopped_after_corridor', \
        (phase_seq(log), m.abort_reason)
    assert events(log, 'waypoint_clipped')
    gap = min(wall[0][0] - world.flu(r['p'])[0] - 0.116 for r in log if r['phase'] in ('corridor', 'staged_stop'))
    sp_gap = min(wall[0][0] - world.flu(r['out'].pos_ned)[0] - 0.116 for r in log
                 if r['phase'] == 'corridor' and r['out'].pos_ned is not None)
    assert sp_gap >= Params().vio_wall_standoff_m - 0.02, sp_gap
    assert gap >= Params().vio_wall_standoff_m - 0.05, gap
    assert_no_collision(world, log)
    old = Params(stop_after='corridor', vio_wall_standoff_m=0.0, v_near_waypoint=10.0)
    m0, _, _, log0 = run(scn, old)
    gap0 = min(wall[0][0] - world.flu(r['p'])[0] - 0.116 for r in log0 if r['phase'] in ('corridor', 'abort_land'))
    assert gap0 < 0.6, gap0                                   # the old planner flew the camera right up to the wall
    print('  end wall: corridor end clipped, camera >= %.2f m from the wall (setpoints >= %.2f m) -> stopped_after_'
          'corridor; old planner: camera %.2f m from the wall' % (gap, sp_gap, gap0))


def test_nudge_only_when_stopped_and_sideways():
    """Official lab run_003: a nudge started at 0.94 m/s with the carrot ahead and the drone kept going forward."""
    ph = ((0.65 + 0.116, 0.0, 1.0), 4, None)
    m, px4, world, log = run(Scenario(name='nudge', phantoms=(ph,)))
    nudges = [e for e in events(log, 'parallax_nudge')]
    assert nudges
    for e in nudges:
        r0 = first(log, lambda r: r['t'] >= e['t'])
        assert np.linalg.norm(r0['v']) <= Params().nudge_max_speed + 0.05, np.linalg.norm(r0['v'])
        f0 = world.flu(r0['p'])[0]
        during = [world.flu(r['p'])[0] for r in log if e['t'] <= r['t'] <= e['t'] + Params().nudge_period_s]
        assert max(during) - f0 < 0.05, max(during) - f0       # sideways / up only, never toward the obstacle
    print('  nudge: started at <= %.2f m/s, forward drift during the nudge < 5 cm' % (Params().nudge_max_speed + 0.05))


MAZE = (((-0.3, -0.58, 0.0), (2.0, -0.52, 1.8), 'board_a'),     # left | middle lane, open at the back
        ((3.10, -1.60, 0.0), (3.16, 0.60, 1.8), 'back_wall'),     # 1.1 m behind board_a's end
        ((0.9, -1.68, 0.0), (3.16, -1.62, 1.8), 'board_b'),      # middle | right lane, open at the front
        ((-0.26, -2.80, 0.0), (-0.20, -0.60, 1.8), 'front_wall'),  # 1.1 m in front of board_b's end
        ((-0.2, -2.78, 0.0), (3.16, -2.72, 1.8), 'right_board'))


def test_corridor_uturn_maze():
    """Stage 1 is a 3-lane serpentine (organizers' diagram): left lane front->back, right turn behind the middle board,
    middle lane back->front, right turn in front of the right board, right lane front->back -- two 180 deg U-turns
    through ~1.1 m gaps.  The prior only gives the lane ends; the U-turn waypoints are inserted."""
    pr = prior_dict(corridor=[dict(forward=2.55, left=0.0, up=1.0), dict(forward=0.35, left=-1.1, up=1.0),
                              dict(forward=2.6, left=-2.2, up=1.0)])
    pr['tunnel'] = dict(entry=dict(forward=4.4, left=-2.2, up=1.0), axis_deg=0.0, length_m=1.2, tol_m=0.5)
    over = {'wall_f': (4.7, 4.9), 'wall_l': (-4.4, -4.2)}      # the climbing wall out of the maze for this test
    scn = Scenario(name='maze', prior=pr, extra_boxes=MAZE, true_over=over, t_max=90)
    m, px4, world, log = run(scn, Params(stop_after='corridor'))
    assert phase_seq(log)[-2:] == ['staged_stop', 'done'] and m.sub == 'stopped_after_corridor', (phase_seq(log), m.abort_reason)
    ins = events(log, 'uturn_waypoint_inserted')
    assert len(ins) == 2, ins
    assert_no_collision(world, log)
    cor = [r for r in log if r['phase'] == 'corridor']
    q = np.array([world.flu(r['p']) for r in cor])
    assert q[q[:, 1] > -0.3, 0].max() > 2.3                       # to the back of the left lane
    assert q[(q[:, 1] < -0.8) & (q[:, 1] > -1.4), 0].min() < 0.6   # to the front of the middle lane
    assert q[q[:, 1] < -1.9, 0].max() > 2.3                       # down the right lane
    def in_turn(q):                                            # last 1 m into each U-turn + its crossing leg
        f, l = q[0], q[1]
        return ((l > -0.3 and f >= 1.55) or (f >= 2.3 and -1.1 <= l <= 0.0)          # U1: behind board_a
                or (-1.4 < l < -0.8 and f <= 1.35) or (f <= 0.6 and -2.2 <= l <= -1.1))  # U2: before board_b
    vt = [np.linalg.norm(r['v'][:2]) for r in cor if in_turn(world.flu(r['p']))]
    assert vt and max(vt) <= Params().v_near_waypoint + 0.1, max(vt)
    # VIO: whenever we move toward what the camera faces, nothing measured is closer than ~0.8 m straight ahead
    worst = math.inf
    for r in cor:
        h = np.array([math.cos(r['yaw']), math.sin(r['yaw']), 0.0])
        if float(r['v'] @ h) < 0.2:
            continue
        cam = r['p'] + 0.116 * h
        worst = min(worst, world.map.clearance(cam, cam + h * 1.5, 0.2) + 0.2)
    assert worst >= Params().vio_wall_standoff_m - 0.2, worst
    # without the U-turn insertion the lane-end prior cuts diagonally through the boards
    m0, _, world0, log0 = run(scn, Params(stop_after='corridor', uturn_min_deg=181.0))
    hits = [world0.collision(r['p']) for r in log0 if r['phase'] == 'corridor']
    assert any(hits) or m0.abort_reason, 'the diagonal shortcut should hit a board or get blocked'
    print('  maze: 2 U-turn waypoints inserted, both turns flown at <= %.2f m/s, nearest obstacle ahead while moving '
          'toward it %.2f m, no contact; without insertion: %s'
          % (max(vt), worst, ('contact with ' + str([h for h in hits if h][0])) if any(hits) else m0.abort_reason))


def test_replica_maze_l_corners():
    """Ubuntu sim maze_ideal_truth (report 16.2): with 1.1 m lanes every right-angle corner is 0.43 m from the wall
    ahead; the VIO standoff clipped the corner into a diagonal that cut the next board end -> corridor timeout.
    Now: hold, yaw toward the next leg, fly the L through the true corner.  Geometry = replica/layout.py MAZE_WALLS +
    the taped maze path; obstacles from a depth-like map (only what the camera has seen, 4 s memory)."""
    from replica.layout import MAZE_WALLS, PANEL_BOTTOM, PANEL_TOP, PLY_BOTTOM, PLY_TOP, TAPE, START, CAGE
    def walls_with(extra):                     # physical boards: banner 5 cm, plywood 2 cm thick (+ extra per face)
        return tuple((f0, l0, f1, l1) + ((PANEL_BOTTOM, PANEL_TOP, name, 0.025 + extra) if skin == 'banner'
                                         else (PLY_BOTTOM, PLY_TOP, name, 0.01 + extra))
                     for name, (f0, l0), (f1, l1), skin, _ in MAZE_WALLS)
    walls = walls_with(0.0)
    path = dict(TAPE)['maze_path']
    fence = dict(forward_min=-START[0], forward_max=CAGE - START[0], left_min=-(CAGE - START[1]), left_max=START[1],
                 up_max=2.9)
    pr = prior_dict(fence=fence, corridor=[dict(forward=f, left=l, up=1.0) for f, l in path[1:]])
    f_end, l_end = path[-1]
    pr['tunnel'] = dict(entry=dict(forward=f_end + 1.8, left=l_end, up=1.0), axis_deg=0.0, length_m=1.2, tol_m=0.5)
    base = dict(name='replica_maze', prior=pr, default_course=False, t_max=70, vehicle_r=Params().prop_radius_m)
    out = []
    for label, scn, prm in (
            ('default yaw', Scenario(depth_walls=walls, **base), Params(stop_after='corridor')),
            ('slow yaw 0.5 rad/s', Scenario(depth_walls=walls, **base), Params(stop_after='corridor', yaw_rate=0.5)),
            # official-compat 191052 / 192043: PX4 tracking (MPC_XY_P 0.95, VEL_P_ACC 1.8) overshot the corner 0.21 m
            ('PX4 MPC lag', Scenario(depth_walls=walls, px4_mpc=True, **base), Params(stop_after='corridor')),
            ('PX4 MPC lag, acc ff ignored', Scenario(depth_walls=walls, px4_mpc=True, px4_acc_ff=False, **base),
             Params(stop_after='corridor')),
            ('PX4 MPC lag, boards +2 cm/face', Scenario(depth_walls=walls_with(0.02), px4_mpc=True, **base),
             Params(stop_after='corridor'))):
        m, px4, world, log = run(scn, prm)
        assert m.sub == 'stopped_after_corridor', (label, phase_seq(log), m.abort_reason)
        assert_no_collision(world, log, skip=())             # props: 0.23 m from the body centre
        cor = [r for r in log if r['phase'] == 'corridor']
        t_cor = cor[-1]['t'] - cor[0]['t']
        assert t_cor < Params().phase_timeout_s['corridor'] - 5.0, (label, t_cor)
        holds, rel = events(log, 'vio_corner_hold'), events(log, 'vio_corner_release')
        clips = events(log, 'waypoint_clipped')
        assert all(e['reason'] and e['original'] is not None and e['clipped'] is not None for e in clips), clips
        q = np.array([world.flu(r['p']) for r in cor])
        for cf, cl in path[1:-1]:                              # every corner flown through (an L), never cut
            d = float(np.min(np.hypot(q[:, 0] - cf, q[:, 1] - cl)))
            assert d < 0.25, (label, 'corner not reached', (cf, cl), d)
        worst = math.inf
        for r in cor:
            h = np.array([math.cos(r['yaw']), math.sin(r['yaw']), 0.0])
            if float(r['v'] @ h) < 0.2:
                continue
            cam = r['p'] + 0.116 * h
            worst = min(worst, world.dmap.true_ahead(cam, h))      # true geometry, not the (expiring) map
        # (acc ff ignored: the vehicle lags the carrot by up to the leash and catches up after the VIO hold; the
        #  node publishes the acceleration and PX4 uses it, so this variant only has to stay off the boards)
        assert worst >= (0.25 if 'ignored' in label else Params().vio_wall_standoff_m), (label, worst)
        assert len(rel) >= len(holds), (holds, rel)
        # tight corners (< 0.6 m to the wall beyond): stop ON the point, settle, face the next leg, go
        tight = events(log, 'tight_corner') + events(log, 'stop_corner')   # every right angle stops on the point
        tight_xy = [world.flu(e['corner'])[:2] for e in tight]
        for i in range(1, len(path) - 1):          # every corner with < 0.55 m true space beyond it stops
            c, d = np.array(path[i]), np.array(path[i]) - np.array(path[i - 1])
            d /= np.linalg.norm(d)
            room = min(world.dmap.true_ahead(world.sf.to_ned((c[0], c[1], 1.0)),
                                             np.r_[d[0] * world.sf.fwd[:2] + d[1] * world.sf.left[:2], 0.0], r=0.23),
                       min(((fence['forward_max'] - c[0]) / d[0]) if d[0] > 1e-6 else math.inf,
                           ((fence['forward_min'] - c[0]) / d[0]) if d[0] < -1e-6 else math.inf,
                           ((fence['left_max'] - c[1]) / d[1]) if d[1] > 1e-6 else math.inf,
                           ((fence['left_min'] - c[1]) / d[1]) if d[1] < -1e-6 else math.inf))
            if room < 0.55:                        # (the stop point may be the centred / fence-pulled vertex)
                assert any(np.hypot(*(x - c)) < Params().snap_max_m + 0.05 for x in tight_xy), \
                    (label, 'not tight', c, room, tight_xy)
        settles = events(log, 'corner_settle')       # one settle + go per real corner (a re-centred vertex is
        assert len(settles) == len(events(log, 'corner_go')) >= len(path) - 2, (label, len(settles))   # re-judged)
        over = []
        for e in settles:
            c = world.flu(e['corner'])[:2]
            i = int(np.argmin([np.hypot(cf - c[0], cl - c[1]) for cf, cl in path]))
            d = np.array(path[i]) - np.array(path[i - 1])
            d /= np.linalg.norm(d)
            near = np.hypot(q[:, 0] - c[0], q[:, 1] - c[1]) < 0.5
            over.append(float(max(0.0, ((q[near, :2] - c) @ d).max())))
        assert max(over) <= (0.13 if 'ignored' in label else 0.06), (label, over)
        wall_d = min(world.clearance_xy(r['p']) for r in cor)          # body centre -> nearest board face
        assert wall_d >= Params().prop_radius_m + 0.05, (label, wall_d)
        # feed-forward never toward a wall closer than its stop distance + props + margin (true geometry)
        for r in cor:
            o = r['out']
            sv = float(np.linalg.norm(o.vel_ned[:2]))
            if sv < 0.1:
                continue
            dw = world.dmap.true_ahead(r['p'], np.r_[o.vel_ned[:2] / sv, 0.0], r=Params().prop_radius_m)
            assert sv * sv / (2 * Params().ff_stop_decel) + Params().prop_radius_m <= dw + 0.02, (label, r['t'], sv, dw)
        out.append('%s: corridor %.1f s, %d tight corner(s) overshoot <= %.3f m, nearest board %.2f m, %d hold(s), '
                   '%d end clip(s), nearest wall ahead while moving toward it %.2f m'
                   % (label, t_cor, len(tight), max(over), wall_d, len(holds), len(clips), worst))
    print('  replica maze (layout.py MAZE_WALLS, depth-like map): every corner flown as an L; ' + '; '.join(out))


# v3 replica maze (official-compat report 16-18; layout.py is v4 now): (name, (f0, l0), (f1, l1), banner?)
V3_WALLS = (('lane_a_left', (0.40, 0.60), (1.95, 0.60), True), ('divider_ab', (-0.20, -0.50), (1.10, -0.50), True),
            ('back_wall', (1.95, 0.60), (1.95, -2.00), False), ('divider_bc', (1.95, -1.60), (0.75, -1.60), False))
V3_FENCE = dict(forward_min=-0.75, forward_max=5.35, left_min=-5.2, left_max=0.9, up_max=2.9)


def _depth_walls(walls, extra=0.0):
    return tuple((f0, l0, f1, l1) + ((0.45, 1.75, n, 0.025 + extra) if banner else (0.35, 1.70, n, 0.01 + extra))
                 for n, (f0, l0), (f1, l1), banner in walls)


def _overshoot(q, corner, d_in):
    near = np.hypot(q[:, 0] - corner[0], q[:, 1] - corner[1]) < 0.6
    return float(max(0.0, ((q[near, :2] - corner) @ d_in).max())) if near.any() else 0.0


def test_narrow_lane_seed6_centring():
    """official-compat 194126 (stereo + tape seed 6): corridor point 1 at (0.29, 0.37) sat 0.2 m from the
    lane_a_left face and made a fake 69 deg corner after a 0.47 m leg; the corner hold cycled 8x with setpoints
    0.22-0.36 m from the board -> 7 prop strikes.  Now the prior vertices are centred on the measured lane
    (>= 0.40 m from boards), the kink is no corner, and every real corner stops on the point (PX4 lag model)."""
    corridor = [(0.29, 0.37), (1.55, 0.10), (1.50, -1.00), (0.30, -1.10), (0.20, -2.30), (2.30, -2.30)]
    pr = prior_dict(fence=V3_FENCE, corridor=[dict(forward=f, left=l, up=1.0) for f, l in corridor])
    pr['tunnel'] = dict(entry=dict(forward=4.1, left=-2.3, up=1.0), axis_deg=0.0, length_m=1.2, tol_m=0.5)
    scn = Scenario(name='seed6_lane', prior=pr, default_course=False, depth_walls=_depth_walls(V3_WALLS), t_max=90,
                   vehicle_r=Params().prop_radius_m, px4_mpc=True)
    m, px4, world, log = run(scn, Params(stop_after='corridor'))
    assert m.sub == 'stopped_after_corridor', (phase_seq(log), m.abort_reason)
    assert_no_collision(world, log, skip=())
    cen = events(log, 'waypoint_centred')
    c1 = [e for e in cen if e['index'] == 1]
    prior_clear = world.clearance_xy(world.sf.to_ned((corridor[0][0], corridor[0][1], 1.0)), skip=())
    final_clear = world.clearance_xy(np.asarray(c1[-1]['to']), skip=()) if c1 else 0.0
    assert prior_clear < 0.3 and c1 and final_clear >= Params().lane_clear_m - 0.03, (prior_clear, final_clear, cen)
    v1 = world.flu(c1[-1]['to'])[:2]
    # the kink: no hold / release cycling there any more (194126: 8 cycles), and it is never flown at the prior point
    stops = events(log, 'tight_corner') + events(log, 'stop_corner')
    settles = [world.flu(e['corner'])[:2] for e in events(log, 'corner_settle')]
    assert not [c for c in settles if np.hypot(*(c - np.array(corridor[0]))) < 0.1], settles
    stops = [e for e in stops if any(np.hypot(*(world.flu(e['corner'])[:2] - c)) < 0.05 for c in settles)]
    holds = [e for e in events(log, 'vio_corner_hold') if np.hypot(*(world.flu(e['corner'])[:2] - v1)) < 0.2]
    assert len(holds) <= 1, holds
    cor = [r for r in log if r['phase'] == 'corridor']
    q = np.array([world.flu(r['p']) for r in cor])
    wall_d = min(world.clearance_xy(r['p']) for r in cor)
    assert wall_d >= Params().prop_radius_m + 0.05, wall_d
    sp_d = min(world.clearance_xy(r['out'].pos_ned) for r in cor if r['out'].pos_ned is not None)
    assert sp_d >= Params().radius - 0.02, sp_d                 # setpoints keep the planning radius
    over = []
    gos = events(log, 'corner_go')
    for e in events(log, 'corner_settle'):                      # every real corner: overshoot <= 0.1 m
        c = world.flu(e['carrot'])[:2]                          # (the stop point, incl. a side-step offset)
        before = [world.flu(r['p'])[:2] for r in cor if e['t'] - 0.8 <= r['t'] <= e['t'] - 0.3]
        d = before[-1] - before[0]                              # the leg actually flown into the corner
        t_go = min([g['t'] for g in gos if g['t'] >= e['t']] or [math.inf])        # (after go: the next leg)
        qq = np.array([world.flu(r['p']) for r in cor if e['t'] - 1.5 <= r['t'] <= t_go])
        over.append(_overshoot(qq, c, d / np.linalg.norm(d)))
    assert stops and max(over) <= 0.10, over
    print('  seed-6 lane: point 1 centred %.2f -> %.2f m from the board, no hold/release cycling; %d corners stop '
          '(overshoot <= %.3f m); nearest board %.2f m (setpoints %.2f m), no contact'
          % (prior_clear, final_clear, len(stops), max(over), wall_d, sp_d))


def test_contact_hover_backoff_replan():
    """A board the depth never saw (glass-like) is hit: IMU spike -> hover, back off along where we came from,
    replan (the board is in the map now) -> side step and on through the course."""
    glass = ((0.75, -0.12, 0.0), (0.85, 0.12, 2.2), 'glass')
    scn = Scenario(name='contact', invisible_boxes=(glass,), contact_physics=True, contact_reveal=True)
    m, px4, world, log = run(scn)
    assert px4.contacts and px4.contacts[0][1] == 'glass', px4.contacts
    ce = events(log, 'contact')
    assert ce and ce[0]['t'] - px4.contacts[0][0] <= 0.1, (ce, px4.contacts)
    back = world.sf.from_ned(np.asarray(ce[0]['pos']) + np.asarray(ce[0]['backoff_dir'])) - world.flu(ce[0]['pos'])
    assert back[0] < -0.15, back                                # backs off the way it came (-forward)
    assert events(log, 'contact_replan')
    assert len(px4.contacts) <= 2, px4.contacts
    assert phase_seq(log) == list(PHASES), (phase_seq(log), m.abort_reason)
    t_rep = events(log, 'contact_replan')[0]['t']
    hover = [r for r in log if ce[0]['t'] <= r['t'] < ce[0]['t'] + Params().contact_hover_s]
    assert all(np.linalg.norm(r['out'].vel_ned) < 1e-6 for r in hover)
    print('  contact: hit at %.2f s, detected %.2f s later, hovered %.1f s, backed off %.2f m along -forward, '
          'replanned at %.2f s, %d contact(s) in all, course done'
          % (px4.contacts[0][0], ce[0]['t'] - px4.contacts[0][0], Params().contact_hover_s,
             Params().contact_backoff_m, t_rep, len(px4.contacts)))


def test_corner_phantoms_in_front_of_net():
    """official-compat v4 round A seeds 1/2: stereo phantoms 0.35-0.55 m in front of the north net held the first
    north corner (0.46 m from the net) forever -> corridor_blocked.  Clusters within fence_phantom_m of a fence face
    neither hold nor block after the nudge: we go on slowly, the fence prior keeps us off the net."""
    from replica.layout import MAZE_WALLS, TAPE, START, CAGE
    walls = tuple((n, a, b, skin == 'banner') for n, a, b, skin, _ in MAZE_WALLS)
    path = dict(TAPE)['maze_path']
    fence = dict(forward_min=-START[0], forward_max=CAGE - START[0], left_min=-(CAGE - START[1]),
                 left_max=START[1], up_max=2.9)
    pr = prior_dict(fence=fence, corridor=[dict(forward=f, left=l, up=1.0) for f, l in path[1:]])
    pr['tunnel'] = dict(entry=dict(forward=path[-1][0] + 1.8, left=path[-1][1], up=1.0), axis_deg=0.0,
                        length_m=1.2, tol_m=0.5)
    fmax = fence['forward_max']
    ph = (((fmax - 0.42, 0.0, 1.0), 40, 0.06), ((fmax - 0.40, -1.15, 1.0), 40, 0.06))
    out = []
    for label, prm in (('new', Params(stop_after='corridor')),
                       ('no fence rule', Params(stop_after='corridor', fence_phantom_m=0.0))):
        scn = Scenario(name='net_phantoms', prior=pr, default_course=False, depth_walls=_depth_walls(walls),
                       depth_phantoms=ph, t_max=90, vehicle_r=Params().prop_radius_m)
        m, px4, world, log = run(scn, prm)
        out.append((label, m, log, world))
    label, m, log, world = out[0]
    assert m.sub == 'stopped_after_corridor', (phase_seq(log), m.abort_reason)
    assert_no_collision(world, log, skip=())
    fc = events(log, 'fence_cluster')
    cor = [r for r in log if r['phase'] == 'corridor']
    t_cor = cor[-1]['t'] - cor[0]['t']
    m0 = out[1][1]
    assert m0.sub != 'stopped_after_corridor', 'the old rule should dead-lock at the north corner'
    print('  net phantoms: %d fence cluster(s) passed after the nudge, corridor %.1f s, no contact; without the '
          'rule: %s' % (len(fc), t_cor, m0.abort_reason))


def test_net_phantom_at_takeoff_not_structure():
    """Gazebo v4C (stereo, warped worlds, blind priors): 10/10 never left the start.  A west-net phantom 0.5-0.6 m
    dead ahead, 0.45 m from the net, was 'confirmed' by the vertical takeoff climb, judged structure (reaches_inward),
    kept -> vertical nudge -> blocked -> corridor_blocked 24.5 s after arming.  Climbing does not carve net phantoms:
    only HORIZONTAL camera motion (>= 0.25 m) or a >= 10 deg bearing change confirms; the nudge that cannot go
    sideways becomes an observation manoeuvre (+-20 deg yaw scan, 0.2 m climb).  Here the phantom (9 points 0.45 m from
    the net in our path, joined by a streak reaching 0.86 m in) lives until the camera has moved 0.2 m sideways."""
    from replica.layout import START
    out = {}
    for label, over in (('new', {}), ('climb confirms', dict(confirm_horizontal=False))):
        pr, scn = _venue_world('gate_b')
        pr.pop('walls', None)                                # (like the v4C sim priors: no wall lines)
        scn.prior = pr
        lmax = pr['fence']['left_max']
        ph = (((0.6, lmax - 0.45, 0.8), 9, 0.04, 0.2),) + tuple(   # 9 points 0.45 m from the net (v4C: 10) ...
            ((0.6, lmax - 0.58 - 0.14 * k, 1.18), 3, 0.03, 0.2) for k in range(3))   # + a streak to 0.86 m inward
        scn.depth_phantoms, scn.t_max, scn.px4_mpc = ph, 200, True
        m, px4, world, log = run(scn, Params(stop_after='corridor', **over))
        t_end = world.dmap.ph_vanished[0] if world.dmap.ph_vanished else math.inf
        ev = [e['kind'] for e in events(log) if e['phase'] == 'corridor' and e['t'] < t_end]
        t_n = min([e['t'] for e in events(log, 'parallax_nudge')] or [t_end])
        ev_pre = [e['kind'] for e in events(log) if e['phase'] == 'corridor' and e['t'] <= t_n]   # (the climb alone)
        out[label] = (m.sub or m.abort_reason, 'net_cluster_is_structure' in ev_pre, 'nudge_vertical_only' in ev,
                      world.dmap.ph_vanished[0] if world.dmap.ph_vanished else None, log, world, 'blocked' in ev)
    sub, structure, scan, t_gone, log, world, blk = out['new']
    assert sub == 'stopped_after_corridor', (sub, phase_seq(log))
    assert not structure and not blk and scan and t_gone is not None, (structure, blk, scan, t_gone)
    assert_no_collision(world, log, skip=())
    assert out['climb confirms'][1], out['climb confirms'][:4]   # = the v4C verdict (the climb confirmed it; the
    # re-confirmation sidestep (v4D) now rescues even that, so it is no longer blocked)
    t_cor = first(log, lambda r: r['phase'] == 'corridor')['t']
    print('  net phantom at takeoff: not structure, observation scan (+-%.0f deg, %.1f m up), ignored, carved %.1f s '
          'into the corridor, corridor done; climb counted as a viewpoint (v4C): structure (reaches_inward)'
          % (Params().observe_scan_deg, Params().observe_up_m, t_gone - t_cor))


def test_small_cluster_reconfirm_lateral_parallax():
    """Gazebo v4D: 7/10 stuck at the start behind a 7-10 point stereo mismatch 0.55 m dead ahead in the middle of the
    1.13 m start lane (0.61 m from the west net: not a net phantom; no gap the 0.35 m probe could pass on either side);
    the sides were never seen, so no sidestep, and vertical parallax could not remove it.  Now a kept small cluster
    that still blocks after the nudge gets observe -> sidestep -> re-confirm: yaw toward a side until it is seen free,
    sidestep 0.2-0.3 m there (horizontal baseline), watch the cluster.  Gone = phantom (ignored, back on the path);
    still there after > the map memory = real (normal blocked handling)."""
    out = {}
    for label, vanish in (('phantom', 0.2), ('real', None)):
        pr, scn = _venue_world('gate_b')
        pr.pop('walls', None)
        scn.prior = pr
        col = tuple(((0.55 + 0.116, -0.08, 0.62 + 0.12 * k), 2, 0.02) + ((vanish,) if vanish else ())
                    for k in range(4))                     # a 0.4 m tall column: vertical support -> kept
        scn.depth_phantoms, scn.t_max, scn.px4_mpc = col, 200, True
        m, px4, world, log = run(scn, Params(stop_after='corridor', start_advance_m=0.0))   # (v4K: with the start
        ev = {k: events(log, k) for k in ('reconfirm_start', 'reconfirm_sidestep', 'phantom_cleared', 'cluster_real')}
        # advance a kept cluster this close becomes a sticky obstacle instead: test_start_advance_off_the_take_off_point)
        out[label] = (m, log, world, ev)
    m, log, world, ev = out['phantom']
    assert m.sub == 'stopped_after_corridor', (phase_seq(log), m.abort_reason)
    assert ev['reconfirm_start'] and ev['reconfirm_sidestep'] and ev['phantom_cleared'], ev
    assert_no_collision(world, log, skip=())
    step = abs(ev['reconfirm_sidestep'][0]['offset_m'])
    assert step >= Params().reconfirm_min_step_m - 1e-6, step
    m2, log2, world2, ev2 = out['real']
    assert ev2['cluster_real'] and not ev2['phantom_cleared'], ev2
    assert_no_collision(world2, log2, skip=())
    t_cl = ev['phantom_cleared'][0]['t'] - ev['reconfirm_start'][0]['t']
    print('  small cluster 0.55 m ahead at the start: observe -> sidestep %.2f m -> phantom cleared %.1f s after the '
          'start of the sequence, corridor done; persistent twin: confirmed real (%s), no contact'
          % (step, t_cl, m2.sub or m2.abort_reason))


def test_floating_cluster_vs_board_end_column():
    """Vertical support (user-approved): every real obstacle here stands on the floor.  A small cluster whose connected
    points span < 0.3 m vertically is a floating phantom candidate and must not block on its own (slow pass, re-checked
    while ignored); an edge-on 2 cm board end is a floor-to-top column of 6-11 points and must still block.  Both
    0.55 m ahead of the camera in the middle of the start lane, outside the net band, no prior wall lines."""
    out = {}
    for label in ('floating', 'board end'):
        pr, scn = _venue_world('gate_b')
        pr.pop('walls', None)
        scn.prior = pr
        if label == 'floating':
            scn.depth_phantoms = (((0.55 + 0.116, -0.08, 0.8), 8, 0.04),)          # persistent, never carved
        else:                                                                      # the edge-on end of a 2 cm board:
            scn.depth_walls = scn.depth_walls + ((0.666, -0.08, 0.72, -0.08, 0.35, 1.2, 'thin_end', 0.01, 'a'),)
        scn.t_max, scn.px4_mpc = 200, True                                         # a column, 0.35-1.2 m
        m, px4, world, log = run(scn, Params(stop_after='corridor'))
        out[label] = (m, log, world)
    m, log, world = out['floating']
    assert m.sub == 'stopped_after_corridor', (phase_seq(log), m.abort_reason)
    assert events(log, 'floating_cluster') and not events(log, 'blocked'), events(log)
    assert_no_collision(world, log, skip=())
    m2, log2, world2 = out['board end']
    assert m2.sub != 'stopped_after_corridor' and not events(log2, 'floating_cluster'), (phase_seq(log2), m2.abort_reason)
    if Params().start_advance_m > 0:                 # (v4K: the start advance took the camera into its blind zone ->
        assert events(log2, 'start_sticky') and events(log2, 'start_sticky_blocked'), [e['kind'] for e in events(log2)]
    assert_no_collision(world2, log2, skip=())
    cor = [r for r in log2 if r['phase'] == 'corridor']
    near = min(float(np.hypot(*(world2.flu(r['p'])[:2] - np.array([0.666, -0.08])))) for r in cor)
    assert near >= Params().prop_radius_m + 0.05, near
    print('  floating 8-point cluster 0.55 m ahead: floating_cluster -> slow pass, corridor done; edge-on 2 cm board '
          'end (column) at the same spot: kept real (%s), closest %.2f m, no contact' % (m2.abort_reason or m2.sub, near))


def test_floor_bridged_phantom_is_floating():
    """Gazebo v4G (4755afa4): all 44 kept clusters were 'supported'; the phantoms' support ran from the cluster at
    camera height down to the FLOOR (h 0.00-0.13 m) through ground noise / a sloped stereo streak.  Floor points are
    no support now, and support must be a vertical column (one 0.15 m cell spanning >= 0.3 m): a 6-point phantom
    0.63 m ahead at 0.94 m, tied by a sloped streak to a patch of floor noise, is floating -> slow pass, corridor
    done; the edge-on board end at the same spot is still kept real.  (Old rule, floor + span: kept -> blocked.)"""
    def scene(over, kind):
        pr, scn = _venue_world('gate_b')
        pr.pop('walls', None)
        scn.prior = pr
        c = np.array([0.75, -0.08, 0.94])
        e = np.array([1.25, 0.25, 0.04])                  # (sloped down and toward the west side, short of the turn)
        if kind == 'streak':
            ph = [(tuple(c), 6, 0.04)]
            ph += [(tuple(c + (e - c) * k / 10.0), 1, 0.01) for k in range(1, 11)]       # 1 point per 0.11 m
            ph += [((f, l, 0.03), 1, 0.02) for f in (1.1, 1.25) for l in (0.1, 0.25)]   # floor noise
            scn.depth_phantoms = tuple(ph)
        else:
            scn.depth_walls = scn.depth_walls + ((0.75, -0.08, 0.80, -0.08, 0.35, 1.2, 'thin_end', 0.01, 'a'),)
        scn.t_max, scn.px4_mpc = 200, True
        return run(scn, Params(stop_after='corridor', **dict(dict(start_advance_m=0.0), **over)))   # (the support
    out = {}                                                                                      # rule only)
    for label, over, kind in (('new', {}, 'streak'), ('old', dict(support_floor_up_m=None, support_cell_m=None), 'streak'),
                              ('board', {}, 'board')):
        m, px4, world, log = scene(over, kind)
        out[label] = (m, log, world)
    m, log, world = out['new']
    assert m.sub == 'stopped_after_corridor', (phase_seq(log), m.abort_reason)
    fl = events(log, 'floating_cluster')
    assert fl and not events(log, 'small_cluster_kept'), [e['kind'] for e in events(log)][:40]
    assert_no_collision(world, log, skip=())
    m0, log0, _ = out['old']
    kept0 = events(log0, 'small_cluster_kept')
    assert kept0 and kept0[0]['support']['why'] == 'supported', [e['kind'] for e in events(log0)][:40]
    mb, logb, worldb = out['board']
    assert not events(logb, 'floating_cluster') and mb.sub != 'stopped_after_corridor', (phase_seq(logb), mb.abort_reason)
    assert_no_collision(worldb, logb, skip=())
    sup = fl[0].get('support') or {}
    print('  floor-bridged 6-point phantom 0.63 m ahead: floating (column %s m, %s floor points dropped) -> corridor '
          'done; old rule: kept (span %.2f m) -> %s; board end at the same spot: kept real (%s)'
          % (sup.get('column_m'), sup.get('floor_pts_dropped'), kept0[0]['support']['span_m'],
             m0.abort_reason or m0.sub, mb.abort_reason or mb.sub))


def test_prior_wall_band_needs_column():
    """Gazebo v4J (7d841286, blind2 priors with wall lines 0.3 m off): every kept cluster was 'prior_wall_line' --
    the band r + wall_line_keep_m + tol (~1 m) covers a whole 1.1 m lane -- so the column check never ran and two
    phantoms 0.34 m from W1 that fell on the offset W1 line blocked.  Now the column check decides near prior wall
    lines too: the prior W1 line 0.3 m into the start lane, a 6-point phantom 0.32 m off W1's face (on that line)
    -> floating, corridor done; an edge-on board end at the same spot -> still real, no contact.  (Old:
    prior_line_real=True -> the phantom is kept.)"""
    def scene(over, kind):
        pr, scn = _venue_world('gate_b')
        pr = json.loads(json.dumps(pr))
        for w in pr['walls']:
            if w['name'] == 'W1':
                w['from']['left'] += 0.3
                w['to']['left'] += 0.3
        scn.prior = pr
        if kind == 'phantom':
            scn.depth_phantoms = (((0.8, -0.25, 0.9), 6, 0.04),)
        else:
            scn.depth_walls = scn.depth_walls + ((0.8, -0.25, 0.85, -0.25, 0.35, 1.2, 'thin_end', 0.01, 'a'),)
        scn.t_max, scn.px4_mpc = 200, True
        return run(scn, Params(stop_after='corridor', **over))
    out = {}
    for label, over, kind in (('new', {}, 'phantom'), ('old', dict(prior_line_real=True), 'phantom'), ('board', {}, 'board')):
        out[label] = scene(over, kind)
    m, px4, world, log = out['new']
    assert m.sub == 'stopped_after_corridor', (phase_seq(log), m.abort_reason)
    fl = events(log, 'floating_cluster')
    assert fl and (fl[0].get('support') or {}).get('prior_wall_line'), [e['kind'] for e in events(log)][:30]
    assert_no_collision(world, log, skip=())
    m0, _, _, log0 = out['old']
    k0 = events(log0, 'small_cluster_kept')
    assert k0 and k0[0]['support']['why'] == 'prior_wall_line', [e['kind'] for e in events(log0)][:30]
    mb, _, worldb, logb = out['board']
    assert not events(logb, 'floating_cluster') and mb.sub != 'stopped_after_corridor', (phase_seq(logb), mb.abort_reason)
    assert_no_collision(worldb, logb, skip=())
    print('  phantom on the offset prior W1 line (0.32 m off the face): floating (column %s m) -> corridor done; old: '
          'kept (prior_wall_line) -> %s; board end there: real (%s), no contact'
          % ((fl[0].get('support') or {}).get('column_m'), m0.abort_reason or m0.sub, mb.abort_reason or mb.sub))


def test_corner_vertex_keeps_net_clearance():
    """Gazebo v4J map s2 / s4 (7d841286): wall-relative centring put the first corner 0.40 m from the west / north
    net (the largest min clearance between the net and nearby map points); a 0.15 m overshoot or 0.23 m of VIO error
    then broke 0.28 m.  Vertex moves now keep vertex_fence_clear_m (0.55) from the fence, or at least no closer than
    the vertex already is.  Here a post near W1's north end squeezes the first corner.  (Old: 0.40 m from the net.)"""
    def scene(over):
        pr, scn = _venue_world('gate_b')
        pr.pop('walls', None)
        scn.prior = pr
        scn.depth_walls = scn.depth_walls + ((1.45, -0.25, 1.48, -0.25, 0.0, 1.8, 'post', 0.02, 'a'),)
        scn.t_max, scn.px4_mpc = 120, True
        return pr, run(scn, Params(stop_after='corridor', **over))
    out = {}
    for label, over in (('new', {}), ('old', dict(vertex_fence_clear_m=0.0))):
        pr, (m, px4, world, log) = scene(over)
        F = pr['fence']
        fd = lambda q: min(F['forward_max'] - world.flu(q)[0], F['left_max'] - world.flu(q)[1])
        moves = [fd(np.asarray(e['to'])) for e in events(log, 'waypoint_centred') if e['index'] == 1]
        cor = [r for r in log if r['phase'] == 'corridor']
        out[label] = (min(moves) if moves else math.inf, min(fd(r['p']) for r in cor), world, log)
    vf, net, world, log = out['new']
    assert vf >= Params().standoff_fence_clear_m - 0.02, vf
    assert_no_collision(world, log, skip=())
    vf0, net0 = out['old'][0], out['old'][1]
    assert vf0 < 0.45 and net >= net0, (vf0, net, net0)
    print('  first corner squeezed by a post: vertex >= %s m from the net (old: moved to %.2f m), closest %.3f m '
          '(old %.3f m)' % ('%.2f' % vf if math.isfinite(vf) else 'unmoved 0.50', vf0, net, net0))


def _box_dist(world, p_ned, name):
    """Horizontal distance from the body centre to the named box (inf if there is none)."""
    q = world.flu(p_ned)
    d = [float(np.linalg.norm(np.maximum(np.maximum(lo[:2] - q[:2], 0), q[:2] - hi[:2])))
         for lo, hi, n in world.boxes if n == name]
    return min(d) if d else math.inf


def test_net_layer_is_no_vertical_support():
    """Gazebo v4E (53d6f3e9, 0/10 through the maze): the drone now left the start, then stopped in the first corridor
    behind small clusters 1.0-1.6 m ahead (eyeball s3: 6 points at NED (1.34, 0.06), 0.8 m from the north net, 0.59 m
    from the west net -- outside the 0.5 m net band): net stereo ghosts that the vertical-support rule called
    'supported' because their 3-D component ran (through a ghost streak) into the net itself, floor to top.  The net
    layer (<= net_layer_m from a fence face) is never support or structure: ghost + streak span < 0.3 m -> floating
    -> slow pass.  Here: the west net, a ghost 0.55 m from it in the start lane + a streak into the net.  (Old rule,
    net points as support: 'supported' -> kept -> blocked.)"""
    out = {}
    for label, over in (('new', {}), ('net supports', dict(net_layer_m=-1.0))):
        pr, scn = _venue_world('gate_b')
        pr.pop('walls', None)
        scn.prior = pr
        lmax = pr['fence']['left_max']
        net = (-0.40, lmax + 0.03, 2.0, lmax + 0.03, 0.0, 2.4, 'west_net', 0.005)          # the net, floor to top
        scn.depth_walls = scn.depth_walls + (net,)
        g = (1.0, lmax - 0.55)
        streak = tuple(((g[0], g[1] + 0.12 * k, 0.9 + 0.03 * k), 2, 0.02) for k in range(1, 6))   # into the net
        scn.depth_phantoms = (((g[0], g[1], 0.9), 6, 0.04),) + streak                       # persistent ghosts
        scn.t_max, scn.px4_mpc = 200, True
        m, px4, world, log = run(scn, Params(stop_after='corridor', **over))
        kept = [e for e in events(log, 'small_cluster_kept')
                if e.get('support') and np.linalg.norm(world.flu(e['center'])[:2] - np.array(g)) < 0.3]
        out[label] = (m.sub or m.abort_reason, log, world, kept)
    sub, log, world, kept = out['new']
    assert sub == 'stopped_after_corridor', (sub, phase_seq(log))
    assert not kept and events(log, 'floating_cluster') and not events(log, 'blocked'), \
        (kept, [e['kind'] for e in events(log)][:40])
    assert_no_collision(world, log, skip=())
    sub0, log0, world0, kept0 = out['net supports']
    assert kept0 and kept0[0]['support']['why'] == 'supported', (sub0, kept0)
    print('  ghost 0.55 m from the west net (+ a streak into it): net layer is no support -> floating -> slow pass, '
          'corridor done; with the net as support: kept (span %.2f m) -> %s' % (kept0[0]['support']['span_m'], sub0))


def test_net_ghosts_do_not_move_vertices():
    """v4E: net ghosts in the first corridor's north turn (the 0.94 m gap between W1's north end and the north net is
    full of them 0.4-1.1 m from the net) were 'boards' for lane centring / gap vias: the turn vertex moved toward W1's
    end and the turn leg ran into it.  Lane centring now uses the blocking criteria: no net layer, no floating /
    tiny / ignored clusters -- only points whose footprint spans >= floating_span_m.  A ghost 0.8 m from the north
    net in the lane + a ghost streak into the net, right at the turn.  (Old: ghosts as boards -> the vertex moved,
    corridor_blocked at W1's end.)"""
    out = {}
    for label, over in (('new', {}), ('ghosts as boards', dict(board_support_filter=False))):
        pr, scn = _venue_world('gate_b')
        pr.pop('walls', None)
        scn.prior = pr
        fmax, lmax = pr['fence']['forward_max'], pr['fence']['left_max']
        scn.depth_walls = scn.depth_walls + ((fmax + 0.03, lmax, fmax + 0.03, lmax - 2.3, 0.0, 2.4, 'north_net', 0.005),)
        streak = tuple(((f, -0.06, 0.9 + 0.04 * k), 2, 0.02) for k, f in enumerate((1.5, 1.65, 1.8, 1.95, 2.08)))
        scn.depth_phantoms = (((1.34, -0.06, 0.9), 6, 0.04),) + streak
        scn.t_max, scn.px4_mpc = 200, True
        m, px4, world, log = run(scn, Params(stop_after='corridor', **over))
        cor = [r for r in log if r['phase'] == 'corridor']
        w1 = min(_box_dist(world, r['p'], 'W1') for r in cor)
        out[label] = (m.sub or m.abort_reason, log, world, w1)
    sub, log, world, w1 = out['new']
    assert sub == 'stopped_after_corridor', (sub, phase_seq(log), [e['kind'] for e in events(log)][:40])
    assert not events(log, 'blocked') and w1 >= Params().lane_clear_m - 0.05, (w1, events(log, 'blocked'))
    assert_no_collision(world, log, skip=())
    sub0 = out['ghosts as boards'][0]
    assert sub0 != 'stopped_after_corridor', sub0
    print('  north-turn net ghosts: not boards -> turn vertex stays in the gap, W1 >= %.2f m, corridor done; ghosts '
          'as boards: %s' % (w1, sub0))


def test_tiny_cluster_never_blocks():
    """v4E: kept 'small' clusters of n = 1-3 started re-confirmations / blocked.  Fewer than never_block_n = 3 map
    points never block on their own: slow pass + nudge, then ignored.  A persistent 2-point pair 0.55 m dead ahead in
    the 1.1 m start lane (vertical-support rule off in both runs, so only this rule decides).  (Old: kept ->
    re-confirmed real -> blocked.)"""
    out = {}
    for label, nb in (('new', 3), ('old', 0)):
        pr, scn = _venue_world('gate_b')
        pr.pop('walls', None)
        scn.prior = pr
        scn.depth_phantoms = (((0.666, -0.08, 0.72), 1, 0.0), ((0.666, -0.08, 0.86), 1, 0.0))
        scn.t_max, scn.px4_mpc = 200, True
        m, px4, world, log = run(scn, Params(stop_after='corridor', floating_span_m=0.0, never_block_n=nb))
        out[label] = (m.sub or m.abort_reason, log, world)
    sub, log, world = out['new']
    assert sub == 'stopped_after_corridor', (sub, phase_seq(log))
    assert events(log, 'tiny_cluster') and not events(log, 'blocked') and not events(log, 'reconfirm_start'), \
        [e['kind'] for e in events(log)][:40]
    assert_no_collision(world, log, skip=())
    sub0, log0, _ = out['old']
    assert events(log0, 'small_cluster_kept') and sub0 != 'stopped_after_corridor', (sub0, phase_seq(log0))
    print('  2-point cluster 0.55 m ahead: tiny -> slow pass, corridor done; old rule: kept -> %s' % sub0)


def test_observation_hover_keeps_rear_clearance():
    """Gazebo v4E: observations at the start held at the take-off point, 0.36-0.44 m in front of the barrier behind
    it (never seen); hold drift during the yaw scans took the airframe to 0.24-0.27 m (< 0.28).  Setpoints never went
    behind the take-off point.  Now an observation hover point needs move_clear_m (0.38) known free behind it (own
    track / camera view) and from map points / fence: it moves forward along the path first (camera
    kept out of the stereo blind zone of what it watches).  Barrier 0.30 m behind the start, a kept column 0.95 m
    ahead (a phantom that a sideways baseline removes); sidesteps keep 0.38 m from W1 too.  (Old: no advance.)"""
    from replica.layout import MAZE_WALLS
    out = {}
    for label, over in (('new', dict(start_advance_m=0.0)), ('old', dict(hold_advance_max_m=0.0, start_advance_m=0.0))):
        pr, scn = _venue_world('gate_b')
        pr.pop('walls', None)
        scn.prior = pr
        walls = []
        for w in scn.depth_walls:
            if w[6] == 'barrier':
                w = (-0.30, w[1], -0.30, w[3]) + tuple(w[4:])
            walls.append(w)
        scn.depth_walls = tuple(walls)
        scn.depth_phantoms = tuple(((0.95 + 0.116, -0.08, 0.62 + 0.12 * k), 2, 0.02, 0.2) for k in range(4))
        scn.t_max, scn.px4_mpc = 200, True
        m, px4, world, log = run(scn, Params(stop_after='corridor', **over))
        t0 = min([e['t'] for e in events(log, 'parallax_nudge')] or [math.inf])
        t1 = min([e['t'] for e in events(log) if e['kind'] in ('phantom_cleared', 'cluster_real',
                                                                'reconfirm_no_room', 'reconfirm_timeout')] or [t0 + 20])
        win = [r for r in log if t0 <= r['t'] <= t1]
        bar = min(_box_dist(world, r['p'], 'barrier') for r in win) if win else math.inf
        w1 = min(_box_dist(world, r['p'], 'W1') for r in win) if win else math.inf
        out[label] = (m.sub or m.abort_reason, log, world, bar, w1)
    sub, log, world, bar, w1 = out['new']
    mc = Params().move_clear_m
    assert sub == 'stopped_after_corridor', (sub, phase_seq(log))
    assert events(log, 'hold_advanced') and bar >= mc - 0.02 and w1 >= mc - 0.02, (bar, w1, events(log, 'hold_advanced'))
    assert_no_collision(world, log, skip=())
    bar0 = out['old'][3]
    assert bar0 < mc - 0.02, bar0
    adv = events(log, 'hold_advanced')[0]
    print('  observation hover moved %.2f m forward (rear known free %.2f -> %.2f m): barrier >= %.2f m, W1 >= %.2f m '
          'during observe/sidestep/watch; old: barrier %.2f m' % (float(np.linalg.norm((np.asarray(adv['to']) - np.asarray(adv['from']))[:2])),
                                                                 adv['clear_before_m'], adv['clear_m'], bar, w1, bar0))


def _barrier_start_world(over, wall_to_f, boulder=None, leg_l=-0.55, col_d=0.62, scan=110.0, bar_face=0.35, **scn_over):
    """Take-off point 0.35 m in front of the barrier (venue w1: 0.355 m), which the camera never sees (it collides,
    it is never in the map); first leg 19 deg to the right (eyeball priors), so the re-confirm sidestep's right-hand
    normal points 0.33 back; a kept column (10 points over 0.48 m, a real one) 0.62 m along the leg."""
    fence = dict(forward_min=-1.2, forward_max=3.2, left_min=-1.6, left_max=1.6, up_max=2.9)
    th = math.atan2(-leg_l, 1.6)
    c = np.array([math.cos(th), -math.sin(th)]) * col_d
    pr = prior_dict(fence=fence, cruise_up_m=0.8, corridor=[dict(forward=1.6, left=leg_l, up=0.8)],
                    wall=dict(cross_from=dict(forward=wall_to_f - 0.95, left=0.0), cross_to=dict(forward=wall_to_f, left=0.0),
                              top_up=1.65, tol_m=0.3))
    if boulder is not None:
        pr['boulders'] = [dict(forward=boulder[0], left=boulder[1])]
    pr['tunnel'] = dict(entry=dict(forward=1.6 + 1.25 * math.cos(th), left=leg_l - 1.25 * math.sin(th), up=0.62),
                        axis_deg=math.degrees(-th), length_m=1.2, tol_m=0.5)
    ph = tuple(((c[0], c[1], 0.56 + 0.12 * k), 2, None) for k in range(5))
    scn = Scenario(name='barrier_start', prior=pr, default_course=False, vehicle_r=Params().prop_radius_m,
                   invisible_boxes=(((-bar_face - 0.02, -1.6, 0.0), (-bar_face, 1.6, 1.7), 'barrier'),), phantoms=ph,
                   t_max=80, px4_mpc=True, **scn_over)
    return run(scn, Params(stop_after='corridor', reconfirm_scan_deg=scan, **over))   # (wide scan: the rear-right
                                                                                       # edge is seen, as in eyeball s2)


def test_barrier_behind_start_keepout():
    """Gazebo v4F (4755afa4): 0.264-0.267 m from the barrier behind the start (w1: 0.355 m behind the take-off
    point).  The barrier is the prior `wall` entry (cross_from / cross_to), not a `walls` line or the fence: the
    v4F keep-out never saw it (nor the boulders), and eyeball s2's re-confirm sidestep along a diagonal first leg
    went 0.11 m back toward it.  Now every prior obstacle is in the keep-out -- fence, barrier, boulders, wall lines
    -- plus the never-seen space behind the take-off point: a sidestep / nudge / backoff / pull-back / return target
    keeps move_clear_m, or (already closer, as at this take-off point) gets no closer; returns / pull-backs onto
    our own track are allowed.  The barrier prior 0.5 m too far back (eyeball): the never-seen rear stops the step
    back; an accurate barrier prior alone (rear rule off) stops it too.  (Old: the step went 0.10 m behind the
    take-off point, 0.23 m from the barrier.)"""
    out = {}
    for label, over, wall_to_f in (('rear', dict(start_back_max_m=None), -0.5),
                                   ('prior', dict(start_rear_keep=False, start_back_max_m=None), 0.0),
                                   ('clamp', dict(prior_keepout=False, start_rear_keep=False), -0.5),
                                   ('old', dict(prior_keepout=False, start_rear_keep=False, start_back_max_m=None), -0.5)):
        over = dict(over, start_advance_m=0.0)                       # (the moves' keep-out, not the start advance)
        m, px4, world, log = _barrier_start_world(over, wall_to_f, boulder=(2.4, 1.0))
        cor = [r for r in log if r['phase'] == 'corridor']
        bar = min(_box_dist(world, r['p'], 'barrier') for r in cor)
        spf = min(world.flu(r['out'].pos_ned)[0] for r in cor if r['out'].pos_ned is not None)
        steps = [e['offset_m'] for e in events(log, 'reconfirm_sidestep')]
        out[label] = (m, log, world, bar, spf, steps, _box_dist(world, cor[0]['p'], 'barrier'))
    for label in ('rear', 'prior'):
        m, log, world, bar, spf, steps, bar0 = out[label]
        assert m.sub == 'stopped_after_corridor', (label, phase_seq(log), m.abort_reason)
        assert steps and steps[0] < 0.0, (label, steps)               # the forward (left) side, not back-right
        assert spf >= -0.02 and bar >= bar0 - 0.03, (label, spf, bar, bar0)
        assert_no_collision(world, log, skip=())
    mc, logc, worldc, barc, spfc, stepsc, _ = out['clamp']             # (v4H) the final setpoint clamp alone
    assert spfc >= -Params().start_back_max_m - 1e-6 and barc >= bar0 - 0.07, (spfc, barc)
    assert events(logc, 'start_rear_clip'), stepsc
    m, log, world, bar, spf, steps, bar0 = out['old']
    assert steps and steps[0] > 0.0 and spf < -0.05 and bar < 0.28, (steps, spf, bar)
    # the prior terms themselves: the barrier from `wall` (cross_to 0.35 m from it), a boulder (default 0.23 m radius)
    mp, mr = out['prior'][0], out['rear'][0]
    home = mp.home.copy()
    home[2] -= 0.8
    assert abs(mp._barrier_clear(home) - 0.35) < 0.01 and abs(mr._barrier_clear(home) - 0.85) < 0.01, \
        (mp._barrier_clear(home), mr._barrier_clear(home))
    assert len(mp.prior.boulders) == 1 and mp.prior.boulders[0][2:] == (0.35, 0.23), mp.prior.boulders
    q = mp.sf.to_ned((2.4 - 0.23 - 0.25, 1.0, 0.8))                # 0.25 m from the boulder's side, 0.8 m up
    q_low = mp.sf.to_ned((2.4 - 0.23 - 0.25, 1.0, 0.3))
    assert abs(mp._boulder_clear(q) - (0.8 - 0.35)) < 0.01 and abs(mp._boulder_clear(q_low) - 0.25) < 0.01, \
        (mp._boulder_clear(q), mp._boulder_clear(q_low))
    print('  barrier 0.35 m behind the take-off point, never seen: sidestep %+.2f m (forward side), setpoints >= '
          '%.2f m behind it, barrier >= %.3f m (prior 0.5 m too far: rear rule) / %+.2f m, %.3f m (accurate prior '
          'alone); setpoint clamp alone: held %.2f m behind, barrier %.3f m; old: sidestep %+.2f m, setpoints %.2f m '
          'behind the take-off point, barrier %.3f m'
          % (out['rear'][5][0], max(0.0, -out['rear'][4]), out['rear'][3], out['prior'][5][0], out['prior'][3], -spfc,
             barc, steps[0], -spf, bar))


def test_start_advance_off_the_take_off_point():
    """Coordinator's rule after v4H eyeball s1 (hovered on the take-off point behind a kept cluster 0.6 m ahead, VIO
    drifted 0.14 m back, props hit the barrier): the first observation hover in the start zone moves forward to
    f = min(0.35, cluster - 0.35, seen-free - 0.35) at <= 0.2 m/s, closer to the kept cluster than the camera rule
    allows.  Take-off 0.36 m in front of the never-seen barrier, a persistent real column 0.6 m ahead, the
    estimate drifting 0.15 m forward of the truth (the airframe drifts back).  (Old: hover on the take-off point.)"""
    out = {}
    for label, over in (('new', {}), ('old', dict(start_advance_m=0.0))):
        m, px4, world, log = _barrier_start_world(over, -0.5, leg_l=0.0, col_d=0.6, scan=80.0, bar_face=0.36,
                                                  est_drift=(3.0, 12.0, (0.15, 0.0, 0.0)))
        cor = [r for r in log if r['phase'] == 'corridor']
        bar = min(_box_dist(world, r['p'], 'barrier') for r in cor)
        out[label] = (m, log, world, bar)
    m, log, world, bar = out['new']
    adv = events(log, 'start_advance')
    assert adv, [e['kind'] for e in events(log)][:30]
    f_to = float(world.flu(np.asarray(adv[0]['to']))[0])
    assert 0.15 <= f_to <= 0.35 + 1e-6, f_to
    assert bar >= 0.28, bar
    assert_no_collision(world, log, skip=())
    bar0 = out['old'][3]
    assert bar0 < 0.28 and not events(out['old'][1], 'start_advance'), bar0
    print('  start advance: hover point %.2f m off the take-off point (column 0.6 m ahead), 0.15 m VIO drift back -> '
          'barrier >= %.3f m, no contact; old (hover on the take-off point): %.3f m' % (f_to, bar, bar0))


def test_start_ratchet_keeps_forward_progress():
    """Gazebo v4H map s4 (3ace9551): the observation hover point had moved 0.2 m off the take-off point, then the
    stop-distance pull-back took the carrot 0.09 m back -- allowed, we had flown through there -- and 0.18 m of VIO
    error put the airframe 0.25 m from the barrier.  The final setpoint clamp (takeoff / corridor, within 0.8 m
    sideways of the take-off point): never more than start_back_max_m behind it, and -- until the vehicle first
    gets start_zone_m away -- never back below the most forward commanded position (<= start_ratchet_m).
    Checked on the clamp itself with the setpoint sequence of map s4."""
    m = Mission(CoursePrior.from_dict(prior_dict()), Params())
    m.home = np.zeros(3)
    m.sf = StartFrame(m.home, 0.0)
    m.phase, m.now, m._events = 'corridor', 0.0, []
    m.pos = np.array([0.0, 0.0, -0.8])
    z3 = np.zeros(3)
    def clip(f, l=0.0, vel=(0.0, 0.0, 0.0)):
        q, v, a = m._clip_start_rear(np.array([f, -l, -0.8]), np.array(vel, float), z3.copy())
        return float(q[0]), v
    assert abs(clip(-0.20)[0]) < 1e-9                                # ratchet from the take-off point itself
    assert abs(clip(0.20)[0] - 0.20) < 1e-9                          # hover point moved forward ...
    q, v = clip(0.11, vel=(-0.2, 0.05, 0.0))
    assert abs(q - 0.20) < 1e-9 and v[0] == 0.0 and v[1] == 0.05     # ... the pull-back is held there (map s4)
    assert abs(clip(0.11, l=1.0)[0] - 0.11) < 1e-9                   # outside the start lane: no clamp
    assert abs(clip(0.50)[0] - 0.50) < 1e-9 and abs(clip(0.25)[0] - 0.30) < 1e-9   # ratchet capped at 0.30
    m.pos = np.array([1.2, 0.0, -0.8])                               # left the start zone once ...
    assert abs(clip(0.11)[0] - 0.11) < 1e-9
    m.pos = np.array([0.1, 0.0, -0.8])                               # ... and stays released (only -0.05 holds)
    assert abs(clip(0.0)[0]) < 1e-9 and abs(clip(-0.2)[0] + 0.05) < 1e-9
    assert [e['kind'] for e in m._events] == ['start_rear_clip']
    m2 = Mission(CoursePrior.from_dict(prior_dict()), Params(start_ratchet_m=0.0))
    m2.home, m2.sf, m2.phase, m2.now, m2._events = np.zeros(3), StartFrame(np.zeros(3), 0.0), 'corridor', 0.0, []
    m2.pos = np.array([0.0, 0.0, -0.8])
    m2._clip_start_rear(np.array([0.2, 0.0, -0.8]), z3.copy(), z3.copy())
    assert abs(float(m2._clip_start_rear(np.array([0.11, 0.0, -0.8]), z3.copy(), z3.copy())[0][0]) - 0.11) < 1e-9
    print('  start clamp: from the take-off point -0.20 -> 0.00; after 0.20 forward a pull-back to 0.11 is held at '
          '0.20 (old: 0.11); capped at 0.30; once 1.0 m away only the -0.05 limit stays; outside the start lane untouched')


def _thin_end_uturn(params):
    """U-turn around the END of a thin plywood board (2 cm) whose face returns no depth: only the end column is in
    the map (8 points in the flight band).  Tape-level prior: the crossing leg passes the end at 0.15 m."""
    fence = dict(forward_min=-3.0, forward_max=3.7, left_min=-2.2, left_max=0.8, up_max=2.9)
    board = (-0.6, -0.55, 2.6, -0.55, 0.35, 1.83, 'thin_board', 0.01, 'b')
    pr = prior_dict(fence=fence, corridor=[dict(forward=2.75, left=0.0, up=1.0), dict(forward=2.75, left=-1.1, up=1.0),
                                           dict(forward=0.3, left=-1.1, up=1.0)])
    pr['tunnel'] = dict(entry=dict(forward=0.3 - 1.8, left=-1.1, up=1.0), axis_deg=180.0, length_m=1.2, tol_m=0.5)
    scn = Scenario(name='thin_end_uturn', prior=pr, default_course=False, depth_walls=(board,), t_max=90,
                   vehicle_r=Params().prop_radius_m, px4_mpc=True, contact_physics=True)
    return run(scn, params)


def test_thin_board_end_uturn():
    """official-compat v4 round B seeds 2/3: at the U-turns around W1's north / W2's south end the drone hovered INTO
    the plywood end (2 cm thick, edge-on = 6-11 points): the small-cluster rule ignored it -> 7-10 prop strikes, two
    crashes.  Away from the net such a cluster is real: the U-turn is centred around it, it is never ignored and there
    is no contact.  (Old rule, emulated by declaring the whole arena 'net band': the end is ignored -> strikes.)"""
    end = np.array([2.6, -0.55])
    m, px4, world, log = _thin_end_uturn(Params(stop_after='corridor'))
    assert m.sub == 'stopped_after_corridor', (phase_seq(log), m.abort_reason)
    cor = [r for r in log if r['phase'] == 'corridor']
    near = min(float(np.hypot(*(world.flu(r['p'])[:2] - end))) for r in cor)
    kinds = [e['kind'] for e in events(log) if e['phase'] == 'corridor']
    pts = world.dmap.all
    band = [q for q in pts if abs(world.flu(q)[2] - 1.0) <= 0.45 and np.hypot(*(world.flu(q)[:2] - end)) < 0.05]
    assert len(band) < Params().minor_cluster_n, len(band)       # a 'small' cluster by the old rule
    assert 'minor_ignored' not in kinds and not events(log, 'contact'), kinds
    assert_no_collision(world, log, skip=())
    assert near >= Params().lane_clear_m - 0.05, near
    cen = [e for e in events(log, 'waypoint_centred')]
    m0, _, w0, log0 = _thin_end_uturn(Params(stop_after='corridor', fence_phantom_m=99.0,
                                             board_open_min_pts=Params().minor_cluster_n))
    near0 = min(float(np.hypot(*(w0.flu(r['p'])[:2] - end))) for r in log0 if r['phase'] in FLIGHT_PHASES_T)
    k0 = [e['kind'] for e in events(log0)]
    assert 'minor_ignored' in k0 and (events(log0, 'contact') or near0 < Params().prop_radius_m + 0.01), (k0, near0)
    print('  thin board end at a U-turn (%d points in the band): %d vertex moves, closest %.2f m to the end, no '
          'contact, corridor %.1f s; old rule: ignored -> %d contact(s), closest %.2f m'
          % (len(band), len(cen), near, cor[-1]['t'] - cor[0]['t'], len(events(log0, 'contact')), near0))


class _PtsMap:
    def __init__(self, pts):
        self.points = np.asarray(pts, float)

    def clearance(self, p_from, p_to, radius_m, exclude=None):
        return math.inf


def test_net_board_end_is_structure():
    """The net-phantom tolerance is only for blobs in front of the net: a board that ENDS at the net (W2's north end,
    5 cm from it) is structure once its points were seen from two viewpoints (they reach far past the 0.5 m band), and
    a prior wall line keeps it real even before that; an isolated blob 0.3 m in front of the net stays a phantom.
    Lane centring keeps the board's net end too."""
    fence = dict(forward_min=-1.0, forward_max=3.0, left_min=-2.0, left_max=2.0, up_max=2.9)
    res = {}
    for with_walls in (False, True):
        pr = prior_dict(fence=fence)
        if with_walls:
            pr['walls'] = [dict(**{'from': dict(forward=1.5, left=0.5), 'to': dict(forward=2.95, left=0.5)})]
        m = _mission_at_home(pr)
        sf = m.sf
        board = [sf.to_ned((f, 0.5, u)) for f in np.arange(1.5, 2.951, 0.05) for u in np.arange(0.3, 1.81, 0.05)]
        rng = np.random.default_rng(0)
        blob = [sf.to_ned((2.7, -0.8, 1.0) + rng.uniform(-0.06, 0.06, 3)) for _ in range(30)]
        m.per = Perception(obstacles=_PtsMap(board + blob), obstacles_t=0.0)
        end_c, blob_c = sf.to_ned((2.9, 0.5, 1.0)), sf.to_ned((2.7, -0.8, 1.0))
        north = sf.to_ned((1.0, 0.0, 0.0)) - sf.o

        def view(t, f):
            m.now, m.pos = t, sf.to_ned((f, 0.0, 1.0))
            m.veh = Vehicle(pos_ned=tuple(m.pos), heading=0.0, armed=True, nav_state=NAV_OFFBOARD)
            m._confirm_update()
        view(1.0, 0.6)
        one_view = (m._phantom_ok(end_c, 0.15), m._phantom_ok(blob_c, 0.1))
        view(2.0, 1.0)                                        # moved 0.4 m: points still there = not carved
        m._phantom_memo.clear()
        two_views = (m._phantom_ok(end_c, 0.15), m._phantom_ok(blob_c, 0.1))
        assert two_views == (False, True), (with_walls, two_views)
        assert not m._fence_phantom(end_c, north) and m._fence_phantom(blob_c, north)
        if with_walls:
            assert one_view == (False, True), one_view           # prior wall line: real from the first sight
        else:
            assert one_view == (True, True), one_view            # one viewpoint: not yet confirmed
        B = m._boards_xy(np.array([-3.5, -3.5]), np.array([3.5, 3.5]), end_c[2])
        f_of = lambda q: (q - sf.o[:2]) @ sf.fwd[:2]              # noqa: E731
        fb = [f_of(q) for q in B]
        assert max(fb) > 2.85 and not any(abs(f_of(q) - 2.7) < 0.1 and abs((q - sf.o[:2]) @ sf.left[:2] + 0.8) < 0.1
                                           for q in B), max(fb)
        ev = [e for e in m._events if e['kind'] == 'net_cluster_is_structure']
        res[with_walls] = (one_view[0], ev[0]['why'] if ev else 'prior wall line')
    print('  board end at the net: no prior wall -> discountable after one view (%s), structure after two (%s); with '
          'a prior wall line -> real at once; blob in front of the net stays a phantom; lane centring keeps the '
          'board to %.2f m from the net' % (res[False][0], res[False][1], 3.0 - max(fb)))


def test_slow_tunnel_detector_latches():
    """official-compat v4 round B: the new perception/tunnel.py (conf 0.52-0.54, centre error 0.13 m) gives one frame
    every ~1.9 s, 0.7-1.1 s after its image (stamp = image time).  Every frame arrived older than detect_max_age_s
    (0.5 s) and was dropped -> no latch -> timeout_tunnel_align_detect.  Hover-checked frames may now be up to
    detect_max_latency_s old, and a slow detector latches with 3 frames (~4 s) instead of 5 (~9.5 s)."""
    scn = dict(name='slow_tunnel', tunnel_det_period_s=1.9, tunnel_det_latency_s=0.9, tunnel_conf=(0.50, 0.55))
    m, px4, world, log = run(Scenario(**scn))
    assert phase_seq(log) == list(PHASES), (phase_seq(log), m.abort_reason)
    df = events(log, 'detect_frames')
    assert df and df[0]['n'] == Params().detect_n_min, df
    t0 = first(log, lambda r: r['phase'] == 'tunnel_align')['t']
    t1 = first(log, lambda r: r['phase'] == 'tunnel_traverse')['t']
    m0, _, _, log0 = run(Scenario(**scn), Params(detect_max_latency_s=0.0, detect_n_min=Params().detect_n,
                                                 tunnel_depth_fallback_s=0.0))
    assert m0.abort_reason == 'timeout_tunnel_align_detect', (phase_seq(log0), m0.abort_reason)
    print('  slow tunnel detector (1.9 s period, 0.9 s latency, conf 0.50-0.55): latched with %d frames, align %.1f s; '
          'old age gate: %s' % (df[0]['n'], t1 - t0, m0.abort_reason))


def test_gate_b_chicane_gap_centre():
    """official-compat round B demo 230517 (truth prior): the straight line chicane end (-2.56, -3.43) -> Gate B
    line-up passed 0.05 m from C1's south end -> gate_approach_blocked.  The chicane end leg (judged against the
    line-up point) gets a via point at the centre of the C1-end <-> C2 gap and is rounded around the end; the
    chicane keeps >= 0.40 m from the C1 / C2 ends; the line-up point backs off C1 (standoff 0.8 -> 0.75 m)."""
    from replica.layout import MAZE_WALLS
    W = {n: (np.array(a), np.array(b)) for n, a, b, *_ in MAZE_WALLS}
    c1_end, c2_end = W['C1'][1], W['C2_north'][0]
    pr, scn = _venue_world('gate_b')
    pr['chicane'] = [dict(forward=-1.646, left=-4.834, up=0.8), dict(forward=-2.377, left=-4.499, up=0.8),
                     dict(forward=-2.56, left=-3.432, up=0.8)]
    scn.prior = pr
    m, px4, world, log = run(scn, Params(stop_after='gate_pass'))
    assert m.sub == 'stopped_after_gate_pass', (phase_seq(log), m.abort_reason)
    assert_no_collision(world, log, skip=())
    vias = events(log, 'gap_via')
    moves = [e for e in events(log, 'waypoint_centred') if e['phase'] == 'chicane' and e['gap']]
    assert vias or moves, events(log)
    ch = [world.flu(r['p']) for r in log if r['phase'] == 'chicane']
    d1 = min(float(np.hypot(*(q[:2] - c1_end))) for q in ch)
    d2 = min(float(np.hypot(*(q[:2] - c2_end))) for q in ch)
    assert d1 >= Params().lane_clear_m and d2 >= Params().lane_clear_m, (d1, d2)
    pts = [world.flu(np.asarray(v['via'], float)) for v in vias] + [world.flu(np.asarray(e['to'], float)) for e in moves
                                                                     if e['end']]
    for q in pts:
        assert min(np.hypot(*(q[:2] - c1_end)), np.hypot(*(q[:2] - c2_end))) >= Params().lane_clear_m - 0.02, q
    gp = events(log, 'gate_passed')
    assert gp, events(log)
    gs = events(log, 'gate_standoff_shrunk')
    end_mv = [e for e in moves if e['end']]
    print('  Gate B chicane: %d via(s), %d gap-centre vertex move(s) (chicane end -> %.2f m clearance), chicane >= '
          "%.2f m from C1's end / %.2f m from C2's end, line-up standoff %s, gate lateral %+.2f m"
          % (len(vias), len(moves), end_mv[-1]['clearance_m'] if end_mv else float('nan'), d1, d2,
             gs[0]['standoff_m'] if gs else 'nominal', gp[0]['lat_m']))


def _venue_world(checkpoint4=None):
    """Official course map v4 (COURSE_MAP.md via replica/layout.py) + mission/course_prior.venue_map.json."""
    from replica.layout import MAZE_WALLS, WALL_TOP, PANEL_BOTTOM, PLY_BOTTOM
    with open(os.path.join(HERE, 'course_prior.venue_map.json')) as fh:
        pr = json.load(fh)
    walls = [(f0, l0, f1, l1) + ((PANEL_BOTTOM, WALL_TOP, n, 0.025) if skin == 'banner' else (PLY_BOTTOM, WALL_TOP, n, 0.01))
             for n, (f0, l0), (f1, l1), skin, _ in MAZE_WALLS]
    wc = (-3.2, -2.03, 1.21)                                   # window board (with its hole) in wall C2
    th = math.radians(69.7)
    al = np.array([-math.sin(th), math.cos(th)])               # along the board
    c = np.array(wc[:2])
    def piece(s0, s1, b0, b1, name):
        a, b = c + al * s0, c + al * s1
        return (a[0], a[1], b[0], b[1], b0, b1, name, 0.01)
    walls += [piece(0.375, 0.75, 0.35, WALL_TOP, 'window_board_l'), piece(-0.75, -0.375, 0.35, WALL_TOP, 'window_board_r'),
              piece(-0.375, 0.375, 0.35, wc[2] - 0.30, 'window_board_low'), piece(-0.375, 0.375, wc[2] + 0.30, WALL_TOP,
                                                                                   'window_board_top')]
    te = pr['tunnel']['entry']
    f0, l0, L = te['forward'], te['left'], pr['tunnel']['length_m']
    walls += [(f0, l0 + s * 0.56, f0 - L, l0 + s * 0.56, 0.07, 1.17, 'tunnel_shell', 0.005) for s in (1.0, -1.0)]
    walls += [(f0, l0 + x, f0 - L, l0 + x, 1.15, 1.17, 'tunnel_arch', 0.005)            # the arch over it
              for x in np.arange(-0.5, 0.51, 0.1)]
    boulders = tuple(((b['forward'] - 0.23, b['left'] - 0.23, 0.0), (b['forward'] + 0.23, b['left'] + 0.23, 0.35),
                      'boulder') for b in pr['boulders'])
    if checkpoint4:
        pr['checkpoint4'] = checkpoint4
    over = dict(tunnel_entry=(f0, l0, 0.62), tunnel_axis_deg=180.0, tunnel_length=L, tunnel_ring_gap=L / 2,
                tunnel_radius=0.55, window_center=wc, window_through_deg=69.7, window_w=0.75, window_h=0.60,
                board_half_w=0.75, rock_center=(pr['landing_target']['center']['forward'],
                                                pr['landing_target']['center']['left']), rock_top=0.0)
    return pr, Scenario(name='venue_map', prior=pr, default_course=False, depth_walls=tuple(walls), extra_boxes=boulders,
                        true_over=over, pad_whole_view=True, vehicle_r=Params().prop_radius_m, t_max=260)


def _official_world(params_over=None, **scn_over):
    """Official lab sim geometry (runs 002-008): v4 map, but the north boundary is a WALL at f 1.92 (map 2.134);
    prior = the corrected official_map (fence.forward_max 1.97, north lane ends 1.555 = centre of the W1-end <-> wall
    gap 1.19..1.92, 0.73 m wide).  PX4 like the official sim: ~0.3 m overshoot braking from 0.5 m/s, and a fast yaw
    facing a near wall starts the slow hold oscillation."""
    pr, scn = _venue_world('gate_b')
    pr = json.loads(json.dumps(pr))
    pr['fence']['forward_max'] = 1.97
    for c in pr['corridor']:
        if abs(c['forward'] - 1.676) < 1e-3:
            c['forward'] = 1.555
    walls = []
    for w in scn.depth_walls:
        w = list(w)
        if w[6] == 'W2':                                   # hangs from the north wall
            w[2] = min(w[2], 1.92)
        walls.append(tuple(w))
    walls.append((1.945, 0.53, 1.945, -5.566, 0.0, 2.4, 'north_wall', 0.025))   # face at 1.92 (gap to W1 0.71 m)
    scn.depth_walls, scn.prior, scn.t_max = tuple(walls), pr, 200
    scn.px4_mpc, scn.px4_official, scn.yaw_wall_osc, scn.contact_physics = True, True, True, True
    for k, v in scn_over.items():
        setattr(scn, k, v)
    kw = dict(stop_after='corridor', v_cruise=0.5, standoff_fence_clear_m=0.35)
    kw.update(params_over or {})
    return pr, scn, Params(**kw)


def test_official_first_corner_and_maze():
    """Official lab runs 002-005 in the fake sim: first-corner gap 0.73 m (lab contact = airframe reach 0.28 m ->
    +-0.085 m), PX4 overshoot ~0.3 m from 0.5 m/s, fast yaw facing the wall -> +-0.3 m hold oscillation.  The run_002-
    era behaviour (hard carrot stops, 1.6 rad/s yaw in place, 0.5 m/s at corners) touches a wall; now: soft planned
    stops (carrot never behind the vehicle's own stop point), <= 0.3 m/s within 1 m of corners, corner yaw on the way
    in at <= 0.4 rad/s near walls, the oscillation never starts, >= 0.28 m (+ margin) from everything."""
    pr, scn, prm = _official_world()
    m, px4, world, log = run(scn, prm)
    assert m.sub == 'stopped_after_corridor', (phase_seq(log), m.abort_reason)
    cor = [r for r in log if r['phase'] == 'corridor']
    wd = min(world.clearance_xy(r['p'], skip=()) for r in cor)
    assert not px4.contacts and not events(log, 'contact'), px4.contacts
    assert wd >= Params().prop_radius_m + 0.03, wd
    assert not px4.osc_events, px4.osc_events
    old = dict(yaw_rate_near=1.6, v_near_waypoint=0.5, veh_brake_mps2=0.0, corner_yaw_early=False, a_stop_plan=2.5,
               a_corner_stop=0.5, acc_ff_max=0.0, ff_stop_decel=0.5, veh_lag_s=0.0, hold_drift_m=0.0)
    _, scn0, prm0 = _official_world(old)
    m0, px0, w0, log0 = run(scn0, prm0)
    cor0 = [r for r in log0 if r['phase'] in FLIGHT_PHASES_T]
    wd0 = min(w0.clearance_xy(r['p'], skip=()) for r in cor0)
    assert px0.contacts or wd0 < Params().prop_radius_m or px0.osc_events, (wd0, px0.osc_events, m0.abort_reason)
    print('  official first corner + maze: corridor %.1f s, nearest %.2f m (reach 0.28), no contact, no hold '
          'oscillation; run_002-era control: %s, nearest %.2f m, %d contact(s), oscillation x%d'
          % (cor[-1]['t'] - cor[0]['t'], wd, m0.abort_reason or m0.sub, wd0, len(px0.contacts), len(px0.osc_events)))


def test_tunnel_depth_fallback_official():
    """Official sim: the tunnel is a tiled arch with NO brass hoops -> the hoop detector never fires (run_007: 22/22
    found=false) and stages 4-6 could not be flown.  After tunnel_depth_fallback_s at the standoff the opening is
    measured in the obstacle map (side walls + arch + floor), centred on the prior axis, and the traverse is a narrow
    wall-relative corridor ('tunnel_mode' depth_fallback).  Also run_009: the standoff keeps >= 0.6 m from the
    measured north wall and the 180 deg turn toward the tunnel is made on the way in, away from the wall."""
    pr, scn, prm = _official_world(dict(stop_after='tunnel_traverse'), tunnel_detector_off=True, t_max=320)
    m, px4, world, log = run(scn, prm)
    assert m.sub == 'stopped_after_tunnel_traverse', (phase_seq(log), m.abort_reason)
    tm = events(log, 'tunnel_mode')
    assert tm and tm[0]['mode'] == 'depth_fallback', tm
    assert not px4.contacts, px4.contacts
    fl = [r for r in log if r['phase'] in FLIGHT_PHASES_T]
    wd = min(world.clearance_xy(r['p'], skip=()) for r in fl)
    assert wd >= Params().prop_radius_m + 0.02, wd
    te = pr['tunnel']['entry']
    q, _ = crossing(world, log, te['forward'] - 0.5 * pr['tunnel']['length_m'], -1)   # mid-tunnel
    lat = abs(q[1] - te['left'])
    assert lat <= 0.12, lat
    north = 1.92                                             # the measured north wall (face)
    hov = [world.flu(r['p']) for r in log if r['phase'] == 'tunnel_align']
    far = north - max(h[0] for h in hov)
    assert far >= Params().standoff_wall_clear_m - 0.1, far
    _, scn2, prm2 = _official_world(dict(stop_after='tunnel_traverse'), tunnel_detector_off=True, t_max=320)
    pr2 = json.loads(json.dumps(scn2.prior))
    pr2['tunnel']['entry']['left'] += 0.25                   # prior 0.25 m off to the side (the shell stays put)
    scn2.prior = pr2
    m2, px2, w2, log2 = run(scn2, prm2)
    assert m2.sub == 'stopped_after_tunnel_traverse', (phase_seq(log2), m2.abort_reason)
    tm2 = events(log2, 'tunnel_mode')
    q2, _ = crossing(w2, log2, te['forward'] - 0.5 * pr['tunnel']['length_m'], -1)
    lat2 = abs(q2[1] - te['left'])
    assert tm2 and abs(tm2[0]['lateral_m'] + 0.25) <= 0.08 and lat2 <= 0.12 and not px2.contacts, (tm2, lat2)
    print('  depth-only tunnel (no hoops): %s after %.0f s at the standoff, opening %.2f m wide, centre %+.2f m from '
          'the prior axis, mid-tunnel %.2f m off the true axis, nearest %.2f m, hover >= %.2f m from the north wall, '
          'no contact; prior 0.25 m off: measured centre %+.2f m, mid-tunnel %.2f m off'
          % (tm[0]['mode'], Params().tunnel_depth_fallback_s, tm[0]['width_m'] or float('nan'),
             tm[0]['lateral_m'], lat, wd, far, tm2[0]['lateral_m'], lat2))


def test_hold_stop_point_ahead_official_px4():
    """Official run_002: a hold set the carrot (v 0) at f 1.20 while the vehicle flew 0.55 m/s; with the official
    PX4 lag the vehicle overshot 0.5 m into the wall.  A wall revealed 1.3 m ahead at 0.55 m/s: the setpoint stays at
    or ahead of the vehicle's own stop point (never behind it while it moves), so it stops without overshoot."""
    wall = ((2.6, -1.0, 0.0), (2.7, 1.0, 2.2), 'late_wall')
    pr = prior_dict(corridor=[dict(forward=4.0, left=0.0, up=1.0)])
    pr['tunnel'] = dict(entry=dict(forward=6.0, left=0.0, up=1.0), axis_deg=0.0, length_m=1.2, tol_m=0.5)
    pr['fence'] = dict(forward_min=-1.0, forward_max=7.0, left_min=-2.0, left_max=2.0, up_max=2.9)
    out = {}
    for label, over in (('new', {}), ('hard stop', dict(veh_brake_mps2=0.0, a_stop_plan=2.5, acc_ff_max=0.0,
                                                         ff_stop_decel=50.0))):
        scn = Scenario(name='late_wall', prior=pr, default_course=True, hidden_boxes=(wall,), hidden_reveal_m=1.3,
                       px4_mpc=True, px4_official=True, t_max=30, vehicle_r=Params().prop_radius_m)
        prm = Params(stop_after='corridor', v_cruise=0.55, **over)
        m, px4, world, log = run(scn, prm)
        assert world.revealed, label
        t_rev = world.revealed[0][0]
        after = [r for r in log if r['t'] >= t_rev and r['phase'] == 'corridor']
        behind = max(float(world.flu(r['p'])[0] - world.flu(r['out'].pos_ned)[0]) for r in after
                     if r['out'].pos_ned is not None and np.linalg.norm(r['v'][:2]) > 0.15) if after else 0.0
        f_max = max(world.flu(r['p'])[0] for r in after)
        out[label] = (behind, 2.6 - f_max)
    assert out['new'][0] <= 0.08, out                       # the carrot is never behind the moving vehicle
    assert out['new'][1] >= Params().prop_radius_m + 0.05, out
    assert out['hard stop'][0] > 0.05, out
    print('  late wall at 0.55 m/s (official PX4 lag): setpoint behind the moving vehicle %.2f m, closest %.2f m to '
          'the wall; hard carrot stop: %.2f m behind, closest %.2f m' % (out['new'] + out['hard stop']))


def _offset_prior(pr, seed, amp=0.5):
    """Per-wall prior offsets: every maze wall gets a random (df, dl) in +-amp; each corridor vertex moves with the
    wall nearest to it (the prior polyline is drawn relative to walls that are each off by up to amp)."""
    from replica.layout import MAZE_WALLS
    rng = np.random.default_rng(seed)
    off = {n: rng.uniform(-amp, amp, 2) for n, *_ in MAZE_WALLS}
    pr = json.loads(json.dumps(pr))
    for c in pr['corridor']:
        q = np.array([c['forward'], c['left']])
        best = min(((float(np.linalg.norm(q - (np.array(a) + np.clip((q - np.array(a)) @ (np.array(b) - np.array(a))
                                                                    / max(1e-9, float((np.array(b) - np.array(a)) @
                                                                                      (np.array(b) - np.array(a)))),
                                                                    0.0, 1.0) * (np.array(b) - np.array(a))))), n)
                    for n, a, b, *_ in MAZE_WALLS))
        c['forward'] += float(off[best[1]][0])
        c['left'] += float(off[best[1]][1])
    return pr


def test_wall_relative_prior_offsets():
    """No hand-carried survey: the prior is the organizers' map (+-0.2-0.3 m) or photos (+-0.5-0.8 m).  v4 map,
    every maze wall's prior off by a random +-0.5 m (corridor vertices move with their wall), PX4 velocity loop.  The
    prior polyline is only the topology; vertices go to the centre of the MEASURED lane / turn gap.  Default: 4 seeds
    (0, 5, 7, 10 -- blocked or hit a board before); MISSION_LONG=1: all 20."""
    seeds = range(20) if os.environ.get('MISSION_LONG') else (0, 5, 7, 10)
    out = []
    for sd in seeds:
        pr0, scn = _venue_world('gate_b')
        scn.prior, scn.t_max, scn.px4_mpc = _offset_prior(pr0, sd), 260, True
        m, px4, world, log = run(scn, Params(stop_after='corridor'))
        assert m.sub == 'stopped_after_corridor', (sd, phase_seq(log), m.abort_reason)
        assert_no_collision(world, log, skip=())
        cor = [r for r in log if r['phase'] == 'corridor']
        wd = min(world.clearance_xy(r['p'], skip=()) for r in cor)
        assert wd >= Params().prop_radius_m + 0.02, (sd, wd)
        out.append((sd, cor[-1]['t'] - cor[0]['t'], wd))
    print('  wall-relative, per-wall prior offsets +-0.5 m: %d/%d seeds through corridor + maze, %.0f-%.0f s, nearest '
          '%.2f m (reach 0.28)' % (len(out), len(list(seeds)), min(o[1] for o in out), max(o[1] for o in out),
                                   min(o[2] for o in out)))


def test_move_only_into_seen_free_space():
    """Props hit a board end while nudging sideways / hovering: translation sideways or backward only into space seen
    free in the camera FOV within seen_free_s (or on our own track, or -- in the stereo blind zone -- within the map
    memory).  Unit level: the predicate, and a parallax nudge goes vertical-only when its sides are not seen free."""
    pr = prior_dict()
    m = _mission_at_home(pr)
    sf = m.sf
    m.per = Perception(obstacles=_PtsMap(np.zeros((0, 3))), obstacles_t=0.0)
    m.phase = 'corridor'

    def at(t, f, l, yaw=0.0):
        m.now, m.pos = t, sf.to_ned((f, l, 1.0))
        m.veh = Vehicle(pos_ned=tuple(m.pos), heading=yaw, armed=True, nav_state=NAV_OFFBOARD)
        m._record_view()
    at(0.0, 0.0, 0.0)
    ahead, side, behind = sf.to_ned((1.5, 0.0, 1.0)), sf.to_ned((0.0, 0.8, 1.0)), sf.to_ned((-0.6, 0.0, 1.0))
    assert m._seen_free(ahead) and not m._seen_free(side) and not m._seen_free(behind)
    for k in range(1, 11):                                   # fly 1 m forward
        at(0.1 * k, 0.1 * k, 0.0)
    assert m._seen_free(sf.to_ned((0.3, 0.0, 1.0)))         # our own track (behind us now)
    abreast = sf.to_ned((1.2, 0.5, 1.0))                     # seen early (25 deg), beside us now (out of view)
    assert m._seen_free(abreast)                             # 1 s old: fine
    at(3.5, 1.0, 0.0)                                        # hovered 2.5 s: that view went stale ...
    assert not m._seen_free(abreast)
    assert m._seen_free(sf.to_ned((1.0, 0.0, 1.0))) and m._seen_free(ahead)   # (our track; ahead: in view now)
    f = Follower(sf.to_ned((1.0, 0.0, 1.0)), [sf.to_ned((3.0, 0.0, 1.0))], Params(), 0.5, avoid='lateral')
    f.seen_free = lambda q: False
    assert not f._nudge_side_seen(f.P[0])
    f.seen_free = lambda q: True
    assert f._nudge_side_seen(f.P[0])
    f._nudge_lat = False
    d, v = f._nudge(0.5, 0.0)
    assert abs(float(d @ f._lat_normal(0.0))) < 1e-9 and d[2] < 0       # vertical only
    print('  seen-free: ahead in the FOV yes, beside / behind no, own track yes, stale (> %.1f s) views no; a '
          'nudge whose sides are not seen free is vertical only' % Params().seen_free_s)


def test_hold_drift_backs_away_from_wall():
    """Official run_003: after a fast 90 deg yaw facing the wall at 0.4 m the hold oscillated +-0.3 m (6 s period)
    into the wall.  Forced here (1.6 rad/s yaw in place, no early yaw): the watchdog sees |pos - sp| > 0.15 m and
    growing while holding and backs away from the nearest wall instead of waiting (detection test; see the note)."""
    over = dict(yaw_rate_near=1.6, corner_yaw_early=False)
    out = {}
    for label, extra in (('watchdog', {}), ('no watchdog', dict(hold_drift_m=0.0))):
        _, scn, prm = _official_world(dict(over, **extra))
        m, px4, world, log = run(scn, prm)
        cor = [r for r in log if r['phase'] in FLIGHT_PHASES_T]
        out[label] = (len(px4.osc_events), len(events(log, 'hold_drift')), len(px4.contacts),
                      min(world.clearance_xy(r['p'], skip=()) for r in cor), m.sub or m.abort_reason)
    assert out['watchdog'][0] >= 1 and out['watchdog'][1] >= 1, out          # the oscillation started, was seen
    assert out['watchdog'][4] == 'stopped_after_corridor', out                # and backing away never ends the run
    # NOTE: in this model it does NOT reduce the contacts: 0.36 m from a wall (reach 0.28) a +-0.3 m oscillation
    # touches long before |pos - sp| reaches 0.15 m, and most swings come while the carrot moves.  What works is not
    # starting it: slow (0.4 rad/s) yaw near walls, turned on the way in -- test_official_first_corner_and_maze.
    print('  hold drift (forced fast yaw at the wall): oscillation x%d, watchdog backed away x%d, %d contact(s), '
          'nearest %.2f m (%s); without the watchdog: %d contact(s), nearest %.2f m (%s)'
          % (out['watchdog'][0], out['watchdog'][1], out['watchdog'][2], out['watchdog'][3], out['watchdog'][4],
             out['no watchdog'][2], out['no watchdog'][3], out['no watchdog'][4]))


def test_venue_map_tape_seed_priors_corridor():
    """official-compat v4 round A: tape-level seed priors (ubuntu_compat/course_prior.replica.seed*.json) put
    corridor points near boards; seeds 2/6 were blocked at the first north corner (the next leg passed W1's end at
    0.31 m).  With vertex + leg centring every seed flies the corridor, >= 0.35 m from every board."""
    out = []
    for s in (2, 6):
        f = os.path.join(HERE, '..', 'ubuntu_compat', 'course_prior.replica.seed%d.json' % s)
        if not os.path.exists(f):
            out.append('seed %d: prior file missing, skipped' % s)
            continue
        with open(f) as fh:
            seed = json.load(fh)
        _, scn = _venue_world('gate_b')
        scn.prior, scn.t_max = seed, 120
        m, px4, world, log = run(scn, Params(stop_after='corridor'))
        assert m.sub == 'stopped_after_corridor', (s, phase_seq(log), m.abort_reason)
        assert_no_collision(world, log, skip=())
        cor = [r for r in log if r['phase'] == 'corridor']
        wd = min(world.clearance_xy(r['p'], skip=()) for r in cor)
        assert wd >= Params().radius, (s, wd)
        out.append('seed %d: corridor %.1f s, %d vertex moves, nearest board %.2f m'
                   % (s, cor[-1]['t'] - cor[0]['t'], len(events(log, 'waypoint_centred')), wd))
    print('  v4 tape-level priors: ' + '; '.join(out))


def test_venue_map_full_course():
    """Whole official map (kinematic): launch -> corridor + serpentine + Gate A -> tunnel north->south -> chicane
    around C1 -> checkpoint 4 (Gate B by default; the window as the option) -> around W4 -> pad -> disarm, re-arm ->
    over the 1.73 m barrier -> launch.  Walls from COURSE_MAP.md (layout.py v4), prior course_prior.venue_map.json."""
    out = []
    for cp in ('gate_b', 'window'):
        pr, scn = _venue_world(cp)
        m, px4, world, log = run(scn)
        seq = phase_seq(log)
        want = (['corridor', 'tunnel_align', 'tunnel_traverse', 'chicane'] +
                (['gate_pass'] if cp == 'gate_b' else ['window_align', 'window_traverse']) +
                (['landing_route'] if cp == 'gate_b' else []) +
                ['land_target', 'disarm', 'rearm', 'climb_over_wall', 'return_home', 'land_home', 'done'])
        assert [s for s in seq if s in want] == want and m.abort_reason is None, (cp, seq, m.abort_reason)
        assert_no_collision(world, log, skip=())
        ev = events(log, 'checkpoint4')
        assert ev and ev[0]['choice'] == cp, ev
        # landed inside the outer ring (whole drone: centre within ring - props)
        dis = events(log, 'disarmed_on_target') or [first(log, lambda r: r['phase'] == 'rearm')]
        land_p = world.flu(log[[i for i, r in enumerate(log) if r['phase'] == 'disarm'][0]]['p'])
        lc = pr['landing_target']['center']
        ring = pr['landing_target'].get('ring_radius_m', 0.78)
        d_pad = math.hypot(land_p[0] - lc['forward'], land_p[1] - lc['left'])
        assert d_pad <= ring - Params().prop_radius_m, (cp, d_pad)
        # over the barrier with margin, home within 0.2 m
        bar_f = [f for n, (f0_, l0_), (f1_, l1_), *_ in __import__('replica.layout', fromlist=['MAZE_WALLS']).MAZE_WALLS
                 if n == 'barrier' for f in (f0_,)][0]
        cross = [world.flu(r['p']) for r in log if r['phase'] == 'climb_over_wall']
        cr = [q for q in cross if abs(q[0] - bar_f) < 0.3]
        assert cr and min(q[2] for q in cr) >= 1.727 + 0.25, (cp, min(q[2] for q in cr) if cr else None)
        end = world.flu(log[-1]['p'])
        assert math.hypot(end[0], end[1]) < 0.2, end
        extra = ''
        if cp == 'gate_b':
            gp = events(log, 'gate_passed')
            assert gp and abs(gp[0]['lat_m']) <= 0.5 * pr['gate_b']['width_m'] - Params().prop_radius_m, gp
            extra = ', gate lateral %+.2f m' % gp[0]['lat_m']
        wall_d = min(world.clearance_xy(r['p'], skip=()) for r in log if r['phase'] in FLIGHT_PHASES_T)
        out.append('%s: done in %.1f s, pad %.2f m from the centre (ring %.2f), barrier crossed %.2f m above its top, '
                   'nearest obstacle %.2f m%s' % (cp, log[-1]['t'], d_pad, ring, min(q[2] for q in cr) - 1.727, wall_d,
                                                  extra))
    print('  venue map: ' + '; '.join(out))


def test_pad_observe_replica_region_and_live_fallback():
    """official-compat 17.4 (ray-cast on the v3 replica): from the window exit the best observe spot is south along
    the passage, (-0.1..0.9, -4.6) facing +f.  The GENERIC search (no coordinates in core) must land there; and as the
    replica south of the pad is uncertain (rocks, blocks, photo 17's red rocks), a pose that the live map later shows
    blocked -- at the pose or in the line of sight -- is replaced by another candidate."""
    walls = [(1.2, -3.3, 2.8, -3.3, 0.35, 1.70, 'climb_wall', 0.01)]       # the wall flown over later
    for s0, s1, b0, b1, n in ((-3.05, -3.425, 0.35, 1.70, 'win_l'), (-4.175, -4.55, 0.35, 1.70, 'win_r'),
                              (-3.425, -4.175, 0.35, 0.91, 'win_low'), (-3.425, -4.175, 1.51, 1.70, 'win_top')):
        walls.append((3.0, s0, 3.0, s1, b0, b1, n, 0.01))                   # window board at f 3.0 (hole 0.91-1.51)
    pr = prior_dict(fence=V3_FENCE, landing_target=dict(center=dict(forward=2.2, left=-4.6), top_up=0.0, tol_m=0.5),
                    wall=dict(cross_from=dict(forward=2.0, left=-3.8), cross_to=dict(forward=2.0, left=-2.8),
                              top_up=1.7, tol_m=0.3))
    m = Mission(CoursePrior.from_dict(pr), Params())
    m.now, m.pos, m.veh, m.phase = 0.0, np.zeros(3), Vehicle(), 'land_target'
    if not hasattr(m, '_events'):
        m._events = []
    m._record_home()
    dm = DepthLikeMap(walls, m.sf)
    dm._act = dm.all                                                         # everything measured
    m.per = Perception(obstacles=dm, obstacles_t=0.0)
    m.pos = m.sf.to_ned((2.55, -3.79, 0.55))                                 # where the drone left the window
    rock = m.g['rock']
    z = m._clamp_alt(rock[2] - m.p.observe_above_top_m)
    lo, pref = m._observe_min_dist(z), m._observe_dist()
    pick = m._pick_observe(rock, z, lo, pref)
    assert pick is not None
    q = m.sf.from_ned(pick[0])
    face = math.degrees(math.atan2(*(m.sf.from_ned(rock)[:2] - q[:2])[::-1]))
    # the passage south of the pad behind the window exit (sim: f -0.1..0.9), between the wall line and the net,
    # facing roughly +f -- not north of the window (re-crossing the board) and not behind the wall
    assert -0.2 <= q[0] <= 1.0 and -4.75 <= q[1] <= -3.6 and abs(face) <= 35.0, (q, face)
    # live map: an unexpected rock AT the pose, then one in the line of sight -> other candidates
    def add_rock(f, l, h=0.6, r=0.15):
        pts = [m.sf.to_ned((f + r * math.cos(a), l + r * math.sin(a), u)) for a in np.linspace(0, 2 * math.pi, 16)
               for u in np.arange(0.0, h, 0.05)]
        dm.all = np.vstack([dm.all, pts])
        dm._act = dm.all
    clean = dm.all.copy()
    add_rock(q[0], q[1])                                                    # (1) a rock on the pose itself
    assert not m._observe_ok(pick[0], rock)
    pick2 = m._pick_observe(rock, z, lo, pref, avoid=[pick[0]])
    assert pick2 is not None and m._observe_ok(pick2[0], rock)
    q2 = m.sf.from_ned(pick2[0])
    assert math.hypot(*(q2 - q)[:2]) >= 0.35 and -0.2 <= q2[0] <= 1.0 and -4.75 <= q2[1] <= -3.6, q2
    dm.all = clean.copy()
    dm._act = dm.all
    mid = 0.5 * (q[:2] + m.sf.from_ned(rock)[:2])
    add_rock(mid[0], mid[1], h=0.3, r=0.1)                                  # (2) a low rock in the line of sight
    assert not m._observe_ok(pick[0], rock)
    pick3 = m._pick_observe(rock, z, lo, pref, avoid=[pick[0]])
    assert pick3 is not None and m._observe_ok(pick3[0], rock)
    q3 = m.sf.from_ned(pick3[0])
    print('  pad observe (v3 replica): generic search -> (%.2f, %.2f) facing %+.0f deg (sim ray-cast: -0.1..0.9, '
          '-4.6, +f); rock at the pose -> (%.2f, %.2f); rock in the line of sight -> (%.2f, %.2f)'
          % (q[0], q[1], face, q2[0], q2[1], q3[0], q3[1]))


def test_pad_observe_after_window_exit():
    """official-compat 191052: out of the window the drone was 0.86 m from the flat pad in the alley between the
    climbing wall and the net; with the camera at 0.55 m the whole pad is visible only from >= 1.46 m; reacquire
    turned toward the wall to back off and hit it.  Now: an observe pose >= the FOV distance from the prior pad centre,
    inside the fence, clear, on the pad side of the wall, reached with the camera kept on the pad."""
    pr = prior_dict(landing_target=dict(center=dict(forward=2.35, left=-3.4), top_up=0.0, tol_m=0.5),
                    wall=dict(cross_from=dict(forward=1.2, left=-3.0), cross_to=dict(forward=1.2, left=-1.6),
                              top_up=1.7, tol_m=0.3))
    over = dict(rock_center=(2.35, -3.4), wall_f=(0.3, 3.3), wall_l=(-2.45, -2.25))   # wall face 0.35 m beside the exit
    scn = Scenario(name='pad_alley', prior=pr, true_over=over, pad_whole_view=True, t_max=120,
                   vehicle_r=Params().prop_radius_m)
    m, px4, world, log = run(scn)
    assert phase_seq(log) == list(PHASES), (phase_seq(log), m.abort_reason)
    assert_no_collision(world, log)
    # live fallback: a rock the replica lacks stands on that pose -> seen on the way -> another pose
    o0 = world.flu(events(log, 'observe_pose')[0]['pos'])
    rock_box = ((o0[0] - 0.15, o0[1] - 0.15, 0.0), (o0[0] + 0.15, o0[1] + 0.15, 0.6), 'red_rock')
    scn_r = Scenario(name='pad_alley_rock', prior=pr, true_over=over, pad_whole_view=True, t_max=120,
                     vehicle_r=Params().prop_radius_m, hidden_boxes=(rock_box,), hidden_reveal_m=1.3)
    mr, _, world_r, log_r = run(scn_r)
    assert phase_seq(log_r) == list(PHASES), (phase_seq(log_r), mr.abort_reason)
    assert_no_collision(world_r, log_r)
    inv = events(log_r, 'observe_pose_invalid')
    fb = [e for e in events(log_r, 'observe_pose') if e['reason'] == 'fallback']
    assert inv and fb, (inv, fb, world_r.revealed)
    o1 = world_r.flu(fb[0]['pos'])
    assert math.hypot(o1[0] - o0[0], o1[1] - o0[1]) >= 0.35
    lt = [r for r in log if r['phase'] == 'land_target']
    q0 = world.flu(lt[0]['p'])
    assert math.hypot(q0[0] - 2.35, q0[1] + 3.4) < 1.0                 # the window exit really is too close
    ob = events(log, 'observe_pose')[0]
    o = world.flu(ob['pos'])
    d_obs = math.hypot(o[0] - 2.35, o[1] + 3.4)
    assert d_obs >= ob['min_dist_m'] - 0.01 and ob['min_dist_m'] >= 1.5, ob
    assert FENCE['left_min'] + 0.4 <= o[1] <= -2.45 - 0.4 and ob['reason'] == 'back_off', (o, ob)
    # never toward the wall: nearest wall face stays >= props + margin, and the camera never faces the wall while
    # within 1 m of it
    wd = min(world.clearance_xy(r['p']) for r in lt)
    assert wd >= Params().prop_radius_m + 0.05, wd
    for r in lt:
        q = world.flu(r['p'])
        if -2.45 - q[1] < 1.0 and q[0] < 3.3:
            h = world.sf.from_ned(r['p'] + np.array([math.cos(r['yaw']), math.sin(r['yaw']), 0.0])) - q
            assert h[1] < 0.5, (r['t'], q, h)                          # not facing +left (the wall)
    lat = events(log, 'latched')
    assert any(e.get('detector') == 'landing' for e in lat)
    print('  pad alley: window exit %.2f m from the pad -> observe pose %.2f m away (FOV minimum %.2f m), reached '
          'with the camera on the pad, nearest wall %.2f m, landed and finished the course; hidden rock on that pose '
          '-> live re-check -> fallback (%.2f, %.2f) -> done'
          % (math.hypot(q0[0] - 2.35, q0[1] + 3.4), d_obs, ob['min_dist_m'], wd, o1[0], o1[1]))


def test_example_prior_loads():
    p = CoursePrior.load(os.path.join(HERE, 'course_prior.example.json'))
    assert p.fence['up_max'] > 2.0 and p.tunnel_length_m > 0 and len(p.corridor) >= 1
    bad = prior_dict()
    del bad['fence']['up_max']
    try:
        CoursePrior.from_dict(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('missing fence.up_max accepted')


def test_start_frame_and_latch():
    sf = StartFrame((1.0, 2.0, -0.1), math.radians(90))     # facing east
    assert np.allclose(sf.to_ned((1, 0, 0)), (1, 3, -0.1))
    assert np.allclose(sf.to_ned((0, 1, 0)), (2, 2, -0.1))    # left of east = north
    assert np.allclose(sf.to_ned((0, 0, 1)), (1, 2, -1.1))
    assert np.allclose(sf.from_ned(sf.to_ned((0.3, -0.7, 1.2))), (0.3, -0.7, 1.2))
    lt = Latch(3, 0.15, 10, prior=(0, 0, 0), gate=1.0, prior_dir=(1, 0))
    assert not lt.feed((0.1, 0, 0), (-1, 0.01))               # sign flipped to the prior direction
    assert not lt.feed((0.12, 0, 0), (1, 0.0))
    assert lt.feed((0.11, 0.01, 0), (1, -0.01))
    pos, d, _ = lt.value()
    assert d[0] > 0.99
    assert not lt.feed((0.5, 0, 0), (1, 0))                   # jump -> streak restarts
    assert not lt.feed((2.0, 0, 0), (1, 0)) and lt.rejected == 1   # outside the prior gate


TESTS = [test_start_frame_and_latch, test_example_prior_loads, test_full_run, test_full_run_rotated_start,
         test_start_phase_per_stage, test_pad_approach_land_target,
         test_corridor_obstacle_side_step, test_corridor_fully_blocked_hovers_then_lands,
         test_tunnel_unstable_detection_waits, test_tunnel_prior_error_and_hoop_jump, test_window_clearance_refused_lands, test_tof_gates_disarm,
         test_pose_unhealthy, test_px4_takeover_releases, test_px4_auto_land_stops_setpoints,
         test_geofence_stop_and_land, test_mission_time_limit, test_missing_obstacle_map_caps_speed,
         test_px4_land_detector_needs_commanded_descent, test_touchdown_handoff_to_px4_land,
         test_tunnel_near_net_standoff_shrinks, test_minor_phantom_cluster_slows_not_side_steps,
         test_persistent_small_cluster_never_side_steps, test_view_locked_phantom_carved_by_nudge,
         test_tunnel_align_phantom_before_entry, test_side_step_return_and_corridor_cap,
         test_return_home_high_then_descend, test_window_reacquire_after_cut_by_fov,
         test_window_reacquire_after_oblique_view, test_rock_observe_distance_and_reacquire,
         test_pilot_mode_full_course, test_pilot_mode_takeover_rearm_and_abort,
         test_pilot_mode_height_bias_never_climbs, test_pilot_mode_climb_guard,
         test_corner_yaw_ahead_and_speed_cap, test_vio_wall_standoff_and_waypoint_clip,
         test_nudge_only_when_stopped_and_sideways, test_corridor_uturn_maze, test_replica_maze_l_corners, test_pad_observe_after_window_exit, test_narrow_lane_seed6_centring,
         test_contact_hover_backoff_replan, test_venue_map_full_course, test_pad_observe_replica_region_and_live_fallback,
         test_venue_map_tape_seed_priors_corridor, test_corner_phantoms_in_front_of_net, test_thin_board_end_uturn,
         test_net_phantom_at_takeoff_not_structure, test_small_cluster_reconfirm_lateral_parallax,
         test_floating_cluster_vs_board_end_column, test_net_layer_is_no_vertical_support, test_net_ghosts_do_not_move_vertices,
         test_tiny_cluster_never_blocks,
         test_observation_hover_keeps_rear_clearance,
         test_barrier_behind_start_keepout,
         test_floor_bridged_phantom_is_floating,
         test_prior_wall_band_needs_column,
         test_corner_vertex_keeps_net_clearance,
         test_vertex_moves_only_into_seen_space,
         test_abort_landing_clears_wall_feet,
         test_vio_jump_no_chase,
         test_start_ratchet_keeps_forward_progress,
         test_start_advance_off_the_take_off_point,
         test_net_board_end_is_structure, test_slow_tunnel_detector_latches, test_gate_b_chicane_gap_centre,
         test_official_first_corner_and_maze, test_hold_stop_point_ahead_official_px4, test_tunnel_depth_fallback_official,
         test_wall_relative_prior_offsets, test_move_only_into_seen_free_space, test_hold_drift_backs_away_from_wall,
         test_side_step_into_unseen_space_looks_first,
         test_tunnel_phantoms_excluded_after_latch, test_landing_latch_noisy_confidence,
         test_staged_stops, test_traverse_commit_gate_and_axis_tracking, test_standoff_keeps_fence_clearance,
         test_wall_measurement_raises_climb]


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
