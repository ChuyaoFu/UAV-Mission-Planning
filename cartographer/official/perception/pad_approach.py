"""Landing-pad approach: perception + a small planner helper for mission-core.

The landing target is a flat 0.60 m blue pad with an H on the floor. The forward camera is level
and sees only ~25 deg below the horizon, so the floor is visible from 2.12 x (camera height) ahead
of the camera. The approach therefore looks at the pad from low and close, locks it over several
frames, keeps re-detecting while it closes in, and covers the last < 1 m on VIO:

  goto_standoff    fly (at the current altitude) to `standoff_m` (1.5 m) in front of the PRIOR pad,
                   on the line from the vehicle to the pad, nose toward the pad
  descend_observe  descend there to `observe_cam_h_m` (0.40 m camera height above the floor)
  acquire          hover; detect every frame (search radius around the prior); lock when
                   `lock_n` detections within `lock_window_s` agree within `lock_tol_m`
  scan             nothing within `acquire_timeout_s`: slow yaw sweep +-`scan_half_deg` around the
                   bearing to the prior (wider search radius); lock -> approach, nothing -> handoff
  approach         creep toward the pad (`v_approach`), re-detecting around the locked estimate and
                   refining it; stops when the pad centre reaches the bottom of the image
                   (camera-to-centre = h / tan(bottom edge angle): 2.11 h official, 2.01 h real
                   skybeta rectified; 0.85 / 0.80 m at 0.40 m)
  settle           short hover at that point (last, closest detections)
  blind            fly on VIO to directly above the estimated centre (body = base_link); the
                   segment is camera stop distance + camera offset (~0.97 m at 0.40 m)
  final_descent    vertical descent over the centre (paused while the xy error is large)
  touchdown        body within `touchdown_h_m` of the floor: `land` = True (mission-core lands /
                   disarms)
  handoff          pad not found (or blind segment too long): hold here, `handoff` = True (pilot)

API (all NED, metres, radians; yaw = NED heading, 0 = north, + = clockwise from above):
    appr = PadApproach(prior_center_ned=(n, e[, d]), params=ApproachParams(...), floor_z=0.0)
    out = appr.update(t, frame=Frame or None, body_ned=None, yaw=None)
        frame: common.geometry.Frame (left image, depth, K, R_nc, cam_ned, body_ned, heading).
               None on ticks without a new image: the planner still ramps the setpoint.
        body_ned / yaw: vehicle pose (default: from the frame).
    out.phase, out.setpoint_ned (ramped body position setpoint), out.yaw (heading setpoint),
    out.pad (PadEstimate: center_ned, yaw_ned, confidence, n_frames, locked, spread_m),
    out.detection (this frame's LandingTarget or None), out.land, out.handoff, out.events.
The detector is injectable (`detector=`): f(frame, floor_z, prior_xy, radius_m) ->
(LandingTarget or None, floor z or None). Default: floor.fit_floor + floor.landing_target (pad).
Alternatively (mission-core: the node runs the perception): per new image call
    prior_xy, radius = appr.search()  ->  node: landing_target(frame, floor, prior_center_ned=(*prior_xy, floor_z),
                                                               prior_radius_m=radius)
    appr.observe(t, landing_target_result, floor_fit.d if found else None)
and then appr.update(t, frame=None, body_ned, yaw) every control tick.

Obstacle checks along the goto / approach segments are mission-core's job (ObstacleMap.clearance);
this module only proposes the goals. Pure numpy (+ floor.py), Python 3.9+.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from . import floor as _floor
except ImportError:                                   # imported as a top-level module
    import floor as _floor                            # type: ignore

TAN_HALF_VFOV = (400 - 200.7) / 421.0   # bottom image edge below the axis: official 640x400 (25.3 deg); real rectified (400 - 193.9) / 414.3 = 26.4 deg


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class ApproachParams:
    standoff_m: float = 1.5            # body -> prior pad centre at the observe pose
    observe_cam_h_m: float = 0.40      # camera height above the floor while observing / approaching
    cam_fwd_m: float = 0.116           # camera ahead of base_link (overwritten from Frame when given)
    cam_up_m: float = 0.029            # camera above base_link (idem)
    tan_half_vfov: float = TAN_HALF_VFOV
    stop_extra_m: float = 0.0          # approach stops at camera-to-centre = h / tan(vfov/2) + this
    pad_size_m: float = 0.60
    v_xy: float = 0.30                 # goto speed (m/s)
    v_approach: float = 0.12           # approach creep (m/s)
    v_blind: float = 0.15              # blind segment (m/s)
    v_z: float = 0.20                  # descent to the observe height (m/s)
    v_land: float = 0.15               # final descent (m/s)
    yaw_rate: float = math.radians(30.0)
    pos_tol_m: float = 0.10            # 'arrived' for goto / approach stop
    xy_tol_m: float = 0.05             # over the centre before the final descent
    land_xy_pause_m: float = 0.12      # final descent pauses while |xy error| exceeds this
    z_tol_m: float = 0.05
    settle_speed: float = 0.06         # m/s, 'stopped'
    settle_s: float = 0.6
    search_radius_m: float = 1.0       # detector prior radius around the prior pad (acquire)
    scan_radius_m: float = 1.5         # ... during the yaw scan
    track_radius_m: float = 0.35       # ... around the locked estimate (approach)
    acquire_timeout_s: float = 4.0
    scan_half_deg: float = 35.0
    scan_rate_dps: float = 12.0
    lock_n: int = 4
    lock_window_s: float = 3.0
    lock_tol_m: float = 0.10
    track_gate_m: float = 0.15         # approach detections farther than this from the estimate: rejected
    est_keep: int = 12                 # detections averaged into the estimate
    min_confidence: float = 0.35
    lost_hold_s: float = 1.5           # approach: no accepted detection this long -> hold
    relock_s: float = 3.0              # ... still nothing: blind if short enough, else scan again
    blind_max_m: float = 1.10          # longest acceptable VIO-only segment (else handoff)
    touchdown_h_m: float = 0.15        # body height above the floor that counts as down
    land_push_m: float = 0.10          # final setpoint this far below the touchdown height
    keep_xy: Optional[Tuple[float, float, float, float]] = None   # cage (n0, n1, e0, e1) for the detector


@dataclass
class PadEstimate:
    center_ned: np.ndarray
    yaw_ned: float
    confidence: float
    n_frames: int
    locked: bool
    spread_m: float
    last_seen: float


@dataclass
class ApproachOutput:
    phase: str
    setpoint_ned: np.ndarray
    yaw: float
    pad: Optional[PadEstimate]
    detection: Optional[object]
    land: bool = False
    handoff: bool = False
    events: List[Tuple[str, dict]] = field(default_factory=list)


def default_detector(keep_xy=None):
    """fit_floor (prior = the planner's floor z) + landing_target(pad) around prior_xy."""
    def det(frame, floor_z, prior_xy, radius):
        ff = _floor.fit_floor(frame, floor_z, keep_xy=keep_xy)
        pri = None if prior_xy is None else (float(prior_xy[0]), float(prior_xy[1]), float(floor_z))
        t = _floor.landing_target(frame, ff, target='pad', prior_center_ned=pri, prior_radius_m=radius, keep_xy=keep_xy)
        fz = float(ff.d) if ff.found and np.isfinite(ff.d) else None
        return (t if t.found else None), fz
    return det


class PadApproach:
    def __init__(self, prior_center_ned: Sequence[float], params: Optional[ApproachParams] = None,
                 floor_z: float = 0.0, detector: Optional[Callable] = None):
        self.p = params or ApproachParams()
        self.prior = np.array([float(prior_center_ned[0]), float(prior_center_ned[1])])
        self.floor_z = float(floor_z)
        self._floor_hist: List[float] = []
        self.detector = detector or default_detector(self.p.keep_xy)
        self.phase = 'goto_standoff'
        self.t_phase = None
        self.sp = None                        # ramped body setpoint (3,)
        self.yaw_sp = None
        self.dets: List[tuple] = []           # (t, x, y, yaw, w, conf) all detections (lock search)
        self.acc: List[tuple] = []            # accepted into the estimate
        self.est: Optional[PadEstimate] = None
        self.locked = False
        self.standoff = None
        self.cruise_z = None
        self.scan_targets: List[float] = []
        self.last_acc_t = None
        self.hold_since = None
        self.prev = None                      # (t, body)
        self.speed = 0.0
        self.handoff_pos = None
        self.blind_len = None
        self.t_last: Optional[float] = None
        self._ext = None                      # pending external detection (observe())

    # ------------------------------------------------------------------ geometry helpers
    def body_h_for_cam(self, cam_h):
        return cam_h - self.p.cam_up_m

    def stop_cam_dist(self):
        return self.p.observe_cam_h_m / self.p.tan_half_vfov + self.p.stop_extra_m

    def _observe_z(self):
        return self.floor_z - self.body_h_for_cam(self.p.observe_cam_h_m)

    def _target_xy(self):
        return self.est.center_ned[:2] if self.est is not None else self.prior

    # ------------------------------------------------------------------ estimate
    def _weight(self, det):
        vis = float(det.factors.get('visible', 1.0)) if getattr(det, 'factors', None) else 1.0
        rng = float(det.range_m) if np.isfinite(getattr(det, 'range_m', math.nan)) else 2.0
        return float(det.confidence) * max(vis, 0.3) / max(rng, 0.6) ** 2

    def _recompute(self, t):
        a = np.array([d[1:6] for d in self.acc[-self.p.est_keep:]])
        w = a[:, 3]
        xy = (a[:, :2] * w[:, None]).sum(0) / w.sum()
        spread = float(math.sqrt(max((w * ((a[:, :2] - xy) ** 2).sum(1)).sum() / w.sum(), 0.0)))
        z4 = (w * np.exp(4j * a[:, 2])).sum()
        yaw = float(np.angle(z4) / 4.0) % (math.pi / 2)
        conf = float(min(1.0, np.mean(a[:, 4]) * min(1.0, len(a) / self.p.lock_n)))
        self.est = PadEstimate(np.array([xy[0], xy[1], self.floor_z]), yaw, conf, len(self.acc), self.locked, spread,
                               self.last_acc_t if self.last_acc_t is not None else t)

    def _try_lock(self, t):
        rec = [d for d in self.dets if t - d[0] <= self.p.lock_window_s]
        if len(rec) < self.p.lock_n:
            return False
        a = np.array([d[1:3] for d in rec]); w = np.array([d[4] for d in rec])
        # weighted medoid-ish centre: the detection with most weighted neighbours within lock_tol
        dist = np.hypot(a[:, None, 0] - a[None, :, 0], a[:, None, 1] - a[None, :, 1])
        score = ((dist <= self.p.lock_tol_m) * w[None, :]).sum(1)
        k = int(np.argmax(score))
        members = [rec[j] for j in np.nonzero(dist[k] <= self.p.lock_tol_m)[0]]
        if len(members) < self.p.lock_n:
            return False
        self.locked = True
        self.acc = list(members)
        self.last_acc_t = max(m[0] for m in members)
        self._recompute(t)
        return True

    def search(self):
        """(prior_xy, radius_m) the detector should use in the current phase, or None (pad below
        the image: blind / descent / done). For a node that runs landing_target itself."""
        if self.phase in ('goto_standoff', 'descend_observe', 'acquire'):
            return self.prior.copy(), self.p.search_radius_m
        if self.phase == 'scan':
            return (self.est.center_ned[:2] if self.est is not None else self.prior).copy(), self.p.scan_radius_m
        if self.phase in ('approach', 'settle') and self.est is not None:
            return self.est.center_ned[:2].copy(), self.p.track_radius_m
        return None

    def observe(self, t, detection, floor_z=None):
        """Feed a detection computed outside (mission-core path): `detection` = LandingTarget (or
        None = this image was searched and nothing was found), floor_z = floor NED z from fit_floor
        (or None). Consumed by the next update(); use update(..., frame=None) then."""
        self._ext = (t, detection, floor_z)

    def _detect(self, t, frame, ev):
        sr = self.search()
        if frame is not None:
            if sr is None:
                return None
            det, fz = self.detector(frame, self.floor_z, sr[0], sr[1])
        elif self._ext is not None:
            _, det, fz = self._ext
            self._ext = None
            if sr is None:
                return None
            if det is not None and getattr(det, 'found', True) and \
                    math.hypot(float(det.center_ned[0]) - sr[0][0], float(det.center_ned[1]) - sr[0][1]) > sr[1]:
                det = None                                # outside this phase's search region
        else:
            return None
        if fz is not None and abs(fz - self.floor_z) < 0.15:
            self._floor_hist = (self._floor_hist + [fz])[-15:]
            self.floor_z = float(np.median(self._floor_hist))
        if det is None or not getattr(det, 'found', True) or float(det.confidence) < self.p.min_confidence:
            return None
        x, y = float(det.center_ned[0]), float(det.center_ned[1])
        rec = (t, x, y, float(det.yaw_ned), self._weight(det), float(det.confidence))
        self.dets = [d for d in self.dets if t - d[0] <= max(self.p.lock_window_s, 5.0)] + [rec]
        if self.locked:
            if math.hypot(x - self.est.center_ned[0], y - self.est.center_ned[1]) <= max(self.p.track_gate_m, 2 * self.est.spread_m):
                self.acc.append(rec); self.last_acc_t = t
                self._recompute(t)
            else:
                ev.append(('pad_detection_rejected', dict(xy=[round(x, 3), round(y, 3)], est=[round(float(v), 3) for v in self.est.center_ned[:2]])))
        return det

    # ------------------------------------------------------------------ phases
    def _enter(self, phase, t, ev, **info):
        ev.append(('pad_phase', dict(phase=phase, **info)))
        self.phase = phase; self.t_phase = t

    def _standoff_from(self, body_xy):
        tgt = self._target_xy()
        u = tgt - body_xy
        n = float(np.linalg.norm(u))
        u = u / n if n > 1e-6 else np.array([math.cos(self.yaw_sp or 0.0), math.sin(self.yaw_sp or 0.0)])
        return tgt - u * self.p.standoff_m

    def update(self, t: float, frame=None, body_ned=None, yaw: Optional[float] = None) -> ApproachOutput:
        p = self.p; ev: List[Tuple[str, dict]] = []
        if body_ned is None:
            body_ned = frame.body_ned
        if yaw is None:
            yaw = float(frame.heading) if frame is not None else (self.yaw_sp or 0.0)
        body = np.asarray(body_ned, float).reshape(3)
        if frame is not None:                                  # camera offset in the yaw frame
            d = np.asarray(frame.cam_ned, float) - np.asarray(frame.body_ned, float)
            c, s = math.cos(yaw), math.sin(yaw)
            fwd = c * d[0] + s * d[1]
            if 0.0 <= fwd <= 0.5 and abs(d[2]) <= 0.3:
                p.cam_fwd_m, p.cam_up_m = float(fwd), float(-d[2])
            img = getattr(frame, 'image', None); K = getattr(frame, 'K', None)
            if img is not None and K is not None:                  # floor visible from h / tan ahead
                p.tan_half_vfov = (float(np.asarray(img).shape[0]) - float(K[3])) / float(K[1])
        if self.prev is not None and t > self.prev[0]:
            v = float(np.linalg.norm(body[:2] - self.prev[1][:2])) / (t - self.prev[0])
            self.speed = 0.6 * self.speed + 0.4 * v
        self.prev = (t, body.copy())
        dt = 0.0 if self.t_last is None else max(0.0, t - self.t_last)
        self.t_last = t
        if self.sp is None:
            self.sp = body.copy(); self.yaw_sp = float(yaw); self.t_phase = t
            self.cruise_z = float(body[2])
            self.standoff = self._standoff_from(body[:2])
            ev.append(('pad_phase', dict(phase=self.phase, standoff=[round(float(v), 3) for v in self.standoff])))

        det = self._detect(t, frame, ev)
        if not self.locked and self.phase in ('goto_standoff', 'descend_observe', 'acquire', 'scan') and self._try_lock(t):
            ev.append(('pad_locked', dict(center=[round(float(v), 3) for v in self.est.center_ned[:2]], n=self.est.n_frames,
                                          spread_m=round(self.est.spread_m, 3), phase=self.phase)))
        tgt = self._target_xy()
        goal = self.sp.copy(); goal_yaw = self.yaw_sp; v_xy = p.v_xy; v_z = p.v_z
        bearing = math.atan2(tgt[1] - body[1], tgt[0] - body[0])
        dxy = float(np.hypot(*(tgt - body[:2])))

        if self.phase == 'goto_standoff':
            goal = np.array([self.standoff[0], self.standoff[1], self.cruise_z]); goal_yaw = math.atan2(tgt[1] - self.standoff[1], tgt[0] - self.standoff[0])
            if np.hypot(*(body[:2] - self.standoff)) < p.pos_tol_m and abs(_wrap(yaw - goal_yaw)) < math.radians(10):
                self._enter('descend_observe', t, ev)
        if self.phase == 'descend_observe':
            goal = np.array([self.standoff[0], self.standoff[1], self._observe_z()]); goal_yaw = math.atan2(tgt[1] - self.standoff[1], tgt[0] - self.standoff[0])
            if abs(body[2] - goal[2]) < p.z_tol_m:
                self._enter('acquire', t, ev)
        if self.phase == 'acquire':
            goal = np.array([self.standoff[0], self.standoff[1], self._observe_z()]); goal_yaw = math.atan2(tgt[1] - self.standoff[1], tgt[0] - self.standoff[0])
            if self.locked:
                self._enter('approach', t, ev, center=[round(float(v), 3) for v in tgt])
                self.last_acc_t = t
            elif t - self.t_phase > p.acquire_timeout_s:
                b0 = goal_yaw; a = math.radians(p.scan_half_deg)
                self.scan_targets = [b0 + a, b0 - a, b0]
                self._enter('scan', t, ev, center_yaw_deg=round(math.degrees(b0), 1))
        if self.phase == 'scan':
            goal = np.array([self.standoff[0], self.standoff[1], self._observe_z()])
            if self.locked:
                self._enter('approach', t, ev, center=[round(float(v), 3) for v in self._target_xy()])
                self.last_acc_t = t
            elif self.scan_targets:
                goal_yaw = self.scan_targets[0]
                if abs(_wrap(self.yaw_sp - goal_yaw)) < math.radians(1.0) and abs(_wrap(yaw - goal_yaw)) < math.radians(5.0):
                    self.scan_targets.pop(0)
                if not self.scan_targets:
                    self.handoff_pos = body.copy()
                    self._enter('handoff', t, ev, reason='pad_not_found')
        if self.phase in ('approach', 'settle'):
            tgt = self.est.center_ned[:2]; dxy = float(np.hypot(*(tgt - body[:2])))
            u = (tgt - body[:2]) / max(dxy, 1e-6)
            stop_body = self.stop_cam_dist() + p.cam_fwd_m
            stop = tgt - u * stop_body
            goal = np.array([stop[0], stop[1], self._observe_z()]); goal_yaw = math.atan2(u[1], u[0]); v_xy = p.v_approach
            # the whole pad is in view only while its centre is >= stop + S/2 (+0.1) ahead of the camera;
            # closer, missing detections are expected (it is leaving the bottom of the image): no hold
            expect_full = (dxy - p.cam_fwd_m) >= self.stop_cam_dist() + 0.5 * p.pad_size_m + 0.1
            if self.phase == 'approach' and not expect_full:
                self.hold_since = None
            elif self.phase == 'approach':
                since = t - (self.last_acc_t if self.last_acc_t is not None else self.t_phase)
                if since > p.lost_hold_s:
                    if self.hold_since is None:
                        self.hold_since = t; ev.append(('pad_lost_hold', dict(since_s=round(since, 2))))
                    goal = self.sp.copy()                                  # freeze the carrot
                    if t - self.hold_since > p.relock_s:
                        if dxy <= p.blind_max_m + 0.3:
                            ev.append(('pad_lost_go_blind', dict(dist_m=round(dxy, 3))))
                            self._enter('settle', t, ev)
                        else:
                            self.locked = False; self.dets = []; self.hold_since = None
                            a = math.radians(p.scan_half_deg); b0 = goal_yaw
                            self.scan_targets = [b0 + a, b0 - a, b0]; self.standoff = body[:2].copy()
                            self._enter('scan', t, ev, reason='pad_lost')
                else:
                    if self.hold_since is not None:
                        ev.append(('pad_reacquired', dict(held_s=round(t - self.hold_since, 2))))
                    self.hold_since = None
            if self.phase == 'approach' and np.hypot(*(body[:2] - stop)) < p.pos_tol_m * 0.5 and self.speed < p.settle_speed:
                self._enter('settle', t, ev, cam_to_center_m=round(dxy - p.cam_fwd_m, 3))
            if self.phase == 'settle' and t - self.t_phase >= p.settle_s and self.speed < p.settle_speed:
                self.blind_len = dxy
                if dxy > p.blind_max_m:
                    self.handoff_pos = body.copy()
                    self._enter('handoff', t, ev, reason='blind_segment_too_long', dist_m=round(dxy, 3))
                else:
                    self._enter('blind', t, ev, dist_m=round(dxy, 3), center=[round(float(v), 3) for v in tgt])
        if self.phase == 'blind':
            tgt = self.est.center_ned[:2]
            goal = np.array([tgt[0], tgt[1], self._observe_z()]); goal_yaw = self.yaw_sp; v_xy = p.v_blind
            if np.hypot(*(body[:2] - tgt)) < p.xy_tol_m and self.speed < p.settle_speed:
                self._enter('final_descent', t, ev)
        if self.phase in ('final_descent', 'touchdown'):
            tgt = self.est.center_ned[:2]
            z_down = self.floor_z - p.touchdown_h_m + p.land_push_m
            goal = np.array([tgt[0], tgt[1], z_down]); goal_yaw = self.yaw_sp; v_xy = p.v_blind; v_z = p.v_land
            if np.hypot(*(body[:2] - tgt)) > p.land_xy_pause_m:
                goal[2] = self.sp[2]                                        # pause the descent, re-centre
            if self.phase == 'final_descent' and self.floor_z - body[2] <= p.touchdown_h_m:
                self._enter('touchdown', t, ev, xy_err_m=round(float(np.hypot(*(body[:2] - tgt))), 3))
        if self.phase == 'handoff':
            goal = self.handoff_pos.copy(); goal_yaw = self.yaw_sp

        # ---- ramp the setpoint toward the goal
        if dt > 0:
            dxy_sp = goal[:2] - self.sp[:2]; n = float(np.linalg.norm(dxy_sp))
            if n > v_xy * dt:
                dxy_sp *= v_xy * dt / n
            self.sp[:2] += dxy_sp
            dz = goal[2] - self.sp[2]
            self.sp[2] += float(np.clip(dz, -v_z * dt, v_z * dt))
            dy = _wrap(goal_yaw - self.yaw_sp)
            self.yaw_sp = _wrap(self.yaw_sp + float(np.clip(dy, -p.yaw_rate * dt, p.yaw_rate * dt)))
        out_pad = None
        if self.est is not None:
            out_pad = PadEstimate(self.est.center_ned.copy(), self.est.yaw_ned, self.est.confidence, self.est.n_frames,
                                  self.locked, self.est.spread_m, self.est.last_seen)
            out_pad.center_ned[2] = self.floor_z
        return ApproachOutput(self.phase, self.sp.copy(), float(self.yaw_sp), out_pad, det,
                              land=self.phase == 'touchdown', handoff=self.phase == 'handoff', events=ev)

