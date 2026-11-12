#!/usr/bin/env python3
"""Hand-carried walk-through bag -> course prior (START frame) for mission.core.

    python3 tools/survey_to_prior.py BAG --name venue [--out-dir DIR] [--stops stops.json]
           [--truth ubuntu_compat/course_prior.replica.json] [--list-stops]

BAG is a rosbag2 directory (metadata.yaml + *.mcap) or one .mcap file with the flight-bag topics
(/stereo/left/image, /camera/depth/image_raw + camera_info, /tf_static, PX4 local position, attitude,
distance sensor). The walk follows SURVEY_PROTOCOL.md. Outputs in --out-dir:
    course_prior.<name>.json          the prior; `_comment` lists every field's source, frames, spread
    course_prior.<name>.png           top-down overlay: path, stops, detections, prior (+ truth)
    course_prior.<name>.report.json   stops, detections, warnings, per-field truth errors (--truth)

Pipeline
  1. poses -> still stops (|v| < --still-speed for >= --still-min-s); first stop = START frame
     (mission.core.StartFrame of its median pose and heading, like the mission at arming).
  2. detectors (perception/) on frames of every stop, prior = a hypothesis 1.5-1.8 m ahead of the
     camera (and --seed-prior positions if given); confident results are clustered per kind.
  3. refinement: the best cluster becomes the prior (as in flight) for every frame that sees it
     at a useful range; the final value is the median of the consistent detections.
  4. remaining stops by geometry: wall (side / top / side triple), rock top (pose + ToF), net
     touches (extreme stops) -> fence; --stops overrides any assignment.
  5. corridor = the carried path from the start pad to the tunnel stop, smoothed and simplified.
Missing or doubtful fields are reported as warnings and left null (or taken from --fallback, marked so);
the tool never fills a field from nothing.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from common.geometry import (Frame, camera_pose_ned, OFFICIAL_K, OFFICIAL_R_BASE_CAM,     # noqa: E402
                             OFFICIAL_T_BASE_CAM, SKY_BETA_HALF_SPAN_M)
from mission.core import CoursePrior, StartFrame                                         # noqa: E402
import survey_bag as sb                                                                  # noqa: E402

T_LPOS, T_ATT, T_TOF = '/fmu/out/vehicle_local_position_v1', '/fmu/out/vehicle_attitude', '/fmu/out/distance_sensor'
T_TF, T_INFO, T_DEPTH, T_LEFT = '/tf_static', '/camera/depth/camera_info', '/camera/depth/image_raw', '/stereo/left/image'
KINDS = ('tunnel', 'window', 'landing')
KIND_STOP = {'tunnel': 'tunnel', 'window': 'window', 'landing': 'rock_view'}
LABELS = ('start', 'end', 'tunnel', 'window', 'rock_view', 'rock_top', 'fence_front', 'fence_back', 'fence_left',
          'fence_right', 'wall_from', 'wall_top', 'wall_to', 'skip')
FENCE_DIRS = {'fence_front': ('forward_max', 0, 1.0), 'fence_back': ('forward_min', 0, -1.0),
              'fence_left': ('left_max', 1, 1.0), 'fence_right': ('left_min', 1, -1.0)}
FENCE_YAW = {'fence_front': 0.0, 'fence_back': 180.0, 'fence_left': 90.0, 'fence_right': -90.0}


def _r(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def wrap_deg(a):
    return (float(a) + 180.0) % 360.0 - 180.0


# =========================================================================== 1. poses
class Track:
    """Small topics of the bag (pass A) + header stamps of every image/depth message."""

    def __init__(self):
        self.lpos, self.att, self.tof = [], [], []
        self.depth_meta, self.left_meta = [], []                          # (log_t, stamp, shape)
        self.K = None; self.size = None; self.depth_frame = None
        self.tf = sb.TfTree(); self.R_bc = None; self.t_bc = None; self.tf_source = ''

    def finish(self):
        L = np.array(self.lpos, float).reshape(-1, 10)
        if len(L) < 20:
            raise RuntimeError('bag has %d %s messages: need the PX4 local position' % (len(L), T_LPOS))
        o = np.argsort(L[:, 0], kind='stable'); L = L[o]
        self.t, self.p, self.v, self.heading, self.valid = L[:, 0], L[:, 1:4], L[:, 4:7], L[:, 7], L[:, 8] > 0.5
        A = np.array(self.att, float).reshape(-1, 5)
        self.att_t, self.att_q = (A[:, 0], A[:, 1:5]) if len(A) else (np.zeros(0), np.zeros((0, 4)))
        D = np.array(self.tof, float).reshape(-1, 3)
        self.tof_t, self.tof_d, self.tof_ok = D[:, 0], D[:, 1], D[:, 2] > 0.5
        self.dmeta = np.array([(a, b) for a, b, _ in self.depth_meta], float).reshape(-1, 2)
        self.lmeta = np.array([(a, b) for a, b, _ in self.left_meta], float).reshape(-1, 2)
        # depth i <-> left j by header stamp (the node pairs within 0.04 s)
        self.pair = {}
        if len(self.lmeta) and len(self.dmeta):
            ls = self.lmeta[:, 1]; order = np.argsort(ls); lss = ls[order]
            for i, st in enumerate(self.dmeta[:, 1]):
                k = int(np.searchsorted(lss, st))
                best = None
                for c in (k - 1, k):
                    if 0 <= c < len(lss) and (best is None or abs(lss[c] - st) < abs(lss[best] - st)):
                        best = c
                if best is not None and abs(lss[best] - st) < 0.04 and \
                        self.left_meta[order[best]][2] == self.depth_meta[i][2]:
                    self.pair[i] = int(order[best])
        lookup = None if self.depth_frame is None else self.tf.lookup('base_link', self.depth_frame)
        if lookup is not None:
            self.R_bc, self.t_bc = lookup; self.tf_source = 'tf_static base_link<-%s' % self.depth_frame
        else:
            self.R_bc, self.t_bc = np.asarray(OFFICIAL_R_BASE_CAM, float), np.asarray(OFFICIAL_T_BASE_CAM, float)
            self.tf_source = 'DEFAULT official rig (no tf base_link<-%s in bag)' % self.depth_frame
        if self.K is None:
            self.K = tuple(OFFICIAL_K); self.tf_source += '; DEFAULT K (no camera_info)'

    def cut_divergence(self, max_speed=5.0, max_dist=50.0):
        """Truncate at the first sample a hand walk cannot produce (VIO divergence / fly-away); returns cut time."""
        dt = np.diff(self.t)
        jump = np.r_[0.0, np.linalg.norm(np.diff(self.p, axis=0), axis=1) / np.maximum(dt, 1e-3)]
        bad = (np.linalg.norm(self.v, axis=1) > max_speed) | (jump > 2 * max_speed) | \
              (np.linalg.norm(self.p - self.p[0], axis=1) > max_dist) | ~np.isfinite(self.p).all(1)
        k = np.nonzero(bad)[0]
        if not len(k):
            return None
        i = int(k[0]); t_cut = float(self.t[max(0, i - 1)])
        self.t, self.p, self.v, self.heading, self.valid = self.t[:i], self.p[:i], self.v[:i], self.heading[:i], self.valid[:i]
        keep = self.dmeta[:, 0] <= t_cut if len(self.dmeta) else np.zeros(0, bool)
        self.pair = {d: l for d, l in self.pair.items() if keep[d]}
        return t_cut

    def pos(self, t):
        return np.array([np.interp(t, self.t, self.p[:, k]) for k in range(3)])

    def head(self, t):
        c = np.interp(t, self.t, np.cos(self.heading)); s = np.interp(t, self.t, np.sin(self.heading))
        return math.atan2(s, c)

    def q(self, t):
        if len(self.att_t) == 0:
            h = self.head(t); return (math.cos(h / 2), 0.0, 0.0, math.sin(h / 2))
        k = int(np.clip(np.searchsorted(self.att_t, t), 1, len(self.att_t) - 1))
        k = k - 1 if abs(self.att_t[k - 1] - t) < abs(self.att_t[k] - t) else k
        return tuple(float(c) for c in self.att_q[k])

    def tof_at(self, t0, t1):
        m = (self.tof_t >= t0) & (self.tof_t <= t1)
        ok = m & self.tof_ok
        return (float(np.median(self.tof_d[ok])) if ok.any() else None), int(m.sum()), int(ok.sum())

    def cam(self, t):
        return camera_pose_ned(self.q(t), self.pos(t), self.R_bc, self.t_bc)


def load_track(bag, reader):
    tr = Track()
    topics = {T_LPOS, T_ATT, T_TOF, T_TF, T_INFO, T_DEPTH, T_LEFT}
    F = sb.field
    for topic, t, m in sb.open_bag(bag, topics, reader):
        if topic == T_LPOS:
            tr.lpos.append((t, F(m, 'x'), F(m, 'y'), F(m, 'z'), F(m, 'vx'), F(m, 'vy'), F(m, 'vz'), F(m, 'heading'),
                            float(bool(F(m, 'xy_valid')) and bool(F(m, 'z_valid'))), 0.0))
        elif topic == T_ATT:
            q = F(m, 'q'); tr.att.append((t, float(q[0]), float(q[1]), float(q[2]), float(q[3])))
        elif topic == T_TOF:
            d = float(F(m, 'current_distance'))
            ok = F(m, 'min_distance') <= d <= F(m, 'max_distance') and F(m, 'signal_quality') != 0
            tr.tof.append((t, d, float(ok)))
        elif topic == T_TF:
            tr.tf.add_msg(m)
        elif topic == T_INFO:
            if tr.K is None:
                k = F(m, 'k')
                if k[0] > 0:
                    tr.K = (float(k[0]), float(k[4]), float(k[2]), float(k[5]))
                    tr.size = (int(F(m, 'width')), int(F(m, 'height')))
                    tr.depth_frame = str(F(F(m, 'header'), 'frame_id')).lstrip('/')
        elif topic == T_DEPTH:
            tr.depth_meta.append((t, sb.stamp_s(m), (int(F(m, 'height')), int(F(m, 'width')))))
            if tr.depth_frame is None:
                tr.depth_frame = str(F(F(m, 'header'), 'frame_id')).lstrip('/')
        elif topic == T_LEFT:
            tr.left_meta.append((t, sb.stamp_s(m), (int(F(m, 'height')), int(F(m, 'width')))))
    tr.finish()
    return tr


def iter_frames(bag, reader, tr, want):
    """Stream the bag again; yield (depth index, Frame) for depth indices in `want` that have a paired
    left image. Pose = PX4 pose at the depth message's log time (as the node does)."""
    want = set(int(i) for i in want if int(i) in tr.pair)
    need_left = {tr.pair[i]: i for i in want}
    pend_d, pend_l = {}, {}
    di = li = -1
    for topic, t, m in sb.open_bag(bag, {T_DEPTH, T_LEFT}, reader):
        if topic == T_DEPTH:
            di += 1
            if di in want:
                pend_d[di] = (t, sb.image_array(m))
        else:
            li += 1
            if li in need_left:
                pend_l[li] = sb.image_array(m)
        for i in [i for i in pend_d if tr.pair[i] in pend_l]:
            t_d, d = pend_d.pop(i); img = pend_l.pop(tr.pair[i])
            if d is None or img is None or img.shape != d.shape:
                continue
            R_nc, cam = tr.cam(t_d)
            yield i, Frame(image=img, depth=d, K=tr.K, R_nc=R_nc, cam_ned=cam, stamp=t_d,
                           body_ned=tr.pos(t_d), heading=tr.head(t_d))
        if len(pend_d) > 40:
            for k in sorted(pend_d)[:-20]:
                pend_d.pop(k)
        if len(pend_l) > 40:
            for k in sorted(pend_l)[:-20]:
                pend_l.pop(k)


# =========================================================================== 2. stops
class Stop:
    def __init__(self, idx, t0, t1):
        self.idx, self.t0, self.t1 = idx, float(t0), float(t1)
        self.label = None; self.why = ''; self.user = False
        self.det = {k: 0 for k in KINDS}

    @property
    def dur(self):
        return self.t1 - self.t0

    def summary(self, t_ref):
        return dict(idx=self.idx, t0=_r(self.t0 - t_ref, 2), dur=_r(self.dur, 1), flu=[_r(c, 3) for c in self.flu],
                    yaw_deg=_r(self.yaw_deg, 1), tof=_r(self.tof, 3), surface_up=_r(self.surface_up, 3),
                    std_m=_r(self.std, 3), label=self.label, why=self.why, user=self.user, det=self.det)


def smooth(x, n):
    if n <= 1 or len(x) < n:
        return x
    k = np.ones(n) / n
    pad = np.pad(x, (n // 2, n - 1 - n // 2), mode='edge')
    return np.convolve(pad, k, mode='valid')


def find_stops(tr, v_thr=0.05, min_s=2.0, yaw_rate_dps=12.0, gap_s=0.4):
    t = tr.t
    dt = float(np.median(np.diff(t))) if len(t) > 2 else 0.02
    n = max(1, int(round(0.5 / max(dt, 1e-3))))
    speed = smooth(np.linalg.norm(tr.v, axis=1), n)
    hd = np.unwrap(tr.heading)
    yr = np.abs(np.gradient(hd, t)) if len(t) > 2 else np.zeros_like(t)
    yr = np.degrees(smooth(yr, n))
    still = (speed < v_thr) & (yr < yaw_rate_dps) & tr.valid
    runs, i = [], 0
    while i < len(t):
        if still[i]:
            j = i
            while j + 1 < len(t) and still[j + 1]:
                j += 1
            runs.append([t[i], t[j]]); i = j + 1
        else:
            i += 1
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < gap_s:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [Stop(k, a, b) for k, (a, b) in enumerate([r for r in merged if r[1] - r[0] >= min_s])]


def describe_stops(tr, stops, sf, rest_h, tof_rest):
    for s in stops:
        m = (tr.t >= s.t0) & (tr.t <= s.t1)
        P = tr.p[m]
        s.ned = np.median(P, 0)
        s.std = float(np.max(np.std(P, 0))) if len(P) > 1 else 0.0
        s.flu = sf.from_ned(s.ned)
        h = math.atan2(np.median(np.sin(tr.heading[m])), np.median(np.cos(tr.heading[m])))
        s.heading = h
        s.yaw_deg = wrap_deg(math.degrees(sf.yaw - h))                  # START: CCW from forward (NED heading is CW)
        s.tof, s.n_tof, s.n_tof_ok = tr.tof_at(s.t0, s.t1)
        # surface under the drone (START up): resting -> up - rest_h; ToF corrects a held-above drone
        if s.tof is not None and tof_rest is not None:
            s.surface_up = float(s.flu[2] - (s.tof - tof_rest) - rest_h); s.surface_src = 'pose+tof'
        else:
            s.surface_up = float(s.flu[2] - rest_h); s.surface_src = 'pose (resting assumed, no ToF)'


# =========================================================================== 3. detectors
class Detectors:
    def __init__(self, args, floor_prior_d):
        from perception import floor, tunnel, window
        self.floor, self.tunnel, self.window = floor, tunnel, window
        self.a = args; self.floor_prior_d = floor_prior_d
        tp = {}
        if args.tunnel_radius:
            tp['radius_m'] = args.tunnel_radius
        if args.hoop_spacing:
            tp['hoop_spacing_m'] = args.hoop_spacing
        self.tparams = tunnel.TunnelParams(**tp) if tp else None
        self.errors = 0; self.calls = {k: 0 for k in KINDS}; self.ms = {k: 0.0 for k in KINDS}

    def run(self, kind, fr, prior, axis=None):
        t0 = time.perf_counter()
        try:
            fl = self.floor.fit_floor(fr, prior_floor_d=self.floor_prior_d)
            floor_d = fl.d if fl.found else self.floor_prior_d
            if kind == 'tunnel':
                est = self.tunnel.detect(fr, prior_entry_ned=prior, floor_z_ned=floor_d,
                                         prior_axis_ned=None if axis is None else np.asarray(axis, float)[:2],
                                         params=self.tparams)
                ok = bool(est.found) and est.confidence >= self.a.tunnel_min_conf and bool(getattr(est, 'align_ok', True))
                if not ok:
                    return None
                rings = [np.asarray(r.center_ned, float) for r in est.rings]
                return dict(pos=np.asarray(est.entry_ned, float), conf=float(est.confidence),
                            axis=np.asarray(est.axis_ned, float)[:2], radius=float(est.radius_m),
                            n_rings=len(rings), span=float(np.linalg.norm(rings[-1] - rings[0])) if len(rings) >= 2 else None)
            if kind == 'window':
                est = self.window.detect(fr, self.a.half_span, self.a.half_height, prior_center_ned=prior,
                                         known_size_m=self.a.window_size, floor_z_ned=floor_d)
                if not (est.found and est.confidence >= self.a.window_min_conf):
                    return None
                return dict(pos=np.asarray(est.center_ned, float), conf=float(est.confidence),
                            normal=np.asarray(est.normal_ned, float), w=float(est.width_m), h=float(est.height_m),
                            range=float(est.range_m), clear=bool(est.clearance_ok))
            est = self.floor.landing_target(fr, fl if fl.found else None, prior_center_ned=prior, bright_mode='soft')
            if not (est.found and est.confidence >= self.a.landing_min_conf):
                return None
            lo, sh = (float(x) for x in est.extent_xy_m)
            if not (sh >= self.a.rock_min_short and lo <= self.a.rock_max_aspect * sh):
                return None                                               # rails, strips, board edges
            return dict(pos=np.asarray(est.center_ned, float), conf=float(est.confidence), top_d=float(est.top_d),
                        extent=[float(x) for x in est.extent_xy_m], range=float(est.range_m))
        except Exception as e:  # noqa: BLE001 - one bad frame must not stop the survey
            self.errors += 1
            if self.errors <= 3:
                print('  detector %s error: %r' % (kind, e), file=sys.stderr)
            return None
        finally:
            self.calls[kind] += 1; self.ms[kind] += 1e3 * (time.perf_counter() - t0)


def horiz_dir(R_nc):
    f = np.asarray(R_nc, float)[:, 2].copy(); f[2] = 0.0
    n = np.linalg.norm(f)
    return f / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])


def in_view(fr, p, margin=25, rng=(0.5, 4.0)):
    uv, z = fr.project(np.asarray(p, float)[None])
    H, W = fr.depth.shape
    u, v = uv[0]
    return bool(np.isfinite(u) and rng[0] < z[0] < rng[1] and margin < u < W - margin and 10 < v < H - 10)


def cluster(dets, radius):
    """Greedy clustering by distance to the running median; returns clusters sorted by sum(conf)."""
    cl = []
    for d in sorted(dets, key=lambda d: -d['conf']):
        for c in cl:
            if np.linalg.norm(d['pos'] - c['center']) < radius:
                c['m'].append(d); c['center'] = np.median([x['pos'] for x in c['m']], 0); break
        else:
            cl.append(dict(m=[d], center=d['pos'].copy()))
    for c in cl:
        # a window the vehicle provably fits through (clearance_ok) counts 3x: net/hall texture rarely passes it
        c['score'] = float(sum(x['conf'] * x.get('w', 1.0) * (1.5 if x.get('clear') else 0.5 if 'clear' in x else 1.0)
                               for x in c['m']))
    return sorted(cl, key=lambda c: -c['score'])


# =========================================================================== 4. assembly helpers
def rdp(P, eps):
    if len(P) < 3:
        return [0, len(P) - 1]
    a, b = P[0], P[-1]
    ab = b - a; L = np.linalg.norm(ab)
    if L < 1e-9:
        d = np.linalg.norm(P - a, axis=1)
    else:
        d = np.abs(ab[0] * (P[:, 1] - a[1]) - ab[1] * (P[:, 0] - a[0])) / L
    k = int(np.argmax(d))
    if d[k] <= eps:
        return [0, len(P) - 1]
    left = rdp(P[:k + 1], eps); right = rdp(P[k:], eps)
    return left[:-1] + [k + r for r in right]


def load_stop_labels(path, n):
    txt = Path(path).read_text()
    out = {}
    try:
        js = json.loads(txt)
        if isinstance(js, list):
            out = {i: str(v) for i, v in enumerate(js) if v not in (None, '', 'auto')}
        else:
            out = {int(k): str(v) for k, v in js.items() if not str(k).startswith('_') and v not in (None, '', 'auto')}
    except ValueError:
        for line in txt.splitlines():
            line = line.split('#', 1)[0].split()
            if len(line) >= 2:
                out[int(line[0])] = line[1]
    bad = {k: v for k, v in out.items() if v not in LABELS or not 0 <= k < n}
    if bad:
        raise SystemExit('--stops: bad entries %s (labels: %s; %d stops)' % (bad, ', '.join(LABELS), n))
    return out


def med_spread(P):
    P = np.asarray(P, float)
    c = np.median(P, 0)
    d = np.linalg.norm(P - c, axis=1)
    return c, (float(np.percentile(d, 90)) if len(d) > 1 else 0.0)


# =========================================================================== main survey
class Survey:
    def __init__(self, args):
        self.a = args
        self.warn = []
        self.src = {}                                                     # field -> provenance dict

    def W(self, msg):
        self.warn.append(msg); print('WARNING: ' + msg)

    def note(self, key, source, n=None, spread=None, **kw):
        d = dict(source=source)
        if n is not None:
            d['n'] = int(n)
        if spread is not None:
            d['spread_m'] = _r(spread, 3)
        d.update({k: v for k, v in kw.items() if v is not None})
        self.src[key] = d

    # ------------------------------------------------------------------ pass A
    def load(self):
        a = self.a
        t0 = time.time()
        self.reader = sb.resolve_reader(a.reader, a.bag, {T_LPOS, T_ATT, T_TOF, T_TF, T_INFO, T_DEPTH, T_LEFT})
        print('reading %s (%.0f MB) with reader=%s' % (a.bag, sb.file_size_mb(a.bag), self.reader))
        tr = self.tr = load_track(a.bag, self.reader)
        print('  %d poses, %d attitude, %d ToF, %d depth (%d paired with left), %.1f s; %s' % (
            len(tr.t), len(tr.att_t), len(tr.tof_t), len(tr.dmeta), len(tr.pair), time.time() - t0, tr.tf_source))
        if not len(tr.dmeta):
            self.W('no depth frames in the bag: the detectors cannot run (record %s + %s)' % (T_DEPTH, T_LEFT))
        elif not tr.pair:
            self.W('no depth frame has a left image within 0.04 s: the detectors cannot run')
        elif 'DEFAULT' in tr.tf_source:
            self.W('camera extrinsic/K not in the bag: ' + tr.tf_source)
        self.t_ref = float(tr.t[0])
        t_cut = tr.cut_divergence()
        if t_cut is not None:
            self.W('pose diverges at t=%.1f s (speed > 5 m/s or > 50 m from start: VIO lost / fly-away): '
                   'everything after it is ignored' % (t_cut - self.t_ref))
        stops = self.stops = find_stops(tr, a.still_speed, a.still_min_s, a.still_yaw_rate)
        if not stops:
            raise SystemExit('no still stops found (|v| < %.2f m/s for >= %.1f s): check the bag / --still-speed' % (a.still_speed, a.still_min_s))
        st = stops[0]
        if st.t0 - self.t_ref > 15.0:
            self.W('first still stop starts %.1f s into the bag: START frame = that stop (protocol: record from the pad)' % (st.t0 - self.t_ref))
        m = (tr.t >= st.t0) & (tr.t <= st.t1)
        home = np.median(tr.p[m], 0)
        yaw = math.atan2(np.median(np.sin(tr.heading[m])), np.median(np.cos(tr.heading[m])))
        self.sf = StartFrame(home, yaw)
        self.floor_prior_d = float(home[2] + a.rest_height)
        tof_rest, _, n_ok = tr.tof_at(st.t0, st.t1)
        self.tof_rest = tof_rest
        if tof_rest is None:
            self.W('ToF invalid on the start pad: rock/wall heights assume the drone RESTS on them')
        describe_stops(tr, stops, self.sf, a.rest_height, tof_rest)
        self.note('START', 'stop #0', n=int(m.sum()), spread=stops[0].std, home_ned=[_r(c) for c in home],
                  yaw_deg_ned=_r(math.degrees(yaw), 2), tof_rest=_r(tof_rest))
        self.labels_user = load_stop_labels(a.stops, len(stops)) if a.stops else {}
        for k, v in self.labels_user.items():
            stops[k].label, stops[k].user, stops[k].why = v, True, '--stops'
        if stops[0].label is None:
            stops[0].label, stops[0].why = 'start', 'first still stop'
        last = stops[-1]
        if last is not stops[0] and last.label is None and np.hypot(*last.flu[:2]) < 0.5 and abs(last.flu[2]) < 0.2:
            last.label, last.why = 'end', 'last stop, back on the pad'
        end = next((s for s in stops if s.label == 'end'), None)
        if end is not None and a.closure_correct:
            d = end.ned - stops[0].ned
            w = np.clip((tr.t - stops[0].t1) / max(1e-3, end.t0 - stops[0].t1), 0.0, 1.0)
            tr.p = tr.p - np.outer(w, d)
            describe_stops(tr, stops, self.sf, a.rest_height, tof_rest)
            print('  --closure-correct: removed %s m of drift, linearly in time between the pad stops' % np.round(d, 3).tolist())
            self.note('closure_correct', 'linear in time, translation only', None, None, removed_ned=[_r(x) for x in d])
        if end is not None:
            dyaw = wrap_deg(end.yaw_deg)
            self.closure = dict(dxy=_r(np.hypot(*end.flu[:2])), dup=_r(end.flu[2]), dyaw_deg=_r(dyaw, 2))
            print('  closure (end pad vs start pad): %.3f m horizontal, %.3f m up, %.2f deg yaw' % (
                np.hypot(*end.flu[:2]), end.flu[2], dyaw))
            if np.hypot(*end.flu[:2]) > a.closure_warn_m or abs(dyaw) > 3.0:
                self.W('closure error %.2f m / %.1f deg: VIO drift or the drone was not put back on its marks; '
                       'far fields may be off by up to that' % (np.hypot(*end.flu[:2]), dyaw))
        else:
            self.closure = None
            self.W('walk did not end back on the pad: no drift (closure) check')
        self.print_stops()

    def print_stops(self):
        print('\n  #   t0(s)  dur  forward   left     up   yaw  tof    surf  label')
        for s in self.stops:
            print('%3d %7.1f %4.1f %7.2f %7.2f %6.2f %5.0f %5s %6.2f  %s%s' % (
                s.idx, s.t0 - self.t_ref, s.dur, s.flu[0], s.flu[1], s.flu[2], s.yaw_deg,
                '-' if s.tof is None else '%.2f' % s.tof, s.surface_up, s.label or '', ' (user)' if s.user else ''))
        print()

    # ------------------------------------------------------------------ pass B/C
    def detect(self):
        a, tr, sf = self.a, self.tr, self.sf
        self.dets = {k: [] for k in KINDS}
        self.final = {}
        if a.no_detect:
            self.W('--no-detect: tunnel/window/rock come only from stops or --fallback')
            return
        try:
            self.D = Detectors(a, self.floor_prior_d)
        except ImportError as e:
            self.W('perception detectors not importable (%s): no detector fields' % e)
            return
        # which (stop, kind) to try: labelled stops only their kind; unlabelled stops every kind
        jobs = {}                                                         # depth idx -> list of (kind, stop)
        stop_of = {}
        for s in self.stops:
            if s.label in ('start', 'end', 'skip'):
                continue
            kinds = [k for k in KINDS if KIND_STOP[k] == s.label] if s.label else list(KINDS)
            if s.label and not kinds:
                continue
            idx = [i for i in np.nonzero((tr.dmeta[:, 0] >= s.t0) & (tr.dmeta[:, 0] <= s.t1))[0] if i in tr.pair]
            if not idx:
                continue
            pick = np.unique(np.linspace(0, len(idx) - 1, min(len(idx), a.frames_per_stop)).round().astype(int))
            for k in pick:
                jobs.setdefault(idx[k], []).extend((kd, s) for kd in kinds)
                stop_of[idx[k]] = s
        if a.scan_stride > 0:                                            # moving frames: in case a pause was missed
            busy = [(s.t0, s.t1) for s in self.stops if s.label in ('start', 'end', 'skip')]
            spd = np.linalg.norm(tr.v, axis=1)
            for i in range(0, len(tr.dmeta), a.scan_stride):
                t_ = tr.dmeta[i, 0]
                if i in jobs or i not in tr.pair or any(t0_ <= t_ <= t1_ for t0_, t1_ in busy):
                    continue
                if np.interp(t_, tr.t, spd) > a.scan_max_speed or -np.interp(t_, tr.t, tr.p[:, 2]) + sf.o[2] < 0.25:
                    continue
                jobs.setdefault(i, []).extend((kd, 'scan') for kd in KINDS)
        seeds = self.seed_positions()
        if seeds:
            for i in range(0, len(tr.dmeta), a.refine_stride):
                if i not in tr.pair:
                    continue
                R_nc, cam = tr.cam(tr.dmeta[i, 0])
                fr = Frame(image=None, depth=np.zeros(tr.size[::-1] if tr.size else (400, 640)), K=tr.K, R_nc=R_nc, cam_ned=cam)
                for kd, p in seeds.items():
                    if in_view(fr, p, rng=(1.0, 3.5)):
                        jobs.setdefault(i, []).append((kd, 'seed'))
        t0 = time.time()
        print('pass 1: %d frames, %d detector calls (stop hypotheses%s%s)' % (
            len(jobs), sum(len(v) for v in jobs.values()), ' + moving-frame scan' if a.scan_stride > 0 else '',
            ' + --seed-prior' if seeds else ''))
        for i, fr in iter_frames(a.bag, self.reader, tr, jobs.keys()):
            fwd = horiz_dir(fr.R_nc)
            for kd, s in jobs[i]:
                if s == 'seed':
                    prior = seeds[kd]; axis = None
                else:
                    D = {'tunnel': a.tunnel_standoff, 'window': a.window_standoff, 'landing': a.rock_standoff}[kd]
                    prior = fr.cam_ned + D * fwd
                    if kd == 'landing':
                        prior[2] = self.floor_prior_d
                    axis = fwd if kd == 'tunnel' else None
                r = self.D.run(kd, fr, prior, axis)
                if r is not None:
                    st_ = s if isinstance(s, Stop) else None
                    r.update(t=fr.stamp, i=i, stop=None if st_ is None else st_.idx, cam=fr.cam_ned.copy(), pass_=1,
                             w=1.0 if s != 'scan' else a.scan_weight, src='stop' if st_ else s)
                    self.dets[kd].append(r)
                    if st_ is not None:
                        st_.det[kd] += 1
        print('  pass 1 %.1f s: %s' % (time.time() - t0, ', '.join('%s %d' % (k, len(v)) for k, v in self.dets.items())))
        best = {}
        self.clusters = {}
        for kd in KINDS:
            cl = cluster(self.dets[kd], a.cluster_m)
            self.clusters[kd] = [dict(flu=[_r(x) for x in self.sf.from_ned(c['center'])], n=len(c['m']), score=_r(c['score'], 2),
                                      n_clear=sum(bool(m.get('clear')) for m in c['m']) if kd == 'window' else None,
                                      stops=sorted({m['stop'] for m in c['m'] if m['stop'] is not None}))
                                 for c in cl[:6]]
            cl = [c for c in cl if len(c['m']) >= 2] or cl[:1]
            if not cl:
                continue
            if kd == 'window':                                            # an opening the vehicle fits through wins
                cc = [c for c in cl if sum(bool(m.get('clear')) for m in c['m']) >= 2]
                if cc and cc[0] is not cl[0]:
                    print('  window: preferring the cluster with clearance_ok frames over a stronger one without')
                    cl = [cc[0]] + [c for c in cl if c is not cc[0]]
            best[kd] = cl[0]; upstream = False
            if kd == 'tunnel':
                b0 = cl[0]; ax = np.mean([np.r_[m['axis'], 0.0] for m in b0['m']], 0); ax /= np.linalg.norm(ax)
                up_ = []
                for c in cl[1:]:
                    d_ = c['center'] - b0['center']
                    cax = np.mean([np.r_[m['axis'], 0.0] for m in c['m']], 0); cax /= np.linalg.norm(cax)
                    along = float(d_ @ ax); lat = float(np.linalg.norm(d_[:2] - along * ax[:2]))
                    if len(c['m']) >= 3 and c['score'] >= 0.3 * b0['score'] and -1.6 <= along <= -0.3 and lat < 0.2 \
                            and abs(d_[2]) < 0.15 and cax @ ax > math.cos(math.radians(15)):
                        up_.append((along, c))
                if up_:
                    along, c = min(up_, key=lambda x: x[0])
                    self.W('tunnel: %d detection clusters on one axis; entry = the upstream one (%.2f m before the '
                           'strongest, which is a later hoop)' % (len(up_) + 1, -along))
                    cl = [c] + [x for x in cl if x is not c]
                    best[kd] = c; upstream = True
            if kd == 'tunnel' and not upstream and len(cl) > 1:
                ax = np.mean([np.r_[m['axis'], 0.0] for m in best[kd]['m']], 0); ax /= np.linalg.norm(ax)
                down = [c for c in cl[1:] if 0.3 <= (c['center'] - best[kd]['center']) @ ax <= 1.6 and np.linalg.norm(
                    (c['center'] - best[kd]['center'])[:2] - ((c['center'] - best[kd]['center']) @ ax) * ax[:2]) < 0.2]
                if down:
                    print('  tunnel: %d downstream cluster(s) on the axis = later hoops, ignored' % len(down))
                    cl = [cl[0]] + [c for c in cl[1:] if all(c is not d for d in down)]
            if len(cl) > 1 and not upstream and cl[1]['score'] > 0.5 * cl[0]['score'] and len(cl[1]['m']) >= 2:
                self.W('%s: two detection clusters %.2f m apart (scores %.1f / %.1f): took the stronger; '
                       'label the right stop in --stops if wrong' % (
                           kd, np.linalg.norm(cl[0]['center'] - cl[1]['center']), cl[0]['score'], cl[1]['score']))
        # pass 2: the pass-1 cluster as the prior, every frame that sees it at a useful range
        jobs2 = {}
        rngs = {'tunnel': a.tunnel_range, 'window': a.window_range, 'landing': a.rock_range}
        for i in range(0, len(tr.dmeta), a.refine_stride):
            if i not in tr.pair:
                continue
            R_nc, cam = tr.cam(tr.dmeta[i, 0])
            fr = Frame(image=None, depth=np.zeros((tr.size[1], tr.size[0]) if tr.size else (400, 640)), K=tr.K, R_nc=R_nc, cam_ned=cam)
            for kd, c in best.items():
                if in_view(fr, c['center'], rng=rngs[kd]):
                    jobs2.setdefault(i, []).append(kd)
        for kd in best:
            ids = [i for i, v in jobs2.items() if kd in v]
            if len(ids) > a.max_refine_frames:
                keep = set(np.asarray(ids)[np.unique(np.linspace(0, len(ids) - 1, a.max_refine_frames).round().astype(int))].tolist())
                for i in ids:
                    if i not in keep:
                        jobs2[i].remove(kd)
        jobs2 = {i: v for i, v in jobs2.items() if v}
        t0 = time.time()
        print('pass 2: %d frames, %d detector calls (prior = pass-1 cluster)' % (len(jobs2), sum(len(v) for v in jobs2.values())))
        refined = {k: [] for k in KINDS}
        for i, fr in iter_frames(a.bag, self.reader, tr, jobs2.keys()):
            for kd in jobs2[i]:
                c = best[kd]
                axis = None
                if kd == 'tunnel':
                    axis = np.mean([m['axis'] for m in c['m']], 0)
                r = self.D.run(kd, fr, c['center'], axis)
                if r is not None:
                    r.update(t=fr.stamp, i=i, stop=None, cam=fr.cam_ned.copy(), pass_=2, w=1.0, src='refine')
                    refined[kd].append(r)
        print('  pass 2 %.1f s: %s' % (time.time() - t0, ', '.join('%s %d' % (k, len(v)) for k, v in refined.items())))
        for kd, c in list(best.items()):
            self.dets[kd].extend(refined[kd])
            pool = [d for d in c['m']] + [d for d in refined[kd] if np.linalg.norm(d['pos'] - c['center']) < a.cluster_m]
            seen = set(); mem = []
            for d in pool:
                if d['i'] not in seen:
                    seen.add(d['i']); mem.append(d)
            if kd == 'window' and not any(d['clear'] for d in mem):
                self.W('window: no frame of the chosen cluster confirms clearance_ok (vehicle fits): check it is the window')
            if len(mem) < a.min_frames:
                self.W('%s: only %d consistent detection frames (< --min-frames %d) at %s: not used' % (
                    kd, len(mem), a.min_frames, np.round(self.sf.from_ned(c['center']), 2).tolist()))
                continue
            hi = [d for d in mem if d['conf'] >= a.hi_conf]
            if len(hi) >= a.min_frames and len(hi) < len(mem):
                print('  %s: %d of %d consistent frames have conf >= %.2f: using those' % (kd, len(hi), len(mem), a.hi_conf))
                mem = hi
            self.final[kd] = mem
            # the stop that saw it -> label (unless the user labelled stops)
            cnt = {}
            for d in c['m']:
                if d['stop'] is not None:
                    cnt[d['stop']] = cnt.get(d['stop'], 0) + 1
            if cnt and not any(s.label == KIND_STOP[kd] for s in self.stops):
                k = max(cnt, key=cnt.get)
                if self.stops[k].label is None:
                    self.stops[k].label, self.stops[k].why = KIND_STOP[kd], '%d %s detections' % (cnt[k], kd)
        print('  detector time: %s; errors %d' % (', '.join('%s %d calls %.0f ms/call' % (
            k, self.D.calls[k], self.D.ms[k] / max(1, self.D.calls[k])) for k in KINDS), self.D.errors))

    def seed_positions(self):
        if not self.a.seed_prior:
            return {}
        P = CoursePrior.load(self.a.seed_prior)
        sf = self.sf
        out = {'tunnel': sf.to_ned(P.tunnel_entry), 'window': sf.to_ned(P.window_center)}
        rock = sf.to_ned(P.landing_center); out['landing'] = rock
        return out

    # ------------------------------------------------------------------ geometry assignment
    def assign(self):
        a, S = self.a, self.stops
        free = lambda s: s.label is None
        # wall triple: side / top / side, consecutive (one stray stop allowed)
        if not any(s.label in ('wall_top', 'wall_from', 'wall_to') for s in S):
            best = None
            cand = [s for s in S if free(s)]
            for ii in range(len(cand)):
                for jj in range(ii + 1, min(ii + 4, len(cand))):
                    for kk in range(jj + 1, min(ii + 5, len(cand))):
                        tri = [cand[ii], cand[jj], cand[kk]]
                        T = max(tri, key=lambda s: s.flu[2])
                        A, B = [s for s in tri if s is not T]
                        if T.flu[2] < a.wall_min_up or T.flu[2] < max(A.flu[2], B.flu[2]) + 0.25:
                            continue
                        if min(A.flu[2], B.flu[2]) < 0.35:
                            continue
                        ab = B.flu[:2] - A.flu[:2]; L = np.linalg.norm(ab)
                        if not 0.6 <= L <= 3.5:
                            continue
                        u = (T.flu[:2] - A.flu[:2]) @ ab / L ** 2
                        lat = abs(ab[0] * (T.flu[1] - A.flu[1]) - ab[1] * (T.flu[0] - A.flu[0])) / L
                        if not 0.15 <= u <= 0.85 or lat > 0.6:
                            continue
                        score = abs(u - 0.5) + lat + 0.2 * (kk - ii - 2)
                        if best is None or score < best[0]:
                            best = (score, A, T, B)
            if best is not None:
                _, A, T, B = best
                T.label, T.why = 'wall_top', 'highest of a side/top/side triple'
                rock = self.rock_hint()
                da = np.hypot(*(A.flu[:2] - rock[:2])) if rock is not None else None
                db = np.hypot(*(B.flu[:2] - rock[:2])) if rock is not None else None
                # cross_to = home side (nearer the start pad); cross_from = rock side
                home_first = np.hypot(*A.flu[:2]) < np.hypot(*B.flu[:2])
                to, fr_ = (A, B) if home_first else (B, A)
                if da is not None and ((da < db) != (fr_ is A)):
                    self.W('wall: the side nearer the rock is also nearer the start pad; cross_from = side nearer the rock')
                    fr_, to = (A, B) if da < db else (B, A)
                fr_.label, fr_.why = 'wall_from', 'wall side, rock side'
                to.label, to.why = 'wall_to', 'wall side, home side'
        # rock top: something under the drone 0.08-0.9 m above the floor, near the rock if we saw it
        if not any(s.label == 'rock_top' for s in S):
            rock = self.rock_hint()
            cand = [s for s in S if free(s) and 0.08 <= s.surface_up + a.rest_height <= 0.95 and s.flu[2] < 1.2]
            if rock is not None:
                cand = [s for s in cand if np.hypot(*(s.flu[:2] - rock[:2])) < a.rock_top_radius]
            if cand:
                s = max(cand, key=lambda s: s.dur)
                s.label, s.why = 'rock_top', 'raised surface under the drone (%s)' % s.surface_src
                if len(cand) > 1:
                    self.W('rock_top: %d candidate stops (%s), took the longest #%d' % (
                        len(cand), ', '.join('#%d' % c.idx for c in cand), s.idx))
        # net touches: extreme stops beyond the start pad and every stage object
        start = S[0].flu
        objs = self.object_points()
        for lab, (key, ax, sg) in FENCE_DIRS.items():
            if any(s.label == lab for s in S):
                continue
            cand = [s for s in S if free(s)]
            if not cand:
                continue
            s = max(cand, key=lambda s: sg * s.flu[ax])
            c = sg * s.flu[ax]
            others = [sg * o.flu[ax] for o in S if o is not s and o.label != 'skip']
            obj = [sg * p[ax] for p in objs]
            why = []
            if c < sg * start[ax] + 0.3:
                why.append('not beyond the start pad')
            if obj and c < max(obj) + 0.2:
                why.append('not beyond the stage objects')
            if others and c < max(others) + 0.05:
                why.append('not the outermost stop')
            if abs(wrap_deg(s.yaw_deg - FENCE_YAW[lab])) > a.touch_yaw_tol:
                why.append('nose not toward that net (yaw %.0f deg)' % s.yaw_deg)
            if why:
                self.W('%s: no net-touch stop (best #%d: %s)' % (lab, s.idx, ', '.join(why)))
                continue
            s.label, s.why = lab, 'outermost stop %s' % key
            if not 0.8 <= s.flu[2] <= 2.2:
                self.W('%s #%d at up %.2f m, protocol says chest height: check it is a net touch' % (lab, s.idx, s.flu[2]))
        for s in S:
            if s.label is None:
                s.label, s.why = 'unassigned', ''

    def rock_hint(self):
        if self.final.get('landing'):
            return self.sf.from_ned(np.median([d['pos'] for d in self.final['landing']], 0))
        s = next((s for s in self.stops if s.label == 'rock_view'), None)
        if s is not None:
            return s.flu + self.a.rock_standoff * np.array([math.cos(math.radians(s.yaw_deg)), math.sin(math.radians(s.yaw_deg)), 0.0])
        return None

    def object_points(self):
        pts = []
        for kd, mem in self.final.items():
            if mem:
                pts.append(self.sf.from_ned(np.median([d['pos'] for d in mem], 0)))
        return pts

    # ------------------------------------------------------------------ prior
    def build(self):
        a, sf, S = self.a, self.sf, self.stops
        lab = lambda L: next((s for s in S if s.label == L), None)
        P = {'frame': 'start', 'tolerance_m': a.tol}
        flu = lambda v: dict(forward=_r(v[0]), left=_r(v[1]), up=_r(v[2]))
        deg_of = lambda vec_ned: _r(wrap_deg(math.degrees(math.atan2(vec_ned @ sf.left, vec_ned @ sf.fwd))), 2)
        # ---- tunnel
        T = self.final.get('tunnel')
        if T:
            c, spr = med_spread([d['pos'] for d in T])
            ax = np.mean([np.r_[d['axis'], 0.0] for d in T], 0); ax /= np.linalg.norm(ax)
            axd = [math.degrees(math.atan2(np.r_[d['axis'], 0] @ sf.left, np.r_[d['axis'], 0] @ sf.fwd)) for d in T]
            ad = deg_of(ax)
            ad_spread = float(np.percentile(np.abs([wrap_deg(x - ad) for x in axd]), 90))
            radius = float(np.median([d['radius'] for d in T]))
            # hoop count: the largest one seen in >= 3 frames and >= 20 % of them (a phantom extra hoop in
            # one or two frames must not stretch the tunnel); --n-hoops picks exactly that count
            cnt = {}
            for d in T:
                if d['n_rings'] >= 2 and d['span']:
                    cnt[d['n_rings']] = cnt.get(d['n_rings'], 0) + 1
            ok_n = [k for k, v in cnt.items() if v >= 3 and v >= 0.2 * len(T)]
            nr = a.n_hoops if a.n_hoops in cnt else (max(ok_n) if ok_n else 0)
            spans = [d['span'] for d in T if d['n_rings'] == nr and d['span']]
            length, lsrc = None, None
            if a.tunnel_length:
                length, lsrc = a.tunnel_length, '--tunnel-length'
            elif spans and nr >= 2:
                length = float(np.median(spans)); lsrc = 'detector (%d hoops, first->last, hoop counts seen %s)' % (
                    nr, dict(sorted(cnt.items())))
                if a.n_hoops and nr < a.n_hoops:
                    self.W('tunnel.length_m from %d of %d hoops: too short, pass --tunnel-length' % (nr, a.n_hoops))
                elif not a.n_hoops and len(cnt) > 1:
                    self.W('tunnel.length_m %.2f m from %d hoops (counts seen %s): pass --n-hoops or --tunnel-length if wrong' % (
                        length, nr, dict(sorted(cnt.items()))))
            else:
                self.W('tunnel.length_m: no hoop count seen consistently, pass --tunnel-length')
            P['tunnel'] = dict(entry=flu(sf.from_ned(c)), axis_deg=ad, length_m=_r(length), radius_m=_r(radius), tol_m=a.tol)
            self.note('tunnel.entry', 'detector tunnel', len(T), spr, conf_med=_r(np.median([d['conf'] for d in T]), 2))
            self.note('tunnel.axis_deg', 'detector tunnel', len(T), None, spread_deg=_r(ad_spread, 1))
            self.note('tunnel.length_m', lsrc or 'missing', len(spans) if spans and not a.tunnel_length else None)
            self.note('tunnel.radius_m', 'detector tunnel', len(T), float(np.std([d['radius'] for d in T])))
            if spr > 0.15:
                self.W('tunnel.entry spread %.2f m over %d frames' % (spr, len(T)))
            if len(T) < 3:
                self.W('tunnel.entry from only %d frames' % len(T))
        else:
            P['tunnel'] = None
            self.W('tunnel: no confident detection (stop ~1.8 m in front of the mouth, hoop-centre height)')
        # ---- window
        Wd = self.final.get('window')
        if Wd:
            c, spr = med_spread([d['pos'] for d in Wd])
            nm = [d for d in Wd if d['range'] <= a.window_normal_range] or Wd
            n = np.mean([d['normal'] * np.array([1, 1, 0]) for d in nm], 0); n /= np.linalg.norm(n)
            through = -n
            td = deg_of(through)
            tds = [math.degrees(math.atan2(-d['normal'] @ sf.left, -d['normal'] @ sf.fwd)) for d in nm]
            P['window'] = dict(center=flu(sf.from_ned(c)), through_deg=td, width_m=_r(np.median([d['w'] for d in Wd])),
                               height_m=_r(np.median([d['h'] for d in Wd])), tol_m=a.tol)
            self.note('window.width_m/height_m', 'detector window (opening estimate, informational; not read by the mission)',
                      len(Wd), None, known_size=a.window_size)
            self.note('window.center', 'detector window', len(Wd), spr, conf_med=_r(np.median([d['conf'] for d in Wd]), 2),
                      n_clearance_ok=sum(d['clear'] for d in Wd))
            self.note('window.through_deg', 'detector window (frames <= %.1f m)' % a.window_normal_range, len(nm), None,
                      spread_deg=_r(np.percentile(np.abs([wrap_deg(x - td) for x in tds]), 90), 1))
            if spr > 0.15:
                self.W('window.center spread %.2f m over %d frames' % (spr, len(Wd)))
            if len(Wd) < 3:
                self.W('window.center from only %d frames' % len(Wd))
        else:
            P['window'] = None
            self.W('window: no confident detection (stop ~1.5 m in front, window-centre height, square on)')
        # ---- rock
        Ld = self.final.get('landing'); rt = lab('rock_top')
        lt = {}
        if Ld:
            c, spr = med_spread([d['pos'][:2] for d in Ld])
            cf = sf.from_ned(np.r_[c, 0.0])
            lt['center'] = dict(forward=_r(cf[0]), left=_r(cf[1]))
            top_det = float(np.median([sf.o[2] - d['top_d'] for d in Ld]))
            self.note('landing_target.center', 'detector landing_target', len(Ld), spr,
                      conf_med=_r(np.median([d['conf'] for d in Ld]), 2),
                      extent_m=[_r(np.median([d['extent'][k] for d in Ld]), 2) for k in (0, 1)])
            if spr > 0.15:
                self.W('landing_target.center spread %.2f m over %d frames' % (spr, len(Ld)))
        else:
            top_det = None
            if rt is not None:
                lt['center'] = dict(forward=_r(rt.flu[0]), left=_r(rt.flu[1]))
                self.note('landing_target.center', 'stop #%d rock_top (drone xy, not the detector)' % rt.idx, 1, rt.std)
                self.W('landing_target.center from the rock-top stop only (no detection): accurate to how centred the drone sat')
            else:
                self.W('landing_target: no detection and no rock-top stop')
        if rt is not None:
            lt['top_up'] = _r(rt.surface_up)
            self.note('landing_target.top_up', 'stop #%d rock_top (%s)' % (rt.idx, rt.surface_src), 1, rt.std,
                      detector_top_up=_r(top_det), tof=_r(rt.tof))
            if top_det is not None and abs(top_det - rt.surface_up) > 0.08:
                self.W('rock top: stop says %.2f m, detector %.2f m' % (rt.surface_up, top_det))
        elif top_det is not None:
            lt['top_up'] = _r(top_det)
            self.note('landing_target.top_up', 'detector landing_target', len(Ld), float(np.std([sf.o[2] - d['top_d'] for d in Ld])))
            self.W('landing_target.top_up from the detector only (no rock-top stop)')
        if 'center' in lt and 'top_up' in lt:
            lt['tol_m'] = a.tol
            P['landing_target'] = lt
        else:
            P['landing_target'] = None if not lt else dict(lt, tol_m=a.tol)
        # ---- wall
        wf, wt, wtop = lab('wall_from'), lab('wall_to'), lab('wall_top')
        wl = {}
        if wf is not None and wt is not None:
            wl['cross_from'] = dict(forward=_r(wf.flu[0]), left=_r(wf.flu[1]))
            wl['cross_to'] = dict(forward=_r(wt.flu[0]), left=_r(wt.flu[1]))
            self.note('wall.cross_from', 'stop #%d' % wf.idx, 1, wf.std)
            self.note('wall.cross_to', 'stop #%d' % wt.idx, 1, wt.std)
        else:
            self.W('wall: no side stops (cross_from / cross_to)')
        if wtop is not None:
            # skids resting on the top edge: up - rest_h. A ToF reading of the floor beyond the thin
            # board (surface far below) is ignored; a short reading (held above the edge) corrects.
            top_pose = float(wtop.flu[2] - a.rest_height)
            top = top_pose; why = 'pose (skids on the top edge)'
            if wtop.tof is not None and self.tof_rest is not None and 0.0 <= top_pose - wtop.surface_up <= 0.3:
                top, why = float(wtop.surface_up), 'pose+tof'
            wl['top_up'] = _r(top)
            self.note('wall.top_up', 'stop #%d wall_top (%s)' % (wtop.idx, why), 1, wtop.std, tof=_r(wtop.tof))
        else:
            self.W('wall.top_up: no wall-top stop')
        if wf is not None and wt is not None:
            dep = self.wall_depth_top(wf, wt)
            if dep is not None:
                self.src.setdefault('wall.top_up', {})['depth_check_top_up'] = _r(dep[0])
                self.src['wall.top_up']['depth_check_frames'] = dep[1]
                if 'top_up' not in wl:
                    wl['top_up'] = _r(dep[0]); self.note('wall.top_up', 'depth (highest board points near the crossing)', dep[1], dep[2])
                    self.W('wall.top_up from depth only')
                elif abs(dep[0] - wl['top_up']) > 0.1:
                    self.W('wall top: stop %.2f m vs depth %.2f m' % (wl['top_up'], dep[0]))
        P['wall'] = dict(wl, tol_m=0.3) if len(wl) == 3 else (dict(wl, tol_m=0.3) if wl else None)
        # ---- fence
        fz = {}
        for L_, (key, ax, sg) in FENCE_DIRS.items():
            s = lab(L_)
            if s is None:
                continue
            fz[key] = _r(s.flu[ax] + sg * (a.touch_offset - a.fence_inset))
            self.note('fence.' + key, 'stop #%d net touch (+%.2f m touch offset -%.2f m inset)' % (s.idx, a.touch_offset, a.fence_inset), 1, s.std)
        warned = ' '.join(self.warn)
        for L_, (key, _, _) in FENCE_DIRS.items():
            if key not in fz and L_ + ':' not in warned:
                self.W('fence.%s: no net-touch stop' % key)
        fz['up_max'] = a.up_max
        self.note('fence.up_max', '--up-max (not measured)')
        P['fence'] = fz
        # ---- corridor + cruise height
        P['corridor'], P['cruise_up_m'] = self.corridor(lab('tunnel'))
        self.prior = P
        self.fill_fallback()

    def corridor(self, tun):
        a, tr, sf = self.a, self.tr, self.sf
        t_end, end_xy, why = None, None, 'tunnel stop'
        if tun is not None:
            t_end, end_xy = tun.t0, tun.flu[:2]
        elif self.final.get('tunnel'):                                    # tunnel seen while moving: end the corridor
            e = sf.from_ned(np.median([d['pos'] for d in self.final['tunnel']], 0))   # where the path first nears it
            F = np.array([sf.from_ned(p) for p in tr.p])
            near = np.nonzero((tr.t > self.stops[0].t1) & (np.hypot(*(F[:, :2] - e[:2]).T) < max(2.0, a.tunnel_standoff + 0.2)))[0]
            if len(near):
                t_end, end_xy, why = float(tr.t[near[0]]), F[near[0], :2], 'first pose within 2 m of the detected tunnel entry'
        if t_end is None:
            self.W('corridor: no tunnel stop, corridor left empty')
            self.note('corridor', 'missing'); return [], self.cruise_only()
        m = (tr.t > self.stops[0].t1) & (tr.t < t_end)
        Q = np.array([sf.from_ned(p) for p in tr.p[m]])
        if len(Q) < 10:
            self.note('corridor', 'missing'); return [], self.cruise_only()
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q[:, :2], axis=0), axis=1))]
        if d[-1] < 0.5:
            self.note('corridor', 'path < 0.5 m'); return [], self.cruise_only()
        s = np.arange(0, d[-1], 0.05)
        R = np.stack([np.interp(s, d, Q[:, k]) for k in range(3)], 1)
        n = max(1, int(round(a.corridor_smooth_m / 0.05)))
        Rs = np.stack([smooth(R[:, k], n) for k in range(3)], 1)
        keep = rdp(Rs[:, :2], a.corridor_rdp_m)
        pts = []
        for k in keep[1:-1]:
            p = Rs[k]
            if np.hypot(*p[:2]) < 0.5 or np.hypot(*(p[:2] - end_xy)) < 0.6:
                continue
            if pts and np.hypot(*(p[:2] - pts[-1][:2])) < a.corridor_min_gap_m:
                continue
            pts.append(p)
        lift = Rs[(s > 0.5) & (s < d[-1] - 0.5), 2]
        cruise = a.cruise_up if a.cruise_up else (float(np.clip(round(np.median(lift) / 0.05) * 0.05, 0.5, 1.5)) if len(lift) else 0.8)
        out = [dict(forward=_r(p[0], 2), left=_r(p[1], 2), up=_r(np.clip(round(p[2] / 0.05) * 0.05, 0.4, 1.8), 2)) for p in pts]
        self.corridor_path = Rs
        self.note('corridor', 'path start->%s: %.1f m, smoothed %.2f m, RDP %.2f m' % (
            why, d[-1], a.corridor_smooth_m, a.corridor_rdp_m), len(out))
        self.note('cruise_up_m', '--cruise-up' if a.cruise_up else 'median carried height of the corridor path')
        return out, cruise

    def cruise_only(self):
        if self.a.cruise_up:
            self.note('cruise_up_m', '--cruise-up'); return self.a.cruise_up
        self.W('cruise_up_m: no corridor path to measure it, pass --cruise-up')
        return None

    def wall_depth_top(self, wf, wt):
        """Robust highest depth point near the crossing segment, from frames between the two side stops."""
        a, tr, sf = self.a, self.tr, self.sf
        A, B = wf.flu[:2], wt.flu[:2]
        ab = B - A; L = np.linalg.norm(ab)
        if L < 0.3:
            return None
        u = ab / L
        t_lo, t_hi = min(wf.t0, wt.t0), max(wf.t1, wt.t1)                # both side stops and the wall top between
        idx = list(np.nonzero((tr.dmeta[:, 0] >= t_lo) & (tr.dmeta[:, 0] <= t_hi))[0])
        idx = idx[::max(1, len(idx) // 40)]
        tops = []
        clipped = 0
        for i, fr in iter_frames(a.bag, self.reader, tr, idx):
            P, VU = fr.points_ned(stride=4, max_range=3.0)
            if len(P) < 50:
                continue
            Q = (P - sf.o) @ np.stack([sf.fwd, sf.left, [0, 0, -1.0]], 1)
            rel = Q[:, :2] - A
            s_ = rel @ u; lat = np.abs(rel @ np.array([-u[1], u[0]]))
            k = (s_ > 0.2 * L) & (s_ < 0.8 * L) & (lat < 0.5) & (Q[:, 2] > 0.3)
            if k.sum() >= 20:
                top = float(np.percentile(Q[k, 2], 98))
                if np.any(VU[k][Q[k, 2] >= top - 0.03, 0] < 12):           # board top cut by the image edge
                    clipped += 1; continue
                tops.append(top)
        if len(tops) < 2:
            if clipped:
                print('  wall depth check: top edge out of view in %d frames (camera too low / too close)' % clipped)
            return None
        return float(np.median(tops)), len(tops), float(np.std(tops))

    def fill_fallback(self):
        if not self.a.fallback:
            return
        fb = json.loads(Path(self.a.fallback).read_text())
        P = self.prior
        for sec in ('tunnel', 'window', 'landing_target', 'wall'):
            if fb.get(sec) is None:
                continue
            if P.get(sec) is None:
                P[sec] = {k: v for k, v in fb[sec].items() if not k.startswith('_')}
                self.note(sec, 'FALLBACK %s' % self.a.fallback)
                self.W('%s taken from --fallback' % sec)
            else:
                for k, v in fb[sec].items():
                    if not k.startswith('_') and (P[sec].get(k) is None):
                        P[sec][k] = v; self.note('%s.%s' % (sec, k), 'FALLBACK %s' % self.a.fallback)
                        self.W('%s.%s taken from --fallback' % (sec, k))
        if P.get('cruise_up_m') is None and fb.get('cruise_up_m') is not None:
            P['cruise_up_m'] = fb['cruise_up_m']; self.note('cruise_up_m', 'FALLBACK %s' % self.a.fallback)
            self.W('cruise_up_m taken from --fallback')
        for k in ('forward_min', 'forward_max', 'left_min', 'left_max'):
            if P['fence'].get(k) is None and (fb.get('fence') or {}).get(k) is not None:
                P['fence'][k] = fb['fence'][k]; self.note('fence.' + k, 'FALLBACK %s' % self.a.fallback)
                self.W('fence.%s taken from --fallback' % k)

    # ------------------------------------------------------------------ output
    def comment_lines(self):
        out = ['Generated by tools/survey_to_prior.py from %s (%s reader) on %s.' % (
            Path(self.a.bag).name, self.reader, time.strftime('%Y-%m-%d %H:%M')),
            'START frame = first still stop (#0) of the walk: put the drone on the same pad marks, same heading, for the flight.',
            'Per field: source, n = supporting frames (detector) or stops, spread_m = 90th percentile distance from the median.']
        for k, v in self.src.items():
            out.append('%s: %s' % (k, ', '.join('%s=%s' % (kk, vv) for kk, vv in v.items())))
        if self.closure:
            out.append('closure (end vs start pad): %s' % self.closure)
        for w in self.warn:
            out.append('WARNING: ' + w)
        return out

    def write(self):
        a = self.a
        out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
        js = {'_comment': self.comment_lines()}
        order = ('frame', 'tolerance_m', 'cruise_up_m', 'fence', 'corridor', 'tunnel', 'window', 'landing_target', 'wall')
        for k in order:
            js[k] = self.prior.get(k)
        self.prior_path = out / ('course_prior.%s.json' % a.name)
        self.prior_path.write_text(json.dumps(js, indent=1) + '\n')
        try:
            CoursePrior.from_dict(js); self.loads = True
            print('wrote %s  (CoursePrior.load OK)' % self.prior_path)
        except Exception as e:  # noqa: BLE001
            self.loads = False
            print('wrote %s  -- NOT loadable by the mission yet: %s' % (self.prior_path, e))
        rep = dict(bag=str(a.bag), reader=self.reader, args={k: v for k, v in vars(a).items()},
                   clusters_pass1=getattr(self, 'clusters', {}),
                   start=self.src.get('START'), closure=self.closure, warnings=self.warn, provenance=self.src,
                   stops=[s.summary(self.t_ref) for s in self.stops], loads=self.loads,
                   detections={k: [dict(flu=[_r(c) for c in self.sf.from_ned(d['pos'])], conf=_r(d['conf'], 2),
                                        t=_r(d['t'] - self.t_ref, 2), stop=d['stop'], pass_=d['pass_'])
                                   for d in v] for k, v in self.dets.items()})
        if a.truth:
            rep['truth'] = self.compare(json.loads(Path(a.truth).read_text()))
        self.report_path = out / ('course_prior.%s.report.json' % a.name)
        self.report_path.write_text(json.dumps(rep, indent=1, default=str) + '\n')
        self.png_path = out / ('course_prior.%s.png' % a.name)
        draw(self, self.png_path, json.loads(Path(a.truth).read_text()) if a.truth else None)
        print('wrote %s\nwrote %s' % (self.report_path, self.png_path))

    def compare(self, tru):
        P = self.prior; E = {}
        g = lambda d, *ks: (lambda x: x)(d)

        def v3(d):
            return np.array([d['forward'], d['left'], d.get('up', 0.0)], float)
        if P.get('tunnel') and tru.get('tunnel'):
            e = v3(P['tunnel']['entry']) - v3(tru['tunnel']['entry'])
            E['tunnel.entry'] = dict(err_m=_r(np.linalg.norm(e)), horiz_m=_r(np.hypot(*e[:2])), up_m=_r(e[2]), vec=[_r(c) for c in e])
            E['tunnel.axis_deg'] = dict(err_deg=_r(wrap_deg(P['tunnel']['axis_deg'] - tru['tunnel']['axis_deg']), 2))
            if P['tunnel'].get('length_m'):
                E['tunnel.length_m'] = dict(err_m=_r(P['tunnel']['length_m'] - tru['tunnel']['length_m']))
        if P.get('window') and tru.get('window'):
            e = v3(P['window']['center']) - v3(tru['window']['center'])
            E['window.center'] = dict(err_m=_r(np.linalg.norm(e)), horiz_m=_r(np.hypot(*e[:2])), up_m=_r(e[2]), vec=[_r(c) for c in e])
            E['window.through_deg'] = dict(err_deg=_r(wrap_deg(P['window']['through_deg'] - tru['window']['through_deg']), 2))
        lt, tl = P.get('landing_target') or {}, tru.get('landing_target') or {}
        if lt.get('center') and tl.get('center'):
            e = v3(lt['center'])[:2] - v3(tl['center'])[:2]
            E['landing_target.center'] = dict(err_m=_r(np.linalg.norm(e)), vec=[_r(c) for c in e])
        if lt.get('top_up') is not None and tl.get('top_up') is not None:
            E['landing_target.top_up'] = dict(err_m=_r(lt['top_up'] - tl['top_up']))
        for k in ('forward_min', 'forward_max', 'left_min', 'left_max'):
            if P['fence'].get(k) is not None:
                E['fence.' + k] = dict(err_m=_r(P['fence'][k] - tru['fence'][k]))
        wl, tw = P.get('wall') or {}, tru.get('wall') or {}
        if wl.get('top_up') is not None:
            E['wall.top_up'] = dict(err_m=_r(wl['top_up'] - tw['top_up']))
        if wl.get('cross_from') and wl.get('cross_to'):
            a0, a1 = v3(wl['cross_from'])[:2], v3(wl['cross_to'])[:2]
            b0, b1 = v3(tw['cross_from'])[:2], v3(tw['cross_to'])[:2]
            tm, tdir = (b0 + b1) / 2, (b1 - b0) / np.linalg.norm(b1 - b0)
            side = lambda p: float((p - tm) @ tdir)                       # signed distance from the truth wall plane
            E['wall.cross'] = dict(from_side_m=_r(side(a0)), to_side_m=_r(side(a1)), crosses=bool(side(a0) < 0 < side(a1)),
                                   mid_err_m=_r(np.linalg.norm((a0 + a1) / 2 - tm)),
                                   dir_err_deg=_r(math.degrees(math.acos(np.clip(((a1 - a0) / np.linalg.norm(a1 - a0)) @ tdir, -1, 1))), 1))
        tgt = {'tunnel.entry': 0.2, 'window.center': 0.2, 'landing_target.center': 0.2, 'landing_target.top_up': 0.2,
               'fence.forward_min': 0.3, 'fence.forward_max': 0.3, 'fence.left_min': 0.3, 'fence.left_max': 0.3}
        for k, lim in tgt.items():
            if k in E:
                E[k]['target_m'] = lim; E[k]['pass'] = bool(abs(E[k]['err_m']) <= lim)
            else:
                E[k] = dict(target_m=lim, missing=True)
        print('\nper-field error vs truth (%s):' % Path(self.a.truth).name)
        for k, v in E.items():
            print('  %-24s %s' % (k, ', '.join('%s=%s' % kv for kv in v.items())))
        return E


# =========================================================================== PNG
def draw(sv, path, truth=None):
    import cv2
    P = sv.prior; sf = sv.sf
    path_flu = np.array([sf.from_ned(p) for p in sv.tr.p[::5]])
    path_flu = path_flu[np.all(np.abs(path_flu[:, :2]) < 15.0, axis=1)]          # never let a wild pose size the canvas
    pts = [path_flu[:, :2]]
    fz = P['fence']
    box = [[fz.get('forward_min'), fz.get('left_min')], [fz.get('forward_max'), fz.get('left_max')]]
    for b in box:
        if None not in b:
            pts.append(np.array([b]))
    if truth:
        tf_ = truth['fence']
        pts.append(np.array([[tf_['forward_min'], tf_['left_min']], [tf_['forward_max'], tf_['left_max']]]))
    allp = np.concatenate(pts)
    allp = allp[np.all(np.abs(allp) < 15.0, axis=1)] if np.any(np.all(np.abs(allp) < 15.0, axis=1)) else np.zeros((1, 2))
    fmin, lmin = allp.min(0) - 0.6
    fmax, lmax = allp.max(0) + 0.6
    S = 110.0
    Wd, Hd = int((lmax - lmin) * S) + 1, int((fmax - fmin) * S) + 1
    img = np.full((Hd + 150, max(Wd, 700), 3), 255, np.uint8)
    px = lambda f, l: (int(round((lmax - l) * S)), int(round((fmax - f) * S)))   # forward up, left left
    for g in np.arange(math.floor(fmin), math.ceil(fmax) + 1):
        cv2.line(img, px(g, lmin), px(g, lmax), (235, 235, 235), 1)
    for g in np.arange(math.floor(lmin), math.ceil(lmax) + 1):
        cv2.line(img, px(fmin, g), px(fmax, g), (235, 235, 235), 1)

    def rect(f0, f1, l0, l1, col, th):
        cv2.rectangle(img, px(f1, l1), px(f0, l0), col, th)
    if truth:
        tf_ = truth['fence']; rect(tf_['forward_min'], tf_['forward_max'], tf_['left_min'], tf_['left_max'], (170, 170, 170), 1)
    known = [fz.get(k) for k in ('forward_min', 'forward_max', 'left_min', 'left_max')]
    if None not in known:
        rect(known[0], known[1], known[2], known[3], (0, 0, 200), 2)
    else:
        for k, v in zip(('forward_min', 'forward_max', 'left_min', 'left_max'), known):
            if v is None:
                continue
            if k.startswith('forward'):
                cv2.line(img, px(v, lmin + 0.3), px(v, lmax - 0.3), (0, 0, 200), 2)
            else:
                cv2.line(img, px(fmin + 0.3, v), px(fmax - 0.3, v), (0, 0, 200), 2)
    n = len(path_flu)
    for k in range(n - 1):
        c = int(200 * k / max(1, n - 1))
        cv2.line(img, px(*path_flu[k, :2]), px(*path_flu[k + 1, :2]), (200 - c // 2, 160, 40 + c), 1, cv2.LINE_AA)
    colors = {'tunnel': (0, 140, 255), 'window': (200, 60, 160), 'landing': (60, 170, 60)}
    for kd, ds in sv.dets.items():
        for d in ds:
            f, l, _ = sf.from_ned(d['pos'])
            cv2.circle(img, px(f, l), 2, colors[kd], -1)

    def arrow(f, l, deg, L, col, th=2):
        a = math.radians(deg)
        cv2.arrowedLine(img, px(f, l), px(f + L * math.cos(a), l + L * math.sin(a)), col, th, cv2.LINE_AA, tipLength=0.2)

    def mark(sec, tag, col, tr_=False):
        src = truth if tr_ else P
        s = (src or {}).get(sec)
        if not s:
            return
        th = 1 if tr_ else 2
        cc = (150, 150, 150) if tr_ else col
        if sec == 'tunnel' and s.get('entry'):
            e = s['entry']; cv2.circle(img, px(e['forward'], e['left']), 6, cc, th)
            arrow(e['forward'], e['left'], s['axis_deg'], s.get('length_m') or 1.0, cc, th)
        if sec == 'window' and s.get('center'):
            c = s['center']; a = math.radians(s['through_deg'] + 90); w = (s.get('width_m') or 0.75) / 2
            cv2.line(img, px(c['forward'] - w * math.cos(a), c['left'] - w * math.sin(a)),
                     px(c['forward'] + w * math.cos(a), c['left'] + w * math.sin(a)), cc, th + 1)
            arrow(c['forward'], c['left'], s['through_deg'], 0.6, cc, th)
        if sec == 'landing_target' and s.get('center'):
            c = s['center']; cv2.rectangle(img, px(c['forward'] + 0.25, c['left'] + 0.22), px(c['forward'] - 0.25, c['left'] - 0.22), cc, th)
        if sec == 'wall' and s.get('cross_from') and s.get('cross_to'):
            a_, b_ = s['cross_from'], s['cross_to']
            cv2.arrowedLine(img, px(a_['forward'], a_['left']), px(b_['forward'], b_['left']), cc, th, cv2.LINE_AA, tipLength=0.1)
        if not tr_ and tag:
            anchor = s.get('entry') or s.get('center') or s.get('cross_from')
            if anchor:
                cv2.putText(img, tag, px(anchor['forward'] + 0.12, anchor['left'] - 0.1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    for sec, tag, col in (('tunnel', 'tunnel', colors['tunnel']), ('window', 'window', colors['window']),
                          ('landing_target', 'rock', colors['landing']), ('wall', 'wall', (90, 60, 20))):
        if truth:
            mark(sec, None, col, True)
        mark(sec, tag, col)
    cor = [(0.0, 0.0)] + [(c['forward'], c['left']) for c in P.get('corridor') or []]
    for k, c in enumerate(cor):
        cv2.circle(img, px(*c), 4, (40, 40, 40), -1)
        if k:
            cv2.line(img, px(*cor[k - 1]), px(*c), (40, 40, 40), 1, cv2.LINE_AA)
    if truth:
        for c in truth.get('corridor') or []:
            cv2.drawMarker(img, px(c['forward'], c['left']), (150, 150, 150), cv2.MARKER_TILTED_CROSS, 8, 1)
    for s in sv.stops:
        col = (0, 0, 0) if s.label in ('unassigned', None) else (0, 90, 200)
        cv2.circle(img, px(*s.flu[:2]), 5, col, 1)
        cv2.putText(img, '%d %s' % (s.idx, s.label if s.label != 'unassigned' else '?'), px(s.flu[0] - 0.12, s.flu[1] - 0.08),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    cv2.drawMarker(img, px(0, 0), (0, 0, 0), cv2.MARKER_CROSS, 14, 2)
    arrow(0, 0, 0, 0.5, (0, 0, 0), 1)
    y = Hd + 18
    bp = Path(str(sv.a.bag)).resolve()
    lines = ['%s  START frame: forward up, left = left. grid 1 m.' % (bp.parent.name + '/' + bp.name if bp.name == 'bag' or bp.suffix == '.mcap' else bp.name),
             'path (blue->red by time), stops (index label), detections: orange tunnel, purple window, green rock',
             'prior: red fence, circles/arrows, black corridor' + ('; grey = truth' if truth else ''),
             '%d warnings (see report.json)' % len(sv.warn)]
    for t in lines:
        cv2.putText(img, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA); y += 18
    cv2.imwrite(str(path), img)


# =========================================================================== CLI
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('bag', help='rosbag2 directory or .mcap file of the walk-through')
    p.add_argument('--name', default='survey', help='output = course_prior.<name>.json')
    p.add_argument('--out-dir', default='.', help='output directory (default: current directory)')
    p.add_argument('--stops', help='stop labels: JSON list in stop order, JSON {index: label}, or "index label" lines; '
                                   'labels: ' + ' '.join(LABELS))
    p.add_argument('--list-stops', action='store_true', help='print the still stops and exit (to write --stops)')
    p.add_argument('--truth', help='truth prior JSON -> per-field errors in the report (validation only)')
    p.add_argument('--seed-prior', help='rough prior JSON: extra detector priors (every frame that sees them)')
    p.add_argument('--fallback', help='prior JSON to fill fields the survey could not measure (marked FALLBACK)')
    p.add_argument('--reader', default='auto', choices=('auto', 'rosbag2', 'mcap', 'builtin'))
    p.add_argument('--no-detect', action='store_true', help='skip the detectors (stops + path only)')
    g = p.add_argument_group('stops')
    g.add_argument('--still-speed', type=float, default=0.05, help='m/s (default 0.05)')
    g.add_argument('--still-min-s', type=float, default=2.0)
    g.add_argument('--still-yaw-rate', type=float, default=12.0, help='deg/s')
    g.add_argument('--closure-warn-m', type=float, default=0.15)
    g.add_argument('--closure-correct', action='store_true',
                   help='subtract the end-pad position error linearly in time (needs the end stop on the pad marks)')
    g = p.add_argument_group('vehicle / site')
    g.add_argument('--rest-height', type=float, default=0.05, help='base_link above the surface it rests on (node uses 0.05)')
    g.add_argument('--touch-offset', type=float, default=SKY_BETA_HALF_SPAN_M, help='body centre -> net when touching (m)')
    g.add_argument('--fence-inset', type=float, default=0.0, help='pull every touched fence side in by this (m)')
    g.add_argument('--touch-yaw-tol', type=float, default=50.0, help='a net touch faces its net within this (deg)')
    g.add_argument('--up-max', type=float, default=2.8, help='fence.up_max (not measured by the walk)')
    g.add_argument('--half-span', type=float, default=SKY_BETA_HALF_SPAN_M)
    g.add_argument('--half-height', type=float, default=0.10)
    g.add_argument('--window-size', type=float, nargs=2, metavar=('W', 'H'), help='measured window opening (m)')
    g.add_argument('--tunnel-radius', type=float, help='measured hoop radius (m)')
    g.add_argument('--hoop-spacing', type=float, help='measured hoop spacing (m)')
    g.add_argument('--n-hoops', type=int, help='number of hoops (warns if fewer were seen)')
    g.add_argument('--tunnel-length', type=float, help='first->last hoop (m); overrides the detector span')
    g.add_argument('--cruise-up', type=float, help='cruise_up_m (default: median carried corridor height)')
    g.add_argument('--tol', type=float, default=0.5, help='tol_m written for tunnel/window/rock')
    g = p.add_argument_group('detection')
    g.add_argument('--tunnel-standoff', type=float, default=1.8)
    g.add_argument('--window-standoff', type=float, default=1.5)
    g.add_argument('--rock-standoff', type=float, default=1.6)
    g.add_argument('--tunnel-min-conf', type=float, default=0.4)
    g.add_argument('--window-min-conf', type=float, default=0.5)
    g.add_argument('--landing-min-conf', type=float, default=0.4)
    g.add_argument('--frames-per-stop', type=int, default=8)
    g.add_argument('--scan-stride', type=int, default=6, help='pass 1 also tries every Nth moving frame (0 = stops only)')
    g.add_argument('--scan-weight', type=float, default=0.5, help='cluster weight of a moving-frame detection')
    g.add_argument('--scan-max-speed', type=float, default=1.2, help='m/s; faster frames are not scanned')
    g.add_argument('--refine-stride', type=int, default=2, help='every Nth depth frame is a pass-2 candidate')
    g.add_argument('--max-refine-frames', type=int, default=60, help='per kind')
    g.add_argument('--cluster-m', type=float, default=0.35)
    g.add_argument('--min-frames', type=int, default=5, help='a detector field needs this many consistent frames')
    g.add_argument('--hi-conf', type=float, default=0.6, help='use only frames >= this confidence when >= --min-frames have it')
    g.add_argument('--rock-min-short', type=float, default=0.2, help='rock top: shorter side >= this (m)')
    g.add_argument('--rock-max-aspect', type=float, default=3.0, help='rock top: long/short side <= this')
    g.add_argument('--tunnel-range', type=float, nargs=2, default=(1.0, 3.0))
    g.add_argument('--window-range', type=float, nargs=2, default=(0.9, 3.0))
    g.add_argument('--rock-range', type=float, nargs=2, default=(0.8, 3.5))
    g.add_argument('--window-normal-range', type=float, default=2.5, help='frames this close give through_deg')
    g.add_argument('--rock-top-radius', type=float, default=0.5)
    g.add_argument('--wall-min-up', type=float, default=0.9)
    g = p.add_argument_group('corridor')
    g.add_argument('--corridor-smooth-m', type=float, default=0.4)
    g.add_argument('--corridor-rdp-m', type=float, default=0.15)
    g.add_argument('--corridor-min-gap-m', type=float, default=0.7)
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    sv = Survey(a)
    sv.load()
    if a.list_stops:
        return sv
    sv.detect()
    sv.assign()
    sv.print_stops()
    sv.build()
    sv.write()
    if sv.warn:
        print('\n%d warning(s):' % len(sv.warn))
        for w in sv.warn:
            print('  - ' + w)
    return sv


if __name__ == '__main__':
    main()
