"""Tunnel servo: the vehicle's pose RELATIVE TO THE TUNNEL, every frame, without VIO position.

Why: the traverse used to latch the entry hoop once at the standoff and then fly a straight VIO line through
the tunnel. Inside the tube VIO is at its weakest (dark, low texture, walls close). This module measures,
every frame, where the vehicle is with respect to the tunnel itself, so the controller can drive the
errors to zero directly.

Inputs per frame: the left image, depth (optional), and the camera ATTITUDE (Frame.R_nc). Frame.cam_ned
(the VIO position) is used once, when the track starts, to turn the mission's latch into a first
guess. After that, only the attitude is used: gravity gives roll and pitch, and the filter carries the
axis heading in the attitude's yaw frame.

Measurements, fused in one iterated EKF over [s, l, u, psi, vs, vl, vu]:
  (1) hoops: a model-based edge tracker (RAPiD style). Each hoop is a vertical circle of known radius
      at a known position along a horizontal axis. Every model point searches along its image normal
      for the nearest edge whose orientation matches. Partial arcs and the exit opening's silhouette
      (bright outside, dark mesh) count the same as a full hoop.
      With a single hoop in view, lateral offset and yaw trade off in the image. The filter separates
      them by carrying the axis heading psi between multi-hoop frames, where the hoop-to-hoop parallax
      measures it.
  (2) depth: points on the tube (mesh, fabric, hoop wire) lie at distance R from the axis, so each one
      gives the residual rho - R. This constrains l, u and psi, not s.
  (3) optional ToF: the body height above the floor constrains u, given the axis height above the
      floor (from the latch).
  Without vision, the filter predicts with constant velocity. An optional `odom_ned` (VIO body
  position) makes it dead-reckon on short VIO deltas instead. `mode` then says 'dead_reckoning'; it
  never says 'vision'.

Frames and signs (all vehicle = base_link, not the camera):
  s      metres along the axis from the ENTRY hoop plane (+ toward the exit)
  l      metres LEFT of the centre line, looking toward the exit (+ = vehicle is left of centre)
  u      metres ABOVE the centre line (+ = vehicle is high)
  yaw_err = wrap(vehicle heading - axis heading), NED yaw (+ = nose turned RIGHT, i.e. clockwise from above)
To centre: move right by l, down by u, yaw by -yaw_err. ServoState.correction_ned already gives the NED vector
from the vehicle to its point on the centre line, and the NED axis direction in the caller's attitude frame.

Pure numpy + OpenCV, Python 3.9+.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Optional, Sequence, Tuple

import numpy as np
import cv2

# official base_link(FLU) <- oak_left(optical) extrinsic (common/geometry.py); overridable in ServoParams
_R_BASE_CAM = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
_T_BASE_CAM = (0.116, 0.037, 0.029)
_F = np.diag([1., -1., -1.])          # FLU <-> FRD


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# Small linear algebra without BLAS: multithreaded BLAS stalls for milliseconds on tiny matrices when the CPU is busy
# (measured 5-10 ms per 7x7 inverse at load ~9 on the M4; the flight computer is always busy).
def _dot3(X, v):
    return X[..., 0] * v[0] + X[..., 1] * v[1] + X[..., 2] * v[2]


def _norm3(X):
    return np.sqrt(X[..., 0] ** 2 + X[..., 1] ** 2 + X[..., 2] ** 2)


def _rotT(X, M):
    """X @ M.T for (..., 3) arrays."""
    return np.stack([_dot3(X, M[0]), _dot3(X, M[1]), _dot3(X, M[2])], -1)


def _mm(A, B):
    return np.einsum('ij,jk->ik', A, B)


def _gram(H, w):
    """H^T diag(w) H"""
    return np.einsum('ni,nj->ij', H * w[:, None], H)


def _solve(A, B):
    """Gauss-Jordan with partial pivoting on a small dense system (n <= ~10). B: (n,) or (n, k)."""
    A = [list(map(float, row)) for row in np.asarray(A, float)]
    Bm = np.asarray(B, float)
    vec = Bm.ndim == 1
    Bl = [[float(v)] for v in Bm] if vec else [list(map(float, row)) for row in Bm]
    n = len(A)
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(A[r][c]))
        if abs(A[piv][c]) < 1e-15:
            raise np.linalg.LinAlgError('singular')
        A[c], A[piv] = A[piv], A[c]
        Bl[c], Bl[piv] = Bl[piv], Bl[c]
        inv = 1.0 / A[c][c]
        A[c] = [v * inv for v in A[c]]
        Bl[c] = [v * inv for v in Bl[c]]
        for r in range(n):
            if r != c and A[r][c] != 0.0:
                f = A[r][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
                Bl[r] = [a - f * b for a, b in zip(Bl[r], Bl[c])]
    out = np.array(Bl)
    return out[:, 0] if vec else out


def _inv(A):
    return _solve(A, np.eye(len(A)))


# ------------------------------------------------------------------------------------------------ model / params
@dataclass
class TunnelModel:
    """Tunnel geometry. `entry_ned` and `axis_ned` are in the VIO frame at the latch; they are used only to start
    the track. Hoop positions are metres along the axis from the entry hoop (first entry must be 0)."""
    entry_ned: np.ndarray                    # (3,) centre of the entry hoop
    axis_ned: np.ndarray                     # (2,) horizontal unit vector, entry -> exit
    radius_m: float = 0.80
    hoop_s: Tuple[float, ...] = (0.0, 0.64, 1.28)
    floor_z_ned: Optional[float] = None      # floor under the tunnel, same frame as entry_ned (enables ToF residual)

    def __post_init__(self):
        self.entry_ned = np.asarray(self.entry_ned, float).reshape(3)
        a = np.asarray(self.axis_ned, float).reshape(-1)[:2]
        self.axis_ned = a / max(float(np.linalg.norm(a)), 1e-9)
        self.hoop_s = tuple(sorted(float(x) for x in self.hoop_s))

    @property
    def exit_s(self) -> float:
        return float(self.hoop_s[-1])

    @property
    def axis_height_m(self) -> Optional[float]:
        """Axis height above the floor (metres) when the floor is known."""
        return None if self.floor_z_ned is None else float(self.floor_z_ned - self.entry_ned[2])

    @classmethod
    def from_estimate(cls, est, n_hoops: Optional[int] = None, spacing_m: Optional[float] = None,
                      length_m: Optional[float] = None, floor_z_ned: Optional[float] = None,
                      hoop_s: Optional[Sequence[float]] = None, radius_m: Optional[float] = None) -> 'TunnelModel':
        """Build from a perception.tunnel TunnelEstimate (or the mission's latched copy: entry_ned, axis_ned,
        radius_m, optional rings).

        With `hoop_s` (measured on site, entry = 0): the model is anchored on the FARTHEST detected ring = the exit
        hoop. The detector reliably sees the exit (innermost ring, bright opening inside) but can name the 2nd hoop
        the entry (it did on the 2026-11-08 real recordings), so the entry is placed at exit - hoop_s[-1].
        Without `hoop_s`: hoops come from the detected rings along the axis. The spacing is the median detected
        spacing unless `spacing_m` is given. The list is extended to `n_hoops`, or to `length_m`."""
        if hoop_s is not None:
            e = np.asarray(est.entry_ned, float).reshape(3)
            a = np.asarray(est.axis_ned, float).reshape(-1)[:2]
            a = a / max(float(np.linalg.norm(a)), 1e-9)
            a3 = np.array([a[0], a[1], 0.0])
            far = e.copy()
            for r in getattr(est, 'rings', None) or []:
                c = getattr(r, 'center_ned', None)
                if c is not None and float((np.asarray(c, float) - far) @ a3) > 0:
                    far = np.asarray(c, float).reshape(3).copy()
            hs = tuple(float(x) for x in hoop_s)
            entry = far - (hs[-1] - hs[0]) * a3
            entry[2] = e[2]
            return cls(entry_ned=entry, axis_ned=a, radius_m=float(radius_m if radius_m is not None else est.radius_m),
                       hoop_s=tuple(x - hs[0] for x in hs), floor_z_ned=floor_z_ned)
        e = np.asarray(est.entry_ned, float).reshape(3)
        a = np.asarray(est.axis_ned, float).reshape(-1)[:2]
        a = a / max(float(np.linalg.norm(a)), 1e-9)
        a3 = np.array([a[0], a[1], 0.0])
        ss = [0.0]
        for r in getattr(est, 'rings', None) or []:
            c = getattr(r, 'center_ned', None)
            if c is None:
                continue
            t = float((np.asarray(c, float) - e) @ a3)
            if t > 0.15 and all(abs(t - x) > 0.15 for x in ss):
                ss.append(t)
        ss.sort()
        sp = spacing_m
        if sp is None:
            d = np.diff(ss)
            sp = float(np.median(d)) if len(d) else None
        if sp is not None:
            if n_hoops is not None:
                while len(ss) < n_hoops:
                    ss.append(ss[-1] + sp)
                ss = ss[:n_hoops]
            elif length_m is not None:
                while ss[-1] + sp <= length_m + 0.5 * sp:
                    ss.append(ss[-1] + sp)
        return cls(entry_ned=e, axis_ned=a, radius_m=float(est.radius_m), hoop_s=tuple(ss), floor_z_ned=floor_z_ned)


@dataclass
class ServoParams:
    # extrinsic: base_link (FLU) <- left camera (optical)
    R_base_cam: np.ndarray = field(default_factory=lambda: _R_BASE_CAM.copy())
    t_base_cam: Tuple[float, float, float] = _T_BASE_CAM
    # edges
    work_width: int = 640                 # larger images are resized (K scaled)
    blur_sigma: float = 1.6               # suppresses the mesh weave and the cage net, keeps hoops and the opening
    canny_hi_pct: float = 90.0            # high threshold = this percentile of the gradient magnitude ...
    canny_hi_min: float = 60.0            # ... but at least this (Sobel 3x3 units on uint8)
    canny_lo_frac: float = 0.45
    orient_tol_deg: float = 35.0          # model normal vs image gradient (unsigned)
    line_max_px: float = 7.0              # max width of a wire (pair of opposite-polarity edges)
    # model sampling / matching
    pts_per_hoop: int = 120
    skip_top_deg: float = 25.0            # model points within this of the hoop top are not used (real hoops peak there,
                                          # and the spine bar / fabric flap sit on top)
    min_view_z_m: float = 0.25            # model points closer than this (optical z) are skipped
    border_px: int = 3
    gate_px: Tuple[float, ...] = (24.0, 14.0, 8.0, 5.0, 4.0)   # 1-D search half-length per iteration
    gate_lost_px: Tuple[float, ...] = (40.0, 24.0, 14.0, 8.0, 5.0, 4.0)
    sigma_px: float = 1.2
    hoop_n_eff: float = 24.0              # information of a hoop fit counted as this many independent points: real
                                          # hoops are not perfect coaxial circles (per-frame bias ~3-8 cm at 2 m)
    huber_px: float = 2.0
    min_match_pts: int = 25               # matched hoop points needed for a hoop measurement
    min_cover: float = 0.25               # matched / visible model points
    min_arc_bins: int = 5                 # matches must fall in at least this many 30-degree hoop sectors
    max_rms_px: float = 2.8
    # depth
    use_depth: bool = True
    depth_stride: int = 4
    depth_n_eff: float = 12.0
    depth_max_pts: int = 700
    depth_gate_m: float = 0.20
    depth_sigma0_m: float = 0.02
    depth_sigma_z2: float = 0.015         # + this * z^2 (SGBM-like)
    depth_min_pts: int = 150
    depth_min_arc_deg: float = 90.0       # inliers must cover this much of the cross-section (30 deg bins)
    depth_max_rms_m: float = 0.06
    conflict_min_pts: int = 300           # this much tube depth failing at the hoop fix rejects the hoop fix
    conflict_min_arc_deg: float = 180.0
    depth_alone_max_s: float = 0.6        # depth without hoops only this soon after the last hoop/vision fix
    depth_floor_frac: float = 0.6         # skip points lower than axis - this*R (the floor under the tube)
    depth_max_range_m: float = 4.0
    # ToF
    tof_sigma_m: float = 0.03
    # filter
    acc_sigma: float = 0.8                # m/s^2 white acceleration (vehicle and handheld)
    psi_rw_deg_s: float = 0.5             # axis-heading random walk (absorbs attitude yaw drift)
    init_sigma: Tuple[float, float, float, float] = (0.20, 0.10, 0.10, math.radians(2.0))
    init_vel_sigma: float = 0.3
    max_dt_s: float = 2.0
    odom_sigma_frac: float = 0.05         # dead reckoning on VIO deltas: sigma = frac * step + 0.005 m
    odom_max_speed: float = 1.5           # a VIO delta faster than this is a jump: ignored (constant velocity instead)
    tight_sigma_s_m: float = 0.25         # prior this tight (dead reckoning) is not "lost": no grid search, chi2 gate on
    tight_sigma_lu_m: float = 0.10
    weak_min_pts: int = 10                # corner arcs while coasting: refine dead reckoning only
    weak_min_cover: float = 0.75
    weak_max_rms_px: float = 1.5
    coast_from_exit_m: float = 1.25       # coasting allowed only if the last valid fix was this close to the exit plane
    coast_max_m: float = 1.9              # ... and for at most this much dead-reckoned travel (exit + tunnel_exit_m)
    max_innovation_chi2: float = 40.0     # pose-part Mahalanobis of the update vs the prediction
    n_iter: int = 5
    # validity
    valid_sigma_lu_m: float = 0.06
    valid_sigma_yaw_deg: float = 4.0
    search_on: bool = False              # grid search when lost: OFF (on real clutter it jumped a hoop; see report)
    search_gain: float = 1.25            # a grid-search pose replaces the prediction only if it scores this much better
    reacquire_after_s: float = 0.5       # search a grid around the prediction after this long without a fix
    max_advance_gap_s: float = 0.3       # ok_to_advance needs a valid vision fix at most this old


@dataclass
class ServoState:
    stamp: float
    valid: bool                      # this frame had an accepted vision measurement and small posterior sigmas
    ok_to_advance: bool              # valid fix <= max_advance_gap_s old and sigmas small
    ok_to_coast: bool                # blind last stretch: VIO dead reckoning from a valid fix near the exit (bounded)
    confidence: float
    mode: str                        # 'vision' | 'predict' | 'dead_reckoning' | 'init'
    offset_l: float                  # vehicle left of the centre line [m]
    offset_u: float                  # vehicle above the centre line [m]
    yaw_err: float                   # vehicle heading - axis heading [rad], + = nose right
    progress_m: float                # along the axis from the entry hoop plane [m]
    dist_to_exit_m: float            # exit hoop plane - vehicle, along the axis [m]
    sigma_l: float
    sigma_u: float
    sigma_yaw: float
    sigma_s: float
    since_valid_s: float
    axis_heading: float              # tunnel axis heading in the attitude (VIO-yaw) frame [rad, NED]
    correction_ned: np.ndarray       # NED vector vehicle -> its point on the centre line (horizontal + vertical)
    axis_ned: np.ndarray             # (3,) unit axis in the attitude frame
    velocity_tunnel: np.ndarray      # (vs, vl, vu) m/s
    dr_dist_m: float = 0.0           # VIO dead-reckoned distance since the last valid vision fix
    sources: Tuple[str, ...] = ()    # 'hoops', 'depth', 'tof'
    hoops_seen: Tuple[int, ...] = ()
    cover: float = 0.0
    rms_px: float = float('nan')
    n_hoop_pts: int = 0
    n_depth_pts: int = 0
    cam_offset_l: float = float('nan')
    cam_offset_u: float = float('nan')
    ms: float = 0.0
    debug: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------ the servo
class TunnelServo:
    """One instance per traverse. Call update(frame, ...) for every frame, in time order."""

    def __init__(self, model: TunnelModel, params: Optional[ServoParams] = None):
        self.m = model
        self.p = params or ServoParams()
        self.x = None                 # [s, l, u, psi, vs, vl, vu] (vehicle = base_link)
        self.P = None
        self.t = None
        self.t_valid = None
        self.last_odom = None
        self.dr_dist = 0.0            # dead-reckoned distance since the last valid vision fix
        self.exit_at_fix = None       # dist_to_exit at the last valid vision fix
        th = np.linspace(0.0, 2.0 * math.pi, self.p.pts_per_hoop, endpoint=False)
        # theta = +90 deg is the top of the hoop (sin(theta) multiplies the UP axis)
        self._th = th[np.abs(_wrap(th - math.pi / 2)) > math.radians(self.p.skip_top_deg)]
        tb = np.asarray(self.p.t_base_cam, float)
        self._Rbc = np.asarray(self.p.R_base_cam, float)
        self._t_frd = np.einsum('ij,j->i', _F, tb)

    # ------------------------------------------------------------------ geometry helpers
    def _body_rot(self, R_nc):
        return _mm(np.asarray(R_nc, float), _mm(_F, self._Rbc).T)          # R_nb (FRD body -> NED)

    @staticmethod
    def _frame_axes(psi):
        c, s = math.cos(psi), math.sin(psi)
        a = np.array([c, s, 0.0])
        L = np.array([s, -c, 0.0])
        U = np.array([0.0, 0.0, -1.0])
        return a, L, U

    def _cam_rel_entry(self, pose, d_cam):
        """camera position relative to the entry hoop centre (NED axes, attitude frame)"""
        s, l, u, psi = pose
        a, L, U = self._frame_axes(psi)
        return s * a + l * L + u * U + d_cam, (a, L, U)

    def _project_hoops(self, pose, R_cn, K, d_cam, th=None):
        """-> uv (H, N, 2), z (H, N), normal (H, N, 2) for all hoops."""
        th = self._th if th is None else th
        cam, (a, L, U) = self._cam_rel_entry(pose, d_cam)
        R = self.m.radius_m
        hs = np.asarray(self.m.hoop_s, float)
        C = hs[:, None] * a[None, :] - cam[None, :]                                 # (H, 3) centres rel camera
        cth, sth = np.cos(th), np.sin(th)
        P = C[:, None, :] + R * (cth[None, :, None] * L + sth[None, :, None] * U)    # (H, N, 3)
        Tg = R * (-sth[None, :, None] * L + cth[None, :, None] * U)                  # tangent (H, N, 3)
        Xc = _rotT(P, R_cn)
        Tc = _rotT(np.broadcast_to(Tg, P.shape), R_cn)
        fx, fy, cx, cy = K
        z = Xc[..., 2]
        zs = np.where(z > 1e-6, z, 1e-6)
        u = fx * Xc[..., 0] / zs + cx
        v = fy * Xc[..., 1] / zs + cy
        du = fx * (Tc[..., 0] * zs - Xc[..., 0] * Tc[..., 2]) / (zs * zs)          # d(uv)/dtheta
        dv = fy * (Tc[..., 1] * zs - Xc[..., 1] * Tc[..., 2]) / (zs * zs)
        nn = np.hypot(du, dv) + 1e-12
        normal = np.stack([-dv / nn, du / nn], -1)
        return np.stack([u, v], -1), z, normal

    # ------------------------------------------------------------------ image features
    def _edges(self, img):
        p = self.p
        g = cv2.GaussianBlur(img, (0, 0), p.blur_sigma)
        dx = cv2.Sobel(g, cv2.CV_16S, 1, 0, ksize=3)
        dy = cv2.Sobel(g, cv2.CV_16S, 0, 1, ksize=3)
        fx_, fy_ = dx.astype(np.float32), dy.astype(np.float32)
        mag = cv2.magnitude(fx_, fy_)
        nz = mag[::4, ::4]
        hi = max(p.canny_hi_min, float(np.percentile(nz, p.canny_hi_pct)))
        e = cv2.Canny(dx, dy, p.canny_lo_frac * hi, hi, L2gradient=True) > 0
        return e, fx_, fy_, mag

    def _match(self, uv, normal, ok, feats, gate):
        """1-D search along each model normal for the nearest edge whose orientation matches.
        A thin wire gives two edges of opposite polarity a few px apart: when the nearest edge has such a partner
        within line_max_px, the match is the pair's midpoint (the wire centre). Otherwise it is the single edge
        (silhouette of the opening). -> (offset px along the normal, matched mask, is_line mask)"""
        e, gx, gy, mag = feats
        Hh, Ww = e.shape
        G = int(math.ceil(gate)) + int(self.p.line_max_px)
        ks = np.arange(-G, G + 1, dtype=np.float32)
        q = uv[:, None, :] + ks[None, :, None] * normal[:, None, :]
        good = np.isfinite(q).all(-1)
        qi = np.rint(np.where(good[..., None], q, -1.0)).astype(np.int32)
        inb = good & (qi[..., 0] >= 0) & (qi[..., 0] < Ww) & (qi[..., 1] >= 0) & (qi[..., 1] < Hh) & ok[:, None]
        xi = np.clip(qi[..., 0], 0, Ww - 1)
        yi = np.clip(qi[..., 1], 0, Hh - 1)
        hit = inb & e[yi, xi]
        gn = gx[yi, xi] * normal[:, None, 0] + gy[yi, xi] * normal[:, None, 1]     # signed gradient along the normal
        hit &= np.abs(gn) >= math.cos(math.radians(self.p.orient_tol_deg)) * (mag[yi, xi] + 1e-6)
        ak = np.abs(ks)[None, :]
        near = hit & (ak <= gate)
        anyh = near.any(1)
        big = 1e6
        j1 = np.argmin(np.where(near, ak, big), 1)                       # nearest edge within the gate
        k1 = ks[j1]
        s1 = np.sign(gn[np.arange(len(j1)), j1])
        dk = ks[None, :] - k1[:, None]
        cand = hit & (np.abs(dk) >= 2) & (np.abs(dk) <= self.p.line_max_px) & (np.sign(gn) == -s1[:, None])
        j2 = np.argmin(np.where(cand, np.abs(dk), big), 1)
        has2 = cand[np.arange(len(j2)), j2]
        k = np.where(has2, 0.5 * (k1 + ks[j2]), k1)
        return k, anyh, anyh & has2

    # ------------------------------------------------------------------ depth points
    def _depth_points(self, depth, K, R_nc):
        p = self.p
        st = p.depth_stride
        d = np.asarray(depth, np.float32)[::st, ::st]
        Hh, Ww = d.shape
        fx, fy, cx, cy = K
        v, u = np.mgrid[0:Hh, 0:Ww]
        u = u * st
        v = v * st
        ok = np.isfinite(d) & (d > 0.15) & (d < p.depth_max_range_m)
        if ok.sum() < p.depth_min_pts:
            return None
        z = d[ok].astype(float)
        X = np.stack([(u[ok] - cx) / fx * z, (v[ok] - cy) / fy * z, z], 1)
        return _rotT(X, np.asarray(R_nc, float)), z

    # ------------------------------------------------------------------ public
    def reset(self):
        self.x = None
        self.P = None
        self.t = None
        self.t_valid = None
        self.last_odom = None
        self.dr_dist = 0.0
        self.exit_at_fix = None

    def _init_state(self, frame):
        R_nc = np.asarray(frame.R_nc, float)
        R_nb = self._body_rot(R_nc)
        body = np.asarray(frame.cam_ned, float) - np.einsum('ij,j->i', R_nb, self._t_frd)
        psi = math.atan2(self.m.axis_ned[1], self.m.axis_ned[0])
        a, L, U = self._frame_axes(psi)
        rel = body - self.m.entry_ned
        self.x = np.array([_dot3(rel, a), _dot3(rel, L), _dot3(rel, U), psi, 0.0, 0.0, 0.0])
        sg = list(self.p.init_sigma) + [self.p.init_vel_sigma] * 3
        self.P = np.diag(np.square(sg))

    def _predict(self, dt, odom):
        """-> True when the VIO delta was used (dead reckoning), False for constant velocity."""
        p = self.p
        Fm = np.eye(7)
        d = None
        if odom is not None and self.last_odom is not None:
            d = np.asarray(odom, float) - self.last_odom
            if not np.all(np.isfinite(d)) or _norm3(d) > p.odom_max_speed * max(dt, 0.05) + 0.05:
                d = None                                   # VIO jumped: do not follow it
        if d is not None:
            # dead reckoning on the VIO body delta (short window), expressed in the tunnel frame
            a, L, U = self._frame_axes(self.x[3])
            step = np.array([_dot3(d, a), _dot3(d, L), _dot3(d, U)])
            self.x[:3] += step
            if dt > 1e-3:
                self.x[4:7] = step / dt
            sg = p.odom_sigma_frac * np.abs(step) + 0.005
            Q = np.zeros((7, 7))
            Q[0, 0], Q[1, 1], Q[2, 2] = sg ** 2
            Q[3, 3] = (math.radians(p.psi_rw_deg_s) ** 2) * dt
            Q[4:7, 4:7] = np.eye(3) * (p.acc_sigma * max(dt, 1e-3)) ** 2
            self.P = self.P + Q
            self.dr_dist += float(_norm3(d))
            return True
        for i in range(3):
            Fm[i, 4 + i] = dt
        self.x = np.einsum('ij,j->i', Fm, self.x)
        q = p.acc_sigma ** 2
        Q = np.zeros((7, 7))
        for i in range(3):
            Q[i, i] = q * dt ** 4 / 4.0
            Q[i, 4 + i] = Q[4 + i, i] = q * dt ** 3 / 2.0
            Q[4 + i, 4 + i] = q * dt ** 2
        Q[3, 3] = (math.radians(p.psi_rw_deg_s) ** 2) * dt
        self.P = _mm(_mm(Fm, self.P), Fm.T) + Q
        return False

    def update(self, frame, tof_m: Optional[float] = None, odom_ned: Optional[Sequence[float]] = None,
               image_scale_ok: bool = True) -> ServoState:
        """frame: perception Frame (image, depth, K, R_nc, cam_ned, stamp). tof_m: downward range (body z) to the
        floor, metres (optional). odom_ned: VIO body position (optional; only its short-term deltas are used,
        for prediction when vision is missing)."""
        t0 = time.perf_counter()
        p = self.p
        stamp = float(getattr(frame, 'stamp', 0.0))
        img = np.asarray(frame.image)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        K = tuple(float(k) for k in frame.K)
        depth = getattr(frame, 'depth', None)
        if img.shape[1] > p.work_width and image_scale_ok:
            sc = p.work_width / img.shape[1]
            img = cv2.resize(img, (p.work_width, int(round(img.shape[0] * sc))), interpolation=cv2.INTER_AREA)
            K = (K[0] * sc, K[1] * sc, K[2] * sc, K[3] * sc)
            if depth is not None:
                depth = cv2.resize(np.asarray(depth, np.float32), (img.shape[1], img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        R_nc = np.asarray(frame.R_nc, float)
        R_cn = R_nc.T
        R_nb = self._body_rot(R_nc)
        d_cam = np.einsum('ij,j->i', R_nb, self._t_frd)  # camera - body, NED
        heading = math.atan2(R_nb[1, 0], R_nb[0, 0])
        first = self.x is None
        if first:
            self._init_state(frame)
            self.t = stamp
        dt = float(np.clip(stamp - self.t, 0.0, p.max_dt_s))
        used_dr = False
        if not first:
            used_dr = self._predict(dt, odom_ned)
        if odom_ned is not None:
            self.last_odom = np.asarray(odom_ned, float).copy()
        self.t = stamp
        since = (stamp - self.t_valid) if self.t_valid is not None else float('inf')
        tight = math.sqrt(self.P[0, 0]) <= p.tight_sigma_s_m and math.sqrt(self.P[1, 1]) <= p.tight_sigma_lu_m and \
            math.sqrt(self.P[2, 2]) <= p.tight_sigma_lu_m
        # lost = the prediction cannot be trusted: no fix for a while and no dead reckoning keeping the prior tight
        lost = first or (since > p.reacquire_after_s and not (used_dr and tight))

        feats = self._edges(img)
        x_pred, P_pred = self.x.copy(), self.P.copy()
        dbg = {}
        pts = None
        if p.use_depth and depth is not None:
            pts = self._depth_points(depth, K, R_nc)
        gates = p.gate_lost_px if (lost and not first) else p.gate_px      # track start: stay local to the latch
        # 0) acquisition: depth first. The cylinder fit has no wrong basins from +-10 cm / +-5 deg starts, while hoop
        #    edges on real clutter do (lateral offset and heading trade off at the standoff). Depth seeds l, u, psi.
        if lost and pts is not None:
            P_w = P_pred.copy()
            P_w[1:4, 1:4] = np.diag(np.square([0.25, 0.25, math.radians(8.0)]))
            xd, Pd, infod = self._iekf(x_pred, P_w, R_cn, K, d_cam, feats, pts, None, R_nb, p.gate_px, use_h=False)
            if infod['depth_ok']:
                x_pred = x_pred.copy()
                x_pred[1:4] = xd[1:4]
                dbg['acq_depth'] = np.round(xd[1:4] - self.x[1:4], 3).tolist()
        # 1) hoops (+ ToF) from the prediction
        x_new, P_new, info = self._iekf(x_pred, P_pred, R_cn, K, d_cam, feats, None, tof_m, R_nb, gates)
        h_ok = info['hoops_ok']
        weak = False
        if not h_ok and not lost and used_dr and tight and info['n_hoop'] >= p.weak_min_pts and \
                info['cover'] >= p.weak_min_cover and info['rms'] <= p.weak_max_rms_px:
            weak = True               # a few clean corner arcs of the exit hoop: refine the dead reckoning, never 'valid'
        # 2) lost and no hoops: grid search; a search start is kept only if the hoops then pass
        if p.search_on and lost and not first and not h_ok and len(self._th):
            xs, sc_best, sc_here = self._search(x_pred, R_cn, K, d_cam, feats)
            dbg['search_score'] = (round(sc_best, 3), round(sc_here, 3))
            if sc_best > p.search_gain * sc_here + 0.02:
                P_s = P_pred.copy()
                if first:
                    P_s[:4, :4] = np.diag(np.square([0.15, 0.1, 0.1, math.radians(3.0)]))
                x2, P2, info2 = self._iekf(xs, P_s, R_cn, K, d_cam, feats, None, tof_m, R_nb, gates)
                if info2['hoops_ok'] and (not h_ok or info2['n_hoop'] > 1.15 * info['n_hoop']):
                    x_new, P_new, info, h_ok = x2, P2, info2, True
                    dbg['search'] = np.round(xs[:4] - x_pred[:4], 3).tolist()
        # 3) depth: refines a hoop fix, or carries l/u/psi alone while the track is fresh (never after a search)
        d_ok = False
        if pts is not None and (h_ok or since <= p.depth_alone_max_s):
            x0, P0 = (x_new, P_new) if h_ok else (x_pred, P_pred)
            x3, P3_, info3 = self._iekf(x0, P0, R_cn, K, d_cam, feats, pts, None, R_nb, p.gate_px[-2:],
                                        use_h=False)
            if info3['depth_ok']:
                d_ok = True
                x_new, P_new = x3, P3_
                for kk in ('n_depth', 'depth_arc_deg', 'depth_rms_m'):
                    info[kk] = info3[kk]
            else:
                for kk in ('n_depth', 'depth_arc_deg', 'depth_rms_m'):
                    if kk in info3:
                        info[kk] = info3[kk]
        # 4) conflict: plenty of tube depth that does NOT fit the hoop solution = the hoops locked onto clutter
        #    (seen on the real recordings after a latch 0.2 m too far along the axis). Reject the hoop fix.
        if h_ok and not d_ok and info.get('n_depth', 0) >= p.conflict_min_pts and \
                info.get('depth_arc_deg', 0.0) >= p.conflict_min_arc_deg:
            h_ok = False
            dbg['reject'] = 'depth_conflict'
            x_new, P_new = x_pred, P_pred
        info['hoops_ok'], info['depth_ok'] = h_ok, d_ok
        if not h_ok:
            info['tof_used'] = False
        sources = [n for n, ok in (('hoops', h_ok), ('depth', d_ok), ('hoops_weak', weak and not d_ok)) if ok]
        if sources and info.get('tof_used'):
            sources.append('tof')
        if not sources:
            x_new, P_new = x_pred, P_pred
        accept = bool(sources)
        if accept and not lost:
            dx = x_new[:4] - self.x[:4]
            dx[3] = _wrap(dx[3])
            try:
                chi2 = float(np.sum(dx * _solve(self.P[:4, :4], dx)))
            except np.linalg.LinAlgError:
                chi2 = 0.0
            info['chi2'] = round(chi2, 1)
            if chi2 > p.max_innovation_chi2:
                accept = False
                dbg['reject'] = 'innovation'
        if accept:
            self.x, self.P = x_new, P_new
            self.x[3] = _wrap(self.x[3])
        sl, su = math.sqrt(self.P[1, 1]), math.sqrt(self.P[2, 2])
        sy, ss = math.sqrt(self.P[3, 3]), math.sqrt(self.P[0, 0])
        vision = accept and ('hoops' in sources or 'depth' in sources)
        valid = vision and sl <= p.valid_sigma_lu_m and su <= p.valid_sigma_lu_m and \
            sy <= math.radians(p.valid_sigma_yaw_deg)
        if valid:
            self.t_valid = stamp
            self.dr_dist = 0.0
            self.exit_at_fix = float(self.m.exit_s - self.x[0])
        since = (stamp - self.t_valid) if self.t_valid is not None else float('inf')
        ok_adv = since <= p.max_advance_gap_s and sl <= p.valid_sigma_lu_m and su <= p.valid_sigma_lu_m and \
            sy <= math.radians(p.valid_sigma_yaw_deg)
        # coast: the last metre before the exit is blind (exit hoop outside the FOV, no tube depth). Allowed only on
        # VIO dead reckoning that started from a valid fix close to the exit, for a bounded distance.
        ok_coast = bool(not ok_adv and used_dr and self.exit_at_fix is not None and
                        self.exit_at_fix <= p.coast_from_exit_m and self.dr_dist <= p.coast_max_m)
        if valid:
            mode = 'vision'
        elif first:
            mode = 'init'
        elif ok_coast:
            mode = 'coast'
        elif used_dr:
            mode = 'dead_reckoning'
        else:
            mode = 'predict'
        # confidence: measurement quality x posterior tightness
        hq = 0.0
        if 'hoops' in sources:
            hq = float(np.clip((info['cover'] - 0.2) / 0.4, 0, 1) * np.clip(1.0 - (info['rms'] - 1.5) / 2.5, 0, 1))
        dq = 0.0
        if 'depth' in sources:
            dq = float(np.clip((info['n_depth'] - p.depth_min_pts) / 200.0, 0, 1) *
                       np.clip((info['depth_arc_deg'] - 60.0) / 120.0, 0, 1))
        mq = 1.0 - (1.0 - hq) * (1.0 - dq)
        tq = math.exp(-((sl / 0.04) ** 2 + (su / 0.04) ** 2 + (sy / math.radians(3.0)) ** 2) / 6.0)
        conf = float(mq * tq) if vision else 0.0
        s, l, u, psi = self.x[:4]
        a, L, U = self._frame_axes(psi)
        corr = -(l * L + u * U)
        cam_rel, _ = self._cam_rel_entry(self.x[:4], d_cam)
        dbg.update(info)
        return ServoState(
            stamp=stamp, valid=bool(valid), ok_to_advance=bool(ok_adv), ok_to_coast=ok_coast,
            confidence=round(conf, 3), mode=mode, dr_dist_m=float(self.dr_dist),
            offset_l=float(l), offset_u=float(u), yaw_err=float(_wrap(heading - psi)), progress_m=float(s),
            dist_to_exit_m=float(self.m.exit_s - s), sigma_l=sl, sigma_u=su, sigma_yaw=sy, sigma_s=ss,
            since_valid_s=float(since), axis_heading=float(psi), correction_ned=corr, axis_ned=a,
            velocity_tunnel=self.x[4:7].copy(), sources=tuple(sources) if accept else (),
            hoops_seen=tuple(info.get('hoops_seen', ())), cover=float(info.get('cover', 0.0)),
            rms_px=float(info.get('rms', float('nan'))), n_hoop_pts=int(info.get('n_hoop', 0)),
            n_depth_pts=int(info.get('n_depth', 0)), cam_offset_l=float(_dot3(cam_rel, L)), cam_offset_u=float(_dot3(cam_rel, U)),
            ms=round((time.perf_counter() - t0) * 1e3, 1), debug=dbg)

    # ------------------------------------------------------------------ reacquisition grid
    def _search(self, x0, R_cn, K, d_cam, feats, wide=False):
        """Grid search around x0 (vectorised). Score = orientation-consistent edge hits within ~4 px over the model
        points in view. wide (track start): s +-0.35 m and u +-0.2 m (l, psi barely).
        -> (best x, best score, score at x0)"""
        p = self.p
        e, gx, gy, mag = feats
        Hh, Ww = e.shape
        nb = 6
        ang = np.mod(np.arctan2(gy, gx), math.pi)
        b = np.minimum((ang / (math.pi / nb)).astype(np.int32), nb - 1)
        ker = np.ones((9, 9), np.uint8)
        dil = np.stack([cv2.dilate((e & ((b == i) | (b == (i + 1) % nb) | (b == (i - 1) % nb))).astype(np.uint8), ker) > 0
                        for i in range(nb)], 0)
        th = self._th[::3]
        if wide:
            # less than half a hoop spacing along the axis: the hoop INDEX comes from the model anchoring
            # (TunnelModel.from_estimate), not from edges. l and psi are NOT searched: at the standoff the visible
            # hoops pin only their combination (l + d*psi), and a grid picks wrong pairs on real clutter. The filter
            # separates them from motion parallax during align / approach.
            ds = np.arange(-0.35, 0.351, 0.05)
            dl = np.array([-0.05, 0.0, 0.05])
            du = np.arange(-0.2, 0.201, 0.05)
            dp = np.radians(np.array([-1.5, 0.0, 1.5]))
        else:
            ds = np.arange(-0.3, 0.301, 0.1)
            dl = np.arange(-0.2, 0.201, 0.1)
            du = np.arange(-0.2, 0.201, 0.1)
            dp = np.radians(np.arange(-6.0, 6.01, 3.0))
        G = np.stack(np.meshgrid(ds, dl, du, dp, indexing='ij'), -1).reshape(-1, 4)
        poses = x0[:4][None, :] + G
        sc = self._score_batch(poses, R_cn, K, d_cam, th, dil, nb)
        k0 = int(np.argmin(np.abs(G).sum(1)))
        kb = int(np.argmax(sc))
        x = x0.copy()
        x[:4] = poses[kb]
        return x, float(sc[kb]), float(sc[k0])

    def _score_batch(self, poses, R_cn, K, d_cam, th, dil, nb, chunk=1500):
        """Fraction-like edge score for many poses at once (no BLAS)."""
        out = np.empty(len(poses))
        Hh, Ww = dil.shape[1:]
        R = self.m.radius_m
        hs = np.asarray(self.m.hoop_s, float)
        cth, sth = np.cos(th), np.sin(th)
        fx, fy, cx, cy = K
        for c0 in range(0, len(poses), chunk):
            P_ = poses[c0:c0 + chunk]
            ca, sa = np.cos(P_[:, 3]), np.sin(P_[:, 3])
            a = np.stack([ca, sa, 0 * ca], 1)
            L = np.stack([sa, -ca, 0 * ca], 1)
            U = np.array([0.0, 0.0, -1.0])
            cam = P_[:, 0:1] * a + P_[:, 1:2] * L + P_[:, 2:3] * U[None, :] + d_cam[None, :]
            C = hs[None, :, None] * a[:, None, :] - cam[:, None, :]                       # (n, H, 3)
            ring = R * (cth[None, :, None] * L[:, None, :] + sth[None, :, None] * U[None, None, :])   # (n, T, 3)
            tang = R * (-sth[None, :, None] * L[:, None, :] + cth[None, :, None] * U[None, None, :])
            Pw = C[:, :, None, :] + ring[:, None, :, :]
            Xc = _rotT(Pw, R_cn)
            Tc = _rotT(np.broadcast_to(tang[:, None, :, :], Pw.shape), R_cn)
            z = Xc[..., 2]
            zs = np.where(z > 1e-6, z, 1e-6)
            u = fx * Xc[..., 0] / zs + cx
            v = fy * Xc[..., 1] / zs + cy
            du = fx * (Tc[..., 0] * zs - Xc[..., 0] * Tc[..., 2])
            dv = fy * (Tc[..., 1] * zs - Xc[..., 1] * Tc[..., 2])
            na = np.mod(np.arctan2(du, -dv), math.pi)                                        # normal direction
            nbi = np.minimum((na / (math.pi / nb)).astype(np.int32), nb - 1)
            ok = (z > self.p.min_view_z_m) & (u >= 2) & (u < Ww - 2) & (v >= 2) & (v < Hh - 2)
            ui = np.clip(np.rint(np.where(ok, u, 0)).astype(np.int32), 0, Ww - 1)
            vi = np.clip(np.rint(np.where(ok, v, 0)).astype(np.int32), 0, Hh - 1)
            hit = ok & dil[nbi, vi, ui]
            n_ok = ok.reshape(len(P_), -1).sum(1)
            out[c0:c0 + chunk] = hit.reshape(len(P_), -1).sum(1) / (n_ok + 15.0)
        return out

    # ------------------------------------------------------------------ the iterated EKF update
    def _iekf(self, x_pred, P_pred, R_cn, K, d_cam, feats, pts, tof_m, R_nb, gates, use_h=True, use_d=True,
              x_start=None):
        """Iterated EKF update (information form) with hoop, depth and ToF residuals.
        -> (x, P, info); info['hoops_ok'] / ['depth_ok'] say whether each source passed its quality test."""
        p = self.p
        e = feats[0]
        Hh, Ww = e.shape
        Pinv = _inv(P_pred)
        x = (x_pred if x_start is None else x_start).copy()
        steps = np.array([0.01, 0.01, 0.01, 0.002])
        R = self.m.radius_m
        info = dict(cover=0.0, rms=float('nan'), n_hoop=0, n_depth=0, depth_arc_deg=0.0)
        use_d = use_d and pts is not None
        use_t = tof_m is not None and self.m.axis_height_m is not None
        H_all = None
        for it, gate in enumerate(gates):
            rows_H, rows_r, rows_w = [], [], []
            # ---- hoops
            if use_h:
                uv, z, nrm = self._project_hoops(x[:4], R_cn, K, d_cam)
                nh, npt = z.shape
                uvf, zf, nf = uv.reshape(-1, 2), z.reshape(-1), nrm.reshape(-1, 2)
                bp = p.border_px
                vis = (zf > p.min_view_z_m) & (uvf[:, 0] >= bp) & (uvf[:, 0] < Ww - bp) & (uvf[:, 1] >= bp) & \
                    (uvf[:, 1] < Hh - bp)
                k, hit, isline = self._match(uvf, nf, vis, feats, gate)
                m = vis & hit
                n_vis = int(vis.sum())
                if m.sum() >= 6:
                    J = np.zeros((int(m.sum()), 7))
                    for j in range(4):
                        xj = x[:4].copy()
                        xj[j] += steps[j]
                        uvj, _, _ = self._project_hoops(xj, R_cn, K, d_cam)
                        duv = (uvj.reshape(-1, 2)[m] - uvf[m]) / steps[j]
                        J[:, j] = np.sum(duv * nf[m], 1)
                    r = -k[m].astype(float)                     # residual = n . (uv - edge)
                    a_ = np.abs(r)
                    w = np.where(a_ <= p.huber_px, 1.0, p.huber_px / np.maximum(a_, 1e-9)) / p.sigma_px ** 2
                    w *= min(1.0, p.hoop_n_eff / max(len(r), 1))     # residuals are correlated along an arc
                    rows_H.append(J)
                    rows_r.append(r)
                    rows_w.append(w)
                    hidx = np.repeat(np.arange(nh), npt)
                    thb = np.tile((np.mod(self._th, 2 * math.pi) // (math.pi / 6)).astype(int), nh)
                    arc_bins = len(np.unique((hidx * 12 + thb)[m]))
                    covs = []
                    for hh in range(nh):
                        vv = vis & (hidx == hh)
                        nvv = int(vv.sum())
                        covs.append((int((m & (hidx == hh)).sum()) / nvv) if nvv >= 8 else 0.0)
                    info.update(cover=float(m.sum()) / max(n_vis, 1), n_hoop=int(m.sum()),
                                line_frac=float((isline & m).sum()) / max(int(m.sum()), 1),
                                rms=float(np.sqrt(np.mean(np.minimum(r * r, 25.0)))),
                                arc_bins=arc_bins,
                                hoops_seen=tuple(i for i, c in enumerate(covs) if c >= p.min_cover),
                                cover_per_hoop=[round(c, 2) for c in covs], n_vis=n_vis)
                else:
                    info.update(cover=0.0, n_hoop=int(m.sum()), n_vis=n_vis)
            # ---- depth
            if use_d:
                P3, zz = pts
                s, l, u, psi = x[:4]
                a, L, U = self._frame_axes(psi)
                cam, _ = self._cam_rel_entry(x[:4], d_cam)
                C0 = -cam                                      # entry centre rel camera
                rel = P3 - C0
                t = _dot3(rel, a)
                radial = rel - t[:, None] * a
                rho = _norm3(radial)
                hup = _dot3(rel, U)
                gate_d = p.depth_gate_m if it > 0 else p.depth_gate_m * 1.5
                sel = (t >= self.m.hoop_s[0] - 0.05) & (t <= self.m.exit_s + 0.05) & \
                      (np.abs(rho - R) < gate_d) & (hup > -p.depth_floor_frac * R)
                idx = np.nonzero(sel)[0]
                if len(idx) > p.depth_max_pts:
                    idx = idx[np.linspace(0, len(idx) - 1, p.depth_max_pts).astype(int)]
                if len(idx) >= p.depth_min_pts:
                    def rho_of(pose):
                        cm, _ = self._cam_rel_entry(pose, d_cam)
                        a2, _, _ = self._frame_axes(pose[3])
                        rl = P3[idx] + cm
                        rr_ = rl - _dot3(rl, a2)[:, None] * a2
                        return _norm3(rr_)
                    r0 = rho_of(x[:4])
                    J = np.zeros((len(idx), 7))
                    for j in (1, 2, 3):
                        xj = x[:4].copy()
                        xj[j] += steps[j]
                        J[:, j] = (rho_of(xj) - r0) / steps[j]
                    rr = r0 - R
                    sg = p.depth_sigma0_m + p.depth_sigma_z2 * zz[idx] ** 2
                    a_ = np.abs(rr) / sg
                    w = np.where(a_ <= 2.0, 1.0, 2.0 / np.maximum(a_, 1e-9)) / sg ** 2
                    w *= min(1.0, p.depth_n_eff / max(len(rr), 1))   # the tube is not a perfect cylinder
                    rows_H.append(J)
                    rows_r.append(rr)
                    rows_w.append(w)
                    ang = np.degrees(np.arctan2(_dot3(radial[idx], U), _dot3(radial[idx], L)))
                    good = np.abs(rr) < 2.5 * sg
                    bins = np.unique(((ang[good] + 180.0) // 30).astype(int))
                    info.update(n_depth=int(good.sum()), depth_arc_deg=30.0 * len(bins),
                                depth_rms_m=float(np.sqrt(np.mean(np.minimum(rr * rr, 0.04)))))
                else:
                    info.update(n_depth=0, depth_arc_deg=0.0)
            # ---- ToF
            if use_t:
                tilt_c = max(R_nb[2, 2], 0.5)
                h_meas = float(tof_m) * tilt_c
                rr = np.array([(self.m.axis_height_m + x[2]) - h_meas])
                J = np.zeros((1, 7))
                J[0, 2] = 1.0
                rows_H.append(J)
                rows_r.append(rr)
                rows_w.append(np.array([1.0 / p.tof_sigma_m ** 2]))
            if not rows_H:
                break
            H = np.concatenate(rows_H)
            rv = np.concatenate(rows_r)
            wv = np.concatenate(rows_w)
            A = Pinv + _gram(H, wv)
            dx = x_pred - x
            dx[3] = _wrap(dx[3])
            g = np.einsum('ni,n->i', H, wv * (rv + np.einsum('ni,i->n', H, dx)))
            try:
                step = _solve(A, g)
            except np.linalg.LinAlgError:
                break
            x = x_pred - step
            H_all = (A,)
        # ---- per-source quality
        info['hoops_ok'] = bool(use_h and info['n_hoop'] >= p.min_match_pts and info['cover'] >= p.min_cover and
                                info['rms'] <= p.max_rms_px and info.get('arc_bins', 0) >= p.min_arc_bins)
        info['depth_ok'] = bool(use_d and info['n_depth'] >= p.depth_min_pts and
                                info['depth_arc_deg'] >= p.depth_min_arc_deg and
                                info.get('depth_rms_m', 1.0) <= p.depth_max_rms_m)
        info['tof_used'] = bool(use_t)
        if H_all is None:
            return x_pred, P_pred, info
        try:
            P_new = _inv(H_all[0])
        except np.linalg.LinAlgError:
            return x_pred, P_pred, info
        return x, 0.5 * (P_new + P_new.T), info


# ------------------------------------------------------------------------------------------------ helpers for core
def servo_velocity(st: ServoState, v_fwd_max: float = 0.25, k_lat: float = 1.0, k_yaw: float = 1.0,
                   v_corr_max: float = 0.15, yaw_rate_max: float = 0.4, exit_margin_m: float = 0.5):
    """Reference control law (mission-core may use its own): NED velocity + yaw rate from the servo state.
    Forward speed only while ok_to_advance, tapered when far off-centre; it stops `exit_margin_m` past the exit.
    Returns (v_ned (3,), yaw_rate rad/s, advancing bool)."""
    a = np.asarray(st.axis_ned, float)
    corr = np.asarray(st.correction_ned, float)
    vc = k_lat * corr
    n = float(np.linalg.norm(vc))
    if n > v_corr_max:
        vc *= v_corr_max / n
    off = math.hypot(st.offset_l, st.offset_u)
    adv = bool(st.ok_to_advance) and st.dist_to_exit_m > -exit_margin_m
    v_f = v_fwd_max * float(np.clip(1.0 - (off - 0.05) / 0.10, 0.0, 1.0)) * float(
        np.clip(1.0 - (abs(st.yaw_err) - math.radians(5)) / math.radians(10), 0.0, 1.0)) if adv else 0.0
    yr = float(np.clip(-k_yaw * st.yaw_err, -yaw_rate_max, yaw_rate_max))
    return v_f * a + vc, yr, v_f > 0.0
