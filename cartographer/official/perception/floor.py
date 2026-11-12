"""Floor plane, raised-obstacle points and the white landing block from one
OAK-D frame (horizontal camera, depth aligned to the left image).

Pure numpy + OpenCV: no ROS, no clock reads, deterministic (fixed RNG seed).
`frame` is duck-typed like `common.geometry.Frame` (image, depth, K, R_nc,
cam_ned); `image` may be uint8 grey (the real OAK-D left camera) or BGR (then
saturation is used as an extra "white/grey" cue).

Conventions (PX4 local NED, z down):
  * floor plane: n . p = n . anchor, `normal_ned` = unit normal pointing DOWN
    into the floor (nominally (0, 0, 1)); `d` = NED z of the floor directly
    below the camera at fit time, anchor = (cam_x, cam_y, d).
  * height above floor: h(p) = n . (anchor - p)  (> 0 above the floor).
    Use `FloorFit.height_above(points)` instead of re-deriving it.

Why the plane is fitted (not assumed level): a 1 deg attitude/extrinsic error
moves the floor 7 cm at 4 m; fitting n absorbs that, so painted rocks stay at
h ~ 0 out to the obstacle range.

All (N,3)x(3,) products, 2x2/3x3 least squares and covariances are written out
element-wise on purpose: numpy hands tiny matmuls to multithreaded BLAS, which
stalls for tens of ms when the CPU is shared (seen at load 12 on the dev Mac).
"""
from dataclasses import dataclass, field
from functools import lru_cache
import math

import numpy as np
import cv2

OAKD_BASELINE_M = 0.075          # OAK-D stereo baseline
_DOWN = np.array([0.0, 0.0, 1.0])
_NAN3 = (math.nan, math.nan, math.nan)


# --------------------------------------------------------------------------- results
@dataclass
class FloorFit:
    found: bool                      # True only if the plane was confirmed by depth this frame
    d: float                         # NED z of the floor under the camera (NaN = unknown)
    normal_ned: np.ndarray           # (3,) unit, pointing down into the floor
    inlier_frac: float               # floor inliers / valid depth points (<= max_range)
    anchor_ned: np.ndarray = field(default_factory=lambda: np.array(_NAN3))
    n_inliers: int = 0
    n_cells: int = 0                 # distinct 10 cm XY cells covered by inliers
    tilt_deg: float = math.nan       # angle between normal and NED down
    sigma0: float = 0.015            # residual model sigma(z) = sigma0 + sigma1 * z  [m]
    sigma1: float = 0.008            # (defaults = prior-only plane: ToF error + ~0.5 deg attitude)
    floor_brightness: float = math.nan   # median grey of inlier pixels
    source: str = 'none'             # 'depth' | 'depth+prior' | 'prior' | 'none'
    d_stereo: float = math.nan       # floor z under the camera of the stereo plane itself (= anchor z)
    reason: str = ''

    @property
    def usable(self):
        """A plane is available (depth fit, or the prior when the fit failed)."""
        return bool(np.isfinite(self.d))

    def height_above(self, p_ned):
        """Height (m, + up) of NED point(s) above this plane."""
        p = np.asarray(p_ned, float)
        return _dot3(np.asarray(self.anchor_ned, float) - p, np.asarray(self.normal_ned, float))

    def sigma_at(self, z):
        return self.sigma0 + self.sigma1 * np.asarray(z)


@dataclass
class ObstacleCloud:
    points_ned: np.ndarray           # (N,3) raised 3-D structure only
    height_above_floor: np.ndarray   # (N,)
    pixels_vu: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), int))  # full-res (v,u)
    floor_used: bool = False         # False -> no plane was available, cloud is empty
    n_below_floor: int = 0           # returns clearly below the floor (glossy-floor mirror images), dropped


@dataclass
class LandingTarget:
    found: bool
    center_ned: np.ndarray           # (3,) centre of the visible top face (z = top surface)
    top_d: float                     # NED z of the top face
    extent_m: float                  # long horizontal side of the visible top face (m)
    confidence: float                # 0..1
    extent_xy_m: np.ndarray = field(default_factory=lambda: np.array([math.nan, math.nan]))  # (long, short)
    yaw_ned: float = math.nan        # direction of the long side (rad, NED yaw, mod pi)
    height_above_floor: float = math.nan
    range_m: float = math.nan        # horizontal distance camera -> centre
    truncated: bool = False          # top face touches the left/right/bottom image border
    n_top_points: int = 0
    reason: str = ''                 # found=False: rejection of the candidate that got furthest
    factors: dict = field(default_factory=dict)   # confidence factors (debug / logging)
    kind: str = ''                   # 'pad' (flat landing pad) | 'block' (raised white block) | '' (none)


_STAGES = ('nothing_raised', 'no_bright_raised_blob', 'too_low', 'too_tall', 'top_not_visible',
           'too_few_top_points', 'top_not_bright', 'outside_prior', 'no_flat_top', 'bad_size',
           'top_not_level', 'low_confidence')


def _no_target(reason):
    return LandingTarget(False, np.array(_NAN3), math.nan, math.nan, 0.0, reason=reason)


# --------------------------------------------------------------------------- shared helpers
def _dot3(a, v):
    """a (...,3) . v (3,) element-wise (no BLAS)."""
    v0, v1, v2 = (float(x) for x in v)
    return a[..., 0] * v0 + a[..., 1] * v1 + a[..., 2] * v2


def _rotate(a, R):
    """Row vectors a (...,3) -> (R @ a.T).T element-wise (no BLAS)."""
    return np.stack([_dot3(a, R[0]), _dot3(a, R[1]), _dot3(a, R[2])], -1)


def _solve_ls(cols, y, w=None, ridge=None, ridge_mean=None):
    """Weighted linear least squares via normal equations: minimise
    sum w^2 (sum_k b_k cols_k - y)^2 + sum_k ridge_k (b_k - ridge_mean_k)^2. Tiny k only."""
    k = len(cols)
    ww = np.ones_like(y) if w is None else w * w
    A = np.empty((k, k)); b = np.empty(k)
    for i in range(k):
        wi = ww * cols[i]
        b[i] = float(np.sum(wi * y))
        for j in range(i, k):
            A[i, j] = A[j, i] = float(np.sum(wi * cols[j]))
    if ridge is not None:
        ridge = np.asarray(ridge, float)
        Rm = ridge if ridge.ndim == 2 else np.diag(ridge)          # full or diagonal penalty
        A += Rm
        if ridge_mean is not None:
            b += Rm @ np.asarray(ridge_mean, float)
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


def _pca2(xy):
    """Mean, (var_long, var_short), long axis, short axis of 2-D points (closed form)."""
    mu = xy.mean(0)
    cx, cy = xy[:, 0] - mu[0], xy[:, 1] - mu[1]
    sxx, syy, sxy = float(np.mean(cx * cx)), float(np.mean(cy * cy)), float(np.mean(cx * cy))
    half_tr = 0.5 * (sxx + syy)
    disc = math.sqrt(max(half_tr * half_tr - (sxx * syy - sxy * sxy), 0.0))
    th = 0.5 * math.atan2(2 * sxy, sxx - syy)
    return mu, (half_tr + disc, max(half_tr - disc, 0.0)), np.array([math.cos(th), math.sin(th)]), \
        np.array([-math.sin(th), math.cos(th)])


@lru_cache(maxsize=16)
def _rays(h, w, stride, K):
    fx, fy, cx, cy = K
    v = np.arange(0, h, stride, dtype=np.float32)
    u = np.arange(0, w, stride, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    r = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], -1)
    r.setflags(write=False)
    return r


class _Grid:
    """Strided depth grid of one frame: optical depth, validity, NED points on demand."""

    def __init__(self, frame, stride, max_range, min_range=0.15, far_limit=20.0, keep_xy=None):
        dep = np.asarray(frame.depth, np.float32)
        self.H, self.W = dep.shape
        self.stride = stride
        self.K = tuple(float(k) for k in frame.K)
        self.rays = _rays(self.H, self.W, stride, self.K)
        z = dep[::stride, ::stride]
        with np.errstate(invalid='ignore'):
            self.any = np.isfinite(z) & (z > min_range) & (z < far_limit)   # for edge tests
            self.valid = self.any & (z < max_range)                         # for outputs
        self.z = np.where(self.any, z, 0).astype(np.float32)
        self.R = np.asarray(frame.R_nc, np.float64)
        self.cam = np.asarray(frame.cam_ned, np.float64).reshape(3)
        self._frame = frame
        self._pall = None
        if keep_xy is not None:                 # arena geofence: depth outside = no depth
            x0, x1, y0, y1 = (float(v) for v in keep_xy)
            P = self.points_all()
            inside = (P[..., 0] >= x0) & (P[..., 0] <= x1) & (P[..., 1] >= y0) & (P[..., 1] <= y1)
            self.any &= inside
            self.valid &= inside

    def points(self, mask):
        """NED points (N,3) float64 of grid cells where mask is True."""
        if self._pall is not None:
            return self._pall[mask].astype(np.float64)
        pc = (self.rays[mask] * self.z[mask][:, None]).astype(np.float64)
        return _rotate(pc, self.R) + self.cam

    def points_all(self):
        """(Hs,Ws,3) float32 NED points of every grid cell (camera position where no depth)."""
        if self._pall is None:
            p = _rotate(self.rays * self.z[..., None], self.R)
            p += self.cam.astype(np.float32)
            self._pall = p
        return self._pall

    def heights(self, floor):
        """Height above `floor` for every grid cell (garbage where invalid)."""
        n = np.asarray(floor.normal_ned, np.float64)
        m = [float(np.dot(self.R[:, j], n)) for j in range(3)]          # R^T n
        c0 = float(np.dot(n, np.asarray(floor.anchor_ned, np.float64) - self.cam))
        return c0 - _dot3(self.rays, m) * self.z

    def vu(self, mask):
        vv, uu = np.nonzero(mask)
        return np.stack([vv * self.stride, uu * self.stride], -1)

    def grey(self):
        """(grey, saturation or None) sampled on the grid."""
        img = np.asarray(self._frame.image)
        sat = None
        if img.ndim == 3:
            if img.shape[:2] != (self.H, self.W):
                img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
            sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[::self.stride, ::self.stride, 1]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[:2] != (self.H, self.W):
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        return img[::self.stride, ::self.stride], sat


def _shift(a, k, axis, fill):
    """out[i] = a[i - k] along axis (k may be negative); vacated cells = fill."""
    out = np.full_like(a, fill)
    src = [slice(None)] * a.ndim; dst = [slice(None)] * a.ndim
    if k > 0:
        src[axis] = slice(0, -k); dst[axis] = slice(k, None)
    else:
        src[axis] = slice(-k, None); dst[axis] = slice(0, k)
    out[tuple(dst)] = a[tuple(src)]
    return out


def _neighbour_jumps(z, valid, rel, abs_, reach=2):
    """Depth jumps to the nearest valid neighbour (within `reach` cells) on each side:
    (left, right, up, down) booleans 'jump > rel*z + abs'. Holes next to mixed pixels
    are common, hence the reach."""
    tol = rel * z + abs_
    out = []
    for axis, sgn in ((1, 1), (1, -1), (0, 1), (0, -1)):
        zn = np.zeros_like(z); vn = np.zeros_like(valid)
        for k in range(reach, 0, -1):               # nearer neighbours override
            vk = _shift(valid, sgn * k, axis, False)
            zn = np.where(vk, _shift(z, sgn * k, axis, 0), zn); vn |= vk
        out.append(valid & vn & (np.abs(z - zn) > tol))
    return out


def _flying_pixels(z, valid, rel=0.08, abs_=0.03):
    """Mixed/flying pixels: depth differs a lot from BOTH neighbours along an axis."""
    l, r, u, d = _neighbour_jumps(z, valid, rel, abs_)
    return valid & ((l & r) | (u & d))


def _vertical_surface(Pg, valid, ks=(1, 2)):
    """Grid points whose neighbour k grid rows higher in the image (any k in ks) is
    nearly straight above them (|dz| > horizontal offset): walls, panels, rock flanks,
    block faces and mirror images of walls in a glossy floor. Floor points have that
    neighbour far behind them instead. Longer baselines beat stereo noise at range."""
    out = np.zeros(valid.shape, bool)
    for k in ks:
        a, b = Pg[k:], Pg[:-k]
        ok = valid[k:] & valid[:-k]
        dz = np.abs(a[..., 2] - b[..., 2])
        dxy = np.hypot(a[..., 0] - b[..., 0], a[..., 1] - b[..., 1])
        out[k:] |= ok & (dz > dxy)
    return out


def _cell_ids(p, cam, cell, half):
    n = int(math.ceil(2 * half / cell))
    ix = np.clip(((p[:, 0] - cam[0] + half) / cell).astype(np.int64), 0, n - 1)
    iy = np.clip(((p[:, 1] - cam[1] + half) / cell).astype(np.int64), 0, n - 1)
    return ix * n + iy, n * n


def _minor_std(xy):
    if len(xy) < 3:
        return 0.0
    return math.sqrt(_pca2(xy)[1][1])


def brightness_gate(grey, floor_ref, sat=None, *, bright_min=90.0, bright_delta=25.0,
                    bright_ratio=1.25, max_sat=70):
    """'White / light-grey' pixel test, relative to the floor brightness of the same
    image (auto-exposure invariant). Returns (mask, threshold)."""
    ref = float(floor_ref) if np.isfinite(floor_ref) else float(np.median(grey))
    # capped below saturation: with a bright floor (auto-exposure / gain up) 1.25 x floor
    # would exceed 255 and nothing could pass; a clipped white top still does
    thr = min(max(bright_min, ref + bright_delta, ref * bright_ratio), 245.0)
    m = np.asarray(grey) >= thr
    if sat is not None:
        m &= np.asarray(sat) <= max_sat
    return m, thr


# --------------------------------------------------------------------------- floor
def _fit_plane(P, Z, cells, ncell, cam, tau0, tau1, n, c, max_tilt_cos, iters=3, slope_sigma=0.0175,
               prior_d=None, prior_sigma=0.03):
    """Re-select inliers and weighted-LS fit z = a + b*dx + e*dy.

    Stereo noise is spatially correlated, so the data is weighted as if each 10 cm
    XY cell were one independent sample (weights * sqrt(cells / points)); a 1 deg
    level prior on the slope then matters when the floor is seen only far away or
    sparsely (the slope would otherwise be extrapolated ~3 m back to the camera),
    while a well-observed attitude/extrinsic tilt still dominates. `prior_d`
    (ToF / last fit) is a soft observation of a = floor z under the camera.
    Returns (n, c, a, inl) or None."""
    dx = P[:, 0] - cam[0]; dy = P[:, 1] - cam[1]
    tau = tau0 + tau1 * Z
    w0 = 2.5 / tau                                  # tau ~ 2.5 sigma
    mean = np.array([0.0 if prior_d is None else prior_d, 0.0, 0.0])
    inl = None
    for _ in range(iters):
        inl = np.abs(_dot3(P, n) - c) < tau
        n_in = int(inl.sum())
        if n_in < 8:
            return None
        n_cells = int(np.count_nonzero(np.bincount(cells[inl], minlength=ncell)))
        w = w0[inl] * math.sqrt(n_cells / n_in)
        # level prior on the slope, per direction, tightened when the inliers barely extend
        # along it compared with their distance from the camera: a far narrow strip (floor
        # along the cage edge, with the net's base rail in it) cannot measure a tilt, and a
        # wrong tilt is extrapolated to the point under the camera
        ix, iy = dx[inl], dy[inl]
        mx, my = float(ix.mean()), float(iy.mean())
        dist = max(math.hypot(mx, my), 0.3)
        ur = np.array([mx, my]) / dist if dist > 0.3 else np.array([1.0, 0.0])
        ul = np.array([-ur[1], ur[0]])
        R2 = np.zeros((2, 2))
        for u_ in (ur, ul):
            spread = float(np.std((ix - mx) * u_[0] + (iy - my) * u_[1]))
            sig = slope_sigma * min(1.0, max(0.1, 4.0 * spread / dist))
            R2 += np.outer(u_, u_) / sig ** 2
        ridge = np.zeros((3, 3)); ridge[1:, 1:] = R2
        if prior_d is not None:
            ridge[0, 0] = prior_sigma ** -2
        sol = _solve_ls([np.ones(n_in), dx[inl], dy[inl]], P[inl, 2], w, ridge, mean)
        if sol is None:
            return None
        a, b, e = (float(x) for x in sol)
        n = np.array([-b, -e, 1.0]); n /= math.sqrt(b * b + e * e + 1.0)
        if n[2] < max_tilt_cos:
            return None
        c = float(n[0] * cam[0] + n[1] * cam[1] + n[2] * a)
    inl = np.abs(_dot3(P, n) - c) < tau
    return n, c, a, inl


def fit_floor(frame, prior_floor_d=None, *, stride=4, max_range=5.0, prior_window_m=0.15,
              band_m=(0.04, 2.6), max_tilt_deg=6.0, min_inliers=30, min_cells=15,
              min_spread_m=0.04, n_hyp=160, max_samples=3000, cell_m=0.10, prior_sigma_m=0.03,
              stereo_sigma_frac=0.05, keep_xy=None, seed=0):
    """Dominant floor plane from depth.

    Hypotheses: half are horizontal planes through one random point (robust when
    the floor is a small, sparse fraction of the depth -- glossy floor, mostly wall),
    half are 3-point planes within `max_tilt_deg` of NED-down (absorb attitude error).
    Score = number of distinct 10 cm XY cells covered by inliers, so a horizontal
    slice through a wall (a 1-D line in XY) cannot beat a 2-D floor patch.
    With `prior_floor_d` (ToF or last fit) only points within `prior_window_m`
    (+1.5 deg tilt allowance, i.e. +2.6 cm per metre of range) of it are considered;
    if nothing is confirmed there the result is found=False with d = prior
    (source 'prior'). Pass None to search the whole band below the camera.

    Two floor heights: the plane (anchor_ned / normal_ned, `d_stereo`) is fitted to
    the stereo returns only, so heights of other stereo points relative to it are
    self-consistent; `d` is the best METRIC floor z under the camera = inverse-variance
    fusion of d_stereo (sigma = 1 cm + `stereo_sigma_frac` x camera height) with the
    prior (sigma `prior_sigma_m`). This is a weighting, not a correction: nothing is
    shifted. The height-proportional term models that from higher up the floor is
    seen only far away at grazing angles, where a few outliers (SGBM mismatches next
    to the net in the venue replica) lift or tilt the plane: there d_stereo p95 was
    8 cm vs 2.8 cm fused. Without a prior, d = d_stereo.
    Prior: ToF-derived floor z when the ToF sees the floor (not over a rock / the
    block); otherwise the last good d. Chaining d alone lets stereo errors creep in.
    keep_xy = (n_min, n_max, e_min, e_max): arena box in NED; depth outside it (the
    hall seen through the net, the net itself) is ignored.
    """
    prior = None if prior_floor_d is None or not np.isfinite(prior_floor_d) else float(prior_floor_d)
    g = _Grid(frame, stride, max_range, keep_xy=keep_xy)
    cam = g.cam
    max_tilt_cos = math.cos(math.radians(max_tilt_deg))

    def fail(reason):
        if prior is None:
            return FloorFit(False, math.nan, _DOWN.copy(), 0.0, reason=reason)
        return FloorFit(False, prior, _DOWN.copy(), 0.0, anchor_ned=np.array([cam[0], cam[1], prior]),
                        source='prior', reason=reason)

    valid_flat = np.flatnonzero(g.valid.ravel())
    n_valid = len(valid_flat)
    if n_valid < min_inliers:
        return fail('no_depth')
    Pg = g.points_all()
    vertical = _vertical_surface(Pg, g.any, ks=(1, 2, 4))[g.valid]
    P = Pg[g.valid].astype(np.float64)
    Z = g.z[g.valid].astype(np.float64)
    if prior is not None:
        cand = np.abs(P[:, 2] - prior) < prior_window_m + Z * math.tan(math.radians(1.5))
    else:
        dz = P[:, 2] - cam[2]
        cand = (dz > band_m[0]) & (dz < band_m[1])
    cand &= ~vertical
    ci = np.flatnonzero(cand)
    if len(ci) < min_inliers:
        return fail('no_floor_candidates')
    Pc, Zc = P[ci], Z[ci]
    cells_c, ncell = _cell_ids(Pc, cam, cell_m, max_range)

    rng = np.random.default_rng(seed)
    M = min(len(ci), max_samples)
    sub = rng.choice(len(ci), M, replace=False) if len(ci) > M else np.arange(len(ci))
    S, SZ, Scell = Pc[sub], Zc[sub], cells_c[sub]

    n1 = n_hyp // 2
    Nh = [np.tile(_DOWN, (n1, 1))]
    Ch = [S[rng.integers(0, M, n1), 2]]
    if prior is not None:
        Nh.append(_DOWN[None]); Ch.append(np.array([prior]))
    tri = rng.integers(0, M, (n_hyp - n1, 3))
    nn = np.cross(S[tri[:, 1]] - S[tri[:, 0]], S[tri[:, 2]] - S[tri[:, 0]])
    nrm = np.linalg.norm(nn, axis=1)
    ok = nrm > 1e-9
    nn = nn[ok] / nrm[ok, None]
    nn *= np.sign(nn[:, 2:3] + 1e-12)
    keep = nn[:, 2] > max_tilt_cos
    Nh.append(nn[keep]); Ch.append(np.sum(nn[keep] * S[tri[ok][keep, 0]], 1))
    Nh = np.vstack(Nh); Ch = np.concatenate(Ch)

    order = np.argsort(Scell, kind='stable')                            # group samples by XY cell
    S, SZ, Scell = S[order], SZ[order], Scell[order]
    starts = np.flatnonzero(np.r_[True, Scell[1:] != Scell[:-1]])
    dist = S[:, 0:1] * Nh[:, 0] + S[:, 1:2] * Nh[:, 1] + S[:, 2:3] * Nh[:, 2] - Ch
    inl = np.abs(dist) < (0.03 + 0.02 * SZ)[:, None]                     # (M, H)
    score = np.logical_or.reduceat(inl, starts, axis=0).sum(0) + 1e-3 * inl.sum(0)

    # refine the best few distinct hypotheses on all candidates
    z_at_cam = (Ch - Nh[:, 0] * cam[0] - Nh[:, 1] * cam[1]) / Nh[:, 2]
    picked = []
    for h in np.argsort(-score):
        if len(picked) >= 3:
            break
        if all(abs(z_at_cam[h] - z_at_cam[k]) > 0.04 for k in picked):
            picked.append(h)
    best = None
    for h in picked:
        r = _fit_plane(Pc, Zc, cells_c, ncell, cam, 0.03, 0.02, Nh[h], Ch[h], max_tilt_cos, iters=2)
        if r is None:
            continue
        r = _fit_plane(Pc, Zc, cells_c, ncell, cam, 0.02, 0.008, r[0], r[1], max_tilt_cos, iters=2)
        if r is None:
            continue
        # tighter last stage: low flat tops right next to the floor (5-7 cm frame beams,
        # rock feet) otherwise stay inside the band and lift a sparsely seen floor
        r2 = _fit_plane(Pc, Zc, cells_c, ncell, cam, 0.015, 0.006, r[0], r[1], max_tilt_cos, iters=2)
        if r2 is not None and r2[3].sum() >= min_inliers:
            r = r2
        n, c, a, il = r
        n_in = int(il.sum())
        n_cells = int(np.count_nonzero(np.bincount(cells_c[il], minlength=ncell)))
        spread = _minor_std(Pc[il, :2])
        if n_in < min_inliers or n_cells < min_cells or spread < min_spread_m or a - cam[2] < 0.04:
            continue
        # with a prior the plane must also agree with it UNDER the camera (window + 2 sigma of the
        # stereo d), not only inside the window at range: a tilted plane through a far strip can
        # otherwise extrapolate tens of cm off
        if prior is not None and abs(a - prior) > prior_window_m + 2 * stereo_sigma_frac * max(prior - cam[2], 0.1):
            continue
        if best is None or n_cells > best[4]:
            best = (n, c, a, il, n_cells)
    if best is None:
        return fail('no_plane')
    n, c, a, il, n_cells = best

    # object bases: candidates 1-4 grid rows straight below a point clearly above the
    # plane at the same depth are rock/wall/frame feet, not floor (they matter when the
    # glossy floor is sparse but objects are dense) -> refit once without them
    m = [float(np.dot(g.R[:, j], n)) for j in range(3)]
    hg = (c - float(np.dot(n, cam))) - _dot3(g.rays, m) * g.z
    raised = g.any & (hg > 0.08)
    base = np.zeros_like(raised)
    for k in (1, 2, 3, 4):
        base |= _shift(raised, k, 0, False) & (np.abs(g.z - _shift(g.z, k, 0, 0)) < 0.1 * g.z + 0.05)
    keep = ~base[g.valid][ci]
    if (~keep).any() and keep.sum() >= min_inliers:
        r = _fit_plane(Pc[keep], Zc[keep], cells_c[keep], ncell, cam, 0.015, 0.006, n, c, max_tilt_cos, iters=2)
        if r is not None:
            il_k = np.zeros(len(ci), bool); il_k[np.flatnonzero(keep)[r[3]]] = True
            nc_k = int(np.count_nonzero(np.bincount(cells_c[il_k], minlength=ncell)))
            if il_k.sum() >= min_inliers and nc_k >= min_cells:
                n, c, a, il, n_cells = r[0], r[1], r[2], il_k, nc_k

    # noise model sigma(z) = s0 + s1*z from the residual MAD of all candidates in a wide band
    # (6 cm + 3 %), per range bin: tracks the real depth quality (SGM on a glossy floor is
    # far worse than the fitting band assumes) without being truncated by that band
    res = np.abs(_dot3(Pc, n) - c)
    wide = res < 0.06 + 0.03 * Zc
    zc_, sg_, wt_ = [], [], []
    min_bin = max(30, int(0.05 * wide.sum()))       # small bins are mostly object feet, not floor
    for lo, hi in ((0.0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 99.0)):
        mb = wide & (Zc >= lo) & (Zc < hi)
        if mb.sum() >= min_bin:
            zc_.append(float(np.median(Zc[mb]))); sg_.append(1.4826 * float(np.median(res[mb]))); wt_.append(float(mb.sum()))
    s0, s1 = 0.02, 0.008
    if len(zc_) >= 2 and max(zc_) - min(zc_) > 0.5:
        sol = _solve_ls([np.ones(len(zc_)), np.array(zc_)], np.array(sg_), np.sqrt(wt_))
        if sol is not None:
            s0, s1 = float(sol[0]), float(sol[1])
    if not zc_ or s1 < 0 or len(zc_) < 2:            # flat model: weighted median of the bins
        if zc_:
            o = np.argsort(sg_); cw = np.cumsum(np.array(wt_)[o])
            s0, s1 = float(np.array(sg_)[o][np.searchsorted(cw, cw[-1] / 2)]), 0.0
    s0 = float(np.clip(s0, 0.004, 0.08)); s1 = float(np.clip(s1, 0.0, 0.05))

    grey, _ = g.grey()
    fb = float(np.median(grey.ravel()[valid_flat[ci[il]]]))
    d_metric, source = float(a), 'depth'
    if prior is not None:                            # metric d: fuse with the prior (ToF)
        var_s = (0.01 + stereo_sigma_frac * max(float(a) - cam[2], 0.1)) ** 2
        var_p = prior_sigma_m ** 2
        d_metric = (float(a) * var_p + prior * var_s) / (var_p + var_s)
        source = 'depth+prior'
    return FloorFit(True, d_metric, n, float(il.sum()) / n_valid,
                    anchor_ned=np.array([cam[0], cam[1], a]), n_inliers=int(il.sum()), n_cells=n_cells,
                    tilt_deg=math.degrees(math.acos(min(1.0, n[2]))), sigma0=s0, sigma1=s1,
                    floor_brightness=fb, source=source, d_stereo=float(a))


# --------------------------------------------------------------------------- obstacles
def obstacles(frame, floor, min_height_m=0.05, max_range=4.0, *, stride=2, k_sigma=4.0,
              min_component_px=3, keep_xy=None):
    """Only 3-D structure above the floor: real rocks, frame bases, panels, walls.

    A point is kept when its height above the fitted plane exceeds
    max(min_height_m, k_sigma * sigma(z)) (painted rocks / floor print sit at h~0),
    it is not a flying pixel at a depth edge, and its raised blob has at least
    `min_component_px` grid pixels (speckle). Points far below the floor are glossy
    floor mirror images and are dropped (counted in n_below_floor).
    """
    if floor is None or not np.isfinite(getattr(floor, 'd', math.nan)):
        return ObstacleCloud(np.zeros((0, 3)), np.zeros(0))
    g = _Grid(frame, stride, max_range, keep_xy=keep_xy)
    valid = g.valid & ~_flying_pixels(g.z, g.any)
    h = g.heights(floor)
    thr = np.maximum(min_height_m, k_sigma * floor.sigma_at(g.z))
    raised = valid & (h > thr)
    below = valid & (h < -thr)
    if raised.any() and min_component_px > 1:
        nlab, lab, stats, _ = cv2.connectedComponentsWithStats(raised.astype(np.uint8), connectivity=8)
        good = stats[:, cv2.CC_STAT_AREA] >= min_component_px
        good[0] = False
        raised = good[lab]
    return ObstacleCloud(g.points(raised), h[raised].astype(np.float64), g.vu(raised), True,
                         int(below.sum()))


# --------------------------------------------------------------------------- landing target
def landing_block(frame, floor, *, stride=2, max_range=4.5, min_top_h=0.10, max_top_h=1.0,
                   min_extent_m=0.25, min_minor_m=0.15, max_extent_m=2.0, disp_noise_px=0.15,
                   min_area_px=12, min_top_points=10, min_bright_top=0.6, min_confidence=0.25,
                   prior_center_ned=None, prior_radius_m=1.0, bright_mode='gate', keep_xy=None,
                   hole_fill='auto', hole_min_px=40, hole_min_frac=0.3, hole_fill_factor=0.85, hole_floor_win=5,
                   hole_max_extent_m=1.5,
                   **bright_kw):
    """Large white / light-grey block standing on the floor with a flat, visible top.

    Candidates are depth-continuous raised blobs (no depth jump inside, h above the
    noise threshold). A candidate is accepted only if
      * its top face (points within a thin band below the blob's top height) is
        mostly bright relative to the floor of the same image (and unsaturated if
        BGR; bright depth holes inside the top's image box count, since an
        over-exposed white top has no stereo texture), and min_top_h <= top height
        <= max_top_h, camera above the top;
      * the top face is a 2-D horizontal patch: its radial depth extent exceeds what
        stereo noise alone gives a vertical edge (4.5 sigma_z + 5 cm, sigma_z =
        disp_noise_px * z^2 / (fx * B)), and long/short sides are within limits.
    Painted floor patches fail "raised"; bright walls/panels fail "flat top".

    hole_fill ('auto' = on when prior_center_ned is given): stereo (SGM) gives a
    textureless / over-exposed white top
    almost no depth, only a thin raised rim, often a single edge. For a raised blob
    whose heights peak at one level (>= 35 % of it within 5 cm, >= 5 points; a wall or
    board does not peak), the top region is grown from that rim through depth holes,
    except holes with valid floor-level pixels around them (`hole_floor_win` grid window); it is
    projected onto the horizontal plane at the rim height (<= 1 m from the rim) and goes
    through the same size / prior / confidence checks (x 0.85, factors['kind'] =
    'hole_fill', long side <= hole_max_extent_m). The rim must be a straight edge
    (>= ~0.12 m) with >= 85 % of the filled top on one side of it: a frame bar at a fixed
    height over a featureless (hole) floor has holes on both sides. Paint has no raised
    rim; panels / boards have spread rim heights. Without a position prior a featureless
    floor next to some raised edge can still pass, hence 'auto'.

    bright_mode='soft': brightness only scales the confidence (>= 0.3) instead of
    gating. For a mono sensor whose response makes the orange floor as bright as
    the grey-white stone (red/NIR-heavy): then any raised flat top of the right
    size qualifies, so use it together with prior_center_ned.
    """
    if floor is None or not np.isfinite(getattr(floor, 'd', math.nan)):
        return _no_target('no_floor')
    if hole_fill == 'auto':                      # a fill-in at a known place: only with a position prior
        hole_fill = prior_center_ned is not None
    g = _Grid(frame, stride, max_range, keep_xy=keep_xy)
    h = g.heights(floor)
    sig = floor.sigma_at(g.z)
    # the raised mask only has to separate the block from the floor (brightness, flat top and
    # size do the discriminating), so it stays low even when the floor depth is very noisy
    thr = np.clip(2.0 * sig, 0.05, 0.08)
    l, r, u, d = _neighbour_jumps(g.z, g.any, 0.10, 0.03)
    edge = l | r | u | d
    raised = g.valid & ~edge & (h > thr)
    grey, sat = g.grey()
    grey = grey.astype(np.float32)

    on_floor = g.valid & (np.abs(h) < np.maximum(0.03, 2.0 * sig))
    if on_floor.sum() >= 50:
        floor_ref = float(np.median(grey[on_floor]))
    elif np.isfinite(floor.floor_brightness):
        floor_ref = float(floor.floor_brightness)
    else:
        floor_ref = float(np.median(grey[g.valid])) if g.valid.any() else math.nan
    bright, _ = brightness_gate(grey, floor_ref, sat, **bright_kw)

    if not raised.any():
        return _no_target('nothing_raised')
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(raised.astype(np.uint8), connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA]
    nbright = np.bincount(lab[raised], weights=bright[raised], minlength=nlab)
    m1 = cv2.blur(grey, (5, 5)); m2 = cv2.blur(grey * grey, (5, 5))
    lowtex = np.sqrt(np.maximum(m2 - m1 * m1, 0)) < 20.0

    Pg = g.points_all()
    vertical = _vertical_surface(Pg, g.any, ks=(2, 4))    # front/side faces of the block, walls
    hc = float(floor.height_above(g.cam))          # camera height above floor
    fx = g.K[0]
    Hs, Ws = g.z.shape
    best, best_reason = None, 'no_bright_raised_blob'

    def note(reason):
        nonlocal best_reason
        if _STAGES.index(reason) > _STAGES.index(best_reason):
            best_reason = reason

    def score(P, h_top, top_grey, z_med, rows, cols, n_top, grounded, slope, tex_frac, kind, f_kind, extra):
        """Shared acceptance + confidence for a top-face point set P (NED). Returns a LandingTarget or None."""
        if prior_center_ned is not None:
            pc = np.asarray(prior_center_ned, float)
            if math.hypot(*(np.median(P[:, :2], 0) - pc[:2])) > prior_radius_m:
                note('outside_prior'); return None
        rad = np.hypot(P[:, 0] - g.cam[0], P[:, 1] - g.cam[1])
        r2, r98 = np.percentile(rad, [2, 98])
        sigma_z = disp_noise_px * z_med * z_med / (fx * OAKD_BASELINE_M)
        need = max(min_minor_m, 4.5 * sigma_z + 0.05) if kind == 'depth' else min_minor_m
        if r98 - r2 < need:
            note('no_flat_top'); return None
        xy = P[:, :2]
        mu, _, ax_long, ax_short = _pca2(xy)
        cx_, cy_ = xy[:, 0] - mu[0], xy[:, 1] - mu[1]
        s_l = cx_ * ax_long[0] + cy_ * ax_long[1]; s_s = cx_ * ax_short[0] + cy_ * ax_short[1]
        l2, l98 = np.percentile(s_l, [2, 98]); q2, q98 = np.percentile(s_s, [2, 98])
        ext_l = (l98 - l2) / 0.96; ext_s = (q98 - q2) / 0.96
        if ext_l < min_extent_m or ext_s < min_minor_m or ext_l > (max_extent_m if kind == 'depth' else min(max_extent_m, hole_max_extent_m)):
            note('bad_size'); return None
        if slope > 15.0:
            note('top_not_level'); return None
        centre_xy = mu + ax_long * (l2 + l98) / 2 + ax_short * (q2 + q98) / 2
        truncated = int(((cols <= 0) | (cols >= Ws - 1) | (rows >= Hs - 1)).sum()) >= 3     # top face cut
        f_pts = min(1.0, max(0.2, n_top / 40.0))
        f_flat = min(1.0, (r98 - r2) / (1.5 * need))
        # contrast judged against the headroom the exposure leaves: a white top clipped at 255
        # over a 200-grey floor is as bright as the sensor can show it
        top_med = float(np.median(top_grey))
        if floor_ref >= 240.0 and top_med >= 245.0:  # over-exposed frame: contrast says nothing
            f_bright = 0.75
        else:
            need_c = min(60.0, max(15.0, 0.6 * (255.0 - floor_ref)))
            f_bright = min(1.0, max(0.2 if bright_mode == 'gate' else 0.3, (top_med - floor_ref) / need_c))
        f_range = min(1.0, max(0.4, 1.0 - 0.6 * max(0.0, z_med - 2.5) / max(max_range - 2.5, 1e-6)))
        f_tex = 0.7 + 0.3 * tex_frac
        # truncation mostly costs position accuracy (reported in `truncated`), not identity: a
        # mild factor, so a block whose near edge sits on the image bottom does not flicker
        conf = f_pts * f_flat * f_bright * f_range * f_tex * f_kind * (0.85 if truncated else 1.0) * (1.0 if grounded else 0.5)
        if conf < min_confidence:
            note('low_confidence'); return None
        # level-floor assumption: the fitted tilt is attitude error, so measure the top
        # from the floor under the camera instead of using the points' raw NED z
        top_z = float(floor.d) - h_top
        return LandingTarget(True, np.array([centre_xy[0], centre_xy[1], top_z]), top_z, float(ext_l),
                             float(conf), extent_xy_m=np.array([ext_l, ext_s]),
                             yaw_ned=float(math.atan2(ax_long[1], ax_long[0]) % math.pi),
                             height_above_floor=h_top,
                             range_m=float(math.hypot(centre_xy[0] - g.cam[0], centre_xy[1] - g.cam[1])),
                             truncated=bool(truncated), n_top_points=int(n_top),
                             factors=dict(pts=f_pts, flat=f_flat, bright=f_bright, range=f_range, tex=f_tex,
                                          grounded=grounded, floor_ref=floor_ref, kind=kind, kind_factor=f_kind,
                                          radial_m=float(r98 - r2), radial_need_m=need, slope_deg=slope, **extra))

    for k in range(1, nlab):
        if area[k] < min_area_px:
            continue
        x0, y0, bw, bh = stats[k, :4]
        sl = (slice(y0, y0 + bh), slice(x0, x0 + bw))         # work inside the blob's bbox
        # cheap pre-filter: enough bright pixels (with depth, or bright depth holes in the bbox)
        # for a top face at all. Not a fraction of the blob: from low altitude the dark front
        # face dominates the blob of a bright-topped block
        if bright_mode == 'gate' and nbright[k] + int((bright[sl] & ~g.any[sl]).sum()) < min_top_points:
            continue
        mk = lab[sl] == k
        hk = h[sl][mk].astype(np.float64)
        h_top = float(np.percentile(hk, 97))
        if h_top < min_top_h:
            note('too_low'); continue
        if h_top > max_top_h:
            note('too_tall'); continue
        if hc - h_top < 0.04:
            note('top_not_visible'); continue
        zk = g.z[sl][mk].astype(np.float64)
        band = 0.02 + 0.006 * float(np.median(zk))
        top = (hk >= h_top - band) & ~vertical[sl][mk]
        if top.sum() < min_top_points:
            note('too_few_top_points'); continue
        btop = bright[sl][mk][top]
        # an over-exposed white top is textureless -> stereo holes exactly on its brightest
        # part; bright no-depth pixels inside the top face's image box count as bright top
        rows, cols = np.nonzero(mk)
        rows, cols = rows[top] + y0, cols[top] + x0
        bb = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
        bright_holes = ~g.any[bb] & bright[bb]
        n_bh = int(bright_holes.sum())
        top_grey = np.concatenate([grey[sl][mk][top], grey[bb][bright_holes]])
        if bright_mode == 'gate' and (btop.sum() + n_bh) / (len(btop) + n_bh) < min_bright_top:
            note('top_not_bright'); continue
        P = Pg[sl][mk][top].astype(np.float64)
        xy = P[:, :2]; mu = xy.mean(0)
        coef = _solve_ls([np.ones(len(xy)), xy[:, 0] - mu[0], xy[:, 1] - mu[1]], hk[top], ridge=np.array([0.0, 1e-6, 1e-6]))
        slope = 90.0 if coef is None else math.degrees(math.atan(math.hypot(coef[1], coef[2])))
        grounded = float(hk.min()) <= max(0.15, float(np.max(thr[sl][mk])) + 0.05) or y0 + bh >= Hs
        cand = score(P, h_top, top_grey, float(np.median(zk[top])), rows, cols, int(top.sum()), grounded, slope,
                     float(lowtex[sl][mk][top].mean()), 'depth', 1.0, {})
        if cand is not None and (best is None or cand.confidence > best.confidence):
            best = cand

    # ---- hole-filled tops. SGM leaves a textureless / over-exposed top as a depth hole with
    # only a thin valid rim (often a single edge), so the raised blob above is just that rim
    # (bad_size / no_flat_top). Here: take the top height from a rim whose heights peak at one
    # level, grow the top region from it through depth holes, but not through holes that have
    # valid FLOOR-level pixels around them (a glossy/plain floor is a partial hole too), and
    # put the region on the horizontal plane at the rim height. Brightness is not needed to
    # outline the region (in the official-compat sim the top is as bright as the floor next
    # to it); it still gates in bright_mode='gate' and scales the confidence.
    if hole_fill:
        nvec = np.asarray(floor.normal_ned, np.float64)
        mrow = [float(np.dot(g.R[:, j], nvec)) for j in range(3)]           # R^T n
        c0 = float(np.dot(nvec, np.asarray(floor.anchor_ned, np.float64) - g.cam))
        hole = ~g.any
        k7 = np.ones((hole_floor_win, hole_floor_win), np.float32)
        grow = 60                                                            # grid px searched around a rim
        for k in range(1, nlab):
            if area[k] < 6:
                continue
            x0, y0, bw, bh = stats[k, :4]
            sl = (slice(y0, y0 + bh), slice(x0, x0 + bw))
            mk = lab[sl] == k
            hr = h[sl][mk].astype(np.float64)
            h_top = float(np.median(hr[hr >= np.percentile(hr, 50)]))         # upper cluster of the rim
            if h_top < min_top_h or h_top > max_top_h or hc - h_top < 0.04:
                continue
            rim_top = mk & (np.abs(h[sl] - h_top) <= 0.05)
            n_rim = int(rim_top.sum())
            if n_rim < 5 or n_rim < 0.35 * mk.sum():                       # heights must peak: a wall/board does not
                continue
            ya, yb, xa, xb = max(y0 - grow, 0), min(y0 + bh + grow, Hs), max(x0 - grow, 0), min(x0 + bw + grow, Ws)
            W_ = (slice(ya, yb), slice(xa, xb))
            hw, vw, hole_w = h[W_], g.valid[W_], hole[W_]
            band = 0.03 + 0.01 * hc
            floorlike = (vw & (hw < max(0.05, 0.4 * h_top))).astype(np.float32)
            toplike = vw & (np.abs(hw - h_top) <= band)
            nF = cv2.filter2D(floorlike, -1, k7, borderType=cv2.BORDER_CONSTANT)
            nT = cv2.filter2D(toplike.astype(np.float32), -1, k7, borderType=cv2.BORDER_CONSTANT)
            cand = (hole_w & ~((nF >= 2) & (nF > 2 * nT))) | toplike
            seed = np.zeros(cand.shape, bool); seed[y0 - ya:y0 - ya + bh, x0 - xa:x0 + bw - xa] = rim_top
            # brightness continuity: hole pixels must look like the holes right next to the rim
            # (a darker floor behind / beside the block stops the region; a floor as bright as
            # the top, as in the official-compat sim, does not)
            gw = grey[W_]
            near_rim = cv2.dilate(seed.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & cand & hole_w
            if near_rim.sum() >= 5:
                top_ref = float(np.median(gw[near_rim]))
                cand &= ~hole_w | (gw >= top_ref - max(25.0, 0.15 * top_ref))
            n2, lab2 = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
            ids = np.unique(lab2[seed & cand]); ids = ids[ids > 0]
            if not len(ids):
                continue
            region = np.isin(lab2, ids)
            n_hole = int((region & hole_w).sum())
            if n_hole < hole_min_px or n_hole < hole_min_frac * region.sum():
                continue                                        # nothing to fill: the depth path covers it
            rays = g.rays[W_][region].astype(np.float64)
            den = rays[:, 0] * mrow[0] + rays[:, 1] * mrow[1] + rays[:, 2] * mrow[2]
            with np.errstate(divide='ignore', invalid='ignore'):
                zt = (c0 - h_top) / den                             # optical depth of the plane h = h_top
            P = _rotate(rays * np.where(np.isfinite(zt), zt, 0)[:, None], g.R) + g.cam
            rim_xy = Pg[sl][rim_top][:, :2].astype(np.float64).mean(0)
            okz = (np.isfinite(zt) & (den > 1e-6) & (zt > 0.2) & (zt < max_range)
                   & (np.hypot(P[:, 0] - rim_xy[0], P[:, 1] - rim_xy[1]) <= 1.0))
            if okz.sum() < min_top_points:
                note('too_few_top_points'); continue
            P = P[okz]
            # a block top lies on ONE side of its edge: a frame bar / beam at a fixed height has
            # (featureless, bright) floor holes on both sides, a rock tip is not an edge at all
            rim_pts = Pg[sl][rim_top][:, :2].astype(np.float64)
            rmu, (rl1, rl2), rax, rnr = _pca2(rim_pts)
            if rl1 < 16.0 * max(rl2, 1e-6) or 4.0 * math.sqrt(rl1) < 0.12:
                note('no_flat_top'); continue                     # rim is not a straight edge >= ~0.12 m
            sd = (P[:, 0] - rmu[0]) * rnr[0] + (P[:, 1] - rmu[1]) * rnr[1]
            off = np.abs(sd) > 0.03
            if off.sum() < min_top_points or max((sd[off] > 0).mean(), (sd[off] < 0).mean()) < 0.85:
                note('no_flat_top'); continue
            rows, cols = np.nonzero(region)
            rows, cols = rows[okz] + ya, cols[okz] + xa
            top_grey = grey[W_][region][okz]
            if bright_mode == 'gate' and float(np.mean(bright[W_][region][okz])) < min_bright_top:
                note('top_not_bright'); continue
            grounded = float(hr.min()) <= max(0.15, float(np.max(thr[sl][mk])) + 0.05) or y0 + bh >= Hs
            cand_t = score(P, h_top, top_grey, float(np.median(zt[okz])), rows, cols, min(int(okz.sum()), 4 * n_rim),
                           grounded, 0.0, 1.0, 'hole_fill', hole_fill_factor,
                           dict(rim_points=n_rim, hole_frac=float(n_hole / region.sum())))
            if cand_t is not None and (best is None or cand_t.confidence > best.confidence):
                best = cand_t
    if best is not None:
        best.kind = 'block'
    return best if best is not None else _no_target(best_reason)


# --------------------------------------------------------------------------- flat landing pad
PAD_SIZE_M = 0.60


def landing_target(frame, floor, *, target='pad', **kw):
    """Final landing target. target='pad' (default since 2026-11-07: a flat ~0.6 m blue square
    pad with a dark H and 4 dark edge bars on the floor, see `landing_pad`), 'block' (raised
    white block, `landing_block`) or 'any' (both, higher confidence wins). Keyword arguments
    go to the chosen detector; ones it does not know are ignored, so the mission's existing
    call (prior_center_ned=..., bright_mode='soft') keeps working."""
    def call(fn):
        import inspect
        ok = inspect.signature(fn).parameters
        has_var = any(p.kind == p.VAR_KEYWORD for p in ok.values())
        return fn(frame, floor, **{k: v for k, v in kw.items() if has_var or k in ok})
    if target == 'pad':
        return call(landing_pad)
    if target == 'block':
        return call(landing_block)
    a, b = call(landing_pad), call(landing_block)
    if a.found and (not b.found or a.confidence >= b.confidence):
        return a
    return b if b.found else a


def _ncc_clean(x):
    """matchTemplate(TM_CCOEFF_NORMED) returns NaN / +-FLT_MAX where the window is constant
    (saturated glare: real flight 010807 frames 95-99 scored 3.4e38). Anything outside [-1, 1] -> 0."""
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x[np.abs(x) > 1.0001] = 0.0
    return x


def _pad_template(s_px, inner=0.90):
    """Interior pattern of the pad in pad units (side = s_px cells): body 1, H and the four
    edge bars 0 (the thin dashed white circle is below the patch resolution and left out).
    Proportions measured from the venue photos (H ~0.33 S tall, strokes ~0.065 S; bars
    0.2 S x 0.045 S, centred 0.39 S from the middle, parallel to their edge)."""
    n = int(round(s_px * inner)) | 1
    c = (n - 1) / 2.0
    u, v = np.meshgrid((np.arange(n) - c) / s_px, (np.arange(n) - c) / s_px)   # u: cols, v: rows
    T = np.ones((n, n), np.float32)
    hw, hh, sw = 0.12, 0.165, 0.033                              # H half width, half height, half stroke
    T[(np.abs(np.abs(u) - hw) <= sw) & (np.abs(v) <= hh)] = 0.0             # uprights (along v)
    T[(np.abs(u) <= hw) & (np.abs(v) <= sw)] = 0.0                          # crossbar
    bl, bw, bd = 0.10, 0.0225, 0.39                                         # bar half length, half width, offset
    for a, b in ((u, v), (v, u)):
        T[(np.abs(a) <= bl) & (np.abs(np.abs(b) - bd) <= bw)] = 0.0
    return T


def _occluded_holes(dep, zlim, max_px, min_gap=0):
    """Depth holes enclosed by occluders: stereo leaves the textureless face of a board / post
    empty and only its edges get depth. A hole pixel p is occluded when its nearest valid
    neighbours on both sides along the row, or both along the column (each within max_px), are
    nearer than zlim[p] - i.e. clearly in front of the floor point p would see. (Testing the
    neighbour against its own floor depth fails along columns: above the horizon everything is
    'in front of the floor', so a floor hole between a block below and the far hall above was
    masked.)"""
    valid = np.isfinite(dep) & (dep > 0.1)
    hole = ~valid
    out = np.zeros(dep.shape, bool)
    if not hole.any():
        return out
    d0 = np.where(valid, dep, np.inf)
    for axis in (1, 0):
        n = dep.shape[axis]
        ar = np.arange(n, dtype=np.int16).reshape((1, -1) if axis == 1 else (-1, 1))
        bef = np.maximum.accumulate(np.where(valid, ar, np.int16(-1)), axis=axis)
        aft = np.flip(np.minimum.accumulate(np.flip(np.where(valid, ar, np.int16(n)), axis=axis), axis=axis), axis=axis)
        ok = hole & (bef >= 0) & (aft < n) & (ar - bef <= max_px) & (aft - ar <= max_px) & (aft - bef > min_gap)
        db = np.take_along_axis(d0, np.clip(bef, 0, n - 1).astype(np.intp), axis)
        da = np.take_along_axis(d0, np.clip(aft, 0, n - 1).astype(np.intp), axis)
        out |= ok & (db < zlim) & (da < zlim)
    return out


def _occlusion_image(dep, K, R, cam, n, anc, rel=0.05, hole_px=100, hole_rel=0.10, hole_min=0):
    """Pixels whose depth is clearly nearer than the floor plane along their ray (something
    raised in front of the floor): depth < z_floor - max(0.08, rel z_floor). Plus depth holes
    bracketed by pixels clearly nearer than their floor point (< z_floor - max(0.15, hole_rel z_floor))
    within hole_px:
    SGBM leaves the textureless face of a board / post empty, and an unmasked board then smears
    across the top-down view (replica v3: stereo pad hits 5/11 -> 8/11 with this rule)."""
    H, W = dep.shape
    fx, fy, cx, cy = K
    nc = np.array([float(R[:, k] @ n) for k in range(3)])
    num = float(n @ (np.asarray(anc, float) - cam))
    u = (np.arange(W, dtype=np.float32) - cx) / fx; v = (np.arange(H, dtype=np.float32) - cy) / fy
    den = nc[0] * u[None, :] + nc[1] * v[:, None] + nc[2]
    with np.errstate(divide='ignore', invalid='ignore'):
        zfl = np.where(den > 1e-6, num / den, np.inf).astype(np.float32)
        occ = np.isfinite(dep) & (dep > 0.1) & (dep < zfl - np.maximum(0.08, rel * zfl))
    if hole_px > 0:
        with np.errstate(invalid='ignore'):
            occ = occ | _occluded_holes(dep, zfl - np.maximum(0.15, hole_rel * zfl), hole_px, hole_min)
    return occ


def landing_pad(frame, floor, *, prior_center_ned=None, prior_radius_m=1.0, pad_size_m=PAD_SIZE_M,
                res=0.02, search_fwd=(1.0, 4.0), search_lat=1.6, yaw_step_deg=5.0,
                min_confidence=0.35, max_range=4.5, keep_xy=None, sq_ncc_min=0.60, sq_margin_min=0.03, sq_support=1.2,
                occlusion='depth', occl_rel=0.05, occl_hole_px=100, occl_hole_rel=0.10, occl_hole_min=0, sq_inview_min=0.95, ipm_plane='stereo', debug=False, **_ignored):
    """Flat landing pad on the floor, from the MONO left image (depth cannot separate it).

    The floor plane (FloorFit: depth fit or prior-only) is used to resample the image onto a
    metric top-down patch (inverse perspective, `res` m per cell) around the position prior
    (+-(prior_radius_m + pad)) or, without one, over search_fwd x +-search_lat ahead of the
    camera. Cells whose depth is clearly nearer than the floor (rocks, cinder blocks, panels)
    and cells outside the image are invalid. Two polarity-free cues over yaw (0-180 deg) x
    position:
      * pattern: |normalised cross-correlation| of the pad interior (uniform body with a bold
        H and 4 edge bars) - works whether the body is lighter or darker than the H;
      * filled square: |mean inside the square - mean of the ring around it| in floor-noise
        units, with fill / ring purity checks - works when the body differs from the floor
        (either way) even if the H barely shows.
    Black tape outlines (inside = floor), painted rocks (not square, no H) and raised blocks
    (masked by depth) fail. Returns LandingTarget(kind='pad', top_d = floor z at the centre,
    height_above_floor = 0, extent_m = pad_size_m, yaw_ned of one edge mod 90 deg).

    Visibility: a level camera at height h sees the floor only beyond 2.12 h (OAK-D 640x400,
    fy 421, cy 200.7); the whole pad needs its near edge there, so the pad centre must be
    >= 2.12 h + S/2 ahead (h = 0.8 m -> 2.0 m).
    """
    if floor is None or not np.isfinite(getattr(floor, 'd', math.nan)):
        return _no_target('no_floor')
    K = tuple(float(k) for k in frame.K)
    img = np.asarray(frame.image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dep = np.asarray(frame.depth, np.float32)
    if img.shape[:2] != dep.shape:
        img = cv2.resize(img, (dep.shape[1], dep.shape[0]), interpolation=cv2.INTER_AREA)
    H_img, W_img = dep.shape
    R = np.asarray(frame.R_nc, np.float64); cam = np.asarray(frame.cam_ned, np.float64).reshape(3)
    n = np.asarray(floor.normal_ned, np.float64); anc = np.asarray(floor.anchor_ned, np.float64)
    if ipm_plane == 'metric' or not (np.all(np.isfinite(n)) and np.all(np.isfinite(anc))):
        # level plane at the metric floor z (ToF-fused d)
        n = np.array([0.0, 0.0, 1.0]); anc = np.array([cam[0], cam[1], float(floor.d)])
    fwd = np.array([R[0, 2], R[1, 2]]); fwd /= max(np.linalg.norm(fwd), 1e-9)
    lat = np.array([-fwd[1], fwd[0]])                                          # to the right
    S = float(pad_size_m)
    # ---- grid (a along fwd, b along lat), centred on the prior or a search box ahead
    if prior_center_ned is not None:
        pc = np.asarray(prior_center_ned, float)[:2]
        half = prior_radius_m + 0.6 * S
        a0 = float((pc - cam[:2]) @ fwd) - half; a1 = a0 + 2 * half
        bc = float((pc - cam[:2]) @ lat); b0 = bc - half; b1 = bc + half
    else:
        a0, a1 = search_fwd; b0, b1 = -search_lat, search_lat
    a0 = max(a0, 0.3)
    na, nb = int(round((a1 - a0) / res)), int(round((b1 - b0) / res))
    if na < 10 or nb < 10:
        return _no_target('outside_view')
    A = a1 - (np.arange(na) + 0.5) * res                       # rows: far (top) -> near (bottom)
    Bv = b0 + (np.arange(nb) + 0.5) * res
    AA, BB = np.meshgrid(A, Bv, indexing='ij')
    X = cam[0] + AA * fwd[0] + BB * lat[0]; Y = cam[1] + AA * fwd[1] + BB * lat[1]
    Z = (float(n @ anc) - n[0] * X - n[1] * Y) / n[2]              # on the floor plane

    def to_cam(Z):
        pc_ = np.stack([X - cam[0], Y - cam[1], Z - cam[2]], -1)
        zc = _dot3(pc_, R[:, 2]); xc = _dot3(pc_, R[:, 0]); yc = _dot3(pc_, R[:, 1])
        with np.errstate(divide='ignore', invalid='ignore'):
            mu = (K[0] * xc / zc + K[2]).astype(np.float32); mv = (K[1] * yc / zc + K[3]).astype(np.float32)
        ok = (zc > 0.3) & (zc < max_range) & (mu >= 1) & (mu < W_img - 2) & (mv >= 1) & (mv < H_img - 2)
        if keep_xy is not None:
            x0, x1, y0, y1 = (float(v) for v in keep_xy)
            ok &= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        return zc, mu, mv, ok

    zc, mu, mv, valid = to_cam(Z)
    if valid.sum() < 0.25 * (S / res) ** 2:
        return _no_target('outside_view')
    inview = valid.copy()                                       # in the image (and the cage), occluded or not
    # occlusion: a valid depth clearly nearer than the floor point = something raised in front
    ui = np.clip(np.round(mu), 0, W_img - 1).astype(int); vi = np.clip(np.round(mv), 0, H_img - 1).astype(int)
    if occlusion != 'none':
        # only the image window the grid samples (+ the hole bracket reach): same result, less work
        m = int(occl_hole_px) + 2
        r0, r1 = max(0, int(vi[valid].min()) - m), min(H_img, int(vi[valid].max()) + m + 1)
        c0, c1 = max(0, int(ui[valid].min()) - m), min(W_img, int(ui[valid].max()) + m + 1)
        occ_img = _occlusion_image(dep[r0:r1, c0:c1], (K[0], K[1], K[2] - c0, K[3] - r0), R, cam, n, anc,
                                   occl_rel, occl_hole_px, occl_hole_rel, occl_hole_min)
        occl = valid & occ_img[np.clip(vi - r0, 0, r1 - r0 - 1), np.clip(ui - c0, 0, c1 - c0 - 1)]
        valid &= ~occl
    blur = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.8)
    patch = cv2.remap(blur, np.where(valid, mu, 0).astype(np.float32), np.where(valid, mv, 0).astype(np.float32),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if debug:
        landing_pad.last_ipm = dict(patch=patch.copy(), valid=valid.copy(), occluded=(occl.copy() if occlusion != 'none' else None),
                                    a1=a1, b0=b0, res=res, fwd=fwd, lat=lat, cam=cam.copy())
    med = float(np.median(patch[valid]))
    mad = 1.4826 * float(np.median(np.abs(patch[valid] - med))) + 1.0
    # fill invalid cells from their own neighbourhood (normalised Gaussian averaging), not with the
    # global median: with stereo holes a large dark region (board / net / background) can stay
    # valid, and a dark global fill punched black blotches into the pad (pattern NCC 0.5 -> 0.2)
    wv = valid.astype(np.float32)
    num = cv2.GaussianBlur(patch * wv, (0, 0), 3.0); den = cv2.GaussianBlur(wv, (0, 0), 3.0)
    fill = np.where(den > 1e-3, num / np.maximum(den, 1e-3), med)
    patch = np.where(valid, patch, fill).astype(np.float32)
    if debug:
        landing_pad.last_ipm['filled'] = patch.copy()
    vmask = valid.astype(np.float32)
    # ---- search over yaw x position
    s_px = S / res
    T = _pad_template(s_px)
    tn = T.shape[0]
    best_p, best_q = None, None                                # (score, row, col, yaw) per cue
    ctr = ((nb - 1) / 2.0, (na - 1) / 2.0)
    # filled square (+ floor ring) and an equal-area disk: a pad is square, a painted rock is a blob
    m_sq = int(round(sq_support * s_px)) | 1
    cq = (m_sq - 1) / 2.0
    uq, vq = np.meshgrid(np.arange(m_sq) - cq, np.arange(m_sq) - cq)
    T_sq = ((np.abs(uq) <= 0.5 * s_px) & (np.abs(vq) <= 0.5 * s_px)).astype(np.float32)
    T_dk = (np.hypot(uq, vq) <= s_px / math.sqrt(math.pi)).astype(np.float32)
    # don't-care: the H (centre) and the circle + edge bars (annulus); their contrast to the body
    # flips with the sensor response, the body-vs-floor step is what the square measures
    rq = np.hypot(uq, vq) / s_px
    M_q = ~((rq <= 0.25) | ((rq >= 0.32) & (rq <= 0.42)))
    M_q = M_q.astype(np.uint8)
    if na > m_sq and nb > m_sq:
        ncc_dk = np.abs(_ncc_clean(cv2.matchTemplate(patch, T_dk, cv2.TM_CCOEFF_NORMED, mask=M_q)))
    else:
        ncc_dk = None
    off_q = m_sq // 2
    r_lim = (prior_radius_m / res) if prior_center_ned is not None else np.inf

    _near = {}

    def near_ctr(shape, off):
        key = (shape, off)
        if key not in _near:
            rr_, cc_ = np.mgrid[0:shape[0], 0:shape[1]]
            _near[key] = (np.hypot(rr_ + off - ctr[1], cc_ + off - ctr[0]) <= r_lim).astype(np.float32)
        return _near[key]

    def sq_score(pr, vr, yaw):
        """|NCC| with the filled-square template (either polarity), only where the square beats
        the disk at the same (unrotated) place and the template area is mostly visible."""
        if ncc_dk is None or pr.shape[0] <= m_sq or pr.shape[1] <= m_sq:
            return None
        ncc = _ncc_clean(cv2.matchTemplate(pr, T_sq, cv2.TM_CCOEFF_NORMED, mask=M_q))
        visq = cv2.boxFilter(vr, -1, (m_sq, m_sq), normalize=True, borderType=cv2.BORDER_CONSTANT)
        visq = visq[off_q:off_q + ncc.shape[0], off_q:off_q + ncc.shape[1]]
        sc = np.abs(ncc) * (visq >= 0.6) * near_ctr(ncc.shape, off_q)
        flat = sc.ravel()
        top8 = np.argpartition(flat, -8)[-8:] if flat.size > 8 else np.arange(flat.size)
        order = top8[np.argsort(flat[top8])[::-1]]
        best_any = None
        Minv = cv2.invertAffineTransform(cv2.getRotationMatrix2D(ctr, float(yaw), 1.0))
        for k in order:
            r_, c_ = divmod(int(k), sc.shape[1])
            if sc.flat[k] <= 0:
                break
            cx, cy = Minv @ np.array([c_ + off_q, r_ + off_q, 1.0])
            ri, ci = int(round(cy)) - off_q, int(round(cx)) - off_q
            if not (0 <= ri < ncc_dk.shape[0] and 0 <= ci < ncc_dk.shape[1]):
                continue
            margin = float(sc.flat[k] - ncc_dk[ri, ci])
            cand = (float(sc.flat[k]), r_ + off_q, c_ + off_q, float(np.sign(ncc.flat[k])), float(visq.flat[k]), margin, 4)
            if margin >= sq_margin_min:
                return cand
            if best_any is None:
                best_any = cand[:5] + (-1.0, 4)              # remembered for diagnostics only
        return best_any

    def eval_yaw(yaw, do_pat, do_sq):
        nonlocal best_p, best_q
        M = cv2.getRotationMatrix2D(ctr, float(yaw), 1.0)
        pr = cv2.warpAffine(patch, M, (nb, na), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=med)
        vr = cv2.warpAffine(vmask, M, (nb, na), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if do_pat and pr.shape[0] > tn and pr.shape[1] > tn:
            ncc = _ncc_clean(cv2.matchTemplate(pr, T, cv2.TM_CCOEFF_NORMED))
            vis = cv2.boxFilter(vr, -1, (tn, tn), normalize=True, borderType=cv2.BORDER_CONSTANT)
            off = tn // 2
            vis_c = vis[off:off + ncc.shape[0], off:off + ncc.shape[1]]
            sc = np.abs(ncc) * (vis_c >= 0.6) * near_ctr(ncc.shape, off)
            k = int(np.argmax(sc))
            r_, c_ = divmod(k, sc.shape[1])
            if best_p is None or sc.flat[k] > best_p[0]:
                best_p = (float(sc.flat[k]), r_ + off, c_ + off, float(yaw), float(np.sign(ncc.flat[k])), float(vis_c.flat[k]))
        if do_sq:
            q = sq_score(pr, vr, yaw % 90.0)
            if q is not None and (best_q is None or q[0] > best_q[0]):
                best_q = q + (float(yaw),)

    # coarse yaw (pattern 0-180, square 0-90: it repeats every 90 deg), then refine both
    coarse = 2.0 * yaw_step_deg
    for yaw in np.arange(0.0, 180.0, coarse):
        eval_yaw(yaw, True, yaw < 90.0)
    for d in (-yaw_step_deg, yaw_step_deg, -0.5 * yaw_step_deg, 0.5 * yaw_step_deg):
        if best_p is not None:
            eval_yaw((best_p[3] + d) % 180.0, True, False)
        if best_q is not None:
            eval_yaw((best_q[-1] + d) % 90.0, False, True)
    if best_p is None and best_q is None:
        return _no_target('outside_view')

    def to_world(row, col, yaw):
        Minv = cv2.invertAffineTransform(cv2.getRotationMatrix2D(ctr, yaw, 1.0))
        cx, cy = Minv @ np.array([col, row, 1.0])
        a = a1 - (cy + 0.5) * res; b = b0 + (cx + 0.5) * res
        xy = cam[:2] + a * fwd + b * lat
        # template u axis (rotated image x) back in the patch: direction of one pad edge
        du = Minv[:, 0]; d_world = du[1] * (-fwd) + du[0] * lat
        return xy, float(math.atan2(d_world[1], d_world[0]) % (math.pi / 2)), a

    def in_view_at(row, col, yaw, size):
        """Fraction of a size x size cells disk around a (rotated-patch) match that is in the image."""
        Minv = cv2.invertAffineTransform(cv2.getRotationMatrix2D(ctr, float(yaw), 1.0))
        cx, cy = Minv @ np.array([col, row, 1.0])
        rr_, cc_ = np.ogrid[0:na, 0:nb]
        disk = (rr_ - cy) ** 2 + (cc_ - cx) ** 2 <= (0.5 * size) ** 2
        return float(inview[disk].sum()) / max(float(math.pi * (0.5 * size) ** 2), 1.0)

    # ---- pattern cue
    c_pat, xy_p, yaw_p, a_p = 0.0, None, None, None
    if best_p is not None and best_p[0] > 0:
        xy_p, yaw_p, a_p = to_world(best_p[1], best_p[2], best_p[3])
        c_pat = float(np.clip((best_p[0] - 0.35) / 0.35, 0.0, 1.0))
    # ---- filled-square cue: |NCC| with square + ring, beating the disk; never alone above 0.8
    c_sq, xy_q, purity, yaw_q, a_q = 0.0, None, 0.0, None, None
    if best_q is not None and best_q[0] > 0:
        con, r_, c_, sgn_q, v_in, margin_q, _, yaw_q = best_q
        purity = margin_q
        xy_q, yaw_q, a_q = to_world(r_, c_, yaw_q)
        c_sq = 0.8 * float(np.clip((con - sq_ncc_min) / 0.15, 0.0, 1.0)) * float(np.clip(margin_q / (2 * sq_margin_min), 0.0, 1.0))
    # ---- combine: the pattern (H + bars) is specific, a filled square is not: a decent pattern
    # match decides the place; the square decides only when the pattern is weak (body about as
    # dark as the H, e.g. a red/NIR-heavy mono response). Agreeing cues reinforce.
    if xy_p is not None and (c_pat >= 0.3 or c_pat >= c_sq or xy_q is None):
        xy, yaw_o, a_o, vis = xy_p, yaw_p, a_p, best_p[5]
        conf = c_pat if xy_q is None or np.hypot(*(xy_q - xy_p)) > 0.15 else 1 - (1 - c_pat) * (1 - c_sq)
    elif xy_q is not None:
        xy, yaw_o, a_o, vis = xy_q, yaw_q, a_q, best_q[4]
        conf = c_sq if xy_p is None or np.hypot(*(xy_p - xy_q)) > 0.15 else 1 - (1 - c_pat) * (1 - c_sq)
        # the square alone must have its whole support (square + ring) inside the image: the image
        # border is a straight edge in the top-down view and completes a 'square' on bright floor
        # sheets / glare (real skybeta flights 010430 / 010807: 6 of 10 false hits). Occlusion by
        # legs / rails only lowers `visible` (and the confidence).
        if in_view_at(best_q[1], best_q[2], best_q[-1], m_sq) < sq_inview_min:
            t = _no_target('square_truncated')
            t.factors = dict(pattern=c_pat, ncc=best_p[0] if best_p else 0.0, square=c_sq,
                             square_ncc=best_q[0], square_vs_disk=purity)
            return t
    else:
        return _no_target('no_pad')
    in_view = in_view_at(*((best_p[1], best_p[2], best_p[3], tn) if xy is xy_p else (best_q[1], best_q[2], best_q[-1], m_sq)))
    # partial visibility (occluded by legs / rails, or cut by the image border) lowers the
    # confidence linearly: 100 % -> x1.0, 80 % -> x0.85, 60 % -> x0.7 (below 60 % no match at all)
    f_vis = float(np.clip((vis - 0.6) / 0.4, 0.0, 1.0)) * 0.3 + 0.7
    f_range = float(np.clip(1.0 - 0.5 * max(0.0, a_o - 3.0), 0.5, 1.0))
    conf = float(conf * f_vis * f_range)
    if prior_center_ned is not None and math.hypot(*(xy - np.asarray(prior_center_ned, float)[:2])) > prior_radius_m:
        return _no_target('outside_prior')
    if conf < min_confidence:
        t = _no_target('low_confidence')
        t.factors = dict(pattern=c_pat, ncc=best_p[0] if best_p else 0.0, square=c_sq,
                         square_ncc=best_q[0] if best_q else 0.0, square_vs_disk=purity)
        return t
    z_floor = float((n @ anc - n[0] * xy[0] - n[1] * xy[1]) / n[2])
    return LandingTarget(True, np.array([xy[0], xy[1], z_floor]), z_floor, S, conf, extent_xy_m=np.array([S, S]),
                         yaw_ned=yaw_o, height_above_floor=0.0, range_m=float(math.hypot(*(xy - cam[:2]))),
                         truncated=bool(vis < 0.95), n_top_points=int(valid.sum()), kind='pad',
                         factors=dict(pattern=c_pat, ncc=best_p[0] if best_p else 0.0, square=c_sq,
                                      square_ncc=best_q[0] if best_q else 0.0, square_vs_disk=purity,
                                      body_vs_floor=(best_q[3] if best_q else 0.0), visible=vis, in_view=in_view, range=f_range,
                                      floor_mad=mad))
