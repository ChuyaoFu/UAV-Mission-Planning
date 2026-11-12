"""Tunnel-mouth detector for the official DroneHacks stack (perception/tunnel.py).

Contract (INTERFACES.md):
    detect(frame, prior_entry_ned=None)
        -> TunnelEstimate(found, entry_ned (3,), axis_ned (2,) unit, radius_m, confidence)

The tunnel is a row of thin brass hoops standing in vertical planes, lined with black mesh
(invisible to stereo) and covered on top by a rock-print fabric. The detector therefore works
on the IMAGE of the wire, not on the mesh depth:

  1. Thin bright ridges at full and half resolution: Hessian line response + NMS + a
     "both sides darker" test (or: clearly brighter than both sides of a step edge, for the
     entrance hoop between bright background and black mesh). Dark net cords, the bright side of
     step edges and tiny components (gaps between net cords) are rejected.
  2. Ellipse hypotheses, in roll-compensated coordinates: 2 points + tangents (axis-aligned
     ellipse, minimal sample), 3 points + tangents (general), and direct fits on single / pairs /
     triples of the largest components (a near hoop is often two separate side arcs). Scored by
     tangent-consistent angular coverage; greedy shortlist on UNEXPLAINED support; refined by a
     direct ellipse fit.
  3. 3-D: each ellipse is fitted as a VERTICAL circle seen by the calibrated, gravity-aligned
     camera (shape only: centre direction, axis yaw, radius/distance; Sampson residuals; small
     robust LM). Absolute scale: floor height (hoop bottom = floor - bottom_gap), optional measured
     radius, and (entrance only, gated) stereo depth on the fabric over the hoop top.
     A "ring" whose inside is a solid surface at ring depth (printed circle) is rejected.
  4. Coaxial grouping (axis / radius / height), pair-RANSAC centre line, axis = centre line +
     hoop normals weighted by view-angle-dependent sigmas. Hoops lie on a regular lattice along
     the axis: off-lattice fits (hybrids across two hoops) are dropped and hoops in front of the
     front-most one are predicted and verified directly against the ridge pixels, so the
     entrance (hoop nearest the camera) is recovered even when the 2-D stage missed it.

Pure numpy / OpenCV (no scipy needed); no ROS, no clock reads in the result (timings only in
debug). Tested with the repo .venv (Python 3.9, numpy 1.26, OpenCV 4.11) and Python 3.12
(numpy 2.5, OpenCV 5.0). Hot products are elementwise on purpose (multithreaded BLAS on
mid-size matmuls stalls on a loaded CPU).
"""
from dataclasses import dataclass, field
import math
import time
from typing import Optional, Tuple

import numpy as np
import cv2


# ----------------------------------------------------------------------------- parameters
@dataclass
class TunnelParams:
    # hoop geometry (measure on site; defaults are guesses from the venue photo)
    radius_m: Optional[float] = None          # measured hoop radius -> strong scale prior
    radius_sigma_m: float = 0.015            # tape-measured hoop radius
    bottom_gap_m: Optional[float] = 0.08      # hoop lowest point above the floor; None = unknown
    floor_sigma_m: float = 0.04               # floor height + gap uncertainty (1 sigma)
    depth_gate: float = 0.06                  # |ln(depth / floor-or-radius scale)| above which depth is dropped
    default_floor_z_ned: float = 0.0          # used if no floor_z_ned is passed and depth has none
    radius_range_m: Tuple[float, float] = (0.30, 0.90)   # r < 0.3 m is unflyable for a 0.4 m quad (also rejects floor rings fitted as small hoops)
    range_m: Tuple[float, float] = (0.4, 6.0)
    max_view_angle_deg: float = 72.0          # hoop normal vs line of sight
    # ridge extraction
    polarity: str = 'bright'                  # 'bright' wire, or 'both' (wire darker than a bright
                                              # background behind it; also picks up dark net cords)
    blur_sigma: float = 0.8
    ridge_min: float = 5.0                    # Hessian response (grey levels / px^2, Sobel scaled)
    line_contrast: float = 9.0                # centre minus brighter side, grey levels
    max_points: int = 3000
    min_component_px: int = 8
    # ellipse hypotheses
    n_hypotheses: int = 420
    seed: int = 7
    n_bins: int = 72
    inlier_px: float = 1.6
    tangent_tol_deg: float = 16.0
    min_support_bins: int = 10
    min_visible_frac: float = 0.25
    min_cover: float = 0.20
    min_axis_px: float = 12.0
    min_axis_ratio: float = 0.2
    max_2d_candidates: int = 6
    # 3-D fit
    fit_points: int = 110
    max_rms_px: float = 1.6
    # grouping / prior / decision
    group_axis_deg: float = 25.0
    group_lateral_m: float = 0.25
    group_radius_frac: float = 0.12          # hoops of one tunnel share a radius
    group_height_m: float = 0.15
    group_min_span_m: float = 0.12
    group_max_span_m: float = 3.0
    line_outlier_m: float = 0.06              # hoop centre off the line through the others
    hoop_spacing_m: Optional[float] = None    # measured hoop spacing (lets a lone hoop be extended)
    min_hoop_spacing_m: float = 0.25
    prior_tol_m: float = 0.5                  # prior error accepted without confidence penalty
    prior_sigma_m: float = 0.4                # decay of the prior factor beyond prior_tol_m
    max_inferred_hoops: int = 3
    n_hoops: Optional[int] = None             # measured number of hoops (bounds the back-inference)
    prior_gate_m: float = 1.5
    min_confidence: float = 0.2
    single_min_support: int = 20              # a lone hoop (no coaxial partner) needs more evidence
    single_min_cover: float = 0.35
    align_max_view_deg: float = 45.0          # lone hoop: max view angle for align_ok


# ----------------------------------------------------------------------------- results
@dataclass
class Ellipse2D:
    conic_n: np.ndarray            # (6,) conic in normalised image coords (see _Norm)
    center_px: np.ndarray          # (2,)
    axes_px: np.ndarray            # (2,) semi-axes along `angle` and `angle + 90 deg`
    angle: float                   # rad, image x towards image y
    support: int                   # covered angular bins (tangent-consistent inliers)
    visible: int                   # bins whose ellipse point lies inside the image
    cover: float                   # support / visible
    inliers: np.ndarray            # indices into the ridge point set
    score: float

    def points(self, n=180):
        phi = np.linspace(-np.pi, np.pi, n, endpoint=False)
        ca, sa = math.cos(self.angle), math.sin(self.angle)
        a1, a2 = self.axes_px
        x = self.center_px[0] + a1*np.cos(phi)*ca - a2*np.sin(phi)*sa
        y = self.center_px[1] + a1*np.cos(phi)*sa + a2*np.sin(phi)*ca
        return np.stack([x, y], -1)


@dataclass
class Ring:
    center_ned: np.ndarray         # (3,) hoop centre
    axis_ned: np.ndarray           # (2,) unit horizontal hoop normal, pointing away from camera
    radius_m: float
    distance_m: float              # camera -> hoop centre
    rms_px: float
    inlier_frac: float
    view_angle_deg: float          # hoop normal vs line of sight (0 = head-on)
    ambiguous: bool                # +-yaw mirror solution fits about as well
    alt_axis_ned: np.ndarray       # the mirror solution's axis
    scale_source: str
    scale_detail: dict
    ellipse: Ellipse2D
    score: float
    shape: np.ndarray              # (alpha, beta, psi, rho) unit-distance shape
    fit_px: np.ndarray = None      # (M,2) ridge pixels used in the 3-D fit


@dataclass
class TunnelEstimate:
    found: bool
    entry_ned: np.ndarray          # (3,) centre of the entrance hoop (NED)
    axis_ned: np.ndarray           # (2,) unit horizontal tunnel axis, entrance -> exit
    radius_m: float
    confidence: float
    rings: list = field(default_factory=list)       # hoops of the chosen group, entrance first
    candidates: list = field(default_factory=list)  # all 2-D ellipse candidates (debug / drawing)
    scale_source: str = ''
    axis_source: str = ''
    debug: dict = field(default_factory=dict)
    # geometry good enough for the FINAL alignment: >= 2 coaxial hoops (centre line), or a lone hoop
    # seen within align_max_view_deg of head-on. Lone hoops seen side-on can be 0.1-0.2 m off the
    # axis with high confidence (venue replica), so they are for coarse steering only.
    align_ok: bool = False


def _not_found(debug=None, candidates=None):
    return TunnelEstimate(False, np.full(3, np.nan), np.full(2, np.nan), float('nan'), 0.0,
                          [], candidates or [], '', '', debug or {})


# ----------------------------------------------------------------------------- 1. ridges
_DIRS = ((1, 0), (1, 1), (0, 1), (-1, 1))      # NMS neighbour for normal angle 0/45/90/135 deg


def _shift(a, dx, dy):
    """b[y, x] = a[y + dy, x + dx] with edge padding."""
    h, w = a.shape
    p = np.pad(a, 1, mode='edge')
    return p[1+dy:1+dy+h, 1+dx:1+dx+w]


@dataclass
class Ridges:
    xy: np.ndarray        # (N,2) float pixel coords (x, y)
    t: np.ndarray         # (N,2) unit tangent
    s: np.ndarray         # (N,) ridge strength
    lab: np.ndarray       # (N,) connected-component label
    shape: tuple


def _ridges_one_scale(gray, p, min_comp):
    """Ridge pixels of one image scale -> xs, ys, nx, ny (normal), strength, component label."""
    g = gray.astype(np.float32)
    if p.blur_sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), p.blur_sigma)
    ixx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
    iyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
    ixy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
    tr = 0.5*(ixx + iyy)
    df = 0.5*(ixx - iyy)
    disc = np.sqrt(df*df + ixy*ixy)
    s = disc - tr                                   # -lambda_min (across a bright line)
    s -= np.maximum(0.0, -(tr + disc))              # suppress bright blobs (both curvatures < 0)
    th = 0.5*np.arctan2(2*ixy, ixx - iyy) + 0.5*np.pi   # normal direction, [0, pi]
    q = np.round(th/(np.pi/4)).astype(np.int32) % 4
    cand = s > p.ridge_min
    cand[:3, :] = False
    cand[-3:, :] = False
    cand[:, :3] = False
    cand[:, -3:] = False
    keep = np.zeros(s.shape, bool)
    for k, (dx, dy) in enumerate(_DIRS):
        m = cand & (q == k)
        if m.any():
            keep |= m & (s >= _shift(s, dx, dy)) & (s >= _shift(s, -dx, -dy))
    ys, xs = np.nonzero(keep)
    H, W = g.shape
    e = np.zeros(0)
    if len(xs) == 0:
        return e, e, e, e, e, np.zeros(0, int)
    tt = th[ys, xs]
    nx, ny = np.cos(tt), np.sin(tt)
    c0 = g[ys, xs]
    ok = np.zeros(len(xs), bool)
    for w in (2, 3):
        x1 = np.clip(np.round(xs + w*nx).astype(int), 0, W-1)
        y1 = np.clip(np.round(ys + w*ny).astype(int), 0, H-1)
        x2 = np.clip(np.round(xs - w*nx).astype(int), 0, W-1)
        y2 = np.clip(np.round(ys - w*ny).astype(int), 0, H-1)
        s1, s2 = g[y1, x1], g[y2, x2]
        hi, lo = np.maximum(s1, s2), np.minimum(s1, s2)
        # a line: both sides darker; or a line on a step edge (the entrance hoop against a
        # bright background on one side and the black mesh on the other): strictly brighter
        # than the bright side and much brighter than the dark side
        ok |= (c0 - hi > p.line_contrast) | ((c0 - hi > 0.4*p.line_contrast) & (c0 - lo > 3*p.line_contrast))
    xs, ys, nx, ny, sv = xs[ok], ys[ok], nx[ok], ny[ok], s[ys[ok], xs[ok]]
    mask = np.zeros((H, W), np.uint8)
    mask[ys, xs] = 255
    n, lab = cv2.connectedComponents(mask, connectivity=8)
    lab = lab[ys, xs]
    # drop tiny components (gaps between net cords, noise); size = ridge pixels in the component
    keep = np.bincount(lab, minlength=n)[lab] >= min_comp
    return xs[keep], ys[keep], nx[keep], ny[keep], sv[keep], lab[keep]


def extract_ridges(gray, p=None):
    """Thin bright line pixels (brass wire) with tangent directions, at full and half resolution
    (a near hoop's wire is 3-5 px wide and breaks up at full resolution)."""
    p = p or TunnelParams()
    H, W = gray.shape[:2]
    imgs = [gray] if p.polarity == 'bright' else [gray, 255 - gray]
    parts, off = [], 0
    for im in imgs:
        for scale, img_s, mc in ((1, im, p.min_component_px),
                                 (2, cv2.pyrDown(im), max(4, p.min_component_px//2))):
            x, y, nx_, ny_, sv_, lab_ = _ridges_one_scale(img_s, p, mc)
            if scale == 2:
                x, y = 2.0*x + 0.5, 2.0*y + 0.5
            parts.append((x.astype(float), y.astype(float), nx_, ny_, sv_, lab_ + off))
            off += (int(lab_.max()) + 1) if len(lab_) else 0
    xs, ys, nx, ny, sv, lab = (np.concatenate([q[i] for q in parts]) for i in range(6))
    lab = lab.astype(np.int64)
    if len(xs) > p.max_points:                      # uniform thinning keeps every component
        idx = np.linspace(0, len(xs) - 1, p.max_points).astype(int)
        xs, ys, nx, ny, sv, lab = xs[idx], ys[idx], nx[idx], ny[idx], sv[idx], lab[idx]
    return Ridges(np.stack([xs, ys], -1), np.stack([-ny, nx], -1), sv.astype(np.float64), lab, (H, W))


# ----------------------------------------------------------------------------- 2. ellipses
class _Norm:
    """Normalised, roll-compensated image coords: pn = Rr (x - c) / S. With the camera roll
    removed, vertical hoops project to ellipses whose axes are close to the image axes."""
    def __init__(self, shape, roll=0.0):
        H, W = shape
        self.c = np.array([W/2.0, H/2.0])
        self.S = max(W, H)/2.0
        self.roll = roll
        cr, sr = math.cos(roll), math.sin(roll)
        self.Rr = np.array([[cr, -sr], [sr, cr]])

    def fwd(self, xy):
        return ((xy - self.c)/self.S) @ self.Rr.T

    def back(self, pn):
        x, y = pn[..., 0], pn[..., 1]
        R = self.Rr
        return np.stack([self.c[0] + self.S*(x*R[0, 0] + y*R[1, 0]),
                         self.c[1] + self.S*(x*R[0, 1] + y*R[1, 1])], -1)


def _ellipse_params(c):
    """Conic coefficients (n,6) -> ok, centre (n,2), semi-axes a1 (along ang), a2, ang."""
    c = np.atleast_2d(c)
    sgn = np.where(c[:, 0] + c[:, 2] < 0, -1.0, 1.0)
    A, B, C, D, E, F = (c*sgn[:, None]).T
    det = 4*A*C - B*B
    with np.errstate(all='ignore'):
        x0 = (B*E - 2*C*D)/det
        y0 = (B*D - 2*A*E)/det
        Fc = A*x0*x0 + B*x0*y0 + C*y0*y0 + D*x0 + E*y0 + F
        root = np.sqrt((A - C)**2 + B*B)
        l1 = 0.5*(A + C + root)
        l2 = 0.5*(A + C - root)
        a1 = np.sqrt(-Fc/l1)
        a2 = np.sqrt(-Fc/l2)
    ok = (det > 0) & (l2 > 0) & (Fc < 0) & np.isfinite(a1) & np.isfinite(a2) & np.isfinite(x0) & np.isfinite(y0)
    ang = 0.5*np.arctan2(B, A - C)
    return ok, np.stack([x0, y0], -1), a1, a2, ang


def _conics_axis_aligned(P, T):
    """2 points + tangents (H,2,2) -> (H,6) axis-aligned conics (B = 0), the minimal sample."""
    x, y, tx, ty = P[..., 0], P[..., 1], T[..., 0], T[..., 1]
    one, zero = np.ones_like(x), np.zeros_like(x)
    rp = np.stack([x*x, y*y, x, y, one], -1)
    rt = np.stack([2*x*tx, 2*y*ty, tx, ty, zero], -1)
    _, _, vt = np.linalg.svd(np.concatenate([rp, rt], 1))
    v = vt[:, -1, :]
    return np.stack([v[:, 0], 0*v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4]], 1)


def _conics_general(P, T):
    """3 points + tangents (H,3,2) -> (H,6) least-squares general conics."""
    x, y, tx, ty = P[..., 0], P[..., 1], T[..., 0], T[..., 1]
    one, zero = np.ones_like(x), np.zeros_like(x)
    rp = np.stack([x*x, x*y, y*y, x, y, one], -1)
    rt = np.stack([2*x*tx, y*tx + x*ty, 2*y*ty, tx, ty, zero], -1)
    _, _, vt = np.linalg.svd(np.concatenate([rp, rt], 1))
    return vt[:, -1, :]


def _fit_conic_direct(x, y):
    """Halir & Flusser numerically stable direct ellipse fit -> (6,) or None."""
    D1 = np.stack([x*x, x*y, y*y], 1)
    D2 = np.stack([x, y, np.ones_like(x)], 1)
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    try:
        Tm = -np.linalg.solve(S3, S2.T)
        M = S1 + S2 @ Tm
        M = np.array([M[2]/2, -M[1], M[0]/2])
        w, v = np.linalg.eig(M)
    except np.linalg.LinAlgError:
        return None
    v = np.real(v)
    cond = 4*v[0]*v[2] - v[1]**2
    good = np.nonzero(cond > 0)[0]
    if len(good) == 0:
        return None
    a1 = v[:, good[0]]
    c = np.concatenate([a1, Tm @ a1])
    n = np.linalg.norm(c)
    return c/n if n > 0 and np.all(np.isfinite(c)) else None


class _PointSet:
    """Ridge points in normalised coords. Products are written elementwise on purpose: a
    multithreaded BLAS on these mid-size matmuls stalls badly on a loaded CPU (measured
    30 ms -> 600+ ms at load 15), and the flight computer is a loaded CPU."""
    def __init__(self, P, T):
        self.P, self.T = P, T

    def geom(self, cn, idx=None):
        """-> first-order distance (normalised units) and |sin| tangent misalignment, (N,H)."""
        P, T = (self.P, self.T) if idx is None else (self.P[idx], self.T[idx])
        x, y = P[:, 0:1], P[:, 1:2]
        A, B, C, D, E, F = (cn[None, :, k] for k in range(6))
        gx = 2*A*x + B*y + D
        gy = B*x + 2*C*y + E
        f = (A*x + B*y + D)*x + (C*y + E)*y + F
        gn = np.sqrt(gx*gx + gy*gy) + 1e-12
        return np.abs(f)/gn, np.abs(gx*T[:, 0:1] + gy*T[:, 1:2])/gn


def _coverage(P, inl, ctr, a1, a2, ang, nrm, shape, nb):
    """Angular bins covered by inliers and bins visible in the image, per conic (sparse)."""
    Hn = len(ctr)
    pi, hi = np.nonzero(inl)
    ca, sa = np.cos(ang), np.sin(ang)
    rx = P[pi, 0] - ctr[hi, 0]
    ry = P[pi, 1] - ctr[hi, 1]
    u1 = (rx*ca[hi] + ry*sa[hi])/a1[hi]
    u2 = (-rx*sa[hi] + ry*ca[hi])/a2[hi]
    b = ((np.arctan2(u2, u1) + np.pi)*(nb/(2*np.pi))).astype(np.int64) % nb
    hits = np.zeros((Hn, nb), bool)
    hits[hi, b] = True
    phi = -np.pi + (np.arange(nb) + 0.5)*(2*np.pi/nb)
    ex = ctr[:, 0:1] + a1[:, None]*np.cos(phi)*ca[:, None] - a2[:, None]*np.sin(phi)*sa[:, None]
    ey = ctr[:, 1:2] + a1[:, None]*np.cos(phi)*sa[:, None] + a2[:, None]*np.sin(phi)*ca[:, None]
    pix = nrm.back(np.stack([ex, ey], -1))
    H, W = shape
    vis = (pix[..., 0] >= 0) & (pix[..., 0] < W) & (pix[..., 1] >= 0) & (pix[..., 1] < H)
    return (hits & vis).sum(1), vis.sum(1)


def _components(rd, roi):
    """Ridge components (index arrays ordered along their principal direction), largest first."""
    labs, inv, cnt = np.unique(rd.lab, return_inverse=True, return_counts=True)
    order = np.argsort(-cnt)
    comps = []
    for k in order[:80]:
        idx = np.nonzero(inv == k)[0]
        pts = rd.xy[idx]
        mu = pts.mean(0)
        if roi is not None and not (roi[0] <= mu[0] <= roi[2] and roi[1] <= mu[1] <= roi[3]):
            continue
        d = pts - mu
        _, _, vt = np.linalg.svd(d, full_matrices=False)
        comps.append(idx[np.argsort(d @ vt[0])])
    return comps


def _sample(comps, p, rng):
    """Index pairs (2-point axis-aligned hypotheses) and triples (3-point general hypotheses)."""
    if not comps:
        return np.zeros((0, 2), int), np.zeros((0, 3), int)
    sizes = np.array([len(c) for c in comps], float)
    w = sizes/sizes.sum()
    n2 = int(p.n_hypotheses*0.75)
    n3 = p.n_hypotheses - n2
    A = rng.choice(len(comps), n2, p=w)
    B = np.where(rng.random(n2) < 0.35, A, rng.choice(len(comps), n2, p=w))
    pairs = []
    for a, b in zip(A, B):
        ia, ib = comps[a], comps[b]
        if a == b:                                   # far-apart points of one arc
            n = len(ia)
            pairs.append((ia[rng.integers(0, max(1, (35*n)//100))], ia[rng.integers(min(n - 1, (65*n)//100), n)]))
        else:
            pairs.append((ia[rng.integers(len(ia))], ib[rng.integers(len(ib))]))
    trip = []
    A = rng.choice(len(comps), n3, p=w)
    B = rng.choice(len(comps), n3, p=w)
    for a, b in zip(A, B):
        ia, ib = comps[a], comps[b]
        n = len(ia)
        if a == b:
            trip.append((ia[rng.integers(0, max(1, n//5))], ia[rng.integers(2*n//5, max(2*n//5 + 1, 3*n//5))],
                         ia[rng.integers(min(n - 1, 4*n//5), n)]))
        else:
            trip.append((ia[rng.integers(0, max(1, (3*n)//10))], ia[rng.integers(min(n - 1, (7*n)//10), n)],
                         ib[rng.integers(len(ib))]))
    return np.array(pairs, int).reshape(-1, 2), np.array(trip, int).reshape(-1, 3)


def _union_fits(comps, P, k_pair=12, k_trip=6):
    """Deterministic hypotheses: direct ellipse fits on single large components and on the
    unions of every pair / triple of the largest ones. A near hoop is often seen only as two
    near-vertical side arcs (top under the fabric, bottom out of frame) in separate components;
    random 2-3 point samples rarely pair them, a union fit always does."""
    out = []
    K = min(k_pair, len(comps))
    sets = [(i,) for i in range(K) if len(comps[i]) >= 15]
    sets += [(i, j) for i in range(K) for j in range(i + 1, K)]
    kt = min(k_trip, len(comps))
    sets += [(i, j, k) for i in range(kt) for j in range(i + 1, kt) for k in range(j + 1, kt)]
    for st in sets:
        idx = np.concatenate([comps[i] for i in st])
        if len(idx) > 400:
            idx = idx[np.linspace(0, len(idx) - 1, 400).astype(int)]
        c = _fit_conic_direct(P[idx, 0], P[idx, 1])
        if c is not None:
            out.append(c)
    return np.array(out).reshape(-1, 6)


def detect_ellipses(gray, p=None, roi=None, ridges=None, roll=0.0):
    """2-D stage: ellipse candidates of thin bright arcs. Returns (candidates, ridges).
    roll: camera roll (rad) about the optical axis; image 'down' should be gravity."""
    p = p or TunnelParams()
    rd = ridges if ridges is not None else extract_ridges(gray, p)
    shape = rd.shape
    if len(rd.xy) < 12:
        return [], rd
    nrm = _Norm(shape, roll)
    P = nrm.fwd(rd.xy)
    T = rd.t @ nrm.Rr.T
    ps = _PointSet(P, T)
    rng = np.random.default_rng(p.seed)
    comps = _components(rd, roi)
    if roi is not None and sum(len(c) for c in comps) < 40:
        comps = _components(rd, None)
    pairs, trip = _sample(comps, p, rng)
    cn = np.concatenate([_conics_axis_aligned(P[pairs], T[pairs]) if len(pairs) else np.zeros((0, 6)),
                         _conics_general(P[trip], T[trip]) if len(trip) else np.zeros((0, 6)),
                         _union_fits(comps, P)])
    if len(cn) == 0:
        return [], rd
    cn /= np.linalg.norm(cn, axis=1, keepdims=True) + 1e-300
    H, W = shape
    amin, amax = p.min_axis_px/nrm.S, 1.6*max(W, H)/nrm.S
    lim_y = 1.6*H/max(W, H) + 0.6

    def plausible(ok, ctr, a1, a2):
        big, small = np.maximum(a1, a2), np.minimum(a1, a2)
        with np.errstate(invalid='ignore'):
            return ok & (small > amin) & (big < amax) & (small > p.min_axis_ratio*big) & \
                (np.abs(ctr[:, 0]) < 1.6) & (np.abs(ctr[:, 1]) < lim_y)

    ok, ctr, a1, a2, ang = _ellipse_params(cn)
    ok = plausible(ok, ctr, a1, a2)
    if not ok.any():
        return [], rd
    cn, ctr, a1, a2, ang = cn[ok], ctr[ok], a1[ok], a2[ok], ang[ok]
    sub = np.arange(len(P)) if len(P) <= 700 else np.linspace(0, len(P) - 1, 700).astype(int)
    tol = p.inlier_px/nrm.S
    sin_tol = math.sin(math.radians(p.tangent_tol_deg))
    dist, serr = ps.geom(cn, sub)
    inl = (dist < 1.5*tol) & (serr < sin_tol)
    sup, vis = _coverage(P[sub], inl, ctr, a1, a2, ang, nrm, shape, p.n_bins)
    sc = sup*(0.3 + sup/np.maximum(vis, 1))
    # greedy shortlist on UNEXPLAINED support (sequential-RANSAC style): duplicates and hybrids
    # of an already picked strong hoop lose their score, so a partly visible near hoop survives
    top = []
    n_inl = np.maximum(inl.sum(0), 1)
    alive = sup >= max(4, p.min_support_bins//2)
    explained = np.zeros(inl.shape[0], bool)
    eff = sc.copy()
    for _ in range(16):
        eff[~alive] = -1
        h = int(np.argmax(eff))
        if eff[h] <= 0.5*p.min_support_bins:
            break
        top.append(h)
        alive[h] = False
        explained |= inl[:, h]
        new = (inl & ~explained[:, None]).sum(0)
        eff = sc*(new/n_inl)
    top = np.array(top, int)
    if len(top) == 0:
        return [], rd
    # batched refinement on all points: inliers -> direct ellipse fit, three rounds
    C = cn[top].copy()
    alive = np.ones(len(C), bool)
    for thr in (2.5*tol, tol, tol):
        d, se = ps.geom(C)
        M = (d < thr) & (se < sin_tol)
        for k in np.nonzero(alive)[0]:
            m = M[:, k]
            if m.sum() < 8:
                alive[k] = False
                continue
            c2 = _fit_conic_direct(P[m, 0], P[m, 1])
            if c2 is None:
                alive[k] = False
                continue
            C[k] = c2
    C = C[alive]
    if len(C) == 0:
        return [], rd
    ok, ctr, a1, a2, ang = _ellipse_params(C)
    ok = plausible(ok, ctr, a1, a2)
    C, ctr, a1, a2, ang = C[ok], ctr[ok], a1[ok], a2[ok], ang[ok]
    if len(C) == 0:
        return [], rd
    d, se = ps.geom(C)
    M = (d < tol) & (se < sin_tol)
    sup, vis = _coverage(P, M, ctr, a1, a2, ang, nrm, shape, p.n_bins)
    refined = []
    for k in range(len(C)):
        s1, v1 = int(sup[k]), int(vis[k])
        if v1 < p.min_visible_frac*p.n_bins or s1 < p.min_support_bins:
            continue
        cover = s1/max(v1, 1)
        if cover < p.min_cover:
            continue
        cpx = nrm.back(ctr[k])
        refined.append(Ellipse2D(C[k], cpx, nrm.S*np.array([a1[k], a2[k]]), float(ang[k] - roll),
                                 s1, v1, cover, np.nonzero(M[:, k])[0], s1*(0.3 + cover)))
    refined.sort(key=lambda e: -e.score)
    out = []
    for e in refined:
        dup = False
        for f in out:
            ov = len(np.intersect1d(e.inliers, f.inliers, assume_unique=True))
            if ov > 0.4*min(len(e.inliers), len(f.inliers)):
                dup = True
                break
            if (np.linalg.norm(e.center_px - f.center_px) < 6 and
                    abs(max(e.axes_px) - max(f.axes_px)) < 0.06*max(f.axes_px)):
                dup = True
                break
        if not dup:
            out.append(e)
        if len(out) >= p.max_2d_candidates:
            break
    return out, rd


# ----------------------------------------------------------------------------- 3. 3-D hoop
def _kmat(K):
    fx, fy, cx, cy = K
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])


def _dir(a, b):
    cb = math.cos(b)
    return np.array([cb*math.cos(a), cb*math.sin(a), math.sin(b)])


def _cross(a, b):
    return np.stack([a[:, 1]*b[:, 2] - a[:, 2]*b[:, 1], a[:, 2]*b[:, 0] - a[:, 0]*b[:, 2],
                     a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]], 1)


def _model_conics(X, KR):
    """Pixel conics (B,3,3) of vertical circles. X (B,4) = (alpha, beta, psi, rho): centre at
    unit distance along dir(alpha, beta) (NED), normal yaw psi, radius rho (relative).
    KR = K @ R_cn. Uses inv(H) ~ rows (h2 x h3, h3 x h1, h1 x h2)."""
    X = np.atleast_2d(X)
    a, b, psi, rho = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    cb = np.cos(b)
    d = np.stack([cb*np.cos(a), cb*np.sin(a), np.sin(b)], 1)
    eh = np.stack([-np.sin(psi), np.cos(psi), np.zeros_like(psi)], 1)
    k1 = rho[:, None]*(eh @ KR.T)
    k2 = rho[:, None]*KR[:, 2][None, :]
    k3 = d @ KR.T
    c23 = _cross(k2, k3)
    c31 = _cross(k3, k1)
    c12 = _cross(k1, k2)
    Q = c23[:, :, None]*c23[:, None, :] + c31[:, :, None]*c31[:, None, :] - c12[:, :, None]*c12[:, None, :]
    return Q/(np.linalg.norm(Q.reshape(len(X), -1), axis=1)[:, None, None] + 1e-300)


def _sampson(Q, U):
    """Q (B,3,3), U (M,3) homogeneous pixels -> signed first-order distances (B,M) in pixels."""
    QU = np.einsum('bij,mj->bmi', Q, U)
    f = np.einsum('bmi,mi->bm', QU, U)
    return f/(2*np.hypot(QU[..., 0], QU[..., 1]) + 1e-12)


def _lm(U, x0, KR, free=(0, 1, 2, 3), iters=25):
    """Robust (soft-L1) Levenberg-Marquardt with a batched forward-difference Jacobian."""
    x = np.array(x0, float)
    x[1] = float(np.clip(x[1], -1.3, 1.3))
    x[3] = float(np.clip(x[3], 0.02, 3.0))
    free = list(free)

    def res(xx):
        return _sampson(_model_conics(xx, KR), U)

    def cost(r):
        return float(np.sum(2*(np.sqrt(1 + r*r) - 1)))

    r = res(x)[0]
    c = cost(r)
    lam = 1e-3
    for _ in range(iters):
        h = np.array([2e-4, 2e-4, 1e-3, 1e-3*max(x[3], 0.05)])[free]
        X = np.repeat(x[None], len(free) + 1, 0)
        for k, j in enumerate(free):
            X[k + 1, j] += h[k]
        R = res(X)
        J = (R[1:] - R[0])/h[:, None]
        w = 1/np.sqrt(1 + R[0]*R[0])
        A = (J*w) @ J.T
        g = (J*w) @ R[0]
        step_ok = False
        for _ in range(8):
            try:
                dx = -np.linalg.solve(A + lam*np.diag(np.diag(A) + 1e-12), g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            xn = x.copy()
            xn[free] += dx
            xn[1] = float(np.clip(xn[1], -1.3, 1.3))
            xn[3] = float(np.clip(xn[3], 0.02, 3.0))
            rn = res(xn)[0]
            cn = cost(rn)
            if cn < c:
                x, r, c = xn, rn, cn
                lam = max(lam*0.2, 1e-7)
                step_ok = True
                break
            lam *= 8
        if not step_ok or np.max(np.abs(dx/h)) < 0.05:
            break
    return x, r


def _fit_shape(U, x0, KR, fix_psi=None):
    """Fit the vertical-circle shape to homogeneous pixel points U (M,3)."""
    x0 = np.array(x0, float)
    if fix_psi is not None:
        x0[2] = fix_psi
    x, res = _lm(U, x0, KR, free=(0, 1, 3) if fix_psi is not None else (0, 1, 2, 3))
    a = np.abs(res)
    good = a < 3.0
    rms = float(np.sqrt(np.mean(res[good]**2))) if good.any() else 99.0
    return x, rms, float(np.mean(a < 2.0)), float(np.sum(np.minimum(res*res, 9.0)))


def _wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def _sample_depth(depth, uv, rad=1):
    H, W = depth.shape
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    out = []
    for du in range(-rad, rad + 1):
        for dv in range(-rad, rad + 1):
            uu, vv = u + du, v + dv
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            val = np.full(len(u), np.nan)
            val[ok] = depth[vv[ok], uu[ok]]
            out.append(val)
    return np.stack(out, 1)


def _project(Km, R_cn, cam, P):
    pc = (P - cam) @ R_cn.T
    z = pc[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = (pc @ Km.T)[:, :2]/z[:, None]
    return uv, z


def _scale_estimates(x, frame, Km, R_cn, cam, floor_z, p, depth, use_depth=False):
    """Absolute distance t of the hoop centre from several sources -> (t, source, detail).
    Depth is sampled on the fabric just above the hoop's top arc. That is only meaningful for the
    ENTRANCE hoop (the fabric drapes over it; above an inner hoop the image shows the entrance's
    fabric flap, ~0.5 m nearer), so it is used for the entrance only (use_depth=True) or when no
    other source exists, and dropped when it disagrees with floor / radius by > depth_gate."""
    a, b, psi, rho = x
    d = _dir(a, b)
    est = {}
    if p.bottom_gap_m is not None and floor_z is not None:
        num = floor_z - p.bottom_gap_m - cam[2]
        den = d[2] + rho
        if num > 0.12 and den > 0.03:
            est['floor'] = (num/den, max(0.015, p.floor_sigma_m/num))
    if p.radius_m is not None:
        est['radius'] = (p.radius_m/rho, max(0.01, p.radius_sigma_m/p.radius_m))
    if depth is not None and use_depth:
        # depth ON the wire itself (entrance hoop only): pose- and floor-independent distance. Sample the
        # unit-scale model circle, take the nearest return in a 3x3 window (thin wire in front of a farther
        # background) and require a tight, consistent majority spread over >= 3 quadrants of the circle (a
        # fabric flap over the top alone must not decide it). A VIO height drift of a few cm biases the
        # floor-based scale by ~10 % (ubuntu-compat v4B flight); this source does not care.
        eh = np.array([-math.sin(psi), math.cos(psi), 0.0])
        th = np.linspace(0, 2*np.pi, 96, endpoint=False)
        P = cam + d + rho*(np.cos(th)[:, None]*eh + np.sin(th)[:, None]*np.array([0, 0, 1.0]))
        uv, z = _project(Km, R_cn, cam, P)
        Hd, Wd = depth.shape
        okp = (z > 0.05) & (uv[:, 0] >= 2) & (uv[:, 0] < Wd - 2) & (uv[:, 1] >= 2) & (uv[:, 1] < Hd - 2)
        if okp.sum() >= 12:
            vals = _sample_depth(depth, uv[okp], rad=1)
            vals = np.where(np.isfinite(vals) & (vals > 0.05), vals, np.inf)
            r = vals.min(1)/z[okp]
            quad = ((th[okp] // (np.pi/2)).astype(int)) % 4
            fin = np.isfinite(r)
            r, quad = r[fin], quad[fin]
            if len(r) >= 12:
                med = float(np.median(r))
                good = np.abs(r - med) < 0.06*med
                if good.sum() >= 12 and good.mean() >= 0.5 and len(set(quad[good].tolist())) >= 3:
                    med = float(np.median(r[good]))
                    est['wire'] = (med, 0.015 + 0.01*med)
    if depth is not None and (use_depth or not est):
        eh = np.array([-math.sin(psi), math.cos(psi), 0.0])
        th = -np.pi/2 + np.linspace(-0.8, 0.8, 17)       # upper arc (fabric drapes over it)
        P = cam + d + rho*(np.cos(th)[:, None]*eh + np.sin(th)[:, None]*np.array([0, 0, 1.0]))
        uv, z = _project(Km, R_cn, cam, P)
        okp = z > 0
        if okp.sum() >= 5:
            samp = []
            for dv in (0.0, -3.0, -6.0):
                vals = _sample_depth(depth, uv[okp] + np.array([0, dv]))
                samp.append(vals/z[okp, None])
            r = np.concatenate([s.ravel() for s in samp])
            r = r[np.isfinite(r) & (r > 0)]
            if len(r) >= 8:
                med = float(np.median(r))
                mad = float(np.median(np.abs(r - med)))
                if mad < 0.08*med and np.mean(np.abs(r - med) < 0.1*med) > 0.5:
                    est['depth'] = (med, 0.04 + 0.02*med)
    if not est:
        return None, 'none', est
    tag = ''
    if 'depth' in est and len(est) >= 2:
        other = [k for k in est if k != 'depth']
        w = np.array([1/est[k][1]**2 for k in other])
        m0 = float(np.sum(w*np.array([math.log(est[k][0]) for k in other]))/np.sum(w))
        if abs(math.log(est['depth'][0]) - m0) > p.depth_gate:
            est = dict(est)
            est['depth_rejected'] = est.pop('depth')
            tag = '(depth-conflict)'
    keys = [k for k in est if k != 'depth_rejected']
    if len(keys) == 1:
        return est[keys[0]][0], keys[0] + tag, est
    # consensus: the largest set of sources that agree pairwise within 2.5 sigma (ties: more weight).
    # A drifted floor input is outvoted by depth on the wire + fabric; a wire estimate that slid off the
    # wire is outvoted by floor + measured radius.
    lt = {k: math.log(est[k][0]) for k in keys}
    sg = {k: est[k][1] for k in keys}

    def agree(a, b):
        return abs(lt[a] - lt[b]) <= 2.5*math.sqrt(sg[a]**2 + sg[b]**2)
    best = None
    for mask in range(1, 1 << len(keys)):
        sub = [keys[i] for i in range(len(keys)) if mask >> i & 1]
        if all(agree(a, b) for i, a in enumerate(sub) for b in sub[i + 1:]):
            key = (len(sub), sum(1/sg[k]**2 for k in sub))
            if best is None or key > best[0]:
                best = (key, sub)
    sub = best[1]
    w = np.array([1/sg[k]**2 for k in sub])
    m = float(np.sum(w*np.array([lt[k] for k in sub]))/np.sum(w))
    dropped = [k for k in keys if k not in sub]
    name = '+'.join(sub) + ('(-%s)' % ','.join(dropped) if dropped else '')
    if len(sub) == 1 and dropped:
        name = sub[0] + '(conflict)'
    return math.exp(m), name + tag, est


def _interior_is_solid(e, depth, t_center):
    """True if the inner half of the ellipse is a flat surface at about hoop depth
    (a printed / painted circle), never true for an open tunnel mouth. (A board right in front of
    a real hoop can also trigger it; making the test plane-aware was tried on the venue replica
    and let more false circles through than it recovered, so it stays conservative.)"""
    if depth is None:
        return False
    g = np.linspace(-0.5, 0.5, 9)
    gx, gy = np.meshgrid(g, g)
    m = gx**2 + gy**2 <= 0.25
    ca, sa = math.cos(e.angle), math.sin(e.angle)
    a1, a2 = e.axes_px
    x = e.center_px[0] + a1*gx[m]*ca - a2*gy[m]*sa
    y = e.center_px[1] + a1*gx[m]*sa + a2*gy[m]*ca
    vals = _sample_depth(depth, np.stack([x, y], -1), rad=0)[:, 0]
    ok = np.isfinite(vals) & (vals > 0.05)
    if ok.mean() < 0.6:
        return False
    v = vals[ok]
    med = float(np.median(v))
    iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
    return iqr < 0.1*med and (t_center is None or abs(med - t_center) < 0.3*t_center)


def _conic_inliers(Q, rd, tol_px, p):
    """Ridge pixels within tol_px of a pixel conic and tangent-consistent with it."""
    U = np.column_stack([rd.xy, np.ones(len(rd.xy))])
    QU = U @ Q
    f = np.einsum('ij,ij->i', U, QU)
    g = QU[:, :2]
    gn = np.hypot(g[:, 0], g[:, 1]) + 1e-12
    sin_tol = math.sin(math.radians(p.tangent_tol_deg))
    return (np.abs(f)/(2*gn) < tol_px) & (np.abs(g[:, 0]*rd.t[:, 0] + g[:, 1]*rd.t[:, 1])/gn < sin_tol)


def _fit_ring(e, rd, frame, Km, R_cn, cam, floor_z, p, depth):
    U = rd.xy[e.inliers]
    if len(U) > p.fit_points:
        cpos = e.center_px
        phi = np.arctan2(U[:, 1] - cpos[1], U[:, 0] - cpos[0])
        U = U[np.argsort(phi)][np.linspace(0, len(U) - 1, p.fit_points).astype(int)]
    Uh = np.column_stack([U, np.ones(len(U))])
    # initialisation from the ellipse: centre ray, vertical / horizontal half extents
    a1, a2 = e.axes_px
    ca, sa = math.cos(e.angle), math.sin(e.angle)
    hw = math.sqrt((a1*ca)**2 + (a2*sa)**2)
    hh = math.sqrt((a1*sa)**2 + (a2*ca)**2)
    fx, fy, cx, cy = frame.K
    dc = np.array([(e.center_px[0] - cx)/fx, (e.center_px[1] - cy)/fy, 1.0])
    dc /= np.linalg.norm(dc)
    dn = R_cn.T @ dc
    alpha = math.atan2(dn[1], dn[0])
    beta = math.asin(np.clip(dn[2], -1, 1))
    rho0 = hh*dc[2]/fy
    delta = math.acos(min(1.0, hw/max(hh, 1e-6)))
    delta = max(delta, math.radians(8))
    KR = Km @ R_cn
    x, rms, fr, cost = _fit_shape(Uh, [alpha, beta, alpha + delta, rho0], KR)
    if rms > 2*p.max_rms_px and fr < 0.4:           # clearly not a vertical circle: stop early
        return None, 'rms %.2f inl %.2f' % (rms, fr)
    # mirror (+/- yaw) solution: only meaningful when the hoop is seen obliquely
    dd = _dir(x[0], x[1])
    los0 = dd[:2]/max(np.linalg.norm(dd[:2]), 1e-9)
    view0 = math.degrees(math.acos(min(1.0, abs(math.cos(x[2])*los0[0] + math.sin(x[2])*los0[1]))))
    alt_x, alt_cost = x, cost
    if view0 > 6.0:
        xa, rmsa, fra, costa = _fit_shape(Uh, [alpha, beta, alpha - delta, rho0], KR)
        if costa < cost:
            x, rms, fr, cost, alt_x, alt_cost = xa, rmsa, fra, costa, x, cost
        else:
            alt_x, alt_cost = xa, costa
    dpsi = abs(_wrap(2*(alt_x[2] - x[2])))/2       # yaw difference modulo pi
    ambiguous = bool(dpsi > math.radians(6) and alt_cost < 1.5*cost + 0.5)
    # re-select inliers tightly around the fitted conic from ALL ridge pixels and refit: removes
    # points of a neighbouring coaxial hoop a few px away (a 3 % size bias = ~4 % distance bias)
    m = _conic_inliers(_model_conics(x, KR)[0], rd, 1.2, p)
    if m.sum() >= 12:
        U = rd.xy[m]
        if len(U) > p.fit_points:
            U = U[np.linspace(0, len(U) - 1, p.fit_points).astype(int)]
        Uh = np.column_stack([U, np.ones(len(U))])
        x2, rms2, fr2, cost2 = _fit_shape(Uh, x, KR)
        if rms2 <= max(rms, 0.8):
            x, rms, fr = x2, rms2, fr2
    if rms > p.max_rms_px or fr < 0.6:
        return None, 'rms %.2f inl %.2f' % (rms, fr)
    t, src, det = _scale_estimates(x, frame, Km, R_cn, cam, floor_z, p, depth)
    if t is None:
        return None, 'no scale'
    d = _dir(x[0], x[1])
    C = cam + t*d
    r = t*x[3]
    n = np.array([math.cos(x[2]), math.sin(x[2])])
    los = d[:2]/max(np.linalg.norm(d[:2]), 1e-9)
    if n @ los < 0:
        n = -n
    na = np.array([math.cos(alt_x[2]), math.sin(alt_x[2])])
    if na @ los < 0:
        na = -na
    view = math.degrees(math.acos(np.clip(n @ los, -1, 1)))
    if not (p.radius_range_m[0] <= r <= p.radius_range_m[1]):
        return None, 'radius %.2f' % r
    if not (p.range_m[0] <= t <= p.range_m[1]):
        return None, 'range %.2f' % t
    if view > p.max_view_angle_deg:
        return None, 'view %.0f' % view
    if floor_z is not None and not (-0.05 < floor_z - C[2] < 1.6 + r):
        return None, 'height %.2f' % (floor_z - C[2])
    if _interior_is_solid(e, depth, t*(R_cn @ d)[2]):
        return None, 'solid interior'
    score = e.support*(0.3 + e.cover)*math.exp(-0.5*(rms/1.0)**2)
    return Ring(C, n, r, t, rms, fr, view, ambiguous, na, src, det, e, score, x, U), 'ok'


# ----------------------------------------------------------------------------- 4. groups
def _compatible(a, b, p):
    na = [a.axis_ned] + ([a.alt_axis_ned] if a.ambiguous else [])
    nb = [b.axis_ned] + ([b.alt_axis_ned] if b.ambiguous else [])
    cos_t = math.cos(math.radians(p.group_axis_deg))
    pairs = [(u, v) for u in na for v in nb if abs(u @ v) >= cos_t]
    if not pairs:
        return False
    if abs(a.radius_m - b.radius_m) > p.group_radius_frac*max(a.radius_m, b.radius_m):
        return False
    dC = b.center_ned[:2] - a.center_ned[:2]
    for u, v in pairs:
        m = u + np.sign(u @ v)*v
        m /= np.linalg.norm(m)
        along = dC @ m
        lat = abs(dC @ np.array([-m[1], m[0]]))
        if not (p.group_min_span_m < abs(along) < p.group_max_span_m):
            continue
        if lat > p.group_lateral_m + math.tan(math.radians(12))*abs(along):
            continue
        if abs(a.center_ned[2] - b.center_ned[2]) > p.group_height_m + 0.05*abs(along):
            continue
        return True
    return False


def _line_fit(C):
    """Principal direction and centroid of 2-D points."""
    c0 = C.mean(0)
    _, _, vt = np.linalg.svd(C - c0, full_matrices=False)
    return vt[0], c0


def _normal_sigma(r):
    """Rough 1-sigma (rad) of a hoop's normal yaw: ill-conditioned head-on, scaled by fit rms."""
    return math.radians(1.0)/max(math.sin(math.radians(r.view_angle_deg)), 0.15) * \
        (3 if r.ambiguous else 1)*max(1.0, r.rms_px/0.6)


def _fuse_axis(rs, Cs):
    """Group axis from the centre line and the hoop normals, weighted by rough 1-sigma models.
    Per-hoop distance errors (~3 %) move centres along the line of sight, i.e. sideways to the
    axis when the tunnel is seen obliquely; normals are good when oblique, poor head-on.
    Returns axis, line sigma, span, normalised normal residuals."""
    u, c0 = _line_fit(Cs)
    span = float(np.ptp((Cs - c0) @ u))
    sv = math.sin(math.radians(float(np.mean([r.view_angle_deg for r in rs]))))
    sig_line = math.sqrt(2)*(0.03*sv + 0.01)/max(span, 0.1)
    vecs, ws, ns, sg = [u], [1/sig_line**2], [], []
    for r in rs:
        n = r.axis_ned
        if r.ambiguous and abs(r.alt_axis_ned @ u) > abs(n @ u):
            n = r.alt_axis_ned
        sig = _normal_sigma(r)
        vecs.append(n)
        ws.append(1/sig**2)
        ns.append(n)
        sg.append(sig)
    axis = _axis_mean(vecs, ws)
    res = np.array([math.acos(min(1.0, abs(float(n @ axis))))/s for n, s in zip(ns, sg)])
    return axis, sig_line, span, res


def _axis_mean(vecs, weights):
    """Sign-invariant weighted mean direction of 2-D unit vectors (doubled angle)."""
    s = sum(w*np.array([math.cos(2*math.atan2(v[1], v[0])), math.sin(2*math.atan2(v[1], v[0]))])
            for v, w in zip(vecs, weights))
    ang = 0.5*math.atan2(s[1], s[0])
    return np.array([math.cos(ang), math.sin(ang)])


def _floor_from_depth(frame, R_nc, cam):
    d = frame.depth
    if d is None:
        return None
    fx, fy, cx, cy = frame.K
    st = 6
    dd = d[::st, ::st]
    vv, uu = np.mgrid[0:d.shape[0]:st, 0:d.shape[1]:st]
    ok = np.isfinite(dd) & (dd > 0.1) & (dd < 5.0)
    if ok.sum() < 60:
        return None
    z = dd[ok]
    zn = ((uu[ok] - cx)/fx*R_nc[2, 0] + (vv[ok] - cy)/fy*R_nc[2, 1] + R_nc[2, 2])*z + cam[2]
    zn = zn[zn > cam[2] + 0.25]
    if len(zn) < 60:
        return None
    hist, edges = np.histogram(zn, bins=np.arange(zn.min(), zn.max() + 0.02, 0.02))
    if len(hist) == 0:
        return None
    k = int(np.argmax(hist))
    zc = 0.5*(edges[k] + edges[k + 1])
    band = zn[np.abs(zn - zc) < 0.03]
    if len(band) < 30 or len(band) < 0.25*len(zn):
        return None
    return float(np.mean(band))


def detect(frame, prior_entry_ned=None, *, floor_z_ned=None, prior_axis_ned=None, params=None):
    """Tunnel entrance from the brass hoops.

    frame          : common.geometry.Frame (grayscale image, depth aligned to it, K, R_nc, cam_ned)
    prior_entry_ned: coarse measured entrance centre (NED, +-0.5 m) -> search ROI + score prior + gate
    floor_z_ned    : floor height in NED (e.g. from the downward ToF or floor.fit_floor); if None it is
                     estimated from depth, else params.default_floor_z_ned
    prior_axis_ned : optional coarse tunnel heading (2,) used only to resolve a mirror ambiguity
    """
    p = params or TunnelParams()
    t0 = time.perf_counter()
    img = frame.image
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    R_nc = np.asarray(frame.R_nc, float)
    R_cn = R_nc.T
    cam = np.asarray(frame.cam_ned, float)
    Km = _kmat(frame.K)
    depth = None
    if frame.depth is not None:
        depth = np.asarray(frame.depth, np.float32)
        if depth.shape != img.shape[:2] or not np.isfinite(depth).any():
            depth = None
    dbg = {}
    floor_src = 'arg'
    floor_z = floor_z_ned
    if floor_z is None:
        floor_z = _floor_from_depth(frame, R_nc, cam) if depth is not None else None
        floor_src = 'depth'
        if floor_z is None:
            floor_z, floor_src = p.default_floor_z_ned, 'default'
    dbg['floor_z'] = floor_z
    dbg['floor_src'] = floor_src
    roi = None
    prior = None if prior_entry_ned is None else np.asarray(prior_entry_ned, float)
    if prior is not None:
        uv, z = _project(Km, R_cn, cam, prior[None])
        if z[0] > 0.3:
            R = Km[0, 0]*(p.prior_gate_m + p.radius_range_m[1])/z[0]
            roi = (uv[0, 0] - R, uv[0, 1] - R, uv[0, 0] + R, uv[0, 1] + R)
    t1 = time.perf_counter()
    rd = extract_ridges(img, p)
    t2 = time.perf_counter()
    g_c = R_cn[:, 2]                                 # gravity (NED down) in the optical frame
    roll = 0.5*np.pi - math.atan2(g_c[1], g_c[0])
    cands, rd = detect_ellipses(img, p, roi=roi, ridges=rd, roll=roll)
    t3 = time.perf_counter()
    rings, why = [], []
    for e in cands:
        ring, msg = _fit_ring(e, rd, frame, Km, R_cn, cam, floor_z, p, depth)
        why.append(msg)
        if ring is not None:
            rings.append(ring)
    t4 = time.perf_counter()
    dbg.update(n_ridge=len(rd.xy), n_cand=len(cands), reject=why, roi=roi)
    if not rings:
        dbg['ms'] = dict(total=1e3*(time.perf_counter() - t0), ridge=1e3*(t2 - t1),
                         ellipse=1e3*(t3 - t2), fit3d=1e3*(t4 - t3))
        return _not_found(dbg, cands)
    # coaxial groups (greedy, best ring first)
    order = sorted(range(len(rings)), key=lambda i: -rings[i].score)
    groups, used = [], set()
    for i in order:
        if i in used:
            continue
        g = [i]
        used.add(i)
        for j in order:
            if j not in used and all(_compatible(rings[j], rings[k], p) for k in g):
                g.append(j)
                used.add(j)
        groups.append(g)
    best = None
    for g in groups:
        rs = [rings[i] for i in g]
        res = _resolve_group(rs, frame, Km, R_cn, cam, floor_z, p, depth, prior, prior_axis_ned, rd)
        if res is None:
            continue
        if best is None or res['score'] > best['score']:
            best = res
    dbg['n_rings'] = len(rings)
    dbg['n_groups'] = len(groups)
    dbg['ms'] = dict(total=1e3*(time.perf_counter() - t0), ridge=1e3*(t2 - t1),
                     ellipse=1e3*(t3 - t2), fit3d=1e3*(t4 - t3))
    if best is None:
        return _not_found(dbg, cands)
    dbg.update(best['debug'])
    found = best['confidence'] >= p.min_confidence
    rs = best['rings']
    align_ok = bool(found and (len(rs) >= 2 or rs[0].view_angle_deg <= p.align_max_view_deg))
    return TunnelEstimate(found, best['entry'], best['axis'], best['radius'], best['confidence'],
                          rs, cands, best['scale_source'], best['axis_source'], dbg, align_ok)


def _extend_front(rs, axis, rd, frame, Km, R_cn, cam, floor_z, p, depth):
    """Coaxial lattice + extrapolate-and-verify.
    Hoops sit on a regular lattice along the axis (spacing s). Candidate spacings come from the
    measured value or from all pairwise gaps / 1, 2, 3. For each s: keep the hoops on the lattice
    of the most reliable hoop (off-lattice fits are hybrids across two hoops), then predict hoops at
    1..4 spacings nearer to the camera, test each predicted conic directly against the ridge pixels
    and refine it if supported. The best-explained lattice wins. Recovers the entrance when the
    2-D stage missed it (hoops a few px apart far away, blur, low contrast).
    Returns (rings sorted along the axis, n_added, stop reason)."""
    al = np.array([r.center_ned[:2] @ axis for r in rs])
    radius = float(np.median([r.radius_m for r in rs]))
    q = [r.rms_px*(1 + 3*abs(r.radius_m - radius)/radius)*(2 if r.ambiguous else 1) for r in rs]
    iref = int(np.argmin(q))
    ref = rs[iref]
    if p.hoop_spacing_m is not None:
        cands = [p.hoop_spacing_m]
    else:
        cands = []
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                g = abs(al[i] - al[j])
                for n in (1, 2, 3):
                    if g/n >= p.min_hoop_spacing_m and all(abs(g/n - c) > 0.03 for c in cands):
                        cands.append(g/n)
        cands = sorted(cands, reverse=True)[:6]
    if not cands:
        return rs, 0, 'no spacing', None
    psi = math.atan2(axis[1], axis[0])
    KR = Km @ R_cn
    U = np.column_stack([rd.xy, np.ones(len(rd.xy))])
    sin_tol = math.sin(math.radians(p.tangent_tol_deg))
    # 1. spacing = the lattice that explains the most hoop support with the fewest empty slots
    best = None
    for sp in cands:
        kk = (al - al[iref])/sp
        on = np.abs(kk - np.round(kk)) <= 0.2
        lat = {}
        for i in np.nonzero(on)[0]:                  # one hoop per lattice slot (best support)
            k = int(round(kk[i]))
            if k not in lat or rs[i].ellipse.support > lat[k].ellipse.support:
                lat[k] = rs[i]
        ks = sorted(lat)
        empty = (ks[-1] - ks[0] + 1) - len(ks)
        sup = [r.ellipse.support for r in lat.values()]
        score = sum(sup) - 0.6*empty*float(np.mean(sup))
        if best is None or score > best[0] + 1e-9:
            best = (score, sp, [lat[k] for k in ks])
    _, sp0, kept = best
    # 2. extrapolate-and-verify in front of the front-most lattice hoop, chaining; pixels within
    #    1 px of an accepted hoop's fitted conic cannot support a new one (no phantoms)
    used = np.zeros(len(rd.xy), bool)
    for r in kept:
        used |= _conic_inliers(_model_conics(r.shape, KR)[0], rd, 1.0, p)
    for sp in (sp0, sp0/2, sp0/3):                     # a missing middle hoop inflates the spacing
        if sp < p.min_hoop_spacing_m:
            break
        added, why, u2 = [], '', used.copy()
        for _ in range(4):
            prev = added[-1] if added else kept[0]
            res = _test_front(prev, sp, axis, psi, radius, U, rd, KR, frame, Km, R_cn, cam, floor_z,
                              p, depth, sin_tol, rd.shape, u2)
            if isinstance(res, str):
                why = res
                break
            added.append(res)
            u2 |= _conic_inliers(_model_conics(res.shape, KR)[0], rd, 1.0, p)
        if added:
            break
    out = kept + added
    out.sort(key=lambda r: r.center_ned[:2] @ axis)
    return out, len(added), why + (' s=%.2f' % sp), (p.hoop_spacing_m or sp0)


def _test_front(ent, spacing, axis, psi, radius, U, rd, KR, frame, Km, R_cn, cam, floor_z, p, depth,
                sin_tol, shape, used):
    """Verify a hoop predicted `spacing` in front of `ent`. The prediction inherits the group's
    axis error (inner hoops are partly hidden by the near side wall, so their normals can be off
    by several degrees), so a small grid of yaw / lateral offsets is searched before the LM refit
    (yaw free). Returns a Ring or a reason string."""
    C0 = ent.center_ned - spacing*np.array([axis[0], axis[1], 0.0])
    if (cam[:2] - C0[:2]) @ axis > -0.3:              # camera at / past that plane: nothing in front
        return 'camera past plane'
    if float(np.linalg.norm(C0 - cam)) < p.range_m[0]:
        return 'too close'
    side = np.array([-axis[1], axis[0], 0.0])
    X = []
    for dl in (-0.08, -0.04, 0.0, 0.04, 0.08):
        C = C0 + dl*side
        t = float(np.linalg.norm(C - cam))
        d = (C - cam)/t
        for dp in (-10, -5, 0, 5, 10):
            X.append([math.atan2(d[1], d[0]), math.asin(np.clip(d[2], -1, 1)), psi + math.radians(dp), radius/t])
    X = np.array(X)
    Qs = _model_conics(X, KR)
    # only ridge pixels inside the (padded) box of the central prediction
    Qc = Qs[len(X)//2]
    okc, ctr, a1, a2, ang = _ellipse_params((Qc[[0, 0, 1, 0, 1, 2], [0, 1, 1, 2, 2, 2]]*np.array([1, 2, 1, 2, 2, 1]))[None])
    if not okc[0]:
        return 'bad prediction'
    R = 1.25*max(a1[0], a2[0]) + 12
    box = np.nonzero((np.abs(rd.xy[:, 0] - ctr[0, 0]) < R) & (np.abs(rd.xy[:, 1] - ctr[0, 1]) < R) & ~used)[0]
    if len(box) < 12:
        return 'few points %d' % len(box)
    Ub = U[box]
    QU = np.einsum('bij,mj->bmi', Qs, Ub)
    f = np.einsum('bmi,mi->bm', QU, Ub)
    g = QU[..., :2]
    gn = np.hypot(g[..., 0], g[..., 1]) + 1e-12
    tb = rd.t[box]
    M = (np.abs(f)/(2*gn) < 2.0) & (np.abs(g[..., 0]*tb[None, :, 0] + g[..., 1]*tb[None, :, 1])/gn < sin_tol)
    nb = M.sum(1)
    kbest = int(np.argmax(nb))
    if nb[kbest] < 12:
        return 'few points %d' % nb[kbest]
    x = X[kbest]
    m = np.zeros(len(U), bool)
    m[box[M[kbest]]] = True
    for tol in (1.6, 1.2):                            # refit (yaw free), re-select, refit
        xf, rms, fr, _ = _fit_shape(U[m], x, KR)
        if rms > p.max_rms_px or fr < 0.6:
            return 'fit rms %.2f inl %.2f' % (rms, fr)
        x = xf
        m = _conic_inliers(_model_conics(x, KR)[0], rd, tol, p)
        if (m & used).sum() > 0.6*max(1, m.sum()):     # mostly another hoop's pixels: phantom
            return 'overlaps a hoop'
        m &= ~used
        if m.sum() < 12:
            return 'few points after refit'
    Q = _model_conics(x, KR)[0]
    coef = Q[[0, 0, 1, 0, 1, 2], [0, 1, 1, 2, 2, 2]]*np.array([1, 2, 1, 2, 2, 1])
    ok, ctr, a1, a2, ang = _ellipse_params(coef[None])
    if not ok[0]:
        return 'bad conic'
    sup, vis = _pix_coverage(rd.xy[m], ctr[0], a1[0], a2[0], ang[0], shape, p.n_bins)
    if vis < p.min_visible_frac*p.n_bins or sup < p.min_support_bins or sup < 0.3*vis:
        return 'support %d/%d' % (sup, vis)
    ts, src, det = _scale_estimates(x, frame, Km, R_cn, cam, floor_z, p, depth)
    if ts is None:
        return 'no scale'
    Cn = cam + ts*_dir(x[0], x[1])
    rn = ts*x[3]
    if abs(rn - radius) > p.group_radius_frac*radius:
        return 'radius %.2f vs %.2f' % (rn, radius)
    if np.linalg.norm(Cn - C0) > 0.2*spacing + 0.06:
        return 'moved %.2f' % np.linalg.norm(Cn - C0)
    n = np.array([math.cos(x[2]), math.sin(x[2])])
    if n @ axis < 0:
        n = -n
    if math.degrees(math.acos(min(1.0, abs(float(n @ axis))))) > 15:
        return 'normal off'
    e = Ellipse2D(coef, ctr[0], np.array([a1[0], a2[0]]), float(ang[0]), int(sup), int(vis),
                  sup/max(vis, 1), np.nonzero(m)[0], sup*(0.3 + sup/max(vis, 1)))
    d = _dir(x[0], x[1])
    los = d[:2]/max(np.linalg.norm(d[:2]), 1e-9)
    view = math.degrees(math.acos(np.clip(abs(n @ los), -1, 1)))
    Um = rd.xy[m]
    if len(Um) > p.fit_points:
        Um = Um[np.linspace(0, len(Um) - 1, p.fit_points).astype(int)]
    return Ring(Cn, n, rn, ts, rms, fr, view, False, n.copy(), src, det, e,
                e.score*math.exp(-0.5*(rms/1.0)**2), x, Um)


def _pix_coverage(xy, ctr, a1, a2, ang, shape, nb):
    """Covered / visible angular bins of one pixel-space ellipse."""
    ca, sa = math.cos(ang), math.sin(ang)
    rx, ry = xy[:, 0] - ctr[0], xy[:, 1] - ctr[1]
    phi = np.arctan2((-rx*sa + ry*ca)/a2, (rx*ca + ry*sa)/a1)
    b = ((phi + np.pi)*(nb/(2*np.pi))).astype(np.int64) % nb
    hit = np.zeros(nb, bool)
    hit[b] = True
    ph = -np.pi + (np.arange(nb) + 0.5)*(2*np.pi/nb)
    ex = ctr[0] + a1*np.cos(ph)*ca - a2*np.sin(ph)*sa
    ey = ctr[1] + a1*np.cos(ph)*sa + a2*np.sin(ph)*ca
    H, W = shape
    vis = (ex >= 0) & (ex < W) & (ey >= 0) & (ey < H)
    return int((hit & vis).sum()), int(vis.sum())


def _hoop_observable(C, axis, radius, Km, R_cn, cam, shape, p):
    """Would a hoop at C be seen by the detector? (in front of the camera, not too close, and at
    least half of its circle inside the image)"""
    if (cam[:2] - C[:2]) @ axis > -0.05:
        return False
    if float(np.linalg.norm(C - cam)) < p.range_m[0]:
        return False
    th = np.linspace(0, 2*np.pi, 72, endpoint=False)
    eh = np.array([-axis[1], axis[0], 0.0])
    P = C + radius*(np.cos(th)[:, None]*eh + np.sin(th)[:, None]*np.array([0.0, 0.0, 1.0]))
    uv, z = _project(Km, R_cn, cam, P)
    H, W = shape
    inside = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return bool(inside.mean() >= 0.5)


def _resolve_group(rs, frame, Km, R_cn, cam, floor_z, p, depth, prior, prior_axis, rd):
    """Axis, entrance hoop and confidence for one coaxial group."""
    dbgs = {}
    los = rs[0].center_ned[:2] - cam[:2]
    if len(rs) >= 2:
        Cs = np.array([r.center_ned[:2] for r in rs])
        if len(rs) >= 3:                              # pair-RANSAC: the line with most hoops on it
            best_in, best_cost = None, None
            for i in range(len(rs)):
                for j in range(i + 1, len(rs)):
                    u = Cs[j] - Cs[i]
                    if np.linalg.norm(u) < 1e-6:
                        continue
                    u = u/np.linalg.norm(u)
                    d = np.abs((Cs - Cs[i]) @ np.array([-u[1], u[0]]))
                    inl = d <= p.line_outlier_m
                    w = np.array([r.score for r in rs])
                    key = (int(inl.sum()), float(w[inl].sum()))
                    cost = float(np.sum(np.minimum(d, p.line_outlier_m)))
                    if best_in is None or key > best_key or (key == best_key and cost < best_cost):
                        best_in, best_key, best_cost = inl, key, cost
            if best_in is not None and best_in.sum() >= 2 and not best_in.all():
                dbgs['dropped_off_line'] = int((~best_in).sum())
                rs = [r for r, k in zip(rs, best_in) if k]
                Cs = Cs[best_in]
        while True:
            axis, sig_line, span, res = _fuse_axis(rs, Cs)
            k = int(np.argmax(res))
            if len(rs) <= 2 or res[k] <= 4.0:
                break
            dbgs.setdefault('dropped_normal', []).append(round(float(res[k]), 1))
            rs = [r for i, r in enumerate(rs) if i != k]
            Cs = np.delete(Cs, k, 0)
        axis_src = 'centres(%d)+normals' % len(rs)
        dbgs['span_m'] = span
        dbgs['axis_line_sigma_deg'] = math.degrees(sig_line)
        dbgs['normal_resid_sigma'] = [round(float(x), 1) for x in res]
    else:
        r = rs[0]
        if r.ellipse.support < p.single_min_support or r.ellipse.cover < p.single_min_cover:
            return None
        axis = r.axis_ned
        axis_src = 'normal'
        if r.ambiguous:
            axis_src = 'normal(ambiguous)'
            if prior_axis is not None:
                pa = np.asarray(prior_axis, float)
                pa = pa/np.linalg.norm(pa)
                if abs(r.alt_axis_ned @ pa) > abs(axis @ pa):
                    axis = r.alt_axis_ned
                axis_src = 'normal(prior-resolved)'
    if axis @ los < 0:
        axis = -axis
    along = [r.center_ned[:2] @ axis for r in rs]
    order = np.argsort(along)
    rs = [rs[i] for i in order]
    spacing = p.hoop_spacing_m
    if len(rs) >= 2 or p.hoop_spacing_m is not None:
        rs, n_ext, why, sp_est = _extend_front(rs, axis, rd, frame, Km, R_cn, cam, floor_z, p, depth)
        spacing = spacing or sp_est
        dbgs['extended_front'] = n_ext
        dbgs['extend_stop'] = why
        if n_ext and len(rs) >= 2:                    # the recovered entrance improves the axis
            axis2, _, _, _ = _fuse_axis(rs, np.array([r.center_ned[:2] for r in rs]))
            axis = axis2 if axis2 @ axis > 0 else -axis2
            rs.sort(key=lambda r: r.center_ned[:2] @ axis)
    ent = rs[0]
    # refit the entrance hoop with the group axis held fixed (better centre when head-on)
    psi = math.atan2(axis[1], axis[0])
    entry, radius, src = ent.center_ned, ent.radius_m, ent.scale_source
    rms = ent.rms_px
    U = ent.fit_px
    x = ent.shape
    if U is not None and len(U) >= 8 and abs(_wrap(2*(psi - ent.shape[2]))) > 1e-6:
        Uh = np.column_stack([U, np.ones(len(U))])
        x2, rms2, fr2, _ = _fit_shape(Uh, ent.shape, Km @ R_cn, fix_psi=psi)
        if rms2 < max(p.max_rms_px, 1.3*ent.rms_px):
            x, rms = x2, rms2
    t, src2, det = _scale_estimates(x, frame, Km, R_cn, cam, floor_z, p, depth, use_depth=True)
    if t is not None:                                 # entrance: fabric depth allowed (gated)
        entry = cam + t*_dir(x[0], x[1])
        radius = t*x[3]
        src = src2
        dbgs['entry_scale'] = {k: (round(v[0], 3), round(v[1], 3)) for k, v in det.items()}
        # the other hoops were scaled from the same floor input; if depth on the entrance says that input is
        # biased (VIO height drift, wrong floor), apply the same correction to them (positions, radii)
        used = src.split('(')[0].split('+')
        if 'floor' in det and 'wire' in used and abs(math.log(t/det['floor'][0])) > 0.03:
            kf = t/det['floor'][0]
            for r in rs[1:]:
                if r.scale_source.startswith('floor'):
                    r.center_ned = cam + kf*(r.center_ned - cam)
                    r.radius_m *= kf
                    r.distance_m *= kf
            dbgs['floor_scale_correction'] = round(kf, 3)
    radii = [r.radius_m for r in rs]
    if len(rs) >= 2:
        radius = float(np.median(radii + [radius]))
    # entrance back-inference (needs a prior and a spacing): close to the tunnel, or inside it, the
    # entrance hoop is larger than the field of view or behind the camera, and the front-most
    # VISIBLE hoop is the 2nd/3rd one. If the prior says the entrance is k spacings nearer and every
    # skipped lattice slot is unobservable (behind / at the camera, or < half inside the image),
    # step back k slots. A skipped slot that should have been visible stops the inference.
    n_inf, blocked = 0, False
    if prior is not None and spacing:
        a3 = np.array([axis[0], axis[1], 0.0])
        k = int(round(float((entry[:2] - prior[:2]) @ axis)/spacing))
        kmax = p.max_inferred_hoops if p.n_hoops is None else max(0, p.n_hoops - len(rs))
        # stepping 0.6 m along a weak head-on single-hoop normal moves the entry sideways
        axis_ok = len(rs) >= 2 or prior_axis is not None or _normal_sigma(rs[0]) <= math.radians(3)
        for j in range(1, k + 1):
            if _hoop_observable(entry - j*spacing*a3, axis, radius, Km, R_cn, cam, frame.image.shape[:2], p):
                break                                 # should have been seen: the prior is off
            if j > kmax or not axis_ok:
                blocked = True                        # hidden, but cannot be placed reliably
                break
            n_inf = j
        if n_inf:
            entry = entry - n_inf*spacing*a3
            dbgs['entry_inferred'] = n_inf
        if blocked:
            dbgs['entry_maybe_behind'] = k - n_inf
    # confidence
    def _cov(r):
        return min(1.0, r.ellipse.cover/0.45)*min(1.0, r.ellipse.support/30.0)
    cov = _cov(ent)
    if len(rs) >= 2:                                  # a huge, partly visible entrance confirmed by
        cov = max(cov, 0.85*max(_cov(r) for r in rs))  # the lattice: let the group's evidence count
    c_fit = math.exp(-0.5*(rms/1.2)**2)
    c_n = 1.0 if len(rs) >= 2 else 0.7
    c_amb = 0.6 if (len(rs) == 1 and ent.ambiguous and prior_axis is None) else 1.0
    c_scale = 1.0 if ('+' in src and 'conflict' not in src) else (0.85 if src in ('floor', 'radius', 'depth', 'wire') else 0.6)
    if src.startswith('wire') and 'conflict' in src:
        c_scale = 0.85                                 # direct depth on the hoop beat a (pose-dependent) floor scale
    if ent.scale_detail.get('floor') is None and p.radius_m is None and dbgs.get('floor_src') == 'default':
        c_scale *= 0.7
    c_prior = 1.0
    if prior is not None:
        dh = float(np.linalg.norm(entry[:2] - prior[:2]))
        dbgs['prior_dist_m'] = dh
        if dh > p.prior_gate_m:
            return None
        # no penalty inside the prior's own tolerance, decay beyond it
        c_prior = math.exp(-0.5*(max(0.0, dh - p.prior_tol_m)/p.prior_sigma_m)**2)
        if len(rs) == 1 and dh <= p.prior_tol_m:     # a lone hoop corroborated by the prior
            c_n = 0.85
    c_inf = (0.9 if n_inf else 1.0)*(0.6 if blocked else 1.0)
    conf = float(np.clip(cov*c_fit*c_n*c_amb*c_scale*c_prior*c_inf, 0, 1))
    score = sum(r.score for r in rs)*(1 + 0.3*(len(rs) - 1))*c_prior
    dbgs['entry_rms_px'] = rms
    return dict(entry=np.asarray(entry, float), axis=axis, radius=float(radius), confidence=conf,
                rings=rs, scale_source=src, axis_source=axis_src, score=score, debug=dbgs)


# ----------------------------------------------------------------------------- drawing
def draw(image, est, candidates=True):
    """BGR debug image: grey = 2-D candidates, green = entrance hoop, yellow = other hoops."""
    vis = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    vis = vis.copy()
    if candidates:
        for e in est.candidates:
            cv2.polylines(vis, [np.round(e.points()).astype(np.int32)], True, (160, 160, 160), 1, cv2.LINE_AA)
    for i, r in enumerate(est.rings):
        col = (0, 255, 0) if i == 0 else (0, 220, 255)
        cv2.polylines(vis, [np.round(r.ellipse.points()).astype(np.int32)], True, col, 2, cv2.LINE_AA)
    if est.found:
        txt = 'entry %.2f %.2f %.2f r=%.2f conf=%.2f' % (est.entry_ned[0], est.entry_ned[1], est.entry_ned[2],
                                                          est.radius_m, est.confidence)
        cv2.putText(vis, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return vis
