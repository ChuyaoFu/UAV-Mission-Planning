"""perception/window.py -- tan board with a rectangular opening ("window").

Contract (INTERFACES.md):
    detect(frame, half_span_m, half_height_m, prior_center_ned=None)
        -> WindowEstimate(found, center_ned (3,), normal_ned (3,), width_m, height_m,
                          clearance_ok, confidence)

Pipeline (depth first, the grey image assists):
  1. Board plane: RANSAC over depth points in gravity-aligned coordinates. A vertical plane
     is a line in the NED x/y plane, so hypotheses are 2-point lines (floor points scatter,
     the board collapses onto a line). Refit by least squares in INVERSE depth (1/z is
     linear in the ray for a plane and stereo noise is ~constant in 1/z), constrained
     vertical (boards stand on the floor). Normal points to the camera.
  2. Every depth ray (stride 2) is intersected with the plane; its hit gets plane
     coordinates (h to the observer's right, v down, metres) and a label from the measured
     depth: on-board / clearly beyond / no return / in front. Labels are binned into a
     metric grid ON THE BOARD, so oblique views give true sizes, not pixel sizes.
     Weak-texture board: no-return pixels whose grey level / local texture match this
     frame's on-plane pixels count as board, but only inside depth-board gaps < fill_m.
  3. Hole cells = observed, not (morphologically closed) board, not occluded in front.
     Each connected hole -> max-weight axis-aligned rectangle (hole +1, board -3). It is a
     window only if a ring around it is board on all four sides; a ring that is mostly
     outside the field of view => "cut_by_fov"; mostly open => the gap beside / below a
     board or a slit between boards, not a window.
  4. Image refinement: each edge is re-located with sub-pixel grey-level gradients along a
     profile sampled ON THE PLANE (depth edges are fattened/blocky; image edges are sharp).
  5. Fallbacks: (a) plane found but no depth hole -> image quads (find_window_quads: pairs
     of near-vertical / near-horizontal Hough lines, scored for 4-side edge support, a
     uniform ring and a grey-level step on every side) back-projected onto the plane; the
     inside must not be on the plane and no side may see through; (b) no plane but the
     window size was measured on site (known_size_m) -> planar PnP of the image quad.

Pure: numpy + OpenCV only, no ROS, no clock reads, deterministic (fixed RANSAC seed).
"""
from dataclasses import dataclass, field
import math
import warnings
from typing import Tuple

import cv2
import numpy as np

DOWN = np.array([0.0, 0.0, 1.0])


@dataclass
class WindowParams:
    stride_fit: int = 4            # depth stride for the plane RANSAC
    stride_grid: int = 2           # depth stride for hole labelling
    min_range: float = 0.25
    max_range: float = 6.0
    noise_a: float = 0.02          # on-plane threshold = a + b*z^2 (OAK-D 7.5 cm baseline)
    noise_b: float = 0.012
    beyond_min: float = 0.15       # "clearly beyond the board" (m)
    ransac_iters: int = 160
    ransac_sample: int = 3000
    seed: int = 7
    vertical_board: bool = True    # boards stand on the floor: fit a vertical plane
    floor_band_m: float = 0.15     # floor points (+- this) are kept out of the board fit
    max_tilt_deg: float = 12.0     # (vertical_board=False) tilt allowed before forcing vertical
    max_view_deg: float = 65.0     # max angle between board normal and line of sight
    min_plane_pts: int = 150
    max_planes: int = 3
    cell_min: float = 0.02
    cell_max: float = 0.06
    close_m: float = 0.06          # fill board-depth gaps narrower than this
    ring_m: float = 0.08           # enclosure ring width
    min_size_m: float = 0.20       # smaller openings are never the race window
    max_window_range: float = 4.5
    board_look: bool = True        # no-return pixels that LOOK like the board count as board
    fill_m: float = 0.30           # ... but only inside depth-board gaps smaller than this
    max_size_m: float = 1.6        # the venue opening is ~0.75 x 0.6 m
    max_aspect: float = 3.0        # width/height and height/width
    min_sill_m: float = 0.15       # opening bottom edge must be this far above the floor
    floor_clear_m: float = 0.10    # measured points this close to the floor are never board
    ring_outer_m: float = 0.05     # band beyond the ring that must not see through (thin rails)
    min_edges_no_prior: int = 2    # without a prior, >= this many edges must show on the image
    prior_search_m: float = 0.6    # known-size search radius around the prior on the board
    fallback_max_range: float = 3.5  # image fallbacks only when the prior is this close
    side_step_min: float = 25.0    # without a prior, >= min_step_sides edges must be a grey-level
    min_step_sides: int = 3        # step of this size (board vs scene behind), see _side_steps
    min_fill: float = 0.75
    board_frac: float = 0.5        # a cell is board if this fraction of its rays hit the plane
    ring_board_min: float = 0.55   # board fraction a side ring needs
    ring_fov_max: float = 0.30     # unobserved fraction that means "cut by the FOV"
    prior_gate_m: float = 1.5
    prior_sigma_m: float = 0.7
    edge_refine: bool = True
    edge_min_contrast: float = 10.0
    edge_mad_m: float = 0.008      # per-edge residual scatter allowed after the line fit (m)
    edge_max_tilt_deg: float = 10.0  # opening edges may be this far off level / plumb in the board
    min_conf: float = 0.25
    clear_min_conf: float = 0.4
    clear_max_range: float = 3.2   # beyond this the size is not trusted for clearance_ok
    size_unc_a: float = 0.003      # per-side size uncertainty a + b*range^2 (m), subtracted from
    size_unc_b: float = 0.0015     # the slack before comparing with margin_m (sim p95 ~ this)
    image_fallback: bool = True    # plane+image / PnP fallbacks (10-60 ms extra when they run)


@dataclass
class WindowEstimate:
    found: bool
    center_ned: np.ndarray
    normal_ned: np.ndarray         # unit, horizontal-ish, points from the board to the camera side
    width_m: float
    height_m: float
    clearance_ok: bool
    confidence: float
    corners_ned: np.ndarray = field(default_factory=lambda: np.full((4, 3), np.nan))  # TL,TR,BR,BL seen from the camera side
    right_ned: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))         # board axis to the observer's right
    lateral_slack_m: float = float('nan')    # width/2 - half_span (alignment budget incl. margin)
    vertical_slack_m: float = float('nan')   # height/2 - half_height
    range_m: float = float('nan')
    source: str = ''               # depth | depth+image | plane+image | image+size
    reason: str = ''               # why not found / why not clear
    sides: Tuple = ()              # enclosure status per side (left,right,top,bottom)
    plane_inliers: int = 0
    edges_refined: int = 0         # sides re-located on the grey image (0..4)
    inplane_tilt_deg: float = float('nan')  # opening's rotation in the board plane vs gravity (diagnostic)


def _nan3():
    return np.full(3, np.nan)


def _empty(reason, plane_inliers=0):
    return WindowEstimate(False, _nan3(), _nan3(), float('nan'), float('nan'), False, 0.0,
                          reason=reason, plane_inliers=plane_inliers)


def _gray(img):
    if img is None:
        return None
    img = np.asarray(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


_RAY_CACHE = {}


def _rays(shape, K, stride):
    """Optical-frame ray directions with z=1 for a strided pixel grid (cached)."""
    key = (shape, tuple(K), stride)
    r = _RAY_CACHE.get(key)
    if r is None:
        fx, fy, cx, cy = K
        vv, uu = np.mgrid[0:shape[0]:stride, 0:shape[1]:stride]
        r = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones(uu.shape)], -1).reshape(-1, 3)
        if len(_RAY_CACHE) > 16:
            _RAY_CACHE.clear()
        _RAY_CACHE[key] = r
    return r


def _thr(p, z):
    return p.noise_a + p.noise_b * z * z


# ----------------------------------------------------------------------------- plane

def _find_planes(P, z, cam, prior, p, floor_z=None):
    """Sequential vertical-plane RANSAC. P (N,3) NED points, z (N,) optical depth."""
    rng = np.random.default_rng(p.seed)
    thr = _thr(p, z)
    X = P[:, :2]
    qn = (P - cam) / z[:, None]                    # NED ray directions, optical z = 1
    iz = 1.0 / z
    # floor points lie along every vertical plane's foot line and tilt far / noisy fits
    remaining = (np.ones(len(P), bool) if floor_z is None else np.abs(P[:, 2] - floor_z) >= p.floor_band_m)
    cos_view = math.cos(math.radians(p.max_view_deg))
    sin_tilt = math.sin(math.radians(p.max_tilt_deg))
    planes = []
    for _ in range(p.max_planes):
        idx = np.flatnonzero(remaining)
        if len(idx) < p.min_plane_pts:
            break
        sub = idx if len(idx) <= p.ransac_sample else rng.choice(idx, p.ransac_sample, replace=False)
        i = sub[rng.integers(0, len(sub), p.ransac_iters)]
        j = sub[rng.integers(0, len(sub), p.ransac_iters)]
        a, b = X[i], X[j]
        dv = b - a
        L = np.hypot(dv[:, 0], dv[:, 1])
        ok = L > 0.15
        n2 = np.stack([-dv[:, 1], dv[:, 0]], 1) / np.maximum(L, 1e-9)[:, None]
        dd = np.einsum('ij,ij->i', n2, a)
        mid = 0.5 * (a + b) - cam[:2]
        los = mid / np.maximum(np.hypot(mid[:, 0], mid[:, 1]), 1e-9)[:, None]
        ok &= np.abs(np.einsum('ij,ij->i', n2, los)) >= cos_view
        if prior is not None:
            ok &= np.abs(n2 @ prior[:2] - dd) < p.prior_gate_m
        if not ok.any():
            break
        n2, dd = n2[ok], dd[ok]
        R = np.abs(X[sub] @ n2.T - dd)
        T = thr[sub][:, None]
        score = np.where(R < T, 1.0 - (R / T) ** 2, 0.0).sum(0)
        k = int(np.argmax(score))
        inl = remaining & (np.abs(X @ n2[k] - dd[k]) < thr)
        n = c = None
        for _it in range(3):
            # Refit in INVERSE depth: for a plane, 1/z = m . ray is linear in m and stereo
            # noise is ~constant in 1/z (disparity). A metric PCA fit is biased on oblique,
            # far boards because the noise lies along the rays, not along the normal.
            if inl.sum() < p.min_plane_pts:
                break
            m = np.linalg.lstsq(qn[inl], iz[inl], rcond=None)[0]
            if p.vertical_board or abs(m[2]) > sin_tilt * np.linalg.norm(m):
                m = np.append(np.linalg.lstsq(qn[inl][:, :2], iz[inl], rcond=None)[0], 0.0)
            mm = float(np.linalg.norm(m))
            if mm < 1e-9:
                break
            delta, nd = 1.0 / mm, m / mm           # plane: nd . (X - cam) = delta
            qd = qn @ nd
            with np.errstate(divide='ignore', invalid='ignore'):
                tpl = np.where(qd > 1e-6, delta / qd, np.inf)
            inl = remaining & (np.abs(z - tpl) < thr)
            if not inl.any():
                break
            n = -nd                                # towards the camera
            Pc = P[inl].mean(0)
            c = Pc - ((Pc - cam) @ nd - delta) * nd
        n_in = int(inl.sum())
        if c is None or n_in < p.min_plane_pts:
            remaining &= ~inl
            continue
        los = c - cam
        los /= max(np.linalg.norm(los), 1e-9)
        if abs(n @ los) < cos_view * 0.9:
            remaining &= ~inl
            continue
        planes.append({'n': n, 'c': c, 'n_in': n_in})
        remaining &= ~(np.abs((P - c) @ n) < 2.0 * thr)
    return planes


def _floor_z(dz_grid, rn, cam, fy, stride, gap=3):
    """NED z of the floor from the strided depth grid, or None. Floor pixels are those whose
    NED height barely changes between image rows `gap` grid steps apart (a vertical surface
    at depth z changes by gap*stride*z/fy there; the test uses 35 % of that, no absolute
    floor, so a close board never passes); the floor is the histogram peak of their
    heights, below the camera and spread in x AND y."""
    Hs, Ws = dz_grid.shape
    if Hs <= gap + 1:
        return None
    rz = rn[:, 2].reshape(Hs, Ws)
    Z = cam[2] + dz_grid * rz
    dZ = np.abs(Z[gap:] - Z[:-gap])
    zc = dz_grid[gap:]
    ok = np.isfinite(dZ) & np.isfinite(zc) & (zc > 0.05)
    horiz = np.zeros(dz_grid.shape, bool)
    horiz[gap:] = ok & (dZ < 0.35 * gap * stride * np.where(ok, zc, 0.0) / fy)
    zz = Z[horiz & (Z > cam[2] + 0.2)]
    if len(zz) < 80:
        return None
    hist, edges = np.histogram(zz, bins=np.arange(cam[2] + 0.2, zz.max() + 0.06, 0.03))
    if len(hist) == 0 or hist.max() < 80:
        return None
    k = int(np.argmax(hist))
    zc0 = 0.5 * (edges[k] + edges[k + 1])
    band = horiz & (np.abs(Z - zc0) < 0.05)
    X = cam[0] + dz_grid * rn[:, 0].reshape(Hs, Ws)
    Y = cam[1] + dz_grid * rn[:, 1].reshape(Hs, Ws)
    xy = np.stack([X[band], Y[band]], 1)
    ev = np.linalg.eigvalsh(np.cov(xy.T)) if len(xy) > 2 else np.zeros(2)
    if math.sqrt(max(ev[0], 0.0)) < 0.10:          # a horizontal surface spreads in x AND y
        return None
    return float(np.median(Z[band]))


def _plane_axes(n):
    e_v = DOWN - (DOWN @ n) * n
    nv = np.linalg.norm(e_v)
    if nv < 1e-6:
        return None, None
    e_v = e_v / nv
    e_h = np.cross(n, e_v)                         # observer's right when looking along -n
    return e_h, e_v


# ----------------------------------------------------------------------------- grid

class _Grid:
    pass


def _plane_grid(depth, K, R_nc, cam, plane, p, gray=None, lstd=None, floor_z=None):
    n, c = plane['n'], plane['c']
    e_h, e_v = _plane_axes(n)
    if e_h is None:
        return None
    s = p.stride_grid
    rc = _rays(depth.shape, K, s)
    rn = rc @ R_nc.T                               # NED direction, optical z = 1
    dz = depth[::s, ::s].reshape(-1).astype(np.float64)   # same dtypes on numpy 1.x and 2.x
    denom = rn @ n
    num = (c - cam) @ n
    with np.errstate(divide='ignore', invalid='ignore'):
        t = num / denom
    hit = (denom < -1e-6) & (t > p.min_range) & (t < p.max_range * 1.5)
    t = np.where(hit, t, np.nan)                   # no inf*0 below
    rel = cam - c
    h = rel @ e_h + t * (rn @ e_h)
    v = rel @ e_v + t * (rn @ e_v)
    valid = np.isfinite(dz) & (dz > 0.05)
    r = np.where(valid, dz, 0.0) - t
    thr = _thr(p, t)
    board = hit & valid & (np.abs(r) <= thr)
    beyond = hit & valid & (r > np.maximum(p.beyond_min, 2.5 * thr))
    front = hit & valid & (r < -np.maximum(2.0 * thr, 0.08))
    if floor_z is not None:
        # the floor crosses every vertical plane at its foot line; there it matches the plane
        # depth but it is NOT board (else the gap under a board / table looks enclosed)
        # (floor seen THROUGH an opening is beyond the plane and stays a hole)
        foot = board & (np.abs(cam[2] + dz * rn[:, 2] - floor_z) < p.floor_clear_m)
        board &= ~foot
        front |= foot
    inval = hit & ~valid
    look = np.zeros_like(inval)
    if p.board_look and gray is not None and board.sum() > 50:
        # weak-texture board: stereo gives no return, but the grey level and the lack of
        # texture still say "tan board" (learned from this frame's on-plane pixels).
        Ig = gray[::s, ::s].reshape(-1).astype(np.float32)
        Lg = lstd[::s, ::s].reshape(-1)               # local grey-level std (7x7)
        lb = Lg[board]
        sthr = max(4.0, 2.5 * float(np.median(lb)))
        ib = Ig[board & (Lg < sthr)]
        if len(ib) > 50:
            med = float(np.median(ib))
            sig = max(3.0, 1.48 * float(np.median(np.abs(ib - med))))
            look = inval & (np.abs(Ig - med) < max(3.0 * sig, 8.0)) & (Lg < sthr)
    if board.sum() < p.min_plane_pts // 2:
        return None
    # cell size from the ray footprint on the board (oblique / far -> coarser)
    rnn = np.linalg.norm(rn[board], axis=1)
    cosv = np.abs(denom[board]) / rnn
    fp = s * t[board] * rnn / K[0] / np.maximum(cosv, 0.3)
    cell = float(np.clip(1.25 * np.percentile(fp, 90), p.cell_min, p.cell_max))
    hb, vb = h[board], v[board]
    pad = p.ring_m + 0.25
    h_lo = max(np.percentile(hb, 0.5) - pad, -4.0)
    h_hi = min(np.percentile(hb, 99.5) + pad, 4.0)
    v_lo = max(np.percentile(vb, 0.5) - pad, -4.0)
    v_hi = min(np.percentile(vb, 99.5) + pad, 4.0)
    Wg = int(math.ceil((h_hi - h_lo) / cell))
    Hg = int(math.ceil((v_hi - v_lo) / cell))
    if Wg < 4 or Hg < 4:
        return None
    ix = np.floor((h - h_lo) / cell)
    iy = np.floor((v - v_lo) / cell)
    inb = hit & (ix >= 0) & (ix < Wg) & (iy >= 0) & (iy < Hg)
    flat = (np.where(inb, iy, 0) * Wg + np.where(inb, ix, 0)).astype(np.int64)

    def cnt(m):
        m = m & inb
        return np.bincount(flat[m], minlength=Hg * Wg).reshape(Hg, Wg).astype(np.float64)

    nb, nf, ni, nx, nt, nl = cnt(board), cnt(beyond), cnt(inval), cnt(front), cnt(hit), cnt(look)
    empty = nt == 0
    if empty.any():                                # ray-coverage gaps: borrow 3x3 neighbours
        def fill(a):
            return np.where(empty, cv2.boxFilter(a, -1, (3, 3), normalize=False,
                                                 borderType=cv2.BORDER_CONSTANT), a)
        nb, nf, ni, nx, nt, nl = fill(nb), fill(nf), fill(ni), fill(nx), fill(nt), fill(nl)
    G = _Grid()
    G.n, G.c, G.e_h, G.e_v = n, c, e_h, e_v
    G.cell, G.h_lo, G.v_lo, G.Wg, G.Hg = cell, h_lo, v_lo, Wg, Hg
    G.observed = nt > 0
    G.B = (nb > 0) & (nb >= p.board_frac * nt)
    k = 2 * int(round(p.close_m / cell / 2.0)) + 1
    # board by appearance only, and only inside small (< fill_m) gaps of the depth board:
    # never turns open space beside / under a board or a whole opening into board
    kf = 2 * int(round(p.fill_m / cell / 2.0)) + 1
    near_b = cv2.morphologyEx(G.B.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((kf, kf), np.uint8)).astype(bool)
    G.BL = ~G.B & (nl >= 0.5 * nt) & (nt > 0) & near_b
    Bx = G.B | G.BL
    G.Bc = cv2.morphologyEx(Bx.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((k, k), np.uint8)).astype(bool) if k > 1 else Bx
    G.Hm = G.observed & ~G.Bc & (nx < 0.5 * nt)
    G.beyond_frac = np.where(nt > 0, nf / np.maximum(nt, 1), 0.0)
    G.beyond_cell = G.observed & (nf >= 0.5 * nt)
    G.n_board_rays = int(board.sum())
    return G


def _max_rect(S, x0, y0, x1, y1, lo_x, hi_x, lo_y, hi_y, mn):
    """Coordinate ascent on the 4 edges of [x0,x1)x[y0,y1) maximising the sum of the
    weight image whose (H+1,W+1) integral image is S."""
    for _ in range(8):
        old = (x0, y0, x1, y1)
        c = np.arange(lo_x, x1 - mn + 1)
        if len(c):
            x0 = int(c[np.argmax(S[y1, x1] - S[y0, x1] - S[y1, c] + S[y0, c])])
        c = np.arange(x0 + mn, hi_x + 1)
        if len(c):
            x1 = int(c[np.argmax(S[y1, c] - S[y0, c] - S[y1, x0] + S[y0, x0])])
        c = np.arange(lo_y, y1 - mn + 1)
        if len(c):
            y0 = int(c[np.argmax(S[y1, x1] - S[c, x1] - S[y1, x0] + S[c, x0])])
        c = np.arange(y0 + mn, hi_y + 1)
        if len(c):
            y1 = int(c[np.argmax(S[c, x1] - S[y0, x1] - S[c, x0] + S[y0, x0])])
        if (x0, y0, x1, y1) == old:
            break
    return x0, y0, x1, y1


def _band(G, arr, ys, xs, outside_val):
    """Mean of arr over the band rows ys=(a,b), cols xs=(a,b); out-of-grid cells count as
    outside_val."""
    (ya, yb), (xa, xb) = ys, xs
    total = max((yb - ya) * (xb - xa), 1)
    ca, cb = max(ya, 0), min(yb, G.Hg)
    da, db = max(xa, 0), min(xb, G.Wg)
    inside = arr[ca:cb, da:db] if (cb > ca and db > da) else np.zeros((0,))
    n_in = inside.size
    return (float(inside.sum()) + outside_val * (total - n_in)) / total


def _sides(G, rect, p):
    x0, y0, x1, y1 = rect
    tr = max(2, int(math.ceil(p.ring_m / G.cell)))
    # trim 10 % of each side length (corner cells are ambiguous)
    dx = max(0, int(0.1 * (x1 - x0)))
    dy = max(0, int(0.1 * (y1 - y0)))
    bands = {
        'left': ((y0 + dy, y1 - dy), (x0 - tr, x0)),
        'right': ((y0 + dy, y1 - dy), (x1, x1 + tr)),
        'top': ((y0 - tr, y0), (x0 + dx, x1 - dx)),
        'bottom': ((y1, y1 + tr), (x0 + dx, x1 - dx)),
    }
    to = tr + max(1, int(math.ceil(p.ring_outer_m / G.cell)))
    outer = {
        'left': ((y0 + dy, y1 - dy), (x0 - to, x0 - tr)),
        'right': ((y0 + dy, y1 - dy), (x1 + tr, x1 + to)),
        'top': ((y0 - to, y0 - tr), (x0 + dx, x1 - dx)),
        'bottom': ((y1 + tr, y1 + to), (x0 + dx, x1 - dx)),
    }
    out = []
    unobs = ~G.observed
    for name in ('left', 'right', 'top', 'bottom'):
        ys, xs = bands[name]
        fb = _band(G, G.Bc, ys, xs, 0.0)
        fu = _band(G, unobs, ys, xs, 1.0)
        fo = _band(G, G.beyond_cell, ys, xs, 0.0)   # positive depth evidence of a gap
        # one band further out must not see through: a hoop, a pole, a table leg or rail
        # (a few cm) is not a board around an opening
        ys2, xs2 = outer[name]
        fthin = _band(G, G.Hm, ys2, xs2, 0.0)
        if fb >= p.ring_board_min and fthin > 0.5:
            st = 'thin'
        elif fb >= p.ring_board_min:
            st = 'board'
        elif fu >= p.ring_fov_max:
            st = 'fov'
        else:
            st = 'open'
        out.append((name, st, fb, fu, fo, fthin))
    return out


def _grid_candidates(G, p):
    Hm8 = G.Hm.astype(np.uint8)
    nlab, lab, st, _ = cv2.connectedComponentsWithStats(Hm8, connectivity=4)
    if nlab <= 1:
        return []
    Wt = np.where(G.Hm, 1.0, np.where(G.Bc, -3.0, -1.0))
    S = np.zeros((G.Hg + 1, G.Wg + 1))
    S[1:, 1:] = Wt.cumsum(0).cumsum(1)
    tr = max(2, int(math.ceil(p.ring_m / G.cell)))
    mn = max(2, int(math.ceil(p.min_size_m / G.cell)))
    mx = int(math.ceil(p.max_size_m / G.cell))
    out = []
    for L in range(1, nlab):
        x, y, w, h, area = (int(a) for a in st[L])
        if w < mn or h < mn or area < 0.5 * mn * mn:
            continue
        x0, y0, x1, y1 = x, y, x + w, y + h
        x0, y0, x1, y1 = _max_rect(S, x0, y0, x1, y1, max(0, x0 - tr), min(G.Wg, x1 + tr),
                                   max(0, y0 - tr), min(G.Hg, y1 + tr), mn)
        if x1 - x0 < mn or y1 - y0 < mn or x1 - x0 > mx or y1 - y0 > mx:
            continue
        fill = float(G.Hm[y0:y1, x0:x1].mean())
        sides = _sides(G, (x0, y0, x1, y1), p)
        beyond = float(G.beyond_frac[y0:y1, x0:x1].mean())
        out.append({'rect': (x0, y0, x1, y1), 'fill': fill, 'sides': sides, 'beyond': beyond,
                    'h0': G.h_lo + x0 * G.cell, 'h1': G.h_lo + x1 * G.cell,
                    'v0': G.v_lo + y0 * G.cell, 'v1': G.v_lo + y1 * G.cell})
    return out


# ----------------------------------------------------------------------------- image edges

def _refine_edges(img_f, K, R_nc, cam, G, hv, p, return_tilt=False):
    """Re-locate the 4 edges (h0,h1,v0,v1) on the grey image. Profiles are sampled on the
    board plane, so the result is metric. Returns (new hv, n_sides_refined)."""
    fx, fy, cx, cy = K
    H, W = img_f.shape
    h0, h1, v0, v1 = hv
    search = 1.5 * G.cell + 0.02
    out = list(hv)
    nref = 0
    slopes = [None] * 4
    rng_cam = np.linalg.norm(G.c + 0.5 * (h0 + h1) * G.e_h + 0.5 * (v0 + v1) * G.e_v - cam)
    px_per_m = fx / max(rng_cam, 0.2)
    nstep = int(np.clip(4 * search * px_per_m, 21, 121)) | 1
    offs = np.linspace(-search, search, nstep)
    step = offs[1] - offs[0]
    q = max(1, int(round(1.5 / (step * px_per_m))))
    for k, (fixed, a, b, axis) in enumerate(((h0, v0, v1, 'h'), (h1, v0, v1, 'h'),
                                             (v0, h0, h1, 'v'), (v1, h0, h1, 'v'))):
        ns = 24
        along = np.linspace(a + 0.15 * (b - a), b - 0.15 * (b - a), ns)
        if axis == 'h':
            hh = fixed + offs[None, :] + 0.0 * along[:, None]
            vv = along[:, None] + 0.0 * offs[None, :]
        else:
            vv = fixed + offs[None, :] + 0.0 * along[:, None]
            hh = along[:, None] + 0.0 * offs[None, :]
        X = G.c + hh[..., None] * G.e_h + vv[..., None] * G.e_v
        pc = (X - cam) @ R_nc
        z = pc[..., 2]
        if np.any(z < 0.05):
            continue
        u = (fx * pc[..., 0] / z + cx).astype(np.float32)
        w = (fy * pc[..., 1] / z + cy).astype(np.float32)
        inside = (u >= 0) & (u <= W - 1) & (w >= 0) & (w <= H - 1)
        prof = cv2.remap(img_f, u, w, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float64)
        g = np.abs(np.gradient(prof, axis=1))
        g[:, :q] = 0
        g[:, -q:] = 0
        kk = np.argmax(g, axis=1)
        rows = np.arange(ns)
        lo = np.clip(kk - q, 0, nstep - 1)
        hi = np.clip(kk + q, 0, nstep - 1)
        contrast = np.abs(prof[rows, hi] - prof[rows, lo])
        okr = inside.all(axis=1) & (contrast >= p.edge_min_contrast) & (kk > q) & (kk < nstep - 1 - q)
        if okr.sum() < max(4, int(0.4 * ns)):
            continue
        gm, g0, gp = g[rows, kk - 1], g[rows, kk], g[rows, np.minimum(kk + 1, nstep - 1)]
        den = gm - 2 * g0 + gp
        delta = np.where(den < -1e-9, 0.5 * (gm - gp) / np.where(den < -1e-9, den, -1.0), 0.0)
        e = offs[kk] + np.clip(delta, -0.5, 0.5) * step
        e = e[okr]
        al = along[okr] - 0.5 * (a + b)            # along-edge position, 0 at the edge middle
        # robust line e = c0 + c1*al: the real opening may be rotated a few degrees in the board
        # plane against gravity (board not level, camera roll / extrinsic error); a constant
        # offset would then scatter by ~ tan(rot) * edge length and be rejected
        tol = max(p.edge_mad_m, 1.5 / px_per_m)
        good = np.ones(len(e), bool)
        c0, c1 = float(np.median(e)), 0.0
        for _it in range(3):
            if good.sum() < 4:
                break
            A = np.stack([np.ones(int(good.sum())), al[good]], 1)
            c0, c1 = (float(x) for x in np.linalg.lstsq(A, e[good], rcond=None)[0])
            res = e - (c0 + c1 * al)
            mad = float(np.median(np.abs(res[good])))
            good = np.abs(res) <= max(3.0 * mad, 0.5 * step, 0.5 * tol)
        res = e - (c0 + c1 * al)
        if (good.sum() < max(4, int(0.3 * ns)) or float(np.median(np.abs(res[good]))) > tol
                or abs(c1) > math.tan(math.radians(p.edge_max_tilt_deg))):
            continue
        out[k] = fixed + c0
        slopes[k] = c1
        nref += 1
    if out[1] - out[0] < 0.5 * (h1 - h0) or out[3] - out[2] < 0.5 * (v1 - v0):
        return (list(hv), 0, float('nan')) if return_tilt else (list(hv), 0)
    if return_tilt:
        # in-plane rotation of the opening against gravity (deg, + = clockwise as seen from the
        # camera side): vertical sides give dh/dv, horizontal sides -dv/dh
        t = [math.degrees(math.atan(slopes[k])) for k in (0, 1) if slopes[k] is not None]
        t += [-math.degrees(math.atan(slopes[k])) for k in (2, 3) if slopes[k] is not None]
        return out, nref, (float(np.median(t)) if t else float('nan'))
    return out, nref


def _side_steps(img_f, K, R_nc, cam, G, hv, band=(0.02, 0.08)):
    """Grey-level step across each edge (left, right, top, bottom), sampled ON THE PLANE:
    |median(ring band outside) - median(band inside)|. A board around an opening is a
    surface distinct from the scene behind it; a 'hole' in a net / mesh wall or in a
    patchy depth surface shows the same thing on both sides. NaN where not in view."""
    fx, fy, cx, cy = K
    H, W = img_f.shape
    h0, h1, v0, v1 = hv
    a, b = band
    out = []
    for side in ('left', 'right', 'top', 'bottom'):
        vals = []
        for sgn in (-1.0, 1.0):                    # -1 outside, +1 inside
            d = np.linspace(a, b, 4)
            if side in ('left', 'right'):
                along = np.linspace(v0 + 0.15 * (v1 - v0), v1 - 0.15 * (v1 - v0), 12)
                e = h0 - sgn * d if side == 'left' else h1 + sgn * d
                hh, vv = np.meshgrid(e, along)
            else:
                along = np.linspace(h0 + 0.15 * (h1 - h0), h1 - 0.15 * (h1 - h0), 12)
                e = v0 - sgn * d if side == 'top' else v1 + sgn * d
                vv, hh = np.meshgrid(e, along)
            X = G.c + hh.reshape(-1, 1) * G.e_h + vv.reshape(-1, 1) * G.e_v
            pc = (X - cam) @ R_nc
            z = pc[:, 2]
            ok = z > 0.05
            u = fx * pc[:, 0] / np.where(ok, z, 1.0) + cx
            w = fy * pc[:, 1] / np.where(ok, z, 1.0) + cy
            ok &= (u >= 0) & (u <= W - 1) & (w >= 0) & (w <= H - 1)
            vals.append(float(np.median(img_f[np.round(w[ok]).astype(int), np.round(u[ok]).astype(int)]))
                        if ok.sum() >= 8 else float('nan'))
        out.append(abs(vals[0] - vals[1]))
    return out


def _known_size_search(img_f, K, R_nc, cam, G, prior, known, p, res=0.01):
    """With a board plane, a prior and the measured window size: rectify the board around
    the prior (res m/px, orthographic on the plane), slide a known-size rectangle and keep
    the centre whose WEAKEST side has the strongest edge (|dI/dh| on the vertical sides,
    |dI/dv| on the horizontal ones). Independent of depth inside the opening, which stereo
    (SGM) tends to fill with the board's disparity. Returns hv=(h0,h1,v0,v1) or None."""
    fx, fy, cx, cy = K
    H, W = img_f.shape
    ww, hh = float(known[0]), float(known[1])
    hp, vp = float((prior - G.c) @ G.e_h), float((prior - G.c) @ G.e_v)
    R = p.prior_search_m
    h_lo, h_hi = hp - R - ww / 2 - 0.1, hp + R + ww / 2 + 0.1
    v_lo, v_hi = vp - R - hh / 2 - 0.1, vp + R + hh / 2 + 0.1
    nh, nv = int((h_hi - h_lo) / res), int((v_hi - v_lo) / res)
    if nh < 10 or nv < 10:
        return None
    hs = h_lo + res * np.arange(nh)
    vs = v_lo + res * np.arange(nv)
    Hh, Vv = np.meshgrid(hs, vs)
    X = G.c + Hh[..., None] * G.e_h + Vv[..., None] * G.e_v
    pc = (X - cam) @ R_nc
    z = pc[..., 2]
    okz = z > 0.05
    u = np.where(okz, fx * pc[..., 0] / np.where(okz, z, 1.0) + cx, -1.0).astype(np.float32)
    w = np.where(okz, fy * pc[..., 1] / np.where(okz, z, 1.0) + cy, -1.0).astype(np.float32)
    inside = okz & (u >= 0) & (u <= W - 1) & (w >= 0) & (w <= H - 1)
    if inside.mean() < 0.3:
        return None
    Rimg = cv2.remap(img_f, u, w, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    gx = np.abs(cv2.Sobel(Rimg, cv2.CV_32F, 1, 0, ksize=3)) * inside
    gy = np.abs(cv2.Sobel(Rimg, cv2.CV_32F, 0, 1, ksize=3)) * inside
    kv = max(3, int(0.7 * hh / res)) | 1           # inner 70 % of each side
    kh = max(3, int(0.7 * ww / res)) | 1
    A = cv2.boxFilter(gx, -1, (1, kv), borderType=cv2.BORDER_CONSTANT)
    B = cv2.boxFilter(gy, -1, (kh, 1), borderType=cv2.BORDER_CONSTANT)
    dx, dy = int(round(ww / 2 / res)), int(round(hh / 2 / res))
    # candidate centres within R of the prior
    ci = np.arange(dy, nv - dy)
    cj = np.arange(dx, nh - dx)
    if len(ci) == 0 or len(cj) == 0:
        return None
    CI, CJ = np.meshgrid(ci, cj, indexing='ij')
    sides = np.stack([A[CI, CJ - dx], A[CI, CJ + dx], B[CI - dy, CJ], B[CI + dy, CJ]], 0)
    score = sides.min(0).astype(np.float64)
    near = (hs[CJ] - hp) ** 2 + (vs[CI] - vp) ** 2 <= R * R
    score = np.where(near, score, -1.0)
    k = np.unravel_index(int(np.argmax(score)), score.shape)
    if score[k] <= 0:
        return None
    hc, vc = hs[CJ[k]], vs[CI[k]]
    return (hc - ww / 2, hc + ww / 2, vc - hh / 2, vc + hh / 2)


# ----------------------------------------------------------------------------- image quads

def _prep_image(gray):
    """Grey image at <= 800 px width, thin dark lines (net strings) closed, blurred."""
    g = _gray(gray)
    H0, W0 = g.shape
    s = 1.0
    if W0 > 800:
        s = 640.0 / W0
        g = cv2.resize(g, (640, int(round(H0 * s))), interpolation=cv2.INTER_AREA)
    k = max(3, int(round(g.shape[1] / 213.0)) | 1)
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    g = cv2.GaussianBlur(g, (5, 5), 1.2)
    return g, s


def _lines(segs, up, vertical, max_lines=24):
    """Hough segments near 'up' (vertical=True, x = a*y + b) or across it (y = a*x + b),
    collinear ones merged. Rows: a, b, lo, hi (extent along the free coordinate), support."""
    if segs is None:
        return np.zeros((0, 5))
    S = segs.reshape(-1, 4).astype(np.float64)
    d = S[:, 2:] - S[:, :2]
    ln = np.hypot(d[:, 0], d[:, 1])
    cu = np.abs(d @ up) / np.maximum(ln, 1e-9)
    if vertical:
        S, ln = S[cu > math.cos(math.radians(15))], ln[cu > math.cos(math.radians(15))]
        S = S[:, [1, 0, 3, 2]]                     # work in (y, x): x = a*y + b
    else:
        S, ln = S[cu < math.sin(math.radians(35))], ln[cu < math.sin(math.radians(35))]
    out = []
    for i in np.argsort(-ln):
        p0, p1 = S[i, :2], S[i, 2:]
        if abs(p1[0] - p0[0]) < 1e-6:
            continue
        a = (p1[1] - p0[1]) / (p1[0] - p0[0])
        b = p0[1] - a * p0[0]
        lo, hi = min(p0[0], p1[0]), max(p0[0], p1[0])
        mid = 0.5 * (lo + hi)
        for L in out:
            if abs(L[0] - a) < 0.06 and abs(L[0] * mid + L[1] - (a * mid + b)) < 3.0:
                L[2], L[3], L[4] = min(L[2], lo), max(L[3], hi), L[4] + ln[i]
                break
        else:
            out.append([a, b, lo, hi, ln[i]])
    out = np.array(out).reshape(-1, 5)
    return out[np.argsort(-out[:, 4])][:max_lines]


def _bilinear(Q, s, t):
    """Points of quads Q (C,4,2) TL,TR,BR,BL at params s (right), t (down); s,t broadcast
    to (C, M) or (M,)."""
    s = np.asarray(s, float)
    t = np.asarray(t, float)
    if s.ndim == 1:
        s, t = s[None, :], t[None, :]
    TL, TR, BR, BL = (Q[:, i, None, :] for i in range(4))
    s, t = s[..., None], t[..., None]
    return (1 - s) * (1 - t) * TL + s * (1 - t) * TR + s * t * BR + (1 - s) * t * BL


def _sample(img, P, fill=np.nan):
    H, W = img.shape
    x = np.round(P[..., 0]).astype(np.int64)
    y = np.round(P[..., 1]).astype(np.int64)
    ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    out = np.full(x.shape, fill, np.float64)
    out[ok] = img[y[ok], x[ok]]
    return out


def _score_quads(Q, g, DT, H, W):
    """Window-likeness of image quads: edge support on all 4 sides, a uniform 'board' ring
    around (similar grey level on all sides, few edges), and a grey-level step on every
    side (board | opening; rejects thin bars and the board-strip-plus-floor box)."""
    C = len(Q)
    if C == 0:
        return np.zeros(0), {}
    u = np.linspace(0.06, 0.94, 20)
    sides = [(u, np.zeros_like(u)), (np.ones_like(u), u), (u, np.ones_like(u)), (np.zeros_like(u), u)]
    sup = np.stack([(_sample(DT, _bilinear(Q, s_, t_), np.inf) <= 1.5).mean(1) for s_, t_ in sides], 1)
    v = np.linspace(0.1, 0.9, 12)[None, :]
    wq = 0.5 * (np.linalg.norm(Q[:, 1] - Q[:, 0], axis=1) + np.linalg.norm(Q[:, 2] - Q[:, 3], axis=1))
    hq = 0.5 * (np.linalg.norm(Q[:, 3] - Q[:, 0], axis=1) + np.linalg.norm(Q[:, 2] - Q[:, 1], axis=1))
    ring_vals, ring_edge, inner_vals = [], [], []
    for k in (1.0, 2.0):                           # rings close to the edge (occluders, narrow board)
        rt = (k * np.maximum(3.0, 0.035 * hq) / np.maximum(hq, 1.0))[:, None] + 0 * v
        rsd = (k * np.maximum(3.0, 0.035 * wq) / np.maximum(wq, 1.0))[:, None] + 0 * v
        vv = v + 0 * rt
        outer = [(vv, -rt), (1 + rsd, vv), (vv, 1 + rt), (-rsd, vv)]
        inner = [(vv, rt), (1 - rsd, vv), (vv, 1 - rt), (rsd, vv)]
        ring_vals.append(np.stack([_sample(g, _bilinear(Q, s_, t_)) for s_, t_ in outer], 1))  # (C,4,12)
        inner_vals.append(np.stack([_sample(g, _bilinear(Q, s_, t_)) for s_, t_ in inner], 1))
        dts = np.stack([_sample(DT, _bilinear(Q, s_, t_), np.inf) for s_, t_ in outer], 1)
        ring_edge.append(dts <= 1.0)
    rv = np.concatenate(ring_vals, 2)                                               # (C,4,24)
    iv = np.concatenate(inner_vals, 2)
    re = np.concatenate(ring_edge, 2)
    out_frac = np.isnan(rv).mean(2)                                                 # (C,4)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        side_med = np.nanmedian(rv, 2)                                              # (C,4)
        side_in = np.nanmedian(iv, 2)
        spread = np.nanmax(side_med, 1) - np.nanmin(side_med, 1)
        # every side must be a grey-level STEP board|opening, not a thin line or a texture edge
        contrast = np.nanmin(np.abs(side_med - side_in), 1)
    # an opening shows the far scene (texture/edges); a painted or dark panel is uniform
    gi = np.linspace(0.15, 0.85, 8)
    si, ti = np.meshgrid(gi, gi)
    in_dt = _sample(DT, _bilinear(Q, si.ravel(), ti.ravel()), np.inf)
    texture = np.clip(4.0 * (in_dt <= 2.0).mean(1), 0.0, 1.0)
    ring_edge = re.mean((1, 2))
    area = np.abs(0.5 * np.sum(Q[:, :, 0] * np.roll(Q[:, :, 1], -1, 1) - np.roll(Q[:, :, 0], -1, 1) * Q[:, :, 1], 1))
    score = (np.clip((sup.min(1) - 0.3) / 0.6, 0, 1) * np.clip(contrast / 20.0, 0, 1) *
             np.clip(1.0 - spread / 40.0, 0, 1) * np.clip(1.0 - 1.5 * ring_edge, 0, 1) *
             (0.4 + 0.6 * texture))
    score = np.where(np.isfinite(score) & (out_frac.max(1) < 0.5), score, 0.0)
    quality = score.copy()                         # 0..1, size-independent
    score = score * (area / float(H * W)) ** 0.25  # mild preference for the larger opening
    return score, {'support': sup, 'contrast': contrast, 'spread': spread, 'ring_edge': ring_edge,
                   'texture': texture, 'quality': quality}


def _refine_quad(E, q, tol=2.5):
    """Least-squares lines through the Canny pixels near each side; corners re-intersected."""
    ys, xs = np.nonzero(E)
    if len(xs) == 0:
        return q
    P = np.stack([xs, ys], 1).astype(np.float64)
    lines = []
    for i in range(4):
        a, b = q[i].astype(np.float64), q[(i + 1) % 4].astype(np.float64)
        d = b - a
        L = np.linalg.norm(d)
        if L < 5:
            return q
        d /= L
        nrm = np.array([-d[1], d[0]])
        rel = P - a
        al = rel @ d
        m = (np.abs(rel @ nrm) < tol) & (al > 0.05 * L) & (al < 0.95 * L)
        if m.sum() < 0.3 * L:
            lines.append((a, d))
            continue
        vx, vy, x0, y0 = cv2.fitLine(P[m].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], float), np.array([vx, vy], float)))
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[(i - 1) % 4], lines[i]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return q
        tt = np.linalg.solve(A, p2 - p1)
        out.append(p1 + tt[0] * d1)
    out = np.array(out, np.float32)
    if np.max(np.linalg.norm(out - q, axis=1)) > 6.0:
        return q
    return out


def _demote_outer(Q, score, top=40, tol=4.0):
    """The opening is the innermost board-bounded rectangle: a quad that encloses a
    decent-scoring quad and shares >= 2 of its sides (opening + a strip of board up to the
    board edge or an occluder) never outranks that inner quad."""
    score = score.copy()
    idx = [int(c) for c in np.argsort(-score)[:top] if score[c] > 0]
    if len(idx) < 2:
        return score
    box = {c: (Q[c][:, 0].min(), Q[c][:, 1].min(), Q[c][:, 0].max(), Q[c][:, 1].max()) for c in idx}
    for a in idx:
        ax0, ay0, ax1, ay1 = box[a]
        for b in idx:
            if a == b or score[b] < 0.5 * score[a]:
                continue
            bx0, by0, bx1, by1 = box[b]
            if not (bx0 >= ax0 - tol and by0 >= ay0 - tol and bx1 <= ax1 + tol and by1 <= ay1 + tol):
                continue
            if (bx1 - bx0) * (by1 - by0) > 0.85 * (ax1 - ax0) * (ay1 - ay0):
                continue
            shared = sum(abs(x - y) < tol for x, y in ((ax0, bx0), (ay0, by0), (ax1, bx1), (ay1, by1)))
            if shared >= 2:
                score[a] = min(score[a], 0.9 * score[b])
    return score


def find_window_quads(gray, max_quads=5, up_px=None, max_score=200):
    """Image-only rectangular openings (no depth). Pairs of near-vertical and pairs of
    near-horizontal Hough lines form quads; each is scored for edge support on all four
    sides, a uniform board ring around it, a grey-level step on every side and some
    texture inside (an opening shows the far hall). A quad enclosing a decent inner quad
    that shares two of its sides is demoted (opening + board strip up to an occluder).
    Returns [{quad (4,2) TL,TR,BR,BL in input pixels, score, ...}], best first.
    up_px: image direction of world-up (default (0,-1), level camera)."""
    if gray is None:
        return []
    g, s = _prep_image(gray)
    H, W = g.shape
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    hi = max(40.0, float(np.percentile(mag, 85)))
    E = cv2.Canny(g, 0.4 * hi, hi, L2gradient=True)
    DT = cv2.distanceTransform((E == 0).astype(np.uint8), cv2.DIST_L2, 3)
    up = np.array([0.0, -1.0]) if up_px is None else np.asarray(up_px, float) / np.linalg.norm(up_px)
    segs = cv2.HoughLinesP(E, 1, np.pi / 180, threshold=max(20, int(0.03 * W)),
                           minLineLength=max(15, int(0.045 * W)), maxLineGap=max(4, int(0.012 * W)))
    V = _lines(segs, up, True)
    Hz = _lines(segs, up, False)
    if len(V) < 2 or len(Hz) < 2:
        return []
    V = V[np.argsort(V[:, 0] * (H / 2.0) + V[:, 1])]
    Hz = Hz[np.argsort(Hz[:, 0] * (W / 2.0) + Hz[:, 1])]
    av, bv = V[:, 0][:, None], V[:, 1][:, None]
    ah, bh = Hz[:, 0][None, :], Hz[:, 1][None, :]
    X = (av * bh + bv) / (1.0 - av * ah)                                         # (Nv,Nh)
    Y = ah * X + bh

    def ov(lo, hi_, a, b):
        return np.clip(np.minimum(hi_, b) - np.maximum(lo, a), 0, None) / np.maximum(b - a, 1e-6)

    # ovH[k,i,j]: horizontal k covers [X[i,k], X[j,k]]; ovV[i,k,l]: vertical i covers [Y[i,k], Y[i,l]]
    XT = X.T                                                                    # (Nh,Nv)
    ovH = ov(Hz[:, 2][:, None, None], Hz[:, 3][:, None, None], XT[:, :, None], XT[:, None, :])
    ovV = ov(V[:, 2][:, None, None], V[:, 3][:, None, None], Y[:, :, None], Y[:, None, :])
    Nv, Nh = len(V), len(Hz)
    i, j, k, l = np.meshgrid(np.arange(Nv), np.arange(Nv), np.arange(Nh), np.arange(Nh), indexing='ij')
    keep = (i < j) & (k < l)
    i, j, k, l = i[keep], j[keep], k[keep], l[keep]
    w_ = X[j, k] - X[i, k]
    h_ = Y[i, l] - Y[i, k]
    keep = ((ovH[k, i, j] >= 0.4) & (ovH[l, i, j] >= 0.4) & (ovV[i, k, l] >= 0.4) & (ovV[j, k, l] >= 0.4)
            & (w_ > 0.04 * W) & (h_ > 0.04 * H) & (X[j, l] - X[i, l] > 0.04 * W) & (Y[j, l] - Y[j, k] > 0.04 * H))
    i, j, k, l = i[keep], j[keep], k[keep], l[keep]
    if len(i) == 0:
        return []
    Q = np.stack([np.stack([X[i, k], Y[i, k]], 1), np.stack([X[j, k], Y[j, k]], 1),
                  np.stack([X[j, l], Y[j, l]], 1), np.stack([X[i, l], Y[i, l]], 1)], 1)
    wq = 0.5 * (np.linalg.norm(Q[:, 1] - Q[:, 0], axis=1) + np.linalg.norm(Q[:, 2] - Q[:, 3], axis=1))
    hq = 0.5 * (np.linalg.norm(Q[:, 3] - Q[:, 0], axis=1) + np.linalg.norm(Q[:, 2] - Q[:, 1], axis=1))
    ok = (hq / np.maximum(wq, 1e-6) > 0.25) & (hq / np.maximum(wq, 1e-6) < 4.0)
    Q = Q[ok]
    if len(Q) > max_score:                         # cheap pre-score: edge support, 8 samples/side
        u8 = np.linspace(0.1, 0.9, 8)
        pre = np.stack([(_sample(DT, _bilinear(Q, s_, t_), np.inf) <= 1.5).mean(1) for s_, t_ in
                        ((u8, 0 * u8), (1 + 0 * u8, u8), (u8, 1 + 0 * u8), (0 * u8, u8))], 1)
        key = pre.min(1) + 0.1 * pre.mean(1)
        Q = Q[np.argsort(-key, kind='stable')[:max_score]]
    score, feat = _score_quads(Q, g.astype(np.float32), DT, H, W)
    score = _demote_outer(Q, score)
    order = np.argsort(-score)
    out, boxes = [], []
    for c in order:
        if score[c] <= 0 or len(out) >= max_quads:
            break
        q = Q[c]
        x0, y0 = q.min(0)
        x1, y1 = q.max(0)
        dup = False
        for (a0, b0, a1, b1) in boxes:
            iw = max(0.0, min(x1, a1) - max(x0, a0))
            ih = max(0.0, min(y1, b1) - max(y0, b0))
            inter = iw * ih
            if inter / ((x1 - x0) * (y1 - y0) + (a1 - a0) * (b1 - b0) - inter) > 0.5:
                dup = True
                break
        if dup:
            continue
        boxes.append((x0, y0, x1, y1))
        qr = _refine_quad(E, q.astype(np.float32))
        out.append({'quad': (qr / s).astype(np.float32), 'score': float(score[c]),
                    'support': feat['support'][c].round(2).tolist(),
                    'contrast': float(feat['contrast'][c]), 'spread': float(feat['spread'][c]),
                    'texture': float(feat['texture'][c]), 'quality': float(feat['quality'][c])})
    return out


# ----------------------------------------------------------------------------- assembly

def _estimate(G, hv, cam, half_span_m, half_height_m, margin_m, conf, source, sides,
              plane_inliers, p, reason='', nref=0, tilt=float('nan')):
    h0, h1, v0, v1 = hv
    width, height = h1 - h0, v1 - v0
    center = G.c + 0.5 * (h0 + h1) * G.e_h + 0.5 * (v0 + v1) * G.e_v
    corners = np.array([G.c + hh * G.e_h + vv * G.e_v
                        for hh, vv in ((h0, v0), (h1, v0), (h1, v1), (h0, v1))])
    lat = width / 2.0 - half_span_m
    ver = height / 2.0 - half_height_m
    rng_m = float(np.linalg.norm(center - cam))
    unc = p.size_unc_a + p.size_unc_b * rng_m * rng_m
    fits = lat - unc >= margin_m and ver - unc >= margin_m
    clear = bool(conf >= p.clear_min_conf and fits and rng_m <= p.clear_max_range)
    why = reason
    if not clear and not why:
        why = ('low_confidence' if conf < p.clear_min_conf else
               'too_small_for_body' if not fits else 'too_far_to_confirm')
    return WindowEstimate(True, center, G.n.copy(), float(width), float(height), clear,
                          float(conf), corners_ned=corners, right_ned=G.e_h.copy(),
                          lateral_slack_m=float(lat), vertical_slack_m=float(ver),
                          range_m=rng_m, source=source,
                          reason=why, sides=tuple(sides), plane_inliers=int(plane_inliers),
                          edges_refined=int(nref), inplane_tilt_deg=float(tilt))


def _prior_q(center, prior, p):
    if prior is None:
        return 1.0
    d = float(np.linalg.norm(center - prior))
    if d > p.prior_gate_m:
        return 0.0
    return math.exp(-0.5 * (d / p.prior_sigma_m) ** 2)


def _size_q(w, h, known):
    if known is None:
        return 1.0
    return math.exp(-0.5 * (((w - known[0]) / 0.12) ** 2 + ((h - known[1]) / 0.12) ** 2))


def _ray_plane_hv(G, K, R_nc, cam, uv):
    fx, fy, cx, cy = K
    uv = np.asarray(uv, float).reshape(-1, 2)
    rc = np.stack([(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy, np.ones(len(uv))], -1)
    rn = rc @ R_nc.T
    den = rn @ G.n
    if np.any(den > -1e-6):
        return None
    t = ((G.c - cam) @ G.n) / den
    X = cam + t[:, None] * rn - G.c
    return np.stack([X @ G.e_h, X @ G.e_v], -1)


def _up_px(R_nc):
    """Image direction of world-up (NED -z) for a camera with attitude R_nc."""
    u = np.asarray(R_nc, float).T @ np.array([0.0, 0.0, -1.0])
    n = math.hypot(u[0], u[1])
    return None if n < 0.3 else np.array([u[0] / n, u[1] / n])


def _geom_reject(G, hv, floor_z, p):
    """Shape checks shared by all paths: size, aspect, and the bottom edge must be a sill
    above the floor (the gap under a board, under a table, beside a tunnel all reach the
    floor)."""
    w, h = hv[1] - hv[0], hv[3] - hv[2]
    if w > p.max_size_m or h > p.max_size_m:
        return 'too_large'
    if max(w / max(h, 1e-6), h / max(w, 1e-6)) > p.max_aspect:
        return 'bad_aspect'
    if floor_z is not None:
        z_bottom = G.c[2] + hv[3] * G.e_v[2] + max(abs(hv[0]), abs(hv[1])) * abs(G.e_h[2])
        if floor_z - z_bottom < p.min_sill_m:
            return 'touches_floor'
    return None


def _prior_in_view(prior, K, R_nc, cam, shape, p):
    if prior is None:
        return True
    pc = (prior - cam) @ R_nc
    if pc[2] < p.min_range or np.linalg.norm(prior - cam) > p.fallback_max_range:
        return False
    fx, fy, cx, cy = K
    u, v = fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy
    H, W = shape
    return bool(-0.1 * W <= u <= 1.1 * W and -0.1 * H <= v <= 1.1 * H)


def detect(frame, half_span_m, half_height_m, prior_center_ned=None, known_size_m=None,
           margin_m=0.06, params=None, floor_z_ned=None):
    """Detect the window. half_span_m / half_height_m: vehicle half extents incl. props.
    prior_center_ned: coarse measured window centre (+-0.5 m) or None.
    known_size_m: optional (width, height) measured on site (enables the image-only PnP
    fallback and breaks ties). margin_m: clearance required on every side for clearance_ok.
    floor_z_ned: floor height (NED z, e.g. from the down ToF); estimated from depth if None.
    """
    p = params or WindowParams()
    prior = None if prior_center_ned is None else np.asarray(prior_center_ned, float).reshape(3)
    K = tuple(float(x) for x in frame.K)
    R_nc = np.asarray(frame.R_nc, float)
    cam = np.asarray(frame.cam_ned, float).reshape(3)
    depth = np.asarray(frame.depth, np.float32)
    gray = _gray(frame.image)
    img_f, lstd = None, None
    if gray is not None:
        img_f = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
        m1 = cv2.boxFilter(img_f, -1, (7, 7))
        m2 = cv2.boxFilter(img_f * img_f, -1, (7, 7))
        lstd = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))   # local grey-level std
        if not p.edge_refine:
            img_f = None

    # 1. planes
    s = p.stride_fit
    rc = _rays(depth.shape, K, s)
    dz = depth[::s, ::s].reshape(-1)
    ok = np.isfinite(dz) & (dz > p.min_range) & (dz < p.max_range)
    P = cam + (rc[ok] * dz[ok][:, None]) @ R_nc.T
    if floor_z_ned is not None:
        floor_z = float(floor_z_ned)
    else:
        with np.errstate(invalid='ignore'):
            floor_z = _floor_z(np.where(ok, dz.astype(np.float64), np.nan).reshape(len(range(0, depth.shape[0], s)), -1),
                               rc @ R_nc.T, cam, K[1], s)
    planes = _find_planes(P, dz[ok], cam, prior, p, floor_z) if ok.sum() >= p.min_plane_pts else []

    # 2-3. depth holes on each plane
    cands, grids, reasons, cut_strong = [], [], [], False
    for pl in planes:
        G = _plane_grid(depth, K, R_nc, cam, pl, p, gray, lstd, floor_z)
        if G is None:
            continue
        grids.append((G, pl))
        plane_q = min(1.0, pl['n_in'] / 300.0)
        for cd in _grid_candidates(G, p):
            stats = [x[1] for x in cd['sides']]
            if any(x == 'open' for x in stats):
                reasons.append('not_enclosed')
                continue
            if any(x == 'thin' for x in stats):
                reasons.append('thin_frame')
                continue
            if any(x == 'fov' for x in stats):
                reasons.append('cut_by_fov')
                cut_strong = cut_strong or sum(x == 'board' for x in stats) >= 2
                continue
            if cd['fill'] < p.min_fill:
                reasons.append('not_rectangular')
                continue
            hv = (cd['h0'], cd['h1'], cd['v0'], cd['v1'])
            bad = _geom_reject(G, hv, floor_z, p)
            if bad:
                reasons.append(bad)
                continue
            center = G.c + 0.5 * (hv[0] + hv[1]) * G.e_h + 0.5 * (hv[2] + hv[3]) * G.e_v
            if np.linalg.norm(center - cam) > p.max_window_range:
                reasons.append('too_far')
                continue
            pq = _prior_q(center, prior, p)
            if pq <= 0.0:
                reasons.append('far_from_prior')
                continue
            ring = min(x[2] for x in cd['sides'])
            ring_q = float(np.clip((ring - 0.4) / 0.5, 0.0, 1.0))
            fill_q = float(np.clip((cd['fill'] - 0.6) / 0.35, 0.0, 1.0))
            sq = _size_q(hv[1] - hv[0], hv[3] - hv[2], known_size_m)
            score = plane_q * math.sqrt(ring_q * fill_q) * pq * sq * (0.8 + 0.2 * min(1.0, 2 * cd['beyond']))
            if score > 0:
                cands.append((score, len(cands), G, pl, cd, hv))
    cands.sort(key=lambda c: (-c[0], c[1]))
    best = cands[0] if cands else None
    for score, _i, G, pl, cd, hv in cands[:3]:      # best first; a rejected one falls through
        nref, tilt = 0, float('nan')
        if img_f is not None:
            hv, nref, tilt = _refine_edges(img_f, K, R_nc, cam, G, hv, p, return_tilt=True)
        rng_m = float(np.linalg.norm(G.c + 0.5 * (hv[0] + hv[1]) * G.e_h + 0.5 * (hv[2] + hv[3]) * G.e_v - cam))
        range_q = float(np.clip(1.0 - 0.5 * (rng_m - 3.0) / max(p.max_window_range - 3.0, 1e-3), 0.5, 1.0))
        conf = float(np.clip(score * (0.75 + 0.25 * nref / 4.0) * range_q, 0.0, 1.0))
        if prior is None and img_f is not None and nref < p.min_edges_no_prior:
            # a real opening has crisp board|hall edges; depth-only "holes" (fabric crossing
            # the plane slab, under tables) do not
            conf = 0.0
            reasons.append('no_image_edges')
        elif prior is None and img_f is not None and conf > 0:
            # ... and a board that is visibly a different surface from the scene behind it:
            # a 'hole' in a net / mesh wall (stereo returns on the cage net) or in a patchy
            # depth surface shows the same grey level on both sides of its edges
            steps = _side_steps(img_f, K, R_nc, cam, G, hv)
            if sum(x >= p.side_step_min for x in steps if np.isfinite(x)) < p.min_step_sides:
                conf = 0.0
                reasons.append('no_board_step')
        if conf >= p.min_conf:
            return _estimate(G, hv, cam, half_span_m, half_height_m, margin_m, conf,
                             'depth+image' if nref else 'depth', [x[:2] for x in cd['sides']],
                             pl['n_in'], p, nref=nref, tilt=tilt)
        if conf > 0:
            reasons.append('low_confidence')

    # 4. fallbacks (pointless when depth already saw the opening cut by the FOV)
    if best is None and cut_strong:
        return _empty('cut_by_fov', max(pl['n_in'] for pl in planes))
    # the image fallbacks cost 20-40 ms: with a prior, only when the prior window is in view
    # and within fallback_max_range (farther detections are not used for alignment anyway)
    run_fb = p.image_fallback and gray is not None and _prior_in_view(prior, K, R_nc, cam, depth.shape, p)
    quads = None
    # 4a. prior + measured size: known-size rectangle search on the rectified board
    #     (4b below: plane + Hough image quads; 4c: PnP when no plane)
    if run_fb and grids and prior is not None and known_size_m is not None and img_f is not None:
        bestk = None
        for G, pl in grids:
            if abs(float((prior - G.c) @ G.n)) > p.prior_gate_m:
                continue
            hv = _known_size_search(img_f, K, R_nc, cam, G, prior, known_size_m, p)
            if hv is None:
                continue
            hv, nref, tilt = _refine_edges(img_f, K, R_nc, cam, G, hv, p, return_tilt=True)
            if nref < 3 or _geom_reject(G, hv, floor_z, p):
                continue
            steps = _side_steps(img_f, K, R_nc, cam, G, hv)
            if sum(x >= p.side_step_min for x in steps if np.isfinite(x)) < p.min_step_sides:
                continue
            x0 = int(round((hv[0] - G.h_lo) / G.cell)); x1 = int(round((hv[1] - G.h_lo) / G.cell))
            y0 = int(round((hv[2] - G.v_lo) / G.cell)); y1 = int(round((hv[3] - G.v_lo) / G.cell))
            sides = _sides(G, (x0, y0, x1, y1), p)
            if (any(x[1] == 'fov' for x in sides) or min(x[2] for x in sides) < 0.2
                    or max(x[4] for x in sides) > 0.25):
                continue
            center = G.c + 0.5 * (hv[0] + hv[1]) * G.e_h + 0.5 * (hv[2] + hv[3]) * G.e_v
            pq = _prior_q(center, prior, p)
            if pq <= 0 or np.linalg.norm(center - cam) > p.max_window_range:
                continue
            ringb = float(np.mean([x[2] for x in sides]))
            conf = float(0.7 * min(1.0, pl['n_in'] / 300.0) * (0.5 + 0.5 * ringb) * (0.5 + 0.5 * nref / 4.0) * pq)
            if bestk is None or conf > bestk[0]:
                bestk = (conf, G, pl, hv, sides, nref, tilt)
        if bestk is not None and bestk[0] >= p.min_conf:
            conf, G, pl, hv, sides, nref, tilt = bestk
            return _estimate(G, hv, cam, half_span_m, half_height_m, margin_m, conf, 'prior+size',
                             [x[:2] for x in sides], pl['n_in'], p, nref=nref, tilt=tilt)
    # 4b. board plane + Hough image quads back-projected onto it
    if run_fb and grids:
        quads = find_window_quads(gray, max_quads=8, up_px=_up_px(R_nc))
        bestq = None
        for G, pl in grids:
            for qd in quads:
                hvq = _ray_plane_hv(G, K, R_nc, cam, qd['quad'])
                if hvq is None:
                    continue
                (htl, vtl), (htr, vtr), (hbr, vbr), (hbl, vbl) = hvq
                h0, h1 = 0.5 * (htl + hbl), 0.5 * (htr + hbr)
                v0, v1 = 0.5 * (vtl + vtr), 0.5 * (vbl + vbr)
                w, h = h1 - h0, v1 - v0
                if w < p.min_size_m or h < p.min_size_m or w > p.max_size_m or h > p.max_size_m:
                    continue
                if max(abs(htl - hbl), abs(htr - hbr)) > 0.15 * w or max(abs(vtl - vtr), abs(vbl - vbr)) > 0.15 * h:
                    continue                       # not a level rectangle on this plane
                x0 = int(round((h0 - G.h_lo) / G.cell)); x1 = int(round((h1 - G.h_lo) / G.cell))
                y0 = int(round((v0 - G.v_lo) / G.cell)); y1 = int(round((v1 - G.v_lo) / G.cell))
                sides = _sides(G, (x0, y0, x1, y1), p)
                if any(x[1] in ('fov', 'thin') for x in sides) or _geom_reject(G, (h0, h1, v0, v1), floor_z, p):
                    continue
                # weak board depth: accept sparse board around it, but never a side whose
                # depth clearly sees through (slit between boards, space beside / under a board)
                # or whose board is only a thin rail (the band beyond it is open)
                if (min(x[2] for x in sides) < 0.2 or max(x[4] for x in sides) > 0.25
                        or max(x[5] for x in sides) > 0.5):
                    continue
                # ... and its inside must not be board (a rectangle painted / printed on a
                # wall, or a checker block of the far hall, lies ON its plane)
                xa, xb = max(x0, 0), min(x1, G.Wg)
                ya, yb = max(y0, 0), min(y1, G.Hg)
                if xb - xa < 2 or yb - ya < 2:
                    continue
                img_inside = 1.0
                if G.Bc[ya:yb, xa:xb].mean() > 0.3 or G.Hm[ya:yb, xa:xb].mean() < 0.5:
                    # Stereo (SGM) smooths the board's disparity into a dark, weakly textured
                    # opening, so depth may say 'board' inside a real window. Trust the image
                    # instead only with a prior (and the measured size), and only if every
                    # edge is a strong board|scene step; never without a prior (printed
                    # rectangles on walls / banners look the same to depth).
                    if prior is None or img_f is None:
                        continue
                    if known_size_m is not None and (abs(w - known_size_m[0]) > 0.15
                                                     or abs(h - known_size_m[1]) > 0.15):
                        continue
                    steps = _side_steps(img_f, K, R_nc, cam, G, (h0, h1, v0, v1))
                    if sum(x >= p.side_step_min for x in steps if np.isfinite(x)) < p.min_step_sides:
                        continue
                    img_inside = 0.8
                ringb = float(np.mean([x[2] for x in sides]))
                hv = (h0, h1, v0, v1)
                center = G.c + 0.5 * (h0 + h1) * G.e_h + 0.5 * (v0 + v1) * G.e_v
                pq = _prior_q(center, prior, p)
                sc = qd['score'] * ringb * pq * _size_q(w, h, known_size_m) * img_inside
                if pq > 0 and (bestq is None or sc > bestq[0]):
                    bestq = (sc, G, pl, hv, sides, img_inside)
        if bestq is not None:
            sc, G, pl, hv, sides, img_inside = bestq
            nref, tilt = 0, float('nan')
            if img_f is not None:
                hv, nref, tilt = _refine_edges(img_f, K, R_nc, cam, G, hv, p, return_tilt=True)
            ringb = float(np.mean([x[2] for x in sides]))
            conf = float(np.clip(0.75 * img_inside * min(1.0, pl['n_in'] / 300.0) * (0.5 + 0.5 * ringb) * (0.5 + 0.5 * nref / 4.0)
                                 * _prior_q(G.c + 0.5 * (hv[0] + hv[1]) * G.e_h + 0.5 * (hv[2] + hv[3]) * G.e_v, prior, p), 0, 1))
            if conf >= p.min_conf:
                return _estimate(G, hv, cam, half_span_m, half_height_m, margin_m, conf,
                                 'plane+image', [x[:2] for x in sides], pl['n_in'], p, nref=nref, tilt=tilt)
            reasons.append('plane_image_low_conf')

    # 4c. no usable board plane near the prior: image quad + measured size -> PnP
    near_plane = prior is not None and any(abs(float((prior - G.c) @ G.n)) <= p.prior_gate_m for G, _ in grids)
    if run_fb and known_size_m is not None and not near_plane:
        if quads is None:
            quads = find_window_quads(gray, max_quads=8, up_px=_up_px(R_nc))
        est = _pnp_fallback(quads, depth, K, R_nc, cam, known_size_m, prior, half_span_m,
                            half_height_m, margin_m, p)
        if est is not None:
            return est
    if not planes:
        return _empty('no_board_plane')
    # most frequent reason; ties -> first seen (a set would order by per-process string hash)
    why = 'no_hole' if not reasons else max(reasons, key=reasons.count)
    return _empty(why, max(pl['n_in'] for pl in planes))


def _pnp_fallback(quads, depth, K, R_nc, cam, known_size_m, prior, half_span_m, half_height_m,
                  margin_m, p):
    """No usable board plane: planar PnP of image quads with the site-measured size. A
    candidate whose inside has depth ON or IN FRONT of the implied board plane is a
    rectangle on some surface (hall wall, poster), not an opening."""
    fx, fy, cx, cy = K
    Km = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    w, h = float(known_size_m[0]), float(known_size_m[1])
    obj = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]])
    best = None
    gi = np.linspace(0.1, 0.9, 9)
    si, ti = np.meshgrid(gi, gi)
    Hd, Wd = depth.shape
    for qd in quads:
        okp, rvec, tvec = cv2.solvePnP(obj, qd['quad'].astype(np.float64), Km, None,
                                       flags=cv2.SOLVEPNP_IPPE)
        if not okp:
            continue
        Rm, _ = cv2.Rodrigues(rvec)
        center = cam + R_nc @ tvec.reshape(3)
        n = R_nc @ (-Rm[:, 2])
        n[2] = 0.0                                 # boards are vertical
        if np.linalg.norm(n) < 0.5:
            continue
        n /= np.linalg.norm(n)
        if n @ (cam - center) < 0:
            n = -n
        rng_m = float(np.linalg.norm(center - cam))
        if not (p.min_range < rng_m < p.max_window_range):
            continue
        uv = _bilinear(qd['quad'][None].astype(np.float64), si.ravel(), ti.ravel())[0]
        ui = np.clip(np.round(uv[:, 0]).astype(int), 0, Wd - 1)
        vi = np.clip(np.round(uv[:, 1]).astype(int), 0, Hd - 1)
        z = depth[vi, ui]
        okz = np.isfinite(z) & (z > 0.05)
        if okz.sum() >= 8:
            nc, t0 = Rm[:, 2], tvec.reshape(3)
            ray = np.stack([(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy, np.ones(len(uv))], -1)
            tp = (nc @ t0) / (ray @ nc)
            near = z[okz] < tp[okz] + np.maximum(_thr(p, tp[okz]), 0.1)
            if near.mean() > 0.4:
                continue
        pq = _prior_q(center, prior, p)
        sc = qd['quality'] * pq
        if pq > 0 and (best is None or sc > best[0]):
            best = (sc, center, n, qd['quality'])
    if best is None:
        return None
    sc, center, n, quality = best
    G = _Grid()
    G.n = n
    G.e_h, G.e_v = _plane_axes(n)
    G.c = center
    conf = float(np.clip(0.45 * quality * _prior_q(center, prior, p), 0, 0.45))
    if conf < p.min_conf:
        return None
    return _estimate(G, (-w / 2, w / 2, -h / 2, h / 2), cam, half_span_m, half_height_m,
                     margin_m, conf, 'image+size', (), 0, p,
                     reason='size_from_site_measurement')
