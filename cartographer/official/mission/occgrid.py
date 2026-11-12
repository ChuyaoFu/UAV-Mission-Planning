"""Persistent 2-D occupancy grid for map-mode maze navigation (maze-planner-design.md, design v1).

Pure Python + numpy, no ROS, nothing here reads a clock (every stamp comes from the caller), same style as core.py.

* ``GridFrame``: a 2-D rigid frame on the PX4 NED xy plane (origin + two unit axes).  The grid is axis-aligned in
  it, so a fence that is a rotated rectangle in NED (start pose changed by the organisers) is still a box here.
  ``GridFrame.from_start(home_ned, yaw)`` = the mission START frame (u = forward, v = left).
* ``Scan``: one horizontal scan -- per bearing the horizontal range of the nearest return inside the flight band.
  Scan sources are pluggable:
    - ``Scan`` from the node: ``scan_from_laserscan`` converts /team13/depth_scan (realdrone/depth_scan.py: per
      depth column the nearest return in the flight band; NaN = no information, never carved) -- the primary source;
    - ``PointsScanSource`` (works today, no node change): a pseudo-scan from the node's ObstacleMap points inside
      the current camera field of view.
* ``OccGrid``: log-odds grid over the fence box.  Ray update (free cells along the ray, +occupancy at the hit), the
  camera's stereo blind zone (``min_range``) is never updated, the fence border is permanently occupied.
  Multi-view confirmation: a cell is *confirmed* occupied only once its log-odds reach ``l_conf`` AND it was hit
  from two camera positions >= ``conf_baseline_m`` apart or from bearings >= ``conf_angle_deg`` apart.  Free-space
  stereo noise is view dependent: later rays through it carve it before a second view confirms it.  Everything
  with occupancy evidence that is not confirmed is *suspect* (planning avoids it, never trusts it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------- frames
class GridFrame:
    """2-D frame on the NED xy plane: (u, v) = ((p - o) . ax, (p - o) . ay)."""

    def __init__(self, origin_ned, ax, ay):
        self.o = np.asarray(origin_ned, float).reshape(-1)[:2].copy()
        self.ax = np.asarray(ax, float).reshape(2) / float(np.linalg.norm(ax))
        self.ay = np.asarray(ay, float).reshape(2) / float(np.linalg.norm(ay))
        self.M = np.vstack([self.ax, self.ay])          # (2, 2): NED xy -> (u, v)

    @classmethod
    def from_start(cls, home_ned, yaw: float) -> 'GridFrame':
        """The mission START frame (core.StartFrame): u = forward, v = left."""
        c, s = math.cos(yaw), math.sin(yaw)
        return cls(home_ned, (c, s), (s, -c))

    def to_uv(self, p_ned) -> np.ndarray:
        p = np.asarray(p_ned, float)
        return (p[..., :2] - self.o) @ self.M.T

    def to_ned_xy(self, uv) -> np.ndarray:
        q = np.asarray(uv, float)
        return self.o + q @ self.M

    def dir_uv(self, yaw_ned) -> np.ndarray:
        """Unit vector(s) of NED yaw angle(s) in (u, v)."""
        y = np.asarray(yaw_ned, float)
        d = np.stack([np.cos(y), np.sin(y)], axis=-1)
        return d @ self.M.T


# ---------------------------------------------------------------- scans
@dataclass
class Scan:
    """One horizontal scan.  bearings: rad relative to ``heading`` in the NED yaw sense (positive = clockwise seen
    from above = to the right).  ranges: horizontal range (m) of the nearest return in the flight band; inf / nan =
    no return within ``max_range`` (the ray is carved to ``max_range``).  valid False = no information (skipped)."""
    stamp: float
    cam_ned: Any
    heading: float
    bearings: Any
    ranges: Any
    valid: Any = None
    max_range: float = 3.0
    hit_ned: Any = None          # optional (n, 2+) actual NED position of each bin's return (nan: none); the hit
                                 # cell is taken from it instead of (bearing, range) -- a pseudo-scan bin's nearest
                                 # point may lie at another bearing (grazing walls would be pulled into the lane)
    carve: Any = None            # optional (n,) free range for rays with no return (default max_range)


def scan_from_points(stamp, cam_ned, heading, points_ned, band_d, hfov_deg=66.0, edge_deg=3.0, bin_deg=1.0,
                     min_range=0.40, max_range=3.0, noreturn_m=2.0, voxel_m=0.05, hole_bins=3) -> Scan:
    """Pseudo-scan from 3-D obstacle points (NED): points inside the flight band (NED z in band_d = (top, bottom)),
    inside the horizontal FOV (minus ``edge_deg`` per side) and min_range..max_range (horizontal) -> nearest range
    per ``bin_deg`` bearing bin.  Each point covers the bearings of its voxel (``voxel_m``: 5 cm voxels are 3 deg
    apart at 1 m -- without this the bins between them are holes that carve through the wall or see the wall behind).
    Bins with no point carve to ``noreturn_m`` at most, and no farther than the nearest return within ``hole_bins``
    bins (a textureless patch or a voxel hole in a wall must not erase it)."""
    cam = np.asarray(cam_ned, float).reshape(3)
    half = 0.5 * hfov_deg - edge_deg
    nb = max(1, int(round(2.0 * half / bin_deg)))
    edges = np.linspace(-half, half, nb + 1)
    centres = np.radians(0.5 * (edges[:-1] + edges[1:]))
    rng = np.full(nb, np.inf)
    P = np.asarray(points_ned, float).reshape(-1, 3)
    hit_xy = np.full((nb, 2), np.nan)
    if len(P):
        d = P - cam
        lo, hi = min(band_d), max(band_d)
        r = np.hypot(d[:, 0], d[:, 1])
        b = np.degrees(np.arctan2(d[:, 1], d[:, 0]) - heading)
        b = (b + 180.0) % 360.0 - 180.0
        m = (P[:, 2] >= lo) & (P[:, 2] <= hi) & (r >= min_range) & (r <= max_range) & (np.abs(b) < half)
        hit_xy = np.full((nb, 2), np.nan)
        if m.any():
            bm, rm, Pm = b[m], r[m], P[m]
            w = np.degrees(np.arctan2(0.5 * voxel_m * math.sqrt(2.0), rm))     # half angular width of the voxel
            lo_k = np.clip(np.floor((bm - w + half) / (2.0 * half) * nb).astype(int), 0, nb - 1)
            hi_k = np.clip(np.floor((bm + w + half) / (2.0 * half) * nb).astype(int), 0, nb - 1)
            ks, rs, ps = [], [], []
            idx = np.arange(len(bm))
            for o in range(int((hi_k - lo_k).max()) + 1):
                sel = lo_k + o <= hi_k
                ks.append(lo_k[sel] + o)
                rs.append(rm[sel])
                ps.append(idx[sel])
            K, R, I = np.concatenate(ks), np.concatenate(rs), np.concatenate(ps)
            order = np.lexsort((R, K))                       # per bin, nearest first
            ub, first = np.unique(K[order], return_index=True)
            rng[ub] = R[order][first]
            hit_xy[ub] = Pm[I[order][first], :2]
    carve = np.full(nb, float(noreturn_m))
    fin = np.isfinite(rng)
    if fin.any() and hole_bins > 0:
        pad = np.concatenate([np.full(hole_bins, np.inf), rng, np.full(hole_bins, np.inf)])
        near = np.min(np.stack([pad[o:o + nb] for o in range(2 * hole_bins + 1)]), axis=0)
        carve = np.where(~fin & np.isfinite(near), np.minimum(carve, near), carve)
    return Scan(stamp=float(stamp), cam_ned=cam.copy(), heading=float(heading), bearings=centres, ranges=rng,
                valid=np.ones(nb, bool), max_range=float(noreturn_m), hit_ned=hit_xy, carve=carve)


def scan_from_laserscan(stamp, R_nc, cam_ned, angle_min, angle_increment, ranges, range_max, carve_max_m=3.0) -> Scan:
    """/team13/depth_scan (realdrone/depth_scan.py, sensor_msgs/LaserScan) -> Scan.  Scan frame = the rectified left
    camera centre, x = optical z, y = optical -x, z = optical -y, angle counter-clockwise from x; R_nc / cam_ned: that
    camera (optical -> NED) at the scan stamp (the node has them for the depth image of the same stamp).
    finite = nearest in-band return (a hit at its real NED position), +inf = free to range_max (carved to
    min(range_max, carve_max_m)), NaN = no in-band data (no information: never carved)."""
    r = np.asarray(ranges, float).reshape(-1)
    n = len(r)
    a = float(angle_min) + float(angle_increment) * np.arange(n)
    d_opt = np.stack([-np.sin(a), np.zeros(n), np.cos(a)], axis=1)       # scan (cos, sin, 0) -> optical
    d = d_opt @ np.asarray(R_nc, float).T
    h = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
    cam = np.asarray(cam_ned, float).reshape(3)
    fin = np.isfinite(r)
    free = np.isposinf(r)
    rh = np.where(fin, np.where(fin, r, 0.0) * h, np.inf)
    hit = np.full((n, 2), np.nan)
    hit[fin] = cam[None, :2] + r[fin, None] * d[fin, :2]
    carve = np.where(free, min(float(range_max), carve_max_m) * h, 0.0)
    return Scan(stamp=float(stamp), cam_ned=cam.copy(), heading=0.0, bearings=np.arctan2(d[:, 1], d[:, 0]),
                ranges=rh, valid=fin | free, max_range=float(max(carve_max_m, np.nanmax(rh[fin]) if fin.any() else 0.0)),
                hit_ned=hit, carve=carve)


class PointsScanSource:
    """Fallback scan source (no node change): pseudo-scan from the node's ObstacleMap.

    Uses only voxels refreshed within ``fresh_s`` of the newest stamp when the map exposes them (``vox`` dict of
    [hits, last_stamp, x, y, z], node/obstacle_map.py): remembered points (4 s memory) are not re-observations and
    must not collect "views".  Otherwise ``.points``."""

    def __init__(self, hfov_deg=66.0, band_up=(0.30, 1.50), bin_deg=1.0, min_range=0.40, max_range=3.0,
                 noreturn_m=2.0, fresh_s=0.25, min_hits=3):
        self.hfov_deg, self.band_up, self.bin_deg = hfov_deg, band_up, bin_deg
        self.min_range, self.max_range, self.noreturn_m = min_range, max_range, noreturn_m
        self.fresh_s, self.min_hits = fresh_s, min_hits

    def points(self, obstacles) -> np.ndarray:
        vox = getattr(obstacles, 'vox', None)
        if isinstance(vox, dict) and vox:
            V = np.asarray(list(vox.values()), float).reshape(-1, 5)
            V = V[V[:, 0] >= self.min_hits]
            if len(V):
                V = V[V[:, 1] >= V[:, 1].max() - self.fresh_s]
            return V[:, 2:5]
        pts = getattr(obstacles, 'points', None)
        return np.zeros((0, 3)) if pts is None else np.asarray(pts, float).reshape(-1, 3)

    def __call__(self, stamp, obstacles, cam_ned, heading, floor_d) -> Optional[Scan]:
        if obstacles is None:
            return None
        band = (floor_d - self.band_up[1], floor_d - self.band_up[0])
        return scan_from_points(stamp, cam_ned, heading, self.points(obstacles), band, hfov_deg=self.hfov_deg,
                                bin_deg=self.bin_deg, min_range=self.min_range, max_range=self.max_range,
                                noreturn_m=self.noreturn_m)


# ---------------------------------------------------------------- distance transform
def edt(mask: np.ndarray) -> np.ndarray:
    """Exact Euclidean distance (in cells) from every cell centre to the nearest True cell centre; inf if none.
    Two passes (column distances, then a broadcast lower envelope per row): O(H W^2) numpy, ~2 ms at 122 x 122."""
    H, W = mask.shape
    if not mask.any():
        return np.full(mask.shape, np.inf)
    idx = np.arange(H)[:, None] * np.ones((1, W))
    last = np.where(mask, idx, -np.inf)
    last = np.maximum.accumulate(last, axis=0)
    nxt = np.where(mask, idx, np.inf)
    nxt = np.minimum.accumulate(nxt[::-1], axis=0)[::-1]
    g = np.minimum(idx - last, nxt - idx)                     # per column, distance along the column (cells)
    g2 = np.where(np.isfinite(g), g * g, np.inf)
    j = np.arange(W, dtype=float)
    dj2 = (j[:, None] - j[None, :]) ** 2                      # (W, W)
    out = np.empty((H, W))
    step = max(1, int(4_000_000 // max(1, W * W)))
    for a in range(0, H, step):
        blk = g2[a:a + step]                                  # (h, W)
        out[a:a + step] = np.min(blk[:, None, :] + dj2[None, :, :], axis=2)
    return np.sqrt(out)


# ---------------------------------------------------------------- the grid
class OccGrid:
    """Persistent log-odds occupancy grid over the fence box (grid frame (u, v) metres).

    box = (u_min, u_max, v_min, v_max): the fence (inside).  Cells whose centre lies outside the box are the fence:
    permanently occupied (they also form the planner's outer wall)."""

    def __init__(self, frame: GridFrame, box: Sequence[float], res=0.05, l_free=0.4, l_occ=0.85, l_min=-4.0,
                 l_max=6.0, l_conf=2.0, l_susp=0.5, l_known_free=-0.2, conf_baseline_m=0.25, conf_angle_deg=15.0,
                 min_range=0.40, hit_keep_m=None, margin_cells=2, keep_k=0.0, stale_s=None):
        self.frame = frame
        self.box = tuple(float(x) for x in box)
        self.res = float(res)
        self.l_free, self.l_occ, self.l_min, self.l_max = l_free, l_occ, l_min, l_max
        self.l_conf, self.l_susp, self.l_known_free = l_conf, l_susp, l_known_free
        self.conf_baseline_m, self.conf_angle = conf_baseline_m, math.radians(conf_angle_deg)
        self.min_range = float(min_range)
        self.hit_keep_m = self.res if hit_keep_m is None else float(hit_keep_m)
        self.keep_k = float(keep_k)       # range-dependent keep-back before a hit (m per m^2): real stereo range noise
                                          # (half320 SGBM ~3 % at 2 m) must not let one ray carve the next ray's hit
        self.stale_s = stale_s            # VIO drifts ~3 % of the distance flown (real hand-carried loops 2026-11-08):
                                          # a confirmed cell not hit again for stale_s counts as suspect only (still
                                          # avoided, never trusted as a confirmed wall) until it is re-confirmed
        u0, u1, v0, v1 = self.box
        m = int(margin_cells)
        self.u0 = u0 - m * self.res
        self.v0 = v0 - m * self.res
        self.nu = int(math.ceil((u1 - u0) / self.res)) + 2 * m
        self.nv = int(math.ceil((v1 - v0) / self.res)) + 2 * m
        cu = self.u0 + (np.arange(self.nu) + 0.5) * self.res
        cv = self.v0 + (np.arange(self.nv) + 0.5) * self.res
        self.cu, self.cv = cu, cv
        inside = (cu[:, None] >= u0) & (cu[:, None] <= u1) & (cv[None, :] >= v0) & (cv[None, :] <= v1)
        self.fence = ~inside
        self.fence[0, :] = self.fence[-1, :] = self.fence[:, 0] = self.fence[:, -1] = True
        self.L = np.zeros((self.nu, self.nv))
        self.seen = np.zeros((self.nu, self.nv), bool)
        self.view0 = np.full((self.nu, self.nv, 2), np.nan)   # first-hit camera position (u, v)
        self.ang0 = np.full((self.nu, self.nv), np.nan)       # first-hit bearing (grid frame, rad)
        self.multi = np.zeros((self.nu, self.nv), bool)
        self.last_hit = np.full((self.nu, self.nv), -np.inf)
        self.extra = np.zeros((self.nu, self.nv), bool)       # caller-set permanent occupancy (e.g. a prior keep-out)
        self.prior = np.zeros((self.nu, self.nv), bool)       # soft prior layer (the prior's wall lines): costly, never
                                                              # blocking; cleared where a ray with a return behind it
                                                              # passes (the wall is not there: the maze changed)
        self.scans = 0
        self.last_stamp = None

    # ---- coordinates
    def cell_of_uv(self, uv) -> np.ndarray:
        q = np.asarray(uv, float)
        i = np.floor((q[..., 0] - self.u0) / self.res).astype(int)
        j = np.floor((q[..., 1] - self.v0) / self.res).astype(int)
        return np.stack([i, j], axis=-1)

    def uv_of_cell(self, ij) -> np.ndarray:
        c = np.asarray(ij, float)
        return np.stack([self.u0 + (c[..., 0] + 0.5) * self.res, self.v0 + (c[..., 1] + 0.5) * self.res], axis=-1)

    def to_cell(self, p_ned) -> np.ndarray:
        return self.cell_of_uv(self.frame.to_uv(p_ned))

    def to_ned_xy(self, ij) -> np.ndarray:
        return self.frame.to_ned_xy(self.uv_of_cell(ij))

    def in_grid(self, ij) -> np.ndarray:
        c = np.asarray(ij)
        return (c[..., 0] >= 0) & (c[..., 0] < self.nu) & (c[..., 1] >= 0) & (c[..., 1] < self.nv)

    # ---- update
    def update_scan(self, scan: Scan, step_m=None) -> dict:
        """One scan: free along each valid ray from min_range to (hit - hit_keep_m) or to max_range (no return), +l_occ
        at the hit cell.  Fence cells never change; nothing inside min_range of the camera changes."""
        cam = np.asarray(scan.cam_ned, float).reshape(-1)
        cuv = self.frame.to_uv(cam)
        b = np.asarray(scan.bearings, float).reshape(-1)
        r = np.asarray(scan.ranges, float).reshape(-1)
        ok = np.ones(len(b), bool) if scan.valid is None else np.asarray(scan.valid, bool).reshape(-1).copy()
        ok &= np.isfinite(b)
        hit = ok & np.isfinite(r) & (r <= scan.max_range + 1e-9)
        ok &= ~(hit & (r < self.min_range))                     # a return inside the blind zone: no information
        hit &= ok
        dirs = self.frame.dir_uv(scan.heading + b)              # (n, 2)
        nr = getattr(scan, 'carve', None)
        nr = np.full(len(b), float(scan.max_range)) if nr is None else np.minimum(np.asarray(nr, float), scan.max_range)
        keep = np.maximum(self.hit_keep_m, self.keep_k * np.where(np.isfinite(r), r, 0.0) ** 2)
        r_free = np.where(hit, r - keep, nr - self.hit_keep_m)
        r_free = np.where(ok, r_free, -1.0)
        step = 0.5 * self.res if step_m is None else float(step_m)
        top = float(r_free.max()) if len(r_free) else -1.0
        n_free = n_hit = 0
        free_idx = np.zeros(0, int)
        if top >= self.min_range:
            S = np.arange(self.min_range, top + 1e-9, step)
            m = S[None, :] <= r_free[:, None]
            P = cuv[None, None, :] + S[None, :, None] * dirs[:, None, :]
            ij = self.cell_of_uv(P[m])
            ing = self.in_grid(ij)
            free_idx = np.unique(ij[ing, 0] * self.nv + ij[ing, 1])
        if top >= self.min_range and self.prior.any():          # prior cells seen through (a return BEHIND them)
            mh = m & hit[:, None]
            if mh.any():
                ijh = self.cell_of_uv(P[mh])
                ok = self.in_grid(ijh)
                self.prior[ijh[ok, 0], ijh[ok, 1]] = False
        hit_idx = np.zeros(0, int)
        if hit.any():
            H = cuv[None, :] + r[hit, None] * dirs[hit]
            if scan.hit_ned is not None:              # the return's own position (pseudo-scan)
                hx = np.asarray(scan.hit_ned, float).reshape(len(b), -1)[hit, :2]
                okx = np.isfinite(hx).all(axis=1)
                H[okx] = self.frame.to_uv(hx[okx])
            ij = self.cell_of_uv(H)
            ing = self.in_grid(ij)
            hit_idx, first = np.unique(ij[ing, 0] * self.nv + ij[ing, 1], return_index=True)
            Hc = H[ing][first]
        fence = self.fence.ravel()
        if len(free_idx):
            free_idx = free_idx[~fence[free_idx]]
            if len(hit_idx):
                free_idx = np.setdiff1d(free_idx, hit_idx, assume_unique=True)
            L = self.L.ravel()
            L[free_idx] = np.maximum(L[free_idx] - self.l_free, self.l_min)
            gone = free_idx[L[free_idx] < 0.0]                   # carved below even: forget its views
            if len(gone):
                self.view0.reshape(-1, 2)[gone] = np.nan
                self.ang0.ravel()[gone] = np.nan
                self.multi.ravel()[gone] = False
            self.seen.ravel()[free_idx] = True
            n_free = len(free_idx)
        if len(hit_idx):
            keep = ~fence[hit_idx]
            hit_idx, Hc = hit_idx[keep], Hc[keep]
            L = self.L.ravel()
            L[hit_idx] = np.minimum(L[hit_idx] + self.l_occ, self.l_max)
            self.seen.ravel()[hit_idx] = True
            self.last_hit.ravel()[hit_idx] = float(scan.stamp)
            v0 = self.view0.reshape(-1, 2)
            a0 = self.ang0.ravel()
            ang = np.arctan2(Hc[:, 1] - cuv[1], Hc[:, 0] - cuv[0])
            new = np.isnan(a0[hit_idx])
            v0[hit_idx[new]] = cuv
            a0[hit_idx[new]] = ang[new]
            old = hit_idx[~new]
            if len(old):
                base = np.linalg.norm(v0[old] - cuv, axis=1)
                da = np.abs((ang[~new] - a0[old] + math.pi) % (2.0 * math.pi) - math.pi)
                conf = (base >= self.conf_baseline_m) | (da >= self.conf_angle)
                self.multi.ravel()[old[conf]] = True
            n_hit = len(hit_idx)
        self.scans += 1
        self.last_stamp = scan.stamp
        return dict(free=n_free, hit=n_hit)

    def mark_free(self, uv, radius_m=0.25):
        """Cells under the vehicle are free (it is there; the camera never sees its own position)."""
        c = self.cell_of_uv(np.asarray(uv, float))
        R = int(math.ceil(radius_m / self.res))
        a0, a1 = max(0, c[0] - R), min(self.nu, c[0] + R + 1)
        b0, b1 = max(0, c[1] - R), min(self.nv, c[1] + R + 1)
        if a0 >= a1 or b0 >= b1:
            return
        ii, jj = np.mgrid[a0:a1, b0:b1]
        m = (np.hypot(ii - c[0], jj - c[1]) * self.res <= radius_m) & ~self.fence[a0:a1, b0:b1]
        sl = (slice(a0, a1), slice(b0, b1))
        L = self.L[sl]
        L[m] = np.minimum(L[m], self.l_known_free - self.l_free)
        self.seen[sl][m] = True

    def set_prior(self, mask: np.ndarray):
        """The soft prior layer (e.g. the prior's wall lines); see cost_map's prior_cost."""
        self.prior = np.asarray(mask, bool) & ~self.fence & ~self.extra

    def prior_live(self) -> np.ndarray:
        """Prior cells not yet seen through and not confirmed (a confirmed cell is a hard block anyway)."""
        return self.prior & ~self.occupied()

    def set_extra(self, mask: np.ndarray):
        """Caller-set permanent occupancy (e.g. a prior keep-out the planner must respect); confirmed, never carved."""
        self.extra = np.asarray(mask, bool) & ~self.fence

    # ---- classification
    def occupied(self) -> np.ndarray:
        """Confirmed (multi-view, hit within stale_s of the latest scan) occupied, the fence, caller-set extra cells."""
        conf = (self.L >= self.l_conf) & self.multi
        if self.stale_s is not None and self.last_stamp is not None:
            conf &= self.last_hit >= float(self.last_stamp) - float(self.stale_s)
        return conf | self.fence | self.extra

    def suspect(self) -> np.ndarray:
        """Occupancy evidence that is not confirmed (single view, or not enough hits yet)."""
        return (self.L >= self.l_susp) & ~self.occupied()

    def known_free(self) -> np.ndarray:
        return self.seen & (self.L <= self.l_known_free) & ~self.fence & ~self.extra

    def unknown(self) -> np.ndarray:
        return ~(self.known_free() | self.occupied() | self.suspect())

    def clearance(self, which='occupied') -> np.ndarray:
        """Distance (m) from each cell centre to the nearest 'occupied' (confirmed + fence) or 'suspect' cell,
        minus half a cell (the measured surface lies anywhere in its cell)."""
        m = self.occupied() if which == 'occupied' else self.suspect()
        d = edt(m) * self.res
        return np.maximum(d - 0.5 * self.res, 0.0)

    def stats(self) -> dict:
        return dict(scans=self.scans, occupied=int((self.occupied() & ~self.fence).sum()),
                    suspect=int(self.suspect().sum()), free=int(self.known_free().sum()),
                    unknown=int((self.unknown() & ~self.fence).sum()))
