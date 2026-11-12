"""Rolling obstacle map in NED from stereo depth. Pure numpy/scipy.

Keeps measured 3-D points that stand above the floor (painted floor texture is
flat and therefore never an obstacle), voxelized, with a short memory so
structure that just left the narrow horizontal camera view (e.g. a wall beside
the vehicle) is still respected. A voxel counts as an obstacle only after it has
been observed in `min_hits` separate frames: single-frame stereo mismatches on
periodic textures (tile grids, nets) must not stop the vehicle.
Unobserved space is NOT treated as an obstacle (user decision: the route ahead
is free unless measured otherwise); the arena net, which stereo barely sees, is
handled by a geofence elsewhere.

Free-space carving (stereo phantoms): SGBM on the regular cage net and on dark
low-texture stone edges produces *consistent* wrong depths, so `min_hits` alone
keeps them. Every frame also re-projects the stored voxels; a voxel whose pixel
now reports a return clearly BEHIND it (>= carve_margin_m) was seen through and
loses `carve_penalty` hits, and is dropped at 0. Real surfaces are never seen
through, so they keep their hits (Ubuntu official-compat replay 2026-11-07:
phantoms in the planner probe 3-7 -> 0-1, real voxels kept 86-94 %).

`exclude` (clearance/blocked): optional list of cylinders
(center_ned, axis_ned, radius_m, half_length_m) whose points are ignored, e.g.
the inside of a latched tunnel/window opening, where stereo phantoms from the
mesh and the background otherwise block the alignment probe.
"""
import math
import numpy as np
from scipy.spatial import cKDTree


class ObstacleMap:
    def __init__(self, voxel_m=0.05, memory_s=4.0, floor_margin_m=0.10, max_range_m=4.0, min_range_m=0.40,
                 min_hits=3, max_points=60000, carve=True, carve_margin_m=0.25, carve_penalty=2):
        self.voxel = voxel_m; self.memory = memory_s; self.floor_margin = floor_margin_m
        self.max_range = max_range_m; self.min_range = min_range_m; self.min_hits = min_hits
        self.max_points = max_points
        self.carve = carve; self.carve_margin = carve_margin_m; self.carve_penalty = carve_penalty
        self.carved = 0                                                  # voxels removed by carving (diagnostics)
        self.vox = {}               # key -> [hits, last_stamp, x, y, z]
        self.tree = None; self.points = np.zeros((0, 3)); self.frames = 0

    def insert(self, frame, floor_d, stamp, stride=4):
        """Add obstacle evidence from one Frame. floor_d: floor height (NED d, metres)."""
        pts, _ = frame.points_ned(stride=stride, max_range=self.max_range)
        if len(pts):                                                     # own airframe/props and sub-minimum stereo range
            pts = pts[np.linalg.norm(pts - frame.cam_ned, axis=1) > self.min_range]
        if len(pts):
            pts = pts[pts[:, 2] < floor_d - self.floor_margin]          # above the floor (NED: smaller d)
        if len(pts):
            keys = np.floor(pts / self.voxel).astype(np.int64)
            keys, idx = np.unique(keys, axis=0, return_index=True)
            for k, p in zip(map(tuple, keys), pts[idx]):
                v = self.vox.get(k)
                if v is None or stamp - v[1] > self.memory:
                    self.vox[k] = [1, stamp, p[0], p[1], p[2]]
                else:
                    v[0] += 1; v[1] = stamp; v[2], v[3], v[4] = p
        if self.carve and self.vox:
            self._carve(frame)
        self.frames += 1
        if self.frames % 5 == 0:                                          # prune stale voxels
            self.vox = {k: v for k, v in self.vox.items() if stamp - v[1] <= self.memory}
        conf = [v[2:] for v in self.vox.values() if v[0] >= self.min_hits and stamp - v[1] <= self.memory]
        allp = np.asarray(conf, float).reshape(-1, 3)
        if len(allp) > self.max_points:
            allp = allp[np.random.default_rng(0).choice(len(allp), self.max_points, replace=False)]
        self.points = allp
        self.tree = cKDTree(allp) if len(allp) else None

    def _carve(self, frame):
        """Remove hits from voxels this frame sees through (depth return clearly behind them)."""
        keys = list(self.vox.keys())
        P = np.array([self.vox[k][2:5] for k in keys], float)
        R = np.asarray(frame.R_nc, float); c = np.asarray(frame.cam_ned, float)
        pc = (P - c) @ R                                                  # NED -> optical camera
        z = pc[:, 2]
        fx, fy, cx, cy = frame.K
        h, w = frame.depth.shape[:2]
        ok = (z > self.min_range) & (z < self.max_range)
        if not ok.any():
            return
        u = np.full(len(z), -1, int); v = np.full(len(z), -1, int)
        u[ok] = np.round(fx * pc[ok, 0] / z[ok] + cx).astype(int)
        v[ok] = np.round(fy * pc[ok, 1] / z[ok] + cy).astype(int)
        ok &= (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1)
        if not ok.any():
            return
        d = np.full(len(z), np.nan)
        d[ok] = frame.depth[v[ok], u[ok]]
        thru = ok & np.isfinite(d) & (d > 0) & (d >= z + self.carve_margin)
        for i in np.nonzero(thru)[0]:
            rec = self.vox[keys[i]]
            rec[0] -= self.carve_penalty
            if rec[0] <= 0:
                del self.vox[keys[i]]; self.carved += 1

    @staticmethod
    def _outside(points, exclude):
        """Mask of points outside every exclusion cylinder (center, axis, radius, half_length)."""
        keep = np.ones(len(points), bool)
        for c, ax, r, hl in exclude or ():
            ax = np.asarray(ax, float); n = float(np.linalg.norm(ax))
            if n < 1e-9:
                continue
            ax = ax / n; rel = points - np.asarray(c, float)
            along = rel @ ax; radial = np.linalg.norm(rel - along[:, None] * ax[None, :], axis=1)
            keep &= ~((np.abs(along) <= hl) & (radial <= r))
        return keep

    def _tree_for(self, exclude):
        if not exclude or self.tree is None:
            return self.tree
        pts = self.points[self._outside(self.points, exclude)]
        return cKDTree(pts) if len(pts) else None

    def blocked(self, p_ned, radius_m, exclude=None):
        tree = self._tree_for(exclude)
        if tree is None:
            return False
        return bool(tree.query_ball_point(np.asarray(p_ned, float), radius_m, return_length=True) > 0)

    def clearance(self, p_from, p_to, radius_m, step_m=0.05, exclude=None):
        """Free distance along the segment before a sample comes within radius of an obstacle.
        exclude: optional cylinders (center_ned, axis_ned, radius_m, half_length_m) whose points are ignored."""
        a = np.asarray(p_from, float); b = np.asarray(p_to, float); L = float(np.linalg.norm(b - a))
        tree = self._tree_for(exclude)
        if tree is None:
            return math.inf
        if L < 1e-6:
            return 0.0 if tree.query_ball_point(a, radius_m, return_length=True) > 0 else math.inf
        n = max(2, int(math.ceil(L / step_m)) + 1)
        s = np.linspace(0., L, n); samples = a + (b - a)[None, :] * (s / L)[:, None]
        d, _ = tree.query(samples, k=1)
        hit = np.nonzero(d < radius_m)[0]
        return math.inf if len(hit) == 0 else float(s[hit[0]])

    def nearest_point(self, p_ned):
        if self.tree is None:
            return None
        _, i = self.tree.query(np.asarray(p_ned, float), k=1)
        return self.points[i]

    def nearest(self, p_ned):
        if self.tree is None:
            return math.inf
        d, _ = self.tree.query(np.asarray(p_ned, float), k=1)
        return float(d)
