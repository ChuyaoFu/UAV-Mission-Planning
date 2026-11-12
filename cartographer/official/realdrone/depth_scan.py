#!/usr/bin/env python3
"""Depth image -> planar 2-D scan for the maze planner's occupancy grid (numpy only; no ROS).

Input: the FINAL depth (metres along the optical z axis, invalid = 0) of a rectified pinhole camera K = (fx, fy, cx,
cy), i.e. after the net / noise filters. For every image column the nearest valid return whose height above the
camera centre (camera up = -y optical) lies in [h_min, h_max] is kept; its range in the camera's x-z plane is
z / cos(azimuth) (all pixels of a column share one azimuth). Columns are binned to uniform angles at the centre
column's spacing (1 / fx rad), so no bin is empty and edge bins take the nearest of their 2-3 columns.

Scan frame = the camera centre with x forward (optical z), y left (optical -x), z up (optical -y): the LaserScan
convention (beams in the frame's x-y plane, angle counter-clockwise from x). Bin values (REP 117):
  finite   nearest in-band return, metres (<= range_max)
  +inf     the nearest in-band return is beyond range_max (free to at least range_max along that ray)
  NaN      no valid in-band pixel at all in the bin's columns (unknown: texture-less, occluded, or nothing there)
A bin mixing columns takes finite first, then NaN, then +inf (unknown wins over free).
intensities = support: number of in-band pixels within `support_tol` (relative) of the reported range in the bin's
columns, in OUTPUT-image pixels (x px_scale^2 when the scan runs at a lower working resolution); 0 for inf / NaN.
"""
import math

import numpy as np

NAN_CODE = np.float32(1e6)            # bin-merge codes: finite < unknown < free
INF_CODE = np.float32(2e6)


class DepthScan:
    def __init__(self, K, size, h_min=-0.25, h_max=0.35, range_max=5.0, range_min=0.0, support_tol=0.10, px_scale=1.0):
        fx, fy, cx, cy = (float(v) for v in K)
        W, H = (int(v) for v in size)
        if not h_min < h_max:
            raise ValueError(f'depth scan band needs h_min < h_max, got {h_min}, {h_max}')
        self.K, self.size = (fx, fy, cx, cy), (W, H)
        self.h_min, self.h_max, self.range_max, self.range_min = float(h_min), float(h_max), float(range_max), float(range_min)
        self.support_tol, self.px_area = float(support_tol), float(px_scale) ** 2
        # rows: z interval whose height a_v * z falls in [h_min, h_max]; a_v = height per metre of depth (up +)
        a = -(np.arange(H, dtype=np.float64) - cy) / fy
        with np.errstate(divide='ignore', invalid='ignore'):
            q1, q2 = self.h_min / a, self.h_max / a
        lo = np.where(a > 0, q1, q2); hi = np.where(a > 0, q2, q1)
        flat = a == 0
        lo[flat] = 0.0 if self.h_min <= 0 <= self.h_max else np.inf
        hi[flat] = np.inf if self.h_min <= 0 <= self.h_max else -np.inf
        lo = np.maximum(lo, max(1e-3, self.range_min))          # also excludes invalid (0) depth
        rows = np.nonzero(hi >= lo)[0]                           # rows that can hold an in-band return
        self.r0, self.r1 = (int(rows[0]), int(rows[-1]) + 1) if rows.size else (0, 0)
        self.lo = lo[self.r0:self.r1, None].astype(np.float32); self.hi = np.minimum(hi, 1e9)[self.r0:self.r1, None].astype(np.float32)
        # columns: azimuth, z -> range factor, uniform-angle bin
        t = (np.arange(W, dtype=np.float64) - cx) / fx
        self.inv_cos = np.sqrt(1.0 + t * t).astype(np.float32)
        phi = -np.arctan(t)                                      # scan angle: left of the optical axis = +
        self.angle_increment = 1.0 / fx
        self.angle_min = float(phi[-1])
        ib = np.rint((phi - self.angle_min) / self.angle_increment).astype(np.int64)
        self.n = int(ib.max()) + 1
        self.angle_max = self.angle_min + (self.n - 1) * self.angle_increment
        order = np.argsort(ib, kind='stable')                    # = columns right -> left
        self.order = order; self.ib = ib
        self.starts = np.r_[0, np.nonzero(np.diff(ib[order]))[0] + 1]
        assert self.starts.size == self.n, 'empty scan bin'      # column spacing < 1 / fx everywhere

    def meta(self):
        return dict(angle_min=self.angle_min, angle_max=self.angle_max, angle_increment=self.angle_increment, n=self.n,
                    range_min=self.range_min, range_max=self.range_max, h_min=self.h_min, h_max=self.h_max,
                    rows=[self.r0, self.r1], size=list(self.size))

    def __call__(self, depth):
        """depth (H, W) float32 -> (ranges float32[n], intensities float32[n]), bin 0 = angle_min (rightmost)"""
        z = np.asarray(depth, np.float32)
        assert z.shape == (self.size[1], self.size[0]), (z.shape, self.size)
        zb = z[self.r0:self.r1]
        zz = np.where((zb >= self.lo) & (zb <= self.hi), zb, np.float32(np.inf))
        zmin = zz.min(axis=0) if zz.shape[0] else np.full(z.shape[1], np.inf, np.float32)
        r = zmin * self.inv_cos
        code = np.where(np.isfinite(r), np.where(r <= self.range_max, r, INF_CODE), NAN_CODE).astype(np.float32)
        binv = np.minimum.reduceat(code[self.order], self.starts)
        # support of the bin's range, counted per column against the bin value (in z units of that column)
        thr = np.where(binv[self.ib] < NAN_CODE, binv[self.ib] * np.float32(1.0 + self.support_tol), 0) / self.inv_cos
        sup = np.count_nonzero(zz <= thr.astype(np.float32), axis=0) if zz.shape[0] else np.zeros(z.shape[1], np.int64)
        inten = np.add.reduceat(sup[self.order], self.starts).astype(np.float32) * np.float32(self.px_area)
        ranges = np.where(binv < NAN_CODE, binv, np.where(binv == INF_CODE, np.float32(np.inf), np.float32(np.nan)))
        return ranges.astype(np.float32), inten


def scan_points(ranges, angle_min, angle_increment):
    """finite bins -> (N, 2) x forward / y left in the scan frame (tests, tools)"""
    a = angle_min + angle_increment * np.arange(len(ranges))
    ok = np.isfinite(ranges)
    return np.stack([ranges[ok] * np.cos(a[ok]), ranges[ok] * np.sin(a[ok])], 1)


# the scan frame's orientation in the optical camera frame: columns = scan x, y, z in optical coordinates
R_OPTICAL_SCAN = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
Q_OPTICAL_SCAN_XYZW = (0.5, -0.5, 0.5, 0.5)                    # same rotation as a quaternion (x, y, z, w)
assert abs(math.sqrt(sum(c * c for c in Q_OPTICAL_SCAN_XYZW)) - 1.0) < 1e-12
