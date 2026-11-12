#!/usr/bin/env python3
"""Cheap source-side filters against free-space stereo noise (numpy + cv2; no ROS). Applied to the final disparity /
depth after the matcher's own post-processing (and after the net filter). Evaluated in tools/noise_eval.py.

  min_support(disp, max_size, max_diff_px)  connected components in disparity (cv2.filterSpeckles on the FINAL map, i.e.
                                            after the texture gate and the median, unlike SGBM's internal speckle pass):
                                            a component of <= max_size px whose neighbours differ by <= max_diff_px is
                                            dropped.
  TemporalCheck                              a return is kept only if a recent frame saw the same surface: the 3-D point
                                            is moved into the previous camera(s) with the pose and compared with that
                                            frame's depth there (+-tol relative, 3x3 neighbourhood min/max).
"""
import numpy as np
import cv2


def min_support(disp, max_size, max_diff_px):
    """disp float32 px (<= 0 invalid) -> keep mask (bool)"""
    d16 = np.round(np.clip(disp, 0, 2000) * 16.0).astype(np.int16)
    cv2.filterSpeckles(d16, 0, int(max_size), int(round(max_diff_px * 16)))
    return d16 > 0


class TemporalCheck:
    """depth (m, 0 invalid) + camera pose (R_nc optical -> world, cam position) per frame -> keep mask.
    A pixel of frame t is kept if it agrees with ANY of the last `nprev` frames within max_age_s; frames without a
    usable history keep everything (start-up, pose gap)."""

    def __init__(self, K, tol=0.05, nprev=1, win=3, max_age_s=0.5, min_depth=0.2, require=False):
        self.K = tuple(float(x) for x in K)
        self.tol, self.nprev, self.win, self.max_age, self.min_depth = float(tol), int(nprev), int(win), float(max_age_s), float(min_depth)
        self.require = bool(require)                    # True: a return needs a matching previous return
        self.hist = []                                  # (stamp, depth, dmin, dmax, R_nc, cam)
        self._g = {}

    def _grid(self, shape):
        g = self._g.get(shape)
        if g is None:
            v, u = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
            fx, fy, cx, cy = self.K
            g = self._g[shape] = ((u - cx) / fx, (v - cy) / fy)
        return g

    def __call__(self, depth, R_nc, cam, stamp):
        z = np.asarray(depth, np.float32)
        H, W = z.shape
        ok = z > self.min_depth
        keep = np.ones(z.shape, bool)
        prev = [h for h in self.hist if 0 < stamp - h[0] <= self.max_age][-self.nprev:]
        if prev:
            agree = np.zeros(z.shape, bool)
            xr, yr = self._grid(z.shape)
            Pc = np.stack([xr * z, yr * z, z], -1)                         # camera t
            Pw = Pc @ np.asarray(R_nc, np.float32).T + np.asarray(cam, np.float32)
            fx, fy, cx, cy = self.K
            for (_, zp, dmin, dmax, Rp, cp) in prev:
                Q = (Pw - np.asarray(cp, np.float32)) @ np.asarray(Rp, np.float32)   # into camera t-k
                zq = Q[..., 2]
                with np.errstate(divide='ignore', invalid='ignore'):
                    mu = (fx * Q[..., 0] / zq + cx).astype(np.float32); mv = (fy * Q[..., 1] / zq + cy).astype(np.float32)
                lo = cv2.remap(dmin, mu, mv, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                hi = cv2.remap(dmax, mu, mv, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                inside = (zq > self.min_depth) & (mu >= 0) & (mu <= W - 1) & (mv >= 0) & (mv <= H - 1)
                agree |= inside & (hi > 0) & (zq >= lo * (1 - self.tol)) & (zq <= hi * (1 + self.tol))
                # out of the previous view: no evidence -> keep; no previous return there: keep unless `require`
                agree |= ~inside if self.require else (~inside | (hi <= 0))
            keep = ~ok | agree
        k = np.ones((self.win, self.win), np.uint8)
        zz = np.where(ok, z, 0).astype(np.float32)
        dmax = cv2.dilate(zz, k)
        dmin = cv2.erode(np.where(ok, z, 1e4).astype(np.float32), k); dmin[dmin >= 1e4] = 0
        self.hist.append((float(stamp), zz, dmin, dmax, np.asarray(R_nc, np.float32), np.asarray(cam, np.float32)))
        self.hist = self.hist[-max(self.nprev, 1) - 1:]
        return keep


# ----------------------------------------------------------------------------------------------- ROS parameters
def coerce(value, like):
    """parameter value -> the type of `like`: an integer-looking 1 for a double, '150' or 150.0 for an int, 'true' / 1
    for a bool, anything for a str."""
    if isinstance(like, bool):
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'on')
        return bool(value)
    if isinstance(like, int):
        return int(round(float(value)))
    if isinstance(like, float):
        return float(value)
    if isinstance(like, str):
        return '' if value is None else str(value)
    return value


def declare_typed(node, name, default):
    """Declare a ROS 2 parameter that accepts an override of ANY type, then coerce it to the default's type. A plain
    declare_parameter(name, 1.0) given `-p name:=1` (an INTEGER) raises InvalidParameterTypeException and the node
    dies (official-compat 2026-11-08: no stereo, no depth)."""
    from rcl_interfaces.msg import ParameterDescriptor
    return coerce(node.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True)).value, default)
