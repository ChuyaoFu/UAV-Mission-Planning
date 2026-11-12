"""Floor guideline detection on a bird's-eye (inverse perspective) view.

Pure numpy/OpenCV. Inputs: one grayscale left-camera image, its intrinsics,
camera extrinsics (base_link FLU <- optical), PX4 attitude (FRD->NED) and NED
position, and the floor height (NED d). Output: line hypotheses in NED.
The guideline is a long narrow band; texture blobs are short or wide.
"""
import math
import numpy as np
import cv2

F_FLU_FRD = np.diag([1., -1., -1.])


def quat_wxyz_to_mat(w, x, y, z):
    n = math.sqrt(w*w + x*x + y*y + z*z); w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class Detector:
    def __init__(self, x_range=(0.45, 3.0), y_half=1.25, res=0.01, band_kernel_m=0.21,
                 min_len_m=0.35, max_gap_m=0.08, min_response=14):
        self.x0, self.x1 = x_range; self.yh = y_half; self.res = res
        self.rows = int(round((self.x1-self.x0)/res)); self.cols = int(round(2*y_half/res))
        k = max(3, int(round(band_kernel_m/res)) | 1)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.min_len = int(round(min_len_m/res)); self.max_gap = int(round(max_gap_m/res)); self.min_response = min_response
        # grid in the level, heading-aligned local frame: x forward, y right
        r = np.arange(self.rows); c = np.arange(self.cols)
        self.gx = (self.x1 - (r+.5)*res)[:, None] * np.ones((1, self.cols))
        self.gy = (-y_half + (c+.5)*res)[None, :] * np.ones((self.rows, 1))

    def cell_to_local(self, row, col):
        return self.x1 - (row+.5)*self.res, -self.yh + (col+.5)*self.res

    def ipm(self, image, K, R_bc, t_bc, q_wxyz, p_ned, heading, d_floor, depth=None, dK=None):
        fx, fy, cx, cy = K
        R_nb = quat_wxyz_to_mat(*q_wxyz)                 # FRD -> NED
        R_nc = R_nb @ F_FLU_FRD @ R_bc                    # optical -> NED
        cam = np.asarray(p_ned, float) + R_nb @ (F_FLU_FRD @ np.asarray(t_bc, float))
        ch, sh = math.cos(heading), math.sin(heading)
        n = p_ned[0] + self.gx*ch - self.gy*sh
        e = p_ned[1] + self.gx*sh + self.gy*ch
        pts = np.stack([n - cam[0], e - cam[1], np.full_like(n, d_floor - cam[2])], -1)
        pc = pts @ R_nc                                   # NED -> optical (R_nc^T applied row-wise)
        z = pc[..., 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = fx*pc[..., 0]/z + cx; v = fy*pc[..., 1]/z + cy
        h, w = image.shape[:2]
        valid = (z > .05) & (u >= 0) & (u <= w-1) & (v >= 0) & (v <= h-1)
        u = np.where(valid, u, -1).astype(np.float32); v = np.where(valid, v, -1).astype(np.float32)
        top = cv2.remap(image, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if depth is not None:
            # keep only cells whose measured stereo depth agrees with the floor hypothesis
            dfx, dfy, dcx, dcy = dK if dK is not None else K
            du = np.round(dfx*pc[..., 0]/np.where(z > .05, z, 1) + dcx).astype(int)
            dv = np.round(dfy*pc[..., 1]/np.where(z > .05, z, 1) + dcy).astype(int)
            dh, dw = depth.shape[:2]
            inside = valid & (du >= 0) & (du < dw) & (dv >= 0) & (dv < dh)
            meas = np.full(z.shape, np.nan, np.float32)
            meas[inside] = depth[dv[inside], du[inside]]
            agree = np.isfinite(meas) & (meas > 0) & (np.abs(meas - z) < .06 + .08*z)
            # walls are nearer than the floor point behind them: a clearly shorter return means not floor
            nearer = np.isfinite(meas) & (meas > 0) & (meas < z - (.06 + .08*z))
            self.floor_agree = agree
            valid = valid & ~nearer
        return top, valid

    def detect(self, top, valid):
        """Return [(length_m, (x_a, y_a), (x_b, y_b))] in the local level frame (x fwd, y right)."""
        g = top.copy(); g[~valid] = int(np.median(top[valid])) if valid.any() else 0
        th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, self.kernel)
        th[~cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)] = 0
        level = max(self.min_response, float(np.percentile(th[valid], 96)) if valid.any() else 255)
        mask = (th >= level).astype(np.uint8)*255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        segs = cv2.HoughLinesP(mask, 1, math.pi/180, threshold=max(15, self.min_len//2),
                               minLineLength=self.min_len, maxLineGap=self.max_gap)
        out = []
        if segs is None:
            return out, mask
        for c0, r0, c1, r1 in segs.reshape(-1, 4):
            a = self.cell_to_local(r0, c0); b = self.cell_to_local(r1, c1)
            out.append((math.hypot(a[0]-b[0], a[1]-b[1]), a, b))
        return out, mask


def merge(segments, angle_tol_deg=8., offset_tol_m=0.07):
    """Cluster collinear segments -> [dict(dir, point, length, s_min, s_max)] (local frame, dir unsigned)."""
    lines = []
    for length, a, b in sorted(segments, key=lambda s: -s[0]):
        d = np.array([b[0]-a[0], b[1]-a[1]]); d /= max(np.linalg.norm(d), 1e-9)
        if d[0] < 0: d = -d                                 # point forward-ish
        mid = (np.array(a)+np.array(b))/2
        for L in lines:
            if abs(float(L['dir'] @ d)) > math.cos(math.radians(angle_tol_deg)):
                nrm = np.array([-L['dir'][1], L['dir'][0]])
                if abs(float((mid - L['point']) @ nrm)) < offset_tol_m:
                    for p in (a, b):
                        s = float((np.array(p) - L['point']) @ L['dir']); L['s_min'] = min(L['s_min'], s); L['s_max'] = max(L['s_max'], s)
                    L['length'] += length; break
        else:
            sa = float((np.array(a)-mid) @ d); sb = float((np.array(b)-mid) @ d)
            lines.append(dict(dir=d, point=mid, length=length, s_min=min(sa, sb), s_max=max(sa, sb)))
    return sorted(lines, key=lambda L: -L['length'])


def local_to_ned(p_ned, heading, x, y):
    ch, sh = math.cos(heading), math.sin(heading)
    return np.array([p_ned[0] + x*ch - y*sh, p_ned[1] + x*sh + y*ch])
