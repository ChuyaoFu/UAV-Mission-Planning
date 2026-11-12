#!/usr/bin/env python3
"""Model-based cage-net phantom filter for stereo depth (numpy + cv2; no ROS), after Barrois, Konrad, Woehler & Gross,
"Resolving stereo matching errors due to repetitive structures using model information", Pattern Recognition Letters
31 (2010) 1683-1692.  [paper: outputs/claude-official/refs/, not redistributed]

Paper -> here
  2.2   plane model of the repetitive structure.           ADAPTED: the planes are KNOWN - the prior's fence box
        (start frame: forward/left min/max, up_max; floor at 0) placed in the camera with the VIO pose. In disparity
        space every face is affine, d = a*u + b*v + c (paper Eqs. 4-6); inside the box the net seen at a pixel is the
        face with the largest positive disparity, unless the floor plane is nearer (no net there).
  2.2.2 plane offset c_s from the model-free stereo points with the "fair" M-estimator in DISPARITY space (residuals
        are Gaussian there, Eq. 7).                       ADOPTED per face: IRLS on the SGBM points within +-0.3 m (the
        prior's tolerance) of the predicted face; the normal comes from prior + pose instead of the FFT wavelengths.
  2.2.1 FFT detection of repetitive structures: horizontal lines of fixed length, half overlap, the strongest non-DC
        peak if its significance (mean / std of the spectrum) exceeds a threshold -> wavelength.
                                                          ADOPTED on the FULL-RES (640) rectified left image, 64-px
        Hann-windowed segments with half overlap, rows pooled 2x, spectra of 3 neighbouring pooled rows averaged; only
        the segments that hold a candidate pixel are transformed (cost). No vertical pass: only horizontal periodicity
        aliases along the epipolar line.
  2.4   decision feedback E_t = E_SSD + lambda_e * E_d inside the repetitive regions, then the matcher's own constraints.
                                                          ADAPTED as a post-filter (SGBM unchanged): in a repetitive
        region a return NEARER than the net model is dropped when the right image matches the model disparity about as
        well as SGBM's choice (SAD_model <= ratio * SAD_d + floor: the model term decides a near-tie, like lambda_e
        between 0.001 and 0.01) and its offset from the model is ~k * wavelength (k = 1..3, an alias of the mesh). A real
        object in front of the net matches clearly better at its own disparity and stays. Returns on the model are
        kept (that is the net); returns behind it are outside the cage and left alone.
  3     limit: the wavelength must exceed the model's disparity uncertainty.
                                                          ADOPTED as a per-pixel guard: no decision where the wavelength
        (work px) < max(min_wavelength, 2 x model disparity uncertainty) - the refined fit residual, or the prior's
        +-0.3 m if a face could not be refined; detection runs at 640 so the fine mesh is not aliased at 320.

Use:  nm = NetModel(fence); bad, info = nm(left640, disp, right, K_disp, baseline, R_nc, cam_ned)
      disp/right at the depth's working size (320x200 node, 640x400 compat), K_disp = (fx, fy, cx, cy) there,
      R_nc/cam_ned = optical camera -> start-frame NED (floor at d = 0) at the image stamp.
"""
import numpy as np
import cv2

FACES = ('forward_min', 'forward_max', 'left_min', 'left_max', 'up_max')
DEFAULTS = dict(fence_tol_m=0.3, max_shift_m=0.4, fit_min_px=150, fair_c=1.0, irls_iters=4, seg=64, min_sig=3.0,
                tol_in_px=1.5, alias_tol=0.25, alias_k=(1, 2, 3), cost_ratio=1.5, cost_floor=3.0, cost_win=(7, 5),
                min_wavelength_px=2.0, require_alias=True, floor_margin_m=0.05, max_sigma_px=3.0)


def face_planes_ned(fence):
    """fence box faces in start-frame NED as (normal n_w, offset h_w) with n_w . X = h_w (+ the floor, last)"""
    F = fence
    return [(np.array([1.0, 0, 0]), float(F['forward_min'])), (np.array([1.0, 0, 0]), float(F['forward_max'])),
            (np.array([0, 1.0, 0]), -float(F['left_min'])), (np.array([0, 1.0, 0]), -float(F['left_max'])),
            (np.array([0, 0, 1.0]), -float(F.get('up_max', 1e3)))]


class NetModel:
    def __init__(self, fence, mask_kw=None, **kw):
        o = dict(DEFAULTS); o.update(kw); self.o = o
        self.fence = {k: float(fence[k]) for k in FACES if k in fence}
        self.faces = face_planes_ned(self.fence)
        self.floor = (np.array([0, 0, 1.0]), -float(o['floor_margin_m']))     # d = -margin (just above the floor)
        self._grid = {}
        from net_mask import NetMask
        self.mask = NetMask(**(mask_kw or {}))

    def params(self):
        return dict(self.o, fence=self.fence, mask=self.mask.params())

    def inside(self, cam_ned, margin=0.05):
        F = self.fence; n, e, d = cam_ned
        return (F['forward_min'] + margin < n < F['forward_max'] - margin and
                -F['left_max'] + margin < e < -F['left_min'] - margin and -d < F.get('up_max', 1e3) - margin)

    def _uv(self, shape):
        g = self._grid.get(shape)
        if g is None:
            v, u = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
            g = self._grid[shape] = (u, v)
        return g

    @staticmethod
    def disparity_plane(n_w, h_w, R_nc, cam_ned, K, fb):
        """plane n_w . X = h_w (NED) -> (a, b, c, h_c) with d(u, v) = a u + b v + c = fb (n_c . ray) / h_c"""
        n_c = np.asarray(R_nc).T @ n_w
        h_c = h_w - float(n_w @ np.asarray(cam_ned))
        fx, fy, cx, cy = K
        a = fb * n_c[0] / (fx * h_c); b = fb * n_c[1] / (fy * h_c)
        c = fb * (n_c[2] - n_c[0] * cx / fx - n_c[1] * cy / fy) / h_c
        return a, b, c, h_c

    def model_disparity(self, shape, K, fb, R_nc, cam_ned, h_scale=None):
        """-> (d_net [shape], face id [shape] (-1 = floor / none), planes [(a, b, c, h_c)])"""
        u, v = self._uv(shape)
        planes, best, fid = [], np.zeros(shape, np.float32), np.full(shape, -1, np.int8)
        for i, (n_w, h_w) in enumerate(self.faces):
            a, b, c, h_c = self.disparity_plane(n_w, h_w, R_nc, cam_ned, K, fb)
            if h_scale is not None:                       # refined distance: d scales with 1 / h_c
                s = h_scale[i]; a, b, c, h_c = a * s, b * s, c * s, h_c / s
            planes.append((a, b, c, h_c))
            di = (a * u + b * v + c).astype(np.float32)
            m = di > best
            best[m] = di[m]; fid[m] = i
        a, b, c, _ = self.disparity_plane(self.floor[0], self.floor[1], R_nc, cam_ned, K, fb)
        floor = a * u + b * v + c
        m = floor >= best                                  # the floor is hit first: no net at this pixel
        best[m] = 0.0; fid[m] = -1
        return best, fid, planes

    def refine(self, disp, fid, planes, stride=2):
        """paper Eq. 7, per face: robust ("fair") fit of the face distance in disparity space to the SGBM returns within
        +-fence_tol of the predicted face. -> (scale per face (1 / h multiplier), residual sigma per face [px] or None)"""
        o = self.o; scale = [1.0] * len(planes); sig = [None] * len(planes)
        d = disp[::stride, ::stride]; f = fid[::stride, ::stride]
        u, v = self._uv(disp.shape); u = u[::stride, ::stride]; v = v[::stride, ::stride]
        for i, (a, b, c, h_c) in enumerate(planes):
            if h_c <= 0:
                continue
            k = (a * u + b * v + c) * h_c                     # fb (n_c . ray): d = k / h
            lo = k / (h_c + o['fence_tol_m']); hi = k / max(h_c - o['fence_tol_m'], 0.05)
            m = (f == i) & (d > 0) & (d >= lo) & (d <= hi)
            if int(m.sum()) < o['fit_min_px']:
                continue
            kk, dd = k[m].astype(np.float64), d[m].astype(np.float64)
            w = 1.0 / h_c
            for _ in range(o['irls_iters']):
                r = dd - kk * w
                om = 1.0 / (1.0 + np.abs(r) / o['fair_c'])
                w = float((om * kk * dd).sum() / max((om * kk * kk).sum(), 1e-9))
            h_new = 1.0 / max(w, 1e-6)
            if abs(h_new - h_c) > o['max_shift_m']:
                continue
            scale[i] = h_c / h_new
            r = dd - kk * w
            sig[i] = float(1.4826 * np.median(np.abs(r - np.median(r))))
        return scale, sig

    def predict(self, disp, K, baseline, R_nc, cam_ned):
        """-> (model net disparity at disp's size (0 = no net / camera outside the fence), info): fence projection +
        per-face robust offset refinement (paper Eqs. 5-7)."""
        info = dict(model=False)
        if not self.inside(cam_ned):
            return None, info
        fb = K[0] * baseline
        d_net, fid, planes = self.model_disparity(disp.shape, K, fb, R_nc, cam_ned)
        scale, sig = self.refine(disp, fid, planes)
        shift = [None if s == 1.0 else round(p[3] * (1.0 / s - 1.0), 3) for p, s in zip(planes, scale)]   # metres, + = farther
        if any(s != 1.0 for s in scale):
            d_net, fid, planes = self.model_disparity(disp.shape, K, fb, R_nc, cam_ned, h_scale=scale)
        # paper's limit: no model where its disparity uncertainty is not small against the mesh period; we do not know
        # the period here, so drop faces whose fit residual is large (> max_sigma_px) and keep the rest
        for i, sg in enumerate(sig):
            if sg is not None and sg > self.o['max_sigma_px']:
                d_net[fid == i] = 0.0
        info.update(model=True, shift_m=shift,
                    sigma_px=[None if x is None else round(x, 2) for x in sig])
        return d_net, info

    def __call__(self, left640, disp, right, K, baseline, R_nc, cam_ned, left_work=None):
        """-> (bad mask at disp's size, info). The decision is net_mask.NetMask's (periodicity score + local far
        support + cost check) with the model: candidates must be nearer than the fitted fence, and a tie at the model
        disparity also counts as ambiguous (decision feedback)."""
        H, W = disp.shape
        d_net, info = self.predict(disp, K, baseline, R_nc, cam_ned)
        if d_net is None:
            return np.zeros(disp.shape, bool), info
        lw = left_work if left_work is not None else cv2.resize(left640, (W, H), interpolation=cv2.INTER_AREA)
        tol = self.o['tol_in_px'] * (W / 320.0)
        bad = self.mask(lw, disp, right, model_disp=d_net, model_tol=self.o['tol_in_px'])
        info['dropped'] = int(bad.sum()); info['tol_px'] = round(tol, 2)
        return bad, info

    def wavelengths(self, left640, pts640):
        """paper 2.2.1 at full res: for candidate pixels (N, 2) [x, y] in 640 coords -> (wavelength px @640, significance)"""
        o = self.o; N = o['seg']; half = N // 2
        img = left640.astype(np.float32)
        H, W = img.shape
        pooled = cv2.resize(img, (W, H // 2), interpolation=cv2.INTER_AREA)
        Hp = pooled.shape[0]
        xs, ys = pts640[:, 0], pts640[:, 1]
        j = np.clip(np.round((xs - half) / half).astype(int), 0, (W - N) // half)
        r = np.clip(ys // 2, 1, Hp - 2).astype(int)
        key = r * 10000 + j
        uk, inv = np.unique(key, return_inverse=True)
        rr, jj = uk // 10000, uk % 10000
        idx = (jj * half)[:, None] + np.arange(N)[None, :]
        segs = np.stack([pooled[(rr + dr)[:, None], idx] for dr in (-1, 0, 1)], 1)       # (M, 3, N)
        segs = segs - segs.mean(-1, keepdims=True)
        A = np.abs(np.fft.rfft(segs * np.hanning(N).astype(np.float32), axis=-1)).mean(1)   # (M, N/2+1)
        kmin = max(2, int(np.floor(N / 40.0))); kmax = min(N // 2 - 1, int(np.ceil(N / 3.0)))
        band = A[:, 2:]
        mu, sd = band.mean(1), band.std(1) + 1e-6
        Ak = A[:, kmin:kmax + 1]
        kk = Ak.argmax(1) + kmin
        sig = (A[np.arange(len(kk)), kk] - mu) / sd
        a0, a1, a2 = A[np.arange(len(kk)), kk - 1], A[np.arange(len(kk)), kk], A[np.arange(len(kk)), np.minimum(kk + 1, N // 2)]
        den = a0 - 2 * a1 + a2
        off = np.where(np.abs(den) > 1e-9, 0.5 * (a0 - a2) / den, 0.0)
        lam = N / np.clip(kk + np.clip(off, -0.5, 0.5), 1.0, None)
        return lam[inv], sig[inv]

    def model_tie(self, left, right, disp, d_net, ys, xs):
        """decision feedback as a post-filter: True where the right image matches the MODEL disparity about as well as
        SGBM's (SAD over cost_win): the model term decides the near-tie."""
        o = self.o; wh = o['cost_win'][1] // 2
        r0 = max(0, int(ys.min()) - wh); r1 = min(disp.shape[0], int(ys.max()) + 1 + wh)
        Ww = disp.shape[1]
        xg = np.arange(Ww, dtype=np.float32)[None, :]
        yg = np.repeat(np.arange(r0, r1, dtype=np.float32)[:, None], Ww, 1)
        L = left[r0:r1].astype(np.float32)
        cur = cv2.remap(right, xg - disp[r0:r1], yg, cv2.INTER_LINEAR).astype(np.float32)
        mod = cv2.remap(right, xg - d_net[r0:r1], yg, cv2.INTER_LINEAR).astype(np.float32)
        sc = cv2.blur(np.abs(L - cur), o['cost_win']); sm = cv2.blur(np.abs(L - mod), o['cost_win'])
        yy = ys - r0
        return sm[yy, xs] <= o['cost_ratio'] * sc[yy, xs] + o['cost_floor']


# ----------------------------------------------------------------------------------------------- pose plumbing
F_FLU_FRD = np.diag([1.0, -1.0, -1.0])


def quat_xyzw_to_R(x, y, z, w):
    n = (x * x + y * y + z * z + w * w) ** 0.5; x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_fence(path):
    """a course prior JSON ({'frame': 'start', 'fence': {...}}) or a bare fence dict -> fence dict"""
    import json
    d = json.loads(open(path).read())
    f = d.get('fence', d)
    assert all(k in f for k in FACES[:4]), f'no fence in {path}'
    return f


class OdomPose:
    """odometry (base_link in an odom frame with z up: nav_msgs/Odometry from /visual_slam/tracking/odometry) ->
    camera pose in the prior's START frame NED (origin on the floor under the launch point, n = forward at start).
    anchor 'origin': the odom frame already is the start frame (compat: ev_bridge odom starts at base_link on the pad);
    anchor 'first' : the first odometry pose received is the start pose (the node is launched with the drone on the
                     takeoff spot, as the mission's home is taken there; position + yaw, level).
    base_height_m: base_link above the floor at the start (only the roof / floor planes depend on it)."""

    def __init__(self, anchor='first', base_height_m=0.05, max_dt=0.1, keep=120):
        assert anchor in ('origin', 'first'), anchor
        self.anchor, self.base_h, self.max_dt, self.keep = anchor, float(base_height_m), float(max_dt), int(keep)
        self.hist = []          # (t, p (3,), R (3,3)) base_link in odom
        self.p0 = None; self.Rz0 = None

    def add(self, t, p, q_xyzw):
        p = np.asarray(p, float); R = quat_xyzw_to_R(*q_xyzw)
        if self.p0 is None:
            if self.anchor == 'origin':
                self.p0, self.Rz0 = np.zeros(3), np.eye(3)
            else:
                yaw = np.arctan2(R[1, 0], R[0, 0]); c, s = np.cos(-yaw), np.sin(-yaw)
                self.p0, self.Rz0 = p.copy(), np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        self.hist.append((float(t), p, R))
        if len(self.hist) > self.keep:
            del self.hist[0]

    def camera(self, t, R_bc, t_bc):
        """-> (R_nc, cam_ned) at stamp t (nearest odometry within max_dt) or None. R_bc, t_bc: base_link <- optical."""
        if not self.hist:
            return None
        k = min(range(len(self.hist)), key=lambda i: abs(self.hist[i][0] - t))
        tk, p, R = self.hist[k]
        if abs(tk - t) > self.max_dt:
            return None
        R_sb = self.Rz0 @ R; p_s = self.Rz0 @ (p - self.p0) + np.array([0, 0, self.base_h])
        R_nc = F_FLU_FRD @ R_sb @ np.asarray(R_bc, float)
        cam = F_FLU_FRD @ (p_s + R_sb @ np.asarray(t_bc, float))
        return R_nc, cam
