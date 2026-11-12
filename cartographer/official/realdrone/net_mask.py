#!/usr/bin/env python3
"""Repetitive-texture (cage net) phantom mask for stereo depth (numpy + cv2 only; no ROS).

Why: a net is a grid of thin cords. Along the epipolar line the left image repeats with the cord spacing p (pixels), so
the matcher can lock onto the neighbouring cell: disparity d_true + k*p, a point 0.4-0.6x the net's range, floating in
free space. These aliases pass the left-right check, repeat every frame and survive SGBM's uniqueness ratio.

Rule (both must hold, evaluated at an internal width of ~320 px; inputs at 640 are pooled 2x first):
  1. periodic texture: the horizontal-gradient autocorrelation r_t of the rectified left image (box sums over `win`
     31x5 px, normalised to [-1, 1]) has a PEAK at some lag p in [3, 24] px: r_p minus the lowest r at any shorter lag
     (floored at 0) > `score_min`;
  2. the pixel is much NEARER than its surroundings: disparity - (smallest valid disparity within `far_win`, 21x41 px)
     > max(`d_margin`, `pfrac` * p). An alias sits at least one period in front of the correctly matched net (or the
     hall seen through it); a wall/board is as near as its own neighbourhood, so it keeps its depth.
Masked pixels become invalid (depth 0). Edges of boards in front of the net stay: their own texture is not periodic.

Use: m = NetMask(); bad = m(left_rect_work_u8, disparity_work_px)      # same size; disparity 0 = invalid
     depth[NetMask.upsample(bad, depth.shape)] = 0
     or NetMask().apply_depth(left_u8, depth_m, fx, baseline_m) for a depth image aligned to a rectified left (compat).
Timing (320x200): ~2-3 ms on a fast x86 core; see realdrone-REPORT.md section 5.4.
"""
import numpy as np
import cv2

DEFAULTS = dict(score_min=0.25, d_margin=6.0, pfrac=0.5, lag_min=3, lag_max=24, win=(31, 5), far_win=(21, 41),
                energy_min=36.0, score_dilate=None, norm='geo', row_step=2, score_erode=(9, 5),
                lag_thin=False, cost_ratio=1.5, cost_floor=3.0, cost_win=(7, 5))


class NetMask:
    def __init__(self, score_min=DEFAULTS['score_min'], d_margin=DEFAULTS['d_margin'], pfrac=DEFAULTS['pfrac'],
                 lag_min=DEFAULTS['lag_min'], lag_max=DEFAULTS['lag_max'], win=DEFAULTS['win'],
                 far_win=DEFAULTS['far_win'], energy_min=DEFAULTS['energy_min'], score_dilate=DEFAULTS['score_dilate'],
                 norm=DEFAULTS['norm'], row_step=DEFAULTS['row_step'], score_erode=DEFAULTS['score_erode'],
                 lag_thin=DEFAULTS['lag_thin'], cost_ratio=DEFAULTS['cost_ratio'], cost_floor=DEFAULTS['cost_floor'],
                 cost_win=DEFAULTS['cost_win']):
        self.score_min, self.d_margin, self.pfrac = float(score_min), float(d_margin), float(pfrac)
        self.lags = list(range(int(lag_min), int(lag_max) + 1))
        if lag_thin:                     # long periods: every 2nd lag (the peak there is several px wide at 320)
            self.lags = [t for t in self.lags if t <= 12 or t % 2 == 0]
        self.lag_thin = bool(lag_thin)
        self.cost_ratio, self.cost_floor = float(cost_ratio), float(cost_floor)
        self.cost_win = (int(cost_win[0]), int(cost_win[1]))
        self.win = (int(win[0]), int(win[1]))
        self.kfar = np.ones((int(far_win[1]), int(far_win[0])), np.uint8)
        self.energy_min = float(energy_min)
        self.kdil = None if not score_dilate else np.ones((int(score_dilate[1]), int(score_dilate[0])), np.uint8)
        assert norm in ('geo', 'max'), norm
        self.norm = norm
        self.row_step = max(1, int(row_step))
        self.score_erode = None if not score_erode else (int(score_erode[0]), int(score_erode[1]))
        self._buf = {}

    def params(self):
        return dict(score_min=self.score_min, d_margin=self.d_margin, pfrac=self.pfrac, lag_min=self.lags[0],
                    lag_max=self.lags[-1], win=list(self.win), far_win=[self.kfar.shape[1], self.kfar.shape[0]],
                    energy_min=self.energy_min,
                    score_dilate=None if self.kdil is None else [self.kdil.shape[1], self.kdil.shape[0]], norm=self.norm,
                    row_step=self.row_step, score_erode=None if self.score_erode is None else list(self.score_erode),
                    lag_thin=self.lag_thin, cost_ratio=self.cost_ratio, cost_floor=self.cost_floor,
                    cost_win=list(self.cost_win))

    def _bufs(self, shape):
        b = self._buf.get(shape)
        if b is None:
            b = self._buf[shape] = tuple(np.zeros(shape, np.float32) for _ in range(6)) + (np.zeros(shape, bool),)
        return b

    def periodicity(self, img):
        """-> (score, period px) per pixel of `img`: best normalised gradient autocorrelation peak over the lags.
        r_t(x) = sum_W g(x'-h) g(x'-h+t) / sqrt(E(x-h) E(x-h+t)), h = t//2, E = sum_W g^2, g = horizontal Sobel
        (Cauchy-Schwarz: |r| <= 1; normalising by E(x) alone over-scored pixels next to a strong edge because the
        lagged product reaches t/2 px past the energy window). norm='max' divides by max_{|s|<=lag_max/2} E(x+s).
        Score at lag t = r_t - max(0, min_{1<=s<t} r_s): a real period dips before it peaks; a slanted edge or an
        obliquely viewed board (horizontally stretched texture) only decays and scored high at lags 3-7 without it.
        Computed on rows pooled by `row_step` (net cords are vertical; halves the cost), flat contiguous products."""
        H0, W = img.shape
        rs = self.row_step
        small = cv2.resize(img, (W, H0 // rs), interpolation=cv2.INTER_AREA) if rs > 1 else img
        g = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
        H = g.shape[0]
        win = (self.win[0], max(1, int(round(self.win[1] / rs))) | 1)
        e = cv2.blur(g * g, win)
        prod, best, per, den, c, tmp, upd = self._bufs(g.shape)
        best.fill(0.0); per.fill(0.0)
        if self.norm == 'max':
            norm = cv2.dilate(e, np.ones((1, 2 * (self.lags[-1] // 2) + 1), np.uint8))
            np.maximum(norm, self.energy_min, out=norm)
        else:
            sq = np.sqrt(np.maximum(e, self.energy_min)).ravel()
        run_min = np.full(g.shape, np.inf, np.float32)
        gf, pf, df, N = g.ravel(), prod.ravel(), den.ravel(), g.size
        for t in sorted(set(range(1, self.lags[0])) | set(self.lags)):
            if t >= W:
                break
            h = t // 2
            np.multiply(gf[:N - t], gf[t:], out=pf[h:N - t + h])            # g(x-h) * g(x-h+t), centred at x
            prod[:, :h] = 0.0; prod[:, W - t + h:] = 0.0                      # pairs that wrap across rows
            cv2.blur(prod, win, dst=c)
            if self.norm == 'geo':
                np.multiply(sq[:N - t], sq[t:], out=df[h:N - t + h])
                den[:, :h] = np.inf; den[:, W - t + h:] = np.inf
                np.divide(c, den, out=c)
            else:
                np.divide(c, norm, out=c)
            if t >= self.lags[0]:   # (t in self.lags)
                np.maximum(run_min, 0.0, out=tmp); np.subtract(c, tmp, out=tmp)   # score_t (dip over s < t)
                np.minimum(run_min, c, out=run_min)
                np.add(best, np.float32(0.05), out=c)          # harmonics (2p, 3p) tie with p: keep the fundamental
                np.greater(tmp, c, out=upd)
                np.maximum(best, tmp, out=best)
                # lags rise, so argmax = max(t * upd); uint8 view: numpy's bool->float cast is ~6x slower on x86
                np.multiply(upd.view(np.uint8), np.float32(t), out=c); np.maximum(per, c, out=per)
            else:
                np.minimum(run_min, c, out=run_min)
        if self.score_erode is not None:     # periodic across a neighbourhood, not just from one side of an edge
            ew, eh = self.score_erode
            best = cv2.erode(best, np.ones((max(1, int(round(eh / rs))) | 1, ew), np.uint8))
        if rs > 1:
            best = cv2.resize(best, (W, H0), interpolation=cv2.INTER_NEAREST)
            per = cv2.resize(per, (W, H0), interpolation=cv2.INTER_NEAREST)
        return best, per

    def __call__(self, img, disp, right=None, model_disp=None, model_tol=1.5):
        """img: rectified left uint8; disp: float32 disparity in px at img's size (<= 0 invalid); right: the rectified
        right uint8 (optional, enables the cost check). -> bool mask (True = repetitive-texture phantom).
        model_disp (net_model.py): the fence model's net disparity at img's size (0 = no net there). Then only returns
        more than `model_tol` px (at 320) NEARER than the net are candidates, and the decision feedback is a tie at the
        model disparity (Barrois et al. 2010, E_t = E_SSD + lambda_e E_d) instead of one period farther.
        Inputs wider than 400 px are pooled by an integer factor to ~320 first."""
        H, W = disp.shape
        k = max(1, int(round(W / 320.0)))
        if k > 1:
            img = cv2.resize(img, (W // k, H // k), interpolation=cv2.INTER_AREA)
            if right is not None:
                right = cv2.resize(right, (W // k, H // k), interpolation=cv2.INTER_AREA)
            d = disp[:H // k * k:k, :W // k * k:k] * (1.0 / k)
            md = None if model_disp is None else model_disp[:H // k * k:k, :W // k * k:k] * (1.0 / k)
        else:
            d = disp; md = model_disp
        bad = np.zeros(d.shape, bool)
        ok = d > 0
        far = cv2.erode(np.where(ok, d, np.float32(1e4)).astype(np.float32), self.kfar)
        D = d - far
        cand = ok & (D > self.d_margin)                 # necessary condition: nearer than the far support
        if md is not None:
            cand &= (md > 0) & (d - md > model_tol)     # and nearer than the fence model
        if cand.any():
            # periodicity / cost only on the bounding box of the candidates plus the filters' reach (same result as
            # the full frame inside it; a frame without near-in-front-of-far pixels costs ~nothing)
            rs = self.row_step; Hh, Ww = d.shape
            mr = rs * (self.win[1] // 2 + (self.score_erode[1] // 2 if self.score_erode else 0) + 2)
            mc = self.win[0] // 2 + self.lags[-1] // 2 + (self.score_erode[0] // 2 if self.score_erode else 0) + 2
            if self.kdil is not None:
                mr += self.kdil.shape[0] // 2; mc += self.kdil.shape[1] // 2
            rows = np.flatnonzero(cand.any(1)); cols = np.flatnonzero(cand.any(0))
            r0 = max(0, rows[0] - mr); r0 -= r0 % rs; r1 = min(Hh, rows[-1] + 1 + mr)
            r1 = min(Hh, r0 + max(rs, (r1 - r0 + rs - 1) // rs * rs))
            c0 = max(0, cols[0] - mc); c1 = min(Ww, cols[-1] + 1 + mc)
            score, per = self.periodicity(np.ascontiguousarray(img[r0:r1, c0:c1]))
            if self.kdil is not None:
                score = cv2.dilate(score, self.kdil)
            sub = cand[r0:r1, c0:c1] & (score > self.score_min) & (D[r0:r1, c0:c1] > np.maximum(self.d_margin, self.pfrac * per))
            if right is not None and self.cost_ratio > 0 and sub.any():
                sub &= self.ambiguous(img, right, d, per, (r0, r1, c0, c1), md)
            bad[r0:r1, c0:c1] = sub
        if k > 1:
            bad = self.upsample(bad, (H, W))
        return bad

    def ambiguous(self, img, right, d, per, box=None, md=None):
        """Cost check: True where the right image matches about as well one period FARTHER (d - p) as at the chosen d,
        i.e. the match could be the neighbouring net cell. A near board edge next to the net fails it (its own texture
        does not repeat at p), so it keeps its depth. SAD over `cost_win`, alt <= cost_ratio * cur + noise floor.
        box = (r0, r1, c0, c1): evaluate only there (per has the box's shape)."""
        r0, r1, c0, c1 = box if box is not None else (0, d.shape[0], 0, d.shape[1])
        db = d[r0:r1, c0:c1]
        xs = np.arange(c0, c1, dtype=np.float32)[None, :]
        ys = np.repeat(np.arange(r0, r1, dtype=np.float32)[:, None], c1 - c0, 1)
        Lf = img[r0:r1, c0:c1].astype(np.float32)
        cur = cv2.remap(right, xs - db, ys, cv2.INTER_LINEAR).astype(np.float32)
        alt = cv2.remap(right, xs - (db - per), ys, cv2.INTER_LINEAR).astype(np.float32)
        sc = cv2.blur(np.abs(Lf - cur), self.cost_win); sa = cv2.blur(np.abs(Lf - alt), self.cost_win)
        if md is not None:
            # model mode (Barrois et al. 2010 decision feedback): ONLY a tie at the fence model's disparity counts. A real
            # periodic object in front of the net (a calibration grid, a striped shirt) also ties at d - p, but its
            # patch does not reappear at the net's disparity, so it keeps its depth.
            mb = md[r0:r1, c0:c1]
            mod = cv2.remap(right, xs - mb, ys, cv2.INTER_LINEAR).astype(np.float32)
            sm = cv2.blur(np.abs(Lf - mod), self.cost_win)
            return (mb > 0) & (sm <= self.cost_ratio * sc + self.cost_floor)
        return (db - per > 0) & (sa <= self.cost_ratio * sc + self.cost_floor)

    @staticmethod
    def upsample(mask, shape):
        H, W = shape
        if mask.shape == (H, W):
            return mask
        return cv2.resize(mask.view(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

    def apply_depth(self, left, depth, fx, baseline, right=None):
        """depth (m, aligned to the rectified left, invalid <= 0 / NaN) -> (depth with phantoms set to 0, mask);
        pass the rectified right image to enable the cost check."""
        z = np.nan_to_num(np.asarray(depth, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        disp = np.where(z > 0, np.float32(fx * baseline) / np.maximum(z, 1e-6), 0).astype(np.float32)
        bad = self(left, disp, right)
        out = z.copy(); out[bad] = 0.0
        return out, bad
