#!/usr/bin/env python3
"""Calibration sanity on real record_manual.py flights (offline; numpy + cv2).
 1. rectification row error: SIFT matches between the rectified left / right (node's RectifyModel at 640 and at the
    full 1024 width); vertical offset y_r - y_l per match (median, MAD, by image region).
 2. plane fits on the node's OFF depth (real_eval.py --save-npz, half320 working resolution): RANSAC in disparity space
    (d = a u + b v + c is exact for a plane), residual RMS in px and metres; the floor plane (world normal within 10 deg
    of up) gives the camera height from stereo vs from VIO; board planes seen from several distances give the world
    offset vs distance slope (a depth scale error k shows up as slope (k - 1) / k).
Usage: python3 calib_check.py FLIGHT_DIR NPZ TF_JSON OUT_JSON"""
import csv, json, sys
from pathlib import Path

import numpy as np
import cv2

HERE = Path(__file__).resolve().parent; RD = HERE.parent
sys.path.insert(0, str(RD)); sys.path.insert(0, str(HERE))
from stereo_depth_node import RectifyModel  # noqa: E402
from real_eval import tf_lookup, load_flight  # noqa: E402


def row_error(fdir, rows, ci, R_lr, t_lr, width, step):
    L, R = ci['left'], ci['right']
    K_l = np.array(L['K']).reshape(3, 3); K_r = np.array(R['K']).reshape(3, 3)
    m = RectifyModel(K_l, np.zeros(5), K_r, np.zeros(5), (1024, 640), R_lr, t_lr, width=width)
    sift = cv2.SIFT_create(nfeatures=1500)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    dys, xs, ys = [], [], []
    for r in rows[::step]:
        l = cv2.imread(str(Path(fdir) / r['left']), 0); rr = cv2.imread(str(Path(fdir) / r['right']), 0)
        Lr, Rr = m.apply(l, rr)
        k1, d1 = sift.detectAndCompute(Lr, None); k2, d2 = sift.detectAndCompute(Rr, None)
        if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
            continue
        for a, b in (p for p in bf.knnMatch(d1, d2, k=2) if len(p) == 2):
            if a.distance > 0.7 * b.distance:
                continue
            (x1, y1), (x2, y2) = k1[a.queryIdx].pt, k2[a.trainIdx].pt
            dx = x1 - x2
            if 0.5 <= dx <= 96 * width / 640.0 and abs(y2 - y1) < 4:
                dys.append(y2 - y1); xs.append(x1 / width); ys.append(y1 / Lr.shape[0])
    dys, xs, ys = map(np.asarray, (dys, xs, ys))
    if dys.size < 50:
        return dict(width=width, n=int(dys.size))
    med = float(np.median(dys)); mad = float(1.4826 * np.median(np.abs(dys - med)))
    reg = {}
    for nm_, sel in (('left', xs < 1 / 3), ('centre', (xs >= 1 / 3) & (xs < 2 / 3)), ('right', xs >= 2 / 3),
                     ('top', ys < 1 / 3), ('bottom', ys >= 2 / 3)):
        if sel.sum() > 20:
            reg[nm_] = round(float(np.median(dys[sel])), 3)
    # linear model dy = e + f x + g y (residual rotation / scale between the rectified rows)
    A = np.stack([np.ones_like(xs), xs - 0.5, ys - 0.5], 1); coef = np.linalg.lstsq(A, dys, rcond=None)[0]
    return dict(width=width, n=int(dys.size), median_px=round(med, 3), mad_px=round(mad, 3),
                p95_abs_px=round(float(np.percentile(np.abs(dys - med), 95)), 3), by_region=reg,
                fit_offset_px=round(float(coef[0]), 3), fit_dy_across_width_px=round(float(coef[1]), 3),
                fit_dy_across_height_px=round(float(coef[2]), 3))


def ransac_planes(z, K, fb, max_planes=3, min_px=3000, thr=0.35, iters=300, rng=None):
    """disparity-space planes on z (work res): [(a, b, c, inlier mask, rms_px)]"""
    rng = rng or np.random.default_rng(0)
    fx, fy, cx, cy = K; H, W = z.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    d = np.where(z > 0, fb / np.maximum(z, 1e-6), 0)
    ok = (z > 0.3) & (z < 4.0)
    out = []
    for _ in range(max_planes):
        idx = np.flatnonzero(ok)
        if idx.size < min_px:
            break
        U, V, Dd = u.ravel()[idx], v.ravel()[idx], d.ravel()[idx]
        best, bn = None, 0
        for _ in range(iters):
            s = rng.choice(idx.size, 3, replace=False)
            A = np.stack([U[s], V[s], np.ones(3)], 1)
            if abs(np.linalg.det(A)) < 1e-6:
                continue
            p = np.linalg.solve(A, Dd[s])
            nin = int((np.abs(U * p[0] + V * p[1] + p[2] - Dd) < thr).sum())
            if nin > bn:
                best, bn = p, nin
        if best is None or bn < min_px:
            break
        inl = np.abs(U * best[0] + V * best[1] + best[2] - Dd) < thr
        A = np.stack([U[inl], V[inl], np.ones(inl.sum())], 1)
        p = np.linalg.lstsq(A, Dd[inl], rcond=None)[0]                  # refit on the inliers
        res = Dd - (U * p[0] + V * p[1] + p[2]); inl = np.abs(res) < thr
        p = np.linalg.lstsq(np.stack([U[inl], V[inl], np.ones(inl.sum())], 1), Dd[inl], rcond=None)[0]
        res = Dd[inl] - (U[inl] * p[0] + V[inl] * p[1] + p[2])
        mask = np.zeros(H * W, bool); mask[idx[inl]] = True
        # the largest connected piece only (one physical surface)
        mm = mask.reshape(H, W).astype(np.uint8)
        nl, cc, st, _ = cv2.connectedComponentsWithStats(mm, connectivity=8)
        if nl > 1:
            k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            mm = cc == k
        else:
            mm = mm.astype(bool)
        if mm.sum() >= min_px:
            # residuals over the whole surface region (closed inlier piece), not only the inliers: robust sigma, the
            # part a quadratic in (u, v) explains (lens / rectification bowl), share beyond 0.5 px
            reg = cv2.morphologyEx(mm.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool) & (z > 0)
            uu, vv = u[reg], v[reg]
            r = d[reg] - (uu * p[0] + vv * p[1] + p[2])
            keep = np.abs(r) < 3.0
            uu, vv, r = uu[keep], vv[keep], r[keep]
            sig = float(1.4826 * np.median(np.abs(r - np.median(r))))
            Q = np.stack([np.ones_like(uu), uu, vv, uu * uu, uu * vv, vv * vv], 1) / np.array([1, W, H, W * W, W * H, H * H])
            rq = r - Q @ np.linalg.lstsq(Q, r, rcond=None)[0]
            curv = float(np.sqrt(max(np.mean(r * r) - np.mean(rq * rq), 0)))
            zz = z[reg][keep]
            out.append(dict(p=p, mask=mm, rms_px=sig, npx=int(reg.sum()), curv_px=curv, beyond05=float(np.mean(np.abs(r) > 0.5)),
                            rms_m=float(np.median(sig * zz * zz / fb)), zmed=float(np.median(zz))))
        ok &= ~mask.reshape(H, W)
    return out


def plane_metric(p, K, fb):
    fx, fy, cx, cy = K; a, b, c = p
    m = np.array([a * fx, b * fy, c + a * cx + b * cy]); h = fb / np.linalg.norm(m)
    return m * h / fb, h                                               # unit normal (camera), distance (m)


def main():
    fdir, npz, tfj, outp = sys.argv[1:5]
    rows, ci = load_flight(fdir); tf = json.load(open(tfj))
    T_lr = tf_lookup(tf, ci['left']['frame_id'], ci['right']['frame_id'])
    res = dict(flight=fdir)
    res['row_error'] = [] if '--no-rows' in sys.argv else [row_error(fdir, rows, ci, T_lr[:3, :3], T_lr[:3, 3], w, max(1, len(rows) // 60)) for w in (640, 1024)]
    print(json.dumps(res['row_error']), flush=True)
    Z = np.load(npz); D, Rw, cw, t, floor = Z['Doff'], Z['Rw'], Z['cw'], Z['t'], float(Z['floor'])
    K = tuple(float(x) for x in Z['K']); fb = K[0] * 0.0752856   # fx_work * baseline
    T_bl = tf_lookup(tf, 'base_link', ci['left']['frame_id']); fb = K[0] * float(np.linalg.norm(T_lr[:3, 3]))
    floor_pts, boards, rms = [], [], []
    rng = np.random.default_rng(0)
    for i in range(0, len(D), 1):
        for pl in ransac_planes(D[i].astype(np.float32), K, fb, rng=rng):
            n_c, h = plane_metric(pl['p'], K, fb)
            n_w = Rw[i] @ n_c                                          # normal pointing away from the camera
            rms.append((pl['rms_px'], pl['rms_m'], pl['zmed'], pl['npx'], pl['curv_px'], pl['beyond05']))
            if abs(n_w[2]) > np.cos(np.radians(10)) and n_w[2] < 0:   # floor seen from above (normal points down)
                floor_pts.append((i, h, cw[i][2] - floor, pl['rms_px']))
            elif abs(n_w[2]) < np.sin(np.radians(20)):                 # vertical board / wall
                d_w = float(n_w @ cw[i] + h)                           # world plane: n_w . X = d_w
                boards.append((i, n_w, d_w, h, pl['rms_px'], t[i]))
    rms = np.array(rms)
    res['planes'] = dict(n=int(len(rms)), note='robust sigma (1.4826 MAD) of disparity residuals over each plane region',
                         curvature_px_median=round(float(np.median(rms[:, 4])), 3),
                         beyond_0p5px_median=round(float(np.median(rms[:, 5])), 3),
                         rms_px_median=round(float(np.median(rms[:, 0])), 3),
                         rms_px_p90=round(float(np.percentile(rms[:, 0], 90)), 3),
                         rms_m_median=round(float(np.median(rms[:, 1])), 4),
                         by_distance={f'{lo}-{hi} m': dict(n=int(s.sum()), rms_px=round(float(np.median(rms[s, 0])), 3),
                                                            rms_mm=round(1e3 * float(np.median(rms[s, 1])), 1))
                                      for lo, hi in ((0.3, 1.0), (1.0, 2.0), (2.0, 4.0))
                                      for s in [(rms[:, 2] >= lo) & (rms[:, 2] < hi)] if s.sum() >= 3}) if len(rms) else {}
    if floor_pts:
        F = np.array(floor_pts)
        A = np.stack([F[:, 2], np.ones(len(F))], 1); k, b0 = np.linalg.lstsq(A, F[:, 1], rcond=None)[0]
        res['floor_height'] = dict(n=len(F), vio_range_m=[round(float(F[:, 2].min()), 2), round(float(F[:, 2].max()), 2)],
                                   stereo_vs_vio_slope=round(float(k), 4), offset_m=round(float(b0), 3),
                                   resid_rms_m=round(float(np.std(F[:, 1] - (k * F[:, 2] + b0))), 3),
                                   median_diff_m=round(float(np.median(F[:, 1] - F[:, 2])), 3))
    # boards: cluster by world normal (15 deg) + offset (0.25 m); per cluster with >= 0.8 m distance span: slope of the
    # world offset vs distance (0 if the depth scale is right)
    used = np.zeros(len(boards), bool); cl = []
    for j in range(len(boards)):
        if used[j]:
            continue
        g = [k for k in range(len(boards)) if not used[k] and boards[k][1] @ boards[j][1] > np.cos(np.radians(15))
             and abs(boards[k][2] - boards[j][2]) < 0.25]
        used[g] = True
        hs = np.array([boards[k][3] for k in g]); dw = np.array([boards[k][2] for k in g])
        if len(g) >= 8 and np.ptp(hs) >= 0.8:
            A = np.stack([hs, np.ones(len(g))], 1); s = np.linalg.lstsq(A, dw, rcond=None)[0][0]
            cl.append(dict(n=len(g), dist_range_m=[round(float(hs.min()), 2), round(float(hs.max()), 2)],
                           offset_vs_dist_slope=round(float(s), 4), implied_depth_scale=round(float(1 / (1 - s)), 4),
                           offset_spread_m=round(float(np.std(dw)), 3),
                           frames=[int(boards[g[0]][0]), int(boards[g[-1]][0])]))
    res['board_scale_whole_flight'] = sorted(cl, key=lambda c: -c['n'])[:6]
    # tracks: same world plane (normal < 10 deg, offset < 0.15 m) in frames <= 1.5 s apart; per track <= 10 s with a
    # >= 0.5 m distance span: fit stereo distance h = s * x + b, x = -n . c (VIO camera displacement along the normal)
    order = sorted(range(len(boards)), key=lambda k: boards[k][5]); tracks = []
    for k in order:
        i, n_w, d_w, h, _, tk = boards[k]
        best = None
        for tr in tracks:
            j = tr[-1]
            if 0 < tk - boards[j][5] <= 1.5 and boards[j][1] @ n_w > np.cos(np.radians(10)) and abs(boards[j][2] - d_w) < 0.15 \
                    and tk - boards[tr[0]][5] <= 10.0:
                best = tr; break
        (best.append(k) if best is not None else tracks.append([k]))
    fits = []
    for tr in tracks:
        hs = np.array([boards[k][3] for k in tr])
        if len(tr) < 5 or np.ptp(hs) < 0.5:
            continue
        nm = np.mean([boards[k][1] for k in tr], 0); nm /= np.linalg.norm(nm)
        x = np.array([-(nm @ cw[boards[k][0]]) for k in tr])
        A = np.stack([x, np.ones_like(x)], 1); (sc, b0), *_ = np.linalg.lstsq(A, hs, rcond=None)
        fits.append(dict(n=len(tr), t=[round(float(boards[tr[0]][5]), 1), round(float(boards[tr[-1]][5]), 1)],
                         dist_m=[round(float(hs.min()), 2), round(float(hs.max()), 2)], scale=round(float(sc), 4),
                         resid_m=round(float(np.std(hs - (sc * x + b0))), 3)))
    w_ = np.array([f['n'] * (f['dist_m'][1] - f['dist_m'][0]) ** 2 for f in fits]) if fits else np.zeros(0)
    res['approach_scale'] = dict(tracks=sorted(fits, key=lambda f: -f['n'])[:10], n_tracks=len(fits),
                                 weighted_median_scale=round(float(np.median(np.repeat([f['scale'] for f in fits],
                                                                                         np.maximum(1, (w_ / max(w_.min(), 1e-9)).astype(int))))), 4) if fits else None,
                                 median_scale=round(float(np.median([f['scale'] for f in fits])), 4) if fits else None)
    Path(outp).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
