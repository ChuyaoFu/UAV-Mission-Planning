#!/usr/bin/env python3
"""Window detector + depth-only gap ("Gate B") measurement on REAL skybeta-05 recordings.

Input: tools/record_manual.py flights (raw 1024x640 pairs + VIO pose per pair). Depth comes from the exact node pipeline
(realdrone/stereo_depth_node.DepthPipeline, preset half320, noise filters min-support 150 + speckle 800, out 640x400).
The VIO pose (cuVSLAM odom: world z up, base_link FLU) is turned into the window module's NED convention
(x = odom y, y = odom x, z = -odom z; a gravity-aligned z-down frame, yaw arbitrary).

Per frame:
  * perception.window.detect without a prior, then with a simulated site prior (median of the no-prior detections in
    the segment + a 0.35 m offset) and the median measured size as known_size_m;
  * measure_gap(): depth-only opening between two vertical edges in a horizontal band at the camera height (or at
    the window's height), from the attitude only (VIO-independent): left / right edge, centre, width, normal.
Consistency: centres moved to the VIO world must stay put; spread over sliding <= 1 s windows and frame-to-frame.

Usage: python3 eval_window_real.py FLIGHT_DIR --tf tf_static.json --frames 130:170 --out OUT_DIR [--pngs 140,150]
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
OFFICIAL = HERE.parent
sys.path.insert(0, str(OFFICIAL)); sys.path.insert(0, str(OFFICIAL / 'realdrone')); sys.path.insert(0, str(OFFICIAL / 'realdrone' / 'tools'))
from common.geometry import Frame  # noqa: E402
from perception import window as win  # noqa: E402
from real_eval import load_flight, tf_lookup  # noqa: E402
from stereo_depth_node import DepthPipeline, OPTIONS, quat_xyzw_to_mat  # noqa: E402

M_ENU2NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
HALF_SPAN, HALF_HEIGHT = 0.20, 0.10


def make_pipe(ci, R_lr, t_lr):
    L, R = ci['left'], ci['right']
    K_l = np.array(L['K']).reshape(3, 3); K_r = np.array(R['K']).reshape(3, 3)
    D_l = np.array(L['D'][:8], float); D_r = np.array(R['D'][:8], float)
    if not np.any(D_l): D_l = np.zeros(5)
    if not np.any(D_r): D_r = np.zeros(5)
    return DepthPipeline(OPTIONS['half320'], K_l, D_l, K_r, D_r, (L['width'], L['height']), R_lr, t_lr, out_width=640,
                         max_depth_m=10.0, speckle=(800, 2), minsup=(150, 1.0))


class Recording:
    def __init__(self, fdir, tf_path):
        self.fdir = Path(fdir)
        self.rows, ci = load_flight(fdir)
        tf = json.load(open(tf_path))
        lo, ro = ci['left']['frame_id'], ci['right']['frame_id']
        T_lr = tf_lookup(tf, lo, ro)
        self.pipe = make_pipe(ci, T_lr[:3, :3], T_lr[:3, 3])
        T_bl = tf_lookup(tf, 'base_link', lo)
        R_lrect = quat_xyzw_to_mat(*self.pipe.model.rect_in_left_quat())
        self.R_bc = T_bl[:3, :3] @ R_lrect; self.t_bc = T_bl[:3, 3]
        self.t = np.array([int(r['stamp_ns']) * 1e-9 for r in self.rows]); self.t -= self.t[0]

    def frame(self, i):
        r = self.rows[i]
        L = cv2.imread(str(self.fdir / r['left']), 0); R = cv2.imread(str(self.fdir / r['right']), 0)
        d, left_out, _ = self.pipe.run(L, R)
        depth = np.where(d > 0, d, np.nan).astype(np.float32)
        p = np.array([float(r[k]) for k in ('odom_x', 'odom_y', 'odom_z')])
        q = [float(r[k]) for k in ('odom_qx', 'odom_qy', 'odom_qz', 'odom_qw')]
        R_wb = quat_xyzw_to_mat(*q)
        R_wc = R_wb @ self.R_bc; c_w = p + R_wb @ self.t_bc
        return Frame(image=left_out, depth=depth, K=self.pipe.K_out, R_nc=M_ENU2NED @ R_wc, cam_ned=M_ENU2NED @ c_w,
                     stamp=float(self.t[i]))


# ------------------------------------------------------------------------------------------- depth-only gap
def measure_gap(fr, band=(-0.30, 0.30), z_ref=None, max_range=4.0, min_w=0.4, max_w=2.0, bin_deg=0.5,
                stride=2, far_margin=0.6):
    """Opening between two vertical structures, from depth and the camera attitude only.

    Points whose height is within `band` of z_ref (NED z; default the camera height) are taken in the level frame
    (forward = horizontal projection of the optical axis, right, metres from the camera). Per 0.5 deg bearing bin the
    nearest return (10th percentile) gives a polar scan. A gap = run of bins with no return or returns farther than
    far_margin beyond both flanks, flanked by near returns (<= max_range). Edge points = the flank bins' nearest
    points. Returns the gap nearest straight ahead (dict) or None."""
    fx, fy, cx, cy = fr.K
    H, W = fr.depth.shape
    d = fr.depth[::stride, ::stride]
    vv, uu = np.mgrid[0:H:stride, 0:W:stride]
    ok = np.isfinite(d) & (d > 0.3)
    rc = np.stack([(uu[ok] - cx) / fx * d[ok], (vv[ok] - cy) / fy * d[ok], d[ok]], -1)
    R = np.asarray(fr.R_nc, float); cam = np.asarray(fr.cam_ned, float)
    P = rc @ R.T                                   # NED offsets from the camera
    zr = 0.0 if z_ref is None else float(z_ref) - cam[2]
    m = (P[:, 2] >= zr + band[0]) & (P[:, 2] <= zr + band[1])
    f = R[:, 2].copy(); f[2] = 0.0
    if np.linalg.norm(f) < 0.3:
        return None
    f /= np.linalg.norm(f); rgt = np.array([-f[1], f[0], 0.0])
    a = P[m] @ f; b = P[m] @ rgt
    keep = (a > 0.2) & (np.hypot(a, b) <= max_range + 3.0)
    a, b = a[keep], b[keep]
    if len(a) < 30:
        return None
    th = np.degrees(np.arctan2(b, a)); rho = np.hypot(a, b)
    half = math.degrees(math.atan(max(cx, W - cx) / fx))
    nb = int(2 * half / bin_deg) + 1
    k = np.clip(((th + half) / bin_deg).astype(int), 0, nb - 1)
    near = np.full(nb, np.nan); cnt = np.bincount(k, minlength=nb)
    order = np.lexsort((rho, k))
    ks, rs = k[order], rho[order]
    starts = np.searchsorted(ks, np.arange(nb))
    pts = [None] * nb
    for i in range(nb):
        n = cnt[i]
        if n >= 3:
            j = starts[i] + int(0.1 * n)
            near[i] = rs[j]
            pts[i] = (a[order][j], b[order][j])
    best = None
    i = 0
    while i < nb:
        if np.isfinite(near[i]) and near[i] <= max_range:
            j = i + 1
            while j < nb and not (np.isfinite(near[j]) and near[j] <= max_range and near[j] < near[i] + far_margin):
                j += 1
            if j < nb and j - i >= 3:
                rl, rr = near[i], near[j]
                inner = near[i + 1:j]
                if np.all(~np.isfinite(inner) | (inner > max(rl, rr) + far_margin)):
                    pl, pr = np.array(pts[i]), np.array(pts[j])
                    w = float(np.linalg.norm(pl - pr))
                    if min_w <= w <= max_w:
                        ctr = 0.5 * (pl + pr)
                        brg = abs(math.degrees(math.atan2(ctr[1], ctr[0])))
                        if best is None or brg < best['bearing_deg']:
                            # pl has the smaller bearing = the LEFT edge (right is +b)
                            seg = pr - pl
                            nrm = np.array([seg[1], -seg[0]]) / max(np.linalg.norm(seg), 1e-9)   # horizontal normal
                            if nrm @ (-ctr) < 0:
                                nrm = -nrm                                                    # towards the camera
                            to_ned = lambda q: cam + q[0] * f + q[1] * rgt + np.array([0, 0, zr])
                            best = dict(bearing_deg=brg, width_m=w, left_lvl=pl, right_lvl=pr, center_lvl=ctr,
                                        range_m=float(np.linalg.norm(ctr)),
                                        left_ned=to_ned(pl), right_ned=to_ned(pr), center_ned=to_ned(ctr),
                                        normal_ned=nrm[0] * f + nrm[1] * rgt, n_inner_unknown=int(np.sum(~np.isfinite(inner))),
                                        n_bins=int(j - i - 1))
            i = j
        else:
            i += 1
    return best


# ------------------------------------------------------------------------------------------- metrics
def spread_1s(t, X, win_s=1.0):
    """median over frames of the max distance from the window mean, for sliding windows of <= win_s with >= 2 frames"""
    out = []
    for i in range(len(t)):
        sel = (t >= t[i]) & (t <= t[i] + win_s)
        if sel.sum() >= 2:
            Y = X[sel]; out.append(float(np.max(np.linalg.norm(Y - Y.mean(0), axis=1))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('flight'); ap.add_argument('--tf', required=True); ap.add_argument('--frames', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--pngs', default='')
    ap.add_argument('--name', default='')
    a = ap.parse_args()
    rec = Recording(a.flight, a.tf)
    i0, i1 = (int(x) for x in a.frames.split(':'))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    name = a.name or Path(a.flight).name
    frames = {}
    rows = []
    for i in range(i0, min(i1, len(rec.rows))):
        fr = rec.frame(i)
        frames[i] = fr
        e = win.detect(fr, HALF_SPAN, HALF_HEIGHT)
        g = measure_gap(fr)
        rows.append(dict(i=i, t=fr.stamp, e=e, gap=g))
    # simulated site prior: median of the no-prior detections + 0.35 m; measured size = their median
    det = [r for r in rows if r['e'].found]
    prior = known = None
    if det:
        C = np.array([r['e'].center_ned for r in det]); prior = np.median(C, 0) + np.array([0.25, -0.2, 0.15])
        known = (float(np.median([r['e'].width_m for r in det])), float(np.median([r['e'].height_m for r in det])))
    for r in rows:
        fr = frames[r['i']]
        r['ep'] = win.detect(fr, HALF_SPAN, HALF_HEIGHT, prior_center_ned=prior, known_size_m=known) if prior is not None else None
        ref = r['ep'] if (r['ep'] is not None and r['ep'].found) else (r['e'] if r['e'].found else None)
        r['gap_win'] = measure_gap(fr, band=(-0.08, 0.08), z_ref=ref.center_ned[2]) if ref is not None else None
    # ---- summary
    summ = dict(name=name, frames=[i0, i1], n=len(rows), prior=None if prior is None else prior.round(3).tolist(),
                known_size=None if known is None else [round(known[0], 3), round(known[1], 3)])
    for key in ('e', 'ep'):
        F = [r for r in rows if r[key] is not None and r[key].found]
        s = dict(found=len(F))
        if F:
            t = np.array([r['t'] for r in F]); C = np.array([r[key].center_ned for r in F])
            N = np.array([r[key].normal_ned for r in F])
            s.update(sources=sorted(set(r[key].source for r in F)),
                     width_med=float(np.median([r[key].width_m for r in F])), width_iqr=float(np.subtract(*np.percentile([r[key].width_m for r in F], [75, 25]))),
                     height_med=float(np.median([r[key].height_m for r in F])), height_iqr=float(np.subtract(*np.percentile([r[key].height_m for r in F], [75, 25]))),
                     center_spread_all_m=float(np.max(np.linalg.norm(C - np.median(C, 0), axis=1))),
                     center_dev_p50_m=float(np.median(np.linalg.norm(C - np.median(C, 0), axis=1))),
                     center_dev_p90_m=float(np.percentile(np.linalg.norm(C - np.median(C, 0), axis=1), 90)),
                     spread_1s_p50_m=float(np.median(spread_1s(t, C))) if len(spread_1s(t, C)) else None,
                     spread_1s_max_m=float(np.max(spread_1s(t, C))) if len(spread_1s(t, C)) else None,
                     normal_dev_p90_deg=float(np.percentile(np.degrees(np.arccos(np.clip(N @ (np.median(N, 0) / np.linalg.norm(np.median(N, 0))), -1, 1))), 90)),
                     clear_ok=int(sum(r[key].clearance_ok for r in F)),
                     reasons_not_found={})
        nf = [r[key].reason for r in rows if r[key] is not None and not r[key].found]
        s['reasons_not_found'] = {k: nf.count(k) for k in sorted(set(nf))}
        summ['window_' + ('noprior' if key == 'e' else 'prior')] = s
    for key in ('gap', 'gap_win'):
        G = [r for r in rows if r[key] is not None]
        s = dict(found=len(G))
        if G:
            t = np.array([r['t'] for r in G]); C = np.array([r[key]['center_ned'] for r in G])
            Wd = np.array([r[key]['width_m'] for r in G])
            s.update(width_med=float(np.median(Wd)), width_iqr=float(np.subtract(*np.percentile(Wd, [75, 25]))),
                     center_dev_p50_m=float(np.median(np.linalg.norm(C - np.median(C, 0), axis=1))),
                     center_dev_p90_m=float(np.percentile(np.linalg.norm(C - np.median(C, 0), axis=1), 90)),
                     spread_1s_p50_m=float(np.median(spread_1s(t, C))) if len(spread_1s(t, C)) else None)
        summ[key] = s
    # per-frame table
    tab = []
    for r in rows:
        d = dict(i=r['i'], t=round(r['t'], 2))
        for key in ('e', 'ep'):
            e = r[key]
            if e is None:
                continue
            d[key] = dict(found=e.found, src=e.source, reason=e.reason, conf=round(e.confidence, 2), clear=e.clearance_ok,
                          c=None if not e.found else e.center_ned.round(3).tolist(),
                          n=None if not e.found else e.normal_ned.round(3).tolist(),
                          wh=None if not e.found else [round(e.width_m, 3), round(e.height_m, 3)], range=None if not e.found else round(e.range_m, 2))
        for key in ('gap', 'gap_win'):
            g = r[key]
            d[key] = None if g is None else dict(w=round(g['width_m'], 3), c=g['center_ned'].round(3).tolist(), brg=round(g['bearing_deg'], 1),
                                                 range=round(g['range_m'], 2), l=g['left_ned'].round(3).tolist(), r=g['right_ned'].round(3).tolist())
        tab.append(d)
    json.dump(dict(summary=summ, frames=tab), open(out / f'{name}_{i0}_{i1}.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1))
    # ---- annotated PNGs
    for s_ in [x for x in a.pngs.split(',') if x]:
        i = int(s_)
        if i not in frames:
            continue
        r = next(x for x in rows if x['i'] == i); fr = frames[i]
        im = cv2.cvtColor(fr.image, cv2.COLOR_GRAY2BGR)
        dv = np.where(np.isfinite(fr.depth), np.clip(fr.depth / 5.0 * 255, 0, 255), 0).astype(np.uint8)
        dv = cv2.applyColorMap(dv, cv2.COLORMAP_TURBO); dv[~np.isfinite(fr.depth)] = 0
        lines = []
        for key, col in (('e', (0, 200, 255)), ('ep', (0, 255, 0))):
            e = r[key]
            if e is not None and e.found:
                uv, z = fr.project(e.corners_ned)
                for img in (im, dv):
                    cv2.polylines(img, [uv.astype(np.int32)], True, col, 2)
                lines.append('%s %s w%.2f h%.2f r%.2f c%.2f %s' % ('noprior' if key == 'e' else 'prior', e.source, e.width_m, e.height_m, e.range_m, e.confidence, 'CLEAR' if e.clearance_ok else e.reason))
            elif e is not None:
                lines.append('%s: not found (%s)' % ('noprior' if key == 'e' else 'prior', e.reason))
        for key, col in (('gap', (255, 0, 255)), ('gap_win', (255, 255, 0))):
            g = r[key]
            if g is None:
                lines.append('%s: none' % key); continue
            for P in (g['left_ned'], g['right_ned']):
                uv, z = fr.project(P[None])
                if z[0] > 0:
                    u, v = int(uv[0, 0]), int(uv[0, 1])
                    for img in (im, dv):
                        cv2.line(img, (u, max(0, v - 60)), (u, min(399, v + 60)), col, 2)
            lines.append('%s w%.2f range %.2f brg %.1f' % (key, g['width_m'], g['range_m'], g['bearing_deg']))
        for k_, s2 in enumerate([f'{name} #{i} t={fr.stamp:.1f}s'] + lines):
            cv2.putText(im, s2, (6, 18 + 18 * k_), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
            cv2.putText(im, s2, (6, 18 + 18 * k_), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.imwrite(str(out / f'{name}_f{i:03d}.png'), np.hstack([im, dv]))


if __name__ == '__main__':
    main()
