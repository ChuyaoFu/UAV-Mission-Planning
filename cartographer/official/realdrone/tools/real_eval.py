#!/usr/bin/env python3
"""Real skybeta stereo (tools/record_manual.py flights: raw 1024x640 gray pairs + VIO pose per pair) through the EXACT
stereo node pipeline (stereo_depth_node.DepthPipeline: rectify -> half320 SGBM -> net filter -> noise filters), filter
sets OFF vs ON, with truth-free labels from multi-view consistency:

  every near return (range < near_m) of the OFF depth is moved to the world with the VIO pose and projected into other
  frames of the same flight seen from >= min_baseline_m away (within +-window_s); there the OFF depth (3x3 min / max)
  either CONFIRMS it (same depth within tol), SEES THROUGH it (the ray continues >= tol + 0.10 m beyond it: free space)
  or gives no evidence (occluded, invalid, out of view).
      phantom = see >= 2 and confirm <= see / 3        real = confirm >= 2 and confirm >= see        else unknown
  plus: points more than 0.15 m below the floor are phantoms (geometry), whatever the votes.
A filter set removes a pixel's return if its own depth has no return there; recall / loss are over the OFF labels.
Mission-relevant = height above the floor > 0.10 m and range 0.4..near_m (ObstacleMap's mapped points).
Small floating cluster = 8-connected component of mission-relevant near returns of < small_px (640-scale) px whose
3-px ring has < 30 % returns within 25 % of its depth.

Caveats: moving people are "seen through" when they move (counted as phantoms); VIO drift over window_s widens tol.
Usage: python3 real_eval.py FLIGHT_DIR --tf tf_static.json --out OUT.json [--fence mission/course_prior.site.json
       --anchor first|last|<index>] [--pngs DIR]
"""
import argparse, csv, json, math, sys, time
from pathlib import Path

import numpy as np
import cv2

HERE = Path(__file__).resolve().parent; RD = HERE.parent
sys.path.insert(0, str(RD)); sys.path.insert(0, str(RD.parent))
from stereo_depth_node import DepthPipeline, OPTIONS, quat_xyzw_to_mat  # noqa: E402
from net_mask import NetMask  # noqa: E402
from net_model import NetModel, OdomPose, load_fence, quat_xyzw_to_R  # noqa: E402

CONFIGS = {                     # name -> (net, speckle, minsup)
    'off': (None, None, None),
    'net_mask': ('mask', None, None),
    'net_model': ('model', None, None),
    'noise': (None, (800, 2), (150, 1.0)),
    'mask+noise': ('mask', (800, 2), (150, 1.0)),
    'model+noise': ('model', (800, 2), (150, 1.0)),
    'minsup150': (None, None, (150, 1.0)),
    'speckle800': (None, (800, 2), None),
}


# ------------------------------------------------------------------------------------------------------------- TF
def tf_lookup(tf, target, source):
    """4x4 T(target <- source) from {'parent->child': [tx, ty, tz, qx, qy, qz, qw]} (child pose in parent)"""
    edges = {}
    for k, v in tf.items():
        p, c = k.split('->')
        T = np.eye(4); T[:3, :3] = quat_xyzw_to_mat(*v[3:]); T[:3, 3] = v[:3]
        edges.setdefault(p, []).append((c, T)); edges.setdefault(c, []).append((p, np.linalg.inv(T)))
    seen = {target: np.eye(4)}; todo = [target]
    while todo:
        f = todo.pop()
        for c, T in edges.get(f, []):
            if c not in seen:
                seen[c] = seen[f] @ T; todo.append(c)
    return seen[source]


def _png_complete(path):
    try:
        with open(path, 'rb') as f:
            f.seek(-12, 2); return f.read(12)[4:8] == b'IEND'
    except OSError:
        return False


def load_flight(fdir):
    fdir = Path(fdir)
    rows = []
    for r in csv.DictReader(open(fdir / 'pairs.csv')):
        if all(_png_complete(fdir / r[k]) for k in ('left', 'right')):    # flight 010807 ends in truncated files
            rows.append(r)
    ci = json.load(open(fdir / 'camera_info.json'))
    return rows, ci


def make_pipe(ci, R_lr, t_lr, net, speckle, minsup, fence):
    L, R = ci['left'], ci['right']
    K_l = np.array(L['K']).reshape(3, 3); K_r = np.array(R['K']).reshape(3, 3)
    D_l = np.array(L['D'][:8], float); D_r = np.array(R['D'][:8], float)
    if not np.any(D_l): D_l = np.zeros(5)
    if not np.any(D_r): D_r = np.zeros(5)
    return DepthPipeline(OPTIONS['half320'], K_l, D_l, K_r, D_r, (L['width'], L['height']), R_lr, t_lr, out_width=640,
                         max_depth_m=10.0, net_mask=NetMask() if net == 'mask' else None,
                         net_model=NetModel(fence) if net == 'model' else None, speckle=speckle, minsup=minsup)


# ---------------------------------------------------------------------------------------------------- evaluation
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('flight'); ap.add_argument('--tf', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--fence', default=str(RD.parent / 'mission/course_prior.site.json'))
    ap.add_argument('--anchor', default='first', help='start-pad pose for net_model: first | last | <row index>')
    ap.add_argument('--near-m', type=float, default=2.0); ap.add_argument('--window-s', type=float, default=4.0)
    ap.add_argument('--min-baseline-m', type=float, default=0.25); ap.add_argument('--nref', type=int, default=8)
    ap.add_argument('--tol', type=float, default=0.06); ap.add_argument('--small-px', type=int, default=600)
    ap.add_argument('--configs', default=','.join(CONFIGS)); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--save-npz', default='')
    a = ap.parse_args()
    rows, ci = load_flight(a.flight)
    if a.limit: rows = rows[:a.limit]
    tf = json.load(open(a.tf))
    lo, ro = ci['left']['frame_id'], ci['right']['frame_id']
    T_lr = tf_lookup(tf, lo, ro); R_lr, t_lr = T_lr[:3, :3], T_lr[:3, 3]
    T_bl = tf_lookup(tf, 'base_link', lo)
    fence = load_fence(a.fence)
    cfgs = [c for c in a.configs.split(',') if c]
    pipes = {c: make_pipe(ci, R_lr, t_lr, *CONFIGS[c], fence) for c in cfgs}
    P0 = pipes['off']
    R_lrect = quat_xyzw_to_mat(*P0.model.rect_in_left_quat())
    R_bc = T_bl[:3, :3] @ R_lrect; t_bc = T_bl[:3, 3]            # base_link <- rectified left camera
    fx, fy, cx, cy = P0.Kw; W, H = P0.work_size; fb = P0.fx * P0.baseline
    # start-frame anchor for net_model (prior's start pad): first pose, last pose (landed on the pad) or a row index
    ai = 0 if a.anchor == 'first' else (len(rows) - 1 if a.anchor == 'last' else int(a.anchor))
    op = OdomPose(anchor='first', base_height_m=0.05, max_dt=1e9)
    ra = rows[ai]; op.add(0.0, [float(ra[k]) for k in ('odom_x', 'odom_y', 'odom_z')],
                          [float(ra[k]) for k in ('odom_qx', 'odom_qy', 'odom_qz', 'odom_qw')])
    n = len(rows); t = np.array([int(r['stamp_ns']) * 1e-9 for r in rows]); t -= t[0]
    Rw = np.zeros((n, 3, 3)); cw = np.zeros((n, 3))
    Doff = np.zeros((n, H, W), np.float16); keep = {c: np.zeros((n, H, W), bool) for c in cfgs if c != 'off'}
    Lw = np.zeros((n, H, W), np.uint8); info = []; tim = {c: [] for c in cfgs}
    t0 = time.time()
    for i, r in enumerate(rows):
        L = cv2.imread(str(Path(a.flight) / r['left']), 0); R = cv2.imread(str(Path(a.flight) / r['right']), 0)
        p = np.array([float(r[k]) for k in ('odom_x', 'odom_y', 'odom_z')]); q = [float(r[k]) for k in ('odom_qx', 'odom_qy', 'odom_qz', 'odom_qw')]
        R_wb = quat_xyzw_to_R(*q); Rw[i] = R_wb @ R_bc; cw[i] = p + R_wb @ t_bc
        op.hist = [(float(t[i]), p, R_wb)]
        pose = op.camera(float(t[i]), R_bc, t_bc)
        rec = dict(i=i, t=round(float(t[i]), 3), vo=int(r['vo_state']))
        for c, pp in pipes.items():
            d, left_out, tm = pp.run(L, R, pose if CONFIGS[c][0] == 'model' else None)
            tim[c].append(1e3 * (tm['match'] + tm['mask']))
            dw = d[::2, ::2]
            if c == 'off':
                Doff[i] = dw; Lw[i] = cv2.resize(left_out, (W, H), interpolation=cv2.INTER_AREA)
            else:
                keep[c][i] = dw > 0
            if CONFIGS[c][0] == 'model':
                rec['model'] = pp.model_info
        info.append(rec)
        if i % 100 == 0:
            print(f'  {i}/{n} {time.time() - t0:.0f}s', flush=True)
    # -------------------------------------------------- floor height in the VIO frame (lowest dense horizontal level)
    zs = []
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    rx, ry = (u - cx) / fx, (v - cy) / fy
    for i in range(0, n, 3):
        z = Doff[i].astype(np.float32); m = (z > 0) & (z < 4.0)
        Pc = np.stack([rx[m] * z[m], ry[m] * z[m], z[m]], 1)
        zs.append((Pc @ Rw[i].T + cw[i])[:, 2])
    zs = np.concatenate(zs)
    hist, edges = np.histogram(zs, bins=np.arange(np.percentile(zs, 0.5) - 0.1, np.percentile(zs, 60), 0.02))
    pk = np.nonzero(hist > 0.25 * hist.max())[0][0]                 # lowest strong level
    pk = pk + int(np.argmax(hist[pk:pk + 8]))
    floor = float(0.5 * (edges[pk] + edges[pk + 1]))
    # -------------------------------------------------- per-frame neighbourhood depth stats (3x3) for the references
    k3 = np.ones((3, 3), np.uint8)
    def refstats(j):
        z = Doff[j].astype(np.float32); ok = z > 0
        dmax = cv2.dilate(np.where(ok, z, 0).astype(np.float32), k3)
        dmin = cv2.erode(np.where(ok, z, 1e4).astype(np.float32), k3)
        nv = cv2.boxFilter(ok.astype(np.float32), -1, (3, 3), normalize=False)
        return dmin, dmax, nv
    cache = {}
    # -------------------------------------------------- labels of the OFF near returns
    lab = np.zeros((n, H, W), np.int8)          # 0 none, 1 phantom, 2 real, 3 unknown (near returns only)
    hgt = np.zeros((n, H, W), np.float16)
    nref_used = []
    for i in range(n):
        z = Doff[i].astype(np.float32)
        rng = z * np.sqrt(1 + rx * rx + ry * ry)
        near = (z > 0) & (rng < a.near_m)
        ys, xs = np.nonzero(near)
        if ys.size == 0:
            continue
        zz = z[ys, xs]
        Pw = np.stack([rx[ys, xs] * zz, ry[ys, xs] * zz, zz], 1) @ Rw[i].T + cw[i]
        hgt[i][ys, xs] = Pw[:, 2] - floor
        # references: frames within the window seen from >= min_baseline away, nearest in time first, both sides
        cand = [j for j in np.argsort(np.abs(t - t[i])) if j != i and abs(t[j] - t[i]) <= a.window_s
                and np.linalg.norm(cw[j] - cw[i]) >= a.min_baseline_m][:a.nref]
        nref_used.append(len(cand))
        see = np.zeros(ys.size, np.int16); conf = np.zeros(ys.size, np.int16)
        for j in cand:
            if j not in cache:
                cache[j] = refstats(j)
                if len(cache) > 64: cache.pop(next(iter(cache)))
            dmin, dmax, nv = cache[j]
            Q = (Pw - cw[j]) @ Rw[j]
            zq = Q[:, 2]
            with np.errstate(divide='ignore', invalid='ignore'):
                mu = np.rint(fx * Q[:, 0] / zq + cx); mv = np.rint(fy * Q[:, 1] / zq + cy)
            ins = (zq > 0.2) & (mu >= 0) & (mu <= W - 1) & (mv >= 0) & (mv <= H - 1)
            mi, vi, ui = np.nonzero(ins)[0], mv[ins].astype(int), mu[ins].astype(int)
            lo_, hi_, nn = dmin[vi, ui], dmax[vi, ui], nv[vi, ui]
            ze = zq[mi]; tl = a.tol * ze + 0.05
            c_ = (hi_ > 0) & (ze >= lo_ - tl) & (ze <= hi_ + tl)
            s_ = (nn >= 5) & (lo_ < 1e4) & (lo_ > ze + tl + 0.10)
            conf[mi] += c_; see[mi] += s_ & ~c_
        L_ = np.full(ys.size, 3, np.int8)
        L_[(conf >= 2) & (conf >= see)] = 2
        L_[(see >= 2) & (3 * conf <= see)] = 1
        L_[hgt[i][ys, xs].astype(np.float32) < -0.15] = 1
        lab[i][ys, xs] = L_
    # -------------------------------------------------- statistics
    mis = (lab > 0) & (hgt.astype(np.float32) > 0.10)
    rng_all = Doff.astype(np.float32) * np.sqrt(1 + rx * rx + ry * ry)[None]
    mis &= rng_all >= 0.4
    allreal = lab == 2
    def frac(x, y): return round(float(x) / max(float(y), 1.0), 4)
    S = dict(flight=str(a.flight), n_frames=n, duration_s=round(float(t[-1]), 1), floor_z_vio=round(floor, 3),
             refs_per_frame=round(float(np.mean(nref_used)), 2) if nref_used else 0, configs={})
    ph = mis & (lab == 1); re = mis & (lab == 2); un = mis & (lab == 3)
    S['off_labels'] = dict(mission_near_px640_per_frame=round(4 * float(mis.sum()) / n, 1),
                           phantom=frac(ph.sum(), mis.sum()), real=frac(re.sum(), mis.sum()), unknown=frac(un.sum(), mis.sum()),
                           phantom_px640_per_frame=round(4 * float(ph.sum()) / n, 1),
                           below_floor_px640_per_frame=round(4 * float(((lab > 0) & (hgt.astype(np.float32) < -0.15)).sum()) / n, 1))
    # small floating clusters per config
    def clusters(valid_i, i):
        m = (mis[i] & valid_i).astype(np.uint8)
        nl, cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        out = []
        z = Doff[i].astype(np.float32)
        for k in range(1, nl):
            area = st[k, cv2.CC_STAT_AREA]
            if 4 * area >= a.small_px:
                continue
            x0, y0, w0, h0 = st[k, :4]
            xa, ya, xb, yb = max(x0 - 4, 0), max(y0 - 4, 0), min(x0 + w0 + 4, W), min(y0 + h0 + 4, H)
            sub = cc[ya:yb, xa:xb] == k
            ring = cv2.dilate(sub.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~sub
            zc = float(np.median(z[ya:yb, xa:xb][sub]))
            zr = z[ya:yb, xa:xb][ring]
            if np.count_nonzero(np.abs(zr - zc) < 0.25 * zc) >= 0.3 * ring.sum():
                continue                                            # attached to a surface: not floating
            ls = lab[i][ya:yb, xa:xb][sub]
            out.append(int(np.bincount(ls, minlength=4)[1:].argmax()) + 1)
        return out
    for c in cfgs:
        kv = (Doff > 0) if c == 'off' else keep[c]
        rem_ph = ph & ~kv; rem_re = re & ~kv
        cl = []
        for i in range(n):
            cl += clusters(kv[i], i)
        cl = np.array(cl, int)
        S['configs'][c] = dict(
            phantom_removed=frac(rem_ph.sum(), ph.sum()), real_near_lost=frac(rem_re.sum(), re.sum()),
            real_all_lost=frac((allreal & ~kv).sum(), allreal.sum()),
            phantom_left_px640_per_frame=round(4 * float((ph & kv).sum()) / n, 1),
            mission_near_left_px640_per_frame=round(4 * float((mis & kv).sum()) / n, 1),
            floating_clusters_per_100_frames=round(100 * cl.size / n, 1),
            floating_by_label=dict(phantom=int((cl == 1).sum()), real=int((cl == 2).sum()), unknown=int((cl == 3).sum())),
            ms_match_plus_filter_mac=round(float(np.median(tim[c])), 1))
    if any(CONFIGS[c][0] == 'model' for c in cfgs):
        mi_ = [r['model'] for r in info if r.get('model', {}).get('model')]
        sh = [s for m in mi_ for s in (m.get('shift_m') or []) if s is not None]
        S['net_model'] = dict(frames_with_model=len(mi_), frames=n, refined_face_shifts_m=dict(
            n=len(sh), median_abs=round(float(np.median(np.abs(sh))), 3) if sh else None,
            p90_abs=round(float(np.percentile(np.abs(sh), 90)), 3) if sh else None),
            dropped_px_per_frame=round(float(np.mean([m.get('dropped', 0) for m in mi_])), 1) if mi_ else 0, anchor=a.anchor)
    print(json.dumps(S, indent=1))
    Path(a.out).write_text(json.dumps(S, indent=1))
    if a.save_npz:
        np.savez_compressed(a.save_npz, Doff=Doff, lab=lab, hgt=hgt, Lw=Lw, t=t, Rw=Rw, cw=cw, floor=floor,
                            **{'keep_' + c.replace('+', '_'): keep[c] for c in keep}, K=np.array(P0.Kw))


if __name__ == '__main__':
    main()
