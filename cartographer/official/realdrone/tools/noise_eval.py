#!/usr/bin/env python3
"""Offline evaluation of cheap source-side filters against the free-space stereo noise that still blocks the mission
(official-compat v4G: `small_cluster_kept` clusters > 0.3 m from any real object).

Input: netmask_extract.py --events npz (full-rate stereo pairs before each mission event, truth + VIO pose, the flight's
prior). Per event, every filter config runs over the window in order (the temporal check needs it), the node's
ObstacleMap is replayed (truth pose, as analyze_obstacles.py) and at the event time we count the confirmed points within
RADIUS of the cluster the mission kept. Events are classed NOISE (cluster centre > 0.3 m from every solid and from the
cage planes, flight's own asflown layout) or REAL. Per frame, pixel classes from truth (netmask_eval.label) give the
removed share of free-space points (< 2 m), wall/board points and board ends.

Baseline = the v4G source: compat SGBM (oak_stereo_model, 640, MODE_SGBM) + net filter (net_model with the prior fence
and VIO pose; net_mask texture fallback without a pose) - what flew.
Configs (name -> what changes):
  base
  spk<W>_<R>   SGBM speckleWindowSize W, speckleRange R            (compat default 100, 2)
  tex<X>       post texture gate texture_min_dn X                  (default 1.2)
  ms<N>_<D>    depth_filters.min_support(max_size N px, max_diff D px) on the final disparity
  tmp<T>_<K>[s] depth_filters.TemporalCheck(tol T %, last K frames, s = require a previous return)

Usage: python3 noise_eval.py --events DIR_or_npz... --flights FLIGHTS_ROOT --configs base spk400_2 ... [--json out]
"""
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1]))
import netmask_eval as E                                   # noqa: E402  (label, CS, solids_dist, SM, constants)
from net_mask import NetMask                                # noqa: E402
from net_model import NetModel                              # noqa: E402
from depth_filters import min_support, TemporalCheck        # noqa: E402

RADIUS, NOISE_M = 0.35, 0.30


def sgbm(L, R, fx, B, speckle=None, texture_min=None):
    """oak_stereo_model.sgbm_depth with overrides -> depth (0 invalid)"""
    SM = E.SM
    q = dict(SM.SGBM)
    if speckle:
        q['speckleWindowSize'], q['speckleRange'] = int(speckle[0]), int(speckle[1])
    mode = getattr(cv2, 'STEREO_SGBM_' + q.pop('mode'))
    d16 = cv2.StereoSGBM_create(mode=mode, **q).compute(L, R)
    P = dict(SM.POST)
    if texture_min is not None:
        P['texture_min_dn'] = float(texture_min)
    valid = d16 > 0
    disp = d16.astype(np.float32) / 16.0
    qn = 2 ** P['subpixel_bits']; disp = np.round(disp * qn) / qn; disp[~valid] = 0
    b = cv2.GaussianBlur(L.astype(np.float32), (0, 0), P['texture_blur_sigma'])
    w = (P['texture_win'], P['texture_win'])
    std = np.sqrt(np.maximum(cv2.blur(b * b, w) - cv2.blur(b, w) ** 2, 0))
    valid &= std >= P['texture_min_dn']
    disp[~valid] = 0
    disp = cv2.medianBlur(disp, P['median_ksize'])
    valid &= disp >= P['min_disp_px']
    disp[~valid] = 0
    return disp


def parse(name):
    if '+' in name:                                   # combination, e.g. spk800_2+ms150_1
        out = {}
        for part in name.split('+'):
            out.update(parse(part))
        return out
    if name == 'base':
        return {}
    if name.startswith('spk'):
        w, r = name[3:].split('_'); return dict(speckle=(int(w), int(r)))
    if name.startswith('tex'):
        return dict(texture_min=float(name[3:]))
    if name.startswith('ms'):
        n, d = name[2:].split('_'); return dict(minsup=(int(n), float(d)))
    if name.startswith('tmp'):
        s = name.endswith('s'); t, k = name[3:].rstrip('s').split('_')
        return dict(temporal=(float(t) / 100.0, int(k), s))
    raise ValueError(name)


class Chain:
    """one config over one event window (stateful: temporal history)"""
    def __init__(self, cfg, K, B, fence, require_support_cls=None):
        self.cfg, self.K, self.B = cfg, K, B
        self.nm = NetMask(); self.md = NetModel(fence) if fence else None
        tp = cfg.get('temporal')
        self.tc = None
        if tp:
            self.tc = TemporalCheck(K, tol=tp[0], nprev=tp[1], require=tp[2])
        self.ms_extra = []

    def __call__(self, L, R, pose_vio, pose_truth, stamp, base_disp=None):
        c = self.cfg
        if base_disp is not None and not c.get('speckle') and c.get('texture_min') is None:
            disp = base_disp.copy()
        else:
            disp = sgbm(L, R, self.K[0], self.B, c.get('speckle'), c.get('texture_min'))
            # net filter as flown
            bad = None
            if self.md is not None and pose_vio is not None:
                bad, _ = self.md(L, disp, R, self.K, self.B, pose_vio[0], pose_vio[1])
            if bad is None:
                bad = self.nm(L, disp, R)
            disp[bad] = 0
        t0 = time.perf_counter()
        if c.get('minsup'):
            disp[~min_support(disp, *c['minsup'])] = 0
        if self.tc is not None and pose_vio is not None:
            z = np.where(disp > 0, self.K[0] * self.B / np.maximum(disp, 1e-6), 0).astype(np.float32)
            keep = self.tc(z, pose_vio[0], pose_vio[1], stamp)
            disp[~keep] = 0
        self.ms_extra.append(time.perf_counter() - t0)
        return np.where(disp > 0, self.K[0] * self.B / np.maximum(disp, 1e-6), 0).astype(np.float32), disp



def classify_event(L, ev):
    c = np.asarray(ev['center'], float) + np.array([0, 0, -0.05])
    a, b, h = c[0] + L.START[0], c[1] + L.START[1], -c[2]
    dsol = float(E.solids_dist(L, np.array([a]), np.array([b]), np.array([h]))[0])
    dnet = float(min(a, L.CAGE - a, b, L.CAGE - b, L.CAGE_H - h))
    return ('real' if min(dsol, abs(dnet)) <= NOISE_M or dnet < 0 else 'noise'), round(dsol, 3), round(dnet, 3)


def run(files, flights_root, configs, before=4.5):
    sys.path.insert(0, str(HERE.parents[1]))
    from node.obstacle_map import ObstacleMap
    from common.geometry import Frame
    rows, pix, cost = [], {c: {} for c in configs}, {c: [] for c in configs}
    for f in files:
        z = np.load(f, allow_pickle=True)
        Z = {k: z[k] for k in z.files}
        flight = str(Z['flight']); events = json.loads(str(Z['events']))
        if not events:
            continue
        Lay = E.CS.load_layout(Path(flights_root) / flight)
        K = tuple(float(x) for x in Z['K']); B = float(Z['baseline']); st = Z['stamp']
        fence = json.loads(str(Z['prior']))['fence'] if str(Z['prior']) else None
        for ev in events:
            idx = [i for i in range(len(st)) if ev['t'] - before <= st[i] <= ev['t'] + 1e-3]
            if len(idx) < 10:
                continue
            kind, dsol, dnet = classify_event(Lay, ev)
            c_ev = np.asarray(ev['center'], float) + np.array([0, 0, -0.05])
            row = dict(flight=flight, t=round(ev['t'], 2), n_mission=ev['n'], kind=kind, d_solid=dsol, d_net=dnet, frames=len(idx))
            chains = {c: Chain(parse(c), K, B, fence) for c in configs}
            maps = {c: ObstacleMap() for c in configs}
            base_disp = {}
            for i in idx:
                pv = (Z['R_nc_vio'][i], Z['cam_vio'][i]); pt = (Z['R_nc'][i], Z['cam_ned'][i])
                lab = None
                for c in configs:
                    d, disp = chains[c](Z['left'][i], Z['right'][i], pv, pt, float(st[i]), base_disp.get(i))
                    if c == 'base':
                        base_disp[i] = disp.copy(); d_base = d
                        lab = E.label(Lay, d, K, pt[0], pt[1])
                    maps[c].insert(Frame(image=None, depth=d, K=K, R_nc=pt[0], cam_ned=pt[1]), 0.0, float(st[i]))
                    if lab is not None:
                        removed = ((d_base > 0) & ~(d > 0))[::E.STRIDE, ::E.STRIDE]
                        mapped = lab['ok'] & (lab['r3'] > 0.40) & (lab['h'] > 0.10)
                        sel = dict(free=lab['phantom'] & mapped & (lab['r3'] < E.NEAR_M), real=lab['real'] & mapped,
                                   end=lab['real_end'] & mapped)
                        acc = pix[c].setdefault(kind, {k: [0, 0] for k in sel})
                        for k, m in sel.items():
                            acc[k][0] += int(m.sum()); acc[k][1] += int((m & removed).sum())
            for c in configs:
                P_ = maps[c].points
                n_at = 0
                if len(P_):
                    n_at = int((np.linalg.norm(P_ - c_ev, axis=1) < RADIUS).sum())
                row[c] = n_at
                cost[c] += chains[c].ms_extra
            rows.append(row)
            print('  ', {k: row[k] for k in ('flight', 't', 'kind', 'd_solid', 'd_net')}, {c: row[c] for c in configs}, flush=True)
    return rows, pix, cost


def summarize(rows, pix, cost, configs):
    out = {}
    for c in configs:
        o = {}
        for kind in ('noise', 'real'):
            rr = [r for r in rows if r['kind'] == kind and r.get('base', 0) > 0]
            o[kind] = dict(events=len([r for r in rows if r['kind'] == kind]), reproduced=len(rr),
                           cleared=sum(1 for r in rr if r[c] == 0), pts_base=sum(r['base'] for r in rr), pts=sum(r[c] for r in rr))
            o[kind]['pts_removed'] = round(1 - o[kind]['pts'] / max(o[kind]['pts_base'], 1), 3)
        pc = {}
        for kind, acc in pix[c].items():
            pc[kind] = {k: round(v[1] / max(v[0], 1), 4) for k, v in acc.items()}
        allacc = {}
        for acc in pix[c].values():
            for k, v in acc.items():
                allacc.setdefault(k, [0, 0]); allacc[k][0] += v[0]; allacc[k][1] += v[1]
        o['pixels_removed'] = {k: round(v[1] / max(v[0], 1), 4) for k, v in allacc.items()}
        o['ms_extra_mean'] = round(1e3 * float(np.mean(cost[c])), 2) if cost[c] else 0.0
        out[c] = o
    return out


def replica(replica_dir, configs, edge_px=6):
    """single-frame configs on the venue-replica frames (per-pixel truth): free = return < 0.8x the nearest true surface
    and < 2 m (any surface), real = within 5 % on a solid, edge = real within edge_px of a true depth step > 15 %"""
    files = [f for f in sorted(glob.glob(str(Path(replica_dir) / '*.npz'))) if not Path(f).name.startswith('compare')]
    cage = json.loads((Path(replica_dir).parent / 'index.json').read_text())['world_labels']['cage']
    fence = dict(forward_min=cage['n_range'][0], forward_max=cage['n_range'][1], left_min=-cage['e_range'][1],
                 left_max=-cage['e_range'][0], up_max=cage['height'])
    acc = {c: {k: [0, 0] for k in ('free', 'real', 'edge')} for c in configs}
    for f in files:
        z = np.load(f, allow_pickle=True)
        L, R = z['image'], z['image_right']; k = np.asarray(z['K'], float).ravel(); K = tuple(k); B = 0.075
        full = z['depth_full'].astype(np.float32); surf = z['surface']; fin = np.isfinite(full)
        ff = np.where(fin, full, 50.0); jump = np.zeros(full.shape, bool)
        for ax in (0, 1):
            r = np.abs(np.diff(ff, axis=ax)) / np.minimum(ff[:-1] if ax == 0 else ff[:, :-1], ff[1:] if ax == 0 else ff[:, 1:])
            j = r > 0.15
            if ax == 0:
                jump[:-1] |= j; jump[1:] |= j
            else:
                jump[:, :-1] |= j; jump[:, 1:] |= j
        near_edge = cv2.dilate(jump.view(np.uint8), np.ones((2 * edge_px + 1,) * 2, np.uint8)).astype(bool)
        pose = (z['R_nc'], z['cam_ned'])
        d0 = None
        for c in configs:
            cf = parse(c)
            if cf.get('temporal'):
                continue
            d, _ = Chain(cf, K, B, fence)(L, R, pose, pose, 0.0)
            if c == 'base':
                d0 = d
                fx, fy, cx, cy = K
                v, u = np.mgrid[0:full.shape[0], 0:full.shape[1]]
                pc = np.stack([(u - cx) / fx * d0, (v - cy) / fy * d0, d0], -1)
                P = pc @ np.asarray(z['R_nc']).T + np.asarray(z['cam_ned'])
                mapped = (d0 > 0) & (np.linalg.norm(pc, axis=-1) > 0.40) & (-P[..., 2] > 0.10)
                g = np.where(fin, full, 0)
                sel = dict(free=mapped & fin & (d0 < 2.0) & (d0 < 0.8 * g),
                           real=mapped & fin & np.isin(surf, E.SOLID) & (np.abs(d0 - g) <= np.maximum(0.05, 0.05 * g)))
                sel['edge'] = sel['real'] & near_edge
            rm = (d0 > 0) & ~(d > 0)
            for kk, m in sel.items():
                acc[c][kk][0] += int(m.sum()); acc[c][kk][1] += int((m & rm).sum())
    return {c: {k: round(v[1] / max(v[0], 1), 4) for k, v in a.items()} for c, a in acc.items() if not parse(c).get('temporal')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', nargs='+', default=[]); ap.add_argument('--flights')
    ap.add_argument('--replica', help='venue-replica frames dir (single-frame configs only)')
    ap.add_argument('--configs', nargs='+', default=['base']); ap.add_argument('--json')
    a = ap.parse_args()
    if 'base' not in a.configs:
        a.configs = ['base'] + a.configs
    files = [f for p in a.events for f in (sorted(glob.glob(str(Path(p) / '*.npz'))) if Path(p).is_dir() else [p])]
    t0 = time.time()
    out = dict(configs=a.configs)
    if files:
        rows, pix, cost = run(files, a.flights, a.configs)
        out.update(summary=summarize(rows, pix, cost, a.configs), events=rows, pix_counts=pix,
                   cost_ms={c: [round(1e3 * x, 3) for x in v] for c, v in cost.items()})
        print(json.dumps(out['summary'], indent=1))
    if a.replica:
        out['replica'] = replica(a.replica, a.configs)
        print(json.dumps(out['replica'], indent=1))
    out['wall_s'] = round(time.time() - t0, 1)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
