#!/usr/bin/env python3
"""Offline evaluation of the repetitive-texture (net) mask on official-compat flight frames (no ROS needed).

Frames: netmask_extract.py npz (noisy OAK-like left/right as published by the compat stack + truth camera pose in
layout NED). Truth: every valid depth pixel (stride 2) is back-projected with the truth pose and labelled against the
flight's own layout (FLIGHT/sim/asflown/layout.py, the same classes as ubuntu_compat/analyze_obstacles.py):
  net     <= NET_TOL of a cage wall / roof plane, or outside the cage (the net, the hall behind it)
  floor   height < FLOOR_H
  real    <= TOL of a course solid (boards, window board, blocks, rocks, tunnel shell)
  struct  TOL..STRUCT_M of a solid: the boards' unmodelled wooden A-frames (end posts, braces +-0.18 m, foot rails
          +-0.30 m) -> excluded from every count
  phantom everything else (free space, > STRUCT_M from every solid)
`real_end`: real points within END_M of a board's vertical end or of the window hole (edges that must survive).
Metrics per pipeline x mask: phantom points with range < 2 m (removed share), real points lost, real_end lost,
net points lost (the net itself at its true range).

Usage: python3 netmask_eval.py --frames DIR_or_npz... --flights FLIGHTS_ROOT [--pipes compat half320] [--json out]
"""
import argparse, glob, json, math, sys, time, types
from pathlib import Path
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
OFF = HERE.parents[1]
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(OFF / 'ubuntu_compat'))
sys.modules.setdefault('analyze_flight', types.SimpleNamespace(read=None, stamp=None))
import course_solids as CS                      # noqa: E402
from analyze_obstacles import solids_dist        # noqa: E402
import oak_stereo_model as SM                    # noqa: E402
from stereo_depth_node import DepthPipeline, OPTIONS   # noqa: E402

TOL, NET_TOL, FLOOR_H, END_M, STRUCT_M, STRIDE, NEAR_M = 0.12, 0.20, 0.08, 0.15, 0.32, 2, 2.0
CLASSES = ('phantom', 'real', 'real_end', 'net', 'floor', 'struct')


def ends_mask(L, a, b, h):
    """real points near a board's vertical end (|along| >= half_len - END_M) or the window hole rim"""
    m = np.zeros(a.shape, bool)
    for name, kind, q in CS.solids(L):
        if kind != 'obox':
            continue
        ca, cb, hl, hp, yaw, h0, h1 = q; ya = math.radians(yaw); da, db = a - ca, b - cb
        al = da * math.cos(ya) + db * math.sin(ya); pe = -da * math.sin(ya) + db * math.cos(ya)
        near = (np.abs(pe) <= hp + TOL) & (np.abs(al) <= hl + TOL) & (h >= h0 - TOL) & (h <= h1 + TOL)
        if name.startswith('window'):
            m |= near                                          # the window board's pieces all border the hole
        else:
            m |= near & (np.abs(al) >= hl - END_M)
    return m


def label(L, depth, K, R_nc, cam, stride=STRIDE, with_ray=True):
    """-> dict class -> bool mask over the strided grid, plus range z (strided)"""
    fx, fy, cx, cy = K
    z = depth[::stride, ::stride].astype(np.float32)
    H, W = z.shape
    v, u = np.mgrid[0:H, 0:W] * stride
    ok = np.isfinite(z) & (z > 0)
    zz = np.where(ok, z, 0)
    pc = np.stack([(u - cx) / fx * zz, (v - cy) / fy * zz, zz], -1)
    P = pc @ np.asarray(R_nc).T + np.asarray(cam)
    a = P[..., 0] + L.START[0]; b = P[..., 1] + L.START[1]; h = -P[..., 2]
    dn = np.minimum.reduce([a, L.CAGE - a, b, L.CAGE - b, L.CAGE_H - h])
    net = ok & (dn < NET_TOL)
    floor = ok & ~net & (h < FLOOR_H)
    rest = ok & ~net & ~floor
    sd = np.full(z.shape, np.inf)
    sd[rest] = solids_dist(L, a[rest], b[rest], h[rest])
    real = rest & (sd <= TOL)
    ph = rest & (sd > STRUCT_M)
    ph_net = np.zeros(z.shape, bool)
    if with_ray and ph.any():
        ph_net[ph] = behind_is_net(L, P[ph], np.asarray(cam, float))
    return dict(phantom=ph, real=real, real_end=real & ends_mask(L, a, b, h), net=net, floor=floor,
                struct=rest & ~real & ~ph, ph_net=ph_net, z=z, ok=ok, h=h, r3=np.linalg.norm(pc, axis=-1))


def behind_is_net(L, P0, cam, step=0.05, tol=0.06):
    """For free-space points P0 (N,3 layout NED): True if the line of sight beyond the point reaches the cage
    boundary (net / hall) without passing within `tol` of a course solid = a phantom in front of the net."""
    if not len(P0):
        return np.zeros(0, bool)
    u = P0 - cam; u /= np.linalg.norm(u, axis=1, keepdims=True)
    hit = np.zeros(len(P0), bool); alive = np.ones(len(P0), bool)
    p = P0.copy()
    for _ in range(int(9.0 / step)):
        p[alive] += u[alive] * step
        a = p[:, 0] + L.START[0]; b = p[:, 1] + L.START[1]; h = -p[:, 2]
        out = alive & ((a < 0) | (a > L.CAGE) | (b < 0) | (b > L.CAGE) | (h > L.CAGE_H) | (h < 0))
        hit |= out & (h >= 0); alive &= ~out
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        sd = solids_dist(L, a[idx], b[idx], h[idx])
        alive[idx[sd < tol]] = False
    return hit


def pipe_compat(K, B):
    def run(L_, R_):
        d, _, _ = SM.sgbm_depth(L_, R_, K[0], baseline=B)
        return np.nan_to_num(d, nan=0.0).astype(np.float32), None
    return run


def pipe_node(preset, K, B, shape, **over):
    o = dict(OPTIONS[preset]); o.update(over)
    Km = np.array([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1.0]])
    p = DepthPipeline(o, Km, np.zeros(5), Km, np.zeros(5), (shape[1], shape[0]), np.eye(3), np.array([B, 0, 0]),
                      rectified_input=True)
    def run(L_, R_):
        d, _, tim = p.run(L_, R_)
        return d, tim
    run.pipe = p
    return run


def load_frames(paths):
    out = []
    for pth in paths:
        for f in (sorted(glob.glob(str(Path(pth) / '*.npz'))) if Path(pth).is_dir() else [pth]):
            z = np.load(f, allow_pickle=True)
            out.append(dict(file=f, flight=str(z['flight']), left=z['left'], right=z['right'], depth=z['depth'],
                            R_nc=z['R_nc'], cam=z['cam_ned'], K=tuple(float(x) for x in z['K']),
                            B=float(z['baseline']), stamp=z['stamp'],
                            R_vio=z['R_nc_vio'] if 'R_nc_vio' in z.files else None,
                            cam_vio=z['cam_vio'] if 'cam_vio' in z.files else None,
                            fence=json.loads(str(z['prior']))['fence'] if 'prior' in z.files and str(z['prior']) else None))
    return out


# ----------------------------------------------------------------------------------------------- evaluation
from net_mask import NetMask   # noqa: E402
from net_model import NetModel   # noqa: E402

CFG = dict(method='texture', mask_kw={}, model_kw={})


class Runner:
    """one pipeline (compat = the sim's 640 SGBM, half320 = the real-drone node) -> (d0 unfiltered, d1 filtered, ms).
    method: texture (net_mask.NetMask), model (net_model.NetModel: fence + pose), both."""
    def __init__(self, pn, K, B, shape, fence):
        m = CFG['method']
        self.pn, self.K, self.B = pn, K, B
        self.nm = NetMask(**CFG['mask_kw']) if m in ('texture', 'both') else None
        self.md = NetModel(fence, mask_kw=CFG['mask_kw'], **CFG['model_kw']) if m in ('model', 'both') and fence else None
        if pn == 'half320':
            self.p0 = pipe_node('half320', K, B, shape)
            Km = np.array([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1.0]])
            self.p1 = DepthPipeline(dict(OPTIONS['half320']), Km, np.zeros(5), Km, np.zeros(5), (shape[1], shape[0]),
                                    np.eye(3), np.array([B, 0, 0]), rectified_input=True, net_mask=self.nm,
                                    net_model=self.md)
        self.info = []

    def __call__(self, L_, R_, pose):
        if self.pn == 'compat':
            d0, _ = pipe_compat(self.K, self.B)(L_, R_)
            t0 = time.perf_counter()
            disp = np.where(d0 > 0, np.float32(self.K[0] * self.B) / np.maximum(d0, 1e-6), 0).astype(np.float32)
            bad = np.zeros(d0.shape, bool)
            if self.nm is not None:
                bad |= self.nm(L_, disp, R_)
            if self.md is not None and pose is not None:
                b2, info = self.md(L_, np.where(bad, 0, disp).astype(np.float32), R_, self.K, self.B, pose[0], pose[1])
                bad |= b2; self.info.append(info)
            d1 = d0.copy(); d1[bad] = 0.0
            return d0, d1, time.perf_counter() - t0
        d0 = self.p0(L_, R_)[0]
        d1, _, tim = self.p1.run(L_, R_, pose)
        if self.md is not None:
            self.info.append(self.p1.model_info)
        return d0, d1, tim['mask']


def frame_pose(fr, i):
    """the pose the node would use: the flight's VIO if extracted, else truth"""
    if fr.get('R_vio') is not None:
        return (fr['R_vio'][i], fr['cam_vio'][i])
    return (fr['R_nc'][i], fr['cam'][i])

GROUPS = ('ph_net', 'ph_other', 'real', 'real_end', 'net')


def frame_counts(lab, removed):
    """mapped points (what ObstacleMap.insert keeps: range > 0.40 m, > 0.10 m above the floor), per class:
    [total, removed]; phantoms only within NEAR_M."""
    mapped = lab['ok'] & (lab['r3'] > 0.40) & (lab['h'] > 0.10)
    near = mapped & (lab['r3'] < NEAR_M)
    sel = dict(ph_net=lab['phantom'] & lab['ph_net'] & near, ph_other=lab['phantom'] & ~lab['ph_net'] & near,
               real=lab['real'] & mapped, real_end=lab['real_end'] & mapped, net=lab['net'] & mapped)
    return {k: np.array([int(m.sum()), int((m & removed).sum())]) for k, m in sel.items()}


def map_phantoms(mp, L, cam):
    """confirmed ObstacleMap points -> number of phantom points (free space, > STRUCT_M from solids, not net/floor)
    within NEAR_M of the camera"""
    P = mp.points
    if not len(P):
        return (0, 0)
    a = P[:, 0] + L.START[0]; b = P[:, 1] + L.START[1]; h = -P[:, 2]
    dn = np.minimum.reduce([a, L.CAGE - a, b, L.CAGE - b, L.CAGE_H - h])
    ph = (dn >= NET_TOL) & (h >= FLOOR_H) & (solids_dist(L, a, b, h) > STRUCT_M) & (np.linalg.norm(P - cam, axis=1) < NEAR_M)
    if not ph.any():
        return (0, 0)
    pn = behind_is_net(L, P[ph], np.asarray(cam, float))
    return (int(pn.sum()), int((~pn).sum()))


def eval_flights(frames, flights_root, every=2, mask_kw=None, pipes=('compat', 'half320'), with_map=True):
    sys.path.insert(0, str(OFF))
    from node.obstacle_map import ObstacleMap
    from common.geometry import Frame
    res = {}
    for pn in pipes:
        res[pn] = dict(tot={k: np.zeros(2, int) for k in GROUPS}, flights={}, ms_mask=[], map=dict(off=[], on=[]))
    for fr in frames:
        L = CS.load_layout(Path(flights_root) / fr['flight'])
        K, B = fr['K'], fr['B']
        runs = {pn: Runner(pn, K, B, fr['left'][0].shape, fr.get('fence')) for pn in pipes}
        maps = {pn: (ObstacleMap(), ObstacleMap()) for pn in pipes}
        for i in range(0, len(fr['left']), every):
            Lf, Rf = fr['left'][i], fr['right'][i]
            for pn in pipes:
                d0, d1, tm = runs[pn](Lf, Rf, frame_pose(fr, i))
                lab = label(L, d0, K, fr['R_nc'][i], fr['cam'][i])
                removed = ((d0 > 0) & ~(d1 > 0))[::STRIDE, ::STRIDE]
                c = frame_counts(lab, removed)
                r = res[pn]; r['ms_mask'].append(1e3 * tm)
                ff = r['flights'].setdefault(fr['flight'], {k: np.zeros(2, int) for k in GROUPS})
                for k, v in c.items():
                    r['tot'][k] += v; ff[k] += v
                if with_map:
                    for mk, dd in (('off', d0), ('on', d1)):
                        mp = maps[pn][0 if mk == 'off' else 1]
                        mp.insert(Frame(image=None, depth=dd, K=K, R_nc=fr['R_nc'][i], cam_ned=fr['cam'][i]), 0.0,
                                  float(fr['stamp'][i]))
                        r['map'][mk].append(map_phantoms(mp, L, fr['cam'][i]))
        print(f"  {fr['flight']}: " + ' | '.join(f"{pn} net-ph {res[pn]['flights'][fr['flight']]['ph_net'][1]}/"
                                                 f"{res[pn]['flights'][fr['flight']]['ph_net'][0]}" for pn in pipes), flush=True)
    out = {}
    for pn, r in res.items():
        frac = lambda v: round(float(v[1]) / max(int(v[0]), 1), 4)
        o = dict(points={k: [int(v[0]), int(v[1])] for k, v in r['tot'].items()},
                 removed={k: frac(v) for k, v in r['tot'].items()},
                 per_flight={f: {k: frac(v) for k, v in d.items() if k in ('ph_net', 'real')} for f, d in r['flights'].items()},
                 ms_mask_mean=round(float(np.mean(r['ms_mask'])), 2), ms_mask_p90=round(float(np.percentile(r['ms_mask'], 90)), 2))
        if with_map and r['map']['off']:
            o['map'] = dict(frames=len(r['map']['off']), note='confirmed ObstacleMap points (frames at 2.5 Hz, '
                            'min_hits 3, memory 4 s) that are phantoms within 2 m of the camera; net = in front of the net')
            for j, kind in ((0, 'net'), (1, 'other')):
                off = np.array([x[j] for x in r['map']['off']]); on = np.array([x[j] for x in r['map']['on']])
                o['map'][kind] = dict(pts_per_frame_off=round(float(off.mean()), 2), pts_per_frame_on=round(float(on.mean()), 2),
                                      frames_with_any_off=round(float((off > 0).mean()), 3),
                                      frames_with_any_on=round(float((on > 0).mean()), 3))
        out[pn] = o
    return out


def pipe_node_masked(preset, K, B, shape, nm):
    o = dict(OPTIONS[preset])
    Km = np.array([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1.0]])
    p = DepthPipeline(o, Km, np.zeros(5), Km, np.zeros(5), (shape[1], shape[0]), np.eye(3), np.array([B, 0, 0]),
                      rectified_input=True, net_mask=nm)
    def run(L_, R_):
        d, _, tim = p.run(L_, R_)
        return d, tim
    run.pipe = p
    return run


SURF = ['other', 'floor', 'banner_panel', 'brass_ring', 'tunnel_fabric', 'black_mesh', 'cage_net', 'tan_board',
        'landing_pad', 'cinder_block', 'rock', 'hall', 'floor_tape']
SOLID = [SURF.index(n) for n in ('banner_panel', 'brass_ring', 'tunnel_fabric', 'tan_board', 'cinder_block', 'rock')]
NETS = [SURF.index(n) for n in ('cage_net', 'hall')]


def replica_eval(replica_dir, mask_kw=None, pipes=('compat', 'half320'), edge_px=6):
    """venue-replica frames (per-pixel truth): per frame group, mapped points (range > 0.40 m, > 0.10 m above floor):
    net phantoms (< 2 m, < 0.8x the nearest true surface, true surface = cage net / hall), real (on a solid, within 5 %),
    edge = real within `edge_px` px of a true depth step > 15 % (board ends, window hole, gate posts, hoops)."""
    files = [f for f in sorted(glob.glob(str(Path(replica_dir) / '*.npz'))) if not Path(f).name.startswith('compare')]
    acc = {}
    runs = None
    cage = json.loads((Path(replica_dir).parent / 'index.json').read_text())['world_labels']['cage']
    fence = dict(forward_min=cage['n_range'][0], forward_max=cage['n_range'][1], left_min=-cage['e_range'][1],
                 left_max=-cage['e_range'][0], up_max=cage['height'])
    for f in files:
        z = np.load(f, allow_pickle=True)
        grp = Path(f).name.split('_')[0]
        Lf, Rf = z['image'], z['image_right']
        k = np.asarray(z['K'], float).ravel(); K = (k[0], k[1], k[2], k[3]); B = 0.075
        if runs is None:
            runs = {pn: Runner(pn, K, B, Lf.shape, fence) for pn in pipes}
        full = z['depth_full'].astype(np.float32); surf = z['surface']
        jump = np.zeros(full.shape, bool)
        ff = np.where(np.isfinite(full), full, 50.0)
        for ax in (0, 1):
            r = np.abs(np.diff(ff, axis=ax)) / np.minimum(ff[:-1] if ax == 0 else ff[:, :-1], ff[1:] if ax == 0 else ff[:, 1:])
            j = r > 0.15
            if ax == 0:
                jump[:-1] |= j; jump[1:] |= j
            else:
                jump[:, :-1] |= j; jump[:, 1:] |= j
        near_edge = cv2.dilate(jump.view(np.uint8), np.ones((2 * edge_px + 1, 2 * edge_px + 1), np.uint8)).astype(bool)
        fx, fy, cx, cy = K
        v, u = np.mgrid[0:full.shape[0], 0:full.shape[1]]
        for pn in pipes:
            d0, d1, _ = runs[pn](Lf, Rf, (z['R_nc'], z['cam_ned']))
            ok = d0 > 0
            pc = np.stack([(u - cx) / fx * d0, (v - cy) / fy * d0, d0], -1)
            P = pc @ np.asarray(z['R_nc']).T + np.asarray(z['cam_ned'])
            mapped = ok & (np.linalg.norm(pc, axis=-1) > 0.40) & (-P[..., 2] > 0.10)
            fin = np.isfinite(full)
            ph = mapped & fin & (d0 < NEAR_M) & (d0 < 0.8 * np.where(fin, full, 0)) & np.isin(surf, NETS)
            real = mapped & fin & np.isin(surf, SOLID) & (np.abs(d0 - np.where(fin, full, 0)) <= np.maximum(0.05, 0.05 * np.where(fin, full, 0)))
            rm = ok & ~(d1 > 0)
            a = acc.setdefault(pn, {}).setdefault(grp, {c: [0, 0] for c in ('ph_net', 'real', 'edge')})
            for c, m in (('ph_net', ph), ('real', real), ('edge', real & near_edge)):
                a[c][0] += int(m.sum()); a[c][1] += int((m & rm).sum())
    out = {}
    for pn, g in acc.items():
        tot = {c: [sum(v[c][0] for v in g.values()), sum(v[c][1] for v in g.values())] for c in ('ph_net', 'real', 'edge')}
        g = dict(g, ALL=tot)
        out[pn] = {grp: {c: dict(n=v[0], removed=round(v[1] / max(v[0], 1), 4)) for c, v in d.items()} for grp, d in g.items()}
    return dict(frames=len(files), groups=out)


def event_eval(paths, flights_root, mask_kw=None, pipes=('compat', 'half320'), before=4.5, radius=0.35):
    """Full-rate frames around each mission `small_cluster_kept` event (netmask_extract.py --events): replay the
    node's ObstacleMap over [t - before, t] with and without the mask; at the event time count the confirmed phantom
    points (free space, in front of the net or not) within `radius` of the cluster the mission kept (centre in PX4 NED;
    layout d = PX4 d - 0.05) and within 2 m of the camera."""
    sys.path.insert(0, str(OFF))
    from node.obstacle_map import ObstacleMap
    from common.geometry import Frame
    rows = []
    for f in paths:
        z = np.load(f, allow_pickle=True)
        flight = str(z['flight']); events = json.loads(str(z['events']))
        if not events:
            continue
        L = CS.load_layout(Path(flights_root) / flight)
        K = tuple(float(x) for x in z['K']); B = float(z['baseline']); st = z['stamp']
        fence = json.loads(str(z['prior']))['fence'] if 'prior' in z.files and str(z['prior']) else None
        runs = {pn: Runner(pn, K, B, z['left'][0].shape, fence) for pn in pipes}
        has_vio = 'R_nc_vio' in z.files
        cache = {}
        for ev in events:
            idx = [i for i in range(len(st)) if ev['t'] - before <= st[i] <= ev['t'] + 1e-3]
            if len(idx) < 10:
                continue
            c_ev = np.asarray(ev['center'], float) + np.array([0, 0, -0.05])
            row = dict(flight=flight, t=round(ev['t'], 2), n_mission=ev['n'], frames=len(idx))
            for pn in pipes:
                maps = (ObstacleMap(), ObstacleMap())
                for i in idx:
                    key = (pn, i)
                    if key not in cache:
                        pose = (z['R_nc_vio'][i], z['cam_vio'][i]) if has_vio else (z['R_nc'][i], z['cam_ned'][i])
                        d0, d1, _ = runs[pn](z['left'][i], z['right'][i], pose)
                        cache[key] = (d0, d1)
                    for mp, dd in zip(maps, cache[key]):
                        mp.insert(Frame(image=None, depth=dd, K=K, R_nc=z['R_nc'][i], cam_ned=z['cam_ned'][i]), 0.0, float(st[i]))
                cam = z['cam_ned'][idx[-1]]
                for tag, mp in zip(('off', 'on'), maps):
                    P = mp.points
                    if not len(P):
                        row[f'{pn}_{tag}_at_event'] = 0; row[f'{pn}_{tag}_net2m'] = 0; continue
                    a = P[:, 0] + L.START[0]; b = P[:, 1] + L.START[1]; h = -P[:, 2]
                    dn = np.minimum.reduce([a, L.CAGE - a, b, L.CAGE - b, L.CAGE_H - h])
                    ph = (dn >= NET_TOL) & (h >= FLOOR_H) & (solids_dist(L, a, b, h) > STRUCT_M)
                    row[f'{pn}_{tag}_at_event'] = int((ph & (np.linalg.norm(P - c_ev, axis=1) < radius)).sum())
                    row[f'{pn}_{tag}_ph2m'] = int((ph & (np.linalg.norm(P - cam, axis=1) < NEAR_M)).sum())
            rows.append(row)
            print('  ', row, flush=True)
    summ = {}
    for pn in pipes:
        rep = [r for r in rows if r.get(f'{pn}_off_at_event', 0) > 0]
        summ[pn] = dict(events=len(rows), reproduced=len(rep),
                        cleared_by_mask=sum(1 for r in rep if r[f'{pn}_on_at_event'] == 0),
                        pts_at_event_off=sum(r[f'{pn}_off_at_event'] for r in rows),
                        pts_at_event_on=sum(r[f'{pn}_on_at_event'] for r in rows),
                        events_with_phantom_within_2m_off=sum(1 for r in rows if r.get(f'{pn}_off_ph2m', 0) > 0),
                        events_with_phantom_within_2m_on=sum(1 for r in rows if r.get(f'{pn}_on_ph2m', 0) > 0),
                        ph2m_pts_off=sum(r.get(f'{pn}_off_ph2m', 0) for r in rows), ph2m_pts_on=sum(r.get(f'{pn}_on_ph2m', 0) for r in rows))
    return dict(summary=summ, events=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', nargs='+', default=[]); ap.add_argument('--flights')
    ap.add_argument('--replica', help='venue-replica frames dir: edge / window / gate check with per-pixel truth')
    ap.add_argument('--events', nargs='+', default=[], help='netmask_extract.py --events npz files (full-rate windows)')
    ap.add_argument('--every', type=int, default=2); ap.add_argument('--pipes', nargs='+', default=['compat', 'half320'])
    ap.add_argument('--score-min', type=float); ap.add_argument('--d-margin', type=float); ap.add_argument('--pfrac', type=float)
    ap.add_argument('--far-win', type=int, nargs=2); ap.add_argument('--score-dilate', type=int, nargs=2)
    ap.add_argument('--norm', choices=('geo', 'max')); ap.add_argument('--row-step', type=int)
    ap.add_argument('--score-erode', type=int, nargs=2); ap.add_argument('--lag-thin', type=int)
    ap.add_argument('--win', type=int, nargs=2); ap.add_argument('--cost-ratio', type=float, help='0 = no cost check')
    ap.add_argument('--cost-floor', type=float)
    ap.add_argument('--method', choices=('texture', 'model', 'both'), default='texture')
    ap.add_argument('--model', nargs='*', default=[], help='NetModel overrides key=value (json values)')
    ap.add_argument('--no-map', action='store_true'); ap.add_argument('--json')
    a = ap.parse_args()
    kw = {k: v for k, v in (('score_min', a.score_min), ('d_margin', a.d_margin), ('pfrac', a.pfrac),
                            ('far_win', a.far_win), ('score_dilate', a.score_dilate), ('norm', a.norm),
                            ('row_step', a.row_step), ('score_erode', a.score_erode),
                            ('lag_thin', None if a.lag_thin is None else bool(a.lag_thin)), ('win', a.win),
                            ('cost_ratio', a.cost_ratio), ('cost_floor', a.cost_floor)) if v is not None}
    t0 = time.time()
    CFG.update(method=a.method, mask_kw=kw, model_kw={k: json.loads(v) for k, v in (x.split('=', 1) for x in a.model)})
    out = dict(method=a.method, mask=NetMask(**kw).params() if a.method != 'model' else None,
               model=NetModel(dict(forward_min=-1, forward_max=1, left_min=-1, left_max=1), **CFG['model_kw']).params()
               if a.method != 'texture' else None, every=a.every)
    if a.frames:
        out['eval'] = eval_flights(load_frames(a.frames), a.flights, a.every, kw, a.pipes, not a.no_map)
    if a.replica:
        out['replica'] = replica_eval(a.replica, kw, a.pipes)
    if a.events:
        files = [f for pth in a.events for f in (sorted(glob.glob(str(Path(pth) / '*.npz'))) if Path(pth).is_dir() else [pth])]
        out['events'] = event_eval(files, a.flights, kw, a.pipes)
    out['wall_s'] = round(time.time() - t0, 1)
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
