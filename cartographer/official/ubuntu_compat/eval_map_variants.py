#!/usr/bin/env python3
"""Offline what-if for ObstacleMap on recorded flights (sim-side evidence; node code untouched).

  python3 eval_map_variants.py [--code ROOT] [--out JSON] FLIGHT_DIR [FLIGHT_DIR ...]   -> printed table + JSON
  --code ROOT: take node/obstacle_map.py + common/ from ROOT (e.g. the profile with the NEW map) instead of each
  flight's code.tar.gz; then the variants are base (carve=False), mycarve (this file's Carve on carve=False),
  new_h3 / new_h4 (the new ObstacleMap's own carving, min_hits 3 / 4).

Replays every recorded depth frame (same pose rule as the node) through:
  base      ObstacleMap() as flown (min_hits 3, memory 4 s, max_range 4 m)
  hits6     min_hits 6
  carve     base + free-space clearing: a confirmed voxel that projects into the current depth image with a
            valid return >= 0.25 m BEHIND it (the camera sees through it) loses 2 hits; deleted at 0
  carve6    carve + min_hits 6
  cluster   base, but the probe only counts voxels that have >= 8 confirmed neighbours within 0.15 m
            (a planner-side 'ignore tiny clusters' rule)
Metrics per variant: median phantom voxels in the map, share of 0.5 s samples with a phantom inside the
heading probe (1.7 m x 0.35 m), phantoms in the probe at each recorded avoid/blocked/abort event, and the
median number of REAL voxels (recall vs base).
"""
import json, math, sys, tarfile, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_flight import read, stamp  # noqa: E402
from analyze_obstacles import solids_dist, TOL, NET_TOL, R_SOFT  # noqa: E402
from probe_events import seg_dist, PROBE_M  # noqa: E402


def make_variants(ObstacleMap, cKDTree):
    class Carve(ObstacleMap):
        def insert(self, frame, floor_d, stamp, stride=4):
            super().insert(frame, floor_d, stamp, stride)
            if not self.vox:
                return
            keys = list(self.vox.keys()); V = np.array([self.vox[k][2:5] for k in keys])
            uv, z = frame.project(V)
            H, W = frame.depth.shape
            ok = (z > self.min_range) & (z < self.max_range) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            idx = np.where(ok)[0]
            if len(idx):
                d = frame.depth[uv[idx, 1].astype(int), uv[idx, 0].astype(int)]
                through = (d > 0) & np.isfinite(d) & (d > z[idx] + 0.25)
                for i in idx[through]:
                    v = self.vox[keys[i]]; v[0] -= 2
                    if v[0] <= 0:
                        del self.vox[keys[i]]
            conf = [v[2:] for v in self.vox.values() if v[0] >= self.min_hits and stamp - v[1] <= self.memory]
            allp = np.asarray(conf, float).reshape(-1, 3)
            self.points = allp; self.tree = cKDTree(allp) if len(allp) else None
    return {'base': lambda: ObstacleMap(), 'hits6': lambda: ObstacleMap(min_hits=6),
            'carve': lambda: Carve(), 'carve6': lambda: Carve(min_hits=6), 'cluster': lambda: ObstacleMap()}


def run(F, L, code=None):
    if code is None:
        tmp = Path(tempfile.mkdtemp(prefix='mv_'))
        with tarfile.open(F / 'code.tar.gz') as tf:
            tf.extractall(tmp)
    else:
        tmp = Path(code)
    sys.path.insert(0, str(tmp))
    for m in [m for m in list(sys.modules) if m.startswith(('node', 'common'))]:
        del sys.modules[m]
    from node.obstacle_map import ObstacleMap
    from common.geometry import Frame, camera_pose_ned
    from scipy.spatial import cKDTree
    sys.path.remove(str(tmp))
    P = json.loads((HERE / 'profile.json').read_text())
    c = P['camera']; K = tuple(c['K']); R_bc = np.asarray(c['R_base_optical'], float); t_bc = tuple(c['left_in_base_m'])
    ev = []
    for ln in (F / 'mission.log').read_text().splitlines():
        if ln.startswith('{') and '"event"' in ln:
            e = json.loads(ln.replace('NaN', 'null'))
            if e['event'].get('kind') in ('avoid', 'blocked', 'abort'):
                ev.append((e['t'], e['event']['kind']))
    A = read(F / 'bag', ['/camera/depth/image_raw', '/fmu/out/vehicle_local_position_v1', '/fmu/out/vehicle_attitude'])
    lp = A['/fmu/out/vehicle_local_position_v1']; att = A['/fmu/out/vehicle_attitude']
    lp_t = np.array([r for r, _ in lp]); at_t = np.array([r for r, _ in att])
    floor_d = float(lp[0][1].z) + 0.05
    if code is None:
        V = make_variants(ObstacleMap, cKDTree)
    else:
        import inspect
        assert 'carve' in inspect.signature(ObstacleMap.__init__).parameters, 'new ObstacleMap expected'
        class Off(ObstacleMap):
            def __init__(self, **kw):
                super().__init__(carve=False, **kw)
        Mine = make_variants(Off, cKDTree)['carve']
        V = {'base': lambda: ObstacleMap(carve=False), 'mycarve': Mine,
             'new_h3': lambda: ObstacleMap(min_hits=3), 'new_h4': lambda: ObstacleMap(min_hits=4)}
    maps = {k: f() for k, f in V.items()}
    stats = {k: dict(ph=[], real=[], probe_hit=0, n=0, ev=[]) for k in maps}
    next_t, ek = None, 0
    for rt, m in A['/camera/depth/image_raw']:
        i = int(np.searchsorted(lp_t, rt)) - 1; j = int(np.searchsorted(at_t, rt)) - 1
        if i < 0 or j < 0:
            continue
        lpm, atm = lp[i][1], att[j][1]
        d = np.frombuffer(bytes(m.data), np.float32).reshape(m.height, -1)[:, :m.width]
        pos = np.array([lpm.x, lpm.y, lpm.z]); R_nc, cam = camera_pose_ned(tuple(float(x) for x in atm.q), pos, R_bc, t_bc)
        s = stamp(m); fr = Frame(image=None, depth=d, K=K, R_nc=R_nc, cam_ned=cam)
        for mp in maps.values():
            mp.insert(fr, floor_d, s)
        at_event = []
        while ek < len(ev) and s >= ev[ek][0]:
            at_event.append(ev[ek]); ek += 1
        sample = next_t is None or s >= next_t
        if not (sample or at_event):
            continue
        if sample:
            next_t = s + 0.5
        hd = float(lpm.heading); u = np.array([math.cos(hd), math.sin(hd), 0.0])
        for k, mp in maps.items():
            pts = mp.points
            if not len(pts):
                if sample:
                    stats[k]['ph'].append(0); stats[k]['real'].append(0); stats[k]['n'] += 1
                for e in at_event:
                    stats[k]['ev'].append((e[0], e[1], 0))
                continue
            aa = pts[:, 0] + L.START[0]; bb = pts[:, 1] + L.START[1]; hh = -(pts[:, 2] - 0.05)
            dn = np.minimum.reduce([aa, L.CAGE - aa, bb, L.CAGE - bb, L.CAGE_H - hh])
            net = dn < NET_TOL; real = (~net) & (solids_dist(L, aa, bb, hh) <= TOL); ph = ~(net | real)
            dist, _ = seg_dist(pts, pos, pos + PROBE_M * u); inn = dist < R_SOFT
            if k == 'cluster' and inn.any():
                nb = mp.tree.query_ball_point(pts[inn], 0.15, return_length=True)
                keep = np.zeros(len(pts), bool); keep[np.where(inn)[0][nb >= 9]] = True; inn = keep
            n_ph = int((inn & ph).sum())
            if sample:
                st = stats[k]; st['ph'].append(int(ph.sum())); st['real'].append(int(real.sum())); st['n'] += 1
                st['probe_hit'] += int(n_ph > 0)
            for e in at_event:
                stats[k]['ev'].append((e[0], e[1], n_ph))
    out = {}
    for k, st in stats.items():
        out[k] = dict(phantom_med=int(np.median(st['ph'])) if st['ph'] else 0, real_med=int(np.median(st['real'])) if st['real'] else 0,
                      samples_with_phantom_in_probe=f"{st['probe_hit']}/{st['n']}",
                      events=[(round(t, 2), kd, n) for t, kd, n in st['ev']])
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--code'); ap.add_argument('--out', default='/compat/run/map_variants.json')
    ap.add_argument('flights', nargs='+'); a = ap.parse_args()
    sys.path.insert(0, '/replica'); import layout as L
    res = {}
    for f in a.flights:
        F = Path(f); res[F.name] = run(F, L, a.code)
        b = res[F.name]['base']
        print(f'== {F.name}')
        for k, r in res[F.name].items():
            rec = (r['real_med'] / b['real_med']) if b['real_med'] else float('nan')
            evs = ' '.join(f"{t}{kd[0]}:{n}" for t, kd, n in r['events'][:12])
            print(f"  {k:8s} phantom_med {r['phantom_med']:5d}  real_med {r['real_med']:5d} (recall {rec:4.2f})  "
                  f"probe-hit samples {r['samples_with_phantom_in_probe']:>7s}  events(t kind:phantoms in probe) {evs}")
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
