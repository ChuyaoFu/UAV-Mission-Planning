#!/usr/bin/env python3
"""Rebuild the node's obstacle map from a recorded flight and classify every confirmed voxel
against the venue-replica geometry (sim-side truth; flight code never sees layout.py).

  python3 analyze_obstacles.py FLIGHT_DIR [--every 0.5]     (in the container; needs a NON-light bag)
  -> FLIGHT_DIR/sim/obstacles.json, sim/obstacles_timeline.csv, sim/obstacles_topdown.png

Map rebuild = what node/mission_node.py does on every /camera/depth/image_raw: ObstacleMap (from the flight's own
code.tar.gz) .insert(Frame(depth, K, camera_pose_ned(newest PX4 attitude/position, TF base_link<-oak_left)),
floor_d = PX4 z at rest + 0.05, stamp). Map points are in PX4 NED (start-anchored); layout NED = same n/e,
d_layout = d_px4 - 0.05 (base_link rests 5 cm above the floor).
Classes per confirmed voxel: real (<= TOL of a solid: panels, boards, wall, stone, rocks, tunnel shell),
net (<= NET_TOL of a cage wall/roof plane, or outside the cage), phantom (everything else).
Corridor: the planner's soft radius = half_span 0.20 + clearance_margin 0.15 = 0.35 m (3-D, like ObstacleMap.clearance)
around the vehicle's own PX4 path; 'ahead' = within 0.35 m of the path the vehicle flies in the next 3 s.
"""
import argparse, csv, io, json, math, sys, tarfile, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_flight import read, stamp  # noqa: E402

TOL, NET_TOL, R_SOFT, R_HARD, AHEAD_S = 0.12, 0.20, 0.35, 0.25, 3.0


def solids_dist(L, a, b, h):
    """distance (m) of cage-coord points to the nearest replica solid (approx.)"""
    d = np.full(a.shape, np.inf)
    for kind, q in L._solids():
        if kind == 'obox':
            ca, cb, hl, hp, yaw, h0, h1 = q
            ya = math.radians(yaw); da, db = a - ca, b - cb
            al = da * math.cos(ya) + db * math.sin(ya); pe = -da * math.sin(ya) + db * math.cos(ya)
            ex = np.maximum(np.abs(al) - hl, 0); ey = np.maximum(np.abs(pe) - hp, 0)
            ez = np.maximum(np.maximum(h0 - h, h - h1), 0)
            d = np.minimum(d, np.sqrt(ex ** 2 + ey ** 2 + ez ** 2))
        elif kind == 'cyl':
            ca, cb, r, h0, h1 = q
            er = np.maximum(np.hypot(a - ca, b - cb) - r, 0); ez = np.maximum(np.maximum(h0 - h, h - h1), 0)
            d = np.minimum(d, np.hypot(er, ez))
        else:                                              # tunnel: hoops + mesh + fabric shell, open ends
            import course_solids as CS                     # any horizontal axis (layout v3 east, v4 north -> south)
            tg = CS.tunnel_geom(L); (ea, eb0), (ua, ub), (ra, rb) = tg['entry'], tg['axis'], tg['right']
            s_ax = (a - ea) * ua + (b - eb0) * ub; rr = np.hypot((a - ea) * ra + (b - eb0) * rb, h - tg['centre_h'])
            eb = np.maximum(np.maximum(-0.02 - s_ax, s_ax - tg['length'] - 0.02), 0)
            d = np.minimum(d, np.hypot(np.abs(rr - 0.555), eb))
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('flight'); ap.add_argument('--every', type=float, default=0.5)
    a_ = ap.parse_args(); F = Path(a_.flight)
    tmp = Path(tempfile.mkdtemp(prefix='obs_'))
    with tarfile.open(F / 'code.tar.gz') as tf:
        tf.extractall(tmp)
    sys.path.insert(0, str(tmp))
    from node.obstacle_map import ObstacleMap
    from common.geometry import Frame, camera_pose_ned
    import course_solids as _cs; L = _cs.load_layout(F)          # the layout this flight flew (warped worlds too)
    P = json.loads((HERE / 'profile.json').read_text())
    c = P['camera']; K = tuple(c['K']); R_bc = np.asarray(c['R_base_optical'], float); t_bc = tuple(c['left_in_base_m'])
    A = read(F / 'bag', ['/camera/depth/image_raw', '/fmu/out/vehicle_local_position_v1', '/fmu/out/vehicle_attitude',
                         '/fmu/out/vehicle_status_v1', '/fmu/out/vehicle_land_detected'])
    if not A['/camera/depth/image_raw']:
        print('no depth in bag (light flight?)'); return 1
    lp = [(rt, m) for rt, m in A['/fmu/out/vehicle_local_position_v1']]
    att = [(rt, m) for rt, m in A['/fmu/out/vehicle_attitude']]
    lp_t = np.array([r for r, _ in lp]); at_t = np.array([r for r, _ in att])
    st = [(rt, m.arming_state == 2) for rt, m in A['/fmu/out/vehicle_status_v1']]
    mp = ObstacleMap(); floor_d = None
    lp0 = lp[0][1]; floor_d = float(lp0.z) + 0.05                       # node: measured at rest before arming
    # EKF path (sim time = depth stamp domain via receive-time interpolation)
    path_rt = lp_t; path = np.array([[m.x, m.y, m.z] for _, m in lp])
    events = []
    for ln in (F / 'mission.log').read_text().splitlines():
        if ln.startswith('{') and '"event"' in ln:
            e = json.loads(ln.replace('NaN', 'null'))
            ev = e['event']; k = ev.get('kind') or ev.get('type')
            if k in ('avoid', 'blocked', 'abort', 'phase', 'standoff_clipped', 'standoff_shrunk', 'corridor_blocked', 'geofence', 'fence_clip'):
                events.append(dict(t=e['t'], kind=k, **{x: ev.get(x) for x in ('to', 'offset_m', 'axis', 'reason', 'phase') if x in ev}))
    status = []
    for ln in (F / 'mission.log').read_text().splitlines():
        if ln.startswith('{') and '"phase"' in ln and '"event"' not in ln:
            j = json.loads(ln.replace('NaN', 'null')); status.append((j['t'], j['phase'], j.get('map_pts')))
    rows, all_ph, last_cls = [], [], None
    next_t = None; t_first = None; dep = A['/camera/depth/image_raw']
    for rt, m in dep:
        i = int(np.searchsorted(lp_t, rt)) - 1; j = int(np.searchsorted(at_t, rt)) - 1
        if i < 0 or j < 0:
            continue
        lpm, atm = lp[i][1], att[j][1]
        d = np.frombuffer(bytes(m.data), np.float32).reshape(m.height, -1)[:, :m.width]
        pos = np.array([lpm.x, lpm.y, lpm.z]); q = tuple(float(x) for x in atm.q)
        R_nc, cam = camera_pose_ned(q, pos, R_bc, t_bc)
        s = stamp(m); t_first = t_first or s
        mp.insert(Frame(image=None, depth=d, K=K, R_nc=R_nc, cam_ned=cam), floor_d, s)
        if next_t is None or s >= next_t:
            next_t = s + a_.every
            pts = mp.points
            if len(pts):
                aa = pts[:, 0] + L.START[0]; bb = pts[:, 1] + L.START[1]; hh = -(pts[:, 2] - 0.05)
                dn = np.minimum.reduce([aa, L.CAGE - aa, bb, L.CAGE - bb, L.CAGE_H - hh])
                net = dn < NET_TOL
                real = (~net) & (solids_dist(L, aa, bb, hh) <= TOL)
                ph = ~(net | real)
                # corridor: 3-D distance to the vehicle's past+future PX4 path, and to the next AHEAD_S seconds only
                fut = path[(path_rt >= rt) & (path_rt <= rt + AHEAD_S)]
                def mind(P_, Q_):
                    if not len(P_) or not len(Q_):
                        return np.full(len(P_), np.inf)
                    Qs = Q_[::max(1, len(Q_) // 60)]
                    return np.sqrt(((P_[:, None, :] - Qs[None, :, :]) ** 2).sum(-1)).min(1)
                phd_f = mind(pts[ph], fut)
                rows.append(dict(t=round(s, 2), map_pts=int(len(pts)), real=int(real.sum()), net=int(net.sum()), phantom=int(ph.sum()),
                                 phantom_ahead_soft=int((phd_f < R_SOFT).sum()), phantom_ahead_hard=int((phd_f < R_HARD).sum()),
                                 phantom_min_ahead_m=round(float(phd_f.min()), 3) if len(phd_f) else None,
                                 veh=[round(float(x), 2) for x in pos]))
                if ph.any():
                    all_ph.append(np.c_[np.full(ph.sum(), s), pts[ph]])
                last_cls = (pts, net, real, ph)
    PH = np.concatenate(all_ph) if all_ph else np.zeros((0, 4))
    # where the phantoms sit: unique voxels over the whole flight, distance to the whole flown path, near which wall
    res = dict(flight=F.name, frames=len(dep), samples=len(rows))
    if rows:
        mx = max(rows, key=lambda r: r['phantom'])
        res['map_pts_max'] = max(r['map_pts'] for r in rows); res['map_pts_med'] = int(np.median([r['map_pts'] for r in rows]))
        res['phantom_max'] = mx['phantom']; res['phantom_max_t'] = mx['t']
        res['phantom_med'] = int(np.median([r['phantom'] for r in rows]))
        res['phantom_frac_med'] = round(float(np.median([r['phantom'] / max(1, r['map_pts']) for r in rows])), 3)
        res['net_med'] = int(np.median([r['net'] for r in rows]))
        hits = [r for r in rows if r['phantom_ahead_soft'] > 0]
        res['samples_with_phantom_ahead_soft'] = len(hits)
        res['samples_with_phantom_ahead_hard'] = sum(1 for r in rows if r['phantom_ahead_hard'] > 0)
        res['phantom_ahead_soft_times'] = [(r['t'], r['phantom_ahead_soft'], r['phantom_min_ahead_m']) for r in hits][:40]
    if len(PH):
        vox = np.unique(np.floor(PH[:, 1:] / 0.05).astype(np.int64), axis=0) * 0.05 + 0.025
        aa = vox[:, 0] + L.START[0]; bb = vox[:, 1] + L.START[1]; hh = -(vox[:, 2] - 0.05)
        dwall = np.c_[aa, L.CAGE - aa, bb, L.CAGE - bb]
        side = np.array(['S', 'N', 'W', 'E'])[dwall.argmin(1)]
        Qs = path[::max(1, len(path) // 400)]
        dpath = np.sqrt(((vox[:, None, :] - Qs[None, :, :]) ** 2).sum(-1)).min(1)
        res['phantom_unique_voxels'] = int(len(vox))
        res['phantom_nearest_wall_counts'] = {k: int((side == k).sum()) for k in 'SNWE'}
        res['phantom_dist_to_net_m'] = dict(p10=round(float(np.percentile(dwall.min(1), 10)), 2), p50=round(float(np.median(dwall.min(1))), 2))
        res['phantom_height_m'] = dict(p10=round(float(np.percentile(hh, 10)), 2), p50=round(float(np.median(hh)), 2), p90=round(float(np.percentile(hh, 90)), 2))
        res['phantom_dist_to_path_m'] = dict(min=round(float(dpath.min()), 3), p10=round(float(np.percentile(dpath, 10)), 2), p50=round(float(np.median(dpath)), 2))
        res['phantom_voxels_within_soft_of_path'] = int((dpath < R_SOFT).sum())
        res['phantom_voxels_within_hard_of_path'] = int((dpath < R_HARD).sum())
    res['mission_events'] = [e for e in events if e['kind'] != 'phase']
    res['phases'] = [dict(t=e['t'], to=e.get('to')) for e in events if e['kind'] == 'phase']
    res['node_map_pts_med'] = int(np.median([s_[2] for s_ in status if s_[2] is not None])) if status else None
    (F / 'sim').mkdir(exist_ok=True)
    (F / 'sim' / 'obstacles.json').write_text(json.dumps(res, indent=1, default=str))
    with open(F / 'sim' / 'obstacles_timeline.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['t']); w.writeheader(); [w.writerow(r) for r in rows]
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 7))
        C = L.CAGE
        ax.plot([0, C, C, 0, 0], [0, 0, C, C, 0], 'b-', lw=1, label='cage net')
        for kind, q in L._solids():
            if kind == 'obox':
                ca, cb, hl, hp, yaw, h0, h1 = q; ya = math.radians(yaw)
                u = np.array([math.cos(ya), math.sin(ya)]); v = np.array([-math.sin(ya), math.cos(ya)])
                cs = [np.array([ca, cb]) + sa * hl * u + sb * hp * v for sa, sb in ((1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1))]
                ax.plot([p[1] for p in cs], [p[0] for p in cs], color='0.4', lw=1)
            elif kind == 'cyl':
                ax.add_patch(plt.Circle((q[1], q[0]), q[2], color='0.5', fill=False))
        import course_solids as CS
        tg = CS.tunnel_geom(L); (ea, eb0), (ua, ub), (ra, rb), ln, rad = tg['entry'], tg['axis'], tg['right'], tg['length'], tg['radius']
        cs = [(ea + s_ * ln * ua + q_ * rad * ra, eb0 + s_ * ln * ub + q_ * rad * rb) for s_, q_ in ((0, -1), (1, -1), (1, 1), (0, 1), (0, -1))]
        ax.plot([c_[1] for c_ in cs], [c_[0] for c_ in cs], color='goldenrod')
        if len(PH):
            ax.scatter(bb, aa, s=2, c='red', alpha=0.4, label=f'phantom voxels ({len(vox)})')
        if last_cls is not None:
            pts, net, real, ph = last_cls
        ax.plot(path[:, 1] + L.START[1], path[:, 0] + L.START[0], 'k-', lw=1, label='flown path (PX4)')
        for e in events:
            if e['kind'] == 'avoid':
                k = int(np.argmin(np.abs(np.array([r['t'] for r in rows]) - e['t']))) if rows else None
                if k is not None:
                    vp = rows[k]['veh']; ax.plot(vp[1] + L.START[1], vp[0] + L.START[0], 'm^', ms=6)
        ax.set_xlabel('b = east of west net (m)'); ax.set_ylabel('a = north of south net (m)')
        ax.set_aspect('equal'); ax.set_xlim(-0.5, C + 0.5); ax.set_ylim(-0.5, C + 0.5); ax.legend(loc='lower right', fontsize=7)
        ax.set_title(f'{F.name}: phantom voxels (red), avoid events (magenta)', fontsize=8)
        fig.savefig(F / 'sim' / 'obstacles_topdown.png', dpi=110, bbox_inches='tight')
    except Exception as e:  # noqa: BLE001
        res['plot_err'] = repr(e)[:120]
    print(json.dumps({k: v for k, v in res.items() if k not in ('phases', 'phantom_ahead_soft_times', 'mission_events')}, default=str))
    print('events:', json.dumps(res['mission_events'], default=str)[:1500])
    return 0


if __name__ == '__main__':
    sys.exit(main())
