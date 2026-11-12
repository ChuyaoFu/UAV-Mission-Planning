#!/usr/bin/env python3
"""Evidence for avoid/blocked/abort events of one flight: what the node's obstacle map held in front of the
vehicle at that moment (sim-side; for the ObstacleMap discussion).

  python3 probe_events.py FLIGHT_DIR [--variant NAME]  -> FLIGHT_DIR/sim/events_evidence.json + sim/evidence/*.png

Probe = segment from the vehicle along its heading, PROBE_M long (mission probe_min_m 1.2 + probe_margin_m 0.5),
radius R_SOFT 0.35 m (half_span + clearance_margin), as in ObstacleMap.clearance. Each voxel in it is classified
real / net / phantom against layout.py (see analyze_obstacles.py). PNG: left image (as recorded) + stereo depth,
probe voxels projected (red = phantom, green = real, blue = net).
"""
import argparse, json, math, sys, tarfile, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_flight import read, stamp  # noqa: E402
from analyze_obstacles import solids_dist, TOL, NET_TOL, R_SOFT  # noqa: E402

PROBE_M = 1.7


def seg_dist(P, a, b):
    ab = b - a; t = np.clip(((P - a) @ ab) / max(1e-9, ab @ ab), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1), t * np.linalg.norm(ab)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('flight'); a_ = ap.parse_args(); F = Path(a_.flight)
    tmp = Path(tempfile.mkdtemp(prefix='ev_'))
    with tarfile.open(F / 'code.tar.gz') as tf:
        tf.extractall(tmp)
    sys.path.insert(0, str(tmp))
    from node.obstacle_map import ObstacleMap
    from common.geometry import Frame, camera_pose_ned
    sys.path.insert(0, '/replica'); import layout as L
    P = json.loads((HERE / 'profile.json').read_text())
    c = P['camera']; K = tuple(c['K']); R_bc = np.asarray(c['R_base_optical'], float); t_bc = tuple(c['left_in_base_m'])
    ev = []
    for ln in (F / 'mission.log').read_text().splitlines():
        if ln.startswith('{') and '"event"' in ln:
            e = json.loads(ln.replace('NaN', 'null')); k = e['event'].get('kind')
            if k in ('avoid', 'blocked', 'abort'):
                ev.append(dict(t=e['t'], kind=k, phase=e['event'].get('phase'), offset_m=e['event'].get('offset_m'),
                               free_m=e['event'].get('free_m'), reason=e['event'].get('reason')))
    # de-duplicate dithering avoid events: keep the first of every 1 s
    keep, last = [], -9
    for e in ev:
        if e['kind'] != 'avoid' or e['t'] - last > 1.0:
            keep.append(e)
            if e['kind'] == 'avoid':
                last = e['t']
    ev = keep
    A = read(F / 'bag', ['/camera/depth/image_raw', '/stereo/left/image', '/fmu/out/vehicle_local_position_v1', '/fmu/out/vehicle_attitude'])
    lp = A['/fmu/out/vehicle_local_position_v1']; att = A['/fmu/out/vehicle_attitude']
    lp_t = np.array([r for r, _ in lp]); at_t = np.array([r for r, _ in att])
    lefts = {round(stamp(m), 3): m for _, m in A['/stereo/left/image']}
    mp = ObstacleMap(); floor_d = float(lp[0][1].z) + 0.05
    out, k = [], 0
    (F / 'sim' / 'evidence').mkdir(parents=True, exist_ok=True)
    for rt, m in A['/camera/depth/image_raw']:
        i = int(np.searchsorted(lp_t, rt)) - 1; j = int(np.searchsorted(at_t, rt)) - 1
        if i < 0 or j < 0:
            continue
        lpm, atm = lp[i][1], att[j][1]
        d = np.frombuffer(bytes(m.data), np.float32).reshape(m.height, -1)[:, :m.width]
        pos = np.array([lpm.x, lpm.y, lpm.z]); q = tuple(float(x) for x in atm.q)
        R_nc, cam = camera_pose_ned(q, pos, R_bc, t_bc); s = stamp(m)
        mp.insert(Frame(image=None, depth=d, K=K, R_nc=R_nc, cam_ned=cam), floor_d, s)
        while k < len(ev) and s >= ev[k]['t']:
            e = ev[k]; k += 1
            hd = float(lpm.heading); u = np.array([math.cos(hd), math.sin(hd), 0.0])
            pts = mp.points
            rec = dict(e, veh_ned=[round(float(x), 2) for x in pos], heading_deg=round(math.degrees(hd), 1), map_pts=int(len(pts)))
            if len(pts):
                dist, along = seg_dist(pts, pos, pos + PROBE_M * u)
                inn = dist < R_SOFT
                Q = pts[inn]
                aa = Q[:, 0] + L.START[0]; bb = Q[:, 1] + L.START[1]; hh = -(Q[:, 2] - 0.05)
                dn = np.minimum.reduce([aa, L.CAGE - aa, bb, L.CAGE - bb, L.CAGE_H - hh]) if len(Q) else np.zeros(0)
                net = dn < NET_TOL; real = (~net) & (solids_dist(L, aa, bb, hh) <= TOL); ph = ~(net | real)
                rec.update(probe_voxels=int(inn.sum()), probe_real=int(real.sum()), probe_net=int(net.sum()), probe_phantom=int(ph.sum()))
                if inn.any():
                    first = float(along[inn].min())
                    rec['first_hit_along_m'] = round(first, 2)
                    if ph.any():
                        rec['phantom_centroid_cage_abh'] = [round(float(x), 2) for x in (aa[ph].mean(), bb[ph].mean(), hh[ph].mean())]
                        rec['phantom_h_range'] = [round(float(hh[ph].min()), 2), round(float(hh[ph].max()), 2)]
                        rec['phantom_nearest_net_m'] = round(float(dn[ph].min()), 2)
                        rec['phantom_first_along_m'] = round(float(along[inn][ph].min()), 2)
                    if real.any():
                        rec['real_first_along_m'] = round(float(along[inn][real].min()), 2)
                # evidence frame
                try:
                    import cv2
                    Lm = lefts.get(round(s, 3))
                    img = (np.frombuffer(bytes(Lm.data), np.uint8).reshape(Lm.height, Lm.step)[:, :Lm.width] if Lm is not None
                           else np.zeros(d.shape, np.uint8))
                    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    dv = np.clip(d / 4.0 * 255, 0, 255).astype(np.uint8); dv[d <= 0] = 0
                    dvis = cv2.applyColorMap(dv, cv2.COLORMAP_TURBO); dvis[d <= 0] = 0
                    fr = Frame(image=None, depth=d, K=K, R_nc=R_nc, cam_ned=cam)
                    if len(Q):
                        uv, z = fr.project(Q)
                        for (uu, vv), zz, cls in zip(uv, z, np.where(ph, 2, np.where(real, 1, 0))):
                            if zz > 0 and 0 <= uu < d.shape[1] and 0 <= vv < d.shape[0]:
                                col = [(255, 80, 0), (0, 200, 0), (0, 0, 255)][cls]
                                cv2.circle(vis, (int(uu), int(vv)), 2, col, -1); cv2.circle(dvis, (int(uu), int(vv)), 2, col, -1)
                    txt = f"t={s:.2f} {e['kind']} {e['phase']} probe: {rec['probe_phantom']} phantom / {rec['probe_real']} real / {rec['probe_net']} net"
                    both = np.hstack([vis, dvis]); cv2.putText(both, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    name = f"t{s:06.2f}_{e['kind']}.png"; cv2.imwrite(str(F / 'sim' / 'evidence' / name), both); rec['png'] = name
                except Exception as ex:  # noqa: BLE001
                    rec['png_err'] = repr(ex)[:100]
            out.append(rec)
    (F / 'sim' / 'events_evidence.json').write_text(json.dumps(out, indent=1, default=str))
    for r in out:
        print(json.dumps({k_: v for k_, v in r.items() if k_ not in ('png',)}, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
