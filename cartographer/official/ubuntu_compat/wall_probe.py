#!/usr/bin/env python3
"""What did the node's wall-top measurement see? (sim-side evidence)

  python3 wall_probe.py FLIGHT_DIR       -> FLIGHT_DIR/sim/wall_probe.json (+ sim/evidence/wall_t*.png)

Rebuilds the node's ObstacleMap (flight's own code.tar.gz) and, at every `wall_top` event in mission.log, applies the
node's measure_wall_top rule (points with 0 < s < L along wall_from->wall_to and |lateral| < 0.8 m; top = 2nd
percentile of NED d). The highest 2 % (the points that set the top) are classified against replica truth:
wall board (within 0.12 m of the WALL solid), other real solid, cage net/frame (within 0.2 m of a net plane), phantom.
"""
import json, math, sys, tarfile, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_flight import read, stamp  # noqa: E402
from analyze_obstacles import solids_dist, TOL, NET_TOL  # noqa: E402


def main():
    F = Path(sys.argv[1])
    tmp = Path(tempfile.mkdtemp(prefix='wall_'))
    with tarfile.open(F / 'code.tar.gz') as tf:
        tf.extractall(tmp)
    sys.path.insert(0, str(tmp))
    from node.obstacle_map import ObstacleMap
    from common.geometry import Frame, camera_pose_ned
    from mission.core import CoursePrior, StartFrame
    sys.path.insert(0, '/replica'); import layout as L
    P = json.loads((HERE / 'profile.json').read_text())
    c = P['camera']; K = tuple(c['K']); R_bc = np.asarray(c['R_base_optical'], float); t_bc = tuple(c['left_in_base_m'])
    prior = CoursePrior.from_dict(json.loads((F / 'prior.json').read_text()))
    evs, home, yaw = [], None, None
    for ln in (F / 'mission.log').read_text().splitlines():
        if ln.startswith('{') and '"event"' in ln:
            e = json.loads(ln.replace('NaN', 'null'))
            if e['event'].get('type') == 'wall_top':
                evs.append((e['t'], e['event']['d'], e['event']['n']))
            if e['event'].get('kind') == 'home':
                home = np.asarray(e['event']['home_ned'], float); yaw = math.radians(e['event']['yaw_deg'])
    sf = StartFrame(home, yaw); a_ = sf.to_ned(prior.wall_cross_from)[:2]; b_ = sf.to_ned(prior.wall_cross_to)[:2]
    ab = b_ - a_; Lw = float(np.linalg.norm(ab)); u = ab / Lw; n_ = np.array([-u[1], u[0]])
    A = read(F / 'bag', ['/camera/depth/image_raw', '/stereo/left/image', '/fmu/out/vehicle_local_position_v1', '/fmu/out/vehicle_attitude'])
    lp = A['/fmu/out/vehicle_local_position_v1']; att = A['/fmu/out/vehicle_attitude']
    lp_t = np.array([r for r, _ in lp]); at_t = np.array([r for r, _ in att])
    lefts = {round(stamp(m), 3): m for _, m in A['/stereo/left/image']}
    mp = ObstacleMap(); floor_d = float(lp[0][1].z) + 0.05; out, k = [], 0
    wl = L.WALL
    for rt, m in A['/camera/depth/image_raw']:
        i = int(np.searchsorted(lp_t, rt)) - 1; j = int(np.searchsorted(at_t, rt)) - 1
        if i < 0 or j < 0:
            continue
        lpm = lp[i][1]; pos = np.array([lpm.x, lpm.y, lpm.z])
        R_nc, cam = camera_pose_ned(tuple(float(x) for x in att[j][1].q), pos, R_bc, t_bc)
        d = np.frombuffer(bytes(m.data), np.float32).reshape(m.height, -1)[:, :m.width]; s = stamp(m)
        fr = Frame(image=None, depth=d, K=K, R_nc=R_nc, cam_ned=cam)
        mp.insert(fr, floor_d, s)
        while k < len(evs) and s >= evs[k][0]:
            t_e, d_e, n_e = evs[k]; k += 1
            pts = mp.points
            rel = pts[:, :2] - a_; ss = rel @ u; lat = np.abs(rel @ n_)
            sel = pts[(ss > 0) & (ss < Lw) & (lat < 0.8)]
            if len(sel) < 20:
                continue
            top = float(np.percentile(sel[:, 2], 2.0)); hi = sel[sel[:, 2] <= top]
            aa = hi[:, 0] + L.START[0]; bb = hi[:, 1] + L.START[1]; hh = -(hi[:, 2] - 0.05)
            dn = np.minimum.reduce([aa, L.CAGE - aa, bb, L.CAGE - bb, L.CAGE_H - hh])
            net = dn < NET_TOL
            dw = np.sqrt(np.maximum(np.abs(aa - wl['a']) - wl['width'] / 2, 0) ** 2 + np.maximum(np.abs(bb - wl['b']) - 0.03, 0) ** 2
                         + np.maximum(np.maximum(wl['bottom'] - hh, hh - wl['top']), 0) ** 2)
            wall = (~net) & (dw <= TOL); real = (~net) & (~wall) & (solids_dist(L, aa, bb, hh) <= TOL); ph = ~(net | wall | real)
            rec = dict(t=t_e, node_d=d_e, node_n=n_e, replay_top_d=round(top, 3), replay_n=int(len(sel)),
                       top_h_above_floor=round(-(top - 0.05), 2), truth_top_h=wl['top'], veh_h=round(-(pos[2] - 0.05), 2),
                       top2pct=dict(n=int(len(hi)), wall=int(wall.sum()), other_real=int(real.sum()), net=int(net.sum()), phantom=int(ph.sum())),
                       top2pct_mean_abh=[round(float(x), 2) for x in (aa.mean(), bb.mean(), hh.mean())],
                       top2pct_dist_to_wall_plane_b=round(float(np.median(np.abs(bb - wl['b']))), 2))
            try:
                import cv2
                Lm = lefts.get(round(s, 3))
                if Lm is not None and (len(out) == 0 or t_e - out[-1]['t'] > 0.3):
                    img = np.frombuffer(bytes(Lm.data), np.uint8).reshape(Lm.height, Lm.step)[:, :Lm.width]
                    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    dv = np.clip(d / 4.0 * 255, 0, 255).astype(np.uint8); dvis = cv2.applyColorMap(dv, cv2.COLORMAP_TURBO); dvis[d <= 0] = 0
                    uv, z = fr.project(hi)
                    for (uu, vv), zz, cl in zip(uv, z, np.where(ph, 2, np.where(net, 0, 1))):
                        if zz > 0 and 0 <= uu < d.shape[1] and 0 <= vv < d.shape[0]:
                            col = [(255, 80, 0), (0, 200, 0), (0, 0, 255)][cl]
                            cv2.circle(vis, (int(uu), int(vv)), 2, col, -1); cv2.circle(dvis, (int(uu), int(vv)), 2, col, -1)
                    both = np.hstack([vis, dvis])
                    cv2.putText(both, f"t={t_e:.2f} wall top h={rec['top_h_above_floor']} m (truth {wl['top']}): top 2%: {rec['top2pct']}",
                                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                    (F / 'sim' / 'evidence').mkdir(parents=True, exist_ok=True)
                    nm = f'wall_t{t_e:06.2f}.png'; cv2.imwrite(str(F / 'sim' / 'evidence' / nm), both); rec['png'] = nm
            except Exception as ex:  # noqa: BLE001
                rec['png_err'] = repr(ex)[:80]
            out.append(rec)
    (F / 'sim' / 'wall_probe.json').write_text(json.dumps(out, indent=1, default=str))
    for r in out:
        print(json.dumps({k_: v for k_, v in r.items() if k_ != 'png'}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
