#!/usr/bin/env python3
"""Numbers for one official-compat flight folder (runs in the container: rosbag2_py + mcap).

  python3 analyze_flight.py FLIGHT_DIR          -> FLIGHT_DIR/sim/analysis.json + printed summary

Uses bag/ (flight_run: PX4 in/out, estimator, TF) and sim/sidebag/ (Gazebo truth /gz/gt/odometry, raw OpenVINS).
Frames: Gazebo world x = north, y = west, z = up, start pad at the origin, vehicle facing +x at start;
PX4 NED (EKF origin = start); start frame f = x, l = y, u = z - z_rest.
"""
import json, math, sys
from pathlib import Path
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


def read(bag, topics):
    out = {t: [] for t in topics}
    if not Path(bag).exists():
        return out
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    r.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    cls = {t: get_message(types[t]) for t in topics if t in types}
    while r.has_next():
        t, data, rt = r.read_next()
        out[t].append((rt * 1e-9, deserialize_message(data, cls[t])))
    return out


def stamp(m):
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def main():
    F = Path(sys.argv[1]); res = {'flight': F.name}
    A = read(F / 'bag', ['/clock', '/fmu/out/vehicle_local_position_v1', '/fmu/in/trajectory_setpoint', '/fmu/out/distance_sensor',
                         '/fmu/out/vehicle_status_v1', '/estimator/healthy', '/fmu/out/vehicle_land_detected'])
    S = read(F / 'sim' / 'sidebag', ['/gz/gt/odometry', '/ov_msckf/odomimu'])
    clk = [(rt, m.clock.sec + m.clock.nanosec * 1e-9) for rt, m in A['/clock']]
    if len(clk) > 10:
        res['rtf'] = round((clk[-1][1] - clk[0][1]) / (clk[-1][0] - clk[0][0]), 3)
    w2s = (lambda rt: np.interp(rt, [c[0] for c in clk], [c[1] for c in clk])) if clk else (lambda rt: rt)
    # phases from mission.log
    ph = []
    for ln in (F / 'mission.log').read_text().splitlines():
        if ln.startswith('{') and '"kind": "phase"' in ln:
            e = json.loads(ln.replace('NaN', 'null'))['event']; ph.append((e['t'], e['phase'], e['to'], e.get('reason')))
    res['phases'] = [dict(t=round(t, 2), frm=a, to=b, why=w) for t, a, b, w in ph]
    # truth
    gt = np.array([[stamp(m), m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z] for _, m in S['/gz/gt/odometry']])
    lp = np.array([[w2s(rt), m.x, m.y, m.z] for rt, m in A['/fmu/out/vehicle_local_position_v1']])
    sp = np.array([[w2s(rt), *m.position] for rt, m in A['/fmu/in/trajectory_setpoint']])
    if len(gt) and len(lp):
        z0 = float(np.median(gt[gt[:, 0] < gt[0, 0] + 3, 3]))
        g_ned = np.c_[gt[:, 0], gt[:, 1], -gt[:, 2], -(gt[:, 3] - z0)]          # truth in NED (start-anchored)
        g_fr = g_ned                                                             # truth in PX4's frame (spawn-anchored)
        try:
            sp_ = json.loads((F / 'sim' / 'stack.json').read_text()).get('spawn', '')
        except Exception:  # noqa: BLE001
            sp_ = ''
        if sp_:
            sf, sl, sy = (float(v) for v in sp_.split(',')); cy, sy_ = math.cos(math.radians(sy)), math.sin(math.radians(sy))
            dx, dy = gt[:, 1] - sf, gt[:, 2] - sl                                 # gz x = f, y = l (course start frame)
            fp, lp_ = cy * dx + sy_ * dy, -sy_ * dx + cy * dy
            g_fr = np.c_[gt[:, 0], fp, -lp_, -(gt[:, 3] - z0)]; res['spawn'] = sp_
        gi = np.c_[[np.interp(lp[:, 0], g_fr[:, 0], g_fr[:, k]) for k in (1, 2, 3)]].T
        err = lp[:, 1:4] - gi
        air = gi[:, 2] < -0.15
        res['truth_rest_base_height_m'] = round(z0, 4)
        res['px4_est_err'] = dict(xy_max=round(float(np.hypot(err[:, 0], err[:, 1]).max()), 3),
                                  xy_rms=round(float(np.sqrt((err[:, 0]**2 + err[:, 1]**2).mean())), 3),
                                  z_max=round(float(np.abs(err[:, 2]).max()), 3),
                                  xy_end=round(float(np.hypot(*err[-1, :2])), 3), z_end=round(float(err[-1, 2]), 3))
        res['truth_max_height_m'] = round(float(-g_ned[:, 3].min()), 3)
        res['truth_path_len_m'] = round(float(np.linalg.norm(np.diff(g_ned[:, 1:3], axis=0), axis=1).sum()), 2)
        res['truth_end_xy_from_start_m'] = round(float(np.hypot(g_ned[-1, 1], g_ned[-1, 2])), 3)
        res['truth_max_xy_from_start_m'] = round(float(np.hypot(g_ned[:, 1], g_ned[:, 2]).max()), 3)
        res['truth_end_fl'] = [round(float(g_ned[-1, 1]), 3), round(float(-g_ned[-1, 2]), 3)]
        if len(sp):
            # hover offset: during steady hover segments (|v| < 0.1, setpoint constant) truth/estimate vs setpoint z
            spi = np.c_[[np.interp(lp[:, 0], sp[:, 0], sp[:, k]) for k in (1, 2, 3)]].T
            vel = np.array([[m.vx, m.vy, m.vz] for _, m in A['/fmu/out/vehicle_local_position_v1']])
            spd = np.gradient(spi[:, 2], lp[:, 0])
            hov = air & (np.linalg.norm(vel, axis=1) < 0.08) & (np.abs(spd) < 0.01)
            if A['/fmu/out/distance_sensor']:              # free hover only: not resting/pushing on the stone
                dst = np.array([(w2s(rt), m.current_distance) for rt, m in A['/fmu/out/distance_sensor']])
                hov &= np.interp(lp[:, 0], dst[:, 0], np.nan_to_num(dst[:, 1], posinf=9.0, neginf=0.0)) > 0.3
            if hov.sum() > 20:
                res['hover'] = dict(samples=int(hov.sum()),
                                    est_below_sp_cm=round(float(np.median(lp[hov, 3] - spi[hov, 2])) * 100, 1),
                                    truth_below_sp_cm=round(float(np.median(gi[hov, 2] - spi[hov, 2])) * 100, 1),
                                    truth_xy_hold_std_cm=round(float(np.hypot(*(gi[hov, :2] - gi[hov, :2].mean(0)).T).std()) * 100, 1))
            track = np.linalg.norm(lp[air, 1:3] - spi[air, :2], axis=1) if air.any() else np.zeros(1)
            res['xy_track_err_est'] = dict(median=round(float(np.median(track)), 3), p95=round(float(np.percentile(track, 95)), 3))
    if len(S['/ov_msckf/odomimu']) and len(gt):
        vo = np.array([[stamp(m), m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z] for _, m in S['/ov_msckf/odomimu']])
        res['vio_samples'] = len(vo)
    ds = [(w2s(rt), m.current_distance, m.min_distance <= m.current_distance <= m.max_distance and m.signal_quality != 0)
          for rt, m in A['/fmu/out/distance_sensor']]
    if ds:
        res['tof_valid_frac'] = round(sum(1 for d in ds if d[2]) / len(ds), 3)
        if len(gt):
            dsa = np.array([(t, d) for t, d, ok in ds if ok])
            if len(dsa):
                gz_ = np.interp(dsa[:, 0], gt[:, 0], gt[:, 3])
                res['tof_minus_truth_height_med_m'] = round(float(np.median(dsa[:, 1] - gz_)), 3)
    # sim-side truth: clearances to the replica course (layout.py; never visible to flight code)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent)); import course_solids as _cs
        L = _cs.load_layout(F); res['layout_flown'] = L.__flown_path__
        if len(gt):
            a = gt[:, 1] + L.START[0]; b = -gt[:, 2] + L.START[1]; hgt = gt[:, 3]
            sys.path.insert(0, str(Path(__file__).resolve().parent)); import course_solids as CS
            tg = CS.tunnel_geom(L); (ea, eb), (ua, ub), (ra, rb) = tg['entry'], tg['axis'], tg['right']
            s_ax = (a - ea) * ua + (b - eb) * ub; r_ax = np.hypot((a - ea) * ra + (b - eb) * rb, hgt - tg['centre_h'])
            inn = (s_ax > 0) & (s_ax < tg['length']) & (r_ax < tg['radius'] + 0.3)
            if inn.any():
                lat_t = ((a - ea) * ra + (b - eb) * rb)[inn]; ver_t = (hgt - tg['centre_h'])[inn]
                res['tunnel_pass'] = dict(samples=int(inn.sum()), max_radial_off_m=round(float(r_ax[inn].max()), 3),
                                          max_lateral_off_m=round(float(lat_t[np.argmax(np.abs(lat_t))]), 3),
                                          max_vertical_off_m=round(float(ver_t[np.argmax(np.abs(ver_t))]), 3),
                                          min_clear_to_mesh_m=round(float(tg['mesh_r'] - r_ax[inn].max() - 0.19), 3))
            w_ = L.WINDOW
            if 'normal_ne' in w_:                            # v4: yawed board, normal to the approach side
                nrm = w_['normal_ne']; al = (math.cos(math.radians(w_['yaw'])), math.sin(math.radians(w_['yaw'])))
            else:                                            # v3: board along east, crossed north -> south
                nrm, al = (1.0, 0.0), (0.0, 1.0)
            sd = (a - w_['a']) * nrm[0] + (b - w_['b']) * nrm[1]; lat_w = (a - w_['a']) * al[0] + (b - w_['b']) * al[1]
            cross = np.where((np.diff(np.sign(sd)) != 0) & (np.abs(lat_w[:-1]) < 1.0))[0]
            if len(cross):
                i = cross[0]; db, dh = lat_w[i], hgt[i] - w_['hole_c']
                res['window_pass'] = dict(lateral_off_m=round(float(db), 3), vertical_off_m=round(float(dh), 3),
                                          clear_side_m=round(w_['hole_w'] / 2 - abs(db) - 0.19, 3),
                                          clear_vert_m=round(w_['hole_h'] / 2 - abs(dh) - 0.05, 3))
            gx, gy0, gy1 = getattr(L, 'MAZE_EXIT_FT', (11.0, 16.9, 20.0))     # maze exit = Gate A (truth crossing eastward)
            if hasattr(L, 'GATE_B_ROUTE_FT'):
                bx = gx * L.FT; cra = np.where((np.diff(np.sign(b - bx)) > 0) & (a[:-1] > gy0 * L.FT - 0.3) & (a[:-1] < gy1 * L.FT + 0.3) & (hgt[:-1] > 0.15))[0]
                if len(cra):
                    i = cra[0]
                    res['gate_a_pass'] = dict(t=round(float(gt[i, 0]), 1), offset_from_centre_m=round(float(a[i] - (gy0 + gy1) / 2 * L.FT), 3),
                                              height_m=round(float(hgt[i]), 3))
            if hasattr(L, 'GATE_B_ROUTE_FT'):                # v4 Gate B: open gap at map x 10.5 ft, y 8.1 -> 11.5 ft (crossed westward)
                gbx, ga0, ga1 = 10.5 * L.FT, 8.1 * L.FT, 11.5 * L.FT
                crg = np.where((np.diff(np.sign(b - gbx)) != 0) & (a[:-1] > ga0 - 0.3) & (a[:-1] < ga1 + 0.3))[0]
                if len(crg):
                    i = crg[0]; off = a[i] - (ga0 + ga1) / 2
                    res['gate_b_pass'] = dict(t=round(float(gt[i, 0]), 1), offset_from_centre_m=round(float(off), 3), height_m=round(float(hgt[i]), 3),
                                              prop_clear_each_side_m=round((ga1 - ga0) / 2 - abs(off) - 0.23, 3))
            wl = L.WALL
            if 'y' in wl:                                    # v4: barrier = plane a = y ft, b from x0 to x1 ft
                pa = wl['y'] * L.FT; crw = np.where((np.diff(np.sign(a - pa)) != 0) & (b[:-1] > wl['x0'] * L.FT - 0.3) & (b[:-1] < wl['x1'] * L.FT + 0.3))[0]
            else:                                            # v3: board along north at b = wl['b']
                near_wl = np.abs(a - wl['a']) < wl['width'] / 2 + 0.5
                crw = np.where((np.diff(np.sign(b - wl['b'])) != 0) & near_wl[:-1])[0]
            if len(crw):
                res['wall_pass'] = dict(height_m=round(float(hgt[crw[0]]), 3), skids_over_top_m=round(float(hgt[crw[0]] - 0.05 - wl['top']), 3))
            if hasattr(L, 'STONE'):                          # v1/v2 white stone block
                ta, tb, hl_n, hl_e, top = L.STONE['a'], L.STONE['b'], L.STONE['len_n'] / 2, L.STONE['len_e'] / 2, L.STONE['height']
            else:                                            # v3 flat landing pad (NED n/e)
                ta, tb = L.PAD['n'] + L.START[0], L.PAD['e'] + L.START[1]; hl_n = hl_e = L.PAD['size'] / 2; top = L.PAD['thick']
            on = (np.abs(a - ta) < hl_n) & (np.abs(b - tb) < hl_e) & (np.abs(hgt - (top + 0.05)) < 0.02)
            if on.any():
                res['stone_touchdown'] = dict(seconds=round(float(np.ptp(gt[on, 0])), 1), off_center_m=round(float(np.hypot(np.median(a[on]) - ta, np.median(b[on]) - tb)), 3))
            res['min_dist_to_net_m'] = round(float(min(a.min(), L.CAGE - a.max(), b.min(), L.CAGE - b.max())), 3)
            if hasattr(L, '_solids'):                        # base_link -> nearest physical course solid (airborne)
                sys.path.insert(0, str(Path(__file__).resolve().parent)); import course_solids
                k = np.where(hgt > 0.15)[0][::10]
                if len(k):
                    d = [course_solids.clearance(L, a[i], b[i], hgt[i]) for i in k]; j = int(np.argmin([x[0] for x in d]))
                    res['min_dist_to_course_solid_m'] = dict(d=round(float(d[j][0]), 3), solid=d[j][1], t=round(float(gt[k[j], 0]), 1),
                                                             f=round(float(gt[k[j], 1]), 2), l=round(float(gt[k[j], 2]), 2),
                                                             note='base_link centre; props reach 0.19 m (axis) - 0.23 m (diagonal)')
    except Exception as e:  # noqa: BLE001
        res['course_truth_err'] = repr(e)[:120]
    hb = [m.data for _, m in A['/estimator/healthy']]
    if hb:
        res['healthy_true_frac'] = round(sum(hb) / len(hb), 3)
    st = [(w2s(rt), m.nav_state, m.arming_state) for rt, m in A['/fmu/out/vehicle_status_v1']]
    res['nav_states'] = sorted(set(s[1] for s in st))
    (F / 'sim').mkdir(exist_ok=True)
    (F / 'sim' / 'analysis.json').write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != 'phases'}, default=str))
    print('phases:', ' -> '.join(f"{p['to']}@{p['t']}" for p in res['phases']))


if __name__ == '__main__':
    main()
