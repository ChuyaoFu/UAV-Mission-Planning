#!/usr/bin/env python3
"""How well damped is the simulated vehicle? Braking overshoot and yaw-induced drift, from one flight (sim truth).

  python3 dynamics_probe.py FLIGHT_DIR [...]     -> FLIGHT_DIR/sim/dynamics.json + one line per flight

Official lab reference (official-sim-test, 2026-11-07): ~0.3 m overshoot braking from 0.5 m/s; +-0.25 m oscillation after
a fast 90 deg yaw. If Gazebo is much better damped, it understates the corner risk.
  braking   the position setpoint (/fmu/in/trajectory_setpoint) moving >= 0.4 m/s comes to rest (<= 0.03 m/s) within 1 s and
            HOLDS (within 0.05 m) for >= 1.5 s; overshoot = furthest truth position PAST the stop point along the incoming
            direction during that hold (<= 3 s)
  yaw       the setpoint yaw turns >= 60 deg within 2 s while the position setpoint moves <= 0.05 m;
            and the vehicle starts within 0.1 m of it; drift = max horizontal truth distance from the held setpoint
            during the turn + up to 2 s while the setpoint stays put
"""
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_flight import read, stamp  # noqa: E402


def probe(F):
    F = Path(F)
    A = read(F / 'bag', ['/clock', '/fmu/in/trajectory_setpoint'])
    S = read(F / 'sim' / 'sidebag', ['/gz/gt/odometry'])
    clk = [(rt, m.clock.sec + m.clock.nanosec * 1e-9) for rt, m in A['/clock']]
    w2s = lambda rt: float(np.interp(rt, [c[0] for c in clk], [c[1] for c in clk]))
    sp = np.array([[w2s(rt), m.position[0], -m.position[1], -m.yaw] for rt, m in A['/fmu/in/trajectory_setpoint']
                   if np.all(np.isfinite(m.position[:2]))])
    g = np.array([[stamp(m), m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z] for _, m in S['/gz/gt/odometry']])
    air = g[:, 3] > 0.3
    t = sp[:, 0]; dt = np.gradient(t); v = np.hypot(np.gradient(sp[:, 1]), np.gradient(sp[:, 2])) / np.maximum(dt, 1e-3)
    vs = np.convolve(v, np.ones(5) / 5, 'same')
    gt = lambda tt, k: np.interp(tt, g[:, 0], g[:, k])
    brakes, i = [], 0
    while i < len(t) - 1:
        if vs[i] >= 0.4:
            j = i
            while j < len(t) - 1 and t[j] - t[i] <= 1.0 and vs[j] > 0.03:
                j += 1
            if vs[j] <= 0.03 and t[j] - t[i] <= 1.0 and np.interp(t[j], g[:, 0], air.astype(float)) > 0.5:
                k0 = max(i - 5, 0); u = sp[i, 1:3] - sp[k0, 1:3]
                stop = sp[j, 1:3]; h = j                        # the stop must HOLD (setpoint within 0.05 m) for >= 1.5 s
                while h < len(t) - 1 and np.linalg.norm(sp[h + 1, 1:3] - stop) <= 0.05 and t[h + 1] - t[j] <= 3.0:
                    h += 1
                if np.linalg.norm(u) > 1e-3 and t[h] - t[j] >= 1.5:
                    u = u / np.linalg.norm(u)
                    w = (g[:, 0] >= t[j]) & (g[:, 0] <= t[h])
                    if w.any():
                        past = ((g[w, 1:3] - stop) @ u).max()
                        v_truth = float(np.hypot(*np.gradient(g[(g[:, 0] > t[j] - 0.3) & (g[:, 0] < t[j]), 1:3], axis=0).mean(0))
                                        / max(np.median(np.diff(g[:, 0])), 1e-3)) if ((g[:, 0] > t[j] - 0.3) & (g[:, 0] < t[j])).sum() > 2 else None
                        brakes.append(dict(t=round(float(t[j]), 2), v_sp_before=round(float(vs[i]), 2),
                                           v_truth_at_stop=None if v_truth is None else round(v_truth, 2), overshoot_m=round(float(past), 3)))
                i = j + 1; continue
        i += 1
    yaws, i = [], 0
    yw = np.unwrap(sp[:, 3])
    while i < len(t) - 1:
        j = i
        while j < len(t) - 1 and t[j + 1] - t[i] <= 2.0:
            j += 1
        if abs(math.degrees(yw[j] - yw[i])) >= 60 and np.linalg.norm(sp[j, 1:3] - sp[i, 1:3]) <= 0.05 \
                and np.interp(t[i], g[:, 0], air.astype(float)) > 0.5:
            hold = sp[i, 1:3]; h = j                            # held window: setpoint stays within 0.05 m (<= 2 s after the turn)
            while h < len(t) - 1 and np.linalg.norm(sp[h + 1, 1:3] - hold) <= 0.05 and t[h + 1] - t[j] <= 2.0:
                h += 1
            w = (g[:, 0] >= t[i]) & (g[:, 0] <= t[h])
            d = np.hypot(g[w, 1] - hold[0], g[w, 2] - hold[1])
            d0 = float(np.hypot(gt(t[i], 1) - hold[0], gt(t[i], 2) - hold[1]))
            if d0 > 0.1 or not w.any():                         # vehicle not settled at the hold point: not a clean event
                i = j + 1; continue
            yaws.append(dict(t=round(float(t[i]), 2), yaw_deg=round(math.degrees(yw[j] - yw[i]), 1), dur_s=round(float(t[j] - t[i]), 2),
                             drift_max_m=round(float(d.max()), 3), offset_at_start_m=round(d0, 3)))
            i = j + 1; continue
        i += 1
    res = dict(flight=F.name, brakes=brakes, yaws=yaws)
    ob = [b['overshoot_m'] for b in brakes]; yd = [y['drift_max_m'] for y in yaws]
    res['summary'] = dict(n_brakes=len(ob), overshoot_median_m=round(float(np.median(ob)), 3) if ob else None,
                          overshoot_max_m=round(float(max(ob)), 3) if ob else None,
                          n_yaws=len(yd), yaw_drift_median_m=round(float(np.median(yd)), 3) if yd else None,
                          yaw_drift_max_m=round(float(max(yd)), 3) if yd else None)
    (F / 'sim' / 'dynamics.json').write_text(json.dumps(res, indent=1))
    return res


if __name__ == '__main__':
    for f in sys.argv[1:]:
        try:
            r = probe(f); print(r['flight'], json.dumps(r['summary']))
        except Exception as e:  # noqa: BLE001
            print(Path(f).name, 'probe failed:', repr(e)[:120])
