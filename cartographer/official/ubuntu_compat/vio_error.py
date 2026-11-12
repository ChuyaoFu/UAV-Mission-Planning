#!/usr/bin/env python3
"""VIO (and PX4 EKF) versus Gazebo truth for one flight, split horizontal / vertical.

  python3 vio_error.py FLIGHT_DIR      -> FLIGHT_DIR/sim/vio_error.json + one summary line

VIO = /visual_slam/tracking/odometry (ev_bridge output, odom frame anchored at base_link on the pad);
truth = /gz/gt/odometry (sim/sidebag) re-anchored at its own first sample on the pad (same convention);
PX4 = /fmu/out/vehicle_local_position_v1 (NED -> FLU). Errors are sampled at the VIO stamps while airborne
(truth height > 0.15 m) and over the whole flight; also the error at landing (last sample).
"""
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_flight import read, stamp  # noqa: E402


def main():
    F = Path(sys.argv[1])
    A = read(F / 'bag', ['/visual_slam/tracking/odometry', '/fmu/out/vehicle_local_position_v1', '/clock'])
    S = read(F / 'sim' / 'sidebag', ['/gz/gt/odometry'])
    g = np.array([[stamp(m), m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z,
                   m.pose.pose.orientation.w, m.pose.pose.orientation.z] for _, m in S['/gz/gt/odometry']])
    y0 = 2 * math.atan2(g[0, 5], g[0, 4]); c, s = math.cos(-y0), math.sin(-y0)
    d = g[:, 1:4] - g[0, 1:4]; gt = np.c_[g[:, 0], c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1], d[:, 2]]
    vio = np.array([[stamp(m), m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z]
                    for _, m in A['/visual_slam/tracking/odometry']])
    clk = [(rt, m.clock.sec + m.clock.nanosec * 1e-9) for rt, m in A['/clock']]
    w2s = lambda rt: float(np.interp(rt, [x[0] for x in clk], [x[1] for x in clk]))
    px4 = np.array([[w2s(rt), m.x, -m.y, -m.z] for rt, m in A['/fmu/out/vehicle_local_position_v1'] if m.xy_valid and m.z_valid])
    res = {'flight': F.name}
    for name, est in (('vio', vio), ('px4', px4)):
        est = est[(est[:, 0] >= gt[0, 0]) & (est[:, 0] <= gt[-1, 0])]
        ti = np.c_[[np.interp(est[:, 0], gt[:, 0], gt[:, k]) for k in (1, 2, 3)]].T
        e = est[:, 1:4] - ti; exy = np.hypot(e[:, 0], e[:, 1]); ez = np.abs(e[:, 2])
        air = ti[:, 2] > 0.15
        path = float(np.linalg.norm(np.diff(ti[air][:, :2], axis=0), axis=1).sum()) if air.sum() > 2 else 0.0
        res[name] = dict(samples=int(len(e)), airborne=int(air.sum()),
                         xy_rms_cm=round(100 * float(np.sqrt((exy[air] ** 2).mean())), 1), xy_max_cm=round(100 * float(exy[air].max()), 1),
                         z_rms_cm=round(100 * float(np.sqrt((ez[air] ** 2).mean())), 1), z_max_cm=round(100 * float(ez[air].max()), 1),
                         end_xy_cm=round(100 * float(exy[-1]), 1), end_z_cm=round(100 * float(ez[-1]), 1),
                         path_m=round(path, 2), drift_pct_of_path=round(100 * float(exy[-1]) / path, 2) if path > 0 else None)
    (F / 'sim' / 'vio_error.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == '__main__':
    main()
