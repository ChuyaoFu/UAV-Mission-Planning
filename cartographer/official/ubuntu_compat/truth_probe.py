#!/usr/bin/env python3
"""Sim-side truth around given times of one flight: pose, attitude, speed, nearest layout solids.

  python3 truth_probe.py FLIGHT_DIR T [T ...]        (T = sim time s, as in mission.log)
  python3 truth_probe.py FLIGHT_DIR --nearest         (time of minimum clearance to any course solid + the solid)

Truth = /gz/gt/odometry (sim/sidebag), start frame f = gz x, l = gz y. Course = physical solids (course_solids.py from /replica/layout.py).
Never visible to flight code.
"""
import math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_flight import read, stamp  # noqa: E402
import course_solids as _cs  # noqa: E402
L = _cs.load_layout(sys.argv[1] if len(sys.argv) > 1 else None)   # the layout this flight flew


def solid_dists(a, b, h):
    import course_solids
    return sorted(((course_solids.dist(kind, q, a, b, h), name, kind, q) for name, kind, q in course_solids.solids(L)),
                  key=lambda x: x[0])


def fl_of(ca, cb):
    return round(ca - L.START[0], 3), round(-(cb - L.START[1]), 3)


def main():
    F = Path(sys.argv[1]); S = read(F / 'sim' / 'sidebag', ['/gz/gt/odometry'])
    g = []
    for _, m in S['/gz/gt/odometry']:
        p, q, v = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear
        roll = math.degrees(math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y)))
        pitch = math.degrees(math.asin(max(-1, min(1, 2 * (q.w * q.y - q.z * q.x)))))
        yaw = math.degrees(math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
        g.append((stamp(m), p.x, p.y, p.z, roll, pitch, yaw, v.x, v.y, v.z))
    g = np.array(g)
    ts = []
    if '--nearest' in sys.argv:
        k = np.where(g[:, 3] > 0.15)[0][::5]
        d = [solid_dists(g[i, 1] + L.START[0], -g[i, 2] + L.START[1], g[i, 3])[0][0] for i in k]
        ts.append(float(g[k[int(np.argmin(d))], 0]))
    ts += [float(x) for x in sys.argv[2:] if x != '--nearest']
    for t in ts:
        i = int(np.argmin(np.abs(g[:, 0] - t)))
        f, l, h, r, p, y = g[i, 1], g[i, 2], g[i, 3], g[i, 4], g[i, 5], g[i, 6]
        print(f't {g[i, 0]:.2f}: f {f:.3f} l {l:.3f} h {h:.3f}  roll {r:.1f} pitch {p:.1f} yaw {y:.1f}  '
              f'v {math.hypot(g[i, 7], g[i, 8]):.2f} m/s')
        for dd, k, kind, q in solid_dists(f + L.START[0], -l + L.START[1], h)[:3]:
            c = fl_of(q[0], q[1])
            desc = (f'box centre f,l {c} half {q[2]:.2f}x{q[3]:.2f} yaw {q[4]:.0f} h {q[5]:.2f}-{q[6]:.2f}' if kind == 'obox'
                    else f'cyl centre f,l {c} r {q[2]:.2f} h {q[3]:.2f}-{q[4]:.2f}')
            print(f'    {k}: {dd:.3f} m  ({desc})')
        w = (g[:, 0] > t - 1.5) & (g[:, 0] < t + 1.5)
        print(f'    +-1.5 s: max |roll| {np.abs(g[w, 4]).max():.1f}  max |pitch| {np.abs(g[w, 5]).max():.1f}  '
              f'max |a_xy| {np.abs(np.gradient(np.hypot(g[w, 7], g[w, 8]), g[w, 0])).max():.2f} m/s^2')


if __name__ == '__main__':
    main()
