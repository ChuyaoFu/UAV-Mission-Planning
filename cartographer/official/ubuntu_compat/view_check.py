#!/usr/bin/env python3
"""Where was the true target in the camera image while the mission hovered to detect it? (sim-side)

  python3 view_check.py FLIGHT_DIR      -> per detect window (status hint detector = window / landing / tunnel):
      median vehicle position (start frame), heading error to the true target, range, and the true target centre's
      pixel position + whether the target's extent fits inside the 640x400 image (OFFICIAL_K, horizontal camera).
Truth from ubuntu_compat/course_prior.replica.json + replica layout sizes; start frame from the mission 'home' event.
"""
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_flight import read  # noqa: E402


def main():
    F = Path(sys.argv[1]); sys.path.insert(0, '/replica'); import layout as L
    P = json.loads((HERE / 'profile.json').read_text()); fx, fy, cx, cy = P['camera']['K']; t_bc = P['camera']['left_in_base_m']
    T = json.loads((HERE / 'course_prior.replica.json').read_text())
    home, yaw0, st = None, None, []
    for ln in (F / 'mission.log').read_text().splitlines():
        if not ln.startswith('{'):
            continue
        j = json.loads(ln.replace('NaN', 'null'))
        if 'event' in j and j['event'].get('kind') == 'home':
            home = np.asarray(j['event']['home_ned'], float); yaw0 = math.radians(j['event']['yaw_deg'])
        elif 'phase' in j and 'event' not in j:
            st.append((j['t'], (j.get('hints') or {}).get('detector'), j['phase'], j.get('sub'), j['pos'], j.get('v')))
    c0, s0 = math.cos(yaw0), math.sin(yaw0)
    to_ned = lambda f, l, u: home + np.array([f * c0 + l * s0, f * s0 - l * c0, -u])
    w = L.WINDOW
    s = L.STONE if hasattr(L, 'STONE') else dict(len_e=L.PAD['size'])     # v3: flat pad
    tgt = {'window': (to_ned(T['window']['center']['forward'], T['window']['center']['left'], T['window']['center']['up']), (w['hole_w'], w['hole_h'])),
           'landing': (to_ned(T['landing_target']['center']['forward'], T['landing_target']['center']['left'], T['landing_target']['top_up']), (s['len_e'], 0.05)),
           'tunnel': (to_ned(T['tunnel']['entry']['forward'], T['tunnel']['entry']['left'], T['tunnel']['entry']['up']), (2 * L.TUNNEL['radius'], 2 * L.TUNNEL['radius']))}
    A = read(F / 'bag', ['/fmu/out/vehicle_local_position_v1'])
    hd = [(m.timestamp * 1e-6, float(m.heading)) for _, m in A['/fmu/out/vehicle_local_position_v1']]
    out = []
    for kind in ('tunnel', 'window', 'landing'):
        rows = [r for r in st if r[1] == kind and (r[5] or 0) < 0.15]          # hovering while this detector is wanted
        if not rows:
            continue
        pos = np.median(np.array([r[4] for r in rows], float), 0)
        t_mid = rows[len(rows) // 2][0]
        h = float(np.median([x[1] for x in hd if abs(x[0] - t_mid) < 2.0])) if hd else 0.0
        R = np.array([[math.cos(h), -math.sin(h), 0], [math.sin(h), math.cos(h), 0], [0, 0, 1.]])   # body FRD -> NED (level)
        cam = pos + R @ np.array([t_bc[0], -t_bc[1], -t_bc[2]])
        p, (wd, ht) = tgt[kind]; d = R.T @ (p - cam)                         # FRD: x fwd, y right, z down
        u = cx + fx * d[1] / d[0] if d[0] > 0 else float('nan'); v = cy + fy * d[2] / d[0] if d[0] > 0 else float('nan')
        half_u = fx * (wd / 2) / max(d[0], 1e-3); half_v = fy * (ht / 2) / max(d[0], 1e-3)
        bearing = math.degrees(math.atan2(d[1], d[0])); elev = math.degrees(math.atan2(-d[2], math.hypot(d[0], d[1])))
        fits = (u - half_u >= 0) and (u + half_u <= 640) and (v - half_v >= 0) and (v + half_v <= 400)
        out.append(dict(kind=kind, hover_s=round(len(rows) * 0.55, 1), range_m=round(float(np.linalg.norm(d)), 2),
                        bearing_deg=round(bearing, 1), elevation_deg=round(elev, 1),
                        pixel=None if not (np.isfinite(u) and np.isfinite(v)) else [round(u), round(v)],
                        half_size_px=[round(half_u), round(half_v)], target_in_image=bool(fits),
                        hover_start_frame=[round(float(x), 2) for x in ((pos - home) @ np.array([c0, s0, 0]), (pos - home) @ np.array([s0, -c0, 0]), -(pos - home)[2])]))
    (F / 'sim' / 'view_check.json').write_text(json.dumps(out, indent=1))
    for r in out:
        print(F.name, json.dumps(r))


if __name__ == '__main__':
    main()
