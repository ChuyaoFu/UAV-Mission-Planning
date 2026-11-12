#!/usr/bin/env python3
"""Corner overshoot, prop clearance, contacts and landing progress of one flight (sim side, Gazebo truth).

  python3 corner_check.py FLIGHT_DIR          -> FLIGHT_DIR/sim/corner_check.json + summary

* Official metric (official lab grading): truth clearance of the vehicle CENTRE to the nearest obstacle while
  airborne; < 0.28 m (airframe reach) = FAIL even without physical contact -> official_min_centre_clearance_m,
  official_violations (intervals).
* Corners = interior vertices of start (0,0) + prior.json corridor (heading change > 20 deg). Per corner: the
  furthest the truth base_link went PAST the corner point along the incoming leg (m; > 0 = overshoot), the
  largest sideways offset from the incoming leg's line while within 0.6 m, and the minimum prop clearance there.
* Prop clearance = for each of the 4 rotor centres (body (+-0.10, +-0.10, 0.03), full attitude), Euclidean distance
  to the nearest PHYSICAL course solid (course_solids.py) or the net, minus the 0.0889 m prop radius; inside the
  tunnel (between first and last hoop) the distance to the mesh shell is used instead. < 0 = the disc overlaps.
* Contacts = IMU (/oak/imu) |acc| > 15 m/s^2 while airborne (truth height > 0.12 m), grouped within 0.3 s, AND a prop
  disc within 0.08 m of a solid / net / tunnel at that moment (else listed as imu_spikes_no_contact). Landing = phases reached after the window, last land_target events, result.
"""
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_flight import read, stamp  # noqa: E402
import course_solids as C  # noqa: E402
L = None                                     # set in main(): the layout this flight flew

ROTORS = np.array([[0.10, -0.10, 0.03], [-0.10, 0.10, 0.03], [0.10, 0.10, 0.03], [-0.10, -0.10, 0.03]])
PROP_R = 0.0889


def rotm(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def prop_clear(p, q):
    """min clearance (m) of the 4 prop discs + the name of the nearest thing."""
    R = rotm(q); best = (9.0, '')
    tg = C.tunnel_geom(L)
    for r in ROTORS:
        f, l, h = p + R @ r
        a, b = f + L.START[0], -l + L.START[1]
        d, n = C.clearance(L, a, b, h)
        net = min(a + L.NET_OFFSET, L.CAGE + L.NET_OFFSET - a, b + L.NET_OFFSET, L.CAGE + L.NET_OFFSET - b, L.CAGE_H - h)
        if net < d:
            d, n = net, 'net'
        s_ax, rr = C.tunnel_local(L, a, b, h)
        if 0.0 <= s_ax <= tg['length'] and rr < tg['mesh_r'] + 0.3:
            d, n = min((d, n), (abs(tg['mesh_r'] - rr), 'tunnel'))
        best = min(best, (d - PROP_R, n))
    return best


OFFICIAL_REACH = 0.28                        # official lab: truth clearance centre -> nearest obstacle < 0.28 m = contact (FAIL)


def centre_clear(p):
    """truth clearance (m) of the vehicle CENTRE to the nearest obstacle (physical solids, net, roof, tunnel shell)."""
    f, l, h = p; a, b = f + L.START[0], -l + L.START[1]
    d, n = C.clearance(L, a, b, h)
    net = min(a + L.NET_OFFSET, L.CAGE + L.NET_OFFSET - a, b + L.NET_OFFSET, L.CAGE + L.NET_OFFSET - b, L.CAGE_H - h)
    if net < d:
        d, n = net, 'net'
    tg = C.tunnel_geom(L); s_ax, rr = C.tunnel_local(L, a, b, h)
    if 0.0 <= s_ax <= tg['length'] and rr < tg['mesh_r'] + 0.3:
        d, n = min((d, n), (abs(tg['mesh_r'] - rr), 'tunnel'))
    return d, n


def main():
    global L
    F = Path(sys.argv[1]); prior = json.loads((F / 'prior.json').read_text())
    L = C.load_layout(F)
    S = read(F / 'sim' / 'sidebag', ['/gz/gt/odometry', '/oak/imu'])
    g = np.array([[stamp(m), m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z,
                   m.pose.pose.orientation.w, m.pose.pose.orientation.x, m.pose.pose.orientation.y, m.pose.pose.orientation.z]
                  for _, m in S['/gz/gt/odometry']])
    z0 = float(np.median(g[g[:, 0] < g[0, 0] + 3, 3]))
    air = g[:, 3] > 0.12
    k = np.where(air)[0][::4]                                       # ~ every 4th truth sample
    pc = [(float(g[i, 0]),) + prop_clear(g[i, 1:4], g[i, 4:8]) for i in k]
    tt = np.array([x[0] for x in pc]); dd = np.array([x[1] for x in pc])
    res = {'flight': F.name, 'z_rest': round(z0, 3)}
    j = int(dd.argmin()); res['min_prop_clearance_m'] = dict(d=round(float(dd[j]), 3), what=pc[j][2], t=round(tt[j], 2),
                                                                f=round(float(np.interp(tt[j], g[:, 0], g[:, 1])), 2),
                                                                l=round(float(np.interp(tt[j], g[:, 0], g[:, 2])), 2))
    cc = [(float(g[i, 0]),) + centre_clear(g[i, 1:4]) for i in k]                # official metric, same samples
    cd = np.array([x[1] for x in cc]); jc = int(cd.argmin())
    res['official_min_centre_clearance_m'] = dict(d=round(float(cd[jc]), 3), what=cc[jc][2], t=round(cc[jc][0], 2),
                                                  fail=bool(cd[jc] < OFFICIAL_REACH), reach_m=OFFICIAL_REACH)
    viol, cur = [], None                      # intervals with centre clearance < 0.28 m
    for (t, d, n) in cc:
        if d < OFFICIAL_REACH:
            if cur is None or t - cur['t1'] > 0.5:
                cur = dict(t0=round(t, 2), t1=t, min=d, what=n); viol.append(cur)
            cur['t1'] = t
            if d < cur['min']: cur['min'], cur['what'] = d, n
    res['official_violations'] = [dict(t0=v['t0'], t1=round(v['t1'], 2), min_m=round(v['min'], 3), what=v['what']) for v in viol]
    neg = dd < 0
    res['prop_overlap_s'] = round(float(neg.sum()) * float(np.median(np.diff(tt))) if len(tt) > 1 else 0.0, 2)
    # corners
    P = [np.zeros(2)] + [np.array([c['forward'], c['left']]) for c in prior['corridor']]
    paths = [('corridor', P)]
    if prior.get('chicane'):                                        # v4: tunnel exit -> chicane -> window centre
        te, ax = prior['tunnel']['entry'], math.radians(prior['tunnel']['axis_deg'])
        ex = np.array([te['forward'], te['left']]) + prior['tunnel']['length_m'] * np.array([math.cos(ax), math.sin(ax)])
        cp = prior.get('checkpoint4') or ('gate_b' if prior.get('gate_b') else 'window')
        wc = prior['gate_b']['center'] if cp == 'gate_b' else prior['window']['center']   # chicane leads to checkpoint 4
        paths.append(('chicane', [ex] + [np.array([c['forward'], c['left']]) for c in prior['chicane']] + [np.array([wc['forward'], wc['left']])]))
    out = []
    for path, P in paths:
      for i in range(1, len(P) - 1):
          u, w = P[i] - P[i - 1], P[i + 1] - P[i]
          if np.linalg.norm(u) < 1e-6 or np.linalg.norm(w) < 1e-6:
              continue
          ang = math.degrees(math.acos(max(-1, min(1, float(u @ w) / np.linalg.norm(u) / np.linalg.norm(w)))))
          if ang < 20:
              continue
          u = u / np.linalg.norm(u); nrm = np.array([-u[1], u[0]])
          rel = g[:, 1:3] - P[i]; near = air & (np.linalg.norm(rel, axis=1) < 0.6)
          if not near.any():
              out.append(dict(path=path, vertex=i, corner=P[i].round(3).tolist(), turn_deg=round(ang), reached=False)); continue
          past = rel[near] @ u; side = rel[near] @ nrm
          w_ = (tt >= g[near, 0].min()) & (tt <= g[near, 0].max())
          out.append(dict(path=path, vertex=i, corner=P[i].round(3).tolist(), turn_deg=round(ang), reached=True,
                          max_past_corner_m=round(float(past.max()), 3), max_side_off_m=round(float(np.abs(side).max()), 3),
                          t_window=[round(float(g[near, 0].min()), 1), round(float(g[near, 0].max()), 1)],
                          min_prop_clearance_m=round(float(dd[w_].min()), 3) if w_.any() else None))
    res['corners'] = out
    # contacts: IMU spikes while airborne
    imu = np.array([[stamp(m), m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z] for _, m in S['/oak/imu']])
    an = np.linalg.norm(imu[:, 1:], axis=1); hi = np.interp(imu[:, 0], g[:, 0], g[:, 3]) > 0.12
    ts = imu[(an > 15) & hi, 0]; ev = []
    for t in ts:
        if not ev or t - ev[-1][-1] > 0.3:
            ev.append([t, t])
        else:
            ev[-1][-1] = t
    cont = []
    for t0, t1 in ev:
        i = int(np.argmin(np.abs(g[:, 0] - t0)))
        d, n = prop_clear(g[i, 1:4], g[i, 4:8])
        cont.append(dict(t=round(t0, 2), peak_acc=round(float(an[(imu[:, 0] >= t0) & (imu[:, 0] <= t1)].max()), 1),
                         f=round(float(g[i, 1]), 2), l=round(float(g[i, 2]), 2), h=round(float(g[i, 3]), 2), nearest=n, prop_clear_m=round(d, 3)))
    # a spike is a CONTACT only with a prop disc within 0.08 m of something (else: hard manoeuvre, e.g. stop at a corner)
    res['contacts'] = [c_ for c_ in cont if c_['prop_clear_m'] < 0.08]
    res['imu_spikes_no_contact'] = [c_ for c_ in cont if c_['prop_clear_m'] >= 0.08]
    # landing progress
    ph, lt = [], []
    for ln in (F / 'mission.log').read_text().splitlines():
        if not ln.startswith('{'):
            continue
        e = json.loads(ln.replace('NaN', 'null')); ev_ = e.get('event', e)
        if ev_.get('kind') == 'phase':
            ph.append((round(ev_['t'], 1), ev_['to'], ev_.get('reason')))
        elif ev_.get('phase') == 'land_target' and ev_.get('kind') not in ('minor_obstacle',):
            lt.append({k: v for k, v in ev_.items() if k in ('t', 'kind', 'reason', 'attempt', 'observe_dist_m', 'reasons', 'free_m', 'center', 'moved')})
    res['phases'] = ph; res['land_target_events'] = lt[-12:]
    try:
        res['result'] = json.loads((F / 'result.json').read_text())
    except Exception:  # noqa: BLE001
        pass
    (F / 'sim' / 'corner_check.json').write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ('min_prop_clearance_m', 'prop_overlap_s', 'official_min_centre_clearance_m')}))
    for v in res['official_violations']:
        print('official < 0.28 m', v)
    for c in out:
        print('corner', c)
    for c in res['contacts']:
        print('contact', c)
    for c in res['imu_spikes_no_contact']:
        print('imu spike (no contact)', c)
    print('phases:', ' -> '.join(f'{p[1]}@{p[0]}' for p in ph))
    for e in lt[-6:]:
        print('land', json.dumps(e, default=str)[:200])


if __name__ == '__main__':
    main()
