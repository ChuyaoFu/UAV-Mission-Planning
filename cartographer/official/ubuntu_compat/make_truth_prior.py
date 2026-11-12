#!/usr/bin/env python3
"""Truth-derived course priors for the venue replica (SIM-SIDE reference only; real runs use measured priors).

  python3 make_truth_prior.py [REPLICA_DIR]   (default ../replica; needs numpy for layout.py)
      -> course_prior.replica.json  (full course, truth)   +   course_prior.hover.json (hover-and-land check)

Layout v4 (2026-11-07, organiser top-view map, cartographer/official/COURSE_MAP.md): the route is the layout's
ROUTE_FT (map path + venue-replica's continuation through the stages), split as
  corridor = ROUTE_FT after the launch up to the point before the tunnel's north-face approach
             (first corridor, serpentine maze, Gate A, east along the north net to above the tunnel: the mission
             adds the tunnel standoff itself);
  tunnel   = layout TUNNEL (first hoop = entry, axis_ab -> axis_deg, length);
  chicane  = ROUTE_FT after the tunnel exit up to the branch point GATE_B_ROUTE_FT[0] (around C1's south end),
             + a point in the middle of the C1-end / C2 gap (the straight line to Gate B clips C1's end);
  gate_b   = Gate B (x 10.5 ft, y 8.1-11.5), checkpoint4 'gate_b' (mission default), exit_m 0.45, after = around W4's
             south end (our route; >= 0.40 m from boards); window.after = ROUTE_FT after the window (to the pad);
  window   = layout WINDOW centre at hole height, through = -normal_ne;   landing = PAD centre;
  wall     = the barrier (map y 11.5 ft): crossed on the straight line pad -> launch, 0.6 m before / 0.35 m after;
  gate_b, boulders, walls = extra keys in the format of mission/course_prior.venue_map.json (walls = the map's wall
             segments barrier, W1-W4, C1, C2_north; walls_tol_m 0.3; the window board is not listed, as in venue_map).
'up' values are above base_link at rest = floor + 0.05 m (the node uses floor_d = home + 0.05); so the floor pad's
top is top_up = thick - 0.05 = -0.045 (mission-core's example writes 0 = "floor"; 4.5 cm difference).
Every corridor/chicane goal is checked: Euclidean clearance >= 0.35 m to every physical solid and to the net.
"""
import json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARGS = [x for x in sys.argv[1:] if not x.startswith('--name')]
NAME = next((x.split('=', 1)[1] for x in sys.argv[1:] if x.startswith('--name=')), None)
REP = Path(ARGS[0]) if ARGS else HERE.parent / 'replica'
sys.path.insert(0, str(REP)); sys.path.insert(0, str(HERE))
import layout as L  # noqa: E402
import course_solids  # noqa: E402

B = 0.05                                     # base_link rest height above the floor
CRUISE = 0.8
S = L.START


def fl(a, b):
    return round(a - S[0], 3), round(-(b - S[1]), 3)


def solid_clearance(a, b, h):
    """Euclidean distance (m) from cage point (a, b, h) to the nearest physical course solid (course_solids.py;
    tunnel excluded). (layout.pose_is_free expands boxes per axis, which is stricter at wall ends.)"""
    return course_solids.clearance(L, a, b, h)[0]


def net_clearance(a, b):
    return min(a, L.CAGE - a, b, L.CAGE - b)


def split_route():
    R = [L.ft_ab(*p) for p in L.ROUTE_FT]                     # cage (a, b)
    tg = course_solids.tunnel_geom(L); (ea, eb), (ua, ub), ln = tg['entry'], tg['axis'], tg['length']
    xa, xb = ea + ua * ln, eb + ub * ln
    d_in = [math.hypot(a - ea, b - eb) for a, b in R]; d_out = [math.hypot(a - xa, b - xb) for a, b in R]
    i_in = min(i for i, d in enumerate(d_in) if d < 0.5)      # approach point at the tunnel's entry face
    i_out = min(i for i, d in enumerate(d_out) if d < 0.1)    # the exit face itself
    w = L.WINDOW
    i_win = min(range(len(R)), key=lambda i: math.hypot(R[i][0] - w['a'], R[i][1] - w['b']))
    corridor = R[1:i_in]
    br = L.ft_ab(*L.GATE_B_ROUTE_FT[0])                       # branch point: Gate B route leaves the window route here
    i_br = min(range(len(R)), key=lambda i: math.hypot(R[i][0] - br[0], R[i][1] - br[1]))
    assert math.hypot(R[i_br][0] - br[0], R[i_br][1] - br[1]) < 0.05, 'GATE_B_ROUTE_FT[0] is not on ROUTE_FT'
    # + one point in the middle of the C1 south end - C2 north gap (~1 m wide at map y 6.3 ft): the straight line
    # from the branch point to Gate B's line-up passes 0.05-0.2 m from C1's south end (demo 20261107-230517 blocked)
    chicane = R[i_out + 1:i_br + 1] + [L.ft_ab(11.52, 5.66)]
    return corridor, chicane, R[i_win + 1:]


def coarse(prior):
    """COARSE prior for a randomised maze: checkpoint order + coarse regions only. No maze walls, no corridor
    waypoints; the maze is a region with an exit (Gate A, +-1 m); later checkpoints as on the map."""
    P = json.loads(json.dumps(prior))
    x0, x1, y0, y1 = getattr(L, 'MAZE_REGION_FT', (0.0, 11.0, 11.5, 20.0)); gx, gy0, gy1 = getattr(L, 'MAZE_EXIT_FT', (11.0, 16.9, 20.0))
    (f0, l0), (f1, l1) = fl(*L.ft_ab(x0, y0)), fl(*L.ft_ab(x1, y1))
    ef, el = fl(*L.ft_ab(gx, (gy0 + gy1) / 2))
    P['corridor'] = []
    P.pop('walls', None); P.pop('walls_tol_m', None)
    P['maze'] = {"region": {"forward_min": min(f0, f1), "forward_max": max(f0, f1), "left_min": min(l0, l1), "left_max": max(l0, l1)},
                 "exit": {"forward": ef, "left": el, "tol_m": 1.0}, "exit_through_deg": -90.0,
                 "_comment": "maze interior unknown (organisers may change it); leave the region through the exit (Gate A) heading east (-l)"}
    P['_scale'] = 'maze-coarse'
    te, tb, wn = P['tunnel']['entry'], P.get('gate_b'), P['window']
    P['checkpoints'] = [                                   # mission-core schema (CoursePrior.checkpoints); map mode flies
        {"name": "gate_a", "kind": "gate", "center": {"forward": ef, "left": el}, "through_deg": -90.0, "search_m": 1.0, "order": 1},
        {"name": "tunnel", "kind": "tunnel", "center": {"forward": te['forward'], "left": te['left']}, "order": 2}]   # those before the tunnel
    if tb:
        P['checkpoints'].append({"name": "gate_b", "kind": "gate", "center": {"forward": tb['center']['forward'], "left": tb['center']['left']},
                                 "through_deg": tb['through_deg'], "search_m": 1.0, "order": 3})
    P['checkpoints'] += [{"name": "window", "kind": "gate", "center": {"forward": wn['center']['forward'], "left": wn['center']['left']},
                          "through_deg": wn['through_deg'], "search_m": 1.0, "order": 4},
                         {"name": "pad", "kind": "region", "center": dict(P['landing_target']['center']), "search_m": 0.78, "order": 5}]
    P['checkpoint_order'] = ['maze', 'gate_a', 'tunnel', 'chicane', 'gate_b_or_window', 'pad', 'barrier', 'home']
    P['_comment'] = ('COARSE prior (random-maze test bed): checkpoint order + coarse regions only; NO maze walls, NO corridor '
                     'waypoints; maze region + exit (Gate A) +-1 m; later stages from the organiser map. ' + P['_comment'][:120])
    return P


def main():
    t, w, wl, pad = L.TUNNEL, L.WINDOW, L.WALL, L.PAD
    tg = course_solids.tunnel_geom(L)
    corridor, chicane, after_window = split_route()
    # Gate B landing route ('after'): around W4's SOUTH end (our route choice; the layout's GATE_B_ROUTE_FT passes 0.3 m
    # from it), map ft -> cage; then the landing observe-pose search takes over
    gate_after = [L.ft_ab(9.1, 8.75), L.ft_ab(8.2, 6.4)]
    for name, pts, need in (('corridor', corridor, 0.35), ('chicane', chicane, 0.40), ('gate_b.after', gate_after, 0.40),
                            ('window.after', after_window, 0.40)):
        for a, b in pts:
            d = solid_clearance(a, b, CRUISE + B + 0.029); n = net_clearance(a, b)
            assert min(d, n) >= need, (f'{name} goal closer than {need} m to a solid/net', fl(a, b), round(d, 3), round(n, 3))
    te = fl(*tg['entry'])
    ax_f, ax_l = tg['axis'][0], -tg['axis'][1]
    nn, ne = w['normal_ne']                                   # NED, to the approach side; travel = -normal
    wc = fl(w['a'], w['b'])
    pf, pl = round(pad['n'], 3), round(-pad['e'], 3)
    wall_f = wl['y'] * L.FT - S[0]                            # barrier plane (map y) in the start frame

    def on_line(f):                                           # point at forward f on the line pad -> launch (0, 0)
        return {"forward": round(f, 3), "left": round(pl * f / pf, 3)}
    g0, g1 = L.ft_ab(10.5, 8.1), L.ft_ab(10.5, 11.5)          # Gate B (COURSE_MAP.md): x 10.5, y 8.1 -> 11.5
    gb = fl((g0[0] + g1[0]) / 2, (g0[1] + g1[1]) / 2)
    C = L.CAGE
    prior = {
        "_comment": ("Venue replica layout v4 (replica/layout.py = organiser top-view map, COURSE_MAP.md) in the START frame "
                     "(forward = north from the launch, left = west, up = above base_link at rest = floor + 0.05 m). "
                     "corridor = map path (first corridor, serpentine maze, Gate A) + east along the north net to above the tunnel; "
                     "chicane = tunnel exit -> around C1's south end to the Gate B / window branch point; checkpoint4 gate_b (default), gate_b.after = around W4's south end, window.after = layout path to the pad (format of course_prior.venue_map.json); "
                     "fence = cage net. TRUTH prior: sim reference only. Generated by ubuntu_compat/make_truth_prior.py."),
        "frame": "start", "tolerance_m": 0.5, "cruise_up_m": CRUISE,
        "fence": {"forward_min": round(-S[0], 3), "forward_max": round(C - S[0], 3), "left_min": round(-(C - S[1]), 3),
                  "left_max": round(S[1], 3), "up_max": round(L.CAGE_H - B - 0.05, 2)},
        "corridor": [{"forward": f, "left": l, "up": CRUISE} for f, l in (fl(a, b) for a, b in corridor)],
        "tunnel": {"entry": {"forward": te[0], "left": te[1], "up": round(t['centre_h'] - B, 3)},
                   "axis_deg": round(math.degrees(math.atan2(ax_l, ax_f)) % 360 if abs(abs(math.degrees(math.atan2(ax_l, ax_f))) - 180) < 1e-6 else math.degrees(math.atan2(ax_l, ax_f)), 1), "length_m": round(tg['length'], 3), "tol_m": 0.5},
        "chicane": [{"forward": f, "left": l, "up": CRUISE} for f, l in (fl(a, b) for a, b in chicane)],
        "gate_b": {"center": {"forward": gb[0], "left": gb[1], "up": 0.9}, "through_deg": 90.0,
                   "width_m": round(math.hypot(g1[0] - g0[0], g1[1] - g0[1]), 3),
                   "exit_m": 0.45, "after": [{"forward": f, "left": l, "up": CRUISE} for f, l in (fl(a, b) for a, b in gate_after)]},
        "window": {"center": {"forward": wc[0], "left": wc[1], "up": round(w['hole_c'] - B, 3)},
                   "through_deg": round(math.degrees(math.atan2(ne, -nn)), 1), "tol_m": 0.5,
                   "after": [{"forward": f, "left": l, "up": CRUISE} for f, l in (fl(a, b) for a, b in after_window)]},
        "checkpoint4": "gate_b",
        "landing_target": {"center": {"forward": pf, "left": pl}, "top_up": round(pad['thick'] - B, 3), "tol_m": 0.5,
                           "ring_radius_m": pad.get('ring_r')},
        "wall": {"cross_from": on_line(wall_f - 0.6), "cross_to": on_line(wall_f + 0.35),
                 "top_up": round(wl['top'] - B, 3), "tol_m": 0.3},
        "boulders": [{"forward": f, "left": l, "height_m": h, "radius_m": r} for (a, b, h, r) in L.BOULDERS for f, l in [fl(a, b)]],
        "walls_tol_m": 0.3,
        "walls": [{"name": n, "from": {"forward": round(p0[0], 3), "left": round(p0[1], 3)}, "to": {"forward": round(p1[0], 3), "left": round(p1[1], 3)}}
                  for (n, p0, p1, _sk, _ro) in L.MAZE_WALLS],
    }
    if NAME:                                                  # per-world priors (random mazes): truth + COARSE, no hover
        (HERE / f'course_prior.{NAME}.json').write_text(json.dumps(prior, indent=1) + '\n')
        (HERE / f'course_prior.{NAME}.coarse.json').write_text(json.dumps(coarse(prior), indent=1) + '\n')
        print(json.dumps({k: prior[k] for k in ('corridor', 'tunnel', 'checkpoint4')}))
        return
    (HERE / 'course_prior.replica.json').write_text(json.dumps(prior, indent=1) + '\n')
    hov = json.loads(json.dumps(prior))
    hov['_comment'] = ("Hover-and-land check in the venue replica (layout v4): tunnel standoff (entry - standoff along axis 0 deg) "
                       "= above the launch, so mission.core takes off, hovers in tunnel_align (no tunnel in view) until its phase "
                       "timeout and lands. Later stages copied from course_prior.replica.json (never reached).")
    hov['corridor'] = [{"forward": 0.0, "left": 0.0, "up": CRUISE}]
    hov['tunnel'] = {"entry": {"forward": 1.8, "left": 0.0, "up": CRUISE}, "axis_deg": 0.0, "length_m": 1.2, "tol_m": 0.5}
    for k in ('chicane', 'gate_b', 'checkpoint4'):
        hov.pop(k, None)
    (HERE / 'course_prior.hover.json').write_text(json.dumps(hov, indent=1) + '\n')
    print(json.dumps({k: prior[k] for k in ('corridor', 'tunnel', 'chicane', 'gate_b', 'window', 'checkpoint4', 'landing_target', 'wall')}))


if __name__ == '__main__':
    main()
