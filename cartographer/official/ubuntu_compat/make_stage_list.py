#!/usr/bin/env python3
"""Per-STAGE test list for run_blind.sh --custom: each course stage started on its own, from a spawn pose computed from
the prior (course start frame), so stages can be flown separately and in parallel.

  python3 make_stage_list.py OUT_LIST --prior PRIOR (rel. to the code profile) --replica DIR [--stages a,b,..] [--reps N]
         [--nav-mode prior|map]                                    (in the container: truth clearance check of each spawn)

Stages (spawn = vehicle start pose F,L,YAW in the course start frame; start_phase -> stop_after):
  tunnel   1.1 m before the entry on the axis, facing along it          tunnel_align -> tunnel_traverse
  chicane  0.45 m past the tunnel exit on the axis, facing the 1st chicane point beyond the exit (else Gate B)   chicane -> chicane
  gate_b   1.0 m before the Gate B centre against through_deg           gate_pass -> gate_pass
  window   0.9 m before the window centre against through_deg           window_align -> window_traverse
           (if the prior's checkpoint4 is gate_b: flown on a copy with checkpoint4 = window, ubuntu_compat/<prior>.cp4window.json)
  pad      gate_b.after[-1] if checkpoint4 = gate_b, else window.after[0] / 0.8 m past the window; facing the pad   land_target
  wall     wall.cross_from, facing cross_to                             climb_over_wall -> done (incl. return_home, land_home)
  home     0.8 m past wall.cross_to (home side), facing the launch (0, 0)   return_home -> done
Each spawn must keep >= 0.35 m (vehicle centre) from every physical solid and the net in THIS world (truth); otherwise it is
moved back along its heading (home: away from the wall) in 0.1 m steps (reported). PARAMS: the mission-core interface for a stage start (Params
start_phase + the start pose in the prior frame) - the key names are set here in one place (START_PHASE_KEY, START_POSE_KEY).
"""
import argparse, importlib.util, json, math, sys
from pathlib import Path

START_PHASE_KEY = 'start_phase'          # mission-core Params field (interface agreed with mission-core 2026-11-08)
START_POSE_KEY = 'start_pose'            # (forward, left, yaw_deg CCW from prior forward): vehicle at arming, prior frame
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import course_solids as C  # noqa: E402


PHASES = {'tunnel': ('tunnel_align', 'tunnel_traverse'), 'chicane': ('chicane', 'chicane'), 'gate_b': ('gate_pass', 'gate_pass'),
          'window': ('window_align', 'window_traverse'), 'route': ('landing_route', 'landing_route'), 'pad': ('land_target', 'land_target'),
          'wall': ('climb_over_wall', None), 'home': ('return_home', None),     # start_phase, stop_after (None: to done)
          'chwin': ('chicane', 'window_traverse')}      # window stage from just past the tunnel exit (chicane spawn)
SPAWN_OF = {'chwin': 'chicane'}                         # stages that reuse another stage's spawn


def heading(p, q):
    return math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))


def stages(P):
    t = P['tunnel']; e = (t['entry']['forward'], t['entry']['left']); a = math.radians(t['axis_deg']); ln = t.get('length_m', 1.524)
    u = (math.cos(a), math.sin(a)); ex = (e[0] + ln * u[0], e[1] + ln * u[1])
    out = {'tunnel': ((e[0] - 1.1 * u[0], e[1] - 1.1 * u[1]), t['axis_deg'], 'tunnel_align', 'tunnel_traverse')}
    ch = [(c['forward'], c['left']) for c in P.get('chicane') or []]
    for i, c in enumerate(ch):                              # prior consistency: a chicane point inside the tunnel tube span
        s_ = (c[0] - e[0]) * u[0] + (c[1] - e[1]) * u[1]; lat = abs(-(c[0] - e[0]) * u[1] + (c[1] - e[1]) * u[0])
        if 0.0 < s_ < ln and lat < 0.9:
            print(f'# WARNING prior: chicane[{i}] {c} lies {s_:.2f} m into the tunnel (length {ln}), {lat:.2f} m off its axis')
    sp = (ex[0] + 0.45 * u[0], ex[1] + 0.45 * u[1])
    nxt = [c for c in ch if (c[0] - ex[0]) * u[0] + (c[1] - ex[1]) * u[1] > 0.3]
    tgt = nxt[0] if nxt else ((P['gate_b']['center']['forward'], P['gate_b']['center']['left']) if P.get('gate_b') else None)
    out['chicane'] = (sp, heading(sp, tgt) if tgt else t['axis_deg'], 'chicane', 'chicane')
    if P.get('gate_b'):
        g = P['gate_b']; c = (g['center']['forward'], g['center']['left']); th = math.radians(g['through_deg'])
        out['gate_b'] = ((c[0] - 1.0 * math.cos(th), c[1] - 1.0 * math.sin(th)), g['through_deg'], 'gate_pass', 'gate_pass')
    w = P['window']; c = (w['center']['forward'], w['center']['left']); th = math.radians(w['through_deg'])
    out['window'] = ((c[0] - 0.9 * math.cos(th), c[1] - 0.9 * math.sin(th)), w['through_deg'], 'window_align', 'window_traverse')
    pc = (P['landing_target']['center']['forward'], P['landing_target']['center']['left'])
    gb_after = (P.get('gate_b') or {}).get('after') or []
    wa = (w.get('after') or [None])[0]
    if P.get('checkpoint4') == 'gate_b' and gb_after:       # checkpoint 4 = Gate B: the pad is approached by gate_b.after
        ps = (gb_after[-1]['forward'], gb_after[-1]['left'])
    else:
        ps = (wa['forward'], wa['left']) if wa else (c[0] + 0.8 * math.cos(th), c[1] + 0.8 * math.sin(th))
    out['pad'] = (ps, heading(ps, pc), 'land_target', 'land_target')
    wl = P['wall']; cf, ct = (wl['cross_from']['forward'], wl['cross_from']['left']), (wl['cross_to']['forward'], wl['cross_to']['left'])
    out['wall'] = (cf, heading(cf, ct), 'climb_over_wall', None)          # climb_over_wall is no stop_after phase: to done
    uc = math.radians(heading(cf, ct)); hs = (ct[0] + 0.8 * math.cos(uc), ct[1] + 0.8 * math.sin(uc))
    out['home'] = (hs, heading(hs, (0.0, 0.0)), 'return_home', None, (math.cos(uc), math.sin(uc)))   # retreat away from the wall
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--prior', required=True); ap.add_argument('--replica', required=True)
    ap.add_argument('--stages', default='tunnel,chicane,gate_b,window,pad,wall,home'); ap.add_argument('--reps', type=int, default=1)
    ap.add_argument('--nav-mode', default=None); ap.add_argument('--profile', default='/compat/profile')
    a = ap.parse_args()
    P = json.loads((Path(a.profile) / a.prior).read_text())
    spec = importlib.util.spec_from_file_location('layout', Path(a.replica) / 'layout.py'); L = importlib.util.module_from_spec(spec)
    sys.modules['layout'] = L; spec.loader.exec_module(L)
    S = stages(P); lines = []
    helper = None                                           # mission-core's own spawns (same poses as its tests)
    try:
        sys.path.insert(0, a.profile); import mission.core as MC
        helper = getattr(MC, 'stage_start_pose', None)
    except Exception as e:  # noqa: BLE001
        print(f'# mission.core not importable ({e!r:.80}): own spawns')
    for name in a.stages.split(','):
        sk = SPAWN_OF.get(name, name)
        if sk not in S and name != 'route':
            print(f'# stage {name}: not available for this prior'); continue
        start, stop = PHASES[name]; up = 0.0; src = 'own'
        if helper:
            try:
                sp = helper(P, start); f, l, yaw = (float(v) for v in sp[:3]); up = float(sp[3]) if len(sp) > 3 else 0.0; src = 'mission.core'
            except Exception as e:  # noqa: BLE001
                print(f'# stage {name}: stage_start_pose failed ({e!r:.100})')
                if sk not in S:
                    continue
        if src == 'own':
            if sk not in S:
                continue
            (f, l), yaw = S[sk][0], S[sk][1]
        moved = 0.0
        rv = S[sk][4] if src == 'own' and len(S[sk]) > 4 else (-math.cos(math.radians(yaw)), -math.sin(math.radians(yaw)))
        for _ in range(12):                                 # truth clearance of the spawn (vehicle centre, hover height)
            ca, cb = f + L.START[0], -l + L.START[1]
            d = min(C.clearance(L, ca, cb, 0.85)[0], ca, L.CAGE - ca, cb, L.CAGE - cb)
            if d >= 0.35 or src != 'own':                   # mission-core's spawns are flown as given (only reported)
                break
            f += 0.1 * rv[0]; l += 0.1 * rv[1]; moved += 0.1
        if up > 0.01:
            print(f'# stage {name}: spawn up_m {up} (raised): the replica spawns on the floor')
        params = {START_PHASE_KEY: start, START_POSE_KEY: [round(f, 3), round(l, 3), round(yaw, 1)] + ([round(up, 3)] if up > 0.01 else [])}
        if stop:
            params['stop_after'] = stop
        if a.nav_mode:
            params['nav_mode'] = a.nav_mode
        prior = a.prior
        if name == 'window' and P.get('checkpoint4') != 'window':     # checkpoint 4 = Gate B in this prior: window variant
            q = dict(P, checkpoint4='window'); q['_stage_variant'] = f'{a.prior} with checkpoint4 = window (make_stage_list.py)'
            prior = f"ubuntu_compat/{Path(a.prior).stem}.cp4window.json"; (Path(a.profile) / prior).write_text(json.dumps(q, indent=1))
            print(f'# window stage: prior {prior} (checkpoint4 gate_b -> window)')
        for r in range(1, a.reps + 1):
            lines.append(f"{name}{r} {a.replica} {prior} {f:.3f},{l:.3f},{yaw:.1f} {json.dumps(params)}")
        print(f'{name:8s} spawn f {f:6.2f} l {l:6.2f} yaw {yaw:6.1f} ({src})  clearance {d:.2f} m{" LOW" if d < 0.35 else ""}'
              f'{f" (moved back {moved:.1f} m)" if moved else ""}  {start} -> {stop or "done"}')
    Path(a.out).write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
