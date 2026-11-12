#!/usr/bin/env python3
"""One line per flight for the round report (reads sim/seed_report.json, analysis.json, obstacles.json,
gate_replay.json, events_evidence.json and mission.log).

  python3 round_table.py FLIGHT_DIR [...]       -> markdown table rows on stdout + JSON list (round_table.json in cwd)
"""
import collections, json, sys
from pathlib import Path


def j(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return {}


rows = []
for f in sys.argv[1:]:
    F = Path(f); s = j(F / 'sim/seed_report.json'); a = j(F / 'sim/analysis.json'); o = j(F / 'sim/obstacles.json')
    g = j(F / 'sim/gate_replay.json').get('summary', {}); ev = j(F / 'sim/events_evidence.json') or []
    kinds = collections.Counter()
    plane = []
    for ln in (F / 'mission.log').read_text().splitlines():
        if ln.startswith('{') and '"event"' in ln:
            e = json.loads(ln.replace('NaN', 'null'))['event']; k = e.get('kind') or e.get('type')
            if k in ('avoid', 'blocked', 'minor_obstacle', 'look_before_side_step', 'parallax_nudge', 'nudge', 'standoff_clipped',
                     'standoff_shrunk', 'corridor_goal_pulled_in', 'commit', 'wall_higher', 'abort'):
                kinds[k] += 1
            if k == 'plane_crossing':
                plane.append({x: e.get(x) for x in ('stage', 'lateral_m', 'vertical_m') if x in e})
    blocked = [(r['t'], r.get('free_m'), r.get('probe_phantom'), r.get('probe_real')) for r in ev if r['kind'] == 'blocked']
    acc = None
    r = dict(flight=F.name, seed=s.get('seed'), finished=s.get('finished'), abort=s.get('abort'), t_done=s.get('time_to_done_s'),
             prior=s.get('prior_err_m'), latched=s.get('latched_err'), tunnel=a.get('tunnel_pass'), window=a.get('window_pass'),
             stone=a.get('stone_touchdown'), wall=a.get('wall_pass'), home=s.get('home_err_m'), ekf_xy_max=(a.get('px4_est_err') or {}).get('xy_max'),
             phantom_med=o.get('phantom_med'), phantom_frac=o.get('phantom_frac_med'), events=dict(kinds), blocked=blocked, plane_crossing=plane,
             gate={k: (v.get('correct'), v.get('correct_rejected'), v.get('false'), v.get('false_accepted'), v.get('gate_needed_for_all_correct_m'))
                   for k, v in g.items()})
    rows.append(r)
    t = r['tunnel'] or {}; w = r['window'] or {}; st = r['stone'] or {}; wl = r['wall'] or {}; p = r['prior'] or {}
    print(f"| {F.name} | {r['seed']} | {p.get('tunnel')}/{p.get('window')}/{p.get('landing')} | "
          f"{r['finished']}{'' if not r['abort'] else ' ' + r['abort']} | {r['t_done']} | "
          f"{t.get('max_radial_off_m')} | {w.get('lateral_off_m')}/{w.get('vertical_off_m')} (clr {w.get('clear_side_m')}) | "
          f"{st.get('off_center_m')} | {wl.get('skids_over_top_m')} | {r['home']} | {dict(kinds)} | {r['phantom_frac']} | "
          f"rej {sum(v[1] or 0 for v in r['gate'].values())} |")
Path('round_table.json').write_text(json.dumps(rows, indent=1, default=str))
