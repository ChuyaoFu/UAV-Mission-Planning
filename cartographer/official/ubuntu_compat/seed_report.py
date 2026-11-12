#!/usr/bin/env python3
"""Per-flight stage table for the prior-perturbation (anti-overfitting) runs.

  python3 seed_report.py FLIGHT_DIR [...]      -> table on stdout + FLIGHT_DIR/sim/seed_report.json

For each stage: prior error (flown prior - replica truth, start frame), the detector's latched estimate error
(latched - truth) and the outcome. Truth = ubuntu_compat/course_prior.replica.json (generated from replica/layout.py).
Latched centres come from mission.log 'latched' events (PX4 NED, start-anchored: f = n, l = -e, up = -d).
"""
import json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRUTH = json.loads((HERE / 'course_prior.replica.json').read_text())
STAGE_OF = [('corridor', 'corridor'), ('tunnel', 'tunnel'), ('window', 'window'), ('land', 'landing'), ('landing', 'landing'),
            ('wall', 'wall'), ('return', 'return_home'), ('home', 'return_home'), ('geofence', 'geofence'), ('pose', 'estimator')]


def flu(p):
    return [p['forward'], p['left'], p.get('up')]


def main():
    rows = []
    for f in sys.argv[1:]:
        F = Path(f); prior = json.loads((F / 'prior.json').read_text()); res = json.loads((F / 'result.json').read_text())
        meta = json.loads((F / 'meta.json').read_text())
        ana = json.loads((F / 'sim' / 'analysis.json').read_text()) if (F / 'sim' / 'analysis.json').exists() else {}
        stack = json.loads((F / 'sim' / 'stack.json').read_text()) if (F / 'sim' / 'stack.json').exists() else {}
        seed = prior.get('_seed', 'truth')
        pe = {}
        for key, tp, fp in (('tunnel', TRUTH['tunnel']['entry'], prior['tunnel']['entry']),
                            ('window', TRUTH['window']['center'], prior['window']['center']),
                            ('landing', TRUTH['landing_target']['center'], prior['landing_target']['center'])):
            pe[key] = round(math.hypot(fp['forward'] - tp['forward'], fp['left'] - tp['left']), 3)
        pe['tunnel_axis_deg'] = round(prior['tunnel']['axis_deg'] - TRUTH['tunnel']['axis_deg'], 2)
        pe['window_dir_deg'] = round(prior['window']['through_deg'] - TRUTH['window']['through_deg'], 2)
        pe['corridor_max'] = round(max(math.hypot(a['forward'] - b['forward'], a['left'] - b['left'])
                                       for a, b in zip(prior['corridor'], TRUTH['corridor'])), 3)
        pe['wall_max'] = round(max(math.hypot(prior['wall'][k]['forward'] - TRUTH['wall'][k]['forward'],
                                              prior['wall'][k]['left'] - TRUTH['wall'][k]['left']) for k in ('cross_from', 'cross_to')), 3)
        lat, phases, abort = {}, [], None
        for ln in (F / 'mission.log').read_text().splitlines():
            if not (ln.startswith('{') and '"event"' in ln):
                continue
            e = json.loads(ln.replace('NaN', 'null'))['event']
            if e.get('kind') == 'latched':
                c = e['center']; f_, l_, u_ = c[0], -c[1], -c[2]
                tk = {'tunnel': TRUTH['tunnel']['entry'], 'window': TRUTH['window']['center'],
                      'landing': dict(TRUTH['landing_target']['center'], up=TRUTH['landing_target']['top_up'])}[e['detector']]
                lat[e['detector']] = dict(xy_err=round(math.hypot(f_ - tk['forward'], l_ - tk['left']), 3),
                                          z_err=round(u_ - tk['up'], 3) if tk.get('up') is not None else None, t=e.get('t'))
            if e.get('kind') == 'phase':
                phases.append(e['to'])
            if e.get('kind') == 'abort':
                abort = e.get('reason')
        stage = next((s for k, s in STAGE_OF if abort and k in abort), None) if abort else None
        t_arm = next((p['t'] for p in ana.get('phases', []) if p['to'] == 'takeoff'), None)
        t_done = next((p['t'] for p in ana.get('phases', []) if p['to'] == 'done'), None)
        r = dict(flight=F.name, mode=f"{stack.get('images', 'gz')}/{stack.get('depth', 'ideal')}", seed=seed,
                 finished=res.get('finished'), abort=abort, failed_stage=stage, last_phase=phases[-3:] if phases else None,
                 time_to_done_s=round(t_done - t_arm, 1) if t_done and t_arm else None, prior_err_m=pe, latched_err=lat,
                 tunnel_pass=ana.get('tunnel_pass'), window_pass=ana.get('window_pass'), stone=ana.get('stone_touchdown'),
                 wall=ana.get('wall_pass'), home_err_m=ana.get('truth_end_xy_from_start_m') if t_done else None,
                 code_sha=meta.get('code_sha'))
        (F / 'sim' / 'seed_report.json').write_text(json.dumps(r, indent=1, default=str))
        rows.append(r)
        print(json.dumps(r, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
