#!/usr/bin/env python3
"""One markdown row per v4 flight from sim/corner_check.json + sim/analysis.json + sim/vio_error.json + result.json.

  python3 v4_summary.py FLIGHT_DIR [...]      (run corner_check.py, analyze_flight.py, vio_error.py first)

Columns: flight | prior | result (+ reason) | furthest stage passed (maze > tunnel > cp4 (gate B / window) > pad > barrier > home) | arm -> end time | contacts (IMU > 15 m/s^2 airborne, nearest thing) |
maze corners: min prop clearance per corridor corner (m) | tunnel max off-axis | Gate B crossing offset + prop clearance | window lateral / vertical |
pad touchdown off-centre (+ whole drone inside the 0.78 m ring?) | return home error | VIO xy RMS.
With several flights: a pass-rate table per prior level (share of flights past each stage, contacts, crashes).
"""
import collections, json, sys
from pathlib import Path

# stage k is PASSED when the flight reached the phase that follows it
# staged_stop = a maze-only run (stop_after corridor) completed the maze
STAGES = [('maze', ('tunnel_align', 'staged_stop')), ('tunnel', ('chicane', 'window_align', 'gate_pass')),
          ('cp4', ('landing_route', 'land_target')), ('pad', ('disarm',)), ('barrier', ('return_home',)), ('home', ('done',))]


def j(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return {}


def row(F):
    F = Path(F); c = j(F / 'sim/corner_check.json'); a = j(F / 'sim/analysis.json'); v = j(F / 'sim/vio_error.json')
    r = j(F / 'result.json'); pr = j(F / 'prior.json')
    ph = c.get('phases', [])
    t_arm = next((t for t, to, _ in ph if to == 'takeoff'), None); t_end = ph[-1][0] if ph else None
    last = ph[-1][1] if ph else '?'; why = ph[-1][2] if ph else ''
    res = 'done' if last == 'done' else f'{last} ({why})'
    reached = [p[1] for p in ph]
    cont = c.get('contacts', [])
    cs = '; '.join(f"t{x['t']:.0f} {x['nearest']}" for x in cont) or '0'
    mz = ' / '.join(f"{x['min_prop_clearance_m']:.2f}" if x.get('reached') and x.get('min_prop_clearance_m') is not None else '-'
                    for x in c.get('corners', []) if x.get('path') == 'corridor')
    tp = a.get('tunnel_pass', {}); wp = a.get('window_pass', {}); sd = a.get('stone_touchdown', {})
    tun = f"{100 * tp['max_radial_off_m']:.1f} cm" if tp else ('-' if 'tunnel_traverse' not in reached else '?')
    win = f"{100 * wp['lateral_off_m']:+.1f} / {100 * wp['vertical_off_m']:+.1f} cm" if wp else '-'
    gp = a.get('gate_b_pass', {})
    gate = f"{100 * gp['offset_from_centre_m']:+.1f} cm (prop clear {gp['prop_clear_each_side_m']:.2f} m)" if gp else '-'
    pad = (f"{100 * sd['off_center_m']:.1f} cm ({'in ring' if sd['off_center_m'] + 0.23 < 0.78 else 'OUT of ring'})" if sd else '-')
    home = f"{100 * a['truth_end_xy_from_start_m']:.1f} cm" if last == 'done' and 'truth_end_xy_from_start_m' in a else '-'
    vio = f"{v['vio']['xy_rms_cm']:.1f} cm" if v.get('vio') else '-'
    tag = f"{pr.get('_scale', 'truth')} {pr.get('_seed', '')}".strip()
    dur = f"{t_end - t_arm:.1f} s" if t_arm is not None and t_end is not None else '-'
    passed = []
    for name, nxt in STAGES:
        if any(n in reached for n in nxt) or (name == 'maze' and a.get('gate_a_pass')): passed.append(name)   # maze: or truth crossed the exit
        else: break
    cp4 = 'gate B' if 'gate_pass' in reached else ('window' if 'window_align' in reached else '-')
    stage = (passed[-1] if passed else 'none') + (f' ({cp4})' if 'cp4' in passed else '')
    crash = 'released' in reached
    mp = c.get('min_prop_clearance_m') or {}
    oc = c.get('official_min_centre_clearance_m') or {}
    ocs = (f"{oc['d']:.2f} m ({oc.get('what', '')}){' **FAIL**' if oc.get('fail') else ''}" if oc else '-')
    mpc = f"{mp['d']:.2f} m ({mp.get('what', '')})" if mp else '-'
    return (f"| {F.name} | {tag} | {res} | {stage} | {dur} | {cs}{' **CRASH**' if crash else ''} | {ocs} | {mpc} | {mz} | {tun} | {gate} | {win} | {pad} | {home} | {vio} |",
            dict(flight=F.name, prior=tag, scale=pr.get('_scale', 'truth'), result=res, done=last == 'done', passed=passed, cp4=cp4,
                 crash=crash, duration_s=dur, contacts=len(cont), phases=reached, min_prop=mp.get('d'),
                 official_fail=bool(oc.get('fail')), official_min=oc.get('d'),
                 tunnel=tp, gate_b=gp, window=wp, pad=sd, home=home, vio=vio))


def pass_table(rows):
    """per prior scale: share of flights that passed each stage, contacts, crashes, median min prop clearance"""
    g = collections.defaultdict(list)
    for d in rows: g[d['scale']].append(d)
    out = ['', '| prior level | n | ' + ' | '.join(n for n, _ in STAGES) + ' | flights with contact | official < 0.28 m (FAIL) | crashes | official-clean full course | median min centre clearance | median min prop clearance |',
           '|---|---|' + '---|' * len(STAGES) + '---|---|---|---|---|---|']
    for k, ds in sorted(g.items()):
        n = len(ds); mp = sorted(x['min_prop'] for x in ds if x['min_prop'] is not None)
        cells = [f"{sum(1 for x in ds if st in x['passed'])}/{n}" for st, _ in STAGES]
        oc = sorted(x['official_min'] for x in ds if x['official_min'] is not None)
        clean = sum(1 for x in ds if 'home' in x['passed'] and not x['official_fail'] and not x['contacts'])
        out.append(f"| {k} | {n} | " + ' | '.join(cells) + f" | {sum(1 for x in ds if x['contacts'])} | {sum(1 for x in ds if x['official_fail'])} | "
                   f"{sum(1 for x in ds if x['crash'])} | {clean}/{n} | " + (f"{oc[len(oc) // 2]:.2f} m" if oc else '-') + ' | '
                   + (f"{mp[len(mp) // 2]:.2f} m" if mp else '-') + ' |')
    return '\n'.join(out)


if __name__ == '__main__':
    print('| flight | prior | result | furthest stage passed | arm→end | contacts | official centre clearance (< 0.28 m = FAIL) | min prop clearance | maze corner prop clearance (m) | tunnel off-axis | Gate B offset | window lat / vert | pad off-centre | home | VIO xy RMS |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    out = []
    for f in sys.argv[1:]:
        md, d = row(f); print(md); out.append(d)
    if len(out) > 1:
        print(pass_table(out))
    print(json.dumps(out), file=sys.stderr)
