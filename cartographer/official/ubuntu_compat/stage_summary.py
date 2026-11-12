#!/usr/bin/env python3
"""One markdown row per PER-STAGE flight (run_stages.sh) + a per-stage table, from meta.json (params: start_phase /
stop_after), sim/stack.json (spawn), sim/corner_check.json (phases, contacts, official clearance), sim/analysis.json.

  python3 stage_summary.py FLIGHT_DIR [...]      (corner_check.py and analyze_flight.py first; run_blind does it)

success = the flight reached its expected end (staged_stop "stop_after_<stop>", or done when there is no stop_after)
          AND the stage's truth metric exists (tunnel crossed / Gate B crossed / window crossed / pad touchdown / wall crossed)
clean   = success AND 0 contacts AND official centre clearance >= 0.28 m
stage time = entry into start_phase -> end (staged_stop / done / aborted)
"""
import ast, collections, json, re, sys
from pathlib import Path

STAGE_OF = {'tunnel_align': 'tunnel', 'tunnel_traverse': 'tunnel', 'chicane': 'chicane', 'gate_pass': 'gate_b',
            'window_align': 'window', 'window_traverse': 'window', 'landing_route': 'pad', 'land_target': 'pad',
            'climb_over_wall': 'wall', 'rearm': 'wall', 'return_home': 'home', 'land_home': 'home'}
ORDER = ['tunnel', 'chicane', 'gate_b', 'window', 'pad', 'wall', 'home']


def j(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return {}


def metric(stage, a):
    tp, gp, wp, sd, wl = (a.get(k) or {} for k in ('tunnel_pass', 'gate_b_pass', 'window_pass', 'stone_touchdown', 'wall_pass'))
    if stage == 'tunnel':
        return bool(tp), (f"max lat {100 * tp.get('max_lateral_off_m', float('nan')):+.1f} / vert {100 * tp.get('max_vertical_off_m', float('nan')):+.1f} cm, "
                          f"radial {100 * tp['max_radial_off_m']:.1f} cm, prop-to-mesh {tp['min_clear_to_mesh_m']:.2f} m") if tp else 'not crossed'
    if stage == 'gate_b':
        return bool(gp), (f"plane offset {100 * gp['offset_from_centre_m']:+.1f} cm, h {gp['height_m']:.2f} m, prop clear {gp['prop_clear_each_side_m']:.2f} m"
                          if gp else 'not crossed')
    if stage == 'window':
        return bool(wp), (f"plane offset lat {100 * wp['lateral_off_m']:+.1f} / vert {100 * wp['vertical_off_m']:+.1f} cm, "
                          f"clear side {wp['clear_side_m']:.2f} / vert {wp['clear_vert_m']:.2f} m" if wp else 'not crossed')
    if stage == 'pad':
        return bool(sd), (f"landing error {100 * sd['off_center_m']:.1f} cm ({'in' if sd['off_center_m'] + 0.23 < 0.78 else 'OUT of'} ring), "
                          f"{sd['seconds']:.0f} s on pad" if sd else 'no pad touchdown')
    if stage in ('wall', 'home'):
        home = a.get('truth_end_xy_from_start_m')
        hs = f"home error {100 * home:.1f} cm" if home is not None else 'home error -'
        if stage == 'wall':
            return bool(wl), (f"height margin (skids over top) {100 * wl['skids_over_top_m']:+.1f} cm, {hs}" if wl else f'not crossed, {hs}')
        return True, hs
    return True, '-'


def row(F):
    F = Path(F); m = j(F / 'meta.json'); c = j(F / 'sim/corner_check.json'); a = j(F / 'sim/analysis.json'); st = j(F / 'sim/stack.json')
    p = m.get('params') or {}
    if isinstance(p, str):
        try:
            p = ast.literal_eval(p) or {}
        except Exception:  # noqa: BLE001
            try:
                p = json.loads(p)
            except Exception:  # noqa: BLE001
                p = {}
    start, stop = p.get('start_phase', ''), p.get('stop_after', '')
    parts = F.name.split('_', 2)                            # <stamp>_<TAG>_<list name><rep>: group by list name
    stage = re.sub(r'_?\d+$', '', parts[2]) if len(parts) > 2 else STAGE_OF.get(start, start or 'full')
    mstage = STAGE_OF.get(stop) if stop else STAGE_OF.get(start, 'full')   # which truth metric applies
    ph = c.get('phases', [])
    reached = [x[1] for x in ph]; last = ph[-1][1] if ph else '?'; why = ph[-1][2] if ph else ''
    t0 = next((x[0] for x in ph if x[1] == start), None); t_end = ph[-1][0] if ph else None
    end_ok = ('staged_stop' in reached and any(f'stop_after_{stop}' in str(x[2]) for x in ph if x[1] == 'staged_stop')) if stop else last == 'done'
    has, met = metric(mstage or 'full', a)
    cont = c.get('contacts', []); oc = c.get('official_min_centre_clearance_m') or {}
    ok = end_ok and has; clean = ok and not cont and not oc.get('fail')
    res = 'done' if last == 'done' else f'{last} ({why})'
    ocs = f"{oc['d']:.2f} m ({oc.get('what', '')}){' **FAIL**' if oc.get('fail') else ''}" if oc else '-'
    cs = '; '.join(f"t{x['t']:.0f} {x['nearest']}" for x in cont) or '0'
    dur = f"{t_end - t0:.1f} s" if t0 is not None and t_end is not None else ('start phase never entered' if start else '-')
    verdict = 'PASS' if clean else ('pass, not clean' if ok else 'FAIL')
    return (f"| {stage} | {F.name} | {st.get('spawn', '') or 'launch'} | {start} → {stop or 'done'} | {res} | **{verdict}** | {dur} | {cs} | {ocs} | {met} |",
            dict(stage=stage, flight=F.name, ok=ok, clean=clean, contacts=len(cont), official_fail=bool(oc.get('fail')), official_min=oc.get('d'),
                 crash='released' in reached, metric=met, duration=dur, result=res))


def table(rows):
    g = collections.defaultdict(list)
    for d in rows:
        g[d['stage']].append(d)
    out = ['', '| stage | n | success | clean (0 contacts, ≥ 0.28 m) | flights with contact | official < 0.28 m (FAIL) | crashes | min centre clearance |',
           '|---|---|---|---|---|---|---|---|']
    for k in sorted(g, key=lambda s: (ORDER.index(s) if s in ORDER else 99, s)):
        ds = g[k]; n = len(ds); oc = [x['official_min'] for x in ds if x['official_min'] is not None]
        out.append(f"| {k} | {n} | {sum(x['ok'] for x in ds)}/{n} | {sum(x['clean'] for x in ds)}/{n} | {sum(1 for x in ds if x['contacts'])} | "
                   f"{sum(x['official_fail'] for x in ds)} | {sum(x['crash'] for x in ds)} | " + (f"{min(oc):.2f} m" if oc else '-') + ' |')
    return '\n'.join(out)


if __name__ == '__main__':
    print('| stage | flight | spawn (f,l,yaw) | phases | result | verdict | stage time | contacts | official centre clearance (< 0.28 m = FAIL) | stage metric |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    out = []
    for f in sys.argv[1:]:
        md, d = row(f); print(md); out.append(d)
    if out:
        print(table(out))
    print(json.dumps(out), file=sys.stderr)
