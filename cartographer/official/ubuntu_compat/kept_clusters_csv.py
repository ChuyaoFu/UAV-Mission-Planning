#!/usr/bin/env python3
"""Every small_cluster_kept AND floating_cluster event of some flights as one CSV row (column `kind`), with sim-side
truth (the layout each flight flew). All `support.*` fields the event carries are flattened into support_<key> columns.

  python3 kept_clusters_csv.py OUT.csv FLIGHT_DIR [...]        (in the container)

Columns
  flight, t, phase                  event time (sim s) and mission phase
  n, free_m                         cluster points; free distance along the path the cluster leaves (event fields)
  center_n/e/d                      cluster centre, PX4 NED (start-anchored), as logged
  f, l, h                           same in the start frame; h = height above the floor (d_layout = d_px4 - 0.05)
  radius_m                          NOT logged by the event (empty)
  support_why, support_span_m, support_comp_n, net_pts_dropped, support_fence_m, support_low_h, support_high_h
                                    the event's vertical-support record (low/high = lowest/highest connected point, as height)
  truth                             'real' if the centre is within 0.30 m of a physical solid (course_solids.py), else 'phantom'
  nearest_real, nearest_real_m      nearest physical solid (board / window board / block / boulder) and Euclidean distance
  net_m                             distance to the nearest cage net wall (truth cage)
  fence_m                           distance to the prior's fence box (same cage in blind priors)
  prior_structure, prior_structure_m nearest structure the PRIOR knows: barrier line (cross_from/to midpoint plane), boulders,
                                    tunnel entry, window centre, Gate B centre (horizontal distance; maze walls are not in the prior)
"""
import csv, json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import course_solids as C  # noqa: E402


def prior_structs(P):
    out = []
    w = P.get('wall')
    if w:
        fa, fb = w['cross_from']['forward'], w['cross_to']['forward']
        out.append(('barrier', ('fplane', (fa + fb) / 2)))
    for i, b in enumerate(P.get('boulders', [])):
        out.append((f'boulder{i}', ('pt', b['forward'], b['left'])))
    out.append(('tunnel_entry', ('pt', P['tunnel']['entry']['forward'], P['tunnel']['entry']['left'])))
    out.append(('window', ('pt', P['window']['center']['forward'], P['window']['center']['left'])))
    if P.get('gate_b'):
        out.append(('gate_b', ('pt', P['gate_b']['center']['forward'], P['gate_b']['center']['left'])))
    return out


def main():
    out = Path(sys.argv[1]); rows = []
    for f in sys.argv[2:]:
        F = Path(f); L = C.load_layout(F); P = json.loads((F / 'prior.json').read_text()); fz = P['fence']
        ps = prior_structs(P)
        for ln in open(F / 'mission.log'):
            if not ln.startswith('{'):
                continue
            e = json.loads(ln.replace('NaN', 'null')).get('event', {})
            if e.get('kind') not in ('small_cluster_kept', 'floating_cluster'):
                continue
            n_, e_, d_ = e['center']; fw, lf, h = n_, -e_, -d_ - 0.05
            ca, cb = fw + L.START[0], lf * -1 + L.START[1]
            dr, name = C.clearance(L, ca, cb, h)
            net = min(ca, L.CAGE - ca, cb, L.CAGE - cb)
            fence = min(fw - fz['forward_min'], fz['forward_max'] - fw, lf - fz['left_min'], fz['left_max'] - lf)
            best = (9e9, '')
            for nm, g in ps:
                d = abs(fw - g[1]) if g[0] == 'fplane' else math.hypot(fw - g[1], lf - g[2])
                best = min(best, (d, nm))
            s = e.get('support') or {}
            rows.append(dict(flight=F.name, kind='kept' if e['kind'] == 'small_cluster_kept' else 'floating', t=round(e['t'], 2), phase=e.get('phase'), n=e.get('n'), free_m=e.get('free_m'),
                             center_n=n_, center_e=e_, center_d=d_, f=round(fw, 3), l=round(lf, 3), h=round(h, 3), radius_m=e.get('radius_m', s_.get('radius_m', '')) if (s_ := (e.get('support') or {})) is not None else '',
                             support_why=s.get('why', ''), support_span_m=s.get('span_m', ''), support_comp_n=s.get('comp_n', ''),
                             net_pts_dropped=s.get('net_pts_dropped', ''), support_fence_m=json.dumps(s.get('fence_m', '')),
                             support_low_h=round(-s['low'][2] - 0.05, 3) if s.get('low') else '',
                             support_high_h=round(-s['high'][2] - 0.05, 3) if s.get('high') else '',
                             truth='real' if dr <= 0.30 else 'phantom', nearest_real=name, nearest_real_m=round(dr, 3),
                             net_m=round(net, 3), fence_m=round(fence, 3), prior_structure=best[1], prior_structure_m=round(best[0], 3),
                             **{f'support_{k}': (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in s.items()
                                if k not in ('why', 'span_m', 'comp_n', 'net_pts_dropped', 'fence_m', 'low', 'high')}))
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval=''); w.writeheader(); w.writerows(rows)
    ph = [r for r in rows if r['truth'] == 'phantom' and r['kind'] == 'kept']
    print(f'{out}: {sum(1 for r in rows if r["kind"] == "kept")} kept / {sum(1 for r in rows if r["kind"] == "floating")} floating clusters, {len(ph)} phantoms; phantom net distance median '
          f"{sorted(r['net_m'] for r in ph)[len(ph) // 2] if ph else '-'} m; support_why counts "
          + json.dumps({k: sum(1 for r in rows if r['support_why'] == k) for k in sorted(set(r['support_why'] for r in rows))}))


if __name__ == '__main__':
    main()
