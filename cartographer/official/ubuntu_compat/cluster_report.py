#!/usr/bin/env python3
"""Per flight: floating_cluster / small_cluster_kept counts, the kept clusters (n, distance to the nearest cage net,
truth clearance to the nearest real solid = is it a phantom?), where the flight ended, and its official-metric fails.

  python3 cluster_report.py FLIGHT_DIR [...]        (in the container)
"""
import json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import course_solids as C  # noqa: E402


def main():
    for f in sys.argv[1:]:
        F = Path(f); L = C.load_layout(F)
        a = json.loads((F / 'sim/analysis.json').read_text()) if (F / 'sim/analysis.json').exists() else {}
        cc = json.loads((F / 'sim/corner_check.json').read_text()) if (F / 'sim/corner_check.json').exists() else {}
        fl, kept = 0, []
        for ln in open(F / 'mission.log'):
            if not ln.startswith('{'):
                continue
            e = json.loads(ln.replace('NaN', 'null')).get('event', {}); k = e.get('kind')
            if k == 'floating_cluster':
                fl += 1
            elif k == 'small_cluster_kept':
                n_, e_, d_ = e['center']; ca, cb, h = n_ + L.START[0], e_ + L.START[1], -d_ - 0.05   # NED (PX4, base_link) -> cage
                net = min(ca, L.CAGE - ca, cb, L.CAGE - cb)
                solid = C.clearance(L, ca, cb, h)[0]
                kept.append(dict(t=round(e['t'], 1), n=e.get('n'), f=round(n_, 2), l=round(-e_, 2), net_m=round(net, 2), solid_m=round(solid, 2)))
        viol = cc.get('official_violations', [])
        print(f"{F.name}: floating {fl}, kept {len(kept)}"
              f" (n {min((k['n'] for k in kept), default='-')}-{max((k['n'] for k in kept), default='-')},"
              f" net dist {min((k['net_m'] for k in kept), default='-')}-{max((k['net_m'] for k in kept), default='-')} m,"
              f" phantoms (> 0.3 m from any solid) {sum(1 for k in kept if k['solid_m'] > 0.3)}/{len(kept)})"
              f" | end f,l {a.get('truth_end_fl')} | < 0.28 m: " + ('; '.join(f"{v['what']} {v['min_m']} m @t{v['t0']}" for v in viol) or 'none'))
        if kept:
            k = kept[-1]
            print(f"    last kept: t {k['t']} n {k['n']} at f {k['f']} l {k['l']} (net {k['net_m']} m, nearest solid {k['solid_m']} m)")


if __name__ == '__main__':
    main()
