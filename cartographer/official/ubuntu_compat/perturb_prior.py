#!/usr/bin/env python3
"""Anti-overfitting: a course prior with realistic on-site errors instead of replica truth.

  python3 perturb_prior.py SEED [--scale tape|eyeball] [--src course_prior.replica.json]
      tape    (default) -> course_prior.replica.seed<SEED>.json          xy disc 0.4 m, z +-0.10 m, +-8 deg,  tops +-0.10 m
      eyeball           -> course_prior.replica.eyeball.seed<SEED>.json  xy disc 0.8 m, z +-0.15 m, +-12 deg, tops +-0.15 m
      (tape = measured on site with a tape; eyeball = estimated from photos)

Every stage point gets its OWN random error (stdlib random, reproducible per seed), sizes from --scale:
  corridor goals, tunnel entry, window centre, landing centre, wall cross_from, wall cross_to:
      xy offset uniform in a disc of radius R_XY; z offset uniform +-DZ where the point has a height
  tunnel axis_deg, window through_deg: +-DANG
  landing top_up, wall top_up: +-DTOP
  (layout v4) chicane goals, gate_b centre (+ through_deg), gate_b.after, window.after, boulders: same sizes, drawn after the above
Unchanged: fence box, cruise_up_m, tolerances. The applied offsets (start frame forward/left/up metres, degrees) go
into `_comment` and `_offsets`; the last stdout line is a compact one-line summary for flight notes.
"""
import argparse, json, math, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCALES = {'tape': (0.4, 0.1, 8.0, 0.1), 'eyeball': (0.8, 0.15, 12.0, 0.15)}   # R_XY, DZ, DANG, DTOP


def main():
    global R_XY, DZ, DANG, DTOP
    ap = argparse.ArgumentParser(); ap.add_argument('seed', type=int); ap.add_argument('--src', default=str(HERE / 'course_prior.replica.json'))
    ap.add_argument('--scale', choices=sorted(SCALES), default='tape')
    a = ap.parse_args()
    R_XY, DZ, DANG, DTOP = SCALES[a.scale]
    rng = random.Random(a.seed)
    src = json.loads(Path(a.src).read_text()); p = json.loads(json.dumps(src))
    offs = {}

    def xy(name, pt, z=True):
        r = R_XY * math.sqrt(rng.random()); th = rng.uniform(-math.pi, math.pi)
        df, dl = r * math.cos(th), r * math.sin(th)
        pt['forward'] = round(pt['forward'] + df, 3); pt['left'] = round(pt['left'] + dl, 3)
        o = {'df': round(df, 3), 'dl': round(dl, 3), 'dxy': round(r, 3)}
        if z and 'up' in pt:
            dz = rng.uniform(-DZ, DZ); pt['up'] = round(pt['up'] + dz, 3); o['dz'] = round(dz, 3)
        offs[name] = o

    for i, c in enumerate(p['corridor']):
        xy(f'corridor[{i}]', c)
    xy('tunnel.entry', p['tunnel']['entry'])
    d = rng.uniform(-DANG, DANG); p['tunnel']['axis_deg'] = round(p['tunnel']['axis_deg'] + d, 2); offs['tunnel.axis_deg'] = round(d, 2)
    xy('window.center', p['window']['center'])
    d = rng.uniform(-DANG, DANG); p['window']['through_deg'] = round(p['window']['through_deg'] + d, 2); offs['window.through_deg'] = round(d, 2)
    xy('landing.center', p['landing_target']['center'], z=False)
    d = rng.uniform(-DTOP, DTOP); p['landing_target']['top_up'] = round(p['landing_target']['top_up'] + d, 3); offs['landing.top_up'] = round(d, 3)
    xy('wall.cross_from', p['wall']['cross_from'], z=False)
    xy('wall.cross_to', p['wall']['cross_to'], z=False)
    d = rng.uniform(-DTOP, DTOP); p['wall']['top_up'] = round(p['wall']['top_up'] + d, 3); offs['wall.top_up'] = round(d, 3)
    for i, c in enumerate(p.get('chicane', [])):                       # layout v4 extra keys (venue_map format)
        xy(f'chicane[{i}]', c)
    if 'gate_b' in p:
        xy('gate_b.center', p['gate_b']['center'])
        d = rng.uniform(-DANG, DANG); p['gate_b']['through_deg'] = round(p['gate_b']['through_deg'] + d, 2); offs['gate_b.through_deg'] = round(d, 2)
    for i, c in enumerate((p.get('gate_b') or {}).get('after', [])):
        xy(f'gate_b.after[{i}]', c)
    for i, c in enumerate(p['window'].get('after', [])):
        xy(f'window.after[{i}]', c)
    for i, c in enumerate(p.get('boulders', [])):
        xy(f'boulders[{i}]', c, z=False)
    assert p['fence'] == src['fence'] and p['cruise_up_m'] == src['cruise_up_m']
    p['_comment'] = (f"PERTURBED prior, scale {a.scale}, seed {a.seed}: course_prior.replica.json (replica truth) + independent errors per stage point "
                     f"(xy disc {R_XY} m, z +-{DZ} m, angles +-{DANG} deg, top heights +-{DTOP} m); fence and cruise height unchanged. "
                     f"Offsets (start frame f/l/up m): " + json.dumps(offs))
    p['_offsets'] = offs; p['_seed'] = a.seed; p['_scale'] = a.scale
    out = HERE / (f'course_prior.replica.seed{a.seed}.json' if a.scale == 'tape' else f'course_prior.replica.{a.scale}.seed{a.seed}.json')
    out.write_text(json.dumps(p, indent=1) + '\n')
    short = lambda k: (f"{k.split('.')[0].replace('corridor', 'c')}"
                       + (f" {offs[k]['df']:+.2f},{offs[k]['dl']:+.2f}" + (f" z{offs[k]['dz']:+.2f}" if 'dz' in offs[k] else '')
                          if isinstance(offs[k], dict) else f" {offs[k]:+g}"))
    keys = [k for k in offs]
    print(out)
    print(f"prior {a.scale} seed {a.seed}: " + '; '.join(
        (k.replace('corridor', 'c').replace('.center', '').replace('.entry', '').replace('cross_', '')
         + (f" {offs[k]['df']:+.2f}/{offs[k]['dl']:+.2f}" + (f"/z{offs[k]['dz']:+.2f}" if 'dz' in offs[k] else '')
            if isinstance(offs[k], dict) else f" {offs[k]:+g}")) for k in keys))


if __name__ == '__main__':
    main()
