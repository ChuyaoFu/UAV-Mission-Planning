#!/usr/bin/env python3
"""BLIND course priors: what we can know without walking the venue = the organiser's top-view map, with every map
element off by its own random amount (topology unchanged: same turn order, same openings, same checkpoint choice).

  python3 make_blind_prior.py --level map|eyeball --seeds 1-8 [--base course_prior.replica.json]
      map     -> course_prior.blind.seed<N>.json          element offset uniform in a disc of radius 0.3 m,
                                                           angles +-5 deg, heights +-0.10 m
      eyeball -> course_prior.blind.eyeball.seed<N>.json  element offset magnitude uniform 0.5-0.8 m (random direction),
                                                           angles +-10 deg, heights +-0.15 m

--base must be the MAP prior: make_truth_prior.py run on the unwarped layout (/replica = COURSE_MAP.md transcribed),
never on a warped world. Element = a wall / gate / window / pad / tunnel / boulder: every waypoint the route derives
from one element moves WITH it (one shared offset, + 0.05 m per-point transcription jitter):
  W1_end    corridor[0..1]  turn around W1's north end         W2_end     corridor[2..3]  U-turn around W2's south end
  gate_A    corridor[4..5]  W3 north end / Gate A              tunnel     corridor[6], tunnel entry (+ axis, up), chicane[0]
  C1_end    chicane[1..2]   around C1's south end              C1C2_gap   chicane[3]
  gate_B    gate_b centre (+ through), gate_b.after[0]          W4_end     gate_b.after[1]
  window    window centre (+ through, up), window.after[0]      pad        landing centre, window.after[1]
  barrier   wall cross_from / cross_to (+ top_up)              boulder<i> each boulder
  walls (set blind2): each map wall moves with its element (barrier, W1 -> W1_end, W2 -> W2_end, W3 -> gate_A, C1 -> C1_end,
            C2_north -> gate_B, W4 -> W4_end) and turns by half that element's angle error about its centre
Unchanged: fence (cage net), cruise height, tolerances, checkpoint4. Offsets -> '_offsets', '_comment'.
"""
import argparse, json, math, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEVELS = {'map': dict(r=(0.0, 0.3), ang=5.0, dz=0.10, disc=True),
          'eyeball': dict(r=(0.5, 0.8), ang=10.0, dz=0.15, disc=False)}
JITTER = 0.05


def parse_seeds(s):
    out = []
    for part in s.split(','):
        a, _, b = part.partition('-'); out += list(range(int(a), int(b or a) + 1))
    return out


def make(base, level, seed):
    P = json.loads(json.dumps(base)); cfg = LEVELS[level]; rng = random.Random(f'blind-{level}-{seed}')
    generic = not (len(P['corridor']) == 7 and P.get('walls') is not None and {w['name'] for w in P['walls']} >= {'W1', 'W2', 'W3'})
    assert len(P.get('chicane', [])) == 4, 'base prior shape changed: update the element map'
    elems = {}

    def elem(name):
        if name not in elems:
            if cfg['disc']:
                r = cfg['r'][1] * math.sqrt(rng.random())
            else:
                r = rng.uniform(*cfg['r'])
            th = rng.uniform(-math.pi, math.pi)
            elems[name] = dict(df=round(r * math.cos(th), 3), dl=round(r * math.sin(th), 3), r=round(r, 3),
                               dz=round(rng.uniform(-cfg['dz'], cfg['dz']), 3), dang=round(rng.uniform(-cfg['ang'], cfg['ang']), 2))
        return elems[name]

    def move(pt, name, z=True):
        e = elem(name); jr = JITTER * math.sqrt(rng.random()); jt = rng.uniform(-math.pi, math.pi)
        pt['forward'] = round(pt['forward'] + e['df'] + jr * math.cos(jt), 3)
        pt['left'] = round(pt['left'] + e['dl'] + jr * math.sin(jt), 3)
        if z and 'up' in pt:
            pt['up'] = round(pt['up'] + e['dz'], 3)
    c = P['corridor']
    if not generic:
        for i, n in ((0, 'W1_end'), (1, 'W1_end'), (2, 'W2_end'), (3, 'W2_end'), (4, 'gate_A'), (5, 'gate_A'), (6, 'tunnel')):
            move(c[i], n, z=False)
    else:                                              # random maze: each corridor point moves with its nearest maze wall
        mw = [w for w in P.get('walls', []) if w['name'] not in ('barrier', 'C1', 'C2_north', 'W4')]
        def near(q):
            best = (9e9, 'gate_A')
            for w in mw:
                a_, b_ = w['from'], w['to']; vx, vy = b_['forward'] - a_['forward'], b_['left'] - a_['left']
                t = max(0, min(1, ((q['forward'] - a_['forward']) * vx + (q['left'] - a_['left']) * vy) / max(vx * vx + vy * vy, 1e-9)))
                d = math.hypot(q['forward'] - a_['forward'] - t * vx, q['left'] - a_['left'] - t * vy)
                best = min(best, (d, 'wall_' + w['name'] if w['name'] != 'W3' else 'gate_A'))
            return best[1]
        for i, q in enumerate(c):
            move(q, 'tunnel' if i == len(c) - 1 else near(q), z=False)
    move(P['tunnel']['entry'], 'tunnel'); P['tunnel']['axis_deg'] = round(P['tunnel']['axis_deg'] + elem('tunnel')['dang'], 2)
    ch = P['chicane']
    for i, n in ((0, 'tunnel'), (1, 'C1_end'), (2, 'C1_end'), (3, 'C1C2_gap')):
        move(ch[i], n, z=False)
    if P.get('gate_b'):
        move(P['gate_b']['center'], 'gate_B', z=False)
        P['gate_b']['through_deg'] = round(P['gate_b']['through_deg'] + elem('gate_B')['dang'], 2)
        aft = P['gate_b'].get('after', [])
        if len(aft) > 0: move(aft[0], 'gate_B', z=False)
        if len(aft) > 1: move(aft[1], 'W4_end', z=False)
    w = P['window']
    move(w['center'], 'window'); w['through_deg'] = round(w['through_deg'] + elem('window')['dang'], 2)
    wa = w.get('after', [])
    if len(wa) > 0: move(wa[0], 'window', z=False)
    if len(wa) > 1: move(wa[1], 'pad', z=False)
    move(P['landing_target']['center'], 'pad', z=False)
    move(P['wall']['cross_from'], 'barrier', z=False); move(P['wall']['cross_to'], 'barrier', z=False)
    P['wall']['top_up'] = round(P['wall']['top_up'] + elem('barrier')['dz'], 3)
    for i, b in enumerate(P.get('boulders', [])):
        move(b, f'boulder{i}', z=False)
    WALL_ELEM = {'barrier': 'barrier', 'W1': 'W1_end', 'W2': 'W2_end', 'W3': 'gate_A', 'C1': 'C1_end', 'C2_north': 'gate_B', 'W4': 'W4_end'}
    if generic:
        WALL_ELEM.update({w['name']: 'wall_' + w['name'] for w in P.get('walls', []) if w['name'] not in WALL_ELEM})
    for wd in P.get('walls', []):                        # drawn LAST: earlier element offsets stay identical to set 'blind'
        e = elem(WALL_ELEM.get(wd['name'], 'wall_' + wd['name']))
        a, b = wd['from'], wd['to']; cf, cl = (a['forward'] + b['forward']) / 2, (a['left'] + b['left']) / 2
        r = math.radians(0.5 * e['dang'])                # half the element's angle error as wall rotation
        for q in (a, b):
            df, dl = q['forward'] - cf, q['left'] - cl
            q['forward'] = round(cf + df * math.cos(r) - dl * math.sin(r) + e['df'], 3)
            q['left'] = round(cl + df * math.sin(r) + dl * math.cos(r) + e['dl'], 3)
    assert P['fence'] == base['fence'] and P['corridor'][0] is not base['corridor'][0]
    P['_comment'] = (f"BLIND prior, level {level}, seed {seed}: organiser-map prior ({base.get('_comment', '')[:60]}...) with every "
                     f"map element off by its own random offset ({'disc 0.3 m' if level == 'map' else '0.5-0.8 m'}; angles +-{cfg['ang']} deg; "
                     f"heights +-{cfg['dz']} m; per-point jitter {JITTER} m). Topology unchanged. Element offsets (f/l m): "
                     + json.dumps({k: (v['df'], v['dl']) for k, v in elems.items()}))
    P['_offsets'] = elems; P['_seed'] = seed; P['_scale'] = f'blind-{level}'
    return P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--level', choices=sorted(LEVELS), required=True)
    ap.add_argument('--seeds', default='1-8'); ap.add_argument('--base', default=str(HERE / 'course_prior.replica.json'))
    ap.add_argument('--out', default=None, help='single output file name (one seed), e.g. course_prior.maze3.blind.json')
    ap.add_argument('--set', default='blind', help="output set name: blind (old, no walls) or blind2 (with the map's walls)")
    a = ap.parse_args(); base = json.loads(Path(a.base).read_text())
    assert not str(base.get('_scale', '')).startswith('blind') and '_seed' not in base, '--base must be the unperturbed map prior'
    for s in parse_seeds(a.seeds):
        P = make(base, a.level, s)
        if a.set == 'blind':
            P.pop('walls', None); P.pop('walls_tol_m', None)      # the original set had no walls
        else:
            assert P.get('walls'), '--set with walls needs a base prior that has them (make_truth_prior.py)'
        P['_scale'] = f'{a.set}-{a.level}'
        out = HERE / (a.out if a.out else (f'course_prior.{a.set}.seed{s}.json' if a.level == 'map' else f'course_prior.{a.set}.eyeball.seed{s}.json'))
        out.write_text(json.dumps(P, indent=1) + '\n')
        big = sorted(P['_offsets'].items(), key=lambda kv: -kv[1]['r'])[:4]
        print(out.name, '| largest element offsets:', ', '.join(f"{k} {v['r']:.2f} m" for k, v in big))


if __name__ == '__main__':
    main()
