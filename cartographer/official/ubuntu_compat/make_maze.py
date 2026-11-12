#!/usr/bin/env python3
"""RANDOMISED MAZE worlds: the organisers may change the maze layout and the start position; the checkpoint order
(maze -> Gate A -> tunnel -> chicane -> Gate B / window -> pad -> barrier -> home) stays. Sim-side test bed.

  python3 make_maze.py SEED [--src /replica] [--dst /compat/maze]        (in the container)
      -> DST/m<SEED>/replica/{layout.py (+ MAZE block), build_world.py, ..., world/, maze.json}
         DST/m<SEED>/common -> /compat/profile/common
  fly it:  fly_compat.sh NAME --replica /compat/maze/m<SEED>/replica ...   (priors: make_truth_prior.py / make_blind_prior.py)

Kept from layout v4: cage + net, barrier (map y 11.5), W3 (x 11, y 11.6-16.9 = the maze's east wall) and Gate A
(x 11, y 16.9-20, the exit), tunnel, C1, C2 + window, W4, Gate B, pad, boulders, everything outside the maze.
Generated inside the maze region (map ft: x 0-11, y 11.5-20, = 3.35 x 2.59 m):
  type NS3  3 north-south lanes, 2 dividers (one hanging from the north net, one standing on the barrier, order
            random), the walk ends going north in the east lane to Gate A          (the v4 map is one instance)
  type NS2  2 wider north-south lanes, 1 divider hanging from the north net (start in the NW, walk south first)
  (no dead ends: with >= 1.1 m lanes in a 3.35 x 2.59 m region a stub either blocks the path or leaves a pocket
   narrower than a lane)
  type EW2  2 east-west lanes, 1 divider attached to W3; start in the SE, walk west, turn north, east to Gate A
  lane widths (each >= 1.1 m clear), the open gap at each divider end (1.1-1.4 m) and the start position (inside the
  first lane, >= 0.45 m from every wall / net) are random; the start heading stays north (map: the drone faces north).
  Divider skins / A-frame blocks like the map's walls; the floor tape follows the solution path (centre line).
The start frame moves with the start: START (cage coords of the launch) is the new start; every prior is derived
from this layout by make_truth_prior.py.  maze.json records the type, walls, start, solution path.
"""
import argparse, importlib.util, json, math, os, random, shutil, subprocess, sys
from pathlib import Path

FT = 0.3048
LANE = 1.1 / FT + 0.17          # >= 1.1 m clear lane in feet (+ board thickness + margin)
X0, X1, Y0, Y1 = 0.0, 11.0, 11.5, 20.0      # maze region (map ft): west net, W3, barrier, north net
YB, YN = 11.6, 19.85                         # board ends: on the barrier side / hanging from the north net
GATE_A = (11.0, 16.9, 20.0)                  # exit: x 11, y 16.9-20 (kept)


def gen(rng):
    kind = rng.choice(['NS3', 'NS3', 'NS2', 'EW2'])
    walls, stubs, path, start = [], [], [], None
    if kind == 'NS3':
        spare = (X1 - X0) - 3 * LANE                       # ft of extra width to distribute
        w = [LANE + spare * f for f in _split(rng, 3)]
        xs = [X0 + w[0], X0 + w[0] + w[1]]
        g1, g2 = rng.uniform(1.1, 1.4) / FT, rng.uniform(1.1, 1.4) / FT
        # the east lane must be walked northward (Gate A at its north end): lane 2 southward, lane 1 northward
        # -> divider 1 hangs from the north net? no: lane1 north -> turn at the NORTH end -> divider 1 stands on the barrier
        walls += [('D1', (xs[0], YB), (xs[0], Y1 - g1)), ('D2', (xs[1], Y0 + g2), (xs[1], YN))]
        cx = [X0 + w[0] / 2, xs[0] + w[1] / 2, xs[1] + w[2] / 2]
        sy = rng.uniform(Y0 + 1.6, Y0 + 3.2); sx = cx[0] + rng.uniform(-1, 1) * max(0.0, w[0] / 2 - 1.5)
        start = (sx, sy)
        ty, by = (Y1 + (Y1 - g1)) / 2, (Y0 + (Y0 + g2)) / 2
        path = [(sx, sy), (cx[0], ty), (cx[1], ty), (cx[1], by), (cx[2], by), (cx[2], (GATE_A[1] + GATE_A[2]) / 2), (X1, (GATE_A[1] + GATE_A[2]) / 2)]
    elif kind == 'NS2':
        spare = (X1 - X0) - 2 * LANE
        w = [LANE + spare * f for f in _split(rng, 2)]
        xd = X0 + w[0]; g = rng.uniform(1.1, 1.4) / FT
        # east lane walked north -> west lane walked south -> start at the NW, divider stands on the barrier... no:
        # west lane south, turn at the SOUTH end -> the divider hangs from the north net
        walls += [('D1', (xd, Y0 + g), (xd, YN))]
        cx = [X0 + w[0] / 2, xd + w[1] / 2]
        sy = rng.uniform(Y1 - 3.0, Y1 - 1.6); sx = cx[0] + rng.uniform(-1, 1) * max(0.0, w[0] / 2 - 1.5)
        start = (sx, sy); by = (Y0 + (Y0 + g)) / 2
        path = [(sx, sy), (cx[0], by), (cx[1], by), (cx[1], (GATE_A[1] + GATE_A[2]) / 2), (X1, (GATE_A[1] + GATE_A[2]) / 2)]
    else:                                                      # EW2
        spare = (Y1 - Y0) - 2 * LANE
        h = [LANE + spare * f for f in _split(rng, 2)]
        yd = Y0 + h[0]; g = rng.uniform(1.1, 1.4) / FT
        walls += [('D1', (X0 + g, yd), (X1 - 0.1, yd))]        # attached to W3, open gap at the west net
        cy = [Y0 + h[0] / 2, yd + h[1] / 2]
        sx = rng.uniform(X1 - 3.2, X1 - 1.6); sy = cy[0] + rng.uniform(-1, 1) * max(0.0, h[0] / 2 - 1.5)
        start = (sx, sy); wx = (X0 + (X0 + g)) / 2
        path = [(sx, sy), (wx, cy[0]), (wx, cy[1]), (X1 - 0.8, cy[1]), (X1, (GATE_A[1] + GATE_A[2]) / 2)]
    return kind, walls + stubs, start, path


def _split(rng, k):
    r = [rng.random() for _ in range(k)]; s = sum(r)
    return [x / s for x in r]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('seed', type=int); ap.add_argument('--src', default='/replica')
    ap.add_argument('--dst', default='/compat/maze')
    a = ap.parse_args(); rng = random.Random(5000 + a.seed)
    spec = importlib.util.spec_from_file_location('layout_src', Path(a.src) / 'layout.py')
    L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
    kind, maze_walls, start, path = gen(rng)
    keep = [w for w in L.WALLS_FT if w[0] not in ('W1', 'W2')]          # W3 = east wall of the maze region, kept
    skins = {n: s for n, _a, _b, s, _r in L.WALLS_FT}
    walls = keep + [(n, (round(p0[0], 3), round(p0[1], 3)), (round(p1[0], 3), round(p1[1], 3)),
                     rng.choice(['banner', 'plywood']), f'random maze wall ({kind})') for n, p0, p1 in maze_walls]
    old_blocks = [b for b in L._BLOCKS_FT if not (b[0] < 10.0 and b[1] > 11.7)]   # drop blocks of the removed W1 / W2
    blocks = list(old_blocks)
    for n, p0, p1 in maze_walls:                                          # one block at the attached end of each wall
        q = p0 if (abs(p0[1] - YB) < 0.2 or abs(p0[0] - X1 + 0.1) < 0.2) else p1
        horiz = abs(p1[1] - p0[1]) < 1e-6
        blocks.append((round(q[0] + (0.0 if horiz else 0.4), 3), round(q[1] + (0.4 if horiz else 0.0), 3), 90.0 if horiz else 0.0))
    # our route = launch -> solution path -> Gate A -> the v4 continuation (east along the north net to the tunnel ...)
    i_cont = L.ROUTE_FT.index((17.6, 18.5))
    route = [tuple(round(v, 3) for v in start)] + [tuple(round(v, 3) for v in p) for p in path[1:]] + L.ROUTE_FT[i_cont:]
    root = Path(a.dst) / f'm{a.seed}'; rep = root / 'replica'
    if root.exists():
        shutil.rmtree(root)
    rep.mkdir(parents=True)
    for f in Path(a.src).glob('*.py'):
        shutil.copy2(f, rep / f.name)
    shutil.copytree(Path(a.src) / 'assets', rep / 'assets')
    os.symlink('/compat/profile/common', root / 'common')
    block = f'''

# ================= RANDOM MAZE - generated by ubuntu_compat/make_maze.py {a.seed} (type {kind}) =================
# Maze walls W1 / W2 replaced, new start; everything outside the maze region as layout v4 (see maze.json).
LAUNCH_FT = {tuple(round(v, 3) for v in start)}
START = (LAUNCH_FT[1] * FT, LAUNCH_FT[0] * FT)
WALLS_FT = {json.dumps(walls)}
WALLS_FT = [(n, tuple(p0), tuple(p1), sk, ro) for n, p0, p1, sk, ro in WALLS_FT]
MAZE_WALLS = [(n, ft_fl(*p0), ft_fl(*p1), skin, role) for (n, p0, p1, skin, role) in WALLS_FT]
PANELS = _panels_from_walls(WALLS_FT)
WINDOW['a'], WINDOW['b'] = ft_ab(WINDOW_FT['cx'], WINDOW_FT['cy'])
TUNNEL['entry_ab'] = ft_ab(*TUNNEL['entry_ft']); TUNNEL['exit_ab'] = ft_ab(*TUNNEL['exit_ft'])
_pad_n, _pad_e = ft_ned(3.3, 7.35)[:2]
PAD['n'], PAD['e'] = float(_pad_n), float(_pad_e)
_BLOCKS_FT = {json.dumps(blocks)}
CINDER_BLOCKS = [(*ft_ned(x, y)[:2], yaw) for x, y, yaw in _BLOCKS_FT]
TAPE_FT = [('maze_path', {json.dumps([tuple(round(v, 3) for v in p) for p in path])})]
TAPE = [(n, [ft_fl(*p) for p in pts]) for n, pts in TAPE_FT]
ROUTE_FT = {json.dumps(route)}
ROUTE_FT = [tuple(p) for p in ROUTE_FT]
MAZE_EXIT_FT = {GATE_A}
MAZE_REGION_FT = {(X0, X1, Y0, Y1)}
MAZE_SEED, MAZE_TYPE = {a.seed}, '{kind}'
'''
    with open(rep / 'layout.py', 'a') as f:
        f.write(block)
    info = dict(seed=a.seed, type=kind, start_ft=start, walls=[(n, p0, p1) for n, p0, p1 in maze_walls], solution_ft=path)
    (rep / 'maze.json').write_text(json.dumps(info, indent=1))
    # sanity: the new layout imports, the solution keeps >= 0.45 m from every maze wall, start >= 0.45 m from walls
    spec = importlib.util.spec_from_file_location('layout_new', rep / 'layout.py')
    N = importlib.util.module_from_spec(spec); spec.loader.exec_module(N)
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import course_solids as C
    worst = min(C.clearance(N, *N.ft_ab(*p), 0.85)[0] for p in [start] + [_lerp(path[i], path[i + 1], t) for i in range(len(path) - 1) for t in (0.25, 0.5, 0.75)])
    assert worst >= 0.40, ('solution path too close to a wall', kind, round(worst, 3))
    r = subprocess.run([sys.executable, str(rep / 'build_world.py'), '--out', str(rep / 'world')], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-3000:]); sys.exit(1)
    print(f'{rep}: type {kind}, {len(maze_walls)} maze walls, start ft ({start[0]:.2f}, {start[1]:.2f}), solution clearance >= {worst:.2f} m')


def _lerp(p, q, t):
    return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))


if __name__ == '__main__':
    main()
