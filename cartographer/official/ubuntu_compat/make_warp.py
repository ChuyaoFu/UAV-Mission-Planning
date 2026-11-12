#!/usr/bin/env python3
"""A WARPED venue replica: the organiser map is not the venue. Every element moves independently by a small random
amount (sim-side "map error"), so map-derived (blind) priors fly in a world that differs from the map.

  python3 make_warp.py SEED [--src /replica] [--dst /compat/warp] [--wall 0.15] [--rot 2] [--pad 0.2]   (in the container)
      -> DST/w<SEED>/replica/{layout.py (+ WARP block), build_world.py, ..., world/, warp.json}
         DST/w<SEED>/common -> /compat/profile/common   (the replica builder imports common.geometry)
  then fly in it:  fly_compat.sh NAME --replica /compat/warp/w<SEED>/replica ...

Offsets (map feet / degrees, all independent, reproducible per seed; written to warp.json):
  walls     each: translation uniform in a disc of radius --wall m, rotation +-rot deg about its centre;
            the barrier only along its normal (map y; it stays one straight line from the west net); wall ends that
            touch a net stay attached (W2's north end, the barrier's west end); endpoints kept inside the cage
  window    translation disc --wall*0.8 m, board yaw +-1.5*rot deg (moves its normal / crossing direction too)
  tunnel    translation disc --wall*0.8 m (entry and exit together), axis +-rot deg about the middle
  pad       translation disc --pad m;   boulders: each disc --pad m;   cinder blocks: with their nearest wall
  tape (floor line), cage, launch point: unchanged.
Sim-side only: flight code never sees layout.py.
"""
import argparse, importlib.util, json, math, os, random, shutil, subprocess, sys
from pathlib import Path

FT = 0.3048


def disc(rng, r_m):
    r = r_m * math.sqrt(rng.random()); th = rng.uniform(-math.pi, math.pi)
    return r * math.cos(th) / FT, r * math.sin(th) / FT          # feet (x east, y north)


def rot(p, c, deg):
    a = math.radians(deg); dx, dy = p[0] - c[0], p[1] - c[1]
    return (c[0] + dx * math.cos(a) - dy * math.sin(a), c[1] + dx * math.sin(a) + dy * math.cos(a))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('seed', type=int); ap.add_argument('--src', default='/replica')
    ap.add_argument('--dst', default='/compat/warp'); ap.add_argument('--wall', type=float, default=0.15)
    ap.add_argument('--rot', type=float, default=2.0); ap.add_argument('--pad', type=float, default=0.2)
    a = ap.parse_args(); rng = random.Random(1000 + a.seed)
    spec = importlib.util.spec_from_file_location('layout_src', Path(a.src) / 'layout.py')
    L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
    cage_ft = L.CAGE / FT; lo, hi = 0.0, cage_ft
    off = {}
    walls = []
    for name, p0, p1, skin, role in L.WALLS_FT:
        if name == 'barrier':                                   # along its normal only; west end stays on the net
            dx, dy = 0.0, rng.uniform(-a.wall, a.wall) / FT; d = 0.0
        else:
            dx, dy = disc(rng, a.wall); d = rng.uniform(-a.rot, a.rot)
        c = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        q0, q1 = rot(p0, c, d), rot(p1, c, d)
        q0, q1 = (q0[0] + dx, q0[1] + dy), (q1[0] + dx, q1[1] + dy)
        for i, (p, q) in enumerate(((p0, q0), (p1, q1))):      # ends touching a net stay attached
            if p[0] <= 0.05 or p[0] >= cage_ft - 0.05:
                q = (p[0], q[1])
            if p[1] <= 0.05 or p[1] >= cage_ft - 0.25:         # (W2 stops 5 cm short of the north net)
                q = (q[0], p[1])
            q = (min(max(q[0], lo), hi), min(max(q[1], lo), hi - 0.15))
            if i == 0: q0 = q
            else: q1 = q
        walls.append((name, (round(q0[0], 4), round(q0[1], 4)), (round(q1[0], 4), round(q1[1], 4)), skin, role))
        off[name] = dict(dx_ft=round(dx, 3), dy_ft=round(dy, 3), rot_deg=round(d, 2),
                         moved_m=round(FT * max(math.dist(p0, q0), math.dist(p1, q1)), 3))
    wf = dict(L.WINDOW_FT); wdx, wdy = disc(rng, a.wall * 0.8); wrot = rng.uniform(-1.5 * a.rot, 1.5 * a.rot)
    wf['cx'], wf['cy'] = wf['cx'] + wdx, wf['cy'] + wdy
    wf['along'] = rot(wf['along'], (0, 0), wrot); wf['cross'] = rot(wf['cross'], (0, 0), wrot)
    off['window'] = dict(dx_ft=round(wdx, 3), dy_ft=round(wdy, 3), rot_deg=round(wrot, 2), moved_m=round(FT * math.hypot(wdx, wdy), 3))
    e0, e1 = L.TUNNEL['entry_ft'], L.TUNNEL['exit_ft']; tdx, tdy = disc(rng, a.wall * 0.8); trot = rng.uniform(-a.rot, a.rot)
    tc = ((e0[0] + e1[0]) / 2, (e0[1] + e1[1]) / 2)
    te0 = rot(e0, tc, trot); te1 = rot(e1, tc, trot)
    te0, te1 = (te0[0] + tdx, te0[1] + tdy), (te1[0] + tdx, te1[1] + tdy)
    off['tunnel'] = dict(dx_ft=round(tdx, 3), dy_ft=round(tdy, 3), rot_deg=round(trot, 2), moved_m=round(FT * math.hypot(tdx, tdy), 3))
    pad_ft = (3.3, 7.35); pdx, pdy = disc(rng, a.pad); pad_w = (pad_ft[0] + pdx, pad_ft[1] + pdy)
    off['pad'] = dict(dx_ft=round(pdx, 3), dy_ft=round(pdy, 3), moved_m=round(FT * math.hypot(pdx, pdy), 3))
    bould = []
    for i, (x, y) in enumerate(L.BOULDERS_FT):
        bx, by = disc(rng, a.pad); bould.append((round(x + bx, 3), round(y + by, 3)))
        off[f'boulder{i}'] = dict(dx_ft=round(bx, 3), dy_ft=round(by, 3), moved_m=round(FT * math.hypot(bx, by), 3))

    def nearest_wall(x, y):
        best = None
        for (name, p0, p1, *_r) in L.WALLS_FT:
            vx, vy = p1[0] - p0[0], p1[1] - p0[1]; t = max(0, min(1, ((x - p0[0]) * vx + (y - p0[1]) * vy) / (vx * vx + vy * vy)))
            d = math.hypot(x - p0[0] - t * vx, y - p0[1] - t * vy)
            if best is None or d < best[0]: best = (d, name)
        return best[1]
    blocks = []
    for x, y, yw in L._BLOCKS_FT:
        o = off[nearest_wall(x, y)]; blocks.append((round(x + o['dx_ft'], 3), round(y + o['dy_ft'], 3), yw))
    # ---------------------------------------------------------------- write the warped replica
    root = Path(a.dst) / f'w{a.seed}'; rep = root / 'replica'
    if root.exists():
        shutil.rmtree(root)
    rep.mkdir(parents=True)
    for f in Path(a.src).glob('*.py'):
        shutil.copy2(f, rep / f.name)
    shutil.copytree(Path(a.src) / 'assets', rep / 'assets')
    os.symlink('/compat/profile/common', root / 'common')
    block = f'''

# ================= WARP (sim-side map error) - generated by ubuntu_compat/make_warp.py {a.seed} =================
# The organiser map above is NOT the venue: every element below moved independently (see warp.json).
WALLS_FT = {json.dumps(walls)}
WALLS_FT = [(n, tuple(p0), tuple(p1), sk, ro) for n, p0, p1, sk, ro in WALLS_FT]
MAZE_WALLS = [(n, ft_fl(*p0), ft_fl(*p1), skin, role) for (n, p0, p1, skin, role) in WALLS_FT]
PANELS = _panels_from_walls(WALLS_FT)
_bar = [w for w in WALLS_FT if w[0] == 'barrier'][0]
WALL.update(x0=_bar[1][0], x1=_bar[2][0], y=_bar[1][1])
WINDOW_FT = {json.dumps(dict(cx=wf['cx'], cy=wf['cy'], along=list(wf['along']), cross=list(wf['cross'])))}
WINDOW['a'], WINDOW['b'] = ft_ab(WINDOW_FT['cx'], WINDOW_FT['cy'])
WINDOW['yaw'] = math.degrees(math.atan2(WINDOW_FT['along'][0], WINDOW_FT['along'][1]))
WINDOW['normal_ne'] = (-WINDOW_FT['cross'][1], -WINDOW_FT['cross'][0])
TUNNEL['entry_ft'], TUNNEL['exit_ft'] = {tuple(round(v, 4) for v in te0)}, {tuple(round(v, 4) for v in te1)}
TUNNEL['entry_ab'] = ft_ab(*TUNNEL['entry_ft']); TUNNEL['exit_ab'] = ft_ab(*TUNNEL['exit_ft'])
_ax = np.subtract(TUNNEL['exit_ab'], TUNNEL['entry_ab']); TUNNEL['length'] = float(np.linalg.norm(_ax))
TUNNEL['axis_ab'] = tuple(float(v) for v in _ax / TUNNEL['length'])
_pad_n, _pad_e = ft_ned({pad_w[0]:.4f}, {pad_w[1]:.4f})[:2]
PAD['n'], PAD['e'] = float(_pad_n), float(_pad_e)
BOULDERS_FT = {json.dumps(bould)}
BOULDERS = [(*ft_ab(x, y), BOULDER_H, 0.23) for x, y in BOULDERS_FT]
ROCKS = BOULDERS
_BLOCKS_FT = {json.dumps(blocks)}
CINDER_BLOCKS = [(*ft_ned(x, y)[:2], yaw) for x, y, yaw in _BLOCKS_FT]
WARP_SEED = {a.seed}
'''
    with open(rep / 'layout.py', 'a') as f:
        f.write(block)
    (rep / 'warp.json').write_text(json.dumps(dict(seed=a.seed, src=a.src, wall_m=a.wall, rot_deg=a.rot, pad_m=a.pad, offsets=off), indent=1))
    r = subprocess.run([sys.executable, str(rep / 'build_world.py'), '--out', str(rep / 'world')], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:]); sys.exit(1)
    print(f'{rep}: world built; largest moves (m):',
          ', '.join(f"{k} {v['moved_m']}" for k, v in sorted(off.items(), key=lambda kv: -kv[1]['moved_m'])[:6]))


if __name__ == '__main__':
    main()
