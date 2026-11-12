#!/usr/bin/env python3
"""Physical course solids of the venue replica (sim side only) + Euclidean clearance + tunnel geometry.

layout._solids() is a COARSE list for camera-pose rejection: boards carry +0.02 m on each face (banner 5 cm board
-> half 0.045; plywood 2 cm window -> half 0.03). Collision geometry and contact analysis need the physical size,
so everything is rebuilt here from layout.PANELS / WINDOW (/ WALL in v3) / CINDER_BLOCKS / ROCKS: physical board
thickness, each board solid from the floor to its top (the A-frame legs and the open gap under a board are not
modelled - conservative for collisions). The tunnel is not in this list: see tunnel_geom() / build_world.py.

Works with layout v3 (WALL separate, window facing north) and v4 (organiser map: barrier is one of PANELS, window
board yawed as part of chicane wall C2, tunnel along TUNNEL['axis_ab'], boulders).

  solids(L)              -> [(name, 'obox', (a, b, half_along, half_perp, yaw_deg, h0, h1)) | (name, 'cyl', (a, b, r, h0, h1))]
  clearance(L, a, b, h)  -> (distance_m, name) to the nearest solid, cage coordinates (a north, b east, h up)
  tunnel_geom(L)         -> dict(entry (a, b), axis (ua, ub), right (ra, rb), length, centre_h, radius, mesh_r)
  tunnel_local(L, a, b, h) -> (s along the axis from the entry, radial distance from the axis)
  load_layout(FLIGHT)    -> the layout module that flight flew (sim/asflown/layout.py, else /replica)
"""
import importlib.util, math, sys
from pathlib import Path


def load_layout(flight=None):
    """The layout module a flight actually flew: FLIGHT/sim/asflown/layout.py (recorded by fly_compat.sh since the
    warped-world support) if present, else /replica/layout.py. Registered as sys.modules['layout']."""
    cand = Path(flight) / 'sim' / 'asflown' / 'layout.py' if flight else None
    p = cand if cand is not None and cand.exists() else Path('/replica/layout.py')
    spec = importlib.util.spec_from_file_location('layout', p)
    m = importlib.util.module_from_spec(spec); sys.modules['layout'] = m; spec.loader.exec_module(m)
    m.__flown_path__ = str(p)
    return m


def tunnel_geom(L):
    t = L.TUNNEL
    if 'axis_ab' in t:                                           # v4
        (ea, eb), (ua, ub), ln = t['entry_ab'], t['axis_ab'], t['length']
    else:                                                        # v3: axis along +east at a = t['a']
        hp = L.tunnel_hoops(); ea, eb, ua, ub, ln = t['a'], hp[0], 0.0, 1.0, hp[-1] - hp[0]
    return dict(entry=(ea, eb), axis=(ua, ub), right=(-ub, ua), length=ln, centre_h=t['centre_h'],
                radius=t['radius'], mesh_r=t['mesh_r'])


def tunnel_local(L, a, b, h):
    g = tunnel_geom(L); (ea, eb), (ua, ub), (ra, rb) = g['entry'], g['axis'], g['right']
    s = (a - ea) * ua + (b - eb) * ub
    return s, math.hypot((a - ea) * ra + (b - eb) * rb, h - g['centre_h'])


def solids(L):
    out = []
    for p in L.PANELS:
        out.append((p['name'], 'obox', (p['a'], p['b'], p['w'] / 2, p['t'] / 2, p['yaw'], 0.0, p['top'])))
    w = L.WINDOW; hw, hh, ht = w['hole_w'] / 2, w['hole_h'] / 2, w['thick'] / 2
    side = (w['width'] / 2 - hw) / 2; off = (w['width'] / 2 + hw) / 2
    if 'yaw' in w:                                               # v4: board long axis at yaw (deg, north -> east)
        y = w['yaw']; ua, ub = math.cos(math.radians(y)), math.sin(math.radians(y))
    else:                                                        # v3: board along east (yaw 90)
        y, ua, ub = 90.0, 0.0, 1.0
    out.append(('window_side_neg', 'obox', (w['a'] - off * ua, w['b'] - off * ub, side, ht, y, 0.0, w['top'])))
    out.append(('window_side_pos', 'obox', (w['a'] + off * ua, w['b'] + off * ub, side, ht, y, 0.0, w['top'])))
    out.append(('window_below', 'obox', (w['a'], w['b'], hw, ht, y, 0.0, w['hole_c'] - hh)))
    out.append(('window_above', 'obox', (w['a'], w['b'], hw, ht, y, w['hole_c'] + hh, w['top'])))
    wl = getattr(L, 'WALL', None)
    if wl and 'a' in wl and 'width' in wl:                       # v3: separate wall board (v4: the barrier is a PANEL)
        out.append(('wall', 'obox', (wl['a'], wl['b'], wl['width'] / 2, wl['thick'] / 2, 0.0, 0.0, wl['top'])))
    for k, (n, e, yw) in enumerate(L.CINDER_BLOCKS):
        out.append((f'block{k}', 'obox', (n + L.START[0], e + L.START[1], L.BLOCK[0] / 2, L.BLOCK[1] / 2, yw, 0.0, L.BLOCK[2])))
    for k, (a, b, h, r) in enumerate(L.ROCKS):
        out.append((f'rock{k}', 'cyl', (a, b, r, 0.0, h)))
    if getattr(L, 'ASTRONAUT_H', 0.0) > 0:                       # v3: astronaut on top of ROCKS[0]
        r0 = L.ROCKS[0]
        out.append(('astronaut', 'cyl', (r0[0], r0[1], 0.03, r0[2], r0[2] + L.ASTRONAUT_H)))
    return out


def dist(kind, q, a, b, h):
    if kind == 'obox':
        ca, cb, hl, hp, yaw, h0, h1 = q; ya = math.radians(yaw); da, db = a - ca, b - cb
        al = da * math.cos(ya) + db * math.sin(ya); pe = -da * math.sin(ya) + db * math.cos(ya)
        ex, ey, ez = max(abs(al) - hl, 0), max(abs(pe) - hp, 0), max(h0 - h, h - h1, 0)
        return math.sqrt(ex * ex + ey * ey + ez * ez)
    ca, cb, r, h0, h1 = q
    return math.hypot(max(math.hypot(a - ca, b - cb) - r, 0), max(h0 - h, h - h1, 0))


def clearance(L, a, b, h, _cache={}):
    key = id(L)
    if key not in _cache:
        _cache[key] = solids(L)
    return min((dist(kind, q, a, b, h), name) for name, kind, q in _cache[key])
