"""Venue replica layout: the single source of truth for the world geometry and the
ground-truth labels.

Version 4 (2026-11-07): follows the organizers' TOP-VIEW course map, transcribed in
cartographer/official/COURSE_MAP.md (1-ft grid, about +-0.3 ft / +-0.1 m). Map coordinates
are kept here verbatim in FEET (x = east, y = north, origin = SW corner of the 20 x 20 ft
arena) and converted with FT = 0.3048. What the map does not give (board heights above the
floor, A-frames, skins, tunnel look, window board size, pad look, boulder height, props) is
estimated from the venue photos 1-18; those values are marked "photo" below.

Frames
  map (ft):         x east, y north from the arena's SW corner.
  cage coords (m):  a = north (= y*FT), b = east (= x*FT), h = height above floor.
  NED (common/geometry.py): origin = the launch point on the floor, drone nose = +n at take-off,
                    n = a - START[0], e = b - START[1], d = -h.
  start frame:      f = n (forward), l = -e (left).    (= COURSE_MAP.md "start frame")
  Gazebo world:     x = n, y = -e, z = h   (FLU world; NED = diag(1,-1,-1) gz).

Ground-truth labels are for OFFLINE evaluation only and must never be imported by flight code.
"""
import math
import numpy as np

FT = 0.3048
CAGE = 20 * FT                # 6.096 m square net cage (map)
CAGE_H = 3.0                  # INTERFACES.md
TARP = 0.25                   # orange tarp sticks out past the cage net (photo)
FLOOR_SIZE = CAGE + 2 * TARP  # floor texture covers this square
LAUNCH_FT = (1.74, 13.0)      # map: launch point, drone faces north
START = (LAUNCH_FT[1] * FT, LAUNCH_FT[0] * FT)    # cage coords (a, b) of the launch
START_BOX = 0.0               # no marking at the real start (user, 2026-11-07)
START_PAD_SIZE = START_BOX


def ft_ab(x, y):
    """map feet (x east, y north) -> cage coords (a north, b east) in metres."""
    return y * FT, x * FT


def ft_ned(x, y, h=0.0):
    a, b = ft_ab(x, y)
    return np.array([a - START[0], b - START[1], -h], float)


def ft_fl(x, y):
    """map feet -> start frame (f forward = north, l left = west)."""
    return (y - LAUNCH_FT[1]) * FT, -(x - LAUNCH_FT[0]) * FT


# --- walls (map): all walls are 68 in = 1.73 m tall ------------------------------------
# Board look from the photos: plywood with a black strip at the bottom, or plywood covered with a
# printed sci-fi banner, standing on wooden A-frames with floor runners. Board bottom above the
# floor (open gap under the board, photos 3, 7, 10, 12, 17) is a photo estimate.
WALL_TOP = 68 * 0.0254        # 1.727 m (map)
WALLS_FT = [
    # name,      (x0, y0),      (x1, y1),     skin,      role
    ('barrier',  (0.0, 11.5),   (15.2, 11.5), 'plywood', 'separates north (corridor, maze, tunnel) from south (chicane, landing); flown over on the return'),
    ('W1',       (3.7, 11.6),   (3.7, 16.9),  'banner',  'right wall of the first corridor (x 0-3.7)'),
    ('W2',       (7.35, 14.8),  (7.35, 19.85), 'plywood', 'maze wall hanging from the north net (map y 20; stops 5 cm short of the net)'),
    ('W3',       (11.0, 11.6),  (11.0, 16.9), 'banner',  'maze wall; Gate A = gap x 11, y 16.9-20'),
    ('C1',       (14.9, 11.2),  (13.2, 6.3),  'plywood', 'chicane wall (diagonal)'),
    ('C2_north', (10.5, 8.1),   (9.254, 4.805), 'plywood', 'chicane wall C2 north of the window board; Gate B = gap x 10.5, y 8.1-11.5'),
    ('W4',       (6.8, 8.0),    (6.8, 11.3),  'banner',  'short wall between the Gate B side and the landing area'),
]
PANEL_W, PANEL_BOTTOM, PANEL_TOP, PANEL_T = 1.60, 0.45, WALL_TOP, 0.05     # banner boards (photo)
PLY_W, PLY_BOTTOM, PLY_TOP, PLY_T, PLY_STRIP = 2.44, 0.35, WALL_TOP, 0.02, 0.18   # plywood boards (photo)
PANEL_GAP = 0.02              # gap between neighbouring boards of one wall
FRAME_FOOT = 0.60             # wooden foot rails, perpendicular to the board (photo)
MAZE_WALLS = [(n, ft_fl(*p0), ft_fl(*p1), skin, role) for (n, p0, p1, skin, role) in WALLS_FT]   # start frame


def _panels_from_walls(walls_ft):
    """Split every wall into boards (banner <= PANEL_W, plywood <= PLY_W); cage coords + yaw."""
    out, k = [], 0
    for name, (x0, y0), (x1, y1), skin, role in walls_ft:
        a0, b0 = ft_ab(x0, y0)
        a1, b1 = ft_ab(x1, y1)
        ln = math.hypot(a1 - a0, b1 - b0)
        wmax = PANEL_W if skin == 'banner' else PLY_W
        n = max(1, math.ceil(ln / wmax - 1e-6))
        yaw = math.degrees(math.atan2(b1 - b0, a1 - a0))
        yaw = (yaw + 90.0) % 180.0 - 90.0
        bot, top, thk = (PANEL_BOTTOM, PANEL_TOP, PANEL_T) if skin == 'banner' else (PLY_BOTTOM, PLY_TOP, PLY_T)
        for i in range(n):
            t = (i + 0.5) / n
            out.append(dict(name=f'{name}_{i}', wall=name, skin=skin, source='map (COURSE_MAP.md)', status='map',
                            a=a0 + t * (a1 - a0), b=b0 + t * (b1 - b0), yaw=yaw, w=ln / n - PANEL_GAP,
                            bottom=bot, top=top, t=thk, mirror=bool(k % 2)))
            k += 1
    return out


PANELS = _panels_from_walls(WALLS_FT)

# --- window: a plywood board with a rectangular hole, part of chicane wall C2 ----------------
# Map: C2 runs (10.5, 8.1) -> (7.5, 0.0); window section ~(7.8, 1.0)-(9.0, 4.0), centre ~(8.4, 2.5);
# crossing direction WNW, normal (-0.937, 0.347). Board width, hole size / height, black strip:
# photos 3, 7, 17 (hole / board ~0.50 wide, ~0.46 tall; board width 1.5 m assumed).
_C2 = np.array([10.5 - 7.5, 8.1 - 0.0]); _C2 /= np.linalg.norm(_C2)          # along C2 (NNE), map x/y
WINDOW_FT = dict(cx=8.4, cy=2.5, along=(float(_C2[0]), float(_C2[1])), cross=(-0.937, 0.347))
WINDOW = dict(width=1.50, bottom=0.35, top=WALL_TOP, thick=0.02, hole_w=0.75, hole_h=0.60, hole_c=1.21, strip_h=0.18)
WINDOW['a'], WINDOW['b'] = ft_ab(WINDOW_FT['cx'], WINDOW_FT['cy'])
WINDOW['yaw'] = math.degrees(math.atan2(WINDOW_FT['along'][0], WINDOW_FT['along'][1]))   # long axis, north->east
# normal pointing to the APPROACH side (ESE), in NED (n, e):
WINDOW['normal_ne'] = (-WINDOW_FT['cross'][1], -WINDOW_FT['cross'][0])

# --- the wall flown over on the return = the barrier (map y = 11.5 ft, x 0 -> 15.2) ----------
WALL = dict(name='barrier', x0=0.0, x1=15.2, y=11.5, top=WALL_TOP, bottom=PLY_BOTTOM, thick=PLY_T, strip_h=PLY_STRIP)

# --- tunnel (map): box x 15.2-20, y 9.0-14.0, entered at the NORTH face, exit SOUTH face ---------
# "dark inside". Look from photos 2, 16: full brass hoops on U-feet + black mesh liner + rock-print
# fabric on top. Hoops at both faces and in the middle; radius / height are photo estimates.
TUNNEL = dict(entry_ft=(17.6, 14.0), exit_ft=(17.6, 9.0), n_hoops=3, radius=0.55, centre_h=0.62,
              tube_r=0.006, mesh_r=0.545, fabric_r=0.565,
              mesh_deg=(115.0, 300.0),     # mesh arc, deg clockwise from the top looking along the axis
              fabric_deg=(-75.0, 125.0))   # fabric arc (drapes further down the right side)
TUNNEL['entry_ab'] = ft_ab(*TUNNEL['entry_ft'])
TUNNEL['exit_ab'] = ft_ab(*TUNNEL['exit_ft'])
_ax = np.subtract(TUNNEL['exit_ab'], TUNNEL['entry_ab'])
TUNNEL['length'] = float(np.linalg.norm(_ax))
TUNNEL['axis_ab'] = tuple(float(v) for v in _ax / TUNNEL['length'])      # (north, east) unit = NED (n, e)

# --- landing (map): ring target centre (3.3, 7.35) ft, outer ring radius ~0.78 m ---------------
# The photos (17, 18) show a flat light-blue folding pad (~0.6 m) with a dark H, a white ring and 4
# dark bars. Until the user confirms: the blue H pad at the ring centre + a printed ring (r 0.78 m).
_pad_n, _pad_e = ft_ned(3.3, 7.35)[:2]
PAD = dict(n=float(_pad_n), e=float(_pad_e), size=0.60, thick=0.005, yaw=20.0, ring_r=0.78, ring_w=0.05)

# --- floor tape: the map's red path line = black tape in the photos (10-15) --------------------
TAPE_W = 0.048
TAPE_FT = [('maze_path', [(1.74, 13.0), (1.74, 18.5), (5.5, 18.5), (5.5, 13.2), (9.2, 13.2), (9.2, 18.5), (11.0, 18.5)])]
TAPE = [(n, [ft_fl(*p) for p in pts]) for n, pts in TAPE_FT]              # start frame (f, l)

# --- boulders (map): diameter 18 in = 0.46 m; height + look (grey stone) from the photos ----------
BOULDERS_FT = [(1.6, 1.8), (5.6, 2.2), (12.6, 2.3), (16.6, 1.9), (18.3, 4.9)]
BOULDER_H = 0.35
BOULDERS = [(*ft_ab(x, y), BOULDER_H, 0.23) for x, y in BOULDERS_FT]      # (a, b, h, base radius)
ROCKS = BOULDERS              # 3-D obstacles (labels 'rocks'); the photos' red rock props are not on the map

# grey cinder blocks weighing down the A-frame runners at wall ends (photo; not on the map)
BLOCK = (0.30, 0.20, 0.15)
_BLOCKS_FT = [(4.1, 11.9, 0.0), (4.1, 16.6, 0.0), (7.0, 15.2, 0.0), (11.4, 11.9, 0.0), (10.6, 16.6, 0.0),
              (14.5, 10.9, 70.0), (13.6, 6.0, 70.0), (7.2, 8.3, 0.0), (7.2, 11.0, 0.0), (9.9, 5.4, 70.0),
              (2.2, 11.15, 90.0), (12.6, 11.15, 90.0)]
CINDER_BLOCKS = [(*ft_ned(x, y)[:2], yaw) for x, y, yaw in _BLOCKS_FT]    # NED n, e, yaw (deg)
ASTRONAUT_H = 0.0             # the photos' astronaut figurine is not on the map: left out

# --- flight route (map path + our continuation through the stages) ------------------------------
ROUTE_FT = [(1.74, 13.0), (1.74, 18.5), (5.5, 18.5), (5.5, 13.2), (9.2, 13.2), (9.2, 18.5), (11.0, 18.5),   # map
            (17.6, 18.5), (17.6, 14.6), (17.6, 9.0), (17.6, 7.6),       # to the tunnel's north face, through it
            (16.5, 5.2), (13.0, 4.6), (11.0, 3.6), (10.74, 1.63),       # around C1's south end to the window
            (8.4, 2.5), (6.5, 3.2), (3.8, 6.6)]                         # through the window (WNW) to the pad
GATE_B_ROUTE_FT = [(13.0, 4.6), (11.6, 9.8), (10.5, 9.8), (8.6, 9.8), (7.4, 7.2), (3.8, 7.0)]   # alternative

# --- cage frame + net ---------------------------------------------------------------
PIPE_R = 0.017
RAIL_H = (PIPE_R, 1.90, CAGE_H)
POSTS_PER_SIDE = 3            # posts every CAGE/3
NET_CELL, NET_CORD, NET_OFFSET = 0.06, 0.0025, 0.03

# --- hall outside the cage -----------------------------------------------------------
HALL = dict(len_a=26.0, len_b=22.0, height=6.5)
TABLES = [(-1.8, 1.5, 90.0), (-1.8, 4.0, 90.0), (2.5, 7.6, 0.0), (7.6, 3.0, 90.0)]
PLYBOX = dict(a=4.5, b=7.5, len_a=2.0, len_b=0.6, height=2.1)


def cage_to_ned(a, b, h=0.0):
    return np.array([a - START[0], b - START[1], -h], float)


def ned_to_gz(p):
    p = np.asarray(p, float)
    return np.array([p[..., 0], -p[..., 1], -p[..., 2]]).T if p.ndim > 1 else np.array([p[0], -p[1], -p[2]])


def cage_to_gz(a, b, h=0.0):
    return ned_to_gz(cage_to_ned(a, b, h))


def ned_to_floor_uv(n, e):
    """Floor texture coords in metres: u east from the NW tarp corner, v south."""
    a, b = n + START[0], e + START[1]
    return b + TARP, (CAGE + TARP) - a


def floor_uv_to_ned(u, v):
    a, b = (CAGE + TARP) - v, u - TARP
    return cage_to_ned(a, b)


def fl_to_ned(f, l, h=0.0):
    """Start frame (f forward, l left) -> NED."""
    return np.array([f, -l, -h], float)


def pad_corners_ned():
    """Landing-pad corners (NED, on its top surface), counter-clockwise seen from above."""
    p = PAD
    y = math.radians(p['yaw'])
    u = np.array([math.cos(y), math.sin(y)]); v = np.array([-math.sin(y), math.cos(y)])
    c = np.array([p['n'], p['e']]); h = p['size'] / 2
    return [np.array([*(c + su * h * u + sv * h * v), -p['thick']]) for su, sv in ((1, 1), (1, -1), (-1, -1), (-1, 1))]


def tunnel_hoops():
    """Hoop centres in cage coords (a, b), entry first."""
    t = TUNNEL
    return [tuple(np.array(t['entry_ab']) + np.array(t['axis_ab']) * t['length'] * i / (t['n_hoops'] - 1))
            for i in range(t['n_hoops'])]


def tunnel_right():
    """Unit (a, b) vector to the right of the tunnel axis (looking along it, seen from above)."""
    ua, ub = TUNNEL['axis_ab']
    return (-ub, ua)


def window_frame():
    """Window board local frame in NED: centre (3,), along (3,) (board long axis), normal (3,) to
    the approach side, up (3,)."""
    w = WINDOW
    c = cage_to_ned(w['a'], w['b'], w['hole_c'])
    y = math.radians(w['yaw'])
    along = np.array([math.cos(y), math.sin(y), 0.0])
    normal = np.array([w['normal_ne'][0], w['normal_ne'][1], 0.0])
    return c, along, normal, np.array([0, 0, -1.0])


def labels_world(painted_rocks=None):
    """Static ground truth (NED). painted_rocks: list from textures.json."""
    t, w, wl, p = TUNNEL, WINDOW, WALL, PAD
    hoops = tunnel_hoops()
    wc, w_along, w_normal, _ = window_frame()
    bar0, bar1 = ft_ned(wl['x0'], wl['y']), ft_ned(wl['x1'], wl['y'])
    out = {
        'frame': 'NED, origin = launch point on the floor (map (1.74, 13.0) ft), metres; start frame f = n, l = -e',
        'source': 'organizers top-view course map (cartographer/official/COURSE_MAP.md) + venue photos 1-18',
        'estimated': True,
        'cage': {'n_range': [-START[0], CAGE - START[0]], 'e_range': [-START[1], CAGE - START[1]],
                 'height': CAGE_H, 'floor_d': 0.0},
        'start_pad': {'center_ned': [0.0, 0.0, 0.0], 'size_m': 0.0, 'launch_ft': list(LAUNCH_FT),
                      'note': 'no marking at the start; the drone starts here heading +n (north)'},
        'tunnel': {
            'entry_ned': cage_to_ned(*hoops[0], t['centre_h']).tolist(),
            'exit_ned': cage_to_ned(*hoops[-1], t['centre_h']).tolist(),
            'axis_ned': list(t['axis_ab']),
            'radius_m': t['radius'], 'inner_radius_m': t['mesh_r'], 'length_m': t['length'],
            'hoop_centers_ned': [cage_to_ned(a, b, t['centre_h']).tolist() for a, b in hoops],
            'box_ft': {'x': [15.2, 20.0], 'y': [9.0, 14.0]},
            'note': 'enter at the north face, exit at the south face (map); full brass hoops at both faces + middle (photos)'},
        'window': {
            'center_ned': wc.tolist(), 'normal_ned': w_normal.tolist(), 'along_ned': w_along.tolist(),
            'width_m': w['hole_w'], 'height_m': w['hole_h'],
            'board_width_m': w['width'], 'board_bottom_h': w['bottom'], 'board_top_h': w['top'],
            'black_strip_h': [w['bottom'], w['bottom'] + w['strip_h']], 'center_ft': [WINDOW_FT['cx'], WINDOW_FT['cy']],
            'note': 'part of chicane wall C2; normal points to the approach side (ESE), crossing direction = -normal (WNW)'},
        'gates': {'gate_a': {'from_ned': ft_ned(11.0, 16.9).tolist(), 'to_ned': ft_ned(11.0, 20.0).tolist()},
                  'gate_b': {'from_ned': ft_ned(10.5, 8.1).tolist(), 'to_ned': ft_ned(10.5, 11.5).tolist()}},
        'landing_target': {
            'type': 'flat_pad',
            'center_ned': [p['n'], p['e'], -p['thick'] / 2], 'top_center_ned': [p['n'], p['e'], -p['thick']],
            'top_d': -p['thick'], 'extent_m': [p['size'], p['size'], p['thick']], 'yaw_deg': p['yaw'],
            'corners_ned': [c.tolist() for c in pad_corners_ned()], 'ring_radius_m': p['ring_r'],
            'look': 'light blue square, dark H, white ring + dashed inner ring, 4 dark bars; printed ring r 0.78 m around it',
            'note': 'map ring target centre (3.3, 7.35) ft; pad look from photos 17-18 until the user confirms'},
        'wall': {
            'name': 'barrier', 'center_ned': ((bar0 + bar1) / 2 + np.array([0, 0, -(wl['top'] + wl['bottom']) / 2])).tolist(),
            'normal_ned': [-1.0, 0.0, 0.0], 'width_m': float(np.linalg.norm(bar1 - bar0)),
            'bottom_h': wl['bottom'], 'top_h': wl['top'], 'top_d': -wl['top'],
            'top_edge_ned': [(bar0 + np.array([0, 0, -wl['top']])).tolist(), (bar1 + np.array([0, 0, -wl['top']])).tolist()],
            'note': 'map barrier y = 11.5 ft, x 0-15.2, flown over on the return (landing -> launch); normal to the landing side'},
        'rocks': [{'base_ned': cage_to_ned(a, b).tolist(), 'apex_ned': cage_to_ned(a, b, h).tolist(),
                   'height_m': h, 'base_radius_m': r, 'kind': 'boulder'} for (a, b, h, r) in ROCKS],
        'panels': [{'name': q['name'], 'wall': q['wall'], 'skin': q['skin'], 'source': q['source'],
                    'center_ned': cage_to_ned(q['a'], q['b'], (q['bottom'] + q['top']) / 2).tolist(),
                    'yaw_deg': q['yaw'], 'width_m': q['w'], 'bottom_h': q['bottom'], 'top_h': q['top'],
                    'thick_m': q['t']} for q in PANELS],
        'maze_walls': [{'name': n, 'start_ft': list(p0), 'end_ft': list(p1), 'start_ned': ft_ned(*p0)[:2].tolist(),
                        'end_ned': ft_ned(*p1)[:2].tolist(), 'start_fl': list(ft_fl(*p0)), 'end_fl': list(ft_fl(*p1)),
                        'skin': sk, 'role': role} for (n, p0, p1, sk, role) in WALLS_FT],
        'maze_path_fl': dict(TAPE)['maze_path'],
        'route_ft': [list(p) for p in ROUTE_FT],
        'tape': [{'name': n, 'width_m': TAPE_W, 'polyline_ned': [[f, -l] for f, l in pts]} for n, pts in TAPE],
        'cinder_blocks': [{'center_ned': [n, e, -BLOCK[2] / 2], 'yaw_deg': y, 'size_m': list(BLOCK)}
                          for (n, e, y) in CINDER_BLOCKS],
    }
    if painted_rocks is not None:
        out['painted_rocks_flat'] = [{'center_ned': floor_uv_to_ned(r['u_m'], r['v_m']).tolist(),
                                      'size_m': r['size_m']} for r in painted_rocks]
    return out


# ----------------------------------------------------------------------------- collision
def _solids():
    """Coarse solid volumes for camera-pose rejection: list of
    (kind, params) with kind 'obox' (centre a,b, half-len along/perp, yaw, h0, h1)
    or 'cyl' (a, b, r, h0, h1) or 'tunnel'."""
    S = []
    for p in PANELS:
        S.append(('obox', (p['a'], p['b'], p['w'] / 2, p['t'] / 2 + 0.02, p['yaw'], 0.0, p['top'])))
    w = WINDOW
    hw, hh = w['hole_w'] / 2, w['hole_h'] / 2
    y = math.radians(w['yaw'])
    ua, ub = math.cos(y), math.sin(y)
    for s0, half, h0, h1 in (((w['width'] / 2 + hw) / 2, (w['width'] / 2 - hw) / 2, 0.0, w['top']),
                             (-(w['width'] / 2 + hw) / 2, (w['width'] / 2 - hw) / 2, 0.0, w['top']),
                             (0.0, hw, 0.0, w['hole_c'] - hh), (0.0, hw, w['hole_c'] + hh, w['top'])):
        S.append(('obox', (w['a'] + s0 * ua, w['b'] + s0 * ub, half, 0.03, w['yaw'], h0, h1)))
    for (n, e, yw) in CINDER_BLOCKS:
        S.append(('obox', (n + START[0], e + START[1], BLOCK[0] / 2, BLOCK[1] / 2, yw, 0.0, BLOCK[2])))
    for (a, b, h, r) in ROCKS:
        S.append(('cyl', (a, b, r, 0.0, h)))
    S.append(('tunnel', None))
    return S


def pose_is_free(p_ned, margin=0.15):
    """True if the camera position is outside all solids (+margin) and inside the cage."""
    a, b, h = p_ned[0] + START[0], p_ned[1] + START[1], -p_ned[2]
    if not (margin < a < CAGE - margin and margin < b < CAGE - margin and 0.05 < h < CAGE_H - margin):
        return False
    for kind, q in _solids():
        if kind == 'obox':
            ca, cb, hl, hp, yaw, h0, h1 = q
            if not (h0 - margin <= h <= h1 + margin):
                continue
            ya = math.radians(yaw)
            da, db = a - ca, b - cb
            along = da * math.cos(ya) + db * math.sin(ya)
            perp = -da * math.sin(ya) + db * math.cos(ya)
            if abs(along) <= hl + margin and abs(perp) <= hp + margin:
                return False
        elif kind == 'cyl':
            ca, cb, r, h0, h1 = q
            if h <= h1 + margin and math.hypot(a - ca, b - cb) <= r + margin:
                return False
        else:
            t = TUNNEL
            ea, eb = t['entry_ab']; ua, ub = t['axis_ab']; ra, rb = tunnel_right()
            s = (a - ea) * ua + (b - eb) * ub
            if -margin - 0.1 <= s <= t['length'] + margin + 0.1:
                lat = (a - ea) * ra + (b - eb) * rb
                rr = math.hypot(lat, h - t['centre_h'])
                inside = rr < t['radius'] - margin and -0.1 <= s <= t['length'] + 0.1
                if not inside and rr < t['fabric_r'] + margin + 0.05:
                    return False
    return True
