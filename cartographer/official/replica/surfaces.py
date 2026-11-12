"""Per-pixel surface type for a replica frame, from the two ground-truth depth maps
and the layout geometry (no extra render needed).

  depth_full  : nearest surface incl. cage net + tunnel black mesh
  depth_ideal : same scene without the net / mesh
A pixel whose depth_full is clearly in front of depth_ideal hit the net or the mesh;
every other pixel is classified by where its 3-D point lies (layout.py primitives).
"""
import math
import numpy as np
import layout as L

NAMES = ['other', 'floor', 'banner_panel', 'brass_ring', 'tunnel_fabric', 'black_mesh', 'cage_net',
         'tan_board', 'landing_pad', 'cinder_block', 'rock', 'hall', 'floor_tape']
ID = {n: i for i, n in enumerate(NAMES)}


def _points(depth, K, R_nc, cam):
    fx, fy, cx, cy = K
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth
    pc = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], -1)
    P = pc @ np.asarray(R_nc).T + np.asarray(cam)
    a = P[..., 0] + L.START[0]; b = P[..., 1] + L.START[1]; h = -P[..., 2]
    return a, b, h


def classify(depth_ideal, depth_full, K, R_nc, cam_ned):
    lab = np.zeros(depth_ideal.shape, np.uint8)
    fin_i, fin_f = np.isfinite(depth_ideal), np.isfinite(depth_full)
    a, b, h = _points(np.where(fin_i, depth_ideal, 0), K, R_nc, cam_ned)
    af, bf, hf = _points(np.where(fin_f, depth_full, 0), K, R_nc, cam_ned)
    T = L.TUNNEL
    ea, eb = T['entry_ab']; ua, ub = T['axis_ab']; ra, rb = L.tunnel_right()
    t_s = lambda aa, bb: (aa - ea) * ua + (bb - eb) * ub          # along the tunnel axis from the entry hoop
    t_l = lambda aa, bb: (aa - ea) * ra + (bb - eb) * rb          # lateral (right) offset from the axis
    in_tunnel_box = lambda aa, bb, hh: ((np.abs(t_l(aa, bb)) < T['fabric_r'] + 0.12) & (t_s(aa, bb) > -0.2)
                                        & (t_s(aa, bb) < T['length'] + 0.2) & (hh < T['centre_h'] + T['fabric_r'] + 0.12))
    mesh_hit = fin_f & (~fin_i | (depth_full < np.where(fin_i, depth_ideal, np.inf) - np.maximum(0.02, 0.01 * depth_full)))
    lab[mesh_hit & in_tunnel_box(af, bf, hf)] = ID['black_mesh']
    lab[mesh_hit & ~in_tunnel_box(af, bf, hf)] = ID['cage_net']
    ok = fin_i & ~mesh_hit
    # hall: outside the cage + tarp
    hall = ok & ((a < -0.3) | (a > L.CAGE + 0.3) | (b < -0.3) | (b > L.CAGE + 0.3))
    lab[hall] = ID['hall']
    rest = ok & ~hall
    lab[rest & (h < 0.012)] = ID['floor']
    # banner panels
    for p in L.PANELS:
        y = math.radians(p['yaw'])
        da, db = a - p['a'], b - p['b']
        along = da * math.cos(y) + db * math.sin(y); perp = -da * math.sin(y) + db * math.cos(y)
        m = rest & (np.abs(perp) < p['t'] / 2 + 0.02) & (np.abs(along) < p['w'] / 2 + 0.01) & \
            (h > p['bottom'] - 0.01) & (h < p['top'] + 0.01)
        lab[m] = ID['banner_panel'] if p['skin'] == 'banner' else ID['tan_board']   # plywood maze boards
    # window board (tan plywood, oriented along chicane wall C2); the barrier etc. are in PANELS
    w = L.WINDOW
    y = math.radians(w['yaw'])
    da, db = a - w['a'], b - w['b']
    wal, wpe = da * math.cos(y) + db * math.sin(y), -da * math.sin(y) + db * math.cos(y)
    m = rest & (np.abs(wpe) < 0.03) & (np.abs(wal) < w['width'] / 2 + 0.01) & (h > w['bottom'] - 0.01) & (h < w['top'] + 0.01)
    lab[m] = ID['tan_board']
    # landing pad (flat, on the floor) and floor tape: both inside the floor band
    pd = L.PAD
    y = math.radians(pd['yaw'])
    pn, pe = a - L.START[0] - pd['n'], b - L.START[1] - pd['e']
    pu, pv = pn * math.cos(y) + pe * math.sin(y), -pn * math.sin(y) + pe * math.cos(y)
    lab[rest & (h < 0.012) & (np.abs(pu) < pd['size'] / 2) & (np.abs(pv) < pd['size'] / 2)] = ID['landing_pad']
    n_, e_ = a - L.START[0], b - L.START[1]
    for _, pts in L.TAPE:
        for (f0, l0), (f1, l1) in zip(pts[:-1], pts[1:]):
            d = np.array([f1 - f0, -(l1 - l0)]); ln = np.linalg.norm(d); d = d / ln
            rn, re = n_ - f0, e_ + l0
            t = np.clip(rn * d[0] + re * d[1], -L.TAPE_W / 2, ln + L.TAPE_W / 2)
            dist = np.hypot(rn - t * d[0], re - t * d[1])
            lab[rest & (h < 0.012) & (dist < L.TAPE_W / 2) & (lab == ID['floor'])] = ID['floor_tape']
    # cinder blocks
    bl, bw, bh = L.BLOCK
    for (bn, be, yaw) in L.CINDER_BLOCKS:
        y = math.radians(yaw)
        dn, de = n_ - bn, e_ - be
        u_, v_ = dn * math.cos(y) + de * math.sin(y), -dn * math.sin(y) + de * math.cos(y)
        lab[rest & (np.abs(u_) < bl / 2 + 0.015) & (np.abs(v_) < bw / 2 + 0.015) & (h > 0.012) & (h < bh + 0.015)] = ID['cinder_block']
    # rocks
    for (ra, rb, rh, rr) in L.ROCKS:
        m = rest & (np.hypot(a - ra, b - rb) < rr * 1.35 + 0.03) & (h > 0.012) & (h < rh + 0.03)
        lab[m] = ID['rock']
    # tunnel: brass rings, then fabric
    rr = np.hypot(t_l(a, b), h - T['centre_h'])
    for i_h in range(T['n_hoops']):
        s_h = T['length'] * i_h / (T['n_hoops'] - 1)
        m = rest & (np.abs(t_s(a, b) - s_h) < 0.025) & (np.abs(rr - T['radius']) < 0.025)
        lab[m] = ID['brass_ring']
    m = rest & in_tunnel_box(a, b, h) & (rr > T['mesh_r'] - 0.02) & (lab != ID['brass_ring']) & (h > 0.012)
    lab[m] = ID['tunnel_fabric']
    return lab
