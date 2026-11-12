#!/usr/bin/env python3
"""Generate the Gazebo Harmonic venue replica (meshes + two SDF worlds) from layout.py.

Usage:  python build_world.py [--out world] [--textures assets/textures]
Writes  OUT/meshes/*.obj, OUT/textures/*, OUT/venue_full.sdf, OUT/venue_depth.sdf.

venue_full.sdf  : everything + left/right RGB cameras (1280x800, 2x supersampled, 7.5 cm baseline) + depth camera.
venue_depth.sdf : same scene WITHOUT the cage net and the tunnel's black mesh
                  (stereo depth barely sees thin black mesh, INTERFACES.md) + depth camera.
All mesh vertices are written directly in Gazebo world coordinates (x=n, y=-e, z=h).
"""
import argparse, math, os, shutil, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import layout as L                                    # noqa: E402
sys.path.insert(0, os.path.dirname(HERE))
from common.geometry import OFFICIAL_K, OFFICIAL_SIZE   # noqa: E402

RNG = np.random.default_rng(11)
SS = 2
BASELINE = 0.075           # OAK-D left -> right camera, along the optical +x (right) axis                    # RGB supersampling factor (area-averaged back to 640x400)


def G(a, b, h):
    """cage coords -> gz world (vectorised)."""
    a, b, h = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float), np.asarray(h, float))
    return np.stack([a - L.START[0], -(b - L.START[1]), h], -1)


# ----------------------------------------------------------------------------- mesh
class Mesh:
    def __init__(self):
        self.v, self.t, self.n, self.f = [], [], [], []

    def _add(self, P, T, N):
        base = len(self.v)
        self.v += list(map(tuple, P)); self.t += list(map(tuple, T)); self.n += list(map(tuple, N))
        return base

    def tri(self, P, T, N=None, out=None):
        """One triangle, P (3,3) gz coords, T (3,2) image uv (v from top).
        Winding is fixed so the geometric normal agrees with N / points away from `out`."""
        P = np.asarray(P, float); T = np.asarray(T, float)
        g = np.cross(P[1] - P[0], P[2] - P[0])
        if np.linalg.norm(g) < 1e-12:
            return
        if N is None:
            gn = g / np.linalg.norm(g)
            if out is not None and np.dot(gn, P.mean(0) - out) < 0:
                gn = -gn
            N = np.repeat(gn[None], 3, 0)
        N = np.asarray(N, float)
        order = [0, 1, 2] if np.dot(g, N.sum(0)) >= 0 else [0, 2, 1]
        b = self._add(P[order], T[order], N[order])
        self.f.append((b, b + 1, b + 2))

    def grid(self, P, T, out=None, double=False, flat=False, flip=False):
        """Parametric surface P (m,n,3), T (m,n,2). Normals point away from `out`
        (a point) or follow the grid orientation (flip to reverse)."""
        m, n = P.shape[:2]
        du = np.gradient(P, axis=1); dv = np.gradient(P, axis=0)
        N = np.cross(du, dv)
        N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12
        if flip:
            N = -N
        if out is not None:
            s = np.sign(np.sum(N * (P - out), -1, keepdims=True)); s[s == 0] = 1
            N = N * s
        for i in range(m - 1):
            for j in range(n - 1):
                for (q0, q1, q2) in (((i, j), (i, j + 1), (i + 1, j + 1)), ((i, j), (i + 1, j + 1), (i + 1, j))):
                    PP = np.array([P[q0], P[q1], P[q2]]); TT = np.array([T[q0], T[q1], T[q2]])
                    NN = None if flat else np.array([N[q0], N[q1], N[q2]])
                    if flat:
                        g = np.cross(PP[1] - PP[0], PP[2] - PP[0]); g /= np.linalg.norm(g) + 1e-12
                        if np.dot(g, (N[q0] + N[q1] + N[q2])) < 0:
                            g = -g
                        NN = np.repeat(g[None], 3, 0)
                    self.tri(PP, TT, NN)
                    if double:
                        self.tri(PP, TT, -NN)

    def quad(self, P, T=((0, 0), (1, 0), (1, 1), (0, 1)), normal=None, double=False):
        P = np.asarray(P, float); T = np.asarray(T, float)
        if normal is None:
            normal = np.cross(P[1] - P[0], P[3] - P[0])
        nn = np.asarray(normal, float); nn = nn / np.linalg.norm(nn)
        Nn = np.repeat(nn[None], 3, 0)
        for idx in ((0, 1, 2), (0, 2, 3)):
            self.tri(P[list(idx)], T[list(idx)], Nn)
            if double:
                self.tri(P[list(idx)], T[list(idx)], -Nn)

    def box(self, c, R, half, uv_scale=None, faces='all'):
        """Oriented box in gz coords: centre c, rotation R (cols = box axes), half extents."""
        c = np.asarray(c, float); half = np.asarray(half, float)
        for ax in range(3):
            if faces == 'long' and ax == int(np.argmax(half)):
                continue
            for s in (-1, 1):
                u, w = [k for k in range(3) if k != ax]
                nrm = R[:, ax] * s
                fc = c + nrm * half[ax]
                cu, cw = R[:, u] * half[u], R[:, w] * half[w]
                P = [fc - cu - cw, fc + cu - cw, fc + cu + cw, fc - cu + cw]
                if uv_scale:
                    su, sw = 2 * half[u] / uv_scale, 2 * half[w] / uv_scale
                    T = ((0, sw), (su, sw), (su, 0), (0, 0))
                else:
                    T = ((0, 1), (1, 1), (1, 0), (0, 0))
                self.quad(P, T, nrm)

    def write(self, path):
        with open(path, 'w') as f:
            f.write('# venue replica mesh (gz world coords)\n')
            for v in self.v:
                f.write('v %.5f %.5f %.5f\n' % v)
            for t in self.t:
                f.write('vt %.5f %.5f\n' % (t[0], 1.0 - t[1]))      # OBJ: v from bottom
            for n in self.n:
                f.write('vn %.4f %.4f %.4f\n' % n)
            for a, b, c in self.f:
                f.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (a + 1, a + 1, a + 1, b + 1, b + 1, b + 1, c + 1, c + 1, c + 1))
        return len(self.f)


def tube(mesh, pts, r, seg=8):
    """Polyline tube with outward normals (pipes, hoop feet)."""
    for p0, p1 in zip(pts[:-1], pts[1:]):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        ax = p1 - p0; ln = np.linalg.norm(ax)
        if ln < 1e-9:
            continue
        ax /= ln
        tmp = np.array([0, 0, 1.]) if abs(ax[2]) < 0.9 else np.array([1., 0, 0])
        e1 = np.cross(ax, tmp); e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)
        th = np.linspace(0, 2 * np.pi, seg + 1)
        ring = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
        for k in range(seg):
            P = np.array([p0 + r * ring[k], p0 + r * ring[k + 1], p1 + r * ring[k + 1], p1 + r * ring[k]])
            N = np.array([ring[k], ring[k + 1], ring[k + 1], ring[k]])
            T = np.array([[k / seg, 0], [(k + 1) / seg, 0], [(k + 1) / seg, 1], [k / seg, 1]])
            mesh.tri(P[[0, 1, 2]], T[[0, 1, 2]], N[[0, 1, 2]])
            mesh.tri(P[[0, 2, 3]], T[[0, 2, 3]], N[[0, 2, 3]])


def rot_yaw_gz(yaw_deg_ned):
    """Columns (along, right, up) in gz coords for a long axis at NED yaw (deg):
    along = (cos, sin) in (n, e); right = along turned 90 deg clockwise (seen from above)."""
    y = math.radians(yaw_deg_ned)
    along = np.array([math.cos(y), -math.sin(y), 0.0])
    right = np.array([-math.sin(y), -math.cos(y), 0.0])
    return np.stack([along, right, [0, 0, 1.0]], 1)


# ----------------------------------------------------------------------------- builders
def build_floor():
    m = Mesh()
    lo, hi = -L.TARP, L.CAGE + L.TARP
    P = [G(lo, lo, 0), G(lo, hi, 0), G(hi, hi, 0), G(hi, lo, 0)]
    # image uv: u = (b + TARP)/FS (east), v = (CAGE + TARP - a)/FS (from the north edge)
    uv = lambda a, b: ((b + L.TARP) / L.FLOOR_SIZE, (L.CAGE + L.TARP - a) / L.FLOOR_SIZE)
    T = [uv(lo, lo), uv(lo, hi), uv(hi, hi), uv(hi, lo)]
    m.quad(P, T, normal=(0, 0, 1))
    return m


def build_carpet():
    m = Mesh(); H = L.HALL
    ca, cb = L.CAGE / 2, L.CAGE / 2
    a0, a1, b0, b1 = ca - H['len_a'] / 2, ca + H['len_a'] / 2, cb - H['len_b'] / 2, cb + H['len_b'] / 2
    P = [G(a0, b0, -0.004), G(a0, b1, -0.004), G(a1, b1, -0.004), G(a1, b0, -0.004)]
    T = [(0, 0), (H['len_b'] / 2, 0), (H['len_b'] / 2, H['len_a'] / 2), (0, H['len_a'] / 2)]
    m.quad(P, T, normal=(0, 0, 1))
    return m


def build_hall():
    walls, ceil = Mesh(), Mesh(); H = L.HALL
    ca, cb = L.CAGE / 2, L.CAGE / 2
    a0, a1, b0, b1, h = ca - H['len_a'] / 2, ca + H['len_a'] / 2, cb - H['len_b'] / 2, cb + H['len_b'] / 2, H['height']
    centre = G(ca, cb, h / 2)
    for (pa, pb, qa, qb) in ((a1, b0, a1, b1), (a1, b1, a0, b1), (a0, b1, a0, b0), (a0, b0, a1, b0)):
        ln = math.hypot(qa - pa, qb - pb)
        P = [G(pa, pb, 0), G(qa, qb, 0), G(qa, qb, h), G(pa, pb, h)]
        T = [(0, 1), (ln / 26.0, 1), (ln / 26.0, 0), (0, 0)]
        nrm = centre - (P[0] + P[2]) / 2; nrm[2] = 0
        walls.quad(P, T, normal=nrm)
    ceil.quad([G(a0, b0, h), G(a0, b1, h), G(a1, b1, h), G(a1, b0, h)], normal=(0, 0, -1))
    return walls, ceil


def build_net():
    m = Mesh(); c, h, off, w = L.CAGE, L.CAGE_H, L.NET_OFFSET, L.NET_CORD / 2
    I = np.eye(3)
    ticks = np.arange(0, c + 1e-6, L.NET_CELL)
    hticks = np.arange(L.NET_CELL / 2, h, L.NET_CELL)
    for side in range(4):
        for t in ticks:                                  # vertical cords
            a, b = [(-off, t), (c + off, t), (t, -off), (t, c + off)][side]
            m.box(G(a, b, h / 2), I, (w, w, h / 2), faces='long')
        for z in hticks:                                 # horizontal cords
            if side < 2:
                a = -off if side == 0 else c + off
                m.box(G(a, c / 2, z), I, (w, c / 2, w), faces='long')
            else:
                b = -off if side == 2 else c + off
                m.box(G(c / 2, b, z), I, (c / 2, w, w), faces='long')
    for t in ticks:                                      # ceiling
        m.box(G(t, c / 2, h + off), I, (w, c / 2, w), faces='long')
        m.box(G(c / 2, t, h + off), I, (c / 2, w, w), faces='long')
    return m


def build_cage_frame():
    m = Mesh(); c, h, r = L.CAGE, L.CAGE_H, L.PIPE_R
    ticks = np.linspace(0, c, L.POSTS_PER_SIDE + 1)
    posts = set()
    for t in ticks:
        posts |= {(0.0, t), (c, t), (t, 0.0), (t, c)}
    for (a, b) in posts:
        tube(m, [G(a, b, 0), G(a, b, h)], r)
    for z in L.RAIL_H:
        for (p, q) in (((0, 0), (c, 0)), ((c, 0), (c, c)), ((c, c), (0, c)), ((0, c), (0, 0))):
            tube(m, [G(p[0], p[1], z), G(q[0], q[1], z)], r)
    for t in ticks[1:-1]:                                # ceiling cross pipes
        tube(m, [G(t, 0, h), G(t, c, h)], r)
        tube(m, [G(0, t, h), G(c, t, h)], r)
    return m


def build_panels():
    """Maze boards: banner skin (panel.jpg) and plywood skin (board texture with the black strip),
    each on a wooden A-frame with floor runners."""
    banner, ply, wood = Mesh(), Mesh(), Mesh()
    for p in L.PANELS:
        tex = banner if p['skin'] == 'banner' else ply
        R = rot_yaw_gz(p['yaw'])
        c = G(p['a'], p['b'], (p['bottom'] + p['top']) / 2)
        hw, ht, hh = p['w'] / 2, p['t'] / 2, (p['top'] - p['bottom']) / 2
        along, right, up = R[:, 0], R[:, 1], R[:, 2]
        for s in (-1, 1):                                 # two printed faces
            fc = c + s * right * ht
            # seen from the outside of face s, image-left must map to the viewer's left
            l, r_ = (-along, along) if s > 0 else (along, -along)
            if p['mirror']:
                l, r_ = r_, l
            P = [fc + l * hw - up * hh, fc + r_ * hw - up * hh, fc + r_ * hw + up * hh, fc + l * hw + up * hh]
            tex.quad(P, ((0, 1), (1, 1), (1, 0), (0, 0)), normal=s * right)
        for s in (-1, 1):                                 # thin edges
            fc = c + s * along * hw
            P = [fc - right * ht - up * hh, fc + right * ht - up * hh, fc + right * ht + up * hh, fc - right * ht + up * hh]
            tex.quad(P, ((0.0, 1), (0.02, 1), (0.02, 0), (0.0, 0)), normal=s * along)
        top = c + up * hh
        tex.quad([top - along * hw - right * ht, top + along * hw - right * ht, top + along * hw + right * ht,
                  top - along * hw + right * ht], ((0, 0), (1, 0), (1, 0.02), (0, 0.02)), normal=up)
        # wooden A-frame: end posts, foot rails, floor rail
        base = G(p['a'], p['b'], 0)
        for s in (-1, 1):
            pc = base + along * s * (hw - 0.03) - right * (ht + 0.025)
            wood.box(pc + up * (p['top'] + 0.02) / 2, R, (0.02, 0.02, (p['top'] + 0.02) / 2))
            wood.box(pc + up * 0.0175, R, (0.02, L.FRAME_FOOT / 2, 0.0175))
            for k in (-1, 1):                              # short diagonal braces (as boxes)
                wood.box(pc + right * k * 0.18 + up * 0.22, R, (0.015, 0.015, 0.2))
        wood.box(base - right * (ht + 0.025) + up * 0.0175, R, (hw, 0.02, 0.0175))
        wood.box(base - right * (ht + 0.025) + up * (p['bottom'] - 0.03), R, (hw, 0.02, 0.02))
    return banner, ply, wood


def _tunnel_point(s_, th, rho, h_off=0.0):
    """Tunnel local -> gz: s_ along the axis from the entry hoop, th from the top clockwise looking
    along the axis (90 deg = right), rho radius from the axis."""
    T = L.TUNNEL
    ea, eb = T['entry_ab']; ua, ub = T['axis_ab']; ra, rb = L.tunnel_right()
    st, ct = np.sin(th), np.cos(th)
    return G(ea + s_ * ua + rho * st * ra, eb + s_ * ub + rho * st * rb, T['centre_h'] + rho * ct + h_off)


def build_tunnel():
    """Brass hoops (entry face, middle, exit face) on U-feet + black mesh liner + rock-print fabric,
    along the tunnel axis from layout.TUNNEL (map: enter north face, exit south face)."""
    T = L.TUNNEL
    brass, mesh, fabric = Mesh(), Mesh(), Mesh()
    hc, R0, tr = T['centre_h'], T['radius'], T['tube_r']
    ua, ub = T['axis_ab']; ra, rb = L.tunnel_right()
    axis_g = G(ua, ub, 0) - G(0, 0, 0)
    th = np.linspace(0, 2 * np.pi, 121)
    ph = np.linspace(0, 2 * np.pi, 9)
    for i_h in range(T['n_hoops']):
        s_h = T['length'] * i_h / (T['n_hoops'] - 1)
        for i in range(len(th) - 1):
            for j in range(len(ph) - 1):
                q = [(th[i], ph[j]), (th[i + 1], ph[j]), (th[i + 1], ph[j + 1]), (th[i], ph[j + 1])]
                P, N, UV = [], [], []
                for (t_, p_) in q:
                    radial = _tunnel_point(0, t_, 1.0) - _tunnel_point(0, t_, 0.0)      # unit radial dir (gz)
                    P.append(_tunnel_point(s_h + tr * math.sin(p_), t_, R0 + tr * math.cos(p_)))
                    N.append(radial * math.cos(p_) + axis_g * math.sin(p_))
                    UV.append((t_ / (2 * np.pi), p_ / (2 * np.pi)))
                P, N, UV = np.array(P), np.array(N), np.array(UV)
                brass.tri(P[[0, 1, 2]], UV[[0, 1, 2]], N[[0, 1, 2]])
                brass.tri(P[[0, 2, 3]], UV[[0, 2, 3]], N[[0, 2, 3]])
        c = _tunnel_point(s_h, 0, 0.0); bottom = hc - R0
        tube(brass, [np.array([c[0], c[1], 0.0]), np.array([c[0], c[1], bottom])], 0.006)
        foot = [c + (G(k1 * 0.17 * ra + k2 * 0.06 * ua, k1 * 0.17 * rb + k2 * 0.06 * ub, 0) - G(0, 0, 0))
                for k1, k2 in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))]
        tube(brass, [np.array([f[0], f[1], 0.006]) for f in foot], 0.006)
    # black mesh liner (semi-transparent), arc from mesh_deg[0] to mesh_deg[1]
    thm = np.radians(np.linspace(*T['mesh_deg'], 40))
    ss = np.linspace(-0.02, T['length'] + 0.02, 13)
    P = np.array([[_tunnel_point(s_, t_, T['mesh_r']) for s_ in ss] for t_ in thm])
    mesh.grid(P, np.stack(np.meshgrid(thm, ss, indexing='ij'), -1), double=True)
    # rock-print fabric over the top, wrinkled, extends past the end hoops and droops
    thf = np.radians(np.linspace(*T['fabric_deg'], 60))
    ss = np.linspace(-0.10, T['length'] + 0.10, 30)
    TH, SS = np.meshgrid(thf, ss, indexing='ij')
    wr = 0.012 * np.sin(7 * TH + 3 * SS) + 0.008 * np.sin(13 * SS + 2 * TH)
    over = np.clip(np.maximum(-SS, SS - T['length']), 0, None)
    rr = T['fabric_r'] + wr - 0.35 * over * (1 + 0.4 * np.cos(TH))
    P = np.array([[_tunnel_point(SS[i, j], TH[i, j], rr[i, j], -0.9 * over[i, j]) for j in range(TH.shape[1])]
                  for i in range(TH.shape[0])])
    arc = (TH - TH.min()) * T['fabric_r']
    fabric.grid(P, np.stack([(SS - SS.min()) / 0.75, arc / 0.75], -1), double=True)
    return brass, mesh, fabric


def build_window():
    """Window board of chicane wall C2: tan plywood with the rectangular hole and the black strip,
    on two legs with foot rails; oriented along layout.WINDOW['yaw'] (map: C2 direction)."""
    win, wood = Mesh(), Mesh()
    w = L.WINDOW
    y = math.radians(w['yaw'])
    along = G(math.cos(y), math.sin(y), 0) - G(0, 0, 0)
    nrm_ab = (-math.sin(y), math.cos(y))                        # board normal (cage a, b)
    nrm = G(*nrm_ab, 0) - G(0, 0, 0)
    c = G(w['a'], w['b'], 0)
    W, bot, top, th = w['width'], w['bottom'], w['top'], w['thick']
    hw, hh, hcz = w['hole_w'] / 2, w['hole_h'] / 2, w['hole_c']
    P = lambda s_, h, side: c + along * s_ + nrm * side * th / 2 + np.array([0, 0, h])
    uv = lambda s_, h, side: ((s_ + W / 2) / W if side > 0 else 1 - (s_ + W / 2) / W, (top - h) / (top - bot))
    rects = [(-W / 2, -hw, bot, top), (hw, W / 2, bot, top), (-hw, hw, bot, hcz - hh), (-hw, hw, hcz + hh, top)]
    for (s1, s2, h1, h2) in rects:
        for side in (-1, 1):
            Q = [P(s1, h1, side), P(s2, h1, side), P(s2, h2, side), P(s1, h2, side)]
            win.quad(Q, [uv(s1, h1, side), uv(s2, h1, side), uv(s2, h2, side), uv(s1, h2, side)], normal=nrm * side)
    edges = [(-W / 2, bot, -W / 2, top), (W / 2, bot, W / 2, top), (-W / 2, bot, W / 2, bot), (-W / 2, top, W / 2, top),
             (-hw, hcz - hh, -hw, hcz + hh), (hw, hcz - hh, hw, hcz + hh), (-hw, hcz - hh, hw, hcz - hh), (-hw, hcz + hh, hw, hcz + hh)]
    for (sa, ha, sb, hb) in edges:
        win.quad([P(sa, ha, -1), P(sb, hb, -1), P(sb, hb, 1), P(sa, ha, 1)], [(0, 0.5), (0.01, 0.5), (0.01, 0.51), (0, 0.51)], double=True)
    Rw = rot_yaw_gz(w['yaw'])
    for s_ in (-W / 2 - 0.02, W / 2 + 0.02):
        pc = c + along * s_
        wood.box(pc + np.array([0, 0, (top + 0.02) / 2]), Rw, (0.02, 0.02, (top + 0.02) / 2))
        wood.box(pc + np.array([0, 0, 0.0175]), Rw, (0.02, L.FRAME_FOOT / 2, 0.0175))
    return win, wood


def N2G(n, e, h):
    return np.array([n, -e, h], float)


def build_landing_pad():
    """Flat blue landing pad: textured top + plain thin sides (texture from make_textures)."""
    m = Mesh(); p = L.PAD
    top = [N2G(c[0], c[1], p['thick']) for c in L.pad_corners_ned()]        # (+u+v, +u-v, -u-v, -u+v)
    # image: u (column) along -v_pad .. ; keep the H upright relative to the pad's own axes
    T = [(1, 0), (0, 0), (0, 1), (1, 1)]
    m.quad(top, T, normal=(0, 0, 1))
    bot = [q - np.array([0, 0, p['thick']]) for q in top]
    for i in range(4):
        j = (i + 1) % 4
        nrm = (top[i] + top[j]) / 2 - N2G(p['n'], p['e'], p['thick']); nrm[2] = 0
        m.quad([bot[i], bot[j], top[j], top[i]], ((0.02, 0.5), (0.03, 0.5), (0.03, 0.51), (0.02, 0.51)), normal=nrm)
    return m


def build_blocks():
    m = Mesh(); bl, bw, bh = L.BLOCK
    for (n, e, yaw) in L.CINDER_BLOCKS:
        m.box(N2G(n, e, bh / 2), rot_yaw_gz(yaw), (bl / 2, bw / 2, bh / 2), uv_scale=0.3)
    return m


def build_astronaut():
    """Small white astronaut figurine on top of rock 0 (photos 10, 12): body + head boxes."""
    m = Mesh()
    if not getattr(L, 'ASTRONAUT_H', 0):
        return m
    a, b, h, r = L.ROCKS[0]
    I = np.eye(3)
    m.box(G(a, b, h + 0.035), I, (0.018, 0.013, 0.035))
    m.box(G(a, b, h + 0.08), I, (0.015, 0.015, 0.013))
    return m


def build_tape():
    """Black floor tape (start box + flight path), 1.5 mm above the floor."""
    m = Mesh(); w = L.TAPE_W / 2; z = 0.0015
    for name, pts in L.TAPE:
        P = [N2G(f, -l, z) for f, l in pts]
        for p0, p1 in zip(P[:-1], P[1:]):
            d = p1 - p0; d /= np.linalg.norm(d)
            n = np.array([-d[1], d[0], 0.0])
            q0, q1 = p0 - d * w, p1 + d * w                 # overlap the joints
            m.quad([q0 - n * w, q1 - n * w, q1 + n * w, q0 + n * w], normal=(0, 0, 1))
    return m


def build_rocks():
    """Map boulders (0.46 m): rounded, faceted grey stones (dome profile)."""
    m = Mesh()
    for k, (a, b, h, r0) in enumerate(L.ROCKS):
        rng = np.random.default_rng(100 + k)
        nseg = 11
        th = np.linspace(0, 2 * np.pi, nseg + 1)
        ts = np.array([0.0, 0.25, 0.5, 0.7, 0.85, 0.95, 1.0])
        jitter = 1 + 0.12 * rng.standard_normal((len(ts), nseg))
        jitter = np.concatenate([jitter, jitter[:, :1]], 1)
        R = r0 * np.sqrt(np.clip(1 - ts ** 2, 0, None))[:, None] * jitter
        R[0] *= 0.92; R[-1] = 0.15 * r0
        AA = a + R * np.cos(th)[None]
        BB = b + R * np.sin(th)[None]
        HH = np.repeat((h * ts)[:, None], nseg + 1, 1) * (1 + 0.05 * rng.standard_normal((len(ts), 1)))
        HH[0] = 0.0
        P = G(AA, BB, HH)
        UV = np.stack([np.repeat((th / np.pi)[None], len(ts), 0), np.repeat((ts * h / 0.3)[:, None], nseg + 1, 1)], -1)
        m.grid(P, UV, out=G(a, b, h * 0.3), flat=True)
        top = G(AA[-1].mean(), BB[-1].mean(), HH[-1].mean())
        for j in range(nseg):
            m.tri([P[-1, j], P[-1, j + 1], top], [(0, 0), (0.1, 0), (0.05, 0.1)], out=G(a, b, 0))
    return m


def build_ring():
    """Printed ring around the landing pad (map: ring target, outer radius ~0.78 m), on the floor."""
    m = Mesh(); p = L.PAD
    r1, r0 = p['ring_r'], p['ring_r'] - p['ring_w']
    t = np.linspace(0, 2 * np.pi, 97)
    for i in range(len(t) - 1):
        q = [N2G(p['n'] + r * math.cos(tt), p['e'] + r * math.sin(tt), 0.0012) for r, tt in
             ((r0, t[i]), (r1, t[i]), (r1, t[i + 1]), (r0, t[i + 1]))]
        m.quad(q, normal=(0, 0, 1))
    return m


def build_props():
    """Hall tables + plywood box outside the cage (background structure)."""
    top, legs, ply = Mesh(), Mesh(), Mesh()
    for (a, b, yaw) in L.TABLES:
        R = rot_yaw_gz(yaw)
        c = G(a, b, 0)
        top.box(c + np.array([0, 0, 0.74]), R, (0.9, 0.375, 0.02))
        for sa in (-0.85, 0.85):
            for sb in (-0.32, 0.32):
                legs.box(c + R[:, 0] * sa + R[:, 1] * sb + np.array([0, 0, 0.36]), R, (0.015, 0.015, 0.36))
    p = L.PLYBOX
    ply.box(G(p['a'], p['b'], p['height'] / 2), np.eye(3), (p['len_a'] / 2, p['len_b'] / 2, p['height'] / 2), uv_scale=1.0)
    return top, legs, ply


# ----------------------------------------------------------------------------- SDF
def material(color=(1, 1, 1), albedo=None, rough=0.8, metal=0.0, emissive_map=None, emissive=None, double=False):
    c = ' '.join('%.3f' % x for x in color)
    s = f'<material><ambient>{c} 1</ambient><diffuse>{c} 1</diffuse><specular>0.2 0.2 0.2 1</specular>'
    if emissive is not None:
        s += '<emissive>%s 1</emissive>' % ' '.join('%.3f' % x for x in emissive)
    if double:
        s += '<double_sided>true</double_sided>'
    s += f'<pbr><metal><roughness>{rough}</roughness><metalness>{metal}</metalness>'
    if albedo:
        s += f'<albedo_map>{albedo}</albedo_map>'
    if emissive_map:
        s += f'<emissive_map>{emissive_map}</emissive_map>'
    s += '</metal></pbr></material>'
    return s


def visual(name, mesh_uri, mat, transparency=None, shadows=True):
    tr = f'<transparency>{transparency}</transparency>' if transparency is not None else ''
    cs = '' if shadows else '<cast_shadows>false</cast_shadows>'
    return (f'<visual name="{name}">{tr}{cs}<geometry><mesh><uri>{mesh_uri}</uri></mesh></geometry>{mat}</visual>\n')


def camera_sdf(with_rgb):
    fx, fy, cx, cy = OFFICIAL_K
    W, H = OFFICIAL_SIZE
    def cam(name, typ, topic, w, h, fx_, fy_, cx_, cy_, fmt, near, far, pose='0 0 0 0 0 0'):
        hfov = 2 * math.atan(w / (2 * fx_))
        return (f'<sensor name="{name}" type="{typ}"><pose>{pose}</pose><always_on>1</always_on><update_rate>20</update_rate>'
                f'<visualize>false</visualize><topic>{topic}</topic><camera><horizontal_fov>{hfov:.9f}</horizontal_fov>'
                f'<image><width>{w}</width><height>{h}</height><format>{fmt}</format></image>'
                f'<clip><near>{near}</near><far>{far}</far></clip>'
                f'<lens><intrinsics><fx>{fx_:.9f}</fx><fy>{fy_:.9f}</fy><cx>{cx_:.9f}</cx><cy>{cy_:.9f}</cy><s>0</s></intrinsics></lens>'
                f'</camera></sensor>\n')
    # gz puts pixel centres at i+0.5 -> +0.5 px on cx/cy gives the OpenCV convention
    s = ''
    if with_rgb:
        s += cam('rgb', 'camera', 'replica/rgb', W * SS, H * SS, fx * SS, fy * SS, cx * SS + 0.5 * SS, cy * SS + 0.5 * SS,
                 'R8G8B8', 0.03, 60)
        # right eye: same intrinsics, BASELINE to the right (camera link y = left -> -BASELINE)
        s += cam('rgb_right', 'camera', 'replica/rgb_right', W * SS, H * SS, fx * SS, fy * SS, cx * SS + 0.5 * SS,
                 cy * SS + 0.5 * SS, 'R8G8B8', 0.03, 60, pose=f'0 {-BASELINE} 0 0 0 0')
    s += cam('depth', 'depth_camera', 'replica/depth', W, H, fx, fy, cx + 0.5, cy + 0.5, 'R_FLOAT32', 0.05, 40)
    return f'<model name="cam"><static>true</static><pose>0 0 1 0 0 0</pose><link name="link">\n{s}</link></model>\n'


def lights():
    s = ''
    ca, cb = L.CAGE / 2, L.CAGE / 2
    k = 0
    for da in (-1.6, 1.6):
        for db in (-1.6, 1.6):
            p = G(ca + da, cb + db, 4.6)
            s += (f'<light type="spot" name="chandelier_{k}"><pose>{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} 0 0 0</pose>'
                  '<diffuse>1.0 0.80 0.58 1</diffuse><specular>0.9 0.75 0.55 1</specular>'
                  '<attenuation><range>14</range><constant>0.25</constant><linear>0.04</linear><quadratic>0.012</quadratic></attenuation>'
                  '<direction>0 0 -1</direction><spot><inner_angle>0.7</inner_angle><outer_angle>1.35</outer_angle><falloff>0.9</falloff></spot>'
                  '<cast_shadows>true</cast_shadows><intensity>1.6</intensity></light>\n')
            k += 1
    H = L.HALL
    for (a, b) in ((ca, cb - H['len_b'] / 2 + 1.0), (ca, cb + H['len_b'] / 2 - 1.0), (ca - H['len_a'] / 2 + 1.0, cb),
                   (ca + H['len_a'] / 2 - 1.0, cb)):
        p = G(a, b, 2.6)
        s += (f'<light type="point" name="cove_{k}"><pose>{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} 0 0 0</pose>'
              '<diffuse>1.0 0.78 0.5 1</diffuse><specular>0.3 0.25 0.2 1</specular>'
              '<attenuation><range>12</range><constant>0.4</constant><linear>0.08</linear><quadratic>0.02</quadratic></attenuation>'
              '<cast_shadows>false</cast_shadows><intensity>1.0</intensity></light>\n')
        k += 1
    s += ('<light type="directional" name="fill"><pose>0 0 10 0 0 0</pose><diffuse>0.28 0.25 0.22 1</diffuse>'
          '<specular>0.05 0.05 0.05 1</specular><direction>0.25 0.15 -1</direction><cast_shadows>false</cast_shadows></light>\n')
    return s


def world_sdf(full, prefix):
    M = lambda n: f'{prefix}meshes/{n}.obj'
    X = lambda n: f'{prefix}textures/{n}'
    v = ''
    v += visual('floor', M('floor'), material(albedo=X('floor.jpg'), rough=0.32))
    v += visual('carpet', M('carpet'), material(albedo=X('carpet.jpg'), rough=0.95))
    v += visual('hall_walls', M('hall_walls'), material(albedo=X('hall.jpg'), emissive_map=X('hall_emissive.jpg'), rough=0.9), shadows=False)
    v += visual('hall_ceiling', M('hall_ceiling'), material(color=(0.25, 0.17, 0.10), rough=0.9), shadows=False)
    v += visual('cage_frame', M('cage_frame'), material(color=(0.035, 0.035, 0.04), rough=0.45))
    if full:
        v += visual('net', M('net'), material(color=(0.02, 0.02, 0.022), rough=0.9), shadows=False)
        v += visual('tunnel_mesh', M('tunnel_mesh'), material(color=(0.01, 0.01, 0.012), rough=0.9, double=True), transparency=0.45)
    v += visual('panels', M('panels'), material(albedo=X('panel.jpg'), rough=0.55))
    v += visual('maze_boards', M('maze_boards'), material(albedo=X('board_wall.jpg'), rough=0.85))
    v += visual('panel_frames', M('panel_frames'), material(color=(0.86, 0.70, 0.46), rough=0.8))
    # brass reads bright gold in the hall (photo 2); with no environment map a metallic PBR
    # material renders almost black, so: low metalness + a warm emissive term standing in for
    # the reflected hall light.
    v += visual('tunnel_hoops', M('tunnel_hoops'), material(color=(0.95, 0.78, 0.45), rough=0.35, metal=0.3, emissive=(0.42, 0.33, 0.15)))
    v += visual('tunnel_fabric', M('tunnel_fabric'), material(albedo=X('fabric.jpg'), rough=0.35, double=True))
    v += visual('window_board', M('window_board'), material(albedo=X('board_window.jpg'), rough=0.85))
    v += visual('board_legs', M('board_legs'), material(color=(0.86, 0.70, 0.46), rough=0.8))
    v += visual('landing_pad', M('landing_pad'), material(albedo=X('landing_pad.jpg'), rough=0.6))
    v += visual('cinder_blocks', M('cinder_blocks'), material(albedo=X('stone.jpg'), rough=0.95))
    v += visual('floor_tape', M('floor_tape'), material(color=(0.02, 0.02, 0.025), rough=0.5), shadows=False)
    v += visual('boulders', M('rocks'), material(albedo=X('stone.jpg'), rough=0.95))
    v += visual('landing_ring', M('landing_ring'), material(color=(0.92, 0.92, 0.92), rough=0.6), shadows=False)
    v += visual('table_tops', M('table_tops'), material(color=(0.32, 0.18, 0.10), rough=0.5))
    v += visual('table_legs', M('table_legs'), material(color=(0.05, 0.05, 0.05), rough=0.5))
    v += visual('plywood_box', M('plywood_box'), material(albedo=X('board_wall.jpg'), rough=0.85))
    name = 'replica'
    return f'''<?xml version="1.0"?>
<!-- DroneHacks 2026 venue replica ({'full: RGB + depth' if full else 'depth pass: no cage net / tunnel mesh'}).
     Generated by cartographer/official/replica/build_world.py; all dimensions are photo estimates. -->
<sdf version="1.9">
<world name="{name}">
  <physics name="fixed" type="ignored"><max_step_size>0.01</max_step_size><real_time_factor>1.0</real_time_factor></physics>
  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
  <scene><ambient>0.30 0.26 0.22 1</ambient><background>0.12 0.09 0.06 1</background><grid>false</grid><shadows>true</shadows></scene>
{lights()}
  <model name="venue"><static>true</static><link name="link">
{v}  </link></model>
{camera_sdf(full)}
</world>
</sdf>
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'world'))
    ap.add_argument('--textures', default=os.path.join(HERE, 'assets', 'textures'))
    ap.add_argument('--prefix', default='', help="URI prefix for meshes/textures (default: relative to the SDF)")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'meshes'), exist_ok=True)
    tdir = os.path.join(a.out, 'textures')
    os.makedirs(tdir, exist_ok=True)
    for f in os.listdir(a.textures):
        shutil.copy2(os.path.join(a.textures, f), os.path.join(tdir, f))
    walls, ceil = build_hall()
    panels, maze_boards, panel_frames = build_panels()
    hoops, tmesh, fabric = build_tunnel()
    win, legs = build_window()
    ttop, tlegs, ply = build_props()
    meshes = dict(floor=build_floor(), carpet=build_carpet(), hall_walls=walls, hall_ceiling=ceil, net=build_net(),
                  cage_frame=build_cage_frame(), panels=panels, maze_boards=maze_boards, panel_frames=panel_frames, tunnel_hoops=hoops,
                  tunnel_mesh=tmesh, tunnel_fabric=fabric, window_board=win, board_legs=legs,
                  landing_pad=build_landing_pad(), landing_ring=build_ring(), cinder_blocks=build_blocks(),
                  floor_tape=build_tape(), rocks=build_rocks(), table_tops=ttop, table_legs=tlegs, plywood_box=ply)
    for k, m in meshes.items():
        n = m.write(os.path.join(a.out, 'meshes', k + '.obj'))
        print(f'  {k:14s} {n:6d} tris')
    for full, fn in ((True, 'venue_full.sdf'), (False, 'venue_depth.sdf')):
        with open(os.path.join(a.out, fn), 'w') as f:
            f.write(world_sdf(full, a.prefix))
    print('world ->', a.out)


if __name__ == '__main__':
    main()
