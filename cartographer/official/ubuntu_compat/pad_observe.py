#!/usr/bin/env python3
"""Where can the horizontal OAK camera see the WHOLE flat landing pad? (sim side, replica truth)

  python3 pad_observe.py [--heights 0.35,0.45,0.55,0.8] [--top 8]     (in the container: /replica + profile.json)

Candidates: base_link poses around the pad centre every 15 deg at distances d, d+0.5, d+1.0 (d = the distance at
which the pad's near edge enters the vertical FOV), camera facing the pad centre (camera 0.116 m ahead of base_link).
A candidate must keep >= 0.40 m from every physical solid (course_solids.py) and from the net.

Visibility = 7 x 7 sample points on the pad (true corners from layout.pad_corners_ned(), yaw included), each
  * inside the image (K from profile.json; camera horizontal, so the vertical half-FOV below the horizon ~25 deg), and
  * not occluded: ray camera -> point tested (Moller-Trumbore) against the replica's own VISUAL meshes
    (/replica/world/meshes: maze_boards, panels, window_board, wall_board, board_legs (window/wall posts + foot
    rails), panel_frames (maze A-frames, struts, rails, under-board brace), cinder_blocks, rocks, astronaut,
    plywood_box). Vertices are in the start frame (x = f, y = l, z = up). The tunnel and net are not tested.
  * 'skirt risk': the ray passes UNDER a maze board (inside its footprint) lower than 0.2 m. Photos 13/14 show some
    maze boards' black strip hanging close to the floor (cloth skirt?), so such rays may be blocked in reality.
Prints, per camera height, the best candidates (visible fraction, skirt-risk fraction, clearance).
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/replica'); sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout as L  # noqa: E402
import course_solids as C  # noqa: E402

MESHES = ('maze_boards', 'panels', 'window_board', 'wall_board', 'board_legs', 'panel_frames', 'cinder_blocks',
          'rocks', 'astronaut', 'plywood_box')
CAM_FWD = 0.116


def load_obj(path):
    V, T = [], []
    for ln in open(path):
        if ln.startswith('v '):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith('f '):
            idx = [int(t.split('/')[0]) - 1 for t in ln.split()[1:]]
            T += [(idx[0], idx[i], idx[i + 1]) for i in range(1, len(idx) - 1)]
    V = np.array(V); T = np.array(T, int)
    return V[T]                                                     # (n, 3, 3)


def blocked(o, d, tri, eps=1e-9):
    """o (3,), d (m, 3) = target - o; True per ray if any triangle is hit at 0 < t < 0.995."""
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    out = np.zeros(len(d), bool)
    for i in range(len(d)):
        p = np.cross(d[i], e2); det = (e1 * p).sum(1); ok = np.abs(det) > eps
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0); s = o - v0
        u = (s * p).sum(1) * inv; q = np.cross(s, e1); v = (q * d[i]).sum(1) * inv; t = (e2 * q).sum(1) * inv
        out[i] = bool((ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-4) & (t < 0.995)).any())
    return out


def skirt_risk(o, d):
    """True per ray if it passes under a maze board footprint lower than 0.2 m."""
    out = np.zeros(len(d), bool)
    ts = np.linspace(0.02, 0.98, 80)
    for i in range(len(d)):
        P = o + ts[:, None] * d[i]; low = P[:, 2] < 0.2
        if not low.any():
            continue
        a, b = P[low, 0] + L.START[0], -P[low, 1] + L.START[1]
        for p in L.PANELS:
            ya = math.radians(p['yaw']); da, db = a - p['a'], b - p['b']
            if ((np.abs(da * math.cos(ya) + db * math.sin(ya)) <= p['w'] / 2) &
                    (np.abs(-da * math.sin(ya) + db * math.cos(ya)) <= p['t'] / 2 + 0.03)).any():
                out[i] = True; break
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--heights', default='0.35,0.45,0.55,0.8'); ap.add_argument('--top', type=int, default=8)
    A = ap.parse_args()
    P = json.load(open(Path(__file__).resolve().parent / 'profile.json'))['camera']
    fx, fy, cx, cy = P['K']; W, H = P['width'], P['height']
    half_down = math.degrees(math.atan((H - cy) / fy))
    tri = np.concatenate([load_obj(f'/replica/world/meshes/{m}.obj') for m in MESHES if Path(f'/replica/world/meshes/{m}.obj').exists()])
    cn = np.array([[c[0], -c[1]] for c in L.pad_corners_ned()])        # (f, l) of the 4 true corners
    u, v = np.meshgrid(np.linspace(0.03, 0.97, 7), np.linspace(0.03, 0.97, 7))
    pts = ((1 - u) * (1 - v))[..., None] * cn[0] + (u * (1 - v))[..., None] * cn[1] + (u * v)[..., None] * cn[2] + ((1 - u) * v)[..., None] * cn[3]
    pts = np.c_[pts.reshape(-1, 2), np.full(49, L.PAD['thick'])]
    pc = cn.mean(0); far = float(np.max(np.linalg.norm(cn - pc, axis=1)))
    print(f'{len(tri)} occluder triangles; pad centre f {pc[0]:.2f} l {pc[1]:.2f}; vertical half-FOV below horizon {half_down:.1f} deg')
    for h in [float(x) for x in A.heights.split(',')]:
        dmin = h / math.tan(math.radians(half_down)) + far
        rows = []
        for ang in range(0, 360, 15):
            for d in (dmin, dmin + 0.5, dmin + 1.0):
                a = math.radians(ang); cam = np.array([pc[0] + d * math.cos(a), pc[1] + d * math.sin(a), h])
                yaw = a + math.pi; bf, bl = cam[0] - CAM_FWD * math.cos(yaw), cam[1] - CAM_FWD * math.sin(yaw)
                clear = min(bf + L.START[0], L.CAGE - L.START[0] - bf, bl + L.CAGE - L.START[1], L.START[1] - bl, C.clearance(L, bf + L.START[0], -bl + L.START[1], h)[0])
                if clear < 0.40:
                    continue
                D = pts - cam
                fwd = D[:, 0] * math.cos(yaw) + D[:, 1] * math.sin(yaw); lat = -D[:, 0] * math.sin(yaw) + D[:, 1] * math.cos(yaw)
                px = cx - fx * lat / np.maximum(fwd, 1e-6); py = cy - fy * D[:, 2] / np.maximum(fwd, 1e-6)
                infov = (fwd > 0.05) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
                occ = blocked(cam, D, tri); sk = skirt_risk(cam, D)
                vis = infov & ~occ
                rows.append((vis.mean(), -(vis & sk).mean(), clear, ang, d, bf, bl, math.degrees(yaw), infov.mean(), occ.mean(), (vis & sk).mean()))
        rows.sort(reverse=True)
        good = [r for r in rows if r[0] >= 0.95 and r[10] == 0]
        print(f'\ncamera h {h:.2f} m: near edge in view from d >= {dmin:.2f} m; {len(rows)} free poses, '
              f'{len(good)} see >= 95 % of the pad with no skirt-risk ray')
        for r in rows[:A.top]:
            print(f'   from {r[3]:3d} deg d {r[4]:.2f}: base f {r[5]:5.2f} l {r[6]:5.2f} yaw {((r[7] + 180) % 360) - 180:6.0f} | clear {r[2]:.2f} | '
                  f'visible {100 * r[0]:3.0f} % (in FOV {100 * r[8]:3.0f} %, occluded {100 * r[9]:3.0f} %) | skirt-risk {100 * r[10]:3.0f} %')


if __name__ == '__main__':
    main()
