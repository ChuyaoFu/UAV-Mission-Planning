#!/usr/bin/env python3
"""Camera poses for the replica render: a route through every stage at three
heights (0.4/0.8/1.2 m, one frame per ~0.25 m) plus extra views of the tunnel
mouth, the window, the landing stone and the wall climb.

Usage: python poses.py [--out world/poses.json]
Each pose: id, group, cam_ned (3), R_nc (3x3, optical->NED), body_ned, heading,
roll/pitch (deg), and the Gazebo camera-link pose (xyz + quaternion wxyz).
Camera = OAK-D left, horizontal, rigidly on the body (common/geometry.py extrinsic).
"""
import argparse, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import layout as L                                                     # noqa: E402
from common.geometry import F_FLU_FRD, OFFICIAL_R_BASE_CAM, OFFICIAL_T_BASE_CAM   # noqa: E402

F_WORLD = np.diag([1., -1., -1.])          # NED <-> gz world
R_LINK_OPT = OFFICIAL_R_BASE_CAM           # optical -> camera link (x fwd, y left, z up), same as FLU


def rot_ned(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx                     # FRD -> NED


def mat_to_quat_wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1) * 2; w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z]); return q / np.linalg.norm(q)


def make_pose(pid, group, cam_ned, yaw, pitch=0.0, roll=0.0):
    R_nb = rot_ned(yaw, pitch, roll)
    R_nc = R_nb @ F_FLU_FRD @ OFFICIAL_R_BASE_CAM
    cam = np.asarray(cam_ned, float)
    body = cam - R_nb @ (F_FLU_FRD @ np.asarray(OFFICIAL_T_BASE_CAM, float))
    R_gz_link = F_WORLD @ R_nc @ R_LINK_OPT.T
    return dict(id=pid, group=group, cam_ned=cam.round(5).tolist(), R_nc=R_nc.round(9).tolist(),
                body_ned=body.round(5).tolist(), heading=round(float(math.atan2(math.sin(yaw), math.cos(yaw))), 6),
                roll_deg=round(math.degrees(roll), 3), pitch_deg=round(math.degrees(pitch), 3),
                gz_xyz=(F_WORLD @ cam).round(5).tolist(), gz_quat_wxyz=mat_to_quat_wxyz(R_gz_link).round(9).tolist())


def smooth_route(pts, radius=0.45, step=0.02):
    """Polyline (cage a,b) with circular-ish corner blends -> dense points + tangents."""
    pts = [np.array(p, float) for p in pts]
    dense = [pts[0]]
    for i in range(1, len(pts) - 1):
        p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
        d0 = (p1 - p0) / np.linalg.norm(p1 - p0); d1 = (p2 - p1) / np.linalg.norm(p2 - p1)
        r = min(radius, 0.45 * np.linalg.norm(p1 - p0), 0.45 * np.linalg.norm(p2 - p1))
        a, b = p1 - d0 * r, p1 + d1 * r
        dense += list(np.linspace(dense[-1], a, max(2, int(np.linalg.norm(a - dense[-1]) / step)))[1:])
        for t in np.linspace(0, 1, 15)[1:]:            # quadratic Bezier corner
            dense.append((1 - t) ** 2 * a + 2 * (1 - t) * t * p1 + t ** 2 * b)
    dense += list(np.linspace(dense[-1], pts[-1], max(2, int(np.linalg.norm(pts[-1] - dense[-1]) / step)))[1:])
    D = np.array(dense)
    seg = np.linalg.norm(np.diff(D, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    return D, s


# route through the stages (cage coords a=north, b=east)
# route = the organizers' map path through the maze + our continuation (layout.ROUTE_FT, feet)
ROUTE = [L.ft_ab(x, y) for x, y in L.ROUTE_FT]       # cage coords (a, b)
HEIGHTS = (0.4, 0.8, 1.2)


def route_poses(step=0.25, seed=3):
    rng = np.random.default_rng(seed)
    D, s = smooth_route(ROUTE)
    out = []
    for h in HEIGHTS:
        for sk in np.arange(0, s[-1] + 1e-6, step):
            i = int(np.clip(np.searchsorted(s, sk), 1, len(D) - 1))
            t = (sk - s[i - 1]) / max(s[i] - s[i - 1], 1e-9)
            p = D[i - 1] + t * (D[i] - D[i - 1])
            d = D[i] - D[i - 1]
            yaw = math.atan2(d[1], d[0])                       # NED heading: atan2(east, north)
            cam = L.cage_to_ned(p[0], p[1], h)
            if not L.pose_is_free(cam, margin=0.12):
                continue
            pitch, roll = np.clip(rng.normal(0, math.radians(1.2), 2), -math.radians(3), math.radians(3))
            out.append((f'route_h{int(h * 100):03d}_s{int(round(sk * 100)):04d}', 'route', cam, yaw, float(pitch), float(roll)))
    return out


def look_yaw(cam, target):
    return math.atan2(target[1] - cam[1], target[0] - cam[0])


def extra_poses():
    out = []
    T = L.TUNNEL                                           # tunnel: views of the entry face (north)
    (ea, eb), (ua, ub), (ra, rb) = T['entry_ab'], T['axis_ab'], L.tunnel_right()
    entry = L.cage_to_ned(ea, eb, T['centre_h'])
    for d in (0.6, 1.0, 1.5):
        for lat in (-0.35, 0.0, 0.35):
            for h in (0.45, 0.8):
                cam = L.cage_to_ned(ea - d * ua + lat * ra, eb - d * ub + lat * rb, h)
                out.append((f'tunnel_d{int(d * 100):03d}_l{int(lat * 100):+04d}_h{int(h * 100):03d}', 'tunnel', cam, look_yaw(cam, entry)))
    wc, w_along, w_normal, _ = L.window_frame()             # window: views from the approach side (ESE)
    for d in (0.8, 1.4, 2.0):
        for lat in (-0.45, 0.0, 0.45):
            for h in (1.0, 1.3):
                cam = np.array([wc[0] + d * w_normal[0] + lat * w_along[0], wc[1] + d * w_normal[1] + lat * w_along[1], -h])
                out.append((f'window_d{int(d * 100):03d}_l{int(lat * 100):+04d}_h{int(h * 100):03d}', 'window', cam, look_yaw(cam, wc)))
    Pd = L.PAD                                             # landing pad: views from where its top is in view
    pc = np.array([Pd['n'], Pd['e'], 0.0])
    for d in (1.2, 1.6):                                   # pad is 1.0 m from the west net (map): stay close + low
        for k in range(8):
            az = math.radians(45 * k)
            for h in (0.45, 0.7):
                cam = np.array([Pd['n'] + d * math.cos(az), Pd['e'] + d * math.sin(az), -h])
                out.append((f'pad_d{int(d * 100):03d}_az{45 * k:03d}_h{int(h * 100):03d}', 'pad', cam, look_yaw(cam, pc)))
    # framed pad views: the camera is level and sees only ~25 deg below the horizon, so the flat pad is
    # fully in frame only from range >= h / tan(23 deg) + half-diagonal; keep views with a clear line of
    # sight to the pad centre and all four corners
    corners = [np.asarray(c, float) for c in L.pad_corners_ned()]

    def los(cam, tgt):
        for t in np.linspace(0.05, 0.95, 40):
            p = cam + (tgt - cam) * t
            if -p[2] > 0.06 and not L.pose_is_free(p, margin=0.02):
                return False
        return True
    for h, extra in ((0.3, 0.0), (0.3, 0.4), (0.45, 0.0), (0.45, 0.4), (0.7, 0.0), (0.7, 0.4)):
        d = math.ceil((h / math.tan(math.radians(23.0)) + 0.45) * 20) / 20 + extra
        for k in range(12):
            az = math.radians(30 * k)
            cam = np.array([Pd['n'] + d * math.cos(az), Pd['e'] + d * math.sin(az), -h])
            if all(los(cam, c) for c in corners + [pc]):
                out.append((f'padf_d{int(round(d * 100)):03d}_az{30 * k:03d}_h{int(h * 100):03d}', 'pad', cam, look_yaw(cam, pc)))
    Wl = L.WALL                                            # barrier: climb on the landing side, cross north
    xb = 2.5                                               # crossing at map x 2.5 ft (between pad and launch)
    for h in np.arange(0.4, 2.21, 0.3):
        cam = L.ft_ned(xb, Wl['y'] - 0.9 / L.FT, h)
        out.append((f'wall_climb_h{int(round(h * 100)):03d}', 'wall', cam, 0.0))
    for k, dy in enumerate(np.arange(-0.9, 0.91, 0.25)):
        cam = L.ft_ned(xb, Wl['y'] + dy / L.FT, 2.1)
        out.append((f'wall_over_k{k:02d}', 'wall', cam, 0.0))
    # return to the launch over the barrier, then the descent onto it
    p0, p1 = L.ft_ned(xb, Wl['y'] + 1.0 / L.FT, 2.1), L.cage_to_ned(*L.START, 2.1)
    yaw_r = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    nseg = max(1, int(np.linalg.norm(p1[:2] - p0[:2]) / 0.25))
    for k in range(nseg + 1):
        out.append((f'return_k{k:02d}', 'return', p0 + (p1 - p0) * k / nseg, yaw_r))
    for h in (1.5, 1.0, 0.6):
        out.append((f'return_descent_h{int(h * 100):03d}', 'return', L.cage_to_ned(*L.START, h), 0.0))
    for k, (x, y) in enumerate(L.GATE_B_ROUTE_FT[1:4]):    # Gate B (checkpoint 4 alternative)
        for h in (0.8, 1.2):
            cam = L.ft_ned(x + 1.2, y, h)
            out.append((f'gateb_k{k}_h{int(h * 100):03d}', 'gateb', cam, math.radians(180 + 90)))
    for yaw_deg in (0, 30, -30):                            # on the start pad before take-off
        cam = L.cage_to_ned(*L.START, 0.12)
        out.append((f'start_pad_yaw{yaw_deg:+03d}', 'start', cam, math.radians(yaw_deg)))
    return [(pid, g, cam, yaw, 0.0, 0.0) for (pid, g, cam, yaw) in out]


# outside-the-cage viewpoints that roughly mimic the venue photos (for the photo
# comparison only; the drone never flies there): id, camera (a, b, h), look-at (a, b, h)
COMPARE = [('compare_photo1_overview', L.ft_ab(-3.5, -2.5) + (2.0,), L.ft_ab(9.0, 10.0) + (0.4,)),
           ('compare_photo10_start_maze', L.ft_ab(-2.5, 10.0) + (1.4,), L.ft_ab(4.0, 15.5) + (0.3,)),
           ('compare_photo12_corridor', L.ft_ab(-2.2, 12.5) + (1.4,), L.ft_ab(4.5, 17.0) + (0.5,)),
           ('compare_photo13_maze_lane', L.ft_ab(9.2, 21.8) + (1.3,), L.ft_ab(9.2, 13.0) + (0.4,)),
           ('compare_photo16_tunnel', L.ft_ab(17.6, 21.8) + (1.0,), L.ft_ab(17.6, 14.0) + (0.6,)),
           ('compare_photo17_window_pad', L.ft_ab(1.0, -2.5) + (1.6,), L.ft_ab(5.5, 5.0) + (0.2,))]


def compare_poses():
    out = []
    for pid, c, t in COMPARE:
        cam, tgt = L.cage_to_ned(*c), L.cage_to_ned(*t)
        d = tgt - cam
        pitch = math.atan2(-d[2], math.hypot(d[0], d[1]))     # NED: target below -> nose down (<0)
        out.append(make_pose(pid, 'compare', cam, math.atan2(d[1], d[0]), pitch, 0.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'world', 'poses.json'))
    a = ap.parse_args()
    poses, skipped = [], []
    for (pid, g, cam, yaw, pitch, roll) in route_poses() + extra_poses():
        if not L.pose_is_free(cam, margin=0.10):
            skipped.append(pid); continue
        poses.append(make_pose(pid, g, cam, yaw, pitch, roll))
    poses += compare_poses()
    with open(a.out, 'w') as f:
        json.dump({'poses': poses}, f, indent=0)
    from collections import Counter
    print(len(poses), 'poses', dict(Counter(p['group'] for p in poses)), 'skipped extras:', skipped)


if __name__ == '__main__':
    main()
