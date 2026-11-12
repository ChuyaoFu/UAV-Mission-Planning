#!/usr/bin/env python3
"""Top-down obstacle map (START frame: forward/left) from OUR OWN depth in a flight bag.

Used to set a coarse course prior from the drone's own sensors (like eyeballing the venue):
  python3 tools/look_map.py <flight_dir>/bag [--cell 0.1] [--hmin 0.3 --hmax 1.4]
Prints an ASCII grid (# = measured obstacle, . = seen free/unknown) plus per-row lateral extents.
"""
import argparse, math, os, sys
import numpy as np

DEV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DEV)
from common.geometry import Frame, camera_pose_ned, OFFICIAL_K, OFFICIAL_R_BASE_CAM, OFFICIAL_T_BASE_CAM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('--cell', type=float, default=0.1)
    ap.add_argument('--hmin', type=float, default=0.3)
    ap.add_argument('--hmax', type=float, default=1.4)
    ap.add_argument('--every', type=int, default=3)
    a = ap.parse_args()
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=a.bag, storage_id='mcap'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    cls = {k: get_message(v) for k, v in types.items()}
    home = None; pos = None; q = None; yaw0 = None; K = OFFICIAL_K; nd = 0; pts = []
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic == '/fmu/out/vehicle_local_position_v1':
            m = deserialize_message(data, cls[topic]); pos = np.array([m.x, m.y, m.z], float)
            if home is None:
                home = pos.copy(); yaw0 = float(m.heading)
        elif topic == '/fmu/out/vehicle_attitude':
            m = deserialize_message(data, cls[topic]); q = tuple(float(c) for c in m.q)
        elif topic == '/camera/depth/camera_info':
            m = deserialize_message(data, cls[topic]); K = (m.k[0], m.k[4], m.k[2], m.k[5])
        elif topic == '/camera/depth/image_raw' and pos is not None and q is not None:
            nd += 1
            if nd % a.every or pos[2] - home[2] > -0.5:          # only while airborne (>= 0.5 m up)
                continue
            m = deserialize_message(data, cls[topic])
            if m.encoding == '32FC1':
                d = np.frombuffer(bytes(m.data), np.float32).reshape(m.height, -1)[:, :m.width]
            else:
                d = np.frombuffer(bytes(m.data), np.uint16).reshape(m.height, -1)[:, :m.width].astype(np.float32) / 1e3
            R_nc, cam = camera_pose_ned(q, pos, np.array(OFFICIAL_R_BASE_CAM, float), np.array(OFFICIAL_T_BASE_CAM, float))
            p, _ = Frame(image=None, depth=d, K=K, R_nc=R_nc, cam_ned=cam).points_ned(stride=4, max_range=5.0)
            pts.append(p)
    if not pts:
        print('no airborne depth frames'); return 1
    P = np.vstack(pts) - home
    c, s = math.cos(yaw0), math.sin(yaw0)
    f = P[:, 0] * c + P[:, 1] * s; l = P[:, 0] * s - P[:, 1] * c; up = -P[:, 2]
    keep = (up > a.hmin) & (up < a.hmax)
    f, l = f[keep], l[keep]
    print(f'{len(pts)} depth frames, {keep.sum()} points between {a.hmin} and {a.hmax} m up; yaw0 {math.degrees(yaw0):.1f} deg')
    fs = np.arange(-0.5, 4.0 + 1e-9, a.cell); ls = np.arange(2.0, -3.0 - 1e-9, -a.cell)
    H, _, _ = np.histogram2d(f, l, bins=[np.append(fs, fs[-1] + a.cell), np.sort(np.append(ls, ls[-1] - a.cell))])
    H = H[:, ::-1]                                                       # columns: left (+) ... right (-)
    print('rows = forward (m), cols = left +2.0 ... right -3.0 (cell %.2f m); # = >=3 hits, + = 1-2' % a.cell)
    for i in range(len(fs) - 1, -1, -1):
        row = ''.join('#' if H[i, j] >= 3 else ('+' if H[i, j] > 0 else '.') for j in range(H.shape[1]))
        occ = np.nonzero(H[i] >= 3)[0]
        left = [ls[j] for j in occ if ls[j] > 0]; right = [ls[j] for j in occ if ls[j] < 0]
        ext = f'L {min(left):+.1f}' if left else 'L  -  '
        ext += f' R {max(right):+.1f}' if right else ' R  -  '
        print(f'{fs[i]:+5.1f} {row} {ext}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
