#!/usr/bin/env python3
"""Extract stereo evaluation frames from an official-compat flight (runs where rosbag2_py + mcap are available).

  python3 netmask_extract.py FLIGHT_DIR OUT.npz [--tmax 120] [--nmax 3.0] [--every 3] [--cap 90]

Inputs: FLIGHT_DIR/bag (/stereo/left/image, /camera/depth/image_raw = the compat SGBM depth, /stereo/left/camera_info)
and FLIGHT_DIR/sim/sidebag (/stereo/right/image, /gz/gt/odometry).
Pose: Gazebo truth re-anchored at its first sample on the pad and rotated by the initial yaw (= the start frame, the
convention of vio_error.py), converted to a PX4-like NED attitude + position, then the camera pose with the compat
profile extrinsic (common.geometry.camera_pose_ned). Layout NED = that NED with d - 0.05 (base_link 5 cm above floor).
Frames: from the first image while t - t0 <= tmax and the vehicle's north offset <= nmax (start pose + first corridor),
every `every`-th stereo pair, at most `cap`.
Output npz: left, right (N,H,W uint8), depth (N,H,W float16, compat SGBM), stamp (N), R_nc (N,3,3), cam_ned (N,3)
(layout NED), K (fx, fy, cx, cy), baseline, flight.
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

F_FLU_FRD = np.diag([1.0, -1.0, -1.0])


def read(bag, topics):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    r.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    out = {t: [] for t in topics}; cls = {t: get_message(types[t]) for t in topics if t in types}
    while r.has_next():
        t, data, _ = r.read_next()
        out[t].append(deserialize_message(data, cls[t]))
    return out


def st(m):
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def img(m):
    a = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.step)
    return a[:, :m.width].copy()


def quat_to_mat(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('flight'); ap.add_argument('out'); ap.add_argument('--tmax', type=float, default=120.0)
    ap.add_argument('--nmax', type=float, default=3.0); ap.add_argument('--every', type=int, default=3)
    ap.add_argument('--cap', type=int, default=90); ap.add_argument('--profile', default='')
    ap.add_argument('--events', type=float, nargs=2, metavar=('BEFORE_S', 'AFTER_S'),
                    help='instead of tmax/nmax/every: every frame within [t-BEFORE, t+AFTER] of each small_cluster_kept '
                         'event in FLIGHT/mission.log (mission t = sim time)')
    a = ap.parse_args(); F = Path(a.flight)
    prof = Path(a.profile) if a.profile else F / 'sim' / 'asflown' / 'profile.json'
    P = json.loads(prof.read_text())['camera']
    R_bc = np.asarray(P['R_base_optical'], float); t_bc = np.asarray(P['left_in_base_m'], float)
    A = read(F / 'bag', ['/stereo/left/image', '/camera/depth/image_raw', '/stereo/left/camera_info',
                         '/visual_slam/tracking/odometry'])
    vio = np.array([[st(m), m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z,
                     m.pose.pose.orientation.w, m.pose.pose.orientation.x, m.pose.pose.orientation.y,
                     m.pose.pose.orientation.z] for m in A['/visual_slam/tracking/odometry']])
    S = read(F / 'sim' / 'sidebag', ['/stereo/right/image', '/gz/gt/odometry'])
    ci = A['/stereo/left/camera_info'][0]; K = (ci.k[0], ci.k[4], ci.k[2], ci.k[5])
    B = -ci.p[3] / ci.p[0] if ci.p[3] else float(P.get('baseline_m', 0.075))
    rights = {round(st(m), 4): m for m in S['/stereo/right/image']}
    depths = {round(st(m), 4): m for m in A['/camera/depth/image_raw']}
    g = S['/gz/gt/odometry']
    gt = np.array([[st(m), m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z,
                    m.pose.pose.orientation.w, m.pose.pose.orientation.x, m.pose.pose.orientation.y,
                    m.pose.pose.orientation.z] for m in g])
    y0 = 2 * math.atan2(gt[0, 7], gt[0, 4]); c, s = math.cos(-y0), math.sin(-y0)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    lefts = A['/stereo/left/image']; t0 = st(lefts[0])
    wins, events = [], []
    if a.events:
        for ln in (F / 'mission.log').read_text().splitlines():
            if ln.startswith('{') and 'small_cluster_kept' in ln:
                e = json.loads(ln.replace('NaN', 'null'))['event']
                events.append(dict(t=e['t'], n=e.get('n'), center=e.get('center'), free_m=e.get('free_m')))
                wins.append((e['t'] - a.events[0], e['t'] + a.events[1]))
    out = dict(left=[], right=[], depth=[], stamp=[], R_nc=[], cam_ned=[], R_nc_vio=[], cam_vio=[], vio_dt=[])
    k = 0
    for m in lefts:
        t = st(m)
        if wins:
            if not any(w0 <= t <= w1 for w0, w1 in wins):
                continue
        elif t - t0 > a.tmax:
            break
        key = round(t, 4)
        if key not in rights or t < gt[0, 0] or t > gt[-1, 0]:
            continue
        i = int(np.searchsorted(gt[:, 0], t)); i = min(max(i, 1), len(gt) - 1)
        j = i if abs(gt[i, 0] - t) < abs(gt[i - 1, 0] - t) else i - 1
        w = float(np.clip((t - gt[i - 1, 0]) / max(gt[i, 0] - gt[i - 1, 0], 1e-9), 0, 1))
        p = Rz @ ((1 - w) * gt[i - 1, 1:4] + w * gt[i, 1:4] - gt[0, 1:4])
        if not wins:
            if p[0] > a.nmax:
                break
            k += 1
            if (k - 1) % a.every:
                continue
        R_sb = Rz @ quat_to_mat(*gt[j, 4:8])                       # start-frame FLU <- body FLU
        R_nb = F_FLU_FRD @ R_sb @ F_FLU_FRD                         # NED <- FRD (PX4-like attitude)
        p_ned = F_FLU_FRD @ p
        R_nc = R_nb @ F_FLU_FRD @ R_bc
        cam = p_ned + R_nb @ (F_FLU_FRD @ t_bc) + np.array([0, 0, -0.05])
        out['left'].append(img(m)); out['right'].append(img(rights[key]))
        dm = depths.get(key)
        out['depth'].append(np.frombuffer(bytes(dm.data), np.float32).reshape(dm.height, -1)[:, :dm.width].astype(np.float16)
                            if dm is not None else np.zeros((m.height, m.width), np.float16))
        out['stamp'].append(t); out['R_nc'].append(R_nc); out['cam_ned'].append(cam)
        # the flight's own VIO (ev_bridge odom: start frame FLU, origin base_link on the pad) - what the node would use
        iv = int(np.argmin(np.abs(vio[:, 0] - t))) if len(vio) else None
        if iv is not None:
            R_sbv = quat_to_mat(*vio[iv, 4:8]); R_nbv = F_FLU_FRD @ R_sbv @ F_FLU_FRD
            out['R_nc_vio'].append(R_nbv @ F_FLU_FRD @ R_bc)
            out['cam_vio'].append(F_FLU_FRD @ vio[iv, 1:4] + R_nbv @ (F_FLU_FRD @ t_bc) + np.array([0, 0, -0.05]))
            out['vio_dt'].append(vio[iv, 0] - t)
        if len(out['left']) >= a.cap and not wins:
            break
    np.savez_compressed(a.out, **{k_: np.asarray(v) for k_, v in out.items()}, K=np.asarray(K), baseline=B,
                        flight=F.name, events=json.dumps(events),
                        prior=(F / 'prior.json').read_text() if (F / 'prior.json').exists() else '')
    print(json.dumps(dict(flight=F.name, frames=len(out['left']), K=K, baseline=B,
                          t_span=round(out['stamp'][-1] - out['stamp'][0], 1) if out['stamp'] else 0,
                          n_max=round(float(max(c_[0] for c_ in out['cam_ned'])), 2) if out['cam_ned'] else None)))


if __name__ == '__main__':
    sys.exit(main())
