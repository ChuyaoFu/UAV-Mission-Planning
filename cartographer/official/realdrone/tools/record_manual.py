#!/usr/bin/env python3
"""Team 13 manual-flight recorder: the pilot flies (RC, Position), we only RECORD. Publishes nothing.

Writes <out>/
  pairs/<stamp_ns>_L.png, _R.png   raw OAK left/right (1024x640 bgr8 -> gray, lossless PNG), time-paired, --rate Hz
  pairs.csv                       stamp, files, VIO odom pose at the stamp (nearest <= 50 ms), vo_state, PX4 local z
  camera_info.json                both CameraInfo (K, D, R, P, size) as published
  stats.json                      frames written / dropped, MB, duration
The raw bgr8 pair is ~100 MB/s; gray PNG at 10 Hz is ~5-9 MB/s. Stops by itself after --minutes or when the disk
has < --min-free-gb left. Poses/PX4/IMU/TF go to a small bag recorded next to it (team13_onboard.sh record).
"""
import argparse
import bisect
import json
import os
import queue
import shutil
import signal
import sys
import threading
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--rate', type=float, default=10.0, help='stereo pairs per second to keep')
    ap.add_argument('--minutes', type=float, default=15.0, help='stop after this long')
    ap.add_argument('--min-free-gb', type=float, default=15.0, help='stop when the disk has less free space')
    ap.add_argument('--png-level', type=int, default=1)
    a = ap.parse_args()

    import cv2
    import numpy as np
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    os.makedirs(os.path.join(a.out, 'pairs'), exist_ok=True)
    csv = open(os.path.join(a.out, 'pairs.csv'), 'a', buffering=1)
    if csv.tell() == 0:
        csv.write('stamp_ns,left,right,odom_x,odom_y,odom_z,odom_qx,odom_qy,odom_qz,odom_qw,odom_dt_ms,vo_state\n')
    rclpy.init()
    node = rclpy.create_node('team13_record_manual')
    S = dict(left={}, right={}, info={}, odom=[], odom_t=[], vo=-1, written=0, dropped=0, unpaired=0, bytes=0, last_keep=0)
    lock = threading.Lock()
    work = queue.Queue(maxsize=16)
    stop = threading.Event()

    def to_gray(m):
        buf = np.frombuffer(m.data, np.uint8)
        if m.encoding in ('bgr8', 'rgb8'):
            img = buf.reshape(m.height, m.step // 3 if m.step else m.width, 3)[:, :m.width]
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if m.encoding == 'bgr8' else cv2.COLOR_RGB2GRAY)
        if m.encoding in ('mono8', '8UC1'):
            return buf.reshape(m.height, m.step or m.width)[:, :m.width]
        raise ValueError('unsupported encoding ' + m.encoding)

    def stamp_ns(m):
        return m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec

    def on_img(side):
        other = 'right' if side == 'left' else 'left'

        def cb(m):
            t = stamp_ns(m)
            with lock:
                S[side][t] = m
                best = min(S[other], key=lambda k: abs(k - t), default=None)
                if best is None or abs(best - t) > 5_000_000:      # wait for the other side (or it was dropped)
                    for d in (S['left'], S['right']):
                        for k in sorted(d)[:-10]:
                            del d[k]; S['unpaired'] += 1
                    return
                om = S[other].pop(best); S[side].pop(t)
                if t - S['last_keep'] < 1e9 / a.rate:
                    return
                S['last_keep'] = t
                lm, rm = (m, om) if side == 'left' else (om, m)
            try:
                work.put_nowait((stamp_ns(lm), lm, rm))
            except queue.Full:
                S['dropped'] += 1
        return cb

    def on_info(side):
        def cb(m):
            if side not in S['info']:
                S['info'][side] = dict(frame_id=m.header.frame_id, width=m.width, height=m.height, distortion_model=
                                       m.distortion_model, D=list(m.d), K=list(m.k), R=list(m.r), P=list(m.p))
                json.dump(S['info'], open(os.path.join(a.out, 'camera_info.json'), 'w'), indent=1)
        return cb

    def on_odom(m):
        t = stamp_ns(m); p, q = m.pose.pose.position, m.pose.pose.orientation
        with lock:
            S['odom_t'].append(t); S['odom'].append((p.x, p.y, p.z, q.x, q.y, q.z, q.w))
            if len(S['odom_t']) > 600:
                del S['odom_t'][:300]; del S['odom'][:300]

    def on_vo(m):
        S['vo'] = int(m.vo_state)

    def writer():
        while not (stop.is_set() and work.empty()):
            try:
                t, lm, rm = work.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                lp, rp = f'pairs/{t}_L.png', f'pairs/{t}_R.png'
                for path, msg in ((lp, lm), (rp, rm)):
                    full = os.path.join(a.out, path)
                    cv2.imwrite(full, to_gray(msg), [cv2.IMWRITE_PNG_COMPRESSION, a.png_level])
                    S['bytes'] += os.path.getsize(full)
                with lock:
                    i = bisect.bisect_left(S['odom_t'], t)
                    cand = [j for j in (i - 1, i) if 0 <= j < len(S['odom_t'])]
                    j = min(cand, key=lambda j: abs(S['odom_t'][j] - t)) if cand else None
                    pose = S['odom'][j] if j is not None else (float('nan'),) * 7
                    dt = abs(S['odom_t'][j] - t) / 1e6 if j is not None else float('nan')
                csv.write('%d,%s,%s,%s,%.1f,%d\n' % (t, lp, rp, ','.join('%.5f' % v for v in pose), dt, S['vo']))
                S['written'] += 1
            except Exception as e:      # keep recording whatever happens to one pair
                print('[record] write error: %s' % e, flush=True)

    try:
        from isaac_ros_visual_slam_interfaces.msg import VisualSlamStatus
        node.create_subscription(VisualSlamStatus, '/visual_slam/status', on_vo, 10)
    except ImportError:
        pass
    node.create_subscription(Image, '/oak/left/image_raw', on_img('left'), qos_profile_sensor_data)
    node.create_subscription(Image, '/oak/right/image_raw', on_img('right'), qos_profile_sensor_data)
    node.create_subscription(CameraInfo, '/oak/left/camera_info', on_info('left'), qos_profile_sensor_data)
    node.create_subscription(CameraInfo, '/oak/right/camera_info', on_info('right'), qos_profile_sensor_data)
    node.create_subscription(Odometry, '/visual_slam/tracking/odometry', on_odom, 50)
    th = threading.Thread(target=writer, daemon=True); th.start()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop.set())
    t0 = time.time(); last_print = t0
    print(f'[record] recording stereo pairs at {a.rate} Hz to {a.out} (stop: team13_onboard.sh stop, or after '
          f'{a.minutes} min)', flush=True)
    why = 'signal'
    while not stop.is_set():
        rclpy.spin_once(node, timeout_sec=0.1)
        now = time.time()
        if now - t0 > a.minutes * 60:
            why = 'time limit'; break
        if now - last_print > 10:
            last_print = now
            free = shutil.disk_usage(a.out).free / 1e9
            print('[record] %4.0f s  pairs %d  dropped %d  unpaired %d  %.0f MB  disk free %.1f GB  vo_state %d' % (
                now - t0, S['written'], S['dropped'], S['unpaired'], S['bytes'] / 1e6, free, S['vo']), flush=True)
            if free < a.min_free_gb:
                why = 'disk low'; break
    stop.set(); th.join(timeout=10)
    stats = dict(why=why, seconds=round(time.time() - t0, 1), pairs=S['written'], dropped=S['dropped'],
                 unpaired=S['unpaired'], mb=round(S['bytes'] / 1e6, 1), rate=a.rate)
    json.dump(stats, open(os.path.join(a.out, 'stats.json'), 'w'), indent=1)
    print('[record] stopped (%s): %s' % (why, stats), flush=True)
    node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
