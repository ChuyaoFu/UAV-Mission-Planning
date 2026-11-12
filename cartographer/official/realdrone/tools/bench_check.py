#!/usr/bin/env python3
"""Read-only bench check of the team13 stereo depth + mission (dry run) on the drone. Publishes nothing.
Usage (container): python3 bench_check.py [seconds] [out.npz] [setpoint_topic]
Reports per topic: count, rate by header stamp and by arrival, stamp age; TF base_link -> depth frame;
depth stats (valid fraction, centre-ROI median); setpoint vs odometry (ENU) agreement."""
import json, math, sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
OUT = sys.argv[2] if len(sys.argv) > 2 else None
SP_TOPIC = sys.argv[3] if len(sys.argv) > 3 else '/team13/dry_run/setpoint_pose'


def st(m): return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


class Check(Node):
    def __init__(self):
        super().__init__('team13_bench_check')
        self.rx = {}; self.last = {}
        for t, typ, q in (('/camera/depth/image_raw', Image, qos_profile_sensor_data), ('/stereo/left/image', Image, qos_profile_sensor_data),
                          ('/camera/depth/camera_info', CameraInfo, qos_profile_sensor_data), ('/stereo/left/camera_info', CameraInfo, qos_profile_sensor_data),
                          *(((SP_TOPIC, PoseStamped, 10),) if SP_TOPIC != '/skybeta/setpoint/pose' else ()),
                          ('/skybeta/setpoint/pose', PoseStamped, 10),
                          ('/visual_slam/tracking/odometry', Odometry, 10)):
            self.create_subscription(typ, t, lambda m, t=t: self.on(t, m), q)
        from tf2_ros import Buffer, TransformListener
        self.buf = Buffer(); self.tfl = TransformListener(self.buf, self)
        self.sp_err = []; self.depth_stats = []; self.saved = None

    def on(self, t, m):
        now = time.time()
        self.rx.setdefault(t, []).append((now, st(m), m.header.frame_id)); self.last[t] = m
        if t == '/camera/depth/image_raw':
            d = np.frombuffer(bytes(m.data), np.float32).reshape(m.height, m.width)
            h, w = d.shape; c = d[h // 3:2 * h // 3, w // 3:2 * w // 3]; cv = c[c > 0]
            self.depth_stats.append((float(np.mean(d > 0)), float(np.median(cv)) if cv.size > 50 else None,
                                     float(np.percentile(d[d > 0], 5)) if np.any(d > 0) else None))
            L = self.last.get('/stereo/left/image')
            if L is not None and st(L) == st(m) and self.saved is None and len(self.depth_stats) > 20:
                self.saved = (np.frombuffer(bytes(L.data), np.uint8).reshape(L.height, L.width).copy(), d.copy())
        if t == SP_TOPIC and '/visual_slam/tracking/odometry' in self.last:
            o = self.last['/visual_slam/tracking/odometry'].pose.pose.position; p = m.pose.position
            self.sp_err.append(math.dist((o.x, o.y, o.z), (p.x, p.y, p.z)))


rclpy.init(); n = Check(); t0 = time.time()
while time.time() - t0 < DUR:
    rclpy.spin_once(n, timeout_sec=0.05)
rep = {}
for t, v in sorted(n.rx.items()):
    a = np.array([x[0] for x in v]); s = np.array([x[1] for x in v])
    r = dict(n=len(v), frame=v[-1][2])
    if len(v) > 2:
        r.update(hz_arrival=round((len(v) - 1) / (a[-1] - a[0]), 2), hz_stamp=round((len(v) - 1) / max(s[-1] - s[0], 1e-9), 2),
                 gap_max_ms=round(1e3 * float(np.max(np.diff(a))), 1), arrival_minus_stamp_ms_med=round(1e3 * float(np.median(a - s)), 1))
    rep[t] = r
dl, dd = n.rx.get('/camera/depth/image_raw', []), n.rx.get('/stereo/left/image', [])
rep['depth_left_same_stamp'] = len(set(x[1] for x in dl) & set(x[1] for x in dd))
if dl:
    try:
        x = n.buf.lookup_transform('base_link', dl[-1][2], rclpy.time.Time())
        tr, q = x.transform.translation, x.transform.rotation
        rep['tf_base_link_to_depth'] = dict(t=[round(tr.x, 4), round(tr.y, 4), round(tr.z, 4)], q_xyzw=[round(q.x, 4), round(q.y, 4), round(q.z, 4), round(q.w, 4)])
    except Exception as e:
        rep['tf_base_link_to_depth'] = f'FAIL {e}'
if n.depth_stats:
    ds = np.array([(a, b if b is not None else np.nan, c if c is not None else np.nan) for a, b, c in n.depth_stats])
    rep['depth'] = dict(valid_mean=round(float(np.nanmean(ds[:, 0])), 3), centre_median_m=round(float(np.nanmedian(ds[:, 1])), 3),
                        p5_median_m=round(float(np.nanmedian(ds[:, 2])), 3))
if '/camera/depth/camera_info' in n.last:
    ci = n.last['/camera/depth/camera_info']; rep['depth_info'] = dict(w=ci.width, h=ci.height, k=[round(c, 2) for c in ci.k], frame=ci.header.frame_id)
if n.sp_err:
    rep['setpoint_minus_odom_m'] = dict(n=len(n.sp_err), med=round(float(np.median(n.sp_err)), 4), max=round(float(np.max(n.sp_err)), 4))
rep['contract_setpoint_publishers_now'] = n.count_publishers('/skybeta/setpoint/pose')
print(json.dumps(rep, indent=1))
if OUT and n.saved is not None:
    np.savez_compressed(OUT, left=n.saved[0], depth=n.saved[1]); print('saved', OUT)
