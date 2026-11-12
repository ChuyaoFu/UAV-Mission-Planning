#!/usr/bin/env python3
"""Read-only: grab N synchronized left/right image_mono frames + camera_info + TF into an .npz.
Usage (inside the skybeta container): python3 capture_pair.py OUT.npz [N]"""
import sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

out = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def st(m): return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


class Cap(Node):
    def __init__(self):
        super().__init__('team13_capture_pair')
        self.L = {}; self.R = {}; self.info = {}; self.pairs = []
        for side, d in (('left', self.L), ('right', self.R)):
            self.create_subscription(Image, f'/oak/{side}/image_mono', lambda m, d=d: self.on_img(m, d), qos_profile_sensor_data)
            self.create_subscription(CameraInfo, f'/oak/{side}/camera_info', lambda m, s=side: self.info.__setitem__(s, m), qos_profile_sensor_data)

    def on_img(self, m, d):
        img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.step)[:, :m.width].copy()
        d[round(st(m), 3)] = img
        common = sorted(set(self.L) & set(self.R))
        for t in common:
            self.pairs.append((t, self.L.pop(t), self.R.pop(t)))
        for dd in (self.L, self.R):
            for k in sorted(dd)[:-10]: dd.pop(k)


rclpy.init(); n = Cap(); t0 = time.time()
while len(n.pairs) < N * 10 and time.time() - t0 < 20: rclpy.spin_once(n, timeout_sec=0.05)
pairs = n.pairs[::10][:N]
from tf2_ros import Buffer, TransformListener
buf = Buffer(); TransformListener(buf, n); t1 = time.time(); tf = {}
while time.time() - t1 < 3:
    rclpy.spin_once(n, timeout_sec=0.05)
for a, b in (('oak_left_camera_optical_frame', 'oak_right_camera_optical_frame'), ('base_link', 'oak_left_camera_optical_frame')):
    try:
        x = buf.lookup_transform(a, b, rclpy.time.Time()); tr, q = x.transform.translation, x.transform.rotation
        tf[a + '->' + b] = [tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w]
    except Exception as e: print('tf fail', a, b, e)
info = {s: dict(K=list(m.k), D=list(m.d), R=list(m.r), P=list(m.p), w=m.width, h=m.height, model=m.distortion_model, frame=m.header.frame_id) for s, m in n.info.items()}
np.savez_compressed(out, stamps=np.array([p[0] for p in pairs]), left=np.stack([p[1] for p in pairs]), right=np.stack([p[2] for p in pairs]),
                    info=np.array(repr(info)), tf=np.array(repr(tf)))
print('saved', out, len(pairs), 'pairs', {k: v[:3] for k, v in tf.items()}, list(info))
