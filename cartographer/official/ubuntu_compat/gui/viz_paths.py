#!/usr/bin/env python3
"""Live display helper for the official-compat GUI (RViz): paths + error label in ONE frame (`odom`).

  /viz/vio_path    nav_msgs/Path  VIO (OpenVINS via ev_bridge, /visual_slam/tracking/odometry), frame odom
  /viz/truth_path  nav_msgs/Path  Gazebo truth (/gz/gt/odometry) moved into odom: odom = base_link pose at the first
                                   truth sample (vehicle at rest on the pad = where ev_bridge anchors odom)
  /viz/px4_path    nav_msgs/Path  PX4 EKF (/fmu/out/vehicle_local_position_v1, NED -> FLU)
  /viz/error       visualization_msgs/Marker  text above the vehicle: VIO - truth (cm) now / max, and PX4 - truth
Sim-side display only (truth is never fed to the flight stack).
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker

PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
LATCH = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                   history=HistoryPolicy.KEEP_LAST, depth=1)


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class VizPaths(Node):
    def __init__(self):
        super().__init__('viz_paths')
        self.paths = {k: Path() for k in ('vio', 'truth', 'px4')}
        for p in self.paths.values():
            p.header.frame_id = 'odom'
        self.pub = {k: self.create_publisher(Path, f'/viz/{k}_path', LATCH) for k in self.paths}
        self.pub_err = self.create_publisher(Marker, '/viz/error', 10)
        self.last = {k: None for k in self.paths}; self.cur = {k: None for k in self.paths}
        self.anchor = None; self.err_max = 0.0; self.px4_err_max = 0.0
        self.create_subscription(Odometry, '/visual_slam/tracking/odometry', self.on_vio, 20)
        self.create_subscription(Odometry, '/gz/gt/odometry', self.on_truth, 20)
        from px4_msgs.msg import VehicleLocalPosition
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.on_px4, PX4_QOS)
        self.create_timer(0.2, self.tick)

    def add(self, key, stamp, xyz, q=None):
        p = np.asarray(xyz, float); self.cur[key] = p
        if self.last[key] is not None and np.linalg.norm(p - self.last[key]) < 0.02:
            return
        self.last[key] = p
        ps = PoseStamped(); ps.header.frame_id = 'odom'; ps.header.stamp = stamp
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(c) for c in p)
        if q is not None:
            ps.pose.orientation = q
        else:
            ps.pose.orientation.w = 1.0
        self.paths[key].poses.append(ps)

    def on_vio(self, m):
        p = m.pose.pose.position; self.add('vio', m.header.stamp, (p.x, p.y, p.z), m.pose.pose.orientation)

    def on_truth(self, m):
        p = m.pose.pose.position; q = m.pose.pose.orientation
        if self.anchor is None:
            self.anchor = (np.array([p.x, p.y, p.z]), yaw_of(q))
        o, y0 = self.anchor; c, s = math.cos(-y0), math.sin(-y0)
        d = np.array([p.x, p.y, p.z]) - o
        self.add('truth', m.header.stamp, (c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]))

    def on_px4(self, m):
        if m.xy_valid and m.z_valid:
            self.add('px4', self.get_clock().now().to_msg(), (m.x, -m.y, -m.z))

    def tick(self):
        now = self.get_clock().now().to_msg()
        for k, p in self.paths.items():
            p.header.stamp = now; self.pub[k].publish(p)
        t, v, x = self.cur['truth'], self.cur['vio'], self.cur['px4']
        if t is None:
            return
        txt = []
        if v is not None:
            e = float(np.linalg.norm(v - t)); self.err_max = max(self.err_max, e)
            txt.append(f'VIO err {100 * e:.0f} cm (max {100 * self.err_max:.0f})')
        if x is not None:
            e2 = float(np.linalg.norm(x - t)); self.px4_err_max = max(self.px4_err_max, e2)
            txt.append(f'PX4 err {100 * e2:.0f} cm (max {100 * self.px4_err_max:.0f})')
        mk = Marker(); mk.header.frame_id = 'odom'; mk.header.stamp = now; mk.ns = 'err'; mk.id = 0
        mk.type = Marker.TEXT_VIEW_FACING; mk.action = Marker.ADD
        mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = float(t[0]), float(t[1]), float(t[2]) + 0.5
        mk.pose.orientation.w = 1.0; mk.scale.z = 0.22; mk.color.r = mk.color.g = mk.color.b = mk.color.a = 1.0
        mk.text = '\n'.join(txt); self.pub_err.publish(mk)


def main():
    rclpy.init(); n = VizPaths()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
