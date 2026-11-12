#!/usr/bin/env python3
"""VIO -> PX4 external vision + /estimator/healthy for the official-compat profile
(stands in for the official onboard/vio/ev_bridge.py + cuVSLAM, whose rules are not yet captured).

source:=vio (default)  OpenVINS /ov_msckf/odomimu (IMU pose in the gravity-aligned 'global' frame,
                       JPL q_GtoI published as xyzw == Hamilton R_world_imu; twist = IMU-frame velocity)
source:=gt             Gazebo ground truth /gz/gt/odometry (debug only; stack.json records which source flew)
Both are re-anchored at their first sample: odom = gravity aligned, origin/yaw of base_link at start.
Outputs
  /fmu/in/vehicle_visual_odometry  px4_msgs/VehicleOdometry, NED/FRD of base_link, ~30 Hz, stamps = system clock us
                                   (PX4 clamps to arrival with UXRCE_DDS_SYNCT=0; EKF2_EV_DELAY compensates latency)
  /estimator/healthy               std_msgs/Bool at 10 Hz: true only while the source is fresh (< max_age_s sim),
                                   finite, slow enough and without a position jump
  /tf odom->base_link              from the same pose (sim-time stamp of the VIO sample)
  /visual_slam/tracking/odometry   nav_msgs/Odometry of base_link in odom (official cuVSLAM topic NAME, OpenVINS DATA)
"""
import json, math
from pathlib import Path
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster

P = json.loads((Path(__file__).resolve().parent / 'profile.json').read_text())
PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
F = np.diag([1., -1., -1.])          # FLU <-> FRD and world-FLU <-> NED


def q_to_R(x, y, z, w):
    n = math.sqrt(x*x + y*y + z*z + w*w); x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def R_to_q_wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2; return (0.25*s, (R[2, 1]-R[1, 2])/s, (R[0, 2]-R[2, 0])/s, (R[1, 0]-R[0, 1])/s)
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return ((R[2, 1]-R[1, 2])/s, 0.25*s, (R[0, 1]+R[1, 0])/s, (R[0, 2]+R[2, 0])/s)
    if i == 1:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return ((R[0, 2]-R[2, 0])/s, (R[0, 1]+R[1, 0])/s, 0.25*s, (R[1, 2]+R[2, 1])/s)
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return ((R[1, 0]-R[0, 1])/s, (R[0, 2]+R[2, 0])/s, (R[1, 2]+R[2, 1])/s, 0.25*s)


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


class EvBridge(Node):
    def __init__(self):
        super().__init__('ev_bridge')
        e = P['estimator']
        self.source = self.declare_parameter('source', 'vio').value
        self.rate = float(self.declare_parameter('ev_rate_hz', 30.0).value)
        self.max_age, self.max_speed, self.max_jump = e['max_age_s'], e['max_speed_mps'], e['max_jump_m']
        self.t_bi = np.asarray(P['imu']['in_base_m'], float)        # IMU in base_link (axes aligned)
        from px4_msgs.msg import VehicleOdometry
        self.VO = VehicleOdometry
        self.pub_ev = self.create_publisher(VehicleOdometry, '/fmu/in/vehicle_visual_odometry', PX4_QOS)
        self.pub_h = self.create_publisher(Bool, e['healthy_topic'], 10)
        self.pub_odom = self.create_publisher(Odometry, '/visual_slam/tracking/odometry', 10)
        self.tfb = TransformBroadcaster(self)
        self.sys_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
        topic = '/ov_msckf/odomimu' if self.source == 'vio' else '/gz/gt/odometry'
        self.create_subscription(Odometry, topic, self.on_odom, 10)
        self.last = None; self.rx_st = -1e9; self.anchor = None; self.reset_counter = 0
        self.ok = False; self.why = 'no_data'; self.n = 0; self.last_log = -1e9
        # health is published on a steady (wall) timer so it never starves; freshness is judged in sim time
        self.create_timer(1.0 / e['healthy_rate_hz'], self.on_timer, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(f'ev_bridge source={self.source} topic={topic}')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def base_pose(self, m):
        """-> (stamp, p_world(base, FLU world), R_world_base, v_world(base), w_body)"""
        st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        o, p = m.pose.pose.orientation, m.pose.pose.position
        R = q_to_R(o.x, o.y, o.z, o.w); pw = np.array([p.x, p.y, p.z])
        tw = m.twist.twist
        v_child = np.array([tw.linear.x, tw.linear.y, tw.linear.z]); w = np.array([tw.angular.x, tw.angular.y, tw.angular.z])
        if self.source == 'vio':                       # child = IMU, axes == body axes
            v = R @ v_child; p_base = pw - R @ self.t_bi
            w_body = w if np.all(np.isfinite(w)) and np.any(w != 0) else np.zeros(3)
            v_base = v - R @ np.cross(w_body, self.t_bi)
        else:                                          # gz odometry: child = model base, twist in child frame
            p_base = pw; v_base = R @ v_child; w_body = w
        # odom frame = gravity-aligned, origin and yaw of base_link at the first sample (like cuVSLAM odom);
        # OpenVINS' own global frame has an arbitrary yaw (180 deg here)
        if self.anchor is None and np.all(np.isfinite(p_base)) and np.all(np.isfinite(R)):
            self.anchor = (p_base.copy(), Rz(-yaw_of(R)))
            self.get_logger().info(f'odom anchor: p0={np.round(p_base, 3).tolist()} yaw0={math.degrees(yaw_of(R)):.1f} deg')
        if self.anchor is not None:
            p0, Ra = self.anchor
            p_base = Ra @ (p_base - p0); R = Ra @ R; v_base = Ra @ v_base
        return st, p_base, R, v_base, w_body

    def on_odom(self, m):
        st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.rx_st = st                                # freshness only; OpenVINS odomimu comes at IMU rate (480 Hz)
        if self.last is not None and st - self.last[0] < 1.0 / self.rate - 1e-4:
            return                                     # process (and check jumps) at ev_rate_hz only
        st, p, R, v, w = self.base_pose(m)
        good = bool(np.all(np.isfinite(p)) and np.all(np.isfinite(R)) and np.all(np.isfinite(v)))
        why = None if good else 'non_finite'
        if good and np.linalg.norm(v) > self.max_speed:
            why = 'too_fast'
        if good and self.last is not None:
            dt = max(1e-3, st - self.last[0])
            if np.linalg.norm(p - self.last[1]) > self.max_jump + self.max_speed * dt:
                why = 'jump'; self.reset_counter = (self.reset_counter + 1) % 256
        self.last = (st, p, R, v, w, why); self.n += 1
        if why is None:
            self.publish(st, p, R, v, w)

    def publish(self, st, p, R, v, w):
        us = self.sys_clock.now().nanoseconds // 1000
        e = self.VO(); e.timestamp = us; e.timestamp_sample = us
        e.pose_frame = self.VO.POSE_FRAME_NED
        pn = F @ p; e.position = [float(c) for c in pn]
        Rn = F @ R @ F; q = R_to_q_wxyz(Rn); e.q = [float(c) for c in q]
        e.velocity_frame = self.VO.VELOCITY_FRAME_NED; e.velocity = [float(c) for c in F @ v]
        e.angular_velocity = [float(c) for c in F @ w]
        e.position_variance = [0.0025] * 3; e.orientation_variance = [0.0025] * 3; e.velocity_variance = [0.01] * 3
        e.reset_counter = self.reset_counter; e.quality = 100
        self.pub_ev.publish(e)
        sec, nsec = int(st), int(round((st - int(st)) * 1e9))
        t = TransformStamped(); t.header.stamp.sec, t.header.stamp.nanosec = sec, min(nsec, 999999999)
        t.header.frame_id, t.child_frame_id = 'odom', 'base_link'
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (float(c) for c in p)
        qw, qx, qy, qz = R_to_q_wxyz(R)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = qx, qy, qz, qw
        self.tfb.sendTransform(t)
        o = Odometry(); o.header = t.header; o.child_frame_id = 'base_link'
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = (float(c) for c in p)
        o.pose.pose.orientation = t.transform.rotation
        vb = R.T @ v
        o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.linear.z = (float(c) for c in vb)
        o.twist.twist.angular.x, o.twist.twist.angular.y, o.twist.twist.angular.z = (float(c) for c in w)
        self.pub_odom.publish(o)

    def on_timer(self):
        now = self.now()
        if self.last is None:
            ok, why = False, 'no_data'
        elif now - self.rx_st > self.max_age:
            ok, why = False, f'stale_{now - self.rx_st:.2f}s'
        elif self.last[5] is not None:
            ok, why = False, self.last[5]
        else:
            ok, why = True, 'ok'
        if ok != self.ok or why.split('_')[0] != self.why.split('_')[0] or now - self.last_log > 10.0:
            self.get_logger().info(f'healthy={ok} ({why}) n={self.n}'); self.last_log = now
        self.ok, self.why = ok, why
        self.pub_h.publish(Bool(data=ok))


def main():
    rclpy.init(); n = EvBridge()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
