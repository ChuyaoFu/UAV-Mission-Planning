#!/usr/bin/env python3
"""OFF-DRONE integration harness: a fake skybeta drone for the team13 stack + the organizers' real flight_interface.

Publishes what skybeta-05 publishes (names/frames/types as measured on the drone 2026-11-07):
  /oak/{left,right}/image_mono + camera_info  30 Hz, the REAL captured pairs (capture_pair.py .npz), cycled
  /tf_static  base_link -> oak_left_camera_optical_frame -> oak_right_camera_optical_frame (captured values)
  /visual_slam/tracking/odometry  ENU odom, base_link, body FLU twist, stamp = image stamp
  fake PX4 (resolved topic names, PX4 QoS): vehicle_status(_v1), vehicle_land_detected, vehicle_local_position
     (dist_bottom = height + 0.10, as measured landed); consumes /fmu/in/{offboard_control_mode,trajectory_setpoint,
     vehicle_command} from flight_interface: DO_SET_MODE(1,6) -> OFFBOARD if armed + heartbeat; offboard position
     control v = 0.95 * err (z 1.0), |a| <= 3; heartbeat lost > 1 s -> POSCTL; COM_DISARM_PRFLT 10 s; auto-disarm 2 s
     after landing.
  scripted pilot: arm (ALTCTL) at --arm-at, Position 2 s later; when the vehicle has hovered at the 0.3 m floor clamp
     for 2 s (mission asking to land) -> takes over (POSCTL), descends 0.5 m/s, lands; optional re-arm cycle.
Prints JSON events + a summary. Run ONLY in an isolated container (never on the drone's DDS domain).
"""
import argparse, ast, json, math, sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLandDetected,
                          VehicleLocalPosition, VehicleStatus)

ap = argparse.ArgumentParser()
ap.add_argument('--npz', required=True); ap.add_argument('--duration', type=float, default=80.0)
ap.add_argument('--arm-at', type=float, default=8.0); ap.add_argument('--rearm', action='store_true')
ap.add_argument('--no-switch', action='store_true', help='pilot arms but never switches to Position (expect PRFLT disarm)')
ap.add_argument('--takeover-at', type=float, default=None, help='pilot takes over (Position) at this time if in Offboard, then lands')
A = ap.parse_args()

try:
    from skybeta_flight_interface.topics import resolve_endpoints
    T = resolve_endpoints()
except Exception:  # noqa: BLE001
    T = dict(vehicle_status='/fmu/out/vehicle_status_v1', vehicle_local_position='/fmu/out/vehicle_local_position',
             vehicle_land_detected='/fmu/out/vehicle_land_detected', offboard_control_mode='/fmu/in/offboard_control_mode',
             trajectory_setpoint='/fmu/in/trajectory_setpoint', vehicle_command='/fmu/in/vehicle_command')
QOS_OUT = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST, depth=5)
QOS_IN = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=10)
ALTCTL, POSCTL, OFFBOARD, LOITER = 1, 2, 14, 4


def ev(t, **k):
    print(json.dumps(dict(t=round(t, 2), fake=k), default=str), flush=True)


class FakeDrone(Node):
    def __init__(self):
        super().__init__('team13_fake_drone')
        z = np.load(A.npz); self.L, self.R = z['left'], z['right']
        info = ast.literal_eval(str(z['info'])); tf = ast.literal_eval(str(z['tf']))
        self.info = {}
        for s in ('left', 'right'):
            i = info[s]; m = CameraInfo(); m.width, m.height = i['w'], i['h']; m.distortion_model = i['model']
            m.d = [float(c) for c in i['D']]; m.k = [float(c) for c in i['K']]; m.r = [float(c) for c in i['R']]
            m.p = [float(c) for c in i['P']]; m.header.frame_id = i['frame']; self.info[s] = m
        self.static = StaticTransformBroadcaster(self); ts = []
        for key, (parent, child) in (('base_link->oak_left_camera_optical_frame', ('base_link', 'oak_left_camera_optical_frame')),
                                     ('oak_left_camera_optical_frame->oak_right_camera_optical_frame',
                                      ('oak_left_camera_optical_frame', 'oak_right_camera_optical_frame'))):
            v = tf[key]; t = TransformStamped(); t.header.frame_id, t.child_frame_id = parent, child
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = v[:3]
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = v[3:]
            ts.append(t)
        self.static.sendTransform(ts)
        self.pub_img = {s: self.create_publisher(Image, f'/oak/{s}/image_mono', 10) for s in ('left', 'right')}
        self.pub_info = {s: self.create_publisher(CameraInfo, f'/oak/{s}/camera_info', 10) for s in ('left', 'right')}
        self.pub_odom = self.create_publisher(Odometry, '/visual_slam/tracking/odometry', 10)
        self.pub_status = self.create_publisher(VehicleStatus, T['vehicle_status'], QOS_OUT)
        self.pub_land = self.create_publisher(VehicleLandDetected, T['vehicle_land_detected'], QOS_OUT)
        self.pub_lpos = self.create_publisher(VehicleLocalPosition, T['vehicle_local_position'], QOS_OUT)
        self.create_subscription(OffboardControlMode, T['offboard_control_mode'], self.on_ocm, QOS_IN)
        self.create_subscription(TrajectorySetpoint, T['trajectory_setpoint'], self.on_sp, QOS_IN)
        self.create_subscription(VehicleCommand, T['vehicle_command'], self.on_cmd, QOS_IN)
        self.t0 = time.time(); self.k = 0
        self.p = np.zeros(3); self.v = np.zeros(3); self.yaw_ned = math.pi / 2     # ENU yaw 0 (x east)
        self.armed = False; self.nav = LOITER; self.landed = True; self.airborne_once = False
        self.ocm_t = None; self.sp = None; self.arm_t = None; self.landed_since = None; self.floor_since = None
        self.pilot = 'idle'; self.stats = dict(events=[]); self.max_alt = 0.0; self.cycle = 0
        self.create_timer(1 / 30.0, self.cam_tick); self.create_timer(1 / 50.0, self.px4_tick)
        self.create_timer(0.1, self.status_tick)

    def now(self): return time.time() - self.t0

    def mark(self, what, **k):
        self.stats['events'].append((round(self.now(), 2), what)); ev(self.now(), what=what, **k)

    # ---- flight_interface -> PX4
    def on_ocm(self, m): self.ocm_t = self.now()
    def on_sp(self, m): self.sp = m

    def on_cmd(self, m):
        if m.command == 176 and int(m.param1) == 1 and int(m.param2) == 6:
            ok = self.armed and self.ocm_t is not None and self.now() - self.ocm_t < 0.5
            if ok and self.nav != OFFBOARD:
                self.nav = OFFBOARD; self.mark('offboard_entered')
            elif not ok:
                self.mark('offboard_refused')
        else:
            self.mark('vehicle_command', command=int(m.command), p1=m.param1, p2=m.param2)

    # ---- sensors
    def cam_tick(self):
        st = self.get_clock().now().to_msg(); i = (self.k // 10) % len(self.L); self.k += 1
        for s, img in (('left', self.L[i]), ('right', self.R[i])):
            m = Image(); m.header.stamp = st; m.header.frame_id = self.info[s].header.frame_id
            m.height, m.width = img.shape; m.encoding = 'mono8'; m.step = img.shape[1]; m.data = img.tobytes()
            self.pub_img[s].publish(m); ci = self.info[s]; ci.header.stamp = st; self.pub_info[s].publish(ci)
        o = Odometry(); o.header.stamp = st; o.header.frame_id = 'odom'; o.child_frame_id = 'base_link'
        pe = (self.p[1], self.p[0], -self.p[2]); ye = math.pi / 2 - self.yaw_ned
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = pe
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = math.sin(ye / 2), math.cos(ye / 2)
        ve = np.array([self.v[1], self.v[0], -self.v[2]]); c, s_ = math.cos(ye), math.sin(ye)
        o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.linear.z = c * ve[0] + s_ * ve[1], -s_ * ve[0] + c * ve[1], ve[2]
        self.pub_odom.publish(o)

    # ---- PX4 + pilot
    def px4_tick(self):
        dt, t = 1 / 50.0, self.now()
        # scripted pilot
        if self.pilot == 'idle' and t > A.arm_at + self.cycle * 0:
            self.armed, self.nav, self.arm_t, self.pilot = True, ALTCTL, t, 'armed'; self.mark('pilot_arm_altctl')
        elif self.pilot == 'armed' and t > self.arm_t + 2.0 and not A.no_switch:
            self.nav, self.pilot = POSCTL, 'auto'; self.mark('pilot_switch_position')
        elif self.pilot == 'auto' and self.nav == OFFBOARD and self.airborne_once:
            alt = -self.p[2]
            if alt < 0.36 and abs(self.v[2]) < 0.05 and self.sp is not None and -self.sp.position[2] < 0.36:
                self.floor_since = self.floor_since or t
                if t - self.floor_since > 2.0:
                    self.nav, self.pilot = POSCTL, 'landing'; self.mark('pilot_takeover_land', alt=round(alt, 3))
            else:
                self.floor_since = None
        elif self.pilot == 'rearm_wait' and t > self.rearm_t:
            self.armed, self.nav, self.arm_t, self.pilot = True, ALTCTL, t, 'armed'; self.mark('pilot_rearm_altctl')
        if A.takeover_at is not None and t > A.takeover_at and self.nav == OFFBOARD and self.pilot == 'auto':
            self.nav, self.pilot = POSCTL, 'landing'; self.mark('pilot_takeover_sticks', alt=round(-self.p[2], 3))
        # offboard heartbeat loss
        if self.nav == OFFBOARD and (self.ocm_t is None or t - self.ocm_t > 1.0):
            self.nav = POSCTL; self.mark('offboard_lost_heartbeat')
        # dynamics
        a_cmd = -2.0 * self.v
        if self.armed and self.nav == OFFBOARD and self.sp is not None and all(math.isfinite(c) for c in self.sp.position):
            err = np.array(self.sp.position, float) - self.p
            vd = np.array([0.95 * err[0], 0.95 * err[1], 1.0 * err[2]])
            h = np.linalg.norm(vd[:2]); vd[:2] *= min(1.0, 12.0 / h) if h > 0 else 1.0
            vd[2] = float(np.clip(vd[2], -3.0, 1.5)); a_cmd = (vd - self.v) / 0.25
            if math.isfinite(self.sp.yaw): self.yaw_ned = float(self.sp.yaw)
        elif self.armed and self.pilot == 'landing':
            a_cmd = (np.array([0.0, 0.0, 0.5]) - self.v) / 0.25
        an = np.linalg.norm(a_cmd); a_cmd *= min(1.0, 3.0 / an) if an > 0 else 1.0
        if not self.armed: a_cmd = -self.v / dt
        self.v = self.v + a_cmd * dt; self.p = self.p + self.v * dt
        if self.p[2] > 0.0: self.p[2] = 0.0; self.v[2] = min(self.v[2], 0.0)
        alt = -self.p[2]; self.max_alt = max(self.max_alt, alt)
        was = self.landed
        self.landed = alt < 0.02 and abs(self.v[2]) < 0.1
        if alt > 0.05 and self.armed: self.airborne_once = True
        if was and not self.landed: self.mark('takeoff', since_arm=round(t - self.arm_t, 2))
        if self.landed and not was: self.mark('touchdown')
        # disarm rules
        if self.armed and not self.airborne_once and t - self.arm_t > 10.0:
            self.armed, self.nav = False, LOITER; self.mark('disarm_COM_DISARM_PRFLT'); self.pilot = 'done'
        if self.armed and self.airborne_once and self.landed:
            self.landed_since = self.landed_since or t
            if t - self.landed_since > 2.0:
                self.armed, self.nav = False, LOITER; self.mark('auto_disarm_landed'); self.landed_since = None
                self.airborne_once = False
                if A.rearm and self.cycle == 0:
                    self.cycle, self.pilot, self.rearm_t = 1, 'rearm_wait', t + 3.0
                else:
                    self.pilot = 'done'
        else:
            self.landed_since = None

    def status_tick(self):
        us = int(self.get_clock().now().nanoseconds / 1000)
        s = VehicleStatus(); s.timestamp = us; s.nav_state = self.nav; s.arming_state = 2 if self.armed else 1
        self.pub_status.publish(s)
        l = VehicleLandDetected(); l.timestamp = us; l.landed = bool(self.landed); self.pub_land.publish(l)
        p = VehicleLocalPosition(); p.timestamp = us; p.x, p.y, p.z = (float(c) for c in self.p)
        p.vx, p.vy, p.vz = (float(c) for c in self.v); p.xy_valid = p.z_valid = p.v_xy_valid = p.v_z_valid = True
        p.heading = float(self.yaw_ned); p.dist_bottom_valid = True; p.dist_bottom = float(-self.p[2] + 0.10)
        self.pub_lpos.publish(p)


def main():
    rclpy.init(); n = FakeDrone(); t_end = time.time() + A.duration
    try:
        while time.time() < t_end and rclpy.ok():
            rclpy.spin_once(n, timeout_sec=0.01)
    finally:
        ev(n.now(), what='summary', events=n.stats['events'], max_alt=round(n.max_alt, 3), topics=T)
        n.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
