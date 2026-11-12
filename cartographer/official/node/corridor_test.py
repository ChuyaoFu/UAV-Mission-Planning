#!/usr/bin/env python3
"""Official-lab integration test: take off, fly forward along the start heading
using a carrot with velocity feed-forward, stop on measured obstacles, land.
Native ROS 2 / px4_msgs only (same interface rules as the official refnav):
PX4 stamps on the system clock, mission logic on the node clock (sim time).
SITL use: python3 node/corridor_test.py --ros-args -p use_sim_time:=true -p goal_forward:=2.0
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.geometry import Frame, camera_pose_ned, quat_wxyz_to_mat   # noqa: E402
from node.obstacle_map import ObstacleMap                               # noqa: E402

PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
NAV_OFFBOARD, NAV_AUTO_LAND = 14, 18


class CorridorTest(Node):
    def __init__(self):
        super().__init__('corridor_test')
        P = lambda n, v: self.declare_parameter(n, v).value
        self.height = float(P('height', 0.8)); self.goal_forward = float(P('goal_forward', 2.0))
        self.goal_left = float(P('goal_left', 0.0)); self.speed = float(P('speed', 0.8))
        self.lead = float(P('lead', 0.9)); self.radius = float(P('radius', 0.28)); self.stop = float(P('stop', 0.5))
        self.timeout = float(P('timeout', 60.0))
        from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint, VehicleAttitude, VehicleCommand,
                                  VehicleLandDetected, VehicleLocalPosition, VehicleStatus)
        self._OCM, self._TSP, self._CMD = OffboardControlMode, TrajectorySetpoint, VehicleCommand
        self.pub_ocm = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', PX4_QOS)
        self.pub_sp = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', PX4_QOS)
        self.pub_cmd = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', PX4_QOS)
        self.sys_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
        self.pos = None; self.vel = np.zeros(3); self.heading = None; self.pos_ok = False; self.q = None
        self.armed = None; self.nav = None; self.landed = None; self.health = (None, False)
        self.K = None; self.R_bc = None; self.t_bc = None
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.on_lpos, PX4_QOS)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', lambda m: setattr(self, 'q', tuple(float(c) for c in m.q)), PX4_QOS)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1', self.on_status, PX4_QOS)
        self.create_subscription(VehicleLandDetected, '/fmu/out/vehicle_land_detected', lambda m: setattr(self, 'landed', bool(m.landed)), PX4_QOS)
        self.create_subscription(Bool, '/estimator/healthy', lambda m: setattr(self, 'health', (self.now(), bool(m.data))), 10)
        self.create_subscription(CameraInfo, '/camera/depth/camera_info', self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/depth/image_raw', self.on_depth, qos_profile_sensor_data)
        from tf2_ros import Buffer, TransformListener
        self.tf_buffer = Buffer(); self.tf_listener = TransformListener(self.tf_buffer, self)
        self.map = ObstacleMap(); self.floor_d = None; self.depth_t = None
        self.phase = 'wait'; self.t_phase = None; self.sp = None; self.home = None; self.goal = None; self.s = 0.
        self.t_cmd = -1e9; self.offboard_seen = False; self.land_auto = False; self.result = None; self.done_at = None
        self.offset = 0.; self.side_target = None; self.side_t = 0.; self.t_last = None; self.log_t = -1e9; self.bad_since = None; self.clear_log = None; self.t_start = None
        self.create_timer(0.05, self.tick)

    # ---------------- inputs
    def now(self): return self.get_clock().now().nanoseconds * 1e-9

    def on_lpos(self, m):
        self.pos = np.array([m.x, m.y, m.z]); self.vel = np.array([m.vx, m.vy, m.vz]); self.heading = m.heading
        self.pos_ok = bool(m.xy_valid and m.z_valid and math.isfinite(m.heading))

    def on_status(self, m):
        self.armed = m.arming_state == 2; self.nav = int(m.nav_state)

    def on_info(self, m):
        if m.k[0] > 0: self.K = (m.k[0], m.k[4], m.k[2], m.k[5])

    def on_depth(self, m):
        if self.K is None or self.q is None or self.pos is None: return
        if self.R_bc is None:
            from rclpy.time import Time
            try: tf = self.tf_buffer.lookup_transform('base_link', m.header.frame_id, Time())
            except Exception: return
            r, t = tf.transform.rotation, tf.transform.translation
            self.R_bc = quat_wxyz_to_mat(r.w, r.x, r.y, r.z); self.t_bc = (t.x, t.y, t.z)
        if m.encoding == '32FC1':
            d = np.frombuffer(bytes(m.data), np.float32).reshape(m.height, -1)[:, :m.width]
        else:
            d = np.frombuffer(bytes(m.data), np.uint16).reshape(m.height, -1)[:, :m.width].astype(np.float32) / 1000.
        R_nc, cam = camera_pose_ned(self.q, self.pos, self.R_bc, self.t_bc)
        f = Frame(image=None, depth=d, K=self.K, R_nc=R_nc, cam_ned=cam)
        floor_d = self.floor_d if self.floor_d is not None else self.pos[2] + 0.05
        self.map.insert(f, floor_d, self.now()); self.depth_t = self.now()

    # ---------------- outputs
    def us(self): return self.sys_clock.now().nanoseconds // 1000

    def command(self, cmd, p1=0., p2=0.):
        c = self._CMD(); c.timestamp = self.us(); c.command = cmd; c.param1 = float(p1); c.param2 = float(p2)
        c.target_system = c.target_component = c.source_system = c.source_component = 1; c.from_external = True
        self.pub_cmd.publish(c)

    def stream(self, vel=None):
        o = self._OCM(); o.timestamp = self.us(); o.position = True; self.pub_ocm.publish(o)
        s = self._TSP(); s.timestamp = o.timestamp; s.position = [float(c) for c in self.sp[:3]]
        s.velocity = [math.nan]*3 if vel is None else [float(vel[0]), float(vel[1]), math.nan]
        s.acceleration = [math.nan]*3; s.yaw = float(self.sp[3]); self.pub_sp.publish(s)

    def set_phase(self, p, why=''):
        self.get_logger().info(f'{self.phase} -> {p} {why}'); self.phase = p; self.t_phase = self.now()

    def land(self, why):
        if self.phase not in ('land', 'done'):
            self.result = why; self.set_phase('land', why)

    def tick(self):
        now = self.now()
        if now <= 0: return
        dt = 0. if self.t_last is None else min(max(now - self.t_last, 0.), .25); self.t_last = now
        if self.t_phase is None: self.t_phase = now; self.t_start = now
        hb_t, hb_ok = self.health
        healthy = hb_t is not None and now - hb_t < 1.0 and hb_ok
        airborne = self.phase in ('takeoff', 'go', 'hold')
        if airborne:
            if self.offboard_seen and self.nav not in (NAV_OFFBOARD,):
                self.get_logger().error(f'PX4 left Offboard (nav {self.nav}); yielding'); self.phase = 'done'; self.result = 'yielded'
            self.bad_since = None if healthy else (self.bad_since or now)
            if self.bad_since is not None and now - self.bad_since > 1.0: self.land('estimator unhealthy')
            elif self.depth_t is None or now - self.depth_t > 2.0: self.land('depth silent')
            elif now - self.t_start > self.timeout: self.land('timeout')
        vel = None
        if self.phase == 'wait':
            if self.armed is False and self.pos_ok and healthy and self.depth_t is not None and now - self.depth_t < .5:
                self.home = np.r_[self.pos, self.heading]; self.sp = list(self.home); self.floor_d = self.pos[2] + 0.05
                c, s = math.cos(self.heading), math.sin(self.heading)
                self.goal = self.home[:2] + np.array([self.goal_forward*c - self.goal_left*s, self.goal_forward*s + self.goal_left*c])
                self.set_phase('arm', f'home {np.round(self.home, 2).tolist()} goal {np.round(self.goal, 2).tolist()}')
            elif now - self.t_phase > 120: self.get_logger().error('not ready'); self.phase = 'done'; self.result = 'not ready'
        elif self.phase == 'arm':
            if self.armed and self.nav == NAV_OFFBOARD:
                self.offboard_seen = True; self.set_phase('takeoff')
            elif now - self.t_phase > 1.0 and now - self.t_cmd > 1.0:
                self.command(176, 1., 6.); self.command(400, 1.); self.t_cmd = now
            if now - self.t_phase > 30: self.phase = 'done'; self.result = 'arm timeout'
        elif self.phase == 'takeoff':
            target = self.home[2] - self.height
            self.sp[2] = max(target, self.sp[2] - 0.6*dt)
            err = abs(self.pos[2] - target); settled = abs(self.vel[2]) < 0.08 and now - self.t_phase > 5.0
            if err < 0.12 or (settled and err < 0.2): self.set_phase('go', f'z {self.pos[2]:.2f} err {err:.2f}')
        elif self.phase == 'go':
            start0 = self.home[:2]; d = self.goal - start0; L = float(np.linalg.norm(d)); u = d / max(L, 1e-9)
            nrm = np.array([-u[1], u[0]])                                  # +: right of the track (NED)
            z_c = self.home[2] - self.height
            v_al = max(float(self.vel[:2] @ u), 0.)
            stop_d = 0.3 + v_al*0.3 + v_al**2/(2*1.5)                       # reaction + braking at 1.5 m/s^2
            proj = max(float((self.pos[:2] - start0) @ u), 0.)
            win = max(0.2, min(stop_d + self.radius + 0.6, L - proj + self.radius))   # never judge beyond the goal
            tree = self.map.tree
            def margin(o):                                                  # min obstacle distance along a parallel lane
                if tree is None: return math.inf
                ss = np.linspace(0., win, 14)
                pts = np.c_[(start0 + nrm*o + u*proj)[None, :] + u[None, :]*ss[:, None], np.full(14, z_c)]
                return float(tree.query(pts, k=1)[0].min())
            best_o, best_sc, best_m = self.offset, -math.inf, margin(self.offset)
            for o in np.arange(-0.6, 0.6001, 0.05):
                if abs(o - self.offset) > 0.35: continue
                m = margin(o); sc = min(m, 1.0) - 0.15*abs(o - self.offset)
                if m >= self.radius and sc > best_sc: best_o, best_sc, best_m = o, sc, m
            if best_m < self.radius:
                self.sp[0], self.sp[1] = self.pos[0], self.pos[1]
                q = self.map.nearest_point(np.r_[self.pos[:2] + u*0.5, z_c]); rel = None if q is None else np.round(q - self.pos, 2).tolist()
                self.result_hold = 'blocked'; self.set_phase('hold', f'no lane with {self.radius:.2f} m clearance; nearest rel {rel}')
            else:
                self.offset += float(np.clip(best_o - self.offset, -0.4*dt, 0.4*dt))  # glide toward the best lane
                self.s = min(max(self.s, proj) + self.speed*dt, L, proj + self.lead)
                carrot = np.r_[start0 + nrm*self.offset + u*self.s, z_c]
                clr = self.map.clearance(self.pos, carrot, self.radius); dist = float(np.linalg.norm(carrot[:2] - self.pos[:2]))
                if clr < dist:
                    k = max(0., clr - 0.05) / max(dist, 1e-9); carrot[:2] = self.pos[:2] + (carrot[:2] - self.pos[:2]) * k
                self.sp[0], self.sp[1] = carrot[0], carrot[1]
                vel = u * self.speed if (best_m > self.radius + 0.15 and self.s < L - 0.3) else None
                if self.s >= L - 1e-6 and np.linalg.norm(self.pos[:2] - (start0 + nrm*self.offset + u*L)) < 0.15:
                    self.result_hold = 'goal reached'; self.set_phase('hold', 'goal reached')
                self.clear_log = dict(lane=round(self.offset, 2), best=round(best_o, 2), margin=round(min(best_m, 9), 2), stop=round(stop_d, 2))
        elif self.phase == 'hold':
            if now - self.t_phase > 1.5: self.land(getattr(self, 'result_hold', 'hold'))
        elif self.phase == 'land':
            if self.nav == NAV_AUTO_LAND: self.land_auto = True
            if not self.land_auto and now - self.t_cmd > 1.0: self.command(21); self.t_cmd = now
            if self.land_auto and self.armed is False: self.set_phase('done', self.result)
            elif now - self.t_phase > 60: self.set_phase('done', 'land not confirmed')
        if self.phase in ('arm', 'takeoff', 'go', 'hold') or (self.phase == 'land' and not self.land_auto):
            if self.sp is not None: self.stream(vel)
        if now - self.log_t > 0.5 and self.pos is not None:
            self.log_t = now
            print(json.dumps(dict(t=round(now, 2), phase=self.phase, pos=np.round(self.pos, 2).tolist(), v=round(float(np.linalg.norm(self.vel[:2])), 2),
                                  sp=None if self.sp is None else np.round(self.sp, 2).tolist(), clr=self.clear_log, map_pts=len(self.map.points),
                                  healthy=healthy, nav=self.nav)), flush=True)
        if self.phase == 'done':
            if self.done_at is None: self.done_at = now; self.get_logger().info(f'done: {self.result}')
            elif now - self.done_at > 1.0: raise SystemExit(0)


def main():
    rclpy.init(); n = CorridorTest()
    try: rclpy.spin(n)
    except SystemExit: pass
    finally:
        n.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == '__main__':
    main()
