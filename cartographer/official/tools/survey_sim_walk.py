#!/usr/bin/env python3
"""SIM ONLY: fly a slow scripted "walk-through" (SURVEY_PROTOCOL.md stops) and record it like a flight bag,
as a stand-in for the hand-carried survey walk. Input for tools/survey_to_prior.py validation.

    python3 tools/survey_sim_walk.py --plan tools/survey_walk.replica.json --out flights/<stamp>_survey_walk \
        --ros-args -p use_sim_time:=true

The plan is in the START frame (forward/left/up from the pad, like the course prior); START = pose at arming.
Each step: {"name", "f", "l", "u", "yaw" (deg CCW from forward, null = keep), "hold" (s)}. The vehicle
moves a carrot at --speed (0.3 m/s) along straight legs and holds still at every step with hold > 0.
The path must be collision-free by construction (no obstacle avoidance here). The bag (flight-bag topics
incl. stereo-left + depth) starts >= 6 s before arming so the pad stop defines the START frame like a walk,
and ends >= 6 s after PX4 LAND reports landed.
"""
import argparse, json, math, os, signal, subprocess, sys, time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
TOPICS = ['/clock', '/tf', '/tf_static', '/fmu/out/vehicle_local_position_v1', '/fmu/out/vehicle_attitude',
          '/fmu/out/vehicle_status_v1', '/fmu/out/vehicle_land_detected', '/fmu/out/distance_sensor',
          '/fmu/out/vehicle_odometry', '/fmu/in/trajectory_setpoint', '/fmu/in/offboard_control_mode',
          '/fmu/in/vehicle_command', '/estimator/healthy', '/visual_slam/tracking/odometry',
          '/stereo/left/image', '/stereo/left/camera_info', '/camera/depth/image_raw', '/camera/depth/camera_info']


class Walk(Node):
    def __init__(self, plan, speed, yaw_rate, log):
        super().__init__('survey_sim_walk')
        from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLandDetected,
                                  VehicleLocalPosition, VehicleStatus)
        self._OCM, self._TSP, self._CMD = OffboardControlMode, TrajectorySetpoint, VehicleCommand
        self.pub_ocm = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', PX4_QOS)
        self.pub_sp = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', PX4_QOS)
        self.pub_cmd = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', PX4_QOS)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.on_lpos, PX4_QOS)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1', self.on_status, PX4_QOS)
        self.create_subscription(VehicleLandDetected, '/fmu/out/vehicle_land_detected',
                                 lambda m: setattr(self, 'landed', bool(m.landed)), PX4_QOS)
        self.create_subscription(Bool, '/estimator/healthy', lambda m: setattr(self, 'healthy', bool(m.data)), 10)
        self.sys_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
        self.plan, self.speed, self.yaw_rate, self.log = plan, speed, math.radians(yaw_rate), log
        self.pos = None; self.heading = None; self.armed = False; self.nav = None; self.landed = True; self.healthy = False
        self.home = None; self.state = 'wait'; self.t_state = None; self.k = 0; self.sp = None; self.yaw_sp = None
        self.t_ready = None; self.t_cmd = -1e9; self.done = False; self.jump = False
        self.create_timer(0.05, self.tick)

    def now(self): return self.get_clock().now().nanoseconds * 1e-9
    def us(self): return self.sys_clock.now().nanoseconds // 1000

    def on_lpos(self, m):
        if m.xy_valid and m.z_valid:
            p = np.array([m.x, m.y, m.z], float)
            if self.pos is not None and np.linalg.norm(p - self.pos) > 0.5:
                self.jump = True                                          # estimator reset / VIO divergence
            self.pos = p; self.heading = float(m.heading)

    def on_status(self, m): self.armed = m.arming_state == 2; self.nav = int(m.nav_state)

    def cmd(self, c, p1=0.0, p2=0.0):
        msg = self._CMD(); msg.timestamp = self.us(); msg.command = int(c); msg.param1 = float(p1); msg.param2 = float(p2)
        msg.target_system = msg.target_component = msg.source_system = msg.source_component = 1
        msg.from_external = True; self.pub_cmd.publish(msg)

    def to_ned(self, f, l, u):
        o, y = self.home
        return o + np.array([f * math.cos(y) + l * math.sin(y), f * math.sin(y) - l * math.cos(y), -u])

    def publish(self, p, yaw):
        o = self._OCM(); o.timestamp = self.us(); o.position = True; self.pub_ocm.publish(o)
        s = self._TSP(); s.timestamp = o.timestamp; s.position = [float(c) for c in p]
        s.velocity = [math.nan] * 3; s.acceleration = [math.nan] * 3; s.jerk = [math.nan] * 3
        s.yaw = float(yaw); s.yawspeed = math.nan; self.pub_sp.publish(s)

    def event(self, **kw):
        kw['t'] = round(self.now(), 2)
        if self.pos is not None:
            kw['pos'] = np.round(self.pos, 3).tolist()
        line = json.dumps(kw); print(line, flush=True); self.log.write(line + '\n'); self.log.flush()

    def set_state(self, s, **kw):
        self.state, self.t_state = s, self.now(); self.event(state=s, **kw)

    def tick(self):
        now = self.now()
        if now <= 0 or self.pos is None:
            return
        if self.t_state is None:
            self.t_state = now
        st = self.state
        if st == 'wait':                                                  # >= 6 s still on the pad, healthy
            if self.healthy and self.landed and not self.armed:
                self.t_ready = self.t_ready or now
                if now - self.t_ready > 6.0:
                    self.home = (self.pos.copy(), self.heading); self.sp = self.pos.copy(); self.yaw_sp = self.heading
                    self.set_state('arm', home=np.round(self.pos, 3).tolist(), yaw_deg=round(math.degrees(self.heading), 2))
            else:
                self.t_ready = None
            return
        if st == 'arm':
            self.publish(self.sp, self.yaw_sp)
            if now - self.t_state > 1.0 and now - self.t_cmd > 1.0:
                self.t_cmd = now
                if self.nav != 14:
                    self.cmd(176, 1, 6)
                elif not self.armed:
                    self.cmd(400, 1)
            if self.armed and self.nav == 14:
                self.set_state('fly'); self.k = 0; self.leg = None
            if now - self.t_state > 20.0:
                self.set_state('abort', why='arm_timeout'); self.done = True
            return
        if st == 'fly':
            # 2026-11-07 walk: VIO drifted 0.34 m, the "touch" goal 0.25 m from the net pushed the vehicle into the
            # net, it crashed and PX4 fell back to ALTCTL while this script kept waiting. Stop on any sign of that.
            why = ('offboard_lost' if self.nav != 14 else 'disarmed' if not self.armed else
                   'pose_jump' if self.jump else None)
            if why:
                self.set_state('abort', why=why, nav=self.nav, armed=self.armed)
                if self.armed and self.nav != 18:
                    self.cmd(21)
                self.done = True
                return
            if self.leg is None:
                if self.k >= len(self.plan):
                    self.set_state('land'); return
                w = self.plan[self.k]
                goal = self.to_ned(w['f'], w['l'], w['u'])
                yaw_t = self.yaw_sp if w.get('yaw') is None else self.home[1] - math.radians(w['yaw'])   # START deg CCW -> NED CW
                self.leg = dict(a=self.sp.copy(), b=goal, s=0.0, L=float(np.linalg.norm(goal - self.sp)),
                                y0=self.yaw_sp, y1=yaw_t, hold=float(w.get('hold', 0.0)), name=w.get('name', ''), t_hold=None)
            g = self.leg
            if g['s'] < g['L']:
                g['s'] = min(g['L'], g['s'] + self.speed * 0.05)
            frac = 1.0 if g['L'] < 1e-6 else g['s'] / g['L']
            self.sp = g['a'] + (g['b'] - g['a']) * frac
            dy = (g['y1'] - g['y0'] + math.pi) % (2 * math.pi) - math.pi
            step = self.yaw_rate * 0.05
            cur = (self.yaw_sp - g['y0'] + math.pi) % (2 * math.pi) - math.pi
            cur = cur + max(-step, min(step, dy - cur))
            self.yaw_sp = g['y0'] + cur
            self.publish(self.sp, self.yaw_sp)
            if g['s'] >= g['L'] and g.get('t_end') is None:
                g['t_end'] = now
            late = g.get('t_end') is not None and now - g['t_end'] > 8.0       # blocked (e.g. by the net): hold here
            if late and not g.get('late_logged'):
                g['late_logged'] = True; self.event(state='not_arrived', name=g['name'], k=self.k,
                                                    err_m=round(float(np.linalg.norm(self.pos - g['b'])), 3))
            arrived = g['s'] >= g['L'] and (np.linalg.norm(self.pos - g['b']) < 0.08 or late) and abs(dy - cur) < 0.03
            if g['hold'] <= 0:
                if g['s'] >= g['L'] and (np.linalg.norm(self.pos - g['b']) < 0.25 or late) and abs(dy - cur) < 0.2:
                    self.k += 1; self.leg = None
            elif arrived:
                if g['t_hold'] is None:
                    g['t_hold'] = now; self.event(state='hold', name=g['name'], k=self.k)
                elif now - g['t_hold'] >= g['hold']:
                    self.event(state='hold_end', name=g['name'], k=self.k); self.k += 1; self.leg = None
            if now - self.t_state > 900:
                self.set_state('land', why='walk_timeout')
            return
        if st == 'land':
            if now - self.t_cmd > 1.0 and self.nav != 18 and self.armed:
                self.t_cmd = now; self.cmd(21)
            if not self.armed and self.landed:
                self.set_state('landed')
            return
        if st == 'landed' and now - self.t_state > 6.0:
            self.done = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--speed', type=float, default=0.3); ap.add_argument('--yaw-rate', type=float, default=30.0)
    ap.add_argument('--timeout', type=float, default=900.0)
    a, ros_args = ap.parse_known_args()
    plan = json.loads(Path(a.plan).read_text())['steps']
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    (out / 'plan.json').write_text(Path(a.plan).read_text())
    bag_log = open(out / 'bag.log', 'w')
    bag = subprocess.Popen(['ros2', 'bag', 'record', '-o', str(out / 'bag'), '-s', 'mcap',
                            '--storage-preset-profile', 'zstd_fast', '--topics', *TOPICS],
                           stdout=bag_log, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(3.0)
    rclpy.init(args=[sys.argv[0]] + [x for x in ros_args if x != '--'])
    log = open(out / 'walk.log', 'w')
    n = Walk(plan, a.speed, a.yaw_rate, log)
    t0 = time.time(); rc = 0
    try:
        while rclpy.ok() and not n.done and time.time() - t0 < a.timeout:
            rclpy.spin_once(n, timeout_sec=0.05)
        if not n.done:
            rc = 1; n.event(state='timeout_land')
            t1 = time.time()
            while time.time() - t1 < 30 and n.armed:
                n.cmd(21); rclpy.spin_once(n, timeout_sec=0.5)
    except KeyboardInterrupt:
        rc = 2
    finally:
        time.sleep(2.0)
        try:
            os.killpg(bag.pid, signal.SIGINT); bag.wait(20)
        except Exception:  # noqa: BLE001
            bag.kill()
        log.close(); n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print('[survey_sim_walk] rc=%d out=%s' % (rc, out), flush=True)
    return rc


if __name__ == '__main__':
    sys.exit(main())
