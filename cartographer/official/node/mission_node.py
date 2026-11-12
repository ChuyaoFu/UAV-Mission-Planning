#!/usr/bin/env python3
"""ROS 2 shell for mission.core in the official lab (native px4_msgs only).

Inputs : PX4 local position/attitude/status/land-detected/distance sensor,
         /estimator/healthy, depth + CameraInfo + TF -> ObstacleMap.
Outputs: OffboardControlMode + TrajectorySetpoint (position, velocity/accel
         feed-forward, yaw) and VehicleCommand, exactly as mission.core asks.
PX4 stamps use the system clock; mission time is the node clock (sim time).
Usage: python3 node/mission_node.py --ros-args -p use_sim_time:=true -p prior:=mission/course_prior.official_sim.json

io:=skybeta (real drone, organizers' contract; realdrone/skybeta_io.py): pose from cuVSLAM odometry in the
odom-derived NED-like frame, state from /skybeta/state + /fmu/out (read only), setpoints ONLY on
/skybeta/setpoint/pose, VehicleCommands logged and never sent, Params.pilot_mode forced on unless params_json sets it.
  python3 node/mission_node.py --ros-args -p io:=skybeta -p prior:=mission/<site prior>.json
The default io:=px4 is the sim path and is unchanged.
"""
import json, math, sys, threading, time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import Bool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.geometry import Frame, camera_pose_ned, quat_wxyz_to_mat   # noqa: E402
from node.obstacle_map import ObstacleMap                               # noqa: E402
from mission.core import DETECTOR_FOR_PHASE, CoursePrior, Mission, Params, Perception, Vehicle   # noqa: E402
try:                                                                    # detectors are optional: without them the
    from perception import floor as floor_mod, tunnel as tunnel_mod, window as window_mod   # noqa: E402
except ImportError as _e:                                               # mission still flies the corridor
    floor_mod = tunnel_mod = window_mod = None; print(json.dumps(dict(event=dict(type='detectors_unavailable', err=str(_e)))), flush=True)

PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
NAN3 = [math.nan]*3


class MissionNode(Node):
    def __init__(self):
        super().__init__('official_mission')
        P = lambda n, v: self.declare_parameter(n, v).value
        prior = CoursePrior.load(str(ROOT / P('prior', 'mission/course_prior.official_sim.json')))
        params = Params()
        user_params = json.loads(P('params_json', '{}'))
        for k, v in user_params.items():                                 # e.g. '{"v_cruise": 0.8}'
            setattr(params, k, type(getattr(params, k))(v))
        self.io_mode = P('io', 'px4')                                    # 'px4' (sims, default) | 'skybeta' (real drone)
        if self.io_mode not in ('px4', 'skybeta'):
            raise ValueError(f'io must be px4 or skybeta, not {self.io_mode!r}')
        if self.io_mode == 'skybeta' and 'pilot_mode' not in user_params:
            if not hasattr(params, 'pilot_mode'):
                raise RuntimeError('io:=skybeta needs mission.core Params.pilot_mode (pilot-arm contract): update mission/')
            params.pilot_mode = True
        self.mission = Mission(prior, params)
        self.radius = params.radius
        self.sys_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
        self.veh = Vehicle(); self.lpos_t = None; self.health = (None, False)
        self.floor_below = float(P('floor_below_base_m', 0.05))         # start floor = base_link at rest + this
        if self.io_mode == 'skybeta':
            from realdrone.skybeta_io import SkybetaIO
            self.io = SkybetaIO(self, self.veh, self.event)
            self.event(dict(type='io_skybeta', pilot_mode=bool(getattr(params, 'pilot_mode', False))))
        else:
            self.io = None
            self._init_px4_io()
        self._init_perception(P, params)

    def event(self, e):
        print(json.dumps(dict(t=round(self.now(), 2), event=e), default=str), flush=True)

    def _init_px4_io(self):
        from px4_msgs.msg import (DistanceSensor, OffboardControlMode, TrajectorySetpoint, VehicleAttitude,
                                  VehicleCommand, VehicleLandDetected, VehicleLocalPosition, VehicleStatus)
        self._OCM, self._TSP, self._CMD = OffboardControlMode, TrajectorySetpoint, VehicleCommand
        self.pub_ocm = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', PX4_QOS)
        self.pub_sp = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', PX4_QOS)
        self.pub_cmd = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', PX4_QOS)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.on_lpos, PX4_QOS)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.on_att, PX4_QOS)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1', self.on_status, PX4_QOS)
        self.create_subscription(VehicleLandDetected, '/fmu/out/vehicle_land_detected', lambda m: setattr(self.veh, 'landed', bool(m.landed)), PX4_QOS)
        self.create_subscription(DistanceSensor, '/fmu/out/distance_sensor', self.on_range, PX4_QOS)
        self.create_subscription(Bool, '/estimator/healthy', lambda m: setattr(self, 'health', (self.now(), bool(m.data))), 10)

    def _init_perception(self, P, params):
        self.K = None; self.R_bc = None; self.t_bc = None
        self.create_subscription(CameraInfo, '/camera/depth/camera_info', self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/depth/image_raw', self.on_depth, qos_profile_sensor_data)
        from tf2_ros import Buffer, TransformListener
        self.tf_buffer = Buffer(); self.tf_listener = TransformListener(self.tf_buffer, self)
        self.map = ObstacleMap(min_hits=int(P('map_min_hits', 3)), carve=bool(P('map_carve', True)),
                               carve_margin_m=float(P('map_carve_margin_m', 0.25)))
        self.map_t = None; self.floor_d = None
        self.per = Perception()
        # maze planner (Params.nav_mode 'map'): the stereo node's planar scan (realdrone/depth_scan.py, -p depth_scan:=true)
        # paired with the camera pose of the depth image of the same stamp -> Perception.scan (else the core falls back
        # to a pseudo-scan from the ObstacleMap).  No publisher: nothing changes.
        self.scan_sync = None
        if bool(P('depth_scan', True)):
            try:                                                           # (needs mission/occgrid.py: cores >= aec89530)
                from node.depth_scan_sync import DepthScanSync
                self.scan_sync = DepthScanSync()
            except ImportError as e:
                self.event(dict(type='depth_scan_off', reason=str(e)))
        if self.scan_sync is not None:
            self.create_subscription(LaserScan, str(P('depth_scan_topic', '/team13/depth_scan')), self.on_scan,
                                     qos_profile_sensor_data)
        # detectors: newest (left image, depth) pair -> worker thread; the mission says which one it needs
        self.half_height = float(P('half_height_m', 0.10)); self.half_span = params.half_span_m
        self.det_period = float(P('detect_period_s', 0.15)); self.left = None; self.want = None
        self.det_frame = None; self.det_evt = threading.Event(); self.det_last_log = {}
        self.create_subscription(Image, '/stereo/left/image', self.on_left, qos_profile_sensor_data)
        threading.Thread(target=self._det_loop, daemon=True).start()
        self.last_phase = None; self.log_t = -1e9; self.done_t = None
        self.create_timer(0.05, self.tick)

    def now(self): return self.get_clock().now().nanoseconds * 1e-9
    def us(self): return self.sys_clock.now().nanoseconds // 1000

    # ---------------- inputs
    def on_lpos(self, m):
        v = self.veh; v.pos_ned = (m.x, m.y, m.z); v.vel_ned = (m.vx, m.vy, m.vz); v.heading = float(m.heading)
        self._xy_ok = bool(m.xy_valid and m.z_valid and math.isfinite(m.heading)); self.lpos_t = self.now()

    def on_att(self, m): self.veh.q_wxyz = tuple(float(c) for c in m.q)

    def on_status(self, m): self.veh.armed = m.arming_state == 2; self.veh.nav_state = int(m.nav_state)

    def on_range(self, m):
        ok = m.min_distance <= m.current_distance <= m.max_distance and m.signal_quality != 0
        self.veh.tof_m = float(m.current_distance) if ok else None; self.veh.tof_t = self.now()

    def on_info(self, m):
        if m.k[0] > 0: self.K = (m.k[0], m.k[4], m.k[2], m.k[5])

    @staticmethod
    def stamp_s(m): return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9

    def on_left(self, m):
        if m.encoding not in ('mono8', '8UC1'): return
        img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.step)[:, :m.width]
        self.left = (self.stamp_s(m), img)

    def on_depth(self, m):
        if self.K is None or self.lpos_t is None: return
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
        pos, q = np.asarray(self.veh.pos_ned, float), self.veh.q_wxyz
        if self.io is not None:                                            # skybeta: odometry pose at the image stamp
            pq = self.io.pose_at(self.stamp_s(m))
            if pq is None: return
            pos, q = np.asarray(pq[0], float), pq[1]
        if self.floor_d is None and not self.veh.armed and self.veh.landed:
            self.floor_d = pos[2] + self.floor_below                       # measured start floor (base_link at rest)
        R_nc, cam = camera_pose_ned(q, pos, self.R_bc, self.t_bc)
        if self.scan_sync is not None:
            for sc in self.scan_sync.add_pose(self.stamp_s(m), R_nc, cam):
                self._set_scan(sc)
        self.map.insert(Frame(image=None, depth=d, K=self.K, R_nc=R_nc, cam_ned=cam),
                        self.floor_d if self.floor_d is not None else pos[2] + self.floor_below, self.now())
        self.map_t = self.now()
        self.per.obstacles = self.map; self.per.obstacles_t = self.map_t
        L = self.left
        if floor_mod and self.want and L is not None and abs(L[0] - self.stamp_s(m)) < 0.04 and L[1].shape == d.shape:
            self.det_frame = (self.want, Frame(image=L[1], depth=d, K=self.K, R_nc=R_nc, cam_ned=cam, stamp=self.now()))
            self.det_evt.set()

    def on_scan(self, m):
        sc = self.scan_sync.add_scan(self.stamp_s(m), m.angle_min, m.angle_increment,
                                     np.asarray(m.ranges, np.float32), m.range_max)
        if sc is not None:
            self._set_scan(sc)

    def _set_scan(self, sc):
        if getattr(self.per, 'scan', None) is None:                      # (cores before aec89530 have no field:
            self.event(dict(type='depth_scan_on', n=int(len(sc.bearings))))   # the attribute is then just ignored)
        self.per.scan, self.per.scan_t = sc, self.now()

    def _detect(self, kind, fr):
        g = getattr(self.mission, 'g', None) or {}
        fl = floor_mod.fit_floor(fr, prior_floor_d=self.floor_d)
        if fl.found: self.per.floor, self.per.floor_t = fl, fr.stamp
        floor_d = fl.d if fl.found else self.floor_d
        if kind == 'tunnel':
            ax = g.get('tunnel_axis')
            return tunnel_mod.detect(fr, prior_entry_ned=g.get('tunnel_entry'), floor_z_ned=floor_d,
                                     prior_axis_ned=None if ax is None else np.asarray(ax, float)[:2])
        if kind == 'window':
            return window_mod.detect(fr, self.half_span, self.half_height, prior_center_ned=g.get('window_center'))
        if kind == 'landing':
            return floor_mod.landing_target(fr, fl if fl.found else None, prior_center_ned=g.get('rock'), bright_mode='soft')
        return None

    def _det_loop(self):
        last = 0.0
        while True:
            self.det_evt.wait(); self.det_evt.clear()
            item, self.det_frame = self.det_frame, None
            if item is None or time.monotonic() - last < self.det_period: continue
            kind, fr = item; last = time.monotonic(); t0 = time.perf_counter()
            try:
                est = self._detect(kind, fr)
            except Exception as e:  # noqa: BLE001 - a detector bug must not stop the flight loop
                if time.monotonic() - self.det_last_log.get('err', -1e9) > 2.0:
                    self.det_last_log['err'] = time.monotonic()
                    print(json.dumps(dict(t=round(self.now(), 2), event=dict(type='detector_error', kind=kind, err=repr(e)[:300]))), flush=True)
                continue
            if est is None: continue
            setattr(self.per, kind, est); setattr(self.per, kind + '_t', fr.stamp)
            found = bool(getattr(est, 'found', False)); key = (kind, found)
            if self.det_last_log.get('key') != key or time.monotonic() - self.det_last_log.get('t', -1e9) > 2.0:
                self.det_last_log.update(key=key, t=time.monotonic())
                c = getattr(est, 'center_ned', getattr(est, 'entry_ned', None))
                print(json.dumps(dict(t=round(fr.stamp, 2), event=dict(type='detection', kind=kind, found=found,
                      conf=getattr(est, 'confidence', None), center=None if c is None else np.round(np.asarray(c, float), 2).tolist(),
                      ms=round(1e3 * (time.perf_counter() - t0), 1), why=getattr(est, 'reason', None))), default=str), flush=True)

    def measure_wall_top(self, now, lat_m=0.5, fence_keep_m=0.3, max_raise_m=0.4):
        """Highest confirmed obstacle at the planned wall crossing -> Perception.wall_top_d (NED d).
        mission.core only ever RAISES its prior wall height with this. Guards (Ubuntu official-compat round 5,
        two aborts with the cage net read as a 2.4-2.7 m "wall"):
          - only points within `lat_m` of the crossing line, strictly between its ends;
          - points within `fence_keep_m` of a fence side are dropped (the net belongs to the geofence);
          - points higher than prior top + `max_raise_m` are dropped (prior top is measured to about 0.15 m)."""
        g = getattr(self.mission, 'g', None); sf = getattr(self.mission, 'sf', None); pts = self.map.points
        if not g or sf is None or len(pts) < 20 or now - (self.map_t or -1e9) > 1.0: return
        prior = self.mission.prior; F = prior.fence
        a = np.asarray(g['wall_from'], float)[:2]; b = np.asarray(g['wall_to'], float)[:2]
        ab = b - a; L = float(np.linalg.norm(ab))
        if L < 0.1: return
        u = ab / L; rel = pts[:, :2] - a
        s_ = rel @ u; lat = np.abs(rel @ np.array([-u[1], u[0]]))
        d0 = pts - sf.o                                                     # start frame (forward, left, up)
        f_, l_, up = d0 @ sf.fwd, d0 @ sf.left, -d0[:, 2]
        inside = ((f_ > F['forward_min'] + fence_keep_m) & (f_ < F['forward_max'] - fence_keep_m) &
                  (l_ > F['left_min'] + fence_keep_m) & (l_ < F['left_max'] - fence_keep_m))
        keep = (s_ > 0.0) & (s_ < L) & (lat < lat_m) & inside & (up <= prior.wall_top_up + max_raise_m)
        sel = pts[keep, 2]
        if len(sel) < 20: return
        top = float(np.percentile(sel, 2.0))                               # robust highest point (NED: smallest d)
        if self.per.wall_top_d is None or abs(top - self.per.wall_top_d) > 0.03:
            print(json.dumps(dict(t=round(now, 2), event=dict(type='wall_top', d=round(top, 3), n=int(len(sel)),
                                                              up=round(float(sf.o[2] - top), 3)))), flush=True)
        self.per.wall_top_d, self.per.wall_t = top, now

    # ---------------- main loop
    def command(self, c):
        msg = self._CMD(); msg.timestamp = self.us(); msg.command = int(c.command)
        msg.param1 = float(c.param1); msg.param2 = float(c.param2)
        msg.target_system = msg.target_component = msg.source_system = msg.source_component = 1
        msg.from_external = True; self.pub_cmd.publish(msg)

    def tick(self):
        now = self.now()
        if now <= 0 or self.lpos_t is None: return
        if self.io is not None:                                             # skybeta: Vehicle from odometry + state
            odom_fresh = self.io.update(now)
        else:
            hb_t, hb_ok = self.health
            self.veh.pose_ok = bool(getattr(self, '_xy_ok', False) and hb_t is not None and now - hb_t < 1.0 and hb_ok
                                    and now - self.lpos_t < 0.5)
        out = self.mission.step(now, self.veh, self.per)
        self.want = out.hints.get('detector') or DETECTOR_FOR_PHASE.get(out.phase)
        if self.want == 'wall':                                             # wall top from the depth obstacle map
            self.want = None; self.measure_wall_top(now)
        if self.io is not None:
            self.io.publish(out, now, odom_fresh)                           # /skybeta/setpoint/pose only
            self.io.log_commands(out.commands, now)                         # never sent
            for e in out.events:
                if isinstance(e, dict) and e.get('kind') == 'pilot_land_requested':
                    self.get_logger().warn(f'PILOT: LAND NOW ({e.get("why")}) -- the mission is holding at the floor clamp')
        elif out.pos_ned is not None:
            o = self._OCM(); o.timestamp = self.us(); o.position = True
            o.velocity = out.vel_ned is not None; o.acceleration = out.acc_ned is not None; self.pub_ocm.publish(o)
            s = self._TSP(); s.timestamp = o.timestamp
            s.position = [float(c) for c in out.pos_ned]
            s.velocity = NAN3 if out.vel_ned is None else [float(c) for c in out.vel_ned]
            s.acceleration = NAN3 if out.acc_ned is None else [float(c) for c in out.acc_ned]
            s.jerk = NAN3; s.yaw = math.nan if out.yaw is None else float(out.yaw); s.yawspeed = math.nan
            self.pub_sp.publish(s)
        for c in ([] if self.io is not None else out.commands):
            self.command(c)
        for e in out.events:
            print(json.dumps(dict(t=round(now, 2), event=e), default=lambda x: np.round(np.asarray(x, float), 3).tolist()), flush=True)
        if out.phase != self.last_phase:
            self.get_logger().info(f'phase {self.last_phase} -> {out.phase} {out.sub}'); self.last_phase = out.phase
        if now - self.log_t > 0.5:
            self.log_t = now
            print(json.dumps(dict(t=round(now, 2), phase=out.phase, sub=out.sub, pos=np.round(self.veh.pos_ned, 2).tolist(),
                                  v=round(float(np.linalg.norm(np.asarray(self.veh.vel_ned)[:2])), 2),
                                  sp=None if out.pos_ned is None else np.round(out.pos_ned, 2).tolist(),
                                  map_pts=len(self.map.points), pose_ok=self.veh.pose_ok, nav=self.veh.nav_state,
                                  hints={k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.hints.items()
                                         if isinstance(v, (int, float, str, bool))},
                                  **({} if self.io is None else dict(io=self.io.summary()))), default=str), flush=True)
        if out.phase in ('done', 'aborted', 'released'):
            if self.done_t is None: self.done_t = now; self.get_logger().info(f'finished: {out.phase} {out.sub}')
            elif now - self.done_t > 1.5 and not (self.io is not None and self.veh.armed):
                raise SystemExit(0)                                         # skybeta: keep streaming until disarmed


    def land_now(self, why):
        """Interrupted while possibly airborne: hand PX4 a LAND before the setpoint stream stops."""
        if not (self.veh.armed and not self.veh.landed):
            return
        if self.io is not None:                                             # skybeta has no land command: the interface
            self.event(dict(type='interrupt_no_land', why=why,              # keeps holding our last pose; pilot lands
                            note='skybeta: flight_interface holds the last pose; the PILOT must land'))
            return
        print(json.dumps(dict(t=round(self.now(), 2), event=dict(type='interrupt_land', why=why))), flush=True)
        t_end, t_cmd = time.monotonic() + 25.0, 0.0
        while time.monotonic() < t_end and not self.veh.landed:              # stay until PX4 reports landed
            if time.monotonic() - t_cmd > 1.0 and self.veh.nav_state != 18:  # 18 = AUTO_LAND
                msg = self._CMD(); msg.timestamp = self.us(); msg.command = 21  # VEHICLE_CMD_NAV_LAND
                msg.target_system = msg.target_component = msg.source_system = msg.source_component = 1
                msg.from_external = True; self.pub_cmd.publish(msg); t_cmd = time.monotonic()
            rclpy.spin_once(self, timeout_sec=0.05)
        print(json.dumps(dict(t=round(self.now(), 2), event=dict(type='interrupt_land_end', landed=self.veh.landed))), flush=True)


def main():
    import signal
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    n = MissionNode()
    stop = {'why': None}
    signal.signal(signal.SIGINT, lambda *_: stop.update(why='SIGINT'))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(why='SIGTERM'))
    try:
        while rclpy.ok() and stop['why'] is None:
            rclpy.spin_once(n, timeout_sec=0.05)
        if stop['why']:
            n.land_now(stop['why'])
    except SystemExit:
        pass
    finally:
        n.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == '__main__':
    main()
