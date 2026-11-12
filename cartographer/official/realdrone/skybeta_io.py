"""io:=skybeta adapter for node/mission_node.py: the organizers' skybeta contract on the real drone.

The mission runs in a NED-like frame that is cuVSLAM's `odom` (ENU, startup-pose aligned) permuted:
    x_ned = y_enu, y_ned = x_enu, z_ned = -z_enu,   heading = pi/2 - yaw_enu
    q_ned_frd = NED_ENU_Q * q_enu_flu * AIRCRAFT_BASELINK_Q   (skybeta frames.py / px4_odom_bridge)
so ObstacleMap points, detector outputs and setpoints all live in ONE odom-derived frame.

Inputs  /visual_slam/tracking/odometry  pose (base_link in odom) + body FLU twist, 30 Hz; a stamp-indexed history
                                         gives the pose at the exact depth-image stamp (same OAK image clock)
        /visual_slam/status              vo_state (1 = tracking)
        /skybeta/state (JSON 5 Hz)       armed, landed, nav_state, offboard_active, origin_enu, local_enu, clamped
        /fmu/out/vehicle_status_v1       nav_state / arming_state, faster than the JSON (same definition)
        /fmu/out/vehicle_land_detected   landed
        /fmu/out/vehicle_local_position  dist_bottom (+valid) -> Vehicle.tof_m (EKF terrain estimate, not raw ToF)
Outputs /skybeta/setpoint/pose only (ENU odom + the live PX4-local minus odom offset, see sky_frame_comp; yaw as a
        quaternion; all-zero quaternion = do not command yaw).
        While offboard_active is false: the CURRENT pose (keeps the interface's "setpoint in the last 1 s" Offboard
        condition true and makes a fresh Offboard start hold where the vehicle is; every pose is floored at 0.3 m).
        Velocity / acceleration feed-forward cannot be sent with a pose (the interface has no such field) -> dropped.
        Setpoints further than max_lead_m from the vehicle are pulled in (PX4 flies a pose at up to 12 m/s).
        VehicleCommands from mission.core are NEVER sent; they are logged as events.
Never publishes to /fmu/in/*.
"""
import bisect
import json
import math
from collections import deque

import numpy as np

#: what a skybeta flight records (tools/flight_run.py --io skybeta and the team13_nav launch). Reading /fmu/in/* is fine.
BAG_TOPICS = ['/tf', '/tf_static', '/oak/imu/data', '/visual_slam/tracking/odometry', '/visual_slam/status',
              '/skybeta/state', '/skybeta/setpoint/pose', '/skybeta/setpoint/twist',
              '/fmu/out/vehicle_local_position', '/fmu/out/vehicle_attitude', '/fmu/out/vehicle_status_v1',
              '/fmu/out/vehicle_land_detected', '/fmu/in/trajectory_setpoint', '/fmu/in/offboard_control_mode',
              '/fmu/in/vehicle_visual_odometry', '/oak/left/camera_info', '/oak/right/camera_info']
BAG_IMAGE_TOPICS = ['/stereo/left/image', '/stereo/left/camera_info', '/camera/depth/image_raw', '/camera/depth/camera_info']

NED_ENU_Q = (0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0)     # wxyz, == skybeta frames.NED_ENU_Q
AIRCRAFT_BASELINK_Q = (0.0, 1.0, 0.0, 0.0)                 # wxyz, == skybeta frames.AIRCRAFT_BASELINK_Q
NAV_OFFBOARD = 14
ARMING_ARMED = 2
VO_SUCCESS = 1


def quat_mul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def enu_to_ned(v):
    return (float(v[1]), float(v[0]), -float(v[2]))


ned_to_enu = enu_to_ned                                     # the permutation is its own inverse


def q_enu_flu_to_ned_frd(q_xyzw):
    x, y, z, w = (float(c) for c in q_xyzw)
    return quat_mul(quat_mul(NED_ENU_Q, (w, x, y, z)), AIRCRAFT_BASELINK_Q)


def yaw_of_wxyz(q):
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


def rot_wxyz(q):
    w, x, y, z = q; n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0; w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def odom_to_ned(p_enu, q_xyzw, v_flu):
    """cuVSLAM odometry (ENU position, ENU<-FLU quaternion xyzw, body FLU linear twist) ->
    (pos_ned, q_wxyz NED<-FRD, heading, vel_ned)."""
    pos = enu_to_ned(p_enu)
    q = q_enu_flu_to_ned_frd(q_xyzw)
    x, y, z, w = q_xyzw
    v_enu = rot_wxyz((w, x, y, z)) @ np.asarray(v_flu, float)
    return pos, q, yaw_of_wxyz(q), enu_to_ned(v_enu)


def setpoint_enu(pos_ned, yaw_ned):
    """Mission setpoint (NED-like) -> ((x, y, z) ENU, quaternion xyzw); yaw None -> all-zero quaternion."""
    p = ned_to_enu(pos_ned)
    if yaw_ned is None or not math.isfinite(float(yaw_ned)):
        return p, (0.0, 0.0, 0.0, 0.0)
    ye = wrap_pi(math.pi / 2.0 - float(yaw_ned))
    return p, (0.0, 0.0, math.sin(ye / 2.0), math.cos(ye / 2.0))


def limit_lead(sp_ned, pos_ned, max_lead_m):
    """Pull a setpoint in to at most max_lead_m from the vehicle (3-D). Returns (setpoint, clipped?)."""
    sp = np.asarray(sp_ned, float); p = np.asarray(pos_ned, float); d = sp - p; n = float(np.linalg.norm(d))
    if max_lead_m <= 0 or n <= max_lead_m:
        return sp, False
    return p + d * (max_lead_m / n), True


class PoseHistory:
    """Odometry poses by image stamp (the OAK clock shared with the depth images)."""

    def __init__(self, n=90):
        self.t = deque(maxlen=n); self.v = deque(maxlen=n)

    def add(self, stamp, pos_ned, q_wxyz):
        if self.t and stamp <= self.t[-1]:
            return
        self.t.append(stamp); self.v.append((np.asarray(pos_ned, float), tuple(q_wxyz)))

    def at(self, stamp, tol=0.06):
        """Pose at `stamp`: linear position interpolation, nearest quaternion; None if outside the history."""
        if not self.t:
            return None
        ts = list(self.t); i = bisect.bisect_left(ts, stamp)
        if i < len(ts) and abs(ts[i] - stamp) < 1e-4:
            return self.v[i]
        if 0 < i < len(ts):
            t0, t1 = ts[i - 1], ts[i]
            a = (stamp - t0) / (t1 - t0) if t1 > t0 else 0.0
            p = self.v[i - 1][0] * (1 - a) + self.v[i][0] * a
            return p, (self.v[i - 1][1] if a < 0.5 else self.v[i][1])
        j = 0 if i == 0 else len(ts) - 1
        return self.v[j] if abs(ts[j] - stamp) <= tol else None


class SkybetaIO:
    def __init__(self, node, veh, log_event):
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from std_msgs.msg import String
        self.node, self.veh, self.log_event = node, veh, log_event
        P = lambda n, v: node.declare_parameter(n, v).value
        self.odom_topic = P('sky_odom_topic', '/visual_slam/tracking/odometry')
        self.max_lead_m = float(P('sky_max_lead_m', 1.5))
        self.odom_max_age = float(P('sky_odom_max_age_s', 0.25))
        self.vo_max_age = float(P('sky_vo_max_age_s', 1.0))
        self.status_max_age = float(P('sky_status_max_age_s', 0.6))
        self.frame_id = P('sky_frame_id', 'odom')
        self.diverge_warn_m = float(P('sky_diverge_warn_m', 0.3))
        # flight_interface (anchor_setpoints false) rotates our ENU setpoint straight into PX4's local frame, which is
        # NOT cuVSLAM odom: on skybeta-05 2026-11-08 local - odom was (0, 0, -0.37..-0.52) m and the vehicle sat ~0.4 m
        # above every setpoint. Add the live (low-passed) local - odom offset from /skybeta/state to each setpoint.
        self.frame_comp = bool(P('sky_frame_comp', True)); self.frame_off = None; self.frame_off_t = None
        self.frame_off_logged = None
        px4_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=5)
        self.PoseStamped = PoseStamped
        self.pub = node.create_publisher(PoseStamped, P('sky_setpoint_topic', '/skybeta/setpoint/pose'), 10)
        node.create_subscription(Odometry, self.odom_topic, self.on_odom, 10)
        node.create_subscription(String, '/skybeta/state', self.on_state, 10)
        try:
            from isaac_ros_visual_slam_interfaces.msg import VisualSlamStatus
            node.create_subscription(VisualSlamStatus, '/visual_slam/status', self.on_vo, 10)
            self.vo_available = True
        except ImportError:
            self.vo_available = False
            self.log_event(dict(type='sky_no_vslam_status_type', note='pose_ok from odometry freshness only'))
        topics = {'vehicle_status': '/fmu/out/vehicle_status_v1', 'vehicle_land_detected': '/fmu/out/vehicle_land_detected',
                  'vehicle_local_position': '/fmu/out/vehicle_local_position'}
        try:
            from skybeta_flight_interface.topics import resolve_endpoints
            topics.update({k: v for k, v in resolve_endpoints().items() if k in topics})
        except Exception:  # noqa: BLE001 - outside the skybeta image: keep the 1.16 spellings
            pass
        from px4_msgs.msg import VehicleLandDetected, VehicleLocalPosition, VehicleStatus
        node.create_subscription(VehicleStatus, topics['vehicle_status'], self.on_status, px4_qos)
        node.create_subscription(VehicleLandDetected, topics['vehicle_land_detected'], self.on_land, px4_qos)
        node.create_subscription(VehicleLocalPosition, topics['vehicle_local_position'], self.on_lpos, px4_qos)
        self.topics = topics
        self.hist = PoseHistory()
        self.odom_rx_t = None; self.odom_ok = False; self.vo = (None, None); self.state = {}; self.state_t = None
        self.status = None; self.status_t = None; self.land_t = None; self.lpos = None
        self.last_cmd_log = {}; self.lead_clip_t = -1e9; self.diverge_t = -1e9; self.last_off = None
        self.n_pub = 0; self.dropped_ff = False; self.last_sp = None
        self.log_event(dict(type='sky_io_up', topics=topics, odom=self.odom_topic, max_lead_m=self.max_lead_m))

    def now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ inputs
    def on_odom(self, m):
        p, q, t = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear
        vals = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        if not all(math.isfinite(float(c)) for c in vals):
            self.odom_ok = False; return
        pos, qn, hdg, vel = odom_to_ned((p.x, p.y, p.z), (q.x, q.y, q.z, q.w), (t.x, t.y, t.z))
        v = self.veh
        v.pos_ned, v.q_wxyz, v.heading = pos, qn, hdg
        v.vel_ned = vel if all(math.isfinite(c) for c in vel) else (0.0, 0.0, 0.0)
        self.hist.add(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, pos, qn)
        self.odom_rx_t = self.now(); self.odom_ok = True
        self.node.lpos_t = self.odom_rx_t                        # mission_node gates depth/tick on this

    def on_vo(self, m):
        self.vo = (self.now(), int(m.vo_state))

    def on_state(self, m):
        try:
            self.state = json.loads(m.data); self.state_t = self.now()
        except ValueError:
            return
        o, loc = self.state.get('odom_enu'), self.state.get('local_enu')
        if o and loc and all(math.isfinite(float(c)) for c in list(o) + list(loc)):
            raw = np.asarray(loc, float) - np.asarray(o, float)
            if float(np.max(np.abs(raw))) < 1.0:          # sanity: ignore a bogus PX4 local position
                self.frame_off = raw if self.frame_off is None else 0.7 * self.frame_off + 0.3 * raw
                self.frame_off_t = self.now()
        if o and loc and self.now() - self.diverge_t > 2.0:
            d = math.dist(o, loc)
            if d > self.diverge_warn_m:
                self.diverge_t = self.now()
                self.log_event(dict(type='sky_vio_ekf_diverge', d=round(d, 3), odom_enu=o, local_enu=loc))
        if self.state.get('clamped'):
            k = 'clamp:' + ','.join(self.state['clamped'])
            if self.now() - self.last_cmd_log.get(k, -1e9) > 2.0:
                self.last_cmd_log[k] = self.now(); self.log_event(dict(type='sky_clamped', why=self.state['clamped']))

    def on_status(self, m):
        self.status = (int(m.nav_state), int(m.arming_state)); self.status_t = self.now()

    def on_land(self, m):
        self.veh.landed = bool(m.landed); self.land_t = self.now()

    def on_lpos(self, m):
        self.lpos = m
        ok = bool(m.dist_bottom_valid) and math.isfinite(float(m.dist_bottom))
        self.veh.tof_m = float(m.dist_bottom) if ok else None
        self.veh.tof_t = self.now()

    # ------------------------------------------------------------------ per tick
    def update(self, now):
        """Fill the Vehicle fields that are not set directly by the callbacks."""
        v, s = self.veh, self.state
        fresh_status = self.status_t is not None and now - self.status_t < self.status_max_age
        fresh_state = self.state_t is not None and now - self.state_t < 1.0
        if fresh_status:
            v.nav_state, arm = self.status; v.armed = arm == ARMING_ARMED
            v.offboard_active = v.nav_state == NAV_OFFBOARD          # == flight_interface.offboard_active
        elif fresh_state:
            v.armed = bool(s.get('armed', False)); v.nav_state = int(s.get('nav_state', -1))
            v.offboard_active = bool(s.get('offboard_active', False))
        else:
            v.offboard_active = False
        if (self.land_t is None or now - self.land_t > 1.0) and fresh_state and s.get('landed') is not None:
            v.landed = bool(s['landed'])
        origin = s.get('origin_enu')
        if origin:
            v.clamp_origin_d = -float(origin[2])
        vo_t, vo_state = self.vo
        vo_ok = (not self.vo_available) or (vo_t is not None and now - vo_t < self.vo_max_age and vo_state == VO_SUCCESS)
        odom_fresh = self.odom_rx_t is not None and now - self.odom_rx_t < self.odom_max_age and self.odom_ok
        v.pose_ok = bool(odom_fresh and vo_ok)
        if self.last_off is not None and v.offboard_active != self.last_off:
            self.log_event(dict(type='sky_offboard', active=v.offboard_active, armed=v.armed, nav=v.nav_state))
        self.last_off = v.offboard_active
        return odom_fresh

    def pose_at(self, stamp):
        return self.hist.at(stamp)

    def publish(self, out, now, odom_fresh):
        """Send this tick's setpoint (see module doc). Returns what was sent, for logging."""
        v = self.veh
        if not odom_fresh:
            return None                                               # no pose -> no stream -> no Offboard entry
        if not v.offboard_active or out.pos_ned is None:
            if v.offboard_active:
                return None                                           # the interface keeps holding our last pose
            sp, yaw = np.asarray(v.pos_ned, float), None              # waiting / released: hold where we are
        else:
            sp, clipped = limit_lead(out.pos_ned, v.pos_ned, self.max_lead_m)
            if clipped and now - self.lead_clip_t > 1.0:
                self.lead_clip_t = now
                self.log_event(dict(type='sky_lead_clipped', asked=np.round(np.asarray(out.pos_ned, float), 3).tolist(),
                                    sent=np.round(sp, 3).tolist(), max_lead_m=self.max_lead_m))
            yaw = out.yaw
            if (out.vel_ned is not None or out.acc_ned is not None) and not self.dropped_ff:
                self.dropped_ff = True
                self.log_event(dict(type='sky_feedforward_dropped', note='pose setpoints carry no velocity/acceleration'))
        if not all(math.isfinite(float(c)) for c in sp):
            return None
        p, q = setpoint_enu(sp, yaw)
        if self.frame_comp and self.frame_off is not None and now - self.frame_off_t < 1.0:
            p = tuple(float(a + b) for a, b in zip(p, self.frame_off))
            if self.frame_off_logged is None or float(np.max(np.abs(self.frame_off - self.frame_off_logged))) > 0.05:
                self.frame_off_logged = self.frame_off.copy()
                self.log_event(dict(type='sky_frame_comp', local_minus_odom=np.round(self.frame_off, 3).tolist()))
        m = self.PoseStamped(); m.header.stamp = self.node.get_clock().now().to_msg(); m.header.frame_id = self.frame_id
        m.pose.position.x, m.pose.position.y, m.pose.position.z = p
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = q
        self.pub.publish(m); self.n_pub += 1; self.last_sp = sp
        return sp

    def log_commands(self, commands, now):
        for c in commands:
            if now - self.last_cmd_log.get(c.name, -1e9) > 1.0:
                self.last_cmd_log[c.name] = now
                self.log_event(dict(type='sky_command_not_sent', name=c.name, command=int(c.command),
                                    p1=float(c.param1), p2=float(c.param2)))

    def summary(self):
        s = self.state
        return dict(off=self.veh.offboard_active, arm=self.veh.armed, landed=self.veh.landed, tof=None if self.veh.tof_m is None
                    else round(self.veh.tof_m, 3), vo=self.vo[1], clamped=s.get('clamped'), frozen=s.get('frozen'),
                    link=s.get('px4_link'), n_pub=self.n_pub)
