#!/usr/bin/env python3
"""Team 13 axis test on the skybeta contract: small, slow, one-axis-at-a-time moves to check our control chain.

Publishes ONLY /skybeta/setpoint/pose (frame odom, ENU, yaw quaternion). Never /fmu/in/*. Run next to the organizers'
flight_interface (team13.launch.py mission:=false stereo:=false), which owns Offboard entry and the 0.3-3.0 m envelope.

Sequence, all relative to the pose and heading at the moment Offboard becomes active (forward = that heading):
  takeoff to ALT -> forward +D, back -> left +D, back -> up +DZ, back -> yaw +Y, back -> yaw -Y, back
  -> 2 loops of r 0.6 m around home (fixed heading) -> hover over home 6 s (VIO drift vs the tape mark)
  -> descend to the 0.3 m floor (monotone, never rises) -> "PILOT: LAND NOW".
Roll / pitch cannot be commanded through the contract: they appear during the forward (pitch) and left (roll) moves.

Safety: setpoints are a slow time ramp (never pulled toward the vehicle, so they cannot ratchet); the test stops publishing for
good (pilot keeps control, the interface holds the last pose) when Offboard is lost, odometry is stale > 0.5 s, the
vehicle is > R_MAX m from home horizontally, or > ALT + DZ + 0.5 m above home.
"""
import argparse
import glob
import json
import math
import os
import sys
import time


def yaw_of(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class AxisPlan:
    """Pure logic (no ROS): targets in the start frame (f, l, u, yaw offset) and a ramped setpoint."""

    def __init__(self, alt=0.8, d=0.5, dz=0.3, yaw_deg=45.0, hold_s=3.0, v=0.2, vz=0.2, wz_deg=20.0,
                 floor_up=0.3, r_max=1.2, lead=0.25, circle_r=0.6, loops=2, v_circle=0.25):
        y = math.radians(yaw_deg)
        r_max = max(r_max, circle_r + 0.6)
        self.floor_up, self.alt, self.r_max, self.lead = floor_up, alt, r_max, lead
        self.circle_r, self.loops, self.v_circle, self.theta = circle_r, loops, v_circle, 0.0
        self.up_max = alt + dz + 0.5
        self.v, self.vz, self.wz = v, vz, math.radians(wz_deg)
        A = alt
        self.steps = [('takeoff', (0, 0, A, 0), 4.0), ('fwd', (d, 0, A, 0), hold_s), ('fwd_back', (0, 0, A, 0), hold_s),
                      ('left', (0, d, A, 0), hold_s), ('left_back', (0, 0, A, 0), hold_s),
                      ('up', (0, 0, A + dz, 0), hold_s), ('up_back', (0, 0, A, 0), hold_s),
                      ('yaw_pos', (0, 0, A, y), hold_s), ('yaw_pos_back', (0, 0, A, 0), hold_s),
                      ('yaw_neg', (0, 0, A, -y), hold_s), ('yaw_neg_back', (0, 0, A, 0), hold_s)]
        if circle_r > 0 and loops > 0:              # VIO check: loops around home at fixed heading, then hover
            self.steps += [('circle_in', (circle_r, 0, A, 0), 1.0), ('circle', None, 0.0),
                           ('home_check', (0, 0, A, 0), 6.0)]
        self.steps += [('descend', (0, 0, floor_up, 0), 2.0)]
        self.i, self.t_arrive, self.sp, self.done, self.why = 0, None, None, False, ''

    def start(self, up_now):
        self.sp = [0.0, 0.0, max(up_now, self.floor_up), 0.0]   # start at the interface floor, like the mission

    def step(self, dt, now, pos_flu, yaw_off):
        """pos_flu = vehicle (f, l, u) relative to home; returns (f, l, u, yaw_off) setpoint or None when finished."""
        if self.done:
            return None
        f, l, u = pos_flu
        if math.hypot(f, l) > self.r_max:
            return self._stop('guard_radius %.2f m' % math.hypot(f, l))
        if u > self.up_max:
            return self._stop('guard_ceiling %.2f m' % u)
        name, tgt, hold = self.steps[self.i]
        sp = self.sp
        if name == 'circle':                        # time-parametrised circle centred on home, heading fixed
            self.theta += self.v_circle / self.circle_r * dt
            th = min(self.theta, 2.0 * math.pi * self.loops)
            sp[0], sp[1] = self.circle_r * math.cos(th), self.circle_r * math.sin(th)
            if self.theta >= 2.0 * math.pi * self.loops:
                self.i += 1; self.t_arrive = None
            return list(sp)
        # takeoff climbs the setpoint at 0.6 m/s (0.3 -> 0.8 m in < 1 s), like the mission's first hover, so PX4 sees a
        # clear climb demand; a 0.2 m/s ramp only asked for ~0.25 m and spooled up so gently it looked underpowered
        vz = 0.6 if name == 'takeoff' else self.vz
        lim = (self.v * dt, self.v * dt, vz * dt, self.wz * dt)
        for k in range(4):
            e = tgt[k] - sp[k]
            sp[k] += max(-lim[k], min(lim[k], e))
        # The setpoint is a pure time ramp toward the target and is NEVER pulled toward the vehicle: on skybeta-05 the
        # vehicle sat ~0.4 m above its setpoint (PX4 vs VIO height offset), and any 'relative to the vehicle' rule
        # ratchets the setpoint upward (real flight 2026-11-08 climbed to 2.6 m).
        reached = all(abs(tgt[k] - sp[k]) < 1e-6 for k in range(4))
        if reached and self.t_arrive is None:
            self.t_arrive = now
        if reached and now - self.t_arrive >= hold:
            if name == 'descend':
                self.done, self.why = True, 'at_floor'
                return list(sp)
            self.i += 1; self.t_arrive = None
        return list(sp)

    def _stop(self, why):
        self.done, self.why = True, why
        return None

    @property
    def phase(self):
        return self.steps[self.i][0] if not self.done else 'done:' + self.why


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--alt', type=float, default=0.8, help='hover height above the start, m')
    ap.add_argument('--d', type=float, default=0.5, help='forward / left step, m')
    ap.add_argument('--dz', type=float, default=0.3, help='up step, m')
    ap.add_argument('--yaw', type=float, default=45.0, help='yaw step, deg')
    ap.add_argument('--hold', type=float, default=3.0, help='hold at each target, s')
    ap.add_argument('--v', type=float, default=0.2, help='ramp speed, m/s')
    ap.add_argument('--circle', type=float, default=0.6, help='VIO loop radius around home, m (0 = no loops)')
    ap.add_argument('--loops', type=int, default=2, help='number of loops')
    ap.add_argument('--vc', type=float, default=0.25, help='loop speed, m/s')
    ap.add_argument('--log-dir', default='', help='default: newest /root/team_ws/team13_flights/*axis*')
    a = ap.parse_args()

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String

    log_dir = a.log_dir
    if not log_dir:
        t_start = time.time() - 5.0
        for _ in range(40):
            c = sorted(d for d in glob.glob('/root/team_ws/team13_flights/*axis*') if os.path.getmtime(d) > t_start)
            if c:
                log_dir = c[-1]; break
            time.sleep(0.5)
    log = open(os.path.join(log_dir or '.', 'axis_test.log'), 'a', buffering=1)

    def out(**kw):
        kw['t'] = round(time.time(), 3)
        log.write(json.dumps(kw) + '\n')

    rclpy.init()
    node = rclpy.create_node('team13_axis_test')
    pub = node.create_publisher(PoseStamped, '/skybeta/setpoint/pose', 10)
    S = dict(pos=None, yaw=None, odom_t=0.0, state={}, state_t=0.0, off=None)

    def on_odom(m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        if all(math.isfinite(v) for v in (p.x, p.y, p.z, q.x, q.y, q.z, q.w)):
            S['pos'], S['yaw'], S['odom_t'] = (p.x, p.y, p.z), yaw_of(q.x, q.y, q.z, q.w), time.time()

    def on_state(m):
        try:
            S['state'], S['state_t'] = json.loads(m.data), time.time()
            o, loc = S['state'].get('odom_enu'), S['state'].get('local_enu')
            if o and loc:     # PX4 local - cuVSLAM odom: the interface sends our setpoint in PX4's frame
                raw = [float(b) - float(a) for a, b in zip(o, loc)]
                if max(abs(v) for v in raw) < 1.0:        # sanity: ignore a bogus PX4 local position
                    S['off'] = raw if S['off'] is None else [0.7 * p + 0.3 * r for p, r in zip(S['off'], raw)]
        except ValueError:
            pass

    node.create_subscription(Odometry, '/visual_slam/tracking/odometry', on_odom, 10)
    node.create_subscription(String, '/skybeta/state', on_state, 10)

    plan = AxisPlan(alt=a.alt, d=a.d, dz=a.dz, yaw_deg=a.yaw, hold_s=a.hold, v=a.v, circle_r=a.circle, loops=a.loops,
                    v_circle=a.vc)
    home, yaw0, was_active, last_phase, finished_t = None, None, False, None, None
    print('[axis] waiting for odometry + /skybeta/state; pilot: ALTITUDE, throttle low, ARM -> POSITION', flush=True)
    out(event='start', args=vars(a))
    rate_dt, t_prev = 0.05, time.time()
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=rate_dt)
        now = time.time(); dt = min(0.2, now - t_prev); t_prev = now
        fresh = S['pos'] is not None and now - S['odom_t'] < 0.5
        active = bool(S['state'].get('offboard_active')) and now - S['state_t'] < 1.0
        if finished_t is not None:
            if now - finished_t > 30.0 or now - S['state_t'] > 5.0:
                break
            continue
        if not fresh:
            if was_active:
                print('[axis] STOP: odometry stale -> no more setpoints, PILOT HAS CONTROL', flush=True)
                out(event='stop', why='odom_stale'); finished_t = now
            continue
        x, y, z = S['pos']
        if not active:
            if was_active:
                print('[axis] Offboard lost -> test over, no more setpoints (pilot has control)', flush=True)
                out(event='stop', why='offboard_lost', phase=plan.phase); finished_t = now
                continue
            m = PoseStamped(); m.header.frame_id = 'odom'; m.header.stamp = node.get_clock().now().to_msg()
            ox, oy, oz = S['off'] if (S['off'] and now - S['state_t'] < 1.0) else (0.0, 0.0, 0.0)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x + ox, y + oy, z + oz
            m.pose.orientation.z, m.pose.orientation.w = math.sin(S['yaw'] / 2), math.cos(S['yaw'] / 2)
            pub.publish(m)                      # keep the "setpoint in the last 1 s" Offboard condition true
            continue
        if not was_active:
            was_active, home, yaw0 = True, (x, y, z), S['yaw']
            plan.start(0.0)
            print('[axis] Offboard active: home (%.2f, %.2f, %.2f) yaw %.1f deg' % (x, y, z, math.degrees(yaw0)), flush=True)
            out(event='offboard', home=home, yaw0=yaw0)
        c, s = math.cos(yaw0), math.sin(yaw0)
        dx, dy = x - home[0], y - home[1]
        pos_flu = (c * dx + s * dy, -s * dx + c * dy, z - home[2])
        sp = plan.step(dt, now, pos_flu, wrap(S['yaw'] - yaw0))
        if plan.phase != last_phase:
            last_phase = plan.phase
            print('[axis] phase %s  pos f/l/u (%.2f, %.2f, %.2f) yaw %+.1f' % (plan.phase, *pos_flu,
                  math.degrees(wrap(S['yaw'] - yaw0))), flush=True)
            out(event='phase', phase=plan.phase, pos_flu=pos_flu)
            if plan.phase == 'home_check':
                print('[axis] back over HOME by VIO: look how far the drone is from the tape mark = VIO drift', flush=True)
        if sp is None:
            print('[axis] STOP (%s): no more setpoints, PILOT HAS CONTROL' % plan.why, flush=True)
            out(event='stop', why=plan.why); finished_t = now
            continue
        f, l, u, yo = sp
        m = PoseStamped(); m.header.frame_id = 'odom'; m.header.stamp = node.get_clock().now().to_msg()
        m.pose.position.x = home[0] + c * f - s * l
        m.pose.position.y = home[1] + s * f + c * l
        m.pose.position.z = home[2] + u
        ox, oy, oz = S['off'] if (S['off'] and now - S['state_t'] < 1.0) else (0.0, 0.0, 0.0)
        m.pose.position.x += ox; m.pose.position.y += oy; m.pose.position.z += oz
        yaw_sp = wrap(yaw0 + yo)
        m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw_sp / 2), math.cos(yaw_sp / 2)
        pub.publish(m)
        st = S['state']
        out(phase=plan.phase, sp_flu=[round(v, 3) for v in sp], pos_flu=[round(v, 3) for v in pos_flu],
            yaw_off=round(math.degrees(wrap(S['yaw'] - yaw0)), 1), odom_enu=st.get('odom_enu'), local_enu=st.get('local_enu'),
            off=[round(v, 3) for v in S['off']] if S['off'] else None)
        if plan.done:
            print('[axis] at the 0.3 m floor. PILOT: LAND NOW (switch to Position, throttle down, disarm)', flush=True)
            out(event='pilot_land_now'); finished_t = now
    out(event='exit')
    node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
