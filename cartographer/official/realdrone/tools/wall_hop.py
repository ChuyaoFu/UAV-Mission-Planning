#!/usr/bin/env python3
"""Team 13 wall hop on the skybeta contract: over the wall, land behind it, then back over and land at the start.

Same control path as tools/axis_test.py (validated on skybeta-05 2026-11-08): ONLY /skybeta/setpoint/pose (odom ENU +
the live PX4-local minus odom offset), cuVSLAM odometry, the organizers' flight_interface (0.3 m floor, 3.0 m ceiling,
one Offboard per arming). Never /fmu/in/*.

Two legs = two armings (the course: land behind the wall, disarm, re-arm, fly back). Frame = start pose and heading at
the FIRST Offboard; place the drone FACING the wall, square to it. Do NOT restart the stack between the legs.
  leg 1: takeoff 0.8 m -> climb in place to CRUISE = wall top + margin -> (only once actually at CRUISE) fly forward to
         the landing spot -> descend to the 0.3 m floor -> "PILOT: LAND NOW, then DISARM"
  leg 2 (pilot re-arms, Altitude -> Position): takeoff -> climb in place to CRUISE -> fly back to the start -> descend
         -> "PILOT: LAND NOW".
Horizontal motion only advances while the vehicle is really within 0.2 m of CRUISE (never crosses the wall low).
Stops for good (no more setpoints, pilot has control, the interface holds the last pose) on: Offboard lost mid-leg,
odometry stale, leaving the corridor box (|left| > 0.8 m, forward < -0.8 m or > landing + 0.8 m), above CRUISE + 0.4 m,
CRUISE not reached within 20 s, or leg 2 starting more than 1.0 m from the landing spot.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

FLOOR_UP = 0.3          # flight_interface min_altitude_m
CEILING_UP = 3.0        # flight_interface max_altitude_m


def yaw_of(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


class WallHop:
    """Pure logic (no ROS). Targets in the start frame (forward, left, up), metres."""

    def __init__(self, wall_dist, wall_h, land_dist, margin=0.5, land_left=0.0, v=0.3, vz=0.3, hold=2.0,
                 wall_thick=0.15):
        self.cruise = wall_h + margin
        if self.cruise > CEILING_UP - 0.4:
            raise ValueError(f'cruise {self.cruise:.2f} m is too close to the {CEILING_UP} m ceiling')
        if land_dist < wall_dist + wall_thick + 0.6:
            raise ValueError('landing spot must be >= 0.6 m beyond the far face of the wall')
        if wall_dist < 0.8:
            raise ValueError('start at least 0.8 m from the wall')
        self.wall_dist, self.wall_h, self.land = wall_dist, wall_h, (land_dist, land_left)
        self.v, self.vz, self.hold = v, vz, hold
        self.f_min, self.f_max, self.l_max = -0.8, land_dist + 0.8, 0.8 + abs(land_left)
        C, L = self.cruise, self.land
        self.legs = [
            [('takeoff', (0, 0, 0.8), 1.0), ('climb', (0, 0, C), hold), ('cross', (L[0], L[1], C), hold),
             ('descend', (L[0], L[1], FLOOR_UP), 2.0)],
            [('takeoff', (L[0], L[1], 0.8), 1.0), ('climb', (L[0], L[1], C), hold), ('return', (0, 0, C), hold),
             ('descend', (0, 0, FLOOR_UP), 2.0)],
        ]
        self.leg, self.i, self.sp, self.t_arrive, self.t_step = 0, 0, None, None, None
        self.leg_done, self.done, self.why = False, False, ''

    def start_leg(self, pos_flu, now):
        """Called at each Offboard entry. Returns None, or a reason to refuse."""
        if self.leg == 1:
            d = math.hypot(pos_flu[0] - self.land[0], pos_flu[1] - self.land[1])
            if d > 1.0:
                return f'leg 2 starts {d:.2f} m from the landing spot (VIO drift or moved drone)'
        self.i, self.t_arrive, self.t_step, self.leg_done = 0, None, now, False
        self.sp = [pos_flu[0], pos_flu[1], max(pos_flu[2], FLOOR_UP)]
        return None

    @property
    def phase(self):
        if self.done:
            return 'done:' + self.why
        return f'leg{self.leg + 1}:' + (self.legs[self.leg][self.i][0] if not self.leg_done else 'landing')

    def _stop(self, why):
        self.done, self.why = True, why
        return None

    def step(self, dt, now, pos_flu):
        if self.done:
            return None
        f, l, u = pos_flu
        if not (self.f_min <= f <= self.f_max) or abs(l) > self.l_max:
            return self._stop(f'guard_box f {f:.2f} l {l:.2f}')
        if u > self.cruise + 0.4:
            return self._stop(f'guard_ceiling {u:.2f} m')
        if self.leg_done:
            return list(self.sp)
        name, tgt, hold = self.legs[self.leg][self.i]
        sp = self.sp
        vz = 0.6 if name == 'takeoff' else self.vz
        sp[2] += max(-vz * dt, min(vz * dt, tgt[2] - sp[2]))
        # horizontal progress only while the vehicle is really at cruise height (never cross the wall low)
        if name in ('cross', 'return') and u >= self.cruise - 0.2:
            dx, dy = tgt[0] - sp[0], tgt[1] - sp[1]
            n = math.hypot(dx, dy)
            k = 1.0 if n <= self.v * dt else self.v * dt / n
            sp[0] += dx * k; sp[1] += dy * k
        reached = all(abs(tgt[k] - sp[k]) < 1e-6 for k in range(3))
        if name == 'climb':
            reached = reached and u >= self.cruise - 0.2
            if not reached and now - self.t_step > 20.0:
                return self._stop(f'climb_not_reached {u:.2f} < {self.cruise - 0.2:.2f} m')
        if reached and self.t_arrive is None:
            self.t_arrive = now
        if reached and now - self.t_arrive >= hold:
            if name == 'descend':
                self.leg_done = True
            else:
                self.i += 1; self.t_arrive = None; self.t_step = now
        return list(sp)

    def leg_finished(self):
        """The pilot landed / disarmed after a completed leg."""
        if self.leg == 0:
            self.leg, self.leg_done = 1, False
        else:
            self.done, self.why = True, 'both_legs_complete'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wall-dist', type=float, required=True, help='start -> NEAR face of the wall, m (tape)')
    ap.add_argument('--wall-h', type=float, required=True, help='wall top above the start floor, m (barrier 68 in = 1.73)')
    ap.add_argument('--land-dist', type=float, required=True, help='start -> landing spot behind the wall, m')
    ap.add_argument('--land-left', type=float, default=0.0, help='landing spot left (+) / right (-) of the line, m')
    ap.add_argument('--margin', type=float, default=0.5, help='clearance above the wall top, m')
    ap.add_argument('--v', type=float, default=0.3, help='horizontal speed, m/s')
    ap.add_argument('--log-dir', default='')
    ap.add_argument('--check', action='store_true', help='only validate the geometry and exit')
    a = ap.parse_args()
    try:
        plan = WallHop(a.wall_dist, a.wall_h, a.land_dist, margin=a.margin, land_left=a.land_left, v=a.v)
    except ValueError as e:
        print('[wall] REJECTED: %s' % e, flush=True); return 2
    if a.check:
        print('[wall] geometry ok: cruise %.2f m, landing %.2f m fwd' % (plan.cruise, a.land_dist), flush=True); return 0

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String

    log_dir = a.log_dir
    if not log_dir:
        t_start = time.time() - 5.0
        for _ in range(40):
            c = sorted(d for d in glob.glob('/root/team_ws/team13_flights/*wall*') if os.path.getmtime(d) > t_start)
            if c:
                log_dir = c[-1]; break
            time.sleep(0.5)
    log = open(os.path.join(log_dir or '.', 'wall_hop.log'), 'a', buffering=1)

    def out(**kw):
        kw['t'] = round(time.time(), 3)
        log.write(json.dumps(kw) + '\n')

    def say(msg):
        print('[wall] ' + msg, flush=True); out(msg=msg)

    rclpy.init()
    node = rclpy.create_node('team13_wall_hop')
    pub = node.create_publisher(PoseStamped, '/skybeta/setpoint/pose', 10)
    S = dict(pos=None, yaw=None, odom_t=0.0, state={}, state_t=0.0, off=None)

    def on_odom(m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        if all(math.isfinite(v) for v in (p.x, p.y, p.z, q.x, q.y, q.z, q.w)):
            S['pos'], S['yaw'], S['odom_t'] = (p.x, p.y, p.z), yaw_of(q.x, q.y, q.z, q.w), time.time()

    def on_state(m):
        try:
            S['state'], S['state_t'] = json.loads(m.data), time.time()
        except ValueError:
            return
        o, loc = S['state'].get('odom_enu'), S['state'].get('local_enu')
        if o and loc:   # the interface flies our setpoint in PX4's local frame: add (local - odom)
            raw = [float(b) - float(c) for c, b in zip(o, loc)]
            if max(abs(v) for v in raw) < 1.0:
                S['off'] = raw if S['off'] is None else [0.7 * p + 0.3 * r for p, r in zip(S['off'], raw)]

    node.create_subscription(Odometry, '/visual_slam/tracking/odometry', on_odom, 10)
    node.create_subscription(String, '/skybeta/state', on_state, 10)

    def publish(x, y, z, yaw, now):
        ox, oy, oz = S['off'] if (S['off'] and now - S['state_t'] < 1.0) else (0.0, 0.0, 0.0)
        m = PoseStamped(); m.header.frame_id = 'odom'; m.header.stamp = node.get_clock().now().to_msg()
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x + ox, y + oy, z + oz
        m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        pub.publish(m)

    say(f'cruise {plan.cruise:.2f} m (wall top {a.wall_h:.2f} + {a.margin:.2f}), wall at {a.wall_dist:.2f} m, '
        f'landing at {a.land_dist:.2f} m fwd / {a.land_left:+.2f} m left. Drone must FACE the wall. '
        'Pilot: ALTITUDE, throttle low, ARM -> POSITION, hands off')
    out(event='start', args=vars(a), cruise=plan.cruise)
    home, yaw0, active_prev, armed_prev, last_phase, end_t = None, None, False, None, None, None
    t_prev = time.time()
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        now = time.time(); dt = min(0.2, now - t_prev); t_prev = now
        st = S['state']; fresh_state = now - S['state_t'] < 1.0
        active = bool(st.get('offboard_active')) and fresh_state
        armed = bool(st.get('armed')) if fresh_state else None
        fresh = S['pos'] is not None and now - S['odom_t'] < 0.5
        if end_t is not None:
            if now - end_t > 30.0 or now - S['state_t'] > 5.0:
                break
            continue
        if not fresh:
            if active_prev:
                say('STOP: odometry stale -> no more setpoints, PILOT TAKE OVER')
                out(event='stop', why='odom_stale'); end_t = now
            continue
        x, y, z = S['pos']
        if home is not None:
            c, s = math.cos(yaw0), math.sin(yaw0)
            dx, dy = x - home[0], y - home[1]
            pos_flu = (c * dx + s * dy, -s * dx + c * dy, z - home[2])
        # ---- not in Offboard: keep a live "hold here" setpoint for the interface; handle leg transitions
        if not active:
            if active_prev:                       # Offboard just ended
                if plan.leg_done:
                    say(f'leg {plan.leg + 1} handed to the pilot for the landing')
                else:
                    say(f'Offboard lost during {plan.phase} -> test over, PILOT HAS CONTROL')
                    out(event='stop', why='offboard_lost', phase=plan.phase); end_t = now; continue
            if plan.leg_done and armed is False:  # landed and disarmed after a completed leg
                plan.leg_finished()
                out(event='leg_finished', phase=plan.phase)
                if plan.done:
                    say('both legs complete. Done.'); end_t = now; continue
                say('leg 1 complete. Pilot: re-ARM on ALTITUDE, then POSITION -> leg 2 (back over the wall)')
            active_prev = False
            publish(x, y, z, S['yaw'], now)
            continue
        # ---- Offboard active
        if not active_prev:
            if home is None:
                home, yaw0 = (x, y, z), S['yaw']
                pos_flu = (0.0, 0.0, 0.0)
                out(event='home', home=home, yaw0=yaw0)
            refuse = plan.start_leg(pos_flu, now)
            if refuse:
                say('REFUSING leg 2: ' + refuse + ' -> no setpoints, PILOT TAKE OVER')
                out(event='stop', why='refuse_leg2', detail=refuse); end_t = now; continue
            say(f'Offboard active: leg {plan.leg + 1} from f/l/u ({pos_flu[0]:.2f}, {pos_flu[1]:.2f}, {pos_flu[2]:.2f})')
        active_prev = True
        sp = plan.step(dt, now, pos_flu)
        if plan.phase != last_phase:
            last_phase = plan.phase
            say('phase %s  pos f/l/u (%.2f, %.2f, %.2f)' % (plan.phase, *pos_flu))
            if plan.leg_done:
                say('at the 0.3 m floor. PILOT: LAND NOW (Position, throttle down), then DISARM')
        if sp is None:
            say(f'STOP ({plan.why}): no more setpoints, PILOT TAKE OVER')
            out(event='stop', why=plan.why); end_t = now; continue
        f, l, u = sp
        c, s = math.cos(yaw0), math.sin(yaw0)
        publish(home[0] + c * f - s * l, home[1] + s * f + c * l, home[2] + u, yaw0, now)
        out(phase=plan.phase, sp=[round(v, 3) for v in sp], pos=[round(v, 3) for v in pos_flu],
            off=[round(v, 3) for v in S['off']] if S['off'] else None)
    out(event='exit')
    node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
