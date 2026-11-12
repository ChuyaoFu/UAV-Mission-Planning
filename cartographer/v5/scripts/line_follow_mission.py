#!/usr/bin/python3
"""Stage 1: fast floor-guideline following. SITL arming only.

Route prior (user decision, 2026-11-06): the course is a single continuous dark
floor line and the route ahead along it is free for the whole aircraft; task
obstacles are handled per stage. So this controller does not require the whole
body box to be observed free. It tracks the measured line axis (existing
stereo-supported GuidelineDetector output on /nav/guideline_hint) with a carrot
ahead on the line, velocity feed-forward along the axis and yaw along the axis so
the forward camera keeps seeing the line.

Uses only measured line segments, accepted BODY odometry and native PX4 state.
No CAD, truth, scene files or preset coordinates.
"""
import os, json, math, time, argparse
from pathlib import Path
import numpy as np
import rclpy
from px4_msgs.msg import VehicleStatus
from navigate import Navigator, install_stop_handlers, wrap
from measured_line_reference import short_forward_reference, camera_horizontal_half_fov_rad


def line_carrot(hint, body, preferred_xy, *, now, window_rad, lookahead_m, max_age_s=.5):
    """Carrot on the measured line: nearest line point + axis*lookahead.

    Returns dict(target_xy, axis, normal, lateral_m, sources) or None.
    The axis sign follows preferred_xy (camera forward before acquisition,
    previous axis afterwards), so the vehicle never reverses along the line.
    """
    try:
        stamp = float((hint or {}).get('stamp', math.nan))
        if math.isfinite(stamp) and 0 < stamp-now < .15:
            now = stamp  # image stamp slightly ahead of this node's /clock sample (slow-RTF SITL)
        r = short_forward_reference(hint, body, preferred_xy, now=now, lookahead_m=lookahead_m,
                                    max_step_m=10*lookahead_m, max_axis_turn_rad=window_rad, max_age_s=max_age_s)
    except ValueError:
        return None
    if r is None:
        return None
    axis = np.array(r['axis_xy_unit']); normal = np.array(r['normal_xy_unit'])
    anchor = np.asarray(body, float)[:2] + normal*r['lateral_reference_error_m']
    return dict(target_xy=anchor + axis*lookahead_m, axis=axis, normal=normal,
                lateral_m=float(r['lateral_reference_error_m']), sources=r['source_segment_indices'],
                stamp=r['stamp'])


class LineFollow(Navigator):
    def __init__(self, output, a):
        super().__init__(output, 'navigation_check')
        self.a = a; self.axis = None; self.events = []; self.line_log = []; self.last_line = None
        self.window = camera_horizontal_half_fov_rad(self.camera_K, self.camera_resolution[0])
        self.along = 0.

    def ev(self, name, **kw):
        self.events.append(dict(stamp=self.now(), event=name, **kw))
        self.get_logger().info(name + ' ' + json.dumps(kw, default=float)[:240])

    def check(self):
        """Base check, but a short VIO feature gap does not abort line following.

        The line measurement (same-frame camera pose) keeps lateral/heading
        corrections; PX4 must still report a fresh local position. Bounded by
        --vio-gap-s; beyond that the base abort-and-land path applies.
        """
        if self.control_alignment.fault: raise RuntimeError('Control coordinate alignment lost: '+self.control_alignment.fault)
        if self.cancelled: raise RuntimeError('Trial cancelled; controlled landing requested')
        if self.healthy():
            if getattr(self, 'gap_since', None) is not None: self.ev('vio_recovered', gap_s=self.now()-self.gap_since)
            self.gap_since = None; return
        fresh_native = self.px is not None and 0 <= self.now()-self.px_t <= self.c['max_observation_age_s']
        fresh_body = self.body is not None and 0 <= self.now()-self.odom_t < .5
        if self.phase == 'LINE_FOLLOW' and fresh_native and fresh_body:
            if getattr(self, 'gap_since', None) is None:
                self.gap_since = self.now(); self.ev('vio_gap_continue', health=(self.health or {}).get('reason'))
            if self.now()-self.gap_since <= self.a.vio_gap_s: return
            self.ev('vio_gap_exhausted', gap_s=self.now()-self.gap_since)
        super().check()

    def camera_forward_xy(self):
        return self.q.apply(self.optical_in_body.apply([0., 0., 1.]))[:2]

    def acquire(self, now):
        preferred = self.axis if self.axis is not None else self.camera_forward_xy()
        window = self.window if self.axis is None else math.radians(self.a.track_window_deg)
        return line_carrot(self.guideline, self.body, preferred, now=now, window_rad=window, lookahead_m=self.a.lookahead)

    def run(self):
        self.begin_baseline_alignment_qualification(); start = time.monotonic()
        while time.monotonic()-start < 45:
            self.pump()
            if self.cancelled: raise RuntimeError('cancelled before alignment; did not arm')
            if self.healthy() and self.map is not None and self.qualify_baseline_alignment_once(establish=True): break
        else: raise RuntimeError('No qualified stationary landed alignment; did not arm')
        self.home = self.body.copy(); self.yaw = self.q.as_euler('xyz')[2]; hold = self.home.copy()
        self.cruise_z = self.home[2] + self.a.height
        for _ in range(40): self.check(); self.send(hold, self.yaw)
        self.enter_offboard_and_arm(hold); self.run_start = self.now(); self.ev('armed', home=self.home.tolist())
        self.phase_set('TAKEOFF'); climb = hold.copy(); climb[2] = self.cruise_z; deadline = self.now()+20; rearms = 0
        while abs(self.body[2]-self.cruise_z) > .08:
            self.check()
            if not self.armed and self.body[2]-self.home[2] < .05:
                # SITL ESC-arming failure detector can disarm right after arming (ULog
                # fd_esc_arming_failure). Re-arm on the ground; no parameter change.
                rearms += 1; self.ev('disarmed_on_ground_rearm', attempt=rearms)
                if rearms > 3: raise RuntimeError('arming not retained after 3 retries')
                self.enter_offboard_and_arm(hold); deadline = self.now()+20; continue
            if not self.armed: raise RuntimeError('disarmed in the air during takeoff')
            self.send(climb, self.yaw)
            if self.now() > deadline: raise RuntimeError('takeoff timeout')
        self.ev('cruise_height', z=float(self.body[2]))
        self.phase_set('LINE_FOLLOW'); begin = self.now(); lost_since = None; last_xy = self.body[:2].copy()
        while self.now()-begin < self.a.seconds:
            self.check(); now = self.now(); c = self.acquire(now)
            if c is not None:
                if self.axis is None: self.ev('line_acquired', axis=c['axis'].tolist(), lateral_m=c['lateral_m'])
                self.axis = c['axis']; self.last_line = c; lost_since = None
                target = np.array([*c['target_xy'], self.cruise_z]); speed = self.a.speed
                if abs(c['lateral_m']) > .25: speed *= .5  # still converging onto the line
                velocity = np.array([*(self.axis*speed), 0.])
                if len(self.line_log) == 0 or now-self.line_log[-1]['t'] > .2:
                    self.line_log.append(dict(t=now, body=self.body.tolist(), target=target.tolist(), axis=self.axis.tolist(),
                                              lateral_m=c['lateral_m'], sources=c['sources'], v=self.body_velocity.tolist()))
            elif self.axis is not None and (lost_since is None or now-lost_since < self.a.coast_s):
                # Continuous-line prior: keep going along the last measured axis briefly.
                if lost_since is None: lost_since = now; self.ev('line_lost_coasting', axis=self.axis.tolist())
                target = np.array([*(self.body[:2] + self.axis*self.a.lookahead*.5), self.cruise_z])
                velocity = np.array([*(self.axis*self.a.speed*.5), 0.])
            else:
                if self.axis is not None and lost_since is not None and now-lost_since > self.a.lost_land_s:
                    self.ev('line_lost_stop', lost_s=now-lost_since); break
                if lost_since is None and self.axis is not None: lost_since = now
                self.brake(replan=False); continue
            self.brake_hold = None
            yaw = math.atan2(self.axis[1], self.axis[0])
            self.send(target, yaw, velocity=velocity)
            step = float(np.linalg.norm(self.body[:2]-last_xy)); self.along += step if step < .3 else 0.; last_xy = self.body[:2].copy()
        self.ev('line_follow_end', along_m=self.along, elapsed_s=self.now()-begin, body=self.body.tolist())
        hold = self.body.copy(); end = self.now()+1.5
        while self.now() < end: self.check(); self.send(hold, self.yaw if self.axis is None else math.atan2(self.axis[1], self.axis[0]))
        self.complete = True

    def finish(self):
        try: super().finish()
        finally:
            Path(str(self.output)+'.line.json').write_text(json.dumps(dict(args=vars(self.a), events=self.events, line=self.line_log,
                along_m=self.along, home=None if self.home is None else self.home.tolist()), indent=1, default=float))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--height', type=float, default=.5, help='cruise height above measured home (m)')
    p.add_argument('--speed', type=float, default=.8, help='feed-forward speed along the line (m/s)')
    p.add_argument('--lookahead', type=float, default=1.0)
    p.add_argument('--track-window-deg', type=float, default=35.)
    p.add_argument('--coast-s', type=float, default=1.5)
    p.add_argument('--lost-land-s', type=float, default=6.)
    p.add_argument('--seconds', type=float, default=90.)
    p.add_argument('--vio-gap-s', type=float, default=6.)
    a = p.parse_args()
    assert Path('/.dockerenv').exists() and os.environ.get('ROS_DOMAIN_ID') == '42' and os.environ.get('SITL_AUTOMATION_AUTHORIZED') == '1', 'SITL only'
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO); n = LineFollow(a.output, a); install_stop_handlers(n)
    try: n.run()
    except Exception as e: n.failure = str(e); n.get_logger().error(str(e))
    finally: n.finish(); n.destroy_node(); rclpy.shutdown()
