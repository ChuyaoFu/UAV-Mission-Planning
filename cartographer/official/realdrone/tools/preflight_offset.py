#!/usr/bin/env python3
"""Pre-flight check: PX4 local position vs cuVSLAM odometry must agree before any team code flies.

flight_interface flies every /skybeta/setpoint/pose in PX4's local frame (anchor_setpoints false). When PX4's EKF origin
or baro height has drifted from cuVSLAM odom (stack restarted without an FC power cycle, long idle), the vehicle flies
off by that offset: skybeta-05 2026-11-08 saw 0.46 m, then 1.33-1.53 m. Exit 0 if |local - odom| <= --max-m (0.5)
on every axis (warning above --warn-m 0.15: the live compensation corrects it), 3 if larger, 4 if a topic is missing. Fix: power-cycle the flight controller, then `docker rm -f skybeta-stack`,
keep the drone still ~1-2 min.
"""
import argparse
import math
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-m', type=float, default=0.5, help='refuse above this (live compensation handles less)')
    ap.add_argument('--warn-m', type=float, default=0.15, help='warn above this')
    ap.add_argument('--timeout', type=float, default=6.0)
    a = ap.parse_args()
    import rclpy
    from nav_msgs.msg import Odometry
    from px4_msgs.msg import VehicleLocalPosition
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    rclpy.init()
    n = rclpy.create_node('team13_preflight_offset')
    S = {}
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
    n.create_subscription(Odometry, '/visual_slam/tracking/odometry',
                          lambda m: S.__setitem__('odom', (m.pose.pose.position.x, m.pose.pose.position.y,
                                                           m.pose.pose.position.z)), 10)
    n.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position',
                          lambda m: S.__setitem__('local', (m.y, m.x, -m.z)), qos)   # NED -> ENU
    t0 = time.time()
    while time.time() - t0 < a.timeout and not ('odom' in S and 'local' in S):
        rclpy.spin_once(n, timeout_sec=0.1)
    n.destroy_node(); rclpy.shutdown()
    if 'odom' not in S or 'local' not in S:
        print('[preflight] MISSING %s (is the stack up and VIO tracking?)' % [k for k in ('odom', 'local') if k not in S])
        return 4
    d = [l - o for o, l in zip(S['odom'], S['local'])]
    msg = '[preflight] VIO odom (%.2f, %.2f, %.2f)  PX4 local (%.2f, %.2f, %.2f)  local-odom (%+.2f, %+.2f, %+.2f) m' % (
        *S['odom'], *S['local'], *d)
    if max(abs(v) for v in d) > a.max_m or not all(math.isfinite(v) for v in d):
        print(msg + '\n[preflight] REFUSING: PX4 and VIO disagree by more than %.2f m -> the drone would fly at the '
              'wrong height.\n  Fix: power-cycle the flight controller (battery), then: docker rm -f skybeta-stack ; '
              'keep the drone still 1-2 min, retry.' % a.max_m)
        return 3
    if max(abs(v) for v in d) > a.warn_m:
        print(msg + '\n[preflight] WARNING: offset above %.2f m. OK to fly: our setpoints (skybeta_io, axis, wall) and '
              'team13_mission (frame_comp) add the live PX4-minus-VIO offset. Look for "frame_comp" in the log. '
              'For a clean run: power-cycle the FC + restart the stack.' % a.warn_m)
        return 0
    print(msg + '  OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
