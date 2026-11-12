#!/usr/bin/env python3
"""Exit 0 if the vehicle (Gazebo truth /gz/gt/odometry) rests at the spawn pose (start pad centre), else 1.

  python3 at_spawn.py [--tol 0.10]

Used by fly_compat.sh --reuse: the mission's START frame is the pose at arming, so a vehicle left where an earlier
flight landed silently shifts every course prior by that offset. Sim-side only (truth never reaches flight code).
"""
import argparse, math, sys, time
import rclpy
from nav_msgs.msg import Odometry


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tol', type=float, default=0.10); ap.add_argument('--timeout', type=float, default=10.0)
    a = ap.parse_args()
    rclpy.init(); n = rclpy.create_node('at_spawn'); got = []
    n.create_subscription(Odometry, '/gz/gt/odometry', lambda m: got.append(m), 5)
    t0 = time.time()
    while not got and time.time() - t0 < a.timeout:
        rclpy.spin_once(n, timeout_sec=0.2)
    if not got:
        print('[at_spawn] no /gz/gt/odometry'); sys.exit(2)
    p, q = got[-1].pose.pose.position, got[-1].pose.pose.orientation
    yaw = math.degrees(2 * math.atan2(q.z, q.w)); d = math.hypot(p.x, p.y)
    ok = d <= a.tol and abs((yaw + 180) % 360 - 180) < 5 and p.z < 0.1
    print(f'[at_spawn] truth x {p.x:.3f} y {p.y:.3f} z {p.z:.3f} yaw {yaw:.1f} deg -> {"AT SPAWN" if ok else "NOT at spawn"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
