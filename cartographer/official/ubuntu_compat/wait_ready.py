#!/usr/bin/env python3
"""Block until the official-compat stack is ready to fly: /estimator/healthy true for 3 s,
PX4 local position valid, vehicle disarmed and landed (preflight flag only logged).
  python3 wait_ready.py [--timeout 150]      exit 0 = ready, 1 = timeout
"""
import argparse, math, sys, time
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

Q = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
               history=HistoryPolicy.KEEP_LAST, depth=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--timeout', type=float, default=150.0); a = ap.parse_args()
    rclpy.init(); n = Node('wait_ready')
    from px4_msgs.msg import VehicleLocalPosition, VehicleStatus, VehicleLandDetected
    s = dict(h=False, h_since=None, lp=False, pf=False, armed=None, landed=False)
    def on_h(m):
        if m.data and not s['h']:
            s['h_since'] = time.monotonic()
        s['h'] = bool(m.data)
    n.create_subscription(Bool, '/estimator/healthy', on_h, 10)
    n.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1',
                          lambda m: s.update(lp=bool(m.xy_valid and m.z_valid and math.isfinite(m.heading))), Q)
    n.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1',
                          lambda m: s.update(pf=bool(m.pre_flight_checks_pass), armed=m.arming_state == 2), Q)
    n.create_subscription(VehicleLandDetected, '/fmu/out/vehicle_land_detected', lambda m: s.update(landed=bool(m.landed)), Q)
    t0 = time.monotonic(); last = 0
    while time.monotonic() - t0 < a.timeout:
        rclpy.spin_once(n, timeout_sec=0.1)
        hok = s['h'] and s['h_since'] is not None and time.monotonic() - s['h_since'] > 3.0
        # pre_flight_checks_pass is for the CURRENT mode (Hold after boot, needs a global position that a
        # GPS-denied vehicle never has); the mission arms in Offboard, which needs only local position.
        if hok and s['lp'] and s['armed'] is False and s['landed']:
            print(f'[wait_ready] READY after {time.monotonic() - t0:.1f} s', flush=True); return 0
        if time.monotonic() - last > 5:
            last = time.monotonic(); print(f'[wait_ready] {time.monotonic() - t0:.0f}s {s}', flush=True)
    print(f'[wait_ready] TIMEOUT {s}', flush=True); return 1


if __name__ == '__main__':
    sys.exit(main())
