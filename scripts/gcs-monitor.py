#!/opt/px4-venv/bin/python3
"""SITL-only GCS heartbeat. Does not send flight commands or position data."""
import time
from pymavlink import mavutil
link=mavutil.mavlink_connection('udpout:127.0.0.1:18570',source_system=251,source_component=190)
while True:
    link.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,mavutil.mavlink.MAV_AUTOPILOT_INVALID,0,0,mavutil.mavlink.MAV_STATE_ACTIVE)
    while link.recv_match(blocking=False):pass
    time.sleep(1)
