#!/usr/bin/python3
"""SITL ONLY: short offboard smoke test; record the actual estimator mode.

This is a diagnostic trajectory, not a race planner or obstacle-avoidance policy.
"""
import json
import math
import os
import time
import subprocess
import fcntl
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from px4_msgs.msg import VehicleLocalPosition, VehicleStatus, VehicleCommand, VehicleCommandAck, OffboardControlMode, TrajectorySetpoint

assert Path('/.dockerenv').exists() and os.environ.get('ROS_DOMAIN_ID')=='42', 'Run only in the isolated SITL container/domain'
flight_lock=open('/work/data/flight-smoke.lock','w')
fcntl.flock(flight_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
px4bin=Path('/work/src/PX4-Autopilot/build/px4_sitl_default/bin')
params={k:float(subprocess.check_output([str(px4bin/'px4-param'),'show','-q',k],text=True).strip()) for k in ['EKF2_GPS_CTRL','EKF2_EV_CTRL','MPC_LAND_SPEED']}
if os.environ.get('NAV_TEST_REQUIRE_VISION')=='1':
    assert params['EKF2_GPS_CTRL']==0 and params['EKF2_EV_CTRL']==13, params
    assert 'never published' in subprocess.check_output([str(px4bin/'px4-listener'),'sensor_gps','-n','1'],text=True)
print('SITL estimator parameters:',params,flush=True)
rclpy.init()
node = Node('sim_baseline_flight', parameter_overrides=[Parameter('use_sim_time', value=True)])
state = {'pos': None, 'status': None, 'ack': [],'health':None}
subs = [node.create_subscription(String,'/carto/localization_health',lambda m:state.update(health=json.loads(m.data)),10),node.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', lambda m: state.update(pos=m), qos_profile_sensor_data),
        node.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1', lambda m: state.update(status=m), qos_profile_sensor_data),
        node.create_subscription(VehicleCommandAck, '/fmu/out/vehicle_command_ack', lambda m: state['ack'].append([m.command,m.result]), qos_profile_sensor_data)]
mode = node.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
traj = node.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
cmd = node.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)
def stamp():
    return node.get_clock().now().nanoseconds//1000
def command(c, p1=0., p2=0.):
    m=VehicleCommand(timestamp=stamp(),command=c,param1=float(p1),param2=float(p2),target_system=1,target_component=1,source_system=250,source_component=1,from_external=True)
    cmd.publish(m)
def tick(p):
    mode.publish(OffboardControlMode(timestamp=stamp(),position=True))
    m=TrajectorySetpoint(timestamp=stamp(),position=[float(v) for v in p],velocity=[math.nan]*3,acceleration=[math.nan]*3,jerk=[math.nan]*3,yaw=initial_yaw,yawspeed=math.nan)
    traj.publish(m)
    deadline=time.monotonic()+.05
    while time.monotonic()<deadline:
        rclpy.spin_once(node,timeout_sec=min(.01,max(0,deadline-time.monotonic())))
def wait(pred, seconds, p=None):
    start=time.monotonic()
    while time.monotonic()-start<seconds:
        if p is None: rclpy.spin_once(node,timeout_sec=.05)
        else: tick(p)
        if pred(): return True
    return False
if not wait(lambda: state['pos'] is not None and state['pos'].xy_valid and state['pos'].z_valid, 40):
    raise SystemExit('No valid PX4 position; did not arm')
p=state['pos']
origin=np.array([p.x,p.y,p.z])
initial_yaw=float(p.heading)+float(os.environ.get('NAV_TEST_YAW_OFFSET','0'))
takeoff=origin+np.array([0,0,-.65])
wait(lambda: False, 3, takeoff)
command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,1,6)
command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,1)
if not wait(lambda: state['status'] is not None and state['status'].arming_state==VehicleStatus.ARMING_STATE_ARMED,8,takeoff):
    raise SystemExit('Arming denied: '+repr(state['ack']))
rows=[];reached_count=0;arm_stamp=stamp();failure=None
try:
    forward=[.4*math.cos(initial_yaw),.4*math.sin(initial_yaw),-.65] if os.environ.get('NAV_TEST_BODY_FORWARD')=='1' else [.5,0,-.65]
    for offset in [[0,0,-.65],forward,[0,0,-.65]]:
        goal=origin+offset
        reached=None
        start=time.monotonic()
        while time.monotonic()-start<25:
            tick(goal)
            p=state['pos']
            if not state['health'] or not state['health']['valid']:raise RuntimeError('Localization gate rejected: '+str(state['health']))
            if not p.xy_valid or not p.z_valid:raise RuntimeError('PX4 position became invalid')
            dist=np.linalg.norm(np.array([p.x,p.y,p.z])-goal)
            rows.append([stamp(),p.x,p.y,p.z,*goal.tolist()])
            if dist<.1:
                if reached is None: reached=time.monotonic()
                if time.monotonic()-reached>2: break
            else: reached=None
        else: raise RuntimeError('Waypoint timeout')
        print('Reached',goal,flush=True)
        reached_count+=1
except Exception as error:
    failure=str(error)
    print('ABORT:',failure,flush=True)
finally:
    land_stamp=stamp()
    command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
    landed=wait(lambda: state['status'].arming_state!=VehicleStatus.ARMING_STATE_ARMED,30)
    Path(os.environ.get('CARTO_FLIGHT_RESULT','/work/cartographer/v3/results/smoke-01.json')).write_text(json.dumps({'origin_ned':origin.tolist(),'rows':rows,'acks':state['ack'],'parameters':params,'arm_stamp_us':arm_stamp,'land_command_stamp_us':land_stamp,'end_stamp_us':stamp(),'disarmed':landed,'reached_waypoints':reached_count,'passed':landed and reached_count==3 and failure is None,'failure':failure,'scope':'short diagnostic only; never a full-course acceptance'},indent=2))
    if not landed:raise RuntimeError('Landing timeout: vehicle is still armed')
node.destroy_node()
rclpy.shutdown()

if failure:raise SystemExit(failure)
