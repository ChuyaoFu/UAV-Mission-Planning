#!/usr/bin/python3
"""Isolated command regression: a braking anchor must not follow pose drift."""
import os,json,tempfile
from types import SimpleNamespace as S
import numpy as np,rclpy
from scipy.spatial.transform import Rotation
from px4_msgs.msg import TrajectorySetpoint,VehicleStatus
from navigate import Navigator
assert os.environ.get('ROS_DOMAIN_ID')=='43'
os.environ['USE_SIM_TIME']='0';rclpy.init()
with tempfile.TemporaryDirectory() as directory:
 n=Navigator(directory+'/result.json');commands=[];goals=[]
 sub=n.create_subscription(TrajectorySetpoint,'/fmu/in/trajectory_setpoint',lambda m:commands.append(m),10)
 from geometry_msgs.msg import PoseStamped
 goal_sub=n.create_subscription(PoseStamped,'/nav/exploration_goal',lambda m:goals.append(m),10)
 n.body=np.array([0.,0.,.5]);n.q=Rotation.identity();n.px=S(x=0.,y=0.,z=-.5,heading=0.);n.health={'reason':'isolated measured-state fixture'};n.status=S(arming_state=VehicleStatus.ARMING_STATE_DISARMED)
 try:
  n.pump(.2);n.brake();n.goals=[{'position':[.5,0.,.5]}];n.map=S(stamp=n.now());n.body_velocity=np.array([.2,0.,0.]);n.body=np.array([.15,0.,.5]);n.px.x=.15;n.brake();n.pump(.1)
  assert commands,'No braking setpoints';assert all(abs(m.position[0])<1e-5 for m in commands),'Braking anchor moved with measured drift'
  assert not goals,'Replanned while still moving'
  n.body_velocity=np.zeros(3);n.brake();assert not goals,'Replanned before regaining the anchor'
  n.body=np.array([.01,0.,.5]);n.px.x=.01;n.map.stamp=n.now();n.brake();n.pump(.1);assert goals,'Did not request a new measured-state plan after settling'
  from quadrotor_msgs.msg import PositionCommand
  old=PositionCommand();old.trajectory_id=7;n.trajectory(old);n.brake_hold=None;n.brake();n.publish_goal(np.array([.5,0.,.5]),{'source':'isolated goal transition'});n.trajectory(old);assert n.reference is None,'Old trajectory accepted after a goal change during braking'
  new=PositionCommand();new.trajectory_id=8;n.trajectory(new);assert n.reference is new,'Fresh trajectory was not accepted'
  print(json.dumps({'passed':True,'old_trajectory_rejected_across_braking_goal_change':True,'braking_anchor_fixed_despite_15cm_drift':True,'replanning_waits_for_settle':True,'vehicle_armed':False},indent=2))
 finally:n.worker.close();n.destroy_node();rclpy.shutdown()
