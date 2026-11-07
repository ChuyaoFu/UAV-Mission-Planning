#!/usr/bin/python3
"""Behavioral ROS2 regression: goals retain height, callbacks never nested-spin.
Isolated domain 43 only. No PX4 node, simulator state, or flight command.
"""
import os,signal,subprocess,time,json,math
from camera_geometry import load_rig
WIDTH,HEIGHT=load_rig()[3]
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data,QoSProfile,ReliabilityPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from traj_utils.msg import Bspline
assert os.environ.get('ROS_DOMAIN_ID')=='43'
env=dict(os.environ,USE_SIM_TIME='0');env.pop('DISABLE_VERIFIED_LAUNCH',None);log=open('/tmp/v5-planner-test.log','w');proc=subprocess.Popen(['python3',os.path.join(os.path.dirname(__file__),'ego_launch.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
rclpy.init();n=Node('planner_behavior_test');trajectories=[];sub=n.create_subscription(Bspline,'/planning/bspline',lambda m:trajectories.append(m),10);odom=n.create_publisher(Odometry,'/nav/body_odom',10);goals=n.create_publisher(PoseStamped,'/nav/exploration_goal',5);depth=n.create_publisher(Image,'/nav/measured_depth',QoSProfile(depth=2,reliability=ReliabilityPolicy.RELIABLE));pose=n.create_publisher(PoseStamped,'/nav/camera_pose',10)
def feed(z,seconds,yaw=0.,body_speed=0.):
 end=time.monotonic()+seconds
 while time.monotonic()<end:
  m=Odometry();m.header.stamp=n.get_clock().now().to_msg();m.header.frame_id='world';m.child_frame_id='base_link';m.pose.pose.position.z=z;m.pose.pose.orientation.z=math.sin(yaw/2);m.pose.pose.orientation.w=math.cos(yaw/2);m.twist.twist.linear.x=body_speed;odom.publish(m);p=PoseStamped();p.header=m.header;p.pose=m.pose.pose;pose.publish(p);d=Image();d.header=m.header;d.encoding='32FC1';d.width=WIDTH;d.height=HEIGHT;d.step=WIDTH*4;d.data=np.zeros((HEIGHT,WIDTH),dtype='<f4').tobytes();depth.publish(d)
  until=time.monotonic()+.1
  while time.monotonic()<until:rclpy.spin_once(n,timeout_sec=.01)
def goal(x,z):
 m=PoseStamped();m.header.stamp=n.get_clock().now().to_msg();m.header.frame_id='world';m.pose.position.x=x;m.pose.position.z=z;m.pose.orientation.w=1.;goals.publish(m)
try:
 feed(0.,3.);feed(.65,1.);goal(.08,.65);feed(.65,1.);assert trajectories,'Valid short observed-free trajectory not produced'
 # B-spline endpoint is the last three control points' weighted combination.
 t=trajectories[-1];z=(t.pos_pts[-3].z+4*t.pos_pts[-2].z+t.pos_pts[-1].z)/6;assert abs(z-.65)<.02,('Goal altitude changed',z)
 goal(2.,.65);feed(.65,.3);goal(-.08,.65);feed(.65,1.)
 assert n.count_publishers('/planning/bspline')==1,'Planner crashed when a new goal interrupted failed planning'
 feed(.65,.7);goal(.05,.45);feed(.65,2.5);assert any(abs((t.pos_pts[-3].z+4*t.pos_pts[-2].z+t.pos_pts[-1].z)/6-.45)<.09 for t in trajectories),'New height goal was not respected'
 feed(.65,.5,yaw=math.pi/2,body_speed=.02);before=len(trajectories);goal(.08,.65);feed(.65,1.5,yaw=math.pi/2,body_speed=.02)
 assert len(trajectories)>before,'No trajectory for rotated body-velocity regression'
 t=trajectories[before];dt=t.knots[1]-t.knots[0];derivative=np.array([t.pos_pts[2].x-t.pos_pts[0].x,t.pos_pts[2].y-t.pos_pts[0].y,t.pos_pts[2].z-t.pos_pts[0].z])/(2*dt)
 assert derivative[1]>.005 and abs(derivative[0])<derivative[1]*.3,('Body velocity not transformed to world',derivative)
 print('endpoints',[(t.pos_pts[-3].z+4*t.pos_pts[-2].z+t.pos_pts[-1].z)/6 for t in trajectories])
 print(json.dumps({'passed':True,'body_twist_transformed_to_world':True,'goal_height_preserved':True,'recovery_goal_without_nested_spin':True,'bspline_messages':len(trajectories)},indent=2))
finally:
 os.killpg(proc.pid,signal.SIGINT)
 try:proc.wait(timeout=8)
 except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=5)
 n.destroy_node();rclpy.try_shutdown();log.close()
