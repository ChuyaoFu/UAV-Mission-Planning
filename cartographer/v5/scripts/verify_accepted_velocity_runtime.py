#!/usr/bin/python3
"""Domain-43 ROS regression for accepted VIO twist and exposure geometry.
Uses isolated measurements only; publishes no arm or trajectory command.
"""
import os,time,json
import numpy as np,rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import VehicleCommand
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation
from robot_geometry import BODY_FROM_IMU,IMU_IN_BODY
from fusion import Fusion
from sensor_frames import Frames
assert os.environ.get('ROS_DOMAIN_ID')=='43';os.environ['USE_SIM_TIME']='0';rclpy.init();fusion=Fusion();frames=Frames();monitor=Node('accepted_velocity_regression');executor=SingleThreadedExecutor();[executor.add_node(n) for n in [fusion,frames,monitor]];accepted=[];body=[];camera=[];depth=[];commands=[]
subs=[monitor.create_subscription(Odometry,'/carto/accepted_odom',lambda m:accepted.append(m),10),monitor.create_subscription(Odometry,'/nav/body_odom',lambda m:body.append(m),10),monitor.create_subscription(PoseStamped,'/nav/camera_pose',lambda m:camera.append(m),10),monitor.create_subscription(Image,'/nav/measured_depth',lambda m:depth.append(m),10),monitor.create_subscription(VehicleCommand,'/fmu/in/vehicle_command',lambda m:commands.append(m),10)]
def pump(seconds):
 end=time.monotonic()+seconds
 while time.monotonic()<end:executor.spin_once(timeout_sec=.005)
def state(position):
 m=Odometry();m.header.stamp=fusion.get_clock().now().to_msg();m.header.frame_id='global';m.child_frame_id='imu';r=Rotation.from_euler('z',.4);ri=r*BODY_FROM_IMU;p=np.asarray(position)+r.apply(IMU_IN_BODY);q=ri.as_quat();m.pose.pose.position.x=float(p[0]);m.pose.pose.position.y=float(p[1]);m.pose.pose.position.z=float(p[2]);m.pose.pose.orientation.x=float(q[0]);m.pose.pose.orientation.y=float(q[1]);m.pose.pose.orientation.z=float(q[2]);m.pose.pose.orientation.w=float(q[3]);w=np.array([.1,.2,.3]);vi=BODY_FROM_IMU.inv().apply(np.cross(BODY_FROM_IMU.apply(w),IMU_IN_BODY));m.twist.twist.linear.x=float(vi[0]);m.twist.twist.linear.y=float(vi[1]);m.twist.twist.linear.z=float(vi[2]);m.twist.twist.angular.x=float(w[0]);m.twist.twist.angular.y=float(w[1]);m.twist.twist.angular.z=float(w[2]);m.pose.covariance=(np.eye(6)*.001).ravel().tolist();m.twist.covariance=(np.eye(6)*.0001).ravel().tolist();return m
try:
 pump(.8);fusion.departed=True;fusion.depart_start=fusion.now()-10.;first=state([0.,0.,.5]);t1=first.header.stamp.sec+first.header.stamp.nanosec*1e-9;fusion.features_cb(Float64MultiArray(data=[t1,t1,8.]));fusion.vio_cb(first);pump(.035);second=state([.08,0.,.5]);t2=second.header.stamp.sec+second.header.stamp.nanosec*1e-9;fusion.features_cb(Float64MultiArray(data=[t2,t2,8.]));fusion.vio_cb(second);pump(.04)
 assert len(accepted)==2 and len(body)==2,'Accepted-pose path not connected'
 for m in body:
  v=m.twist.twist.linear;assert np.linalg.norm([v.x,v.y,v.z])<1e-10,'Pose reset leaked into physical velocity';assert m.child_frame_id=='base_link';assert np.linalg.eigvalsh(np.asarray(m.twist.covariance).reshape(6,6)).min()>-1e-12
 assert abs(body[-1].pose.pose.position.x-body[0].pose.pose.position.x-.08)<1e-9
 image=Image();tm=(t1+t2)/2;image.header.stamp.sec=int(tm);image.header.stamp.nanosec=round((tm-int(tm))*1e9);image.header.frame_id='left_optical';image.width=1;image.height=1;image.step=4;image.encoding='32FC1';image.data=np.array([np.nan],dtype='<f4').tobytes();frames.image(image);pump(.04);assert camera and depth and image.header.frame_id=='left_optical' and depth[-1].header.frame_id=='left_optical';assert np.isnan(np.frombuffer(depth[-1].data,dtype='<f4')[0]),'Invalid depth changed'
 before=len(accepted);pump(.025);fusion.last_update=fusion.now()-10.;stale=state([.09,0.,.5]);fusion.vio_cb(stale);pump(.04);assert fusion.reason=='unobservable';assert len(accepted)==before and len(body)==2,'Unobservable state bypassed existing acceptance gate';pump(.025);bad_velocity=state([.09,0.,.5]);tv=bad_velocity.header.stamp.sec+bad_velocity.header.stamp.nanosec*1e-9;fusion.features_cb(Float64MultiArray(data=[tv,tv,8.]));bad_velocity.twist.twist.linear.x=float('nan');fusion.vio_cb(bad_velocity);pump(.04);assert len(accepted)==before and len(body)==2,'Invalid twist published as usable navigation velocity';assert not commands
 print(json.dumps({'passed':True,'pose_reset_m':.08,'maximum_published_body_speed_mps':max(np.linalg.norm([m.twist.twist.linear.x,m.twist.twist.linear.y,m.twist.twist.linear.z]) for m in body),'calibrated_angular_lever_removed':True,'unobservable_publication_blocked':True,'invalid_velocity_publication_blocked':True,'exposure_header_preserved':True,'NaN_depth_preserved':True,'vehicle_armed':False},indent=2))
finally:
 executor.shutdown();[n.destroy_node() for n in [fusion,frames,monitor]];rclpy.shutdown()
