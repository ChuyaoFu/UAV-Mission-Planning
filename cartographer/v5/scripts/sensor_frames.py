#!/usr/bin/python3
"""Measured odometry/body and exposure-time optical pose for the online map."""
import os,json,time,math,copy
from collections import deque
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy
from geometry_msgs.msg import PoseStamped,TransformStamped
from nav_msgs.msg import Odometry,Path as RosPath
from sensor_msgs.msg import Image
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from frames import PoseHistory
from camera_geometry import load_rig
from robot_geometry import IMU_IN_BODY,BODY_FROM_IMU,body_position,body_rotation,body_velocity

def stamp(m):return m.header.stamp.sec+m.header.stamp.nanosec*1e-9
class Frames(Node):
 def __init__(self):
  super().__init__('measured_navigation_frames',parameter_overrides=[Parameter('use_sim_time',value=os.environ.get('USE_SIM_TIME','1')=='1')]);self.history=PoseHistory();self.pending=deque(maxlen=2);self.K,self.T,_,self.res=load_rig();self.gauge=False;self.pending_slam=deque(maxlen=16)
  self.tf=StaticTransformBroadcaster(self);self.odom=self.create_publisher(Odometry,'/nav/body_odom',10);self.depth=self.create_publisher(Image,'/nav/measured_depth',2);self.pose=self.create_publisher(PoseStamped,'/nav/camera_pose',2);self.path=self.create_publisher(RosPath,'/nav/measured_path',1);self.trace=RosPath();self.last_path=-1.
  self.subs=[self.create_subscription(Odometry,'/carto/accepted_odom',self.imu,20),self.create_subscription(Image,'/stereo/depth/image',self.image,qos_profile_sensor_data),self.create_subscription(PoseStamped,'/tracked_pose',self.map_pose,5)]
 def imu(self,m):
  p=m.pose.pose.position;q=m.pose.pose.orientation;t=stamp(m);position=np.array([p.x,p.y,p.z]);rotation=Rotation.from_quat([q.x,q.y,q.z,q.w]);self.history.append(t,position,rotation.as_quat());self.drain_slam_pose();body=body_position(position,rotation);bq=body_rotation(rotation).as_quat()
  v=m.twist.twist.linear;w=m.twist.twist.angular;velocity_body=body_velocity([v.x,v.y,v.z],[w.x,w.y,w.z]);angular_body=BODY_FROM_IMU.apply([w.x,w.y,w.z]);o=Odometry();o.header=copy.deepcopy(m.header);o.header.frame_id='world';o.child_frame_id='base_link';o.pose.pose=copy.deepcopy(m.pose.pose);o.pose.pose.orientation.x=float(bq[0]);o.pose.pose.orientation.y=float(bq[1]);o.pose.pose.orientation.z=float(bq[2]);o.pose.pose.orientation.w=float(bq[3]);o.pose.pose.position.x=float(body[0]);o.pose.pose.position.y=float(body[1]);o.pose.pose.position.z=float(body[2]);o.twist.twist.linear.x=float(velocity_body[0]);o.twist.twist.linear.y=float(velocity_body[1]);o.twist.twist.linear.z=float(velocity_body[2]);o.twist.twist.angular.x=float(angular_body[0]);o.twist.twist.angular.y=float(angular_body[1]);o.twist.twist.angular.z=float(angular_body[2])
  # OpenVINS pose uncertainty is [global translation, local IMU rotation].
  # Move the pose reference and express angular errors/twist in the body frame.
  def skew(v):
   x,y,z=v;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
  r=BODY_FROM_IMU.as_matrix();jp=np.eye(6);jp[:3,3:]=rotation.as_matrix()@skew(BODY_FROM_IMU.inv().apply(IMU_IN_BODY));jp[3:,3:]=r;jv=np.zeros((6,6));jv[:3,:3]=r;jv[:3,3:]=skew(IMU_IN_BODY)@r;jv[3:,3:]=r;o.pose.covariance=(jp@np.asarray(m.pose.covariance).reshape(6,6)@jp.T).ravel().tolist();o.twist.covariance=(jv@np.asarray(m.twist.covariance).reshape(6,6)@jv.T).ravel().tolist();self.odom.publish(o)
  if t-self.last_path>.2:
   p=PoseStamped();p.header=o.header;p.pose=o.pose.pose;self.trace.header=o.header;self.trace.poses.append(p);self.trace.poses=self.trace.poses[-3000:];self.path.publish(self.trace);self.last_path=t
  self.drain()
 def image(self,m):self.pending.append(m);self.drain()
 def drain(self):
  while self.pending:
   m=self.pending[0];t=stamp(m)
   if not self.history.items or t>self.history.items[-1][0]:return
   self.pending.popleft();sample=self.history.sample(t)
   if sample is None or self.get_clock().now().nanoseconds*1e-9-t>.25:continue
   p,r=sample;camera=p+r.apply(self.T[:3,3]);rotation=r*Rotation.from_matrix(self.T[:3,:3]);pose=PoseStamped();pose.header=copy.deepcopy(m.header);pose.header.frame_id='world';pose.pose.position.x=float(camera[0]);pose.pose.position.y=float(camera[1]);pose.pose.position.z=float(camera[2]);q=rotation.as_quat();pose.pose.orientation.x=float(q[0]);pose.pose.orientation.y=float(q[1]);pose.pose.orientation.z=float(q[2]);pose.pose.orientation.w=float(q[3]);self.pose.publish(pose);self.depth.publish(m)
 def map_pose(self,m):
  if self.gauge:return
  self.pending_slam.append(m);self.drain_slam_pose()
 def drain_slam_pose(self):
  # SLAM's extrapolated pose can arrive ahead of accepted BODY history. Queue
  # its ORIGINAL stamp until an actual estimate brackets it; never restamp.
  while self.pending_slam and not self.gauge:
   m=self.pending_slam[0]
   if not self.history.items or stamp(m)>self.history.items[-1][0]:return
   self.pending_slam.popleft();s=self.history.sample(stamp(m))
   if s is None:continue
   p,r=s;q=m.pose.orientation;mp=m.pose.position;rotation=r*Rotation.from_quat([q.x,q.y,q.z,q.w]).inv();translation=p-rotation.apply([mp.x,mp.y,mp.z]);v=TransformStamped();v.header=copy.deepcopy(m.header);v.header.frame_id='world';v.child_frame_id='map';v.transform.translation.x=float(translation[0]);v.transform.translation.y=float(translation[1]);v.transform.translation.z=float(translation[2]);q=rotation.as_quat();v.transform.rotation.x=float(q[0]);v.transform.rotation.y=float(q[1]);v.transform.rotation.z=float(q[2]);v.transform.rotation.w=float(q[3]);self.tf.sendTransform(v);self.gauge=True
if __name__=='__main__':
 rclpy.init();n=Frames()
 try:rclpy.spin(n)
 finally:n.destroy_node();rclpy.shutdown()
