#!/usr/bin/python3
"""Cartographer continuous odom TF -> PX4; never forwards map corrections.
Cartographer supplies no covariance: variances below are configured measurement
noise, not claimed estimator covariance. Gate uses timing, geometry and jumps.
"""
import json,math,time
import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import VehicleOdometry
D=np.diag([1.,-1.,-1.])
class Adapter(Node):
 def __init__(self):
  super().__init__('carto_to_px4',parameter_overrides=[Parameter('use_sim_time',value=True)])
  self.health=None;self.last=None;self.start=None;self.accepted=0;self.rejected=0;self.reason='no_pose';self.lastgood=-1;self.fault=None
  self.pub=self.create_publisher(VehicleOdometry,'/fmu/in/vehicle_visual_odometry',10)
  self.posepub=self.create_publisher(PoseStamped,'/carto/local_pose',10)
  self.hp=self.create_publisher(String,'/carto/localization_health',10)
  self.subs=[self.create_subscription(TFMessage,'/tf',self.tf,100),self.create_subscription(String,'/carto/depth_health',lambda m:setattr(self,'health',json.loads(m.data)),20)]
  self.create_timer(.1,self.report)
 def report(self):
  now=self.get_clock().now().nanoseconds*1e-9
  self.hp.publish(String(data=json.dumps({'stamp':now,'valid':self.reason=='ok' and 0<=now-self.lastgood<.2,'reason':self.reason,'accepted':self.accepted,'rejected':self.rejected,'last_good':self.lastgood,'variance_source':'configured measurement noise, Cartographer exposes no covariance'})))
 def tf(self,m):
  for t in m.transforms:
   if t.header.frame_id!='carto_odom' or t.child_frame_id!='vio_imu':continue
   ts=t.header.stamp.sec+t.header.stamp.nanosec*1e-9;now=self.get_clock().now().nanoseconds*1e-9
   p=t.transform.translation;q=t.transform.rotation;p=np.array([p.x,p.y,p.z]);q=np.array([q.x,q.y,q.z,q.w])
   reason='ok';h=self.health
   if not np.isfinite(np.r_[p,q]).all() or abs(np.linalg.norm(q)-1)>.01:reason='nonfinite_pose'
   elif not -.05<now-ts<.15:reason='pose_age'
   elif not h or not -.05<ts-h['stamp']<.2:reason='depth_age'
   elif not h['usable']:reason='depth_coverage'
   elif h['normal_rank_ratio']<.015:reason='geometric_degeneracy'
   if self.last:
    dt=ts-self.last[0]
    if dt<=0:
     if dt<-.001:self.fault='clock_or_trajectory_reset'
     continue
    if np.linalg.norm(p-self.last[1])>max(.08,2*dt):reason='pose_jump'
    if (Rotation.from_quat(q)*Rotation.from_quat(self.last[2]).inv()).magnitude()>max(.15,3*dt):reason='rotation_jump'
   if reason in ('pose_jump','rotation_jump'):self.fault=reason
   if self.fault:reason=self.fault
   self.last=(ts,p,q);self.reason=reason
   if reason!='ok':self.rejected+=1;continue
   self.lastgood=ts;self.accepted+=1
   qr=Rotation.from_matrix(D@Rotation.from_quat(q).as_matrix()@D).as_quat()
   out=VehicleOdometry(timestamp=int(now*1e6),timestamp_sample=int(ts*1e6),pose_frame=VehicleOdometry.POSE_FRAME_FRD,position=(D@p).astype(np.float32).tolist(),q=[float(qr[3]),float(qr[0]),float(qr[1]),float(qr[2])],velocity_frame=VehicleOdometry.VELOCITY_FRAME_UNKNOWN,velocity=[math.nan]*3,angular_velocity=[math.nan]*3,position_variance=[.0025]*3,orientation_variance=[.0025]*3,velocity_variance=[math.nan]*3,quality=1)
   self.pub.publish(out)
   pose=PoseStamped(header=t.header);pose.pose.position.x=float(p[0]);pose.pose.position.y=float(p[1]);pose.pose.position.z=float(p[2]);pose.pose.orientation.x=float(q[0]);pose.pose.orientation.y=float(q[1]);pose.pose.orientation.z=float(q[2]);pose.pose.orientation.w=float(q[3]);self.posepub.publish(pose)
rclpy.init();n=Adapter()
try:rclpy.spin(n)
finally:print({'accepted':n.accepted,'rejected':n.rejected,'last_reason':n.reason});n.destroy_node();rclpy.shutdown()
