#!/usr/bin/python3
"""Cartographer local SLAM + actual visual-inertial motion prior.
No truth/GPS input. True MSCKF update counts monitor visual validity.
"""
import copy,json,math
import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2,Range
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String,Bool,Float64MultiArray
from px4_msgs.msg import VehicleOdometry,VehicleStatus
from rclpy.qos import qos_profile_sensor_data
D=np.diag([1.,-1.,-1.])
class Fusion(Node):
 def __init__(self):
  super().__init__('carto_vio_fusion',parameter_overrides=[Parameter('use_sim_time',value=True)])
  self.depth=None;self.cad=None;self.line=None;self.vio=None;self.last_update=-1.;self.features=0;self.accepted=0;self.rejected=0;self.pose=None;self.last=None;self.reason='startup';self.lastgood=-1;self.fault=None;self.armed=False;self.arm_start=-1;self.departed=False;self.depart_start=-1;self.range=None;self.ground_range=None;self.align=None;self.origin=None;self.correction=np.zeros(3);self.bridge_std=math.inf
  self.motion=self.create_publisher(Bool,'/vio/in_motion',10)
  self.odom=self.create_publisher(Odometry,'/carto/vio_odom',20);self.ev=self.create_publisher(VehicleOdometry,'/fmu/in/vehicle_visual_odometry',10);self.pp=self.create_publisher(PoseStamped,'/carto/local_pose',10);self.hp=self.create_publisher(String,'/carto/localization_health',10)
  self.subs=[self.create_subscription(String,'/line/localization_health',lambda m:setattr(self,'line',json.loads(m.data)),10),self.create_subscription(String,'/cad/localization_health',lambda m:setattr(self,'cad',json.loads(m.data)),10),self.create_subscription(Range,'/carto/down_range',self.range_cb,10),self.create_subscription(VehicleStatus,'/fmu/out/vehicle_status_v1',self.status_cb,qos_profile_sensor_data),self.create_subscription(Odometry,'/ov_msckf/odomimu',self.vio_cb,100),self.create_subscription(Float64MultiArray,'/ov_msckf/visual_update_health',self.features_cb,20),self.create_subscription(String,'/carto/depth_health',lambda m:setattr(self,'depth',json.loads(m.data)),20),self.create_subscription(TFMessage,'/tf',self.tf,100)]
  self.create_timer(.1,self.report)
 def status_cb(self,m):
  armed=m.arming_state==VehicleStatus.ARMING_STATE_ARMED
  if armed and not self.armed:self.arm_start=self.now()
  self.armed=armed
  if not armed:self.departed=False;self.depart_start=-1

 def range_cb(self,m):
  self.range=m.range
  if not self.armed:self.ground_range=m.range
  elif self.ground_range is not None and m.range>self.ground_range+.06 and not self.departed:self.departed=True;self.depart_start=self.now()
  self.motion.publish(Bool(data=self.departed))
 def now(self):return self.get_clock().now().nanoseconds*1e-9
 def features_cb(self,m):
  if len(m.data)!=3:return
  self.features=int(m.data[2])
  if self.features>=2:self.last_update=float(m.data[1])
 def vio_cb(self,m):
  ts=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
  cov=np.asarray(m.pose.covariance).reshape(6,6)
  if -.05<self.now()-ts<.2 and np.isfinite(cov).all() and np.max(np.diag(cov)[:3])<.04:
   self.vio=m;o=copy.deepcopy(m);o.header.frame_id='vio_global';o.child_frame_id='vio_imu';self.odom.publish(o)
 def report(self):
  now=self.now();h={'stamp':now,'valid':self.reason in ['ok_geometry','ok_visual','bounded_inertial','stationary_initialized','ok_cad_geometry'] and now-self.lastgood<.2,'reason':self.reason,'accepted':self.accepted,'rejected':self.rejected,'last_good':self.lastgood,'accepted_visual_update_features':self.features,'visual_update_age':now-self.last_update,'inertial_increment_std_bound_m':self.bridge_std,'cad_supported':bool(self.cad and self.cad['supported'] and now-self.cad['stamp']<.4),'depth_rank':None if self.depth is None else self.depth['normal_rank_ratio'],'source':'continuous VIO with gated Cartographer geometry correction; no truth','configured_noise_not_carto_covariance':True}
  self.hp.publish(String(data=json.dumps(h)))
 def tf(self,m):
  for t in m.transforms:
   if t.header.frame_id!='carto_odom' or t.child_frame_id!='vio_imu':continue
   ts=t.header.stamp.sec+t.header.stamp.nanosec*1e-9;now=self.now();p=t.transform.translation;q=t.transform.rotation;p=np.array([p.x,p.y,p.z]);q=np.array([q.x,q.y,q.z,q.w]);reason='ok_geometry';h=self.depth
   vio_fresh=self.vio is not None and now-(self.vio.header.stamp.sec+self.vio.header.stamp.nanosec*1e-9)<.15
   visual_fresh=vio_fresh and now-self.last_update<.5 and np.max(np.array(self.vio.pose.covariance).reshape(6,6).diagonal()[:3])<.01
   stationary=vio_fresh and not self.departed and np.linalg.norm([self.vio.twist.twist.linear.x,self.vio.twist.twist.linear.y,self.vio.twist.twist.linear.z])<.03
   inertial_age=now-max(self.last_update,self.depart_start)
   velocity_variance=math.inf if self.vio is None else float(np.max(np.array(self.vio.twist.covariance).reshape(6,6).diagonal()[:3]))
   self.bridge_std=math.sqrt(max(.0004,velocity_variance))*max(0.,inertial_age)+.025*max(0.,inertial_age)**2
   bounded=vio_fresh and inertial_age<3.0 and self.bridge_std<.08
   line_fresh=self.line is not None and now-self.line['stamp']<.3 and self.line['lateral_supported']
   line_full=line_fresh and self.line['supported']
   if line_fresh and vio_fresh and inertial_age<3. and self.bridge_std<.12:bounded=True
   cad_supported=vio_fresh and (line_full or self.cad is not None and now-self.cad['stamp']<.40 and self.cad['supported'])
   geom=h is not None and now-h['stamp']<.2 and h['usable'] and h['normal_rank_ratio']>.015
   if not -.05<now-ts<.2:reason='stale_pose'
   elif not np.isfinite(np.r_[p,q]).all() or abs(np.linalg.norm(q)-1)>.01:reason='invalid_pose'
   elif not geom:
    reason='ok_cad_geometry' if cad_supported else ('ok_visual' if visual_fresh else ('stationary_initialized' if stationary else ('bounded_inertial' if bounded else 'unobservable')))
   if self.last:
    dt=ts-self.last[0]
    if dt<=0:continue
    if np.linalg.norm(p-self.last[1])>max(.15,3*dt):self.fault='pose_jump'
   self.last=(ts,p,q)
   if self.fault:reason=self.fault
   self.reason=reason
   if self.vio is not None:
    vp=self.vio.pose.pose.position;vq=self.vio.pose.pose.orientation;vp=np.array([vp.x,vp.y,vp.z]);vq=Rotation.from_quat([vq.x,vq.y,vq.z,vq.w])
    if self.align is None:self.align=Rotation.from_quat(q)*vq.inv();self.origin=p-self.align.apply(vp)
    prediction=self.align.apply(vp)+self.origin
    if geom and visual_fresh:
     residual=p-prediction-self.correction
     if np.linalg.norm(residual)<.2:self.correction+=residual*min(.01,.001/max(np.linalg.norm(residual),1e-8))
    p=prediction+self.correction;q=(self.align*vq).as_quat()
   if reason not in ['ok_geometry','ok_visual','bounded_inertial','stationary_initialized','ok_cad_geometry']:self.rejected+=1;continue
   self.lastgood=ts;self.accepted+=1
   v=self.vio.twist.twist.linear;vb=np.array([v.x,v.y,v.z]);vc=np.array(self.vio.twist.covariance).reshape(6,6).diagonal()[:3]
   velocity_ok=np.isfinite(vb).all() and np.isfinite(vc).all() and vc.max()<.09
   qr=Rotation.from_matrix(D@Rotation.from_quat(q).as_matrix()@D).as_quat();variance=.0016 if geom else (.0036 if visual_fresh else .0064)
   self.ev.publish(VehicleOdometry(timestamp=int(now*1e6),timestamp_sample=int(ts*1e6),pose_frame=VehicleOdometry.POSE_FRAME_FRD,position=(D@p).astype(np.float32).tolist(),q=[float(qr[3]),float(qr[0]),float(qr[1]),float(qr[2])],velocity_frame=VehicleOdometry.VELOCITY_FRAME_BODY_FRD if velocity_ok else VehicleOdometry.VELOCITY_FRAME_UNKNOWN,velocity=(D@vb).astype(np.float32).tolist() if velocity_ok else [math.nan]*3,angular_velocity=[math.nan]*3,position_variance=[variance]*3,orientation_variance=[.0025]*3,velocity_variance=np.maximum(vc,.0025).astype(np.float32).tolist() if velocity_ok else [math.nan]*3,quality=1))
   pose=PoseStamped(header=t.header);pose.pose.position.x=float(p[0]);pose.pose.position.y=float(p[1]);pose.pose.position.z=float(p[2]);pose.pose.orientation.x=float(q[0]);pose.pose.orientation.y=float(q[1]);pose.pose.orientation.z=float(q[2]);pose.pose.orientation.w=float(q[3]);self.pp.publish(pose)
rclpy.init();n=Fusion()
try:rclpy.spin(n)
finally:n.destroy_node();rclpy.shutdown()
