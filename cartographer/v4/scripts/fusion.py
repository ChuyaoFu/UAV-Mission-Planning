#!/usr/bin/python3
"""Continuous VIO control odometry, independent of sparse SLAM scan arrivals.
Cartographer local scans provide bounded corrections; map loop poses never
enter PX4. No evaluation/GPS topic is consumed. Geometry must be measured.
"""
import copy,json,math
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range,PointCloud2
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String,Bool,Float64MultiArray
from px4_msgs.msg import VehicleOdometry,VehicleStatus
from height_odometry import HeightObserver
D=np.diag([1.,-1.,-1.])
def timestamp(h):return h.stamp.sec+h.stamp.nanosec*1e-9
class Fusion(Node):
 def __init__(self):
  super().__init__('carto_vio_fusion',parameter_overrides=[Parameter('use_sim_time',value=True)]);self.height_observer=HeightObserver();self.vio=None;self.depth=None;self.mapping=None;self.cad=None;self.line=None;self.armed=False;self.departed=False;self.ground=None;self.depart_start=-1.;self.last_update=-1.;self.features=0;self.lastgood=-1.;self.last_publish=-1.;self.accepted=0;self.rejected=0;self.carto_rejected=0;self.correction=np.zeros(3);self.carto_transform=None;self.last_carto=-1.;self.last_correction=-1.;self.history=deque(maxlen=600);self.reason='startup';self.bridge_std=math.inf
  self.motion=self.create_publisher(Bool,'/vio/in_motion',10);self.odom=self.create_publisher(Odometry,'/carto/vio_odom',20);self.ev=self.create_publisher(VehicleOdometry,'/fmu/in/vehicle_visual_odometry',10);self.pp=self.create_publisher(PoseStamped,'/carto/local_pose',10);self.hp=self.create_publisher(String,'/carto/localization_health',10)
  self.subs=[self.create_subscription(PointCloud2,'/carto/points',self.height_observer.cloud,qos_profile_sensor_data),self.create_subscription(Odometry,'/ov_msckf/odomimu',self.vio_cb,100),self.create_subscription(Float64MultiArray,'/ov_msckf/visual_update_health',self.features_cb,20),self.create_subscription(Range,'/carto/down_range',self.range_cb,10),self.create_subscription(VehicleStatus,'/fmu/out/vehicle_status_v1',self.status_cb,qos_profile_sensor_data),self.create_subscription(TFMessage,'/tf',self.tf,100)]
  for topic,key in [('/carto/depth_health','depth'),('/mapping/health','mapping'),('/cad/localization_health','cad'),('/line/localization_health','line')]:self.subs.append(self.create_subscription(String,topic,lambda m,k=key:setattr(self,k,json.loads(m.data)),20))
  self.create_timer(.1,self.report)
 def now(self):return self.get_clock().now().nanoseconds*1e-9
 def status_cb(self,m):
  self.armed=m.arming_state==VehicleStatus.ARMING_STATE_ARMED
  if not self.armed:self.departed=False;self.depart_start=-1.
 def range_cb(self,m):
  self.height_observer.ranging(m)
  if not self.armed:self.ground=m.range
  elif self.ground is not None and m.range>self.ground+.06 and not self.departed:self.departed=True;self.depart_start=self.now()
  self.motion.publish(Bool(data=self.departed))
 def features_cb(self,m):
  if len(m.data)==3:
   self.features=int(m.data[2])
   if self.features>=2:self.last_update=float(m.data[1])
 def geometry(self,now):
  return self.mapping is not None and now-self.mapping['stamp']<.25 and self.mapping['structure_points']>=80 and self.depth is not None and now-self.depth['stamp']<.25 and self.depth['usable'] and self.depth['normal_rank_ratio']>.015 and now-self.last_carto<.2
 def classify(self,now,m):
  covariance=np.asarray(m.pose.covariance).reshape(6,6);visual=now-self.last_update<.5 and np.max(covariance.diagonal()[:3])<.01;age=max(0.,now-max(self.last_update,self.depart_start));velocity_variance=float(np.max(np.asarray(m.twist.covariance).reshape(6,6).diagonal()[:3]));self.bridge_std=math.sqrt(max(.0004,velocity_variance))*age+.025*age*age;bounded=age<3. and self.bridge_std<.08
  line_fresh=self.line is not None and now-self.line['stamp']<.3 and self.line['lateral_supported']
  if line_fresh and age<3. and self.bridge_std<.12:bounded=True
  cad=(line_fresh and self.line['supported']) or (self.cad is not None and now-self.cad['stamp']<.4 and self.cad['supported'])
  if self.geometry(now):return 'ok_geometry'
  if cad:return 'ok_cad_geometry'
  if visual:return 'ok_visual'
  v=m.twist.twist.linear
  if not self.departed and np.linalg.norm([v.x,v.y,v.z])<.03:return 'stationary_initialized'
  return 'bounded_inertial' if bounded else 'unobservable'
 def vio_cb(self,m):
  ts=timestamp(m.header);now=self.now();cov=np.asarray(m.pose.covariance).reshape(6,6);p=m.pose.pose.position;q=m.pose.pose.orientation;position=np.array([p.x,p.y,p.z]);quaternion=np.array([q.x,q.y,q.z,q.w])
  if not -.05<now-ts<.2 or not np.isfinite(np.r_[position,quaternion,cov.ravel()]).all() or abs(np.linalg.norm(quaternion)-1)>.01 or cov.diagonal()[:3].max()>=.04:self.reason='invalid_vio';self.rejected+=1;return
  self.vio=m;self.history.append((ts,position,Rotation.from_quat(quaternion)));o=copy.deepcopy(m);o.header.frame_id='vio_global';o.child_frame_id='vio_imu';o.pose.pose.position.z+=self.height_observer.odometry(m);self.odom.publish(o)
  if ts-self.last_publish<.019:return
  self.last_publish=ts;self.reason=self.classify(now,m)
  if self.reason=='unobservable':self.rejected+=1;return
  v=m.twist.twist.linear;vb=np.array([v.x,v.y,v.z]);vc=np.asarray(m.twist.covariance).reshape(6,6).diagonal()[:3];velocity_ok=np.isfinite(np.r_[vb,vc]).all() and vc.max()<.09;qr=Rotation.from_matrix(D@Rotation.from_quat(quaternion).as_matrix()@D).as_quat();pout=position+self.correction;variance=.0016 if self.reason=='ok_geometry' else (.0036 if self.reason=='ok_visual' else .0064)
  self.ev.publish(VehicleOdometry(timestamp=int(now*1e6),timestamp_sample=int(ts*1e6),pose_frame=VehicleOdometry.POSE_FRAME_FRD,position=(D@pout).astype(np.float32).tolist(),q=[float(qr[3]),float(qr[0]),float(qr[1]),float(qr[2])],velocity_frame=VehicleOdometry.VELOCITY_FRAME_BODY_FRD if velocity_ok else VehicleOdometry.VELOCITY_FRAME_UNKNOWN,velocity=(D@vb).astype(np.float32).tolist() if velocity_ok else [math.nan]*3,angular_velocity=[math.nan]*3,position_variance=[variance]*3,orientation_variance=[.0025]*3,velocity_variance=np.maximum(vc,.0025).astype(np.float32).tolist() if velocity_ok else [math.nan]*3,quality=1))
  pose=PoseStamped();pose.header=m.header;pose.header.frame_id='vio_global';pose.pose.position.x=float(pout[0]);pose.pose.position.y=float(pout[1]);pose.pose.position.z=float(pout[2]);pose.pose.orientation=q;self.pp.publish(pose);self.lastgood=ts;self.accepted+=1
 def tf(self,m):
  for t in m.transforms:
   if t.header.frame_id!='carto_odom' or t.child_frame_id!='vio_imu' or not self.history:continue
   ts=timestamp(t.header);sample=min(self.history,key=lambda a:abs(a[0]-ts))
   if abs(sample[0]-ts)>.06:continue
   p=t.transform.translation;q=t.transform.rotation;p=np.array([p.x,p.y,p.z]);rotation=Rotation.from_quat([q.x,q.y,q.z,q.w])
   if not np.isfinite(p).all():self.carto_rejected+=1;continue
   self.last_carto=ts
   if self.carto_transform is None:
    align=sample[2]*rotation.inv();self.carto_transform=(align,sample[1]-align.apply(p))
   align,offset=self.carto_transform;measured=align.apply(p)+offset;residual=measured-sample[1]-self.correction
   if not self.geometry(self.now()) or np.linalg.norm(residual)>.25:self.carto_rejected+=1;continue
   dt=.02 if self.last_correction<0 else min(.1,max(0.,ts-self.last_correction));self.last_correction=ts;self.correction+=residual*min(.02,.02*dt/max(np.linalg.norm(residual),1e-8))
 def report(self):
  now=self.now();self.hp.publish(String(data=json.dumps({'stamp':now,'valid':self.reason not in ['startup','invalid_vio','unobservable'] and now-self.lastgood<.2,'reason':self.reason,'accepted':self.accepted,'rejected':self.rejected,'last_good':self.lastgood,'accepted_visual_update_features':self.features,'visual_update_age':now-self.last_update,'inertial_increment_std_bound_m':self.bridge_std,'cad_supported':bool(self.cad and self.cad['supported'] and now-self.cad['stamp']<.4),'depth_rank':None if self.depth is None else self.depth['normal_rank_ratio'],'mapping_structure':None if self.mapping is None else self.mapping['structure_points'],'carto_height_correction_m':self.height_observer.correction,'carto_height_rejected':self.height_observer.rejected,'carto_height_reacquired':self.height_observer.reacquired,'carto_height_floor_support_age':now-self.height_observer.floor_support,'carto_corrections_rejected':self.carto_rejected,'source':'continuous measured VIO with bounded observed local SLAM correction; no loop-map pose or truth','configured_noise_not_carto_covariance':True})))
rclpy.init();n=Fusion()
try:rclpy.spin(n)
finally:n.destroy_node();rclpy.shutdown()
