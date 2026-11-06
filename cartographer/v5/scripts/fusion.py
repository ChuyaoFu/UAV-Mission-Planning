#!/usr/bin/python3
"""Continuous VIO control odometry, independent of sparse SLAM scan arrivals.
Cartographer observes this odometry for mapping only; scan/loop poses never
enter PX4. No evaluation/GPS topic is consumed. Geometry must be measured.
"""
import copy,json,math,os
import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range,PointCloud2
from std_msgs.msg import String,Bool,Float64MultiArray
from px4_msgs.msg import VehicleOdometry,VehicleStatus,VehicleLandDetected
from height_odometry import HeightObserver
from robot_geometry import BODY_FROM_IMU,body_rotation,body_position,body_velocity,body_velocity_covariance
D=np.diag([1.,-1.,-1.])
def timestamp(h):return h.stamp.sec+h.stamp.nanosec*1e-9
class Fusion(Node):
 def __init__(self):
  super().__init__('carto_vio_fusion',parameter_overrides=[Parameter('use_sim_time',value=__import__('os').environ.get('USE_SIM_TIME','1')=='1')]);self.height_observer=HeightObserver();self.vio=None;self.depth=None;self.mapping=None;self.armed=False;self.departed=False;self.ground=None;self.depart_start=-1.;self.last_update=-1.;self.features=0;self.lastgood=-1.;self.last_publish=-1.;self.last_processed=-1.;self.accepted=0;self.rejected=0;self.reason='startup';self.bridge_std=math.inf
  self.accepted_odom=self.create_publisher(Odometry,'/carto/accepted_odom',10);self.motion=self.create_publisher(Bool,'/vio/in_motion',10);self.odom=self.create_publisher(Odometry,'/carto/vio_odom',20);self.ev=self.create_publisher(VehicleOdometry,'/fmu/in/vehicle_visual_odometry',10);self.pp=self.create_publisher(PoseStamped,'/carto/local_pose',10);self.hp=self.create_publisher(String,'/carto/localization_health',10)
  self.startup_zupt_released=False;self.startup_zupt_release_reason=None;self.native_landed_seen=False;self.native_land_source=-1;self.native_land_received=-1.;self.native_land_topic=os.environ.get('NATIVE_LAND_TOPIC','/fmu/out/vehicle_land_detected')
  self.odometry_source='/nav/home_relative_odomimu' if os.environ.get('NAV_HOME_RELATIVE_ODOMETRY')=='1' else '/ov_msckf/odomimu'
  self.subs=[self.create_subscription(PointCloud2,'/carto/points',self.height_observer.cloud,qos_profile_sensor_data),self.create_subscription(Odometry,self.odometry_source,self.vio_cb,100),self.create_subscription(Float64MultiArray,'/ov_msckf/visual_update_health',self.features_cb,20),self.create_subscription(Range,'/carto/down_range',self.range_cb,10),self.create_subscription(VehicleStatus,'/fmu/out/vehicle_status_v1',self.status_cb,qos_profile_sensor_data)]
  self.subs.append(self.create_subscription(VehicleLandDetected,self.native_land_topic,self.land_cb,qos_profile_sensor_data))
  for topic,key in [('/carto/depth_health','depth'),('/mapping/health','mapping')]:self.subs.append(self.create_subscription(String,topic,lambda m,k=key:setattr(self,k,json.loads(m.data)),20))
  self.height_freeze_sub=self.create_subscription(Bool,'/nav/freeze_global_height',lambda m:self.height_observer.freeze() if m.data else None,10)
  self.create_timer(.1,self.report)
 def now(self):return self.get_clock().now().nanoseconds*1e-9
 def status_cb(self,m):
  self.armed=m.arming_state==VehicleStatus.ARMING_STATE_ARMED
  if not self.armed:self.departed=False;self.depart_start=-1.
 def land_cb(self,m):
  # This cancels a filter-mode assumption. It is NOT an altitude observation,
  # proof of ascent, or permission to mark the ToF departure detector true.
  source=int(m.timestamp);now=self.now()
  if source<=0 or source<=self.native_land_source or not math.isfinite(now) or now<self.native_land_received:return
  self.native_land_source=source;self.native_land_received=now
  if m.landed:self.native_landed_seen=True
  elif self.native_landed_seen and not self.startup_zupt_released:
   self.startup_zupt_released=True;self.startup_zupt_release_reason='ordered native landed=false cancels startup stationary assumption; no ascent measurement'
  if self.startup_zupt_released:self.motion.publish(Bool(data=True))
 def range_cb(self,m):
  self.height_observer.ranging(m)
  if not self.armed:self.ground=m.range
  elif self.ground is not None and m.range>self.ground+.06 and not self.departed:self.departed=True;self.depart_start=self.now()
  self.motion.publish(Bool(data=self.departed or self.startup_zupt_released))
 def features_cb(self,m):
  if len(m.data)==3:
   self.features=int(m.data[2])
   if self.features>=2:self.last_update=float(m.data[1])
 def classify(self,now,m):
  covariance=np.asarray(m.pose.covariance).reshape(6,6);visual=now-self.last_update<.5 and np.max(covariance.diagonal()[:3])<.01;age=max(0.,now-max(self.last_update,self.depart_start));velocity_variance=float(np.max(np.asarray(m.twist.covariance).reshape(6,6).diagonal()[:3]));self.bridge_std=math.sqrt(max(.0004,velocity_variance))*age+.025*age*age;bounded=age<3. and self.bridge_std<.08
  if visual:return 'ok_visual'
  v=m.twist.twist.linear
  if not self.departed and np.linalg.norm([v.x,v.y,v.z])<.03:return 'stationary_initialized'
  return 'bounded_inertial' if bounded else 'unobservable'
 def vio_cb(self,m):
  ts=timestamp(m.header)
  if ts-self.last_processed<.009:return
  self.last_processed=ts;now=self.now();cov=np.asarray(m.pose.covariance).reshape(6,6);p=m.pose.pose.position;q=m.pose.pose.orientation;position=np.array([p.x,p.y,p.z]);quaternion=np.array([q.x,q.y,q.z,q.w])
  if not -.05<now-ts<.2 or not np.isfinite(np.r_[position,quaternion,cov.ravel()]).all() or abs(np.linalg.norm(quaternion)-1)>.01 or cov.diagonal()[:3].max()>=.04:self.reason='invalid_vio';self.rejected+=1;return
  self.vio=m;o=copy.deepcopy(m);o.header.frame_id='vio_global';o.child_frame_id='vio_imu';height_correction=self.height_observer.odometry(m) if self.odometry_source=='/ov_msckf/odomimu' else 0.;o.pose.pose.position.z+=height_correction;self.odom.publish(o)
  if ts-self.last_publish<.019:return
  self.last_publish=ts;self.reason=self.classify(now,m)
  if self.reason=='unobservable':self.rejected+=1;return
  # Preserve the estimator's actual twist for navigation, behind the same
  # accepted-pose gate as external vision. Never differentiate pose resets.
  v=m.twist.twist.linear;w=m.twist.twist.angular;vb=body_velocity([v.x,v.y,v.z],[w.x,w.y,w.z]);vc=np.diag(body_velocity_covariance(m.twist.covariance));velocity_ok=np.isfinite(np.r_[vb,vc]).all() and vc.max()<.09;qr=Rotation.from_matrix(D@body_rotation(Rotation.from_quat(quaternion)).as_matrix()@D).as_quat();pout=position.copy();pout[2]+=height_correction;variance=.0036 if self.reason=='ok_visual' else .0064
  if velocity_ok:self.accepted_odom.publish(o)
  self.ev.publish(VehicleOdometry(timestamp=int(now*1e6),timestamp_sample=int(ts*1e6),pose_frame=VehicleOdometry.POSE_FRAME_FRD,position=(D@body_position(pout,Rotation.from_quat(quaternion))).astype(np.float32).tolist(),q=[float(qr[3]),float(qr[0]),float(qr[1]),float(qr[2])],velocity_frame=VehicleOdometry.VELOCITY_FRAME_BODY_FRD if velocity_ok else VehicleOdometry.VELOCITY_FRAME_UNKNOWN,velocity=(D@vb).astype(np.float32).tolist() if velocity_ok else [math.nan]*3,angular_velocity=[math.nan]*3,position_variance=[variance]*3,orientation_variance=[.0025]*3,velocity_variance=np.maximum(vc,.0025).astype(np.float32).tolist() if velocity_ok else [math.nan]*3,quality=1))
  pose=PoseStamped();pose.header=m.header;pose.header.frame_id='vio_global';pose.pose.position.x=float(pout[0]);pose.pose.position.y=float(pout[1]);pose.pose.position.z=float(pout[2]);pose.pose.orientation=q;self.pp.publish(pose);self.lastgood=ts;self.accepted+=1
 def report(self):
  if self.startup_zupt_released:self.motion.publish(Bool(data=True))
  now=self.now();self.hp.publish(String(data=json.dumps({'stamp':now,'startup_zupt_released':self.startup_zupt_released,'startup_zupt_release_reason':self.startup_zupt_release_reason,'native_land_source_us':self.native_land_source,'native_land_received_ros_s':self.native_land_received,'valid':self.reason not in ['startup','invalid_vio','unobservable'] and now-self.lastgood<.2,'reason':self.reason,'accepted':self.accepted,'rejected':self.rejected,'last_good':self.lastgood,'accepted_visual_update_features':self.features,'visual_update_age':now-self.last_update,'inertial_increment_std_bound_m':self.bridge_std,'depth_rank':None if self.depth is None else self.depth['normal_rank_ratio'],'mapping_structure':None if self.mapping is None else self.mapping['structure_points'],'global_height_frozen':self.height_observer.frozen,'carto_height_correction_m':self.height_observer.correction,'carto_height_rejected':self.height_observer.rejected,'carto_height_reacquired':self.height_observer.reacquired,'carto_height_floor_support_age':now-self.height_observer.floor_support,'odometry_source':self.odometry_source,'global_tof_height_correction_enabled':self.odometry_source=='/ov_msckf/odomimu','map_pose_feedback_enabled':False,'external_vision_reference':'calibrated body origin; PX4 EV lever offset must be zero','source':'native measured VIO; optional global ToF correction only on the legacy path; Cartographer poses do not correct control odometry','configured_noise_not_carto_covariance':True})))
if __name__=='__main__':
 rclpy.init();n=Fusion()
 try:rclpy.spin(n)
 finally:n.destroy_node();rclpy.shutdown()
