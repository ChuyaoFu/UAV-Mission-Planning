"""Bounded actual sensor windows for live floor-seed geometry.
Only actual IMU, range, accepted body odometry and exact exposure camera pose.
No configured camera/body angle, commanded height, GT, CAD or home clone.
"""
from collections import OrderedDict
import threading,time
import numpy as np
from scipy.spatial.transform import Rotation

def transform(position,quaternion_xyzw):
 p=np.asarray(position,float);q=np.asarray(quaternion_xyzw,float)
 if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(np.r_[p,q]).all() or abs(np.linalg.norm(q)-1)>.01:raise ValueError('invalid actual estimated pose')
 T=np.eye(4);T[:3,:3]=Rotation.from_quat(q).as_matrix();T[:3,3]=p;return T

def body_at(s,body):
 if s in body:return body[s],dict(method='exact original bodyOdom stamp',before_stamp_ns=s,after_stamp_ns=s)
 stamps=np.array(sorted(body),np.int64);j=int(np.searchsorted(stamps,s))
 if j==0 or j==len(stamps):return None,None
 a,b=int(stamps[j-1]),int(stamps[j])
 if b-a>60_000_000:return None,None
 A,B=body[a],body[b];w=(s-a)/(b-a);relative=A[:3,:3].T@B[:3,:3];rot=Rotation.from_matrix(relative).as_rotvec()
 if np.linalg.norm(rot)>.15:return None,None
 T=np.eye(4);T[:3,:3]=A[:3,:3]@Rotation.from_rotvec(w*rot).as_matrix();T[:3,3]=(1-w)*A[:3,3]+w*B[:3,3];return T,dict(method='derived SO3/translation interpolation at ORIGINAL range stamp; not restamped odometry',before_stamp_ns=a,after_stamp_ns=b,weight=w,max_gap_ns=60_000_000,position_interpolation_bound_m=2.*((b-a)*1e-9)**2/8.,rotation_interpolation_bound_rad=4.*((b-a)*1e-9)**2/8.)

class LiveSensorWindows:
 def __init__(self,T_imu_from_camera,tof_origin_body_m,capacity_seconds=1.5):
  self.TIC=np.asarray(T_imu_from_camera,float);self.tof_body=np.asarray(tof_origin_body_m,float)
  if self.tof_body.shape!=(3,) or not np.isfinite(self.tof_body).all():raise ValueError('actual calibrated ToF body translation required')
  self.horizon_ns=int(capacity_seconds*1e9);self.imu=OrderedDict();self.ranges=OrderedDict();self.body=OrderedDict();self.camera=OrderedDict();self.lock=threading.RLock();self.health=None;self.health_received=-float('inf');self.last_reason='missing_actual_sensors';self.last_support=dict(exposure_stamp_ns=None,sample_count=0,first_ns=None,last_ns=None,max_gap_ns=None)
 def reset(self):
  with self.lock:self.imu.clear();self.ranges.clear();self.body.clear();self.camera.clear();self.health=None;self.health_received=-float('inf');self.last_support=dict(exposure_stamp_ns=None,sample_count=0,first_ns=None,last_ns=None,max_gap_ns=None)
 def _put(self,store,s,value,maximum):
  store[int(s)]=value
  newest=max(store)
  for k in list(store):
   if k<newest-self.horizon_ns:store.pop(k,None)
  while len(store)>maximum:store.popitem(last=False)
 def add_imu(self,s,accel,gyro):
  a=np.asarray(accel,float);g=np.asarray(gyro,float)
  if a.shape!=(3,) or g.shape!=(3,) or not np.isfinite(np.r_[a,g]).all():return
  with self.lock:self._put(self.imu,s,(a,g),512)
 def add_range(self,s,r,minimum,maximum):
  if not np.isfinite([r,minimum,maximum]).all() or not 0<minimum<=r<=maximum:return
  with self.lock:self._put(self.ranges,s,(float(r),float(minimum),float(maximum)),96)
 def add_body(self,s,T):
  with self.lock:self._put(self.body,s,T,256)
 def add_camera(self,s,T):
  with self.lock:self._put(self.camera,s,T,96)
 def add_health(self,obj):
  if not isinstance(obj,dict) or 'valid' not in obj or 'stamp' not in obj:return
  with self.lock:self.health=dict(obj);self.health_received=time.monotonic()
 def healthy(self,now_ns):
  with self.lock:
   if self.health is None:return False
   try:age=(int(now_ns)-int(round(float(self.health['stamp'])*1e9)))*1e-9
   except (ValueError,TypeError,OverflowError):return False
   return bool(self.health.get('valid') is True and 0<=age<=.5 and time.monotonic()-self.health_received<=.5)
 def diagnostics(self):
  with self.lock:
   buffers={name:dict(length=len(store),first_ns=min(store) if store else None,last_ns=max(store) if store else None) for name,store in (('imu',self.imu),('range',self.ranges),('body',self.body),('camera_pose',self.camera))}
   return dict(last_support=dict(self.last_support),buffers=buffers,last_reason=self.last_reason)
 def floor_for_exposure(self,s,now_ns):
  with self.lock:
   support_stamps=sorted(t for t in self.imu if abs(t-s)<=40_000_000);self.last_support=dict(exposure_stamp_ns=int(s),sample_count=len(support_stamps),first_ns=support_stamps[0] if support_stamps else None,last_ns=support_stamps[-1] if support_stamps else None,max_gap_ns=max((b-a for a,b in zip(support_stamps,support_stamps[1:])),default=None))
   if s not in self.camera:self.last_reason='missing_exact_actual_camera_pose';return None
   # Wait for the complete bounded actual support window; +40ms is actual
   # received sensor evidence and is included in end-to-end input age.
   if not self.imu or max(self.imu)<s+35_000_000 or now_ns<s+40_000_000:self.last_reason='await_actual_40ms_IMU_support';return None
   samples=[(t,a,w) for t,(a,w) in self.imu.items() if abs(t-s)<=40_000_000]
   if len(samples)<2:self.last_reason='insufficient_actual_IMU_support';return None
   force=np.mean([x[1] for x in samples],axis=0);norm=float(np.linalg.norm(force))
   if not 1.<norm<30.:self.last_reason='invalid_actual_specific_force';return None
   upI=force/norm;upC=self.TIC[:3,:3].T@upI;gstamp=min(samples,key=lambda x:abs(x[0]-s))[0]
   if not self.ranges:self.last_reason='missing_actual_range';return None
   ts=min(self.ranges,key=lambda t:abs(t-s));r,lo,hi=self.ranges[ts]
   if abs(ts-s)>50_000_000 or ts>now_ns:self.last_reason='unpaired_actual_range';return None
   WB,meta=body_at(ts,self.body)
   if WB is None:self.last_reason='missing_actual_body_bracket_at_range_stamp';return None
   WC=self.camera[s];origin=WC[:3,:3].T@(WB[:3,:3]@self.tof_body+WB[:3,3]-WC[:3,3]);direction=WC[:3,:3].T@WB[:3,:3]@np.array([0.,0.,-1.]);direction/=np.linalg.norm(direction)
   if upC@direction>-.9:self.last_reason='actual_ToF_ray_not_downward';return None
   self.last_reason='actual_floor_geometry_ready';fields=dict(gravity_up_optical=upC,gravity_stamp_ns=np.int64(gstamp),tof_range_m=np.float64(r),tof_stamp_ns=np.int64(ts),tof_valid=np.bool_(lo<=r<=hi),tof_origin_optical_m=origin,tof_direction_optical=direction)
   provenance=dict(gravity='actual OAK specific force +/-40ms, dynamic acceleration not separated',imu_window_first_ns=min(x[0] for x in samples),imu_window_last_ns=max(x[0] for x in samples),specific_force_norm_mps2=norm,gyro_max_rps=max(float(np.linalg.norm(x[2])) for x in samples),body_pose_for_original_range=meta,ToF_axis='user-declared actual body-down beam; calibrated translation only',camera_pose_stamp_ns=s,native_covariance_attached_to_nav_pose=False)
   return fields,provenance
