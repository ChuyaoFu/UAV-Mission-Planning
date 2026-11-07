#!/usr/bin/env python3
"""Default-OFF ROS2 structural evidence owner. Source-only delivery.
Authoritative raw remains on the existing original raw topic at10Hz. This node
publishes completed inferred packets ONLY on the sole evidence topic: sending
raw-only evidence first would consume message_filters' exact pose pairing.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import copy,json,time,fcntl,hashlib
from pathlib import Path
from collections import OrderedDict
import numpy as np,cv2,yaml
from stream_adapter import GrabCutLatestAdapter
from exact_exposure_assembler import ExactExposureAssembler
from live_inputs import LiveSensorWindows,transform

def ns(header):return int(header.stamp.sec)*1_000_000_000+int(header.stamp.nanosec)
def estimated_pose(msg):
 p=msg.pose.pose if hasattr(msg.pose,'pose') else msg.pose;q=p.orientation;t=p.position;return transform([t.x,t.y,t.z],[q.x,q.y,q.z,q.w])
def load_actual_calibration(rig_path,robot_path,mask_dir):
 if not all((rig_path,robot_path,mask_dir)):raise ValueError('enabled inference requires explicit actual rig/robot/mask paths')
 text=Path(rig_path).read_text();rig=yaml.safe_load('\n'.join(l for l in text.splitlines() if not l.startswith('%YAML')));a,b=rig['cam0'],rig['cam1'];K=np.array(a['intrinsics'],float);T=np.array(a['T_imu_cam'],float);R=np.array(b['T_imu_cam'],float);baseline=float(np.linalg.norm(T[:3,3]-R[:3,3]))
 if tuple(a['resolution'])!=(1280,800) or tuple(b['resolution'])!=(1280,800) or not np.array_equal(K,np.array(b['intrinsics'])) or not np.allclose(T[:3,:3],R[:3,:3]) or not np.allclose(R[:3,3]-T[:3,3],T[:3,0]*baseline,atol=1e-5) or max(abs(np.array(a['distortion_coeffs'])))>1e-8 or max(abs(np.array(b['distortion_coeffs'])))>1e-8:raise ValueError('actual native rectified shared-K positive-X stereo calibration required')
 masks=[]
 for i in (0,1):
  m=cv2.imread(str(Path(mask_dir)/f'self-mask{i}.png'),cv2.IMREAD_GRAYSCALE)
  if m is None or m.shape!=(800,1280):raise ValueError('both actual native self masks required')
  masks.append(m>0)
 robot=json.loads(Path(robot_path).read_text());tof=np.array(robot['tof_in_body_m'],float)
 # The configured body/IMU mount quaternion is intentionally never read.
 return K,T,baseline,masks,tof

def main(args=None):
 import rclpy
 from rclpy.node import Node
 from rclpy.qos import QoSProfile,ReliabilityPolicy,HistoryPolicy
 from sensor_msgs.msg import Image,Imu,Range
 from geometry_msgs.msg import PoseStamped
 from nav_msgs.msg import Odometry
 from std_msgs.msg import String,Float64MultiArray
 from cv_bridge import CvBridge
 class StructuralNode(Node):
  def __init__(self):
   super().__init__('structural_free_ray_evidence')
   def param(k,v):return self.declare_parameter(k,v).value
   self.enabled=bool(param('enable_inferred_free',False));self.model_kind=str(param('model_kind','ground_ipm'));self.include_floor=bool(param('include_floor_completion',True));self.variant=str(param('engine_variant','cold5_freeze1_seedvalidated_240p'));self.topic=str(param('evidence_topic','/nav/depth_evidence'));self.joint_topic=str(param('native_joint_topic','auto'));self.world_frame=str(param('navigation_world_frame','world'));self.max_age=float(param('inferred_input_max_age_s',.5));self.compute_budget=float(param('compute_budget_s',.4));self.rate=float(param('max_hz',2.5));self.raw_max_age=float(param('raw_input_max_age_s',.25));self.voxel=float(param('voxel_resolution_m',.1));self.verified_retreat=bool(param('native_diagonal_retreat_verified',False));self.rig_path=str(param('rig_path',''));self.robot_path=str(param('robot_path',''));self.mask_dir=str(param('mask_dir',''));self.library=str(param('lbd_library',''));self.health_topic=str(param('health_topic','/nav/structural_evidence_health'));self.bridge=CvBridge();self.pub=None;self.owner_fault=False;self.start_mono=time.monotonic();self.last_ros=self.clock_ns();self.last_raw_ns=None;self.last_raw_mono=-float('inf');self.last_health_mono=-float('inf');self.last_publish_age_s=None;self.last_wall_age_s=None;self.last_inferred_stamp=-1;self.pending=OrderedDict();self.worker=None;self.assembler=None;self.inputs=None;self.native_sub=None;self.selected_joint_topic=None;self.last_result={};self.counts=dict(raw_received=0,raw_invalid=0,joined_floor=0,evidence_published=0,inferred_pixels=0,stale=0,missing_floor=0,native_joint_received=0,clock_resets=0,owner_conflicts=0,invalid_inputs=0)
   self.imu_qos_depth=int(param('imu_qos_depth',200));self.body_qos_depth=int(param('body_qos_depth',20));self.range_qos_depth=int(param('range_qos_depth',20));self.camera_pose_qos_depth=int(param('camera_pose_qos_depth',20))
   for name,value,cap in (('imu_qos_depth',self.imu_qos_depth,512),('body_qos_depth',self.body_qos_depth,64),('range_qos_depth',self.range_qos_depth,64),('camera_pose_qos_depth',self.camera_pose_qos_depth,64)):
    if not 1<=value<=cap:raise ValueError(name+' outside bounded [1,'+str(cap)+']')
   self.counts['source_health_rejections']=0
   if not(0<self.raw_max_age<=.25):raise ValueError('authoritative raw freshness is not relaxed')
   if self.enabled:
    if not self.verified_retreat:raise ValueError('set native_diagonal_retreat_verified only for the actual consumer that subtracts sqrt(3)*voxel')
    K,T,B,masks,tof=load_actual_calibration(self.rig_path,self.robot_path,self.mask_dir)
    if not self.library or not Path(self.library).is_file():raise ValueError('actual Linux-built LBD library required')
    self.inputs=LiveSensorWindows(T,tof);self.worker=GrabCutLatestAdapter(T,*masks,self.library,enabled=True,permit_conditional_free=True,clock_ns=self.clock_ns,health_ok_now=self.health_ok,max_rate_hz=self.rate,max_compute_s=self.compute_budget,max_input_age_s=self.max_age,voxel_resolution_m=self.voxel,consumer_reserve_credit_m=np.sqrt(3.)*self.voxel,consumer_quantization_verified=True,engine_variant=self.variant,model_kind=self.model_kind,include_floor_completion=self.include_floor);self.assembler=ExactExposureAssembler(self.worker,K,B)
   lease=hashlib.sha256((os.environ.get('ROS_DOMAIN_ID','0')+'|'+self.topic).encode()).hexdigest()[:16];self.lease=open(f'/tmp/uav-structural-evidence-{os.getuid()}-{lease}.lock','a+')
   try:fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:raise RuntimeError('another structural evidence owner already holds this DDS-domain/topic lease')
   qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST);self.output_qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,history=HistoryPolicy.KEEP_LAST);self.hp=self.create_publisher(String,self.health_topic,1);self.subs=[self.create_subscription(Image,'/stereo/depth/image',self.raw,qos)]
   imu_qos=QoSProfile(depth=self.imu_qos_depth,reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST);body_qos=QoSProfile(depth=self.body_qos_depth,reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST);range_qos=QoSProfile(depth=self.range_qos_depth,reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST);camera_qos=QoSProfile(depth=self.camera_pose_qos_depth,reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST)
   if self.enabled:
    self.subs += [self.create_subscription(Image,'/vio/left/image_raw',lambda m:self.image('left',m),qos),self.create_subscription(Image,'/vio/right/image_raw',lambda m:self.image('right',m),qos),self.create_subscription(Imu,'/vio/imu',self.imu,imu_qos),self.create_subscription(Range,'/carto/down_range',self.range,range_qos),self.create_subscription(Odometry,'/nav/body_odom',self.body,body_qos),self.create_subscription(PoseStamped,'/nav/camera_pose',self.camera,camera_qos),self.create_subscription(String,'/carto/localization_health',self.localization_health,qos)]
   self.timer=self.create_timer(.01,self.tick);self.discovery_timer=self.create_timer(.5,self.discovery)
  def clock_ns(self):return int(self.get_clock().now().nanoseconds)
  def health_ok(self):
   now=self.clock_ns();rawage=float('inf') if self.last_raw_ns is None else (now-self.last_raw_ns)*1e-9
   return bool(self.inputs is not None and self.inputs.healthy(now) and 0<=rawage<=self.raw_max_age and time.monotonic()-self.last_raw_mono<=self.raw_max_age and not self.owner_fault)
  def discovery(self):
   if self.owner_fault:return
   if self.pub is None and time.monotonic()-self.start_mono>=1.0:
    if self.count_publishers(self.topic)>0:self.fail_owner('existing evidence publisher found; stop old planar owner before starting structural');return
    self.pub=self.create_publisher(Image,self.topic,self.output_qos)
    if self.worker:self.worker.start()
   elif self.pub is not None and self.count_publishers(self.topic)>1:self.fail_owner('second evidence publisher discovered; structural owner has stopped publishing');return
   if not self.enabled or self.native_sub is not None:return
   candidates=[self.joint_topic] if self.joint_topic!='auto' else [n for n,t in self.get_topic_names_and_types() if 'std_msgs/msg/Float64MultiArray' in t and n.startswith('/ov_msckf/') and ('clone' in n or 'joint' in n)]
   if len(candidates)==1 and self.count_publishers(candidates[0])==1:
    name=candidates[0];self.native_sub=self.create_subscription(Float64MultiArray,name,self.native_joint,QoSProfile(depth=2,reliability=ReliabilityPolicy.BEST_EFFORT));self.selected_joint_topic=name;self.get_logger().info('actual native joint input: '+name)
  def fail_owner(self,reason):
   self.owner_fault=True;self.counts['owner_conflicts']+=1
   if self.worker:self.worker.enabled=False
   if self.pub is not None:self.destroy_publisher(self.pub);self.pub=None
   self.get_logger().error(reason)
  def valid_image(self,m,encoding):return m.width==1280 and m.height==800 and m.encoding==encoding
  def raw(self,m):
   self.counts['raw_received']+=1;s=ns(m.header)
   if not self.valid_image(m,'32FC1') or m.header.frame_id!='left_optical':self.counts['raw_invalid']+=1;return
   self.last_raw_ns=s;self.last_raw_mono=time.monotonic()
   # NO raw-only message is sent to the FREE evidence synchronizer. The native
   # raw grid_map/depth input remains independently wired to this original raw
   # source at10Hz; this node neither delays nor replaces that stream.
   if not self.enabled:return
   if not self.valid_image(m,'32FC1') or m.header.frame_id!='left_optical':self.counts['raw_invalid']+=1;return
   if not 0<=(self.clock_ns()-s)*1e-9<=self.max_age:self.counts['stale']+=1;return
   try:
    a=self.bridge.imgmsg_to_cv2(m,desired_encoding='passthrough');a.setflags(write=False);self.assembler.offer_image('depth',s,a,m.header.frame_id,m.header);self.pending[s]=None
    while len(self.pending)>24:self.pending.popitem(last=False)
   except (ValueError,TypeError,RuntimeError) as e:self.bad_input(e)
  def image(self,kind,m):
   if not self.valid_image(m,'mono8'):self.counts['invalid_inputs']+=1;return
   s=ns(m.header)
   if not 0<=(self.clock_ns()-s)*1e-9<=self.max_age:return
   try:a=self.bridge.imgmsg_to_cv2(m,desired_encoding='passthrough');a.setflags(write=False);self.assembler.offer_image(kind,s,a,m.header.frame_id,m.header)
   except (ValueError,TypeError,RuntimeError) as e:self.bad_input(e)
  def imu(self,m):self.inputs.add_imu(ns(m.header),[m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z],[m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z])
  def range(self,m):self.inputs.add_range(ns(m.header),float(m.range),float(m.min_range),float(m.max_range))
  def body(self,m):
   try:
    if m.header.frame_id!=self.world_frame:raise ValueError('body odometry is not in configured actual navigation world frame')
    self.inputs.add_body(ns(m.header),estimated_pose(m))
   except (ValueError,TypeError) as e:self.bad_input(e)
  def camera(self,m):
   try:
    if m.header.frame_id!=self.world_frame:raise ValueError('camera pose is not in the same actual navigation world frame')
    self.inputs.add_camera(ns(m.header),estimated_pose(m))
   except (ValueError,TypeError) as e:self.bad_input(e)
  def localization_health(self,m):
   try:self.inputs.add_health(json.loads(m.data))
   except (ValueError,TypeError) as e:self.bad_input(e)
  def native_joint(self,m):
   try:
    a=np.asarray(m.data,np.float64).copy()
    if a.ndim!=1 or len(a)<163 or len(a)>1603 or a[0]!=1 or len(a)!=3+160*int(a[1]):raise ValueError('native joint schema1 required')
    a.setflags(write=False);self.worker.offer_native_joint(a);self.counts['native_joint_received']+=1
   except (ValueError,TypeError,OverflowError) as e:self.bad_input(e)
  def bad_input(self,e):
   self.counts['invalid_inputs']+=1;self.last_result={'input_rejection':str(e)}
  def tick(self):
   now=self.clock_ns()
   if now<self.last_ros:
    self.pending.clear();self.last_inferred_stamp=-1;self.counts['clock_resets']+=1
    if self.inputs:self.inputs.reset()
    if self.worker:self.worker.reset_inputs()
    if self.assembler:self.assembler.parts.clear()
   self.last_ros=now
   if self.enabled and not self.owner_fault:
    for s in list(self.pending):
     if (now-s)*1e-9>self.max_age:self.pending.pop(s,None);self.counts['missing_floor']+=1;continue
     if not self.health_ok():self.counts['source_health_rejections']+=1;continue
     pair=self.inputs.floor_for_exposure(s,now)
     if pair is None:continue
     fields,provenance=pair
     try:self.assembler.offer_floor_geometry(s,source_health=True,**fields);self.counts['joined_floor']+=1;self.pending.pop(s,None)
     except (ValueError,TypeError) as e:self.bad_input(e)
    result=self.worker.poll()
    if result is not None:
     self.last_result={k:v for k,v in result.get('stats',{}).items() if k not in ('triangulation','floor_association','raw_crosschecks')};packet=result.get('packet')
     if packet is not None and result.get('eligible_for_inferred_free') and self.pub is not None and packet['stamp_ns']>self.last_inferred_stamp:
      evidence=self.worker.evidence_32fc2(result);pixels=int(np.isfinite(evidence[...,1]).sum())
      if pixels:
       msg=self.bridge.cv2_to_imgmsg(evidence,encoding='32FC2');msg.header=copy.deepcopy(result['original_header']);age=(self.clock_ns()-packet['stamp_ns'])*1e-9;wallage=result['stats']['job_input_ros_age_s']+time.monotonic()-result['submitted_monotonic']
       if self.health_ok() and 0<=age<=self.max_age and 0<=wallage<=self.max_age:self.pub.publish(msg);self.last_inferred_stamp=packet['stamp_ns'];self.counts['evidence_published']+=1;self.counts['inferred_pixels']=pixels;self.last_publish_age_s=age;self.last_wall_age_s=wallage
       else:self.counts['stale']+=1
   if time.monotonic()-self.last_health_mono>=1.:
    self.last_health_mono=time.monotonic();report=dict(self.counts,inferred_free_enabled=self.enabled,raw_output_waits_for_stereo=False,last_compute_s=self.last_result.get('compute_s'),last_publish_age_s=self.last_publish_age_s,last_wall_publish_age_s=self.last_wall_age_s,last_free_fraction=self.counts['inferred_pixels']/(800.*1280.),max_hz=self.rate,enabled=self.enabled,engine_variant=self.variant,model_kind=self.model_kind,floor_completion_included=self.include_floor,evidence_topic=self.topic,evidence_publisher_claimed=self.pub is not None,owner_fault=self.owner_fault,native_joint_topic=self.selected_joint_topic,raw_topic='/stereo/depth/image',raw_forwarding='existing original native raw input remains independent; no earlyrawonly free-evidence packets',last_raw_age_s=None if self.last_raw_ns is None else (now-self.last_raw_ns)*1e-9,actual_health_ok=self.health_ok(),floor_wait_reason=None if self.inputs is None else self.inputs.last_reason,sensor_qos_depths=dict(imu=self.imu_qos_depth,body=self.body_qos_depth,range=self.range_qos_depth,camera_pose=self.camera_pose_qos_depth,image=1),sensor_windows=None if self.inputs is None else self.inputs.diagnostics(),worker_stats=None if self.worker is None else dict(self.worker.stats),cached_complete=0 if self.worker is None else len(self.worker.frames),cached_partial=0 if self.assembler is None else len(self.assembler.parts),pending_floor_exposures=len(self.pending),last_worker=self.last_result,compute_budget_s=self.compute_budget,inferred_input_max_age_s=self.max_age,raw_input_max_age_s=self.raw_max_age,output_channel1='surface minus0.10m Euclidean/norm, maxray2.5; native must subtract sqrt3*voxel independently',native_diagonal_retreat_verified=self.verified_retreat,voxel_resolution_m=self.voxel,original_header_preserved=True,occupied_endpoints='original raw consumer only; inference NEVER creates occupied endpoints',target_hardware_qualified=False)
    self.hp.publish(String(data=json.dumps(report,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))))
  def close(self):
   if self.worker:self.worker.close()
   fcntl.flock(self.lease,fcntl.LOCK_UN);self.lease.close()
 rclpy.init(args=args);node=None
 try:node=StructuralNode();rclpy.spin(node)
 finally:
  if node is not None:node.close();node.destroy_node()
  if rclpy.ok():rclpy.shutdown()
if __name__=='__main__':main()
