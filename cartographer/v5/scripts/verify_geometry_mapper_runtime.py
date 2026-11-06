#!/usr/bin/python3
"""Optional inferred-free evidence must never become measured hits. Domain43 only."""
import json,os,signal,subprocess,time
import numpy as np,rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy,qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Int8MultiArray
from camera_geometry import load_rig
assert os.environ.get('ROS_DOMAIN_ID')=='43';K,_,_,(W,H)=load_rig();env=dict(os.environ,USE_SIM_TIME='0',DISABLE_VERIFIED_LAUNCH='1',ENABLE_GEOMETRIC_FREE_RAYS='1');log=open('/tmp/v5-geometric-map-test.log','w');proc=subprocess.Popen(['python3',os.path.join(os.path.dirname(__file__),'ego_launch.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
rclpy.init();n=Node('geometric_evidence_mapper_test');states=[];sub=n.create_subscription(Int8MultiArray,'/grid_map/knowledge',lambda m:states.append(np.array(m.data,dtype=np.int8)),QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL));pub=n.create_publisher(Image,'/nav/depth_evidence',qos_profile_sensor_data);raw_pub=n.create_publisher(Image,'/nav/measured_depth',qos_profile_sensor_data);pose=n.create_publisher(PoseStamped,'/nav/camera_pose',10);odom=n.create_publisher(Odometry,'/nav/body_odom',10)
def feed(measured,lower,seconds):
 end=time.monotonic()+seconds
 while time.monotonic()<end:
  t=n.get_clock().now().to_msg();p=PoseStamped();p.header.stamp=t;p.header.frame_id='world';p.pose.position.z=.65;p.pose.orientation.w=1.;o=Odometry();o.header=p.header;o.pose.pose=p.pose;odom.publish(o);pose.publish(p);a=np.empty((H,W,2),np.float32);a[:,:,0]=measured;a[:,:,1]=lower;m=Image();m.header.stamp=t;m.header.frame_id='left_optical';m.height=H;m.width=W;m.encoding='32FC2';m.step=W*8;m.data=a.astype('<f4',copy=False).tobytes();pub.publish(m);raw=Image();raw.header=m.header;raw.height=H;raw.width=W;raw.encoding='32FC1';raw.step=W*4;raw.data=np.full((H,W),measured,dtype='<f4').tobytes();raw_pub.publish(raw);until=time.monotonic()+.1
  while time.monotonic()<until:rclpy.spin_once(n,timeout_sec=.01)
try:
 feed(np.nan,np.nan,3.);assert states and np.all(states[-1]==-1),'Empty inference invented map evidence'
 feed(np.nan,1.,1.);grid=states[-1].reshape(80,80,30);assert (grid==-3).sum()>0,'No inferred-free provenance';assert not np.isin(grid,[0,1]).any(),'Inference became measured evidence';assert np.all(grid[:,:,26:]==-1),'Inferred rays cleared terminal/beyond-surface voxels';inferred=int((grid==-3).sum())
 feed(np.nan,np.nan,1.);assert not (states[-1]==-3).any(),'Inferred support did not expire'
 feed(.7,np.nan,1.);occupied=int((states[-1]==1).sum());assert occupied>0,'Measured hit missing'
 feed(np.nan,1.2,.4);grid=states[-1].reshape(80,80,30);assert (grid==1).sum()==occupied,'Inferred ray erased measured obstacles';assert not (grid[:,:,24:]==-3).any(),'Inferred rays passed through measured occluder'
 print(json.dumps({'passed':True,'inferred_free_voxels':inferred,'inference_never_marked_measured_or_occupied':True,'terminal_quantization_reserve_verified':True,'support_expiration_verified':True,'measured_occluder_preserved':True,'vehicle_armed':False},indent=2))
finally:
 os.killpg(proc.pid,signal.SIGINT)
 try:proc.wait(timeout=8)
 except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=5)
 n.destroy_node();rclpy.try_shutdown();log.close()
