#!/usr/bin/python3
"""Black-box upstream mapper test on isolated ROS domain 43; never arms PX4."""
import json,os,signal,subprocess,time
from camera_geometry import load_rig
K,_,_,(WIDTH,HEIGHT)=load_rig()
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy,qos_profile_sensor_data,ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Int8MultiArray
assert os.environ.get('ROS_DOMAIN_ID')=='43','Synthetic sensors are restricted to test domain 43'
env=dict(os.environ,USE_SIM_TIME='0',DISABLE_VERIFIED_LAUNCH='1');log=open('/tmp/v5-mapper-test.log','w');proc=subprocess.Popen(['python3',os.path.join(os.path.dirname(__file__),'ego_launch.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
rclpy.init();n=Node('mapper_policy_test');states=[];q=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL);sub=n.create_subscription(Int8MultiArray,'/grid_map/knowledge',lambda m:states.append(np.array(m.data,dtype=np.int8)),q);pub=n.create_publisher(Image,'/nav/measured_depth',QoSProfile(depth=2,reliability=ReliabilityPolicy.RELIABLE));pose=n.create_publisher(PoseStamped,'/nav/camera_pose',10);odom=n.create_publisher(Odometry,'/nav/body_odom',10)
def feed(value,seconds):
 end=time.monotonic()+seconds
 while time.monotonic()<end:
  t=n.get_clock().now().to_msg();p=PoseStamped();p.header.stamp=t;p.header.frame_id='world';p.pose.position.z=.65;p.pose.orientation.w=1.;o=Odometry();o.header=p.header;o.pose.pose=p.pose;odom.publish(o);pose.publish(p);m=Image();m.header=p.header;m.height=HEIGHT;m.width=WIDTH;m.encoding='32FC1';m.step=WIDTH*4;m.data=np.full((HEIGHT,WIDTH),value,dtype='<f4').tobytes();pub.publish(m)
  until=time.monotonic()+.1
  while time.monotonic()<until:rclpy.spin_once(n,timeout_sec=.01)
try:
 feed(0.,3.);assert states,'No online-map output';assert np.all(states[-1]==-1),'Zero depth invented free/occupied voxels'
 feed(float('nan'),1.);assert np.all(states[-1]==-1),'NaN depth invented free/occupied voxels'
 feed(5.,1.);assert np.all(states[-1]==-1),'Out-of-range depth cleared unknown space'
 feed(1.,1.);last=states[-1];assert (last==0).sum()>0 and (last==1).sum()>0 and (last==-1).sum()>0,'Valid measured rays missing free/hit/unknown distinction'
 # Every sampled interior point lies on an actually fed ray, before its hit.
 # Endpoint-cell and traversal-cell deduplication must not truncate these rays.
 expected=set()
 for v in range(3,HEIGHT-3,30):
  for u in range(3,WIDTH-3,30):
   endpoint=np.array([(u-K[2])/K[0],(v-K[3])/K[1],1.])
   for fraction in [.137,.337,.537,.737]:
    point=np.array([0.,0.,.65])+fraction*endpoint
    index=np.floor((point-np.array([-4.,-4.,-1.]))/.1).astype(int)
    expected.add(tuple(index))
 grid=last.reshape((80,80,30));missing=[i for i in expected if grid[i]!=0]
 assert not missing,f'{len(missing)} measured ray-interior voxels incorrectly remained blocked'
 print(json.dumps({'passed':True,'ray_interior_voxels_verified':len(expected),'zero_nan_far_remain_unknown':True,'measured_free_cells':int((last==0).sum()),'measured_occupied_cells':int((last==1).sum()),'unknown_cells':int((last==-1).sum()),'operator_launch_disabled':True},indent=2))
finally:
 os.killpg(proc.pid,signal.SIGINT)
 try:proc.wait(timeout=8)
 except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=5)
 n.destroy_node();rclpy.try_shutdown();log.close()
