#!/usr/bin/python3
"""Isolated mapping replay: recorded camera-derived points, IMU and VIO only.
No truth or CAD topics are published; domain 43 is required, never flight 42.
"""
import argparse,json,os,time,subprocess
from pathlib import Path
import numpy as np,rosbag2_py,rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage
from cartographer_ros_msgs.srv import TrajectoryQuery,FinishTrajectory,WriteState
ap=argparse.ArgumentParser();ap.add_argument('source');ap.add_argument('label');ap.add_argument('--rate',type=float,default=2.);ap.add_argument('--structure',action='store_true');ap.add_argument('--bag-root',default='/work/cartographer/v2/data');a=ap.parse_args();assert os.environ.get('ROS_DOMAIN_ID')=='43'
root=Path('/work/cartographer/v3');out=root/'results'/a.label;out.mkdir(exist_ok=True);bag=str(Path(a.bag_root)/a.source)
r=rosbag2_py.SequentialReader();r.open(rosbag2_py.StorageOptions(uri=bag,storage_id='mcap'),rosbag2_py.ConverterOptions('',''));wanted=['/clock','/vio/imu','/carto/points','/ov_msckf/odomimu'];wanted+=['/vio/left/image_raw','/carto/down_range'] if a.structure else [];types={t.name:get_message(t.type) for t in r.get_all_topics_and_types() if t.name in wanted};r.set_filter(rosbag2_py.StorageFilter(topics=wanted));rclpy.init();n=Node('isolated_mapping_replay');pub={t:n.create_publisher(types[t],'/carto/vio_odom' if t.endswith('odomimu') else t,100) for t in types};rows={'map':[],'local':[]}
from mapping_cloud import StructuralFilter
structure=StructuralFilter();structure_health=[]
def vec(p,q,stamp):return [stamp.sec+stamp.nanosec*1e-9,p.x,p.y,p.z,q.x,q.y,q.z,q.w]
def pose(m):rows['map'].append(vec(m.pose.position,m.pose.orientation,m.header.stamp))
def tf(m):
 for t in m.transforms:
  if t.header.frame_id=='carto_odom' and t.child_frame_id=='vio_imu':rows['local'].append(vec(t.transform.translation,t.transform.rotation,t.header.stamp))
subs=[n.create_subscription(PoseStamped,'/tracked_pose',pose,1000),n.create_subscription(TFMessage,'/tf',tf,1000)];time.sleep(2);start=None;wall=None;count=0
while r.has_next():
 topic,data,ts=r.read_next()
 if start is None:start=ts;wall=time.monotonic()
 deadline=wall+(ts-start)*1e-9/a.rate
 while time.monotonic()<deadline:rclpy.spin_once(n,timeout_sec=min(.003,max(0.,deadline-time.monotonic())))
 m=deserialize_message(data,types[topic])
 if topic.endswith('odomimu'):
  if a.structure:structure.odom(m)
  m.header.frame_id='vio_global';m.child_frame_id='vio_imu'
 if a.structure:
  if topic.endswith('image_raw'):structure.image(m);continue
  if topic.endswith('down_range'):structure.ranging(m);continue
  if topic=='/carto/points':
   p=structure.filter(m)
   if structure.health:structure_health.append(structure.health.copy())
   if p is None:continue
   m.width=len(p);m.row_step=len(p)*12;m.data=p.tobytes()
 pub[topic].publish(m);count+=1
 if count%300==0:rclpy.spin_once(n,timeout_sec=0.)
end=time.monotonic()+8
while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.01)
def service(cls,name,req):
 c=n.create_client(cls,name)
 if not c.wait_for_service(timeout_sec=3):raise RuntimeError(name+' missing')
 f=c.call_async(req);rclpy.spin_until_future_complete(n,f,timeout_sec=30)
 if not f.done():raise RuntimeError(name+' timeout')
 return f.result()
f=FinishTrajectory.Request();f.trajectory_id=0;service(FinishTrajectory,'/finish_trajectory',f);time.sleep(3);q=TrajectoryQuery.Request();q.trajectory_id=0;result=service(TrajectoryQuery,'/trajectory_query',q);rows['optimized']=[vec(p.pose.position,p.pose.orientation,p.header.stamp) for p in result.trajectory];w=WriteState.Request();w.filename=str(out/'cartographer.pbstream');w.include_unfinished_submaps=True;service(WriteState,'/write_state',w);(out/'structure-health.json').write_text(json.dumps(structure_health));(out/'trajectory.json').write_text(json.dumps(rows));print(json.dumps({'messages':count,'samples':{k:len(v) for k,v in rows.items()}}));n.destroy_node();rclpy.shutdown()
