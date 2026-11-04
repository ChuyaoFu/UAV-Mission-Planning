#!/usr/bin/python3
"""Isolated exact-input replay. Publishes measured mapping points/IMU/VIO only."""
import os,sys,time,json
from pathlib import Path
import numpy as np,rosbag2_py,rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from cartographer_ros_msgs.srv import TrajectoryQuery,FinishTrajectory,WriteState
assert os.environ.get('ROS_DOMAIN_ID')=='43','Never publish replay into flight domain'
bag=Path(sys.argv[1]);out=Path(sys.argv[2]);rate=float(sys.argv[3]) if len(sys.argv)>3 else 3.;out.mkdir(exist_ok=True,parents=True);r=rosbag2_py.SequentialReader();r.open(rosbag2_py.StorageOptions(uri=str(bag),storage_id='mcap'),rosbag2_py.ConverterOptions('',''));published=['/clock','/vio/imu','/mapping/points','/carto/vio_odom'];wanted=published+['/carto/points','/ov_msckf/odomimu','/carto/down_range','/vio/left/image_raw'];types={t.name:get_message(t.type) for t in r.get_all_topics_and_types() if t.name in wanted};r.set_filter(rosbag2_py.StorageFilter(topics=list(types)));rclpy.init();n=Node('isolated_exact_mapping_replay');pub={t:n.create_publisher(types[t],t,100) for t in published};poses=[]
from mapping_cloud import StructuralFilter
bootstrap=StructuralFilter();bootstrap_count=0
def vec(p):
 q=p.pose.orientation;v=p.pose.position;return [p.header.stamp.sec+p.header.stamp.nanosec*1e-9,v.x,v.y,v.z,q.x,q.y,q.z,q.w]
sub=n.create_subscription(PoseStamped,'/tracked_pose',lambda m:poses.append(vec(m)),1000);time.sleep(2);start=None;wall=None;count=0
while r.has_next():
 topic,data,ts=r.read_next()
 if start is None:start=ts;wall=time.monotonic()
 deadline=wall+(ts-start)*1e-9/rate
 while time.monotonic()<deadline:rclpy.spin_once(n,timeout_sec=min(.002,max(0.,deadline-time.monotonic())))
 if topic in published:pub[topic].publish(deserialize_message(data,types[topic]));count+=1
 elif bootstrap_count<5:
  m=deserialize_message(data,types[topic])
  if topic.endswith('odomimu'):bootstrap.odom(m)
  elif topic.endswith('down_range'):bootstrap.ranging(m)
  elif topic.endswith('image_raw'):bootstrap.image(m)
  elif topic=='/carto/points':
   before=bootstrap.bootstrap_left;p=bootstrap.filter(m)
   if p is not None and bootstrap.bootstrap_left<before:
    m.width=len(p);m.row_step=len(p)*12;m.data=p.tobytes();pub['/mapping/points'].publish(m);bootstrap_count+=1
 if count%100==0:rclpy.spin_once(n,timeout_sec=0.)
end=time.monotonic()+3
while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.01)
def call(cls,name,req):
 c=n.create_client(cls,name);assert c.wait_for_service(timeout_sec=3);f=c.call_async(req);rclpy.spin_until_future_complete(n,f,timeout_sec=45);assert f.done(),name;return f.result()
assert bootstrap_count==5,("Incomplete measured startup",bootstrap_count)
f=FinishTrajectory.Request();f.trajectory_id=0;call(FinishTrajectory,'/finish_trajectory',f);time.sleep(2);q=TrajectoryQuery.Request();q.trajectory_id=0;m=call(TrajectoryQuery,'/trajectory_query',q);rows=[vec(p) for p in m.trajectory];(out/'trajectory.json').write_text(json.dumps({'optimized':rows,'live_map':poses}));w=WriteState.Request();w.filename=str(out/'cartographer.pbstream');w.include_unfinished_submaps=True;call(WriteState,'/write_state',w);(out/'replay.json').write_text(json.dumps({'source':str(bag),'topics':published,'initialization':'Five actual measured ground-floor frames regenerated from original recorded stereo, no synthetic plane','bootstrap_frames':bootstrap_count,'messages':count,'nodes':len(rows),'domain':43,'rate':rate,'truth_or_CAD_published':False},indent=2));print('nodes',len(rows));n.destroy_node();rclpy.shutdown()
