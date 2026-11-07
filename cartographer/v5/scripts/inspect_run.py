#!/usr/bin/python3
"""Offline measured-stream diagnostics. No reference map is read."""
import argparse,json,collections
from pathlib import Path
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from online_map import OnlineMap
p=argparse.ArgumentParser();p.add_argument('bag');p.add_argument('output');a=p.parse_args();r=rosbag2_py.SequentialReader();r.open(rosbag2_py.StorageOptions(uri=a.bag,storage_id='mcap'),rosbag2_py.ConverterOptions('',''));types={x.name:get_message(x.type) for x in r.get_all_topics_and_types()};meta=None;state=None;poses=[];health=[];states=[];mapped=[]
while r.has_next():
 topic,data,t=r.read_next()
 if topic not in ['/nav/body_odom','/grid_map/knowledge','/grid_map/metadata','/carto/localization_health','/carto/mission_state']:continue
 m=deserialize_message(data,types[topic])
 if topic=='/grid_map/metadata':meta=list(m.data)
 elif topic=='/grid_map/knowledge' and meta:
  state=np.array(m.data,dtype=np.int8);grid=OnlineMap(meta,state)
  if poses:mapped.append({'t':meta[8],'position':poses[-1][1:],'safe':grid.traversable(poses[-1][1:]),'component':grid.component(poses[-1][1:]),'largest_safe_component':int(np.bincount(grid.components.ravel())[1:].max()) if grid.n_components else 0,'free':int((state==0).sum()),'occupied':int((state==1).sum())})
 elif topic=='/nav/body_odom':
  v=m.pose.pose.position;poses.append([m.header.stamp.sec+m.header.stamp.nanosec*1e-9,v.x,v.y,v.z])
 elif topic=='/carto/localization_health':health.append(json.loads(m.data))
 elif topic=='/carto/mission_state':states.append({'t':t*1e-9,'phase':m.data})
result={'health_counts':dict(collections.Counter(v['reason'] for v in health)),'maximum_visual_features':max((v['accepted_visual_update_features'] for v in health),default=0),'observed_map_snapshots':mapped[::10],'phases':states,'poses':poses[::10]};Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['poses','observed_map_snapshots']},indent=2));print(mapped[-5:])
if state is not None:np.savez_compressed(Path(a.output).with_suffix('.npz'),metadata=meta,knowledge=state,poses=poses)
