#!/usr/bin/python3
"""Offline sensor-stream timing and velocity diagnostics; no reference poses."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np,rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation
from robot_geometry import body_rotation,body_velocity
p=argparse.ArgumentParser();p.add_argument('bag');p.add_argument('output');a=p.parse_args();reader=rosbag2_py.SequentialReader();reader.open(rosbag2_py.StorageOptions(uri=a.bag,storage_id='mcap'),rosbag2_py.ConverterOptions('',''));types={x.name:x.type for x in reader.get_all_topics_and_types()};topics=['/position_cmd','/planning/bspline','/ov_msckf/odomimu','/ov_msckf/visual_update_health','/carto/local_pose','/nav/body_odom','/nav/camera_pose','/grid_map/metadata','/stereo/health','/carto/localization_health','/carto/mission_state'];reader.set_filter(rosbag2_py.StorageFilter(topics=topics));types={name:get_message(kind) for name,kind in types.items() if name in topics};streams=defaultdict(list);phases=[];velocities={'raw':[],'derived':[]};body=[];health=[]
def stamp(m):return m.header.stamp.sec+m.header.stamp.nanosec*1e-9
while reader.has_next():
 topic,data,t=reader.read_next();m=deserialize_message(data,types[topic]);arrival=t*1e-9;extra={}
 if topic=='/carto/mission_state':phases.append((arrival,m.data));continue
 if topic in ['/carto/localization_health','/stereo/health']:
  d=json.loads(m.data);exposure=d['stamp'];extra=d
  if topic=='/carto/localization_health':health.append((arrival,d))
 elif topic=='/grid_map/metadata':exposure=float(m.data[8]);extra={'published_at':float(m.data[7])}
 elif topic=='/planning/bspline':exposure=m.start_time.sec+m.start_time.nanosec*1e-9
 elif topic=='/ov_msckf/visual_update_health':exposure=float(m.data[0]);extra={'last_update':float(m.data[1]),'features':int(m.data[2])}
 else:exposure=stamp(m)
 streams[topic].append((arrival,exposure,extra))
 if topic in ['/ov_msckf/odomimu','/nav/body_odom']:
  q=m.pose.pose.orientation;r=Rotation.from_quat([q.x,q.y,q.z,q.w]);v=m.twist.twist.linear;w=m.twist.twist.angular
  world=body_rotation(r).apply(body_velocity([v.x,v.y,v.z],[w.x,w.y,w.z])) if topic=='/ov_msckf/odomimu' else r.apply([v.x,v.y,v.z]);velocities['raw' if topic=='/ov_msckf/odomimu' else 'derived'].append((exposure,*world))
  if topic=='/nav/body_odom':pp=m.pose.pose.position;body.append((exposure,pp.x,pp.y,pp.z))

def percentiles(values):return np.percentile(values,[50,95,99,100]).tolist() if len(values) else []
result={'scope':'Measured messages only; no simulator pose, geometry or evaluation results','streams':{},'phases':[]}
for topic,values in streams.items():
 v=np.array([[x[0],x[1]] for x in values]);unique=np.unique(v[:,1]);result['streams'][topic]={'count':len(v),'unique_exposures':len(unique),'max_exposure_gap_s':float(np.diff(unique).max()) if len(unique)>1 else None,'arrival_minus_exposure_p50_p95_p99_max_s':percentiles(v[:,0]-v[:,1])}
for index,(start,phase) in enumerate(phases):
 end=phases[index+1][0] if index+1<len(phases) else float('inf');row={'phase':phase,'start':start,'end':end,'streams':{}}
 for topic,values in streams.items():
  selected=[x for x in values if start<=x[0]<end]
  if not selected:continue
  v=np.array([[x[0],x[1]] for x in selected]);row['streams'][topic]={'count':len(v),'exposure_first_last':v[[0,-1],1].tolist(),'maximum_age_s':float(np.max(v[:,0]-v[:,1]))}
 result['phases'].append(row)
raw=np.array(velocities['raw']);derived=np.array(velocities['derived']);selected=derived[(derived[:,0]>=raw[0,0])&(derived[:,0]<=raw[-1,0])] if len(raw) and len(derived) else np.empty((0,4))
if len(selected):
 predicted=np.column_stack([np.interp(selected[:,0],raw[:,0],raw[:,k]) for k in [1,2,3]]);error=np.linalg.norm(selected[:,1:]-predicted,axis=1);speed=np.linalg.norm(selected[:,1:],axis=1);raw_speed=np.linalg.norm(predicted,axis=1);worst=np.argsort(error)[-12:];result['velocity']={'samples':len(selected),'raw_speed_p50_p95_p99_max_mps':percentiles(raw_speed),'derived_speed_p50_p95_p99_max_mps':percentiles(speed),'disagreement_p50_p95_p99_max_mps':percentiles(error),'derived_above_5cmps_when_raw_below_3cmps_fraction':float(np.mean((speed>.05)&(raw_speed<.03))),'worst':[{'stamp':float(selected[i,0]),'derived_world':selected[i,1:].tolist(),'raw_world':predicted[i].tolist(),'difference_mps':float(error[i])} for i in worst]}
if len(selected) and phases:
 from bisect import bisect_right
 phase_times=[x[0] for x in phases];names=[phases[max(0,bisect_right(phase_times,t)-1)][1] for t in selected[:,0]];result['velocity_by_phase']={}
 for phase in sorted(set(names)):
  ids=np.array([name==phase for name in names]);result['velocity_by_phase'][phase]={'samples':int(ids.sum()),'disagreement_p50_p95_p99_max_mps':percentiles(error[ids]),'raw_speed_p50_p95_p99_max_mps':percentiles(raw_speed[ids]),'derived_speed_p50_p95_p99_max_mps':percentiles(speed[ids]),'derived_above_5cmps_when_raw_below_3cmps_fraction':float(np.mean((speed[ids]>.05)&(raw_speed[ids]<.03)))}
if streams['/carto/local_pose'] and streams['/nav/camera_pose']:
 from bisect import bisect_left
 pose_times=[x[1] for x in streams['/carto/local_pose']];gaps=[]
 for arrival,t,_ in streams['/nav/camera_pose']:
  index=bisect_left(pose_times,t)
  if 0<index<len(pose_times) and pose_times[index]!=t and pose_times[index]-pose_times[index-1]>.1:gaps.append({'exposure':t,'lower_pose':pose_times[index-1],'upper_pose':pose_times[index],'gap_s':pose_times[index]-pose_times[index-1]})
 result['camera_poses_previously_interpolated_across_gaps_above_100ms']={'count':len(gaps),'examples':gaps[:20]}
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({'streams':result['streams'],'velocity':result.get('velocity')},indent=2))
