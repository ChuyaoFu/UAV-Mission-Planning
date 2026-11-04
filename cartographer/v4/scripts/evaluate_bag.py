#!/usr/bin/python3
"""Read MCAP directly for independent post-flight metrics; publishes nothing."""
import sys,json
from pathlib import Path
import numpy as np,rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation
root=Path('/work/cartographer/v4');out=root/'results'/sys.argv[1];reader=rosbag2_py.SequentialReader();reader.open(rosbag2_py.StorageOptions(uri=str(root/'data'/sys.argv[1]),storage_id='mcap'),rosbag2_py.ConverterOptions('',''))
topics={'/carto/local_pose':'local','/tracked_pose':'map','/evaluation/poses':'truth','/tf':'odom','/carto/depth_health':'depth','/carto/localization_health':'health','/cad/localization_health':'cad_health','/line/localization_health':'line_health','/window/localization_health':'window_health','/carto/down_range':'range','/ov_msckf/visual_update_health':'visual_update'};types={t.name:get_message(t.type) for t in reader.get_all_topics_and_types() if t.name in topics};reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)));rows={k:[] for k in topics.values()}
def sec(h):return h.stamp.sec+h.stamp.nanosec*1e-9
def vector(p,q,t):return [t,p.x,p.y,p.z,q.x,q.y,q.z,q.w]
while reader.has_next():
 topic,data,_=reader.read_next();m=deserialize_message(data,types[topic]);key=topics[topic]
 if key.endswith('health') or key=='depth':rows[key].append(json.loads(m.data))
 elif key in ['local','map']:rows[key].append(vector(m.pose.position,m.pose.orientation,sec(m.header)))
 elif key in ['truth','odom']:
  for t in m.transforms:
   if (key=='truth' and t.child_frame_id=='x500_vio_0') or (key=='odom' and t.header.frame_id=='carto_odom' and t.child_frame_id=='vio_imu'):rows[key].append(vector(t.transform.translation,t.transform.rotation,sec(t.header)))
 elif key=='range':rows[key].append([sec(m.header),m.range])
 elif key=='visual_update':rows[key].append(list(m.data))
summary={k+'_samples':len(v) for k,v in rows.items()};summary['source']='direct read of original MCAP; no ROS replay publications'
for key in ['local','map','odom']:
 a=np.array(rows[key])
 if len(a):summary[key+'_position_span_m']=np.ptp(a[:,1:4],axis=0).tolist()
if rows['depth']:summary['depth_mean_valid_fraction']=float(np.mean([v['valid_fraction'] for v in rows['depth']]))
a=np.array(rows['local']);g=np.array(rows['truth'])
if len(a)>5 and len(g)>5:
 gp=g[:,1:4]+Rotation.from_quat(g[:,4:8]).apply([.12,0,.055]);rot=Rotation.from_quat(g[0,4:8])*Rotation.from_quat(a[0,4:8]).inv();interp=np.array([np.interp(a[:,0],g[:,0],gp[:,i]) for i in range(3)]).T;estimate=rot.apply(a[:,1:4]-a[0,1:4])+gp[0];err=np.linalg.norm(estimate-interp,axis=1);summary.update(evaluated_pose='fused continuous local odometry',ate_rmse_m=float(np.sqrt(np.mean(err**2))),ate_max_m=float(err.max()),alignment='fixed initial pose; no fitted trajectory alignment')
(out/'evaluation-offline.json').write_text(json.dumps({'summary':summary,'rows':rows},indent=2));print(json.dumps(summary,indent=2))
