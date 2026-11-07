#!/usr/bin/python3
"""Offline same-recording estimator comparison; never a navigation process."""
import argparse,json
from pathlib import Path
import numpy as np,rosbag2_py
from scipy.spatial.transform import Rotation,Slerp
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--bag',required=True);a=p.parse_args();root=Path(a.run)
robot=json.loads((root/'config/robot.json').read_text());lever=np.array(robot['imu_in_body_m']);mount=Rotation.from_quat(robot.get('imu_in_body_quaternion_xyzw',[0,0,0,1]))
g=np.array(json.loads((root/'evaluation.json').read_text())['poses']);g=g[np.unique(g[:,0],return_index=True)[1]];rows={'/ov_msckf/odomimu':[],'/nav/body_odom':[]};reader=rosbag2_py.SequentialReader();reader.open(rosbag2_py.StorageOptions(uri=a.bag,storage_id='mcap'),rosbag2_py.ConverterOptions('',''))
while reader.has_next():
 topic,data,t=reader.read_next()
 if topic not in rows:continue
 m=deserialize_message(data,Odometry);p=m.pose.pose.position;q=m.pose.pose.orientation;r=Rotation.from_quat([q.x,q.y,q.z,q.w]);position=np.array([p.x,p.y,p.z])
 if topic=='/ov_msckf/odomimu':r=r*mount.inv();position-=r.apply(lever)
 rows[topic].append([m.header.stamp.sec+m.header.stamp.nanosec*1e-9,*position,*r.as_quat()])
arrays={k:np.array(v) for k,v in rows.items()};arrays={k:v[np.unique(v[:,0],return_index=True)[1]] for k,v in arrays.items()};start=max([g[0,0]]+[v[0,0] for v in arrays.values()]);end=min([g[-1,0]]+[v[-1,0] for v in arrays.values()]);times=np.arange(start,end,.05);truth=np.column_stack([np.interp(times,g[:,0],g[:,k]) for k in [1,2,3]]);true_r=Slerp(g[:,0],Rotation.from_quat(g[:,4:8]))([times[0]])[0];result={'scope':'Independent offline evaluation only; both estimators evaluated on the same 20 Hz timestamps with their own initial-pose alignment','samples':len(times),'estimators':{}}
for topic,v in arrays.items():
 position=np.column_stack([np.interp(times,v[:,0],v[:,k]) for k in [1,2,3]]);initial=Slerp(v[:,0],Rotation.from_quat(v[:,4:8]))([times[0]])[0];r=true_r*initial.inv();estimate=r.apply(position-position[0])+truth[0];error=estimate-truth;result['estimators'][topic]={'rmse_3d_m':float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),'rmse_xy_m':float(np.sqrt(np.mean(np.sum(error[:,:2]**2,axis=1)))),'rmse_components_m':np.sqrt(np.mean(error**2,axis=0)).tolist()}
(root/'estimator-ablation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
