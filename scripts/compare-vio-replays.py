#!/usr/bin/python3
import json
from pathlib import Path
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation,Slerp

root=Path('/work');original=root/'results/vio-raw-diagnostic-01'
truth=np.load(original/'sensor-diagnostic.npz')['truth']
rot=Slerp(truth[:,0],Rotation.from_quat(truth[:,4:8]))
flight=json.loads((original/'baseline-flight.json').read_text())
report={}
for cap in [20,25]:
 label=f'vio-replay-cap{cap}'
 r=rosbag2_py.SequentialReader();r.open(rosbag2_py.StorageOptions(uri=str(root/'data'/label),storage_id='mcap'),rosbag2_py.ConverterOptions('cdr','cdr'))
 types={x.name:get_message(x.type) for x in r.get_all_topics_and_types()}
 rows=[]
 while r.has_next():
  topic,b,_=r.read_next()
  if topic!='/ov_msckf/poseimu':continue
  m=deserialize_message(b,types[topic]);p=m.pose.pose.position;q=m.pose.pose.orientation
  rows.append([m.header.stamp.sec+m.header.stamp.nanosec*1e-9,p.x,p.y,p.z,q.x,q.y,q.z,q.w])
 e=np.array(rows);e=e[np.r_[True,np.diff(e[:,0])>0]]
 e=e[(e[:,0]>=truth[0,0])&(e[:,0]<=truth[-1,0])]
 gt=np.column_stack([np.interp(e[:,0],truth[:,0],truth[:,j]) for j in range(1,4)])
 align=rot([e[0,0]]).as_matrix()[0]@Rotation.from_quat(e[0,4:8]).as_matrix().T
 est=(e[:,1:4]-e[0,1:4])@align.T+gt[0]
 error=np.linalg.norm(est-gt,axis=1)
 # Retain the previous whole-trajectory rigid ATE metric for comparison, too.
 x=e[:,1:4]-e[:,1:4].mean(axis=0);y=gt-gt.mean(axis=0)
 u,_,vt=np.linalg.svd(x.T@y);s=np.eye(3);s[2,2]=np.linalg.det(u@vt)
 fitted=np.linalg.norm(x@(u@s@vt)+gt.mean(axis=0)-gt,axis=1)
 d={'samples':len(e),'camera_update_rate_hz':float((len(e)-1)/(e[-1,0]-e[0,0])),
    'initial_alignment_rmse_m':float(np.sqrt(np.mean(error**2))),'initial_alignment_max_m':float(error.max()),
    'rigid_ate_rmse_m':float(np.sqrt(np.mean(fitted**2))),'rigid_ate_max_m':float(fitted.max())}
 for name,mask in [('before_land',(e[:,0]>=flight['arm_stamp_us']*1e-6)&(e[:,0]<flight['land_command_stamp_us']*1e-6)),('landing',(e[:,0]>=flight['land_command_stamp_us']*1e-6)&(e[:,0]<=flight['end_stamp_us']*1e-6))]:
  d[name+'_initial_alignment_max_m']=float(error[mask].max())
 report[label]=d
(root/'results/vio-replay-comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
