#!/usr/bin/python3
"""OFFLINE evaluation only. True poses/scene geometry never enter navigation."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,rosbag2_py
from scipy.spatial.transform import Rotation,Slerp
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--bag',required=True);a=p.parse_args();root=Path(a.run);g=np.array(json.loads((root/'evaluation.json').read_text())['poses']);g=g[np.unique(g[:,0],return_index=True)[1]];rows=[];reader=rosbag2_py.SequentialReader();reader.open(rosbag2_py.StorageOptions(uri=a.bag,storage_id='mcap'),rosbag2_py.ConverterOptions('',''))
while reader.has_next():
 topic,data,t=reader.read_next()
 if topic!='/nav/body_odom':continue
 m=deserialize_message(data,Odometry);p=m.pose.pose.position;q=m.pose.pose.orientation;rows.append([m.header.stamp.sec+m.header.stamp.nanosec*1e-9,p.x,p.y,p.z,q.x,q.y,q.z,q.w])
b=np.array(rows);b=b[(b[:,0]>=g[0,0])&(b[:,0]<=g[-1,0])];i=np.unique(b[:,0],return_index=True)[1];b=b[i];truth=np.stack([np.interp(b[:,0],g[:,0],g[:,k]) for k in range(1,4)],1);r=Slerp(g[:,0],Rotation.from_quat(g[:,4:8]))([b[0,0]])[0]*Rotation.from_quat(b[0,4:8]).inv();translation=truth[0]-r.apply(b[0,1:4]);estimate=r.apply(b[:,1:4])+translation;error=np.linalg.norm(estimate-truth,axis=1)
map_file=root/'online-occupancy.npz'
if not map_file.exists():map_file=root/'measured-diagnostics.npz'
m=np.load(map_file);meta=m['metadata'];shape=tuple(int(x) for x in meta[4:7]);state=m['knowledge'].reshape(shape);ijk=np.argwhere(state==1);points=np.array(meta[:3])+(ijk+.5)*meta[3];points=r.apply(points)+translation
surfaces=[]
for model in ET.parse(root/'evaluation-world.sdf').findall('.//model'):
 pose=np.array([float(v) for v in model.findtext('pose','0 0 0 0 0 0').split()]);size=model.findtext('link/collision/geometry/box/size')
 if size is None:continue
 surfaces.append((model.get('name'),pose,np.array([float(v) for v in size.split()])))
dist=np.full(len(points),np.inf)
for name,pose,size in surfaces:
 local=Rotation.from_euler('xyz',pose[3:]).inv().apply(points-pose[:3]);q=np.abs(local)-size/2;sdf=np.linalg.norm(np.maximum(q,0.),axis=1)+np.minimum(np.max(q,axis=1),0.);dist=np.minimum(dist,np.abs(sdf))
def stats(x):return {} if not len(x) else {'count':len(x),'median_m':float(np.median(x)),'p90_m':float(np.percentile(x,90)),'rmse_m':float(np.sqrt(np.mean(x*x))),'within_15cm_fraction':float(np.mean(x<.15))}
result={'scope':'Offline fixed-initial-pose evaluation; true geometry and poses were unavailable to navigation. Local online occupancy, not a globally loop-optimized map.','alignment':'Initial simultaneous measured and true body poses only; no fitted trajectory/map alignment','odometry_ate':{'rmse_m':float(np.sqrt(np.mean(error**2))),'p90_m':float(np.percentile(error,90)),'max_m':float(error.max())},'occupied_surface_distance_all':stats(dist),'occupied_surface_distance_above_25cm':stats(dist[points[:,2]>.25]),'unknown_voxels':int((state==-1).sum()),'measured_free_voxels':int((state==0).sum()),'operator_verified_launch_voxels':int((state==-2).sum()),'coverage_warning':'Nearest-surface agreement is not map completeness or proof of safe unknown space.'};
if (root/'actual-optimized-structure.npy').exists() and (root/'measured-map-gauge.json').exists():
 gauge=json.loads((root/'measured-map-gauge.json').read_text());optimized=np.load(root/'actual-optimized-structure.npy');optimized=Rotation.from_quat(gauge['world_from_map_quaternion_xyzw']).apply(optimized)+gauge['world_from_map_translation'];optimized=r.apply(optimized)+translation;optimized_dist=np.full(len(optimized),np.inf)
 for name,pose,size in surfaces:
  local=Rotation.from_euler('xyz',pose[3:]).inv().apply(optimized-pose[:3]);q=np.abs(local)-size/2;sdf=np.linalg.norm(np.maximum(q,0.),axis=1)+np.minimum(np.max(q,axis=1),0.);optimized_dist=np.minimum(optimized_dist,np.abs(sdf))
 result['optimized_structure_surface_distance']=stats(optimized_dist)
 result['optimized_structure_alignment']='Saved first-measured SLAM/VIO gauge followed by initial true/measured alignment; no map fit'
(root/'mapping-evaluation.json').write_text(json.dumps(result,indent=2));np.savez_compressed(root/'offline-aligned-map.npz',points=points,surface_distance=dist,estimated_path=estimate,true_path=truth);print(json.dumps(result,indent=2))
