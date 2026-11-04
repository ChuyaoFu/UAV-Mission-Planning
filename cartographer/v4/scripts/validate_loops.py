#!/usr/bin/python3
"""Independent post-flight validation of accepted inter-submap constraints.
Uses recorded truth only here, never in mapping, navigation or map rendering.
"""
import sys,json,struct,gzip,subprocess
from pathlib import Path
import numpy as np,rosbag2_py
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation,Slerp
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
root=Path('/work/cartographer/v4');label=sys.argv[1];out=root/'results'/label;source=root/'carto_ws/src/cartographer';generated=root/'generated_proto';generated.mkdir(exist_ok=True)
if not (generated/'cartographer/mapping/proto/serialization_pb2.py').exists():
 files=[str(p.relative_to(source)) for p in (source/'cartographer').rglob('*.proto')];subprocess.run(['protoc','-I',str(source),'-I','/usr/include','--python_out='+str(generated),*files],cwd=source,check=True)
sys.path.insert(0,str(generated));from cartographer.mapping.proto.serialization_pb2 import SerializationHeader,SerializedData
with (out/'slam/cartographer.pbstream').open('rb') as f:
 assert struct.unpack('<Q',f.read(8))[0]==0x7b1d1f7b5bf501db
 def record():
  n=struct.unpack('<Q',f.read(8))[0];return gzip.decompress(f.read(n))
 header=SerializationHeader();header.ParseFromString(record());graph=None
 for _ in range(3):
  m=SerializedData();m.ParseFromString(record())
  if m.WhichOneof('data')=='pose_graph':graph=m.pose_graph;break
assert graph is not None
constraints=[c for c in graph.constraint if c.tag==1];intra={}
for c in graph.constraint:
 if c.tag==0:intra.setdefault((c.submap_id.trajectory_id,c.submap_id.submap_index),set()).add((c.node_id.trajectory_id,c.node_id.node_index))
times={}
for trajectory in graph.trajectory:
 for node in trajectory.node:times[(trajectory.trajectory_id,node.node_index)]=node.timestamp/1e7-62135596800.
evaluation=json.loads((out/'evaluation-offline.json').read_text());g=np.array(evaluation['rows']['truth']);slerp=Slerp(g[:,0],Rotation.from_quat(g[:,4:8]));needed=set()
for c in constraints:needed.add((c.node_id.trajectory_id,c.node_id.node_index));needed.update(intra[(c.submap_id.trajectory_id,c.submap_id.submap_index)])
r=rosbag2_py.SequentialReader();r.open(rosbag2_py.StorageOptions(uri=str(root/'data'/label),storage_id='mcap'),rosbag2_py.ConverterOptions('',''));r.set_filter(rosbag2_py.StorageFilter(topics=['/mapping/structural_points']));frames=[]
while r.has_next():
 _,data,_=r.read_next();m=deserialize_message(data,PointCloud2)
 if m.width:frames.append((m.header.stamp.sec+m.header.stamp.nanosec*1e-9,np.frombuffer(m.data,dtype='<f4').reshape(-1,3).copy()))
frame_times=np.array([x[0] for x in frames]);world={}
for key in needed:
 t=times[key];i=int(np.argmin(abs(frame_times-t)))
 if abs(frame_times[i]-t)>.06:continue
 actual=frame_times[i];rotation=slerp(actual);body=np.array([np.interp(actual,g[:,0],g[:,j]) for j in [1,2,3]]);world[key]=rotation.apply(frames[i][1]+[.12,0,.055])+body
results=[]
for c in constraints:
 key=(c.node_id.trajectory_id,c.node_id.node_index);sub=(c.submap_id.trajectory_id,c.submap_id.submap_index);members=intra[sub];clouds=[world[k] for k in members if k in world];item={'node_id':list(key),'submap_id':list(sub),'node_stamp_s':times[key],'submap_node_time_span_s':[min(times[k] for k in members),max(times[k] for k in members)],'method':'actual structural points transformed using independent recorded body truth; map poses are not used in this overlap check'}
 if key not in world or not clouds:item.update(verified=False,reason='Insufficient original point records')
 else:
  reference=np.vstack(clouds);distance,_=cKDTree(reference).query(world[key]);overlap=float(np.mean(distance<.12));median=float(np.median(distance));item.update(node_points=len(world[key]),reference_points=len(reference),median_independent_overlap_distance_m=median,fraction_within_12cm=overlap,verified=bool(overlap>.5 and median<.12))
 results.append(item)
result={'accepted_inter_submap_constraints':len(constraints),'all_independent_overlap_checks_passed':all(x['verified'] for x in results),'checks':results,'scope':'post-flight validation of this run; not a guarantee of recall or arbitrary-scene loop detection','truth_input_to_runtime':False};(out/'loop-independent-validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
