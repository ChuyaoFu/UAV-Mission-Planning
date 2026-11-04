#!/usr/bin/python3
"""Read-only post-flight sensor/map separation against independent CAD/truth.
Never publishes. Same original images and poses compare old/new stereo.
"""
import sys,json,time,copy
from pathlib import Path
import numpy as np,cv2,rosbag2_py
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation,Slerp
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from camera_geometry import load_rig
from stereo_quality import Matcher
from mapping_cloud import StructuralFilter
source=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=True,parents=True);bag=Path(sys.argv[3]);cv2.setNumThreads(2)
e=json.load(open(source/'evaluation-offline.json'));g=np.array(e['rows']['truth']);gp=g[:,1:4]+Rotation.from_quat(g[:,4:8]).apply([.12,0,.055]);grot=Slerp(g[:,0],Rotation.from_quat(g[:,4:8]));nodes=np.array(json.load(open(source/'trajectory.json'))['optimized']);r=Rotation.from_quat(g[0,4:8])*Rotation.from_quat(nodes[0,4:8]).inv();offset=gp[0]-r.apply(nodes[0,1:4]);nt=nodes[:,0];nrot=Slerp(nt,Rotation.from_quat(nodes[:,4:8]));cad=np.load('/work/cartographer/results/cad-grid.npz');indices=np.argwhere(cad['surface']);surface=indices*float(cad['res'])+cad['origin'];tree=cKDTree(surface)
K,T,B,resolution=load_rig();matcher=Matcher();f=StructuralFilter();frames={};raw={};old_structure={};results={k:[] for k in ['old_sensor','new_sensor','old_map','new_map']};qual=[];timing=[];last=-1.;selected=set()
reader=rosbag2_py.SequentialReader();reader.open(rosbag2_py.StorageOptions(uri=str(bag),storage_id='mcap'),rosbag2_py.ConverterOptions('',''));wanted=['/vio/left/image_raw','/vio/right/image_raw','/ov_msckf/odomimu','/carto/down_range','/mapping/structural_points'];types={x.name:get_message(x.type) for x in reader.get_all_topics_and_types() if x.name in wanted};reader.set_filter(rosbag2_py.StorageFilter(topics=wanted))
def stamp(m):return m.header.stamp.sec+m.header.stamp.nanosec*1e-9
def project(points,t,key):
 if not len(points) or not (max(g[0,0],nt[0])<=t<=min(g[-1,0],nt[-1])):return
 truth=np.array([np.interp(t,g[:,0],gp[:,i]) for i in range(3)]);world=grot([t])[0].apply(points)+truth;results[key+'_sensor'].append(world)
 pose=np.array([np.interp(t,nt,nodes[:,i]) for i in [1,2,3]]);mapped=nrot([t])[0].apply(points)+pose;results[key+'_map'].append(r.apply(mapped)+offset)
def drain():
 for key in list(frames):
  values=frames[key];t=key*1e-9
  if len(values)<2 or not f.history or f.history[-1][0]<t:continue
  left,right=values[0],values[1];a=np.ndarray((left.height,left.width,3),np.uint8,buffer=left.data,strides=(left.step,3,1)) if left.encoding in ['rgb8','bgr8'] else np.ndarray((left.height,left.width),np.uint8,buffer=left.data,strides=(left.step,1));b=np.ndarray((right.height,right.width,3),np.uint8,buffer=right.data,strides=(right.step,3,1)) if right.encoding in ['rgb8','bgr8'] else np.ndarray((right.height,right.width),np.uint8,buffer=right.data,strides=(right.step,1));a=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY) if a.ndim==3 else a;b=cv2.cvtColor(b,cv2.COLOR_RGB2GRAY) if b.ndim==3 else b
  depth,q=matcher.compute(a,b,K[0],B);v,u=np.mgrid[3:depth.shape[0]-3:6,3:depth.shape[1]-3:6];z=depth[v,u];p=np.stack(((u-K[2])*z/K[0],(v-K[3])*z/K[1],z),-1);p=p@T[:3,:3].T+T[:3,3];p=np.ascontiguousarray(p[np.isfinite(z)&(z>.25)&(z<8)],dtype='<f4')
  from sensor_msgs.msg import PointCloud2
  cloud=PointCloud2();cloud.header=left.header;cloud.point_step=12;cloud.data=p.tobytes();f.image(left);filtered=f.filter(cloud)
  if filtered is not None:project(f.last_structure,t,'new')
  if key in old_structure:project(old_structure.pop(key),t,'old')
  qual.append(dict(q,stamp=t,structure=0 if filtered is None else len(f.last_structure)));timing.append(q['compute_ms']);del frames[key]
while reader.has_next():
 topic,data,_=reader.read_next();m=deserialize_message(data,types[topic]);t=stamp(m);key=round(t*1e9)
 if topic.endswith('odomimu'):f.odom(m);drain()
 elif topic.endswith('down_range'):f.ranging(m)
 elif topic.endswith('structural_points'):
  if key in selected:old_structure[key]=np.frombuffer(m.data,dtype='<f4').reshape(-1,3).copy()
 elif '/image_raw' in topic:
  if '/left/' in topic:
   if t-last<.25:continue
   last=t;selected.add(key);frames.setdefault(key,{})[0]=m
  elif key in selected:frames.setdefault(key,{})[1]=m
  drain()
 # Old structure may arrive just after matching; defer all pair draining until
 # a later odometry event is no guarantee. Independently collect selected old.
 if topic.endswith('structural_points') and key in selected:
  project(np.frombuffer(m.data,dtype='<f4').reshape(-1,3),t,'old');old_structure.pop(key,None)
for k in results:
 p=np.vstack(results[k]) if results[k] else np.empty((0,3));_,ids=np.unique(np.floor(p/.04).astype(np.int32),axis=0,return_index=True);p=p[ids];results[k]=p
np.savez_compressed(out/'geometry-points.npz',**results)
def metric(p):
 if not len(p):return {'points':0}
 d,_=tree.query(p,workers=2);return dict(points=len(p),median_m=float(np.median(d)),p95_m=float(np.percentile(d,95)),over_20cm=float(np.mean(d>.2)))
# All optimized nodes use the actual FIRST initialized node's single gauge,
# including stationary initialization before the recorder started.
valid=(nt>=g[0,0])&(nt<=g[-1,0]);a=nodes[valid];truth=np.array([np.interp(a[:,0],g[:,0],gp[:,i]) for i in range(3)]).T;error=np.linalg.norm(r.apply(a[:,1:4])+offset-truth,axis=1)
summary={'geometry':{k:metric(v) for k,v in results.items()},'optimized_initial_node_gauge_rmse_m':float(np.sqrt(np.mean(error**2))),'optimized_initial_node_gauge_max_m':float(error.max()),'stereo_compute_ms':{'median':float(np.median(timing)),'p95':float(np.percentile(timing,95))},'frames':len(qual),'gauge':'first initialized optimized node aligned to recorded stationary start; no scale/trajectory fit; NOT reset at first structure-bearing airborne node','method':'Same original frames. sensor = independent true exposure poses, map = same optimized trajectory. CAD/truth only in this offline evaluator.'};(out/'geometry-audit.json').write_text(json.dumps(summary,indent=2));(out/'stereo-quality.json').write_text(json.dumps(qual));print(json.dumps(summary,indent=2))
