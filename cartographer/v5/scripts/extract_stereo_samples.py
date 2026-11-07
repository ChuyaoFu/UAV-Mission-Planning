#!/usr/bin/python3
"""Export actual stereo-only samples for isolated offline depth evaluation."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from cv_bridge import CvBridge
from camera_geometry import load_rig
from self_mask import self_masks
from stereo_quality import Matcher
p=argparse.ArgumentParser();p.add_argument('bag');p.add_argument('output');p.add_argument('--offsets',type=float,nargs='+',default=[15.,30.,45.,60.]);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);reader=rosbag2_py.SequentialReader();reader.open(rosbag2_py.StorageOptions(uri=a.bag,storage_id='mcap'),rosbag2_py.ConverterOptions('',''));types={x.name:get_message(x.type) for x in reader.get_all_topics_and_types()};reader.set_filter(rosbag2_py.StorageFilter(topics=['/vio/left/image_raw','/vio/right/image_raw']));buf=[{},{}];bridge=CvBridge();first=None;offsets=iter(sorted(a.offsets));target=next(offsets);K,T,baseline,res=load_rig();masks=self_masks();matcher=Matcher();manifest=[]
while reader.has_next():
 topic,data,receipt=reader.read_next();m=deserialize_message(data,types[topic]);key=m.header.stamp.sec*1000000000+m.header.stamp.nanosec;i=0 if '/left/' in topic else 1
 if first is None:first=key*1e-9
 if key*1e-9-first<target:continue
 buf[i][key]=m
 for old in list(buf[i]):
  if old<key-500000000:del buf[i][old]
 if key not in buf[1-i]:continue
 l,r=buf[0].pop(key),buf[1].pop(key);left=bridge.imgmsg_to_cv2(l,desired_encoding='mono8');right=bridge.imgmsg_to_cv2(r,desired_encoding='mono8');f=out/f'frame-{len(manifest):03d}.npz';measured,_=matcher.compute_scaled(left,right,K[0],baseline,masks,scale=.5);np.savez_compressed(f,left=left,right=right,stamp=key*1e-9,K=K,baseline=baseline,self_mask=masks[0],right_self_mask=masks[1],measured_stereo=measured);manifest.append({'file':f.name,'stamp':key*1e-9,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()});target=next(offsets,None)
 if target is None:break
assert manifest,'No synchronized measured stereo samples';(out/'manifest.json').write_text(json.dumps({'source':'Recorded raw stereo and rectified calibration only; no poses, CAD, truth depth or free-space prior','frames':manifest},indent=2));print(json.dumps(manifest,indent=2))
