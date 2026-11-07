#!/usr/bin/python3
"""Export a few actual monocular-left/stereo-depth pairs; no reference data."""
import argparse,json
from pathlib import Path
import numpy as np,cv2,rosbag2_py
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from cv_bridge import CvBridge
p=argparse.ArgumentParser();p.add_argument('bag');p.add_argument('output');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);r=rosbag2_py.SequentialReader();r.open(rosbag2_py.StorageOptions(uri=a.bag,storage_id='mcap'),rosbag2_py.ConverterOptions('',''));images={};depths={};bridge=CvBridge();last=-1.;count=0;z=None;initial=None
while r.has_next():
 topic,data,t=r.read_next()
 if topic=='/nav/body_odom':
  m=deserialize_message(data,Odometry);z=m.pose.pose.position.z
  if initial is None:initial=z
  continue
 if topic not in ['/vio/left/image_raw','/stereo/depth/image']:continue
 m=deserialize_message(data,Image);stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
 if topic.endswith('image_raw'):images[stamp]=m
 else:depths[stamp]=m
 for k in list(images):
  if k<stamp-1.:images.pop(k,None)
 for k in list(depths):
  if k<stamp-1.:depths.pop(k,None)
 common=sorted(set(images)&set(depths))
 if not common or z is None or z-initial<.40:continue
 k=common[-1]
 if k-last<2.:continue
 gray=bridge.imgmsg_to_cv2(images[k],desired_encoding='mono8');rgb=np.repeat(gray[:,:,None],3,axis=2);depth=bridge.imgmsg_to_cv2(depths[k],desired_encoding='32FC1');np.savez_compressed(out/('frame-%02d.npz'%count),image=rgb,measured_stereo=depth,stamp=k);last=k;count+=1
 if count>=8:break
print(json.dumps({'input_pairs':count,'source':'recorded monochrome stereo-left images and actual passive matching; no scene, pose, or reference depth input','directory':str(out)},indent=2))
