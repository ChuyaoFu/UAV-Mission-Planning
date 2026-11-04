#!/usr/bin/python3
"""Finite-FOV passive stereo diagnostic, not an exact DepthAI firmware emulator."""
import json,time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
class Stereo(Node):
 def __init__(self):
  super().__init__('passive_stereo',parameter_overrides=[Parameter('use_sim_time',value=True)])
  cv2.setNumThreads(2);self.bridge=CvBridge();self.buf=[{},{}];self.count=0
  self.match=cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=5,P1=200,P2=800,disp12MaxDiff=1,uniquenessRatio=12,speckleWindowSize=80,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
  self.pub=self.create_publisher(Image,'/stereo/depth/image',5);self.health=self.create_publisher(String,'/stereo/health',10)
  self.subs=[self.create_subscription(Image,f'/vio/{side}/image_raw',lambda m,i=i:self.cb(m,i),qos_profile_sensor_data) for i,side in enumerate(['left','right'])]
 def cb(self,m,i):
  key=m.header.stamp.sec*1000000000+m.header.stamp.nanosec;self.buf[i][key]=m
  for k in list(self.buf[i]):
   if k<key-1000000000:del self.buf[i][k]
  if key not in self.buf[1-i]:return
  left=self.buf[0].pop(key);right=self.buf[1].pop(key)
  a=self.bridge.imgmsg_to_cv2(left,desired_encoding='mono8');b=self.bridge.imgmsg_to_cv2(right,desired_encoding='mono8')
  disp=self.match.compute(a,b).astype(np.float32)/16
  depth=np.full_like(disp,np.nan);good=disp>0;depth[good]=(a.shape[1]/(2*np.tan(.7))*.075)/disp[good]
  depth[(depth<.25)|(depth>8)]=np.nan
  out=self.bridge.cv2_to_imgmsg(depth,encoding='32FC1');out.header=left.header;out.header.frame_id='depth_optical';self.pub.publish(out)
  h={'stamp':key*1e-9,'valid_fraction':float(np.isfinite(depth).mean()),'image_std':float(a.std()),'mean_luminance':float(a.mean()),'exact_sync':True,'source':'opencv_passive_stereo','model_limitation':'not OAK firmware; ideal pinhole images'}
  self.health.publish(String(data=json.dumps(h)));self.count+=1
  if self.count%100==0:self.get_logger().info(json.dumps(h))
rclpy.init();node=Stereo()
try:rclpy.spin(node)
finally:node.destroy_node();rclpy.shutdown()
