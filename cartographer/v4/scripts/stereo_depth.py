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
from camera_geometry import load_rig
from stereo_quality import Matcher
class Stereo(Node):
 def __init__(self):
  super().__init__('passive_stereo',parameter_overrides=[Parameter('use_sim_time',value=True)])
  cv2.setNumThreads(2);self.bridge=CvBridge();self.buf=[{},{}];self.count=0;self.last_processed=-1.
  self.match=Matcher();self.K,self.T,self.baseline,self.resolution=load_rig()
  self.pub=self.create_publisher(Image,'/stereo/depth/image',5);self.health=self.create_publisher(String,'/stereo/health',10)
  self.subs=[self.create_subscription(Image,f'/vio/{side}/image_raw',lambda m,i=i:self.cb(m,i),qos_profile_sensor_data) for i,side in enumerate(['left','right'])]
 def cb(self,m,i):
  key=m.header.stamp.sec*1000000000+m.header.stamp.nanosec;self.buf[i][key]=m
  for k in list(self.buf[i]):
   if k<key-1000000000:del self.buf[i][k]
  if key not in self.buf[1-i]:return
  left=self.buf[0].pop(key);right=self.buf[1].pop(key)
  if key*1e-9-self.last_processed<.09:return
  self.last_processed=key*1e-9
  a=self.bridge.imgmsg_to_cv2(left,desired_encoding='mono8');b=self.bridge.imgmsg_to_cv2(right,desired_encoding='mono8')
  assert a.shape[::-1]==self.resolution,'Image resolution does not match rectified calibration'
  depth,quality=self.match.compute(a,b,self.K[0],self.baseline)
  out=self.bridge.cv2_to_imgmsg(depth,encoding='32FC1');out.header=left.header;out.header.frame_id='depth_optical';self.pub.publish(out)
  h={'stamp':key*1e-9,'valid_fraction':float(np.isfinite(depth).mean()),'image_std':float(a.std()),'mean_luminance':float(a.mean()),'exact_sync':True,'max_processing_hz':10.,'end_to_end_age_s':self.get_clock().now().nanoseconds*1e-9-key*1e-9,'source':'opencv_passive_stereo','model_limitation':'not OAK firmware; ideal pinhole images'}
  h.update(quality);self.health.publish(String(data=json.dumps(h)));self.count+=1
  if self.count%100==0:self.get_logger().info(json.dumps(h))
rclpy.init();node=Stereo()
try:rclpy.spin(node)
finally:node.destroy_node();rclpy.shutdown()
