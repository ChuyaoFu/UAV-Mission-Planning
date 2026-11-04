#!/usr/bin/python3
"""Observable stereo structure for SLAM, with a small measured floor sample.
No CAD, ground truth, GPS or synthetic points. Live safety retains /carto/points.
"""
import json,math,copy
from collections import deque
import cv2,numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
class StructuralFilter:
 def __init__(self):
  cv2.setNumThreads(1);self.rotation=None;self.prev_z=None;self.height=None;self.range=None;self.images={};self.pitch=Rotation.from_euler('y',30,degrees=True).as_matrix();self.health={};self.last_structure=None;self.bootstrap_left=5
 def odom(self,m):
  p=m.pose.pose.position;q=m.pose.pose.orientation
  self.rotation=Rotation.from_quat([q.x,q.y,q.z,q.w]);z=p.z
  if self.height is None and self.range is not None:self.height=self.measurement()
  if self.height is not None:
   if self.prev_z is not None:self.height+=z-self.prev_z
   if self.range is not None:
    measurement=self.measurement()
    if abs(measurement-self.height)<.08:self.height+=.10*(measurement-self.height)
  self.prev_z=z
 def measurement(self):return self.range*self.rotation.as_matrix()[2,2]+self.rotation.apply([.12,0,.08])[2]
 def ranging(self,m):self.range=m.range
 def image(self,m):
  if m.encoding=='mono8':gray=np.ndarray((m.height,m.width),np.uint8,buffer=m.data,strides=(m.step,1))
  else:
   image=np.ndarray((m.height,m.width,3),np.uint8,buffer=m.data,strides=(m.step,3,1));gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY if m.encoding=='rgb8' else cv2.COLOR_BGR2GRAY)
  gradient=cv2.boxFilter(abs(cv2.Sobel(gray,cv2.CV_32F,1,0)),cv2.CV_32F,(5,5));t=m.header.stamp.sec*1000000000+m.header.stamp.nanosec;self.images[t]=gradient
  for k in list(self.images):
   if k<t-1000000000:del self.images[k]
 def filter(self,m):
  t=m.header.stamp.sec*1000000000+m.header.stamp.nanosec
  if self.height is None or self.rotation is None or t not in self.images:return None
  points=np.frombuffer(m.data,dtype='<f4').reshape(-1,3);camera=(points-[0,.0375,0])@self.pitch;distance=np.linalg.norm(camera,axis=1);uv=np.rint(np.stack([320-camera[:,1]*379.917386/camera[:,0],200-camera[:,2]*379.917386/camera[:,0]],axis=1)).astype(int);gradient=self.images[t];inside=(uv[:,0]>=0)&(uv[:,0]<gradient.shape[1])&(uv[:,1]>=0)&(uv[:,1]<gradient.shape[0]);texture=np.zeros(len(points));texture[inside]=gradient[uv[inside,1],uv[inside,0]];height=points@self.rotation.as_matrix()[2,:]+self.height
  structure=(height>.12)&(height<2.4)&(distance<3.5)&(distance>.3)&(texture>2.)
  p=points[structure]
  if len(p)>10:
   d,_=cKDTree(p).query(p,k=5);p=p[d[:,-1]<.14]
  floor=points[(abs(height)<.05)&(distance<1.3)]
  if len(floor)>40:floor=floor[np.linspace(0,len(floor)-1,40).astype(int)]
  self.health={'stamp':t*1e-9,'raw_points':len(points),'structure_points':len(p),'floor_points':len(floor),'height_m':float(self.height),'usable':len(p)>=60,'source':'actual passive depth, RGB texture, VIO attitude and downward ToF; no CAD/truth'}
  if len(p)<60:
   if self.bootstrap_left>0 and self.height<.25 and len(floor)>=20:
    self.bootstrap_left-=1;self.last_structure=np.empty((0,3),dtype='<f4');return np.ascontiguousarray(floor,dtype='<f4')
   return None
  self.last_structure=np.ascontiguousarray(p,dtype='<f4')
  return np.ascontiguousarray(np.vstack([p,floor]),dtype='<f4')
if __name__=='__main__':
 import rclpy
 from rclpy.node import Node
 from rclpy.parameter import Parameter
 from rclpy.qos import qos_profile_sensor_data
 from sensor_msgs.msg import PointCloud2,Image,Range
 from nav_msgs.msg import Odometry
 from std_msgs.msg import String
 class Mapping(Node):
  def __init__(self):
   super().__init__('measured_structure_cloud',parameter_overrides=[Parameter('use_sim_time',value=True)]);self.f=StructuralFilter();self.p=self.create_publisher(PointCloud2,'/mapping/points',5);self.s=self.create_publisher(PointCloud2,'/mapping/structural_points',5);self.h=self.create_publisher(String,'/mapping/health',10);self.subs=[self.create_subscription(Odometry,'/ov_msckf/odomimu',self.f.odom,100),self.create_subscription(Range,'/carto/down_range',self.f.ranging,10),self.create_subscription(Image,'/vio/left/image_raw',self.f.image,qos_profile_sensor_data),self.create_subscription(PointCloud2,'/carto/points',self.cloud,qos_profile_sensor_data)]
  def cloud(self,m):
   p=self.f.filter(m)
   if self.f.health:self.h.publish(String(data=json.dumps(self.f.health)))
   if p is None:return
   m.width=len(p);m.row_step=len(p)*12;m.data=p.tobytes();self.p.publish(m)
   s=copy.deepcopy(m);p=self.f.last_structure;s.width=len(p);s.row_step=len(p)*12;s.data=p.tobytes();self.s.publish(s)
 rclpy.init();node=Mapping()
 try:rclpy.spin(node)
 finally:node.destroy_node();rclpy.shutdown()
