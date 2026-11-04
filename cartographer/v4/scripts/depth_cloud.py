#!/usr/bin/python3
"""Forward depth -> finite PointCloud2; exposes coverage and surface-normal rank.
ideal: explicit ideal rendering baseline. stereo: actual OpenCV stereo matching.
Neither stream consumes evaluation poses, GPS or previous-flight maps.
"""
import os,json,time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image,PointCloud2,PointField
from std_msgs.msg import String
from camera_geometry import load_rig
class DepthCloud(Node):
 def __init__(self):
  super().__init__('depth_cloud',parameter_overrides=[Parameter('use_sim_time',value=True)])
  self.pub=self.create_publisher(PointCloud2,'/carto/points',5)
  self.health=self.create_publisher(String,'/carto/depth_health',10)
  self.sub=self.create_subscription(Image,os.environ.get('DEPTH_TOPIC','/vio/depth/image'),self.cb,qos_profile_sensor_data)
  self.K,self.T,self.baseline,self.resolution=load_rig()
  self.last=-1;self.count=0;self.first=time.monotonic()
 def cb(self,m):
  ts=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
  if ts<=self.last:return
  self.last=ts
  if m.encoding not in ('32FC1','16UC1'):raise ValueError(m.encoding)
  dtype=np.dtype('>f4' if m.is_bigendian else '<f4') if m.encoding=='32FC1' else np.dtype('>u2' if m.is_bigendian else '<u2')
  d=np.ndarray((m.height,m.width),dtype=dtype,buffer=m.data,strides=(m.step,dtype.itemsize)).astype(np.float32)
  if m.encoding=='16UC1':d*=.001
  assert (m.width,m.height)==self.resolution,'Depth resolution/calibration mismatch'
  f,fy,cx,cy=self.K;v,u=np.mgrid[3:m.height-3:4,3:m.width-3:4]
  z=d[v,u];optical=np.stack(((u-cx)*z/f,(v-cy)*z/fy,z),-1)
  p=optical@self.T[:3,:3].T+self.T[:3,3]
  good=np.isfinite(z)&(z>.25)&(z<8)
  with np.errstate(invalid='ignore'):
   normals=np.cross(p[1:-1,2:]-p[1:-1,:-2],p[2:,1:-1]-p[:-2,1:-1])
  local_good=good[1:-1,1:-1]&good[1:-1,2:]&good[1:-1,:-2]&good[2:,1:-1]&good[:-2,1:-1]
  with np.errstate(invalid='ignore'):
   smooth=(np.abs(z[1:-1,2:]-z[1:-1,:-2])<.15)&(np.abs(z[2:,1:-1]-z[:-2,1:-1])<.15)
  normals[~(local_good&smooth)]=0
  n=normals.reshape(-1,3);length=np.linalg.norm(n,axis=1)
  n=n[np.isfinite(length)&(length>1e-5)]
  if len(n):
   n/=np.linalg.norm(n,axis=1)[:,None];eig=np.linalg.eigvalsh(n.T@n/len(n));rank=float(eig[0]/max(eig[-1],1e-9))
  else:eig=np.zeros(3);rank=0.
  # This is a conservative depth geometry indicator, not SLAM covariance.
  health={'stamp':ts,'source':os.environ.get('DEPTH_SOURCE','ideal_rendered_depth'),'valid_fraction':float(good.mean()),'points':int(good.sum()),'normal_eigenvalues':eig.tolist(),'normal_rank_ratio':rank,'usable':bool(good.mean()>.15 and good.sum()>300),'geometry_full_rank':bool(rank>.015),'frame':'vio_imu'}
  self.health.publish(String(data=json.dumps(health)))
  points=np.ascontiguousarray(p[good],dtype='<f4')
  out=PointCloud2(header=m.header,height=1,width=len(points),fields=[PointField(name=k,offset=4*i,datatype=PointField.FLOAT32,count=1) for i,k in enumerate('xyz')],is_bigendian=False,point_step=12,row_step=12*len(points),data=points.tobytes(),is_dense=True)
  out.header.frame_id='vio_imu'
  if len(points):self.pub.publish(out)
  self.count+=1
  if self.count%100==0:self.get_logger().info(json.dumps(health))
rclpy.init();node=DepthCloud()
try:rclpy.spin(node)
finally:node.destroy_node();rclpy.shutdown()
