#!/usr/bin/python3
"""Rectified camera-info and rigid calibration for RTAB-Map. No pose prior."""
import os,copy
import numpy as np,rclpy
from scipy.spatial.transform import Rotation
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image,CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from camera_geometry import load_rig
class Frames(Node):
 def __init__(self):
  super().__init__('rtab_measured_calibration',parameter_overrides=[Parameter('use_sim_time',value=os.environ.get('USE_SIM_TIME','1')=='1')]);self.K,self.T,self.baseline,self.resolution=load_rig();self.tf=StaticTransformBroadcaster(self);self.pub=[];self.subs=[];self.last=[-1.,-1.]
  transforms=[]
  for i,side in enumerate(['left','right']):
   self.pub.append((self.create_publisher(Image,'/rtab/'+side+'/image_rect',2),self.create_publisher(CameraInfo,'/rtab/'+side+'/camera_info',2)));self.subs.append(self.create_subscription(Image,'/vio/'+side+'/image_raw',lambda m,i=i:self.image(m,i),qos_profile_sensor_data));t=TransformStamped();t.header.frame_id='vio_imu';t.child_frame_id=side+'_optical';p=self.T[:3,3].copy()
   if i==1:p+=self.T[:3,0]*self.baseline
   t.transform.translation.x=float(p[0]);t.transform.translation.y=float(p[1]);t.transform.translation.z=float(p[2]);q=Rotation.from_matrix(self.T[:3,:3]).as_quat();t.transform.rotation.x=float(q[0]);t.transform.rotation.y=float(q[1]);t.transform.rotation.z=float(q[2]);t.transform.rotation.w=float(q[3]);transforms.append(t)
  self.tf.sendTransform(transforms)
 def image(self,m,i):
  t=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
  publish_image=t-self.last[i]>=.09
  if publish_image:self.last[i]=t
  m.header.frame_id=('left' if i==0 else 'right')+'_optical';f,fy,cx,cy=self.K;c=CameraInfo();c.header=m.header;c.width=self.resolution[0];c.height=self.resolution[1];c.distortion_model='plumb_bob';c.d=[0.]*5;c.k=[f,0.,cx,0.,fy,cy,0.,0.,1.];c.r=np.eye(3).ravel().tolist();c.p=[f,0.,cx,(-f*self.baseline if i else 0.),0.,fy,cy,0.,0.,0.,1.,0.];self.pub[i][1].publish(c)
  if publish_image:self.pub[i][0].publish(m)
rclpy.init();n=Frames()
try:rclpy.spin(n)
finally:n.destroy_node();rclpy.try_shutdown()
