#!/usr/bin/python3
"""Bounded, sensor-only relative guideline perception; never corrects pose."""
import os,json
from collections import deque
import numpy as np,rclpy,message_filters
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image,Range
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from scipy.spatial.transform import Rotation
from cv_bridge import CvBridge
from camera_geometry import load_rig
from robot_geometry import TOF_IN_BODY
from frames import PoseHistory
from relative_guideline import GuidelineDetector
from perception_worker import LatestWorker
class Guidelines(Node):
 def __init__(self):
  super().__init__('relative_guideline_perception',parameter_overrides=[Parameter('use_sim_time',value=os.environ.get('USE_SIM_TIME','1')=='1')]);self.history=PoseHistory();self.ranges=deque(maxlen=100);self.pending=deque(maxlen=2);self.K=load_rig()[0];self.bridge=CvBridge();self.detector=GuidelineDetector();self.last=-1.;self.worker=LatestWorker(self.compute);self.pub=self.create_publisher(String,'/nav/guideline_hint',2)
  self.subs=[self.create_subscription(Odometry,'/nav/body_odom',self.body,10),self.create_subscription(Range,'/carto/down_range',lambda m:self.ranges.append(m),10)];self.images=[message_filters.Subscriber(self,kind,topic,qos_profile=qos_profile_sensor_data) for kind,topic in [(Image,'/vio/left/image_raw'),(Image,'/nav/measured_depth'),(PoseStamped,'/nav/camera_pose')]];self.sync=message_filters.TimeSynchronizer(self.images,5);self.sync.registerCallback(self.frames);self.create_timer(.02,self.publish)
 def stamp(self,m):return m.header.stamp.sec+m.header.stamp.nanosec*1e-9
 def body(self,m):
  p=m.pose.pose.position;q=m.pose.pose.orientation;self.history.append(self.stamp(m),[p.x,p.y,p.z],[q.x,q.y,q.z,q.w]);self.drain()
 def frames(self,*messages):self.pending.append(messages);self.drain()
 def drain(self):
  while self.pending:
   image,depth,camera=self.pending[0];t=self.stamp(image)
   if not self.history.items or t>self.history.items[-1][0]:return
   self.pending.popleft()
   if t-self.last<.19 or not self.ranges:continue
   body=self.history.sample(t);range_m=min(self.ranges,key=lambda m:abs(self.stamp(m)-t))
   if body is None or abs(self.stamp(range_m)-t)>.08 or not range_m.min_range<=range_m.range<=range_m.max_range:continue
   p,r=body;floor=(p+r.apply(TOF_IN_BODY))[2]-range_m.range*r.as_matrix()[2,2];self.last=t;self.worker.submit('lines',(image,depth,camera,float(floor)))
 def compute(self,kind,data):
  image,depth,camera,floor=data;p=camera.pose.position;q=camera.pose.orientation;a=self.bridge.imgmsg_to_cv2(image,desired_encoding='mono8');d=self.bridge.imgmsg_to_cv2(depth,desired_encoding='32FC1');segments=self.detector.detect(a,d,self.K,np.array([p.x,p.y,p.z]),Rotation.from_quat([q.x,q.y,q.z,q.w]),floor);return [{'stamp':self.stamp(image),'segments':segments,'floor_z_measured':floor,'source':'relative image/stereo/ToF geometry only; not a pose or free-space correction'}]
 def publish(self):
  for result in self.worker.take():self.pub.publish(String(data=json.dumps(result)))
  if self.worker.error:self.get_logger().error(self.worker.error);self.worker.error=None
if __name__=='__main__':
 import cv2
 cv2.setNumThreads(1);rclpy.init();n=Guidelines()
 try:rclpy.spin(n)
 finally:n.worker.close();n.destroy_node();rclpy.shutdown()
