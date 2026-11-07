#!/usr/bin/python3
"""Actual measured structural cloud, reprojected with current optimized SLAM poses.
A display/export node only. No CAD, navigation prior or evaluation input.
"""
from collections import deque
from frames import PoseHistory
from nav_msgs.msg import Odometry
import numpy as np,rclpy
from scipy.spatial.transform import Rotation,Slerp
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
from sensor_msgs.msg import PointCloud2,PointField
from cartographer_ros_msgs.srv import TrajectoryQuery
class View(Node):
 def __init__(self):
  super().__init__('optimized_structure_map_view',parameter_overrides=[Parameter('use_sim_time',value=__import__('os').environ.get('USE_SIM_TIME','1')=='1')]);self.pending=deque(maxlen=8);self.history=PoseHistory();self.last_keyframe=None;self.clouds=deque(maxlen=600);self.future=None;self.points=None;self.client=self.create_client(TrajectoryQuery,'/trajectory_query');qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE);self.pub=self.create_publisher(PointCloud2,'/mapping/global_cloud',qos);self.sub=self.create_subscription(PointCloud2,'/mapping/structural_points',self.cloud,20);self.odom=self.create_subscription(Odometry,'/nav/body_odom',self.body,10);self.timer=self.create_timer(1.,self.tick)
 def body(self,m):
  p=m.pose.pose.position;q=m.pose.pose.orientation;self.history.append(m.header.stamp.sec+m.header.stamp.nanosec*1e-9,[p.x,p.y,p.z],[q.x,q.y,q.z,q.w]);self.drain()
 def cloud(self,m):self.pending.append(m);self.drain()
 def drain(self):
  while self.pending:
   m=self.pending[0];stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
   if not self.history.items or stamp>self.history.items[-1][0]:return
   self.pending.popleft();self.accept_cloud(m)
 def accept_cloud(self,m):
  if m.point_step!=12 or not m.width:return
  stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9;sample=self.history.sample(stamp)
  if sample is None:return
  p,r=sample
  if self.last_keyframe is not None:
   pp,rr=self.last_keyframe
   if np.linalg.norm(p-pp)<.06 and (r*rr.inv()).magnitude()<.08:return
  self.last_keyframe=(p.copy(),r);self.clouds.append((stamp,np.frombuffer(m.data,dtype='<f4').reshape(-1,3)[::max(1,m.width//800)].copy()))
 def tick(self):
  if self.future is not None or not self.client.service_is_ready() or not self.clouds:return
  request=TrajectoryQuery.Request();request.trajectory_id=0;self.future=self.client.call_async(request);self.future.add_done_callback(self.build)
 def build(self,future):
  self.future=None
  try:response=future.result()
  except Exception as error:self.get_logger().warning(str(error));return
  if response.status.code!=0 or len(response.trajectory)<2:return
  poses=response.trajectory;t=np.array([p.header.stamp.sec+p.header.stamp.nanosec*1e-9 for p in poses]);ids=np.unique(t,return_index=True)[1];t=t[ids];poses=[poses[i] for i in ids];xyz=np.array([[p.pose.position.x,p.pose.position.y,p.pose.position.z] for p in poses]);q=np.array([[p.pose.orientation.x,p.pose.orientation.y,p.pose.orientation.z,p.pose.orientation.w] for p in poses]);clouds=[c for c in self.clouds if t[0]<=c[0]<=t[-1]]
  if not clouds:return
  times=np.array([c[0] for c in clouds]);translations=np.array([np.interp(times,t,xyz[:,i]) for i in range(3)]).T;rotations=Slerp(t,Rotation.from_quat(q))(times);points=np.vstack([r.apply(c[1])+p for r,p,c in zip(rotations,translations,clouds)]);_,ids=np.unique(np.floor(points/.04).astype(np.int32),axis=0,return_index=True);points=np.ascontiguousarray(points[ids],dtype='<f4');self.points=points
  m=PointCloud2();m.header.frame_id='map';m.header.stamp=self.get_clock().now().to_msg();m.height=1;m.width=len(points);m.fields=[PointField(name=name,offset=i*4,datatype=PointField.FLOAT32,count=1) for i,name in enumerate(['x','y','z'])];m.point_step=12;m.row_step=12*len(points);m.is_dense=True;m.data=points.tobytes();self.pub.publish(m)
rclpy.init();node=View()
try:rclpy.spin(node)
finally:node.destroy_node();rclpy.shutdown()
