#!/usr/bin/python3
"""Actual map and measured-state visualization; no scene or truth inputs."""
import json
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String,Int8MultiArray
from visualization_msgs.msg import Marker,MarkerArray
from nav_msgs.msg import Odometry
class View(Node):
 def __init__(self):
  super().__init__('cadfree_live_view',parameter_overrides=[Parameter('use_sim_time',value=True)]);self.health={};self.phase='DISARMED';self.counts={};self.pose=None;self.p=self.create_publisher(MarkerArray,'/carto/view_markers',1)
  self.subs=[self.create_subscription(String,'/carto/localization_health',lambda m:setattr(self,'health',json.loads(m.data)),5),self.create_subscription(String,'/carto/mission_state',lambda m:setattr(self,'phase',m.data),5),self.create_subscription(Odometry,'/nav/body_odom',lambda m:setattr(self,'pose',m.pose.pose),5),self.create_subscription(Int8MultiArray,'/grid_map/knowledge',self.map,1)];self.create_timer(.5,self.draw)
 def map(self,m):
  import numpy as np
  a=np.asarray(m.data,dtype=np.int8);self.counts={'free':int((a==0).sum()),'occupied':int((a==1).sum()),'unknown':int((a==-1).sum()),'launch':int((a==-2).sum()),'inferred':int((a==-3).sum())}
 def draw(self):
  t=Marker();t.header.frame_id='world';t.header.stamp=self.get_clock().now().to_msg();t.id=0;t.type=Marker.TEXT_VIEW_FACING;t.action=Marker.ADD;t.pose.position.z=1.8;t.pose.orientation.w=1.;t.scale.z=.055;t.color.a=1.;t.color.g=1. if self.health.get('valid') else .2;t.color.r=.2 if self.health.get('valid') else 1.;t.color.b=.6;t.text='Online measured map\n'+self.phase+'\n'+self.health.get('reason','waiting')+'\nmeasured free '+str(self.counts.get('free',0))+' / occupied '+str(self.counts.get('occupied',0))+'\ninferred_free '+str(self.counts.get('inferred',0))+'\nunknown '+str(self.counts.get('unknown',0))+' | explicit launch '+str(self.counts.get('launch',0));t.text=t.text.replace(' ','_');markers=[t]
  if self.pose:
   p=Marker();p.header=t.header;p.id=1;p.type=Marker.ARROW;p.action=Marker.ADD;p.pose=self.pose;p.scale.x=.35;p.scale.y=.08;p.scale.z=.08;p.color.a=1.;p.color.r=1.;p.color.g=.7;markers.append(p)
  self.p.publish(MarkerArray(markers=markers))
rclpy.init();n=View()
try:rclpy.spin(n)
finally:n.destroy_node();rclpy.shutdown()
