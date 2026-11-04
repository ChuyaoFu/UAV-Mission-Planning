#!/usr/bin/python3
"""Read only actual SLAM pose/health for RViz; never reads evaluation truth."""
import json
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker,MarkerArray
class View(Node):
 def __init__(self):
  super().__init__('slam_live_view',parameter_overrides=[Parameter('use_sim_time',value=True)]);self.health={};self.pose=None;self.mission='DISARMED / SENSOR CHECK';self.structure={}
  self.pub=self.create_publisher(MarkerArray,'/carto/view_markers',10)
  self.subs=[self.create_subscription(String,'/mapping/health',lambda m:setattr(self,'structure',json.loads(m.data)),10),self.create_subscription(String,'/carto/localization_health',lambda m:setattr(self,'health',json.loads(m.data)),10),self.create_subscription(PoseStamped,'/tracked_pose',lambda m:setattr(self,'pose',m),10),self.create_subscription(String,'/carto/mission_state',lambda m:setattr(self,'mission',m.data),10)];self.create_timer(.2,self.tick)
 def tick(self):
  h=self.health;t=Marker();t.header.frame_id='map';t.header.stamp=self.get_clock().now().to_msg();t.ns='health';t.id=0;t.type=Marker.TEXT_VIEW_FACING;t.action=Marker.ADD;t.pose.position.z=1.2;t.pose.position.x=.8;t.pose.position.y=.9;t.pose.orientation.w=1.;t.scale.z=.08;t.color.a=1.;t.color.r=0.2 if h.get('valid') else 1.;t.color.g=1. if h.get('valid') else .2;t.color.b=.6
  t.text='SLAM: '+str(h.get('reason','waiting'))+'\n'+self.mission+'\nvisual updates '+str(h.get('accepted_visual_update_features',0))+' | structure '+str(self.structure.get('structure_points',0))
  markers=[t]
  if self.pose:
   b=Marker();b.header=self.pose.header;b.ns='aircraft';b.id=1;b.type=Marker.ARROW;b.action=Marker.ADD;b.pose=self.pose.pose;b.scale.x=.28;b.scale.y=.08;b.scale.z=.08;b.color.r=1.;b.color.g=.65;b.color.b=.1;b.color.a=1.;markers.append(b)
  self.pub.publish(MarkerArray(markers=markers))
rclpy.init();n=View()
try:rclpy.spin(n)
finally:n.destroy_node();rclpy.shutdown()
