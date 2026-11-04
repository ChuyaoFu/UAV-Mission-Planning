#!/usr/bin/python3
"""25-ray bounded downward ToF proxy for the official VL53L1X, not added lidar.
This geometric/noise proxy omits reflectance, crosstalk and real ToF firmware.
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan,Range
from px4_msgs.msg import DistanceSensor
class DownRange(Node):
 def __init__(self):
  super().__init__('downward_tof',parameter_overrides=[Parameter('use_sim_time',value=True)])
  self.pub=self.create_publisher(Range,'/carto/down_range',10);self.px=self.create_publisher(DistanceSensor,'/fmu/in/distance_sensor',10)
  self.sub=self.create_subscription(LaserScan,'/carto/down_tof/scan',self.cb,qos_profile_sensor_data)
 def cb(self,m):
  a=np.asarray(m.ranges);a=a[np.isfinite(a)&(a>=.05)&(a<=4.)]
  if not len(a):return
  value=float(np.min(a));r=Range(header=m.header,radiation_type=Range.INFRARED,field_of_view=.471238,min_range=.05,max_range=4.,range=value);self.pub.publish(r)
  stamp=m.header.stamp.sec*1000000+m.header.stamp.nanosec//1000
  self.px.publish(DistanceSensor(timestamp=stamp,device_id=0xCA01,min_distance=.05,max_distance=4.,current_distance=value,variance=.0004,signal_quality=100,type=DistanceSensor.MAV_DISTANCE_SENSOR_LASER,h_fov=.471238,v_fov=.471238,q=[1.,0.,0.,0.],orientation=25,mode=DistanceSensor.MODE_ENABLED))
rclpy.init();n=DownRange()
try:rclpy.spin(n)
finally:n.destroy_node();rclpy.shutdown()
