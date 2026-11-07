"""Lazy native Jazzy reader adapter; imports ROS2 only when explicitly selected."""
from types import SimpleNamespace
class NativeReader:
 def __init__(self,paths,**unused):
  if len(paths)!=1:raise ValueError('One native bag URI required')
  self.path=str(paths[0])
 def __enter__(self):
  import rosbag2_py
  self.reader=rosbag2_py.SequentialReader()
  self.reader.open(rosbag2_py.StorageOptions(uri=self.path,storage_id=''),rosbag2_py.ConverterOptions('',''))
  self.connections=[SimpleNamespace(topic=t.name,msgtype=t.type) for t in self.reader.get_all_topics_and_types()]
  return self
 def messages(self,connections):
  import rosbag2_py
  selected={c.topic:c for c in connections};self.reader.set_filter(rosbag2_py.StorageFilter(topics=list(selected)))
  while self.reader.has_next():
   topic,data,stamp=self.reader.read_next()
   if topic in selected:yield selected[topic],stamp,data
 def deserialize(self,raw,msgtype):
  from rclpy.serialization import deserialize_message
  from sensor_msgs.msg import Image,Imu
  types={'sensor_msgs/msg/Image':Image,'sensor_msgs/msg/Imu':Imu}
  if msgtype not in types:raise ValueError('Unsupported native type: '+msgtype)
  return deserialize_message(raw,types[msgtype])
 def __exit__(self,*unused):self.reader=None
