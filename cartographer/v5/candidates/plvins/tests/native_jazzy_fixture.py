#!/usr/bin/env python3
"""Run under native ROS2 Jazzy. Creates a small synthetic version9 MCAP fixture.
No metadata edit, version spoofing, pose, truth or estimator output is involved.
"""
import argparse,array,json,gc
from pathlib import Path
import rosbag2_py
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image,Imu
from std_msgs.msg import String
import yaml
p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
if a.output.exists():raise ValueError('Fixture URI must not exist')
w=rosbag2_py.SequentialWriter();w.open(rosbag2_py.StorageOptions(uri=str(a.output),storage_id='mcap'),rosbag2_py.ConverterOptions('',''))
for i,(name,kind) in enumerate([('/vio/left/image_raw','sensor_msgs/msg/Image'),('/vio/imu','sensor_msgs/msg/Imu'),('/excluded','std_msgs/msg/String')],1):
 w.create_topic(rosbag2_py.TopicMetadata(id=i,name=name,type=kind,serialization_format='cdr'))
for i in range(76):
 ns=10**9+i*4000000;m=Imu();m.header.stamp.sec=ns//10**9;m.header.stamp.nanosec=ns%10**9;m.header.frame_id='measured_imu'
 m.angular_velocity.x=.1;m.angular_velocity.y=.2;m.angular_velocity.z=.3;m.linear_acceleration.z=9.81
 m.orientation.x=.2;m.orientation.y=.3;m.orientation.z=.4;m.orientation.w=.5
 w.write('/vio/imu',serialize_message(m),ns)
 if i%10==0:
  im=Image();im.header.stamp=m.header.stamp;im.header.frame_id='left_optical';im.width=6;im.height=4;im.encoding='mono8';im.step=8;im.data=array.array('B',range(32));w.write('/vio/left/image_raw',serialize_message(im),ns+1000)
# This payload is deliberately not a serialized String, so exporting it would fail.
w.write('/excluded',b'forbidden test payload',1400000000)
w=None;gc.collect()
info=yaml.safe_load((a.output/'metadata.yaml').read_text())['rosbag2_bagfile_information']
assert info['version']==9 and info['storage_identifier']=='mcap',info
print(json.dumps({'generator':'native rosbag2_py Jazzy SequentialWriter','metadata_version':info['version'],'storage':info['storage_identifier'],'image_count':8,'imu_count':76,'output':str(a.output),'synthetic_fixture_only':True},indent=2))
