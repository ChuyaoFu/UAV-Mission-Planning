#!/usr/bin/python3
import sys
from pathlib import Path
import cv2,rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
label=sys.argv[1];out=Path('/work/cartographer/v2/results')/label/'images';out.mkdir(exist_ok=True)
r=rosbag2_py.SequentialReader();r.open(rosbag2_py.StorageOptions(uri='/work/cartographer/v2/data/'+label,storage_id='mcap'),rosbag2_py.ConverterOptions('',''));r.set_filter(rosbag2_py.StorageFilter(topics=['/vio/left/image_raw']));last=-1;b=CvBridge()
while r.has_next():
 _,data,ts=r.read_next();m=deserialize_message(data,Image);t=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
 if t-last<1.:continue
 last=t;a=b.imgmsg_to_cv2(m,'mono8');cv2.imwrite(str(out/f'left-{t:.1f}.png'),a)
print(out)
