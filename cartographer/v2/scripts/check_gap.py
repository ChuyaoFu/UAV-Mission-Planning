import sys,numpy as np,rosbag2_py
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
r=rosbag2_py.SequentialReader();r.open(rosbag2_py.StorageOptions(uri='/work/cartographer/v2/data/'+sys.argv[1],storage_id='mcap'),rosbag2_py.ConverterOptions('',''));r.set_filter(rosbag2_py.StorageFilter(topics=['/ov_msckf/odomimu']));last=-1
while r.has_next():
 _,data,_=r.read_next();m=deserialize_message(data,Odometry);t=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
 if t<91 or t>99 or t-last<.5:continue
 last=t;print(round(t,3),'poscov',np.array(m.pose.covariance).reshape(6,6).diagonal()[:3].round(6),'velcov',np.array(m.twist.covariance).reshape(6,6).diagonal()[:3].round(6))
