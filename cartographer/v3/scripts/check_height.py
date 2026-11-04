import sys,numpy as np,rosbag2_py
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Range
from scipy.spatial.transform import Rotation
r=rosbag2_py.SequentialReader();r.open(rosbag2_py.StorageOptions(uri='/work/cartographer/v3/data/'+sys.argv[1],storage_id='mcap'),rosbag2_py.ConverterOptions('',''));r.set_filter(rosbag2_py.StorageFilter(topics=['/carto/local_pose','/carto/down_range']));height=None;prev=None;ran=None;last=-1
while r.has_next():
 topic,data,_=r.read_next()
 if topic.endswith('down_range'):ran=deserialize_message(data,Range).range+.025;continue
 m=deserialize_message(data,PoseStamped);q=m.pose.orientation;z=m.pose.position.z-Rotation.from_quat([q.x,q.y,q.z,q.w]).apply([.12,0,.055])[2];t=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
 if ran is None:continue
 if height is None:height=ran;prev=z
 predicted=height+z-prev;innovation=ran-predicted;accept=abs(innovation)<.08
 if accept:predicted+=.12*innovation
 height=predicted;prev=z
 if 119<t<131 and t-last>.4:print(round(t,2),'range',round(ran,3),'filtered',round(height,3),'accepted',accept);last=t
