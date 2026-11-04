#!/usr/bin/python3
import time,json
from pathlib import Path
import cv2,numpy as np,rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String
rclpy.init();n=Node('inspect_v2');bridge=CvBridge();data={};root=Path('/work/cartographer/v4/results')
def image_cb(m):
 a=bridge.imgmsg_to_cv2(m,'mono8');cv2.imwrite(str(root/'left-current.png'),a);p=cv2.goodFeaturesToTrack(a,250,.01,8);data['image']={'std':float(a.std()),'features':0 if p is None else len(p),'encoding':m.encoding}
def odom(m):data['odom']={'stamp':m.header.stamp.sec+m.header.stamp.nanosec*1e-9,'position_variance':np.array(m.pose.covariance).reshape(6,6).diagonal().tolist()}
subs=[n.create_subscription(Image,'/vio/left/image_raw',image_cb,qos_profile_sensor_data),n.create_subscription(Odometry,'/ov_msckf/odomimu',odom,20),n.create_subscription(String,'/carto/localization_health',lambda m:data.update(health=json.loads(m.data)),20)]
end=time.monotonic()+3
while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.05)
(root/'sensor-inspection.json').write_text(json.dumps(data,indent=2));print(json.dumps(data,indent=2));n.destroy_node();rclpy.shutdown()
