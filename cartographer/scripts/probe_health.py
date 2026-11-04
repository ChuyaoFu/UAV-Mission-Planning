#!/usr/bin/python3
import argparse,json,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=3);p.add_argument('--output',required=True);a=p.parse_args()
rclpy.init();n=Node('health_probe');states=[];cloud=[]
subs=[n.create_subscription(String,'/carto/localization_health',lambda m:states.append(json.loads(m.data)),100),n.create_subscription(PointCloud2,'/carto/points',lambda m:cloud.append(m.header.stamp.sec+m.header.stamp.nanosec*1e-9),100)]
end=time.monotonic()+a.seconds
while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.05)
r={'health_samples':len(states),'valid_samples':sum(bool(h['valid']) for h in states),'reasons':sorted(set(h['reason'] for h in states)),'cloud_samples':len(cloud),'last_health':states[-1] if states else None,'probe_wall_seconds':a.seconds};Path(a.output).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2));n.destroy_node();rclpy.shutdown()
