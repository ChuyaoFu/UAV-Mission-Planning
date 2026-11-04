#!/usr/bin/python3
"""Read actual bridged messages and save measurable sensor evidence."""
import json
import time
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu, CameraInfo
from cv_bridge import CvBridge

rclpy.init()
node = Node('sensor_check')
bridge = CvBridge()
counts = defaultdict(int)
stamps = defaultdict(list)
info = {}
out = Path('/work/results/sensor-check')
out.mkdir(parents=True, exist_ok=True)

def callback(kind, msg):
    counts[kind] += 1
    stamps[kind].append(msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9)
    if counts[kind] != 1:
        return
    if isinstance(msg, Image):
        im = bridge.imgmsg_to_cv2(msg)
        info[kind] = {'size': [msg.width, msg.height], 'encoding': msg.encoding, 'frame': msg.header.frame_id}
        if kind == 'depth':
            valid = im[np.isfinite(im) & (im>0)]
            info[kind]['range_m'] = [float(valid.min()), float(valid.max())] if len(valid) else []
        else:
            gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) if im.ndim == 3 else im
            pts = cv2.goodFeaturesToTrack(gray, 300, .01, 8)
            info[kind]['corners'] = 0 if pts is None else len(pts)
            cv2.imwrite(str(out/f'{kind}.png'), cv2.cvtColor(im, cv2.COLOR_RGB2BGR) if im.ndim == 3 else im)
    elif isinstance(msg, Imu):
        info[kind] = {'frame': msg.header.frame_id, 'accel': [msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z]}
    else:
        info[kind] = {'k': list(msg.k), 'frame': msg.header.frame_id}

subs=[]
for kind, topic, typ in [('left','/vio/left/image_raw',Image),('right','/vio/right/image_raw',Image),('depth','/vio/depth/image',Image),('imu','/vio/imu',Imu),('camera_info','/vio/left/camera_info',CameraInfo)]:
    subs.append(node.create_subscription(typ, topic, lambda m,k=kind: callback(k,m), qos_profile_sensor_data))
start=time.monotonic()
while time.monotonic()-start<12:
    rclpy.spin_once(node, timeout_sec=.1)
for k,v in stamps.items():
    if len(v)>1:
        info[k]['rate_sim_hz']=(len(v)-1)/(v[-1]-v[0])
info['counts']=dict(counts)
if stamps['left'] and stamps['right']:
    right=set(stamps['right'])
    info['stereo_exact_stamp_fraction']=sum(x in right for x in stamps['left'])/len(stamps['left'])
(out/'report.json').write_text(json.dumps(info,indent=2))
print(json.dumps(info,indent=2))
node.destroy_node()
rclpy.shutdown()
if not all(counts[k]>20 for k in ['left','right','depth','imu']):
    raise SystemExit('Missing or stalled sensor stream')
