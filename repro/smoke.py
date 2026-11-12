#!/usr/bin/python3
"""Disarmed end-to-end sensor, ROS, PX4 and mapper connectivity smoke test."""
import json,subprocess,time
from collections import Counter
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy,HistoryPolicy
from sensor_msgs.msg import Imu,Image,Range
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from px4_msgs.msg import VehicleStatus
from cartographer_ros_msgs.srv import GetTrajectoryStates
rclpy.init();node=Node('handoff_disarmed_smoke');counts=Counter();stamps={'left':set(),'right':set()};armed=[False];clock=[]
# Best-effort image delivery may lag during Mesa startup. Keep enough samples
# for this observer and wait for the actual pairing criterion before exiting.
sensor_qos=QoSProfile(depth=50,reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST)
def stereo_sync():
 den=min(len(stamps['left']),len(stamps['right']))
 return len(stamps['left']&stamps['right'])/max(1,den)
def cb(key,m):
 counts[key]+=1
 if key in stamps:stamps[key].add(m.header.stamp.sec*1000000000+m.header.stamp.nanosec)
 if key=='status' and m.arming_state==2:armed[0]=True
 if key=='clock':clock.append(m.clock.sec+m.clock.nanosec*1e-9)
subs=[]
for key,topic,typ in [('left','/vio/left/image_raw',Image),('right','/vio/right/image_raw',Image),
 ('imu','/vio/imu',Imu),('range','/carto/down_range',Range),('odom','/ov_msckf/odomimu',Odometry),
 ('status','/fmu/out/vehicle_status_v1',VehicleStatus),('clock','/clock',Clock)]:
 subs.append(node.create_subscription(typ,topic,lambda m,k=key:cb(k,m),sensor_qos))
client=node.create_client(GetTrajectoryStates,'/get_trajectory_states')
start=time.monotonic();future=None;mapper=False
while time.monotonic()-start<55:
 rclpy.spin_once(node,timeout_sec=.05)
 if future is None and client.service_is_ready():future=client.call_async(GetTrajectoryStates.Request())
 if future is not None and future.done():
  res=future.result();mapper=res is not None and res.status.code==0
 if armed[0]:raise RuntimeError('Unexpected arming during a disarmed smoke test')
 if time.monotonic()-start>18 and all(counts[k]>=5 for k in ['left','right','range','odom','status','clock']) and counts['imu']>=100 and mapper and stereo_sync()>.95:break
sync=stereo_sync()
px='/work/src/PX4-Autopilot/build/px4_sitl_default/bin'
gps=subprocess.check_output([px+'/px4-listener','sensor_gps','-n','1'],text=True)
status=subprocess.check_output([px+'/px4-commander','status'],text=True)
passed=all(counts[k]>=5 for k in ['left','right','range','odom','status','clock']) and counts['imu']>=100 and mapper and sync>.95 and len(clock)>1 and clock[-1]>clock[0] and 'never published' in gps and 'Disarmed' in status
report={'passed':passed,'counts':dict(counts),'stereo_matched_fraction':sync,'observation_seconds':time.monotonic()-start,'mapper_service_responded':mapper,'gnss_never_published':'never published' in gps,'disarmed':'Disarmed' in status,'scope':'connectivity and sensor smoke only; no simulated flight commanded'}
out=Path('/work/results/reproduction');out.mkdir(parents=True,exist_ok=True);(out/'smoke.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
node.destroy_node();rclpy.shutdown();raise SystemExit(not passed)
