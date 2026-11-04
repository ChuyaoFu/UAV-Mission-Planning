#!/usr/bin/python3
"""Save actual SLAM results for review. Never loaded into a subsequent flight."""
import argparse,json,time
from pathlib import Path
import numpy as np
import cv2,rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
from cartographer_ros_msgs.srv import WriteState
ap=argparse.ArgumentParser();ap.add_argument('output');a=ap.parse_args();out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
rclpy.init();n=Node('export_actual_slam',parameter_overrides=[Parameter('use_sim_time',value=True)]);state={}
q=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
sub=n.create_subscription(OccupancyGrid,'/map',lambda m:state.update(map=m),q);client=n.create_client(WriteState,'/write_state')
if not client.wait_for_service(timeout_sec=5):raise SystemExit('SLAM write_state unavailable')
request=WriteState.Request();request.filename=str(out/'cartographer.pbstream');request.include_unfinished_submaps=True;future=client.call_async(request);rclpy.spin_until_future_complete(n,future,timeout_sec=15)
if not future.done():raise SystemExit('Map save timed out')
response=future.result();print(response)
end=time.monotonic()+5
while 'map' not in state and time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.1)
if 'map' in state:
 m=state['map'];grid=np.array(m.data,dtype=np.int16).reshape(m.info.height,m.info.width);gray=np.full_like(grid,205,dtype=np.uint8);gray[grid==0]=254;known=grid>0;gray[known]=np.clip(254-grid[known]*2.54,0,254).astype(np.uint8);cv2.imwrite(str(out/'slam-occupancy.png'),np.flipud(gray))
 p=m.info.origin.position;o=m.info.origin.orientation;metadata={'source':'actual Cartographer 3D submap occupancy projection, not CAD','resolution':m.info.resolution,'width':m.info.width,'height':m.info.height,'origin_xyz':[p.x,p.y,p.z],'origin_xyzw':[o.x,o.y,o.z,o.w],'frame':m.header.frame_id,'stamp_s':m.header.stamp.sec+m.header.stamp.nanosec*1e-9,'note':'3D-to-2D projection includes floor; not a 2D navigation cost map'};(out/'map-metadata.json').write_text(json.dumps(metadata,indent=2))
n.destroy_node();rclpy.shutdown()
