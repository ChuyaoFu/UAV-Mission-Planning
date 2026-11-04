#!/usr/bin/python3
import sys,json,time
from pathlib import Path
import numpy as np,rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
out=Path(sys.argv[1]);out.mkdir(exist_ok=True,parents=True);rclpy.init();n=Node('capture_actual_structure_map');state={};q=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE);sub=n.create_subscription(PointCloud2,'/mapping/global_cloud',lambda m:state.update(cloud=m),q);end=time.monotonic()+5
while 'cloud' not in state and time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.1)
if 'cloud' not in state:raise RuntimeError('Actual optimized map cloud unavailable')
m=state['cloud'];p=np.frombuffer(m.data,dtype='<f4').reshape(-1,3).copy();np.savez_compressed(out/'actual-structure-map.npz',points=p)
with (out/'actual-structure-map.ply').open('w') as f:
 f.write(f'ply\nformat ascii 1.0\nelement vertex {len(p)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n');np.savetxt(f,p,fmt='%.5f')
(out/'structure-map-metadata.json').write_text(json.dumps({'points':len(p),'frame':m.header.frame_id,'stamp_s':m.header.stamp.sec+m.header.stamp.nanosec*1e-9,'source':'actual passive stereo structure, reprojected using current optimized Cartographer trajectory','voxel_selection_m':.04,'synthetic_points':False,'cad_or_truth_input':False},indent=2));print('saved actual structure map',len(p));n.destroy_node();rclpy.shutdown()
