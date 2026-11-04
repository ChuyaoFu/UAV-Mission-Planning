#!/usr/bin/python3
"""Read-only evaluator. Truth is used here exclusively for offline statistics."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Range
ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=float,default=30);ap.add_argument('--output',required=True);args=ap.parse_args()
rclpy.init();n=Node('carto_readonly_evaluation',parameter_overrides=[Parameter('use_sim_time',value=True)])
rows={'odom':[],'truth':[],'depth':[],'stereo':[],'map':[],'health':[],'range':[]}
def sec(h):return h.stamp.sec+h.stamp.nanosec*1e-9
def tf(m):
 for t in m.transforms:
  if t.header.frame_id=='carto_odom' and t.child_frame_id=='vio_imu':
   p=t.transform.translation;q=t.transform.rotation;rows['odom'].append([sec(t.header),p.x,p.y,p.z,q.x,q.y,q.z,q.w])
def gt(m):
 for t in m.transforms:
  if t.child_frame_id=='x500_vio_0':
   p=t.transform.translation;q=t.transform.rotation;rows['truth'].append([sec(t.header),p.x,p.y,p.z,q.x,q.y,q.z,q.w])
def pose(m):
 p=m.pose.position;q=m.pose.orientation;rows['map'].append([sec(m.header),p.x,p.y,p.z,q.x,q.y,q.z,q.w])
subs=[n.create_subscription(String,'/carto/localization_health',lambda m:rows['health'].append(json.loads(m.data)),100),n.create_subscription(Range,'/carto/down_range',lambda m:rows['range'].append([sec(m.header),m.range]),100),n.create_subscription(TFMessage,'/tf',tf,100),n.create_subscription(TFMessage,'/evaluation/poses',gt,100),n.create_subscription(PoseStamped,'/tracked_pose',pose,100),n.create_subscription(String,'/carto/depth_health',lambda m:rows['depth'].append(json.loads(m.data)),100),n.create_subscription(String,'/stereo/health',lambda m:rows['stereo'].append(json.loads(m.data)),100)]
end=time.monotonic()+args.seconds
while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.05)
summary={k+'_samples':len(v) for k,v in rows.items()}
for k in ['odom','map']:
 a=np.array(rows[k])
 if len(a):summary[k+'_position_span_m']=np.ptp(a[:,1:4],axis=0).tolist();summary[k+'_duration_s']=float(a[-1,0]-a[0,0])
for k in ['depth','stereo']:
 if rows[k]:summary[k+'_mean_valid_fraction']=float(np.mean([v['valid_fraction'] for v in rows[k]]))
a=np.array(rows['odom']);g=np.array(rows['truth'])
if len(a)>5 and len(g)>5:
 # Initial rigid alignment only. Pose of IMU includes known .15,0,.08 lever arm.
 gp=g[:,1:4]+Rotation.from_quat(g[:,4:8]).apply([.15,0,.08]);rot=Rotation.from_quat(g[0,4:8])*Rotation.from_quat(a[0,4:8]).inv()
 interp=np.stack([np.interp(a[:,0],g[:,0],gp[:,i]) for i in range(3)],axis=1)
 estimate=rot.apply(a[:,1:4]-a[0,1:4])+gp[0];err=np.linalg.norm(estimate-interp,axis=1)
 summary.update(ate_rmse_m=float(np.sqrt(np.mean(err**2))),ate_max_m=float(err.max()),alignment='fixed initial pose; no fitted trajectory alignment')
Path(args.output).write_text(json.dumps({'summary':summary,'rows':rows},indent=2));print(json.dumps(summary,indent=2));n.destroy_node();rclpy.shutdown()
