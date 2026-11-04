#!/usr/bin/python3
"""Evaluate rendered-image VIO against Gazebo truth, without feeding it back."""
import argparse
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry

ap=argparse.ArgumentParser()
ap.add_argument('--seconds',type=float,default=120)
ap.add_argument('--label',default='unspecified')
args=ap.parse_args()
rclpy.init()
node=Node('independent_vio_evaluation')
truth=[]
estimate=[]
frames=set()
def stamp(m): return m.header.stamp.sec+m.header.stamp.nanosec*1e-9
def truth_cb(msg):
    transforms={m.child_frame_id:m for m in msg.transforms}
    frames.update(transforms)
    candidates=[k for k in transforms if k.endswith('vio_rig')]
    if not candidates: return
    key=candidates[0]
    def world(k, visited=None):
        visited=set() if visited is None else visited
        if k not in transforms: return np.eye(4)
        if k in visited: raise ValueError('Cyclic truth transforms')
        visited.add(k)
        m=transforms[k]
        p=m.transform.translation; q=m.transform.rotation
        a=np.eye(4)
        a[:3,:3]=Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
        a[:3,3]=[p.x,p.y,p.z]
        return world(m.header.frame_id,visited)@a
    a=world(key)
    truth.append([stamp(transforms[key]),*a[:3,3].tolist()])
def vio_cb(m):
    p=m.pose.pose.position
    estimate.append([stamp(m),p.x,p.y,p.z])
subs=[node.create_subscription(TFMessage,'/evaluation/poses',truth_cb,qos_profile_sensor_data),node.create_subscription(Odometry,'/ov_msckf/odomimu',vio_cb,qos_profile_sensor_data)]
start=time.monotonic()
while time.monotonic()-start<args.seconds:
    rclpy.spin_once(node,timeout_sec=.05)
out=Path('/work/results/vio-evaluation');out.mkdir(parents=True,exist_ok=True)
(out/'raw.json').write_text(json.dumps({'truth':truth,'estimate':estimate,'truth_frames':sorted(frames)}))
node.destroy_node();rclpy.shutdown()
if len(truth)<20 or len(estimate)<20:
    raise SystemExit(f'Insufficient data: truth={len(truth)}, estimate={len(estimate)}, frames={sorted(frames)}')
gt=np.asarray(truth);est=np.asarray(estimate)
if not np.isfinite(gt).all() or not np.isfinite(est).all():
    failure={'label':args.label,'passed':False,'reason':'Non-finite trajectory samples','nonfinite_truth_rows':int((~np.isfinite(gt).all(axis=1)).sum()),'nonfinite_vio_rows':int((~np.isfinite(est).all(axis=1)).sum())}
    (out/'metrics.json').write_text(json.dumps(failure,indent=2));print(json.dumps(failure,indent=2))
    raise SystemExit('Invalid VIO output; evaluation failed')
gt=gt[np.r_[True,np.diff(gt[:,0])>0]]
est=est[np.r_[True,np.diff(est[:,0])>0]]
est=est[(est[:,0]>=gt[0,0])&(est[:,0]<=gt[-1,0])]
ref=np.column_stack([np.interp(est[:,0],gt[:,0],gt[:,i]) for i in range(1,4)])
pred=est[:,1:]
# One rigid SE(3) alignment, never a scale fit. Used for reporting only.
x=pred-pred.mean(axis=0);y=ref-ref.mean(axis=0)
u,s,vt=np.linalg.svd(x.T@y)
d=np.eye(3);d[2,2]=np.linalg.det(u@vt)
r=u@d@vt
aligned=x@r+ref.mean(axis=0)
error=np.linalg.norm(aligned-ref,axis=1)
path_length=float(np.linalg.norm(np.diff(ref,axis=0),axis=1).sum())
metrics={'duration_s':float(est[-1,0]-est[0,0]),'samples':len(est),'truth_path_length_m':path_length,'ate_rmse_m':float(np.sqrt(np.mean(error**2))),'ate_p95_m':float(np.percentile(error,95)),'max_error_m':float(error.max()),'max_output_gap_s':float(np.diff(est[:,0]).max()),'alignment':'one rigid SE3, no scale; evaluation only','moving_path':path_length>1.0}
metrics['label']=args.label
metrics['passed']=metrics['moving_path'] and metrics['ate_rmse_m']<.1 and metrics['max_error_m']<.3 and metrics['max_output_gap_s']<.2
(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
fig,axs=plt.subplots(1,2,figsize=(11,4))
axs[0].plot(ref[:,0],ref[:,1],label='Gazebo truth (IMU)')
axs[0].plot(aligned[:,0],aligned[:,1],label='OpenVINS, rigid aligned')
axs[0].axis('equal');axs[0].set_xlabel('x (m)');axs[0].set_ylabel('y (m)');axs[0].legend()
axs[1].plot(est[:,0]-est[0,0],error);axs[1].set_xlabel('Simulation time (s)');axs[1].set_ylabel('Position error (m)')
fig.tight_layout();fig.savefig(out/'trajectory.png',dpi=150)
print(json.dumps(metrics,indent=2))
