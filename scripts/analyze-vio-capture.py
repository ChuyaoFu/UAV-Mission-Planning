#!/usr/bin/python3
"""Compare raw sensor timing/kinematics and both estimator output paths.

Ground truth is used only by this offline diagnostic, never by VIO or PX4.
"""
from collections import Counter
import json
from pathlib import Path
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation,Slerp
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/work');label='vio-raw-diagnostic-01';out=root/'results'/label
reader=rosbag2_py.SequentialReader()
reader.open(rosbag2_py.StorageOptions(uri=str(root/'data'/label),storage_id='mcap'),rosbag2_py.ConverterOptions('cdr','cdr'))
types={x.name:get_message(x.type) for x in reader.get_all_topics_and_types()}
data={k:[] for k in ['left','right','imu','truth','pose','odom','msckf','slam']}
def stamp(h):return h.stamp.sec+h.stamp.nanosec*1e-9
def pq(m):
 p=m.position;q=m.orientation
 return [p.x,p.y,p.z,q.x,q.y,q.z,q.w]
while reader.has_next():
 topic,raw,_=reader.read_next()
 if not (topic.startswith('/vio/') or topic.startswith('/ov_msckf/') or topic=='/evaluation/poses'):continue
 m=deserialize_message(raw,types[topic])
 if topic in ['/vio/left/image_raw','/vio/right/image_raw']:
  data['left' if '/left/' in topic else 'right'].append(stamp(m.header))
 elif topic=='/vio/imu':
  data['imu'].append([stamp(m.header),m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z,m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z])
 elif topic=='/evaluation/poses':
  frames={x.child_frame_id:x for x in m.transforms}
  candidates=[k for k in frames if k.endswith('vio_rig')]
  if not candidates:continue
  def world(k):
   if k not in frames:return np.eye(4)
   x=frames[k];p=x.transform.translation;q=x.transform.rotation
   a=np.eye(4);a[:3,:3]=Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix();a[:3,3]=[p.x,p.y,p.z]
   return world(x.header.frame_id)@a
  k=candidates[0];a=world(k)
  data['truth'].append([stamp(frames[k].header),*a[:3,3],*Rotation.from_matrix(a[:3,:3]).as_quat()])
 elif topic in ['/ov_msckf/poseimu','/ov_msckf/odomimu']:
  k='pose' if topic.endswith('/poseimu') else 'odom'
  data[k].append([stamp(m.header),*pq(m.pose.pose)])
 elif topic in ['/ov_msckf/points_msckf','/ov_msckf/points_slam']:
  data['msckf' if topic.endswith('msckf') else 'slam'].append([stamp(m.header),m.width*m.height])
for k,v in data.items():
 a=np.asarray(v)
 if len(a):
  t=a if a.ndim==1 else a[:,0]
  a=a[np.r_[True,np.diff(t)>0]]
 data[k]=a
np.savez_compressed(out/'sensor-diagnostic.npz',**data)
def intervals(a):
 t=a if a.ndim==1 else a[:,0];dt=np.diff(t)
 return {'count':len(t),'rate_sim_hz':float((len(t)-1)/(t[-1]-t[0])),'max_gap_s':float(dt.max()),'interval_counts_ms':dict(Counter(np.round(dt*1000,3)))}
report={'streams':{k:intervals(data[k]) for k in ['left','right','imu','pose','odom']}}
report['stereo_exact_fraction']=sum(x in set(data['right']) for x in data['left'])/len(data['left'])
used=[]
for t in data['left']:
 if not used or t>=used[-1]+1/20:used.append(t)
report['source_throttle_accepts_at_20hz']={'count':len(used),'fraction':len(used)/len(data['left'])}
g=data['truth'];imu=data['imu'];imu=imu[(imu[:,0]>=g[0,0])&(imu[:,0]<=g[-1,0])]
R=Slerp(g[:,0],Rotation.from_quat(g[:,4:8]))
# Rigid orientation+translation from the initial VIO state only, then hold it
# fixed. No time-varying alignment or scale fitting can conceal subsequent drift.
e=data['pose'];e=e[(e[:,0]>=g[0,0])&(e[:,0]<=g[-1,0])]
align=R([e[0,0]]).as_matrix()[0]@Rotation.from_quat(e[0,4:8]).as_matrix().T
offset=np.array([np.interp(e[0,0],g[:,0],g[:,j]) for j in range(1,4)])-align@e[0,1:4]
errors={};estimated={}
for key in ['pose','odom']:
 e=data[key];e=e[(e[:,0]>=g[0,0])&(e[:,0]<=g[-1,0])]
 truth=np.column_stack([np.interp(e[:,0],g[:,0],g[:,j]) for j in range(1,4)])
 est=e[:,1:4]@align.T+offset
 err=np.linalg.norm(est-truth,axis=1)
 estimated[key]=(e[:,0],est)
 errors[key]=(e[:,0],err)
 report[key+'_initial_alignment_error']={'rmse_m':float(np.sqrt(np.mean(err**2))),'max_m':float(err.max()),'peak_s':float(e[err.argmax(),0])}
 # Adjacent state jumps: a large update at low true speed indicates a correction,
 # rather than physical motion or an evaluation interpolation artifact.
 jump=np.linalg.norm(np.diff(est,axis=0),axis=1)
 top=np.argsort(jump)[-5:][::-1]
 report[key+'_largest_adjacent_steps']=[{'time_s':float(e[i+1,0]),'step_m':float(jump[i]),'dt_s':float(e[i+1,0]-e[i,0])} for i in top]
f=json.loads((out/'baseline-flight.json').read_text())
arm=f['arm_stamp_us']*1e-6;land=f['land_command_stamp_us']*1e-6
# At 20 Hz, differentiate a smoothed truth trajectory for an independent IMU
# check. Exclude the floor impact, where finite differences are not reliable.
gp=savgol_filter(g[:,1:4],9,3,axis=0)
gv=np.gradient(gp,g[:,0],axis=0)
ga=np.gradient(gv,g[:,0],axis=0)
t=imu[:,0];gravity=np.array([0,0,-9.8])
aw=R(t).apply(imu[:,4:7])+gravity
ag=np.column_stack([np.interp(t,g[:,0],ga[:,j]) for j in range(3)])
vgt=np.column_stack([np.interp(t,g[:,0],gv[:,j]) for j in range(3)])
mask=(t>=arm+.4)&(t<=land+1.8)
report['imu_vs_truth_preimpact']={'mean_acceleration_residual_world_mss':np.mean((aw-ag)[mask],axis=0).tolist(),'rms_acceleration_residual_world_mss':np.sqrt(np.mean(((aw-ag)[mask])**2,axis=0)).tolist()}
tt=t[mask];a=aw[mask];v=vgt[mask]
integrated=v[0]+cumulative_trapezoid(a,tt,axis=0,initial=0)
report['imu_truth_orientation_integrated_velocity_error_end_world_ms']=(integrated[-1]-v[-1]).tolist()
fig,axes=plt.subplots(4,1,figsize=(11,12),sharex=True)
for j,c in enumerate(['tab:blue','tab:orange','tab:green']):
 axes[0].plot(g[:,0],g[:,j+1],color=c,label='truth '+['x','y','z'][j])
 axes[0].plot(*[estimated['pose'][0],estimated['pose'][1][:,j]],color=c,ls='--')
axes[0].set_ylabel('World position (m)');axes[0].legend(ncol=3)
for k,(ts,err) in errors.items():axes[1].plot(ts,err,label=k)
axes[1].set_ylabel('Position error (m)');axes[1].legend()
for j in range(3):axes[2].plot(tt,integrated[:,j]-v[:,j],label=['x','y','z'][j])
axes[2].set_ylabel('IMU-integrated velocity\nminus truth (m/s)');axes[2].legend()
for k in ['msckf','slam']:
 if len(data[k]):axes[3].plot(data[k][:,0],data[k][:,1],label=k)
axes[3].set_ylabel('Published feature count');axes[3].legend();axes[3].set_xlabel('Simulation time (s)')
for ax in axes:
 ax.axvline(arm,color='gray',ls=':');ax.axvline(land,color='red',ls=':');ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(out/'sensor-diagnostic.png',dpi=150)
(out/'sensor-diagnostic.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
