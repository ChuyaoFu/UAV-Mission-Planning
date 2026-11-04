#!/opt/px4-venv/bin/python3
from pathlib import Path
import numpy as np
from pyulog import ULog
from scipy.spatial.transform import Rotation
p=max(Path('/work/src/PX4-Autopilot/build/px4_sitl_default/rootfs/log').glob('*/*.ulg'),key=lambda p:p.stat().st_mtime)
u=ULog(str(p));ds={d.name:d.data for d in u.data_list}
print('Log',p)
print('Relevant datasets',[k for k in ds if any(s in k for s in ['attitude','odometry','groundtruth'])])
status=ds['vehicle_status'];a=status['timestamp'][status['arming_state']==2]*1e-6
start=float(a.min());print('Armed start',start)
for name in ['vehicle_attitude','vehicle_attitude_groundtruth','vehicle_visual_odometry']:
    if name not in ds:continue
    d=ds[name];print(name,'samples',len(d['timestamp']))
    q=np.column_stack([d[f'q[{i}]'] for i in range(4)])
    good=np.isfinite(q).all(axis=1)&(np.linalg.norm(q,axis=1)>.5)
    t=d['timestamp'][good]*1e-6
    e=Rotation.from_quat(q[good][:,[1,2,3,0]]).as_euler('xyz',degrees=True)
    for dt in [0,3,6,9,12,15,18,21,24,27]:
        i=np.argmin(abs(t-(start+dt)))
        print('after',dt,'RPY',np.round(e[i],2).tolist())
for name in ['vehicle_local_position','vehicle_local_position_groundtruth','vehicle_visual_odometry']:
    if name not in ds:continue
    d=ds[name];t=d['timestamp']*1e-6;print(name,'position samples')
    xyz=np.column_stack([d[k] for k in ['x','y','z']]) if name!='vehicle_visual_odometry' else np.column_stack([d[f'position[{i}]'] for i in range(3)])
    for dt in [0,3,6,9,12,15,18,19,20,21,24]:
        i=np.argmin(abs(t-(start+dt)));print('after',dt,'XYZ',np.round(xyz[i],3).tolist())
