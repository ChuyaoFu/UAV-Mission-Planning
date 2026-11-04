#!/opt/px4-venv/bin/python3
from pathlib import Path
import json
import numpy as np
from pyulog import ULog
from scipy.spatial.transform import Rotation
root=Path('/work')
contact=json.loads((root/'results/vehicle-contacts.json').read_text())
tc=contact['nonfloor_events'][0]['stamp']
p=max((root/'src/PX4-Autopilot/build/px4_sitl_default/rootfs/log').glob('*/*.ulg'),key=lambda p:p.stat().st_mtime)
names=['vehicle_attitude','vehicle_attitude_groundtruth','vehicle_attitude_setpoint','vehicle_local_position','vehicle_local_position_groundtruth','trajectory_setpoint','vehicle_visual_odometry']
u=ULog(str(p),message_name_filter_list=names);ds={d.name:d.data for d in u.data_list}
r={'first_contact':contact['nonfloor_events'][0],'land_command_s':json.loads((root/'results/baseline-flight.json').read_text())['land_command_stamp_us']*1e-6,'samples':[]}
def at(name,t):
    d=ds[name];i=int(np.argmin(abs(d['timestamp']*1e-6-t)));return d,i
for dt in [-2.5,-2,-1.5,-1,-.5,-.1,0,.1,.5,1]:
    t=tc+dt;row={'time_s':t,'relative_to_contact_s':dt}
    for name in ['vehicle_attitude','vehicle_attitude_groundtruth','vehicle_visual_odometry']:
        d,i=at(name,t);q=[d[f'q[{j}]'][i] for j in range(4)]
        row[name+'_rpy_deg']=Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_euler('xyz',degrees=True).tolist()
    for name in ['vehicle_local_position','vehicle_local_position_groundtruth']:
        d,i=at(name,t);row[name+'_xyz']=[float(d[k][i]) for k in ['x','y','z']]
        row[name+'_velocity']=[float(d[k][i]) for k in ['vx','vy','vz']]
    d,i=at('vehicle_visual_odometry',t)
    row['ev_position']=[float(d[f'position[{j}]'][i]) for j in range(3)]
    row['ev_body_velocity']=[float(d[f'velocity[{j}]'][i]) for j in range(3)]
    if 'trajectory_setpoint' in ds:
        d,i=at('trajectory_setpoint',t)
        row['setpoint_position']=[float(d[f'position[{j}]'][i]) for j in range(3)]
        row['setpoint_velocity']=[float(d[f'velocity[{j}]'][i]) for j in range(3)]
    if 'vehicle_attitude_setpoint' in ds:
        d,i=at('vehicle_attitude_setpoint',t)
        q=[d[f'q_d[{j}]'][i] for j in range(4)]
        row['attitude_setpoint_rpy_deg']=Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_euler('xyz',degrees=True).tolist()
    r['samples'].append(row)
(root/'results/contact-timeline.json').write_text(json.dumps(r,indent=2));print('First contact',r['first_contact'],'land command',r['land_command_s'])
for x in r['samples']:
    print(round(x['relative_to_contact_s'],2),'gt_rpy',np.round(x['vehicle_attitude_groundtruth_rpy_deg'],2).tolist(),'gt_xyz',np.round(x['vehicle_local_position_groundtruth_xyz'],3).tolist(),'sp_xyz',x.get('setpoint_position'),'sp_vel',x.get('setpoint_velocity'),'att_sp',np.round(x.get('attitude_setpoint_rpy_deg'),2).tolist())
    print('  local_xyz',np.round(x['vehicle_local_position_xyz'],3).tolist(),'ev_xyz',np.round(x['ev_position'],3).tolist(),'gt_vel',np.round(x['vehicle_local_position_groundtruth_velocity'],3).tolist(),'local_vel',np.round(x['vehicle_local_position_velocity'],3).tolist(),'ev_body_vel',np.round(x['ev_body_velocity'],3).tolist())
