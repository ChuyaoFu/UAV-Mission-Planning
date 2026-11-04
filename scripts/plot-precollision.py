#!/usr/bin/python3
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=json.loads(Path('/work/results/clearance-false-zupt-failed/contact-timeline.json').read_text())
a=[x for x in r['samples'] if x['relative_to_contact_s']<0]
t=np.array([x['relative_to_contact_s'] for x in a])
true_v=[];vio_v=[];pos=[];ev=[]
rg=Rotation.from_euler('xyz',a[0]['vehicle_attitude_groundtruth_rpy_deg'],degrees=True).as_matrix()
re=Rotation.from_euler('xyz',a[0]['vehicle_visual_odometry_rpy_deg'],degrees=True).as_matrix()
align=rg@re.T
for x in a:
    rr=Rotation.from_euler('xyz',x['vehicle_attitude_groundtruth_rpy_deg'],degrees=True).as_matrix()
    true_v.append((rr.T@np.asarray(x['vehicle_local_position_groundtruth_velocity']))[0])
    vio_v.append(x['ev_body_velocity'][0])
    pos.append(x['vehicle_local_position_groundtruth_xyz'][0])
    ev.append((align@np.asarray(x['ev_position']))[0])
pos=np.array(pos);ev=np.array(ev)
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
ax[0].plot(t,true_v,'o-',label='Physical ground truth, body forward')
ax[0].plot(t,vio_v,'o-',label='Reported VIO body-forward velocity')
ax[0].set_ylabel('Velocity (m/s)')
ax[1].plot(t,pos-pos[0],'o-',label='Physical ground-truth displacement')
ax[1].plot(t,ev-ev[0],'o-',label='VIO displacement, fixed frame alignment')
ax[1].set_ylabel('Forward displacement since first sample (m)')
for p in ax:
    p.axvline(0,color='red',ls='--',label='First non-floor contact')
    p.set_xlabel('Time relative to first contact (s)');p.grid(alpha=.2);p.legend(fontsize=8)
fig.suptitle('Recorded localization discrepancy BEFORE the first wall/net contact')
fig.text(.5,.02,f'First contact at simulation t={r["first_contact"]["stamp"]:.3f}s: rotor vs north net. This chart shows the failed diagnostic run.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.07,1,.94]);fig.savefig('/work/results/precollision-diagnosis.png',dpi=160)
print('Saved precollision-diagnosis.png')
