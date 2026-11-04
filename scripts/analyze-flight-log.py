#!/opt/px4-venv/bin/python3
from pathlib import Path
import json
import numpy as np
from pyulog import ULog
root=Path('/work')
log=max((root/'src/PX4-Autopilot/build/px4_sitl_default/rootfs/log').glob('*/*.ulg'),key=lambda p:p.stat().st_mtime)
u=ULog(str(log),message_name_filter_list=['vehicle_status','sensor_combined','estimator_status_flags','sensor_gps'])
report={'ulog':str(log),'datasets':[(d.name,d.multi_id) for d in u.data_list]}
status=next(d.data for d in u.data_list if d.name=='vehicle_status')
armed=status['arming_state']==2
times=status['timestamp'][armed]*1e-6
report['armed_interval_s']=[float(times.min()),float(times.max())]
for d in u.data_list:
    if d.name=='sensor_combined':
        a=np.column_stack([d.data[f'accelerometer_m_s2[{i}]'] for i in range(3)])
        norms=np.linalg.norm(a,axis=1);i=int(norms.argmax())
        report['max_acceleration']={'m_s2':float(norms[i]),'timestamp_s':float(d.data['timestamp'][i]*1e-6)}
    if d.name=='estimator_status_flags':
        report['fusion_flags']={k:bool(v[-1]) for k,v in d.data.items() if k in ['cs_ev_pos','cs_ev_yaw','cs_ev_vel','cs_ev_hgt','cs_gnss_pos','cs_gnss_vel','cs_gps_hgt','cs_yaw_align']}
raw=json.loads((root/'results/vio-evaluation/raw.json').read_text())
gt=np.asarray(raw['truth']);est=np.asarray(raw['estimate'])
for name,mask in [('armed', (est[:,0]>=times.min())&(est[:,0]<=times.max()+.2)),('before_landing', (est[:,0]>=times.min())&(est[:,0]<=times.max()-2))]:
    e=est[mask]
    ref=np.column_stack([np.interp(e[:,0],gt[:,0],gt[:,i]) for i in range(1,4)])
    x=e[:,1:]-e[:,1:].mean(axis=0);y=ref-ref.mean(axis=0)
    l,_,r=np.linalg.svd(x.T@y);dd=np.eye(3);dd[2,2]=np.linalg.det(l@r)
    aligned=x@(l@dd@r)+ref.mean(axis=0);err=np.linalg.norm(aligned-ref,axis=1)
    report[name]={'interval_s':[float(e[0,0]),float(e[-1,0])],'ate_rmse_m':float(np.sqrt(np.mean(err**2))),'max_error_m':float(err.max())}
print(json.dumps(report,indent=2));(root/'results/flight-log-analysis.json').write_text(json.dumps(report,indent=2))
