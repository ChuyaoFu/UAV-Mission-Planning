#!/usr/bin/python3
"""Independent OFFLINE replay evaluation. Never an estimator/navigation input."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('replays',nargs='+');a=p.parse_args();root=Path(a.run);g=np.asarray(json.load(open(root/'evaluation.json'))['poses']);g=g[np.unique(g[:,0],return_index=True)[1]];robot=json.load(open(root/'config/robot.json'));mount=Rotation.from_quat(robot['imu_in_body_quaternion_xyzw']);lever=np.array(robot['imu_in_body_m']);rows=[]
for path in a.replays:
 d=json.load(open(path));b=np.array(d['poses']);b=b[(b[:,0]>=g[0,0])&(b[:,0]<=g[-1,0])];b=b[np.unique(b[:,0],return_index=True)[1]]
 if len(b)<100:raise ValueError('Insufficient replay poses')
 rb=Rotation.from_quat(b[:,4:8])*mount.inv();body=b[:,1:4]-rb.apply(lever);truth=np.column_stack([np.interp(b[:,0],g[:,0],g[:,k]) for k in [1,2,3]]);rg=Slerp(g[:,0],Rotation.from_quat(g[:,4:8]))([b[0,0]])[0];align=rg*rb[0].inv();estimate=align.apply(body-body[0])+truth[0];e=estimate-truth
 rows.append({'replay':str(path),'histogram':d['histogram'],'fast_threshold':d.get('fast_threshold',10),'samples':len(b),'coverage':d['moving_visual_update_coverage_fraction'],'update_age_p50_p95_max_s':d['moving_update_age_p50_p95_max_s'],'rmse_3d_m':float(np.sqrt(np.mean(np.sum(e*e,axis=1)))),'rmse_xy_m':float(np.sqrt(np.mean(np.sum(e[:,:2]**2,axis=1)))),'maximum_position_error_m':float(np.linalg.norm(e,axis=1).max())})
out={'role':'Offline independent scoring only; fixed first-pose alignment, no fitted trajectory or geometry alignment','comparisons':rows};(root/'vio-replay-evaluation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
