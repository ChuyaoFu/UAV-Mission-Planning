#!/usr/bin/env python3
"""Independent offline scoring only. Never imported by navigation or PL-VINS."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run',type=Path);p.add_argument('odometry',type=Path);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
run=json.loads((a.run/'evaluation.json').read_text());g=np.asarray(run['poses'],float)
g=g[np.unique(g[:,0],return_index=True)[1]]
robot=json.loads((a.run/'config/robot.json').read_text())
mount=Rotation.from_quat(robot['imu_in_body_quaternion_xyzw']);lever=np.asarray(robot['imu_in_body_m'])
rows=[]
with a.odometry.open() as f:
 for r in csv.DictReader(f):rows.append([int(r['stamp_ns'])*1e-9,*[float(r[k]) for k in ['x','y','z','qx','qy','qz','qw']]])
b=np.asarray(rows,float)
if len(b)<2:raise ValueError('No initialized trajectory to evaluate')
if not np.isfinite(b).all() or np.any(np.diff(b[:,0])<=0):raise ValueError('Invalid/nonmonotonic estimator trajectory; do not hide resets')
b=b[(b[:,0]>=g[0,0])&(b[:,0]<=g[-1,0])]
if len(b)<2:raise ValueError('No reference time overlap')
rb=Rotation.from_quat(b[:,4:8])*mount.inv()
body=b[:,1:4]-rb.apply(lever)
reference=np.column_stack([np.interp(b[:,0],g[:,0],g[:,k]) for k in [1,2,3]])
rg=Slerp(g[:,0],Rotation.from_quat(g[:,4:8]))(b[:,0])
align=rg[0]*rb[0].inv()
estimate=align.apply(body-body[0])+reference[0]
error=estimate-reference;norm=np.linalg.norm(error,axis=1);angular=(rg.inv()*(align*rb)).magnitude()
result={'role':'Independent offline scoring only; no reference feedback to estimator/navigation',
 'alignment':'First overlapping BODY pose only. Fixed metric scale; no trajectory fit or reset alignment.',
 'source_run':str(a.run),'candidate_odometry':str(a.odometry),'samples':len(b),
 'reference_interval_s':[float(g[0,0]),float(g[-1,0])],'estimated_interval_s':[float(b[0,0]),float(b[-1,0])],
 'gap_p50_p95_max_s':np.percentile(np.diff(b[:,0]),[50,95,100]).tolist(),
 'rmse_3d_m':float(np.sqrt(np.mean(norm**2))),'rmse_xy_m':float(np.sqrt(np.mean(np.sum(error[:,:2]**2,axis=1)))),
 'position_error_p50_p95_max_m':np.percentile(norm,[50,95,100]).tolist(),
 'orientation_error_p50_p95_max_deg':np.rad2deg(np.percentile(angular,[50,95,100])).tolist(),
 'full_reference_interval_covered':bool(b[0,0]-g[0,0]<=.1 and g[-1,0]-b[-1,0]<=.1),
 'qualification':'A partial recorded replay, not a successful mission or hardware qualification'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
