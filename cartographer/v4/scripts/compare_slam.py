#!/usr/bin/python3
"""Independent offline SLAM evaluation; no publications or fitted trajectory."""
import json,sys,re
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
root=Path('/work/cartographer/v4');label=sys.argv[1];data=json.loads((root/'results'/label/'trajectory.json').read_text());source=json.loads(Path(sys.argv[2] if len(sys.argv)>2 else '/work/cartographer/v2/results/course-21/evaluation-offline.json').read_text());g=np.array(source['rows']['truth']);gp=g[:,1:4]+Rotation.from_quat(g[:,4:8]).apply([.12,0,.055]);result={};aligned={};judge_path=root/'results'/label/'judge.json';flight_window=json.loads(judge_path.read_text())['result'] if judge_path.exists() else None
for name,values in data.items():
 a=np.array(values)
 if len(a)<3:continue
 a=a[(a[:,0]>=g[0,0])&(a[:,0]<=g[-1,0])];a=a[np.argsort(a[:,0])];a=a[np.r_[True,np.diff(a[:,0])>1e-5]];reference=np.array([np.interp(a[:,0],g[:,0],gp[:,i]) for i in range(3)]).T;anchor=np.asarray(values[0]) if name=='optimized' else a[0]
 if name=='optimized' and anchor[0]<g[0,0]:
  # Recorder starts after preflight initialization. The vehicle is confirmed
  # disarmed and stationary before arming; preserve that FIRST node's gauge.
  anchor_reference=gp[0];gquat=g[0,4:8]
 else:
  anchor_reference=np.array([np.interp(anchor[0],g[:,0],gp[:,i]) for i in range(3)]);gquat=g[np.argmin(abs(g[:,0]-anchor[0])),4:8]
 rot=Rotation.from_quat(gquat)*Rotation.from_quat(anchor[4:8]).inv();estimate=rot.apply(a[:,1:4]-anchor[1:4])+anchor_reference;err=np.linalg.norm(estimate-reference,axis=1);result[name]={'samples':len(a),'first_stamp_s':float(a[0,0]),'last_stamp_s':float(a[-1,0]),'rmse_m':float(np.sqrt(np.mean(err**2))),'max_m':float(err.max()),'p95_m':float(np.percentile(err,95)),'span_xyz_m':np.ptp(estimate,axis=0).tolist(),'final_error_m':float(err[-1])}
 if flight_window is not None and 'elapsed_s' in flight_window:
  active=(a[:,0]>=flight_window['start_time_s'])&(a[:,0]<=flight_window['start_time_s']+flight_window['elapsed_s'])
  if active.any():result[name].update(flight_samples=int(active.sum()),flight_rmse_m=float(np.sqrt(np.mean(err[active]**2))),flight_max_m=float(err[active].max()))
 aligned[name]={'time':a[:,0].tolist(),'estimate':estimate.tolist(),'reference':reference.tolist(),'error':err.tolist()}
(root/'results'/label/'slam-metrics.json').write_text(json.dumps({'metrics':result,'alignment':'FIRST initialization node gauge for optimized trajectory, recorded stationary first pose for live/local outputs; no scale or trajectory fit; no reset at first airborne node','source':'independent original recorded truth; evaluation only'},indent=2));(root/'results'/label/'slam-comparison.json').write_text(json.dumps(aligned));print(json.dumps(result,indent=2))
