#!/usr/bin/python3
"""Independent offline SLAM evaluation; no publications or fitted trajectory."""
import json,sys,re
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
root=Path('/work/cartographer/v3');label=sys.argv[1];data=json.loads((root/'results'/label/'trajectory.json').read_text());source=json.loads(Path(sys.argv[2] if len(sys.argv)>2 else '/work/cartographer/v2/results/course-21/evaluation-offline.json').read_text());g=np.array(source['rows']['truth']);gp=g[:,1:4]+Rotation.from_quat(g[:,4:8]).apply([.12,0,.055]);result={};aligned={}
for name,values in data.items():
 a=np.array(values)
 if len(a)<3:continue
 a=a[(a[:,0]>=g[0,0])&(a[:,0]<=g[-1,0])];a=a[np.argsort(a[:,0])];a=a[np.r_[True,np.diff(a[:,0])>1e-5]];reference=np.array([np.interp(a[:,0],g[:,0],gp[:,i]) for i in range(3)]).T;gquat=g[np.argmin(abs(g[:,0]-a[0,0])),4:8];rot=Rotation.from_quat(gquat)*Rotation.from_quat(a[0,4:8]).inv();estimate=rot.apply(a[:,1:4]-a[0,1:4])+reference[0];err=np.linalg.norm(estimate-reference,axis=1);result[name]={'samples':len(a),'first_stamp_s':float(a[0,0]),'last_stamp_s':float(a[-1,0]),'rmse_m':float(np.sqrt(np.mean(err**2))),'max_m':float(err.max()),'p95_m':float(np.percentile(err,95)),'span_xyz_m':np.ptp(estimate,axis=0).tolist(),'final_error_m':float(err[-1])};aligned[name]={'time':a[:,0].tolist(),'estimate':estimate.tolist(),'reference':reference.tolist(),'error':err.tolist()}
(root/'results'/label/'slam-metrics.json').write_text(json.dumps({'metrics':result,'alignment':'fixed initial pose only, no scale or trajectory fit','source':'independent original recorded truth; evaluation only'},indent=2));(root/'results'/label/'slam-comparison.json').write_text(json.dumps(aligned));print(json.dumps(result,indent=2))
