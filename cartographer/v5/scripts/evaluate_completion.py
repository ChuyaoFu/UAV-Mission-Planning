#!/usr/bin/python3
"""Independent geometry-only evaluation of saved learned outputs.
Never imported by prediction or navigation; no reference data is sent to them.
"""
import argparse,json,os,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('inputs');p.add_argument('predictions');p.add_argument('output');a=p.parse_args();root=Path(a.run);os.environ['STEREO_CALIBRATION']=str(root/'config/gazebo_vio/kalibr_imucam_chain.yaml');os.environ['ROBOT_CALIBRATION']=str(root/'config/robot.json')
from camera_geometry import load_rig
from robot_geometry import IMU_IN_BODY,BODY_FROM_IMU
g=np.array(json.loads((root/'evaluation.json').read_text())['poses']);g=g[np.unique(g[:,0],return_index=True)[1]];slerp=Slerp(g[:,0],Rotation.from_quat(g[:,4:8]));K,T,baseline,res=load_rig();surfaces=[]
for model in ET.parse(root/'evaluation-world.sdf').findall('.//model'):
 size=model.findtext('link/collision/geometry/box/size')
 if size:
  pose=np.array([float(v) for v in model.findtext('pose').split()]);surfaces.append((pose,np.array([float(v) for v in size.split()])))
rows=[]
for f in sorted(Path(a.predictions).glob('frame-*.npz')):
 d=np.load(f);input=np.load(Path(a.inputs)/f.name);stamp=float(d['stamp']);pred=d['inferred_depth'];R=slerp([stamp])[0];body=np.array([np.interp(stamp,g[:,0],g[:,i]) for i in range(1,4)]);camera=body+R.apply(IMU_IN_BODY+BODY_FROM_IMU.apply(T[:3,3]));rot=R*BODY_FROM_IMU*Rotation.from_matrix(T[:3,:3]);v,u=np.mgrid[:pred.shape[0],:pred.shape[1]];rays=rot.apply(np.stack([(u-K[2])/K[0],(v-K[3])/K[1],np.ones_like(u)],axis=-1).reshape(-1,3));truth=np.full(len(rays),np.inf)
 for pose,size in surfaces:
  rb=Rotation.from_euler('xyz',pose[3:]).inv();origin=rb.apply(camera-pose[:3]);directions=rb.apply(rays)
  with np.errstate(divide='ignore',invalid='ignore'):
   t0=(-size/2-origin)/directions;t1=(size/2-origin)/directions
  near=np.maximum(np.minimum(t0,t1).max(1),.001);far=np.maximum(t0,t1).min(1);truth=np.minimum(truth,np.where(far>=near,near,np.inf))
 truth=truth.reshape(pred.shape);measured=input['measured_stereo'];missing=~(np.isfinite(measured)&(measured>=.3)&(measured<=2.4));valid=np.isfinite(truth)&(truth>=.3)&(truth<=4)&np.isfinite(pred);error=pred[valid&missing]-truth[valid&missing]
 rows.append({'frame':f.name,'pixels_missing_in_stereo':len(error),'absolute_error_median_m':float(np.median(np.abs(error))),'absolute_error_p90_m':float(np.percentile(np.abs(error),90)),'positive_overestimate_p99_m':float(np.percentile(np.maximum(error,0),99)),'unsafe_overestimate_above_25cm_fraction':float(np.mean(error>.25)),'predicted_depth_below_30cm_fraction':float(np.mean(pred[valid&missing]<.3))})
result={'scope':'Independent evaluation only, not a depth input. Prediction process has no reference geometry/pose access.','frames':rows,'navigation_enabled':False};Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
