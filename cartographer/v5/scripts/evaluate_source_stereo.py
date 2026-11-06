#!/usr/bin/python3
"""Independent source-mesh ray scoring of frozen predictions. Evaluator only.
This file and its meshes/poses are denied to the prediction and navigation profiles.
"""
import argparse,hashlib,json,os,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,trimesh
from scipy.spatial.transform import Rotation,Slerp
from trimesh.ray.ray_pyembree import RayMeshIntersector
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('inputs');p.add_argument('predictions');p.add_argument('output');p.add_argument('--pattern',default='frame-*.npz');p.add_argument('--candidate-kind',choices=['learned','elas','planar'],default='learned');p.add_argument('--world',default='/work/sim/worlds/dronehacks.sdf');p.add_argument('--model-root',default='/work/sim/models');a=p.parse_args();run=Path(a.run);os.environ['STEREO_CALIBRATION']=str(run/'config/gazebo_vio/kalibr_imucam_chain.yaml');os.environ['ROBOT_CALIBRATION']=str(run/'config/robot.json')
from camera_geometry import load_rig
from robot_geometry import IMU_IN_BODY,BODY_FROM_IMU
K,T,baseline,res=load_rig();from evaluation_geometry import load_collision_mesh
scene,hashes=load_collision_mesh(a.world,a.model_root);ray=RayMeshIntersector(scene);gt=np.array(json.loads((run/'evaluation.json').read_text())['poses']);gt=gt[np.unique(gt[:,0],return_index=True)[1]];slerp=Slerp(gt[:,0],Rotation.from_quat(gt[:,4:8]));rows=[]
def score(depth,truth):
 known=np.isfinite(truth)&(truth>.25)&(truth<8);valid=known&np.isfinite(depth)&(depth>.25)&(depth<8);error=depth[valid]-truth[valid]
 return {'reference_rays':int(known.sum()),'accepted_rays':int(valid.sum()),'coverage_fraction':float(valid.sum()/max(1,known.sum())),'absolute_error_median_p90_p99_m':np.percentile(np.abs(error),[50,90,99]).tolist() if len(error) else [],'unsafe_overestimate_above_25cm_fraction':float(np.mean(error>.25)) if len(error) else None,'positive_overestimate_p99_m':float(np.percentile(np.maximum(error,0),99)) if len(error) else None}
for f in sorted(Path(a.predictions).glob(a.pattern)):
 pred=np.load(f);input=np.load(Path(a.inputs)/f.name);stamp=float(input['stamp']);R=slerp([stamp])[0];body=np.array([np.interp(stamp,gt[:,0],gt[:,i]) for i in range(1,4)]);origin=body+R.apply(IMU_IN_BODY+BODY_FROM_IMU.apply(T[:3,3]));optical=R*BODY_FROM_IMU*Rotation.from_matrix(T[:3,:3]);v,u=np.mgrid[4:res[1]:8,4:res[0]:8];directions=optical.apply(np.column_stack([(u.ravel()-K[2])/K[0],(v.ravel()-K[3])/K[1],np.ones(u.size)]));locations,indices,triangles=ray.intersects_location(np.repeat(origin[None],len(directions),axis=0),directions,multiple_hits=False);truth=np.full(u.size,np.nan);truth[indices]=np.linalg.norm(locations-origin,axis=1)/np.linalg.norm(directions[indices],axis=1);truth=truth.reshape(u.shape);rows.append({'frame':f.name,'stamp':stamp,'classical_measured_stereo':score(input['measured_stereo'][v,u],truth),a.candidate_kind+'_filtered':score(pred['inferred_depth'][v,u],truth),a.candidate_kind+'_raw':score(pred['raw_inferred_depth'][v,u],truth)});print(json.dumps(rows[-1]),flush=True)
result={'candidate_kind':a.candidate_kind,'prediction_pattern':a.pattern,'scope':'Independent scoring only; native poses and physical collision meshes were never supplied to prediction or flight. Includes low-visibility net collision surfaces. Not Orin timing or complete-course validation.','mesh_hashes':hashes,'mesh_triangles':len(scene.faces),'frames':rows,'navigation_enabled':False};Path(a.output).write_text(json.dumps(result,indent=2))
