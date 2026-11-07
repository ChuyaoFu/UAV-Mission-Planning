#!/usr/bin/env python3
"""Independent frozen-replay scoring ONLY. Never a navigation input/module.

Align only first BODY translation and gravity-axis yaw. Do not fit scale,
trajectory, roll or pitch. Simulator poses remain confined to this evaluator.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('replay',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    document=json.loads(a.replay.read_text());estimate=np.asarray(document['home_poses_ns_xyz_qxyzw_cov36'],float)
    truth=np.asarray(json.loads((a.run/'evaluation.json').read_text())['poses'],float)
    truth=truth[np.unique(truth[:,0],return_index=True)[1]]
    if estimate.ndim!=2 or estimate.shape[1]!=44 or len(estimate)<100:raise ValueError('Actual home BODY posterior stream required')
    times=estimate[:,0]*1e-9;keep=(times>=truth[0,0])&(times<=truth[-1,0]);estimate=estimate[keep];times=times[keep]
    if len(estimate)<100 or not np.isfinite(estimate).all() or (np.diff(times)<=0).any():raise ValueError('Invalid/insufficient frozen posterior')
    true_position=np.column_stack([np.interp(times,truth[:,0],truth[:,i]) for i in [1,2,3]])
    true_rotation=Slerp(truth[:,0],Rotation.from_quat(truth[:,4:8]))(times);relative_rotation=Rotation.from_quat(estimate[:,4:8])
    alignment=Rotation.from_euler('z',true_rotation[0].as_euler('xyz')[2]-relative_rotation[0].as_euler('xyz')[2])
    position=alignment.apply(estimate[:,1:4]-estimate[0,1:4])+true_position[0]
    error=position-true_position;distance=np.linalg.norm(error,axis=1)
    covariance=estimate[:,8:].reshape(-1,6,6)
    sigma=3*np.sqrt(np.maximum(0,np.linalg.eigvalsh(covariance[:,:3,:3])[:,-1]))
    attitude=(true_rotation.inv()*(alignment*relative_rotation)).magnitude()
    report={'role':'Independent evaluator only; no output fed to navigation','alignment':'first BODY translation and gravity-axis yaw only; no scale/trajectory/roll/pitch fit',
        'samples':len(times),'rmse_3d_m':float(np.sqrt(np.mean(distance**2))),'maximum_position_error_m':float(distance.max()),
        'position_error_p50_p95_m':np.percentile(distance,[50,95]).tolist(),'orientation_error_p50_p95_max_deg':np.degrees(np.percentile(attitude,[50,95,100])).tolist(),
        'posterior_position_three_sigma_maxeig_p50_p95_max_m':np.percentile(sigma,[50,95,100]).tolist(),
        'fraction_position_error_within_three_sigma_maxeig_proxy':float(np.mean(distance<=sigma+1e-9)),
        'accuracy_qualification':False,'scope':'Short original-course launch recording, not physical sensor qualification, Orin timing or full-course success'}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
