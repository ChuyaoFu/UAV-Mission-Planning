#!/usr/bin/env python3
"""Read actual robot calibration, emit an isolated config fragment only."""
import argparse,json,hashlib,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--robot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads(a.robot.read_text());t=np.asarray(r['imu_in_body_m'],float);q=np.asarray(r['imu_in_body_quaternion_xyzw'],float)
if t.shape!=(3,) or q.shape!=(4,) or not np.isfinite(np.r_[t,q]).all() or abs(np.linalg.norm(q)-1)>1e-6:raise ValueError('Explicit actual finite T_BODY_IMU required')
R=Rotation.from_quat(q);C=R.inv();lever=-C.apply(t);out=np.r_[lever,C.as_quat()]
assert np.linalg.norm(t+R.apply(lever))<1e-14 and np.linalg.norm((R*C).as_rotvec())<1e-14
if a.output.exists():raise ValueError('Refusing overwrite')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('# Isolated optional telemetry; no estimator prior/noise change\nretain_home_clone: true\nhome_relative_imu_from_body: ['+', '.join(format(x,'.17e') for x in out)+']\n')
a.output.with_suffix('.provenance.json').write_text(json.dumps({'robot_input':str(a.robot),'sha256':hashlib.sha256(a.robot.read_bytes()).hexdigest(),'definition':'T_IMU_BODY=inverse(T_BODY_IMU) with configured actual lever and Hamilton quaternion','T_imu_body_xyz_xyzw':out.tolist(),'no_config_installed':True},indent=2)+'\n');print(a.output)
