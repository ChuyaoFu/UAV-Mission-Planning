"""Rectified stereo calibration shared by projection and matching.
The input must be the calibration of the rectified images actually published.
"""
import os
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1]
def load_rig():
 if os.environ.get('HARDWARE_MODE')=='1' and not os.environ.get('STEREO_CALIBRATION'):raise ValueError('Hardware mode requires the actual rectified camera calibration; simulator calibration is forbidden')
 path=Path(os.environ.get('STEREO_CALIBRATION',ROOT/'config/gazebo_vio/kalibr_imucam_chain.yaml'))
 text=path.read_text();rig=yaml.safe_load('\n'.join(x for x in text.splitlines() if not x.startswith('%YAML')))
 left,right=rig['cam0'],rig['cam1'];K=np.array(left['intrinsics'],float);T=np.array(left['T_imu_cam'],float);Tr=np.array(right['T_imu_cam'],float)
 if not np.allclose(K,right['intrinsics']) or not np.allclose(T[:3,:3],Tr[:3,:3]) or np.max(np.abs(left['distortion_coeffs']))>1e-8 or np.max(np.abs(right['distortion_coeffs']))>1e-8:raise ValueError('Expected rectified stereo images with shared intrinsics and parallel optical axes')
 baseline=np.linalg.norm(T[:3,3]-Tr[:3,3]);assert baseline>0
 if not np.allclose(Tr[:3,3]-T[:3,3],T[:3,0]*baseline,atol=1e-5):raise ValueError('Rectified right camera must lie along positive left optical X; verify left/right ordering and extrinsics')
 return K,T,baseline,tuple(left['resolution'])
