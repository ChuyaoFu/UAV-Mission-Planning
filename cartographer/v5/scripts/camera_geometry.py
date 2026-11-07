"""Rectified stereo calibration shared by projection and matching.
The input must be the calibration of the rectified images actually published.
"""
import os
from pathlib import Path
import numpy as np,yaml
from scipy.spatial.transform import Rotation
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

def bearing_visible(bearing,trial_yaw,body_rotation,optical_in_body,K,resolution,margin_px=20.):
 """Can a horizontal exploration direction be observed at a proposed yaw?
 Uses the real rigid optical rotation and measured body tilt. This checks a
 direction at infinity, not obstacle clearance or a target's near visibility.
 """
 current_yaw=body_rotation.as_euler('xyz')[2]
 world_from_optical=Rotation.from_euler('z',trial_yaw-current_yaw)*body_rotation*optical_in_body
 ray=world_from_optical.inv().apply([np.cos(bearing),np.sin(bearing),0.])
 if not np.isfinite(ray).all() or ray[2]<=0:return False
 uv=ray[:2]/ray[2]*np.asarray(K[:2])+K[2:]
 return bool(margin_px<=uv[0]<=resolution[0]-margin_px and margin_px<=uv[1]<=resolution[1]-margin_px)
