"""Measured rigid kit mounting; no field geometry. Hardware needs its own file."""
import json,os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def load_robot():
 if os.environ.get('HARDWARE_MODE')=='1' and not os.environ.get('ROBOT_CALIBRATION'):raise ValueError('Hardware requires measured ROBOT_CALIBRATION')
 path=Path(os.environ.get('ROBOT_CALIBRATION',Path(__file__).resolve().parents[1]/'config/robot.json'));data=json.loads(path.read_text())
 for key in ['imu_in_body_m','tof_in_body_m']:
  a=np.asarray(data[key],float)
  if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError('Invalid '+key)
 if os.environ.get('HARDWARE_MODE')=='1' and 'imu_in_body_quaternion_xyzw' not in data:raise ValueError('Hardware requires the measured IMU-to-body mounting rotation')
 q=np.asarray(data.get('imu_in_body_quaternion_xyzw',[0,0,0,1]),float)
 if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>.001:raise ValueError('Invalid IMU-to-body mounting quaternion')
 return data
ROBOT=load_robot();IMU_IN_BODY=np.asarray(ROBOT['imu_in_body_m'],float);TOF_IN_BODY=np.asarray(ROBOT['tof_in_body_m'],float);BODY_FROM_IMU=Rotation.from_quat(ROBOT.get('imu_in_body_quaternion_xyzw',[0,0,0,1]));IMU_FROM_TOF=BODY_FROM_IMU.inv().apply(IMU_IN_BODY-TOF_IN_BODY)
def body_rotation(imu_rotation):return imu_rotation*BODY_FROM_IMU.inv()
def body_position(imu_position,imu_rotation):return np.asarray(imu_position)-body_rotation(imu_rotation).apply(IMU_IN_BODY)

def body_velocity(velocity_in_imu,angular_velocity_in_imu):
 """Remove the reference-point velocity caused by the rigid IMU lever arm."""
 return BODY_FROM_IMU.apply(velocity_in_imu)-np.cross(BODY_FROM_IMU.apply(angular_velocity_in_imu),IMU_IN_BODY)
def body_velocity_covariance(twist_covariance):
 x,y,z=IMU_IN_BODY;skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]]);r=BODY_FROM_IMU.as_matrix();j=np.concatenate([r,skew@r],axis=1);return j@np.asarray(twist_covariance).reshape(6,6)@j.T
