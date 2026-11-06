"""Measured rigid sensor offsets; hardware must explicitly select its own file."""
import json,os
from pathlib import Path
import numpy as np

def load_robot():
 if os.environ.get('HARDWARE_MODE')=='1' and not os.environ.get('ROBOT_CALIBRATION'):
  raise ValueError('Hardware requires measured ROBOT_CALIBRATION')
 path=Path(os.environ.get('ROBOT_CALIBRATION',Path(__file__).resolve().parents[1]/'config/robot.json'))
 data=json.loads(path.read_text())
 for key in ['imu_in_body_m','tof_in_body_m']:
  a=np.asarray(data[key],float)
  if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError('Invalid '+key)
 return data
ROBOT=load_robot()
IMU_IN_BODY=np.asarray(ROBOT['imu_in_body_m'],float)
TOF_IN_BODY=np.asarray(ROBOT['tof_in_body_m'],float)
IMU_FROM_TOF=IMU_IN_BODY-TOF_IN_BODY
