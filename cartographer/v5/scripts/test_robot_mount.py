import unittest,xml.etree.ElementTree as ET
from types import SimpleNamespace as S
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from robot_geometry import BODY_FROM_IMU,IMU_IN_BODY,TOF_IN_BODY,IMU_FROM_TOF,body_rotation,body_position,body_velocity,body_velocity_covariance
from height_odometry import HeightObserver
ROOT=Path(__file__).resolve().parents[1]
def odom(t,z,q):return S(header=S(stamp=S(sec=int(t),nanosec=0)),pose=S(pose=S(position=S(z=z),orientation=S(x=q[0],y=q[1],z=q[2],w=q[3]))))
class MountTests(unittest.TestCase):
 def test_imu_mount_is_removed_from_body_attitude_and_lever(self):
  world_body=Rotation.from_euler('xyz',[.1,-.2,.8]);world_imu=world_body*BODY_FROM_IMU;np.testing.assert_allclose(body_rotation(world_imu).as_matrix(),world_body.as_matrix(),atol=1e-10);body=np.array([1.,2.,3.]);np.testing.assert_allclose(body_position(body+world_body.apply(IMU_IN_BODY),world_imu),body,atol=1e-10)
 def test_whole_kit_model_matches_mount_calibration(self):
  model=ET.parse(ROOT/'config/model.sdf');p=np.array([float(v) for v in model.findtext(".//link[@name='vio_rig']/pose").split()]);np.testing.assert_allclose(p[:3],IMU_IN_BODY);np.testing.assert_allclose(Rotation.from_euler('xyz',p[3:]).as_matrix(),BODY_FROM_IMU.as_matrix(),atol=1e-9)
 def test_tof_stays_fixed_in_body_when_oak_tilts(self):
  p=np.array([float(v) for v in ET.parse(ROOT/'config/model.sdf').findtext(".//sensor[@name='down_tof']/pose").split()]);np.testing.assert_allclose(IMU_IN_BODY+BODY_FROM_IMU.apply(p[:3]),TOF_IN_BODY,atol=1e-9);axis=(BODY_FROM_IMU*Rotation.from_euler('xyz',p[3:])).apply([1.,0.,0.]);np.testing.assert_allclose(axis,[0,0,-1],atol=1e-9)
 def test_tof_height_uses_body_tilt_not_imu_board_tilt(self):
  h=HeightObserver();q=BODY_FROM_IMU.as_quat();h.ranging(S(header=S(stamp=S(sec=1,nanosec=0)),range=.2,min_range=.05,max_range=4.));h.odometry(odom(1,0.,q));h.ranging(S(header=S(stamp=S(sec=2,nanosec=0)),range=.7,min_range=.05,max_range=4.));h.odometry(odom(2,.5,q));self.assertAlmostEqual(h.correction,0.,places=8)
 def test_rotating_in_place_has_zero_body_origin_velocity(self):
  omega=np.array([.1,-.2,.7]);lever_velocity=np.cross(omega,IMU_IN_BODY);np.testing.assert_allclose(body_velocity(BODY_FROM_IMU.inv().apply(lever_velocity),BODY_FROM_IMU.inv().apply(omega)),np.zeros(3),atol=1e-10)
 def test_self_clear_mount_and_old_obstructed_mount(self):
  import self_mask
  from unittest.mock import patch
  self.assertLess(max(m.mean() for m in self_mask.self_masks()),.001)
  with patch('self_mask.IMU_IN_BODY',np.array([.12,0,.055])):self.assertGreater(min(m.mean() for m in self_mask.self_masks()),.05)
if __name__=='__main__':unittest.main()
