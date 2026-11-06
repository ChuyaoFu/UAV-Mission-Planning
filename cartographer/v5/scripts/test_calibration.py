import unittest,xml.etree.ElementTree as ET
from pathlib import Path
import cv2,numpy as np
from scipy.spatial.transform import Rotation
from camera_geometry import load_rig,ROOT
class CalibrationTests(unittest.TestCase):
 def test_opencv_can_parse_vio_calibration(self):
  f=cv2.FileStorage(str(ROOT/'config/gazebo_vio/kalibr_imucam_chain.yaml'),cv2.FILE_STORAGE_READ);self.assertTrue(f.isOpened());self.assertEqual(f.getNode('cam0').getNode('T_imu_cam').size(),4);f.release()
 def test_projection_matches_actual_simulator_mount(self):
  K,T,b,res=load_rig();sensor=ET.parse(ROOT/'config/model.sdf').find(".//sensor[@name='left']");pose=np.array([float(x) for x in sensor.findtext('pose').split()]);optical=np.array([[0,0,1],[-1,0,0],[0,-1,0]]);np.testing.assert_allclose(T[:3,:3],Rotation.from_euler('xyz',pose[3:]).as_matrix()@optical,atol=1e-8);np.testing.assert_allclose(T[:3,3],pose[:3]);self.assertAlmostEqual(K[0],res[0]/2/np.tan(float(sensor.findtext('camera/horizontal_fov'))/2));self.assertAlmostEqual(b,.075)
if __name__=='__main__':unittest.main()
