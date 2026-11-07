import unittest,numpy as np,cv2
from scipy.spatial.transform import Rotation
from relative_guideline import GuidelineDetector
class GuidelineTests(unittest.TestCase):
 def fixture(self):
  a=np.full((240,640),210,np.uint8);cv2.line(a,(60,120),(580,120),20,12);return a,np.ones(a.shape,np.float32),np.array([300.,300.,320.,120.]),np.array([0.,0.,1.]),Rotation.from_matrix(np.diag([1.,-1.,-1.]))
 def test_measured_line_is_relative_metric_geometry(self):
  a,d,k,p,r=self.fixture();v=GuidelineDetector().detect(a,d,k,p,r,0.);self.assertTrue(v);self.assertGreater(abs(v[0]['endpoints'][1][0]-v[0]['endpoints'][0][0]),1.)
 def test_missing_stereo_cannot_create_a_route_hint(self):
  a,d,k,p,r=self.fixture();d[:]=np.nan;self.assertEqual(GuidelineDetector().detect(a,d,k,p,r,0.),[])
 def test_non_floor_line_is_rejected(self):
  a,d,k,p,r=self.fixture();self.assertEqual(GuidelineDetector().detect(a,d,k,p,r,-1.),[])
 def test_no_absolute_yaw_is_assumed(self):
  a,d,k,p,r=self.fixture();rz=Rotation.from_euler('z',np.pi/2);v=GuidelineDetector().detect(a,d,k,p,rz*r,0.);self.assertTrue(v);self.assertGreater(abs(v[0]['endpoints'][1][1]-v[0]['endpoints'][0][1]),1.)
if __name__=='__main__':unittest.main()
