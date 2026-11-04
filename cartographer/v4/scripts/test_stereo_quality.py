import unittest,cv2,numpy as np
from camera_geometry import load_rig
from stereo_quality import Matcher
class StereoChecks(unittest.TestCase):
 def test_calibration_roundtrip(self):
  K,T,B,shape=load_rig();p=np.array([[.1,.2,1.],[.2,-.1,2.]])
  imu=p@T[:3,:3].T+T[:3,3];recovered=(imu-T[:3,3])@T[:3,:3]
  self.assertTrue(np.allclose(p,recovered));self.assertGreater(B,0)
 def test_rectified_shift_and_occlusion(self):
  rng=np.random.default_rng(7);left=cv2.GaussianBlur(rng.integers(0,256,(240,640),dtype=np.uint8),(3,3),0);right=np.zeros_like(left);right[:,:-12]=left[:,12:];right[:,260:300]=0
  depth,q=Matcher().compute(left,right,380.,.075);expected=380*.075/12
  valid=depth[30:-30,160:240];self.assertGreater(np.isfinite(valid).mean(),.7);self.assertLess(abs(np.nanmedian(valid)-expected),.03)
  self.assertLess(np.isfinite(depth[30:-30,275:300]).mean(),.2)
if __name__=='__main__':unittest.main()
