import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from frames import PoseHistory
class ExposureTests(unittest.TestCase):
 def test_interpolates_pose_at_exposure(self):
  h=PoseHistory();h.append(1.,np.array([0.,0.,0.]),Rotation.from_euler('z',0).as_quat());h.append(1.08,np.array([1.,0.,0.]),Rotation.from_euler('z',90,degrees=True).as_quat());p,r=h.sample(1.04);np.testing.assert_allclose(p,[.5,0,0]);self.assertAlmostEqual(r.as_euler('xyz',degrees=True)[2],45.)
 def test_short_slerp_matches_full_history(self):
  from frames import interpolate_rotation
  from scipy.spatial.transform import Slerp
  times=np.linspace(1.,4.,300);rotations=Rotation.from_euler('xyz',np.column_stack([times*.03,times*.02,times*.5]));reference=Slerp(times,rotations)
  for t in np.linspace(1.,4.,20):self.assertLess((interpolate_rotation(times,rotations.as_quat(),t)*reference([t])[0].inv()).magnitude(),1e-12)
 def test_rejects_unbracketed_exposure(self):
  h=PoseHistory();h.append(1.,np.zeros(3),[0,0,0,1]);self.assertIsNone(h.sample(.9));self.assertIsNone(h.sample(1.1))
 def test_does_not_interpolate_across_tracking_loss(self):
  h=PoseHistory();h.append(1.,np.zeros(3),[0,0,0,1]);h.append(1.6,np.ones(3),[0,0,0,1]);self.assertIsNone(h.sample(1.3));p,r=h.sample(1.6);np.testing.assert_allclose(p,np.ones(3))
 def test_accepts_boundary_gap_despite_float_rounding(self):
  h=PoseHistory();h.append(1.,np.zeros(3),[0,0,0,1]);h.append(1.1,np.ones(3),[0,0,0,1]);self.assertIsNotNone(h.sample(1.05))
if __name__=='__main__':unittest.main()
