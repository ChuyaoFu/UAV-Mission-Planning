import unittest
import numpy as np
from motion_limits import *
class Limits(unittest.TestCase):
 def test_bound(self):
  for distance in [.0,.04,.2,.5,1.]:
   for latency in [.05,.2,.5]:
    v=safe_speed(distance,latency,uncertainty=.03)
    if distance>=.03:self.assertLessEqual(stopping_distance(v,latency),distance+1e-9)
    else:self.assertEqual(v,0.)
 def test_latency_uncertainty(self):
  self.assertLess(safe_speed(.4,.3),safe_speed(.4,.1));self.assertLess(safe_speed(.4,.1,uncertainty=.1),safe_speed(.4,.1,uncertainty=.03))
 def test_visibility(self):
  self.assertEqual(visibility_speed_factor(0),1.);self.assertEqual(visibility_speed_factor(.7),0.);self.assertLess(visibility_speed_factor(.4),visibility_speed_factor(.2))
 def test_slew(self):
  v=slew_velocity(np.zeros(3),np.array([1.,1.,0.]),.05);self.assertAlmostEqual(np.linalg.norm(v),.05)
if __name__=='__main__':unittest.main()
