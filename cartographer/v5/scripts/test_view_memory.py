import unittest,numpy as np
from view_memory import ViewMemory,angle
class ViewTests(unittest.TestCase):
 def test_empty_memory_has_no_surveyed_heading(self):
  self.assertEqual(ViewMemory().choose(1.,np.zeros(3),.7,0.),.7)
 def test_low_feature_or_stale_constraints_do_not_certify_view(self):
  v=ViewMemory();v.add(1.,np.zeros(3),1.,2,0.);v.add(1.,np.zeros(3),1.,40,.7);self.assertEqual(len(v.views),0)
 def test_known_free_translation_can_keep_a_textured_side_view(self):
  v=ViewMemory();v.add(1.,np.zeros(3),1.,50,0.);self.assertAlmostEqual(v.choose(2.,np.zeros(3),0.,.5),1.)
 def test_distant_or_old_views_are_not_reused(self):
  v=ViewMemory();v.add(1.,np.zeros(3),1.,50,0.);self.assertEqual(v.choose(30.,np.zeros(3),0.,0.),0.);self.assertEqual(v.choose(2.,np.array([2.,0.,0.]),0.,0.),0.)
 def test_retained_home_view_requires_same_neighborhood_and_height(self):
  v=ViewMemory(maxlen=32,max_age=1e9,max_distance=.5,height_tolerance=.12);v.add(1.,np.array([0.,0.,.55]),.8,30,0.);self.assertAlmostEqual(v.choose(150.,np.array([.1,0.,.55]),0.,0.),.8);self.assertEqual(v.choose(150.,np.array([1.,0.,.55]),0.,0.),0.);self.assertEqual(v.choose(150.,np.array([.1,0.,.8]),0.,0.),0.)
 def test_exploration_cannot_trade_visible_bearing_for_more_features(self):
  v=ViewMemory();v.add(1.,np.zeros(3),np.pi,200,0.);v.add(1.3,np.zeros(3),.1,8,0.)
  self.assertAlmostEqual(v.choose(2.,np.zeros(3),0.,0.,allowed=lambda yaw:abs(angle(yaw))<.4),.1)
  self.assertAlmostEqual(v.choose(2.,np.zeros(3),0.,0.),np.pi)
 def test_no_admissible_memory_returns_requested_view(self):
  v=ViewMemory();v.add(1.,np.zeros(3),np.pi,200,0.)
  self.assertEqual(v.choose(2.,np.zeros(3),0.,0.,allowed=lambda yaw:abs(angle(yaw))<.4),0.)
if __name__=='__main__':unittest.main()
