import unittest
import numpy as np,cv2
from stereo_quality import Matcher
class StereoTests(unittest.TestCase):
 def test_constant_surface_does_not_invent_depth(self):
  a=np.full((240,400),180,np.uint8);d,_=Matcher().compute(a,a,240.,.075);self.assertFalse(np.isfinite(d).any())
 def test_low_contrast_noise_is_not_range_evidence(self):
  r=np.random.default_rng(2);a=r.integers(179,181,(240,400),dtype=np.uint8);b=np.roll(a,-10,axis=1);d,_=Matcher().compute(a,b,240.,.075);self.assertFalse(np.isfinite(d[:,140:-20]).any())
 def test_rectified_known_disparity_preserves_scale(self):
  r=np.random.default_rng(3);a=r.integers(30,220,(240,400),dtype=np.uint8);b=np.zeros_like(a);b[:,:-10]=a[:,10:];d,_=Matcher().compute(a,b,240.,.075);v=d[20:-20,150:-30];v=v[np.isfinite(v)];self.assertGreater(len(v),10000);self.assertAlmostEqual(float(np.median(v)),1.8,places=2)
 def test_half_resolution_matching_preserves_metric_scale(self):
  a=np.random.default_rng(5).integers(30,220,(240,800),dtype=np.uint8);b=np.zeros_like(a);b[:,:-20]=a[:,20:];d,q=Matcher().compute_scaled(a,b,480.,.075,scale=.5);v=d[20:-20,300:-60];v=v[np.isfinite(v)];self.assertEqual(d.shape,a.shape);self.assertGreater(len(v),10000);self.assertAlmostEqual(float(np.median(v)),1.8,places=2);self.assertEqual(q['native_equivalent_lr_tolerance_px'],.75)
 def test_half_resolution_does_not_fill_unknown(self):
  a=np.full((240,800),180,np.uint8);d,_=Matcher().compute_scaled(a,a,480.,.075,scale=.5);self.assertFalse(np.isfinite(d).any())
if __name__=='__main__':unittest.main()
