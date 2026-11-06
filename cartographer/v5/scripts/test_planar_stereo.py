import unittest
import cv2
import numpy as np
from planar_stereo import PlanarStereo


class PlaneStereoTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        self.shape = (200, 320)
        self.K = np.array([190., 190., 160., 100.])
        self.left = np.full(self.shape, 170, np.uint8)
        self.right = self.left.copy()
        self.measured = np.full(self.shape, np.nan, np.float32)
        self.measured[24:180:8, 32:296:8] = 2.

    def predict(self, measured=None, right=True):
        return PlanarStereo().compute(self.left, self.measured if measured is None else measured,
                                      self.K, .075, right=self.right if right else None)

    def test_constant_plane_is_metric_and_lower_endpoint_is_conservative(self):
        result = self.predict()
        support = np.isfinite(result['inferred_depth'])
        self.assertGreater(support.sum(), 2000)
        np.testing.assert_allclose(result['inferred_depth'][support], 2., atol=1e-4)
        self.assertTrue((result['free_ray_lower_bound'][support] < 2.).all())
        np.testing.assert_array_equal(result['measured_depth'], self.measured)
        self.assertFalse(result['navigation_enabled'])
        self.assertFalse(np.isfinite(result['inferred_depth'][:16]).any())
        self.assertFalse(np.isfinite(result['inferred_depth'][:, :20]).any())

    def test_absent_or_collinear_stereo_does_not_become_a_plane(self):
        missing = np.full(self.shape, np.nan, np.float32)
        self.assertFalse(np.isfinite(self.predict(missing)['inferred_depth']).any())
        line = missing.copy()
        line[100, 40:280] = 2.
        self.assertFalse(np.isfinite(self.predict(line)['inferred_depth']).any())
        self.assertFalse(np.isfinite(self.predict(right=False)['inferred_depth']).any())

    def test_unmeasured_foreground_region_is_not_filled_as_background(self):
        self.left[65:135, 120:200] = 30
        self.right[65:135, 106:186] = 30
        self.measured[60:140, 110:210] = np.nan
        result = self.predict()
        self.assertFalse(np.isfinite(result['inferred_depth'][70:130, 125:195]).any())
        self.assertTrue(np.isfinite(result['inferred_depth'][24:60, 32:100]).any())

    def test_direct_nearer_obstacle_is_preserved(self):
        self.measured[80:120, 140:180] = 1.
        self.left[80:120, 140:180] = 30
        self.right[80:120, 126:166] = 30
        result = self.predict()
        np.testing.assert_allclose(result['measured_depth'][80:120, 140:180], 1.)
        self.assertFalse(np.isfinite(result['inferred_depth'][80:120, 140:180]).any())


if __name__ == '__main__':
    unittest.main()
