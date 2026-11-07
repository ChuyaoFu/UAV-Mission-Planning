import importlib.util
import threading
import time
import unittest
from pathlib import Path
import cv2
import numpy as np
from planar_stereo import PlanarStereo
from planar_evidence_runtime import ExactTripletJoin, LatestOnlyWorker, pack_evidence


class NativeSafetyTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        self.shape = (200, 320)
        self.K = np.array([190., 190., 160., 100.])
        self.left = np.full(self.shape, 170, np.uint8)
        self.right = self.left.copy()
        self.raw = np.full(self.shape, np.nan, np.float32)
        self.raw[24:180:8, 32:296:8] = 2.

    def run_model(self, **kwargs):
        return PlanarStereo().compute(self.left, self.raw, self.K, .075, right=self.right, **kwargs)

    def test_odd_native_measured_obstacle_cannot_be_overwritten_or_cleared(self):
        # This point disappears under nearest-neighbor half-grid downsampling.
        self.raw[101, 151] = 1.
        r = self.run_model()
        np.testing.assert_array_equal(r['measured_depth'], self.raw)
        self.assertFalse(np.isfinite(r['inferred_depth'][98:105, 148:155]).any())
        self.assertFalse(np.isfinite(r['free_ray_lower_bound'][98:105, 148:155]).any())
        self.assertGreater(np.isfinite(r['inferred_depth']).sum(), 1000)

    def test_free_bound_never_passes_known_neighbor_even_inside_surface_tolerance(self):
        self.K[0] = self.K[1] = 3000.
        self.raw[101, 151] = 1.955
        r = self.run_model()
        self.assertFalse(np.isfinite(r['free_ray_lower_bound'][98:105, 148:155]).any())
        self.assertTrue(np.isfinite(r['inferred_depth'][40:60, 210:230]).any())

    def test_no_inference_at_any_native_raw_hit_and_both_self_masks(self):
        rng = np.random.default_rng(8)
        y, x = rng.integers(30, 170, 100), rng.integers(40, 280, 100)
        self.raw[y, x] = 2.
        left_mask = np.zeros(self.shape, bool)
        right_mask = left_mask.copy()
        left_mask[60:90, 70:100] = True
        right_mask[120:150, 160:190] = True
        r = self.run_model(self_mask=left_mask, right_self_mask=right_mask)
        self.assertFalse(np.isfinite(r['inferred_depth'][np.isfinite(self.raw) | left_mask]).any())
        # Right blocked pixels project approximately 7.125px to the left at 2m.
        self.assertFalse(np.isfinite(r['inferred_depth'][123:147, 169:195]).any())

    def test_projected_near_measurement_occludes_background_in_right_image(self):
        self.K[0] = self.K[1] = 760.
        self.raw[80:120, 160:164] = .6
        # Foreground projects at x~65, background at the same right pixels
        # projects from x~94, far outside the left foreground dilation.
        r = self.run_model()
        self.assertFalse(np.isfinite(r['inferred_depth'][83:117, 92:98]).any())
        self.assertTrue(np.isfinite(r['inferred_depth'][50:70, 100:140]).any())

    def test_slanted_plane_uses_native_pixel_ray_and_shorter_endpoint(self):
        y, x = np.indices(self.shape)
        disparity = 7. + 2. * ((x - self.K[2]) / self.K[0]) + .5 * ((y - self.K[3]) / self.K[1])
        truth = (self.K[0] * .075 / disparity).astype(np.float32)
        self.raw[np.isfinite(self.raw)] = truth[np.isfinite(self.raw)]
        r = self.run_model()
        support = np.isfinite(r['inferred_depth'])
        self.assertGreater(support.sum(), 1000)
        np.testing.assert_allclose(r['inferred_depth'][support], truth[support], atol=2e-6)
        self.assertTrue((r['free_ray_lower_bound'][support] < truth[support]).all())
        np.testing.assert_allclose(r['depth_interval_m'][support], r['inferred_depth'][support] - r['free_ray_lower_bound'][support], atol=2e-7)

    def test_native_thin_foreground_and_photo_mismatch_reject(self):
        self.left[100:101, 80:220] = 30
        self.right[100:101, 73:213] = 30
        r = self.run_model()
        self.assertFalse(np.isfinite(r['inferred_depth'][99:102, 90:210]).any())
        self.right[:] = 150
        self.assertFalse(np.isfinite(self.run_model()['inferred_depth']).any())

    def test_expired_compute_deadline_retains_measurements_and_no_partial_inference(self):
        r = self.run_model(deadline_monotonic=time.monotonic() - 1.)
        self.assertTrue(r['budget_exhausted'])
        np.testing.assert_array_equal(r['measured_depth'], self.raw)
        for field in ('inferred_depth', 'free_ray_lower_bound', 'depth_interval_m'):
            self.assertTrue(np.isnan(r[field]).all())
        self.assertTrue((r['plane_id'] == -1).all())

    def test_invalid_calibration_grid_or_types_are_rejected(self):
        for k, baseline in (([0., 190., 160., 100.], .075), (self.K, 0.), ([np.nan]*4, .075), (self.K, -1.)):
            with self.assertRaises(ValueError):
                PlanarStereo().compute(self.left, self.raw, k, baseline, right=self.right)
        with self.assertRaises(ValueError):
            self.run_model(self_mask=np.zeros((10, 10)))
        with self.assertRaises(ValueError):
            self.run_model(self_mask=np.full(self.shape, np.nan))
        with self.assertRaises(ValueError):
            PlanarStereo().compute(np.zeros((802, 1280), np.uint8), np.full((802, 1280), np.nan, np.float32), self.K, .075)
        with self.assertRaises(ValueError):
            PlanarStereo().compute(self.left.astype(float), self.raw, self.K, .075)

    def test_zero_negative_and_infinite_raw_values_do_not_enable_inference(self):
        for x, value in zip((121, 131, 141), (0., -1., np.inf)):
            self.raw[101, x] = value
        r = self.run_model()
        for x in (121, 131, 141):
            self.assertTrue(np.isnan(r['inferred_depth'][101, x]))
        np.testing.assert_array_equal(r['measured_depth'], self.raw)

    def test_degenerate_right_projection_stays_unknown(self):
        # Affine disparity slope approaches focal length: right patch collapses.
        self.raw[:] = np.nan
        y, x = np.indices(self.shape)
        region = (x >= 130) & (x <= 160) & (y >= 30) & (y <= 170) & (y % 4 == 0) & (x % 2 == 0)
        disparity = .98 * (x - 130) + 15.
        self.raw[region] = (self.K[0] * .075 / disparity[region]).astype(np.float32)
        self.assertFalse(np.isfinite(self.run_model()['inferred_depth']).any())



class EvidenceContractTests(unittest.TestCase):
    def test_pack_preserves_measured_and_only_accepts_consistent_free_rays(self):
        measured = np.array([[1., np.nan, np.nan, np.nan, np.nan]], np.float32)
        result = dict(free_ray_lower_bound=np.full((1, 5), 1.9, np.float32), inferred_depth=np.full((1, 5), 2., np.float32), depth_interval_m=np.full((1, 5), .1, np.float32), plane_id=np.zeros((1, 5), np.int16))
        result['depth_interval_m'][0, 2] = .5
        result['plane_id'][0, 3] = -1
        result['free_ray_lower_bound'][0, 4] = 2.1
        r = pack_evidence(measured, result, enabled=True)
        np.testing.assert_array_equal(r[..., 0], measured)
        self.assertTrue(np.isnan(r[0, 0, 1]))
        self.assertAlmostEqual(r[0, 1, 1], 1.9, places=6)
        self.assertTrue(np.isnan(r[0, 2:, 1]).all())
        self.assertTrue(np.isnan(pack_evidence(measured, result)[..., 1]).all())

    def test_exact_join_never_mixes_exposures_and_has_hard_input_bound(self):
        j = ExactTripletJoin(6)
        for i in range(10000):
            self.assertIsNone(j.push(0, i, ('left', i)))
            self.assertLessEqual(len(j.buffers[0]), 6)
        self.assertIsNone(j.push(1, 9998, 'right_old'))
        self.assertIsNone(j.push(2, 9999, 'depth'))
        stamp, triple = j.push(1, 9999, 'right')
        self.assertEqual((stamp, triple), (9999, (('left', 9999), 'right', 'depth')))
        self.assertTrue(all(not b for b in j.buffers))
        self.assertIsNone(j.push(0, 9999, 'duplicate'))
        j.reset()
        for stream in (0, 1):
            self.assertIsNone(j.push(stream, 10, stream))
        self.assertEqual(j.push(2, 10, 2), (10, (0, 1, 2)))

    def test_one_active_one_pending_and_latest_result_only(self):
        started, release = threading.Event(), threading.Event()
        calls = []
        def handler(item):
            calls.append(item)
            if item == 0:
                started.set()
                if not release.wait(1.):
                    raise RuntimeError('test timeout')
            return item * 2
        w = LatestOnlyWorker(handler)
        try:
            w.submit(0)
            self.assertTrue(started.wait(1.))
            for i in range(1, 1001):
                w.submit(i)
            with w.condition:
                self.assertEqual(w.pending, 1000)
                self.assertIsNone(w.ready)
            release.set()
            deadline = time.monotonic() + 2.
            result = None
            while time.monotonic() < deadline:
                with w.condition:
                    done = w.ready is not None and w.ready[0] == 1000
                if done:
                    result = w.take()
                    break
                time.sleep(.001)
            self.assertIsNotNone(result)
            self.assertEqual(calls, [0, 1000])
            self.assertEqual(result[1], 2000)
            self.assertEqual(w.superseded, 999)
            self.assertIsNone(w.take())
        finally:
            release.set()
            w.close()
        self.assertFalse(w.thread.is_alive())

    def test_worker_error_is_reported_and_does_not_kill_worker(self):
        def handler(item):
            if item == 0:
                raise ValueError('expected test error')
            return item
        w = LatestOnlyWorker(handler)
        try:
            for item in (0, 1):
                w.submit(item)
                deadline = time.monotonic() + 1.
                ready = None
                while ready is None and time.monotonic() < deadline:
                    ready = w.take()
                    time.sleep(.001)
                self.assertIsNotNone(ready)
                self.assertEqual(ready[0], item)
                self.assertEqual(ready[3] is not None, item == 0)
            self.assertEqual(w.errors, 1)
        finally:
            w.close()


if __name__ == '__main__':
    unittest.main()
