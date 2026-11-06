import unittest
import cv2
import numpy as np
from task_perception import OpeningDetector


class OpeningTests(unittest.TestCase):
    def rectangle(self, background=4.):
        image = np.full((300, 400), 150, np.uint8)
        cv2.rectangle(image, (90, 60), (310, 250), 220, -1)
        cv2.rectangle(image, (110, 80), (290, 230), 35, -1)
        depth = np.full(image.shape, 2., np.float32)
        depth[80:231, 110:291] = background
        return image, depth, np.array([300., 300., 200., 150.])

    def find(self, rows, shape):
        valid = [r for r in rows if r['kind'] == shape and r['center_camera_m'] is not None]
        self.assertTrue(valid, rows)
        return min(valid, key=lambda r: np.prod(r['rim_dimensions_m']))

    def test_measured_rectangle_reports_relative_metric_geometry(self):
        image, depth, K = self.rectangle()
        row = self.find(OpeningDetector().detect(image, depth, K), 'rectangle')
        self.assertEqual(row['status'], 'rim_and_background_observed')
        np.testing.assert_allclose(row['center_camera_m'], [0., .0333, 2.], atol=.025)
        np.testing.assert_allclose(sorted(row['rim_dimensions_m']), [1., 1.2], atol=.04)
        self.assertLess(row['normal_camera'][2], -.99)
        self.assertTrue(all(a < b for a, b in zip(row['clearance_lower_m'], row['rim_dimensions_m'])))
        self.assertFalse(row['passage_clearance_verified'])

    def test_missing_center_depth_does_not_certify_aperture(self):
        image, depth, K = self.rectangle(np.nan)
        row = self.find(OpeningDetector().detect(image, depth, K), 'rectangle')
        self.assertEqual(row['status'], 'rim_geometry_only')
        self.assertEqual(row['center_evidence']['unknown_fraction'], 1.)
        self.assertFalse(row['center_evidence']['background_observed'])

    def test_painted_rectangle_on_wall_is_not_an_observed_opening(self):
        image, depth, K = self.rectangle(2.)
        row = self.find(OpeningDetector().detect(image, depth, K), 'rectangle')
        self.assertEqual(row['status'], 'interior_obstructed_or_same_surface')
        self.assertFalse(row['center_evidence']['background_observed'])

    def test_nonmeasurement_depth_cannot_establish_rim(self):
        image, depth, K = self.rectangle()
        rows = OpeningDetector().detect(image, depth, K, measured_mask=np.zeros(depth.shape, bool))
        self.assertTrue(rows)
        self.assertTrue(all(r['center_camera_m'] is None and r['clearance_lower_m'] is None for r in rows))

    def test_ellipse_uses_measured_rim_plane_not_preset_dimensions(self):
        image = np.full((300, 400), 150, np.uint8)
        cv2.ellipse(image, (200, 150), (95, 75), 0, 0, 360, 220, -1)
        cv2.ellipse(image, (200, 150), (80, 60), 0, 0, 360, 35, -1)
        mask = np.zeros(image.shape, np.uint8)
        cv2.ellipse(mask, (200, 150), (80, 60), 0, 0, 360, 1, -1)
        depth = np.full(image.shape, 2., np.float32);depth[mask != 0] = 4.
        row = self.find(OpeningDetector().detect(image, depth, [300., 300., 200., 150.]), 'ellipse')
        self.assertEqual(row['status'], 'rim_and_background_observed')
        np.testing.assert_allclose(row['rim_dimensions_m'], [.8, 1.0667], atol=.035)
        self.assertFalse(row['passage_clearance_verified'])

    def test_inferred_center_pixels_are_not_background_evidence(self):
        image, depth, K = self.rectangle()
        direct = np.ones(depth.shape, bool);direct[80:231, 110:291] = False
        row = self.find(OpeningDetector().detect(image, depth, K, measured_mask=direct), 'rectangle')
        self.assertEqual(row['status'], 'rim_geometry_only')
        self.assertFalse(row['center_evidence']['background_observed'])

    def test_tilted_rim_with_different_intrinsics_recovers_metric_size(self):
        angle = np.deg2rad(25.);R = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
        corners = np.array([[-.6,-.5,0.],[.6,-.5,0.],[.6,.5,0.],[-.6,.5,0.]]) @ R.T + [.1,-.05,2.]
        K = np.array([420.,400.,200.,150.]);uv = np.column_stack((corners[:,0]/corners[:,2]*K[0]+K[2],corners[:,1]/corners[:,2]*K[1]+K[3]));image = np.full((300,400), 180,np.uint8);cv2.fillPoly(image,[np.rint(uv).astype(np.int32)],35)
        normal = R @ [0.,0.,-1.];d = -normal @ [.1,-.05,2.];y,x = np.indices(image.shape);rays = np.stack(((x-K[2])/K[0],(y-K[3])/K[1],np.ones(image.shape)),axis=-1);depth=(-d/(rays @ normal)).astype(np.float32);mask=np.zeros(image.shape,np.uint8);cv2.fillPoly(mask,[np.rint(uv).astype(np.int32)],1);depth[mask!=0]+=2.
        row=self.find(OpeningDetector().detect(image,depth,K),'rectangle');self.assertEqual(row['status'],'rim_and_background_observed');np.testing.assert_allclose(sorted(row['rim_dimensions_m']),[1.,1.2],atol=.04);np.testing.assert_allclose(row['normal_camera'],normal,atol=.02);np.testing.assert_allclose(row['center_camera_m'],[.1,-.05,2.],atol=.025)

    def test_blank_image_does_not_produce_an_opening(self):
        image = np.full((300, 400), 150, np.uint8)
        self.assertEqual(OpeningDetector().detect(image, np.full(image.shape, 2.), [300., 300., 200., 150.]), [])


if __name__ == '__main__':
    unittest.main()
