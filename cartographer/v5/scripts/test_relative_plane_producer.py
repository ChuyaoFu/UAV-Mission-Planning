"""Current sensor-frame producer contract."""
import unittest
import task_surfaces as module
from pathlib import Path
import numpy as np
import cv2
from task_observations import Exposure


class ProducerTests(unittest.TestCase):
    def image(self):
        image=np.full((300,400),40,np.uint8)
        cv2.rectangle(image,(145,95),(160,205),220,-1);cv2.rectangle(image,(240,95),(255,205),220,-1);cv2.rectangle(image,(145,142),(255,158),220,-1)
        return image,np.full(image.shape,2.,np.float32),[300,300,200,150]

    def test_raw_camera_plane_needs_no_world_translation_or_global_covariance(self):
        image,depth,K=self.image();detector=module.LandingPlaneDetector()
        T=np.diag([1.,-1.,-1.,1.]);T[2,3]=2.
        exposure=Exposure(2.,2.,2.,2.,T,'e',pose_error_m=.418,calibration_id='rig')
        with self.assertRaises(ValueError):detector.detect(image,depth,K,exposure,2.)
        rows=detector.detect_sensor_frame(image,depth,K,[0,0,-1],2000000001)
        self.assertTrue(any(r['marker_kind']=='h' for r in rows));self.assertEqual(rows[0]['stamp_ns'],2000000001)
        self.assertFalse(rows[0]['landing_area_verified'])

    def test_unchanged_legacy_nominal_geometry_and_vertical_rejection(self):
        image,depth,K=self.image();detector=module.LandingPlaneDetector()
        T=np.diag([1.,-1.,-1.,1.]);T[2,3]=2.
        exposure=Exposure(2.,2.,2.,2.,T,'e',calibration_id='rig')
        old=detector.detect(image,depth,K,exposure,2.)
        local=detector.detect_sensor_frame(image,depth,K,[0,0,-1],2000000000)
        self.assertEqual(len(old),len(local));np.testing.assert_allclose(old[0]['center_camera_m'],local[0]['center_camera_m'])
        self.assertEqual(detector.detect_sensor_frame(image,depth,K,[0,-1,0],2000000000),[])

if __name__=='__main__':unittest.main()
