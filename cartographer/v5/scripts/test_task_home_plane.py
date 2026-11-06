import dataclasses
import unittest
import numpy as np
from task_surfaces import HomePlaneDetector
from task_perception_helper import TaskPerceptionHelper
from task_contact import build_support_patch
from test_task_landing import calibrated_foot
from test_task_surfaces import down_exposure,forward_exposure


class HomePlaneTests(unittest.TestCase):
    def packet(self,stamp=2.,x=0.):
        return dataclasses.replace(down_exposure(stamp,(x,0,2.)),epoch='own-local',pose_error_m=.005)

    def test_unmarked_own_launch_plane_is_reobserved_from_direct_stereo(self):
        detector=HomePlaneDetector();cal=calibrated_foot();depth=np.full((300,400),2.,np.float32)
        rows=detector.detect(depth,[300,300,200,150],self.packet(),cal,2.)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['details']['marker_kind'],'own_launch_plane')
        self.assertFalse(rows[0]['details']['vio_global_z_correction_applied'])
        np.testing.assert_allclose(rows[0]['details']['observed_ground_z_local_m'],0.,atol=.001)

    def test_raised_platform_cannot_be_reclassified_as_home_or_global_drift(self):
        rows=HomePlaneDetector().detect(np.full((300,400),1.37,np.float32),[300,300,200,150],self.packet(),calibrated_foot(),2.)
        self.assertEqual(rows,[])

    def test_missing_depth_wrong_epoch_and_vertical_wall_do_not_anchor_home(self):
        detector=HomePlaneDetector();cal=calibrated_foot();depth=np.full((300,400),2.,np.float32);K=[300,300,200,150]
        self.assertEqual(detector.detect(depth,K,self.packet(),cal,2.,np.zeros(depth.shape,bool)),[])
        self.assertEqual(detector.detect(depth,K,dataclasses.replace(self.packet(),epoch='reset'),cal,2.),[])
        self.assertEqual(detector.detect(depth,K,dataclasses.replace(forward_exposure(2.),epoch='own-local'),cal,2.),[])

    def test_home_helper_track_can_supply_dense_final_contact_support(self):
        cal=calibrated_foot();helper=TaskPerceptionHelper(cal);depth=np.full((300,400),2.,np.float32);image=np.full(depth.shape,40,np.uint8)
        for i,x in enumerate((-.05,0.,.05)):
            now=2+i*.1;exp=self.packet(now,x)
            tracks=helper.observe(image,depth,[300,300,200,150],exp,'home',now)
        self.assertTrue(tracks);self.assertTrue(tracks[0].consistent)
        patch=build_support_patch(depth,[300,300,200,150],exp,tracks[0],now,.48,.01)
        self.assertIsNotNone(patch)
        self.assertEqual(patch.marker_kind,'own_launch_plane')

if __name__=='__main__':unittest.main()
