import unittest
import cv2
import numpy as np
from task_observations import Exposure
from task_surfaces import LandingPlaneDetector,WallTopDetector
from task_perception_helper import TaskPerceptionHelper


def down_exposure(stamp=1.,origin=(0,0,2)):
    T=np.eye(4); T[:3,:3]=np.diag([1.,-1.,-1.]); T[:3,3]=origin
    return Exposure(stamp,stamp,stamp,stamp,T,'local',calibration_id='rig')

def forward_exposure(stamp=1.):
    T=np.eye(4); T[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]]; T[:3,3]=[0,0,1.]
    return Exposure(stamp,stamp,stamp,stamp,T,'local',calibration_id='rig')


def landing_image():
    image=np.full((300,400),40,np.uint8)
    cv2.rectangle(image,(145,95),(160,205),220,-1)
    cv2.rectangle(image,(240,95),(255,205),220,-1)
    cv2.rectangle(image,(145,142),(255,158),220,-1)
    return image,np.full(image.shape,2.,np.float32),[300,300,200,150]


class SurfaceTests(unittest.TestCase):
    def test_h_marker_measured_plane_is_only_a_landing_proposal(self):
        image,depth,K=landing_image()
        rows=LandingPlaneDetector().detect(image,depth,K,down_exposure(),1.)
        h=[r for r in rows if r['marker_kind']=='h']
        self.assertTrue(h,rows)
        np.testing.assert_allclose(h[0]['center_camera_m'],[0,0,2.],atol=.02)
        self.assertFalse(h[0]['landing_area_verified'])
        self.assertFalse(h[0]['semantic_identity_verified'])

    def test_inferred_or_missing_plane_support_cannot_anchor_marker(self):
        image,depth,K=landing_image()
        self.assertEqual(LandingPlaneDetector().detect(image,depth,K,down_exposure(),1.,np.zeros(image.shape,bool)),[])
        depth[:]=np.nan
        self.assertEqual(LandingPlaneDetector().detect(image,depth,K,down_exposure(),1.),[])

    def test_vertical_mark_does_not_become_landing_plane(self):
        image,depth,K=landing_image()
        self.assertEqual(LandingPlaneDetector().detect(image,depth,K,forward_exposure(),1.),[])

    def test_blank_horizontal_plane_does_not_establish_a_dock(self):
        image,depth,K=landing_image(); image[:]=40
        self.assertEqual(LandingPlaneDetector().detect(image,depth,K,down_exposure(),1.),[])

    def test_wall_top_requires_measured_farther_background_above_edge(self):
        image=np.full((300,400),150,np.uint8); depth=np.full(image.shape,2.,np.float32); depth[:150]=4.
        rows=WallTopDetector().detect(image,depth,[300,300,200,150],forward_exposure(),1.)
        self.assertEqual(len(rows),1)
        self.assertGreater(rows[0]['background_pairs'],12)
        self.assertLess(rows[0]['edge_height_upper_local_m'],1.15)
        self.assertFalse(rows[0]['overflight_clearance_verified'])
        depth[:150]=np.nan
        self.assertEqual(WallTopDetector().detect(image,depth,[300,300,200,150],forward_exposure(),1.),[])

    def test_painted_line_and_provenance_mask_cannot_establish_wall_top(self):
        image=np.full((300,400),150,np.uint8); image[:150]=30; depth=np.full(image.shape,2.,np.float32)
        detector=WallTopDetector(); K=[300,300,200,150]
        self.assertEqual(detector.detect(image,depth,K,forward_exposure(),1.),[])
        depth[:150]=4.; mask=np.ones(image.shape,bool); mask[:150]=False
        self.assertEqual(detector.detect(image,depth,K,forward_exposure(),1.,mask),[])

    def test_horizontal_depth_discontinuity_is_not_a_vertical_wall(self):
        image=np.full((300,400),150,np.uint8); depth=np.full(image.shape,2.,np.float32); depth[:150]=4.
        self.assertEqual(WallTopDetector().detect(image,depth,[300,300,200,150],down_exposure(),1.),[])

    def test_helper_preserves_local_direct_support_and_bounded_tracks(self):
        image,depth,K=landing_image(); helper=TaskPerceptionHelper()
        for i,x in enumerate((0,.05,.1)):
            # Move image mark consistently with measured camera translation.
            shifted=cv2.warpAffine(image,np.float32([[1,0,-x*150],[0,1,0]]),(400,300),borderValue=40)
            tracks=helper.observe(shifted,depth,K,down_exposure(1+i*.1,(x,0,2)),'dock',1+i*.1)
        marks=[t for t in tracks if t.details['marker_kind']=='h']
        self.assertTrue(marks)
        self.assertTrue(marks[0].consistent)
        self.assertIn('support_points_local_m',marks[0].details)
        self.assertFalse(marks[0].passage_clearance_verified)
        self.assertLessEqual(len(marks[0].details['support_points_local_m']),400)

    def test_worker_results_are_detached_from_next_mutating_track_update(self):
        image,depth,K=landing_image();helper=TaskPerceptionHelper()
        first=helper.observe(image,depth,K,down_exposure(),'dock',1.)
        self.assertTrue(first)
        count=first[0].observation_count
        helper.observe(image,depth,K,down_exposure(1.1),'dock',1.1)
        self.assertEqual(first[0].observation_count,count)

    def test_cross_and_rotated_cross_are_explicit_bounded_marker_categories(self):
        detector=LandingPlaneDetector()
        base=np.full((300,400),40,np.uint8)
        cv2.rectangle(base,(190,95),(210,205),220,-1)
        cv2.rectangle(base,(145,140),(255,160),220,-1)
        for angle in (0,35):
            matrix=cv2.getRotationMatrix2D((200,150),angle,1.)
            image=cv2.warpAffine(base,matrix,(400,300),borderValue=40)
            rows=detector.detect(image,np.full(image.shape,2.,np.float32),[300,300,200,150],down_exposure(),1.)
            self.assertTrue(any(r['marker_kind']=='cross' for r in rows),rows)

if __name__=='__main__': unittest.main()
