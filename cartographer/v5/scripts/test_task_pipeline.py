import unittest
import cv2
import numpy as np
from task_perception_helper import TaskPerceptionHelper
from task_executive import TaskExecutive,Feedback
from test_task_executive import LIMITS,measured_map,guard_at
from test_task_observations import exposure
from test_task_surfaces import landing_image,down_exposure


def opening_view(origin,inside=4.):
    center=np.array([1.5,0.,.7]); origin=np.array(origin)
    K=np.array([300.,300.,200.,150.]); image=np.full((300,400),180,np.uint8)
    # World y/z corners, observed through the calibrated optical camera.
    corners=center+[[0,.5,.5],[0,-.5,.5],[0,-.5,-.5],[0,.5,-.5]]
    T=exposure(origin=origin).local_from_camera
    camera=(corners-origin)@T[:3,:3]
    uv=np.column_stack((camera[:,0]/camera[:,2]*K[0]+K[2],camera[:,1]/camera[:,2]*K[1]+K[3]))
    poly=np.rint(uv).astype(np.int32)
    cv2.fillPoly(image,[poly],35)
    depth=np.full(image.shape,1.5-origin[0],np.float32)
    cv2.fillPoly(depth,[poly],inside)
    return image,depth,K


class PipelineTests(unittest.TestCase):
    def test_real_detector_tracker_and_executive_share_frame_contract(self):
        helper=TaskPerceptionHelper()
        for i,y in enumerate((-.05,0.,.05)):
            now=1+i*.1;origin=[0,y,.7];image,depth,K=opening_view(origin)
            tracks=helper.observe(image,depth,K,exposure(now,origin,epoch='own-local'),'window',now)
        self.assertEqual(len(tracks),1)
        self.assertTrue(tracks[0].consistent)
        np.testing.assert_allclose(tracks[0].center,[1.5,0,.7],atol=.015)
        executive=TaskExecutive(LIMITS,[0,0,.18],'own-local'); executive.stage='WINDOW'
        f=Feedback(now,origin,'own-local',True,False,0.)
        intent=executive.tick(f,tracks,guard_at(measured_map(),now),now)
        self.assertEqual(intent.action,'move')
        self.assertIsNotNone(intent.certificate)
        self.assertFalse(executive.report()['competition_tasks_complete'])

    def test_unknown_depth_pipeline_preserves_geometry_but_cannot_approach_for_passage(self):
        helper=TaskPerceptionHelper()
        for i,y in enumerate((-.05,0.,.05)):
            now=1+i*.1;origin=[0,y,.7];image,depth,K=opening_view(origin,np.nan)
            tracks=helper.observe(image,depth,K,exposure(now,origin,epoch='own-local'),'window',now)
        self.assertEqual(tracks[0].interior,'unknown')
        executive=TaskExecutive(LIMITS,[0,0,.18],'own-local');executive.stage='WINDOW'
        self.assertEqual(executive.tick(Feedback(now,origin,'own-local',True,False,0.),tracks,guard_at(measured_map(),now),now).action,'explore')

    def test_measured_h_plane_pipeline_covers_full_landing_footprint(self):
        helper=TaskPerceptionHelper(); image,depth,K=landing_image()
        for i,x in enumerate((0.,.05,.1)):
            now=1+i*.1
            shifted=cv2.warpAffine(image,np.float32([[1,0,-x*150],[0,1,0]]),(400,300),borderValue=40)
            tracks=helper.observe(shifted,depth,K,down_exposure(now,(x,0,2)),'dock',now)
        # The helper uses a different reset scope in this fixture. Only the
        # executive with that same measured scope may consume the track.
        executive=TaskExecutive(LIMITS,[1,0,.18],'local');executive.stage='DOCK'
        marks=[t for t in tracks if t.details['marker_kind']=='h']
        self.assertTrue(marks)
        self.assertTrue(executive._landing_support(marks[0]))

if __name__=='__main__':unittest.main()
