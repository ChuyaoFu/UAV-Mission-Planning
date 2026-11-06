import unittest
import numpy as np
from task_observations import Exposure,TargetTracker,opening_interior,passage_views

R=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])

def exposure(stamp=1.,origin=(0,0,0),epoch='local1',**kwargs):
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=origin
    return Exposure(stamp,stamp,stamp,stamp,T,epoch,calibration_id='measured-rig',**kwargs)

def candidate(origin=(0,0,0),interior='free-background'):
    center=R.T@(np.array([2.,0.,0.])-origin)
    evidence={'background_observed':interior=='free-background','blocking_pixels':0,'same_surface_pixels':0}
    if interior=='painted': evidence.update(blocking_pixels=100,same_surface_pixels=100)
    if interior=='blocked': evidence.update(blocking_pixels=10,same_surface_pixels=1)
    return {'kind':'rectangle','center_camera_m':center.tolist(),'normal_camera':[0,0,-1],
            'clearance_lower_m':[1.,1.2],'rim_measured_support':100,
            'geometry_margin_m':.02,'center_evidence':evidence}


class ObservationTests(unittest.TestCase):
    def test_transform_recovers_same_local_target_from_different_poses(self):
        tracker=TargetTracker()
        for i,y in enumerate((0.,.05,.10)):
            rows=tracker.update([candidate((0,y,0))],exposure(1+i*.1,(0,y,0)),'window',1+i*.1)
        self.assertEqual(len(tracker.tracks),1)
        np.testing.assert_allclose(rows[0].center,[2,0,0])
        self.assertTrue(rows[0].consistent)
        self.assertFalse(rows[0].passage_clearance_verified)

    def test_repeated_frames_and_stationary_camera_do_not_establish_geometry(self):
        tracker=TargetTracker()
        for i in range(6):
            rows=tracker.update([candidate()],exposure(1+i*.1),'window',1+i*.1)
        self.assertFalse(rows[0].consistent)
        before=rows[0].observation_count
        self.assertEqual(tracker.update([candidate()],exposure(1.5),'window',1.5),[])
        self.assertEqual(rows[0].observation_count,before)

    def test_duplicate_contours_count_once_per_exposure(self):
        tracker=TargetTracker()
        rows=tracker.update([candidate(),candidate()],exposure(),'window',1.)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0].observation_count,1)

    def test_latest_contradictory_evidence_vetoes_previous_background(self):
        tracker=TargetTracker()
        for i,kind in enumerate(('free-background','free-background','blocked','unknown')):
            origin=np.array([0,i*.05,0])
            row=tracker.update([candidate(origin,kind)],exposure(1+i*.1,origin),'window',1+i*.1)[0]
            self.assertEqual(row.interior,kind)
        self.assertTrue(row.consistent)
        self.assertFalse(row.passage_clearance_verified)

    def test_painted_is_separate_from_foreground_blocker(self):
        self.assertEqual(opening_interior(candidate(interior='painted')),'painted')
        self.assertEqual(opening_interior(candidate(interior='blocked')),'blocked')
        self.assertEqual(opening_interior(candidate(interior='unknown')),'unknown')

    def test_epoch_reset_discards_old_targets(self):
        tracker=TargetTracker()
        a=tracker.update([candidate()],exposure(),'window',1.)[0]
        b=tracker.update([candidate()],exposure(1.1,epoch='reset2'),'window',1.1)[0]
        self.assertNotEqual(a.id,b.id)
        self.assertEqual(len(tracker.tracks),1)
        self.assertFalse(b.consistent)

    def test_stale_future_skew_and_bad_transform_rejected(self):
        for packet,now in ((exposure(),1.7),(exposure(),.9)):
            with self.assertRaises(ValueError): packet.validate(now)
        packet=exposure(); object.__setattr__(packet,'pose_stamp',.9)
        with self.assertRaises(ValueError): packet.validate(1.)
        packet=exposure(); packet.local_from_camera[0,0]=2.
        with self.assertRaises(ValueError): packet.validate(1.)

    def test_bad_provenance_or_pose_rejected(self):
        for field,value in (('calibration_id',''),('tracking_valid',False),('pose_error_m',.3)):
            packet=exposure(); object.__setattr__(packet,field,value)
            with self.assertRaises(ValueError): packet.validate(1.)

    def test_track_capacity_and_stale_pruning(self):
        tracker=TargetTracker(max_tracks=2)
        for i in range(4):
            c=candidate(); c['center_camera_m'][0]+=i
            tracker.update([c],exposure(1+i*.1),'window',1+i*.1)
        self.assertEqual(len(tracker.tracks),2)
        tracker.update([],exposure(4.),'window',4.)
        self.assertEqual(tracker.tracks,{})

    def test_views_account_for_tilted_plane_and_never_certify(self):
        tracker=TargetTracker(); row=tracker.update([candidate()],exposure(),'window',1.)[0]
        row.normal=np.array([.6,0,.8]); view=passage_views(row,.3,.15)
        self.assertAlmostEqual(view['body_support_m'],.3*.6+.15*.8)
        self.assertFalse(view['verified'])
        self.assertGreater(np.dot(np.array(view['before'])-row.center,row.normal),view['body_support_m'])

if __name__=='__main__': unittest.main()
