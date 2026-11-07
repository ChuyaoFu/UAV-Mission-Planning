"""Synthetic sensor-domain fixtures; no actual qualification, ROS or vehicle."""
import unittest
from dataclasses import replace
import numpy as np
from task_relative_contact import *

T_BC=np.diag([1.,-1.,-1.,1.]);MOUNT=(0.,0.,-.023);AXIS=(0.,0.,-1.)
K=(160.,160.,320.,240.);STAMP=2000000000
SENSORS=('stereo','camera_extrinsics','plane_geometry','tof','tof_extrinsics','attitude','velocity','gyro')


def domain(mock_qualified=False):
    # The qualified scope below tests ingestion of external declarations only.
    # Every sensor/qualification record in this file is SYNTHETIC, not real.
    return SensorErrorDomain('synthetic-rig','synthetic-native-grid','a'*64,
        'independent-sensor-bench' if mock_qualified else 'synthetic-test',SENSORS,digest_matrix(T_BC),digest_tof(MOUNT,AXIS),
        (640,480),K,1.,(.25,2.),.005,.001,.001,.001,.003,.001,.001,.001,.001,.001)


def sample(t=STAMP,range_m=.477,armed=True,landed=False):
    return FootRangeSample(t,t,'e','synthetic-rig',range_m,(0.,0.,1.),(0.,0.,0.),(0.,0.,0.),landed,armed,t)


def foot(d):
    c=RelativeFootCalibrator(d,MOUNT,AXIS,diagnostic_only=True)
    for i in range(30):
        s=sample(100000000+i*50000000,.101,False,True)
        assert c.ingest(s,s.stamp_ns),c.last_rejection
    result=c.result(s.stamp_ns);assert result is not None
    return result


def patch(d=None,z=.5,**changes):
    d=domain() if d is None else d
    plane=CurrentPlane(STAMP,'e',d.calibration_id,(0.,0.,z),(0.,0.,1.))
    args=dict(image_stamp_ns=STAMP,depth_stamp_ns=STAMP,up_stamp_ns=STAMP,up_body=(0.,0.,1.),now_ns=STAMP,
              required_halfwidth_m=.46,actual_profile_id=d.camera_profile_id,diagnostic_only=True)
    args.update(changes)
    return build_relative_support(np.full((480,640),z,np.float32),np.ones((480,640),bool),K,T_BC,plane,d,**args)


def veto(p,**changes):
    values=dict(stamp_ns=p.reference_stamp_ns,reference_stamp_ns=p.reference_stamp_ns,epoch='e',source_region_sha256=p.source_region_sha256,
                allows_contact_sweep=True,full_upper_side_sweep_checked=True,unknown_blocked=True,reference_registration_qualified=True,certificate_id='synthetic-map-query',
                mode='descent',body_goal_m=(0,0,float(p.center_body_m[2])+.124),local_goal_error_m=.04,sweep_radius_m=p.requested_halfwidth_m)
    values.update(changes);return GlobalMapVeto(**values)


class RelativeTests(unittest.TestCase):
    def test_same_exposure_common_world_translation_cancels_exactly(self):
        point=np.array([.1,.2,.5,1.]);relative=T_BC@point
        for translation in ([0,0,0],[1000,-2500,700],[.154,.154,.154]):
            W_B=np.eye(4);W_B[:3,3]=translation
            recovered=np.linalg.solve(W_B,W_B@T_BC@point)
            np.testing.assert_allclose(recovered,relative,atol=1e-12)
        p=patch();r=assess_relative_contact(p,sample(),foot(domain()),domain(),MOUNT,AXIS,STAMP)
        self.assertTrue(r['local_geometry_passes']);self.assertFalse(r['global_translation_input_used'])
        self.assertLess(r['local_support_error_m'],.04)
        self.assertFalse(r['motion_certificate_issued']);self.assertFalse(r['native_command_issued'])

    def test_synthetic_only_diagnostic_can_never_be_runtime_contact_qualification(self):
        d=domain();p=patch(d);r=assess_relative_contact(p,sample(),foot(d),d,MOUNT,AXIS,STAMP,global_map_veto=veto(p))
        self.assertTrue(r['local_geometry_passes']);self.assertFalse(r['local_precision_qualified']);self.assertFalse(r['contact_candidate_eligible'])
        with self.assertRaisesRegex(RelativeContractError,'qualification missing'):patch(d,diagnostic_only=False)

    def test_external_contract_mock_requires_both_local_proof_and_global_map_veto(self):
        d=domain(True);p=patch(d);f=foot(d)
        good=assess_relative_contact(p,sample(),f,d,MOUNT,AXIS,STAMP,global_map_veto=veto(p))
        self.assertTrue(good['contact_candidate_eligible']);self.assertFalse(good['motion_certificate_issued'])
        for bad in [None,veto(p,allows_contact_sweep=False),veto(p,unknown_blocked=False),veto(p,full_upper_side_sweep_checked=False),
                    veto(p,reference_registration_qualified=False),veto(p,reference_stamp_ns=STAMP-1),veto(p,source_region_sha256='other'),veto(p,mode='liftoff'),veto(p,body_goal_m=(0,0,-.1)),veto(p,local_goal_error_m=0.),veto(p,sweep_radius_m=.1)]:
            r=assess_relative_contact(p,sample(),f,d,MOUNT,AXIS,STAMP,global_map_veto=bad)
            self.assertTrue(r['local_precision_qualified']);self.assertFalse(r['contact_candidate_eligible'])

    def test_unknown_hole_provenance_and_foreground_native_pixel_all_refuse(self):
        d=domain();plane=CurrentPlane(STAMP,'e',d.calibration_id,(0,0,.5),(0,0,1))
        for kind in ('nan','foreground','provenance','fractional-mask'):
            depth=np.full((480,640),.5,np.float32);mask=np.ones(depth.shape,bool)
            if kind=='nan':depth[240,320]=np.nan
            elif kind=='foreground':depth[240,320]=.45
            elif kind=='provenance':mask[240,320]=False
            else:mask=np.ones(depth.shape)*.5
            with self.assertRaises(RelativeContractError):
                build_relative_support(depth,mask,K,T_BC,plane,d,image_stamp_ns=STAMP,depth_stamp_ns=STAMP,up_stamp_ns=STAMP,up_body=(0,0,1),now_ns=STAMP,required_halfwidth_m=.46,actual_profile_id=d.camera_profile_id,diagnostic_only=True)

    def test_zero_residual_does_not_qualify_absolute_depth_or_plane_normal(self):
        for d in (replace(domain(),optical_z_error_m=None),replace(domain(),optical_z_error_m=.06),replace(domain(),plane_normal_error_rad=None)):
            with self.assertRaises(RelativeContractError):patch(d)

    def test_exact_exposure_current_plane_required_and_fov_cannot_be_clipped(self):
        for kwargs in ({'image_stamp_ns':STAMP+1},{'up_stamp_ns':STAMP+1},{'required_halfwidth_m':1.4},{'actual_profile_id':'different-running-profile'}):
            with self.assertRaises(RelativeContractError):patch(**kwargs)
        d=domain();plane=CurrentPlane(STAMP,'e',d.calibration_id,(0,0,.5),(0,0,1),'averaged-world-track')
        with self.assertRaisesRegex(RelativeContractError,'Current raw plane'):
            build_relative_support(np.full((480,640),.5,np.float32),np.ones((480,640),bool),K,T_BC,plane,d,image_stamp_ns=STAMP,depth_stamp_ns=STAMP,up_stamp_ns=STAMP,up_body=(0,0,1),now_ns=STAMP,required_halfwidth_m=.46,actual_profile_id=d.camera_profile_id,diagnostic_only=True)

    def test_actual_extrinsics_and_independent_stride_are_checked(self):
        with self.assertRaisesRegex(RelativeContractError,'extrinsics'):patch(replace(domain(),body_from_camera_sha256='b'*64))
        self.assertIsNotNone(patch(domain(),z=1.5))
        with self.assertRaisesRegex(RelativeContractError,'Independent measured rays'):patch(replace(domain(),independent_stride_px=2.),z=1.5)
        d=domain();p=patch(d)
        with self.assertRaisesRegex(RelativeContractError,'ToF extrinsics'):
            assess_relative_contact(p,sample(),foot(d),d,(0,0,-.06),AXIS,STAMP)

    def test_historical_support_needs_relative_link_even_one_nanosecond_later(self):
        d=domain();p=patch(d)
        with self.assertRaisesRegex(RelativeContractError,'Historical support'):
            assess_relative_contact(p,sample(STAMP+1),foot(d),d,MOUNT,AXIS,STAMP+1)

    def test_transport_accumulates_error_and_preserves_original_measurement_age(self):
        d=domain();p=patch(d);target=STAMP+20000000;T=np.eye(4);T[2,3]=.005
        link=RelativeMotionLink(STAMP,target,'e',d.calibration_id,tuple(tuple(x) for x in T),.001,.001,'b'*64,'synthetic-test','relative-imu-propagation')
        q=transport_support(p,link,target,diagnostic_only=True)
        self.assertEqual(q.source_stamp_ns,STAMP);self.assertEqual(q.reference_stamp_ns,target)
        self.assertGreater(q.local_error_m,p.local_error_m);self.assertLess(q.safe_halfwidth_m,p.safe_halfwidth_m)
        self.assertEqual(q.motion_qualification_records,('b'*64,));self.assertFalse(q.runtime_qualified)
        r=assess_relative_contact(q,sample(target,.472),foot(d),d,MOUNT,AXIS,target)
        self.assertTrue(r['local_geometry_passes']);self.assertFalse(r['contact_candidate_eligible'])
        for bad in (replace(link,translation_error_m=.05),replace(link,epoch='reset'),replace(link,method='subtract-two-global-marginals'),replace(link,target_stamp_ns=STAMP+300000000)):
            with self.assertRaises(RelativeContractError):transport_support(p,bad,bad.target_stamp_ns,diagnostic_only=True)

    def test_retained_patch_liftoff_requires_qualified_delta_explicit_goal_and_actual_armed(self):
        d=domain(True);p=patch(d,max_support_age_s=2.,static_environment=True)
        target=STAMP+1500000000;T=np.eye(4);T[2,3]=.376
        link=RelativeMotionLink(STAMP,target,'e',d.calibration_id,tuple(tuple(x) for x in T),.001,.001,'b'*64,'independent-relative-motion-bench','relative-imu-propagation')
        q=transport_support(p,link,target);s=sample(target,.101,True,True);f=foot(d)
        v=veto(q,mode='liftoff',body_goal_m=(0,0,.3))
        r=assess_relative_contact(q,s,f,d,MOUNT,AXIS,target,mode='liftoff',liftoff_height_m=.3,global_map_veto=v)
        self.assertTrue(r['contact_candidate_eligible']);self.assertFalse(r['native_command_issued'])
        self.assertEqual(r['source_stamp_ns'],STAMP)
        with self.assertRaises(RelativeContractError):assess_relative_contact(q,replace(s,armed=False),f,d,MOUNT,AXIS,target,mode='liftoff',liftoff_height_m=.3,global_map_veto=v)
        with self.assertRaises(RelativeContractError):assess_relative_contact(q,s,f,d,MOUNT,AXIS,target,mode='liftoff',global_map_veto=v)

    def test_contact_range_tilt_drift_speed_native_and_clearance_errors_refuse(self):
        d=domain();p=patch(d);f=foot(d)
        for bad in [replace(sample(),range_m=.40),replace(sample(),native_received_ns=STAMP-500000000),replace(sample(),armed=False,landed=False),
                    replace(sample(),velocity_body_mps=(.1,0,0)),replace(sample(),velocity_body_mps=(0,0,-.4)),replace(sample(),attitude_stamp_ns=STAMP+1)]:
            with self.assertRaises(RelativeContractError):assess_relative_contact(p,bad,f,d,MOUNT,AXIS,STAMP)
        with self.assertRaises(RelativeContractError):assess_relative_contact(replace(p,center_body_m=(.06,0,-.5)),sample(),f,d,MOUNT,AXIS,STAMP)

    def test_foot_calibration_is_offset_only_without_camera_or_world_position(self):
        d=replace(domain(),optical_z_error_m=None,plane_normal_error_rad=None,camera_profile_id='',body_from_camera_sha256=None,resolution=None,intrinsics=None,independent_stride_px=None,depth_interval_m=None,camera_translation_error_m=None,camera_rotation_error_rad=None)
        f=foot(d);self.assertAlmostEqual(f.offset_m,.124)
        self.assertFalse(hasattr(f,'resting_body_local_m'));self.assertFalse(f.runtime_qualified)
        c=RelativeFootCalibrator(d,MOUNT,AXIS,diagnostic_only=True)
        self.assertFalse(c.ingest(replace(sample(.0),native_received_ns=0,stamp_ns=0,attitude_stamp_ns=0,landed=True,armed=False,range_m=.101,velocity_body_mps=(.03,0,0)),0))
        self.assertIn('Stationary',c.last_rejection)

    def test_foot_calibration_continuous_native_range_window_and_error_bound(self):
        d=domain();c=RelativeFootCalibrator(d,MOUNT,AXIS,diagnostic_only=True)
        for i in range(10):
            s=sample(i*50000000,.101,False,True);self.assertTrue(c.ingest(s,s.stamp_ns))
        self.assertIsNone(c.result(s.stamp_ns))
        self.assertFalse(c.ingest(sample(900000000,.101,False,True),900000000));self.assertEqual(c.diagnostics(900000000)['sample_count'],0)
        c=RelativeFootCalibrator(d,MOUNT,AXIS)
        self.assertFalse(c.ingest(sample(STAMP,.101,False,True),STAMP));self.assertIn('qualification missing',c.last_rejection)

if __name__=='__main__':unittest.main()
