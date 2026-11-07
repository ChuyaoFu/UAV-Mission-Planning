"""Synthetic sensor/qualification fixtures + actual ROBOT model; no course inputs."""
import sys,json,copy,unittest,math,os
from pathlib import Path
from dataclasses import replace
import numpy as np
from task_mechanical_foot import *
from generate_robot_foot_geometry import generate
from test_relative_contact import domain,sample,patch,veto,MOUNT,AXIS,STAMP
from task_relative_contact import RelativeFootCalibrator
GDATA=json.loads(Path(os.environ['ROBOT_FOOT_GEOMETRY']).read_text()) if os.environ.get('ROBOT_FOOT_GEOMETRY') else generate(Path(__file__).resolve().parents[1]/'config/model.sdf')
HASH=GDATA['payload']['source_model_sha256']


def geom(payload=None):
    p=copy.deepcopy(GDATA['payload'] if payload is None else payload)
    return MechanicalFootGeometry(p,canonical_digest(p))


def dom(qualified=True):return replace(domain(qualified),tof_range_error_m=.02,tof_mount_error_m=.005)


def calibrator(d=None,g=None,diagnostic_only=False):
    return MechanicalFootCalibrator(dom() if d is None else d,MOUNT,AXIS,geom() if g is None else g,
        runtime_kind='sitl',loaded_robot_model_sha256=HASH,diagnostic_only=diagnostic_only)


def dwell(c,bias=0.,shift=(0,0,0)):
    for i in range(30):
        t=100000000+i*50000000;s=sample(t,.088+bias+.0001*np.sin(i),False,True)
        p=MechanicalDwellPose(t,'e','synthetic-rig',tuple(np.asarray([0,0,.111])+shift),True)
        assert c.ingest(s,p,t),c.last_rejection
    return c.result(t)


class MechanicalFootTests(unittest.TestCase):
    def test_actual_robot_skid_points_and_strict_support_bound(self):
        points,h,error=geom().validate('sitl',HASH)
        self.assertEqual(len(points),8);self.assertAlmostEqual(h,.111)
        self.assertAlmostEqual(max(abs(points[:,0])),.085);self.assertAlmostEqual(max(abs(points[:,1])),.072)
        self.assertAlmostEqual(geom().payload['point_error_m'],.002)
        self.assertLess(error,.012);self.assertGreater(error,.0117)
        self.assertLess(.02+.005+error,.04)

    def test_support_bound_covers_all_tilt_azimuth_and_all_contacts(self):
        points,h,error=geom().validate('sitl',HASH);theta=geom().payload['contact_tilt_limit_rad']
        for a in np.linspace(0,2*np.pi,721):
            n=np.array([math.sin(theta)*math.cos(a),math.sin(theta)*math.sin(a),math.cos(theta)])
            # Along-up contact difference from scalar [0,0,-h], with worst admitted n.up.
            support=float(np.max(-points@n));difference=(abs(support-h*n[2])+.002)/math.cos(theta)
            self.assertLessEqual(difference,error+1e-12)

    def test_geometry_offset_not_learned_from_tof_noise_or_common_bias(self):
        for bias in (0.,.01):
            c=calibrator();f=dwell(c,bias)
            self.assertIsNotNone(f);self.assertAlmostEqual(f.offset_m,.111)
            self.assertFalse(f.tof_used_to_estimate_mechanical_error);self.assertTrue(f.runtime_qualified)
            self.assertEqual(c.domain.tof_range_error_m,.02);self.assertEqual(c.domain.tof_mount_error_m,.005)
            self.assertLess(f.error_m,.012)
        # Same observed jitter would still trip the existing learned .025+scatter cap.
        old=RelativeFootCalibrator(dom(),MOUNT,AXIS)
        for i in range(30):
            t=100000000+i*50000000;s=sample(t,.088+.0001*np.sin(i),False,True);self.assertTrue(old.ingest(s,t))
        self.assertIsNone(old.result(t))

    def test_tof_is_consistency_not_mechanical_accuracy_qualification(self):
        d=replace(dom(),qualified_sensors=tuple(x for x in dom().qualified_sensors if x!='tof'))
        f=dwell(calibrator(d));self.assertIsNotNone(f);self.assertTrue(f.runtime_qualified)
        # Downstream contact still needs original absolute ToF/stereo/etc qualification.
        p=patch(d);s=sample();r=assess_mechanical_relative_contact(p,s,f,d,MOUNT,AXIS,STAMP,geometry=geom(),runtime_kind='sitl',loaded_robot_model_sha256=HASH)
        self.assertTrue(r['local_geometry_passes']);self.assertFalse(r['local_precision_qualified'])

    def test_continuous_native_pose_tracking_motion_and_tof_vetoes(self):
        for bad in ('armed','airborne','stale_native','badpose','move','tof','duplicate'):
            c=calibrator();self.assertIsNotNone(dwell(c));t=1600000000
            s=sample(t,.088,False,True);p=MechanicalDwellPose(t,'e','synthetic-rig',(0,0,.111),True)
            if bad=='armed':s=replace(s,armed=True)
            elif bad=='airborne':s=replace(s,landed=False)
            elif bad=='stale_native':s=replace(s,native_received_ns=t-400000000)
            elif bad=='badpose':p=replace(p,tracking_valid=False)
            elif bad=='move':p=replace(p,position_local_m=(.013,0,.111))
            elif bad=='tof':s=replace(s,range_m=.2)
            else:s=replace(s,stamp_ns=1550000000,attitude_stamp_ns=1550000000);p=replace(p,stamp_ns=s.stamp_ns)
            self.assertFalse(c.ingest(s,p,t),bad);self.assertEqual(len(c.samples),0)

    def test_global_translation_cancels_in_dwell_but_never_creates_home(self):
        a=dwell(calibrator());b=dwell(calibrator(),shift=(1e6,-2e6,3e6))
        self.assertEqual(a.offset_m,b.offset_m);self.assertEqual(a.error_m,b.error_m)
        self.assertFalse(hasattr(a,'resting_body_local_m'))

    def test_complete_local_contact_and_separate_map_veto(self):
        d=dom();f=dwell(calibrator(d));p=patch(d)
        args=dict(geometry=geom(),runtime_kind='sitl',loaded_robot_model_sha256=HASH)
        no=assess_mechanical_relative_contact(p,sample(),f,d,MOUNT,AXIS,STAMP,**args)
        self.assertTrue(no['local_geometry_passes']);self.assertFalse(no['contact_candidate_eligible'])
        self.assertLess(no['foot_clearance_error_m'],.04);self.assertLess(no['local_goal_error_m'],.04)
        good=veto(p,body_goal_m=tuple(no['body_goal_m']))
        yes=assess_mechanical_relative_contact(p,sample(),f,d,MOUNT,AXIS,STAMP,global_map_veto=good,**args)
        self.assertTrue(yes['contact_candidate_eligible']);self.assertFalse(yes['native_command_issued']);self.assertFalse(yes['motion_certificate_issued'])
        with self.assertRaisesRegex(RelativeContractError,'typed assess_mechanical'):
            assess_relative_contact(p,sample(),f,d,MOUNT,AXIS,STAMP)

    def test_plane_relative_to_body_cone_is_extra_required_not_just_gravity(self):
        d=dom();f=dwell(calibrator(d));p=patch(d);theta=math.radians(5.1)
        p=replace(p,normal_body=(math.sin(theta),0,math.cos(theta)))
        with self.assertRaisesRegex(RelativeContractError,'plane-to-BODY'):
            assess_mechanical_relative_contact(p,sample(),f,d,MOUNT,AXIS,STAMP,geometry=geom(),runtime_kind='sitl',loaded_robot_model_sha256=HASH)

    def test_digest_loaded_model_missing_geometry_and_hardware_domain_fail(self):
        with self.assertRaises(RelativeContractError):geom().validate('sitl','0'*64)
        with self.assertRaises(RelativeContractError):geom().validate('hardware')
        for key,value in [('contact_set_complete',False),('other_collision_geometry_checked',False),('point_error_metric','per-axis'),('other_collision_downward_support_upper_m',.12)]:
            p=copy.deepcopy(GDATA['payload']);p[key]=value
            with self.assertRaises(RelativeContractError):geom(p).validate('sitl',HASH)
        g=geom();g.payload['point_error_m']=.0001
        with self.assertRaises(RelativeContractError):g.validate('sitl',HASH)

    def test_hardware_accepts_only_explicit_measured_record_not_sitl_definition(self):
        p=copy.deepcopy(GDATA['payload']);p['source_scope']='independent-measured-hardware';p['hardware_use_forbidden']=False
        with self.assertRaises(RelativeContractError):geom(p).validate('hardware')
        # MOCK metrology attestation only, not a physical robot measurement.
        p['measurement_record_sha256']='c'*64
        points,h,error=geom(p).validate('hardware');self.assertAlmostEqual(h,.111);self.assertLess(error,.012)

    def test_unqualified_diagnostics_never_gain_runtime_qualification(self):
        d=dom(False);f=dwell(calibrator(d,diagnostic_only=True));self.assertFalse(f.runtime_qualified)
        c=calibrator(d);t=100000000;s=sample(t,.088,False,True);p=MechanicalDwellPose(t,'e','synthetic-rig',(0,0,.111),True)
        self.assertFalse(c.ingest(s,p,t));self.assertIn('qualification missing',c.last_rejection)

if __name__=='__main__':unittest.main()
