import unittest
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation
from task_uncertainty import UncertaintyPolicy
from competition_inputs import BodyHistory,ExactFrameJoiner,PriorControlLedger
from task_safety import MeasuredMapGuard
from task_executive import Feedback,TaskExecutive,MissionLimits
from task_landing import RangePose
from task_contact import ContactLandingAdapter
from online_map import OnlineMap


def mapp():
    shape=(80,80,80)
    return OnlineMap([-2.,-2.,-2.,.05,*shape,0,2.],np.zeros(shape,np.int8),.32,.16)


def guard(m,error=.1542856,reference=.1542856):
    p=UncertaintyPolicy(map_pose_error_m=reference)
    return MeasuredMapGuard(m,'e',.32,.16,extra_collision_margin_m=p.extra(error),uncertainty=p.evidence(error))


class UncertaintyTests(unittest.TestCase):
    def test_invalid_crosscovariance_cannot_pass_on_small_diagonal_alone(self):
        h=BodyHistory();cov=np.eye(6)*1e-6;cov[0,1]=cov[1,0]=.01
        with self.assertRaisesRegex(ValueError,'positive semidefinite'):
            h.add(2000000000,[0,0,0],[0,0,0,1],[0,0,0],[0,0,0],cov)
        self.assertFalse(h.items)

    def test_actual_154_covariance_preserved_and_418_refused(self):
        h=BodyHistory();cov=np.diag([.00261895,.00263843,.00261913,1e-5,1e-5,1e-5])
        h.add(2000000000,[0,0,0],[0,0,0,1],[0,0,0],[0,0,0],cov)
        self.assertAlmostEqual(h.items[-1].error_m,3*np.sqrt(.00263843))
        np.testing.assert_array_equal(h.items[-1].pose_covariance,cov)
        with self.assertRaises(ValueError):h.add(2020000000,[0,0,0],[0,0,0,1],[0,0,0],[0,0,0],np.eye(6)*(.418/3)**2)
        self.assertEqual(len(h.items),1)
        self.assertAlmostEqual(h.observed_position_error_max_m,.418)

    def test_exact_join_admits_154_for_observation_without_clipping(self):
        h=BodyHistory();cov=np.eye(6)*(.154/3)**2
        h.add(2000000000,[0,0,0],[0,0,0,1],[0,0,0],[0,0,0],cov)
        j=ExactFrameJoiner([60,60,32,24],np.eye(4),'e','cal',(64,48))
        for kind,value in [('left',np.zeros((48,64),np.uint8)),('depth',np.ones((48,64))),('camera',np.eye(4))]:j.add(kind,2000000000,value)
        f=j.take(h,2.)[0]
        self.assertAlmostEqual(f.exposure.pose_error_m,.154)
        self.assertEqual(f.exposure.uncertainty_limit_m,.20)
        Feedback(2.,[0,0,0],'e',False,True,0.,position_error_m=.154).validate(2.)

    def test_actual_excess_and_reference_sum_only_credit_declared_margin(self):
        p=UncertaintyPolicy(map_pose_error_m=.154)
        np.testing.assert_allclose(p.extra(.154),[.268,.268,.308])
        np.testing.assert_allclose(np.array([.32,.32,.16])+p.extra(.154),[.588,.588,.468])
        depth_policy=UncertaintyPolicy(map_pose_error_m=.154,map_geometry_error_m=.02)
        np.testing.assert_allclose(depth_policy.extra(.154),[.288,.288,.328])
        self.assertEqual(depth_policy.evidence(.154)['tracking_allowance_m'],.04)
        self.assertFalse(p.evidence(.154)['gauge_covariance_subtracted'])

    def test_full_inflated_swept_body_refuses_new_unknown_shell(self):
        m=mapp();s=m.state.copy();s[tuple(m.idx([0,.5,0]))]=-1
        m=OnlineMap([*m.origin,m.res,*m.shape,0,2.],s,.32,.16)
        self.assertTrue(m.segment([0,0,0],[.2,0,0]))
        self.assertIsNone(guard(m).certify([[0,0,0],[.2,0,0]],2.,'e','inflated'))

    def test_z_shell_and_stopping_tail_are_not_skipped(self):
        for cell,path in [([0,0,.4],[[0,0,0],[.2,0,0]]),([1.05,0,0],[[0,0,0],[.6,0,0]])]:
            m=mapp();s=m.state.copy();s[tuple(m.idx(cell))]=-1
            m=OnlineMap([*m.origin,m.res,*m.shape,0,2.],s,.32,.16)
            self.assertTrue(m.segment(*path))
            self.assertIsNone(guard(m).certify(path,2.,'e','all dimensions'))

    def test_wide_observed_free_pass_and_metadata_keep_actual_error(self):
        c=guard(mapp()).certify([[0,0,0],[.2,0,0]],2.,'e','wide')
        self.assertIsNotNone(c);self.assertTrue(c.covers_motion([0,0,0],[.1,0,0],2.1,'e'))
        self.assertEqual(c.as_dict()['uncertainty']['body_three_sigma_proxy_m'],.1542856)
        self.assertFalse(c.covers_uncertainty(.16))

    def test_contact_global_admission_does_not_relax_local_precision(self):
        a=ContactLandingAdapter([0,0,0],[0,0,-1],'cal')
        rs=RangePose(2.,2.,2.,.1,[0,0,.1],np.eye(3),[0,0,0],[0,0,0],'e',True,False,True,True,.154)
        self.assertIsNone(a.certify(guard(mapp()),None,None,rs,2.))
        self.assertIn('4 cm',a.last_rejection)

    def test_larger_feedback_uncertainty_cannot_retroactively_expand_prior_certificate(self):
        h=BodyHistory()
        for t,error in [(2.,.154),(2.1,.16)]:
            h.add(round(t*1e9),[0,0,0],[0,0,0,1],[0,0,0],[0,0,0],np.eye(6)*(error/3)**2)
        ledger=PriorControlLedger();ledger.add(guard(mapp(),.154).certify([[0,0,0],[0,0,0]],2.,'e','prior envelope'))
        self.assertFalse(ledger.interval(2.,2.1,h,'e').covers_motion([0,0,0],[0,0,0],2.1,'e'))

    def test_policy_uncertainty_expands_aperture_geometry(self):
        from task_observations import TargetTrack
        t=TargetTrack('x','window','rectangle','e',np.zeros(3),np.array([1.,0,0]),np.array([.8,.8]),.03,2.,1.)
        t.boundary=np.array([[0,-.4,-.4],[0,.4,-.4],[0,.4,.4],[0,-.4,.4]])
        e=TaskExecutive(MissionLimits(.32,.16,uncertainty_policy=UncertaintyPolicy()),[0,0,0],'e')
        e.current_position_error_m=.03;self.assertTrue(e._aperture_body_fits(t))
        e.current_position_error_m=.154;self.assertFalse(e._aperture_body_fits(t))

if __name__=='__main__':unittest.main()
