import unittest,dataclasses
from types import SimpleNamespace as NS
import numpy as np
from competition_inputs import BodyHistory,ExactFrameJoiner,NativeState,PriorControlLedger,image_array,range_pose
from task_safety import MeasuredMapGuard
from online_map import OnlineMap
from task_uncertainty import UncertaintyPolicy


def body(history,t,p=(0,0,.7),q=(0,0,0,1)):
    return history.add(int(round(t*1e9)),p,q,[.1,0,0],[0,0,0],np.diag([1e-6]*6),'world')

def raw_header(t):
    ns=int(round(t*1e9));return NS(stamp=NS(sec=ns//1000000000,nanosec=ns%1000000000),frame_id='world')

def free_map(stamp=2.):
    shape=(50,50,35);return OnlineMap([-2.5,-2.5,-1.,.1,*shape,0,stamp],np.zeros(shape,np.int8),.32,.16)


class InputTests(unittest.TestCase):
    def test_position_covariance_is_variance_not_std_or_orientation(self):
        h=BodyHistory();cov=np.diag([.001,.0004,.0001,4.,9.,16.])
        self.assertTrue(h.add(1000000000,[0,0,.7],[0,0,0,1],[0,0,0],[0,0,0],cov))
        self.assertAlmostEqual(h.items[-1].error_m,3*np.sqrt(.001))

    def test_exact_three_sigma_budget_is_inclusive_without_variance_relaxation(self):
        h=BodyHistory(uncertainty_policy=UncertaintyPolicy(admission_limit_m=.15));cov=np.diag([.0025]*3+[1e-6]*3)
        self.assertTrue(h.add(1000000000,[0,0,.7],[0,0,0,1],[0,0,0],[0,0,0],cov))
        self.assertEqual(h.items[-1].error_m,.15)
        cov[0,0]=np.nextafter(.0025,np.inf)
        with self.assertRaisesRegex(ValueError,'uncertainty exceeds'):
            h.add(1020000000,[0,0,.7],[0,0,0,1],[0,0,0],[0,0,0],cov)
        self.assertEqual(len(h.items),1)

    def test_fusion_acceptable_covariance_is_not_automatically_task_acceptable(self):
        # These are numerical examples, NOT covariance measured in the run.
        for variance in (.0036,.0064,.0099,.0399):
            h=BodyHistory(uncertainty_policy=UncertaintyPolicy(admission_limit_m=.15))
            with self.assertRaisesRegex(ValueError,'uncertainty exceeds'):
                h.add(1000000000,[0,0,.7],[0,0,0,1],[0,0,0],[0,0,0],np.diag([variance]*3+[1e-6]*3))
            self.assertEqual(len(h.items),0)

    def test_body_slerp_and_velocity_are_in_body_then_expressed_in_local(self):
        h=BodyHistory();body(h,1.,q=(0,0,0,1));body(h,1.1,q=(0,0,np.sin(.1),np.cos(.1)))
        middle=h.sample(1050000000)
        self.assertAlmostEqual(middle.rotation.as_euler('xyz')[2],.1)
        np.testing.assert_allclose(h.items[-1].velocity,[.1*np.cos(.2),.1*np.sin(.2),0.])
        self.assertIsNone(h.sample(1100000001))

    def test_missing_pose_bracket_and_long_gap_do_not_extrapolate(self):
        h=BodyHistory();body(h,1.);body(h,1.2)
        self.assertIsNone(h.sample(1100000000));self.assertIsNone(h.sample(999999999))

    def test_pose_jump_frame_change_and_time_reset_are_latched_faults(self):
        for action in ('jump','frame','time'):
            h=BodyHistory();body(h,1.)
            if action=='jump':self.assertFalse(body(h,1.02,p=(1,0,.7)))
            elif action=='frame':self.assertFalse(h.add(1020000000,[0,0,.7],[0,0,0,1],[0,0,0],[0,0,0],np.eye(6)*1e-6,'reset_world'))
            else:self.assertFalse(body(h,.99))
            self.assertIsNotNone(h.fault)
            self.assertIsNone(h.sample(1000000000))

    def test_pixel_parser_respects_stride_endian_and_units(self):
        values=np.array([[1.,np.nan],[2.,3.]],dtype='>f4');padded=b''.join(row.tobytes()+b'xxxx' for row in values)
        m=NS(encoding='32FC1',width=2,height=2,step=12,data=padded,is_bigendian=True)
        result=image_array(m,'32FC1',(2,2));np.testing.assert_allclose(result,values,equal_nan=True)
        m.encoding='16UC1'
        with self.assertRaises(ValueError):image_array(m,'32FC1',(2,2))

    def test_exact_join_does_not_nearest_match_a_different_left_exposure(self):
        h=BodyHistory();body(h,2.);T=np.eye(4);T[:3,3]=[0,0,.7]
        j=ExactFrameJoiner([60,60,32,24],np.eye(4),'epoch','cal',(64,48))
        j.add('left',2000000001,np.zeros((48,64),np.uint8));j.add('depth',2000000000,np.ones((48,64)))
        j.add('camera',2000000000,T)
        self.assertEqual(j.take(h,2.),[])
        j.add('left',2000000000,np.zeros((48,64),np.uint8));frames=j.take(h,2.)
        self.assertEqual(len(frames),1);self.assertEqual(frames[0].stamp_ns,2000000000)

    def test_join_requires_matching_exposure_time_calibrated_camera_pose(self):
        h=BodyHistory();body(h,2.);j=ExactFrameJoiner([60,60,32,24],np.eye(4),'epoch','cal',(64,48))
        for key in ('left','depth'):j.add(key,2000000000,np.zeros((48,64)))
        wrong=np.eye(4);wrong[:3,3]=[.1,0,.7];j.add('camera',2000000000,wrong)
        self.assertEqual(j.take(h,2.),[]);self.assertEqual(j.rejected,1)

    def test_native_freshness_uses_receipt_clock_not_px4_boot_microseconds(self):
        native=NativeState();status=NS(timestamp=9000000000,arming_state=1);land=NS(timestamp=8000000000,landed=True,freefall=False)
        native.update('status',status,2.);native.update('land',land,2.)
        self.assertEqual(native.flags(2.1,2),(False,True))
        self.assertIsNone(native.flags(2.4,2))
        self.assertFalse(native.update('land',land,2.4))
        self.assertIsNone(native.flags(2.4,2))

    def test_range_is_joined_to_bracketed_accepted_body_and_actual_native_landed(self):
        h=BodyHistory();body(h,2.);body(h,2.1);s=h.sample(2050000000)
        m=NS(header=raw_header(2.05),range=.1,min_range=.05,max_range=4.)
        self.assertIsNone(range_pose(m,None,'epoch',(False,True),2.06,True))
        self.assertIsNone(range_pose(m,s,'epoch',None,2.06,True))
        packet=range_pose(m,s,'epoch',(False,True),2.06,True)
        self.assertIsNotNone(packet);self.assertEqual(packet.pose_stamp,packet.range_stamp)
        self.assertTrue(packet.landed)


    def test_map_float_roundoff_is_tolerated_but_actual_future_time_is_rejected(self):
        m=free_map(2.+1e-12);g=MeasuredMapGuard(m,'epoch',.32,.16)
        self.assertIsNotNone(g.certify([[0,0,.7],[.1,0,.7]],2.,'epoch','numeric nanosecond resolution'))
        m=free_map(2.+.0001);g=MeasuredMapGuard(m,'epoch',.32,.16)
        self.assertIsNone(g.certify([[0,0,.7],[.1,0,.7]],2.,'epoch','real future map'))

    def test_caches_are_bounded(self):
        j=ExactFrameJoiner([60,60,32,24],np.eye(4),'epoch','cal',(64,48))
        for i in range(100):
            j.add('left',i,np.zeros((1,1)));j.add('depth',i,np.zeros((1,1)));j.add('camera',i,np.eye(4))
        self.assertEqual(len(j.buffers['left']),24);self.assertEqual(len(j.buffers['depth']),6)


class LedgerTests(unittest.TestCase):
    def history(self):
        h=BodyHistory()
        for i in range(5):body(h,2.+i*.05,p=(i*.01,0,.7))
        return h

    def certificate(self,issued=2.):
        m=free_map(issued)
        return MeasuredMapGuard(m,'epoch',.32,.16).certify([[0,0,.7],[.2,0,.7]],issued,'epoch','published')

    def test_complete_prior_control_chain_covers_delayed_accepted_body_feedback(self):
        h=self.history();ledger=PriorControlLedger();ledger.add(self.certificate(2.));ledger.add(self.certificate(2.1))
        a=ledger.interval(2.,2.2,h,'epoch')
        self.assertTrue(a.covers_motion([0,0,.7],[.04,0,.7],2.22,'epoch'))
        self.assertGreaterEqual(ledger.audited_segments,4)

    def test_command_or_map_created_after_motion_cannot_retroactively_authorize_it(self):
        h=self.history();ledger=PriorControlLedger();ledger.add(self.certificate(2.1))
        self.assertFalse(ledger.interval(2.,2.2,h,'epoch').covers_motion([0,0,.7],[.04,0,.7],2.22,'epoch'))

    def test_expired_gap_or_changed_epoch_cannot_be_stitched_into_success(self):
        h=self.history();ledger=PriorControlLedger();ledger.add(dataclasses.replace(self.certificate(),valid_until=2.08));ledger.add(self.certificate(2.15))
        self.assertFalse(ledger.interval(2.,2.2,h,'epoch').covers_motion([0,0,.7],[.04,0,.7],2.22,'epoch'))
        ledger=PriorControlLedger();ledger.add(self.certificate())
        self.assertFalse(ledger.interval(2.,2.2,h,'epoch').covers_motion([0,0,.7],[.04,0,.7],2.22,'reset'))


    def test_missing_accepted_body_interval_cannot_be_hidden_by_exact_endpoints(self):
        h=BodyHistory();body(h,2.,p=(0,0,.7));body(h,2.2,p=(.02,0,.7))
        ledger=PriorControlLedger();ledger.add(self.certificate())
        self.assertFalse(ledger.interval(2.,2.2,h,'epoch').covers_motion([0,0,.7],[.02,0,.7],2.22,'epoch'))

    def test_actual_feedback_positions_must_match_the_accepted_history(self):
        h=self.history();ledger=PriorControlLedger();ledger.add(self.certificate())
        self.assertFalse(ledger.interval(2.,2.2,h,'epoch').covers_motion([.01,0,.7],[.04,0,.7],2.22,'epoch'))

if __name__=='__main__':unittest.main()
