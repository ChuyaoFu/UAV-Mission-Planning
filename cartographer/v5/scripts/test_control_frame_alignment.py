import math,unittest
from dataclasses import replace
from types import SimpleNamespace as NS
import numpy as np
from control_frame_alignment import MeasuredFrameAlignment,NativeMeasurement,position_command_sample,ned_rotation

def native(t=2.,p=(3.,4.,-1.),heading=.7,counters=(0,0,0,0,0),delta_xy=(0.,0.),delta_z=0.):
    return NativeMeasurement(int(t*1e6),t,np.array(p),heading,counters,np.array(delta_xy),delta_z,0,int(t*1e6),np.zeros(3),.01,.01,.01)

def anchored():
    a=MeasuredFrameAlignment();a.observe_body(2.,[1.,-2.,.4],.3,[0,0,0],'world',.01,.01,0.);a.observe_native(native())
    a.establish(2.,anchor_id='synthetic_stationary_measured_pair',stationary_confirmed=True,disarmed_confirmed=True)
    return a

class AlignmentTests(unittest.TestCase):
    def test_nonzero_origins_yaws_and_proper_rotation(self):
        a=anchored();np.testing.assert_allclose(a.position(a.body.position),a.native.position)
        self.assertAlmostEqual(a.yaw(.3),.7);np.testing.assert_allclose(a.rotation@a.rotation.T,np.eye(3),atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(a.rotation),1.)
        for yaw in (-2.,0.,1.5):
            v=np.array([math.cos(yaw),math.sin(yaw),0.]);out=a.derivative(v)
            self.assertAlmostEqual(math.atan2(out[1],out[0]),a.yaw(yaw))
    def test_no_reanchoring_when_measurements_drift(self):
        a=anchored();before=a.position([2,1,.7]);R=a.rotation.copy();t=a.translation.copy()
        a.observe_body(2.05,a.body.position+[.01,0,0],.31,[.2,0,0],'world',.01,.01,.2)
        a.observe_native(native(2.05,p=(3.01,4.,-1.),heading=.69));a.validate(2.05)
        np.testing.assert_array_equal(a.rotation,R);np.testing.assert_array_equal(a.translation,t);np.testing.assert_array_equal(a.position([2,1,.7]),before)
    def test_analytic_pva_derivatives_survive_fixed_transform(self):
        a=anchored();h=1e-4;t=.8
        p=lambda t:np.array([t*t*t,2*t*t,-t])
        v=np.array([3*t*t,4*t,-1.]);acc=np.array([6*t,4.,0.])
        np.testing.assert_allclose((a.position(p(t+h))-a.position(p(t-h)))/(2*h),a.derivative(v),atol=2e-8)
        np.testing.assert_allclose((a.position(p(t+h))-2*a.position(p(t))+a.position(p(t-h)))/h**2,a.derivative(acc),atol=2e-7)
    def test_xy_and_z_reset_add_translation_once_no_rotation(self):
        a=anchored();old=a.position([2,1,.7]);R=a.rotation.copy()
        delta=np.array([.4,-.7,.2]);m=native(2.05,p=a.native.position+delta,counters=(1,1,0,0,0),delta_xy=delta[:2],delta_z=delta[2])
        a.observe_native(m);np.testing.assert_allclose(a.position([2,1,.7]),old+delta);np.testing.assert_array_equal(a.rotation,R)
        self.assertFalse(a.observe_native(m));self.assertEqual(a.revision,1)
    def test_counter_wrap_is_one_explicit_reset(self):
        a=anchored();a.native=replace(a.native,counters=(255,0,0,0,0));a.observe_native(native(2.05,p=(3.2,4,-1),counters=(0,0,0,0,0),delta_xy=(.2,0)))
        self.assertEqual(a.revision,1)
    def test_heading_velocity_missed_reset_and_clock_restart_are_latched(self):
        for change in ({'counters':(0,0,1,0,0)},{'counters':(0,0,0,1,0)},{'counters':(2,0,0,0,0)},{'source_us':1}):
            with self.subTest(change=change):
                a=anchored();m=replace(native(2.05),**change)
                with self.assertRaises(ValueError):a.observe_native(m)
                self.assertIsNotNone(a.fault)
                with self.assertRaises(ValueError):a.validate(2.05)
    def test_unannounced_native_and_body_jumps_are_rejected(self):
        for which in ('native','body','heading'):
            a=anchored()
            with self.assertRaises(ValueError):
                if which=='native':a.observe_native(native(2.05,p=(10,4,-1)))
                elif which=='heading':a.observe_native(native(2.05,heading=1.7))
                else:a.observe_body(2.05,[10,-2,.4],.3,[0,0,0],'world',.01,.01,0.)
    def test_initial_resets_wait_do_not_latch_fault(self):
        a=MeasuredFrameAlignment();a.observe_body(2.,[0,0,0],0.,[0,0,0],'world',.01,.01,0.);a.observe_native(native())
        a.observe_native(native(2.05,counters=(4,3,2,1,1),heading=-2.));self.assertIsNone(a.fault)
        self.assertFalse(a.observe_body(1.,[0,0,0],0.,[0,0,0],'new_world',.01,.01,0.));self.assertIsNone(a.fault);self.assertIsNone(a.native)
    def test_missing_anchor_stationarity_and_stale_data_stop(self):
        a=MeasuredFrameAlignment()
        with self.assertRaises(ValueError):a.validate(2.)
        a=anchored()
        with self.assertRaises(ValueError):a.establish(2.,anchor_id='reanchor',stationary_confirmed=True,disarmed_confirmed=True)
        with self.assertRaises(ValueError):a.validate(2.201)
        a=anchored();a.body=replace(a.body,stamp=2.3)
        with self.assertRaises(ValueError):a.validate(2.)
        a=anchored();a.body=replace(a.body,stamp=2.4)
        with self.assertRaises(ValueError):a.validate(2.4)
    def test_bad_ready_frame_future_and_nonfinite_reference_rejected(self):
        m=NS(header=NS(stamp=NS(sec=2,nanosec=0),frame_id='world'),trajectory_id=1,trajectory_flag=1,
             position=NS(x=1,y=2,z=3),velocity=NS(x=.4,y=-.2,z=.1),acceleration=NS(x=.2,y=.1,z=0))
        result=position_command_sample(m,2.,'world');np.testing.assert_array_equal(result[2],[.4,-.2,.1])
        for mutate in ('frame','future','nan','flag'):
            import copy
            n=copy.deepcopy(m)
            if mutate=='frame':n.header.frame_id='map'
            elif mutate=='future':n.header.stamp.sec=3
            elif mutate=='nan':n.acceleration.x=math.nan
            else:n.trajectory_flag=4
            with self.assertRaises(ValueError):position_command_sample(n,2.,'world')
if __name__=='__main__':unittest.main()
