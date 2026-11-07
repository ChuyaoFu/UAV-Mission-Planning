"""Navigation-only startup qualification; explicit NO-DDS callbacks."""
import os,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from competition_ros_stubs import install
M=install()
import navigate as app
from scipy.spatial.transform import Rotation

def odom(t=2.,p=(1.,-2.,.4),yaw=.3):
    m=M.Odometry();ns=round(t*1e9);m.header=NS(stamp=NS(sec=ns//10**9,nanosec=ns%10**9),frame_id='world')
    m.pose.pose.position=NS(x=p[0],y=p[1],z=p[2]);q=Rotation.from_euler('z',yaw).as_quat();m.pose.pose.orientation=NS(x=q[0],y=q[1],z=q[2],w=q[3]);m.pose.covariance=np.eye(6).ravel()*1e-6
    return m

def px(t=2.,p=(3.,4.,-1.),**kwargs):
    return M.VehicleLocalPosition(timestamp=round(t*1e6),timestamp_sample=round(t*1e6),x=p[0],y=p[1],z=p[2],heading=.7,
          heading_good_for_control=True,xy_reset_counter=0,z_reset_counter=0,heading_reset_counter=0,vxy_reset_counter=0,vz_reset_counter=0,
          delta_xy=[0.,0.],delta_z=0.,ref_timestamp=0,vx=0.,vy=0.,vz=0.,v_xy_valid=True,v_z_valid=True,eph=.02,epv=.02,evh=.01,evv=.01,heading_var=.0001,**kwargs)



class Qualified(Exception):pass

class BaselineAnchorTests(unittest.TestCase):
    def setUp(self):
        with patch.object(app,'ROOT',Path(os.environ['ROBOT_CALIBRATION']).parents[1]):
            self.n=app.Navigator(str(Path.cwd()/'work/baseline-anchor/result.json'))
        self.n.test_time=2.;self.n.map=NS(stamp=2.);self.last_status=-99.;self.last_land=-99.
    def tearDown(self):self.n.worker.close()
    def feed(self,t,*,body=(1.,-2.,.4),native=(3.,4.,-1.),body_v=(0,0,0),body_w=(0,0,0),native_v=(0,0,0),armed=False,landed=True,at_rest=True,freefall=False,counter=0,cadence=False,frame='world'):
        n=self.n;n.test_time=t
        n.localization_health(NS(data=__import__('json').dumps(dict(valid=True,stamp=t,reason='no_dds_fixture',accepted_visual_update_features=20,visual_update_age=0.))))
        if not cadence or t-self.last_status>=.504:
            n.vehicle_status(M.VehicleStatus(timestamp=round(t*1e6),arming_state=2 if armed else 1));self.last_status=t
        if landed is not None and (not cadence or t-self.last_land>=1.):
            n.baseline_land_detected(M.VehicleLandDetected(timestamp=round(t*1e6),landed=landed,at_rest=at_rest,freefall=freefall));self.last_land=t
        m=px(t,native);m.vx,m.vy,m.vz=native_v;m.xy_reset_counter=counter;n.local_position(m)
        m=odom(t,body);m.header.frame_id=frame;m.twist.twist.linear=NS(x=body_v[0],y=body_v[1],z=body_v[2]);m.twist.twist.angular=NS(x=body_w[0],y=body_w[1],z=body_w[2]);n.odom(m)
        n.map=NS(stamp=t)
    def begin(self):self.n.begin_baseline_alignment_qualification()
    def poll(self):return self.n.qualify_baseline_alignment_once(establish=True)
    def assert_no_commands(self):
        self.assertFalse(self.n.traj.messages);self.assertFalse(self.n.mode.messages);self.assertFalse(self.n.command_pub.messages)
    def test_original_one_second_twenty_samples_with_real_native_cadence(self):
        self.begin()
        for i in range(20):self.feed(round(2+i*.05,6),cadence=True);self.assertFalse(self.poll())
        self.assert_no_commands();self.feed(3.,cadence=True);self.assertTrue(self.poll())
        self.assertIsNotNone(self.n.control_alignment.anchor);self.assert_no_commands()
        self.assertEqual(len(self.n.subscriptions['/fmu/out/vehicle_land_detected']),1)
    def test_repeated_poll_and_duplicate_body_do_not_accumulate_dwell(self):
        self.begin();self.feed(2.)
        for _ in range(50):self.assertFalse(self.poll())
        self.assertEqual(len(self.n._baseline_samples),1);self.assert_no_commands()
    def test_disarmed_alone_never_implies_landed(self):
        self.begin()
        for i in range(26):self.feed(round(2+i*.05,6),landed=None);self.assertFalse(self.poll())
        self.assert_no_commands()
    def test_armed_airborne_freefall_or_native_motion_prevents_anchor(self):
        self.begin();clock=2.
        for fields in ({'armed':True},{'landed':False},{'at_rest':False},{'freefall':True},{'native_v':(.031,0,0)},{'body_v':(.031,0,0)},{'body_w':(0,0,.031)}):
            for i in range(23):clock=round(clock+.05,6);self.feed(clock,**fields);self.assertFalse(self.poll())
        self.assertIsNone(self.n.control_alignment.rotation);self.assert_no_commands()
    def test_existing_upright_domain_and_explicit_health_epoch_are_enforced(self):
        self.begin()
        for i in range(12):self.feed(round(2+i*.05,6));self.assertFalse(self.poll())
        self.n.health['home_epoch_ns']=2
        self.assertFalse(self.poll());self.assertLessEqual(len(self.n._baseline_samples),1)
        n=self.n;n.q=Rotation.from_euler('xyz',[3,0,0],degrees=True)
        self.assertFalse(self.poll());self.assertEqual(len(n._baseline_samples),0);self.assert_no_commands()
    def test_actual_spread_loss_restarts_full_dwell(self):
        self.begin()
        for i in range(11):self.feed(round(2+i*.05,6));self.assertFalse(self.poll())
        self.feed(2.55,body=(1.013,-2,.4));self.assertFalse(self.poll())
        for i in range(1,20):self.feed(round(2.55+i*.05,6),body=(1.013,-2,.4));self.assertFalse(self.poll())
        self.feed(3.6,body=(1.013,-2,.4));self.assertTrue(self.poll());self.assert_no_commands()
    def test_native_reset_restarts_unanchored_window_without_poison(self):
        self.begin()
        for i in range(13):self.feed(round(2+i*.05,6));self.assertFalse(self.poll())
        self.feed(2.65,native=(3.2,4,-1),counter=1);self.assertFalse(self.poll())
        for i in range(1,20):self.feed(round(2.65+i*.05,6),native=(3.2,4,-1),counter=1);self.assertFalse(self.poll())
        self.feed(3.7,native=(3.2,4,-1),counter=1);self.assertTrue(self.poll())
        self.assertIsNone(self.n.control_alignment.fault);self.assert_no_commands()
    def test_body_epoch_change_and_stale_native_land_reset_window(self):
        self.begin()
        for i in range(12):self.feed(round(2+i*.05,6));self.assertFalse(self.poll())
        self.feed(2.6,frame='new_world');self.assertFalse(self.poll())
        self.assertIsNone(self.n.control_alignment.rotation)
        self.feed(2.65,frame='new_world');self.assertFalse(self.poll())
        self.n._baseline_land_t=1.;self.assertFalse(self.poll());self.assertEqual(len(self.n._baseline_samples),0)
        self.assert_no_commands()
    def test_original_body_gap_larger_than_150ms_restarts(self):
        self.begin();self.feed(2.);self.poll();self.feed(2.1);self.poll();self.feed(2.3);self.assertFalse(self.poll())
        self.assertLessEqual(len(self.n._baseline_samples),1);self.assert_no_commands()
    def test_run_reaches_prime_only_after_qualified_anchor_before_any_arm(self):
        n=self.n;first=[]
        def pump(*args):self.feed(round(n.test_time+.05,6),cadence=True)
        n.pump=pump
        publish=n.mode.publish
        def checked_publish(msg):
            self.assertIsNotNone(n.control_alignment.anchor);self.assertGreaterEqual(n._baseline_samples[-1][0]-n._baseline_samples[0][0],1.)
            if not first:first.append(n.now())
            publish(msg)
        n.mode.publish=checked_publish
        def finish_prime(hold):raise Qualified()
        n.enter_offboard_and_arm=finish_prime
        with self.assertRaises(Qualified):n.run()
        self.assertGreaterEqual(first[0],3.);self.assertEqual(len(n.mode.messages),50);self.assertFalse(n.command_pub.messages)
    def test_complete_task_manual_anchor_path_has_no_extra_land_subscription(self):
        n=self.n;self.assertFalse(n._baseline_dwell_active);self.assertIsNone(n._baseline_land_subscription)
        n.test_time=2.;n.health=dict(valid=True,stamp=2.,reason='no_dds_fixture');n.local_position(px());n.odom(odom());n.vehicle_status(M.VehicleStatus(timestamp=2000000,arming_state=1))
        n.establish_control_alignment('existing_mechanical_dwell',stationary_confirmed=True)
        self.assertIsNotNone(n.control_alignment.rotation);self.assertNotIn('/fmu/out/vehicle_land_detected',n.subscriptions)
if __name__=='__main__':unittest.main()
