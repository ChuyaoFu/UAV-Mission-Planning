"""Actual Navigator methods with EXPLICIT no-DDS ROS/PX4 stubs."""
import math,unittest,os
from types import SimpleNamespace as NS
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from competition_ros_stubs import install
M=install()
import navigate as app
from navigate import Navigator
from unittest.mock import patch

def odom(t=2.,p=(1.,-2.,.4),yaw=.3):
    m=M.Odometry();ns=round(t*1e9);m.header=NS(stamp=NS(sec=ns//10**9,nanosec=ns%10**9),frame_id='world')
    m.pose.pose.position=NS(x=p[0],y=p[1],z=p[2]);q=Rotation.from_euler('z',yaw).as_quat();m.pose.pose.orientation=NS(x=q[0],y=q[1],z=q[2],w=q[3]);m.pose.covariance=np.eye(6).ravel()*1e-6
    return m

def px(t=2.,p=(3.,4.,-1.),**kwargs):
    return M.VehicleLocalPosition(timestamp=round(t*1e6),timestamp_sample=round(t*1e6),x=p[0],y=p[1],z=p[2],heading=.7,
          heading_good_for_control=True,xy_reset_counter=0,z_reset_counter=0,heading_reset_counter=0,vxy_reset_counter=0,vz_reset_counter=0,
          delta_xy=[0.,0.],delta_z=0.,ref_timestamp=0,vx=0.,vy=0.,vz=0.,v_xy_valid=True,v_z_valid=True,eph=.02,epv=.02,evh=.01,evv=.01,heading_var=.0001,**kwargs)

def command(t=2.,trajectory_id=1):
    return NS(header=NS(stamp=NS(sec=int(t),nanosec=round((t-int(t))*1e9)),frame_id='world'),trajectory_flag=1,trajectory_id=trajectory_id,
              position=NS(x=1.6,y=-1.8,z=.55),velocity=NS(x=.4,y=-.2,z=.1),acceleration=NS(x=.2,y=.1,z=-.1))

class NavigatorBridgeTests(unittest.TestCase):
    def setUp(self):
        with patch.object(app,'ROOT',Path(os.environ['ROBOT_CALIBRATION']).parents[1]):
            self.n=Navigator(str(Path.cwd()/'work/ego-native-bridge/no-dds.json'))
        n=self.n;n.test_time=2.;n.pump=lambda *a,**kw:None
        n.health=dict(valid=True,stamp=2.,reason='synthetic_no_dds',accepted_visual_update_features=20,visual_update_age=0.)
        n.local_position(px());n.odom(odom());n.vehicle_status(M.VehicleStatus(timestamp=2000000,arming_state=1))
        n.establish_control_alignment('synthetic_mechanical_dwell',stationary_confirmed=True)
    def tearDown(self):self.n.worker.close()
    def test_follow_forwards_unclipped_original_native_pva(self):
        n=self.n;n.map=NS(stamp=2.,segment=lambda *a:True);n.phase='RETURN_ON_OBSERVED_MAP';r=command();n.trajectory(r);n.follow()
        msg=n.traj.messages[-1];a=n.control_alignment
        np.testing.assert_allclose(msg.position,a.position([1.6,-1.8,.55]));np.testing.assert_allclose(msg.velocity,a.derivative([.4,-.2,.1]));np.testing.assert_allclose(msg.acceleration,a.derivative([.2,.1,-.1]))
        self.assertEqual(n.reference.header.stamp.sec,2);self.assertEqual(n.ref_t,2.)
    def test_send_fixed_target_not_dragged_by_new_estimates(self):
        n=self.n;goal=np.array([1.5,-1.9,.6]);n.send(goal,.3,[.2,0,0],[.1,0,0]);before=n.traj.messages[-1]
        n.test_time=2.05;n.odom(odom(2.05,p=(1.01,-2.,.4)));n.local_position(px(2.05,p=(3.05,4.,-1.)))
        n.send(goal,.3,[.2,0,0],[.1,0,0]);after=n.traj.messages[-1]
        np.testing.assert_array_equal(after.position,before.position);np.testing.assert_array_equal(after.velocity,before.velocity);np.testing.assert_array_equal(after.acceleration,before.acceleration)
    def test_invalid_initial_px_waits_and_does_not_poison_anchor(self):
        n=self.n;n.control_alignment=type(n.control_alignment)();m=px();m.xy_valid=False;n.local_position(m)
        self.assertIsNone(n.control_alignment.fault);self.assertIsNone(n.control_alignment.native)
        n.local_position(px());n.odom(odom());n.establish_control_alignment('new_qualified_dwell',stationary_confirmed=True)
        n.send(n.body,.3);self.assertTrue(n.traj.messages)
    def test_frd_ev_unaligned_heading_flag_is_not_a_readiness_gate(self):
        n=self.n;n.control_alignment=type(n.control_alignment)();m=px();m.heading_good_for_control=False
        n.local_position(m);n.odom(odom());n.establish_control_alignment('qualified_frd_ev_dwell',stationary_confirmed=True)
        n.send(n.body,.3);self.assertTrue(n.traj.messages)
    def test_native_translation_reset_moves_targets_by_delta_only(self):
        n=self.n;goal=n.body+[.2,0,.1];n.send(goal,.3,[.1,.2,0],[.2,0,0]);first=n.traj.messages[-1]
        n.test_time=2.05;n.odom(odom(2.05));m=px(2.05,p=(3.4,3.8,-.7));m.xy_reset_counter=1;m.z_reset_counter=1;m.delta_xy=[.4,-.2];m.delta_z=.3;n.local_position(m)
        n.send(goal,.3,[.1,.2,0],[.2,0,0]);second=n.traj.messages[-1]
        np.testing.assert_allclose(np.array(second.position)-first.position,[.4,-.2,.3]);np.testing.assert_array_equal(second.velocity,first.velocity);np.testing.assert_array_equal(second.acceleration,first.acceleration)
    def test_unqualified_and_stale_bridge_never_publish(self):
        n=self.n;n.control_alignment.rotation=None
        with self.assertRaises(ValueError):n.send(n.body,.3)
        self.assertFalse(n.traj.messages)
        n=self.n;n.test_time=2.4
        with self.assertRaises(ValueError):n.send(n.body,.3)
        self.assertFalse(n.traj.messages)
    def test_brake_without_replan_latches_one_world_target(self):
        n=self.n;called=[];n.replan_if_held=lambda:called.append(True);n.brake(replan=False);first=n.traj.messages[-1].position
        n.body=n.body+[.02,0,0];n.brake(replan=False);np.testing.assert_array_equal(n.traj.messages[-1].position,first);self.assertFalse(called)
        n.brake();self.assertEqual(called,[True])
    def test_unknown_and_stale_ego_reference_still_brake(self):
        n=self.n;n.map=NS(stamp=2.,segment=lambda *a:False);n.replan_if_held=lambda:None;n.trajectory(command());n.follow()
        msg=n.traj.messages[-1];np.testing.assert_allclose(msg.position,n.control_alignment.position(n.body));self.assertTrue(np.isnan(msg.acceleration).all());self.assertGreater(n.safety_vetoes,0)
        n.reference=command(1.7,2);n.ref_t=1.7;n.map=NS(stamp=2.,segment=lambda *a:True);n.follow();self.assertIn('reference_stale',n.veto_reasons)
if __name__=='__main__':unittest.main()
