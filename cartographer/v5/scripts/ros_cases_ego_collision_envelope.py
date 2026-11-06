"""Actual copied adapter tests; explicit no-DDS fixtures only."""
import os,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from competition_ros_stubs import install
M=install()
sys.modules['rclpy.qos'].DurabilityPolicy.VOLATILE=0
import competition_navigate as app
from task_safety import MeasuredMapGuard
from online_map import OnlineMap
from ego_collision_envelope import Envelope,floor_ns


def odom(t):
    m=M.Odometry();ns=round(t*1e9);m.header.stamp=NS(sec=ns//10**9,nanosec=ns%10**9)
    m.pose.pose.position.z=.124;m.pose.covariance=(np.eye(6)*1e-6).ravel().tolist()
    return m


class AdapterCases(unittest.TestCase):
    def setUp(self):
        self.auth=patch.object(app,'assert_sitl_authorized');self.auth.start()
        self.n=app.CompetitionNavigator(os.environ['COMPETITION_STUB_RESULT'],
            app.CompetitionOptions(.003,.001,.01,map_pose_error_m=0.,require_ego_envelope=True,
                complete_task=True,foot_geometry=str(app.NAV_ROOT/'config/robot-foot-geometry.json'),contact_rig_error_m=.005))
        n=self.n;n.test_time=2.;n.health={'stamp':2.,'valid':True,'reason':'ok_visual','accepted_visual_update_features':20,'visual_update_age':0.}
        n.local_cb(M.VehicleLocalPosition(timestamp=2000000));n.status_cb(M.VehicleStatus(timestamp=2000000));n.land_cb(M.VehicleLandDetected(timestamp=2000000))
        n.odom(odom(1.95));n.odom(odom(2.));n.home=n.body.copy();n.home_pose_error_m=.003;n.yaw=0.
        n.establish_control_alignment('synthetic_stationary_ground_fixture',stationary_confirmed=True)
        shape=(50,50,35);n.map=OnlineMap([-2.5,-2.5,-1.,.1,*shape,0,2.],np.zeros(shape,np.int8),.32,.16)
        n.bootstrap='READY';n.pump=lambda dt=.01:None
        n.ego_envelope_clock=lambda:(round(n.test_time*1e9),round(n.test_time*1e9))
        self.extra=[.1,.11,.12];self.deadline=2.1
        def guard():
            return MeasuredMapGuard(n.map,n.session,.32,.16,max_age_s=.5,
                extra_collision_margin_m=self.extra,
                uncertainty={'causal_forward_reserve':{'valid_until':self.deadline}})
        n.guard=guard

    def tearDown(self):
        for w in (self.n.task_worker,self.n.frontier_worker,self.n.worker):w.close()
        self.auth.stop()

    def ack(self,wire=None):
        self.n.ego_envelope_ack(NS(data=wire or self.n.ego_envelope_pub.messages[-1].data))

    def test_missing_complete_task_forecast_dependency_rejected(self):
        with self.assertRaisesRegex(ValueError,'requires --complete-task'):
            app.CompetitionOptions(.003,.001,.01,require_ego_envelope=True)

    def test_constructor_qos_and_opt_in_cli(self):
        n=self.n;self.assertIn('/nav/ego_collision_envelope_ack',n.subscriptions)
        captured={}
        class Fake:
            failure=None
            def __init__(self,output,options):captured['options']=options
            def run(self):pass
            def finish(self):pass
            def destroy_node(self):pass
        argv=['competition','--output','unused','--tof-error-m','.003','--tof-extrinsic-error-m','.001',
              '--stereo-error-m','.01','--map-pose-error-m','0','--require-ego-envelope',
              '--complete-task','--foot-geometry',str(app.NAV_ROOT/'config/robot-foot-geometry.json'),'--contact-rig-error-m','.005']
        with patch.object(app,'CompetitionNavigator',Fake),patch.object(app,'install_stop_handlers'),patch.object(sys,'argv',argv):app.main()
        self.assertTrue(captured['options'].require_ego_envelope)

    def test_actual_guard_xyz_immutable_and_minimum_deadline(self):
        n=self.n;n.map.stamp=1.55;n.update_ego_envelope();wire=n.ego_envelope_pub.messages[-1].data;r=Envelope.parse(wire)
        self.assertEqual(r.extra,tuple(self.extra));self.assertEqual(r.valid_until_ns,floor_ns(2.05))
        self.extra[0]=.9;self.deadline=2.01
        self.assertEqual(n.ego_envelope_pub.messages[-1].data,wire)
        self.assertEqual(n.ego_envelope.latest_sent,r)

    def test_no_ack_new_goal_brakes_once_and_does_not_follow(self):
        n=self.n;n.update_ego_envelope()
        with patch.object(n,'brake') as brake,patch.object(n,'follow') as follow:
            n.execute_move(n.body+[.05,0,0],'fixture')
        brake.assert_called_once_with(replan=False);follow.assert_not_called()
        self.assertEqual(n.goal_pub.messages,[]);self.assertEqual(n.goals,[])

    def test_previous_ack_allows_goal_after_next_record_sent(self):
        n=self.n;n.update_ego_envelope();self.ack();first=n.ego_envelope.latest_ack
        n.test_time=2.05;self.deadline=2.15;n.update_ego_envelope()
        self.assertEqual(n.ego_envelope.latest_sent.sequence,2)
        self.assertTrue(n.publish_goal(n.body+[.05,0,0],{'source':'fixture'}))
        self.assertEqual(len(n.goal_pub.messages),1)
        self.assertEqual(n.goals[-1]['reason']['ego_envelope_ack']['sequence'],first.sequence)

    def test_expired_or_wrong_payload_ack_never_authorizes(self):
        n=self.n;n.update_ego_envelope();bad=n.ego_envelope.latest_sent.wire().split();bad[-1]='.99';self.ack(' '.join(bad))
        with patch.object(n,'brake') as brake:self.assertFalse(n.publish_goal(n.body,'fixture'))
        brake.assert_called_once_with(replan=False);self.assertEqual(n.goal_pub.messages,[])
        n.test_time=2.1;self.ack()
        self.assertIsNone(n.ego_envelope.latest_ack)

    def test_goal_resend_requires_live_matching_ack(self):
        n=self.n;n.goals=[{'position':n.body.tolist()}];n.last_replan=.5;n.brake_hold=n.body.copy()
        with patch.object(n,'brake') as brake:
            self.assertFalse(n.replan_if_held());brake.assert_called_once_with(replan=False)
        self.assertEqual(n.goal_pub.messages,[])
        n.update_ego_envelope();self.ack();self.assertTrue(n.replan_if_held());self.assertEqual(len(n.goal_pub.messages),1)
        n.test_time=2.1
        n.last_replan=.5
        with patch.object(n,'brake') as brake:self.assertFalse(n.replan_if_held());brake.assert_called_once_with(replan=False)

    def test_follow_keeps_reference_without_ack_but_current_extra_still_vetoes(self):
        from quadrotor_msgs.msg import PositionCommand
        n=self.n;r=PositionCommand();r.header.stamp=NS(sec=2,nanosec=0);r.header.frame_id='world'
        target=n.body+np.array([.02,0.,0.]);r.position=NS(x=target[0],y=target[1],z=target[2])
        n.trajectory(r);self.assertIsNone(n.ego_envelope.latest_ack)
        n.follow();self.assertIs(n.reference,r);self.assertIsNone(n.brake_hold)
        self.assertEqual(len(n.traj.messages),1)
        values=np.zeros(n.map.shape,np.int8);values[tuple(n.map.idx([.45,0.,n.body[2]]))]=1
        n.map=OnlineMap([*n.map.origin,n.map.res,*n.map.shape,0,2.],values,.32,.16)
        self.assertTrue(n.map.segment(n.body,target))
        self.assertFalse(n.map.segment(n.body,target,extra_margin=self.extra))
        with self.assertRaisesRegex(RuntimeError,'stopping body volume is not observed free'):n.follow()
        self.assertEqual(len(n.traj.messages),1)

    def test_inherited_brake_replan_wait_sends_one_brake_command(self):
        n=self.n;n.goals=[{'position':n.body.tolist()}];n.last_replan=.5
        with patch.object(n,'send') as send:n.brake()
        self.assertEqual(send.call_count,1);self.assertEqual(n.goal_pub.messages,[])

    def test_startup_box_send_does_not_depend_on_ego_ack(self):
        n=self.n;n.bootstrap='ARMING';n.engine=NS()
        n.launch_certificate=lambda now:app.LaunchCertificate(now,n.session,n.home,.32,.16,.8,.9)
        n.send(n.body+[0,0,.02],0.)
        self.assertEqual(len(n.traj.messages),1);self.assertIsNone(n.ego_envelope.latest_ack)

    def test_control_tick_publishes_once_without_goal_side_effect(self):
        n=self.n;n.health['valid']=False
        n.control_once();self.assertEqual(len(n.ego_envelope_pub.messages),1)
        self.assertEqual(n.goal_pub.messages,[])

if __name__=='__main__':unittest.main()
