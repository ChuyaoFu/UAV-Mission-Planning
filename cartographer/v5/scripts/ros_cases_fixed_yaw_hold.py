"""Real adapter callbacks with explicit no-DDS message fixtures; no flight qualification."""
import os,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
MAIN=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(MAIN/'cartographer/v5/scripts'))
sys.path.insert(0,os.environ['AUDIT_SOURCE'])
from competition_ros_stubs import install
M=install();sys.modules['rclpy.qos'].DurabilityPolicy.VOLATILE=0
import competition_navigate as app
from online_map import OnlineMap
from task_safety import MeasuredMapGuard

class Callbacks(unittest.TestCase):
    def setUp(self):
        self.auth=patch.object(app,'assert_sitl_authorized');self.auth.start()
        self.n=n=app.CompetitionNavigator(os.environ['AUDIT_RESULT'],app.CompetitionOptions(.003,.001,.01,map_pose_error_m=0.))
        n.pump=lambda dt=.01:None
        self.sample(2.,[0,0,.124]);self.sample(2.05,[0,0,.124])
        n.home=n.body.copy();n.home_pose_error_m=.003;n.yaw=0.;n.run_start=2.
        n.establish_control_alignment('synthetic_stationary_ground_fixture',stationary_confirmed=True)
        shape=(50,50,35);n.map=OnlineMap([-2.5,-2.5,-1.,.1,*shape,0,2.05],np.zeros(shape,np.int8),.32,.16)
        n.engine=NS(stage='EXPLORE',phase='SEARCH');n.bootstrap='READY'
        n.guard=lambda:MeasuredMapGuard(n.map,n.session,.32,.16,max_age_s=.5,extra_collision_margin_m=[.01,.01,.01])
        n.feedback_once=lambda:(NS(armed=False,landed=False),NS(action='explore'),None)
        n.explore=lambda feedback:n.execute_move(n.home+np.array([.3,0,0]),'fixture',turn=True)
        n.launch_certificate=lambda now:app.LaunchCertificate(now,n.session,n.home,.32,.16,.8,1.3)
    def sample(self,t,p):
        n=self.n;n.test_time=t
        n.health={'stamp':t,'valid':True,'reason':'ok_visual','accepted_visual_update_features':20,'visual_update_age':0.}
        n.local_cb(M.VehicleLocalPosition(timestamp=round(t*1e6)))
        n.status_cb(M.VehicleStatus(timestamp=round(t*1e6)))
        n.land_cb(M.VehicleLandDetected(timestamp=round(t*1e6)))
        m=M.Odometry();ns=round(t*1e9);m.header.stamp=NS(sec=ns//10**9,nanosec=ns%10**9)
        m.pose.pose.position=NS(x=p[0],y=p[1],z=p[2]);m.pose.covariance=(np.eye(6)*1e-6).ravel().tolist();n.odom(m)
        if n.map is not None:n.map.stamp=t
    def tearDown(self):
        for w in (self.n.task_worker,self.n.frontier_worker,self.n.worker):w.close()
        self.auth.stop()
    def test_bootstrap_arc_target_fixed_in_both_phases(self):
        n=self.n
        for phase,key in [('ASCEND','takeoff_survey_height_m'),('DESCEND','takeoff_height_m')]:
            with self.subTest(phase=phase):
                n.bootstrap=phase;n.bootstrap_phase_started=n.now()
                p=n.home+np.array([0,0,n.c[key]])
                start=n.body.copy();steps=max(1,int(np.ceil(np.linalg.norm(p-start)/.01)))
                for i in range(1,steps+1):self.sample(n.now()+.05,start+(p-start)*i/steps)
                n.bootstrap_phase_started=n.now();n.control_once()
                self.assertIn(n.bootstrap,('HIGH_ARC','LOW_ARC'))
                self.sample(n.now()+.05,p);n.control_once();first=np.array(n.traj.messages[-1].position)
                self.sample(n.now()+.05,p+[.012,-.008,.006]);n.control_once()
                np.testing.assert_allclose(n.traj.messages[-1].position,first,atol=1e-14)
                self.assertEqual(len(n.goal_pub.messages),0)
    def test_turn_target_fixed_across_actual_control_callbacks(self):
        n=self.n;n.yaw=.8;n.control_once();first=np.array(n.traj.messages[-1].position)
        self.sample(2.1,n.home+[.012,-.008,.006]);n.control_once()
        np.testing.assert_allclose(n.traj.messages[-1].position,first,atol=1e-14)
        self.assertEqual(len(n.goal_pub.messages),0)
        self.assertEqual(len(n.ledger.items),2)
        self.assertLessEqual(abs(n.traj.messages[-1].yaw-n.traj.messages[0].yaw),n.yaw_limiter.rate*.05+1e-12)
    def test_intervening_hold_callback_starts_new_turn_episode(self):
        n=self.n;n.yaw=.8;n.control_once()
        n.feedback_once=lambda:(NS(armed=False,landed=False),NS(action='hold',look_at=None),None)
        self.sample(2.1,n.home+[.006,0,0]);n.control_once()
        n.feedback_once=lambda:(NS(armed=False,landed=False),NS(action='explore'),None)
        self.sample(2.15,n.home+[.018,0,0]);n.control_once()
        np.testing.assert_allclose(n.traj.messages[-1].position,n.control_alignment.position(n.body),atol=1e-14)
    def test_feedback_gap_preserves_same_turn_episode(self):
        n=self.n;n.yaw=.8;n.control_once();first=np.array(n.traj.messages[-1].position)
        # The real control callback receives no new task feedback this tick.
        n.feedback_once=lambda:None;n.last_intent=NS(action='explore',certificate=None);n.last_feedback=n.now()
        n.goals=[{'position':(n.home+np.array([.3,0,0])).tolist()}]
        self.sample(2.1,n.home+[.006,0,0])
        with patch.object(n,'follow') as follow:
            n.control_once();follow.assert_not_called()
        np.testing.assert_allclose(n.traj.messages[-1].position,first,atol=1e-14)
        n.feedback_once=lambda:(NS(armed=False,landed=False),NS(action='explore'),None)
        self.sample(2.15,n.home+[.018,0,0]);n.control_once()
        np.testing.assert_allclose(n.traj.messages[-1].position,first,atol=1e-14)
    def test_actual_stopping_tail_vetoes_even_when_hold_point_is_free(self):
        n=self.n;n.yaw=.8;n.control_once();before=len(n.traj.messages)
        self.sample(2.1,n.home+[.012,0,0]);n.body_velocity=np.array([1.,0,0])
        values=np.zeros(n.map.shape,np.int8);values[n.map.idx([.8,0,n.body[2]])[0],:,:]=1
        n.map=OnlineMap([*n.map.origin,n.map.res,*n.map.shape,0,n.now()],values,.32,.16)
        with self.assertRaisesRegex(RuntimeError,'Actual offboard reference/stopping'):
            n.control_once()
        self.assertEqual(len(n.traj.messages),before)
    def test_turn_exit_then_reentry_captures_new_position(self):
        n=self.n;n.yaw=.8;n.control_once()
        with patch.object(n,'publish_goal',return_value=False):
            n.yaw=0.;self.sample(2.1,n.home+[.006,0,0]);n.control_once()
        n.yaw=.8;self.sample(2.15,n.home+[.018,0,0]);n.control_once()
        np.testing.assert_allclose(n.traj.messages[-1].position,n.control_alignment.position(n.body),atol=1e-14)
    def test_fixed_point_still_rechecks_current_guard_and_stopping(self):
        n=self.n;n.yaw=.8;n.control_once();before=len(n.traj.messages)
        self.sample(2.1,n.home+[.012,0,0])
        with patch.object(n,'guard',return_value=None),self.assertRaisesRegex(RuntimeError,'No measured map'):
            n.control_once()
        self.assertEqual(len(n.traj.messages),before)
        self.assertEqual(len(n.ledger.items),1)

if __name__=='__main__':unittest.main()
