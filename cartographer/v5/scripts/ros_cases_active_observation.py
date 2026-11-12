"""Real dispatch/control callbacks with explicit NO-DDS plant fixtures."""
import unittest,math
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from ros_cases_native_mechanical import TypedNativeCases as Fixture,free_map

class ObservationCases(unittest.TestCase):
    setUp=Fixture.setUp
    tearDown=Fixture.tearDown
    sensors=Fixture.sensors
    install_launch=Fixture.install_launch
    def prepare(self):
        n=self.install_launch();n.bootstrap='READY';n.run_start=n.now();n.last_replan=n.now()
        self.submit=patch.object(n.frontier_worker,'submit');self.submit.start();self.addCleanup(self.submit.stop)
        self.tick(3.5)
        return n
    def tick(self,t,*,position=None,velocity=(0,0,0),visual_age=0.,frontier='pending'):
        n=self.n;p=n.home.copy() if position is None else np.asarray(position)
        self.sensors(t,p,True,False,velocity=velocity);n.health['visual_update_age']=visual_age;n.map=free_map(t)
        if frontier!='pending':
            candidate=None if frontier is None else (p+[float(frontier),0,0],{'fine_observed_step':float(frontier)<.15})
            n.frontier_result=dict(stamp=t,origin=p.copy(),candidate=candidate)
        n.control_once()
    def starts(self):return [e for e in self.n.adapter_events if e['event']=='frontier_observation_started']
    def test_certified_line_reference_precedes_frontier_without_map_fill(self):
        n=self.prepare();forward=n.q.apply(n.body_from_camera[:3,2]);forward[2]=0.;forward/=np.linalg.norm(forward)
        a=n.body+forward*.1;a[2]-=.55;b=a+forward*.5
        n.guideline=dict(stamp=3.55,segments=[dict(endpoints=[a.tolist(),b.tolist()],support_points=20,source='grayscale line with measured stereo floor support')])
        self.tick(3.55,frontier=.1)
        self.assertFalse(self.starts());self.assertTrue(n.goals)
        self.assertTrue(any(e['event']=='guideline_reference' for e in n.adapter_events))
        self.assertEqual(n.goals[-1]['reason']['source'],'measured guideline reference')
        np.testing.assert_array_equal(n.map.state,free_map(3.55).state)
    def test_line_outside_camera_half_fov_does_not_preempt_observation(self):
        # 60 deg off the optical axis: inside the former fixed 85 deg window, outside the
        # calibration's horizontal half-FOV (~40 deg), so it must not become the route.
        n=self.prepare();self.assertLess(n.line_axis_turn_bound,math.radians(60))
        forward=n.q.apply(n.body_from_camera[:3,2]);bearing=math.atan2(forward[1],forward[0])+math.radians(60)
        direction=np.array([math.cos(bearing),math.sin(bearing),0.])
        a=n.body.copy();a[2]-=.55;b=a+direction*.5
        n.guideline=dict(stamp=3.55,segments=[dict(endpoints=[a.tolist(),b.tolist()],support_points=20,source='grayscale line with measured stereo floor support')])
        self.tick(3.55,frontier=.1)
        self.assertFalse(any(e['event']=='guideline_reference' for e in n.adapter_events))
        self.assertEqual(len(self.starts()),1)
    def test_stale_line_does_not_preempt_observation(self):
        n=self.prepare();n.guideline=dict(stamp=2.,segments=[dict(endpoints=[[0,0,0],[1,0,0]],support_points=20,source='grayscale line with measured stereo floor support')])
        self.tick(3.55,frontier=.1)
        self.assertEqual(len(self.starts()),1)
        self.assertFalse(any(e['event']=='guideline_reference' for e in n.adapter_events))
    def test_fine_step_brakes_fixed_position_before_any_scan_yaw(self):
        n=self.prepare();self.tick(3.55,velocity=(.1,0,0),frontier=.1)
        hold=n.brake_hold.copy();first=n.traj.messages[-1].position;self.assertEqual(n._frontier_observation['phase'],'BRAKE');self.assertEqual(len(self.starts()),1)
        self.tick(3.6,position=hold+[.005,0,0],velocity=(.1,0,0));self.assertEqual(n._frontier_observation['phase'],'BRAKE');np.testing.assert_array_equal(n.brake_hold,hold)
        self.tick(3.65,position=hold+[.005,0,0]);self.assertEqual(n._frontier_observation['phase'],'SCAN')
        np.testing.assert_allclose(n.traj.messages[-1].position,first);self.assertLessEqual(abs(n.traj.messages[-1].yaw),n.yaw_limiter.lead);self.assertGreater(abs(n.traj.messages[-1].yaw),1e-6)
        self.assertEqual(len(n.goal_pub.messages),0);self.assertGreater(len(n.ledger.items),2)
    def test_missing_completed_frontier_triggers_but_pending_worker_does_not(self):
        n=self.prepare();self.assertFalse(self.starts());self.tick(3.55,frontier=None)
        self.assertEqual(len(self.starts()),1);self.assertEqual(self.starts()[0]['reason'],'no frontier')
    def test_fresh_measured_line_selects_side_without_a_field_direction(self):
        n=self.prepare();mount=n.body_from_camera[:3,2];bearing=math.atan2(mount[1],mount[0])
        for sign in (1,-1):
            direction=np.array([math.cos(bearing+sign*.4),math.sin(bearing+sign*.4),0.])
            n.guideline=dict(stamp=n.now(),segments=[dict(endpoints=[n.body.tolist(),(n.body+direction).tolist()],support_points=12,source='grayscale line with measured stereo floor support')])
            side,evidence=n.observation_initial_side(n.now());self.assertEqual(side,sign);self.assertIn('measured floor-line',evidence['source'])
        n.guideline['stamp']=n.now()-1.;_,evidence=n.observation_initial_side(n.now());self.assertNotIn('floor-line',evidence['source'])
    def test_normal_certified_frontier_exits_scan_and_existing_goal_path_continues(self):
        n=self.prepare();self.tick(3.55,frontier=.1);self.tick(3.6)
        self.tick(3.65,frontier=.2);self.assertIsNone(n._frontier_observation);self.assertTrue(n.goals)
        reasons=[e.get('reason') for e in n.adapter_events if e['event']=='frontier_observation_finished'];self.assertIn('ordinary certified frontier available',reasons)
    def test_visual_gate_stops_scan_without_forging_measurements(self):
        n=self.prepare();self.tick(3.55,frontier=.1);self.tick(3.6);last=n.last_visual_yaw;forward=n.forward
        self.tick(3.65,visual_age=.36);self.assertIsNone(n._frontier_observation);self.assertEqual(n.last_visual_yaw,last);self.assertIs(n.forward,forward)
        self.tick(3.7,frontier=.1);self.assertEqual(len(self.starts()),1)
    def test_active_observation_missing_feedback_never_follows_old_translation(self):
        n=self.prepare();self.tick(3.55,frontier=.1);n.goals=[{'position':(n.body+[.1,0,0]).tolist()}]
        n.reference=NS(trajectory_id=9) # A newer native auto-replan of the old goal.
        self.sensors(3.6,n.home,True,False);n.map=free_map(3.6)
        with patch.object(n,'feedback_once',return_value=None),patch.object(n,'follow') as follow:
            n.control_once();follow.assert_not_called()
        np.testing.assert_allclose(n.traj.messages[-1].position,n.control_alignment.position(n.brake_hold))
        n.test_time=3.8;n.health['stamp']=3.8;n.owns_vehicle=True;n.armed=True
        with patch.object(n,'healthy',return_value=True),patch.object(n,'feedback_once',return_value=None),patch.object(n,'native_hold') as native_hold,patch.object(n,'follow') as follow:
            n.home_relative_adapter_health['stamp']=3.8;n.control_once();follow.assert_not_called();native_hold.assert_called_once()
        self.assertIsNone(n._frontier_observation);self.assertIsNone(n.reference)
    def test_unknown_body_volume_still_stops_without_a_scan_publication(self):
        from online_map import OnlineMap
        n=self.prepare();self.tick(3.55,frontier=.1);before=len(n.traj.messages)
        self.sensors(3.6,n.home,True,False);m=free_map(3.6);state=m.state.copy();state[tuple(m.idx(n.body))]=-1
        n.map=OnlineMap([*m.origin,m.res,*m.shape,0,3.6],state,.32,.16)
        with self.assertRaises(RuntimeError):n.control_once()
        self.assertEqual(len(n.traj.messages),before)
    def test_scan_is_bounded_and_same_fine_episode_cannot_retrigger(self):
        n=self.prepare();self.tick(3.55,frontier=.1)
        for i in range(1,164):self.tick(round(3.55+i*.05,6))
        self.assertIsNone(n._frontier_observation);self.assertEqual(len(self.starts()),1)
        self.tick(11.8,frontier=.1);self.assertEqual(len(self.starts()),1)
        finishes=[e for e in n.adapter_events if e['event']=='frontier_observation_finished'];self.assertLessEqual(finishes[-1]['elapsed_s'],8.05)

del Fixture
if __name__=='__main__':unittest.main(verbosity=2)
