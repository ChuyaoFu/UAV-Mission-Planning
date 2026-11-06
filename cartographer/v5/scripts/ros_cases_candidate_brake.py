"""Actual adapter callbacks, known/unknown voxel queries; NO DDS or flight."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
from ros_cases_competition import Cases as ExistingCases,M,app,odom,header
from task_uncertainty import UncertaintyPolicy
from online_map import OnlineMap

class CandidateBrakeCases(unittest.TestCase):
    setUp=ExistingCases.setUp
    tearDown=ExistingCases.tearDown
    def world(self,*,body_bad=False,stop_bad=False,moving=False):
        n=self.node;n.bootstrap='READY';n.test_time=2.02;n.health['stamp']=2.02
        n.local_cb(M.VehicleLocalPosition(timestamp=2020000));msg=odom(2.02)
        if moving:msg.twist.twist.linear=NS(x=.1,y=0.,z=0.)
        n.odom(msg)
        # Synthetic causal-reserve fixture matches trial43's recorded proxy;
        # the actual accepted BODY covariance remains unchanged.
        n.forward_position_uncertainty=lambda now:(.06711407941539257,dict(valid_until=now+.1,synthetic_fixture=True))
        n.uncertainty_policy=UncertaintyPolicy(included_xy_margin_m=.06,included_z_margin_m=.04,map_pose_error_m=.055,map_geometry_error_m=.01)
        shape=(160,160,160);state=np.zeros(shape,np.int8);origin=np.array([-.8,-.8,-.6])
        idx=lambda p:tuple(np.floor((p-origin)/.01).astype(int))
        state[idx(n.body+[0,-.46,0])]=-1
        if body_bad:state[idx(n.body)]=-1
        if stop_bad:
            n.body_velocity=np.array([.2,0,0]);state[idx(n.body+[.49,0,0])]=-1
        n.map=OnlineMap([*origin,.01,*shape,0,n.now()],state,.32,.16)
        n.last_received_trajectory_id=7;n.reference=NS(trajectory_id=7);n.ref_t=n.now()
        return n,n.body+[0,-.045,0],np.array([0,.2,0])
    def test_bad_reference_only_brakes_with_actual_certificate_and_no_bad_setpoint(self):
        n,target,v=self.world(moving=True);guard=n.guard();extra=guard.extra
        speed=np.linalg.norm(n.body_velocity);stop=n.body+n.body_velocity*(n.c['max_observation_age_s']+speed/(2*n.c['max_acceleration_mps2']));paths=[(n.body,target),(n.body,target+v*.3),(n.body,stop)]
        self.assertEqual([guard.certify([a,b],n.now(),n.session,'fixture') is not None for a,b in paths],[False,True,True])
        n.send(target,n.yaw,v,np.array([.1,0,0]))
        self.assertEqual(len(n.traj.messages),1);self.assertEqual(len(n.ledger.items),1)
        msg=n.traj.messages[-1];np.testing.assert_allclose(msg.position,n.control_alignment.position(n.body))
        self.assertFalse(np.allclose(msg.position,n.control_alignment.position(target)));self.assertTrue(np.isnan(msg.velocity).all());self.assertTrue(np.isnan(msg.acceleration).all())
        self.assertIsNone(n.reference);self.assertEqual(n.waiting_trajectory_id,7);np.testing.assert_array_equal(n.brake_hold,n.body)
        cert=n.ledger.items[-1];self.assertEqual(cert.guard.extra,extra);self.assertLessEqual(cert.valid_until,guard.map.stamp+guard.max_age_s)
        self.assertTrue(cert.covers_motion(n.body,stop,n.now(),n.session));np.testing.assert_allclose(cert.paths[-1][1],stop)
        self.assertAlmostEqual(cert.valid_until,n.now()+.1)
        self.assertFalse(cert.covers_motion(n.body,target,n.now(),n.session))
        before=n.body.copy();sample=odom(2.07,p=before+[.02,0,0]);sample.twist.twist.linear=NS(x=.1,y=0.,z=0.);n.test_time=2.07;n.health['stamp']=2.07;n.odom(sample)
        audit=n.ledger.interval(2.02,2.07,n.body_history,n.session)
        self.assertTrue(audit.covers_motion(before,n.body,2.07,n.session));self.assertGreater(n.ledger.audited_segments,0)
        self.assertEqual(n.veto_reasons['candidate_reference_volume'],1);self.assertFalse(getattr(n,'_candidate_brake_active',False));self.assertFalse(n.goal_pub.messages)
    def test_bad_velocity_lookahead_only_also_brakes(self):
        n,_,_=self.world();target=n.body+[0,.01,0];v=np.array([0,-.25,0])
        g=n.guard();self.assertIsNotNone(g.certify([n.body,target],n.now(),n.session,'reference'))
        self.assertIsNone(g.certify([n.body,target+v*.3],n.now(),n.session,'lead'))
        n.send(target,n.yaw,v);self.assertEqual(len(n.traj.messages),1);self.assertEqual(len(n.ledger.items),1)
    def test_current_body_unknown_remains_fatal_and_publishes_nothing(self):
        n,target,v=self.world(body_bad=True)
        with self.assertRaisesRegex(RuntimeError,'not observed free'):n.send(target,n.yaw,v)
        self.assertFalse(n.traj.messages);self.assertFalse(n.ledger.items)
    def test_actual_stop_unknown_remains_fatal_with_body_still_safe(self):
        n,target,v=self.world(stop_bad=True);g=n.guard()
        self.assertIsNotNone(g.certify([n.body,n.body],n.now(),n.session,'body'))
        with self.assertRaisesRegex(RuntimeError,'not observed free'):n.send(target,n.yaw,v)
        self.assertFalse(n.traj.messages);self.assertFalse(n.ledger.items)
    def test_brake_recheck_failure_does_not_recurse(self):
        n,target,v=self.world();first=n.guard();second=n.guard()
        class Veto:
            map=second.map;extra=second.extra;uncertainty=second.uncertainty;max_age_s=second.max_age_s
            def certify(self,*args):return None
        with patch.object(n,'guard',side_effect=[first,Veto()]) as calls:
            with self.assertRaisesRegex(RuntimeError,'not observed free'):n.send(target,n.yaw,v)
        self.assertEqual(calls.call_count,2);self.assertFalse(n.traj.messages);self.assertFalse(n.ledger.items);self.assertFalse(n._candidate_brake_active)
    def test_old_hold_is_replaced_and_new_trajectory_can_continue_normal_follow(self):
        n,target,v=self.world();n.brake_hold=target.copy();n.send(target,n.yaw,v)
        def cmd(id,goal):return NS(header=header(n.now()),trajectory_id=id,trajectory_flag=1,position=NS(x=goal[0],y=goal[1],z=goal[2]),velocity=NS(x=0.,y=0.,z=0.),acceleration=NS(x=0.,y=0.,z=0.))
        n.trajectory(cmd(7,n.body+[.02,0,0]));self.assertIsNone(n.reference)
        good=n.body+[.02,0,0];n.goals=[{'position':good.tolist()}];n.last_replan=n.now()-2.
        n.brake();self.assertEqual(len(n.goal_pub.messages),1)
        n.trajectory(cmd(8,good));self.assertIsNotNone(n.reference)
        n.follow();self.assertEqual(len(n.traj.messages),3);self.assertEqual(len(n.ledger.items),3);self.assertIsNone(n.brake_hold)
        np.testing.assert_allclose(n.traj.messages[-1].position,n.control_alignment.position(good))
del ExistingCases # Imported fixture is not another test suite.
if __name__=='__main__':unittest.main(verbosity=2)
