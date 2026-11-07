"""Synthetic measured-state contract tests; these are NOT simulator flights."""
import copy
import unittest
import numpy as np
from online_map import OnlineMap
from task_observations import TargetTrack
from task_safety import MeasuredMapGuard
from task_executive import TaskExecutive,MissionLimits,Feedback

LIMITS=MissionLimits(.25,.12,.18,max_step_m=.12)


def measured_map(now=1.,unknown=False):
    shape=(90,50,40); origin=[-1.,-2.5,-1.]
    state=np.full(shape,-1 if unknown else 0,np.int8)
    return OnlineMap([*origin,.1,*shape,0,now],state,.25,.12)


def guard_at(base,now,epoch='own-local'):
    snap=copy.copy(base); snap.stamp=now
    return MeasuredMapGuard(snap,epoch,.25,.12)


def target(role,center=(1,0,.7),normal=(-1,0,0),now=1.,origin=(0,0,.7),id=None):
    t=TargetTrack(id or role,role,'ellipse' if role.startswith('tunnel') else 'rectangle','own-local',
                  np.asarray(center,float),np.asarray(normal,float),np.array([1.2,1.3]),.025,
                  now,now-.2,[np.array(origin)+[0,-.05,0],np.array(origin)+[0,.05,0],np.array(origin)],
                  observation_count=3,consistent=True,interior='free-background',geometry_verified=True)
    t.history=[{'stamp':now-.2+i*.1,'calibration_id':'measured-rig'} for i in range(3)]
    if role=='dock':
        t.kind='landing_plane'; t.normal=np.array([0.,0.,1.]); t.interior='unknown'
        grid=np.linspace(-.4,.4,17)
        t.details={'marker_kind':'h','support_points_local_m':[[center[0]+x,center[1]+y,center[2]] for x in grid for y in grid]}
    if role=='wall':
        t.kind='wall_top';t.interior='unknown';t.details={'edge_height_upper_local_m':1.0}
    return t


class GuardTests(unittest.TestCase):
    def setUp(self): self.map=measured_map()

    def test_unknown_and_occupied_body_periphery_veto_clear_center_ray(self):
        g=guard_at(self.map,1.)
        a=[0,0,.7]; b=[.5,0,.7]
        self.assertIsNotNone(g.certify([a,b],1.,'own-local','test'))
        # Center ray remains free, but rotor/body envelope meets this cell.
        for value in (-1,1):
            m=measured_map(); m.state[tuple(m.idx([.3,.2,.7]))]=value
            m=OnlineMap([*m.origin,m.res,*m.shape,0,1.],m.state,.25,.12)
            self.assertIsNone(guard_at(m,1.).certify([a,b],1.,'own-local','test'))

    def test_expiry_epoch_outside_and_body_calibration_are_strict(self):
        g=guard_at(self.map,1.)
        self.assertIsNone(g.certify([[0,0,.7],[.2,0,.7]],1.6,'own-local','test'))
        self.assertIsNone(g.certify([[0,0,.7],[.2,0,.7]],1.,'reset','test'))
        self.assertIsNone(g.certify([[0,0,.7],[20,0,.7]],1.,'own-local','test'))
        with self.assertRaises(ValueError): MeasuredMapGuard(self.map,'own-local',.1,.1)

    def test_certificate_checks_prior_snapshot_tube_and_expiry(self):
        g=guard_at(self.map,1.); c=g.certify([[0,0,.7],[.2,0,.7]],1.,'own-local','test')
        self.assertTrue(c.covers_motion([0,0,.7],[.1,0,.7],1.1,'own-local'))
        self.assertFalse(c.covers_motion([0,0,.7],[.1,.1,.7],1.1,'own-local'))
        self.assertFalse(c.covers_motion([0,0,.7],[.1,0,.7],1.5,'own-local'))
        self.assertFalse(c.covers_motion([0,0,.7],[.1,0,.7],1.1,'reset'))

    def test_occupied_floor_is_not_silently_cleared_for_touchdown(self):
        m=measured_map(); state=m.state.copy()
        state[:,:,m.idx([0,0,0])[2]]=1
        floor=OnlineMap([*m.origin,m.res,*m.shape,0,1.],state,.25,.12)
        # The existing voxel grid cannot represent this exact final contact
        # pose with padding. Refuse instead of inventing a contact exception.
        self.assertIsNone(guard_at(floor,1.).certify([[0,0,.5],[0,0,.18]],1.,'own-local','landing'))

    def test_extreme_outside_goal_is_rejected_before_unbounded_sampling(self):
        self.assertIsNone(guard_at(self.map,1.).certify([[0,0,.7],[1e100,0,.7]],1.,'own-local','test'))


class ExecutiveTests(unittest.TestCase):
    def setUp(self):
        self.map=measured_map(); self.now=1.;self.p=np.array([0.,0.,.7]); self.armed=True;self.landed=False;self.tof=.52
        self.exec=TaskExecutive(LIMITS,[0,0,.18],'own-local')

    def feedback(self):
        return Feedback(self.now,self.p.copy(),'own-local',self.armed,self.landed,0. if self.landed else .3,self.tof,body_up_local=[0,0,1])

    def tick(self,tracks=(),advance=.15):
        self.now+=advance
        for t in tracks:
            t.stamp=self.now; t.camera_origins[-1]=self.p.copy()
            t.history.append({'stamp':self.now,'calibration_id':'measured-rig'});t.history=t.history[-16:]
        return self.exec.tick(self.feedback(),tracks,guard_at(self.map,self.now),self.now)

    def follow(self,intent):
        if intent.action=='move':
            self.p=np.array(intent.goal); self.landed=False;self.tof=None
        elif intent.action=='land':
            delta=np.array(intent.goal)-self.p
            self.p+=delta*min(1.,.10/max(1e-9,np.linalg.norm(delta)))
            if np.linalg.norm(self.p-np.array(intent.goal))<1e-6:
                self.landed=True;self.tof=0.
        elif intent.action=='disarm': self.armed=False
        elif intent.action=='arm': self.armed=True


    def test_measured_rim_axes_do_not_require_vertical_clearance_equal_to_rotor_diameter(self):
        t=target('window')
        t.center=np.array([1.,0.,.7]);t.dimensions=np.array([1.,.5])
        t.boundary=t.center+np.array([[0,-.5,-.25],[0,.5,-.25],[0,.5,.25],[0,-.5,.25]])
        self.assertTrue(self.exec._usable(t,1.))
        t.boundary=t.center+np.array([[0,-.2,-.5],[0,.2,-.5],[0,.2,.5],[0,-.2,.5]])
        self.assertFalse(self.exec._usable(t,1.))

    def test_no_target_or_timeout_can_complete_any_task(self):
        for _ in range(80):
            intent=self.tick()
            self.assertEqual(intent.action,'explore')
        self.assertEqual(self.exec.completed,{})
        self.assertFalse(self.exec.report()['competition_tasks_complete'])

    def test_unknown_interior_painted_and_blocked_targets_are_rejected(self):
        for interior in ('unknown','painted','blocked'):
            t=target('tunnel_entry');t.interior=interior
            self.assertEqual(self.tick([t]).action,'explore')
        t.interior='free-background';t.consistent=False
        self.assertEqual(self.tick([t]).action,'explore')

    def test_geometry_cannot_bypass_unknown_full_body_map(self):
        t=target('tunnel_entry')
        m=measured_map(unknown=True)
        result=self.exec.tick(self.feedback(),[t],guard_at(m,self.now),self.now)
        self.assertEqual(result.action,'hold')
        self.assertEqual(self.exec.completed,{})
        self.assertFalse(t.passage_clearance_verified)

    def test_duplicate_and_stale_feedback_does_not_advance(self):
        t=target('tunnel_entry'); first=self.tick([t])
        phase=self.exec.phase
        result=self.exec.tick(self.feedback(),[t],guard_at(self.map,self.now),self.now)
        self.assertEqual(result.action,'hold');self.assertEqual(self.exec.phase,phase)
        result=self.exec.tick(self.feedback(),[t],guard_at(self.map,self.now+1),self.now+1)
        self.assertEqual(result.action,'hold')

    def test_uncommanded_motion_or_tracking_reset_never_earns_crossing(self):
        self.tick();self.p[0]+=.10
        result=self.tick()
        self.assertEqual(result.action,'hold')
        self.assertIsNotNone(self.exec.fault)
        self.assertFalse(self.exec.report()['competition_tasks_complete'])
        executive=TaskExecutive(LIMITS,[0,0,.18],'own-local')
        f=Feedback(1.,[0,0,.7],'reset',True,False,0.)
        self.assertEqual(executive.tick(f,[],guard_at(self.map,1.),1.).action,'hold')
        self.assertEqual(executive.completed,{})

    def test_frontier_motion_requires_prior_map_authorization(self):
        self.tick()
        move=self.exec.authorize_external_step([.1,0,.7],self.feedback(),guard_at(self.map,self.now),self.now)
        self.assertEqual(move.action,'move');self.follow(move)
        self.assertEqual(self.tick().action,'explore')
        self.assertIsNone(self.exec.fault)

    def test_single_tunnel_plane_cannot_complete_tunnel(self):
        t=target('tunnel_entry')
        for _ in range(80):
            intent=self.tick([t]);self.follow(intent)
            if self.exec.stage=='TUNNEL_EXIT': break
        self.assertEqual(self.exec.stage,'TUNNEL_EXIT')
        same=target('tunnel_exit',center=t.center,origin=self.p)
        for _ in range(8): self.assertEqual(self.tick([same]).action,'explore')
        self.assertNotIn('TUNNEL_EXIT',self.exec.completed)

    def test_far_side_pose_without_new_far_side_image_is_not_completion(self):
        t=target('tunnel_entry')
        for _ in range(80):
            intent=self.tick([t] if self.exec.phase!='VERIFY' else [])
            self.follow(intent)
            if self.exec.phase=='VERIFY': break
        self.assertEqual(self.exec.phase,'VERIFY')
        for _ in range(5): self.assertEqual(self.tick().action,'observe')
        self.assertEqual(self.exec.completed,{})

    def test_fresh_blocker_vetoes_acquired_target(self):
        t=target('tunnel_entry');self.tick([t]);t.interior='blocked'
        result=self.tick([t]);self.assertEqual(result.action,'hold')
        self.assertEqual(self.exec.phase,'SEARCH');self.assertIsNone(self.exec.target)

    def test_landing_support_holes_and_unrecognized_mark_rejected(self):
        self.exec.stage='DOCK'; self.exec.phase='SEARCH'
        t=target('dock',center=(1,0,0),normal=(0,0,1))
        t.details['marker_kind']='unrecognized_shape'
        self.assertEqual(self.tick([t]).action,'explore')
        t.details['marker_kind']='h'
        t.details['support_points_local_m']=[p for p in t.details['support_points_local_m'] if p[1]>.1]
        self.assertEqual(self.tick([t]).action,'explore')

    def test_rearm_requires_landed_tof_and_confirmed_disarm(self):
        self.exec.stage='DOCK';self.exec.phase='DISARM'
        t=target('dock',center=(0,0,0));self.exec.target=t
        self.p=np.array([0.,0.,.18]);self.landed=True;self.tof=None
        self.assertEqual(self.tick().action,'hold')
        self.tof=0.
        self.assertEqual(self.tick().action,'disarm')
        self.assertEqual(self.exec.stage,'DOCK')
        self.armed=False
        self.tick();self.assertEqual(self.exec.stage,'REARM')
        self.assertEqual(self.tick().action,'arm')
        self.assertNotIn('REARM',self.exec.completed)
        self.armed=True
        self.assertEqual(self.tick().action,'move')
        self.assertIn('REARM',self.exec.completed)

    def test_wall_crossing_must_put_recorded_home_on_far_side(self):
        self.exec.stage='WALL'; t=target('wall',center=(-.5,0,.8),normal=(1,0,0))
        self.assertEqual(self.tick([t]).action,'explore')
        self.assertIsNone(self.exec.target)


    def test_consistent_flag_alone_is_not_observation_provenance(self):
        t=target('tunnel_entry');t.history=[]
        self.assertEqual(self.tick([t]).action,'explore')
        t.history=[{'stamp':self.now-i*.1,'calibration_id':'rig'} for i in range(3)]
        t.camera_origins=[self.p.copy() for _ in range(3)]
        self.assertEqual(self.tick([t]).action,'explore')


    def test_missing_contact_calibration_holds_at_dock_without_a_fake_default(self):
        limits=MissionLimits(.25,.12)
        executive=TaskExecutive(limits,[0,0,.1],'own-local');executive.stage='DOCK'
        f=self.feedback()
        result=executive.tick(f,[target('dock')],guard_at(self.map,self.now),self.now)
        self.assertEqual(result.action,'hold')
        self.assertIn('calibration',result.reason)
        # The padded map envelope can exceed the measured physical foot offset.
        # The map guard, not a false physical-calibration check, rejects contact.
        self.assertEqual(MissionLimits(.32,.16,.075).foot_to_body_center_m,.075)

    def test_landing_requires_measured_upright_body_attitude(self):
        f=Feedback(1.,[0,0,.18],'own-local',True,True,0.,0.)
        self.assertFalse(self.exec._feedback_on_surface(f,np.array(f.position),np.zeros(3)))
        f=Feedback(1.,[0,0,.18],'own-local',True,True,0.,0.,body_up_local=[.5,0,np.sqrt(.75)])
        self.assertFalse(self.exec._feedback_on_surface(f,np.array(f.position),np.zeros(3)))

    def test_complete_ordered_synthetic_trace_requires_all_gates(self):
        centers={'tunnel_entry':(1,0,.7),'tunnel_exit':(2.2,0,.7),
                 'window':(3.6,0,.7),'dock':(5.,0,0),'wall':(2.5,0,.8)}
        actions=set();stages=[]
        for _ in range(500):
            role=self.exec.desired_role
            tracks=[] if role is None or role=='home' else [target(role,center=centers[role],normal=(1,0,0) if role=='wall' else (-1,0,0),now=self.now+.15,origin=self.p)]
            intent=self.tick(tracks)
            actions.add(intent.action); stages.append(self.exec.stage)
            self.follow(intent)
            if intent.action=='complete': break
        report=self.exec.report()
        self.assertIsNone(self.exec.fault,report)
        self.assertTrue(report['competition_tasks_complete'],(report,self.p,intent.as_dict()))
        self.assertEqual(set(report['completed']),{'TUNNEL_ENTRY','TUNNEL_EXIT','WINDOW','DOCK','REARM','RETAKEOFF','WALL','HOME'})
        self.assertTrue({'arm','disarm','land','move','complete'}.issubset(actions))
        self.assertFalse(self.armed);self.assertTrue(self.landed)
        self.assertLess(np.linalg.norm(self.p-self.exec.home),.06)
        event_names=[e['event'] for e in report['events']]
        self.assertLess(event_names.index('dock_landed_and_disarmed'),event_names.index('rearmed_after_dock_disarm'))
        self.assertEqual(event_names.count('body_before_measured_plane'),4)
        self.assertEqual(event_names.count('body_crossed_measured_plane'),4)
        self.assertEqual(event_names.count('passage_complete'),4)

if __name__=='__main__': unittest.main()
