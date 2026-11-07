"""Dense measured-plane and real OnlineMap terminal contact negative tests."""
import dataclasses
import unittest
import numpy as np
from online_map import OnlineMap
from task_safety import MeasuredMapGuard
from task_contact import build_support_patch,ContactLandingAdapter
from test_task_executive import target
from test_task_surfaces import down_exposure
from test_task_landing import sample,calibrated_foot,MOUNT,AXIS


def floor_map(stamp=2.):
    shape=(60,60,35);origin=[-3.,-3.,-1.];res=.1
    state=np.zeros(shape,np.int8);state[:,:,:10]=-1;state[:,:,10]=1
    return OnlineMap([*origin,res,*shape,0,stamp],state,.32,.16)


def support(depth=None,mask=None):
    image=np.full((300,400),2.,np.float32) if depth is None else depth
    t=target('dock',center=(0,0,0),now=2.)
    exp=dataclasses.replace(down_exposure(2.),epoch='own-local',pose_error_m=.005)
    return build_support_patch(image,[300,300,200,150],exp,t,2.,.48,.01,mask)


class ContactTests(unittest.TestCase):
    def setUp(self):
        self.map=floor_map();self.guard=MeasuredMapGuard(self.map,'own-local',.32,.16)
        self.patch=support();self.cal=calibrated_foot()
        self.adapter=ContactLandingAdapter(MOUNT,AXIS,'actual-kit')
        z=self.cal.offset_m+.30
        self.s=sample(2.,z+MOUNT[2],position=(0,0,z),landed=False,armed=True,velocity_local_mps=[0,0,-.1])
        self.assertIsNotNone(self.patch)


    def test_upsampled_depth_does_not_fake_independent_native_ray_density(self):
        t=target('dock',center=(0,0,0),now=2.)
        exp=dataclasses.replace(down_exposure(2.),epoch='own-local',pose_error_m=.005)
        depth=np.full((300,400),2.,np.float32)
        # Each native pixel would pass alone. Actual matching at half-grid
        # spacing is coarser and must be checked without filling any holes.
        self.assertIsNone(build_support_patch(depth,[300,300,200,150],exp,t,2.,.48,.01,measurement_stride_px=2.))

    def test_legal_floor_contact_works_without_changing_a_single_global_voxel(self):
        before=self.map.state.copy();blocked=self.map.blocked.copy()
        goal=[0,0,self.cal.offset_m]
        self.assertIsNone(self.guard.certify([self.s.position_local_m,goal],2.,'own-local','ordinary free-space descent'))
        cert=self.adapter.certify(self.guard,self.patch,self.cal,self.s,2.)
        self.assertIsNotNone(cert,self.adapter.last_rejection)
        self.assertGreater(cert.exempted_support_voxels,0)
        self.assertFalse(cert.as_dict()['full_body_free'])
        self.assertFalse(cert.as_dict()['global_map_changed'])
        np.testing.assert_array_equal(self.map.state,before)
        np.testing.assert_array_equal(self.map.blocked,blocked)

    def test_unknown_rotor_periphery_and_unrelated_side_obstacle_still_veto(self):
        for state,point in ((-1,[.2,0,.3]),(1,[.2,0,.3]),(-1,[.2,0,.02])):
            values=self.map.state.copy();values[tuple(self.map.idx(point))]=state
            m=OnlineMap([*self.map.origin,self.map.res,*self.map.shape,0,2.],values,.32,.16)
            result=self.adapter.certify(MeasuredMapGuard(m,'own-local',.32,.16),self.patch,self.cal,self.s,2.)
            self.assertIsNone(result,(state,point))

    def test_dense_patch_rejects_hole_edge_missing_or_inferred_single_pixel(self):
        for value in (np.nan,4.,1.94):
            depth=np.full((300,400),2.,np.float32);depth[150,220]=value
            self.assertIsNone(support(depth))
        mask=np.ones((300,400),bool);mask[150,220]=False
        self.assertIsNone(support(mask=mask))

    def test_small_obstacle_in_same_coarse_floor_voxel_invalidates_dense_contact_proof(self):
        # Ordinary occupancy cannot separate this obstacle from the floor voxel.
        # Dense direct-plane proof must refuse it instead of exempting the cell.
        depth=np.full((300,400),2.,np.float32);depth[140:144,220:224]=1.95
        self.assertIsNone(support(depth))

    def test_observed_patch_edge_or_too_low_spatial_resolution_cannot_authorize(self):
        small=dataclasses.replace(self.patch,half_width_m=.3)
        self.assertIsNone(self.adapter.certify(self.guard,small,self.cal,self.s,2.))
        t=target('dock',center=(0,0,0),now=2.)
        exp=dataclasses.replace(down_exposure(2.,(0,0,4)),epoch='own-local',pose_error_m=.005)
        depth=np.full((300,400),4.,np.float32)
        self.assertIsNone(build_support_patch(depth,[300,300,200,150],exp,t,2.,.48,.01))

    def test_monitor_rejects_ToF_plane_disagreement_and_never_corrects_global_z(self):
        bad=dataclasses.replace(self.s,range_m=self.s.range_m-.07)
        self.assertIsNone(self.adapter.certify(self.guard,self.patch,self.cal,bad,2.))
        self.assertIn('agree',self.adapter.last_rejection)
        self.assertIsNone(self.adapter.last_observation)
        good=self.adapter.certify(self.guard,self.patch,self.cal,self.s,2.)
        self.assertIsNotNone(good)
        self.assertFalse(self.adapter.last_observation['vio_global_z_correction_applied'])

    def test_monitor_rejects_drift_fast_descent_tilt_stale_range_and_epoch_reset(self):
        angle=np.deg2rad(5);R=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
        bads=[dict(position_local_m=[.06,0,self.s.position_local_m[2]]),
              dict(velocity_local_mps=[.1,0,-.1]),dict(velocity_local_mps=[0,0,-.4]),
              dict(local_from_body_rotation=R),dict(range_stamp=1.8),dict(epoch='reset'),
              dict(range_valid=False),dict(tracking_valid=False),dict(armed=False)]
        for bad in bads:
            self.assertIsNone(self.adapter.certify(self.guard,self.patch,self.cal,dataclasses.replace(self.s,**bad),2.),bad)

    def test_stale_support_map_and_wrong_robot_calibration_refuse_descent(self):
        self.assertIsNone(self.adapter.certify(self.guard,dataclasses.replace(self.patch,stamp=-20),self.cal,self.s,2.))
        old=MeasuredMapGuard(floor_map(1.),'own-local',.32,.16)
        self.assertIsNone(self.adapter.certify(old,self.patch,self.cal,self.s,2.))
        self.assertIsNone(self.adapter.certify(self.guard,self.patch,dataclasses.replace(self.cal,robot_calibration_id='other'),self.s,2.))

    def test_issued_certificate_audits_only_vertical_near_center_measured_motion(self):
        cert=self.adapter.certify(self.guard,self.patch,self.cal,self.s,2.)
        p=np.asarray(self.s.position_local_m)
        self.assertTrue(cert.covers_motion(p,p+[0,0,-.02],2.1,'own-local'))
        self.assertFalse(cert.covers_motion(p,p+[.08,0,-.02],2.1,'own-local'))
        self.assertFalse(cert.covers_motion(p,[0,0,self.cal.offset_m-.1],2.1,'own-local'))
        self.assertFalse(cert.covers_motion(p,p,2.3,'own-local'))
        self.assertFalse(cert.covers_motion(p,p,2.1,'reset'))



class NativeLandingExecutiveTests(unittest.TestCase):
    def test_occupied_floor_native_land_disarm_rearm_liftoff_with_preserved_vio(self):
        from task_executive import TaskExecutive,MissionLimits,Feedback
        from task_contact import LandingContext
        cal=calibrated_foot();patch=support();adapter=ContactLandingAdapter(MOUNT,AXIS,'actual-kit')
        mission=TaskExecutive(MissionLimits(.32,.16),[0,0,.124],'own-local')
        mission.install_foot_calibration(cal)
        mission.stage='DOCK';mission.phase='LAND';mission.target=target('dock',center=(0,0,0),now=2.)
        p=np.array([0.,0.,cal.offset_m+.30]);now=2.;armed=True;landed=False;mode='descent'
        actions=[];self.intents=[]
        for _ in range(100):
            now+=.1
            velocity=[0,0,0] if landed else [0,0,.2 if mode=='liftoff' else -.2]
            rs=sample(now,p[2]+MOUNT[2],position=p.copy(),landed=landed,armed=armed,velocity_local_mps=velocity)
            ctx=LandingContext(adapter,patch,cal,rs)
            guard=MeasuredMapGuard(floor_map(now),'own-local',.32,.16)
            f=Feedback(now,p.copy(),'own-local',armed,landed,0. if landed else .2)
            intent=mission.tick(f,[],guard,now,landing_context=ctx)
            actions.append(intent.action);self.intents.append({'stamp':now,**intent.as_dict()})
            self.assertIsNone(mission.fault,(mission.report(),intent.as_dict()))
            if intent.action=='land':
                self.assertEqual(intent.certificate.mode,'descent')
                p[2]=max(cal.offset_m,p[2]-.02)
                landed=abs(p[2]-cal.offset_m)<1e-6
            elif intent.action=='disarm':armed=False
            elif intent.action=='arm':armed=True;mode='liftoff'
            elif intent.action=='takeoff':
                self.assertEqual(intent.certificate.mode,'liftoff')
                p[2]=min(cal.offset_m+.30,p[2]+.02);landed=False
            if mission.stage=='WALL':break
        self.assertEqual(mission.stage,'WALL',(mission.report(),intent.as_dict(),now,p))
        self.assertTrue({'land','disarm','arm','takeoff'}.issubset(actions))
        self.assertTrue({'DOCK','REARM','RETAKEOFF'}.issubset(mission.completed))
        self.assertEqual(mission.epoch,'own-local')
        self.assertLess(now-patch.stamp,10.)
        self.assertAlmostEqual(mission.home[2],.124)
        self.assertIsNotNone(mission.completed['DOCK']['proof'])
        self.assertIsNotNone(mission.completed['RETAKEOFF']['proof'])
        self.last_report=mission.report();self.calibration=cal.as_dict()
        import json
        json.dumps(self.last_report)



    def test_hidden_marker_can_use_unchanged_dense_patch_and_fresh_tof_contact_proof(self):
        from task_executive import TaskExecutive,MissionLimits,Feedback
        from task_contact import LandingContext
        cal=calibrated_foot();patch=support();adapter=ContactLandingAdapter(MOUNT,AXIS,'actual-kit')
        mission=TaskExecutive(MissionLimits(.32,.16),[0,0,.124],'own-local');mission.install_foot_calibration(cal)
        mission.stage='DOCK';mission.phase='APPROACH';mission.target=target('dock',center=(0,0,0),now=1.)
        p=np.array([0.,0.,cal.offset_m+.30]);mission.views={'before':p.tolist()}
        rs=sample(2.,p[2]+MOUNT[2],position=p,landed=False,armed=True)
        guard=MeasuredMapGuard(floor_map(),'own-local',.32,.16)
        intent=mission.tick(Feedback(2.,p,'own-local',True,False,0.),[],guard,2.,landing_context=LandingContext(adapter,patch,cal,rs))
        self.assertEqual(intent.action,'land');self.assertEqual(intent.certificate.mode,'descent')
        self.assertFalse(intent.certificate.as_dict()['global_map_changed'])

    def test_contact_veto_does_not_erase_valid_continuous_body_feedback(self):
        from task_executive import TaskExecutive,MissionLimits,Feedback
        from task_contact import LandingContext
        cal=calibrated_foot();patch=support();adapter=ContactLandingAdapter(MOUNT,AXIS,'actual-kit')
        mission=TaskExecutive(MissionLimits(.32,.16),[0,0,.124],'own-local');mission.install_foot_calibration(cal)
        mission.stage='DOCK';mission.phase='LAND';mission.target=target('dock',center=(0,0,0),now=2.)
        p=np.array([0.,0.,cal.offset_m+.30])
        for i in range(10):
            now=2+i*.1;rs=sample(now,p[2]+MOUNT[2],position=p.copy(),landed=False,armed=True,range_valid=False)
            guard=MeasuredMapGuard(floor_map(now),'own-local',.32,.16)
            intent=mission.tick(Feedback(now,p.copy(),'own-local',True,False,0.),[],guard,now,landing_context=LandingContext(adapter,patch,cal,rs))
            self.assertEqual(intent.action,'hold');self.assertIsNone(mission.fault)
            self.assertEqual(mission.last_feedback.stamp,now)
        self.assertEqual(mission.completed,{})

    def test_home_alignment_is_tighter_than_contact_drift_limit_before_land(self):
        from task_executive import TaskExecutive,MissionLimits,Feedback
        cal=calibrated_foot();mission=TaskExecutive(MissionLimits(.32,.16),[0,0,.124],'own-local')
        mission.install_foot_calibration(cal);mission.stage='HOME';mission.phase='SEEK_HOME'
        p=np.array([.044,0.,.424]);guard=MeasuredMapGuard(floor_map(),'own-local',.32,.16)
        # Even before a contact patch arrives, do not enter LAND_HOME at an
        # offset that the real native-LAND monitor would subsequently veto.
        intent=mission.tick(Feedback(2.,p,'own-local',True,False,0.),[],guard,2.)
        self.assertEqual(intent.action,'move');self.assertEqual(mission.phase,'SEEK_HOME')

    def test_landed_flag_does_not_bypass_a_vetoed_contact_monitor(self):
        from task_executive import TaskExecutive,MissionLimits,Feedback
        from task_contact import LandingContext
        cal=calibrated_foot();patch=support();adapter=ContactLandingAdapter(MOUNT,AXIS,'actual-kit')
        mission=TaskExecutive(MissionLimits(.32,.16),[0,0,.124],'own-local');mission.install_foot_calibration(cal)
        mission.stage='DOCK';mission.phase='DISARM';mission.target=target('dock',center=(0,0,0),now=2.)
        rs=sample(2.,cal.offset_m+MOUNT[2],position=(0,0,cal.offset_m),range_valid=False)
        f=Feedback(2.,rs.position_local_m,'own-local',False,True,0.,0.,body_up_local=[0,0,1])
        intent=mission.tick(f,[],MeasuredMapGuard(floor_map(),'own-local',.32,.16),2.,landing_context=LandingContext(adapter,patch,cal,rs))
        self.assertEqual(intent.action,'hold');self.assertNotIn('DOCK',mission.completed)

    def test_staging_climbs_in_observed_free_space_before_crossing_raised_platform_edge(self):
        from task_executive import TaskExecutive,MissionLimits,Feedback
        cal=calibrated_foot();base=floor_map();values=base.state.copy()
        lo=base.idx([.8,-.7,.01]);hi=base.idx([2.2,.7,.6])
        values[lo[0]:hi[0]+1,lo[1]:hi[1]+1,10:hi[2]+1]=1
        mission=TaskExecutive(MissionLimits(.32,.16,max_step_m=.06,max_speed_mps=.4),[0,0,.124],'own-local')
        mission.install_foot_calibration(cal);mission.stage='DOCK'
        t=target('dock',center=(1.5,0,.6),now=2.);p=np.array([0.,0.,.7]);now=2.;horizontal_started=False
        for _ in range(100):
            now+=.15;t.stamp=now
            m=OnlineMap([*base.origin,base.res,*base.shape,0,now],values,.32,.16)
            f=Feedback(now,p.copy(),'own-local',True,False,.4)
            intent=mission.tick(f,[t],MeasuredMapGuard(m,'own-local',.32,.16),now)
            self.assertIsNone(mission.fault,intent.as_dict())
            if intent.action=='move':
                nextp=np.array(intent.goal)
                if abs(nextp[0]-p[0])>.001:
                    horizontal_started=True
                    self.assertGreater(p[2],.6+cal.offset_m+.30-.055)
                p=nextp
            if mission.phase=='LAND':break
        self.assertTrue(horizontal_started)
        self.assertEqual(mission.phase,'LAND',(intent.as_dict(),p))

    def test_liftoff_still_rejects_unknown_or_occupied_overhead(self):
        cal=calibrated_foot();patch=support();adapter=ContactLandingAdapter(MOUNT,AXIS,'actual-kit')
        rs=sample(2.,cal.offset_m+MOUNT[2],position=(0,0,cal.offset_m),armed=True,landed=True)
        for state in (-1,1):
            m=floor_map();values=m.state.copy();values[tuple(m.idx([0,0,.4]))]=state
            changed=OnlineMap([*m.origin,m.res,*m.shape,0,2.],values,.32,.16)
            self.assertIsNone(adapter.certify(MeasuredMapGuard(changed,'own-local',.32,.16),patch,cal,rs,2.,liftoff_height_m=.30))

if __name__=='__main__':unittest.main()
