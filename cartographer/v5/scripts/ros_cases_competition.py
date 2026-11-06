"""Isolated actual-Navigator interface tests with explicit NO-DDS ROS stubs."""
import os,sys,unittest,json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace as NS
import numpy as np
from scipy.spatial.transform import Rotation
from competition_ros_stubs import install
M=install()
EVIDENCE={}
import competition_navigate as app
from task_executive import Intent,Feedback,TaskExecutive,MissionLimits
from task_safety import MeasuredMapGuard
from online_map import OnlineMap
from competition_inputs import ExactFrameJoiner


def header(t,frame='world'):
    ns=int(round(t*1e9));return NS(stamp=NS(sec=ns//1000000000,nanosec=ns%1000000000),frame_id=frame)


def odom(t,p=(0,0,.124)):
    m=M.Odometry();m.header=header(t)
    m.pose.pose.position=NS(x=p[0],y=p[1],z=p[2]);m.pose.covariance=np.diag([1e-6]*6).ravel().tolist()
    return m


def free_map(now=2.):
    shape=(50,50,35);return OnlineMap([-2.5,-2.5,-1.,.1,*shape,0,now],np.zeros(shape,np.int8),.32,.16)


def plain_command(c):
    return {k:getattr(c,k) for k in ('command','timestamp','param1','param2','param3') if hasattr(c,k)}


class Cases(unittest.TestCase):
    def test_actual_154_admission_still_refuses_launch_in_same_box_before_arm(self):
        n=self.node;m=odom(2.02);m.pose.covariance=np.diag([(.154/3)**2]*3+[1e-6]*3).ravel().tolist()
        n.odom(m);self.assertAlmostEqual(n.body_history.items[-1].error_m,.154)
        n.home_pose_error_m=.154;n.engine=TaskExecutive(MissionLimits(.32,.16,.124),n.home,n.session);n.bootstrap='ARMING'
        with self.assertRaisesRegex(RuntimeError,'declared local launch'):
            n.ensure_offboard_ground()
        self.assertFalse(any(m.command==400 for m in n.command_pub.messages))
        self.assertFalse(n.traj.messages)

    def test_actual_send_uses_expanded_current_and_map_uncertainty(self):
        n=self.node;n.bootstrap='READY'
        from task_uncertainty import UncertaintyPolicy
        n.uncertainty_policy=UncertaintyPolicy(map_pose_error_m=.154)
        m=odom(2.02);m.pose.covariance=np.diag([(.154/3)**2]*3+[1e-6]*3).ravel().tolist();n.odom(m)
        world=free_map();state=world.state.copy();state[tuple(world.idx(n.body+[0,.5,0]))]=-1
        n.map=OnlineMap([-2.5,-2.5,-1.,.1,*world.shape,0,2.],state,.32,.16)
        with self.assertRaisesRegex(RuntimeError,'not observed free'):n.send(n.body+[.05,0,0],0.)
        self.assertFalse(n.traj.messages)

    def setUp(self):
        self.auth=patch.object(app,'assert_sitl_authorized');self.auth.start()
        self.node=app.CompetitionNavigator(os.environ.get('COMPETITION_STUB_RESULT',str(Path.cwd()/'work/no-dds-competition-result.json')),app.CompetitionOptions(.003,.001,.01,map_pose_error_m=0.))
        n=self.node;n.test_time=2.;n.health={'stamp':2.,'valid':True,'reason':'ok_visual','accepted_visual_update_features':20,'visual_update_age':0.}
        n.local_cb(M.VehicleLocalPosition(timestamp=2000000));n.status_cb(M.VehicleStatus(timestamp=2000000))
        n.land_cb(M.VehicleLandDetected(timestamp=2000000));n.odom(odom(1.95));n.odom(odom(2.))
        n.map=free_map();n.home=n.body.copy();n.yaw=0.
        n.home_pose_error_m=.003
        # Actual Navigator.send still transforms/publishes; skip only its wall
        # waiting in most cases. Sensor/worker pumping is tested separately.
        self.real_pump=n.pump;n.pump=lambda dt=.01:None

    def tearDown(self):
        n=self.node;n.task_worker.close();n.frontier_worker.close();n.worker.close();self.auth.stop()

    def test_inherits_real_copied_navigator_and_keeps_existing_health_gates(self):
        self.assertTrue(app.Navigator.__module__=='navigate')
        self.assertNotIn('competition_ros_stubs',sys.modules['navigate'].__file__)
        self.assertTrue(self.node.healthy())
        self.node.health['valid']=False;self.assertFalse(self.node.healthy())

    def test_constructor_checks_sitl_before_node_or_publishers(self):
        self.auth.stop()
        with patch.dict(os.environ,{'SITL_AUTOMATION_AUTHORIZED':'0'}):
            with self.assertRaises(PermissionError):app.CompetitionNavigator('unused',app.CompetitionOptions(.003,.001,.01))
        self.auth.start()

    def test_fresh_native_landed_is_required_not_inferred_from_disarmed_status(self):
        n=self.node;n.native.records.pop('land');self.assertIsNone(n.native.flags(2.,2))
        self.assertFalse(n.ensure_offboard_ground())
        self.assertEqual(n.command_pub.messages,[])

    def test_command_ack_denial_blocks_retry_and_does_not_fake_state_confirmation(self):
        n=self.node;n.request(M.VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,(1.,))
        n.ack_cb(M.VehicleCommandAck(timestamp=2000001,command=400,result=2))
        with self.assertRaises(RuntimeError):n.request(400,(1.,))
        self.assertFalse(n.armed)

    def test_native_land_emits_no_offboard_heartbeat_or_position_stream(self):
        n=self.node;n.bootstrap='READY';n.armed=True
        n.status_cb(M.VehicleStatus(timestamp=2000001,arming_state=2,nav_state=18))
        cert=NS(mode='descent',issued=2.,valid_until=2.25)
        f=Feedback(2.,n.body,n.session,True,False,0.)
        n.dispatch(f,Intent('land','DOCK','LAND','certified',certificate=cert),NS())
        self.assertEqual(len(n.command_pub.messages),1);self.assertEqual(n.command_pub.messages[-1].command,21)
        self.assertEqual(n.mode.messages,[]);self.assertEqual(n.traj.messages,[])
        with self.assertRaises(RuntimeError):n.send(n.body.copy(),0.)

    def test_contact_veto_requests_real_native_loiter_without_conflicting_setpoints(self):
        n=self.node;n.bootstrap='READY';n.native_land_requested=True;n.armed=True
        n.status_cb(M.VehicleStatus(timestamp=2000001,arming_state=2,nav_state=18))
        f=Feedback(2.,n.body,n.session,True,False,0.)
        with self.assertRaises(RuntimeError):n.dispatch(f,Intent('hold','DOCK','LAND','missing ToF'),None)
        command=n.command_pub.messages[-1]
        self.assertEqual((command.command,command.param1,command.param2,command.param3),(176,1.,4.,3.))
        self.assertEqual(n.traj.messages,[])
        self.assertFalse(n.native_handoff)
        n.status_cb(M.VehicleStatus(timestamp=2000002,arming_state=2,nav_state=4));n.native_hold('test')
        self.assertTrue(n.native_handoff)

    def test_offboard_mode_is_primed_and_confirmed_before_arm_request(self):
        n=self.node;n.bootstrap='ARMING';n.engine=TaskExecutive(MissionLimits(.32,.16,.124),n.home,n.session)
        n.mode_prime_started=2.
        n.ensure_offboard_ground();self.assertFalse(n.command_pub.messages)
        n.test_time=3.3;n.health['stamp']=3.3;n.odom_t=3.3
        n.local_cb(M.VehicleLocalPosition(timestamp=3300000));n.status_cb(M.VehicleStatus(timestamp=3300000,nav_state=4));n.land_cb(M.VehicleLandDetected(timestamp=3300000))
        self.assertFalse(n.ensure_offboard_ground());self.assertEqual(n.command_pub.messages[-1].command,176)
        n.status_cb(M.VehicleStatus(timestamp=3300001,nav_state=14))
        self.assertTrue(n.ensure_offboard_ground())
        self.assertFalse(any(m.command==400 for m in n.command_pub.messages))

    def test_unknown_actual_offboard_volume_cannot_be_sent_to_px4(self):
        n=self.node;n.bootstrap='READY';m=free_map();m.state[:]=-1
        n.map=OnlineMap([-2.5,-2.5,-1.,.1,*m.shape,0,2.],m.state,.32,.16)
        with self.assertRaises(RuntimeError):n.send(n.body+[0,0,.3],0.)
        self.assertEqual(n.traj.messages,[])

    def test_actual_send_uses_real_ned_conversion_and_records_prior_certificate(self):
        n=self.node;n.bootstrap='READY';n.send(n.body+[.05,0,0],0.)
        self.assertEqual(len(n.traj.messages),1);self.assertEqual(len(n.ledger.items),1)
        np.testing.assert_allclose(n.traj.messages[-1].position,[.05,0,0],atol=1e-8)
        self.assertEqual(n.ledger.items[-1].issued,2.)

    def test_declared_launch_exception_cannot_escape_user_local_box(self):
        n=self.node;n.engine=TaskExecutive(MissionLimits(.32,.16,.124),n.home,n.session);n.bootstrap='ASCEND'
        n.send(n.home+[0,0,.035],0.)
        self.assertEqual(n.ledger.items[-1].as_dict()['kind'],'declared_local_launch_only')
        with self.assertRaises(RuntimeError):n.send(n.home+[.3,0,.2],0.)

    def test_fusion_freeze_requires_fresh_positive_ack_and_never_publishes_false(self):
        n=self.node;n.freeze_required=True
        self.assertFalse(n.freeze_ok());self.assertTrue(n.freeze_pub.messages[-1].data)
        n.health['global_height_frozen']=True;self.assertTrue(n.freeze_ok())
        n.test_time=3.;self.assertFalse(n.freeze_ok())
        self.assertTrue(all(m.data for m in n.freeze_pub.messages))

    def test_wrong_depth_grid_encoding_and_optical_frame_are_rejected(self):
        n=self.node
        m=M.Image(header=header(2.,'ideal_depth'),width=1,height=1,step=4,data=np.ones(1,np.float32).tobytes(),encoding='32FC1')
        n.depth_cb(m);self.assertTrue(n.blockers);self.assertEqual(len(n.joiner.buffers['depth']),0)

    def test_already_armed_startup_never_claims_or_commands_vehicle(self):
        n=self.node;n.status_cb(M.VehicleStatus(timestamp=2000001,arming_state=2))
        with self.assertRaises(RuntimeError):n.bootstrap_once()
        self.assertFalse(n.owns_vehicle);self.assertFalse(n.command_pub.messages)

    def test_disarm_is_not_emitted_without_native_landed_confirmation(self):
        n=self.node;f=Feedback(2.,n.body,n.session,True,False,0.)
        n.dispatch(f,Intent('disarm','DOCK','DISARM','test'),None)
        self.assertFalse(any(m.command==400 for m in n.command_pub.messages))

    def test_finish_success_uses_all_task_evidence_and_native_disarmed_landed_flags(self):
        n=self.node;n.engine=NS(report=lambda:{'competition_tasks_complete':True})
        n.complete=True;n.finish();report=json.loads(n.output.read_text())
        self.assertTrue(report['competition_tasks_complete'])
        n.complete=True;n.native.records.pop('land');n.finish();report=json.loads(n.output.read_text())
        self.assertFalse(report['competition_tasks_complete'])

    def test_callbacks_join_exact_exposure_and_validate_body_camera_rig(self):
        n=self.node
        class Capture:
            def __init__(self):self.jobs=[];self.error=None
            def submit(self,key,value):self.jobs.append((key,value))
            def take(self):return []
            def close(self):pass
        n.task_worker.close();n.task_worker=Capture()
        w,h=n.resolution;left=M.Image(header=header(2.),encoding='mono8',width=w,height=h,step=w,data=bytes(w*h))
        depth=M.Image(header=header(2.,'left_optical'),encoding='32FC1',width=w,height=h,step=w*4,data=np.full((h,w),2.,np.float32).tobytes())
        state=n.body_history.sample(2000000000);T=np.eye(4);T[:3,:3]=state.rotation.as_matrix();T[:3,3]=state.position
        T=T@n.joiner.body_from_camera;q=Rotation.from_matrix(T[:3,:3]).as_quat()
        pose=M.PoseStamped();pose.header=header(1.999);pose.pose.position=NS(x=T[0,3],y=T[1,3],z=T[2,3]);pose.pose.orientation=NS(x=q[0],y=q[1],z=q[2],w=q[3])
        n.left_cb(left);n.depth_cb(depth);n.camera_cb(pose);self.assertFalse(n.task_worker.jobs)
        pose.header=header(2.);n.camera_cb(pose)
        self.assertEqual(len(n.task_worker.jobs),1)
        frame=n.task_worker.jobs[0][1][0]
        self.assertEqual(frame.stamp_ns,2000000000)
        self.assertEqual(frame.exposure.pose_stamp,frame.exposure.depth_stamp)
        np.testing.assert_allclose(frame.exposure.local_from_camera,T,atol=1e-12)

    def test_startup_uses_actual_range_window_and_waits_for_armed_status(self):
        n=self.node;offset=.124;raw=offset+float(app.TOF_IN_BODY[2])
        for i in range(1,36):
            t=2.+i*.1;n.test_time=t;n.health['stamp']=t
            n.local_cb(M.VehicleLocalPosition(timestamp=int(t*1e6)))
            n.status_cb(M.VehicleStatus(timestamp=int(t*1e6),nav_state=14))
            n.land_cb(M.VehicleLandDetected(timestamp=int(t*1e6)))
            n.odom(odom(t));n.map=free_map(t)
            n.range_cb(M.Range(header=header(t),range=raw))
            n.bootstrap_once()
            if i<20:self.assertFalse(n.command_pub.messages)
        self.assertIsNotNone(n.foot);self.assertAlmostEqual(n.foot.offset_m,raw-app.TOF_IN_BODY[2])
        self.assertEqual(n.bootstrap,'ARMING')
        self.assertTrue(any(m.command==400 and m.param1==1 for m in n.command_pub.messages))
        n.status_cb(M.VehicleStatus(timestamp=5500001,nav_state=14,arming_state=2))
        n.bootstrap_once();self.assertEqual(n.bootstrap,'ASCEND')
        before=len(n.traj.messages);n.bootstrap_once();self.assertGreater(len(n.traj.messages),before)
        self.assertEqual(n.engine.epoch,n.session)

    def test_land_request_keeps_offboard_until_actual_land_confirmation_then_stops(self):
        n=self.node;n.bootstrap='READY';n.armed=True
        n.status_cb(M.VehicleStatus(timestamp=2000001,arming_state=2,nav_state=14))
        class Cert:
            issued=2.;valid_until=2.25;mode='descent'
            def covers_motion(self,*args):return True
        intent=Intent('land','DOCK','LAND','valid',certificate=Cert())
        n.last_intent=intent;f=Feedback(2.,n.body,n.session,True,False,0.)
        n.dispatch(f,intent,NS());self.assertGreater(len(n.traj.messages),0)
        count=len(n.traj.messages)
        n.status_cb(M.VehicleStatus(timestamp=2000002,arming_state=2,nav_state=18))
        n.dispatch(f,intent,NS());self.assertEqual(len(n.traj.messages),count)
        self.assertIsNone(n.mode_prime_started)

    def test_late_old_arm_denial_does_not_prevent_a_different_native_hold_command(self):
        n=self.node;n.request(176,(1.,6.));n.ack_cb(M.VehicleCommandAck(timestamp=2000001,command=176,result=2))
        n.native_hold('abort mode request')
        self.assertEqual(n.command_pub.messages[-1].param3,3.)

    def test_manual_takeover_stops_automatic_mode_requests(self):
        n=self.node;n.status_cb(M.VehicleStatus(timestamp=2000001,arming_state=2,nav_state=2,failsafe_and_user_took_over=True))
        with self.assertRaises(RuntimeError):n.control_once()
        n.native_hold('operator took over');self.assertFalse(n.command_pub.messages)

    def test_observe_only_never_enters_bootstrap_or_requests_commands(self):
        n=self.node;n.options=app.replace(n.options,observe_only=True)
        for _ in range(5):n.control_once()
        self.assertFalse(n.command_pub.messages);self.assertFalse(n.mode.messages);self.assertFalse(n.traj.messages)

    def test_exploration_high_texture_hint_cannot_look_outside_target_view_cone(self):
        n=self.node;n.views.choose=lambda now,p,desired,current,**kwargs:desired+np.pi
        chosen=n.frontier_view_yaw(n.body+[1,1,0])
        expected=np.pi/4-np.arctan2(n.body_from_camera[1,2],n.body_from_camera[0,2])
        self.assertLessEqual(abs(app.wrap(chosen-expected)),n.frontier_bearing_bound)
        self.assertAlmostEqual(chosen,expected)

    def test_old_view_memory_api_is_still_constrained_for_exploration(self):
        n=self.node;n.views.choose=lambda now,p,desired,current:desired+np.pi
        chosen=n.frontier_view_yaw(n.body+[0,1,0])
        self.assertAlmostEqual(chosen,np.pi/2,places=6)

    def test_platform_retakeoff_cannot_reuse_the_initial_declared_launch_exception(self):
        n=self.node;n.bootstrap='READY';n.engine=TaskExecutive(MissionLimits(.32,.16,.124),n.home,n.session)
        n.engine.stage='REARM';n.engine.phase='TAKEOFF'
        values=n.map.state.copy();values[:,:,10]=1
        n.map=OnlineMap([-2.5,-2.5,-1.,.1,*n.map.shape,0,2.],values,.32,.16)
        with self.assertRaises(RuntimeError):n.send(n.body+[0,0,.035],0.)
        self.assertFalse(n.traj.messages)

    def test_failure_exit_waits_for_actual_native_hold_confirmation(self):
        n=self.node;n.owns_vehicle=True;n.armed=True;n.bootstrap='READY';n.failure='negative fixture'
        n.status_cb(M.VehicleStatus(timestamp=2000001,arming_state=2,nav_state=18))
        calls=[0]
        def pump(dt=.01):
            calls[0]+=1
            if calls[0]==2:n.status_cb(M.VehicleStatus(timestamp=2000002,arming_state=2,nav_state=4))
        n.pump=pump;n.finish()
        self.assertGreaterEqual(calls[0],2);self.assertTrue(n.native_handoff)
        self.assertFalse(n.traj.messages)
        self.assertFalse(json.loads(n.output.read_text())['competition_tasks_complete'])

    def test_exploration_rejects_horizontal_bearing_outside_real_vertical_fov_when_tilted(self):
        n=self.node;n.q=Rotation.from_euler('y',1.2)
        self.assertIsNone(n.frontier_view_yaw(n.body+[1,0,0]))
        self.assertTrue(any('calibrated camera view' in reason for reason in n.blockers))

    def test_complete_executive_is_connected_through_real_callbacks_and_native_commands(self):
        # This is a NO-DDS protocol trace. Perception supplies synthetic plane
        # fixtures, the first launch is a prevalidated unit-test precondition,
        # and a bounded fake controller moves only according to emitted native
        # commands/setpoints. No real image recall or flight is claimed.
        from dataclasses import replace
        from task_contact import SupportPatch
        from task_landing import RangePose
        from test_task_executive import target
        from competition_inputs import BodyHistory
        n=self.node
        raw=.124+float(app.TOF_IN_BODY[2])
        for i in range(25):
            t=2+i*.05
            rs=RangePose(t,t,t,raw,np.array([0,0,.124]),np.eye(3),np.zeros(3),np.zeros(3),n.session,True,False,True,True)
            self.assertTrue(n.foot_calibrator.ingest(rs,t))
        n.foot=n.foot_calibrator.result(3.2);self.assertIsNotNone(n.foot)
        n.engine=TaskExecutive(MissionLimits(.32,.16,max_step_m=.12,max_speed_mps=.3),[0,0,.124],n.session)
        n.engine.install_foot_calibration(n.foot);n.home=np.array([0.,0.,.124]);n.bootstrap='READY';n.run_start=4.
        n.body_history=BodyHistory();n.native_land_requested=False;n.control_owner='OFFBOARD'
        centers={'tunnel_entry':(1,0,.7),'tunnel_exit':(2.2,0,.7),'window':(3.6,0,.7),'dock':(5.,0,0),'wall':(2.5,0,.8)}
        p=np.array([0.,0.,.7]);velocity=np.zeros(3);armed=True;landed=False;nav=14;clock=4.;seq=5000000
        trajectory_id=[100]
        def planner(goal):
            trajectory_id[0]+=1
            r=NS(header=header(clock),trajectory_id=trajectory_id[0],position=NS(x=goal[0],y=goal[1],z=goal[2]),velocity=NS(x=0.,y=0.,z=0.))
            n.trajectory(r)
        original_publish=n.goal_pub.publish
        def goal_publish(m):
            original_publish(m);planner([m.pose.position.x,m.pose.position.y,m.pose.position.z])
        n.goal_pub.publish=goal_publish
        seen=0;setpoint_seen=0;actions=set()
        for step in range(2500):
            clock+=.1;seq+=1;n.test_time=clock
            n.health={'stamp':clock,'valid':True,'reason':'ok_visual','accepted_visual_update_features':20,'visual_update_age':0.,'global_height_frozen':True}
            n.status_cb(M.VehicleStatus(timestamp=seq,arming_state=2 if armed else 1,nav_state=nav))
            n.land_cb(M.VehicleLandDetected(timestamp=seq,landed=landed))
            n.local_cb(M.VehicleLocalPosition(timestamp=seq,x=p[0],y=-p[1],z=-p[2]))
            m=odom(clock,p);m.twist.twist.linear=NS(x=velocity[0],y=velocity[1],z=velocity[2]);n.odom(m)
            n.map=free_map(clock)
            # Fixture has a real occupied floor, including at the dock/home.
            shape=(80,60,35);values=np.zeros(shape,np.int8);values[:,:,:10]=-1;values[:,:,10]=1
            n.map=OnlineMap([-1.,-3.,-1.,.1,*shape,0,clock],values,.32,.16)
            n.range_cb(M.Range(header=header(clock),range=p[2]+float(app.TOF_IN_BODY[2])))
            role=n.engine.desired_role
            n.latest_tracks=[]
            if role in centers:
                t=target(role,center=centers[role],normal=(1,0,0) if role=='wall' else (-1,0,0),now=clock,origin=p)
                t.epoch=n.session;n.latest_tracks=[t]
            if role in ('dock','home') or n.engine.stage=='REARM':
                center=centers['dock'] if role=='dock' or n.engine.stage=='REARM' else (0,0,0)
                marker='h' if center[0] else 'own_launch_plane'
                n.supports={'measured-patch':SupportPatch(t.id if role=='dock' else 'measured-patch',clock,n.session,n.calibration_id,center,(0,0,1),.48,.015,23000,0.,.005,marker)}
            if n.goals:planner(n.goals[-1]['position'])
            try:n.control_once()
            except Exception as error:raise AssertionError((str(error),clock,n.engine.stage,n.engine.phase,p,nav,armed,landed,n.control_owner,n.map.origin,n.map.shape,n.map.stamp,n.now(),n.map.segment(n.body,n.body),None if n.last_intent is None else (n.last_intent.action,n.last_intent.reason)))
            if n.last_intent is not None:actions.add(n.last_intent.action)
            for c in n.command_pub.messages[seen:]:
                if c.command==400:armed=bool(c.param1)
                elif c.command==21:nav=18
                elif c.command==176:nav=14 if c.param2==6 else 4
            seen=len(n.command_pub.messages)
            old=p.copy()
            if nav==18 and armed:
                p[2]=max(n.foot.offset_m,p[2]-.02);landed=abs(p[2]-n.foot.offset_m)<1e-6
            elif nav==14 and armed and len(n.traj.messages)>setpoint_seen:
                goal=app.D@np.asarray(n.traj.messages[-1].position) if hasattr(app,'D') else np.diag([1,-1,-1])@np.asarray(n.traj.messages[-1].position)
                delta=goal-p;p+=delta*min(1.,.02/max(np.linalg.norm(delta),1e-9));landed=p[2]<=n.foot.offset_m+.003
            setpoint_seen=len(n.traj.messages);velocity=(p-old)/.1
            if n.complete:break
        self.assertTrue(n.complete,(n.engine.report(),n.last_intent.as_dict(),p,clock))
        self.assertEqual(set(n.engine.completed),{'TUNNEL_ENTRY','TUNNEL_EXIT','WINDOW','DOCK','REARM','RETAKEOFF','WALL','HOME'})
        self.assertTrue({'land','arm','disarm','takeoff','complete'}.issubset(actions))
        self.assertGreater(n.ledger.audited_segments,0)
        EVIDENCE['complete_protocol_fixture']={'perception_and_support_are_synthetic_fixtures':True,
            'initial_takeoff_is_prevalidated_fixture':True,'controller_report':n.engine.report(),
            'prior_measured_segments':n.ledger.audited_segments,'commands':[plain_command(c) for c in n.command_pub.messages]}
        self.assertTrue(n.write_result()['competition_tasks_complete'])

    def test_direct_native_hold_yaw_uses_the_same_native_yaw_limiter(self):
        n=self.node;n.yaw_limiter.update(0.,0.,2.);n.test_time=2.05
        n.emergency_hold_ned=([0.,0.,0.],1.);n.emergency_native_counters=(None,None,None)
        n.native_hold('limited emergency hold')
        self.assertGreater(n.traj.messages[-1].yaw,0.)
        self.assertLessEqual(abs(n.traj.messages[-1].yaw),n.yaw_limiter.rate*.05+1e-9)

    def test_launch_stopping_volume_must_remain_inside_declared_box(self):
        n=self.node;n.engine=TaskExecutive(MissionLimits(.32,.16,.124),n.home,n.session);n.bootstrap='ASCEND'
        n.body=n.home+[0,0,.6];n.body_velocity=np.array([0.,0.,.5])
        with self.assertRaises(RuntimeError):n.send(n.body+[0,0,.035],0.)
        self.assertFalse(n.traj.messages)

    def test_duplicate_range_is_not_counted_or_allowed_to_erase_startup_window(self):
        n=self.node;m=M.Range(header=header(2.),range=.099)
        n.range_cb(m);count=len(n.foot_calibrator.samples);n.range_cb(m)
        self.assertEqual(len(n.foot_calibrator.samples),count)

    def test_actual_preflight_flag_must_pass_before_independent_arm_request(self):
        n=self.node;n.bootstrap='ARMING';n.engine=TaskExecutive(MissionLimits(.32,.16,.124),n.home,n.session)
        n.mode_prime_started=0.
        n.status_cb(M.VehicleStatus(timestamp=2000001,nav_state=14,pre_flight_checks_pass=False))
        self.assertFalse(n.ensure_offboard_ground())
        self.assertFalse(any(m.command==400 for m in n.command_pub.messages))
        n.status_cb(M.VehicleStatus(timestamp=2000002,nav_state=14,pre_flight_checks_pass=True))
        self.assertTrue(n.ensure_offboard_ground())

    def test_base_status_and_ack_subscriptions_feed_both_inherited_and_task_state(self):
        n=self.node
        self.assertEqual(len(n.subscriptions['/fmu/out/vehicle_status_v1']),1)
        self.assertEqual(len(n.subscriptions['/fmu/out/vehicle_command_ack']),1)
        n.subscriptions['/fmu/out/vehicle_status_v1'][0](M.VehicleStatus(timestamp=2000001,arming_state=2))
        self.assertTrue(n.armed);self.assertTrue(n.native.flags(2.,2)[0]);self.assertEqual(n.status_t,2.)
        n.subscriptions['/fmu/out/vehicle_command_ack'][0](M.VehicleCommandAck(timestamp=2000001,command=176,result=0))
        self.assertIn(176,n.command_acks);self.assertEqual(len(n.native.acks),1)


class MechanicalObserveCases(unittest.TestCase):
    def setUp(self):
        self.auth=patch.object(app,'assert_sitl_authorized');self.auth.start()
        self.node=app.CompetitionNavigator(os.environ['COMPETITION_STUB_RESULT'],app.CompetitionOptions(.02,.005,.01,
            observe_only=True,foot_geometry=str(app.NAV_ROOT/'config/robot-foot-geometry.json')))
        self.node.pump=lambda dt=.01:None

    def tearDown(self):
        n=self.node;n.task_worker.close();n.frontier_worker.close();n.worker.close();self.auth.stop()

    def feed(self,i,relative=True,armed=False,epoch=123456789):
        n=self.node;t=2.+i*.05;n.test_time=t
        n.health={'stamp':t,'valid':True,'odometry_source':'/nav/home_relative_odomimu' if relative else '/ov_msckf/odomimu',
            'global_tof_height_correction_enabled':not relative,'accepted_visual_update_features':20,'visual_update_age':0.}
        n.home_relative_health_cb(app.String(data=json.dumps(dict(stamp=t,valid=True,home_epoch_ns=epoch,
            pose_covariance_clipped=False,position_prior_changed=False,native_fast_imu_epoch_verified=True))))
        n.local_cb(M.VehicleLocalPosition(timestamp=int(t*1e6)));n.status_cb(M.VehicleStatus(timestamp=int(t*1e6),arming_state=2 if armed else 1))
        n.land_cb(M.VehicleLandDetected(timestamp=int(t*1e6)))
        m=odom(t,(.12,-.2,.137));m.pose.covariance=np.diag([(.154/3)**2]*3+[1e-6]*3).ravel().tolist();n.odom(m)
        n.range_cb(M.Range(header=header(t),range=.111+float(app.TOF_IN_BODY[2])+.0001*np.sin(i)))
        n.control_once()

    def test_actual_callbacks_produce_mechanical_dwell_and_measured_anchor_with_original_p(self):
        n=self.node
        for i in range(30):self.feed(i)
        report=n.write_result();m=report['observed_mechanical_foot_calibration']
        self.assertTrue(m['observed_dwell_ready'],m);self.assertTrue(m['known_home_body_anchor'])
        self.assertAlmostEqual(m['offset_m'],.111);self.assertLess(m['error_m'],.012)
        anchor=m['home_body_anchor'];np.testing.assert_allclose(anchor['position_local_m'],[.12,-.2,.137])
        self.assertAlmostEqual(anchor['position_three_sigma_proxy_m'],.154)
        np.testing.assert_allclose(np.asarray(anchor['pose_covariance'])[:3,:3],np.eye(3)*(.154/3)**2)
        self.assertFalse(anchor['covariance_subtracted']);self.assertFalse(anchor['covariance_averaged'])
        self.assertFalse(m['runtime_contact_qualified']);self.assertFalse(m['legacy_foot_calibration_created'])
        self.assertEqual(len(n.foot_calibrator.samples),0);self.assertIsNone(n.foot);self.assertIsNone(n.engine);self.assertIsNone(n.home)
        n.bootstrap_once();self.assertIsNone(n.engine)
        self.assertFalse(n.command_pub.messages);self.assertFalse(n.mode.messages);self.assertFalse(n.traj.messages)
        EVIDENCE['mechanical_callback_observe_only']=json.loads(n.output.read_text())

    def test_legacy_height_corrected_chain_cannot_become_known_relative_home(self):
        n=self.node
        for i in range(30):self.feed(i,relative=False)
        m=n.write_result()['observed_mechanical_foot_calibration']
        self.assertAlmostEqual(m['offset_m'],.111);self.assertFalse(m['observed_dwell_ready']);self.assertIsNone(m['home_body_anchor'])
        self.assertIn('home-relative',m['last_rejection']);self.assertFalse(n.command_pub.messages)

    def test_actual_native_arm_or_stale_stream_invalidates_observe_readiness(self):
        n=self.node
        for i in range(30):self.feed(i)
        self.assertTrue(n.mechanical_foot_report()['known_home_body_anchor'])
        n.status_cb(M.VehicleStatus(timestamp=4000000,arming_state=2))
        self.assertFalse(n.mechanical_foot_report()['known_home_body_anchor'])
        self.assertEqual(len(n.mechanical_foot_observer.samples),0)
        n.status_cb(M.VehicleStatus(timestamp=4000001,arming_state=1))
        self.assertFalse(n.mechanical_foot_report()['observed_dwell_ready'])
        n.test_time+=1.;self.assertFalse(n.mechanical_foot_report()['observed_dwell_ready'])
        self.assertFalse(n.owns_vehicle);self.assertFalse(n.command_pub.messages)

    def test_reference_epoch_change_cannot_reuse_the_previous_anchor(self):
        n=self.node
        for i in range(30):self.feed(i)
        self.feed(30,epoch=987654321)
        m=n.mechanical_foot_report();self.assertFalse(m['observed_dwell_ready']);self.assertIsNone(m['home_body_anchor'])
        self.assertIn('epoch changed',m['last_rejection'])

    def test_mechanical_mode_cannot_select_native_dispatch_or_private_geometry(self):
        with self.assertRaisesRegex(ValueError,'observe-only'):
            app.CompetitionOptions(.02,.005,.01,foot_geometry='robot-foot-geometry.json')
        from competition_mechanical_observer import MechanicalFootObserver
        with self.assertRaises(ValueError):
            MechanicalFootObserver(app.NAV_ROOT/'config',Path(__file__),app.TOF_IN_BODY,[0,0,-1],'rig',.02,.005)

    def test_health_only_epoch_change_invalidates_anchor_before_next_range(self):
        n=self.node
        for i in range(30):self.feed(i)
        n.home_relative_health_cb(app.String(data=json.dumps(dict(stamp=n.now(),valid=True,home_epoch_ns=987654321,
            pose_covariance_clipped=False,position_prior_changed=False))))
        self.assertFalse(n.mechanical_foot_report()['known_home_body_anchor'])
        self.assertEqual(len(n.mechanical_foot_observer.samples),0)

    def test_duplicate_range_does_not_reset_or_inflate_actual_dwell(self):
        n=self.node
        for i in range(30):self.feed(i)
        count=n.mechanical_foot_observer.packet_count
        n.range_cb(M.Range(header=header(n.now()),range=.111+float(app.TOF_IN_BODY[2])))
        self.assertEqual(n.mechanical_foot_observer.packet_count,count)
        self.assertTrue(n.mechanical_foot_report()['observed_dwell_ready'])

if __name__=='__main__':
    result=unittest.main(verbosity=2,exit=False).result
    if os.environ.get('COMPETITION_STUB_EVIDENCE'):
        EVIDENCE.update({'explicit_ros_stubs':True,'dds_started':False,'simulator_controlled':False,
            'original_course_flight_validated':False,'tests_run':result.testsRun,
            'tests_passed':result.wasSuccessful(),'real_navigator_source':str(Path(sys.modules['navigate'].__file__).resolve())})
        path=Path(os.environ['COMPETITION_STUB_EVIDENCE']);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(EVIDENCE,indent=2))
    sys.exit(not result.wasSuccessful())
