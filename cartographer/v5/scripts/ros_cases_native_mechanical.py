"""Actual adapter with explicit NO-DDS stubs, synthetic perception only."""
import os,sys,unittest,json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from dataclasses import replace
import numpy as np
from competition_ros_stubs import install
M=install()
import competition_navigate as app
from task_executive import MissionLimits
from task_mechanical_executive import MechanicalTaskExecutive
from task_native_contact import *
from competition_inputs import BodyHistory
from online_map import OnlineMap
from ros_cases_competition import header,odom,free_map
from test_task_executive import target


def retained(n,track,clock,role='dock',center=None):
    c=np.asarray(track.center if center is None else center);a=np.linspace(0,2*np.pi,48,endpoint=False)
    landing=np.c_[np.cos(a),np.sin(a)]*.85+c[:2]
    xy=np.array([[x,y] for x in np.linspace(-1.2,1.2,9) for y in np.linspace(-1.2,1.2,9)])+c[:2]
    origin=c[:2].copy();A=np.c_[xy-origin,np.ones(len(xy))]
    source=pose_envelope(n.body_history,int(round(clock*1e9)))
    return RetainedSupport(track.id if role=='dock' else 'own-home',role,'cross' if role=='dock' else 'own_launch_plane',clock,n.session,n.calibration_id,
        tuple(c),(0,0,1),landing,convex_xy(xy),origin,np.array([0.,0.,c[2]]),np.linalg.pinv(A),np.full(len(xy),.005),source,
        ('a' if role=='dock' else 'c')*64,len(xy),.25,.01,0.,.01)


class TypedNativeCases(unittest.TestCase):
    def setUp(self):
        self.auth=patch.object(app,'assert_sitl_authorized');self.auth.start()
        self.n=app.CompetitionNavigator(os.environ['COMPETITION_STUB_RESULT'],app.CompetitionOptions(.02,.005,.02,
            foot_geometry=str(app.NAV_ROOT/'config/robot-foot-geometry.json'),complete_task=True,contact_rig_error_m=.005,map_pose_error_m=0.))
        self.n.pump=lambda dt=.01:None

    def tearDown(self):
        n=self.n;n.task_worker.close();n.frontier_worker.close();n.worker.close();self.auth.stop()

    def sensors(self,t,p,armed=False,landed=True,nav=14,velocity=(0,0,0),native_cadence=False,position_proxy=.003,range_delay=0.):
        n=self.n;n.test_time=t
        n.health=dict(stamp=t,valid=True,reason='synthetic_actual_callback_fixture',odometry_source='/nav/home_relative_odomimu',global_tof_height_correction_enabled=False,
            accepted_visual_update_features=20,visual_update_age=0.,global_height_frozen=True)
        n.home_relative_health_cb(app.String(data=json.dumps(dict(stamp=t,valid=True,home_epoch_ns=123456789,
            pose_covariance_clipped=False,position_prior_changed=False,native_fast_imu_epoch_verified=True))))
        n.local_cb(M.VehicleLocalPosition(timestamp=int(round(t*1e6)),x=p[0],y=-p[1],z=-p[2]))
        # Actual PX4 cadence family: status~0.504s, landed1s. On discrete
        # native transitions also publish an actual new packet explicitly.
        if not native_cadence or getattr(self,'last_status',-99)+.5<=t or getattr(self,'old_native',None)!=(armed,nav):
            n.status_cb(M.VehicleStatus(timestamp=int(round(t*1e6)),arming_state=2 if armed else 1,nav_state=nav));self.last_status=t;self.old_native=(armed,nav)
        if not native_cadence or getattr(self,'last_land',-99)+1.<=t:
            n.land_cb(M.VehicleLandDetected(timestamp=int(round(t*1e6)),landed=landed,at_rest=landed and np.linalg.norm(velocity)<.03));self.last_land=t
        msg=odom(t,p);msg.pose.covariance=np.diag([(position_proxy/3)**2]*3+[1e-6]*3).ravel().tolist();msg.twist.covariance=np.diag([(.0429/3)**2]*3+[1e-6]*3).ravel().tolist();msg.twist.twist.linear=NS(x=velocity[0],y=velocity[1],z=velocity[2]);n.odom(msg)
        n.range_cb(M.Range(header=header(t-range_delay),range=p[2]+float(app.TOF_IN_BODY[2])))

    def camera_frame(self,t):
        n=self.n;w,h=n.resolution;T=np.eye(4);T[:3,:3]=n.q.as_matrix();T[:3,3]=n.body;T=T@n.body_from_camera
        from scipy.spatial.transform import Rotation
        q=Rotation.from_matrix(T[:3,:3]).as_quat();pose=M.PoseStamped();pose.header=header(t)
        pose.pose.position=NS(x=T[0,3],y=T[1,3],z=T[2,3]);pose.pose.orientation=NS(x=q[0],y=q[1],z=q[2],w=q[3])
        n.left_cb(M.Image(header=header(t),encoding='mono8',width=w,height=h,step=w,data=bytes(w*h)))
        n.depth_cb(M.Image(header=header(t,'left_optical'),encoding='32FC1',width=w,height=h,step=w*4,data=np.full((h,w),2.,np.float32).tobytes()))
        n.camera_cb(pose)
        app.CompetitionNavigator.pump(n,.01)

    def test_actual_mechanical_bootstrap_accepts_real_cadence_without_fake_legacy_cal(self):
        n=self.n
        for i in range(45):
            t=2+i*.05;self.sensors(t,[0,0,.111],native_cadence=True);n.map=free_map(t)
            if i%4==0:self.camera_frame(t)
            # Keep native boot in WAIT_READY until enough samples, then inspect
            # the actual source installation before any command progression.
            if n.engine is None:n.bootstrap_once()
            if n.engine is not None:break
        self.assertIsInstance(n.foot,MechanicalStartup);self.assertIsInstance(n.engine,MechanicalTaskExecutive)
        self.assertAlmostEqual(n.foot.offset_m,.111);self.assertLess(n.foot.error_m,.012)
        self.assertEqual(n.bootstrap,'ARMING');self.assertFalse(n.command_pub.messages)
        self.assertGreaterEqual(n.joiner.last,0);self.assertGreater(n.latest_direct_depth_pixels,64)
        self.assertFalse(hasattr(n.foot,'resting_ground_z_local_m'))
        self.assertFalse(n.foot.home_body_anchor['covariance_subtracted'])

    def install_launch(self):
        n=self.n
        for i in range(30):self.sensors(2+i*.05,[0,0,.111])
        foot=MechanicalStartup.from_observer(n.mechanical_foot_report(),n.session,n.calibration_id,n.now())
        n.foot=foot;n.home=np.array([0,0,.111]);n.home_pose_error_m=.003
        n.engine=MechanicalTaskExecutive(MissionLimits(.32,.16),n.home,n.session,foot);n.bootstrap='ASCEND'
        return n

    def test_actual31_growth_is_reserved_before_send_with_ground_contact_prime(self):
        n=self.install_launch();self.sensors(3.5,[0,0,.111],True,True)
        n.map=free_map(n.now());self.assertIsNotNone(n.feedback_once());n.send(n.body+[0,0,.02],0.)
        cert=n.ledger.items[-1];original=cert.as_dict()
        self.assertGreater(cert.uncertainty['body_three_sigma_proxy_m'],.015)
        self.assertAlmostEqual(cert.uncertainty['causal_forward_reserve']['velocity_three_sigma_mps'],.0429)
        self.assertIsNotNone(cert.startup_support)
        self.sensors(3.55,[0,0,.11109607],True,False,position_proxy=.004482459,range_delay=.015)
        n.feedback_once();self.assertIsNone(n.engine.fault)
        self.assertTrue(n.ledger.last_audit['accepted'],n.ledger.last_audit)
        self.assertEqual(cert.as_dict(),original)
        self.assertLess(n.launch_height(n.now(),.85),.8)
        self.assertAlmostEqual(n.body_history.items[-1].error_m,.004482459)

    def test_startup_without_current_tof_or_with_wrong_plane_cannot_prime(self):
        n=self.install_launch();n.current_range=None
        with self.assertRaisesRegex(RuntimeError,'declared local launch'):n.send(n.body.copy(),0.)
        self.assertFalse(n.traj.messages)
        self.sensors(3.5,[0,0,.111],True,True)
        n.current_range=replace(n.current_range,range_m=.8)
        with self.assertRaisesRegex(RuntimeError,'declared local launch'):n.send(n.body.copy(),0.)
        self.assertFalse(n.traj.messages)

    def test_prior_forecast_is_not_updated_by_new_covariance_and_upper_side_stay_blocked(self):
        n=self.install_launch();cert=n.launch_certificate(n.now());old=cert.uncertainty['body_three_sigma_proxy_m']
        self.assertFalse(cert.covers_uncertainty(old+.001))
        self.assertFalse(cert.covers_motion(n.body,n.home+[0,0,.9],n.now(),n.session))
        self.assertFalse(cert.covers_motion(n.body,n.home+[.3,0,0],n.now(),n.session))
        self.assertFalse(cert.covers_motion(n.body,n.body,n.now()+.101,n.session))
        self.assertFalse(cert.covers_motion(n.body,n.body,n.now(),'changed_epoch'))
        self.assertEqual(cert.uncertainty['body_three_sigma_proxy_m'],old)

    def test_current_unknown_veto_aborts_actual_native_land_without_offboard_stream(self):
        n=self.n
        for i in range(30):self.sensors(2+i*.05,[0,0,.111])
        foot=MechanicalStartup.from_observer(n.mechanical_foot_report(),n.session,n.calibration_id,n.now())
        n.foot=foot;n.home=np.array([0,0,.111]);n.home_pose_error_m=.003
        n.engine=MechanicalTaskExecutive(MissionLimits(.32,.16),n.home,n.session,foot)
        n.engine.stage='DOCK';n.engine.phase='LAND';n.bootstrap='READY';n.run_start=4.
        self.sensors(4.,[0,0,.411],True,False,18)
        row=target('dock',center=(0,0,0),now=4.,origin=n.body);row.epoch=n.session;n.engine.target=row
        support=retained(n,row,4.);n.native_supports[support.source_hash]=support
        n.map=free_map(4.);n.map.state[tuple(n.map.idx([.2,0,.3]))]=-1
        n.native_land_requested=True;n.control_owner='LAND';n.owns_vehicle=True
        with self.assertRaisesRegex(RuntimeError,'Native LAND aborted'):n.control_once()
        self.assertFalse(n.traj.messages);self.assertFalse(n.mode.messages)
        self.assertTrue(any(c.command==176 and c.param3==3. for c in n.command_pub.messages))
        self.assertFalse(any(c.command==400 for c in n.command_pub.messages))
        self.assertNotIn('DOCK',n.engine.completed)

    def test_default_off_and_explicit_parameters(self):
        self.assertFalse(app.CompetitionOptions(.02,.005,.02).complete_task)
        with self.assertRaises(ValueError):app.CompetitionOptions(.02,.005,.02,complete_task=True)
        with self.assertRaises(ValueError):app.CompetitionOptions(.02,.005,.02,complete_task=True,foot_geometry='robot-foot-geometry.json')

    def test_complete_measured_protocol_uses_native_land_disarm_rearm_and_retained_support(self):
        n=self.n
        # Actual startup observer callbacks establish the typed robot/home record.
        for i in range(30):self.sensors(2+i*.05,[0,0,.111])
        report=n.mechanical_foot_report();foot=MechanicalStartup.from_observer(report,n.session,n.calibration_id,n.now())
        limits=MissionLimits(.32,.16,max_step_m=.12,max_speed_mps=.3,uncertainty_policy=None)
        n.foot=foot;n.home=np.array([0.,0.,.111]);n.home_pose_error_m=.003
        n.engine=MechanicalTaskExecutive(limits,n.home,n.session,foot);n.bootstrap='READY';n.run_start=4.
        n.body_history=BodyHistory();n.control_owner='OFFBOARD';n.native_land_requested=False
        # Initial launch is a prevalidated fixture, as in original adapter tests.
        p=np.array([0.,0.,.7]);velocity=np.zeros(3);armed=True;landed=False;nav=14;clock=4.
        centers={'tunnel_entry':(1.,0,.7),'tunnel_exit':(2.2,0,.7),'window':(3.6,0,.7),'dock':(5.,0,0.),'wall':(2.5,0,.8)}
        trajectory=[100];seen=0;setpoint_seen=0;actions=set();support_saved=False;home_saved=False
        def planner(goal):
            trajectory[0]+=1;n.trajectory(NS(header=header(clock),trajectory_id=trajectory[0],position=NS(x=goal[0],y=goal[1],z=goal[2]),velocity=NS(x=0.,y=0.,z=0.)))
        publish=n.goal_pub.publish
        n.goal_pub.publish=lambda msg:(publish(msg),planner([msg.pose.position.x,msg.pose.position.y,msg.pose.position.z]))
        for step in range(4400):
            clock=round(clock+.05,6)
            self.sensors(clock,p,armed,landed,nav,velocity,native_cadence=True)
            shape=(80,60,35);values=np.zeros(shape,np.int8);values[:,:,:10]=-1;values[:,:,10]=1
            n.map=OnlineMap([-1.,-3.,-1.,.1,*shape,0,clock],values,.32,.16)
            role=n.engine.desired_role;n.latest_tracks=[]
            if role in centers:
                row=target(role,center=centers[role],normal=(1,0,0) if role=='wall' else (-1,0,0),now=clock,origin=p)
                row.epoch=n.session;n.latest_tracks=[row]
                if role=='dock' and not support_saved:
                    sup=retained(n,row,clock);n.native_supports[sup.source_hash]=sup;support_saved=True
            if role=='home' and not home_saved:
                row=NS(center=np.array([0.,0.,0.]),id='own-home');sup=retained(n,row,clock,'home');n.native_supports[sup.source_hash]=sup;home_saved=True
            if n.goals:planner(n.goals[-1]['position'])
            trajectory_before=len(n.traj.messages)
            try:n.control_once()
            except Exception as error:raise AssertionError((str(error),clock,n.engine.stage,n.engine.phase,p,nav,armed,landed,n.native_contact.last_rejection,None if n.last_intent is None else(n.last_intent.action,n.last_intent.reason)))
            if nav==18 and armed and not landed:self.assertEqual(len(n.traj.messages),trajectory_before,'No offboard stream during confirmed native LAND')
            if n.last_intent is not None:actions.add(n.last_intent.action)
            for command in n.command_pub.messages[seen:]:
                if command.command==400:armed=bool(command.param1)
                elif command.command==21:nav=18
                elif command.command==176:nav=14 if command.param2==6 else 4
            seen=len(n.command_pub.messages);old=p.copy()
            if nav==18 and armed:
                p[2]=max(.111,p[2]-.01);landed=abs(p[2]-.111)<1e-6
            elif nav==14 and armed and len(n.traj.messages)>setpoint_seen:
                goal=np.diag([1,-1,-1])@np.asarray(n.traj.messages[-1].position);delta=goal-p;p+=delta*min(1.,.01/max(np.linalg.norm(delta),1e-9));landed=p[2]<=.114
            setpoint_seen=len(n.traj.messages);velocity=(p-old)/.05
            if n.complete:break
        self.assertTrue(n.complete,(clock,n.engine.report(),n.native_contact.last_rejection,p))
        self.assertEqual(set(n.engine.completed),{'TUNNEL_ENTRY','TUNNEL_EXIT','WINDOW','DOCK','REARM','RETAKEOFF','WALL','HOME'})
        self.assertTrue({'land','disarm','arm','takeoff','complete'}<=actions)
        self.assertGreater(n.ledger.audited_segments,0)
        self.assertTrue(n.write_result()['competition_tasks_complete'])
        Path(os.environ['COMPETITION_STUB_RESULT']).with_name('typed-native-evidence.json').write_text(json.dumps(dict(
            SYNTHETIC_PERCEPTION_AND_MAP=True,actual_DDS=False,actual_flight=False,initial_launch_prevalidated_fixture=True,
            report=plain(n.engine.report()),command_count=len(n.command_pub.messages),prior_audited_segments=n.ledger.audited_segments),indent=2))

if __name__=='__main__':unittest.main(verbosity=2)
