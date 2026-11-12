#!/usr/bin/env python3
"""SITL-only ROS/PX4 adapter for the complete CAD-free TaskExecutive.

Inherits navigation's coordinate conversion, planner following, measured stopping
checks and braking. It never calls Navigator.run() (navigation-only baseline).
Task perception is exact-exposure synchronized and runs in a bounded worker.
Native LAND owns its mode; no offboard setpoint is sent after LAND confirmation.
"""
import os,json,time,math,hashlib,uuid
from pathlib import Path
from dataclasses import dataclass,replace
from collections import deque
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy,ReliabilityPolicy
from std_msgs.msg import Bool,String
from sensor_msgs.msg import Image,Range
from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import VehicleStatus,VehicleLocalPosition,VehicleLandDetected,VehicleCommand,VehicleCommandAck,OffboardControlMode,TrajectorySetpoint
from scipy.spatial.transform import Rotation
from navigate import Navigator,install_stop_handlers,wrap,ROOT as NAV_ROOT
from camera_geometry import load_rig,bearing_visible
from robot_geometry import ROBOT,IMU_IN_BODY,BODY_FROM_IMU,TOF_IN_BODY
from perception_worker import LatestWorker
from task_observations import vector,plain
from task_executive import TaskExecutive,MissionLimits,Feedback
from task_safety import MeasuredMapGuard
from task_perception_helper import TaskPerceptionHelper
from task_landing import RestingFootCalibrator,surface_relative_observation
from task_contact import ContactLandingAdapter,LandingContext,build_support_patch
from competition_inputs import BodyHistory,NativeState,ExactFrameJoiner,PriorControlLedger,stamp_ns,image_array,range_pose
from task_uncertainty import UncertaintyPolicy
from task_relative_contact import existing_foot_calibrator_diagnostics
from ego_collision_envelope import EnvelopeSender,floor_ns,CEILING_NS
from measured_line_reference import short_forward_reference,camera_horizontal_half_fov_rad


def assert_sitl_authorized():
    if not (Path('/.dockerenv').exists() and os.environ.get('ROS_DOMAIN_ID')=='42'
            and os.environ.get('SITL_AUTOMATION_AUTHORIZED')=='1' and os.environ.get('HARDWARE_MODE','0')!='1'
            and os.environ.get('USE_SIM_TIME','1')=='1'):
        raise PermissionError('Competition automatic control remains explicitly authorized container SITL-only')


@dataclass(frozen=True)
class CompetitionOptions:
    tof_error_m: float
    tof_extrinsic_error_m: float
    stereo_error_m: float
    mission_budget_s: float=600.
    initial_wait_s: float=45.
    task_hz: float=3.
    direct_control_hz: float=20.
    native_land_topic: str='/fmu/out/vehicle_land_detected'
    command_ack_topic: str='/fmu/out/vehicle_command_ack'
    observe_only: bool=False
    uncertainty_admission_m: float=.20
    map_pose_error_m: float=.20
    foot_geometry: str=None
    complete_task: bool=False
    contact_rig_error_m: float=None
    retained_support_max_age_s: float=120.
    allow_conditional_free: bool=False
    require_ego_envelope: bool=False

    def __post_init__(self):
        if self.require_ego_envelope and not self.complete_task:raise ValueError("--require-ego-envelope requires --complete-task and its actual causal forecast")
        if self.foot_geometry and not (self.observe_only or self.complete_task):raise ValueError('Mechanical foot requires --observe-only or explicit --complete-task; native remains default off')
        if self.complete_task and (not self.foot_geometry or self.contact_rig_error_m is None or not 0<=self.contact_rig_error_m<=.02 or not 1<=self.retained_support_max_age_s<=180):raise ValueError('Complete typed task requires robot geometry, explicit complete rig-point error and bounded support history')
        UncertaintyPolicy(admission_limit_m=self.uncertainty_admission_m,map_pose_error_m=self.map_pose_error_m)
        if not 0<self.tof_error_m<=.02 or not 0<=self.tof_extrinsic_error_m<=.02 or not 0<self.stereo_error_m<=.02:
            raise ValueError('Explicit validated range/extrinsic/direct-stereo error budgets required')
        if not 60<=self.mission_budget_s<=1800 or not 5<=self.initial_wait_s<=120 or not .5<=self.task_hz<=5 or not 10<=self.direct_control_hz<=50:
            raise ValueError('Invalid bounded competition timing parameters')


class LaunchCertificate:
    """Only the user's declared local takeoff box, anchored to measured home."""
    def __init__(self,now,epoch,home,radius,halfheight,halfwidth,height,extra=(0.,0.,0.),uncertainty=None,startup_support=None):
        self.issued=now;self.valid_until=now+.25;self.epoch=epoch;self.home=np.asarray(home).copy()
        self.radius=radius;self.halfheight=halfheight;self.halfwidth=halfwidth;self.height=height
        self.extra=np.broadcast_to(np.asarray(extra,float),(3,)).copy();self.uncertainty=dict(uncertainty or {})
        self.startup_support=None if startup_support is None else dict(startup_support)
        reserve=self.uncertainty.get('causal_forward_reserve')
        if reserve is not None:self.valid_until=min(self.valid_until,float(reserve['valid_until']))
    def covers_uncertainty(self,error_m):
        return not self.uncertainty or error_m<=self.uncertainty['body_three_sigma_proxy_m']+1e-12
    def covers_motion(self,a,b,now,epoch):
        if epoch!=self.epoch or not self.issued-1e-9<=now<=self.valid_until:return False
        for p in (vector(a),vector(b)):
            if np.any(np.abs(p[:2]-self.home[:2])+self.radius+self.extra[:2]>self.halfwidth):return False
            lower=self.home[2]-.015+self.extra[2]
            if self.startup_support is not None and p[2]<lower:
                if np.linalg.norm(p[:2]-self.home[:2])>self.startup_support['center_xy_enclosure_m']:return False
                lower=self.startup_support['body_lower_enclosure_z_m']
            if not lower<=p[2]<=self.home[2]+self.height-self.halfheight-self.extra[2]:return False
        return True
    def as_dict(self):
        return {'kind':'declared_local_launch_only','issued':self.issued,'valid_until':self.valid_until,
                'epoch':self.epoch,'measured_home_body':self.home.tolist(),'halfwidth_m':self.halfwidth,
                'height_above_resting_body_m':self.height,'extra_collision_margin_xyz_m':self.extra.tolist(),
                'uncertainty':self.uncertainty,'startup_support':self.startup_support,'global_map_changed':False}


class ControlCertificate:
    """Published offboard reference plus measured-velocity stopping envelope."""
    def __init__(self,guard,paths,now):
        self.guard=guard;self.paths=tuple((tuple(a),tuple(b)) for a,b in paths)
        self.issued=now;self.valid_until=min(now+.25,guard.map.stamp+guard.max_age_s);self.epoch=guard.epoch
        reserve=guard.uncertainty.get('causal_forward_reserve')
        if reserve is not None:self.valid_until=min(self.valid_until,float(reserve['valid_until']))
    def covers_motion(self,a,b,now,epoch):
        if epoch!=self.epoch or not self.issued-1e-9<=now<=self.valid_until:return False
        a,b=vector(a),vector(b)
        def distance(p):
            ds=[]
            for x,y in self.paths:
                x,y=np.asarray(x),np.asarray(y);v=y-x
                u=np.clip((p-x)@v/max(v@v,1e-12),0.,1.);ds.append(np.linalg.norm(p-x-u*v))
            return min(ds)
        return max(distance(a),distance(b))<=self.guard.margin and self.guard.map.segment(a,b,extra_margin=self.guard.extra)
    def as_dict(self):
        return {'kind':'actual_offboard_control_and_stopping','issued':self.issued,'valid_until':self.valid_until,
                'epoch':self.epoch,'map_stamp':self.guard.map.stamp,'paths':self.paths,
                'extra_collision_margin_xyz_m':self.guard.extra,'uncertainty':self.guard.uncertainty}
    def covers_uncertainty(self,error_m):
        return not self.guard.uncertainty or error_m<=self.guard.uncertainty['body_three_sigma_proxy_m']+1e-12


class CompetitionNavigator(Navigator):
    def __init__(self,output,options):
        assert_sitl_authorized()
        super().__init__(output,mission='course_corridor')
        self.options=options;self.c['max_mission_seconds']=options.mission_budget_s
        # Conditional static-scene geometry is an explicit model, never raw depth.
        self.c['geometric_free_rays']=bool(options.allow_conditional_free and self.c.get('geometric_free_rays',False))
        self.uncertainty_policy=UncertaintyPolicy(options.uncertainty_admission_m,self.c['xy_margin_m'],
                                                 self.c['vertical_margin_m'],options.map_pose_error_m,
                                                 map_geometry_error_m=options.stereo_error_m)
        self.home_pose_error_m=None
        self.session='accepted-body-'+uuid.uuid4().hex;self.body_history=BodyHistory(uncertainty_policy=self.uncertainty_policy);self.native=NativeState()
        self.ego_envelope=EnvelopeSender(self.session) if options.require_ego_envelope else None
        self.ego_envelope_pub=None;self.ego_envelope_sub=None;self.ego_latest_goal_ack=None
        if self.ego_envelope is not None:
            envelope_qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.VOLATILE)
            # Subscribe before the first record; ACK is not durable authorization.
            self.ego_envelope_sub=self.create_subscription(String,'/nav/ego_collision_envelope_ack',self.ego_envelope_ack,envelope_qos)
            self.ego_envelope_pub=self.create_publisher(String,'/nav/ego_collision_envelope',envelope_qos)
        self.K,T,_,self.resolution=load_rig();body_cam=np.eye(4)
        body_cam[:3,:3]=BODY_FROM_IMU.as_matrix()@T[:3,:3]
        body_cam[:3,3]=IMU_IN_BODY+BODY_FROM_IMU.apply(T[:3,3])
        calibration=json.dumps({'robot':ROBOT,'K':self.K.tolist(),'imu_from_camera':T.tolist(),'resolution':self.resolution},sort_keys=True)
        self.body_from_camera=body_cam
        fx,fy,cx,cy=self.K;w,h=self.resolution
        self.frontier_bearing_bound=max(0.,min(math.atan2(cx,fx),math.atan2(w-1-cx,fx),math.atan2(cy,fy),math.atan2(h-1-cy,fy))-.1)
        # Guideline axis window = this calibration's horizontal half-FOV (was a fixed 85 deg),
        # matching the trial45 actual-frame tests that reject the initial transverse edges.
        self.line_axis_turn_bound=camera_horizontal_half_fov_rad(self.K,w)
        self.calibration_id=hashlib.sha256(calibration.encode()).hexdigest()
        self.joiner=ExactFrameJoiner(self.K,body_cam,self.session,self.calibration_id,self.resolution)
        self.tof_axis=np.asarray(ROBOT.get('tof_axis_body',[0.,0.,-1.]),float)
        self.mechanical_foot_observer=None;self.home_relative_adapter_health=None
        self.native_contact=None;self.native_supports={};self.native_contact_rejections={};self.latest_direct_depth_pixels=0
        if options.foot_geometry:
            from competition_mechanical_observer import MechanicalFootObserver
            self.mechanical_foot_observer=MechanicalFootObserver(NAV_ROOT/'config',options.foot_geometry,
                TOF_IN_BODY,self.tof_axis,self.calibration_id,options.tof_error_m,options.tof_extrinsic_error_m)
            if options.complete_task:
                from task_native_contact import NativeMechanicalContact
                self.native_contact=NativeMechanicalContact(TOF_IN_BODY,self.tof_axis,self.calibration_id,options.retained_support_max_age_s).configure_tof(options.tof_error_m,options.tof_extrinsic_error_m)
        self.foot_calibrator=RestingFootCalibrator(TOF_IN_BODY,self.tof_axis,self.calibration_id,
                                                   options.tof_error_m,options.tof_extrinsic_error_m)
        self.contact=ContactLandingAdapter(TOF_IN_BODY,self.tof_axis,self.calibration_id)
        self.task_perception=TaskPerceptionHelper();self.task_worker=LatestWorker(self.perceive)
        self.frontier_worker=LatestWorker(self.plan_frontier);self.ledger=PriorControlLedger()
        self.engine=None;self.foot=None;self.current_range=None;self.pending_ranges=deque(maxlen=8)
        self.latest_tracks=[];self.forward=None;self.supports={};self.last_frame_submit=-1.;self.last_feedback=-1.;self.last_range_stamp_ns=-1
        self.bootstrap_arc_hold=None;self._turn_hold=None
        self.bootstrap='WAIT_READY';self.bootstrap_started=self.now();self.started_wall=time.monotonic();self.bootstrap_phase_started=self.now()
        self.control_owner='NONE';self.native_land_requested=False;self.native_handoff=False;self.owns_vehicle=False;self.manual_handoff=False;self.emergency_hold_ned=None;self.emergency_native_counters=None
        self.mode_prime_started=None;self.command_requests={};self.last_intent=None;self.active_goal=None
        self.visited=deque(maxlen=256);self.frontier_result=None;self.frontier_pending=False;self.last_frontier_request=-1.;self.frontier_level=0
        self.last_freeze=-1.;self.freeze_required=False;self.height_freeze_ack_at=None;self.blockers={};self.adapter_events=[];self.launch_proofs=[];self.latest_launch_twist=None
        self.freeze_pub=self.create_publisher(Bool,'/nav/freeze_global_height',QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.subs += [self.create_subscription(Image,'/vio/left/image_raw',self.left_cb,qos_profile_sensor_data),
                      self.create_subscription(Image,'/nav/measured_depth',self.depth_cb,QoSProfile(depth=2,reliability=ReliabilityPolicy.RELIABLE)),
                      self.create_subscription(PoseStamped,'/nav/camera_pose',self.camera_cb,qos_profile_sensor_data),
                      self.create_subscription(Range,'/carto/down_range',self.range_cb,qos_profile_sensor_data),
                      self.create_subscription(VehicleLandDetected,options.native_land_topic,self.land_cb,qos_profile_sensor_data),
                      ]
        if self.mechanical_foot_observer is not None:
            self.subs.append(self.create_subscription(String,'/nav/home_relative_adapter_health',self.home_relative_health_cb,5))
        if options.command_ack_topic!='/fmu/out/vehicle_command_ack':
            self.subs.append(self.create_subscription(VehicleCommandAck,options.command_ack_topic,self.command_ack,qos_profile_sensor_data))

    def event(self,name,**details):
        self.adapter_events.append({'stamp':self.now(),'event':name,**plain(details)})
        self.adapter_events=self.adapter_events[-512:]

    def block(self,reason):
        self.blockers[reason]=self.blockers.get(reason,0)+1

    def odom(self,m):
        p=m.pose.pose.position;q=m.pose.pose.orientation;v=m.twist.twist.linear;w=m.twist.twist.angular
        try:
            velocity_error=None
            if self.options.complete_task:
                vc=np.asarray(m.twist.covariance,float).reshape(6,6)
                if not np.isfinite(vc).all() or not np.allclose(vc,vc.T,atol=1e-10) or np.linalg.eigvalsh((vc+vc.T)*.5).min()<-1e-10:raise ValueError('Actual native twist covariance required for causal launch forecast')
                velocity_error=3*math.sqrt(max(0.,np.linalg.eigvalsh(vc[:3,:3]).max()))
            good=self.body_history.add(stamp_ns(m.header),[p.x,p.y,p.z],[q.x,q.y,q.z,q.w],
                                     [v.x,v.y,v.z],[w.x,w.y,w.z],m.pose.covariance,m.header.frame_id)
        except ValueError as error:
            if self.mechanical_foot_observer is not None:self.mechanical_foot_observer.reject('actual accepted BODY rejected: '+str(error))
            self.block(str(error));return
        if not good and self.body_history.fault and self.mechanical_foot_observer is not None:self.mechanical_foot_observer.reject(self.body_history.fault)
        if good:
            if self.options.complete_task:self.latest_launch_twist=(stamp_ns(m.header)*1e-9,velocity_error)
            super().odom(m)
        self.drain_inputs()

    def vehicle_status(self,m):
        # The base subscription binds this override. One ordered callback feeds
        # BOTH the inherited handshake/status fields and the task native state.
        if self.native.update('status',m,self.now()):
            super().vehicle_status(m)
            if self.mechanical_foot_observer is not None and m.arming_state==VehicleStatus.ARMING_STATE_ARMED:self.mechanical_foot_observer.reject('actual native armed transition interrupted disarmed dwell')
            if getattr(m,'failsafe_and_user_took_over',False):self.manual_handoff=True
    def status_cb(self,m):self.vehicle_status(m)
    def local_position(self,m):
        if self.native.update('local',m,self.now()):super().local_position(m)
    def local_cb(self,m):self.local_position(m)
    def land_cb(self,m):
        fresh=self.native.update('land',m,self.now())
        if fresh and self.mechanical_foot_observer is not None and (not m.landed or m.freefall):self.mechanical_foot_observer.reject('actual native landed/freefall transition interrupted dwell')
        self.drain_inputs()
    def command_ack(self,m):
        if getattr(m,'target_system',0) not in (0,250):return
        if self.native.update('ack',m,self.now()):super().command_ack(m)
    def ack_cb(self,m):self.command_ack(m)
    def range_cb(self,m):
        key=stamp_ns(m.header)
        if key<=self.last_range_stamp_ns:return
        self.last_range_stamp_ns=key;self.pending_ranges.append(m);self.drain_inputs()
    def left_cb(self,m):
        try:self.joiner.add('left',stamp_ns(m.header),image_array(m,'mono8',self.resolution))
        except ValueError as error:self.block(str(error));return
        self.drain_inputs()
    def depth_cb(self,m):
        if m.header.frame_id!='left_optical':self.block('raw direct depth optical frame mismatch');return
        try:self.joiner.add('depth',stamp_ns(m.header),image_array(m,'32FC1',self.resolution))
        except ValueError as error:self.block(str(error));return
        self.drain_inputs()
    def camera_cb(self,m):
        if m.header.frame_id!='world':self.block('camera pose must use accepted local world frame');return
        p=m.pose.position;q=m.pose.orientation;T=np.eye(4)
        try:
            quaternion=vector([q.x,q.y,q.z,q.w],4)
            if abs(np.linalg.norm(quaternion)-1)>.001:raise ValueError('Nonunit exposure camera quaternion')
            T[:3,:3]=Rotation.from_quat(quaternion).as_matrix();T[:3,3]=vector([p.x,p.y,p.z])
        except ValueError as error:self.block(str(error));return
        self.joiner.add('camera',stamp_ns(m.header),T);self.drain_inputs()

    def home_relative_health_cb(self,message):
        try:
            data=json.loads(message.data)
            if not isinstance(data,dict):raise ValueError('Expected actual home-relative health mapping')
            self.home_relative_adapter_health=data
            observer=self.mechanical_foot_observer
            if observer is not None:
                if not data.get('valid'):observer.reject('actual home-relative adapter health invalid')
                elif observer.relative_epoch is not None and data.get('home_epoch_ns')!=observer.relative_epoch:observer.reject('Actual home reference epoch changed; adapter restart required')
        except (ValueError,TypeError) as error:
            self.home_relative_adapter_health=None
            if self.mechanical_foot_observer is not None:self.mechanical_foot_observer.reject('invalid actual home-relative health packet')
            self.block('invalid home-relative adapter health: '+str(error))

    def mechanical_foot_report(self):
        if self.mechanical_foot_observer is None:return None
        return self.mechanical_foot_observer.report(self.now(),self.health,self.home_relative_adapter_health,
            native_flags=self.native.flags(self.now(),VehicleStatus.ARMING_STATE_ARMED))

    def drain_inputs(self):
        now=self.now();flags=self.native.flags(now,VehicleStatus.ARMING_STATE_ARMED)
        for m in list(self.pending_ranges):
            key=stamp_ns(m.header);state=self.body_history.sample(key)
            if state is None and self.body_history.items and key>self.body_history.items[-1].stamp_ns:continue
            self.pending_ranges.remove(m)
            rs=range_pose(m,state,self.session,flags,now,self.healthy())
            if rs is None:
                if self.mechanical_foot_observer is not None:self.mechanical_foot_observer.reject('actual matched BODY/range/native input unavailable')
                continue
            self.current_range=rs
            if self.engine is None:
                if self.mechanical_foot_observer is None:self.foot_calibrator.ingest(rs,now)
                else:self.mechanical_foot_observer.ingest(rs,state,self.native,self.body_history,
                    self.health,self.home_relative_adapter_health,now)
        for frame in self.joiner.take(self.body_history,now):
            if frame.exposure.stamp-self.last_frame_submit<1/self.options.task_hz:continue
            self.last_frame_submit=frame.exposure.stamp
            role=None if self.engine is None or self.bootstrap!='READY' else self.engine.desired_role
            radius=self.c['vehicle_radius_m']+self.c['xy_margin_m']
            reference=max(self.uncertainty_policy.map_pose_error_m,self.body_history.observed_position_error_max_m)
            radius+=max(self.uncertainty_policy.extra(frame.exposure.pose_error_m,reference)[:2])
            width=radius+(self.map.res if self.map is not None else self.c['resolution_m'])+self.contact.max_lateral_error
            self.task_worker.submit('frame',(frame,role,self.foot,width))

    def perceive(self,key,job):
        frame,role,foot,width=job;self.task_perception.foot_calibration=None if self.options.complete_task else foot;now=self.now()
        mask=np.isfinite(frame.measured_depth)
        if self.options.complete_task and role=='home' and foot is not None:
            from task_native_home import measured_home_candidates
            rows=measured_home_candidates(frame.measured_depth,self.K,frame.exposure,foot,self.options.stereo_error_m)
            tracks=self.task_perception.tracker.update(rows,frame.exposure,'home',now)
            from task_observations import measured_forward_evidence
            result={'stamp':frame.exposure.stamp,'epoch':self.session,'tracks':tracks,
                'forward_evidence':measured_forward_evidence(frame.measured_depth,self.K,frame.exposure,now,mask)}
        else:result=self.task_perception.frame(frame.image,frame.measured_depth,self.K,frame.exposure,role,now,mask)
        result['native_supports']=[];result['direct_depth_pixels']=int(np.isfinite(frame.measured_depth).sum())
        if self.options.complete_task and foot is not None and role in ('dock','home'):
            from task_native_contact import pose_envelope,build_retained_support
            try:
                source=pose_envelope(self.body_history,frame.stamp_ns)
                anchor={**foot.home_body_anchor,'mechanical_offset_m':foot.offset_m,'mechanical_error_m':foot.error_m}
                for track in result['tracks'][:3]:
                    patch=build_retained_support(frame.image,frame.measured_depth,self.K,frame.exposure,track,
                        self.body_from_camera,source,self.options.stereo_error_m,self.options.contact_rig_error_m,
                        self.c['vehicle_radius_m']+self.c['xy_margin_m'],anchor if role=='home' else None)
                    if patch is not None:result['native_supports'].append(patch)
            except ValueError as error:result['native_support_rejection']=str(error)
        result['support_patches']=[]
        scale=float(self.c.get('stereo_matching_scale',1.))
        stride=max(self.resolution[0]/max(1,round(self.resolution[0]*scale)),
                   self.resolution[1]/max(1,round(self.resolution[1]*scale)),1.)
        for track in result['tracks'][:2]:
            if track.role not in ('dock','home') or not track.consistent:continue
            patch=build_support_patch(frame.measured_depth,self.K,frame.exposure,track,now,width,
                                      self.options.stereo_error_m,mask,measurement_stride_px=stride)
            if patch is not None:result['support_patches'].append(patch)
        return [result]

    def plan_frontier(self,key,job):
        original,p,home,visited,yaw,level,extra,hints=job;m=original.clearance_view(extra)
        relative=float(p[2]-home[2]);cruise=relative+(0,1,-1)[level%3]*m.res
        candidate=m.frontier(p,home,visited,yaw,max_heading_change=math.pi,cruise_height=cruise,guideline_segments=hints)
        if candidate is None:
            candidate=m.frontier(p,home,visited,yaw,max_heading_change=math.pi,cruise_height=relative,allow_altitude_change=True,guideline_segments=hints)
        if self.options.complete_task and self.engine is not None and self.engine.stage=='WALL' and (candidate is None or level%3==0):
            from task_native_contact import observed_vertical_frontier
            candidate=observed_vertical_frontier(m,p,extra=0.)
        return [{'stamp':self.now(),'origin':p,'candidate':candidate,'level':level}]

    def pump(self,dt=.01):
        super().pump(dt)
        for result in self.task_worker.take():
            if result['epoch']!=self.session:continue
            self.latest_tracks=result['tracks'];self.forward=result['forward_evidence'];self.latest_direct_depth_pixels=result.get('direct_depth_pixels',0)
            for support in result['support_patches']:self.supports[support.target_id]=support
            if result.get('native_support_rejection'):self.block('typed raw support: '+result['native_support_rejection'])
            for support in result.get('native_supports',[]):self.native_supports[support.source_hash]=support
            self.native_supports={k:p for k,p in self.native_supports.items() if 0<=self.now()-p.stamp<=self.options.retained_support_max_age_s}
            if len(self.native_supports)>32:self.native_supports=dict(sorted(self.native_supports.items(),key=lambda x:x[1].stamp)[-32:])
            self.supports={key:p for key,p in self.supports.items() if 0<=self.now()-p.stamp<=10.}
            if len(self.supports)>12:self.supports=dict(sorted(self.supports.items(),key=lambda item:item[1].stamp)[-12:])
        for result in self.frontier_worker.take():self.frontier_result=result;self.frontier_pending=False
        if self.task_worker.error:self.block('task worker: '+self.task_worker.error)
        if self.frontier_worker.error:self.block('frontier worker: '+self.frontier_worker.error);self.frontier_pending=False

    def forward_position_uncertainty(self,now):
        if not self.body_history.items:raise RuntimeError('Accepted BODY required for causal uncertainty forecast')
        state=self.body_history.items[-1];error=state.error_m+.001;forecast=None
        if self.options.complete_task:
            twist=self.latest_launch_twist
            if twist is None or abs(twist[0]-state.stamp)>1e-8 or not 0<=now-state.stamp<=.15:raise RuntimeError('Fresh same-stamp native twist covariance required for control forecast')
            horizon=.10+now-state.stamp
            # Same explicit 2m/s2 bounded-acceleration domain as the native-twist
            # bridge. Marginal sum, no independence/covariance cancellation.
            growth=twist[1]*horizon+.5*2.*horizon*horizon;error+=growth
            forecast=dict(horizon_s=horizon,valid_until=now+.10,velocity_three_sigma_mps=twist[1],
                          acceleration_enclosure_mps2=2.,position_growth_reserve_m=growth,
                          source_stamp=state.stamp,model='constant-velocity error enclosure; actual acceleration <=2m/s2')
        return error,forecast

    def guard(self):
        if self.map is None or not self.body_history.items:return None
        try:error,forecast=self.forward_position_uncertainty(self.now())
        except RuntimeError as rejection:self.block(str(rejection));return None
        if not self.uncertainty_policy.admits(error):return None
        reference=max(self.uncertainty_policy.map_pose_error_m,self.body_history.observed_position_error_max_m)
        uncertainty=self.uncertainty_policy.evidence(error,reference)
        if forecast is not None:uncertainty['causal_forward_reserve']=forecast
        return MeasuredMapGuard(self.map,self.session,self.c['vehicle_radius_m']+self.c['xy_margin_m'],
                                self.c['vertical_half_m']+self.c['vertical_margin_m'],
                                max_age_s=self.c['max_observation_age_s'],
                                extra_collision_margin_m=self.uncertainty_policy.extra(error,reference),uncertainty=uncertainty)

    def ego_envelope_clock(self):
        return int(self.get_clock().now().nanoseconds),time.monotonic_ns()

    def ego_envelope_ack(self,message):
        if self.ego_envelope is None:return
        now_ns,steady_ns=self.ego_envelope_clock()
        self.ego_envelope.accept_ack(message.data,now_ns,steady_ns)

    def update_ego_envelope(self):
        if self.ego_envelope is None:return
        now_ns,steady_ns=self.ego_envelope_clock()
        if not self.ego_envelope.expire(now_ns,steady_ns):return
        guard=self.guard()
        if guard is None:
            self.ego_envelope.reason='current actual map guard unavailable';return
        forecast=guard.uncertainty.get('causal_forward_reserve')
        if forecast is None:
            self.ego_envelope.reason='actual causal forecast unavailable';return
        try:
            if guard.epoch!=self.session:raise ValueError('guard body-session epoch mismatch')
            now_ns,steady_ns=self.ego_envelope_clock()
            deadline=min(floor_ns(guard.map.stamp+guard.max_age_s),floor_ns(forecast['valid_until']),now_ns+CEILING_NS)
            record=self.ego_envelope.issue(guard.extra,now_ns,deadline,steady_ns)
        except (ValueError,TypeError,OverflowError) as error:
            self.ego_envelope.reject('cannot issue current envelope: '+str(error));return
        self.ego_envelope_pub.publish(String(data=record.wire()))

    def ego_goal_ready(self):
        if self.ego_envelope is None:return True
        guard=self.guard()
        if guard is None:
            self.ego_envelope.goal_wait('current actual map guard unavailable');self.block('EGO envelope: current actual map guard unavailable');return False
        now_ns,steady_ns=self.ego_envelope_clock()
        try:ready=guard.epoch==self.session and self.ego_envelope.ready(guard.extra,now_ns,steady_ns)
        except (ValueError,TypeError):ready=False
        if not ready:
            self.ego_envelope.goal_wait();self.block('EGO envelope: '+self.ego_envelope.reason)
        return ready

    def publish_goal(self,p,reason):
        if not self.ego_goal_ready():self.brake(replan=False);return False
        if self.ego_envelope is not None:
            reason=dict(reason) if isinstance(reason,dict) else {'source':reason}
            self.ego_latest_goal_ack=dict(self.ego_envelope.latest_ack.metadata(),goal_publish_ns=self.ego_envelope_clock()[0],kind='new_goal')
            reason['ego_envelope_ack']=dict(self.ego_latest_goal_ack)
        super().publish_goal(p,reason)
        return True

    def replan_if_held(self,brake_on_wait=True):
        if self.ego_envelope is None:return super().replan_if_held()
        # Preserve inherited eligibility before gating this actual resend.
        if self.brake_hold is not None and (np.linalg.norm(self.body_velocity)>.05 or np.linalg.norm(self.body-self.brake_hold)>.05):return
        if self.map is None or self.now()-self.map.stamp>self.c['max_observation_age_s']:return
        if not self.goals or self.now()-self.last_replan<=1.:return
        if not self.ego_goal_ready():
            if brake_on_wait:self.brake(replan=False)
            return False
        self.ego_latest_goal_ack=dict(self.ego_envelope.latest_ack.metadata(),goal_publish_ns=self.ego_envelope_clock()[0],kind='held_goal_resend')
        super().replan_if_held()
        return True

    def brake(self,replan=True):
        if self.ego_envelope is None:return super().brake(replan=replan)
        # Same inherited brake state/command; avoid recursive resend attempts.
        if self.brake_hold is None:
            self.brake_hold=self.body.copy();self.waiting_trajectory_id=self.last_received_trajectory_id;self.reference=None
        if replan:self.replan_if_held(brake_on_wait=False)
        self.send(self.brake_hold,self.yaw)

    def launch_certificate(self,now):
        if not self.body_history.items or self.home_pose_error_m is None:raise RuntimeError('Measured startup/reference uncertainty required')
        error,forecast=self.forward_position_uncertainty(now)
        support=self.launch_support(now,error) if self.options.complete_task else None
        evidence=self.uncertainty_policy.evidence(error,self.home_pose_error_m,include_map_geometry=False)
        if forecast is not None:evidence['causal_forward_reserve']=forecast
        return LaunchCertificate(now,self.session,self.home,self.c['vehicle_radius_m']+self.c['xy_margin_m'],
                                 self.c['vertical_half_m']+self.c['vertical_margin_m'],
                                 self.c['operator_verified_launch_halfwidth_m'],self.c['operator_verified_launch_height_m'],
                                 self.uncertainty_policy.extra(error,self.home_pose_error_m,include_map_geometry=False),evidence,support)

    def launch_support(self,now,forecast_error):
        from task_native_contact import MechanicalStartup
        rs=self.current_range;cal=self.foot
        report={'checked_at':now,'forecast_error_m':forecast_error,'reason':None,
                'range_age_s':None if rs is None else now-rs.stamp,
                'range_tracking_valid':None if rs is None else rs.tracking_valid,
                'range_valid':None if rs is None else rs.range_valid,
                'native_status_age_s':None if 'status' not in self.native.records else now-self.native.records['status'][1],
                'native_land_age_s':None if 'land' not in self.native.records else now-self.native.records['land'][1]}
        self.last_launch_support_check=report
        def reject(reason):report['reason']=reason;return None
        if not isinstance(cal,MechanicalStartup):return reject('typed_startup_missing')
        if cal.epoch!=self.session:return reject('startup_epoch_mismatch')
        if rs is None:return reject('matched_range_missing')
        if not rs.range_valid:return reject('range_invalid')
        if not rs.tracking_valid:return reject('tracking_was_invalid_when_range_pose_created')
        if rs.epoch!=self.session:return reject('range_epoch_mismatch')
        if not 0<=now-rs.stamp<=.15:return reject('matched_range_age_outside_0_to_150ms')
        if self.native.flags(now,VehicleStatus.ARMING_STATE_ARMED) is None:return reject('native_flags_unavailable')
        R=np.asarray(rs.local_from_body_rotation);report.update(body_up_z=float(R[2,2]),tof_ray_z=float((R@self.tof_axis)[2]))
        if R[2,2]<math.cos(math.radians(5)):return reject('body_tilt_over_5deg')
        if (R@self.tof_axis)[2]>-math.cos(math.radians(8)):return reject('tof_ray_not_downward_within_8deg')
        clearance=-float((R@(TOF_IN_BODY+self.tof_axis*rs.range_m))[2])-cal.offset_m*R[2,2]
        observed_rise=float(vector(rs.position_local_m)[2]-self.home[2])
        enclosure=forecast_error+self.home_pose_error_m+self.options.tof_error_m+self.options.tof_extrinsic_error_m+cal.error_m
        report.update(clearance_m=clearance,observed_rise_m=observed_rise,plane_enclosure_m=enclosure)
        if abs(observed_rise-clearance)>enclosure:return reject('original_plane_and_current_tof_disagree')
        if clearance < -enclosure:return reject('negative_clearance_outside_enclosure')
        report['reason']='accepted'
        # Only the original measured start's narrow vertical contact column.
        # This is support contact, never a claim that below-floor air is free.
        column=forecast_error+self.home_pose_error_m+.04
        model='original narrow contact column'
        if self.options.allow_conditional_free and self.c.get('geometric_free_rays',False):
            # Explicit static continuous supporting-floor assumption within the
            # already declared local launch box. Uncertainty ERODES its domain.
            extra=self.uncertainty_policy.extra(forecast_error,self.home_pose_error_m,include_map_geometry=False)
            column=max(0.,float(self.c['operator_verified_launch_halfwidth_m']-self.c['vehicle_radius_m']-self.c['xy_margin_m']-max(extra[:2])))
            model='conditional static continuous launch floor inside declared local box; actual ToF agreement required'
        return dict(kind='original_mechanical_home_ToF_support_contact',stamp=rs.stamp,
                    geometry_digest=cal.geometry_digest,foot_clearance_m=clearance,
                    body_lower_enclosure_z_m=float(self.home[2]-enclosure-.015),
                    center_xy_enclosure_m=float(column),support_model_assumption=model,
                    plane_consistency_enclosure_m=enclosure,global_map_changed=False)

    def launch_height(self,now,desired):
        cert=self.launch_certificate(now)
        # Keep arrival and the .1m/s launch stopping tail below the SAME prior
        # full-body upper bound; never enlarge the operator's verified box.
        stopping=.1*(self.c['max_observation_age_s']+.1/(2*self.c['max_acceleration_mps2']))
        ceiling=cert.height-cert.halfheight-cert.extra[2]-.05-stopping
        height=min(float(desired),ceiling)
        if height<.05:raise RuntimeError('Declared launch box cannot contain forecast uncertainty and stopping volume')
        return height

    def request(self,command,params=(),interval=1.):
        now=self.now();old=self.command_requests.get(command)
        denied=(VehicleCommandAck.VEHICLE_CMD_RESULT_DENIED,VehicleCommandAck.VEHICLE_CMD_RESULT_UNSUPPORTED,
                VehicleCommandAck.VEHICLE_CMD_RESULT_FAILED,VehicleCommandAck.VEHICLE_CMD_RESULT_CANCELLED)
        if old and old[1]==tuple(params) and self.native.command_rejected(command,old[0],denied):raise RuntimeError('PX4 command rejected: '+str(command))
        if old and old[1]==tuple(params) and now-old[0]<interval:return
        fields={f'param{i+1}':float(v) for i,v in enumerate(params)}
        m=VehicleCommand(timestamp=int(now*1e6),command=command,target_system=1,target_component=1,
                         source_system=250,source_component=1,from_external=True,**fields)
        self.command_pub.publish(m)
        if command==VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM and params and params[0]==1.:
            self.owns_vehicle=True;self.arm_request_pending=True
        self.command_requests[command]=(now,tuple(params),self.native.sequence);self.event('command_requested',command=command,params=params)

    def send(self,target,yaw,velocity=None,acceleration=None):
        actual_status=self.native.get('status',self.now(),.75)
        current_local=self.native.get('local',self.now(),.35)
        if current_local is None:raise RuntimeError('Fresh native local position required for setpoint frame conversion')
        self.px=current_local
        if self.control_owner=='LAND' or (self.armed and actual_status is not None and actual_status.nav_state==VehicleStatus.NAVIGATION_STATE_AUTO_LAND):
            raise RuntimeError('Offboard setpoint forbidden while native LAND owns control')
        now=self.now();target=vector(target);cert=None
        if self.bootstrap!='READY' and self.engine is not None:
            cert=self.launch_certificate(now)
            speed=np.linalg.norm(self.body_velocity)
            stop=self.body+self.body_velocity*(self.c['max_observation_age_s']+speed/(2*self.c['max_acceleration_mps2']))
            if not cert.covers_motion(self.body,target,now,self.session) or not cert.covers_motion(self.body,stop,now,self.session):
                self.event('startup_envelope_rejected',certificate=cert.as_dict(),body=self.body.tolist(),target=target.tolist(),stop=stop.tolist(),range_feedback=None if self.current_range is None else vars(self.current_range))
                raise RuntimeError('Startup command/stopping projection leaves declared local launch envelope')
        elif self.last_intent is not None and getattr(self.last_intent.certificate,'mode',None) in ('descent','liftoff'):
            cert=self.last_intent.certificate
            if not cert.covers_motion(self.body,target,now,self.session):raise RuntimeError('Contact offboard step leaves measured support certificate')
        else:
            guard=self.guard()
            if guard is None:raise RuntimeError('No measured map for offboard control')
            v=np.zeros(3) if velocity is None else vector(velocity);speed=np.linalg.norm(self.body_velocity)
            stop=self.body+self.body_velocity*(self.c['max_observation_age_s']+speed/(2*self.c['max_acceleration_mps2']))
            paths=[(self.body,target),(self.body,target+v*.3),(self.body,stop)]
            path_certificates=[guard.certify([a,b],now,self.session,'actual offboard and stopping') for a,b in paths]
            if any(c is None for c in path_certificates):
                self.event('offboard_volume_rejected',map_stamp=guard.map.stamp,checked_at=now,
                           map_age_s=now-guard.map.stamp,allowed_age_s=guard.max_age_s,
                           paths=[(a.tolist(),b.tolist()) for a,b in paths],
                           geometrically_free=[guard.map.segment(a,b,extra_margin=guard.extra) for a,b in paths],
                           extra_collision_margin_m=guard.extra,uncertainty=guard.uncertainty)
                # A rejected candidate must not terminate a safely stoppable
                # episode. Reuse the existing fixed hold, with a NEW certificate
                # for the actual published brake; never certify the bad target.
                body_safe=guard.certify([self.body,self.body],now,self.session,'current body before candidate brake')
                if not getattr(self,'_candidate_brake_active',False) and path_certificates[2] is not None and body_safe is not None:
                    self.event('candidate_reference_brake',rejected_target=target.tolist(),waiting_after_trajectory_id=self.last_received_trajectory_id)
                    self.safety_vetoes+=1;self.veto_reasons['candidate_reference_volume']+=1
                    self.brake_hold=None # Latch the just-checked current BODY, not an older hold.
                    self._candidate_brake_active=True
                    try:self.brake(replan=False)
                    finally:self._candidate_brake_active=False
                    return
                raise RuntimeError('Actual offboard reference/stopping body volume is not observed free')
            cert=ControlCertificate(guard,paths,now)
        self.ledger.add(cert)
        super().send(target,yaw,velocity,acceleration)

    def freeze_ok(self):
        if not self.freeze_required:return True
        if self.now()-self.last_freeze>.15:self.freeze_pub.publish(Bool(data=True));self.last_freeze=self.now()
        confirmed=bool(self.health and self.health.get('global_height_frozen',False) and self.now()-self.health.get('stamp',-99)<.25)
        if confirmed and self.height_freeze_ack_at is None:self.height_freeze_ack_at=self.now()
        return confirmed

    def native_hold(self,reason):
        self._turn_hold=None
        self.block(reason)
        if self.manual_handoff:return
        self.request(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,(1.,4.,3.))
        status=self.native.get('status',self.now(),.75)
        self.native_handoff=bool(status is not None and status.nav_state==VehicleStatus.NAVIGATION_STATE_AUTO_LOITER)
        self.control_owner='NATIVE_HOLD_CONFIRMED' if self.native_handoff else 'NATIVE_HOLD_REQUESTED'
        # Keep OFFBOARD alive only if it is STILL the confirmed actual mode.
        # No conflicting offboard stream while requesting native hold from LAND.
        local=self.native.get('local',self.now(),.35)
        if status is not None and status.nav_state==VehicleStatus.NAVIGATION_STATE_OFFBOARD and local is not None and local.xy_valid and local.z_valid:
            counters=tuple(getattr(local,k,None) for k in ('xy_reset_counter','z_reset_counter','heading_reset_counter'))
            if self.emergency_hold_ned is None or counters!=self.emergency_native_counters:
                self.emergency_hold_ned=([float(local.x),float(local.y),float(local.z)],float(local.heading))
                self.emergency_native_counters=counters
            p,yaw=self.emergency_hold_ned;now=int(self.now()*1e6)
            yaw=self.yaw_limiter.update(yaw,float(local.heading),self.now())
            self.mode.publish(OffboardControlMode(timestamp=now,position=True))
            self.traj.publish(TrajectorySetpoint(timestamp=now,position=p,velocity=[math.nan]*3,acceleration=[math.nan]*3,jerk=[math.nan]*3,yaw=yaw,yawspeed=math.nan))

    def contact_context(self,rs):
        if self.engine is None or self.foot is None:return None
        stage=self.engine.stage
        if self.options.complete_task:
            from task_native_contact import NativeContactContext,polygon_margin
            if stage not in ('DOCK','REARM','HOME'):return None
            target=self.engine.target if stage=='DOCK' else self.engine.dock if stage=='REARM' else None
            candidates=[p for p in self.native_supports.values() if p.epoch==self.session and 0<=self.now()-p.stamp<=self.options.retained_support_max_age_s]
            if stage=='HOME':candidates=[p for p in candidates if p.role=='home' and p.marker_kind=='own_launch_plane']
            elif target is None:return None
            else:candidates=[p for p in candidates if p.target_id==target.id]
            if not candidates:return None
            rank=lambda p:polygon_margin(p.landing_boundary_xy,rs.position_local_m[:2])-p.boundary_error_m-p.error_m
            patch=max(candidates,key=rank)
            return NativeContactContext(self.native_contact,patch,self.foot,rs,self.body_history,self.native)
        if stage not in ('DOCK','REARM','HOME'):return None
        anchor=self.engine.target if stage=='DOCK' else self.engine.dock if stage=='REARM' else None
        candidates=list(self.supports.values())
        if stage=='HOME':candidates=[p for p in candidates if p.marker_kind=='own_launch_plane']
        elif anchor is not None:candidates=[p for p in candidates if np.linalg.norm(np.asarray(p.center_local_m)-anchor.center)<anchor.error_m+.03]
        else:return None
        if not candidates:return None
        patch=max(candidates,key=lambda p:p.stamp)
        return LandingContext(self.contact,patch,self.foot,rs)

    def feedback_once(self):
        rs=self.current_range
        if self.engine is None or rs is None or rs.stamp<=self.last_feedback:return None
        if self.now()-rs.stamp>.15 or not self.healthy():return None
        flags=self.native.flags(self.now(),VehicleStatus.ARMING_STATE_ARMED)
        if flags is None:return None
        rs=replace(rs,armed=flags[0],landed=flags[1],tracking_valid=True)
        guard=self.guard()
        if guard is None:return None
        state=self.body_history.sample(int(round(rs.stamp*1e9)))
        if state is None:return None
        if self.engine.last_feedback is not None:
            self.engine.pending=self.ledger.interval(self.engine.last_feedback.stamp,rs.stamp,self.body_history,self.session)
        feedback=Feedback(rs.stamp,state.position,self.session,flags[0],flags[1],float(np.linalg.norm(state.velocity)),
                          body_up_local=state.rotation.as_matrix()[:,2],position_error_m=state.error_m,
                          uncertainty_limit_m=self.uncertainty_policy.admission_limit_m)
        context=self.contact_context(rs)
        tracks=self.latest_tracks if self.bootstrap=='READY' else []
        if not self.options.complete_task and self.engine.stage=='DOCK' and self.engine.phase=='SEARCH':
            # Acquire the complete raw support patch BEFORE flying over a pad
            # that the fixed forward/down camera can no longer see underneath.
            tracks=[t for t in tracks if any(p.target_id==t.id and 0<=self.now()-p.stamp<.75 and
                     self.height_freeze_ack_at is not None and p.stamp>=self.height_freeze_ack_at-1e-9 for p in self.supports.values())]
            if not tracks:self.block('no fresh strict dense support patch after height-freeze acknowledgement')
        intent=self.engine.tick(feedback,tracks,guard,self.now(),
                                self.forward if self.bootstrap=='READY' else None,context)
        self.last_feedback=rs.stamp;self.last_intent=intent
        if self.engine.fault:
            self.event('motion_audit_failed',bootstrap=self.bootstrap,fault=self.engine.fault,audit=self.ledger.last_audit)
            raise RuntimeError(self.engine.fault)
        return feedback,intent,context

    def ensure_offboard_ground(self):
        now=self.now();flags=self.native.flags(now,VehicleStatus.ARMING_STATE_ARMED)
        status=self.native.get('status',now,.75)
        if flags is None or status is None:
            self.block('waiting for fresh native status/landed confirmation before arming');return False
        if not flags[1]:raise RuntimeError('Native landed confirmation required before arming')
        if self.mode_prime_started is None:self.mode_prime_started=now
        self.native_land_requested=False;self.control_owner='OFFBOARD_PRIME'
        self.send(self.body.copy(),self.yaw)
        if now-self.mode_prime_started<1.2:return False
        if status.nav_state!=VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.request(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,(1.,6.));return False
        if not getattr(status,'pre_flight_checks_pass',False):
            self.block('native pre_flight_checks_pass is not true');return False
        self.control_owner='OFFBOARD';return True

    def launch_step(self,goal):
        delta=vector(goal)-self.body
        # Bootstrap uses the complete certified local-box goal, as the native
        # PX4 position controller expects; contact liftoff retains a short tube.
        step=vector(goal) if self.bootstrap!='READY' else self.body+delta*min(1.,.035/max(np.linalg.norm(delta),1e-9))
        self.send(step,self.yaw,delta*min(.10,1.)/max(np.linalg.norm(delta),1e-9))

    def bootstrap_once(self):
        if self.options.observe_only:return
        if self.mechanical_foot_observer is not None and not self.options.complete_task:
            self.block('mechanical foot observe-only: native typed contact/bootstrap not integrated');return
        now=self.now();flags=self.native.flags(now,VehicleStatus.ARMING_STATE_ARMED)
        if self.bootstrap=='WAIT_READY':
            if flags is not None and flags[0]:raise RuntimeError('Startup found already armed vehicle; no control ownership claimed')
            if not self.healthy() or self.map is None or flags is None or not flags[1]:return
            if self.options.complete_task:
                if self.joiner.last<0 or not 0<=now-self.joiner.last*1e-9<=.6 or self.latest_direct_depth_pixels<64:
                    self.block('waiting for actual exact task camera/depth/BODY frame before complete-task arming');return
                from task_native_contact import MechanicalStartup
                report=self.mechanical_foot_report()
                if not report['observed_dwell_ready'] or not report['known_home_body_anchor']:
                    self.block('waiting for actual mechanical/native/home-relative startup dwell: '+str(report['last_rejection']));return
                cal=MechanicalStartup.from_observer(report,self.session,self.calibration_id,now)
                self.foot=cal;self.home=np.asarray(cal.home_body_anchor['position_local_m']);self.yaw=float(self.q.as_euler('xyz')[2])
                self.home_pose_error_m=cal.home_body_anchor['position_three_sigma_proxy_m']
            else:
                cal=self.foot_calibrator.result(now)
                if cal is None:self.block('waiting for actual stationary landed ToF calibration');return
                self.foot=cal;self.home=np.asarray(cal.resting_body_local_m);self.yaw=float(self.q.as_euler('xyz')[2])
                self.home_pose_error_m=max(s.error_m for s in self.body_history.items if cal.first_stamp<=s.stamp<=cal.last_stamp)
            # Reuse the completed measured stationary/native-ground dwell.
            # This anchor stays fixed across takeoff, landing and rearming.
            if self.control_alignment.rotation is None:
                self.establish_control_alignment(self.session,stationary_confirmed=True)
                self.event('control_frame_anchored',anchor=self.control_alignment.anchor)
            limits=MissionLimits(self.c['vehicle_radius_m']+self.c['xy_margin_m'],self.c['vertical_half_m']+self.c['vertical_margin_m'],
                                 max_speed_mps=self.c['max_speed_mps'],uncertainty_policy=self.uncertainty_policy,
                                 home_pose_error_m=self.home_pose_error_m)
            if self.options.complete_task:
                from task_mechanical_executive import MechanicalTaskExecutive
                self.engine=MechanicalTaskExecutive(limits,self.home,self.session,cal)
            else:
                self.engine=TaskExecutive(limits,self.home,self.session);self.engine.install_foot_calibration(cal)
            self.bootstrap='ARMING';self.bootstrap_phase_started=now;self.event('measured_startup_ready',calibration=cal.as_dict())
        if self.bootstrap=='ARMING':
            if now-self.bootstrap_phase_started>15.:raise RuntimeError('Offboard/arming native confirmation timed out')
            if flags is not None and flags[0]:
                arm=self.command_requests.get(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM)
                if arm is None or arm[1]!=(1.,):raise RuntimeError('Vehicle armed externally; automatic takeover refused')
                self.owns_vehicle=True;self.event('native_arm_confirmed',status_source_us=int(self.status.timestamp),request_stamp=arm[0]);self.run_start=now;self.bootstrap='ASCEND';self.bootstrap_phase_started=now;return
            if self.ensure_offboard_ground():self.request(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,(1.,),interval=.5)
            return
        if self.bootstrap in ('ASCEND','DESCEND'):
            if now-self.bootstrap_phase_started>25.:raise RuntimeError('Declared-box launch/settling timed out')
            desired=self.c['takeoff_survey_height_m'] if self.bootstrap=='ASCEND' else self.c['takeoff_height_m']
            height=self.launch_height(now,desired) if self.options.complete_task else desired
            goal=self.home+[0,0,height]
            if np.linalg.norm(self.body-goal)<.05 and np.linalg.norm(self.body_velocity)<.08:
                self.bootstrap='HIGH_ARC' if self.bootstrap=='ASCEND' else 'LOW_ARC';self.bootstrap_phase_started=now
                self.bootstrap_arc_hold=self.body.copy()
            self.launch_step(goal);return
        if self.bootstrap in ('HIGH_ARC','LOW_ARC'):
            if now-self.bootstrap_phase_started>4.:
                self.bootstrap='DESCEND' if self.bootstrap=='HIGH_ARC' else 'READY';self.bootstrap_phase_started=now
                self.bootstrap_arc_hold=None
                self.yaw=float(self.q.as_euler('xyz')[2]);self.visited.append(self.body.copy());return
            current=float(self.q.as_euler('xyz')[2]);age=self.health.get('visual_update_age',99.)
            look=current if age>.35 else self.yaw+.30*math.sin((now-self.bootstrap_phase_started)*math.pi/2)
            self.send(self.bootstrap_arc_hold,look);return

    def frontier_view_yaw(self,goal):
        delta=np.asarray(goal)-self.body
        current=float(self.q.as_euler('xyz')[2])
        if self.options.complete_task and self.health.get('accepted_visual_update_features',0)>=4 and self.health.get('visual_update_age',99.)<=.20:
            guard=self.guard()
            known=None if guard is None else guard.certify([self.body,goal],self.now(),self.session,'retain constrained view for known-space translation')
            if known is not None:
                self.event('frontier_retain_tracked_heading',yaw=current,goal=np.asarray(goal).tolist(),certificate=known.as_dict())
                return current
        if np.linalg.norm(delta[:2])<.01:return float(self.q.as_euler('xyz')[2])
        optical_forward=self.body_from_camera[:3,2]
        mount_bearing=math.atan2(optical_forward[1],optical_forward[0])
        desired=wrap(math.atan2(delta[1],delta[0])-mount_bearing)
        current=float(self.q.as_euler('xyz')[2])
        allowed=lambda yaw:bearing_visible(math.atan2(delta[1],delta[0]),yaw,self.q,
                    Rotation.from_matrix(self.body_from_camera[:3,:3]),self.K,self.resolution,margin_px=20.)
        try:selected=self.views.choose(self.now(),self.body,desired,current,allowed=allowed)
        except TypeError:selected=self.views.choose(self.now(),self.body,desired,current)
        # Also enforce locally for an older ViewMemory API or a bad hint.
        if abs(wrap(selected-desired))>self.frontier_bearing_bound or not allowed(selected):selected=desired
        if not allowed(selected):
            self.block('frontier bearing outside calibrated camera view at measured tilt');return None
        self.event('frontier_view_cone',desired_body_yaw=desired,selected_body_yaw=selected,
                   max_bearing_error_rad=self.frontier_bearing_bound,mount_bearing_rad=mount_bearing)
        return float(selected)

    def observation_initial_side(self,now):
        current=float(self.q.as_euler('xyz')[2]);mount=self.body_from_camera[:3,2]
        bearing=current+math.atan2(mount[1],mount[0]);hint=self.guideline
        if hint and 0<=now-hint.get('stamp',-99)<.4:
            for segment in hint.get('segments',[]):
                ends=np.asarray(segment.get('endpoints',[]),float)
                if ends.shape!=(2,3) or not np.isfinite(ends).all() or segment.get('support_points',0)<8 or segment.get('source')!='grayscale line with measured stereo floor support':continue
                delta=ends[1,:2]-ends[0,:2]
                if np.linalg.norm(delta)<.25:continue
                line=math.atan2(delta[1],delta[0]);turn=min((wrap(line-bearing),wrap(line+math.pi-bearing)),key=abs)
                if turn!=0:return (1 if turn>0 else -1),dict(source='fresh measured floor-line tangent',stamp=hint['stamp'],line_bearing_world_rad=line,turn_from_optical_axis_rad=turn,support_points=segment['support_points'])
        remembered=self.views.choose(now,self.body,current,current)
        if wrap(remembered-current)!=0:return (1 if wrap(remembered-current)>0 else -1),dict(source='previously measured useful view',yaw=remembered)
        side=getattr(self,'_observation_next_side',1);self._observation_next_side=-side
        return side,dict(source='symmetric alternating probe; no measured side preference')

    def finish_frontier_observation(self,reason):
        observation=getattr(self,'_frontier_observation',None)
        if observation is None:return
        self.yaw=float(self.q.as_euler('xyz')[2]);self._frontier_observation=None
        self.reference=None;self.waiting_trajectory_id=self.last_received_trajectory_id
        self.event('frontier_observation_finished',reason=reason,elapsed_s=self.now()-observation['started'])

    def step_frontier_observation(self,now):
        observation=getattr(self,'_frontier_observation',None)
        if observation is None:return False
        if not self.healthy() or self.health.get('visual_update_age',99.)>.35:
            healthy=bool(self.healthy());self.finish_frontier_observation('visual health gate')
            if healthy:self.brake(replan=False)
            return True
        if now-observation['started']>=8.:
            self.finish_frontier_observation('bounded observation completed');return False
        if observation['phase']=='BRAKE':
            if np.linalg.norm(self.body_velocity)>.05 or np.linalg.norm(self.body-observation['hold'])>.05:
                self.brake(replan=False)
                if not np.array_equal(self.brake_hold,observation['hold']):self.finish_frontier_observation('fixed hold replaced by current safety check')
                return True
            observation['phase']='SCAN';observation['yaw']=float(self.q.as_euler('xyz')[2])
            self.event('frontier_observation_scan',yaw=observation['yaw'],side=observation['side'],relative_arc_rad=[-.60,.75])
        current=float(self.q.as_euler('xyz')[2]);side=observation['side'];relative=side*wrap(current-observation['yaw'])
        if not observation['returning'] and relative>=.75-.10:observation['returning']=True
        if observation['returning'] and relative<=-.60+.10:
            self.finish_frontier_observation('bounded arc completed');return False
        direction=-1 if observation['returning'] else 1
        self.yaw=wrap(observation['yaw']+side*float(np.clip(relative+direction*.18,-.60,.75)))
        self.brake(replan=False) # Existing send certifies every actual fixed-hold request.
        if not np.array_equal(self.brake_hold,observation['hold']):self.finish_frontier_observation('fixed hold replaced by current safety check')
        return True

    def request_frontier(self,now,guard):
        if guard is None or now-self.last_frontier_request<=.5:return
        self.frontier_level+=1;self.last_frontier_request=now
        hints=self.guideline['segments'] if self.guideline and now-self.guideline['stamp']<.4 else None
        self.frontier_worker.submit('frontier',(self.map,self.body.copy(),self.home.copy(),list(self.visited),float(self.q.as_euler('xyz')[2]),self.frontier_level,tuple(np.broadcast_to(np.asarray(guard.extra,float),(3,))),hints))

    def explore(self,feedback):
        now=self.now();guard=self.guard();missing=False
        # The continuous measured guideline is the primary route reference.
        # Approaching its supporting axis still needs the ordinary map proof;
        # this does not turn the connector or unseen space into free cells.
        if self.options.complete_task and guard is not None and self.engine.phase=='SEARCH':
            forward=self.q.apply(self.body_from_camera[:3,2])[:2]
            line=short_forward_reference(self.guideline,self.body,forward,now=now,
                                         lookahead_m=.30,max_step_m=.20,max_axis_turn_rad=self.line_axis_turn_bound)
            if line is not None:
                authorized=self.engine.authorize_external_step(line['reference'],feedback,guard,now)
                if authorized.action=='move':
                    chosen_yaw=self.frontier_view_yaw(authorized.goal)
                    if chosen_yaw is not None:
                        self.finish_frontier_observation('certified measured guideline reference available')
                        self.active_goal=None;self.yaw=chosen_yaw
                        if now-getattr(self,'_line_event_stamp',-99.)>=.5:
                            self._line_event_stamp=now
                            self.event('guideline_reference',reference=line,certificate=authorized.certificate.as_dict())
                        self.execute_move(authorized.goal,'measured guideline reference',turn=True);return
        if self.active_goal is not None and (np.linalg.norm(self.body-self.active_goal)<.055 or now-self.last_replan>8.):
            if np.linalg.norm(self.body-self.active_goal)<.055 and not getattr(self,'_active_frontier_fine',False):self._frontier_observation_spent=False
            self.visited.append(self.body.copy());self.active_goal=None
        observing=getattr(self,'_frontier_observation',None) is not None
        if self.active_goal is None:self._turn_hold=None # Preserve fixed-yaw-hold episode reset.
        if self.active_goal is None or observing:
            result=self.frontier_result
            if result is not None:
                self.frontier_result=None
                if now-result['stamp']<.6 and np.linalg.norm(self.body-result['origin'])<.08:
                    if result['candidate'] is None:missing=True
                    else:
                        self.active_goal=np.asarray(result['candidate'][0])
                        self._active_frontier_fine=bool(result['candidate'][1].get('fine_observed_step',False))
            if self.active_goal is None or observing:self.request_frontier(now,guard)
        if observing and self.active_goal is not None and not getattr(self,'_active_frontier_fine',False):
            candidate=self.engine.authorize_external_step(self.active_goal,feedback,guard,now)
            if candidate.action=='move' and self.frontier_view_yaw(candidate.goal) is not None:
                self.finish_frontier_observation('ordinary certified frontier available');observing=False
        needs_observation=missing or (self.active_goal is not None and getattr(self,'_active_frontier_fine',False))
        if self.options.complete_task and needs_observation and not observing and not getattr(self,'_frontier_observation_spent',False) and self.healthy() and self.health.get('visual_update_age',99.)<=.35:
            side,source=self.observation_initial_side(now);self._frontier_observation_spent=True
            self._frontier_observation=dict(started=now,hold=self.body.copy(),phase='BRAKE',side=side,returning=False)
            self.yaw=float(self.q.as_euler('xyz')[2]);self.brake_hold=None;self._turn_hold=None
            self.event('frontier_observation_started',reason='fine observed step' if self.active_goal is not None else 'no frontier',side=side,side_evidence=source,hold=self.body.tolist())
            self.brake(replan=False);return
        if self.step_frontier_observation(now):return
        if self.active_goal is None:self.block('no current certified online frontier');self.brake(replan=False);return
        authorized=self.engine.authorize_external_step(self.active_goal,feedback,guard,now)
        if authorized.action!='move':self.active_goal=None;self.brake(replan=False);return
        chosen_yaw=self.frontier_view_yaw(authorized.goal)
        if chosen_yaw is None:
            self.active_goal=None;self.brake(replan=False);return
        self.yaw=chosen_yaw
        self.execute_move(authorized.goal,'online frontier',turn=True)

    def execute_move(self,goal,reason,turn=False):
        goal=np.asarray(goal);current=float(self.q.as_euler('xyz')[2])
        if turn and abs(wrap(self.yaw-current))>.25:
            # Task feedback may be slower than control; keep this point across
            # intervening hold ticks until the frontier/intent or task changes.
            key=None if self.engine is None else (self.engine.stage,self.engine.phase)
            if self._turn_hold is None or self._turn_hold[0]!=key:
                self._turn_hold=(key,self.body.copy())
            self.send(self._turn_hold[1],current+float(np.clip(wrap(self.yaw-current),-.12,.12)));return
        self._turn_hold=None
        publish=not self.goals
        if self.goals:
            previous=np.asarray(self.goals[-1]['position']);old_delta=previous-self.body;new_delta=goal-self.body
            reached=np.linalg.norm(old_delta)<.055
            aligned=float(old_delta@new_delta)>0 and np.linalg.norm(np.cross(old_delta,new_delta))<=.15*max(np.linalg.norm(old_delta)*np.linalg.norm(new_delta),1e-9)
            if not aligned and not reached:
                if np.linalg.norm(self.body_velocity)>.05:self.brake();return
                publish=True
            elif reached:publish=True
            # Keep an aligned in-flight short goal until it is reached. Do not
            # cancel EGO every time the executive recomputes a receding target.
        if publish and self.publish_goal(goal,{'source':reason,'competition_relative_target':True}) is False:return
        self.follow()

    def dispatch(self,feedback,intent,context):
        action=intent.action;status=self.native.get('status',self.now(),.75)
        if action!='explore':self.finish_frontier_observation('task intent '+action)
        if action not in ('move','explore'):self._turn_hold=None
        if status is None:self.native_hold('stale VehicleStatus');return
        if action=='complete':
            if self.options.complete_task and (feedback.armed or not feedback.landed or self.engine.stage!='COMPLETE'):raise RuntimeError('Actual native home disarm/landed completion required')
            self.complete=True;return
        if action=='land':
            if context is None or getattr(intent.certificate,'mode',None)!='descent':self.native_hold('native LAND requires strict measured contact certificate');return
            self.ledger.add(intent.certificate)
            if not self.native_land_requested:
                self.mode_prime_started=None;self.brake_hold=self.body.copy();self.reference=None
            self.native_land_requested=True
            self.control_owner='LAND' if status.nav_state==VehicleStatus.NAVIGATION_STATE_AUTO_LAND else 'LAND_REQUESTED'
            self.request(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            if self.control_owner=='LAND_REQUESTED':self.brake()
            return
        if action in ('hold','observe') and (self.native_land_requested or status.nav_state==VehicleStatus.NAVIGATION_STATE_AUTO_LAND) and feedback.armed and not feedback.landed:
            self.native_hold('native contact veto: '+intent.reason);raise RuntimeError('Native LAND aborted on task/contact veto')
        if action=='disarm':
            if self.options.complete_task and (intent.certificate is None or not intent.certificate.grounded):self.native_hold('typed grounded certificate missing before disarm');return
            if not feedback.landed:self.native_hold('refused airborne disarm');return
            self.request(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,(0.,));return
        if action=='arm':
            if context is None:self.block('no contact proof for rearming');return
            # A calibrated ground-hold uses the contact certificate explicitly.
            cert=(context.adapter.certify(self.guard(),context.support,context.calibration,context.range_pose,self.now(),body_history=context.body_history,native=context.native) if self.options.complete_task else
                  context.adapter.certify(self.guard(),context.support,context.calibration,context.range_pose,self.now()))
            if cert is None:self.block('ground contact proof unavailable before rearm');return
            self.last_intent=replace(intent,certificate=cert)
            if self.ensure_offboard_ground():self.request(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,(1.,),interval=.5)
            return
        if action=='takeoff':
            arm=self.command_requests.get(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM)
            own_arm=arm is not None and arm[1]==(1.,) and arm[0]>=self.engine.completed.get('DOCK',{}).get('stamp',float('inf'))
            if not feedback.armed or not own_arm or getattr(intent.certificate,'mode',None)!='liftoff':self.native_hold('invalid calibrated liftoff intent');return
            self.native_land_requested=False;self.control_owner='OFFBOARD';self.launch_step(intent.goal);return
        if self.native_land_requested or status.nav_state==VehicleStatus.NAVIGATION_STATE_AUTO_LAND:
            if feedback.landed:return
            self.native_hold('unexpected task intent while native LAND active');return
        self.control_owner='OFFBOARD'
        if action=='move':self.active_goal=None;self.execute_move(intent.goal,'measured '+intent.stage+'/'+intent.phase)
        elif action=='explore':self.explore(feedback)
        else:
            if intent.look_at is not None:
                delta=np.asarray(intent.look_at)-self.body
                if np.linalg.norm(delta[:2])>.08:self.yaw=float(self.q.as_euler('xyz')[2])+float(np.clip(wrap(math.atan2(delta[1],delta[0])-self.q.as_euler('xyz')[2]),-.10,.10))
            self.brake(replan=False)

    def control_once(self):
        self.update_ego_envelope()
        if self.cancelled:raise RuntimeError('Competition cancelled')
        if self.options.observe_only:return
        if self.mechanical_foot_observer is not None and not self.options.complete_task:raise RuntimeError('Mechanical foot dispatch remains disabled without explicit complete-task')
        if self.options.complete_task and self.engine is not None:
            health=self.home_relative_adapter_health;epoch=self.foot.home_body_anchor['home_epoch_ns']
            if not health or not health.get('valid') or not 0<=self.now()-health.get('stamp',-99)<=.35 or health.get('home_epoch_ns')!=epoch:
                self.invalidate_control_alignment('actual home-relative epoch/health continuity lost')
                if self.owns_vehicle:self.native_hold('actual home-relative epoch/health continuity lost')
                raise RuntimeError('Actual retained home reference changed or expired; native typed task stopped')
        if self.manual_handoff:raise RuntimeError('Native PX4 reports user takeover; automatic control stopped')
        if self.body_history.fault:raise RuntimeError(self.body_history.fault)
        if self.cancelled:raise RuntimeError('Competition cancelled')
        if self.engine is None and time.monotonic()-self.started_wall>self.options.initial_wait_s:
            raise RuntimeError('Startup sensor/calibration gate timed out without arming')
        if not self.healthy():
            if self.owns_vehicle:
                self.native_hold('localization health gate failed')
                raise RuntimeError('Accepted localization lost; strict mission audit stopped for native hold')
            return
        result=self.feedback_once()
        if self.owns_vehicle and hasattr(self,'run_start') and self.now()-self.run_start>self.options.mission_budget_s-20:
            raise RuntimeError('Competition budget reserved for safe native handoff')
        if self.bootstrap!='READY':self.bootstrap_once();return
        if self.now()-self.run_start>self.options.mission_budget_s-20:raise RuntimeError('Competition budget reserved for safe hold/landing')
        self.freeze_required|=self.engine.stage in ('DOCK','REARM','WALL','HOME','COMPLETE')
        if not self.freeze_ok():
            if self.options.complete_task and self.native_land_requested:
                self.native_hold('height-freeze acknowledgement lost during typed native contact');raise RuntimeError('Typed native contact freeze acknowledgement lost')
            self.block('waiting for fusion global_height_frozen acknowledgement');self.brake(replan=False);return
        if result is None:
            if self.native_land_requested:
                if self.current_range is None or self.now()-self.current_range.stamp>.15:
                    self.native_hold('fresh matched ToF/body feedback missing during LAND')
                    if self.options.complete_task:raise RuntimeError('Typed native contact current range/BODY expired')
                return
            if getattr(self,'_frontier_observation',None) is not None:
                if self.now()-self.last_feedback<.15:
                    if not self.step_frontier_observation(self.now()):self.brake(replan=False)
                else:
                    self.finish_frontier_observation('matched feedback timeout')
                    if self.owns_vehicle and self.armed:self.native_hold('matched ToF/body feedback stream timed out')
                return
            if self.last_intent is not None and self.now()-self.last_feedback<.15:
                action=self.last_intent.action
                if action in ('move','explore') and self._turn_hold is not None:
                    current=float(self.q.as_euler('xyz')[2])
                    self.send(self._turn_hold[1],current+float(np.clip(wrap(self.yaw-current),-.12,.12)))
                elif action in ('move','explore') and self.goals:self.follow()
                elif action=='takeoff':self.launch_step(self.last_intent.goal)
                elif action=='arm':self.ensure_offboard_ground()
                else:self.brake()
            elif self.owns_vehicle and self.armed:
                self.native_hold('matched ToF/body feedback stream timed out')
            return
        feedback,intent,context=result
        self.phase_set('COMPETITION_'+self.engine.stage+'_'+self.engine.phase) if self.phase!='COMPETITION_'+self.engine.stage+'_'+self.engine.phase else None
        self.dispatch(feedback,intent,context)

    def run(self):
        last=-1.
        while rclpy.ok() and not self.complete:
            self.pump(.005)
            if self.options.observe_only and time.monotonic()-self.started_wall>self.options.initial_wait_s:break
            if time.monotonic()-last>=1/self.options.direct_control_hz:
                self.control_once();last=time.monotonic()

    def write_result(self):
        report={} if self.engine is None else self.engine.report()
        flags=self.native.flags(self.now(),VehicleStatus.ARMING_STATE_ARMED)
        mechanical=self.mechanical_foot_report()
        confirmed_complete=bool(self.complete and not self.failure and flags is not None and not flags[0] and flags[1] and report.get('competition_tasks_complete'))
        report.update({'competition_tasks_complete':confirmed_complete,'mission_profile':'competition',
                       'adapter_failure':self.failure,'native_hold_handoff_confirmed':self.native_handoff,
                       'native_command_acks':self.command_acks,'arm_request_pending':self.arm_request_pending,
                       'observe_only':self.options.observe_only,'manual_handoff':self.manual_handoff,
                       'joined_frames_rejected':self.joiner.rejected,
                       'exact_join_buffers':{k:{'count':len(v),'latest_stamp_ns':None if not v else max(v)} for k,v in self.joiner.buffers.items()},'latest_exact_exposure_stamp_ns':self.joiner.last,
                       'body_history_samples':len(self.body_history.items),
                       'uncertainty_policy':vars(self.uncertainty_policy),
                       'latest_direct_depth_pixels':self.latest_direct_depth_pixels,
                       'latest_body_three_sigma_proxy_m':None if not self.body_history.items else self.body_history.items[-1].error_m,
                       'latest_body_pose_covariance':None if not self.body_history.items else self.body_history.items[-1].pose_covariance,
                       'observed_position_error_max_m':self.body_history.observed_position_error_max_m,
                       'home_pose_error_m':self.home_pose_error_m,
                       'measured_foot_calibration':None if self.foot is None else self.foot.as_dict(),
                       'complete_task_enabled':self.options.complete_task,
                       'typed_native_contact_dispatch_enabled':self.options.complete_task and not self.options.observe_only,
                       'conditional_free_enabled':self.c['geometric_free_rays'],
                       'ego_envelope_required':self.options.require_ego_envelope,
                       'ego_envelope':None if self.ego_envelope is None else self.ego_envelope.report(),
                       'ego_latest_goal_ack':self.ego_latest_goal_ack,
                       'conditional_free_assumptions':'static scene, fitted continuous supporting floor and observed vertical termination; raw occupancy veto' if self.c['geometric_free_rays'] else None,
                       'native_contact_observation':None if self.native_contact is None else self.native_contact.last_observation,
                       'foot_calibration_mode':'learned' if mechanical is None else 'mechanical-native' if self.options.complete_task else 'mechanical-observe-only',
                       'retained_native_supports':[p.as_dict() for p in self.native_supports.values()],
                       'native_contact_last_rejection':None if self.native_contact is None else self.native_contact.last_rejection,
                       'observed_resting_calibration_candidate':None if mechanical is not None or self.foot_calibrator.result(self.now()) is None else self.foot_calibrator.result(self.now()).as_dict(),
                       'observed_mechanical_foot_calibration':mechanical,
                       'observed_home_body_anchor':None if mechanical is None else mechanical['home_body_anchor'],
                       'resting_calibration_diagnostics':existing_foot_calibrator_diagnostics(self.foot_calibrator,self.now()) if mechanical is None else mechanical,
                       'sensor_error_qualification':'interface-only admission settings, not flight-qualified' if self.options.observe_only else 'requires external validated error budgets',
                       'control_owner':self.control_owner,'bootstrap':self.bootstrap,'sensor_epoch':self.session,
                       'blocked_reasons':self.blockers,'adapter_events':self.adapter_events,
                       'prior_control_audited_segments':self.ledger.audited_segments,
                       'last_launch_support_check':getattr(self,'last_launch_support_check',None),
                       'last_prior_control_audit':self.ledger.last_audit,
                       'source':'accepted body + exact measured stereo/camera pose + actual ToF + native PX4 status/landed',
                       'independent_course_scoring_included':False})
        self.output.parent.mkdir(parents=True,exist_ok=True);self.output.write_text(json.dumps(plain(report),indent=2))
        return report

    def finish(self):
        # Do not call Navigator.finish(): it would blindly LAND on failure.
        # Do not abandon an armed OFFBOARD vehicle without confirmed native
        # hold/user takeover. An unresolved handoff keeps the adapter alive and
        # writes an INCOMPLETE report while the primary operator intervenes.
        if self.owns_vehicle and not self.complete and not self.manual_handoff:
            last_report=-1.;warned=False;started=time.monotonic()
            while rclpy.ok():
                flags=self.native.flags(self.now(),VehicleStatus.ARMING_STATE_ARMED)
                arm=self.command_requests.get(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM)
                fresh_after_request=arm is None or self.native.received_sequence.get('status',-1)>arm[2]
                if flags is not None and not flags[0] and fresh_after_request:break
                rs=self.current_range
                safe_contact=False
                if flags is not None and flags[1] and rs is not None and self.foot is not None and rs.range_valid and self.now()-rs.stamp<=.15:
                    clearance=-float((rs.local_from_body_rotation@(TOF_IN_BODY+self.tof_axis*rs.range_m))[2])-self.foot.offset_m
                    safe_contact=abs(clearance)<=.02 and np.linalg.norm(rs.velocity_local_mps)<.06
                try:
                    if safe_contact:self.request(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,(0.,))
                    else:self.native_hold(self.failure or 'competition stopped')
                except RuntimeError as error:self.block(str(error))
                if self.native_handoff or self.manual_handoff:break
                if time.monotonic()-started>5. and not warned:
                    self.failure=(self.failure or '')+'; native HOLD/disarm handoff unconfirmed; adapter remains alive'
                    self.get_logger().error(self.failure);warned=True
                if self.now()-last_report>1.:self.write_result();last_report=self.now()
                self.pump(.05)
        self.write_result()
        self.task_worker.close();self.frontier_worker.close();self.worker.close()


def main():
    import argparse
    from rclpy.signals import SignalHandlerOptions
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--tof-error-m',type=float,required=True);p.add_argument('--tof-extrinsic-error-m',type=float,required=True)
    p.add_argument('--stereo-error-m',type=float,required=True);p.add_argument('--mission-budget-s',type=float,default=600.)
    p.add_argument('--require-ego-envelope',action='store_true',help='Require fresh native ACK of actual atomic guard envelope before EGO free-space goals')
    p.add_argument('--allow-conditional-free',action='store_true',help='Explicit static-scene conditional model; raw obstacles and unknown space veto')
    p.add_argument('--complete-task',action='store_true',help='Default off: explicit typed mechanical complete task/native contact')
    p.add_argument('--contact-rig-error-m',type=float,help='Explicit complete3D rig-point error over admitted support range; no implicit zero')
    p.add_argument('--retained-support-max-age-s',type=float,default=120.)
    p.add_argument('--foot-geometry',help='ROBOT mechanical geometry JSON inside config; observe-only or explicit complete-task')
    p.add_argument('--observe-only',action='store_true');p.add_argument('--land-topic',default='/fmu/out/vehicle_land_detected');p.add_argument('--ack-topic',default='/fmu/out/vehicle_command_ack')
    p.add_argument('--uncertainty-admission-m',type=float,default=.20,help='Estimator proxy admission domain, not a safety/accuracy qualification')
    p.add_argument('--map-pose-error-m',type=float,required=True,help='Explicit retained map-pose bound declaration; never silently zero')
    a=p.parse_args();assert_sitl_authorized();options=CompetitionOptions(a.tof_error_m,a.tof_extrinsic_error_m,a.stereo_error_m,
            mission_budget_s=a.mission_budget_s,native_land_topic=a.land_topic,command_ack_topic=a.ack_topic,observe_only=a.observe_only,
            uncertainty_admission_m=a.uncertainty_admission_m,map_pose_error_m=a.map_pose_error_m,foot_geometry=a.foot_geometry,complete_task=a.complete_task,contact_rig_error_m=a.contact_rig_error_m,retained_support_max_age_s=a.retained_support_max_age_s,allow_conditional_free=a.allow_conditional_free,require_ego_envelope=a.require_ego_envelope)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO);n=CompetitionNavigator(a.output,options);install_stop_handlers(n)
    try:n.run()
    except Exception as error:n.failure=str(error);n.get_logger().error(str(error))
    finally:n.finish();n.destroy_node();rclpy.shutdown()
    if n.failure:raise SystemExit(2)

if __name__=='__main__':main()
