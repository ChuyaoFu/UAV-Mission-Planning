"""Pure CAD-free competition executive. Returns intents, never arms or flies.

The caller must execute only the latest intent and brake on hold/fault, feed
accepted measured body poses, and call authorize_external_step before a frontier
step. Every motion uses a fresh full-body map certificate. The task order and
robot calibration are the only priors. No timeout or travelled distance completes
a task. Completion means all measured evidence gates below were satisfied.
"""
from dataclasses import dataclass, field,replace
from copy import deepcopy
import math
import numpy as np
from task_observations import vector,passage_views,plain
from task_safety import MeasuredMapGuard
from task_landing import rotation,FootCalibration

def rotation_of_contact_up(range_pose):
    return rotation(range_pose.local_from_body_rotation)[:,2].tolist()


@dataclass(frozen=True)
class MissionLimits:
    body_radius_m: float
    body_half_height_m: float
    foot_to_body_center_m: object = None
    max_step_m: float = .30
    max_speed_mps: float = .8
    target_age_s: float = .6
    pose_age_s: float = .2
    max_feedback_gap_s: float = .4
    arrival_m: float = .055
    hover_clearance_m: float = .30
    landing_dwell_s: float = .6
    home_tolerance_m: float = .12
    allowed_dock_markers: tuple = ('h','cross','closed_mark')
    allow_door_alternative: bool = True
    uncertainty_policy: object = None
    home_pose_error_m: float = 0.

    def __post_init__(self):
        positive=[self.body_radius_m,self.body_half_height_m,
                  self.max_step_m,self.max_speed_mps,self.target_age_s,self.pose_age_s,
                  self.max_feedback_gap_s,self.arrival_m,self.hover_clearance_m,
                  self.landing_dwell_s,self.home_tolerance_m]
        if not np.isfinite(positive).all() or min(positive)<=0:
            raise ValueError('Require finite measured body calibration and positive limits')
        if not math.isfinite(self.home_pose_error_m) or not 0<=self.home_pose_error_m<=.5:
            raise ValueError('Invalid measured home uncertainty')
        if self.foot_to_body_center_m is not None and (not math.isfinite(self.foot_to_body_center_m) or self.foot_to_body_center_m<=0):
            raise ValueError('Landing foot offset must be measured, positive, or None')


@dataclass(frozen=True)
class Feedback:
    stamp: float
    position: object
    epoch: str
    armed: bool
    landed: bool
    speed_mps: float
    foot_height_above_surface_m: object = None
    tracking_valid: bool = True
    body_up_local: object = None
    position_error_m: float = .03
    uncertainty_limit_m: float = .20

    def validate(self,now):
        p=vector(self.position)
        if not math.isfinite(self.uncertainty_limit_m) or not .01<=self.uncertainty_limit_m<=.5 or not math.isfinite(self.position_error_m) or not 0<=self.position_error_m<=self.uncertainty_limit_m: raise ValueError('Invalid measured pose error budget')
        if not np.isfinite([self.stamp,now,self.speed_mps]).all() or self.speed_mps<0:
            raise ValueError('Invalid measured feedback')
        if self.foot_height_above_surface_m is not None and (not math.isfinite(self.foot_height_above_surface_m) or self.foot_height_above_surface_m<-.03):
            raise ValueError('Invalid calibrated downward-ToF foot clearance')
        return p


@dataclass
class Intent:
    action: str
    stage: str
    phase: str
    reason: str
    goal: object = None
    look_at: object = None
    certificate: object = None
    evidence: dict = field(default_factory=dict)

    def as_dict(self):
        return {'action':self.action,'stage':self.stage,'phase':self.phase,'reason':self.reason,
                'goal':self.goal,'look_at':self.look_at,
                'certificate':None if self.certificate is None else self.certificate.as_dict(),
                'evidence':self.evidence}


class TaskExecutive:
    def __init__(self,limits,home_body_position,epoch):
        self.limits=limits; self.home=vector(home_body_position).copy()
        if not epoch: raise ValueError('Measured local-pose epoch required')
        self.epoch=epoch; self.stage='TUNNEL_ENTRY'; self.phase='SEARCH'
        self.target=None; self.views=None; self.last_feedback=None; self.pending=None
        self.current_position_error_m=.03
        self.log=[]; self.events_dropped=0; self.completed={}; self.dwell_start=None; self.cross_stamp=None
        self.landing_evidence={};self.takeoff_evidence=None; self.foot_calibration=None; self.motion_audit_intervals=0; self.passage_evidence={}; self.entry=None; self.dock=None; self.fault=None; self.serial=0

    def install_foot_calibration(self,calibration):
        cal=calibration
        if not isinstance(cal,FootCalibration) or cal.epoch!=self.epoch or cal.sample_count<20 or cal.last_stamp-cal.first_stamp<1. or not 0<cal.error_m<=.025 or not .01<=cal.offset_m<=.6 or not cal.robot_calibration_id:
            raise ValueError('Stable measured same-epoch startup calibration required')
        if np.linalg.norm(vector(cal.resting_body_local_m)-self.home)>.03:
            raise ValueError('Startup calibration does not match own recorded resting body pose')
        self.foot_calibration=cal
        self.limits=replace(self.limits,foot_to_body_center_m=cal.offset_m)
        self._event('measured_foot_calibration_installed',cal.last_stamp,calibration=cal.as_dict())

    @property
    def desired_role(self):
        return {'TUNNEL_ENTRY':'tunnel_entry','TUNNEL_EXIT':'tunnel_exit','WINDOW':'window',
                'DOCK':'dock','WALL':'wall','HOME':'home'}.get(self.stage)

    def _event(self,name,stamp,**details):
        self.serial+=1
        event={'event_id':self.serial,'event':name,'stamp':stamp,'epoch':self.epoch,**plain(details)}
        self.log.append(event)
        if len(self.log)>256:
            del self.log[0]; self.events_dropped+=1
        # Evidence grows with actual task events, not frame rate.
        return event

    def _intent(self,action,reason,goal=None,look_at=None,certificate=None):
        return Intent(action,self.stage,self.phase,reason,
                      None if goal is None else vector(goal).tolist(),
                      None if look_at is None else vector(look_at).tolist(),certificate)

    def _hold(self,reason):
        return self._intent('hold',reason)

    def authorize_external_step(self,goal,feedback,guard,now):
        """Call immediately BEFORE executing an independently selected frontier.

        Uses the same immutable map certificate and motion audit as task steps.
        It cannot select an exploration direction or complete a mission stage.
        """
        p=feedback.validate(now)
        if guard.uncertainty and feedback.position_error_m>guard.uncertainty['body_three_sigma_proxy_m']+1e-12:
            return self._hold('current uncertainty exceeds the issued geometry envelope')
        if self.fault or self.phase not in ('SEARCH','SEEK_HOME') or self.last_feedback is None or feedback.stamp != self.last_feedback.stamp or not self._fresh(feedback,now):
            return self._hold('external step is not authorized in current state')
        return self._move(vector(goal),p,guard,now,'measured frontier step')

    def _fresh(self,f,now):
        return f.tracking_valid and f.epoch==self.epoch and 0<=now-f.stamp<=self.limits.pose_age_s

    def _audit_motion(self,f,p,now):
        previous=self.last_feedback
        if previous is None: return True
        dt=f.stamp-previous.stamp
        displacement=np.linalg.norm(p-vector(previous.position))
        if dt<=0: return False
        if dt>self.limits.max_feedback_gap_s or displacement>self.limits.max_speed_mps*dt+.04:
            self.fault='feedback continuity or motion bound lost'; return False
        landed_jitter=previous.landed and f.landed and displacement<=.008
        if displacement>1e-6 and not landed_jitter:
            if self.pending is None or not self.pending.covers_motion(previous.position,p,now,self.epoch):
                self.fault='measured motion lacked a prior measured-volume/contact certificate'; return False
        self.motion_audit_intervals+=1
        return True

    def _move(self,goal,p,guard,now,reason):
        goal=vector(goal); d=goal-p
        step=p+d*min(1.,self.limits.max_step_m/max(np.linalg.norm(d),1e-9))
        certificate=guard.certify([p,step],now,self.epoch,reason)
        if certificate is None:
            self.pending=None
            return self._intent('observe','unknown, occupied, stale, or out-of-map body path',look_at=goal)
        self.pending=certificate
        return self._intent('move',reason,step,certificate=certificate)

    def _usable(self,t,now):
        if t.epoch != self.epoch or not t.consistent or not t.geometry_verified or not 0<=now-t.stamp<=self.limits.target_age_s: return False
        if t.observation_count<3 or len(t.camera_origins)<3: return False
        if len({h.get('stamp') for h in t.history if h.get('calibration_id')})<3: return False
        if max(np.linalg.norm(p-q) for p in t.camera_origins for q in t.camera_origins)<.08: return False
        if t.role in ('window','door','tunnel_entry','tunnel_exit'):
            if t.interior!='free-background': return False
            if not self._aperture_body_fits(t): return False
        if t.role=='dock':
            if self.limits.foot_to_body_center_m is None: return False
            if t.details.get('marker_kind') not in self.limits.allowed_dock_markers: return False
            if not self._landing_support(t): return False
        if t.role=='wall':
            if 'edge_height_upper_local_m' not in t.details or abs(t.normal[2])>.25: return False
        if self.stage=='TUNNEL_EXIT' and self.entry is not None:
            # One observed plane cannot stand for both tunnel entry and exit.
            n=self.entry.normal
            separation=-float(n@(t.center-self.entry.center))
            if separation <= 2*(self.limits.body_radius_m+self.entry.error_m+t.error_m): return False
            if abs(n@t.normal)<math.cos(math.radians(20)): return False
        return True

    def _aperture_body_fits(self,t):
        extra=self._body_uncertainty_extra()
        if t.boundary is None:
            # With no measured in-plane axes, retain the conservative fallback.
            diameter=2*math.hypot(self.limits.body_radius_m+extra[0],self.limits.body_half_height_m+extra[2])
            return bool(min(t.dimensions)>=diameter+2*t.error_m)
        boundary=np.asarray(t.boundary,dtype=float)
        if boundary.ndim!=2 or boundary.shape[1]!=3 or not 4<=len(boundary)<=64 or not np.isfinite(boundary).all():return False
        n=t.normal
        if np.max(np.abs((boundary-t.center)@n))>max(.02,t.error_m):return False
        for a,b in zip(boundary,np.roll(boundary,-1,axis=0)):
            edge=b-a
            if np.linalg.norm(edge)<.01:return False
            inward=np.cross(n,edge);inward/=np.linalg.norm(inward)
            distance=float((t.center-a)@inward)
            if distance<0:inward=-inward;distance=-distance
            support=(self.limits.body_radius_m+max(extra[:2]))*np.linalg.norm(inward[:2])+(self.limits.body_half_height_m+extra[2])*abs(inward[2])
            if distance<support+t.error_m:return False
        # This is a geometry proposal check only; the normal complete-body
        # OnlineMap query remains authoritative, including its AABB padding.
        return True

    def _body_uncertainty_extra(self):
        policy=self.limits.uncertainty_policy
        return np.zeros(3) if policy is None else np.asarray(policy.extra(self.current_position_error_m,0.,include_map_geometry=False))

    def _landing_support(self,t):
        points=np.asarray(t.details.get('support_points_local_m',[]),dtype=float)
        if points.ndim!=2 or points.shape[1]!=3 or len(points)<32 or not np.isfinite(points).all(): return False
        if abs(t.normal[2])<math.cos(math.radians(15)): return False
        radius=self.limits.body_radius_m+t.error_m
        grid=np.linspace(-radius,radius,11)
        xy=np.array([[x,y] for x in grid for y in grid if x*x+y*y<=radius*radius])+t.center[:2]
        distance=np.linalg.norm(xy[:,None,:]-points[None,:,:2],axis=2)
        # Every sampled footprint patch needs direct nearby plane support;
        # convex hulls never fill missing measurements or holes.
        return bool((distance.min(axis=1)<=.06).all())

    def _same_target(self,t):
        if self.target is None or t.role != self.target.role or t.kind != self.target.kind: return False
        return np.linalg.norm(t.center-self.target.center)<self.target.error_m+t.error_m+.08 and abs(t.normal@self.target.normal)>.97

    def _acquire(self,tracks,p,now):
        roles={'window','door'} if self.stage=='WINDOW' and self.limits.allow_door_alternative else {self.desired_role}
        pool=[t for t in tracks if t.role in roles and self._usable(t,now)]
        if not pool: return False
        self.target=deepcopy(min(pool,key=lambda t:np.linalg.norm(t.center-p)))
        t=self.target
        if t.role=='wall':
            if np.dot(p-t.center,t.normal)<0: t.normal=-t.normal
            if float((self.home-t.center)@t.normal)>=-(self.limits.body_radius_m+t.error_m):
                self.target=None; return False
        if t.role in ('dock',):
            goal=t.center+[0,0,self.limits.foot_to_body_center_m+self.limits.hover_clearance_m]
            transit_height=max(float(p[2]),float(goal[2]))
            self.views={'before':np.asarray(goal).tolist(),
                        'staging':np.array([p[0],p[1],transit_height]).tolist(),
                        'overhead':np.array([t.center[0],t.center[1],transit_height]).tolist()}
        else:
            # Freeze plane orientation toward the current observing body.
            if np.dot(p-t.center,t.normal)<0: t.normal=-t.normal
            extra=self._body_uncertainty_extra()
            self.views=passage_views(t,self.limits.body_radius_m+max(extra[:2]),self.limits.body_half_height_m+extra[2])
            if t.role=='wall':
                height=t.details['edge_height_upper_local_m']+self.limits.body_half_height_m+extra[2]+t.error_m+.10
                for name in ('before','after'): self.views[name][2]=height
        self.phase='APPROACH'; self.cross_stamp=None
        self.passage_evidence={'geometry_observations':deepcopy(t.history),
                               'camera_origins_local_m':[p.tolist() for p in t.camera_origins],
                               'audit_start_interval':self.motion_audit_intervals,
                               'plane_center_local_m':t.center.tolist(),'plane_normal_local':t.normal.tolist()}
        self._event('target_acquired',now,target_id=t.id,role=t.role,center_local_m=t.center.tolist(),
                    normal_local=t.normal.tolist(),interior=t.interior,geometry_only=True)
        return True

    def _feedback_on_surface(self,f,p,surface,horizontal_tolerance=None,reference_error_m=0.):
        if self.limits.foot_to_body_center_m is None: return False
        if f.body_up_local is None: return False
        up=vector(f.body_up_local)
        if abs(np.linalg.norm(up)-1.)>.01 or up[2]<math.cos(math.radians(10)): return False
        contact_tolerance=.035 if self.foot_calibration is None else max(.01,min(.025,self.foot_calibration.error_m+.006))
        if not f.landed or f.speed_mps>.06 or f.foot_height_above_surface_m is None or abs(f.foot_height_above_surface_m)>contact_tolerance: return False
        expected=surface+[0,0,self.limits.foot_to_body_center_m]
        tolerance=self.limits.arrival_m if horizontal_tolerance is None else horizontal_tolerance
        relative_xy=0. if self.limits.uncertainty_policy is None else math.sqrt(2)*f.position_error_m+reference_error_m
        relative_z=0. if self.limits.uncertainty_policy is None else f.position_error_m+reference_error_m
        return np.linalg.norm(p[:2]-expected[:2])+relative_xy<=tolerance and abs(p[2]-expected[2])+relative_z<=.06

    def _dwell(self,condition,now):
        if not condition:
            self.dwell_start=None; return False
        if self.dwell_start is None: self.dwell_start=now
        return now-self.dwell_start>=self.limits.landing_dwell_s

    def tick(self,feedback,tracks,guard,now,forward_evidence=None,landing_context=None):
        f=feedback; p=f.validate(now)
        if not isinstance(guard,MeasuredMapGuard) or guard.epoch!=self.epoch or not np.allclose(guard.map.extent,[self.limits.body_radius_m,self.limits.body_radius_m,self.limits.body_half_height_m]):
            return self._hold('map epoch or measured body calibration mismatch')
        if self.fault: return self._hold(self.fault)
        if not self._fresh(f,now):
            return self._hold('stale, future, reset, or invalid measured local pose')
        if self.last_feedback is not None and f.stamp<=self.last_feedback.stamp:
            return self._hold('duplicate or out-of-order feedback')
        if not self._audit_motion(f,p,now): return self._hold(self.fault or 'motion audit failed')
        # A semantic/contact veto must not discard a valid measured pose and
        # later masquerade as a missing VIO interval. No task gate advances.
        self.last_feedback=f
        self.current_position_error_m=f.position_error_m
        if guard.uncertainty and f.position_error_m>guard.uncertainty['body_three_sigma_proxy_m']+1e-12:
            return self._hold('current uncertainty exceeds the issued geometry envelope')
        contact_certificate=None
        if landing_context is not None and self.stage in ('DOCK','HOME','REARM'):
            if f.position_error_m>.04:
                return self._hold('local contact pose precision exceeds unchanged 4 cm gate')
            ctx=landing_context;rs=ctx.range_pose
            if self.foot_calibration is None or ctx.calibration!=self.foot_calibration:
                return self._hold('install the actual startup foot calibration before contact adaptation')
            if rs.epoch!=f.epoch or abs(rs.pose_stamp-f.stamp)>.025 or np.linalg.norm(vector(rs.position_local_m)-p)>.02 or rs.armed!=f.armed or rs.landed!=f.landed:
                return self._hold('contact range/pose and accepted vehicle feedback mismatch')
            if self.stage in ('DOCK','REARM'):
                anchor=self.target if self.stage=='DOCK' else self.dock
                if anchor is not None and (np.linalg.norm(vector(ctx.support.center_local_m)-anchor.center)>anchor.error_m+.03 or abs(vector(ctx.support.normal_local)@anchor.normal)<.99):
                    return self._hold('contact support proof does not match the active measured platform')
            if self.stage=='HOME' and (np.linalg.norm(vector(ctx.support.center_local_m)[:2]-self.home[:2])>self.limits.home_tolerance_m or abs(ctx.support.height(*self.home[:2])-(self.home[2]-ctx.calibration.offset_m))>ctx.support.error_m+ctx.calibration.error_m+.03):
                return self._hold('home contact support does not match the recorded measured start ground')
            rising=self.stage=='REARM' and f.armed
            contact_certificate=ctx.adapter.certify(guard,ctx.support,ctx.calibration,rs,now,
                                      liftoff_height_m=self.limits.hover_clearance_m if rising else None)
            if contact_certificate is not None:
                obs=ctx.adapter.last_observation
                if f.foot_height_above_surface_m is not None and abs(f.foot_height_above_surface_m-obs['foot_clearance_m'])>.02:
                    return self._hold('reported foot clearance disagrees with calibrated measured ToF')
                f=replace(f,foot_height_above_surface_m=obs['foot_clearance_m'],
                          body_up_local=rotation_of_contact_up(rs))
        active_contact=(self.stage=='DOCK' and self.phase in ('LAND','DISARM')) or (self.stage=='HOME' and self.phase in ('LAND_HOME','DISARM_HOME')) or self.stage=='REARM'
        if active_contact and f.position_error_m>.04:
            self.pending=None
            return self._hold('local contact pose precision exceeds unchanged 4 cm gate')
        if landing_context is not None and active_contact and contact_certificate is None:
            self.pending=None
            return self._hold('native contact monitor veto: '+str(landing_context.adapter.last_rejection))
        self.last_feedback=f
        if not f.landed and not f.armed:
            self.fault='airborne feedback without confirmed arming'; return self._hold(self.fault)
        if self.stage in ('TUNNEL_ENTRY','TUNNEL_EXIT','WINDOW','WALL') and f.landed:
            return self._hold('task passages require measured airborne state')
        # Certificates apply only until replaced by a later intent. A measured
        # stationary dwell still needs a fresh full-body holding volume.
        if not f.landed:
            holding=guard.certify([p,p],now,self.epoch,'measured hover volume')
            contact_phase=(self.stage=='DOCK' and self.phase=='LAND') or (self.stage=='HOME' and self.phase=='LAND_HOME') or (self.stage=='REARM' and self.phase=='TAKEOFF')
            if holding is None and contact_phase: holding=contact_certificate
            if holding is None:
                self.pending=None; return self._hold('current body volume lacks free-flight or measured-contact verification')
        else:
            holding=None
        self.pending=holding
        if self.stage=='COMPLETE': return self._intent('complete','all ordered measured mission evidence gates satisfied')
        if self.stage in ('TUNNEL_ENTRY','TUNNEL_EXIT','WINDOW','WALL'):
            return self._passage(f,p,tracks,guard,now,forward_evidence)
        if self.stage=='DOCK': return self._dock(f,p,tracks,guard,now,contact_certificate,landing_context)
        if self.stage=='REARM': return self._rearm(f,p,guard,now,contact_certificate,landing_context)
        if self.stage=='HOME': return self._home(f,p,guard,now,contact_certificate,landing_context)
        return self._hold('invalid stage')

    def _passage(self,f,p,tracks,guard,now,forward_evidence=None):
        if self.phase=='SEARCH':
            if not self._acquire(tracks,p,now):
                return self._intent('explore','seek task target through independently measured free frontiers')
        t=self.target
        matching=[q for q in tracks if self._same_target(q) and 0<=now-q.stamp<=self.limits.target_age_s]
        if matching and any(q.interior in ('blocked','painted') for q in matching):
            self.target=None; self.views=None; self.phase='SEARCH'
            return self._hold('fresh painted or obstructed interior vetoed target')
        if self.phase=='APPROACH':
            if np.linalg.norm(p-np.asarray(self.views['before']))>self.limits.arrival_m:
                return self._move(self.views['before'],p,guard,now,'approach measured target observation side')
            if not matching or not any(self._usable(q,now) for q in matching):
                return self._intent('observe','reobserve target at approach side',look_at=t.center)
            # Crossing intent needs one fresh full-body certificate spanning the
            # entire front-to-back section before the first crossing motion.
            cert=guard.certify([p,self.views['after']],now,self.epoch,'complete target body passage')
            if cert is None: return self._intent('observe','full body passage remains unknown or obstructed',look_at=t.center)
            self.phase='CROSS'
            self.passage_evidence['before']=self._event('body_before_measured_plane',now,target_id=t.id,signed_distance_m=float((p-t.center)@t.normal))
            self.passage_evidence['path']=self._event('passage_path_verified',now,target_id=t.id,certificate=cert.as_dict())
        if self.phase=='CROSS':
            # Revalidate whole remaining sweep, not only the center ray.
            if guard.certify([p,self.views['after']],now,self.epoch,'remaining body passage') is None:
                return self._intent('observe','remaining body passage lost map verification',look_at=t.center)
            signed=float((p-t.center)@t.normal)
            body_position_support=f.position_error_m*(np.abs(t.normal).sum() if self.limits.uncertainty_policy is not None else 1.)
            support=self.views['body_support_m']+t.error_m+body_position_support
            if signed<=-support and np.linalg.norm(p-np.asarray(self.views['after']))<=self.limits.arrival_m:
                self.cross_stamp=now; self.phase='VERIFY'
                self.passage_evidence['crossing']=self._event('body_crossed_measured_plane',now,target_id=t.id,signed_distance_m=signed)
            else:
                return self._move(self.views['after'],p,guard,now,'cross online-verified complete-body passage')
        if self.phase=='VERIFY':
            # Require a NEW measured far-side observation of this same plane.
            observed=any(q.stamp>self.cross_stamp and self._usable(q,now) and q.camera_origins and
                         float((q.camera_origins[-1]-t.center)@t.normal)<-t.error_m for q in matching)
            forward=self._forward_exit_proof(forward_evidence,t,guard,now)
            if not observed and forward is None:
                return self._intent('observe','obtain fresh measured forward depth/map or optional far-side rim view')
            event=self._event('passage_complete',now,target_id=t.id,role=t.role,far_side_observed=True,
                              full_body_crossed=True,verified=True,proof=deepcopy(self.passage_evidence),
                              far_side_observation_stamps=[q.stamp for q in matching if q.stamp>self.cross_stamp],
                              evidence_mode='same_plane_reobserved' if observed else 'continuous_vio_prior_paths_and_new_forward_measurements',
                              forward_proof=forward)
            self.completed[self.stage]=event
            if self.stage=='TUNNEL_ENTRY': self.entry=deepcopy(t); self.stage='TUNNEL_EXIT'
            elif self.stage=='TUNNEL_EXIT': self.stage='WINDOW'
            elif self.stage=='WINDOW': self.stage='DOCK'
            elif self.stage=='WALL': self.stage='HOME'
            self.phase='SEARCH' if self.stage!='HOME' else 'SEEK_HOME'; self.target=None; self.views=None
            return self._hold('ordered passage evidence recorded; seeking next task')
        return self._hold('invalid passage phase')

    def _forward_exit_proof(self,e,t,guard,now):
        if e is None or e.epoch!=self.epoch or not e.calibration_id or e.source!='direct measured stereo only': return None
        if not self.cross_stamp+1e-9<e.stamp<=now or now-e.stamp>self.limits.target_age_s: return None
        if max(abs(e.stamp-e.depth_stamp),abs(e.stamp-e.pose_stamp))>.025: return None
        if not self.cross_stamp+1e-9<guard.map.stamp<=now+1e-9 or now-guard.map.stamp>guard.max_age_s: return None
        if self.motion_audit_intervals<=self.passage_evidence.get('audit_start_interval',self.motion_audit_intervals): return None
        if float((vector(e.camera_origin_local_m)-t.center)@t.normal)>=-t.error_m: return None
        points=np.asarray(e.points_local_m,dtype=float)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all() or len(points)>256: return None
        valid=(points-t.center)@t.normal < -(self.views['body_support_m']+t.error_m)
        q=np.asarray(e.quadrants)
        if q.shape!=(len(points),): return None
        if valid.sum()<48 or len(set(q[valid]))<3: return None
        return {'fresh_forward':e.as_dict(),'fresh_map_stamp':guard.map.stamp,
                'continuous_audited_intervals':self.motion_audit_intervals-self.passage_evidence['audit_start_interval'],
                'far_side_direct_points':int(valid.sum()),'pose_epoch_unchanged':True,
                'prior_complete_body_path_verified':True}

    def _dock(self,f,p,tracks,guard,now,contact_certificate=None,landing_context=None):
        if self.foot_calibration is not None and self.phase in ('LAND','DISARM') and landing_context is None:
            return self._hold('calibrated native landing requires fresh measured contact context')
        if self.limits.foot_to_body_center_m is None:
            return self._hold('measured landing-foot calibration is required before terminal landing')
        if self.phase=='SEARCH':
            if not self._acquire(tracks,p,now): return self._intent('explore','seek marked measured landing plane')
        t=self.target
        if self.phase=='APPROACH':
            for waypoint in ('staging','overhead'):
                if waypoint in self.views:
                    if np.linalg.norm(p-np.asarray(self.views[waypoint]))>self.limits.arrival_m:
                        return self._move(self.views[waypoint],p,guard,now,'map-verified '+waypoint+' for observed platform')
                    del self.views[waypoint]
            approach_tolerance=min(self.limits.arrival_m,.025) if landing_context is not None or self.foot_calibration is not None else self.limits.arrival_m
            if np.linalg.norm(p-np.asarray(self.views['before']))>approach_tolerance:
                return self._move(self.views['before'],p,guard,now,'approach marked landing plane')
            matching=[q for q in tracks if self._same_target(q) and self._usable(q,now)]
            if not matching and contact_certificate is None:
                return self._intent('observe','refresh direct support over full landing footprint',look_at=t.center)
            # A front/down camera can lose the mark beneath the body. A still-
            # valid dense support patch plus current calibrated ToF/pose/map
            # contact certificate is the executable equivalent; no hole filling.
            self.phase='LAND'; self.dwell_start=None
        if self.phase=='LAND':
            if self._dwell(self._feedback_on_surface(f,p,t.center,reference_error_m=t.error_m),now):
                self.phase='DISARM'; self._event('dock_landed',now,target_id=t.id,tof_and_pose_agree=True)
            else:
                goal=t.center+[0,0,self.limits.foot_to_body_center_m]
                cert=contact_certificate if landing_context is not None else guard.certify([p,goal],now,self.epoch,'complete landing descent above measured plane')
                if cert is None: return self._hold('landing descent body volume remains unverified')
                self.pending=cert
                if 'DOCK' not in self.landing_evidence:
                    self.landing_evidence['DOCK']={'initial_descent_certificate':cert.as_dict(),
                        'surface_relative':None if landing_context is None else deepcopy(landing_context.adapter.last_observation)}
                    self._event('dock_descent_authorized',now,proof=deepcopy(self.landing_evidence['DOCK']))
                intent=self._intent('land','native LAND on measured marked support with monitored contact certificate',goal,certificate=cert)
                intent.evidence={'native_land_required':landing_context is not None,'stop_offboard_position_stream_after_native_mode_confirmation':landing_context is not None,
                                 'surface_relative':None if landing_context is None else landing_context.adapter.last_observation}
                return intent
        if self.phase=='DISARM':
            if not self._feedback_on_surface(f,p,t.center,reference_error_m=t.error_m): return self._hold('landing evidence lost before disarm')
            if f.armed: return self._intent('disarm','confirmed stable dock landing')
            self.completed['DOCK']=self._event('dock_landed_and_disarmed',now,target_id=t.id,verified=True,
                proof=deepcopy(self.landing_evidence.get('DOCK')),
                final_measured_body_local_m=p.tolist(),foot_clearance_m=f.foot_height_above_surface_m,
                measured_speed_mps=f.speed_mps,landed=f.landed,armed=f.armed)
            self.dock=deepcopy(t); self.stage='REARM'; self.phase='ARM'; self.dwell_start=None
            return self._hold('dock disarm confirmed; rearm task enabled')
        return self._hold('invalid dock phase')

    def _rearm(self,f,p,guard,now,contact_certificate=None,landing_context=None):
        if self.foot_calibration is not None and landing_context is None:
            return self._hold('calibrated rearm/liftoff requires fresh measured contact context')
        if self.phase=='ARM':
            if not self._feedback_on_surface(f,p,self.dock.center,reference_error_m=self.dock.error_m): return self._hold('dock pose/ToF/landed evidence required before rearm')
            if not f.armed: return self._intent('arm','rearm after confirmed task landing and disarm')
            self.completed['REARM']=self._event('rearmed_after_dock_disarm',now,verified=True,
                previous_disarm_event_id=self.completed['DOCK']['event_id'],
                measured_body_local_m=p.tolist(),foot_clearance_m=f.foot_height_above_surface_m,armed=f.armed,landed=f.landed)
            self.phase='TAKEOFF'
        goal=self.dock.center+[0,0,self.limits.foot_to_body_center_m+self.limits.hover_clearance_m]
        if not f.armed: return self._hold('arming confirmation lost before takeoff')
        relative_error=0. if self.limits.uncertainty_policy is None else math.sqrt(3)*f.position_error_m+self.dock.error_m
        if np.linalg.norm(p-goal)+relative_error<=self.limits.arrival_m and not f.landed:
            self.completed['RETAKEOFF']=self._event('measured_retakeoff',now,verified=True,
                proof=deepcopy(self.takeoff_evidence),measured_body_local_m=p.tolist(),
                foot_clearance_m=f.foot_height_above_surface_m,armed=f.armed,landed=f.landed)
            self.stage='WALL'; self.phase='SEARCH'; self.target=None
            return self._hold('rearm and measured takeoff complete; seek observed wall top')
        if landing_context is not None:
            if contact_certificate is None:return self._hold('calibrated contact liftoff certificate required')
            self.pending=contact_certificate
            if self.takeoff_evidence is None:self.takeoff_evidence=contact_certificate.as_dict()
            return self._intent('takeoff','monitored liftoff from verified measured support',goal,certificate=contact_certificate)
        return self._move(goal,p,guard,now,'take off again through online-verified body volume')

    def _home(self,f,p,guard,now,contact_certificate=None,landing_context=None):
        if self.phase=='SEEK_HOME':
            hover=self.home+[0,0,self.limits.hover_clearance_m]
            arrival=min(self.limits.arrival_m,.025) if landing_context is not None or self.foot_calibration is not None else self.limits.arrival_m
            if np.linalg.norm(p-hover)>arrival:
                route=guard.map.route_step(p,hover,max_step=self.limits.max_step_m)
                if route is None: return self._intent('explore','reobserve measured-free return route to own recorded launch pose')
                return self._move(route[0],p,guard,now,'return through current map to own measured launch pose')
            self.phase='LAND_HOME'; self.dwell_start=None
        if self.phase=='LAND_HOME':
            if self.foot_calibration is not None and landing_context is None:
                return self._hold('calibrated home landing requires fresh measured contact context')
            home_surface=self.home-[0,0,self.limits.foot_to_body_center_m]
            if self._dwell(self._feedback_on_surface(f,p,home_surface,self.limits.home_tolerance_m,math.sqrt(3)*self.limits.home_pose_error_m),now):
                self.phase='DISARM_HOME'; self._event('home_landed',now,home_error_m=float(np.linalg.norm(p-self.home)))
            else:
                cert=contact_certificate if landing_context is not None else guard.certify([p,self.home],now,self.epoch,'observed home landing descent')
                if cert is None: return self._hold('home landing descent remains unknown or blocked')
                self.pending=cert
                if 'HOME' not in self.landing_evidence:
                    self.landing_evidence['HOME']={'initial_descent_certificate':cert.as_dict(),
                        'surface_relative':None if landing_context is None else deepcopy(landing_context.adapter.last_observation)}
                    self._event('home_descent_authorized',now,proof=deepcopy(self.landing_evidence['HOME']))
                intent=self._intent('land','native LAND at own measured starting body pose',self.home,certificate=cert)
                intent.evidence={'native_land_required':landing_context is not None,'surface_relative':None if landing_context is None else landing_context.adapter.last_observation}
                return intent
        if self.phase=='DISARM_HOME':
            if self.foot_calibration is not None and landing_context is None:
                return self._hold('calibrated home disarm requires fresh measured contact context')
            surface=self.home-[0,0,self.limits.foot_to_body_center_m]
            if not self._feedback_on_surface(f,p,surface,self.limits.home_tolerance_m,math.sqrt(3)*self.limits.home_pose_error_m): return self._hold('home landing evidence lost')
            if f.armed: return self._intent('disarm','confirmed stable home landing')
            self.completed['HOME']=self._event('home_landed_and_disarmed',now,verified=True,
                proof=deepcopy(self.landing_evidence.get('HOME')),final_measured_body_local_m=p.tolist(),
                foot_clearance_m=f.foot_height_above_surface_m,armed=f.armed,landed=f.landed)
            required={'TUNNEL_ENTRY','TUNNEL_EXIT','WINDOW','DOCK','REARM','RETAKEOFF','WALL','HOME'}
            if set(self.completed)!=required:
                self.fault='missing ordered task evidence'; return self._hold(self.fault)
            self.stage='COMPLETE'; self.phase='COMPLETE'
            return self._intent('complete','all ordered measured mission evidence gates satisfied')
        return self._hold('invalid home phase')

    def report(self):
        return {'competition_tasks_complete':self.stage=='COMPLETE' and not self.fault,
                'stage':self.stage,'phase':self.phase,'fault':self.fault,'epoch':self.epoch,
                'home_body_local_m':self.home.tolist(),'completed':deepcopy(self.completed),
                'events':deepcopy(self.log),'events_dropped':self.events_dropped,'source':'measured local poses, direct stereo and full-body online-map certificates',
                'preset_course_coordinates':False,'cad_used':False,'ideal_depth_used':False}
