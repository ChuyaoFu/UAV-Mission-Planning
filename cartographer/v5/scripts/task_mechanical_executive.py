"""Default-off mechanical contact extension; legacy executive is unchanged."""
from copy import deepcopy
from dataclasses import replace
import math
import numpy as np
from task_executive import TaskExecutive
from task_safety import MeasuredMapGuard
from task_observations import vector
from task_native_contact import MechanicalStartup,NativeContactContext,observed_vertical_frontier


class MechanicalTaskExecutive(TaskExecutive):
    def __init__(self,limits,home_body_position,epoch,mechanical_foot):
        super().__init__(limits,home_body_position,epoch)
        if not isinstance(mechanical_foot,MechanicalStartup) or mechanical_foot.epoch!=epoch:raise ValueError('Typed mechanical startup/home anchor required')
        if np.linalg.norm(np.asarray(mechanical_foot.home_body_anchor['position_local_m'])-self.home)>1e-9:raise ValueError('Original measured home anchor required')
        self.mechanical_foot=mechanical_foot
        self.limits=replace(limits,foot_to_body_center_m=mechanical_foot.offset_m)
        self._event('typed_mechanical_startup_installed',mechanical_foot.last_stamp,mechanical=mechanical_foot.as_dict())
        self.last_contact_certificate=None;self.ground_receipts_key=None;self.ground_receipts=set()

    def _landing_support(self,t):
        # Proposal only: executable native contact always needs the independent
        # retained raw-plane/boundary/map certificate, not this sparse track.
        p=np.asarray(t.details.get('support_points_local_m',[]))
        return bool(p.ndim==2 and p.shape[1]==3 and len(p)>=32 and np.isfinite(p).all())

    def _move(self,goal,p,guard,now,reason):
        result=super()._move(goal,p,guard,now,reason)
        if result.action=='observe' and abs(vector(goal)[2]-p[2])>guard.map.res:
            step=observed_vertical_frontier(guard.map,p,vector(goal),guard.extra,self.limits.max_step_m)
            if step is not None:return super()._move(step[0],p,guard,now,'proved-free vertical staging toward observed target')
        return result

    def tick(self,feedback,tracks,guard,now,forward_evidence=None,landing_context=None):
        if self.stage not in ('DOCK','REARM','HOME'):
            return super().tick(feedback,tracks,guard,now,forward_evidence,None)
        f=feedback;p=f.validate(now)
        if self.fault:return self._hold(self.fault)
        if not isinstance(guard,MeasuredMapGuard) or guard.epoch!=self.epoch or not np.allclose(guard.map.extent,[self.limits.body_radius_m,self.limits.body_radius_m,self.limits.body_half_height_m]) or not self._fresh(f,now):return self._hold('fresh same-epoch BODY/map required')
        if self.last_feedback is not None and f.stamp<=self.last_feedback.stamp:return self._hold('duplicate or reordered native contact feedback')
        if not self._audit_motion(f,p,now):return self._hold(self.fault or 'prior contact motion certificate missing')
        self.last_feedback=f;self.current_position_error_m=f.position_error_m
        if guard.uncertainty and f.position_error_m>guard.uncertainty['body_three_sigma_proxy_m']+1e-12:return self._hold('actual pose uncertainty grew beyond issued global map envelope')
        if not f.landed and not f.armed:self.fault='airborne without actual native arming';return self._hold(self.fault)
        ctx=landing_context;cert=None
        if ctx is not None:
            if not isinstance(ctx,NativeContactContext) or ctx.calibration!=self.mechanical_foot:return self._hold('typed retained-support context/calibration mismatch')
            rs=ctx.range_pose
            if abs(rs.stamp-f.stamp)>1e-9 or rs.epoch!=self.epoch or rs.armed!=f.armed or rs.landed!=f.landed or np.linalg.norm(vector(rs.position_local_m)-p)>.001:return self._hold('actual native BODY/ToF feedback mismatch')
            rising=self.stage=='REARM' and f.armed
            cert=ctx.adapter.certify(guard,ctx.support,ctx.calibration,rs,now,
                liftoff_height_m=self.limits.hover_clearance_m if rising else None,body_history=ctx.body_history,native=ctx.native)
        self.last_contact_certificate=cert
        active=(self.stage=='DOCK' and self.phase in ('LAND','DISARM')) or self.stage=='REARM' or (self.stage=='HOME' and self.phase in ('LAND_HOME','DISARM_HOME'))
        if active and cert is None:
            self.pending=None;return self._hold('typed native contact veto: '+('support context missing' if ctx is None else str(ctx.adapter.last_rejection)))
        if not f.landed:
            holding=guard.certify([p,p],now,self.epoch,'current complete BODY hover') or cert
            if holding is None:return self._hold('current body upper/side or contact volume unverified')
            self.pending=holding
        else:self.pending=cert
        if self.stage=='DOCK':return self._dock_typed(f,p,tracks,guard,now,ctx,cert)
        if self.stage=='REARM':return self._rearm_typed(f,p,guard,now,ctx,cert)
        return self._home_typed(f,p,guard,now,ctx,cert)

    def _on_support(self,f,cert):
        key=None if cert is None else (self.stage,cert.support.target_id)
        if key!=self.ground_receipts_key:self.ground_receipts_key=key;self.ground_receipts=set()
        good=bool(cert is not None and cert.grounded and f.landed and f.speed_mps<=.06)
        if not good:self.ground_receipts=set();return False
        stamp=cert.observation.get('native_land_source_us')
        if stamp is not None:self.ground_receipts.add(stamp)
        return len(self.ground_receipts)>=2

    def _dock_typed(self,f,p,tracks,guard,now,ctx,cert):
        if self.phase=='SEARCH':
            if not self._acquire(tracks,p,now):return self._intent('explore','seek observed marked pad with measured boundary')
        if self.phase=='APPROACH':
            for name in ('staging','overhead'):
                if name in self.views:
                    if np.linalg.norm(p-self.views[name])>self.limits.arrival_m:return self._move(self.views[name],p,guard,now,'proved-free pad '+name)
                    del self.views[name]
            # Derive permissible lateral alignment from actual pad/body/error
            # budget. No universal2.5cm/4cm cap replaces the measured boundary.
            if cert is None:
                goal=np.asarray(self.views['before']);delta=goal-p
                if np.linalg.norm(delta)>self.limits.arrival_m:return self._move(goal,p,guard,now,'align over observed marked support')
                return self._intent('observe','retain bounded raw planar support before losing camera footprint',look_at=self.target.center)
            if cert.support.target_id!=self.target.id:return self._hold('retained support is not the acquired marked pad')
            self.phase='LAND';self.dwell_start=None
        if self.phase=='LAND':
            if self._dwell(self._on_support(f,cert),now):
                self.phase='DISARM';self._event('dock_landed',now,target_id=self.target.id,typed_native_contact=cert.as_dict())
            else:
                if cert is None:return self._hold('typed native descent remains unverified')
                self.pending=cert
                if 'DOCK' not in self.landing_evidence:self.landing_evidence['DOCK']=cert.as_dict()
                return self._intent('land','native LAND with retained plane, current ToF/BODY/P and full map veto',cert.points[-1],certificate=cert)
        if self.phase=='DISARM':
            if not self._on_support(f,cert):return self._hold('actual native landed/at_rest contact lost before disarm')
            if f.armed:return self._intent('disarm','native landed/at_rest and typed support dwell confirmed',certificate=cert)
            self.completed['DOCK']=self._event('dock_landed_and_disarmed',now,target_id=self.target.id,verified=True,
                proof=deepcopy(self.landing_evidence['DOCK']),typed_contact=cert.as_dict(),landed=True,armed=False)
            self.dock=deepcopy(self.target);self.stage='REARM';self.phase='ARM';self.dwell_start=None
            return self._hold('actual dock disarm confirmed; typed rearm enabled')
        return self._hold('invalid typed dock phase')

    def _rearm_typed(self,f,p,guard,now,ctx,cert):
        if self.phase=='ARM':
            if not self._on_support(f,cert):return self._hold('current grounded support required before rearm')
            if not f.armed:return self._intent('arm','rearm on current typed grounded support',certificate=cert)
            self.completed['REARM']=self._event('rearmed_after_dock_disarm',now,verified=True,
                previous_disarm_event_id=self.completed['DOCK']['event_id'],typed_contact=cert.as_dict(),armed=True,landed=True)
            self.phase='TAKEOFF'
        if not f.armed:return self._hold('native arming lost during liftoff')
        if cert is None or cert.mode!='liftoff':return self._hold('current complete liftoff sweep required')
        clearance=cert.observation['foot_clearance_m'];err=cert.observation['foot_clearance_error_m']
        # Positive measured rise by >=one body/uncertainty-resolved interval,
        # bounded by requested hover height. No fixed world/platform height.
        if not f.landed and clearance-err>=self.limits.hover_clearance_m-self.limits.arrival_m:
            self.completed['RETAKEOFF']=self._event('measured_retakeoff',now,verified=True,
                proof=cert.as_dict(),foot_clearance_m=clearance,foot_clearance_error_m=err,armed=True,landed=False)
            self.stage='WALL';self.phase='SEARCH';self.target=None
            return self._hold('typed measured rearm/liftoff complete; seek observed wall')
        self.pending=cert;self.takeoff_evidence=cert.as_dict()
        return self._intent('takeoff','monitored typed liftoff through current measured upper/side map',cert.points[-1],certificate=cert)

    def _home_typed(self,f,p,guard,now,ctx,cert):
        if self.phase=='SEEK_HOME':
            hover=self.home+[0,0,self.limits.hover_clearance_m]
            if np.linalg.norm(p-hover)>self.limits.arrival_m:
                route=guard.map.route_step(p,hover,max_step=self.limits.max_step_m)
                if route is None:return self._intent('explore','reobserve free route to actual retained home BODY anchor')
                return self._move(route[0],p,guard,now,'measured return route to original home BODY')
            if cert is None:return self._intent('observe','reobserve own measured home supporting surface before native LAND')
            if cert.support.marker_kind!='own_launch_plane':return self._hold('home surface not tied to original measured startup anchor')
            # Full original home/current uncertainty is retained in this check.
            home_error=self.limits.home_pose_error_m+f.position_error_m
            if np.linalg.norm(p[:2]-self.home[:2])+home_error>self.limits.home_tolerance_m:return self._hold('actual home-relative uncertainty exceeds measured home arrival region')
            self.phase='LAND_HOME';self.dwell_start=None
        if self.phase=='LAND_HOME':
            if self._dwell(self._on_support(f,cert),now):self.phase='DISARM_HOME';self._event('home_landed',now,typed_contact=cert.as_dict())
            else:
                if cert is None:return self._hold('typed home contact sweep unverified')
                self.pending=cert;self.landing_evidence['HOME']=cert.as_dict()
                return self._intent('land','native LAND on reobserved measured own home support',cert.points[-1],certificate=cert)
        if self.phase=='DISARM_HOME':
            if not self._on_support(f,cert):return self._hold('native home contact lost before disarm')
            if f.armed:return self._intent('disarm','typed home support and actual native grounded dwell confirmed',certificate=cert)
            self.completed['HOME']=self._event('home_landed_and_disarmed',now,verified=True,proof=cert.as_dict(),landed=True,armed=False)
            self.stage='COMPLETE';self.phase='COMPLETE'
            return self._intent('complete','all ordered measured tasks including typed native contact completed')
        return self._hold('invalid typed home phase')
