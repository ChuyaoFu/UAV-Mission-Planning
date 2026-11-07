"""Actual adapter callback observer for ROBOT mechanical foot geometry.

Read-only: mechanical metrology + actual native/ToF/BODY dwell and measured
home-relative anchor. No learned size, invented error bounds or native control.
"""
from pathlib import Path
from collections import deque
import hashlib,json,math
import numpy as np
from task_mechanical_foot import MechanicalFootGeometry


class MechanicalFootObserver:
    def __init__(self,config_root,geometry_path,mount,axis,calibration_id,range_error,extrinsic_error):
        root=Path(config_root).resolve(strict=True);path=Path(geometry_path)
        if not path.is_absolute():path=root/path
        path=path.resolve(strict=True)
        if root not in path.parents:raise ValueError('ROBOT foot geometry must be an existing file inside config; private scene paths forbidden')
        robot=root/'model.sdf';loaded=hashlib.sha256(robot.read_bytes()).hexdigest()
        self.geometry=MechanicalFootGeometry.from_dict(json.loads(path.read_text()))
        self.points,self.offset,self.error=self.geometry.validate('sitl',loaded)
        self.path=str(path);self.model_hash=loaded;self.mount=np.asarray(mount,float);self.axis=np.asarray(axis,float)
        if self.mount.shape!=(3,) or self.axis.shape!=(3,) or not np.isfinite(np.r_[self.mount,self.axis]).all() or abs(np.linalg.norm(self.axis)-1)>.001:raise ValueError('Actual calibrated BODY ToF extrinsics required')
        if not 0<range_error<=.02 or not 0<=extrinsic_error<=.02:raise ValueError('Original explicit ToF error inputs required')
        self.calibration_id=calibration_id;self.range_error=range_error;self.extrinsic_error=extrinsic_error
        # Known ROBOT geometry is not learned/averaged from these samples.
        # Two individually consistent observations can differ by twice the
        # declared measurement + extrinsic + mechanical support interval.
        self.dwell_spread_limit=2*(range_error+extrinsic_error+self.error)
        self.samples=deque(maxlen=100);self.last_rejection=None;self.anchor=None;self.relative_epoch=None
        self.packet_count=0;self.rejected_count=0

    def reject(self,reason):
        self.samples.clear();self.anchor=None;self.last_rejection=str(reason);self.rejected_count+=1;return False

    @staticmethod
    def chain_evidence(health,relative,now):
        if not isinstance(health,dict) or not health.get('valid') or not 0<=now-float(health.get('stamp',-1))<=.25 or health.get('odometry_source')!='/nav/home_relative_odomimu' or health.get('global_tof_height_correction_enabled') is not False:
            raise ValueError('Actual accepted home-relative odometry chain with ToF height correction disabled required')
        if not isinstance(relative,dict) or not relative.get('valid') or not 0<=now-float(relative.get('stamp',-1))<=.35:
            raise ValueError('Fresh actual home-relative adapter health required')
        epoch=relative.get('home_epoch_ns')
        if not isinstance(epoch,int) or epoch<0 or relative.get('pose_covariance_clipped') is not False or relative.get('position_prior_changed') is not False:
            raise ValueError('Verified home epoch and original unmodified posterior required')
        return epoch

    def ingest(self,rs,state,native,history,health,relative,now):
        self.packet_count+=1
        try:
            epoch=self.chain_evidence(health,relative,now)
            if self.relative_epoch is not None and epoch!=self.relative_epoch:return self.reject('Actual home reference epoch changed; restart adapter session for a new dwell')
            self.relative_epoch=epoch
            flags=native.flags(now,2)
            if flags!=(False,True) or rs.armed or not rs.landed:raise ValueError('Fresh actual native disarmed/landed required')
            if not rs.tracking_valid or not rs.range_valid or state is None or history.fault or not health.get('valid') or not 0<=now-float(health['stamp'])<=.25:raise ValueError('Healthy actual accepted BODY/range tracking required')
            if not 0<=now-rs.stamp<=.15 or abs(rs.stamp-state.stamp)>1e-9 or max(abs(rs.stamp-rs.range_stamp),abs(rs.stamp-rs.pose_stamp))>1e-9:raise ValueError('Exact same-exposure bracketed actual BODY/ToF required')
            R=state.rotation.as_matrix();v=np.asarray(rs.velocity_local_mps);w=np.asarray(rs.angular_velocity_body_radps)
            if np.linalg.norm(v)>.03 or np.linalg.norm(w)>.03 or R[2,2]<math.cos(math.radians(2)):raise ValueError('Observed stationary upright BODY dwell required')
            up=R[2,:];ray=R@self.axis
            if ray[2]>-math.cos(math.radians(5)):raise ValueError('Actual ToF ray is not downward')
            observed=-float(up@(self.mount+self.axis*rs.range_m));predicted=float(np.max(-self.points@up))
            # Declared input intervals only test consistency; they never qualify
            # absolute ToF accuracy or alter independent mechanical geometry E.
            if abs(observed-predicted)>self.range_error+self.extrinsic_error+self.error:raise ValueError('Actual ToF inconsistent with known mechanical foot support')
            position=np.asarray(state.position,float)
            if self.samples:
                if not 0<rs.stamp-self.samples[-1]['stamp']<=.15:raise ValueError('Duplicate/reordered/discontinuous actual dwell')
                if np.linalg.norm(position-self.samples[0]['position'])>.012:raise ValueError('Accepted BODY moved during startup dwell')
                if abs(observed-np.median([x['observed_offset'] for x in self.samples]))>self.dwell_spread_limit:raise ValueError('Observed ToF variation exceeds declared consistency intervals')
            self.samples.append(dict(stamp=rs.stamp,position=position.copy(),observed_offset=observed))
            self.last_rejection=None
            if self.ready(now) and self.anchor is None:
                actual=[s for s in history.items if self.samples[0]['stamp']<=s.stamp<=self.samples[-1]['stamp'] and s.pose_covariance is not None]
                if not actual:raise ValueError('No original accepted BODY covariance sample inside actual dwell')
                a=actual[0]
                self.anchor=dict(source='actual accepted /nav/body_odom from verified home-relative chain',stamp_ns=a.stamp_ns,
                    home_epoch_ns=epoch,sensor_epoch=rs.epoch,calibration_id=self.calibration_id,
                    position_local_m=a.position.tolist(),quaternion_xyzw=a.quaternion.tolist(),
                    pose_covariance=np.asarray(a.pose_covariance).tolist(),position_three_sigma_proxy_m=float(a.error_m),
                    covariance_averaged=False,covariance_subtracted=False,mechanical_geometry_used_to_set_world_position=False,
                    home_relative_health=dict(relative),accepted_body_frame=history.frame)
            return True
        except (ValueError,TypeError,KeyError,OverflowError) as e:return self.reject(e)

    def ready(self,now):
        return bool(len(self.samples)>=20 and self.samples[-1]['stamp']-self.samples[0]['stamp']>=1. and
            0<=now-self.samples[-1]['stamp']<=.15 and np.ptp([x['observed_offset'] for x in self.samples])<=self.dwell_spread_limit)

    def report(self,now,health=None,relative=None,*,native_flags=None):
        reason=self.last_rejection
        ready=self.ready(now)
        try:
            epoch=self.chain_evidence(health,relative,now)
            if self.relative_epoch is not None and epoch!=self.relative_epoch:raise ValueError('Actual home reference epoch changed; adapter restart required')
        except (ValueError,TypeError) as e:ready=False;reason=str(e)
        if native_flags!=(False,True):ready=False;reason='Fresh actual native disarmed/landed required'
        if not ready and reason is None:
            reason=('insufficient continuous samples/span' if len(self.samples)<20 or self.samples[-1]['stamp']-self.samples[0]['stamp']<1. else
                    'observed ToF spread exceeds declared intervals or latest accepted dwell stale/future')
        span=0. if not self.samples else self.samples[-1]['stamp']-self.samples[0]['stamp']
        anchor=self.anchor if ready else None
        return dict(mode='mechanical-observe-only',mechanical_geometry_validated=True,geometry_file=self.path,geometry_digest=self.geometry.calibration_digest,
            robot_model_file_sha256=self.model_hash,actual_runtime_loaded_model_verified=False,offset_m=self.offset,error_m=self.error,admission_error_cap_m=.025,
            foot_contact_points_body_m=self.points.tolist(),sample_count=len(self.samples),span_s=span,
            dwell_spread_limit_m=self.dwell_spread_limit,dwell_reduces_declared_sensor_error=False,
            latest_age_s=None if not self.samples else now-self.samples[-1]['stamp'],last_rejection=reason,
            observed_dwell_ready=ready,home_body_anchor=anchor,known_home_body_anchor=anchor is not None,
            declared_tof_range_error_m=self.range_error,declared_tof_extrinsic_error_m=self.extrinsic_error,
            observed_offset_spread_m=None if not self.samples else float(np.ptp([x['observed_offset'] for x in self.samples])),
            geometry_error_learned_from_tof=False,measurement_accuracy_qualified=False,runtime_contact_qualified=False,
            observed_velocity_and_attitude_are_not_hard_error_bounds=True,
            legacy_foot_calibration_created=False,native_contact_dispatch_enabled=False,motion_certificate_issued=False)
