"""Optional ROBOT-only calibrated contact geometry. No course CAD/world prior.

Mechanical precision never comes from ToF noise, dwell scatter or VIO gauge.
Existing learned-foot path is untouched. All commands/certificates stay external.
"""
from dataclasses import dataclass
from collections import deque
import hashlib,json,math,re
import numpy as np
from scipy.spatial import ConvexHull
from task_relative_contact import (FootOffsetCalibration,RelativeContractError,vec,unit,require,
    tof_geometry,assess_relative_contact)


def canonical_digest(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class MechanicalFootGeometry:
    payload: dict
    calibration_digest: str

    @classmethod
    def from_dict(cls,data):return cls(dict(data['payload']),data['calibration_digest'])

    def validate(self,runtime_kind,loaded_robot_model_sha256=None):
        p=self.payload
        require(canonical_digest(p)==self.calibration_digest,'Mechanical calibration digest mismatch')
        require(p.get('schema')=='robot-mechanical-foot-v1' and p.get('body_frame')=='base_link','Explicit robot BODY mechanical calibration required')
        require(p.get('contact_set_complete') is True and p.get('other_collision_geometry_checked') is True,'Complete physical support set and other collision check required')
        require(p.get('point_error_metric')=='euclidean-per-contact-point-bound','Mechanical2mm must bound each complete3D contact point')
        error=p.get('point_error_m');theta=p.get('contact_tilt_limit_rad')
        require(error is not None and theta is not None and math.isfinite(error) and math.isfinite(theta) and 0<error<=.01 and 0<theta<=math.radians(5),'Independent mechanical/contact tilt bounds required')
        points=np.asarray(p.get('foot_contact_points_body_m'),float)
        require(points.ndim==2 and points.shape[1]==3 and 4<=len(points)<=32 and np.isfinite(points).all(),'Complete finite mechanical contact points required')
        require(np.ptp(points[:,2])<=1e-9 and .01<=-points[0,2]<=.6,'This minimal path requires coplanar rigid foot bottoms')
        hull=ConvexHull(points[:,:2]);require(np.max(hull.equations[:,2])<=1e-9,'BODY axis must lie inside the calibrated foot support hull')
        h=-float(points[0,2]);other=p.get('other_collision_downward_support_upper_m')
        require(other is not None and math.isfinite(other) and other+error<h*math.cos(theta)-error,'Another robot part could become lowest within contact tilt cone')
        if runtime_kind=='sitl':
            require(p.get('source_scope')=='sitl-robot-model-definition','SITL requires actual robot model definition')
            require(re.fullmatch('[0-9a-f]{64}',p.get('source_model_sha256','')) is not None and loaded_robot_model_sha256==p['source_model_sha256'],'Actual loaded ROBOT model digest mismatch')
        elif runtime_kind=='hardware':
            require(not p.get('hardware_use_forbidden',False),'SITL model definition cannot qualify hardware')
            require(p.get('source_scope')=='independent-measured-hardware','Hardware requires actual measured mechanical calibration, not SITL CAD')
            require(re.fullmatch('[0-9a-f]{64}',p.get('measurement_record_sha256','')) is not None,'Actual hardware measurement record required')
        else:raise RelativeContractError('Explicit SITL/hardware calibration domain required')
        # The scalar consumer uses [0,0,-h]; full support can shift laterally.
        # Divide the complete point/tilt support bound by min(n dot up).
        rho=float(np.max(np.linalg.norm(points[:,:2],axis=1)))
        support_error=(error+rho*math.sin(theta))/math.cos(theta)
        require(0<support_error<=.025,'Mechanical complete contact support error exceeds unchanged25mm cap')
        return points,h,float(support_error)


@dataclass(frozen=True)
class MechanicalDwellPose:
    stamp_ns: int
    epoch: str
    calibration_id: str
    position_local_m: tuple
    tracking_valid: bool


@dataclass(frozen=True)
class MechanicalFootCalibration(FootOffsetCalibration):
    mechanical_geometry_digest: str=''
    mechanical_source_scope: str=''
    mechanical_point_error_m: float=0.
    contact_tilt_limit_rad: float=0.
    tof_used_to_estimate_mechanical_error: bool=False


class MechanicalFootCalibrator:
    """Known mechanical dimensions + continuous native/pose/ToF consistency dwell.

    Pose displacement and ToF are consistency vetoes, not precision estimators.
    Their noise is never averaged into or added to the measured robot size error.
    """
    def __init__(self,domain,tof_in_body,tof_axis_body,geometry,*,runtime_kind,
                 loaded_robot_model_sha256=None,diagnostic_only=False):
        self.domain=domain;self.mount=vec(tof_in_body);self.axis=unit(tof_axis_body)
        self.geometry=geometry;self.runtime_kind=runtime_kind;self.loaded_hash=loaded_robot_model_sha256
        self.points,self.offset,self.error=geometry.validate(runtime_kind,loaded_robot_model_sha256)
        self.diagnostic_only=diagnostic_only;self.samples=deque(maxlen=100);self.epoch=None;self.last_rejection=None

    def ingest(self,s,pose,now_ns):
        try:
            # Refuse mutation/rebinding after validation.
            self.geometry.validate(self.runtime_kind,self.loaded_hash)
            up,v,w,hit,tof_error,_=tof_geometry(s,self.domain,self.mount,self.axis,now_ns)
            require(pose.stamp_ns==s.stamp_ns and pose.epoch==s.epoch and pose.calibration_id==s.calibration_id and pose.tracking_valid,'Exact same-epoch tracked BODY pose required for continuous dwell')
            position=vec(pose.position_local_m)
            require(s.landed and not s.armed,'Confirmed disarmed native landed required')
            require(np.linalg.norm(v)+self.domain.speed_error_mps<=.03 and np.linalg.norm(w)+self.domain.gyro_error_radps<=.03,'Stationary velocity/gyro bounds not satisfied')
            require(math.acos(float(np.clip(up[2],-1,1)))+self.domain.up_error_rad<=math.radians(2),'Startup upright tilt/error bound exceeded')
            observed=-float(up@hit);predicted=float(np.max(-self.points@up))
            # ToF remains the original per-exposure range+mount+axis+up check.
            # Mechanical support cone covers admitted contact tilt; no fitted noise.
            require(abs(observed-predicted)<=tof_error+self.error,'Actual ToF disagrees with calibrated mechanical support')
            qualified=self.domain.runtime_qualified(('tof_extrinsics','attitude','velocity','gyro'))
            require(qualified or self.diagnostic_only,'Independent rig/attitude/stationarity qualification missing')
            if self.epoch!=s.epoch:self.samples.clear();self.epoch=s.epoch
            if self.samples:
                require(0<s.stamp_ns-self.samples[-1][0]<=150000000,'Duplicate/reordered/discontinuous mechanical dwell')
                require(np.linalg.norm(position-self.samples[0][3])<=.012,'BODY pose moved during mechanical startup dwell')
                require(abs(observed-np.median([x[1] for x in self.samples]))<=.01,'Unstable startup ToF consistency window')
            self.samples.append((s.stamp_ns,observed,tof_error,position.copy()));self.last_rejection=None;return True
        except (ValueError,TypeError,OverflowError) as e:
            self.samples.clear();self.last_rejection=str(e);return False

    def result(self,now_ns):
        self.geometry.validate(self.runtime_kind,self.loaded_hash)
        if len(self.samples)<20 or self.samples[-1][0]-self.samples[0][0]<1000000000 or not 0<=now_ns-self.samples[-1][0]<=150000000:return None
        if np.ptp([x[1] for x in self.samples])>.012:return None
        return MechanicalFootCalibration(self.offset,self.error,self.epoch,self.domain.calibration_id,
            self.samples[0][0],self.samples[-1][0],len(self.samples),
            self.domain.runtime_qualified(('tof_extrinsics','attitude','velocity','gyro')),
            self.domain.qualification_record_sha256,
            source='independently calibrated ROBOT foot geometry; native/pose/ToF dwell consistency only',
            mechanical_geometry_digest=self.geometry.calibration_digest,mechanical_source_scope=self.geometry.payload['source_scope'],
            mechanical_point_error_m=self.geometry.payload['point_error_m'],contact_tilt_limit_rad=self.geometry.payload['contact_tilt_limit_rad'])

    def diagnostics(self,now_ns):
        return dict(sample_count=len(self.samples),last_rejection=self.last_rejection,
            span_s=0. if not self.samples else (self.samples[-1][0]-self.samples[0][0])*1e-9,
            ready=self.result(now_ns) is not None,geometry_digest=self.geometry.calibration_digest,
            mechanical_offset_m=self.offset,complete_support_error_m=self.error,
            tof_range_error_m=self.domain.tof_range_error_m,tof_mount_error_m=self.domain.tof_mount_error_m,
            tof_is_consistency_only=True,mechanical_error_learned_from_noise=False,global_position_error_used=False)


def assess_mechanical_relative_contact(patch,sample,foot,domain,tof_in_body,tof_axis_body,now_ns,*,geometry,
                                      runtime_kind,loaded_robot_model_sha256=None,**kwargs):
    points,offset,error=geometry.validate(runtime_kind,loaded_robot_model_sha256)
    require(isinstance(foot,MechanicalFootCalibration) and foot.mechanical_geometry_digest==geometry.calibration_digest,'Exact typed mechanical foot calibration required')
    require(abs(foot.offset_m-offset)<=1e-12 and foot.error_m>=error and foot.mechanical_point_error_m==geometry.payload['point_error_m'] and foot.contact_tilt_limit_rad==geometry.payload['contact_tilt_limit_rad'] and foot.mechanical_source_scope==geometry.payload['source_scope'],'Mechanical complete support error/geometry changed')
    theta=geometry.payload['contact_tilt_limit_rad']
    require(math.acos(float(np.clip(unit(patch.normal_body)[2],-1,1)))+patch.normal_error_rad<=theta,'Entire mechanical support cone requires actual plane-to-BODY tilt <=5deg')
    require(math.acos(float(np.clip(unit(sample.up_body)[2],-1,1)))+domain.up_error_rad<=theta,'Actual BODY-to-up tilt outside mechanical calibration cone')
    # Only this wrapper can discharge the extra support-cone premise. The
    # optional base guard prevents accidental direct mechanical-foot consumption.
    scalar=FootOffsetCalibration(foot.offset_m,foot.error_m,foot.epoch,foot.calibration_id,
        foot.first_stamp_ns,foot.last_stamp_ns,foot.sample_count,foot.runtime_qualified,
        foot.qualification_record_sha256,source=foot.source)
    result=assess_relative_contact(patch,sample,scalar,domain,tof_in_body,tof_axis_body,now_ns,**kwargs)
    result.update(mechanical_geometry_digest=geometry.calibration_digest,mechanical_contact_points=len(points),
        mechanical_complete_support_error_m=error,tof_used_to_estimate_mechanical_error=False)
    return result
