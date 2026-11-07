"""Measured resting-foot calibration and surface-relative landing observations.

No ground height, landing-foot offset or platform height is assumed. Inputs are
accepted measured local poses, calibrated ToF extrinsics, actual range/status,
and PX4 landed/armed feedback. Nothing in this module corrects the VIO gauge.
"""
from dataclasses import dataclass
from collections import deque
import math
import numpy as np
from task_observations import vector,unit


def rotation(value):
    R=np.asarray(value,dtype=float)
    if R.shape!=(3,3) or not np.isfinite(R).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-5) or abs(np.linalg.det(R)-1)>1e-5:
        raise ValueError('Expected measured proper body rotation')
    return R


@dataclass(frozen=True)
class RangePose:
    stamp: float
    range_stamp: float
    pose_stamp: float
    range_m: float
    position_local_m: object
    local_from_body_rotation: object
    velocity_local_mps: object
    angular_velocity_body_radps: object
    epoch: str
    landed: bool
    armed: bool
    range_valid: bool
    tracking_valid: bool
    position_error_m: float = .03


@dataclass(frozen=True)
class FootCalibration:
    offset_m: float
    error_m: float
    sample_count: int
    first_stamp: float
    last_stamp: float
    epoch: str
    robot_calibration_id: str
    resting_body_local_m: tuple
    resting_ground_z_local_m: float
    source: str = 'stable disarmed PX4-landed measured ToF and body attitude'

    def as_dict(self): return dict(vars(self))


class RestingFootCalibrator:
    """Bounded continuous upright stationary startup calibration.

    For an exactly upright downward ToF, offset = range - tof_in_body.z.
    The implementation uses actual measured attitude and calibrated ray axis.
    A rejection clears the sample window: separated good fragments never count
    as one stationary dwell. The physical calibration has no deadline fallback.
    """
    def __init__(self,tof_in_body_m,tof_axis_body,robot_calibration_id,
                 range_error_m,extrinsic_error_m,min_samples=20,min_span_s=1.,max_samples=100):
        self.mount=vector(tof_in_body_m);self.axis=unit(tof_axis_body)
        if not robot_calibration_id or not 0<range_error_m<=.02 or not 0<=extrinsic_error_m<=.02:
            raise ValueError('Measured range/extrinsic error budgets and calibration ID required')
        if not 3<=min_samples<=max_samples<=200 or not 0<min_span_s<=10:
            raise ValueError('Invalid bounded calibration window')
        self.robot_calibration_id=robot_calibration_id
        self.range_error=range_error_m;self.extrinsic_error=extrinsic_error_m
        self.min_samples=min_samples;self.min_span=min_span_s
        self.samples=deque(maxlen=max_samples);self.last_rejection=None;self.epoch=None

    def _reject(self,reason):
        self.samples.clear();self.last_rejection=reason;return False

    def ingest(self,sample,now):
        s=sample
        try:
            p=vector(s.position_local_m);R=rotation(s.local_from_body_rotation)
            v=vector(s.velocity_local_mps);w=vector(s.angular_velocity_body_radps)
            stamps=np.array([s.stamp,s.range_stamp,s.pose_stamp,now],float)
            if not np.isfinite(stamps).all() or not 0<=now-s.stamp<=.15 or max(abs(stamps[:3]-s.stamp))>.02:
                return self._reject('stale, future or unsynchronized ToF/body samples')
            if not s.epoch or not s.tracking_valid or not s.range_valid or not s.landed or s.armed:
                return self._reject('confirmed disarmed landed tracking and valid range required')
            if not math.isfinite(s.range_m) or not .015<=s.range_m<=4.:
                return self._reject('invalid actual ToF range')
            if np.linalg.norm(v)>.03 or np.linalg.norm(w)>.03 or R[2,2]<math.cos(math.radians(2)):
                return self._reject('startup body must be stationary and upright')
            ray=R@self.axis
            if ray[2]>-math.cos(math.radians(5)):
                return self._reject('calibrated ToF ray is not downward')
            hit=R@(self.mount+self.axis*s.range_m)
            offset=-float(hit[2])
            if not .01<=offset<=.6:
                return self._reject('measured foot offset outside robot calibration bounds')
            if self.epoch!=s.epoch:
                self.samples.clear();self.epoch=s.epoch
            if self.samples:
                if s.stamp<=self.samples[-1][0]: return self._reject('duplicate or out-of-order sample')
                if s.stamp-self.samples[-1][0]>.15: return self._reject('startup sample continuity lost')
                if np.linalg.norm(p-self.samples[0][2])>.012:
                    return self._reject('body moved during startup calibration')
                if abs(offset-np.median([x[1] for x in self.samples]))>.01:
                    return self._reject('unstable startup range/foot offset')
            self.samples.append((s.stamp,offset,p.copy()))
            self.last_rejection=None;return True
        except (ValueError,TypeError,OverflowError):
            return self._reject('invalid measured startup packet')

    def result(self,now):
        if len(self.samples)<self.min_samples or self.samples[-1][0]-self.samples[0][0]<self.min_span or not -1e-9<=now-self.samples[-1][0]<=.15:
            return None
        offsets=np.array([x[1] for x in self.samples]);positions=np.array([x[2] for x in self.samples])
        if np.ptp(offsets)>.012: return None
        offset=float(np.median(offsets));body=np.median(positions,axis=0)
        error=self.range_error+self.extrinsic_error+float(np.max(np.abs(offsets-offset)))
        if error>.025: return None
        return FootCalibration(offset,error,len(self.samples),self.samples[0][0],self.samples[-1][0],
                               self.epoch,self.robot_calibration_id,tuple(body),float(np.median(positions[:,2]-offsets)))


def surface_relative_observation(sample,now,tof_in_body_m,tof_axis_body,foot_calibration,
                                 plane_center_local_m,plane_normal_local,target_id):
    """Return local plane/ToF agreement and platform-vs-start height. No z reset.

    The normal is measured local ENU. The ToF hit must lie on this same observed
    plane; a new platform must retain its own height instead of shifting VIO.
    """
    s=sample;cal=foot_calibration
    p=vector(s.position_local_m);R=rotation(s.local_from_body_rotation)
    n=unit(plane_normal_local);c=vector(plane_center_local_m)
    if n[2]<0:n=-n
    stamps=np.array([s.stamp,s.range_stamp,s.pose_stamp,now],float)
    if not np.isfinite(stamps).all() or not 0<=now-s.stamp<=.15 or max(abs(stamps[:3]-s.stamp))>.02:
        raise ValueError('Stale or unsynchronized measured range pose')
    if s.epoch!=cal.epoch or not s.tracking_valid or not s.range_valid or not math.isfinite(s.range_m) or not .015<=s.range_m<=4.:
        raise ValueError('Valid same-epoch measured range required')
    if n[2]<math.cos(math.radians(10)):raise ValueError('Landing plane is too tilted')
    axis=unit(tof_axis_body);ray=R@axis
    if ray[2]>-math.cos(math.radians(8)):raise ValueError('Landing ToF is not downward')
    hit=p+R@(vector(tof_in_body_m)+axis*s.range_m)
    height_at_hit=float(c[2]-(n[0]*(hit[0]-c[0])+n[1]*(hit[1]-c[1]))/n[2])
    return {'stamp':s.stamp,'epoch':s.epoch,'target_id':target_id,
            'tof_hit_local_m':hit.tolist(),'plane_residual_m':float(n@(hit-c)),
            'foot_clearance_m':float(p[2]-cal.offset_m-hit[2]),
            'observed_surface_z_local_m':float(hit[2]),'plane_surface_z_local_m':height_at_hit,
            'surface_height_above_start_ground_m':float(hit[2]-cal.resting_ground_z_local_m),
            'vio_global_z_correction_applied':False,
            'source':'measured range, calibrated mounting and preserved local VIO pose'}
