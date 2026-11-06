"""ROS-free measured BODY/world -> PX4 local NED alignment.

An application-qualified stationary anchor is mandatory. No automatic anchoring,
continuous correction, truth input, planner, or inferred clock synchronization.
"""
from dataclasses import dataclass
import math
import numpy as np


def vector(value):
    a=np.asarray(value,float)
    if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError('Finite 3-vector required')
    return a.copy()


def angle(value):
    if not math.isfinite(value):raise ValueError('Finite angle required')
    return math.atan2(math.sin(value),math.cos(value))


def ned_rotation(phi):
    c,s=math.cos(phi),math.sin(phi)
    return np.array([[c,s,0.],[s,-c,0.],[0.,0.,-1.]])


@dataclass(frozen=True)
class BodyMeasurement:
    stamp: float
    position: object
    yaw: float
    velocity: object
    frame: str
    position_error: float
    angular_error: float
    angular_speed: float


@dataclass(frozen=True)
class NativeMeasurement:
    source_us: int
    received: float
    position: object
    heading: float
    counters: tuple
    delta_xy: object
    delta_z: float
    ref_timestamp: int
    sample_us: int
    velocity: object
    position_error: float
    velocity_error: float
    heading_error: float

    @classmethod
    def from_message(cls,m,received):
        if not m.xy_valid or not m.z_valid:raise ValueError('Native local position invalid')
        keys=('xy_reset_counter','z_reset_counter','heading_reset_counter','vxy_reset_counter','vz_reset_counter')
        counters=tuple(int(getattr(m,k)) for k in keys)
        if any(not 0<=c<=255 for c in counters):raise ValueError('Invalid native reset counter')
        d=np.asarray(m.delta_xy,float)
        if d.shape!=(2,) or not np.isfinite(d).all() or not math.isfinite(m.delta_z):raise ValueError('Invalid native reset delta')
        std=np.asarray([m.eph,m.epv,m.evh,m.evv,m.heading_var],float)
        if not np.isfinite(std).all() or (std<0).any():raise ValueError('Native covariance proxies required')
        if not m.v_xy_valid or not m.v_z_valid:raise ValueError('Native velocity invalid')
        return cls(int(m.timestamp),float(received),vector([m.x,m.y,m.z]),angle(float(m.heading)),counters,d.copy(),float(m.delta_z),int(m.ref_timestamp),int(m.timestamp_sample),vector([m.vx,m.vy,m.vz]),
                   3*math.sqrt(2*m.eph*m.eph+m.epv*m.epv),3*math.sqrt(2*m.evh*m.evh+m.evv*m.evv),3*math.sqrt(m.heading_var))


class MeasuredFrameAlignment:
    def __init__(self,max_body_age=.20,max_native_age=.35,max_speed=0.3):
        self.max_body_age=float(max_body_age);self.max_native_age=float(max_native_age);self.max_speed=float(max_speed)
        self.body=None;self.native=None;self.rotation=None;self.translation=None;self.phi=None
        self.fault=None;self.revision=0;self.anchor=None;self.waiting_reason=None;self.initialization_epoch=0

    def reject(self,reason):
        if self.rotation is None:
            self.waiting_reason=str(reason);raise ValueError(self.waiting_reason)
        if self.fault is None:self.fault=str(reason)
        raise ValueError(self.fault)

    def invalidate(self,reason):
        if self.rotation is None:self.restart_initialization(reason)
        elif self.fault is None:self.fault=str(reason)

    def restart_initialization(self,reason):
        self.body=None;self.native=None;self.waiting_reason=str(reason);self.initialization_epoch+=1

    def observe_body(self,stamp,position,yaw,velocity,frame,position_error,angular_error,angular_speed):
        if self.fault:raise ValueError(self.fault)
        try:
            if not math.isfinite(stamp) or not frame:raise ValueError('BODY timestamp/frame required')
            if not np.isfinite([position_error,angular_error,angular_speed]).all() or min(position_error,angular_error,angular_speed)<0:raise ValueError('Actual BODY covariance/motion proxies required')
            sample=BodyMeasurement(float(stamp),vector(position),angle(float(yaw)),vector(velocity),str(frame),float(position_error),float(angular_error),float(angular_speed))
            if self.body is not None:
                old=self.body;dt=sample.stamp-old.stamp
                if dt<0 or sample.frame!=old.frame:
                    if self.rotation is None:self.restart_initialization('BODY time/frame changed before anchor');return False
                    raise ValueError('BODY timestamp/frame reset after anchor')
                if dt==0:
                    if not np.array_equal(sample.position,old.position) or sample.yaw!=old.yaw:raise ValueError('Conflicting duplicate BODY timestamp')
                    return False
                travel=max(self.max_speed,np.linalg.norm(old.velocity),np.linalg.norm(sample.velocity))*dt
                if self.rotation is not None and np.linalg.norm(sample.position-old.position)>travel+old.position_error+sample.position_error:raise ValueError('Unannounced BODY position jump')
                turn=max(old.angular_speed,sample.angular_speed)*dt
                if self.rotation is not None and abs(angle(sample.yaw-old.yaw))>turn+old.angular_error+sample.angular_error:raise ValueError('Unannounced BODY heading jump')
            self.body=sample;self.waiting_reason=None;return True
        except (ValueError,TypeError,AttributeError) as error:self.reject(error)

    def observe_native(self,sample):
        if self.fault:raise ValueError(self.fault)
        try:
            if sample.source_us<=0 or sample.sample_us<=0 or not math.isfinite(sample.received):raise ValueError('Native source/receipt timestamp required')
            if self.native is not None:
                old=self.native
                if sample.source_us<old.source_us:
                    if self.rotation is None:self.restart_initialization('PX4 clock restarted before anchor');return False
                    raise ValueError('PX4 source clock restarted')
                if sample.source_us==old.source_us:
                    if sample.counters!=old.counters or sample.heading!=old.heading or sample.sample_us!=old.sample_us or sample.ref_timestamp!=old.ref_timestamp or not np.array_equal(sample.position,old.position):raise ValueError('Conflicting duplicate PX4 timestamp')
                    return False
                if self.rotation is None:
                    self.native=sample;return True
                steps=tuple((n-o)%256 for n,o in zip(sample.counters,old.counters))
                if any(steps[i] for i in (2,3,4)):raise ValueError('Unsupported PX4 heading or velocity reset; requalification required')
                if any(steps[i] not in (0,1) for i in (0,1)):raise ValueError('Missed/reordered PX4 position reset')
                if sample.ref_timestamp!=old.ref_timestamp:raise ValueError('PX4 global reference changed; requalification required')
                delta=np.r_[sample.delta_xy if steps[0] else np.zeros(2),sample.delta_z if steps[1] else 0.]
                dt=(sample.sample_us-old.sample_us)*1e-6
                if dt<0:raise ValueError('PX4 measurement clock restarted')
                speed=max(self.max_speed,np.linalg.norm(old.velocity),np.linalg.norm(sample.velocity))
                enclosure=old.position_error+sample.position_error+dt*(old.velocity_error+sample.velocity_error)
                if np.linalg.norm(sample.position-old.position-delta)>speed*dt+enclosure:raise ValueError('Unannounced PX4 position jump')
                angular_speed=0. if self.body is None else self.body.angular_speed
                if abs(angle(sample.heading-old.heading))>angular_speed*dt+old.heading_error+sample.heading_error:raise ValueError('Unannounced PX4 heading jump')
                if self.translation is not None and any(steps[:2]):
                    self.translation=self.translation+delta;self.revision+=1
            self.native=sample;return True
        except (ValueError,TypeError,AttributeError) as error:self.reject(error)

    def fresh(self,now):
        if self.fault:raise ValueError(self.fault)
        if self.body is None or self.native is None:raise ValueError('BODY and native local measurements required')
        if not 0<=now-self.body.stamp<=self.max_body_age:raise ValueError('Stale/future BODY measurement')
        if not 0<=now-self.native.received<=self.max_native_age:raise ValueError('Stale/future native local receipt')

    def establish(self,now,*,anchor_id,stationary_confirmed,disarmed_confirmed):
        self.fresh(now)
        if self.rotation is not None:raise ValueError('Alignment already fixed; silent reanchoring forbidden')
        if not anchor_id or not stationary_confirmed or not disarmed_confirmed:raise ValueError('Qualified stationary disarmed anchor required')
        self.phi=angle(self.native.heading+self.body.yaw);self.rotation=ned_rotation(self.phi)
        self.translation=self.native.position-self.rotation@self.body.position
        self.anchor=dict(id=str(anchor_id),body_stamp_ros=self.body.stamp,native_source_us=self.native.source_us,
                         native_received_ros=self.native.received,body_frame=self.body.frame,
                         clocks_synchronized=False,stationary_anchor_qualified_by_caller=True)

    def validate(self,now):
        self.fresh(now)
        if self.rotation is None:raise ValueError('Explicit qualified control alignment has not been established')
        # No drift-based reanchoring or new alignment-error threshold. Input
        # discontinuity checks use original reported covariance proxies above.

    def position(self,p):
        if self.rotation is None:raise ValueError('Control alignment unavailable')
        return self.rotation@vector(p)+self.translation

    def derivative(self,v):
        if self.rotation is None:raise ValueError('Control alignment unavailable')
        return self.rotation@vector(v)

    def inverse_position(self,p):
        if self.rotation is None:raise ValueError('Control alignment unavailable')
        return self.rotation.T@(vector(p)-self.translation)

    def yaw(self,yaw):
        if self.phi is None:raise ValueError('Control alignment unavailable')
        return angle(self.phi-angle(yaw))


def position_command_sample(m,now,frame,max_age=.2):
    """Validate native EGO's original sample; no clipping, scaling or retiming."""
    stamp=float(m.header.stamp.sec)+int(m.header.stamp.nanosec)*1e-9
    if not math.isfinite(stamp) or not 0<=now-stamp<=max_age:raise ValueError('Stale/future PositionCommand')
    if m.header.frame_id!=frame:raise ValueError('PositionCommand/BODY frame mismatch')
    if int(m.trajectory_flag) not in (1,3):raise ValueError('PositionCommand is not ready/completed')
    if int(m.trajectory_id)<0:raise ValueError('Invalid trajectory id')
    values=[vector([getattr(m,k).x,getattr(m,k).y,getattr(m,k).z]) for k in ('position','velocity','acceleration')]
    return stamp,*values
