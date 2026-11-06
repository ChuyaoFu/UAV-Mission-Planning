"""Bounded exposure/native-state joiners for competition_navigate.

No ROS imports, scene geometry or clock-domain guesses. ROS sensor stamps use
integer nanoseconds; PX4 source microseconds are used only for ordering, and
native state freshness uses separately recorded ROS receipt time.
"""
from dataclasses import dataclass
from collections import deque,OrderedDict
from bisect import bisect_left
import math
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from task_observations import Exposure,vector
from task_landing import RangePose
from task_uncertainty import UncertaintyPolicy


def stamp_ns(header):return int(header.stamp.sec)*1000000000+int(header.stamp.nanosec)


def image_array(message,encoding,resolution):
    if message.encoding!=encoding or (message.width,message.height)!=tuple(resolution):
        raise ValueError('Image encoding/grid does not match actual rectified calibration')
    dtype=np.dtype('u1') if encoding=='mono8' else np.dtype('>f4' if message.is_bigendian else '<f4')
    if encoding not in ('mono8','32FC1'):raise ValueError('Only mono8 and metric direct 32FC1 are accepted')
    if message.step<message.width*dtype.itemsize or len(message.data)<message.height*message.step:
        raise ValueError('Malformed image row stride/data')
    return np.ndarray((message.height,message.width),dtype,buffer=message.data,
                      strides=(message.step,dtype.itemsize)).astype(np.uint8 if encoding=='mono8' else np.float32,copy=True)


@dataclass(frozen=True)
class BodyState:
    stamp_ns: int
    position: np.ndarray
    quaternion: np.ndarray
    velocity: np.ndarray
    gyro: np.ndarray
    error_m: float
    position_covariance: object = None
    pose_covariance: object = None

    @property
    def stamp(self):return self.stamp_ns*1e-9
    @property
    def rotation(self):return Rotation.from_quat(self.quaternion)


class BodyHistory:
    def __init__(self,maxlen=400,max_gap_s=.1,uncertainty_policy=None):
        self.items=deque(maxlen=maxlen);self.max_gap_ns=int(max_gap_s*1e9);self.fault=None;self.frame=None
        self.policy=uncertainty_policy or UncertaintyPolicy()
        self.observed_position_error_max_m=0.

    def add(self,stamp,p,q,body_velocity,body_gyro,pose_covariance,frame='world'):
        p=vector(p);q=vector(q,4);v=vector(body_velocity);w=vector(body_gyro)
        cov=np.asarray(pose_covariance,float).reshape(6,6)
        if not np.isfinite(cov).all() or (cov.diagonal()<0).any() or abs(np.linalg.norm(q)-1)>.001:
            raise ValueError('Invalid accepted body covariance/orientation')
        if not np.allclose(cov,cov.T,rtol=1e-7,atol=1e-10) or np.linalg.eigvalsh((cov+cov.T)*.5).min()<-1e-10:
            raise ValueError('Accepted body covariance is not symmetric positive semidefinite')
        position_variance=float(cov.diagonal()[:3].max())
        raw_error=3*math.sqrt(position_variance)
        # Fusion may already have mapped a pose refused by this stricter bridge.
        # Retain that uncertainty; rejecting BODY admission does not erase it.
        self.observed_position_error_max_m=max(self.observed_position_error_max_m,raw_error)
        # Compare in variance units: 3*sqrt(.0025) rounds just above .15.
        # The next representable variance ABOVE .0025 must still be refused.
        if position_variance>self.policy.admission_limit_m**2/9:raise ValueError('Accepted body uncertainty exceeds task budget')
        error=min(self.policy.admission_limit_m,max(.003,3*math.sqrt(position_variance)))
        self.policy.require(error)
        if self.frame is not None and frame!=self.frame:self.fault='accepted body frame changed';return False
        self.frame=frame
        if self.items:
            old=self.items[-1];dt=(stamp-old.stamp_ns)*1e-9
            if dt==0:return False
            if dt<0:self.fault='accepted body timestamp reset/reordered';return False
            bound=max(2.,np.linalg.norm(old.velocity),np.linalg.norm(v))*dt+.06
            if np.linalg.norm(p-old.position)>bound:self.fault='accepted body pose jumped';return False
        R=Rotation.from_quat(q)
        self.items.append(BodyState(int(stamp),p.copy(),q.copy(),R.apply(v),w.copy(),error,cov[:3,:3].copy(),cov.copy()));return True

    def sample(self,t):
        if self.fault or not self.items or not self.items[0].stamp_ns<=t<=self.items[-1].stamp_ns:return None
        h=list(self.items);times=[s.stamp_ns for s in h];i=bisect_left(times,t)
        if times[i]==t:return h[i]
        a,b=h[i-1],h[i]
        if b.stamp_ns-a.stamp_ns>self.max_gap_ns:return None
        u=(t-a.stamp_ns)/(b.stamp_ns-a.stamp_ns)
        R=Slerp([0.,1.],Rotation.from_quat([a.quaternion,b.quaternion]))([u])[0]
        lerp=lambda x,y:x*(1-u)+y*u
        return BodyState(t,lerp(a.position,b.position),R.as_quat(),lerp(a.velocity,b.velocity),
                         lerp(a.gyro,b.gyro),max(a.error_m,b.error_m)+.001,None)


class NativeState:
    def __init__(self):self.records={};self.acks=deque(maxlen=32);self.sequence=0;self.received_sequence={}

    def update(self,kind,message,received):
        source=int(message.timestamp)
        previous=self.records.get(kind)
        if previous and source<=previous[0]:return False
        self.records[kind]=(source,float(received),message)
        self.sequence+=1;self.received_sequence[kind]=self.sequence
        if kind=='ack':self.acks.append((received,message))
        return True

    def get(self,kind,now,max_age):
        record=self.records.get(kind)
        return None if record is None or not 0<=now-record[1]<=max_age else record[2]

    def flags(self,now,armed_constant):
        # PX4 republishes unchanged land state at 1Hz and status at about2Hz.
        # Preserve actual native messages/source order; bounds include those
        # measured periods plus transport allowance, never infer from disarm.
        status=self.get('status',now,.75);land=self.get('land',now,1.25)
        if status is None or land is None:return None
        return status.arming_state==armed_constant,bool(land.landed and not land.freefall)

    def command_rejected(self,command,issued,denied_results):
        return any(received>=issued and m.command==command and m.result in denied_results for received,m in self.acks)


@dataclass(frozen=True)
class SensorFrame:
    stamp_ns: int
    image: np.ndarray
    measured_depth: np.ndarray
    exposure: Exposure


class ExactFrameJoiner:
    """Exact integer header join; body pose must bracket the exposure too."""
    def __init__(self,K,local_body_from_camera,epoch,calibration_id,resolution):
        self.K=vector(K,4);self.body_from_camera=np.asarray(local_body_from_camera,float)
        self.epoch=epoch;self.calibration_id=calibration_id;self.resolution=tuple(resolution)
        self.buffers={k:OrderedDict() for k in ('left','depth','camera')};self.rejected=0;self.last=-1

    def add(self,kind,key,value):
        b=self.buffers[kind];b[int(key)]=value
        limit=24 if kind=='left' else 6
        while len(b)>limit:b.popitem(last=False)

    def take(self,body_history,now):
        out=[];keys=sorted(set(self.buffers['left'])&set(self.buffers['depth'])&set(self.buffers['camera']))
        for key in keys:
            if key<=self.last:
                for b in self.buffers.values():b.pop(key,None)
                continue
            state=body_history.sample(key)
            if state is None:
                if body_history.items and key>body_history.items[-1].stamp_ns:continue
                self.rejected+=1
                for b in self.buffers.values():b.pop(key,None)
                continue
            image=self.buffers['left'].pop(key);depth=self.buffers['depth'].pop(key);actual=self.buffers['camera'].pop(key)
            T=np.eye(4);T[:3,:3]=state.rotation.as_matrix();T[:3,3]=state.position
            predicted=T@self.body_from_camera
            if np.linalg.norm(actual[:3,3]-predicted[:3,3])>.02 or np.linalg.norm(Rotation.from_matrix(actual[:3,:3].T@predicted[:3,:3]).as_rotvec())>.03:
                self.rejected+=1;continue
            exp=Exposure(key*1e-9,key*1e-9,key*1e-9,key*1e-9,actual.copy(),self.epoch,
                         pose_error_m=state.error_m,tracking_valid=True,calibration_id=self.calibration_id,
                         uncertainty_limit_m=body_history.policy.admission_limit_m,
                         position_error_xyz_m=(state.error_m,)*3)
            try:exp.validate(now)
            except ValueError:self.rejected+=1;continue
            out.append(SensorFrame(key,image,depth,exp));self.last=key
        for b in self.buffers.values():
            for key in list(b):
                if now-key*1e-9>.75:b.pop(key,None)
        return out[-2:]


def range_pose(message,state,epoch,native_flags,now,tracking_valid):
    t=stamp_ns(message.header)*1e-9
    if state is None or native_flags is None or abs(t-state.stamp)>.000001:return None
    valid=math.isfinite(message.range) and message.min_range<=message.range<=message.max_range
    if not 0<=now-t<=.15:return None
    armed,landed=native_flags
    return RangePose(t,t,state.stamp,float(message.range),state.position.copy(),state.rotation.as_matrix(),
                     state.velocity.copy(),state.gyro.copy(),epoch,landed,armed,bool(valid),bool(tracking_valid),state.error_m)


class PriorControlLedger:
    """Actual published control certificates, never planned intents.

    Delayed pose feedback is split at measured-body samples AND command issue
    times. Every subinterval needs a certificate issued BEFORE that subinterval.
    A later map/command cannot retroactively bless an already-flown segment.
    """
    def __init__(self,maxlen=64,max_age_s=2.):
        self.items=deque(maxlen=maxlen);self.max_age=max_age_s;self.audited_segments=0;self.last_audit=None

    def add(self,certificate):
        if self.items and certificate.issued<self.items[-1].issued:raise ValueError('Command clock rolled back')
        self.items.append(certificate)
        while len(self.items)>1 and certificate.issued-self.items[1].issued>self.max_age:self.items.popleft()

    def interval(self,start_stamp,end_stamp,history,epoch):
        ledger=self
        class Audit:
            def covers_motion(self,a,b,now,current_epoch):
                report={'start_stamp':start_stamp,'end_stamp':end_stamp,'checked_at':now,
                        'epoch':epoch,'certificate_count':len(ledger.items),'accepted':False}
                ledger.last_audit=report
                def reject(reason,**details):
                    report.update(reason=reason,**details);return False
                if current_epoch!=epoch or end_stamp<start_stamp:return reject('epoch_or_interval')
                start=history.sample(int(round(start_stamp*1e9)));end=history.sample(int(round(end_stamp*1e9)))
                if start is None or end is None:return reject('feedback_body_history_unavailable')
                if np.linalg.norm(vector(a)-start.position)>.002 or np.linalg.norm(vector(b)-end.position)>.002:return reject('feedback_history_position_mismatch')
                times={start_stamp,end_stamp}
                times.update(s.stamp for s in history.items if start_stamp<s.stamp<end_stamp)
                times.update(c.issued for c in ledger.items if start_stamp<c.issued<end_stamp)
                times=sorted(times)
                for t0,t1 in zip(times,times[1:]):
                    details={'segment_start':t0,'segment_end':t1}
                    if t1-t0>history.max_gap_ns*1e-9+1e-9:return reject('body_history_gap',**details)
                    old=[c for c in ledger.items if c.issued<=t0+1e-9]
                    if not old:return reject('no_prior_published_certificate',**details)
                    cert=old[-1]
                    details.update(certificate=cert.as_dict() if hasattr(cert,'as_dict') else
                                   {'issued':cert.issued,'valid_until':cert.valid_until})
                    p=history.sample(int(round(t0*1e9)));q=history.sample(int(round(t1*1e9)))
                    if p is None or q is None:return reject('segment_body_history_unavailable',**details)
                    details.update(position_a=p.position.tolist(),position_b=q.position.tolist(),
                                   error_a_m=p.error_m,error_b_m=q.error_m)
                    if not cert.covers_motion(p.position,q.position,t1,current_epoch):return reject('prior_certificate_time_or_geometry',**details)
                    if hasattr(cert,'covers_uncertainty') and not cert.covers_uncertainty(max(p.error_m,q.error_m)):return reject('prior_certificate_uncertainty',**details)
                    ledger.audited_segments+=1
                report.update(accepted=True,reason='all_segments_have_prior_published_certificates')
                return True
        return Audit()
