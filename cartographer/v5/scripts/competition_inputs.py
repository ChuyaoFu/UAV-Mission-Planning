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

    @property
    def stamp(self):return self.stamp_ns*1e-9
    @property
    def rotation(self):return Rotation.from_quat(self.quaternion)


class BodyHistory:
    def __init__(self,maxlen=400,max_gap_s=.1):
        self.items=deque(maxlen=maxlen);self.max_gap_ns=int(max_gap_s*1e9);self.fault=None;self.frame=None

    def add(self,stamp,p,q,body_velocity,body_gyro,pose_covariance,frame='world'):
        p=vector(p);q=vector(q,4);v=vector(body_velocity);w=vector(body_gyro)
        cov=np.asarray(pose_covariance,float).reshape(6,6)
        if not np.isfinite(cov).all() or (cov.diagonal()<0).any() or abs(np.linalg.norm(q)-1)>.001:
            raise ValueError('Invalid accepted body covariance/orientation')
        error=max(.003,3*math.sqrt(float(cov.diagonal()[:3].max())))
        if error>.15:raise ValueError('Accepted body uncertainty exceeds task budget')
        if self.frame is not None and frame!=self.frame:self.fault='accepted body frame changed';return False
        self.frame=frame
        if self.items:
            old=self.items[-1];dt=(stamp-old.stamp_ns)*1e-9
            if dt==0:return False
            if dt<0:self.fault='accepted body timestamp reset/reordered';return False
            bound=max(2.,np.linalg.norm(old.velocity),np.linalg.norm(v))*dt+.06
            if np.linalg.norm(p-old.position)>bound:self.fault='accepted body pose jumped';return False
        R=Rotation.from_quat(q)
        self.items.append(BodyState(int(stamp),p.copy(),q.copy(),R.apply(v),w.copy(),error));return True

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
                         lerp(a.gyro,b.gyro),max(a.error_m,b.error_m)+.001)


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
        status=self.get('status',now,.5);land=self.get('land',now,.35)
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
                         pose_error_m=state.error_m,tracking_valid=True,calibration_id=self.calibration_id)
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
                     state.velocity.copy(),state.gyro.copy(),epoch,landed,armed,bool(valid),bool(tracking_valid))


class PriorControlLedger:
    """Actual published control certificates, never planned intents.

    Delayed pose feedback is split at measured-body samples AND command issue
    times. Every subinterval needs a certificate issued BEFORE that subinterval.
    A later map/command cannot retroactively bless an already-flown segment.
    """
    def __init__(self,maxlen=64,max_age_s=2.):
        self.items=deque(maxlen=maxlen);self.max_age=max_age_s;self.audited_segments=0

    def add(self,certificate):
        if self.items and certificate.issued<self.items[-1].issued:raise ValueError('Command clock rolled back')
        self.items.append(certificate)
        while len(self.items)>1 and certificate.issued-self.items[1].issued>self.max_age:self.items.popleft()

    def interval(self,start_stamp,end_stamp,history,epoch):
        ledger=self
        class Audit:
            def covers_motion(self,a,b,now,current_epoch):
                if current_epoch!=epoch or end_stamp<start_stamp:return False
                start=history.sample(int(round(start_stamp*1e9)));end=history.sample(int(round(end_stamp*1e9)))
                if start is None or end is None:return False
                if np.linalg.norm(vector(a)-start.position)>.002 or np.linalg.norm(vector(b)-end.position)>.002:return False
                times={start_stamp,end_stamp}
                times.update(s.stamp for s in history.items if start_stamp<s.stamp<end_stamp)
                times.update(c.issued for c in ledger.items if start_stamp<c.issued<end_stamp)
                times=sorted(times)
                for t0,t1 in zip(times,times[1:]):
                    if t1-t0>history.max_gap_ns*1e-9+1e-9:return False
                    old=[c for c in ledger.items if c.issued<=t0+1e-9]
                    if not old:return False
                    cert=old[-1]
                    p=history.sample(int(round(t0*1e9)));q=history.sample(int(round(t1*1e9)))
                    if p is None or q is None or not cert.covers_motion(p.position,q.position,t1,current_epoch):return False
                    ledger.audited_segments+=1
                return True
        return Audit()
