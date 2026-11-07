"""Pure sender-side state for the frozen EGO_ENVELOPE_V1 protocol."""
from dataclasses import dataclass
from collections import Counter
import math,re

PREFIX='EGO_ENVELOPE_V1'
CEILING_NS=100000000
INT64_MAX=(1<<63)-1
UINT64_MAX=(1<<64)-1
EPOCH=re.compile(r'[A-Za-z0-9_.:-]{1,128}\Z',re.ASCII)
UNSIGNED=re.compile(r'[0-9]+\Z',re.ASCII)
DECIMAL=re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z',re.ASCII)


def floor_ns(seconds):
    """Floor the supplied float deadline exactly; never round it later."""
    value=float(seconds)
    if not math.isfinite(value) or value<0:raise ValueError('Invalid actual deadline')
    numerator,denominator=value.as_integer_ratio()
    result=numerator*1000000000//denominator
    if result>INT64_MAX:raise ValueError('Deadline outside native int64 clock')
    return result


def extra_xyz(extra):
    try:values=tuple(float(v) for v in extra)
    except TypeError:values=(float(extra),)*3
    if len(values)!=3 or not all(math.isfinite(v) and 0<=v<=1 for v in values):
        raise ValueError('Actual admitted guard.extra XYZ required')
    return values


@dataclass(frozen=True)
class Envelope:
    epoch:str
    sequence:int
    issued_ns:int
    valid_until_ns:int
    extra:tuple

    def __post_init__(self):
        if not isinstance(self.epoch,str) or not EPOCH.fullmatch(self.epoch):raise ValueError('Invalid body-session epoch')
        if not isinstance(self.sequence,int) or isinstance(self.sequence,bool) or not 0<self.sequence<=UINT64_MAX:raise ValueError('Invalid sequence')
        if any(not isinstance(t,int) or isinstance(t,bool) or not 0<=t<=INT64_MAX for t in (self.issued_ns,self.valid_until_ns)):raise ValueError('Invalid native nanosecond clock')
        if not 0<self.valid_until_ns-self.issued_ns<=CEILING_NS:raise ValueError('Envelope exceeds actual 100ms ceiling')
        object.__setattr__(self,'extra',extra_xyz(self.extra))

    def wire(self):
        return ' '.join([PREFIX,self.epoch,str(self.sequence),str(self.issued_ns),str(self.valid_until_ns),
                         *(format(v,'.17g') for v in self.extra)])

    def metadata(self):
        return dict(epoch=self.epoch,sequence=self.sequence,issued_ns=self.issued_ns,
                    valid_until_ns=self.valid_until_ns,extra_xyz_m=list(self.extra))

    @classmethod
    def parse(cls,wire):
        if not isinstance(wire,str) or not wire.isascii() or len(wire)>512:raise ValueError('Invalid ASCII record')
        tokens=wire.split()
        if len(tokens)!=8 or tokens[0]!=PREFIX:raise ValueError('Eight-token atomic record required')
        if not all(UNSIGNED.fullmatch(t) for t in tokens[2:5]) or not all(DECIMAL.fullmatch(t) for t in tokens[5:]):raise ValueError('Invalid numeric token')
        return cls(tokens[1],int(tokens[2]),int(tokens[3]),int(tokens[4]),tuple(map(float,tokens[5:])))


class EnvelopeSender:
    """No ROS, planner, map mutations or extra padding; records are immutable."""
    def __init__(self,epoch):
        if not isinstance(epoch,str) or not EPOCH.fullmatch(epoch):raise ValueError('Invalid body-session epoch')
        self.epoch=epoch;self.sequence=0;self.sent={};self.steady_deadlines={}
        self.latest_sent=None;self.latest_ack=None;self.ack_floor=0
        self.last_ros_ns=-1;self.last_steady_ns=-1;self.fault=None;self.reason='waiting for native ACK'
        self.counts=Counter(sent=0,accepted=0,rejected=0,expired=0,goal_wait=0)
        self.rejections=Counter()

    def _observe(self,now_ns,steady_ns):
        if any(not isinstance(t,int) or isinstance(t,bool) or t<0 for t in (now_ns,steady_ns)) or now_ns>INT64_MAX:
            self.fault='invalid sender clock'
        elif now_ns<self.last_ros_ns or steady_ns<self.last_steady_ns:
            self.fault='sender clock regressed; new process/session required'
        if self.fault:self.reason=self.fault;return False
        self.last_ros_ns=now_ns;self.last_steady_ns=steady_ns
        for sequence,record in list(self.sent.items()):
            if now_ns>=record.valid_until_ns or steady_ns>=self.steady_deadlines[sequence]:
                self.counts['expired']+=1
                # An unacknowledged later record could already have closed or
                # superseded the native gate. Never fall back behind it.
                self.ack_floor=max(self.ack_floor,sequence)
                del self.sent[sequence];del self.steady_deadlines[sequence]
        return True

    def expire(self,now_ns,steady_ns):
        return self._observe(now_ns,steady_ns)

    def issue(self,extra,now_ns,valid_until_ns,steady_ns):
        if not self._observe(now_ns,steady_ns):raise ValueError(self.fault)
        if self.sequence==UINT64_MAX:raise ValueError('Native sequence exhausted; restart required')
        record=Envelope(self.epoch,self.sequence+1,now_ns,valid_until_ns,extra_xyz(extra))
        self.sequence=record.sequence;self.sent[record.sequence]=record
        self.steady_deadlines[record.sequence]=steady_ns+(valid_until_ns-now_ns)
        self.latest_sent=record;self.counts['sent']+=1
        return record

    def reject(self,reason):
        self.reason=reason;self.counts['rejected']+=1;self.rejections[reason]+=1
        return False

    def accept_ack(self,wire,now_ns,steady_ns):
        if not self._observe(now_ns,steady_ns):return self.reject(self.fault)
        try:record=Envelope.parse(wire)
        except (ValueError,OverflowError):return self.reject('malformed native ACK')
        expected=self.sent.get(record.sequence)
        if record.epoch!=self.epoch:return self.reject('wrong ACK epoch')
        if not record.issued_ns<=now_ns<record.valid_until_ns:return self.reject('expired or future native ACK')
        if expected is None:return self.reject('ACK does not identify a live sent record')
        if record!=expected:return self.reject('ACK full payload mismatch')
        if record.sequence<=self.ack_floor:return self.reject('ACK behind expired possible native state')
        if self.latest_ack is not None and record.sequence<=self.latest_ack.sequence:return self.reject('duplicate or old native ACK')
        self.latest_ack=record;self.counts['accepted']+=1;self.reason='accepted actual native envelope ACK'
        return True

    def ready(self,extra,now_ns,steady_ns):
        if not self._observe(now_ns,steady_ns):return False
        need=extra_xyz(extra);ack=self.latest_ack
        if ack is None or ack.sequence not in self.sent or ack.sequence<=self.ack_floor:
            self.reason='no live matching native ACK';return False
        if not all(have>=want for have,want in zip(ack.extra,need)):
            self.reason='ACK extra does not cover current guard';return False
        # Any newer sent packet might already be active despite ACK delivery
        # delay. Its actual XYZ must cover this guard too; no invented padding.
        if any(not all(have>=want for have,want in zip(r.extra,need))
               for r in self.sent.values() if r.sequence>ack.sequence):
            self.reason='unacknowledged possible native extra is insufficient';return False
        self.reason='live ACK and possible newer native records cover current guard'
        return True

    def goal_wait(self,reason=None):
        self.counts['goal_wait']+=1
        if reason is not None:self.reason=reason

    def report(self):
        return dict(self.counts,epoch=self.epoch,fault=self.fault,reason=self.reason,
                    rejections=dict(self.rejections),live_sent_records=len(self.sent),
                    latest_sent=None if self.latest_sent is None else self.latest_sent.metadata(),
                    latest_ack=None if self.latest_ack is None else self.latest_ack.metadata(),
                    latest_ack_live=bool(self.latest_ack is not None and self.latest_ack.sequence in self.sent and self.latest_ack.sequence>self.ack_floor),
                    expired_ack_floor_sequence=self.ack_floor)
