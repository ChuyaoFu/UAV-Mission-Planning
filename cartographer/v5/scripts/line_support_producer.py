"""Default-off producer of native EGO_LINE_SUPPORT_V1 records (candidate, not wired).

Converts exactly one measured, stereo-supported floor segment of one
/nav/guideline_hint result into one native line-support record. Pure: no ROS,
no map, no planner and no clock reads; the caller passes the native-clock
``now_ns`` it would publish at.

Deliberately NOT done here:
- choosing which detected segment is the task line. The caller passes the
  indices its own measured-line filter accepted (for example
  measured_line_reference's ``source_segment_indices``). Detected segments are
  not all the task line (trial 45 at 62.0 s: transverse bright edges).
- extending, merging or extrapolating endpoints. The native domain is the
  finite measured segment; a reference's axis extension or lateral approach
  leg is not support.
- inventing or re-stamping times. Exposure and ToF stamps must come from the
  hint's versioned ``conditional_support`` sub-record (a proposed
  guideline_node extension; the current hint lacks it, so nothing is produced).
- setting corridor half-width or z bounds. Those are native launch parameters
  with no default and are never seen by this module.
- task-phase / health gating. The caller passes ``allowed=True`` only when its
  own policy permits support; default is no record.

Native acceptance remains authoritative (epoch/frame/declaration match, input
age, order, raw-stream freshness, unknown-only, raw-obstacle precedence).
"""
from dataclasses import dataclass
import math,re

PREFIX='EGO_LINE_SUPPORT_V1'
HINT_SCHEMA='guideline_conditional_support_v1'
# Same literal relative_guideline.GuidelineDetector writes for stereo-floor segments.
MEASURED_SOURCE='grayscale line with measured stereo floor support'
TOKEN=re.compile(r'[A-Za-z0-9_.:-]{1,128}\Z',re.ASCII)
INT64_MAX=(1<<63)-1
UINT64_MAX=(1<<64)-1
# Native/guideline_node ToF association bound (guideline_node drain(): .08 s). Not new evidence.
TOF_SKEW_NS=80000000
# Native LineSupport caps the configured input age at 500 ms.
INPUT_AGE_CEILING_NS=500000000
# Same minimum stereo support as measured_line_reference.short_forward_reference.
MIN_SUPPORT_POINTS=8
WIRE_LIMIT=1024


def _integer(value):
    return isinstance(value,int) and not isinstance(value,bool)


@dataclass(frozen=True)
class ProducerConfig:
    """``epoch``/``frame``/``declaration`` must equal the native launch values
    exactly; ``input_age_ns`` must equal native ``inferred_free_input_max_age_s``."""
    enabled:bool=False
    epoch:str=''
    frame:str=''
    declaration:str=''
    input_age_ns:int=0

    def __post_init__(self):
        if not isinstance(self.enabled,bool):raise ValueError('enabled must be bool')
        if not self.enabled:return
        for name in ('epoch','frame','declaration'):
            value=getattr(self,name)
            if not isinstance(value,str) or not TOKEN.fullmatch(value):
                raise ValueError(f'Enabled producer requires explicit {name} token matching native')
        if not _integer(self.input_age_ns) or not 0<self.input_age_ns<=INPUT_AGE_CEILING_NS:
            raise ValueError('Enabled producer requires the existing native input age (0, 500ms]')


@dataclass(frozen=True)
class Result:
    wire:object
    reason:str
    metadata:dict


class LineSupportProducer:
    """State: last emitted sequence/exposure and the latched home epoch only.

    ``sequence_start`` defaults to 0. A restarted producer talking to a still
    running native map is then rejected by native ordering until restarted
    together (fail-closed). Pass a larger seed (e.g. time.monotonic_ns()) only
    if availability across producer restarts is wanted.
    """
    def __init__(self,config,sequence_start=0):
        if not isinstance(config,ProducerConfig):raise ValueError('ProducerConfig required')
        if not _integer(sequence_start) or not 0<=sequence_start<UINT64_MAX:raise ValueError('Invalid sequence seed')
        self.config=config;self.sequence=sequence_start;self.last_exposure_ns=-1
        self.home_epoch_ns=None;self.fault=None;self.emitted=0;self.refused=0

    def _refuse(self,reason,**extra):
        self.refused+=1
        return Result(None,reason,dict(extra,emitted=False))

    def record(self,hint,segment_indices,*,now_ns,allowed,home_epoch_ns):
        c=self.config
        if not c.enabled:return Result(None,'disabled',dict(emitted=False))
        if self.fault:return self._refuse(self.fault)
        if not _integer(now_ns) or not 0<=now_ns<=INT64_MAX:return self._refuse('invalid native clock')
        if allowed is not True:return self._refuse('caller gate closed')
        if not _integer(home_epoch_ns):return self._refuse('actual home epoch unavailable')
        if self.home_epoch_ns is None:self.home_epoch_ns=home_epoch_ns
        elif home_epoch_ns!=self.home_epoch_ns:
            self.fault='actual home epoch changed; restart producer and native with a new epoch'
            return self._refuse(self.fault)
        if not isinstance(hint,dict):return self._refuse('hint must be a decoded guideline result')
        sub=hint.get('conditional_support')
        if not isinstance(sub,dict) or sub.get('schema')!=HINT_SCHEMA:
            return self._refuse('hint lacks versioned conditional_support timing/frame (guideline_node extension required)')
        if sub.get('frame_id')!=c.frame:return self._refuse('hint frame differs from native map frame')
        exposure,tof=sub.get('exposure_stamp_ns'),sub.get('tof_stamp_ns')
        if not all(_integer(t) and 0<=t<=INT64_MAX for t in (exposure,tof)):return self._refuse('invalid raw source stamps')
        try:stamp=float(hint.get('stamp'))
        except (TypeError,ValueError):return self._refuse('hint stamp missing')
        if not math.isfinite(stamp) or abs(stamp-exposure*1e-9)>1e-6:return self._refuse('hint stamp does not match exposure_stamp_ns')
        if abs(exposure-tof)>TOF_SKEW_NS:return self._refuse('ToF association outside existing 80ms bound')
        if now_ns<exposure or now_ns<tof:return self._refuse('source stamp later than native clock')
        if now_ns-exposure>c.input_age_ns or now_ns-tof>c.input_age_ns:return self._refuse('source older than native input age')
        if exposure<=self.last_exposure_ns:return self._refuse('exposure not newer than last emitted record')
        floor=hint.get('floor_z_measured')
        if not isinstance(floor,(int,float)) or isinstance(floor,bool) or not math.isfinite(floor):return self._refuse('measured floor_z missing')
        floor=float(floor)
        segments=hint.get('segments')
        if not isinstance(segments,list):return self._refuse('hint segments missing')
        try:indices=list(segment_indices)
        except TypeError:return self._refuse('caller-selected segment indices required')
        if not indices or len(set(indices))!=len(indices) or not all(_integer(i) and 0<=i<len(segments) for i in indices):
            return self._refuse('caller-selected segment indices invalid')
        candidates=[]
        for i in indices:
            s=segments[i]
            if not isinstance(s,dict) or s.get('source')!=MEASURED_SOURCE:return self._refuse('selected segment is not a measured stereo-floor segment',index=i)
            e=s.get('endpoints');support=s.get('support_points')
            try:points=[[float(v) for v in p] for p in e]
            except (TypeError,ValueError):return self._refuse('selected segment endpoints invalid',index=i)
            if len(points)!=2 or any(len(p)!=3 for p in points) or not all(math.isfinite(v) for p in points for v in p):
                return self._refuse('selected segment endpoints invalid',index=i)
            if any(abs(p[2]-floor)>1e-9 for p in points):return self._refuse('selected segment is not on the measured floor',index=i)
            if not _integer(support) or support<MIN_SUPPORT_POINTS:return self._refuse('selected segment stereo support below existing minimum',index=i)
            length=math.hypot(points[1][0]-points[0][0],points[1][1]-points[0][1])
            if not math.isfinite(length) or length<=0.:return self._refuse('selected segment has no horizontal extent',index=i)
            candidates.append((length*support,-i,i,points,length,support))
        # Native accepts one record per exposure: keep the strongest selected measured
        # segment (same length*support weighting as the reference); never merge.
        _,_,chosen,points,length,support=max(candidates)
        sequence=self.sequence+1
        if sequence>UINT64_MAX:return self._refuse('sequence exhausted')
        ax,ay=points[0][0],points[0][1];bx,by=points[1][0],points[1][1]
        wire=' '.join([PREFIX,c.epoch,c.frame,c.declaration,str(sequence),str(exposure),str(tof),
                       *(format(v,'.17g') for v in (ax,ay,bx,by,floor))])
        if len(wire)>WIRE_LIMIT or not wire.isascii():return self._refuse('record exceeds native wire limit')
        self.sequence=sequence;self.last_exposure_ns=exposure;self.emitted+=1
        return Result(wire,'emitted',dict(
            emitted=True,sequence=sequence,exposure_ns=exposure,tof_ns=tof,
            chosen_segment_index=chosen,selected_segment_indices=indices,
            segment_xy_length_m=length,support_points=support,floor_z=floor,
            home_epoch_ns=self.home_epoch_ns,epoch=c.epoch,frame=c.frame,declaration=c.declaration,
            domain_dimensions='native launch parameters; not set or seen by producer',
            endpoints_are_measured_segment=True,extrapolated=False,
            free_space_authorization='native conditional support only; raw obstacles and Python guard still veto'))


def parse(wire):
    """Mirror of the native 12-token grammar, for logs and tests only."""
    if not isinstance(wire,str) or not wire.isascii() or len(wire)>WIRE_LIMIT:raise ValueError('Invalid ASCII record')
    t=wire.split()
    if len(t)!=12 or t[0]!=PREFIX:raise ValueError('Twelve-token record required')
    if not all(TOKEN.fullmatch(x) for x in t[1:4]):raise ValueError('Invalid identifier token')
    if not all(re.fullmatch(r'[0-9]+',x,re.ASCII) for x in t[4:7]):raise ValueError('Invalid unsigned token')
    sequence,exposure,tof=(int(x) for x in t[4:7])
    if not 0<sequence<=UINT64_MAX or exposure>INT64_MAX or tof>INT64_MAX:raise ValueError('Integer outside native range')
    values=[float(x) for x in t[7:]]
    if not all(math.isfinite(v) for v in values):raise ValueError('Non-finite geometry')
    return dict(epoch=t[1],frame=t[2],declaration=t[3],sequence=sequence,exposure_ns=exposure,tof_ns=tof,
                ax=values[0],ay=values[1],bx=values[2],by=values[3],floor_z=values[4])
