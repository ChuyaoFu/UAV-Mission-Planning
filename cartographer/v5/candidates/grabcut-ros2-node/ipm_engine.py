"""Explicit single-exposure conditional-ground-IPM stream mode; default OFF.
No native pose is projected into the optical floor fit. Hover is allowed.
"""
import time,resource,platform
import numpy as np,cv2
from ground_ipm import infer_ipm,POLICY
from engine import DeadlineExceeded
class IPMEngine:
 def __init__(self,T_imu_from_camera,left_mask,right_mask,lbd_library,max_cache=24,variant='frozen400p',include_floor=False):
  self.infer=infer_ipm;self.include_floor=bool(include_floor)
  if self.include_floor:
   from ground_ipm_floor import infer_ipm as floor_model
   self.infer=floor_model
  self.m0=np.asarray(left_mask,bool).copy();self.m1=np.asarray(right_mask,bool).copy();self.segmenter=None;self.variant=variant;self.last_stamp=None;cv2.setNumThreads(1)
  if self.m0.shape!=(800,1280) or self.m1.shape!=self.m0.shape:raise ValueError('actual native masks required')
  if variant=='cold5_freeze1_seedvalidated_240p':
   if not self.include_floor:raise ValueError('240p release is explicit IPM+floor policy')
   from optimized_segment_240 import MeasuredSeedGrabCutValidatedReuse240
   from ground_ipm_floor_240 import infer_ipm as floor240_model
   self.segmenter=MeasuredSeedGrabCutValidatedReuse240();self.infer=floor240_model
  elif variant=='cold5_freeze1_seedvalidated_400p':
   from optimized_segment_validated import MeasuredSeedGrabCutValidatedReuse
   self.segmenter=MeasuredSeedGrabCutValidatedReuse()
  elif variant=='cold5_freeze1_400p':
   from optimized_segment import MeasuredSeedGrabCutReuse
   self.segmenter=MeasuredSeedGrabCutReuse()
  elif variant!='frozen400p':raise ValueError('explicit known IPM segmenter policy required')
 def evaluate(self,job,max_compute_s=.4):
  start=time.monotonic();deadline=start+max_compute_s if max_compute_s is not None else float('inf');stats=dict(status='starting',model_kind='ground_ipm',engine_variant=self.variant,opencv=cv2.__version__,ipm_policy=POLICY.copy(),native_joint_required=False,temporal_baseline_required=False)
  def check():
   if time.monotonic()>deadline:raise DeadlineExceeded('compute budget expired; no inference output')
  try:
   frame=max(job['frames'],key=lambda f:int(f['stamp_ns']));s=int(frame['stamp_ns'])
   if any(int(frame[k])!=s for k in ('left_stamp_ns','right_stamp_ns','depth_stamp_ns')):raise ValueError('exact original stereo/depth exposure required')
   if self.last_stamp is not None and s<self.last_stamp and self.segmenter is not None:self.segmenter.reset()
   self.last_stamp=s;packet,meta=self.infer(frame,self.m0,self.m1,self.segmenter,check);stats.update(meta);check()
   if packet is not None:
    packet['candidate_free_prefix_endpoint_z_m']=np.full(frame['left'].shape,np.nan,np.float32);packet['support_stamps_ns']=np.array([s],np.int64);packet['model_exposure_stamp_ns']=s;packet['native_joint_used']=False;packet['model_kind']='ground_ipm';stats['current_exposure_stamp_ns']=s
   return packet,stats
  except DeadlineExceeded as e:stats.update(status='deadline_drop',reason=str(e));return None,stats
  except (ValueError,KeyError,TypeError,RuntimeError,cv2.error) as e:stats.update(status='evidence_rejected',reason=str(e));return None,stats
  finally:
   if self.segmenter is not None:stats['appearance_models']={side:dict(ready=state.background is not None,support_stamp_ns=state.support_stamp_ns,last_validated_stamp_ns=getattr(state,'last_validated_stamp_ns',None)) for side,state in self.segmenter.states.items()}
   stats['compute_s']=time.monotonic()-start;stats['peak_process_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1024 if platform.system()=='Linux' else 1)
