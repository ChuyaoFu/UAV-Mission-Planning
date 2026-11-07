"""Bounded snapshot engine. Native OpenCV executes only in its isolated process.
No transport publishers, map updates, depth fabrication or occupied outputs.
"""
from collections import OrderedDict
import time,math,resource,platform
import numpy as np,cv2
from scipy.spatial.transform import Rotation
from native_relative_clone_covariance import decode,rigid
from temporal_boundary import LBD,triangulate,floor_validate,raw_contradictions
from seeded_grabcut import segment,POLICY
from cut_lines import extract
from conditional_model import surface

class DeadlineExceeded(RuntimeError):pass

def poses_from_joint(record,TIC):
 pairs=decode(record);poses={};residual=0.
 for pair in pairs:
  for t,p,q in ((pair.camera_time0_s,pair.position0,pair.quaternion0),(pair.camera_time1_s,pair.position1,pair.quaternion1)):
   s=int(round(t*1e9));residual=max(residual,abs(t*1e9-s));R=Rotation.from_quat(q).as_matrix();W=np.eye(4);W[:3,:3]=R@TIC[:3,:3];W[:3,3]=p+R@TIC[:3,3];poses[s]=W
 if residual>1.:raise ValueError('native floating camera-time/original image-header mismatch >1ns')
 return poses,int(round(pairs[0].camera_time1_s*1e9)),pairs

class Engine:
 def __init__(self,T_imu_from_camera,left_mask,right_mask,lbd_library,max_cache=24,variant='frozen400p'):
  self.variant=variant;self.segmenter=None
  if variant=='cold5_freeze1_400p':
   from optimized_segment import MeasuredSeedGrabCutReuse
   self.segmenter=MeasuredSeedGrabCutReuse()
  elif variant=='cold5_freeze1_seedvalidated_400p':
   from optimized_segment_validated import MeasuredSeedGrabCutValidatedReuse
   self.segmenter=MeasuredSeedGrabCutValidatedReuse()
  elif variant!='frozen400p':raise ValueError('unknown explicitly versioned engine variant')
  self.TIC=rigid(T_imu_from_camera).copy();self.m0=np.asarray(left_mask,bool).copy();self.m1=np.asarray(right_mask,bool).copy();self.lbd=LBD(lbd_library);self.cache=OrderedDict();self.max_cache=max_cache;self.last_stamp=None;cv2.setNumThreads(1)
  if self.m0.shape!=(800,1280) or self.m1.shape!=self.m0.shape or not 3<=max_cache<=24:raise ValueError('frozen model requires actual1280x800 masks and bounded cache')
  try:
   from threadpoolctl import threadpool_limits
   self.threadpool_limit=threadpool_limits(limits=1)
  except ImportError:self.threadpool_limit=None
 def evaluate(self,job,max_compute_s=.4):
  started=time.monotonic();deadline=started+max_compute_s if max_compute_s is not None else float('inf');stats=dict(status='starting',engine_variant=self.variant,support_view_policy='current+cachedpreferredmax3' if self.segmenter else 'max6metricdistinct',segmentation_cache_hits=0,segmentation_cache_misses=0,opencv=cv2.__version__,lbd_opencv=self.lbd.version,blas_threads_limited=self.threadpool_limit is not None,segmentation_policy=POLICY.copy())
  def check():
   if time.monotonic()>deadline:raise DeadlineExceeded('compute budget expired; no inference output')
  try:
   poses,current,pairs=poses_from_joint(job['joint'],self.TIC);frames={int(f['stamp_ns']):f for f in job['frames']};available=sorted(set(poses)&set(frames));check()
   if current not in frames:raise ValueError('exact current posterior exposure unavailable; no future means for older images')
   if self.last_stamp is not None and current<self.last_stamp:
    self.cache.clear()
    if self.segmenter is not None:self.segmenter.reset()
   self.last_stamp=current
   for s in list(self.cache):
    if current-s>1_500_000_000 or s>current:self.cache.pop(s,None)
   obs=[];selected=[]
   for s in available:
    f=frames[s]
    if np.asarray(f['K']).shape!=(4,) or not np.isfinite(f['K']).all() or min(f['K'][:2])<=0 or not np.isfinite(f['baseline']) or not 0<float(f['baseline'])<=.5:raise ValueError('invalid actual native intrinsics/baseline')
    if not np.array_equal(f['K'],frames[current]['K']) or float(f['baseline'])!=float(frames[current]['baseline']):raise ValueError('calibration changed within native clone window')
    if any(int(f[k])!=s for k in ('left_stamp_ns','right_stamp_ns','depth_stamp_ns')):raise ValueError('left/right/depth headers must match exactly')
    if f['left'].dtype!=np.uint8 or f['right'].dtype!=np.uint8 or f['measured_stereo'].dtype!=np.float32:raise ValueError('original mono8 and float32 raw depth required')
    if f['left'].shape!=self.m0.shape or f['right'].shape!=self.m1.shape or f['measured_stereo'].shape!=self.m0.shape:raise ValueError('actual native masks/images/depth mismatch')
    if len(selected)==0 or all(np.linalg.norm(poses[s][:3,3]-poses[t][:3,3])>=.01 for t in selected):selected.append(s)
   if current not in selected:selected.append(current)
   # Selection is bounded and sensor-motion-only; no episode windows/commands.
   if self.segmenter is not None:
    pool=list(selected);chosen=[current]
    while len(chosen)<3:
     candidates=[s for s in pool if s not in chosen and all(np.linalg.norm(poses[s][:3,3]-poses[t][:3,3])>=.01 for t in chosen)]
     if not candidates:break
     cached=[s for s in candidates if s in self.cache];candidate=max(cached or candidates,key=lambda s:min(np.linalg.norm(poses[s][:3,3]-poses[t][:3,3]) for t in chosen));chosen.append(candidate)
    selected=sorted(chosen)
   elif len(selected)>6:selected=[selected[i] for i in np.unique(np.linspace(0,len(selected)-1,6,dtype=int))]
   if len(selected)<3:raise ValueError('insufficient actual metric-distinct exposures')
   features={}
   # Warm the CURRENT exposure first; a deadline drop never blocks raw traffic.
   for s in sorted(selected,reverse=True):
    check();f=frames[s]
    if s in self.cache:cached=self.cache[s];stats['segmentation_cache_hits']+=1
    else:
     seg,detail,_=(self.segmenter(f,self.m0,'left',current) if self.segmenter is not None else segment(f,self.m0))
     if seg is None:raise ValueError('missing actual geometric seed class')
     lines=extract(seg)
     if not lines:raise ValueError('no connected floor/nonfloor cut boundary')
     line=np.asarray(lines[0]['line'],np.float32);h,w=f['left'].shape
     if min(line[[0,2]])<32 or max(line[[0,2]])>=w-32 or min(line[[1,3]])<32 or max(line[[1,3]])>=h-32:raise ValueError('selected boundary lacks native LBD/border support')
     descriptor,valid=self.lbd.describe(f['left'],line.reshape(1,4))
     if not valid[0]:raise ValueError('native LBD descriptor invalid')
     cached=dict(line=line,descriptor=descriptor[0].copy(),segment=dict(label400=seg['label400'],label_floor_nonfloor_unknown=seg['label_floor_nonfloor_unknown']),detail=detail);self.cache[s]=cached;stats['segmentation_cache_misses']+=1
     while len(self.cache)>self.max_cache:self.cache.popitem(last=False)
     check()
    features[s]=cached
   bystamp={s:dict(frames[s],W=poses[s],self_mask=self.m0) for s in selected}
   for i,s in enumerate(sorted(selected)):obs.append(dict(stamp_ns=s,frame_index=i,line_index=0,line=features[s]['line'],K=frames[s]['K'],W=poses[s]))
   # LBD describes real native pixels. Hamming is diagnostic; the frozen model's
   # actual association remains longest observed boundary + full metric checks.
   ordered=sorted(selected);stats['lbd_consecutive_hamming']=[int(np.unpackbits(features[b]['descriptor']^features[a]['descriptor']).sum()) for a,b in zip(ordered,ordered[1:])];tri=triangulate(obs);stats['triangulation']=tri;check()
   if tri['status']!='metric_temporal_line':raise ValueError(tri['status'])
   floor=floor_validate(tri,bystamp);stats['floor_association']=floor
   if floor['status']!='tof_gravity_consistent_horizontal_junction_hypothesis':raise ValueError(floor['status'])
   checks=[dict(stamp_ns=s,**raw_contradictions(tri,floor,bystamp[s])) for s in tri['selected_stamps']];stats['raw_crosschecks']=checks;check()
   if any(c['raw_farther_contradictions']>=3 for c in checks):raise ValueError('actual farther raw contradiction')
   packet,detail=surface(frames[current],features[current]['segment'],poses[current],tri,floor,self.m0,self.m1,check,self.segmenter,current);check();packet['source_native_joint_record']=np.asarray(job['joint']).copy();packet['observed_metric_line_endpoints_world']=np.array(tri['endpoints_world']);packet['same_posterior_latest_stamp_ns']=current;stats.update(detail);stats['status']='conditional_candidate';stats['current_exposure_stamp_ns']=current;stats['common_observed_span_m']=tri['quality']['common_span_m'];return packet,stats
  except DeadlineExceeded as e:stats['status']='deadline_drop';stats['reason']=str(e);return None,stats
  except (ValueError,KeyError,TypeError,IndexError,RuntimeError,cv2.error) as e:stats['status']='evidence_rejected';stats['reason']=str(e);return None,stats
  finally:
   stats['compute_s']=time.monotonic()-started;stats['peak_process_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1024 if platform.system()=='Linux' else 1);stats['cached_segmentations']=len(self.cache)
