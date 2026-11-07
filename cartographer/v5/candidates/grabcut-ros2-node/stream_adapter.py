"""Default-OFF latest-only adapter. Raw callbacks never run native inference.
Process isolation contains native crashes/deadlines. One job in flight; no backlog.
This module never publishes ROS topics, especially /nav/depth_evidence.
"""
from collections import OrderedDict
import multiprocessing as mp
import threading,time,queue,copy
import numpy as np

FLOOR_FIELDS=('gravity_up_optical','gravity_stamp_ns','tof_range_m','tof_stamp_ns','tof_valid','tof_origin_optical_m','tof_direction_optical')

def _worker(jobs,results,busy,config):
 if config['model_kind']=='ground_ipm':
  from ipm_engine import IPMEngine as Engine
 else:
  from engine import Engine
 try:engine=Engine(config['TIC'],config['left_mask'],config['right_mask'],config['lbd_library'],config['max_cached_frames'],config['engine_variant'],include_floor=config['include_floor_completion']) if config['model_kind']=='ground_ipm' else Engine(config['TIC'],config['left_mask'],config['right_mask'],config['lbd_library'],config['max_cached_frames'],config['engine_variant'])
 except BaseException as e:
  try:results.put_nowait(dict(packet=None,stats=dict(status='worker_initialization_failed',reason=repr(e))))
  except queue.Full:pass
  busy.value=0;return
 while True:
  job=jobs.get()
  if job is None:return
  start=time.monotonic();queue_delay_s=(time.time_ns()-job['submitted_wall_ns'])*1e-9
  try:
   packet,stats=engine.evaluate(job,config['max_compute_s']);stats['job_input_ros_age_s']=job['input_ros_age_s'];stats['input_to_worker_delay_s']=queue_delay_s;stats['worker_queue_delay_s']=queue_delay_s;stats['submit_to_compute_finish_s']=(time.time_ns()-job['submitted_wall_ns'])*1e-9;output=dict(packet=packet,stats=stats,source_health=job['source_health'],original_header=job['original_header'],submitted_monotonic=job['submitted_monotonic'],finished_monotonic=time.monotonic())
   try:results.put_nowait(output)
   except queue.Full:
    try:results.get_nowait()
    except queue.Empty:pass
    try:results.put_nowait(output)
    except queue.Full:pass
  except BaseException as e:
   try:results.put_nowait(dict(packet=None,stats=dict(status='worker_exception',reason=repr(e))))
   except queue.Full:pass
  finally:busy.value=0

class GrabCutLatestAdapter:
 def __init__(self,T_imu_from_camera,left_self_mask,right_self_mask,lbd_library,*,enabled=False,permit_conditional_free=False,clock_ns=None,health_ok_now=None,max_rate_hz=2.,max_compute_s=.4,max_input_age_s=.5,max_cached_frames=24,cache_horizon_s=1.5,max_ray_m=2.5,voxel_resolution_m=.1,consumer_reserve_credit_m=0.,consumer_quantization_verified=False,engine_variant='frozen400p',model_kind='temporal',include_floor_completion=False):
  if model_kind not in ('temporal','ground_ipm'):raise ValueError('explicit known model kind required')
  if not(0<max_rate_hz<=2.5 and 0<max_compute_s<=.5 and 0<max_input_age_s<=.5 and 3<=max_cached_frames<=24 and .3<=cache_horizon_s<=1.5):raise ValueError('bounded latest-worker policy required')
  if not(0<max_ray_m<=2.5 and .01<=voxel_resolution_m<=.5):raise ValueError('invalid native ray/voxel range')
  maximum_credit=np.sqrt(3.)*voxel_resolution_m
  if not(0<=consumer_reserve_credit_m<=maximum_credit+1e-12) or (consumer_reserve_credit_m>0 and not consumer_quantization_verified):raise ValueError('consumer credit requires verified actual mapper subtraction')
  if left_self_mask is None or right_self_mask is None:raise ValueError('both actual native self masks required')
  lm=np.asarray(left_self_mask);rm=np.asarray(right_self_mask)
  if lm.dtype!=bool or rm.dtype!=bool or lm.shape!=(800,1280) or rm.shape!=lm.shape:raise ValueError('actual bool masks at native resolution required')
  self.config=dict(TIC=np.array(T_imu_from_camera,float),left_mask=lm.copy(),right_mask=rm.copy(),lbd_library=str(lbd_library),max_cached_frames=max_cached_frames,max_compute_s=max_compute_s,engine_variant=engine_variant,model_kind=model_kind,include_floor_completion=bool(include_floor_completion));self.enabled=bool(enabled);self.permit_conditional_free=bool(permit_conditional_free);self.clock_ns=clock_ns;self.health_ok_now=health_ok_now;self.max_age=max_input_age_s;self.rate=max_rate_hz;self.horizon_ns=int(cache_horizon_s*1e9);self.maximum=max_cached_frames;self.max_ray=max_ray_m;self.reserve=np.sqrt(3.)*voxel_resolution_m+.1;self.credit=consumer_reserve_credit_m;self.raynorm_cache=None;self.frames=OrderedDict();self.joint=None;self.joints=OrderedDict();self.lock=threading.Lock();self.last_submit=-float('inf');self.last_submitted_stamp=None;self.process=None;self.scheduler=None;self.stop_event=threading.Event();self.ctx=mp.get_context('spawn');self.jobs=self.ctx.Queue(maxsize=1);self.results=self.ctx.Queue(maxsize=1);self.busy=self.ctx.Value('b',0,lock=False);self.last_rejection='disabled';self.stats=dict(offered=0,offer_dropped=0,submitted=0,age_dropped=0,worker_restarts_required=0)
 def start(self):
  if not self.enabled:return False
  if self.process is not None:return self.process.is_alive()
  self.process=self.ctx.Process(target=_worker,args=(self.jobs,self.results,self.busy,self.config),daemon=True);self.process.start();self.scheduler=threading.Thread(target=self._schedule,daemon=True);self.scheduler.start();return True
 def offer_exposure(self,frame,*,original_header=None,pose_health_ok=None):
  """O(1), nonwaiting admission. Caller transfers immutable owned arrays.
  left/right/raw and all sensor metadata retain ORIGINAL native headers/units.
  The existing raw publisher proceeds independently, whether admission succeeds.
  """
  if not self.enabled:return False
  if not self.lock.acquire(False):self.stats['offer_dropped']+=1;return False
  try:
   s=int(frame['stamp_ns'])
   if any(int(frame[k])!=s for k in ('left_stamp_ns','right_stamp_ns','depth_stamp_ns')):self.last_rejection='exact_exposure_mismatch';return False
   if frame.get('left_frame_id')!=frame.get('depth_frame_id') or not frame.get('depth_frame_id'):self.last_rejection='reference_frame_mismatch';return False
   for k in ('left','right','measured_stereo'):
    a=np.asarray(frame[k])
    if a.flags.writeable:raise ValueError('admission requires immutable owned image arrays; no hidden callback copy')
   for k in FLOOR_FIELDS:
    if k not in frame:raise ValueError('missing actual '+k)
   if self.frames and s<next(reversed(self.frames)):self.frames.clear();self.joint=None;self.joints.clear();self.last_submitted_stamp=None
   if s in self.frames:return False
   snapshot=dict(frame)
   for k in ('K','gravity_up_optical','tof_origin_optical_m','tof_direction_optical'):
    snapshot[k]=np.array(frame[k],copy=True);snapshot[k].setflags(write=False)
   self.frames[s]=dict(data=snapshot,header=copy.deepcopy(original_header),health=pose_health_ok)
   for old in list(self.frames):
    if old<s-self.horizon_ns:self.frames.pop(old,None)
   while len(self.frames)>self.maximum:self.frames.popitem(last=False)
   self.stats['offered']+=1;return True
  finally:self.lock.release()
 def offer_native_joint(self,record):
  if not self.enabled:return False
  a=np.asarray(record)
  if a.ndim!=1 or len(a)>1603 or a.flags.writeable:raise ValueError('actual bounded immutable native joint record required')
  if not self.lock.acquire(False):return False
  try:
   if len(a)<163 or a.dtype!=np.float64 or a[0]!=1 or not np.isfinite(a[4]):raise ValueError('native schema1 float64 required')
   s=int(round(float(a[4])*1e9));self.joint=a;self.joints[s]=a
   if len(self.joints)>24:self.joints.popitem(last=False)
   newest=max(self.joints)
   for old in list(self.joints):
    if old<newest-self.horizon_ns:self.joints.pop(old,None)
   return True
  finally:self.lock.release()
 def _schedule(self):
  while not self.stop_event.wait(.01):
   if not self.enabled or self.busy.value or time.monotonic()-self.last_submit<1/self.rate:continue
   if self.process is None or not self.process.is_alive():self.last_rejection='native_process_dead_no_raw_effect';self.stats['worker_restarts_required']+=1;return
   if self.clock_ns is None:self.last_rejection='actual_ROS_clock_required';continue
   if not self.lock.acquire(False):continue
   try:
    common=set(self.frames) if self.config['model_kind']=='ground_ipm' else set(self.joints)&set(self.frames)
    if not common:continue
    s=max(common);record=self.joints.get(s) if self.config['model_kind']=='ground_ipm' else self.joints[s]
    if s==self.last_submitted_stamp:continue
    age=(int(self.clock_ns())-s)*1e-9
    if not 0<=age<=self.max_age:self.last_rejection='stale_before_worker';self.stats['age_dropped']+=1;continue
    current=self.frames[s]
    if self.config['model_kind']=='ground_ipm':
     data=[current['data']]
    else:
     if len(record)<3 or int(record[1])<1 or int(record[1])>10 or len(record)!=3+160*int(record[1]):self.last_rejection='invalid_joint_schema';continue
     rows=np.asarray(record)[3:].reshape(-1,160);ids={int(round(float(t)*1e9)) for t in rows[:,0]}|{s};data=[v['data'] for t,v in self.frames.items() if t in ids]
    job=dict(frames=data,joint=record,source_health=current['health'],original_header=current['header'],input_ros_age_s=age,submitted_monotonic=time.monotonic(),submitted_wall_ns=time.time_ns())
    self.busy.value=1
    try:self.jobs.put_nowait(job)
    except queue.Full:self.busy.value=0;continue
    self.last_submit=time.monotonic();self.last_submitted_stamp=s;self.stats['submitted']+=1
   finally:self.lock.release()
 def poll(self):
  """Return latest result with current ROS+wall age. No topic is published."""
  value=None
  while True:
   try:value=self.results.get_nowait()
   except queue.Empty:break
  if value is None:return None
  packet=value.get('packet');value['eligible_for_inferred_free']=False
  if packet is None:return value
  now=time.monotonic();age=(int(self.clock_ns())-packet['stamp_ns'])*1e-9 if self.clock_ns else float('inf');wall_age=value['stats']['job_input_ros_age_s']+now-value['submitted_monotonic'];value['current_ros_exposure_age_s']=age;value['current_wall_exposure_age_s']=wall_age;fresh=0<=age<=self.max_age and 0<=wall_age<=self.max_age;value['eligible_for_inferred_free']=bool(self.enabled and self.permit_conditional_free and value['source_health'] is True and self.health_ok_now is not None and self.health_ok_now() is True and fresh and value['stats']['compute_s']<=self.config['max_compute_s']);return value
 def evidence_32fc2(self,result):
  """Final admission point. Call immediately before the SINGLE existing publisher.
  Channel0 original raw bits. Channel1 conditional INFERRED free-only axialZ.
  Consumer MUST retain occupied/terminal/whole-body/stopping checks and full age.
  """
  packet=result['packet'];raw=packet['original_measured_depth'];e=np.full(raw.shape+(2,),np.nan,np.float32);e[...,0]=raw
  if not result.get('eligible_for_inferred_free') or not self.enabled or not self.permit_conditional_free or self.health_ok_now is None or self.health_ok_now() is not True:return e
  age=(int(self.clock_ns())-packet['stamp_ns'])*1e-9;wall_age=result['stats']['job_input_ros_age_s']+time.monotonic()-result['submitted_monotonic']
  if not(0<=age<=self.max_age and 0<=wall_age<=self.max_age):return e
  z=packet['inferred_depth'];K=packet['K'];signature=(z.shape,tuple(map(float,K)))
  if self.raynorm_cache is None or self.raynorm_cache[0]!=signature:
   x=(np.arange(z.shape[1],dtype=float)-K[2])/K[0];y=(np.arange(z.shape[0],dtype=float)-K[3])/K[1];norm=np.sqrt(x[None,:]**2+y[:,None]**2+1);norm.setflags(write=False);self.raynorm_cache=(signature,norm)
  norm=self.raynorm_cache[1];distance=z*norm;limited=np.minimum(distance,self.max_ray);lower=(limited-(self.reserve-self.credit))/norm;lower=np.nextafter(lower.astype(np.float32),np.float32(-np.inf));ok=np.isfinite(z)&np.isnan(raw)&np.isfinite(lower)&(lower>.25)&(lower<z)&(limited>self.reserve);e[...,1][ok]=lower[ok];return e
 def reset_inputs(self):
  with self.lock:
   self.frames.clear();self.joint=None;self.joints.clear();self.last_submitted_stamp=None

 def close(self):
  self.enabled=False;self.stop_event.set()
  if self.scheduler:self.scheduler.join(timeout=.2)
  if self.process:
   if self.process.is_alive():
    try:self.jobs.put_nowait(None)
    except queue.Full:pass
   self.process.join(timeout=.2)
   if self.process.is_alive():self.process.terminate();self.process.join(timeout=.2)
  self.frames.clear();self.joint=None;self.joints.clear()
  for q in (self.jobs,self.results):
   try:q.cancel_join_thread();q.close()
   except ValueError:pass
