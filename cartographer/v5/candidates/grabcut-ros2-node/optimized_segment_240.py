"""Experimental CURRENT-SEED-VALIDATED240p GMM reuse; explicit NEW ungraded policy.
Cold trains the original five-iteration model; warm calls native
GC_EVAL_FREEZE_MODEL once with CURRENT geometric hard seeds. Output is a NEW
conditional candidate policy requiring independent grading, not the frozenv1.
A state's learned appearance may support any view in one current causal
snapshot, but never a decision before its actual supporting exposure.
"""
import time,platform,resource
from dataclasses import dataclass
from functools import lru_cache
import numpy as np,cv2

POLICY=dict(height_px=240,cold_iterations=5,warm_iterations=1,mode='GC_EVAL_FREEZE_MODEL',rng_seed=0,floor_height_tolerance_m=.03,horizon_angular_exclusion_deg=.5,max_sensor_skew_s=.05,model_reuse_policy='current measured seed coverage only',coverage_min=.95,coverage_sigma=3.,mono_std_floor_DN=1.)
@dataclass
class GMMState:
 background:object=None
 floor:object=None
 support_stamp_ns:int=None
 decision_stamp_ns:int=None
 last_validated_stamp_ns:int=None
 last_validation:object=None
 signature:object=None
 def reset(self):self.background=self.floor=self.support_stamp_ns=self.signature=None;self.last_validated_stamp_ns=None;self.last_validation=None

@lru_cache(maxsize=2)
def grid(shape,K):
 H,W=shape;fx,fy,cx,cy=K;x=(np.arange(W,dtype=float)-cx)/fx;y=(np.arange(H,dtype=float)-cy)/fy;norm=np.sqrt(x[None,:]**2+y[:,None]**2+1);return x,y,norm


def seed_coverage(image,seeds,model):
 """Current240p mono intensity vs positive-weight learned GMM components.
 OpenCV GMM layout:5weights,5x3means,5x3x3covariances. Project replicated
 RGB to1Dmono with u=(1,1,1)/3; floor standard deviation at1digitalnumber.
 This is appearance support only, not a semantic or geometric probability.
 """
 a=np.asarray(model,float).reshape(-1)
 if len(a)!=65 or not np.isfinite(a).all():return dict(accepted=False,reason='invalid GMM',fraction=0.,pixels=int(seeds.sum()))
 weights=a[:5];active=weights>0
 if not active.any():return dict(accepted=False,reason='no positive GMM component',fraction=0.,pixels=int(seeds.sum()))
 means=a[5:20].reshape(5,3)[active];cov=a[20:65].reshape(5,3,3)[active];u=np.ones(3)/3;mu=means@u;variance=np.einsum('i,kij,j->k',u,cov,u);sigma=np.maximum(np.sqrt(np.maximum(variance,0.)),1.);values=np.asarray(image[seeds],float)
 if not len(values):return dict(accepted=False,reason='no current actual seed',fraction=0.,pixels=0)
 inside=np.abs(values[:,None]-mu[None,:])<=3.*sigma[None,:];covered=inside.any(axis=1);fraction=float(covered.mean());components=[dict(weight=float(w),mono_mean_DN=float(m),projected_std_DN=float(sd),current_seed_matches=int(n)) for w,m,sd,n in zip(weights[active],mu,sigma,inside.sum(0))]
 return dict(accepted=bool(fraction>=.95),fraction=fraction,pixels=int(len(values)),covered_pixels=int(covered.sum()),minimum_fraction=.95,sigma_multiplier=3.,std_floor_DN=1.,seed_median_DN=float(np.median(values)),components=components)

def validate_current_seeds(image,seedF,seedB,bg,fg):
 floor=seed_coverage(image,seedF,fg);nonfloor=seed_coverage(image,seedB,bg)
 return dict(accepted=bool(floor['accepted'] and nonfloor['accepted']),floor=floor,nonfloor=nonfloor)

def segment(d,self_mask=None,*,state=None,decision_stamp_ns=None):
 start=time.perf_counter();L=d['left'];raw=d['measured_stereo'];K=tuple(map(float,d['K']));s=int(d['stamp_ns']);decision=s if decision_stamp_ns is None else int(decision_stamp_ns)
 if s>decision:raise ValueError('a future image cannot support current decision')
 actual_g=np.array(d['gravity_up_optical'],float);actual_g/=np.linalg.norm(actual_g);g=np.array(d.get('fitted_floor_normal_optical',actual_g),float);g/=np.linalg.norm(g);h=float(d['fitted_floor_offset_m']) if 'fitted_floor_offset_m' in d else float(g@(d['tof_origin_optical_m']+float(d['tof_range_m'])*d['tof_direction_optical']));H,W=L.shape;x,y,norm=grid((H,W),K);dot=g[0]*x[None,:]+g[1]*y[:,None]+g[2];valid=np.isfinite(raw)&(raw>0);raw_height=dot*raw-h;hardfloor=valid&(abs(raw_height)<=.03);hardnonfloor=(dot>norm*np.sin(np.radians(.5)))&((actual_g[0]*x[None,:]+actual_g[1]*y[:,None]+actual_g[2])>norm*np.sin(np.radians(.5)));conflict=hardfloor&hardnonfloor
 if self_mask is None:raise ValueError('actual mask required')
 hardfloor&=~self_mask;hardnonfloor&=~self_mask
 if not bool(d['tof_valid']) or h>=-.05 or abs(int(d['tof_stamp_ns'])-s)*1e-9>.05 or abs(int(d['gravity_stamp_ns'])-s)*1e-9>.05:raise ValueError('actual ToF/gravity invalid/skew')
 if (H,W)!=(800,1280):raise ValueError('this prototype retains actual1280x800 native grid')
 size=(384,240);scale_x=size[0]/W;scale_y=size[1]/H;image=cv2.resize(L,size,interpolation=cv2.INTER_AREA);seedF=np.zeros(image.shape,bool);seedB=np.zeros(image.shape,bool)
 # Exact native pixel-centre -> segmentation-cell mapping. No native K/ray edits.
 for source,target in ((hardfloor,seedF),(hardnonfloor,seedB)):
  yy,xx=np.nonzero(source);sx=np.minimum(np.floor((xx+.5)*scale_x).astype(int),size[0]-1);sy=np.minimum(np.floor((yy+.5)*scale_y).astype(int),size[1]-1);target[sy,sx]=True
 mixed=seedF&seedB;seedF[mixed]=False;seedB[mixed]=False;excluded=cv2.resize(self_mask.astype(np.float32),size,interpolation=cv2.INTER_AREA)>0.;seedF&=~excluded;seedB&=~excluded;mask=np.full(image.shape,cv2.GC_PR_BGD,np.uint8);mask[seedF]=cv2.GC_FGD;mask[seedB]=cv2.GC_BGD;before=mask.copy();seed_s=time.perf_counter()-start
 if seedF.sum()<5 or seedB.sum()<5:return None,dict(status='missing_geometric_seed_class'),None
 state=GMMState() if state is None else state;signature=((H,W),K)
 if state.decision_stamp_ns is not None and decision<state.decision_stamp_ns:state.reset()
 state.decision_stamp_ns=decision
 reusable=state.background is not None and state.signature==signature and state.support_stamp_ns<=decision
 if not reusable and s!=decision:raise ValueError('current exposure must initialize appearance model before historical views')
 if reusable:
  validation=validate_current_seeds(image,seedF,seedB,state.background,state.floor)
  if s!=decision and state.last_validated_stamp_ns!=decision:raise ValueError('current actual seeds must validate reused model before historical views')
  if s!=decision and not validation['accepted']:return None,dict(status='historical_seed_coverage_failed',seed_validation=validation,seed_validation_input_stamp_ns=s,last_validated_stamp_ns=state.last_validated_stamp_ns,model_support_stamp_ns=state.support_stamp_ns),None
  warm=validation['accepted']
 else:warm=False
 if s==decision and warm:state.last_validated_stamp_ns=s;state.last_validation=validation
 bg=state.background.copy() if warm else np.zeros((1,65),np.float64);fg=state.floor.copy() if warm else np.zeros((1,65),np.float64);cv2.setRNGSeed(0);t=time.perf_counter();cv2.grabCut(np.repeat(image[:,:,None],3,axis=2),mask,None,bg,fg,1 if warm else 5,cv2.GC_EVAL_FREEZE_MODEL if warm else cv2.GC_INIT_WITH_MASK);graph_s=time.perf_counter()-t
 if not warm:
  state.background=bg.copy();state.floor=fg.copy();state.support_stamp_ns=s;state.signature=signature;validation=validate_current_seeds(image,seedF,seedB,bg,fg)
  if validation['accepted']:state.last_validated_stamp_ns=s;state.last_validation=validation
  else:
   state.last_validated_stamp_ns=None;state.last_validation=validation
   return None,dict(status='current_seed_coverage_failed_after_cold',seed_validation=validation,seed_validation_input_stamp_ns=s,last_validated_stamp_ns=None,model_support_stamp_ns=s,grabcut_s=graph_s),None
 t=time.perf_counter();isfloor=((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD))&~excluded;nonfloor=~isfloor&~excluded
 def connected(region,seeds):
  _,labels=cv2.connectedComponents(region.astype(np.uint8),connectivity=4);ids=np.unique(labels[seeds&region]);ids=ids[ids>0];return np.isin(labels,ids)&region
 cf=connected(isfloor,seedF);cb=connected(nonfloor,seedB);lab=np.zeros(image.shape,np.uint8);lab[cf]=1;lab[cb]=2;native_x=np.minimum(np.floor((np.arange(W)+.5)*scale_x).astype(int),size[0]-1);native_y=np.minimum(np.floor((np.arange(H)+.5)*scale_y).astype(int),size[1]-1);native=lab[native_y[:,None],native_x[None,:]];native[self_mask]=0;post_s=time.perf_counter()-t
 result=dict(label_floor_nonfloor_unknown=native,label400=lab,grabcut_mask400=mask,initial_mask400=before,hard_floor_native=hardfloor,hard_nonfloor_native=hardnonfloor,hard_floor400=seedF,hard_nonfloor400=seedB,gmm_background=bg,gmm_floor=fg,raw_height_relative_tof_floor_m=raw_height.astype(np.float32))
 stats=dict(segmentation_resolution=[384,240],native_resolution=[1280,800],segmentation_scale_xy=[scale_x,scale_y],coordinate_mapping='pixel-centre: seg=(native+.5)*scale-.5; inverse=(seg+.5)/scale-.5',status='experimental_currentseed_validated_warm_GMM' if warm else 'experimental_validated_cold_original5',mode='GC_EVAL_FREEZE_MODEL' if warm else 'GC_INIT_WITH_MASK',iterations=1 if warm else 5,seed_s=seed_s,grabcut_s=graph_s,postprocess_s=post_s,total_segment_s=time.perf_counter()-start,model_support_stamp_ns=state.support_stamp_ns,model_age_s=(decision-state.support_stamp_ns)*1e-9,last_validated_stamp_ns=state.last_validated_stamp_ns,seed_validation_input_stamp_ns=s,seed_validation=validation,decision_stamp_ns=decision,hard_floor_native=int(hardfloor.sum()),hard_nonfloor_native=int(hardnonfloor.sum()),native_seed_conflicts=int(conflict.sum()),mixed400_cells=int(mixed.sum()),hard_floor400=int(seedF.sum()),hard_nonfloor400=int(seedB.sum()),floor_height_camera_m=h,gravity_up_optical=g.tolist(),floor_pixels=int((native==1).sum()),nonfloor_pixels=int((native==2).sum()),unknown_pixels=int((native==0).sum()),seed_floor_retention=float((native[hardfloor]==1).mean()),seed_nonfloor_retention=float((native[hardnonfloor]==2).mean()),raw_floor_classified_nonfloor=int((hardfloor&(native==2)).sum()),raw_nonfloor_classified_floor=int((hardnonfloor&(native==1)).sum()),peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1024 if platform.system()=='Linux' else 1));return result,stats,image


class MeasuredSeedGrabCutValidatedReuse:
 """Explicit new policy: no blind age extension; current seed coverage required.
 Call CURRENT exposure first for each decision snapshot, then any historical
 views. last_validated_stamp never changes pose/range/evidence timestamps.
 """
 def __init__(self,mode='cold5_freeze1_seedvalidated_240p'):
  if mode!='cold5_freeze1_seedvalidated_240p':raise ValueError('explicit validated240p policy required')
  self.mode=mode;self.states={'left':GMMState(),'right':GMMState()}
 def __call__(self,frame,self_mask,camera_side,decision_stamp_ns):
  if camera_side not in self.states:raise ValueError('camera_side must be left/right')
  return segment(frame,self_mask,state=self.states[camera_side],decision_stamp_ns=decision_stamp_ns)
 def reset(self):
  for state in self.states.values():state.reset();state.decision_stamp_ns=None

# Explicit240p import name; legacyclass name remains API-compatible.
MeasuredSeedGrabCutValidatedReuse240 = MeasuredSeedGrabCutValidatedReuse
