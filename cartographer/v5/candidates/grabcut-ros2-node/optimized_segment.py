"""Experimental measured-seed400p GrabCut GMM reuse, default not integrated.
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

POLICY=dict(height_px=400,cold_iterations=5,warm_iterations=1,mode='GC_EVAL_FREEZE_MODEL',rng_seed=0,floor_height_tolerance_m=.03,horizon_angular_exclusion_deg=.5,max_sensor_skew_s=.05,max_model_age_s=1.5)
@dataclass
class GMMState:
 background:object=None
 floor:object=None
 support_stamp_ns:int=None
 decision_stamp_ns:int=None
 signature:object=None
 def reset(self):self.background=self.floor=self.support_stamp_ns=self.signature=None

@lru_cache(maxsize=2)
def grid(shape,K):
 H,W=shape;fx,fy,cx,cy=K;x=(np.arange(W,dtype=float)-cx)/fx;y=(np.arange(H,dtype=float)-cy)/fy;norm=np.sqrt(x[None,:]**2+y[:,None]**2+1);return x,y,norm

def segment(d,self_mask=None,*,state=None,decision_stamp_ns=None):
 start=time.perf_counter();L=d['left'];raw=d['measured_stereo'];K=tuple(map(float,d['K']));s=int(d['stamp_ns']);decision=s if decision_stamp_ns is None else int(decision_stamp_ns)
 if s>decision:raise ValueError('a future image cannot support current decision')
 actual_g=np.array(d['gravity_up_optical'],float);actual_g/=np.linalg.norm(actual_g);g=np.array(d.get('fitted_floor_normal_optical',actual_g),float);g/=np.linalg.norm(g);h=float(d['fitted_floor_offset_m']) if 'fitted_floor_offset_m' in d else float(g@(d['tof_origin_optical_m']+float(d['tof_range_m'])*d['tof_direction_optical']));H,W=L.shape;x,y,norm=grid((H,W),K);dot=g[0]*x[None,:]+g[1]*y[:,None]+g[2];valid=np.isfinite(raw)&(raw>0);raw_height=dot*raw-h;hardfloor=valid&(abs(raw_height)<=.03);hardnonfloor=(dot>norm*np.sin(np.radians(.5)))&((actual_g[0]*x[None,:]+actual_g[1]*y[:,None]+actual_g[2])>norm*np.sin(np.radians(.5)));conflict=hardfloor&hardnonfloor
 if self_mask is None:raise ValueError('actual mask required')
 hardfloor&=~self_mask;hardnonfloor&=~self_mask
 if not bool(d['tof_valid']) or h>=-.05 or abs(int(d['tof_stamp_ns'])-s)*1e-9>.05 or abs(int(d['gravity_stamp_ns'])-s)*1e-9>.05:raise ValueError('actual ToF/gravity invalid/skew')
 if (H,W)!=(800,1280):raise ValueError('this prototype retains actual1280x800 native grid')
 size=(640,400);image=cv2.resize(L,size,interpolation=cv2.INTER_AREA);seedF=hardfloor.reshape(400,2,640,2).any((1,3));seedB=hardnonfloor.reshape(400,2,640,2).any((1,3));mixed=seedF&seedB;seedF[mixed]=False;seedB[mixed]=False;excluded=self_mask.reshape(400,2,640,2).any((1,3));seedF&=~excluded;seedB&=~excluded;mask=np.full(image.shape,cv2.GC_PR_BGD,np.uint8);mask[seedF]=cv2.GC_FGD;mask[seedB]=cv2.GC_BGD;before=mask.copy();seed_s=time.perf_counter()-start
 if seedF.sum()<5 or seedB.sum()<5:return None,dict(status='missing_geometric_seed_class'),None
 state=GMMState() if state is None else state;signature=((H,W),K)
 if state.decision_stamp_ns is not None and decision<state.decision_stamp_ns:state.reset()
 state.decision_stamp_ns=decision
 warm=state.background is not None and state.signature==signature and state.support_stamp_ns<=decision and decision-state.support_stamp_ns<=int(1.5e9)
 bg=state.background.copy() if warm else np.zeros((1,65),np.float64);fg=state.floor.copy() if warm else np.zeros((1,65),np.float64);cv2.setRNGSeed(0);t=time.perf_counter();cv2.grabCut(np.repeat(image[:,:,None],3,axis=2),mask,None,bg,fg,1 if warm else 5,cv2.GC_EVAL_FREEZE_MODEL if warm else cv2.GC_INIT_WITH_MASK);graph_s=time.perf_counter()-t
 if not warm:state.background=bg.copy();state.floor=fg.copy();state.support_stamp_ns=s;state.signature=signature
 t=time.perf_counter();isfloor=((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD))&~excluded;nonfloor=~isfloor&~excluded
 def connected(region,seeds):
  _,labels=cv2.connectedComponents(region.astype(np.uint8),connectivity=4);ids=np.unique(labels[seeds&region]);ids=ids[ids>0];return np.isin(labels,ids)&region
 cf=connected(isfloor,seedF);cb=connected(nonfloor,seedB);lab=np.zeros(image.shape,np.uint8);lab[cf]=1;lab[cb]=2;native=cv2.resize(lab,(W,H),interpolation=cv2.INTER_NEAREST);native[self_mask]=0;post_s=time.perf_counter()-t
 result=dict(label_floor_nonfloor_unknown=native,label400=lab,grabcut_mask400=mask,initial_mask400=before,hard_floor_native=hardfloor,hard_nonfloor_native=hardnonfloor,hard_floor400=seedF,hard_nonfloor400=seedB,gmm_background=bg,gmm_floor=fg,raw_height_relative_tof_floor_m=raw_height.astype(np.float32))
 stats=dict(status='experimental_conditional_warm_GMM' if warm else 'experimental_cold_original5',mode='GC_EVAL_FREEZE_MODEL' if warm else 'GC_INIT_WITH_MASK',iterations=1 if warm else 5,seed_s=seed_s,grabcut_s=graph_s,postprocess_s=post_s,total_segment_s=time.perf_counter()-start,model_support_stamp_ns=state.support_stamp_ns,decision_stamp_ns=decision,hard_floor_native=int(hardfloor.sum()),hard_nonfloor_native=int(hardnonfloor.sum()),native_seed_conflicts=int(conflict.sum()),mixed400_cells=int(mixed.sum()),hard_floor400=int(seedF.sum()),hard_nonfloor400=int(seedB.sum()),floor_height_camera_m=h,gravity_up_optical=g.tolist(),floor_pixels=int((native==1).sum()),nonfloor_pixels=int((native==2).sum()),unknown_pixels=int((native==0).sum()),seed_floor_retention=float((native[hardfloor]==1).mean()),seed_nonfloor_retention=float((native[hardnonfloor]==2).mean()),raw_floor_classified_nonfloor=int((hardfloor&(native==2)).sum()),raw_nonfloor_classified_floor=int((hardnonfloor&(native==1)).sum()),peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1024 if platform.system()=='Linux' else 1));return result,stats,image

class MeasuredSeedGrabCutReuse:
 """Injectable explicit NEW-policy segmenter. Caller still owns deadline/age.

 Usage: segmenter(frame,actual_mask,'left',decision_stamp_ns) -> result,stats,image
 The decision stamp is the latest actual exposure of the SAME causal job;
 this allows processing its older native-clone views without resetting models.
 Cold calls may overrun; the caller must drop that output while retaining this
 object's learned GMM for the next fresh job. Never publish expired results.
 """
 def __init__(self,mode='cold5_freeze1_400p'):
  if mode!='cold5_freeze1_400p':raise ValueError('explicit new400p policy required')
  self.mode=mode;self.states={'left':GMMState(),'right':GMMState()}
 def __call__(self,frame,self_mask,camera_side,decision_stamp_ns):
  if camera_side not in self.states:raise ValueError('camera_side must be left/right')
  return segment(frame,self_mask,state=self.states[camera_side],decision_stamp_ns=decision_stamp_ns)
 def reset(self):
  for state in self.states.values():state.reset();state.decision_stamp_ns=None
