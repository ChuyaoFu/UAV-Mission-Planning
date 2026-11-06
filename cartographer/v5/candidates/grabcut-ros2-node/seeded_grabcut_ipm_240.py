import json,time,hashlib,resource,platform
from pathlib import Path
import numpy as np,cv2
# Pure reusable400p fixed-five-iteration measured-seed core; no import-time IO.
POLICY=dict(height_px=240,iterations=5,rng_seed=0,floor_height_tolerance_m=.03,horizon_angular_exclusion_deg=.5,max_sensor_skew_s=.05,mode='GC_INIT_WITH_MASK',unknown_initial_state='GC_PR_BGD; a probable category only, never a hard geometric nonfloor label')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def segment(d,self_mask=None,side=None,decision_stamp=None):
 L=d['left'];raw=d['measured_stereo'];K=d['K'];s=int(d['stamp_ns']);actual_g=d['gravity_up_optical'].astype(float);actual_g/=np.linalg.norm(actual_g);g=np.asarray(d['fitted_floor_normal_optical'],float);h=float(d['fitted_floor_offset_m']);H,W=L.shape;v,u=np.indices(L.shape);rays=np.stack(((u-K[2])/K[0],(v-K[3])/K[1],np.ones(L.shape)),axis=-1);dot=rays@g;valid=np.isfinite(raw)&(raw>0);raw_height=dot*raw-h;hardfloor=valid&(abs(raw_height)<=POLICY['floor_height_tolerance_m']);hardnonfloor=(dot>np.linalg.norm(rays,axis=-1)*np.sin(np.radians(POLICY['horizon_angular_exclusion_deg'])))&((rays@actual_g)>np.linalg.norm(rays,axis=-1)*np.sin(np.radians(POLICY['horizon_angular_exclusion_deg'])));conflict=hardfloor&hardnonfloor
 if self_mask is None:self_mask=np.zeros(L.shape,bool)
 hardfloor&=~self_mask;hardnonfloor&=~self_mask
 if not bool(d['tof_valid']) or h>=-.05 or abs(int(d['tof_stamp_ns'])-s)*1e-9>POLICY['max_sensor_skew_s'] or abs(int(d['gravity_stamp_ns'])-s)*1e-9>POLICY['max_sensor_skew_s']:raise ValueError('actual ToF/gravity invalid/skew')
 scale=POLICY['height_px']/H;w=int(round(W*scale));size=(w,POLICY['height_px']);image=cv2.resize(L,size,interpolation=cv2.INTER_AREA);seedF=np.zeros(image.shape,bool);seedB=np.zeros(image.shape,bool)
 # Map only genuine native supporting points to400p cells. Mixed cells stay unknown.
 yy,xx=np.nonzero(hardfloor);seedF[np.minimum(((yy+.5)*scale).astype(int),size[1]-1),np.minimum(((xx+.5)*scale).astype(int),size[0]-1)]=True
 yy,xx=np.nonzero(hardnonfloor);seedB[np.minimum(((yy+.5)*scale).astype(int),size[1]-1),np.minimum(((xx+.5)*scale).astype(int),size[0]-1)]=True
 mixed=seedF&seedB;seedF[mixed]=False;seedB[mixed]=False;excluded=cv2.resize(self_mask.astype(np.uint8),size,interpolation=cv2.INTER_AREA)>0;seedF&=~excluded;seedB&=~excluded;mask=np.full(image.shape,cv2.GC_PR_BGD,np.uint8);mask[seedF]=cv2.GC_FGD;mask[seedB]=cv2.GC_BGD
 if seedF.sum()<5 or seedB.sum()<5:return None,dict(status='missing_geometric_seed_class',hard_floor_native=int(hardfloor.sum()),hard_nonfloor_native=int(hardnonfloor.sum()),floor_height_camera_m=h),None
 before=mask.copy();bg=np.zeros((1,65),np.float64);fg=np.zeros((1,65),np.float64);cv2.setRNGSeed(POLICY['rng_seed']);t=time.perf_counter();cv2.grabCut(np.repeat(image[:,:,None],3,axis=2),mask,None,bg,fg,POLICY['iterations'],cv2.GC_INIT_WITH_MASK);seconds=time.perf_counter()-t;isfloor=(mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD);isfloor[excluded]=False;nonfloor=~isfloor&~excluded
 def seed_connected(region,seeds):
  _,labels=cv2.connectedComponents(region.astype(np.uint8),connectivity=4);ids=np.unique(labels[seeds&region]);ids=ids[ids>0];return np.isin(labels,ids)&region
 connectedF=seed_connected(isfloor,seedF);connectedB=seed_connected(nonfloor,seedB);lab=np.zeros(image.shape,np.uint8);lab[connectedF]=1;lab[connectedB]=2;iy=np.minimum(((np.arange(H)+.5)*scale).astype(int),size[1]-1);ix=np.minimum(((np.arange(W)+.5)*scale).astype(int),size[0]-1);native=lab[iy[:,None],ix[None,:]];native[self_mask]=0;result=dict(label_floor_nonfloor_unknown=native,label400=lab,grabcut_mask400=mask,initial_mask400=before,hard_floor_native=hardfloor,hard_nonfloor_native=hardnonfloor,hard_floor400=seedF,hard_nonfloor400=seedB,gmm_background=bg,gmm_floor=fg,raw_height_relative_tof_floor_m=raw_height.astype(np.float32))
 stats=dict(status='segmented_conditional_appearance_only',hard_floor_native=int(hardfloor.sum()),hard_nonfloor_native=int(hardnonfloor.sum()),native_seed_conflicts=int(conflict.sum()),mixed400_cells=int(mixed.sum()),hard_floor400=int(seedF.sum()),hard_nonfloor400=int(seedB.sum()),floor_height_camera_m=h,gravity_up_optical=g.tolist(),floor_pixels=int((native==1).sum()),nonfloor_pixels=int((native==2).sum()),unknown_pixels=int((native==0).sum()),grabcut_s=seconds,seed_floor_retention=float((native[hardfloor]==1).mean()) if hardfloor.any() else None,seed_nonfloor_retention=float((native[hardnonfloor]==2).mean()) if hardnonfloor.any() else None,raw_floor_classified_nonfloor=int((hardfloor&(native==2)).sum()),raw_nonfloor_classified_floor=int((hardnonfloor&(native==1)).sum()),peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1024 if platform.system()=='Linux' else 1))
 return result,stats,image

