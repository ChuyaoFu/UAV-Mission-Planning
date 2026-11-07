"""NEW optional conditional IPM vertical +actual fitted-floor completion. Caller defaultOFF.
No ROS, files, temporal poses, native clones, GT/CAD/commands or occupied outputs.
"""
import time,numpy as np,cv2
from seeded_grabcut_ipm import segment as default_segmenter
from cut_lines import extract
from freeze_wall_hypothesis import component
from temporal_boundary import unit

POLICY=dict(version='conditional-ground-ipm-floor-completion-v1',floor_completion=True,ransac_iterations=96,max_anchors=2400,floor_inlier_distance_m=.02,normal_gravity_degrees=3.,tof_offset_max_m=.05,floor_seed_tolerance_m=.03,min_floor_inliers=40,min_floor_fraction=.12,min_2d_support_m=.02,min_boundary_span_m=.20,max_tof_gravity_skew_s=.05,native_axial_depth_m=[.3,2.4],max_euclidean_ray_m=2.5,vertical_probe_m=[.06,1.],boundary_end_margin_m=.02,extra_euclidean_retreat_m=.1,expected_native_voxel_m=.1)

def fit_floor(frame,mask,check):
 raw=frame['measured_stereo'];K=frame['K'];g=unit(frame['gravity_up_optical']);s=int(frame['stamp_ns']);r=float(frame['tof_range_m']);origin=np.asarray(frame['tof_origin_optical_m'],float);d=unit(frame['tof_direction_optical'])
 if not bool(frame['tof_valid']) or not np.isfinite(r) or r<=0 or d@g>-.9 or any(abs(int(frame[k])-s)*1e-9>POLICY['max_tof_gravity_skew_s'] for k in ('tof_stamp_ns','gravity_stamp_ns')):raise ValueError('invalid/unpaired actual ToF/gravity')
 hit=origin+r*d;h=float(g@hit)
 if h>=-.05:raise ValueError('actual camera not above fitted-floor hypothesis')
 v,u=np.nonzero(np.isfinite(raw)&(raw>=.3)&(raw<=2.4)&~mask);z=raw[v,u];P=np.column_stack(((u-K[2])/K[0],(v-K[3])/K[1],np.ones(len(u))))*z[:,None];keep=abs(P@g-h)<=.08;P=P[keep]
 if len(P)<40:raise ValueError('insufficient actual floor candidates')
 rng=np.random.default_rng(2026);P=P if len(P)<=2400 else P[rng.choice(len(P),2400,replace=False)];best=None;required=max(40,int(np.ceil(.12*len(P))));cos=np.cos(np.radians(3.))
 for i in range(96):
  if i%8==0:check()
  a,b,c=P[rng.choice(len(P),3,replace=False)];cross=np.cross(b-a,c-a);norm=np.linalg.norm(cross)
  if norm<1e-6:continue
  n=cross/norm
  if n@g<0:n=-n
  if n@g<cos:continue
  offset=float(n@a)
  if abs(n@hit-offset)>.05:continue
  inlier=abs(P@n-offset)<=.02;count=int(inlier.sum())
  if count>=required and (best is None or count>best[0]):best=(count,inlier)
 if best is None:raise ValueError('no gravity/ToF-consistent measured floor plane')
 pts=P[best[1]];center=pts.mean(0);_,sv,V=np.linalg.svd(pts-center,full_matrices=False);n=unit(V[-1]);n=n if n@g>=0 else -n;offset=float(n@center);residual=abs(P@n-offset);inlier=residual<=.02;angular=float(np.degrees(np.arccos(np.clip(n@g,-1,1))));toferr=float(n@hit-offset)
 if angular>3 or abs(toferr)>.05 or sv[1]/np.sqrt(len(pts))<.02:raise ValueError('refined measured floor normal/ToF/2D support invalid')
 return n,offset,dict(original_tof_floor_height_m=h,fitted_floor_normal_optical=n.tolist(),fitted_floor_offset_m=offset,actual_gravity_up_optical=g.tolist(),normal_gravity_angle_deg=angular,tof_plane_signed_residual_m=toferr,inlier_points=int(inlier.sum()),sampled_actual_points=len(P),inlier_fraction=float(inlier.mean()),inlier_residual_p50_p95_max_m=np.quantile(residual[inlier],[.5,.95,1.]).tolist(),support_singular_values_m=sv.tolist(),physical_covariance_bound_available=False,uncertainty='RANSAC residual/ToF agreement are diagnostics, not physical range/covariance guarantee')

def _seg(fn,frame,mask,side,s):
 value=fn(frame,mask,side,s)
 if isinstance(value,tuple):return value[0],value[1]
 if isinstance(value,dict) and 'label400' in value:return value,{}
 if isinstance(value,dict) and 'segmentation' in value:return value['segmentation'],value.get('stats',{})
 raise ValueError('segmenter must return segmentation mapping or legacy tuple')

def infer_ipm(frame,left_mask,right_mask,segmenter=None,check=lambda:None):
 started=time.monotonic();meta=dict(policy=POLICY.copy(),status='starting',assumption='Explicit CONDITIONAL ground-supported vertical termination via IPM. Boundary 3D is projected onto measured fitted floor, NOT measured wall/native triangulated line.',navigation_enabled=False,certified_free_ray_pixels=0)
 try:
  check();raw=frame['measured_stereo'];L=frame['left'];R=frame['right'];K=np.asarray(frame['K'],float);B=float(frame['baseline']);s=int(frame['stamp_ns']);m0=np.asarray(left_mask);m1=np.asarray(right_mask)
  if L.shape!=(800,1280) or R.shape!=L.shape or raw.shape!=L.shape or raw.dtype!=np.float32 or L.dtype!=np.uint8 or R.dtype!=np.uint8 or m0.shape!=L.shape or m1.shape!=L.shape or m0.dtype!=bool or m1.dtype!=bool or K.shape!=(4,) or not np.isfinite(K).all() or min(K[:2])<=0 or not 0<B<=.5:raise ValueError('actual native images/raw/K/B/masks required')
  n,h,fit=fit_floor(frame,m0,check);meta['floor_fit']=fit;fn=segmenter or default_segmenter;f=dict(frame,fitted_floor_normal_optical=n,fitted_floor_offset_m=h);seg,stats=_seg(fn,f,m0,'left',s);check();meta['left_segmentation']=stats
  if seg is None:raise ValueError('left appearance seed failure')
  lines=extract(seg)
  if not lines:raise ValueError('no connected observed floor/nonfloor boundary')
  selected=None
  for obs in lines:
   line=obs['line'];uv=line.reshape(2,2);dc=np.c_[(uv[:,0]-K[2])/K[0],(uv[:,1]-K[3])/K[1],np.ones(2)];den=dc@n
   if np.any(den>=-1e-8):continue
   depth=h/den;P=dc*depth[:,None];span=np.linalg.norm(P[1]-P[0])
   if np.any(depth<.3) or np.any(depth>2.4) or span<.2:continue
   selected=(line,P,span);break
  if selected is None:raise ValueError('observed boundary IPM intersection invalid/range/span')
  line,P,span=selected;Q=P.mean(0);t=unit(P[1]-P[0]);up=unit(frame['gravity_up_optical']);normal=unit(np.cross(t,up));normal=normal if normal@Q>=0 else -normal;horizontal=unit(np.cross(up,normal));horizontal=horizontal if horizontal@t>=0 else -horizontal;half=float(abs((P[1]-P[0])@horizontal)*.5);slope=float(t@up/(t@horizontal));meta.update(observed_boundary_pixels=line.tolist(),ipm_ground_endpoints_optical_m=P.tolist(),ipm_common_single_view_horizontal_span_m=2*half,ipm_boundary_3d_span_m=float(span),temporal_baseline_required=False)
  W=np.eye(4);WR=np.eye(4);WR[0,3]=B;lab,cid,edge=component(L,m0,W,K,Q,t,up,span*.5);rlab,rcid,redge=component(R,m1,WR,K,Q,t,up,span*.5);v,u=np.indices(raw.shape);ray=np.stack(((u-K[2])/K[0],(v-K[3])/K[1],np.ones(raw.shape)),axis=-1);den=ray@normal;prediction=np.full(raw.shape,np.nan);ok=den>1e-6;prediction[ok]=(normal@Q)/den[ok];along=(prediction[...,None]*ray-Q)@horizontal;height=(prediction[...,None]*ray-Q)@up-slope*along;norm=np.linalg.norm(ray,axis=-1)
  rz=np.full(raw.shape,np.inf,np.float32);yy,xx=np.nonzero(np.isfinite(raw)&(raw>0));zz=raw[yy,xx];xr=np.rint(xx-K[0]*B/zz).astype(int);valid=(xr>=0)&(xr<raw.shape[1]);np.minimum.at(rz.ravel(),yy[valid]*raw.shape[1]+xr[valid],zz[valid]);rz[~np.isfinite(rz)]=np.nan;rf=dict(frame,left=R,measured_stereo=rz,tof_origin_optical_m=np.asarray(frame['tof_origin_optical_m'])-np.array([B,0,0]),fitted_floor_normal_optical=n,fitted_floor_offset_m=h-n[0]*B);rseg,rstats=_seg(fn,rf,m1,'right',s);check();meta['right_segmentation']=rstats
  roi=(prediction>=.3)&(prediction<=2.4)&(prediction*norm<=2.5)&(height>.06)&(height<1.)&(abs(along)<=half-.02)&(seg['label_floor_nonfloor_unknown']==2)&~edge&~m0
  if cid is None or rcid is None or rseg is None:raise ValueError('unconnected/hole/appearance component')
  roi&=lab==cid;xr=(u-K[0]*B/np.maximum(prediction,1e-6)).astype(np.float32);yr=v.astype(np.float32);rightbad=cv2.remap((m1|redge).astype(np.uint8),xr,yr,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=1)>0;rightlabel=cv2.remap(rlab.astype(np.float32),xr,yr,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0);rightnonfloor=cv2.remap((rseg['label_floor_nonfloor_unknown']==2).astype(np.uint8),xr,yr,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)>0;warped=cv2.remap(R,xr,yr,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);photo=cv2.boxFilter(abs(L.astype(np.float32)-warped.astype(np.float32)),cv2.CV_32F,(5,5));roi&=(xr>=4)&(xr<raw.shape[1]-4)&~rightbad&(rightlabel==rcid)&rightnonfloor&(photo<5)
  positive=np.isfinite(raw)&(raw>0);farther=roi&positive&(raw>prediction+np.maximum(.03,.03*prediction));meta['actual_farther_raw_contradictions']=int(farther.sum())
  if farther.sum()>=3:raise ValueError('actual farther depth contradicts applicable termination component')
  foreground=positive&(raw+np.maximum(.03,.03*raw)<prediction);foreground=cv2.dilate(foreground.astype(np.uint8),np.ones((7,7),np.uint8))>0;roi&=np.isnan(raw)&~foreground;vertical=np.full(raw.shape,np.nan,np.float32);vertical[roi]=prediction[roi].astype(np.float32)
  # NEW optional caller-selected fitted-floor completion. The already-fitted
  # n/h and measured-seed connected floor labels are used, no legacy projector.
  floor_den=ray@n;floor_z=np.full(raw.shape,np.nan);fd=floor_den<-1e-8;floor_z[fd]=h/floor_den[fd];floor_roi=(floor_z>=.3)&(floor_z<=2.4)&(floor_z*norm<=2.5)&(seg['label_floor_nonfloor_unknown']==1)&~edge&~m0
  floor_xr=(u-K[0]*B/np.maximum(floor_z,1e-6)).astype(np.float32);floor_bad=cv2.remap((m1|redge).astype(np.uint8),floor_xr,yr,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=1)>0;right_floor=cv2.remap((rseg['label_floor_nonfloor_unknown']==1).astype(np.uint8),floor_xr,yr,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)>0;floor_warp=cv2.remap(R,floor_xr,yr,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);floor_photo=cv2.boxFilter(abs(L.astype(np.float32)-floor_warp.astype(np.float32)),cv2.CV_32F,(5,5));floor_roi&=(floor_xr>=4)&(floor_xr<raw.shape[1]-4)&~floor_bad&right_floor&(floor_photo<5)
  floor_farther=floor_roi&positive&(raw>floor_z+np.maximum(.03,.03*floor_z));floor_foreground=positive&(raw+np.maximum(.03,.03*raw)<floor_z);floor_veto=cv2.dilate((floor_foreground|floor_farther).astype(np.uint8),np.ones((7,7),np.uint8))>0;floor_rejected=int(floor_farther.sum())>=3
  floor_veto_count=int((floor_roi&floor_veto).sum());floor_roi&=np.isnan(raw)&~floor_veto
  if floor_rejected:floor_roi[:]=False
  floor_depth=np.full(raw.shape,np.nan,np.float32);floor_depth[floor_roi]=floor_z[floor_roi].astype(np.float32);surface=np.fmin(vertical,floor_depth);merged=np.isfinite(surface);consumer=np.full(raw.shape,np.nan,np.float32);consumer[roi]=np.nextafter((prediction[roi]-.1/norm[roi]).astype(np.float32),np.float32(-np.inf));floor_consumer=np.full(raw.shape,np.nan,np.float32);floor_consumer[floor_roi]=np.nextafter((floor_z[floor_roi]-.1/norm[floor_roi]).astype(np.float32),np.float32(-np.inf));consumer=np.fmin(consumer,floor_consumer);consumer[consumer<=.25]=np.nan;free=np.full(raw.shape,np.nan,np.float32)
  meta.update(floor_completion_policy='NEW conditional measured fitted-floor/dual-view connected-floor/photo completion; no occupied endpoints',vertical_inferred_pixels=int(np.isfinite(vertical).sum()),floor_inferred_pixels=int(floor_roi.sum()),floor_farther_raw_contradictions=int(floor_farther.sum()),floor_rejected_by_farther_raw=floor_rejected,floor_foreground_veto_pixels=floor_veto_count,merged_surface_pixels=int(merged.sum()));meta.update(status='conditional_ipm_candidate',surface_pixels=int(np.isfinite(surface).sum()),consumer_input_pixels=int(np.isfinite(consumer).sum()),left_component=cid,right_component=rcid,right_seed_source='DERIVED calibrated reprojection of actual left original stereo hits; seed only, not right sensor raw depth',physical_wall_measurement=False,physical_range_error_bound_available=False)
  if not np.isfinite(surface).any():raise ValueError('empty eligible conditional IPM+floor region')
  packet=dict(stamp_ns=np.int64(s),stamp_s=np.float64(s*1e-9),K=K,baseline=np.float64(B),inferred_depth=surface,conditional_vertical_depth=vertical,conditional_floor_depth=floor_depth,consumer_input_depth=consumer,surface_minus_0p10_over_raynorm=consumer,free_ray_lower_bound=free,original_measured_depth=raw,fitted_floor_normal_optical=n,fitted_floor_offset_m=np.float64(h),ipm_ground_endpoints_optical_m=P,observed_boundary_pixels=line,candidate_qualified=np.bool_(False),navigation_enabled=np.bool_(False),semantics=np.array('NEW CONDITIONAL ground-supported vertical termination plus measured fitted-floor completion; NOT original raw/occupied/certifiedfree'));check();return packet,meta
 except (ValueError,KeyError,TypeError,IndexError,cv2.error) as e:meta.update(status='rejected',reason=str(e));return None,meta
 finally:meta['compute_s']=time.monotonic()-started
