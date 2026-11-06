"""CAD-free multi-exposure metric line diagnosis; no image editing or floor projection.
OpenCV native LSD + pinned PL-VINS LBD, then interpretation-plane intersection.
All reconstructed geometry is inferred. Missing joint pose covariance => no bound.
"""
from dataclasses import dataclass,asdict
import ctypes,math
from pathlib import Path
import cv2,numpy as np

@dataclass(frozen=True)
class Policy:
 min_line_px:float=60.
 max_lines:int=100
 hamming_max:int=30
 ratio:float=.8
 min_views:int=3
 min_baseline_m:float=.08
 min_distinct_view_translation_m:float=.01
 min_parallax_deg:float=1.
 max_reprojection_px:float=2.
 max_normal_rank_ratio:float=.08
 min_common_span_m:float=.20
 min_z:float=.3
 max_z:float=2.4
 horizontal_tolerance_deg:float=3.
 floor_association_tolerance_m:float=.05
 max_sensor_dt_s:float=.05
 max_track_gap_s:float=10.
 native_margin:int=4
 max_vertical_probe_m:float=1.

class LBD:
 def __init__(self,path):
  self.lib=ctypes.CDLL(str(Path(path).resolve()));self.fn=self.lib.boundary_lbd_describe;self.B=ctypes.POINTER(ctypes.c_ubyte);self.F=ctypes.POINTER(ctypes.c_float);self.fn.argtypes=[self.B,ctypes.c_int,ctypes.c_int,self.F,ctypes.c_int,self.B,self.B];self.fn.restype=ctypes.c_int;v=self.lib.boundary_lbd_opencv_version;v.restype=ctypes.c_char_p;self.version=v().decode()
 def describe(self,image,lines):
  image=np.ascontiguousarray(image,np.uint8);lines=np.ascontiguousarray(lines,np.float32).reshape(-1,4);desc=np.zeros((len(lines),32),np.uint8);valid=np.zeros(len(lines),np.uint8)
  if len(lines):
   code=self.fn(image.ctypes.data_as(self.B),image.shape[1],image.shape[0],lines.ctypes.data_as(self.F),len(lines),desc.ctypes.data_as(self.B),valid.ctypes.data_as(self.B))
   if code:raise RuntimeError('native LBD status '+str(code))
  return desc,valid.astype(bool)

def unit(x):
 x=np.asarray(x,float);n=np.linalg.norm(x)
 if not np.isfinite(x).all() or n<1e-12:raise ValueError('invalid vector')
 return x/n

def Kmatrix(K):fx,fy,cx,cy=K;return np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])

def detect(image,lbd,p=Policy(),mask=None):
 a=np.asarray(image)
 if a.ndim!=2 or a.dtype!=np.uint8:raise ValueError('native mono8 required')
 found=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(a)[0];lines=np.empty((0,4),np.float32) if found is None else found.reshape(-1,4);length=np.linalg.norm(lines[:,:2]-lines[:,2:],axis=1);order=np.argsort(-length);lines=lines[order[length[order]>=p.min_line_px]][:p.max_lines];h,w=a.shape
 keep=np.all((lines[:,[0,2]]>=p.native_margin)&(lines[:,[0,2]]<w-p.native_margin),axis=1)&np.all((lines[:,[1,3]]>=p.native_margin)&(lines[:,[1,3]]<h-p.native_margin),axis=1);lines=lines[keep]
 if mask is not None and len(lines):
  if np.asarray(mask).shape!=a.shape:raise ValueError('native self-mask shape mismatch')
  t=np.linspace(0,1,32);uv=lines[:,:2,None]*(1-t)+lines[:,2:,None]*t;u=np.rint(uv[:,0]).astype(int);v=np.rint(uv[:,1]).astype(int);lines=lines[~np.any(mask[v,u],axis=1)]
 desc,valid=lbd.describe(a,lines);return lines[valid],desc[valid]

def matches(desc_a,desc_b,p=Policy()):
 if not len(desc_a) or not len(desc_b):return []
 bf=cv2.BFMatcher(cv2.NORM_HAMMING);k=min(2,len(desc_b));ab=bf.knnMatch(desc_a,desc_b,k=k);ba=bf.match(desc_b,desc_a);reverse={m.queryIdx:m.trainIdx for m in ba};accepted=[]
 for row in ab:
  if not row:continue
  m=row[0]
  if m.distance>=p.hamming_max or reverse.get(m.trainIdx)!=m.queryIdx:continue
  if len(row)>1 and not m.distance<p.ratio*row[1].distance:continue
  accepted.append((m.queryIdx,m.trainIdx,float(m.distance),len(row)>1))
 return accepted

def interpretation(line,K,W):
 a=np.r_[line[:2],1.];b=np.r_[line[2:],1.];lc=np.cross(a,b);nc=unit(Kmatrix(K).T@lc);nw=unit(W[:3,:3]@nc);return nw,float(nw@W[:3,3])

def triangulate(observations,p=Policy(),require_distinct_exposures=True):
 """Each observed line defines a plane through actual camera centre. Floor/ToF
 is NEVER an argument to this solver, including for endpoints or scale."""
 if require_distinct_exposures and len({o['stamp_ns'] for o in observations})<p.min_views:return dict(status='insufficient_exposures')
 centres=np.array([o['W'][:3,3] for o in observations]);unique=[]
 for i in range(len(observations)):
  if all(np.linalg.norm(centres[i]-centres[j])>=p.min_distinct_view_translation_m for j in unique):unique.append(i)
 if len(unique)<p.min_views:return dict(status='insufficient_metric_distinct_views',distinct_views=len(unique),max_camera_baseline_m=float(np.linalg.norm(centres[:,None]-centres,axis=2).max()))
 selected=[observations[i] for i in unique];rows=[interpretation(o['line'],o['K'],o['W']) for o in selected];N=np.array([x[0] for x in rows]);b=np.array([x[1] for x in rows]);_,s,V=np.linalg.svd(N);direction=unit(V[-1]);rank_ratio=float(s[-1]/max(s[1],1e-12)) if len(s)>2 else 0.;parallax=float(np.degrees(np.arccos(np.clip(np.abs(N@N.T),0,1))).max());perp=centres-centres[0];perp-=np.outer(perp@direction,direction);perp_baseline=float(np.linalg.norm(perp[:,None]-perp,axis=2).max())
 quality=dict(distinct_exposures=len({o['stamp_ns'] for o in selected}),distinct_camera_views=len(selected),normal_singular_values=s.tolist(),normal_rank_ratio=rank_ratio,max_plane_parallax_deg=parallax,max_perpendicular_baseline_m=perp_baseline,camera_motion_singular_values=np.linalg.svd(centres-centres.mean(0),compute_uv=False).tolist())
 if rank_ratio>p.max_normal_rank_ratio:return dict(status='world_line_inconsistent_or_dynamic',quality=quality)
 if parallax<p.min_parallax_deg or perp_baseline<p.min_baseline_m or s[1]<1e-5:return dict(status='line_baseline_degenerate',quality=quality)
 Q=np.linalg.lstsq(np.vstack((N,direction)),np.r_[b,0.],rcond=None)[0];intervals=[];errors=[]
 for o in selected:
  W=o['W'];K=o['K'];span=[]
  for uv in o['line'].reshape(2,2):
   dc=np.array([(uv[0]-K[2])/K[0],(uv[1]-K[3])/K[1],1.]);dw=W[:3,:3]@dc;A=np.column_stack((direction,-dw));v=np.linalg.lstsq(A,W[:3,3]-Q,rcond=None)[0];span.append(v[0])
  span=sorted(span);qz=((Q-W[:3,3])@W[:3,:3])[2];tz=(direction@W[:3,:3])[2]
  if abs(tz)<1e-10:
   if not p.min_z<=qz<=p.max_z:return dict(status='line_outside_native_depth_admission',quality=quality)
  else:
   bounds=sorted([(p.min_z-qz)/tz,(p.max_z-qz)/tz]);span=[max(span[0],bounds[0]),min(span[1],bounds[1])]
  intervals.append(span)
 lo=max(v[0] for v in intervals);hi=min(v[1] for v in intervals)
 if hi-lo<p.min_common_span_m:return dict(status='insufficient_common_observed_extent',quality=quality,common_span_m=float(hi-lo))
 endpoints=np.stack((Q+lo*direction,Q+hi*direction));mid=(endpoints[0]+endpoints[1])*.5
 for o in selected:
  pc=(np.stack((endpoints[0],mid,endpoints[1]))-o['W'][:3,3])@o['W'][:3,:3];uv=pc[:,:2]/pc[:,2,None]*np.array(o['K'][:2])+np.array(o['K'][2:]);l=np.cross(np.r_[o['line'][:2],1.],np.r_[o['line'][2:],1.]);errors.extend((np.abs(np.c_[uv,np.ones(3)]@l)/np.linalg.norm(l[:2])).tolist())
 quality['maximum_reprojection_error_px']=max(errors);quality['common_span_m']=float(hi-lo)
 if max(errors)>p.max_reprojection_px:return dict(status='multiview_reprojection_rejected',quality=quality)
 return dict(status='metric_temporal_line',quality=quality,point_world=mid,direction_world=direction,endpoints_world=endpoints,selected_stamps=[o['stamp_ns'] for o in selected],selected_frame_indices=[o['frame_index'] for o in selected],selected_line_indices=[o['line_index'] for o in selected],source='interpretation-plane intersection of actual exposures; no floor projection, no raw-sensor claim')

def floor_validate(tri,frames_by_stamp,p=Policy()):
 if tri['status']!='metric_temporal_line':return dict(status='no_metric_line')
 checks=[];ups=[]
 for stamp in tri['selected_stamps']:
  f=frames_by_stamp[stamp];missing=[k for k in ('gravity_up_optical','gravity_stamp_ns','tof_range_m','tof_stamp_ns','tof_valid','tof_origin_optical_m','tof_direction_optical') if k not in f]
  if missing:return dict(status='missing_actual_floor_evidence',missing=missing,stamp_ns=stamp)
  if not bool(f['tof_valid']) or abs(int(f['tof_stamp_ns'])-stamp)*1e-9>p.max_sensor_dt_s or abs(int(f['gravity_stamp_ns'])-stamp)*1e-9>p.max_sensor_dt_s:return dict(status='invalid_or_unpaired_actual_tof_gravity',stamp_ns=stamp)
  g=unit(f['gravity_up_optical']);d=unit(f['tof_direction_optical']);r=float(f['tof_range_m']);origin=np.asarray(f['tof_origin_optical_m'],float)
  if not np.isfinite(r) or r<=0 or d@g>-.9 or origin.shape!=(3,):return dict(status='tof_not_a_valid_downward_ray',stamp_ns=stamp)
  hit=origin+r*d;h=float(g@hit);W=f['W'];up=W[:3,:3]@g;ups.append(up);pc=(tri['endpoints_world']-W[:3,3])@W[:3,:3];residual=pc@g-h;checks.append(dict(stamp_ns=stamp,tof_floor_height_camera_m=h,endpoint_height_residuals_m=residual.tolist()))
  if h>=-.05 or np.max(np.abs(residual))>p.floor_association_tolerance_m:return dict(status='metric_line_not_on_actual_tof_floor',checks=checks)
 up=unit(np.mean(ups,axis=0));tilt=math.degrees(math.asin(float(np.clip(abs(tri['direction_world']@up),0,1))))
 if tilt>p.horizontal_tolerance_deg:return dict(status='metric_line_not_horizontal',line_tilt_deg=tilt,checks=checks)
 return dict(status='tof_gravity_consistent_horizontal_junction_hypothesis',checks=checks,up_world=up,line_tilt_deg=tilt,association='Local horizontal surface / wall-junction hypothesis only; ToF not a semantic proof of floor or wall')

def raw_contradictions(tri,floor,frame,p=Policy()):
 if floor.get('status')!='tof_gravity_consistent_horizontal_junction_hypothesis':return dict(status='not_floor_qualified')
 W=frame['W'];K=frame['K'];Q=tri['point_world'];t=tri['direction_world'];up=floor['up_world'];n=unit(np.cross(t,up));delta=Q-W[:3,3]
 if n@delta<0:n=-n
 nC=W[:3,:3].T@n;b=float(n@delta);z=frame['measured_stereo'];v,u=np.nonzero(np.isfinite(z)&(z>=p.min_z)&(z<=p.max_z));raw=z[v,u];dc=np.column_stack(((u-K[2])/K[0],(v-K[3])/K[1],np.ones(len(u))));den=dc@nC;predict=np.full(len(u),np.nan);positive=den>1e-6;predict[positive]=b/den[positive];PW=dc@W[:3,:3].T*predict[:,None]+W[:3,3];along=(PW-Q)@t;height=(PW-Q)@up;half=np.linalg.norm(tri['endpoints_world'][1]-tri['endpoints_world'][0])*.5;region=np.isfinite(predict)&(predict>=p.min_z)&(predict<=p.max_z)&(np.abs(along)<=half)&(height>.06)&(height<p.max_vertical_probe_m);tol=np.maximum(.03,.03*predict);farther=region&(raw>predict+tol);closer=region&(raw<predict-tol);# Retain one connected image component seeded only by the independently
 # reconstructed finite metric line and its explicit vertical-wall hypothesis.
 component_id=None;component_seed_fraction=0.
 if 'left' in frame:
  image=frame['left'];edge=cv2.dilate(cv2.Canny(image,8,16),np.ones((3,3),np.uint8))>0
  if 'edge_working_copy' in frame:edge|=cv2.dilate(cv2.Canny(frame['edge_working_copy'],8,16),np.ones((3,3),np.uint8))>0
  if 'self_mask' in frame and frame['self_mask'] is not None:edge|=frame['self_mask']
  edge[:p.native_margin]=True;edge[-p.native_margin:]=True;edge[:,:p.native_margin]=True;edge[:,-p.native_margin:]=True
  _,labels=cv2.connectedComponents((~edge).astype(np.uint8),connectivity=4);seed=[]
  for along_seed in np.linspace(-half*.8,half*.8,9):
   for hseed in (.08,.12,.20):
    world=Q+along_seed*t+hseed*up;point=(world-W[:3,3])@W[:3,:3]
    if point[2]<=0:continue
    uv=point[:2]/point[2]*np.array(K[:2])+np.array(K[2:]);xx,yy=np.rint(uv).astype(int)
    if 0<=xx<image.shape[1] and 0<=yy<image.shape[0] and labels[yy,xx]>0:seed.append(int(labels[yy,xx]))
  if not seed:return dict(status='no_unambiguous_wall_side_image_component',raw_farther_contradictions=0,raw_nonfloor_wall_support=0)
  ids,ct=np.unique(seed,return_counts=True);best=int(np.argmax(ct));component_id=int(ids[best]);component_seed_fraction=float(ct[best]/len(seed))
  if component_seed_fraction<.6:return dict(status='ambiguous_wall_side_image_component',raw_farther_contradictions=0,raw_nonfloor_wall_support=0)
  region&=labels[v,u]==component_id;farther=region&(raw>predict+tol);closer=region&(raw<predict-tol)
 onwall=region&(np.abs(raw-predict)<=tol);realPW=(dc*raw[:,None])@W[:3,:3].T+W[:3,3];realheight=(realPW-Q)@up;wall=onwall&(realheight>.06);return dict(status='raw_farther_surface_contradiction' if int(farther.sum())>=3 else 'no_farther_raw_contradiction_observed',raw_farther_contradictions=int(farther.sum()),farther_raw_floor_plane_support=int((farther&(np.abs(realheight)<.05)).sum()),image_component_id=component_id,image_component_seed_fraction=component_seed_fraction,raw_nearer_foreground=int(closer.sum()),raw_nonfloor_wall_support=int(wall.sum()),unknown_not_evidence=True,wall_normal_world=n,wall_point_world=Q)

def relative_clone_covariance(T0,T1,P12,orientation_error_frame):
 """Exact cross-cov propagation prototype. Covariance order theta0,p0,theta1,p1;
 translations in world; explicit world-left or camera-right rotation tangent.
 This does not map an undocumented native OV/JPL covariance convention."""
 from scipy.spatial.transform import Rotation
 P=np.asarray(P12,float)
 if P.shape!=(12,12) or not np.isfinite(P).all() or not np.allclose(P,P.T,atol=1e-9):raise ValueError('actual symmetric12x12 joint covariance required')
 if orientation_error_frame not in ('world_left','camera_right'):raise ValueError('explicit verified tangent convention required')
 nominal=np.linalg.inv(T0)@T1;J=np.zeros((6,12));eps=1e-6
 def residual(delta):
  out=[]
  for i,T in enumerate((T0,T1)):
   A=np.array(T,copy=True);d=delta[6*i:6*i+6];R=Rotation.from_rotvec(d[:3]).as_matrix();A[:3,:3]=R@A[:3,:3] if orientation_error_frame=='world_left' else A[:3,:3]@R;A[:3,3]+=d[3:];out.append(A)
  relative=np.linalg.inv(out[0])@out[1];return np.r_[Rotation.from_matrix(relative[:3,:3]@nominal[:3,:3].T).as_rotvec(),relative[:3,3]-nominal[:3,3]]
 for k in range(12):
  d=np.zeros(12);d[k]=eps;J[:,k]=(residual(d)-residual(-d))/(2*eps)
 return J@P@J.T,J
