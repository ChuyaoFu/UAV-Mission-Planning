"""Fixed400p/five-iteration conditional model; no IO, publishers or map writes."""
import time
import numpy as np,cv2
from temporal_boundary import unit
from seeded_grabcut import segment
from freeze_wall_hypothesis import component

def surface(frame,segmented,W,tri,floor,m0,m1,check=lambda:None,segmenter=None,decision_stamp_ns=None):
 left=frame['left'];right=frame['right'];raw=frame['measured_stereo'];K=frame['K'];baseline=float(frame['baseline']);W=np.asarray(W,float);Q=np.array(tri['point_world']);t=unit(tri['direction_world']);up=unit(floor['up_world']);half=np.linalg.norm(np.array(tri['endpoints_world'])[1]-np.array(tri['endpoints_world'])[0])*.5;n=unit(np.cross(t,up));delta=Q-W[:3,3]
 if n@delta<0:n=-n
 nc=W[:3,:3].T@n;b=float(n@delta);v,u=np.indices(raw.shape);rays=np.stack(((u-K[2])/K[0],(v-K[3])/K[1],np.ones(raw.shape)),axis=-1);den=rays@nc;prediction=np.full(raw.shape,np.nan);good=den>1e-6;prediction[good]=b/den[good];along=(W[:3,3]-Q)@t+prediction*(rays@(W[:3,:3].T@t));height=(W[:3,3]-Q)@up+prediction*(rays@(W[:3,:3].T@up));leftseg=segmented['label_floor_nonfloor_unknown'];WR=W.copy();WR[:3,3]+=W[:3,:3]@np.array([baseline,0,0]);lab,cid,edge=component(left,m0,W,K,Q,t,up,half);rlab,rcid,redge=component(right,m1,WR,K,Q,t,up,half)
 # Right GrabCut seed depths are explicitly DERIVED by calibrated reprojection
 # of original LEFT raw measurements, never a synthetic right sensor product.
 rz=np.full(raw.shape,np.nan,np.float32);yy,xx=np.nonzero(np.isfinite(raw)&(raw>0));zz=raw[yy,xx];xr=np.rint(xx-K[0]*baseline/zz).astype(int);ok=(xr>=0)&(xr<raw.shape[1]);flat=np.full(raw.size,np.inf,np.float32);np.minimum.at(flat,yy[ok]*raw.shape[1]+xr[ok],zz[ok]);rz=flat.reshape(raw.shape);rz[~np.isfinite(rz)]=np.nan;rightframe=dict(frame,left=right,measured_stereo=rz,tof_origin_optical_m=frame['tof_origin_optical_m']-np.array([baseline,0,0]));check();rseg,rstats,_=(segmenter(rightframe,m1,'right',decision_stamp_ns) if segmenter is not None else segment(rightframe,m1));check()
 roi=(prediction>=.3)&(prediction<=2.4)&(height>.06)&(height<1.)&(abs(along)<=half-.02)&(leftseg==2)&~m0&~edge
 if cid is None or rcid is None or rseg is None:roi[:]=False
 else:roi&=lab==cid
 xr=(u-K[0]*baseline/np.maximum(prediction,1e-6)).astype(np.float32);yr=v.astype(np.float32);rightbad=cv2.remap((m1|redge).astype(np.uint8),xr,yr,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=1)>0;rightlabels=cv2.remap(rlab.astype(np.float32),xr,yr,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0);warped=cv2.remap(right,xr,yr,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);photo=cv2.boxFilter(abs(left.astype(np.float32)-warped.astype(np.float32)),cv2.CV_32F,(5,5));roi&=(xr>=4)&(xr<raw.shape[1]-4)&~rightbad&(rightlabels==rcid)&(photo<5)
 if rseg is not None:
  rnonfloor=cv2.remap((rseg['label_floor_nonfloor_unknown']==2).astype(np.uint8),xr,yr,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)>0;roi&=rnonfloor
 validraw=np.isfinite(raw)&(raw>0);tol=np.maximum(.03,.03*prediction);farther=roi&validraw&(raw>prediction+tol);foreground=validraw&(raw+np.maximum(.03,.03*raw)<prediction);foreground=cv2.dilate(foreground.astype(np.uint8),np.ones((7,7),np.uint8))>0;roi&=np.isnan(raw)&~foreground
 # Any actual farther return in the applicable component rejects the whole candidate.
 rejected=int(farther.sum())>=3
 if rejected:roi[:]=False
 surface=np.full(raw.shape,np.nan,np.float32);surface[roi]=prediction[roi].astype(np.float32);norm=np.linalg.norm(rays,axis=-1);reserve=np.sqrt(3.)*.1+.1;prefix=np.full(raw.shape,np.nan,np.float32);prefix[roi]=(prediction[roi]-reserve/norm[roi]).astype(np.float32);prefix[prefix<=0]=np.nan;free=np.full(raw.shape,np.nan,np.float32)
 return dict(stamp_ns=int(frame['stamp_ns']),stamp_s=float(frame['stamp_ns'])*1e-9,K=K,baseline=baseline,inferred_depth=surface,candidate_free_prefix_endpoint_z_m=prefix,free_ray_lower_bound=free,original_measured_depth=raw,native_world_from_optical=W,support_stamps_ns=np.array(tri['selected_stamps'],np.int64),candidate_qualified=False,navigation_enabled=False,semantics='INFERRED conditional free-only virtual termination; never raw/occupied'),dict(farther_raw_contradictions=int(farther.sum()),surface_pixels=int(np.isfinite(surface).sum()),right_segmentation=rstats,left_component=cid,right_component=rcid,rejected_by_farther_raw=rejected)
