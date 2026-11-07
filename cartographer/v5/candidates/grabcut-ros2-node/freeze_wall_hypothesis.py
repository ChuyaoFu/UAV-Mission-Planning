#!/usr/bin/env python3
"""Freeze a specifically UNQUALIFIED structural surface for independent scoring.
Never emits free-ray bounds, raw anchors or occupied endpoints. Actual native
same-posterior clone means only; no nav pose/covariance mixing.
"""
import argparse,json,hashlib
from pathlib import Path
import cv2,numpy as np
from temporal_boundary import unit

def component(image,mask,W,K,Q,t,up,half):
 edge=cv2.dilate(cv2.Canny(image,8,16),np.ones((3,3),np.uint8))>0;edge|=mask;edge[:4]=True;edge[-4:]=True;edge[:,:4]=True;edge[:,-4:]=True;_,labels=cv2.connectedComponents((~edge).astype(np.uint8),connectivity=4);seed=[]
 for s in np.linspace(-half*.8,half*.8,9):
  for h in (.08,.12,.20):
   pc=(Q+s*t+h*up-W[:3,3])@W[:3,:3]
   if pc[2]<=0:continue
   uv=pc[:2]/pc[2]*K[:2]+K[2:];x,y=np.rint(uv).astype(int)
   if 0<=x<image.shape[1] and 0<=y<image.shape[0] and labels[y,x]>0:seed.append(int(labels[y,x]))
 if not seed:return labels,None,edge
 ids,ct=np.unique(seed,return_counts=True);i=int(np.argmax(ct));return labels,int(ids[i]) if ct[i]/len(seed)>=.6 else None,edge

def main():
 p=argparse.ArgumentParser();p.add_argument('--native-diagnostics',required=True);p.add_argument('--packets',required=True);p.add_argument('--features',required=True);p.add_argument('--mask-dir',required=True);p.add_argument('--track-id',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);doc=json.loads(Path(a.native_diagnostics).read_text());row=next(r for r in doc['best_moving_tracks'] if r['track_id']==a.track_id);tri=row['triangulation'];floor=row['floor_association']
 if floor['status']!='tof_gravity_consistent_horizontal_junction_hypothesis':raise ValueError('not a measured floor-consistent metric line')
 stamp=max(tri['selected_stamps']);frame_index=tri['selected_frame_indices'][tri['selected_stamps'].index(stamp)];source=Path(a.packets)/f'packet-{frame_index:03d}.npz';d=np.load(source);W=np.array(row['native_selected_camera_poses'][str(stamp)],float);K=d['K'];B=float(d['baseline']);L=d['left'];R=d['right'];raw=d['measured_stereo'];Q=np.array(tri['point_world']);t=unit(tri['direction_world']);up=unit(floor['up_world']);half=np.linalg.norm(np.array(tri['endpoints_world'])[1]-np.array(tri['endpoints_world'])[0])*.5;normal=unit(np.cross(t,up));delta=Q-W[:3,3]
 if normal@delta<0:normal=-normal
 nc=W[:3,:3].T@normal;b=float(normal@delta);v,u=np.indices(raw.shape);ray=np.stack(((u-K[2])/K[0],(v-K[3])/K[1],np.ones(raw.shape)),axis=-1);den=ray@nc;z=np.full(raw.shape,np.nan);positive=den>1e-6;z[positive]=b/den[positive];along=(W[:3,3]-Q)@t+z*(ray@(W[:3,:3].T@t));height=(W[:3,3]-Q)@up+z*(ray@(W[:3,:3].T@up));m0=cv2.imread(str(Path(a.mask_dir)/'self-mask0.png'),0)>0;m1=cv2.imread(str(Path(a.mask_dir)/'self-mask1.png'),0)>0;lab,cid,edge=component(L,m0,W,K,Q,t,up,half)
 WR=W.copy();WR[:3,3]+=W[:3,:3]@np.array([B,0,0]);rlab,rcid,redge=component(R,m1,WR,K,Q,t,up,half);roi=(z>=.3)&(z<=2.4)&(height>.06)&(height<1.0)&(np.abs(along)<=half-.02)&~edge&~m0
 if cid is None or rcid is None:roi[:]=False
 else:roi&=lab==cid
 xr=(u-K[0]*B/np.maximum(z,1e-6)).astype(np.float32);yr=v.astype(np.float32);rightmask=cv2.remap((m1|redge).astype(np.uint8),xr,yr,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=1)>0;rightlabels=cv2.remap(rlab.astype(np.float32),xr,yr,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0);warped=cv2.remap(R,xr,yr,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);photo=cv2.boxFilter(np.abs(L.astype(np.float32)-warped.astype(np.float32)),cv2.CV_32F,(5,5));roi&=(xr>=4)&(xr<R.shape[1]-4)&~rightmask&(rightlabels==rcid)&(photo<5)
 validraw=np.isfinite(raw)&(raw>0);foreground=validraw&(raw+np.maximum(.03,.03*raw)<z);foreground=cv2.dilate(foreground.astype(np.uint8),np.ones((7,7),np.uint8))>0;contrad=roi&validraw&(raw>z+np.maximum(.03,.03*z));roi&=np.isnan(raw)&~foreground
 candidate=np.full(raw.shape,np.nan,np.float32);candidate[roi]=z[roi].astype(np.float32);free=np.full(raw.shape,np.nan,np.float32);name=f'track-{a.track_id:03d}-exposure-{stamp}';path=out/(name+'.npz');fe=np.load(Path(a.features)/source.name)
 np.savez_compressed(path,stamp_ns=np.int64(stamp),stamp=np.float64(stamp*1e-9),K=K,baseline=B,inferred_depth=candidate,free_ray_lower_bound=free,original_measured_depth=raw,native_world_from_optical=W,source_native_joint_record=np.array(row['source_native_full_joint_record']),observed_line_pixels=fe['left_lines'][tri['selected_line_indices'][tri['selected_stamps'].index(stamp)]],support_stamps_ns=np.array(tri['selected_stamps'],np.int64),candidate_qualified=np.bool_(False),navigation_enabled=np.bool_(False),semantics=np.array('UNQUALIFIED inferred vertical-wall hypothesis for independent scoring ONLY; no raw/occupied/free bound'))
 info=dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),track_id=a.track_id,exposure_stamp_ns=stamp,native_posterior_record_index=row['native_posterior_epoch'],source_packet=str(source.resolve()),source_packet_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),surface_pixels=int(roi.sum()),observed_line_common_span_m=2*half,left_component=cid,right_component=rcid,current_roi_farther_raw_contradictions=int(contrad.sum()),wall_positive_native_support=max((c.get('raw_nonfloor_wall_support',0) for c in row.get('raw_crosschecks',[])),default=0),candidate_qualified=False,qualification_reason='native temporal metric line/floor evidence exists, but no positive vertical-wall depth support; absence of contradiction is not wall evidence; noncollinear motion/categorical association unresolved',native_mean_covariance_pairing='same actual posterior record only; nav/camera_pose not used for prediction geometry',conditional_model='local vertical plane through independently triangulated horizontal line; only observed horizontal extent and left/right edge-bounded components, max1m local probe, no scene dimensions/heights',free_ray_pixels=0,navigation_enabled=False)
 (out/(name+'.json')).write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info))
if __name__=='__main__':main()
