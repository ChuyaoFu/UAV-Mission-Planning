"""Relative floor-line hints from grayscale and measured stereo only.
No CAD registration, surveyed floor height, pose correction or free-space fill.
"""
import cv2,json,numpy as np
from pathlib import Path
# Optional measured-support gates for higher cruise altitude (absent file = original values).
_GATES=dict(max_depth_m=2.4,floor_band_m=.055,residual_p90_m=.025,contrast_min=12.,min_len_px=60.,clahe=0.)
_OVERRIDE=Path(__file__).resolve().parents[1]/'config/guideline_detector.json'
if _OVERRIDE.exists():_GATES.update({k:float(v) for k,v in json.loads(_OVERRIDE.read_text()).items() if k in _GATES})
class GuidelineDetector:
 def __init__(self):self.lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD);self.gates=dict(_GATES)
 def detect(self,image,depth,K,camera_position,camera_rotation,floor_z):
  if self.gates['clahe']>0:image=cv2.createCLAHE(clipLimit=self.gates['clahe'],tileGridSize=(8,8)).apply(image)
  gray=cv2.resize(image,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA);found=self.lsd.detect(gray)[0];segments=[]
  if found is None:return segments
  for line in found.reshape(-1,4):
   x1,y1,x2,y2=line*2.;length=np.hypot(x2-x1,y2-y1)
   if length<self.gates['min_len_px']:continue
   t=np.linspace(.05,.95,40);u=x1+(x2-x1)*t;v=y1+(y2-y1)*t;normal=np.array([-(y2-y1),x2-x1])/length
   def samples(offset):
    xx=np.clip(np.rint(u+normal[0]*offset).astype(int),0,image.shape[1]-1);yy=np.clip(np.rint(v+normal[1]*offset).astype(int),0,image.shape[0]-1);return image[yy,xx].astype(float)
   if np.median(np.abs(samples(6)-samples(-6)))<self.gates['contrast_min']:continue
   xx=np.clip(np.rint(u).astype(int),0,depth.shape[1]-1);yy=np.clip(np.rint(v).astype(int),0,depth.shape[0]-1);z=depth[yy,xx];good=np.isfinite(z)&(z>.25)&(z<self.gates['max_depth_m'])
   if good.sum()<8:continue
   rays=np.column_stack([(u[good]-K[2])/K[0],(v[good]-K[3])/K[1],np.ones(good.sum())]);points=camera_rotation.apply(rays*z[good,None])+camera_position;points=points[np.abs(points[:,2]-floor_z)<self.gates['floor_band_m']]
   if len(points)<8:continue
   center=points.mean(0);_,_,axes=np.linalg.svd(points[:,:2]-center[:2],full_matrices=False);direction=axes[0];s=(points[:,:2]-center[:2])@direction;span=np.percentile(s,[10,90]);residual=np.abs((points[:,:2]-center[:2])@np.array([-direction[1],direction[0]]))
   if span[1]-span[0]<.25 or np.percentile(residual,90)>self.gates['residual_p90_m']:continue
   ends=[np.r_[center[:2]+direction*a,floor_z].tolist() for a in span];segments.append({'endpoints':ends,'support_points':len(points),'source':'grayscale line with measured stereo floor support'})
  return sorted(segments,key=lambda x:x['support_points'],reverse=True)[:12]
