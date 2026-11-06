"""Resolution-generic cut LSD extraction in ORIGINAL native-pixel units.
Working label names are legacy; their dimensions determine actual scale.
No camera intrinsics/image coordinates are resized or reinterpreted.
"""
import cv2,numpy as np
NATIVE_POLICY=dict(min_length_px=80.,opposite_side_offset_px=6.,border_px=4.,samples_per_side=61,min_side_support=.8,max_lines=8)

def working_to_native(points,working_shape,native_shape=(800,1280)):
 h,w=working_shape;H,W=native_shape;scale=np.array([w/W,h/H],float)
 if min(scale)<=0 or not np.isclose(scale[0],scale[1],atol=1e-12):raise ValueError('isotropic actual working/native shape required')
 return (np.asarray(points,dtype=np.float64)+.5)/scale-.5

def extract(res,native_shape=(800,1280)):
 labels=None
 for key in ('label_working','label_seg','label400'):
  if key in res:labels=np.asarray(res[key]);break
 if labels is None or labels.ndim!=2:raise ValueError('actual working label map required')
 H,W=map(int,native_shape);h,w=labels.shape;scale=np.array([w/W,h/H],float)
 if min(scale)<=0 or not np.isclose(scale[0],scale[1],atol=1e-12):raise ValueError('working labels must preserve native aspect ratio')
 found=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect((labels==1).astype(np.uint8)*255)[0];lines=[]
 if found is None:return lines
 for line_work in found.reshape(-1,4):
  a,b=working_to_native(line_work.reshape(2,2),labels.shape,native_shape);delta=b-a;length=float(np.linalg.norm(delta))
  if length<NATIVE_POLICY['min_length_px']:continue
  t=delta/length;n=np.array([-t[1],t[0]]);sides=[]
  for sign in (-1,1):
   p=a[None,:]+np.linspace(.05,.95,61)[:,None]*delta+sign*6.*n
   native_valid=(p[:,0]>=4)&(p[:,0]<W-4)&(p[:,1]>=4)&(p[:,1]<H-4);xy=np.rint((p+.5)*scale-.5).astype(int);valid=native_valid&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h);values=labels[xy[valid,1],xy[valid,0]];sides.append((float((values==1).mean()) if len(values) else 0.,float((values==2).mean()) if len(values) else 0.))
  support=max(min(sides[0][0],sides[1][1]),min(sides[1][0],sides[0][1]))
  if support>=.8:lines.append(dict(line=np.r_[a,b].astype(np.float32),length=length,side_support=support,working_shape=[h,w],native_shape=[H,W],native_policy=NATIVE_POLICY.copy()))
 return sorted(lines,key=lambda item:-item['length'])[:8]
