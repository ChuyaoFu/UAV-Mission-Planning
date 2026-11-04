"""Passive stereo with independent right-to-left and photometric validation."""
import time
import cv2,numpy as np
class Matcher:
 def __init__(self):
  opts=dict(numDisparities=128,blockSize=5,P1=200,P2=800,disp12MaxDiff=1,uniquenessRatio=12,speckleWindowSize=80,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
  self.left=cv2.StereoSGBM_create(minDisparity=0,**opts);self.right=cv2.StereoSGBM_create(minDisparity=-127,**opts)
 def compute(self,a,b,f,baseline):
  start=time.perf_counter();d=self.left.compute(a,b).astype(np.float32)/16;dr=self.right.compute(b,a).astype(np.float32)/16;y,x=np.indices(d.shape);xr=x-d;inside=(xr>=0)&(xr<d.shape[1]-1)
  sampled=cv2.remap(dr,xr.astype(np.float32),y.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=-128)
  consistency=abs(d+sampled);warped=cv2.remap(b,xr.astype(np.float32),y.astype(np.float32),cv2.INTER_LINEAR)
  photo=cv2.boxFilter(abs(a.astype(np.float32)-warped.astype(np.float32)),cv2.CV_32F,(5,5))
  good=(d>0)&inside&(sampled>-128)&(consistency<.75)&(photo<20.)
  depth=np.full_like(d,np.nan);depth[good]=f*baseline/d[good];depth[(depth<.25)|(depth>8)]=np.nan
  return depth,{'compute_ms':(time.perf_counter()-start)*1000,'left_valid_fraction':float((d>0).mean()),'lr_valid_fraction':float(good.mean()),'lr_max_error_px':.75,'photometric_max_mean':20.}
