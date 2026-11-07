"""Passive stereo with independent right-to-left and photometric validation."""
import time
import cv2,numpy as np
class Matcher:
 def __init__(self):
  opts=dict(numDisparities=128,blockSize=5,P1=200,P2=800,disp12MaxDiff=1,uniquenessRatio=12,speckleWindowSize=80,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
  self.left=cv2.StereoSGBM_create(minDisparity=0,**opts);self.right=cv2.StereoSGBM_create(minDisparity=-127,**opts)
 def compute(self,a,b,f,baseline,self_occlusion=None,lr_tolerance=.75):
  start=time.perf_counter();d=self.left.compute(a,b).astype(np.float32)/16;dr=self.right.compute(b,a).astype(np.float32)/16;y,x=np.indices(d.shape);xr=x-d;inside=(xr>=0)&(xr<d.shape[1]-1)
  sampled=cv2.remap(dr,xr.astype(np.float32),y.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=-128)
  consistency=abs(d+sampled);warped=cv2.remap(b,xr.astype(np.float32),y.astype(np.float32),cv2.INTER_LINEAR)
  photo=cv2.boxFilter(abs(a.astype(np.float32)-warped.astype(np.float32)),cv2.CV_32F,(5,5))
  mean=cv2.boxFilter(a.astype(np.float32),cv2.CV_32F,(9,9));variance=cv2.boxFilter(a.astype(np.float32)**2,cv2.CV_32F,(9,9))-mean**2
  bm=cv2.boxFilter(b.astype(np.float32),cv2.CV_32F,(9,9));bv=cv2.boxFilter(b.astype(np.float32)**2,cv2.CV_32F,(9,9))-bm**2;right_variance=cv2.remap(bv,xr.astype(np.float32),y.astype(np.float32),cv2.INTER_LINEAR)
  texture=(variance>=4.)&(right_variance>=4.)
  good=(d>0)&inside&(sampled>-128)&(consistency<lr_tolerance)&(photo<20.)&texture
  if self_occlusion is not None:
   left_mask,right_mask=self_occlusion;right_block=cv2.remap(right_mask.astype(np.uint8),xr.astype(np.float32),y.astype(np.float32),cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=1);good&=(~left_mask)&(right_block==0)
  depth=np.full_like(d,np.nan);depth[good]=f*baseline/d[good];depth[(depth<.25)|(depth>8)]=np.nan
  return depth,{'compute_ms':(time.perf_counter()-start)*1000,'left_valid_fraction':float((d>0).mean()),'lr_valid_fraction':float(good.mean()),'lr_max_error_px':lr_tolerance,'photometric_max_mean':20.,'minimum_patch_std_gray_levels':2.,'texture_supported_fraction':float(texture.mean())}

 def compute_scaled(self,a,b,f,baseline,self_occlusion=None,scale=1.):
  if scale not in [1.,.5]:raise ValueError('Only native or twofold measured-image reduction is supported')
  if scale==1.:return self.compute(a,b,f,baseline,self_occlusion)
  size=(a.shape[1]//2,a.shape[0]//2)
  if a.shape[1]%2 or a.shape[0]%2:raise ValueError('Twofold stereo requires even image dimensions')
  masks=None if self_occlusion is None else tuple(cv2.resize(m.astype(np.float32),size,interpolation=cv2.INTER_AREA)>0 for m in self_occlusion)
  left=cv2.resize(a,size,interpolation=cv2.INTER_AREA);right=cv2.resize(b,size,interpolation=cv2.INTER_AREA)
  depth,quality=self.compute(left,right,f*scale,baseline,masks,lr_tolerance=.75*scale)
  # Nearest resampling preserves every invalid measurement. It neither fills
  # holes nor adds geometric information. The output keeps the input ray grid.
  depth=cv2.resize(depth,(a.shape[1],a.shape[0]),interpolation=cv2.INTER_NEAREST)
  quality.update(matching_resolution_wh=list(size),matching_scale=scale,output_grid='native image grid, nearest resampling; invalid depth remains invalid',native_equivalent_lr_tolerance_px=.75)
  return depth,quality
