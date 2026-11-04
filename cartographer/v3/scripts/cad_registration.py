#!/usr/bin/python3
"""Robust distance-field registration against published CAD, never flight truth.
Only a small, regularized XY/yaw correction is estimated; floor-only geometry
cannot pull the horizontal position because floor points are excluded.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt,map_coordinates
from scipy.optimize import least_squares
class Registration:
 def __init__(self,path='/work/cartographer/results/cad-grid.npz'):
  data=np.load(path);self.origin=data['origin'];self.res=float(data['res']);surface=data['surface'];self.distance=distance_transform_edt(~surface,sampling=[self.res]*3).astype(np.float32);self.last=-1;self.events=[];self.health=None
 def sample(self,points):return map_coordinates(self.distance,((points-self.origin)/self.res).T,order=1,mode='constant',cval=1.,prefilter=False)
 def update(self,points,pivot,stamp):
  if stamp-self.last<.18:return None
  self.last=stamp
  good=np.isfinite(points).all(1)&(points[:,2]>.16)&(points[:,2]<1.80)&(np.linalg.norm(points-pivot,axis=1)<3.0)
  p=points[good]
  if len(p)<100:return None
  p=p[::max(1,len(p)//350)];initial=self.sample(p);p=p[initial<.30]
  if len(p)<80:return None
  def moved(x):
   c=np.cos(x[2]);s=np.sin(x[2]);r=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
   return (p-pivot)@r.T+pivot+np.r_[x[:2],0.]
  def residual(x):return np.r_[self.sample(moved(x)),x[0]*.3,x[1]*.3,x[2]*.8]
  def jacobian(x):
   columns=[]
   for j,h in enumerate([.003,.003,.002]):
    a=x.copy();b=x.copy();a[j]+=h;b[j]-=h;columns.append((residual(a)-residual(b))/(2*h))
   return np.array(columns).T
  try:result=least_squares(residual,[0.,0.,0.],jac=jacobian,bounds=([-.18,-.18,-.10],[.18,.18,.10]),loss='soft_l1',f_scale=.025,max_nfev=18)
  except (ValueError,FloatingPointError):return None
  if not np.isfinite(result.x).all():return None
  before=float(np.sqrt(np.mean(self.sample(p)**2)));after=float(np.sqrt(np.mean(self.sample(moved(result.x))**2)))
  j=jacobian(result.x)[:-3];weights=1/np.sqrt(1+(self.sample(moved(result.x))/.025)**2);information=j.T@(weights[:,None]*j)
  eigen=np.linalg.eigvalsh(information);std=np.full(3,1e3)
  if eigen[0]>1e-5:
   covariance=np.linalg.inv(information)*max(after,.025)**2*4.;std=np.sqrt(np.maximum(covariance.diagonal(),0))
  supported=after<.06 and std[:2].max()<.08 and std[2]<.06
  self.health={'stamp':stamp,'supported':bool(supported and before<.09 and not np.any(np.abs(result.x)>np.array([.17,.17,.095]))),'fit_rms_m':after,'xy_yaw_std_proxy':std.tolist(),'information_eigenvalues':eigen.tolist(),'points':len(p),'source':'live passive depth registered to published CAD; no truth; approximate Jacobian uncertainty proxy'}
  if not supported or np.any(np.abs(result.x)>np.array([.17,.17,.095])) or before<.018 or after>before*.94 or after>.06:return None
  delta=result.x.copy();translation=np.linalg.norm(delta[:2]);delta[:2]*=min(1.,.012/max(translation,1e-8));delta[2]=np.clip(delta[2],-.006,.006)
  self.events.append({'stamp':stamp,'before_rms_m':before,'after_rms_m':after,'points':len(p),'unfiltered_solution':result.x.tolist(),'applied_xy_yaw':delta.tolist()})
  return delta
