#!/usr/bin/python3
"""Passive image aperture edges registered to the published CAD window.
Uses actual RGB and inertial attitude; never reads evaluation or truth topics.
"""
import json
from pathlib import Path
import cv2,numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
class WindowLocalization:
 def __init__(self):
  self.u=np.array([.34201974,.93969277]);self.n=np.array([.93969277,-.34201974]);self.center=self.u*(-2.295955)+self.n*.334498
  self.edges=[]
  for n in [.283697,.385297]:
   corners=np.array([np.r_[self.u*u+self.n*n,z] for u,z in [(-2.75316,.3048),(-1.83875,.3048),(-1.83875,1.2192),(-2.75316,1.2192)]])
   self.edges.extend(np.concatenate([np.linspace(corners[i],corners[(i+1)%4],35) for i in range(4)]))
  self.edges=np.array(self.edges);self.last=-1;self.events=[];self.health=None;self.debug=None
  pitch=json.loads(Path('/work/cartographer/v2/config/model-manifest.json').read_text())['camera_pitch_down_degrees'];self.optical=Rotation.from_euler('y',pitch,degrees=True)*Rotation.from_matrix([[0,0,1],[-1,0,0],[0,-1,0]])
 def update(self,m,body,rotation):
  stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
  if stamp-self.last<.15 or np.linalg.norm(body[:2]-self.center)>2.1 or (body[:2]-self.center)@self.n<.45:return None
  self.last=stamp
  if m.encoding not in ['rgb8','bgr8','mono8']:return None
  if m.encoding=='mono8':gray=np.ndarray((m.height,m.width),np.uint8,buffer=m.data,strides=(m.step,1))
  else:
   image=np.ndarray((m.height,m.width,3),np.uint8,buffer=m.data,strides=(m.step,3,1));gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY if m.encoding=='rgb8' else cv2.COLOR_BGR2GRAY)
  canny=cv2.Canny(gray,45,110);distance=cv2.distanceTransform((canny==0).astype(np.uint8),cv2.DIST_L2,5)
  def project(a,points):
   rot=Rotation.from_euler('z',a[2])*rotation;camera=body+np.r_[a[:2],0.]+rot.apply([.12,.0375,.055]);p=(rot*self.optical).inv().apply(points-camera);return p[:,:2]/np.maximum(.1,p[:,2,None])*379.9173862805374+[320,200],p[:,2]
  uv,z=project(np.zeros(3),self.edges);ids=(z>.25)&(uv[:,0]>8)&(uv[:,0]<632)&(uv[:,1]>8)&(uv[:,1]<392)
  points=self.edges[ids]
  if len(points)<80:return None
  def residual(a):
   uv,z=project(a,points);d=map_coordinates(distance,uv[:,::-1].T,order=1,mode='constant',cval=60.,prefilter=False)
   return np.r_[np.minimum(d,40.)/379.917,a[0]*.04,a[1]*.04,a[2]*.25]
  def jac(a):
   cols=[]
   for j,h in enumerate([.003,.003,.002]):
    x=a.copy();y=a.copy();x[j]+=h;y[j]-=h;cols.append((residual(x)-residual(y))/(2*h))
   return np.array(cols).T
  bounds=np.array([.55,.55,.14]);fits=[]
  for dy in [0.,-.25,.25]:
   try:fits.append(least_squares(residual,[0.,dy,0.],jac=jac,bounds=(-bounds,bounds),loss='soft_l1',f_scale=.012,max_nfev=24))
   except (ValueError,FloatingPointError):pass
  if not fits:return None
  fit=min(fits,key=lambda f:np.mean(residual(f.x)**2));before=float(np.sqrt(np.mean(residual(np.zeros(3))[:-3]**2))*379.917);after=float(np.sqrt(np.mean(residual(fit.x)[:-3]**2))*379.917)
  j=jac(fit.x)[:-3];eig=np.linalg.eigvalsh(j.T@j);bound=bool(np.any(abs(fit.x)>bounds*.96));supported=after<4.5 and eig[0]>.002 and not bound
  self.health={'stamp':stamp,'supported':bool(supported),'before_px':before,'after_px':after,'fit_delta':fit.x.tolist(),'eigenvalues':eig.tolist(),'points':len(points),'source':'actual passive image window edges and published CAD'};self.debug=(gray,project(fit.x,points)[0],project(np.zeros(3),points)[0])
  if not supported or after>=before*.94:return None
  correction=fit.x.copy();norm=np.linalg.norm(correction[:2]);correction[:2]*=min(1.,.025/max(norm,1e-8));correction[2]=np.clip(correction[2],-.006,.006)
  self.events.append(dict(self.health,applied=correction.tolist()));return correction
