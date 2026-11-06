#!/usr/bin/python3
"""Observed official red floor guideline -> CAD floor geometry registration.
No external marker, no truth; passive RGB + known camera intrinsics + ToF height.
"""
from robot_geometry import IMU_IN_BODY,IMU_FROM_TOF
import json
from pathlib import Path
from camera_geometry import load_rig
import cv2,numpy as np
from scipy.ndimage import distance_transform_edt,map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
class LineLocalization:
 def __init__(self):
  self.res=.005;self.origin=np.array([-3.10,-3.10]);self.mask=np.zeros((1241,1241),np.uint8);vertices=[];faces=[];group=''
  for line in Path('/work/course_source/dronehacks-course-v3.obj').read_text().splitlines():
   a=line.split()
   if not a:continue
   if a[0]=='g':group=line[2:].strip()
   elif a[0]=='v':vertices.append(list(map(float,a[1:4])))
   elif a[0]=='f' and group=='Guideline (zone 1)':faces.append([int(s.split('/')[0])-1 for s in a[1:]])
  v=np.array(vertices)
  for face in faces:
   q=np.rint((v[face,:2]-self.origin)/self.res).astype(np.int32);cv2.fillConvexPoly(self.mask,q,255)
  self.distance=distance_transform_edt(self.mask==0)*self.res;self.last=-1;self.health=None;self.events=[]
  self.K,self.T,self.baseline,self.resolution=load_rig();self.rotation=self.T[:3,:3]

 def sample(self,p):return map_coordinates(self.distance,((p-self.origin)/self.res)[:,::-1].T,order=1,mode='constant',cval=1.,prefilter=False)
 def update(self,m,body,rotation):
  stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
  if stamp-self.last<.09:return None
  self.last=stamp
  if m.encoding not in ['rgb8','bgr8']:return None
  image=np.ndarray((m.height,m.width,3),np.uint8,buffer=m.data,strides=(m.step,3,1));image=image if m.encoding=='rgb8' else image[:,:,::-1]
  hsv=cv2.cvtColor(np.ascontiguousarray(image),cv2.COLOR_RGB2HSV);mask=((hsv[:,:,0]<8)|(hsv[:,:,0]>175))&(hsv[:,:,1]>70)&(hsv[:,:,2]>35)
  y,x=np.nonzero(mask)
  if len(x)<80:return None
  ids=np.arange(0,len(x),max(1,len(x)//500));x=x[ids];y=y[ids]
  fx,fy,cx,cy=self.K;rays=np.stack([(x-cx)/fx,(y-cy)/fy,np.ones(len(x))],axis=1)@self.rotation.T;rays=rotation.apply(rays)
  camera=body+rotation.apply(IMU_IN_BODY+self.T[:3,3]);valid=rays[:,2]<-.12;rays=rays[valid]
  if len(rays)<60:return None
  scale=(.00127-camera[2])/rays[:,2];points=camera+scale[:,None]*rays;points=points[(scale>0)&(scale<4.),:2]
  if len(points)<60 or np.linalg.norm(np.ptp(points,axis=0))<.25:return None
  pivot=body[:2]
  def moved(a):
   c,s=np.cos(a[2]),np.sin(a[2]);rot=np.array([[c,-s],[s,c]]);return (points-pivot)@rot.T+pivot+a[:2]
  def residual(a):return np.r_[self.sample(moved(a)),a[0]*.12,a[1]*.12,a[2]*.5]
  def jac(a):
   cols=[]
   for j,h in enumerate([.003,.003,.002]):
    x=a.copy();y=a.copy();x[j]+=h;y[j]-=h;cols.append((residual(x)-residual(y))/(2*h))
   return np.array(cols).T
  try:fit=least_squares(residual,[0.,0.,0.],jac=jac,bounds=([-.22,-.22,-.10],[.22,.22,.10]),loss='soft_l1',f_scale=.015,max_nfev=18)
  except (ValueError,FloatingPointError):return None
  before=float(np.sqrt(np.mean(self.sample(points)**2)));after=float(np.sqrt(np.mean(self.sample(moved(fit.x))**2)))
  j=jac(fit.x)[:-3];information=j.T@j;eig=np.linalg.eigvalsh(information)
  lateral=after<.025 and eig[-2]>1.;supported=lateral and eig[0]>.5
  self.health={'stamp':stamp,'supported':bool(supported),'lateral_supported':bool(lateral),'fit_rms_m':after,'information_eigenvalues':eig.tolist(),'points':len(points),'source':'observed official red floor line matched to published CAD; passive RGB + ToF, no truth'}
  if not lateral or after>=before*.97:return None
  correction=fit.x.copy();norm=np.linalg.norm(correction[:2]);correction[:2]*=min(1.,.010/max(norm,1e-8));correction[2]=np.clip(correction[2],-.004,.004)
  self.events.append({'stamp':stamp,'before':before,'after':after,'delta':correction.tolist(),'full_xy_yaw_supported':bool(supported)})
  return correction
