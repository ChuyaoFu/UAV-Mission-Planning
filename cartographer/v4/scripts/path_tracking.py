"""Map-independent continuous polyline tracking and collision-checked smoothing."""
import math
import numpy as np
from scipy.ndimage import distance_transform_edt,map_coordinates
from scipy.optimize import minimize
class Follower:
 def __init__(self,points):
  self.p=np.asarray(points,float);self.length=np.linalg.norm(np.diff(self.p,axis=0),axis=1);self.s=np.r_[0.,np.cumsum(self.length)];self.progress=0.
 def point(self,s):return np.array([np.interp(s,self.s,self.p[:,j]) for j in range(3)])
 def project(self,position):
  # Bound arclength advance: a nearby later parallel leg cannot teleport progress.
  candidates=[]
  for i,length in enumerate(self.length):
   if length<1e-6 or self.s[i]>self.progress+.6 or self.s[i+1]<self.progress-.10:continue
   d=self.p[i+1]-self.p[i];u=np.clip(np.dot(position-self.p[i],d)/(length*length),0,1);s=self.s[i]+u*length
   if s<=self.progress+.6:candidates.append((np.linalg.norm(position-self.point(s)),s))
  if candidates:self.progress=max(self.progress,min(candidates)[1])
  return self.progress
 def target(self,position,speed=.7,lookahead=.30):
  s=self.project(np.asarray(position));a=self.point(s);b=self.point(min(self.s[-1],s+lookahead));delta=b-position
  # Curvature speed limit uses planned turning, independent of vehicle model.
  v1=self.point(min(self.s[-1],s+.12))-a;v2=self.point(min(self.s[-1],s+.42))-self.point(min(self.s[-1],s+.30));n1=np.linalg.norm(v1);n2=np.linalg.norm(v2)
  angle=math.acos(np.clip(np.dot(v1,v2)/max(n1*n2,1e-9),-1,1)) if min(n1,n2)>.01 else 0.;curvature=angle/.30;limit=min(speed,math.sqrt(.55/max(curvature,.01)))
  remaining=self.s[-1]-s;limit=min(limit,max(.05,1.0*np.linalg.norm(self.p[-1]-position)))
  velocity=delta*min(3.,limit/max(np.linalg.norm(delta),1e-6))
  return b,velocity,{'progress_m':float(s),'remaining_m':float(remaining),'curvature':float(curvature),'planned_limit_mps':float(limit)}

def smooth(points,grid):
 points=np.asarray(points,float)
 if len(points)<3 or np.ptp(points[:,2])>.20:return points
 length=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))];t=np.linspace(0,length[-1],max(3,int(length[-1]/.12)+1));raw=np.array([np.interp(t,length,points[:,j]) for j in range(3)]).T
 low=min(grid.idx(p)[2] for p in raw);high=max(grid.idx(p)[2] for p in raw);layer=np.any(grid.occupied[:,:,low:high+1],axis=2);dist=(distance_transform_edt(~layer)-distance_transform_edt(layer))*grid.res;gx,gy=np.gradient(dist,grid.res)
 def objective(v):
  p=raw[:,:2].copy();p[1:-1]=v.reshape(-1,2);delta=np.diff(p,axis=0);size=np.sqrt((delta*delta).sum(1)+1e-5);second=np.diff(p,n=2,axis=0);coords=((p-grid.origin[:2])/grid.res).T;d=map_coordinates(dist,coords,order=1,mode='nearest');deficit=np.maximum(0,.07-d);g=np.zeros_like(p);unit=delta/size[:,None];g[:-1]-=unit;g[1:]+=unit
  bend=4.*second;g[:-2]+=bend;g[1:-1]-=2*bend;g[2:]+=bend
  gradient=np.stack([map_coordinates(gx,coords,order=1,mode='nearest'),map_coordinates(gy,coords,order=1,mode='nearest')],axis=1);g-=200.*deficit[:,None]*gradient;g+=.2*(p-raw[:,:2])
  return float(size.sum()+2.*(second*second).sum()+100.*(deficit*deficit).sum()+.1*((p-raw[:,:2])**2).sum()),g[1:-1].ravel()
 result=minimize(objective,raw[1:-1,:2].ravel(),jac=True,method='L-BFGS-B',bounds=[(float(x)-.22,float(x)+.22) for x in raw[1:-1,:2].ravel()],options={'maxiter':120,'ftol':1e-8});candidate=raw.copy();candidate[1:-1,:2]=result.x.reshape(-1,2)
 if all(grid.segment(a,b) for a,b in zip(candidate,candidate[1:])):return candidate
 return points
