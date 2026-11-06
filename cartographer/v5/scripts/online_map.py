"""Bounded online-map queries and frontier viewpoints, no prior map or goals.
State -1=unobserved, 0=measured free, 1=occupied, -2=declared launch envelope.
"""
import math
from collections import deque
import numpy as np
from scipy.ndimage import minimum_filter,label,uniform_filter,distance_transform_edt
class OnlineMap:
 def __init__(self,metadata,values,radius=.32,z_half=.16):
  self.origin=np.array(metadata[:3]);self.res=float(metadata[3]);self.shape=tuple(int(x) for x in metadata[4:7]);self.stamp=float(metadata[8]);self.state=np.asarray(values,dtype=np.int8).reshape(self.shape).copy()
  self.free=(self.state==0)|(self.state==-2)
  self.extent=np.array([radius,radius,z_half]);size=tuple(2*math.floor(d/self.res+.5)+1 for d in self.extent);self.safe=minimum_filter(self.free.astype(np.uint8),size=size,mode='constant',cval=0).astype(bool)
  self.components,self.n_components=label(self.safe)
  self.blocked=np.pad((~self.free).astype(np.int32),((1,0),(1,0),(1,0))).cumsum(0).cumsum(1).cumsum(2)
 def idx(self,p):return np.floor((np.asarray(p)-self.origin)/self.res).astype(int)
 def xyz(self,i):return self.origin+(np.asarray(i)+.5)*self.res
 def inside(self,i):return bool(np.all(i>=0)&np.all(i<self.shape))
 def traversable(self,p):return self.segment(p,p)
 def segment(self,a,b,extra_margin=0.):
  points=np.linspace(a,b,max(2,int(np.linalg.norm(np.asarray(a)-b)/(self.res*.35))+1));lo=self.idx(points-self.extent-extra_margin);hi=self.idx(points+self.extent+extra_margin)+1
  if np.any(lo<0) or np.any(hi>self.shape):return False
  x,y,z=lo.T;X,Y,Z=hi.T;v=self.blocked;total=v[X,Y,Z]-v[x,Y,Z]-v[X,y,Z]-v[X,Y,z]+v[x,y,Z]+v[x,Y,z]+v[X,y,z]-v[x,y,z]
  return bool((total==0).all())
 def component(self,p):
  i=self.idx(p)
  return int(self.components[tuple(i)]) if self.inside(i) else 0
 def frontier(self,position,home,visited,yaw,max_heading_change=.5,cruise_height=.55,allow_altitude_change=False,viewpoint_margin=0.):
  component=self.component(position)
  if component==0:return None
  information=uniform_filter((self.state==-1).astype(np.float32),size=(15,15,7),mode='constant',cval=1.)
  clearance=distance_transform_edt(self.safe)*self.res
  candidates=np.argwhere((self.components==component)&self.safe&(information>.025));p=self.xyz(candidates)
  distance=np.linalg.norm(p-position,axis=1);height=p[:,2]-home[2]
  # Horizontal exploration uses a cruise band, independent of field geometry.
  # A blind ceiling is not a reason to climb; task-specific altitude changes
  # require a separate observed-free active-perception maneuver.
  height_ok=(height>.42)&(height<.85) if allow_altitude_change else (np.abs(height-cruise_height)<.075)
  good=(distance>.15)&(distance<1.4)&height_ok
  p=p[good];candidates=candidates[good];distance=distance[good]
  if not len(p):return None
  novelty=np.full(len(p),2.)
  for v in visited:novelty=np.minimum(novelty,np.linalg.norm(p-np.asarray(v),axis=1) if allow_altitude_change else np.linalg.norm(p[:,:2]-np.asarray(v)[:2],axis=1))
  angular=np.arctan2(p[:,1]-position[1],p[:,0]-position[0])-yaw;angular=np.abs(np.arctan2(np.sin(angular),np.cos(angular)))
  # Exploration goals depend only on measured-map frontiers, travel and history.
  score=np.minimum(clearance[tuple(candidates.T)],.3)*.15+information[tuple(candidates.T)]*3.+np.minimum(novelty,1.)*.8+np.minimum(np.linalg.norm(p[:,:2]-home[:2],axis=1),2.)*.15-distance*.10-angular*.04-2.*np.abs(p[:,2]-home[2]-cruise_height)
  score[(novelty<.12)|((angular>max_heading_change)&(np.linalg.norm(p[:,:2]-position[:2],axis=1)>.08))]=-np.inf
  if not np.isfinite(score).any():return None
  i=int(np.argmax(score));frontier=p[i]
  # A connected free component need not be convex. Follow its online grid path
  # with a short receding horizon, instead of asking the optimizer to jump
  # directly across an unknown concavity or beyond the observed corridor.
  start=tuple(self.idx(position));target=tuple(candidates[i]);parents={start:None};queue=deque([start])
  while queue and target not in parents:
   current=queue.popleft()
   for axis in range(3):
    for direction in [-1,1]:
     neighbor=list(current);neighbor[axis]+=direction;neighbor=tuple(neighbor)
     if neighbor not in parents and self.inside(np.array(neighbor)) and self.components[neighbor]==component:
      parents[neighbor]=current;queue.append(neighbor)
  if target not in parents:return None
  path=[];current=target
  while current!=start:path.append(self.xyz(current));current=parents[current]
  delta=frontier-position;direct=position+delta*min(1.,.4/max(np.linalg.norm(delta),1e-9))
  goal=direct if self.segment(position,direct,extra_margin=viewpoint_margin) else None
  for waypoint in ([] if goal is not None else reversed(path)):
   if np.linalg.norm(waypoint-position)>.45:break
   if not self.segment(position,waypoint,extra_margin=viewpoint_margin):break
   delta=waypoint-position
   if np.linalg.norm(delta[:2])>.08 and abs(math.atan2(math.sin(math.atan2(delta[1],delta[0])-yaw),math.cos(math.atan2(delta[1],delta[0])-yaw)))>max_heading_change:break
   goal=waypoint
  if goal is None:return None
  return goal,{'score':float(score[i]),'unknown_fraction_near_viewpoint':float(information[tuple(candidates[i])]),'frontier_position':frontier.tolist(),'online_grid_path_cells':len(path),'observed_free_altitude_maneuver':allow_altitude_change,'source':'short observed-free grid path to an online frontier; no preset route'}
