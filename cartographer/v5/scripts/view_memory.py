"""Bounded memory of measured useful camera headings; no field/map prior."""
from collections import deque
import math
import numpy as np
def angle(a):return math.atan2(math.sin(a),math.cos(a))
class ViewMemory:
 def __init__(self,maxlen=256,max_age=25.,max_distance=.8,height_tolerance=.18):self.views=deque(maxlen=maxlen);self.max_age=max_age;self.max_distance=max_distance;self.height_tolerance=height_tolerance
 def add(self,stamp,position,yaw,features,update_age):
  if features<6 or update_age>.2:return
  if self.views:
   t,p,a,n=self.views[-1]
   if stamp-t<.25 and np.linalg.norm(position-p)<.06 and abs(angle(yaw-a))<.08:return
  self.views.append((float(stamp),np.asarray(position).copy(),float(yaw),int(features)))
 def choose(self,stamp,position,desired,current):
  candidates=[]
  for t,p,yaw,n in self.views:
   distance=np.linalg.norm(position-p);age=stamp-t
   if age>self.max_age or distance>self.max_distance or abs(float(position[2]-p[2]))>self.height_tolerance:continue
   score=math.log1p(n)-1.5*distance-.015*age-.15*abs(angle(yaw-desired))-.08*abs(angle(yaw-current))
   candidates.append((score,yaw))
  return max(candidates)[1] if candidates else desired
