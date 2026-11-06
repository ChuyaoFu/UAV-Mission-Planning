"""Timestamped sensor-frame geometry. No course/world pose or map inputs."""
from collections import deque
from bisect import bisect_left
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
class PoseHistory:
 def __init__(self,maxlen=300):self.items=deque(maxlen=maxlen)
 def append(self,t,position,quaternion):
  if self.items and t<=self.items[-1][0]:return
  self.items.append((float(t),np.asarray(position,float),np.asarray(quaternion,float)))
 def sample(self,t):
  if not self.items or t<self.items[0][0] or t>self.items[-1][0]:return None
  h=list(self.items);times=[v[0] for v in h];i=bisect_left(times,t)
  if i==0:return h[0][1].copy(),Rotation.from_quat(h[0][2])
  a,b=h[i-1],h[i];u=(t-a[0])/(b[0]-a[0]);p=a[1]+u*(b[1]-a[1]);r=Slerp([a[0],b[0]],Rotation.from_quat([a[2],b[2]]))([t])[0]
  return p,r

def interpolate_rotation(times,quaternions,t):
 """Exact Slerp uses only the two exposure-bracketing samples."""
 i=bisect_left(times,t)
 if i==0:return Rotation.from_quat(quaternions[0])
 if i>=len(times):return Rotation.from_quat(quaternions[-1])
 return Slerp([times[i-1],times[i]],Rotation.from_quat([quaternions[i-1],quaternions[i]]))([t])[0]
