"""Floor-relative height from VIO increments and a single downward ToF.
Each range is applied once at its exposure time. Large surface transitions are
rejected; no terrain map, GPS, CAD or truth. XY and attitude remain raw VIO.
"""
from frames import interpolate_rotation
from robot_geometry import IMU_IN_BODY,IMU_FROM_TOF,body_rotation
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
class HeightObserver:
 def __init__(self):
  self.history=deque(maxlen=300);self.pending=None;self.last=-1.;self.anchor=None;self.correction=0.;self.rejected=0;self.accepted=0;self.ranges=deque(maxlen=100);self.floor_support=-1.;self.reacquired=0;self.large_innovations=deque(maxlen=12)
  self.frozen=False
 def freeze(self):
  # A raised support surface must never redefine the original global floor.
  # Preserve the current gauge through platform landing, disarm and liftoff.
  self.frozen=True;self.pending=None;self.large_innovations.clear()
 def ranging(self,m):
  t=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
  if np.isfinite(m.range) and m.min_range<=m.range<=m.max_range:self.pending=(t,float(m.range));self.ranges.append(self.pending)
 def odometry(self,m):
  t=m.header.stamp.sec+m.header.stamp.nanosec*1e-9;p=m.pose.pose.position;q=m.pose.pose.orientation
  if self.history and t<=self.history[-1][0]:return self.correction
  self.history.append((t,p.z,[q.x,q.y,q.z,q.w]))
  if self.frozen:return self.correction
  if self.pending is None:return self.correction
  stamp,distance=self.pending
  if stamp<=self.last or stamp>t or stamp<self.history[0][0]:return self.correction
  h=list(self.history);times=np.array([a[0] for a in h]);z=float(np.interp(stamp,times,[a[1] for a in h]));rotation=interpolate_rotation(times,[a[2] for a in h],stamp);cosine=body_rotation(rotation).as_matrix()[2,2]
  if cosine<.9:self.rejected+=1;self.last=stamp;return self.correction
  measured=distance*cosine+rotation.apply(IMU_FROM_TOF)[2]
  if self.anchor is None:self.anchor=(z,measured);self.last=stamp;return self.correction
  predicted=self.anchor[1]+z-self.anchor[0]+self.correction;innovation=measured-predicted;dt=min(.1,max(0.,stamp-self.last));self.last=stamp
  if abs(innovation)>.08:
   self.large_innovations.append((stamp,innovation))
   consistent=len(self.large_innovations)>=6 and self.large_innovations[-1][0]-self.large_innovations[0][0]>.15 and np.ptp([x[1] for x in self.large_innovations])<.05
   if stamp-self.floor_support>.25 or not consistent:self.rejected+=1;return self.correction
   self.reacquired+=1
  else:self.large_innovations.clear()
  self.correction+=float(np.clip(.20*innovation,-.08*dt,.08*dt));self.accepted+=1
  return self.correction

 def cloud(self,m):
  if m.point_step!=12 or not self.history or not self.ranges:return
  t=m.header.stamp.sec+m.header.stamp.nanosec*1e-9;h=list(self.history);times=np.array([a[0] for a in h])
  if t<times[0] or t>times[-1]:return
  range_time,distance=min(self.ranges,key=lambda a:abs(a[0]-t))
  if abs(range_time-t)>.08:return
  rotation=interpolate_rotation(times,[a[2] for a in h],t);cosine=body_rotation(rotation).as_matrix()[2,2]
  if cosine<.9:return
  candidate=distance*cosine+rotation.apply(IMU_FROM_TOF)[2];p=np.frombuffer(m.data,dtype='<f4').reshape(-1,3);p=rotation.apply(p);floor=p[np.isfinite(p).all(1)&(abs(p[:,2]+candidate)<.055)]
  # A wide measured floor patch distinguishes floor reacquisition from a
  # small obstacle top. No ideal plane or CAD coordinates are inserted.
  if len(floor)<80:return
  spread=np.percentile(floor[:,:2],95,axis=0)-np.percentile(floor[:,:2],5,axis=0)
  if spread.min()>.55:self.floor_support=t
