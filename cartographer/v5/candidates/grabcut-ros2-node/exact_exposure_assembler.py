"""Small nonpublishing exact-header joiner for the existing ROS evidence owner.
No sensor geometry is invented here: root supplies actual paired floor geometry.
"""
from collections import OrderedDict
import threading
import numpy as np
from stream_adapter import FLOOR_FIELDS

class ExactExposureAssembler:
 def __init__(self,adapter,K,baseline,max_partial=24):
  self.adapter=adapter;self.K=np.array(K,float);self.baseline=float(baseline);self.parts=OrderedDict();self.maximum=int(max_partial);self.lock=threading.Lock()
  if not(3<=self.maximum<=24):raise ValueError('bounded exact-header partial cache required')
 def _trim(self):
  while len(self.parts)>self.maximum:self.parts.popitem(last=False)
 def _complete(self,s):
  p=self.parts[s]
  if not all(k in p for k in ('left','right','depth','floor')):return False
  L,R,D=p['left'],p['right'],p['depth'];floor=p['floor'];frame=dict(left=L['array'],right=R['array'],measured_stereo=D['array'],K=self.K,baseline=self.baseline,stamp_ns=s,left_stamp_ns=L['stamp_ns'],right_stamp_ns=R['stamp_ns'],depth_stamp_ns=D['stamp_ns'],left_frame_id=L['frame_id'],depth_frame_id=D['frame_id'],**floor)
  accepted=self.adapter.offer_exposure(frame,original_header=D['header'],pose_health_ok=p.get('source_health'));self.parts.pop(s,None);return accepted
 def offer_image(self,kind,stamp_ns,array,frame_id,original_header):
  if kind not in ('left','right','depth'):raise ValueError('kind must be left/right/depth')
  if not self.adapter.enabled or not self.lock.acquire(False):return False
  try:
   s=int(stamp_ns);a=np.asarray(array)
   if a.flags.writeable:raise ValueError('caller must transfer immutable owned ROS image array')
   p=self.parts.setdefault(s,{});p[kind]=dict(array=a,frame_id=str(frame_id),stamp_ns=s,header=original_header);self._trim();return self._complete(s) if s in self.parts else False
  finally:self.lock.release()
 def offer_floor_geometry(self,exposure_stamp_ns,*,source_health=None,**actual_fields):
  """Target exposure key is separate from ORIGINAL ToF/gravity sample stamps.
  Actual body/camera/ToF geometry must be derived by the existing sensor owner;
  never fill from configured mounting angle, ideal height, GT or a floor pixel.
  """
  if any(k not in actual_fields for k in FLOOR_FIELDS):raise ValueError('all actual floor fields required')
  if not self.adapter.enabled or not self.lock.acquire(False):return False
  try:
   s=int(exposure_stamp_ns);p=self.parts.setdefault(s,{});p['floor']={k:actual_fields[k] for k in FLOOR_FIELDS};p['source_health']=source_health;self._trim();return self._complete(s) if s in self.parts else False
  finally:self.lock.release()
