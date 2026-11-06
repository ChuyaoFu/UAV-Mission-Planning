"""Known robot self-occlusion only; no scene geometry or absolute pose."""
import os
import numpy as np,cv2
from scipy.spatial.transform import Rotation
from robot_geometry import ROBOT,IMU_IN_BODY,BODY_FROM_IMU
from camera_geometry import load_rig

def self_masks():
 K,T,b,res=load_rig();
 if os.environ.get('HARDWARE_MODE')=='1' and 'self_filter' not in ROBOT:raise ValueError('Hardware requires measured self-occlusion masks or body/prop geometry in ROBOT_CALIBRATION')
 v,u=np.mgrid[:res[1],:res[0]];optical=np.stack([(u-K[2])/K[0],(v-K[3])/K[1],np.ones_like(u)],-1).reshape(-1,3);masks=[];shape=ROBOT.get('self_filter',{})
 for side in range(2):
  t=T[:3,3]+(T[:3,0]*b if side else 0);o=IMU_IN_BODY+BODY_FROM_IMU.apply(t);r=BODY_FROM_IMU*Rotation.from_matrix(T[:3,:3]);d=r.apply(optical);blocked=np.zeros(len(d),bool)
  for center in shape.get('rotor_centers_body_m',[]):
   c=np.asarray(center);offset=o[:2]-c[:2];A=np.sum(d[:,:2]**2,1);B=2*(d[:,:2]@offset);C=np.dot(offset,offset)-shape['rotor_swept_radius_m']**2;disc=B*B-4*A*C
   with np.errstate(divide='ignore',invalid='ignore'):
    lo=(-B-np.sqrt(np.maximum(disc,0)))/(2*A);hi=(-B+np.sqrt(np.maximum(disc,0)))/(2*A);za=(c[2]-shape['rotor_swept_half_height_m']-o[2])/d[:,2];zb=(c[2]+shape['rotor_swept_half_height_m']-o[2])/d[:,2]
   blocked|=(disc>=0)&(np.minimum(hi,np.maximum(za,zb))>=np.maximum(np.maximum(lo,np.minimum(za,zb)),.001))
  for low,high in shape.get('body_boxes_m',[]):
   with np.errstate(divide='ignore',invalid='ignore'):
    a=(np.asarray(low)-o)/d;bnd=(np.asarray(high)-o)/d
   blocked|=np.maximum(np.minimum(a,bnd).max(1),.001)<=np.maximum(a,bnd).min(1)
  mask=cv2.dilate(blocked.reshape(res[1],res[0]).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool);masks.append(mask)
 return tuple(masks)
if __name__=='__main__':
 from pathlib import Path
 root=Path(__file__).resolve().parents[1];m=self_masks()
 for i,mask in enumerate(m):cv2.imwrite(str(root/'config/gazebo_vio'/('self-mask%d.png'%i)),mask.astype(np.uint8)*255)
 print('self-occlusion fractions (white is excluded)',[float(a.mean()) for a in m])
