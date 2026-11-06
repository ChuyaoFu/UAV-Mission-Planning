"""Synthetic raw-depth camera pitched20deg: measured outline + planar error model."""
import unittest,math
from types import SimpleNamespace as NS
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from task_native_contact import *


def scene(holes=False,foreground=False,shift=(0,0,0)):
    w,h=640,480;K=np.array([400.,400.,320.,240.]);pitch=math.radians(20)
    forward=np.array([math.cos(pitch),0,-math.sin(pitch)]);right=np.array([0,-1,0]);down=np.cross(forward,right)
    T=np.eye(4);T[:3,:3]=np.column_stack((right,down,forward));T[:3,3]=np.array([0,0,1.])+shift
    yy,xx=np.indices((h,w));rays=np.stack(((xx-K[2])/K[0],(yy-K[3])/K[1],np.ones((h,w))),axis=-1)
    rw=rays@T[:3,:3].T;z=np.divide(-1.,rw[:,:,2],out=np.full((h,w),np.nan),where=rw[:,:,2]<-.01)
    depth=z.astype(np.float32);points=rw*z[:,:,None]+T[:3,3]
    center=np.array([2.2,0.,0.])+shift;inside=(np.linalg.norm(points[:,:,:2]-center[:2],axis=2)<.75)&np.isfinite(z)
    image=np.full((h,w),40,np.uint8);image[inside]=220
    if holes:depth[::5,::5]=np.nan
    if foreground:
        yy2,xx2=np.nonzero(inside);i=len(xx2)//2;depth[yy2[i]-3:yy2[i]+4,xx2[i]-3:xx2[i]+4]*=.8
    P=np.diag([1e-6]*3+[1e-8]*3);pose=PoseEnvelope(1000000000,np.array([0,0,1.])+shift,np.eye(3),(P,))
    exposure=NS(local_from_camera=T,stamp=1.,epoch='e',calibration_id='rig',pose_error_m=.003)
    track=NS(role='dock',consistent=True,geometry_verified=True,center=center,normal=np.array([0,0,1.]),id='pad',details={'marker_kind':'cross'})
    return image,depth,K,exposure,track,pose


class RetainedBuilder(unittest.TestCase):
    def build(self,**kwargs):
        i,d,k,e,t,p=scene(**kwargs)
        return build_retained_support(i,d,k,e,t,np.eye(4),p,.02,.005,.32)

    def test_pitched20_observed_pad_with_sparse_holes_does_not_need_every_pixel(self):
        p=self.build(holes=True);self.assertIsNotNone(p)
        self.assertGreater(p.unknown_fraction,0);self.assertGreater(p.direct_points,64)
        self.assertGreater(p.half_width_m,.5);self.assertGreater(p.error_m,.02)
        self.assertTrue(p.as_dict()['plane_model_interpolates_unmeasured_surface'])
        self.assertFalse(p.as_dict()['unmeasured_voxels_written_free'])

    def test_real_finite_foreground_is_not_absorbed_as_support(self):
        self.assertIsNone(self.build(foreground=True))

    def test_deterministic_total_error_not_residual_accuracy_or_sqrt_n(self):
        p=self.build();self.assertIsNotNone(p)
        self.assertLess(p.measured_residual_m,1e-5)
        self.assertGreater(p.error_m,.02);self.assertGreater(p.normal_fit_error_rad,0.)

    def test_coordinate_origin_translation_does_not_change_support_precision(self):
        a=self.build();b=self.build(shift=np.array([1000,-2000,3000]))
        self.assertIsNotNone(b)
        self.assertAlmostEqual(a.error_m,b.error_m,places=7)
        np.testing.assert_allclose(np.asarray(b.center_local_m)-np.asarray(a.center_local_m),[1000,-2000,3000],atol=1e-5)

if __name__=='__main__':unittest.main()
