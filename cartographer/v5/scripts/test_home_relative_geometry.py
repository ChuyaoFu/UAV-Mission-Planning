"""Coordinate-chart, reference-point and stale-source regressions."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from home_relative_geometry import body_pose_to_native_imu,align_native_twist,skew,native_anchor_ns,validate_reference,ReferenceChanged


class GeometryTests(unittest.TestCase):
    def test_full_lever_and_covariance_roundtrip_to_existing_body_contract(self):
        rng=np.random.default_rng(18)
        for _ in range(30):
            r=Rotation.random(random_state=rng);mount=Rotation.from_euler('xyz',[.2,.35,-.1]);lever=np.array([.22,.01,.055])
            a=rng.normal(size=(6,6))*.001;p=a@a.T;position=rng.normal(size=3)
            pi,qi,pc=body_pose_to_native_imu(position,r.as_quat(),p,lever,mount)
            ri=Rotation.from_quat(qi)
            np.testing.assert_allclose(pi-r.apply(lever),position,atol=1e-14)
            j=np.eye(6);j[:3,3:]=ri.as_matrix()@skew(mount.inv().apply(lever));j[3:,3:]=mount.as_matrix()
            body_local=j@pc@j.T
            chart=np.eye(6);chart[3:,3:]=r.as_matrix().T
            np.testing.assert_allclose(body_local,chart@p@chart.T,atol=1e-14)

    def test_real_lever_uncertainty_is_not_erased(self):
        p=np.zeros((6,6));p[5,5]=.01
        _,_,out=body_pose_to_native_imu([0,0,0],[0,0,0,1],p,[.22,0,0],Rotation.identity())
        self.assertAlmostEqual(out[1,1],.22**2*.01)
        self.assertAlmostEqual(out[1,5],.22*.01)

    def test_native_twist_rotation_and_declared_age_increase_error(self):
        p=np.eye(6)*.0001;pose=np.eye(6)*.00001
        v,w,c=align_native_twist([1,0,0],[0,0,.2],p,Rotation.from_euler('z',90,degrees=True),Rotation.identity(),.01,pose,pose)
        np.testing.assert_allclose(v,[0,1,0],atol=1e-12)
        np.testing.assert_allclose(w,[0,0,.2],atol=1e-12)
        self.assertTrue((np.diag(c)>np.diag(p)).all())
        for age in [-.001,.020001]:
            with self.assertRaises(ValueError):align_native_twist([0,0,0],[0,0,0],p,Rotation.identity(),Rotation.identity(),age,pose,pose)

    def test_missing_or_non_psd_covariance_refuses(self):
        for p in [np.eye(6)*np.nan,np.diag([-1.,1,1,1,1,1])]:
            with self.assertRaises(ValueError):body_pose_to_native_imu([0,0,0],[0,0,0,1],p,[.22,0,.055],Rotation.identity())

    def test_positive_correlated_alignment_error_is_enclosed(self):
        native=np.eye(6)*1e-8;native[1,1]=.1**2
        attitude=np.zeros((6,6));attitude[5,5]=.0001**2
        _,_,out=align_native_twist([1,0,0],[0,0,0],native,Rotation.identity(),Rotation.identity(),0.,2*attitude,attitude)
        # Compatible joint model: e_v=.1*z, theta_source=.0001*(u+z),
        # theta_current=.0001*u. Its positive cross term must be retained.
        self.assertGreaterEqual(out[1,1],(.1+.0001)**2)

    def test_native_anchor_rounding_matches_half_away_from_zero(self):
        self.assertEqual(native_anchor_ns(1.0000000005),1000000001)
        self.assertEqual(native_anchor_ns(-1.0000000005),-1000000001)

    def test_clock_reference_locks_camera_epoch_and_offset_separately(self):
        mount=Rotation.from_euler('y',20,degrees=True);lever=np.array([.22,0,.055]);inverse=mount.inv()
        def packet(camera,offset):return np.r_[1.,camera,camera+offset,2.,2.+offset,-inverse.apply(lever),inverse.as_quat()]
        with self.assertRaises(ValueError):validate_reference(packet(1.,.125),lever,mount)
        old,epoch=validate_reference(packet(1.,.125),lever,mount,fast_imu_epoch_verified=True)
        self.assertEqual(epoch,1125000000)
        with self.assertRaises(ReferenceChanged):validate_reference(packet(.75,.375),lever,mount,old,True)

if __name__=='__main__':unittest.main()
