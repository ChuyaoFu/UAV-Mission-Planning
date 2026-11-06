import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from relative_clone_covariance import ClonePair,decode


class RelativeCloneTests(unittest.TestCase):
    def pair(self,P=None):
        return ClonePair(1.,1.4,np.array([1.,2.,3.]),np.array([0.,0.,0.,1.]),np.array([1.2,2.1,3.1]),np.array([0.,0.,0.,1.]),np.eye(12)*1e-6 if P is None else P,0.)

    def test_common_translation_prior_cancels_through_real_cross_terms(self):
        base=np.zeros((12,12));base[3:6,3:6]=np.eye(3)*1e-5;base[9:12,9:12]=np.eye(3)*2e-5
        G=np.zeros((12,3));G[3:6]=np.eye(3);G[9:12]=np.eye(3)
        prior=G@np.diag([.0025,.004,.01])@G.T
        _,P=self.pair(base+prior).camera_motion(np.eye(4))
        np.testing.assert_allclose(P[:3,:3],np.eye(3)*3e-5,atol=1e-15)
        independent=base+np.diag(prior.diagonal())
        _,wrong=self.pair(independent).camera_motion(np.eye(4))
        self.assertGreater(wrong[0,0],P[0,0]*100)

    def test_global_translation_of_both_means_does_not_change_relative_motion(self):
        p=self.pair();q=ClonePair(p.camera_time0_s,p.camera_time1_s,p.position0+20,p.quaternion0,p.position1+20,p.quaternion1,p.joint_covariance,0.)
        np.testing.assert_allclose(p.camera_motion(np.eye(4))[0],q.camera_motion(np.eye(4))[0],atol=1e-13)

    def test_measured_camera_lever_is_included(self):
        p=self.pair();T=np.eye(4);T[:3,3]=[.2,.05,0]
        r=Rotation.from_euler('z',90,degrees=True).as_quat()
        p=ClonePair(1.,1.4,np.zeros(3),p.quaternion0,np.zeros(3),r,p.joint_covariance,0.)
        motion,P=p.camera_motion(T)
        np.testing.assert_allclose(motion[:3,3],[-.25,.15,0],atol=1e-12)
        self.assertGreater(P[0,0],2e-6)

    def test_joint_payload_rejects_missing_cross_covariance_and_bad_epochs(self):
        p=self.pair();row=np.r_[p.camera_time0_s,p.camera_time1_s,p.position0,p.quaternion0,p.position1,p.quaternion1,p.joint_covariance.ravel()]
        packet=np.r_[1,1,0,row]
        self.assertEqual(len(decode(packet)),1)
        for bad in [packet[:-1],np.r_[1,1,.1,row],np.r_[2,1,0,row],np.r_[1,11,0,row]]:
            with self.assertRaises(ValueError):decode(bad)
        bad=packet.copy();bad[3+16]=-.1
        with self.assertRaises(ValueError):decode(bad)


if __name__=='__main__':unittest.main()
