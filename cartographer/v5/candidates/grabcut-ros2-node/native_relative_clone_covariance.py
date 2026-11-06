"""Relative camera motion from OpenVINS' actual joint clone posterior.

Keeps the full cross-covariance. Never subtracts a guessed gauge variance or
changes estimator priors. Calibration/noise-model validity remains external.
The native quaternion is JPL G-to-I; its numeric xyzw is Hamilton I-to-world.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


def rigid(T):
    T=np.asarray(T,float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1],atol=1e-10):
        raise ValueError('Finite rigid camera-to-IMU transform required')
    if not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-8) or abs(np.linalg.det(T[:3,:3])-1)>1e-8:
        raise ValueError('Camera-to-IMU rotation is not SO(3)')
    return T


@dataclass(frozen=True)
class ClonePair:
    camera_time0_s: float
    camera_time1_s: float
    position0: np.ndarray
    quaternion0: np.ndarray
    position1: np.ndarray
    quaternion1: np.ndarray
    joint_covariance: np.ndarray
    camera_to_imu_time_offset_s: float

    def camera_motion(self,imu_from_camera):
        T=rigid(imu_from_camera)
        R0=Rotation.from_quat(self.quaternion0).as_matrix();R1=Rotation.from_quat(self.quaternion1).as_matrix()
        def motion(A,B):
            C0=A@T[:3,:3];C1=B@T[:3,:3]
            t=C0.T@(self.position1-self.position0+B@T[:3,3]-A@T[:3,3])
            return C0.T@C1,t
        R,t=motion(R0,R1)
        J=np.zeros((6,12));world_to_camera0=(R0@T[:3,:3]).T
        J[:3,3:6]=-world_to_camera0;J[:3,9:12]=world_to_camera0
        # PoseJPL's local angle perturbation is right-multiplicative in R_WI.
        # Differentiate only the rotation columns, including camera lever arms.
        epsilon=1e-6
        for camera,column in [(0,0),(1,6)]:
            for axis in range(3):
                v=np.zeros(3);v[axis]=epsilon
                delta=Rotation.from_rotvec(v).as_matrix()
                plus=motion(R0@delta,R1) if camera==0 else motion(R0,R1@delta)
                minus=motion(R0@delta.T,R1) if camera==0 else motion(R0,R1@delta.T)
                J[:3,column+axis]=(plus[1]-minus[1])/(2*epsilon)
                J[3:,column+axis]=(Rotation.from_matrix(R.T@plus[0]).as_rotvec()-Rotation.from_matrix(R.T@minus[0]).as_rotvec())/(2*epsilon)
        P=J@self.joint_covariance@J.T;P=(P+P.T)*.5
        if not np.isfinite(P).all() or np.linalg.eigvalsh(P).min()<-1e-10:
            raise ValueError('Invalid propagated joint relative covariance')
        output=np.eye(4);output[:3,:3]=R;output[:3,3]=t
        return output,P


def decode(data,expected_camera_to_imu_time_offset_s=0.):
    """Schema1: header3, then <=10 records of160 original double values.

    Filter times originate in native double camera seconds. Consumers must
    retain/check their residual to original image headers; these are not newly
    measured integer exposure timestamps. No missing covariance is fabricated.
    """
    a=np.asarray(data,float)
    if a.ndim!=1 or len(a)<3 or not np.isfinite(a).all() or a[0]!=1 or a[1]!=int(a[1]) or not 1<=a[1]<=10:
        raise ValueError('Malformed bounded joint-clone schema')
    n=int(a[1])
    if len(a)!=3+160*n or abs(a[2]-expected_camera_to_imu_time_offset_s)>1e-9:
        raise ValueError('Joint-clone length or calibrated time offset mismatch')
    result=[];latest=None;previous=-np.inf
    for row in a[3:].reshape(n,160):
        t0,t1=row[:2];p0=row[2:5];q0=row[5:9];p1=row[9:12];q1=row[12:16];P=row[16:].reshape(12,12)
        if not t0<t1 or t0<=previous or (latest is not None and t1!=latest):
            raise ValueError('Nonordered or mixed clone epochs')
        if max(abs(np.linalg.norm(q0)-1),abs(np.linalg.norm(q1)-1))>1e-6:
            raise ValueError('Nonunit actual clone quaternion')
        if not np.allclose(P,P.T,atol=1e-10,rtol=1e-7) or np.linalg.eigvalsh((P+P.T)*.5).min()<-1e-10:
            raise ValueError('Joint covariance is not symmetric positive semidefinite')
        result.append(ClonePair(t0,t1,p0.copy(),q0.copy(),p1.copy(),q1.copy(),P.copy(),float(a[2])))
        previous=t0;latest=t1
    return result
