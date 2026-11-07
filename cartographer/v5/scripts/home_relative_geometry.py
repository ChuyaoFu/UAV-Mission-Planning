"""Calibrated BODY pose to the existing native-IMU odometry chart.

Pure geometry only. The input pose must come from the actual joint posterior;
this module cannot create a home reference or reduce an estimator covariance.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation


def skew(v):
    x,y,z=np.asarray(v,float)
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def covariance(value):
    p=np.asarray(value,float).reshape(6,6)
    if not np.isfinite(p).all() or not np.allclose(p,p.T,atol=1e-10,rtol=1e-7):
        raise ValueError('Finite symmetric covariance required')
    if np.linalg.eigvalsh((p+p.T)*.5).min() < -64*np.finfo(float).eps*max(1.,np.linalg.norm(p,2)):
        raise ValueError('Positive semidefinite covariance required')
    return p


def body_pose_to_native_imu(position,quaternion,body_covariance,imu_in_body,body_from_imu):
    """Input [p_H, LEFT theta_H]; output [p_H, RIGHT theta_I].

    The IMU lever is moved with its full position/angle cross-covariance.
    Downstream's existing IMU-to-BODY transform therefore removes it once.
    Calibration uncertainty is not claimed to be included in this posterior.
    """
    position=np.asarray(position,float);lever=np.asarray(imu_in_body,float)
    q=np.asarray(quaternion,float)
    if position.shape!=(3,) or lever.shape!=(3,) or q.shape!=(4,) or not np.isfinite(np.r_[position,lever,q]).all() or abs(np.linalg.norm(q)-1)>1e-6:
        raise ValueError('Finite calibrated pose required')
    r=Rotation.from_quat(q);ri=r*body_from_imu
    displacement=r.apply(lever)
    j=np.eye(6);j[:3,3:]=-skew(displacement);j[3:,3:]=ri.as_matrix().T
    p=j@covariance(body_covariance)@j.T
    # Two opposite lever transformations cancel position uncertainty at the
    # actual reference epoch. Finite arithmetic can leave ~1e-18 negatives.
    # Only ADD a conservative roundoff allowance; never clip eigenvalues or
    # reduce any native covariance direction. Larger indefiniteness rejects.
    padding=64*np.finfo(float).eps*max(1.,np.linalg.norm(p,2))
    return position+displacement,ri.as_quat(),(p+p.T)*.5+padding*np.eye(6)


def align_native_twist(linear,angular,twist_covariance,source_rotation,current_rotation,age_s,
                       source_pose_covariance,current_pose_covariance,
                       acceleration_limit_mps2=2.,angular_acceleration_limit_radps2=4.,max_age_s=.020):
    """Re-express a recent actual IMU twist, retaining source-time uncertainty.

    Constant world velocity over the bounded time gap is a declared motion
    model. Acceleration and both attitude marginal bounds are added, never
    subtracted. No pose extrapolation or guessed zero velocity is generated.
    The caller retains the original twist timestamp in its provenance report.
    """
    vectors=np.r_[linear,angular]
    if not np.isfinite(vectors).all() or len(vectors)!=6 or not np.isfinite([age_s,acceleration_limit_mps2,angular_acceleration_limit_radps2,max_age_s]).all():
        raise ValueError('Finite measured twist and motion domain required')
    if not 0<=age_s<=max_age_s or min(acceleration_limit_mps2,angular_acceleration_limit_radps2)<=0 or not 0<max_age_s<=.020:
        raise ValueError('Twist outside bounded causal motion domain')
    r=(current_rotation.inv()*source_rotation).as_matrix()
    j=np.zeros((6,6));j[:3,:3]=r;j[3:,3:]=r
    p=j@covariance(twist_covariance)@j.T
    attitude=sum(3*np.sqrt(max(0.,np.linalg.eigvalsh(covariance(c)[3:,3:]).max()))
                 for c in (source_pose_covariance,current_pose_covariance))
    linear_bound=acceleration_limit_mps2*age_s+attitude*np.linalg.norm(linear)
    angular_bound=angular_acceleration_limit_radps2*age_s+attitude*np.linalg.norm(angular)
    # Treat each bound as an extra standard deviation, conservatively.
    # Linear/angular alignment errors and the existing native error may be
    # correlated. First enclose the complete extra 6D second moment with
    # 2*diag(b_linear^2 I3, b_angular^2 I3), then use Young's PSD inequality
    # with eta=1. This encloses the sum relative to the SUPPLIED native model;
    # it cannot repair missing terms in an upstream covariance approximation.
    if linear_bound or angular_bound:
        p=2*p+4*np.diag([linear_bound**2]*3+[angular_bound**2]*3)
    return r@np.asarray(linear),r@np.asarray(angular),(p+p.T)*.5


class ReferenceChanged(ValueError):pass


def native_anchor_ns(seconds):
    """Match C++ llround(double_seconds*1e9), including half ties."""
    value=float(seconds)*1e9
    if not math.isfinite(value) or abs(value)>=2**63:raise ValueError('Native anchor outside int64 epoch')
    magnitude=abs(value);whole=math.floor(magnitude)
    rounded=whole+int(magnitude-whole>=.5)
    return -rounded if value<0 else rounded


def reference_signature(a):return np.r_[a[1:3],a[2]-a[1],a[5:12]]


def validate_reference(data,imu_in_body,body_from_imu,previous=None,fast_imu_epoch_verified=False):
    a=np.asarray(data,float)
    if a.shape!=(12,) or not np.isfinite(a).all() or a[0]!=1:raise ValueError('Unknown native reference schema')
    clock_tolerance=max(1e-9,4*max(math.ulp(float(v)) for v in a[1:5]))
    if a[1]<0 or a[2]<0 or a[3]<a[1] or a[4]<a[2] or abs((a[4]-a[3])-(a[2]-a[1]))>clock_tolerance:
        raise ValueError('Mixed native reference epochs or calibrated clock')
    expected=body_from_imu.inv();lever=-expected.apply(imu_in_body)
    if not np.allclose(a[5:8],lever,atol=1e-9,rtol=0) or abs(np.linalg.norm(a[8:12])-1)>1e-6 or (Rotation.from_quat(a[8:12])*expected.inv()).magnitude()>1e-8:
        raise ValueError('Native BODY/IMU calibration does not match actual adapter rig')
    if previous is not None and not np.array_equal(reference_signature(a),reference_signature(previous)):
        raise ReferenceChanged('Native home/clock/rig reference changed; explicit process/session restart required')
    if abs(a[2]-a[1])>1e-12 and not fast_imu_epoch_verified:
        raise ValueError('Nonzero offset requires verified native fast-odometry IMU-epoch contract')
    return a.copy(),native_anchor_ns(a[2])
