#pragma once
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <stdexcept>
namespace ov_msckf {
struct HomeRelativePose {
 Eigen::Matrix3d rotation;
 Eigen::Vector3d position;
 Eigen::Matrix<double,6,6> covariance;
 Eigen::Matrix<double,6,12> jacobian;
};
inline HomeRelativePose home_relative_body_pose(const Eigen::Matrix3d&A,const Eigen::Vector3d&p0,
 const Eigen::Matrix3d&B,const Eigen::Vector3d&p1,const Eigen::Matrix3d&C,const Eigen::Vector3d&lever,
 const Eigen::Matrix<double,12,12>&joint) {
 auto skew=[](const Eigen::Vector3d&x){Eigen::Matrix3d S;S<<0.,-x.z(),x.y(),x.z(),0.,-x.x(),-x.y(),x.x(),0.;return S;};
 if(!A.allFinite()||!B.allFinite()||!C.allFinite()||!p0.allFinite()||!p1.allFinite()||!lever.allFinite()||!joint.allFinite())throw std::runtime_error("home_relative: nonfinite actual posterior/calibration");
 if ((joint-joint.transpose()).cwiseAbs().maxCoeff()>1e-9 ||
     Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,12,12>>(joint).eigenvalues().minCoeff() < -1e-10)
   throw std::runtime_error("home_relative: invalid symmetric/PSD joint telemetry");
 const Eigen::Matrix3d Q=A.transpose()*B,W=C.transpose()*A.transpose();
 HomeRelativePose out;out.rotation=C.transpose()*Q*C;
 out.position=C.transpose()*(A.transpose()*(p1-p0)+Q*lever-lever);
 out.jacobian.setZero();
 out.jacobian.block<3,3>(0,0)=C.transpose()*skew(A.transpose()*(p1-p0)+Q*lever);
 out.jacobian.block<3,3>(0,3)=-W;
 out.jacobian.block<3,3>(0,6)=-C.transpose()*Q*skew(lever);
 out.jacobian.block<3,3>(0,9)=W;
 // Pose covariance: BOTH translation and LEFT small-angle error in HOME BODY.
 // This is diag(I,R_relative) times the mixed C0/C1 right-tangent chart.
 out.jacobian.block<3,3>(3,0)=-C.transpose();
 out.jacobian.block<3,3>(3,6)=C.transpose()*Q;
 out.covariance=out.jacobian*joint*out.jacobian.transpose();
 out.covariance=.5*(out.covariance+out.covariance.transpose()).eval();
 if(!out.position.allFinite()||!out.rotation.allFinite()||!out.covariance.allFinite())throw std::runtime_error("home_relative: invalid propagated telemetry");
 if (Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>>(out.covariance).eigenvalues().minCoeff() < -1e-10)
   throw std::runtime_error("home_relative: invalid PSD output telemetry");
 // No clipping, diagonal-only approximation, prior subtraction or P mutation.
 return out;
}
}
