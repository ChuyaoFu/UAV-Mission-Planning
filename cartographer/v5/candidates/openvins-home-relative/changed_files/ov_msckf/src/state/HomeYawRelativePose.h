#pragma once
#include "HomeRelativePose.h"
namespace ov_msckf {
inline HomeRelativePose home_yaw_relative_body_pose(const Eigen::Matrix3d&A,const Eigen::Vector3d&p0,
 const Eigen::Matrix3d&B,const Eigen::Vector3d&p1,const Eigen::Matrix3d&C,const Eigen::Vector3d&lever,
 const Eigen::Matrix<double,12,12>&joint) {
 auto skew=[](const Eigen::Vector3d&x){Eigen::Matrix3d S;S<<0.,-x.z(),x.y(),x.z(),0.,-x.x(),-x.y(),x.x(),0.;return S;};
 if(!A.allFinite()||!B.allFinite()||!C.allFinite()||!p0.allFinite()||!p1.allFinite()||!lever.allFinite()||!joint.allFinite())throw std::runtime_error("home_yaw_relative: nonfinite input");
 if ((joint-joint.transpose()).cwiseAbs().maxCoeff()>1e-9 ||
     Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,12,12>>(joint).eigenvalues().minCoeff() < -1e-10)
   throw std::runtime_error("home_yaw_relative: invalid symmetric/PSD joint telemetry");
 // Home yaw is from BODY attitude, never from pitched IMU axes.
 const Eigen::Vector3d forward=A*C.col(0),up=Eigen::Vector3d::UnitZ();
 const double horizontal2=forward.x()*forward.x()+forward.y()*forward.y();
 if(horizontal2<1e-12)throw std::runtime_error("home_yaw_relative: home BODY yaw undefined near vertical forward axis");
 const double yaw=std::atan2(forward.y(),forward.x());
 const Eigen::Matrix3d Z=Eigen::AngleAxisd(yaw,up).toRotationMatrix();
 const Eigen::RowVector3d atan_gradient(-forward.y()/horizontal2,forward.x()/horizontal2,0.);
 const Eigen::RowVector3d yaw_J=atan_gradient*(-A*skew(C.col(0)));
 HomeRelativePose out;out.rotation=Z.transpose()*B*C;
 out.position=Z.transpose()*(p1+B*lever-p0-A*lever);
 out.jacobian.setZero();
 out.jacobian.block<3,3>(0,0)=-skew(up)*out.position*yaw_J+Z.transpose()*A*skew(lever);
 out.jacobian.block<3,3>(0,3)=-Z.transpose();
 out.jacobian.block<3,3>(0,6)=-Z.transpose()*B*skew(lever);
 out.jacobian.block<3,3>(0,9)=Z.transpose();
 // LEFT orientation error and additive position error, BOTH in gravity-aligned H.
 out.jacobian.block<3,3>(3,0)=-up*yaw_J;
 out.jacobian.block<3,3>(3,6)=Z.transpose()*B;
 out.covariance=out.jacobian*joint*out.jacobian.transpose();
 out.covariance=.5*(out.covariance+out.covariance.transpose()).eval();
 if(!out.position.allFinite()||!out.rotation.allFinite()||!out.covariance.allFinite() ||
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>>(out.covariance).eigenvalues().minCoeff() < -1e-10)
   throw std::runtime_error("home_yaw_relative: invalid propagated telemetry");
 return out;
}
}
