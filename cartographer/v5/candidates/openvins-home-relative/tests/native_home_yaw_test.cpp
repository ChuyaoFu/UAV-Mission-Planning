#include "home_yaw_relative_pose.h"
#include "types/PoseJPL.h"
#include <random>
#include <cassert>
#include <iostream>
using V3=Eigen::Vector3d;using M3=Eigen::Matrix3d;using Joint=Eigen::Matrix<double,12,12>;
void set(ov_type::PoseJPL&p,const M3&R,const V3&t){Eigen::Quaterniond q(R);Eigen::Matrix<double,7,1>x;x<<q.coeffs(),t;p.set_value(x);}
struct Motion{M3 R;V3 t;};
Motion motion(const ov_type::PoseJPL&a,const ov_type::PoseJPL&b,const M3&C,const V3&l){M3 A=a.Rot().transpose(),B=b.Rot().transpose(),body=A*C;double yaw=std::atan2(body(1,0),body(0,0));M3 Z=Eigen::AngleAxisd(yaw,V3::UnitZ()).toRotationMatrix();return {Z.transpose()*B*C,Z.transpose()*(b.pos()+B*l-a.pos()-A*l)};}
int main(){std::mt19937 rng(5491);std::normal_distribution<double>d;auto v=[&](){return V3(d(rng),d(rng),d(rng));};auto r=[&](){V3 a=v();return M3(Eigen::AngleAxisd(d(rng),a.normalized()));};double maxJ=0.;
 for(int k=0;k<200;++k){M3 A=r(),B=r(),C=r();V3 p0=v(),p1=v(),l=.2*v();Joint F;for(int a=0;a<12;++a)for(int b=0;b<12;++b)F(a,b)=d(rng);Joint P=1e-6*F*F.transpose(),before=P;auto out=ov_msckf::home_yaw_relative_body_pose(A,p0,B,p1,C,l,P);assert((P-before).norm()==0.);ov_type::PoseJPL a,b;set(a,A,p0);set(b,B,p1);auto nominal=motion(a,b,C,l);assert((out.position-nominal.t).norm()<1e-13&&(out.rotation-nominal.R).norm()<1e-13);
  Eigen::Matrix<double,6,12>J;for(int col=0;col<12;++col){V3 t[2],phi[2];for(int side=0;side<2;++side){set(a,A,p0);set(b,B,p1);Eigen::VectorXd dx=Eigen::VectorXd::Zero(6);dx[col%6]=(side==0?1.:-1.)*1e-6;if(col<6)a.update(dx);else b.update(dx);auto pert=motion(a,b,C,l);t[side]=pert.t-nominal.t;Eigen::AngleAxisd aa(pert.R*nominal.R.transpose());phi[side]=aa.angle()*aa.axis();}J.block<3,1>(0,col)=(t[0]-t[1])/(2e-6);J.block<3,1>(3,col)=(phi[0]-phi[1])/(2e-6);}maxJ=std::max(maxJ,(J-out.jacobian).cwiseAbs().maxCoeff());assert((J-out.jacobian).cwiseAbs().maxCoeff()<1e-7);assert((out.covariance-J*P*J.transpose()).norm()<1e-10);
  Eigen::Matrix<double,12,3>N=Eigen::Matrix<double,12,3>::Zero();N.block<3,3>(3,0)=M3::Identity();N.block<3,3>(9,0)=M3::Identity();assert((out.jacobian*N).norm()<1e-14);
 }
 // Pitched IMU20deg, BODYnotassumedlevel: yaw frame preserves world-up.
 M3 RBI=Eigen::AngleAxisd(20.*M_PI/180.,V3::UnitY()).toRotationMatrix();M3 C=RBI.transpose();M3 body=Eigen::AngleAxisd(.3,V3::UnitZ()).toRotationMatrix()*Eigen::AngleAxisd(.12,V3::UnitY()).toRotationMatrix()*Eigen::AngleAxisd(-.08,V3::UnitX()).toRotationMatrix();M3 A=body*RBI;V3 lever=-C*V3(.22,0,.055);Joint P;Eigen::Matrix<double,6,6>Q=Eigen::Matrix<double,6,6>::Identity()*.0025;for(int i=0;i<2;++i)for(int j=0;j<2;++j)P.block<6,6>(i*6,j*6)=Q;auto atHome=ov_msckf::home_yaw_relative_body_pose(A,V3::Zero(),A,V3::Zero(),C,lever,P);assert(atHome.position.norm()<1e-15);assert((atHome.covariance.block<3,3>(0,0).norm()<1e-17));assert((atHome.covariance.block<3,3>(3,3).trace()>.004)); // roll/pitch remain uncertain!
 M3 Z=Eigen::AngleAxisd(.3,V3::UnitZ()).toRotationMatrix();assert((Z*V3::UnitZ()-V3::UnitZ()).norm()==0.);assert((atHome.rotation-Z.transpose()*body).norm()<1e-14);
 std::cout<<"PASS200 actual PoseJPL native perturbation checks: gravity-aligned HOME BODY yaw Jacobian, arbitrary home roll/pitch, fixed BODY lever, full joint6cov; pitchedIMU20deg test; initial common-position cancels while real roll/pitch uncertainty remains\n";
 std::cout<<"max_J_abs_error="<<maxJ<<" initial_rollpitch_cov_trace="<<atHome.covariance.block<3,3>(3,3).trace()<<"\n";
}
