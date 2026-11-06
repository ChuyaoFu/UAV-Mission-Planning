#include "home_relative_pose.h"
#include "types/PoseJPL.h"
#include <random>
#include <cassert>
#include <iostream>
using V3=Eigen::Vector3d;using M3=Eigen::Matrix3d;using Joint=Eigen::Matrix<double,12,12>;
void set(ov_type::PoseJPL&pose,const M3&R,const V3&p){Eigen::Quaterniond q(R);Eigen::Matrix<double,7,1>v;v<<q.coeffs(),p;pose.set_value(v);}
struct Motion{M3 R;V3 t;};
Motion relative(const ov_type::PoseJPL&a,const ov_type::PoseJPL&b,const M3&C,const V3&l){const M3 A=a.Rot().transpose(),B=b.Rot().transpose();const M3 home=A*C;return {home.transpose()*B*C,home.transpose()*(b.pos()+B*l-a.pos()-A*l)};}
int main(){
 std::mt19937 generator(6381);std::normal_distribution<double>d;auto vec=[&](){return V3(d(generator),d(generator),d(generator));};auto rot=[&](){V3 axis=vec();return M3(Eigen::AngleAxisd(d(generator),axis.normalized()));};double maxerror=0.,maxgauge=0.;
 for(int k=0;k<200;++k){M3 A=rot(),B=rot(),C=rot();V3 p0=vec(),p1=vec(),l=vec()*.1;Joint F;for(int r=0;r<12;++r)for(int c=0;c<12;++c)F(r,c)=d(generator);Joint P=1e-6*F*F.transpose(),original=P;auto out=ov_msckf::home_relative_body_pose(A,p0,B,p1,C,l,P);assert((P-original).norm()==0.);
  ov_type::PoseJPL a,b;set(a,A,p0);set(b,B,p1);Motion nom=relative(a,b,C,l);assert((out.rotation-nom.R).norm()<2e-14&&(out.position-nom.t).norm()<2e-14);
  Eigen::Matrix<double,6,12>J;
  for(int j=0;j<12;++j){V3 dt[2],dr[2];for(int sign=0;sign<2;++sign){set(a,A,p0);set(b,B,p1);Eigen::VectorXd dx=Eigen::VectorXd::Zero(6);dx[j%6]=(sign==0?1.:-1.)*1e-6;if(j<6)a.update(dx);else b.update(dx);Motion pert=relative(a,b,C,l);dt[sign]=pert.t-nom.t;Eigen::AngleAxisd aa(pert.R*nom.R.transpose());dr[sign]=aa.angle()*aa.axis();}J.block<3,1>(0,j)=(dt[0]-dt[1])/(2e-6);J.block<3,1>(3,j)=(dr[0]-dr[1])/(2e-6);}
  maxerror=std::max(maxerror,(J-out.jacobian).cwiseAbs().maxCoeff());assert(maxerror<3e-9);
  assert((out.covariance-J*P*J.transpose()).norm()<1e-12);assert((Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>>(out.covariance).eigenvalues().minCoeff()>-1e-12));
  Eigen::Matrix<double,12,3>N=Eigen::Matrix<double,12,3>::Zero();N.block<3,3>(3,0)=M3::Identity();N.block<3,3>(9,0)=M3::Identity();Joint common=P+N*(M3::Identity()*.0025)*N.transpose();auto withgauge=ov_msckf::home_relative_body_pose(A,p0,B,p1,C,l,common);maxgauge=std::max(maxgauge,(out.covariance-withgauge.covariance).cwiseAbs().maxCoeff());assert(maxgauge<1e-16);
 }
 // Exact correlated same-pose reference has zero relative uncertainty by
 // coordinate identity, while original global prior remains nonzero.
 Joint same;Eigen::Matrix<double,6,6>P6=Eigen::Matrix<double,6,6>::Identity()*.0025;for(int r=0;r<2;++r)for(int c=0;c<2;++c)same.block<6,6>(6*r,6*c)=P6;auto exact=ov_msckf::home_relative_body_pose(M3::Identity(),V3::Zero(),M3::Identity(),V3::Zero(),M3::Identity(),V3(.2,.03,.04),same);assert(exact.position.norm()==0.&&exact.covariance.norm()<1e-17);assert(same.diagonal().minCoeff()>.002);
 std::cout<<"PASS200 actual PoseJPL native central differences for HOME-frame position+LEFT-angle6cov, nonzero lever, full cross blocks, PSD, unchangedP and correlated identity\n";
 std::cout<<"max_J_abs_error="<<maxerror<<" max_common_translation_cov_abs_change="<<maxgauge<<"\n";
}
