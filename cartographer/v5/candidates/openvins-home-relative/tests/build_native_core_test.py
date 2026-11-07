from pathlib import Path
import re,sys
source=Path(sys.argv[1]).read_text();target=Path(sys.argv[2]);pattern=re.compile(r'^(?:void|bool|Eigen::MatrixXd|std::shared_ptr<Type>) StateHelper::(\w+)\(',re.M);matches=list(pattern.finditer(source));names=['EKFPropagation','EKFUpdate','get_marginal_covariance','marginalize','clone','initialize_invertible','augment_clone','marginalize_old_clone'];bodies={x.group(1):source[x.start():matches[i+1].start() if i+1<len(matches) else len(source)] for i,x in enumerate(matches)}
prefix=r'''
#include "types/IMU.h"
#include "types/PoseJPL.h"
#include "types/Vec.h"
#include <Eigen/Geometry>
#include <map>
#include <mutex>
#include <random>
#include <iostream>
#include <algorithm>
#include <cassert>
#define PRINT_ERROR(...) ((void)0)
#define PRINT_WARNING(...) ((void)0)
using namespace ov_type;
namespace ov_msckf {
struct Options{bool retain_home_clone=false,do_calib_camera_timeoffset=false,do_calib_camera_intrinsics=false;int max_clone_size=11;};
struct Camera{void set_value(const Eigen::MatrixXd&) {}};
class State {
public:
 Options _options;Eigen::MatrixXd _Cov;std::vector<std::shared_ptr<Type>>_variables;
 std::shared_ptr<IMU>_imu=std::make_shared<IMU>();std::shared_ptr<PoseJPL>_home_pose=nullptr;double _home_camera_time=-1.,_timestamp=0.;
 std::map<double,std::shared_ptr<PoseJPL>>_clones_IMU;std::mutex _mutex_state;
 std::shared_ptr<Vec>_calib_dt_CAMtoIMU=std::make_shared<Vec>(1);
 std::map<size_t,std::shared_ptr<Vec>>_cam_intrinsics;std::map<size_t,std::shared_ptr<Camera>>_cam_intrinsics_cameras;
 double margtimestep(){return _clones_IMU.begin()->first;}
 State(bool retain){_options.retain_home_clone=retain;_imu->set_local_id(0);_variables.push_back(_imu);Eigen::Matrix<double,16,1>x=Eigen::Matrix<double,16,1>::Zero();x[3]=1.;x.segment<3>(4)<<.2,-.1,.4;_imu->set_value(x);_imu->set_fej(x);_Cov=Eigen::MatrixXd::Identity(15,15)*.0025;}
};
class StateHelper{
public:
 static void EKFPropagation(std::shared_ptr<State>,const std::vector<std::shared_ptr<Type>>&,const std::vector<std::shared_ptr<Type>>&,const Eigen::MatrixXd&,const Eigen::MatrixXd&);
 static void EKFUpdate(std::shared_ptr<State>,const std::vector<std::shared_ptr<Type>>&,const Eigen::MatrixXd&,const Eigen::VectorXd&,const Eigen::MatrixXd&);
 static Eigen::MatrixXd get_marginal_covariance(std::shared_ptr<State>,const std::vector<std::shared_ptr<Type>>&);
 static void marginalize(std::shared_ptr<State>,std::shared_ptr<Type>);
 static std::shared_ptr<Type>clone(std::shared_ptr<State>,std::shared_ptr<Type>);
 static void initialize_invertible(std::shared_ptr<State>,std::shared_ptr<Type>,const std::vector<std::shared_ptr<Type>>&,const Eigen::MatrixXd&,const Eigen::MatrixXd&,const Eigen::MatrixXd&,const Eigen::VectorXd&);
 static void augment_clone(std::shared_ptr<State>,Eigen::Matrix<double,3,1>);
 static void marginalize_old_clone(std::shared_ptr<State>);
};
'''
tests=r'''
} // namespace ov_msckf
using namespace ov_msckf;
std::vector<std::shared_ptr<Type>> active(std::shared_ptr<State>s,const std::vector<std::shared_ptr<Vec>>&features){std::vector<std::shared_ptr<Type>>v={s->_imu};for(auto&c:s->_clones_IMU)v.push_back(c.second);for(auto&f:features)v.push_back(f);return v;}
int main(){
 auto base=std::make_shared<State>(false),kept=std::make_shared<State>(true);std::vector<std::shared_ptr<Vec>>fb,fk;
 StateHelper::augment_clone(base,Eigen::Vector3d::Zero());StateHelper::augment_clone(kept,Eigen::Vector3d::Zero());assert(kept->_home_pose==kept->_clones_IMU.begin()->second);assert(!base->_home_pose);const auto home=kept->_home_pose;const auto home0=home->value();
 std::mt19937 rng(5841);std::normal_distribution<double>normal;auto randmat=[&](int rows,int cols){Eigen::MatrixXd x(rows,cols);for(int i=0;i<rows;++i)for(int j=0;j<cols;++j)x(i,j)=normal(rng);return x;};
 double maxP=0.,maxmean=0.;int augments=0,marginalized=0;
 for(int step=1;step<=240;++step){
  Eigen::MatrixXd Phi=Eigen::MatrixXd::Identity(15,15);Phi.block<3,3>(3,6)=.05*Eigen::Matrix3d::Identity();Phi.block<3,3>(0,9)=-.05*Eigen::Matrix3d::Identity();Phi.block<3,3>(6,12)=-.05*Eigen::Matrix3d::Identity();Eigen::MatrixXd Q=1e-7*Eigen::MatrixXd::Identity(15,15);
  StateHelper::EKFPropagation(base,{base->_imu},{base->_imu},Phi,Q);StateHelper::EKFPropagation(kept,{kept->_imu},{kept->_imu},Phi,Q);
  Eigen::VectorXd motion=randmat(15,1)*1e-5;base->_imu->update(motion);kept->_imu->update(motion);base->_timestamp=kept->_timestamp=step*.05;
  StateHelper::augment_clone(base,Eigen::Vector3d::Zero());StateHelper::augment_clone(kept,Eigen::Vector3d::Zero());
  std::vector<std::shared_ptr<Type>>ob={base->_imu,base->_clones_IMU.begin()->second,base->_clones_IMU.rbegin()->second},ok={kept->_imu,kept->_clones_IMU.begin()->second,kept->_clones_IMU.rbegin()->second};
  if(!fb.empty()){ob.push_back(fb.back());ok.push_back(fk.back());}
  int size=0;for(auto&t:ob)size+=t->size();Eigen::MatrixXd H=randmat(6,size)*.3;Eigen::VectorXd res=randmat(6,1)*1e-4;Eigen::MatrixXd R=Eigen::MatrixXd::Identity(6,6)*1e-3;
  StateHelper::EKFUpdate(base,ob,H,res,R);StateHelper::EKFUpdate(kept,ok,H,res,R);
  if(step%17==0){auto a=std::make_shared<Vec>(3),b=std::make_shared<Vec>(3);Eigen::MatrixXd hx=randmat(3,15)*.2;Eigen::VectorXd rr=randmat(3,1)*1e-4;Eigen::MatrixXd I=Eigen::MatrixXd::Identity(3,3);StateHelper::initialize_invertible(base,a,{base->_imu},hx,I,I*.001,rr);StateHelper::initialize_invertible(kept,b,{kept->_imu},hx,I,I*.001,rr);fb.push_back(a);fk.push_back(b);++augments;}
  if(fb.size()>3){StateHelper::marginalize(base,fb.front());StateHelper::marginalize(kept,fk.front());fb.erase(fb.begin());fk.erase(fk.begin());++marginalized;}
  StateHelper::marginalize_old_clone(base);StateHelper::marginalize_old_clone(kept);
  assert(base->_clones_IMU.size()==kept->_clones_IMU.size()&&base->_clones_IMU.size()<=11);auto ab=active(base,fb),ak=active(kept,fk);assert(ab.size()==ak.size());
  for(size_t i=0;i<ab.size();++i){maxmean=std::max(maxmean,(ab[i]->value()-ak[i]->value()).cwiseAbs().maxCoeff());assert((ab[i]->fej()-ak[i]->fej()).cwiseAbs().maxCoeff()<1e-12);}
  auto Pb=StateHelper::get_marginal_covariance(base,ab),Pk=StateHelper::get_marginal_covariance(kept,ak);maxP=std::max(maxP,(Pb-Pk).cwiseAbs().maxCoeff());assert(maxmean<1e-11&&maxP<1e-12);
  if(step>=11){assert(kept->_clones_IMU.count(0.)==0);assert(kept->_Cov.rows()==base->_Cov.rows()+6);assert(home->id()>=0);}else assert(kept->_Cov.rows()==base->_Cov.rows());
  assert(Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(kept->_Cov).eigenvalues().minCoeff()>-1e-10);
 }
 assert((home->value()-home0).norm()>1e-6);assert((kept->_Cov.block(home->id(),0,6,15)).norm()>0);auto before=kept->_Cov;StateHelper::marginalize(kept,home);assert((kept->_Cov-before).norm()==0);assert(home->id()>=0); // also protects direct ZUPT-style deletion
 std::cout<<"PASS actual StateHelper bodies:240propagate/clone/EKFupdate/window-evict cycles;14delayed variable initializations/11marginalizations; recent11 same; retainedhome+6only; nonzero cross and updatedhome mean; FEJ unchanged; PSD; direct deletion guard\n";
 std::cout<<"max_active_mean_abs_difference="<<maxmean<<" max_active_cov_abs_difference="<<maxP<<"\n";
}
'''
target.write_text(prefix+'\n'.join(bodies[n] for n in names)+tests)
