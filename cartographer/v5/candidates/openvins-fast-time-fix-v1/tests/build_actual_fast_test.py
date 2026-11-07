from pathlib import Path
import sys
base=Path(sys.argv[1]);candidate=Path(sys.argv[2]);target=Path(sys.argv[3]);old=base.read_text();new=candidate.read_text();a=old.index('bool Propagator::fast_state_propagate(');b=old.index('std::vector<ov_core::ImuData> Propagator::select_imu_readings(',a);c=old.index('void Propagator::predict_and_compute(',b)
legacy=old[a:b].replace('Propagator::fast_state_propagate','Propagator::legacy_fast_state_propagate',1);fixed=new[new.index('bool Propagator::fast_state_propagate('):new.index('std::vector<ov_core::ImuData> Propagator::select_imu_readings(')];select=old[b:c]
header=(base.parent/'Propagator.h').read_text();i=header.index('  static ov_core::ImuData interpolate_data(');j=header.index('\n  }',i)+4;interpolate=header[i:j]
state=(base.parent/'State.h').read_text();i=state.index('  static Eigen::Matrix3d Dm(');j=state.index('\n  }',i)+4;dm=state[i:j];i=state.index('  static Eigen::Matrix3d Tg(');j=state.index('\n  }',i)+4;tg=state[i:j]
prefix=r'''
#include "types/IMU.h"
#include "types/Vec.h"
#include "types/JPLQuat.h"
#include "utils/quat_ops.h"
#include <vector>
#include <mutex>
#include <atomic>
#include <iostream>
#include <cassert>
#define PRINT_WARNING(...) ((void)0)
#define PRINT_DEBUG(...) ((void)0)
using namespace ov_type;using namespace ov_core;
namespace ov_core {struct ImuData {double timestamp;Eigen::Vector3d wm,am;};}
struct StateOptions{enum ImuModel{KALIBR,RPNG};ImuModel imu_model=KALIBR;};
struct State{
 StateOptions _options;double _timestamp=10.;std::shared_ptr<IMU>_imu=std::make_shared<IMU>();Eigen::MatrixXd _Cov=Eigen::MatrixXd::Identity(15,15)*.0025;
 std::shared_ptr<Vec>_calib_dt_CAMtoIMU=std::make_shared<Vec>(1),_calib_imu_dw=std::make_shared<Vec>(6),_calib_imu_da=std::make_shared<Vec>(6),_calib_imu_tg=std::make_shared<Vec>(9);
 std::shared_ptr<JPLQuat>_calib_imu_ACCtoIMU=std::make_shared<JPLQuat>(),_calib_imu_GYROtoIMU=std::make_shared<JPLQuat>();
 State(double offset){Eigen::Matrix<double,16,1>x=Eigen::Matrix<double,16,1>::Zero();x[3]=1.;x.segment<3>(4)<<1.,2.,3.;x.segment<3>(7)<<.5,-.3,.2;_imu->set_value(x);_imu->set_local_id(0);Eigen::Matrix<double,1,1>dt;dt<<offset;_calib_dt_CAMtoIMU->set_value(dt);Eigen::Matrix<double,6,1>d;d<<1,0,0,1,0,1;_calib_imu_dw->set_value(d);_calib_imu_da->set_value(d);}
'''
prefix+=dm+'\n'+tg+r'''
};
struct StateHelper{static Eigen::MatrixXd get_marginal_covariance(std::shared_ptr<State>s,const std::vector<std::shared_ptr<Type>>&){return s->_Cov;}};
class Propagator{
public:
 struct Noise{double sigma_w_2=4e-8,sigma_a_2=.0004,sigma_wb_2=1e-10,sigma_ab_2=1e-8;}_noises;
 Eigen::Vector3d _gravity=Eigen::Vector3d(0,0,9.8);std::atomic<bool>cache_imu_valid{false};double cache_state_time=0.,cache_t_off=0.;Eigen::MatrixXd cache_state_est,cache_state_covariance;std::mutex imu_data_mtx;std::vector<ov_core::ImuData>imu_data;
 bool fast_state_propagate(std::shared_ptr<State>,double,Eigen::Matrix<double,13,1>&,Eigen::Matrix<double,12,12>&);
 bool legacy_fast_state_propagate(std::shared_ptr<State>,double,Eigen::Matrix<double,13,1>&,Eigen::Matrix<double,12,12>&);
 static std::vector<ov_core::ImuData> select_imu_readings(const std::vector<ov_core::ImuData>&,double,double,bool);
'''+interpolate+r'''
};
'''
tests=r'''
void data(Propagator&p,double until,bool varying=false){p.imu_data.clear();for(int i=0;i<=100;++i){double t=9.9+i*.004;if(t>until+1e-10)break;ov_core::ImuData d;d.timestamp=t;d.wm=Eigen::Vector3d::Zero();d.am=Eigen::Vector3d(0,0,9.8);if(varying){d.wm<<.01*sin(t),.02*cos(t),.03;d.am+=Eigen::Vector3d(.1*sin(t),.04*cos(t),.02);}p.imu_data.push_back(d);}}
int main(){
 for(double offset:{.005,-.003,0.}){auto s=std::make_shared<State>(offset);const auto state0=s->_imu->value(),P0=s->_Cov;Propagator p;data(p,10.020);Eigen::Matrix<double,13,1>out;Eigen::Matrix<double,12,12>cov;assert(p.fast_state_propagate(s,10.020,out,cov));assert(p.cache_state_time==10.020&&p.cache_t_off==0.);Eigen::Vector3d expected=state0.block<3,1>(4,0)+state0.block<3,1>(7,0)*(10.020-(10.+offset));assert((out.block<3,1>(4,0)-expected).norm()<2e-13);assert((s->_imu->value()-state0).norm()==0.&&(s->_Cov-P0).norm()==0.&&s->_timestamp==10.);
  data(p,10.024);assert(p.fast_state_propagate(s,10.024,out,cov));expected=state0.block<3,1>(4,0)+state0.block<3,1>(7,0)*(10.024-(10.+offset));assert((out.block<3,1>(4,0)-expected).norm()<2e-13&&p.cache_state_time==10.024);
  // Filter-update cache invalidation: next initial start is camera+offset again.
  s->_timestamp=10.05;p.cache_imu_valid=false;data(p,10.068);assert(p.fast_state_propagate(s,10.068,out,cov));expected=state0.block<3,1>(4,0)+state0.block<3,1>(7,0)*(10.068-(10.05+offset));assert((out.block<3,1>(4,0)-expected).norm()<2e-13&&p.cache_state_time==10.068);
 }
 // Actual old function reproduces wrong first endpoint/extrapolation at+5ms.
 auto state=std::make_shared<State>(.005);Propagator legacy;data(legacy,10.020);Eigen::Matrix<double,13,1>x;Eigen::Matrix<double,12,12>P;assert(legacy.legacy_fast_state_propagate(state,10.020,x,P));assert(std::abs(legacy.cache_state_time-10.025)<1e-14);assert(std::abs(x[4]-(1.+.5*.015))>.002);
 // Zero-offset parity: actual original and fixed full bodies, nontrivial IMU
 // and cache refresh, compare complete prediction and12x12cov exactly.
 auto zero=std::make_shared<State>(0.);Propagator oldp,newp;double maxx=0.,maxP=0.;
 for(int k=0;k<12;++k){if(k==6){zero->_timestamp=10.04;oldp.cache_imu_valid=false;newp.cache_imu_valid=false;}double t=10.020+k*.004;data(oldp,t,true);data(newp,t,true);Eigen::Matrix<double,13,1>a,b;Eigen::Matrix<double,12,12>A,B;assert(oldp.legacy_fast_state_propagate(zero,t,a,A)&&newp.fast_state_propagate(zero,t,b,B));maxx=std::max(maxx,(a-b).cwiseAbs().maxCoeff());maxP=std::max(maxP,(A-B).cwiseAbs().maxCoeff());assert((a-b).norm()==0.&&(A-B).norm()==0.);}
 std::cout<<"PASS actual fast_state_propagate+actual select_imu_readings/interpolate_data: positive/negative/zerooffset first cache, next cache and filter-update refresh; constant-motion epoch oracle; originalStateP/mean untouched; original+5ms error reproduced\n";
 std::cout<<"zerooffset_full13state_max_abs_difference="<<maxx<<" zerooffset_full12x12cov_max_abs_difference="<<maxP<<"\n";
}
'''
target.write_text(prefix+legacy+fixed+select+tests)
