#include "SensorInputAudit.h"
#include <cassert>
#include <cstdio>
#include <iostream>
int main(int argc,char**argv){assert(argc==2);std::remove(argv[1]);ov_core::ImuData imu;imu.timestamp=10.;imu.wm<<.01,.02,.03;imu.am<<.1,.2,9.8;const auto wm=imu.wm,am=imu.am;ov_core::CameraData camera;camera.timestamp=10.004;camera.sensor_ids={0,1};cv::Mat backing(8,10,CV_8UC1,cv::Scalar(42));cv::Mat roi=backing(cv::Rect(1,1,6,6));camera.images={roi,roi.clone()};camera.masks={cv::Mat::zeros(6,6,CV_8UC1),cv::Mat::zeros(6,6,CV_8UC1)};const auto original=camera.images[0].clone();
 {ov_msckf::SensorInputAudit audit(argv[1]);audit.imu(imu);imu.timestamp=10.004;audit.imu(imu);audit.camera(camera);audit.motion();Eigen::MatrixXd x=Eigen::MatrixXd::Zero(16,1),P=Eigen::MatrixXd::Identity(15,15)*.0025;x(3)=1.;const auto P0=P;audit.initialized(9.,x,P);assert((P-P0).norm()==0.);camera.images[1].at<uchar>(1,1)=43;audit.camera(camera);}
 assert((imu.wm-wm).norm()==0.&&(imu.am-am).norm()==0.);assert(cv::norm(camera.images[0],original)==0.);bool rejected=false;try{ov_msckf::SensorInputAudit overwrite(argv[1]);}catch(const std::exception&){rejected=true;}assert(rejected);
 std::cout<<"PASS actual logger header: monotonic ledger, raw IMU values, image/mask payload hashes, noncontinuous images, init full state/P read-only, END completeness, no overwrite\n";
}
