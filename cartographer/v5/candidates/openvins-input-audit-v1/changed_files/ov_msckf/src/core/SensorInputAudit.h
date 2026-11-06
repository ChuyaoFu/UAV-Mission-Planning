#pragma once
#include "utils/sensor_data.h"
#include <fstream>
#include <iomanip>
#include <mutex>
#include <cstdint>
#include <cstring>
#include <stdexcept>
namespace ov_msckf {
// Diagnostic ONLY. Records actual core ENTRY order/values, not intended bag order.
class SensorInputAudit {
 std::ofstream file;std::mutex mutex;std::uint64_t seq=0,imu_count=0,imu_hash=14695981039346656037ULL;double last_imu=-1.;
 static void bytes(std::uint64_t&h,const void*ptr,size_t n){const auto*p=static_cast<const unsigned char*>(ptr);for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ULL;}}
 static std::uint64_t image_hash(const cv::Mat&m){std::uint64_t h=14695981039346656037ULL;int shape[]={m.rows,m.cols,m.type()};bytes(h,shape,sizeof(shape));for(int r=0;r<m.rows;++r)bytes(h,m.ptr(r),m.cols*m.elemSize());return h;}
 void prefix(const char*kind){file<<kind<<','<<seq++<<','<<imu_count<<','<<last_imu<<','<<imu_hash;}
 void finish(){file<<'\n';}
public:
 explicit SensorInputAudit(const std::string&path){std::ifstream existing(path);if(existing.good())throw std::runtime_error("sensor audit refuses existing output");file.open(path,std::ios::out);if(!file)throw std::runtime_error("cannot open explicit sensor audit output");file<<std::scientific<<std::setprecision(17)<<"# SENSOR_INPUT_AUDIT_V1 native_core_entry_only no_truth no_filter_change\n";}
 ~SensorInputAudit(){std::lock_guard<std::mutex>lock(mutex);prefix("END");finish();file.flush();}
 void imu(const ov_core::ImuData&m){std::lock_guard<std::mutex>lock(mutex);const double values[]={m.timestamp,m.wm.x(),m.wm.y(),m.wm.z(),m.am.x(),m.am.y(),m.am.z()};bytes(imu_hash,values,sizeof(values));++imu_count;last_imu=m.timestamp;prefix("IMU");for(double x:values)file<<','<<x;finish();}
 void camera(const ov_core::CameraData&m){std::lock_guard<std::mutex>lock(mutex);prefix("CAM");file<<','<<m.timestamp<<','<<m.sensor_ids.size();for(size_t i=0;i<m.sensor_ids.size();++i){const auto&im=m.images.at(i);file<<','<<m.sensor_ids[i]<<','<<im.rows<<','<<im.cols<<','<<im.type()<<','<<image_hash(im)<<','<<image_hash(m.masks.at(i));}finish();file.flush();}
 void motion(){std::lock_guard<std::mutex>lock(mutex);prefix("MOTION_NOTICE");finish();}
 void initialized(double timestamp,const Eigen::MatrixXd&value,const Eigen::MatrixXd&covariance){std::lock_guard<std::mutex>lock(mutex);prefix("INIT");file<<','<<timestamp<<','<<value.rows()<<','<<value.cols();for(int r=0;r<value.rows();++r)for(int c=0;c<value.cols();++c)file<<','<<value(r,c);file<<','<<covariance.rows()<<','<<covariance.cols();for(int r=0;r<covariance.rows();++r)for(int c=0;c<covariance.cols();++c)file<<','<<covariance(r,c);finish();file.flush();}
};
}
