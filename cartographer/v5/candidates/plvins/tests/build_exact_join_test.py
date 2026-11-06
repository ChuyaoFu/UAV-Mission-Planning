from pathlib import Path
import sys
old,new,out=map(Path,sys.argv[1:])
def extract(p,name):
 s=p.read_text();a=s.index('std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>,');b=s.index('\nvoid imu_callback(',a);return s[a:b].replace('getMeasurements()\n','%s()\n'%name)
prefix=r'''
#include <queue>
#include <vector>
#include <memory>
#include <cassert>
#include <iostream>
#include <utility>
#define ROS_WARN(...) ((void)0)
struct Stamp {long long ns=0;bool operator<(Stamp o)const{return ns<o.ns;}bool operator>(Stamp o)const{return ns>o.ns;}bool operator<=(Stamp o)const{return ns<=o.ns;}double toSec()const{return double(ns);}};
struct Header {Stamp stamp;};
namespace sensor_msgs {struct Imu{Header header;};struct PointCloud{Header header;};using ImuConstPtr=std::shared_ptr<const Imu>;using PointCloudConstPtr=std::shared_ptr<const PointCloud>;}
std::queue<sensor_msgs::ImuConstPtr> imu_buf;
std::queue<sensor_msgs::PointCloudConstPtr> feature_buf,linefeature_buf;
int sum_of_wait=0;
void reset(){imu_buf={};feature_buf={};linefeature_buf={};for(int i=0;i<=100;i++){auto m=std::make_shared<sensor_msgs::Imu>();m->header.stamp.ns=i;imu_buf.push(m);}}
void points(std::initializer_list<int> p,std::initializer_list<int> l){for(int i:p){auto m=std::make_shared<sensor_msgs::PointCloud>();m->header.stamp.ns=i;feature_buf.push(m);}for(int i:l){auto m=std::make_shared<sensor_msgs::PointCloud>();m->header.stamp.ns=i;linefeature_buf.push(m);}}
'''
tests=r'''
int main(){
 reset();points({10,20,30},{10,15,20,30});auto old=baseline();assert(old.size()==3);assert(old[1].second.first->header.stamp.ns!=old[1].second.second->header.stamp.ns);
 reset();points({10,20,30},{10,15,20,30});auto good=getMeasurements();assert(good.size()==3);for(auto&p:good)assert(p.second.first->header.stamp.ns==p.second.second->header.stamp.ns);
 reset();points({10,15,20,30},{10,20,30});good=getMeasurements();assert(good.size()==3);for(auto&p:good)assert(p.second.first->header.stamp.ns==p.second.second->header.stamp.ns);
 reset();points({10},{20});assert(getMeasurements().empty());assert(feature_buf.empty() && linefeature_buf.size()==1);
 reset();points({20},{10});assert(getMeasurements().empty());assert(linefeature_buf.empty() && feature_buf.size()==1);
 reset();points({0,10},{0,10});good=getMeasurements();assert(good.size()==1 && good[0].second.first->header.stamp.ns==10);
 reset();points({100},{100});assert(getMeasurements().empty());assert(feature_buf.size()==1 && linefeature_buf.size()==1);
 std::cout<<"PASS 7 actual getMeasurements cases: baseline mismatch reproduced; exact pairing/drop recovery and unchanged IMU bracketing\n";
}
'''
out.write_text(prefix+extract(old,'baseline')+extract(new,'getMeasurements')+tests)
