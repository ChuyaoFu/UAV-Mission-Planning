#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "line_descriptor_custom.hpp"
#include <chrono>
#include <ctime>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
using cv::line_descriptor::KeyLine;
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
int main(int argc,char**argv){
 if(argc!=2){std::cerr<<"usage: plvins_lsd_probe manifest.csv (stamp_ns,absolute_PGM_path)\n";return 2;}
 cv::setNumThreads(1);
 std::ifstream input(argv[1]);if(!input){return 2;}
 std::cerr<<"OpenCV="<<CV_VERSION<<" requested_threads=1 reported_threads="<<cv::getNumThreads()<<"\n";
 std::cout<<"stamp_ns,point_candidates,line_candidates,matched_lines,line_tracks_ge3,gap_reset,detect_ms,describe_ms,match_ms,total_ms,cpu_ms,orientation_bins\n";
 std::string row;std::vector<KeyLine> previous;cv::Mat previous_desc;std::vector<int> previous_age;long long last=-1;
 while(std::getline(input,row)){
  if(row.empty()||row[0]=='#')continue;
  const auto comma=row.find(',');if(comma==std::string::npos){std::cerr<<"bad manifest\n";return 2;}
  const long long stamp=std::stoll(row.substr(0,comma));const std::string path=row.substr(comma+1);
  if(last>=stamp){std::cerr<<"nonmonotonic measured timestamps\n";return 2;}
  const bool reset=last>=0 && stamp-last>250000000LL;
  if(reset){previous.clear();previous_desc.release();previous_age.clear();}
  cv::Mat raw=cv::imread(path,cv::IMREAD_GRAYSCALE);if(raw.empty()){std::cerr<<"bad measured image "<<path<<"\n";return 2;}
  const auto start=Clock::now();const auto cpu=std::clock();
  cv::Mat img;cv::createCLAHE(3.,cv::Size(8,8))->apply(raw,img);
  std::vector<cv::Point2f> points;cv::goodFeaturesToTrack(img,points,150,.01,30);
  auto detector=cv::line_descriptor::LSDDetectorC::createLSDDetectorC();
  cv::line_descriptor::LSDDetectorC::LSDOptions opts;
  opts.refine=1;opts.scale=.5;opts.sigma_scale=.6;opts.quant=2.;opts.ang_th=22.5;opts.log_eps=1.;opts.density_th=.6;opts.n_bins=1024;opts.min_length=.125*std::min(img.cols,img.rows);
  std::vector<KeyLine> all,lines;auto d0=Clock::now();detector->detect(img,all,2,1,opts);auto d1=Clock::now();
  cv::Mat descriptors,selected;
  auto bd=cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
  if(!all.empty())bd->compute(img,all,descriptors);
  for(size_t i=0;i<all.size();++i)if(all[i].octave==0 && all[i].lineLength>=60){lines.push_back(all[i]);selected.push_back(descriptors.row(int(i)));}
  auto d2=Clock::now();std::vector<int> ages(lines.size(),1);int matched=0,mature=0;
  if(!selected.empty() && !previous_desc.empty()){
   std::vector<cv::DMatch> matches;cv::line_descriptor::BinaryDescriptorMatcher::createBinaryDescriptorMatcher()->match(selected,previous_desc,matches);
   for(const auto&m:matches){if(m.distance>=30)continue;const auto&a=lines[m.queryIdx];const auto&b=previous[m.trainIdx];
    // Exact pinned PL-VINS matching gates, including its asymmetric start/end check.
    const auto serr=a.getStartPoint()-b.getEndPoint();const auto eerr=a.getEndPoint()-b.getEndPoint();
    if(serr.dot(serr)<200*200 && eerr.dot(eerr)<200*200 && std::abs(a.angle-b.angle)<.1){ages[m.queryIdx]=previous_age[m.trainIdx]+1;++matched;}
   }
  }
  auto end=Clock::now();bool bins[6]={};for(size_t i=0;i<lines.size();++i){mature+=ages[i]>=3;double a=std::fmod(lines[i].angle+2*CV_PI,CV_PI);bins[std::min(5,int(a/CV_PI*6))]=true;}int nbin=0;for(bool b:bins)nbin+=b;
  std::cout<<stamp<<","<<points.size()<<","<<lines.size()<<","<<matched<<","<<mature<<","<<int(reset)<<","<<ms(d0,d1)<<","<<ms(d1,d2)<<","<<ms(d2,end)<<","<<ms(start,end)<<","<<1000.*(std::clock()-cpu)/CLOCKS_PER_SEC<<","<<nbin<<"\n";
  previous=lines;previous_desc=selected;previous_age=ages;last=stamp;
 }
}
