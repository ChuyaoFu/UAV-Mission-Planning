#include "track/Grider_FAST.h"
#include "track/Grider_GRID.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cassert>
using ov_core::Grider_FAST;
using ov_core::Grider_GRID;
static void check_mask(const cv::Mat&img,const cv::Mat&mask,const std::vector<cv::KeyPoint>&p){for(auto&k:p){assert(std::isfinite(k.pt.x)&&std::isfinite(k.pt.y));int x=k.pt.x,y=k.pt.y;assert(k.pt.x>=0&&k.pt.y>=0&&x<img.cols&&y<img.rows);assert(mask.at<uchar>(y,x)<=127);}}
static cv::Mat pattern(int rows=240,int cols=320,int contrast=155){cv::Mat im(rows,cols,CV_8UC1,cv::Scalar(60));for(int y=20;y<rows-20;y+=24)for(int x=20;x<cols-20;x+=24)cv::rectangle(im,cv::Rect(x,y,10,10),cv::Scalar(60+contrast),-1);return im;}
int main(int argc,char**argv){
 cv::setNumThreads(1);cv::setRNGSeed(0);std::cout<<"OpenCV="<<CV_VERSION<<" reported_threads="<<cv::getNumThreads()<<"\n";
 cv::Mat im=pattern(),noise(im.size(),CV_8UC1);cv::RNG(443).fill(noise,cv::RNG::UNIFORM,0,256);cv::Mat mask=cv::Mat::zeros(im.size(),CV_8UC1);std::vector<cv::KeyPoint>a,b;
 Grider_FAST::detect_cell(noise,mask,a,25,12,20,true,false);cv::FAST(noise,b,20,true);assert(!a.empty()&&a.size()==b.size());for(size_t i=0;i<a.size();++i)assert(a[i].pt==b[i].pt&&a[i].response==b[i].response);
 Grider_FAST::detect_cell(im,mask,a,25,12,20,true,true);std::vector<cv::Point2f> reference;cv::Mat allowed;cv::compare(mask,cv::Scalar(127),allowed,cv::CMP_LE);cv::goodFeaturesToTrack(im,reference,25,.01,12,allowed,3,false,.04);assert(a.size()==reference.size());for(size_t i=0;i<a.size();++i)assert(a[i].pt==reference[i]);assert(!a.empty()&&a.size()<=25);
 for(size_t i=0;i<a.size();++i)for(size_t j=0;j<i;++j)assert(cv::norm(a[i].pt-a[j].pt)>=12-1e-5);
 mask.setTo(255);Grider_FAST::detect_cell(im,mask,a,25,12,20,true,true);assert(a.empty());
 mask.setTo(128);Grider_FAST::detect_cell(im,mask,a,25,12,20,true,true);assert(a.empty());
 mask.setTo(127);Grider_FAST::detect_cell(im,mask,a,25,12,20,true,true);assert(!a.empty());
 mask.setTo(0);cv::Mat blank(im.size(),CV_8UC1,cv::Scalar(128));Grider_FAST::detect_cell(blank,mask,a,25,12,20,true,true);assert(a.empty());
 cv::Mat low=pattern(240,320,14);Grider_FAST::detect_cell(low,mask,a,25,12,20,true,false);assert(a.empty());Grider_FAST::detect_cell(low,mask,b,25,12,20,true,true);assert(!b.empty());
 std::vector<std::pair<int,int>> locs;for(int y=0;y<4;++y)for(int x=0;x<4;++x)locs.emplace_back(x,y);
 mask.setTo(0);cv::rectangle(mask,cv::Rect(0,0,160,240),cv::Scalar(255),-1);a.clear();Grider_GRID::perform_griding(im,mask,locs,a,64,4,4,20,true,true,12);assert(!a.empty());check_mask(im,mask,a);assert(a.size()<=16*5);
 cv::Mat opposite=255-mask;b.clear();Grider_GRID::perform_griding(im,opposite,locs,b,64,4,4,20,true,true,12);assert(!b.empty());check_mask(im,opposite,b);for(auto&k:a)assert(k.pt.x>=160);for(auto&k:b)assert(k.pt.x<160);
 a.clear();Grider_GRID::perform_griding(blank,mask,locs,a,64,4,4,20,true,true,12);assert(a.empty());
 a.clear();std::vector<std::pair<int,int>> none;Grider_GRID::perform_griding(im,mask,none,a,64,4,4,20,true,true,12);assert(a.empty());
 std::cout<<"PASS actual functions: FAST raw parity; standard GFTT exact parameters/order; minPixelDist; 255/128 excluded and127 allowed; blank0; real lowcontrast corners; opposite-camera masks; post-subpixel mask; original grid quotas; empty cells\n";
 // Fingerprint default FAST grid to compare an independently compiled upstream executable.
 mask.setTo(0);a.clear();Grider_GRID::perform_griding(noise,mask,locs,a,64,4,4,20,true);assert(!a.empty());std::ofstream fp("fast-grid-fingerprint.txt");fp<<std::setprecision(9);for(auto&k:a)fp<<k.pt.x<<","<<k.pt.y<<","<<k.response<<"\n";
 for(int i=1;i<argc;++i){cv::Mat measured=cv::imread(argv[i],cv::IMREAD_GRAYSCALE);assert(!measured.empty());cv::Ptr<cv::CLAHE> clahe=cv::createCLAHE(10.,cv::Size(8,8));cv::Mat preprocessed;clahe->apply(measured,preprocessed);measured=preprocessed;
  cv::Mat zero=cv::imread(std::string(argv[i])+".mask.pgm",cv::IMREAD_GRAYSCALE);const bool supplied_mask=!zero.empty();if(!supplied_mask)zero=cv::Mat::zeros(measured.size(),CV_8UC1);assert(zero.size()==measured.size());std::vector<std::pair<int,int>> grid;for(int y=0;y<5;++y)for(int x=0;x<5;++x)grid.emplace_back(x,y);
  for(bool gftt:{false,true}){std::vector<double> times;size_t count=0;for(int repeat=0;repeat<12;++repeat){std::vector<cv::KeyPoint> pts;auto start=std::chrono::steady_clock::now();Grider_GRID::perform_griding(measured,zero,grid,pts,250,5,5,10,true,gftt,8);auto stop=std::chrono::steady_clock::now();check_mask(measured,zero,pts);count=pts.size();if(repeat>1)times.push_back(std::chrono::duration<double,std::milli>(stop-start).count());}std::sort(times.begin(),times.end());std::cout<<"DETECTOR_PROBE file="<<argv[i]<<" detector="<<(gftt?"GFTT":"FAST")<<" preprocess=CLAHE10_8x8 min_px_dist=8 supplied_mask="<<supplied_mask<<" candidates="<<count<<" median_ms="<<times[times.size()/2]<<" max_ms="<<times.back()<<"\n";}
 }
}
