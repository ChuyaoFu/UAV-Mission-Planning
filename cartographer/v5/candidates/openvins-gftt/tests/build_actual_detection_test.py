from pathlib import Path
import sys
src=Path(sys.argv[1]).read_text();target=Path(sys.argv[2]);mono=src[src.index('void TrackKLT::perform_detection_monocular'):src.index('void TrackKLT::perform_matching')];wrapper=src[src.index('void TrackKLT::extract_grid'):]
prefix=r'''
#include "track/Grider_FAST.h"
#include "track/Grider_GRID.h"
#include <atomic>
#include <chrono>
#include <cassert>
#define PRINT_INFO(...) ((void)0)
using namespace ov_core;
class TrackKLT {
public:
 int num_features=80,grid_x=4,grid_y=4,min_px_dist=12,threshold=20,pyr_levels=2;
 bool use_gftt=true,record_detection_timing=true;
 cv::Size win_size=cv::Size(15,15);std::atomic<size_t>currid{0};
 void perform_detection_monocular(const std::vector<cv::Mat>&,const cv::Mat&,std::vector<cv::KeyPoint>&,std::vector<size_t>&);
 void perform_detection_stereo(const std::vector<cv::Mat>&,const std::vector<cv::Mat>&,const cv::Mat&,const cv::Mat&,size_t,size_t,std::vector<cv::KeyPoint>&,std::vector<cv::KeyPoint>&,std::vector<size_t>&,std::vector<size_t>&);
 void extract_grid(const cv::Mat&,const cv::Mat&,const std::vector<std::pair<int,int>>&,std::vector<cv::KeyPoint>&);
};
'''
tests=r'''
void allowed(const cv::Mat&mask,const std::vector<cv::KeyPoint>&pts){for(const auto&k:pts)assert(mask.at<uchar>((int)k.pt.y,(int)k.pt.x)<=127);}
int main(){
 cv::setNumThreads(1);cv::Mat image(240,320,CV_8UC1,cv::Scalar(60));for(int y=20;y<220;y+=24)for(int x=20;x<300;x+=24)cv::rectangle(image,cv::Rect(x,y,10,10),cv::Scalar(74),-1);
 cv::Mat zero=cv::Mat::zeros(image.size(),CV_8UC1),white(image.size(),CV_8UC1,cv::Scalar(255)),leftmask=zero.clone();cv::rectangle(leftmask,cv::Rect(0,0,160,240),cv::Scalar(255),-1);cv::Mat rightmask=255-leftmask;
 std::vector<cv::Mat>pyr;cv::buildOpticalFlowPyramid(image,pyr,cv::Size(15,15),2);
 TrackKLT t;std::vector<cv::KeyPoint>l,r;std::vector<size_t>li,ri;
 t.perform_detection_monocular(pyr,leftmask,l,li);assert(!l.empty()&&l.size()==li.size());allowed(leftmask,l);
 t.perform_detection_monocular(pyr,white,l,li);assert(l.empty()&&li.empty());
 // The right camera must be able to extract left-half corners even when
 // that region is excluded in the LEFT camera's own mask.
 t.perform_detection_stereo(pyr,pyr,white,rightmask,0,1,l,r,li,ri);assert(l.empty()&&li.empty());assert(!r.empty()&&r.size()==ri.size());allowed(rightmask,r);for(auto&k:r)assert(k.pt.x<160);
 // Original left-to-right KLT remains active with unmodified status gates.
 l.clear();r.clear();li.clear();ri.clear();t.perform_detection_stereo(pyr,pyr,zero,zero,0,1,l,r,li,ri);assert(!l.empty()&&!r.empty());assert(l.size()==li.size()&&r.size()==ri.size());size_t shared=0;for(auto id:li)if(std::find(ri.begin(),ri.end(),id)!=ri.end())++shared;assert(shared>0);
 cv::Mat blank(image.size(),CV_8UC1,cv::Scalar(128));std::vector<cv::Mat> blankpyr;cv::buildOpticalFlowPyramid(blank,blankpyr,cv::Size(15,15),2);l.clear();r.clear();li.clear();ri.clear();t.perform_detection_stereo(blankpyr,blankpyr,zero,zero,0,1,l,r,li,ri);assert(l.empty()&&r.empty());
 std::cout<<"PASS actual unedited extracted mono/stereo detection methods: per-camera self-masks, right mask ownership, lowcontrast real corners, unchanged stereo LK shared IDs, blank0; ASan/UBSan\n";
}
'''
target.write_text(prefix+mono+wrapper+tests)
