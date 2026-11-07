#include "track/Grider_FAST.h"
#include "track/Grider_GRID.h"
#include <fstream>
#include <iomanip>
#include <cassert>
int main(int argc,char**argv){assert(argc==2);cv::setNumThreads(1);cv::Mat image(240,320,CV_8UC1);cv::RNG(443).fill(image,cv::RNG::UNIFORM,0,256);cv::Mat mask=cv::Mat::zeros(image.size(),CV_8UC1);std::vector<std::pair<int,int>>locs;for(int y=0;y<4;++y)for(int x=0;x<4;++x)locs.emplace_back(x,y);std::vector<cv::KeyPoint>pts;ov_core::Grider_GRID::perform_griding(image,mask,locs,pts,64,4,4,20,true);assert(!pts.empty());std::ofstream out(argv[1]);out<<std::setprecision(9);for(auto&k:pts)out<<k.pt.x<<","<<k.pt.y<<","<<k.response<<"\n";}
