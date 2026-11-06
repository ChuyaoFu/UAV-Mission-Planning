#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "line_descriptor_custom.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
extern "C" const char* boundary_lbd_opencv_version(){return CV_VERSION;}
extern "C" int boundary_lbd_describe(const unsigned char* image,int w,int h,const float* lines,int count,unsigned char* out,unsigned char* valid){
 if(!image||!lines||!out||!valid||w<16||h<16||w>1920||h>1200||count<0||count>1024)return -1;
 std::fill(out,out+count*32,0);std::fill(valid,valid+count,0);
 try {
  cv::setNumThreads(1); // Isolated worker only; keep native CPU admission bounded.
  cv::Mat gray(h,w,CV_8UC1,const_cast<unsigned char*>(image));std::vector<cv::line_descriptor::KeyLine> keys;std::vector<int> source_ids;
  for(int i=0;i<count;++i){const float* f=lines+4*i;float dx=f[2]-f[0],dy=f[3]-f[1],length=std::hypot(dx,dy);if(!std::isfinite(length)||length<1||f[0]<1||f[1]<1||f[2]<1||f[3]<1||f[0]>=w-1||f[2]>=w-1||f[1]>=h-1||f[3]>=h-1)continue;
   cv::line_descriptor::KeyLine k;k.startPointX=k.sPointInOctaveX=f[0];k.startPointY=k.sPointInOctaveY=f[1];k.endPointX=k.ePointInOctaveX=f[2];k.endPointY=k.ePointInOctaveY=f[3];k.lineLength=length;k.numOfPixels=int(std::max(std::abs(dx),std::abs(dy)))+1;k.angle=std::atan2(dy,dx);k.class_id=int(keys.size());k.octave=0;k.pt=cv::Point2f((f[0]+f[2])*.5f,(f[1]+f[3])*.5f);k.response=length/std::max(w,h);k.size=std::abs(dx*dy);keys.push_back(k);source_ids.push_back(i);
  }
  if(keys.empty())return 0;cv::Mat descriptors;auto descriptor=cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();descriptor->compute(gray,keys,descriptors);
  if(descriptors.empty())return 0;if(descriptors.cols!=32||descriptors.rows!=int(keys.size())||descriptors.type()!=CV_8UC1)return -2;
  for(int j=0;j<descriptors.rows;++j){int keyid=keys[j].class_id;int id=(keyid>=0&&keyid<int(source_ids.size()))?source_ids[keyid]:-1;if(id>=0&&id<count){std::memcpy(out+32*id,descriptors.ptr(j),32);valid[id]=1;}}
  return 0;
 }catch(...){return -3;}
}
