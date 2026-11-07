#!/usr/bin/python3
"""Original recorded passive left camera -> timestamped video, no synthesis."""
import argparse,json,subprocess
from pathlib import Path
import cv2,numpy as np,rosbag2_py
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rclpy.serialization import deserialize_message
ap=argparse.ArgumentParser();ap.add_argument('bag');ap.add_argument('output');a=ap.parse_args();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);r=rosbag2_py.SequentialReader();r.open(rosbag2_py.StorageOptions(uri=a.bag,storage_id='mcap'),rosbag2_py.ConverterOptions('',''));r.set_filter(rosbag2_py.StorageFilter(topics=['/vio/left/image_raw','/carto/mission_state']));p=None;last=None;start=None;next_frame=None;frames=0;phase='SENSOR CHECK'
while r.has_next():
 topic,data,_=r.read_next()
 if topic.endswith('mission_state'):phase=deserialize_message(data,String).data;continue
 m=deserialize_message(data,Image);t=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
 if m.encoding=='mono8':image=cv2.cvtColor(np.ndarray((m.height,m.width),np.uint8,buffer=m.data,strides=(m.step,1)),cv2.COLOR_GRAY2RGB)
 else:
  image=np.ndarray((m.height,m.width,3),np.uint8,buffer=m.data,strides=(m.step,3,1)).copy()
  if m.encoding=='bgr8':image=image[:,:,::-1].copy()
 cv2.rectangle(image,(0,0),(m.width,47),(15,20,30),-1);cv2.putText(image,f'PASSIVE LEFT | sim {t:.3f}s | {phase}',(8,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(240,245,250),1,cv2.LINE_AA);cv2.putText(image,Path(a.bag).name+' | original camera recording, no truth overlay',(8,39),cv2.FONT_HERSHEY_SIMPLEX,.40,(190,225,230),1,cv2.LINE_AA)
 if p is None:
  p=subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{m.width}x{m.height}','-r','20','-i','-','-an','-c:v','libx264','-preset','fast','-crf','22','-threads','1','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE);start=t;next_frame=t
 while next_frame<t-.025 and last is not None:p.stdin.write(last.tobytes());frames+=1;next_frame+=.05
 if t>=next_frame-.025:p.stdin.write(np.ascontiguousarray(image).tobytes());frames+=1;next_frame+=.05
 last=image
if p is None:raise RuntimeError('No recorded camera frames')
p.stdin.close();assert p.wait(timeout=30)==0
metadata={'source_bag':a.bag,'source_topic':'/vio/left/image_raw','actual_original_frames':True,'fps':20,'frames':frames,'duration_s':frames/20,'sim_start_s':start,'sim_end_s':t,'timing':'resampled original image timestamps at 20 Hz; no acceleration','truth_overlay':False};out.with_suffix('.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata))
