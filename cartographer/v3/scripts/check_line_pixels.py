#!/usr/bin/python3
import sys,json
from pathlib import Path
import cv2,numpy as np,rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
label=sys.argv[1];r=rosbag2_py.SequentialReader();r.open(rosbag2_py.StorageOptions(uri='/work/cartographer/v3/data/'+label,storage_id='mcap'),rosbag2_py.ConverterOptions('',''));r.set_filter(rosbag2_py.StorageFilter(topics=['/vio/left/image_raw']));b=CvBridge();last=-1;rows=[]
while r.has_next():
 _,data,_=r.read_next();m=deserialize_message(data,Image);ts=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
 if ts-last<2:continue
 last=ts;image=b.imgmsg_to_cv2(m,'rgb8');f=image.astype(float);hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV);old=(f[:,:,0]>35)&(f[:,:,0]>2*f[:,:,1])&(f[:,:,0]>2*f[:,:,2]);new=((hsv[:,:,0]<8)|(hsv[:,:,0]>175))&(hsv[:,:,1]>70)&(hsv[:,:,2]>35);rows.append({'stamp':ts,'old_pixels':int(old.sum()),'hsv_pixels':int(new.sum())})
 if len(rows)==10:
  out=Path('/work/cartographer/v3/results')/label;cv2.imwrite(str(out/'rgb-line-check.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR));cv2.imwrite(str(out/'line-mask-check.png'),new.astype(np.uint8)*255)
print(json.dumps(rows,indent=2));(Path('/work/cartographer/v3/results')/label/'line-pixel-statistics.json').write_text(json.dumps(rows,indent=2))
