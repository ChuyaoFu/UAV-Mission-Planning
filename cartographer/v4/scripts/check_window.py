import json,sys
from pathlib import Path
import cv2,numpy as np,rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation
from window_localization import WindowLocalization
label=sys.argv[1];root=Path('/work/cartographer/v4/results')/label;f=json.loads((root/'flight.json').read_text());trace=f['trace'];w=WindowLocalization();r=rosbag2_py.SequentialReader();r.open(rosbag2_py.StorageOptions(uri='/work/cartographer/v4/data/'+label,storage_id='mcap'),rosbag2_py.ConverterOptions('',''));r.set_filter(rosbag2_py.StorageFilter(topics=['/vio/left/image_raw']));last=-1
while r.has_next():
 _,data,_=r.read_next();m=deserialize_message(data,Image);t=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
 if t<118 or t>130 or t-last<1.:continue
 last=t;a=min(trace,key=lambda a:abs(a['stamp']-t));p=np.array(a['position']);rotation=Rotation.from_euler('z',a['yaw']);w.update(m,p,rotation);print(round(t,2),w.health,flush=True)
 if w.debug:
  gray,after,before=w.debug;image=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
  for points,color in [(before,(0,0,255)),(after,(0,255,0))]:
   for x,y in points.astype(int):cv2.circle(image,(x,y),1,color,-1)
  cv2.imwrite(str(root/f'window-fit-{t:.1f}.png'),image)
