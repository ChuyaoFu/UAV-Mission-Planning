#!/usr/bin/python3
"""Read-only replay comparison for height drift and floor reacquisition."""
import sys,json
from pathlib import Path
import numpy as np,rosbag2_py
from scipy.spatial.transform import Rotation
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from height_odometry import HeightObserver
bag=Path(sys.argv[1]);output=Path(sys.argv[2]);e=json.load(open(sys.argv[3]));g=np.array(e['rows']['truth']);truth=g[:,1:4]+Rotation.from_quat(g[:,4:8]).apply([.12,0,.055]);f=HeightObserver();r=rosbag2_py.SequentialReader();r.open(rosbag2_py.StorageOptions(uri=str(bag),storage_id='mcap'),rosbag2_py.ConverterOptions('',''));wanted=['/ov_msckf/odomimu','/carto/down_range','/carto/points'];types={t.name:get_message(t.type) for t in r.get_all_topics_and_types() if t.name in wanted};r.set_filter(rosbag2_py.StorageFilter(topics=list(types)));rows=[];anchor=None
while r.has_next():
 topic,data,_=r.read_next();m=deserialize_message(data,types[topic]);t=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
 if topic.endswith('down_range'):f.ranging(m)
 elif topic.endswith('/points'):f.cloud(m)
 else:
  z=m.pose.pose.position.z;correction=f.odometry(m)
  if anchor is None:anchor=z
  reference=float(np.interp(t,g[:,0],truth[:,2]))
  rows.append([t,z-anchor+truth[0,2]-reference,z+correction-anchor+truth[0,2]-reference,correction,f.rejected,f.reacquired])
a=np.array(rows);health=e['rows']['health'];old_correction=np.interp(a[:,0],[h['stamp'] for h in health],[h.get('carto_height_correction_m',0.) for h in health]);old_error=a[:,1]+old_correction-old_correction[0];result={'previous_observer_z_rmse_m':float(np.sqrt((old_error**2).mean())),'previous_observer_z_final_m':float(old_error[-1]),**{'raw_vio_z_rmse_m':float(np.sqrt((a[:,1]**2).mean())),'corrected_z_rmse_m':float(np.sqrt((a[:,2]**2).mean())),'raw_vio_z_final_m':float(a[-1,1]),'corrected_z_final_m':float(a[-1,2]),'reacquired_updates':f.reacquired,'rejected_ranges':f.rejected,'method':'original raw VIO, passive stereo and single ToF, offline read-only; same initial vertical gauge; independent truth only after processing'}};output.write_text(json.dumps(result,indent=2));output.with_suffix('.samples.json').write_text(json.dumps(rows));print(json.dumps(result,indent=2))
