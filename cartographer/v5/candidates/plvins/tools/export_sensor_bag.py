#!/usr/bin/env python3
"""Allowlisted ROS1/ROS2 bag -> sensor-only ROS1 bag and/or EuRoC directory.
Uses exposure/IMU header stamps, never odometry, orientation truth, depth or CAD.
"""
import argparse,csv,json,sqlite3,tempfile,math,sys
from importlib.metadata import version
if sys.version_info < (3,10):
 raise RuntimeError("Jazzy version9/MCAP exporter requires Python>=3.10 and rosbags==0.10.11")
if tuple(map(int,version("rosbags").split(".")[:3])) < (0,10,3):
 raise RuntimeError("rosbags>=0.10.3 required for rosbag2 version9; tested pin is0.10.11. Never edit metadata version to bypass this check")
from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.rosbag1 import Writer
from rosbags.typesys import get_typestore,Stores

def stamp_ns(msg):return int(msg.header.stamp.sec)*1000000000+int(msg.header.stamp.nanosec)
def export(source,output,image_topic,imu_topic,euroc=False,ros1=True,reader_backend="rosbags"):
 output=Path(output)
 if output.exists() and any(output.iterdir()):raise ValueError('Output must be empty; refusing to overwrite an earlier replay')
 output.mkdir(parents=True,exist_ok=True)
 ts=get_typestore(Stores.ROS1_NOETIC);types=ts.types
 Time=types['builtin_interfaces/msg/Time'];Header=types['std_msgs/msg/Header'];Image=types['sensor_msgs/msg/Image'];Imu=types['sensor_msgs/msg/Imu'];Q=types['geometry_msgs/msg/Quaternion'];V=types['geometry_msgs/msg/Vector3']
 counts={'image':0,'imu':0};gaps={'image':[],'imu':[]};last={};width_height=None
 reader_type=AnyReader
 if reader_backend=='native':
  from native_ros2_reader import NativeReader
  reader_type=NativeReader
 elif reader_backend!='rosbags':raise ValueError('Unknown reader backend')
 with tempfile.TemporaryDirectory(prefix='plvins-sensor-',dir=str(output)) as tmp:
  conn=sqlite3.connect(str(Path(tmp)/'spool.sqlite'));conn.execute('CREATE TABLE sensors(stamp INTEGER, stream INTEGER, payload BLOB, UNIQUE(stamp,stream))')
  with reader_type([Path(source)],default_typestore=get_typestore(Stores.ROS2_JAZZY)) as reader:
   selected=[c for c in reader.connections if c.topic in [image_topic,imu_topic]]
   if {c.topic for c in selected}!={image_topic,imu_topic}:raise ValueError('Both explicit sensor topics must exist in the bag')
   for c in selected:
    expected='sensor_msgs/msg/Image' if c.topic==image_topic else 'sensor_msgs/msg/Imu'
    if c.msgtype!=expected:raise ValueError('Selected topic has wrong type: '+c.topic)
   for connection,record_stamp,raw in reader.messages(connections=selected):
    m=reader.deserialize(raw,connection.msgtype);stamp=stamp_ns(m)
    if stamp<=0:raise ValueError('Nonpositive measured header stamp; will not fabricate timestamps')
    h=Header(0,Time(stamp//1000000000,stamp%1000000000),m.header.frame_id)
    if connection.topic==image_topic:
     if m.encoding!='mono8' or m.width<=0 or m.height<=0 or m.step<m.width or len(m.data)<m.step*m.height:raise ValueError('Expected valid mono8 image; convert calibrated color input explicitly upstream')
     shape=(int(m.width),int(m.height))
     if width_height is not None and shape!=width_height:raise ValueError('Image dimensions changed mid-sequence')
     width_height=shape
     # Remove padding only, preserving every measured pixel.
     pixels=np.asarray(m.data,dtype=np.uint8)[:m.step*m.height].reshape(m.height,m.step)[:,:m.width].copy()
     sensor=Image(h,m.height,m.width,'mono8',0,m.width,pixels.reshape(-1));stream=0
    else:
     w=[m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z];a=[m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z]
     if not all(math.isfinite(float(x)) for x in w+a):raise ValueError('Nonfinite measured IMU')
     # Orientation is not consumed by PL-VINS. Strip it, mark unavailable.
     cov=np.zeros(9);cov[0]=-1
     sensor=Imu(h,Q(0.,0.,0.,1.),cov,V(*map(float,w)),np.asarray(m.angular_velocity_covariance,float),V(*map(float,a)),np.asarray(m.linear_acceleration_covariance,float));stream=1
    payload=bytes(ts.serialize_ros1(sensor,sensor.__msgtype__))
    try:conn.execute('INSERT INTO sensors VALUES(?,?,?)',(stamp,stream,payload))
    except sqlite3.IntegrityError:raise ValueError('Duplicate exposure/IMU header stamp; resolve recording duplicates explicitly')
   conn.commit()
  if euroc:
   import cv2
   camdir=output/'mav0/cam0';imudir=output/'mav0/imu0';(camdir/'data').mkdir(parents=True);imudir.mkdir(parents=True)
   cf=(camdir/'data.csv').open('w');inf=(imudir/'data.csv').open('w');cw=csv.writer(cf);iw=csv.writer(inf)
   cw.writerow(['#timestamp [ns]','filename']);iw.writerow(['#timestamp [ns]','w_RS_S_x [rad s^-1]','w_RS_S_y [rad s^-1]','w_RS_S_z [rad s^-1]','a_RS_S_x [m s^-2]','a_RS_S_y [m s^-2]','a_RS_S_z [m s^-2]'])
  writer=Writer(output/'sensors.bag') if ros1 else None
  try:
   if writer:
    writer.open();connections=[writer.add_connection('/cam0/image_raw','sensor_msgs/msg/Image',typestore=ts),writer.add_connection('/imu0','sensor_msgs/msg/Imu',typestore=ts)]
   for stamp,stream,payload in conn.execute('SELECT stamp,stream,payload FROM sensors ORDER BY stamp,stream DESC'):
    kind='image' if stream==0 else 'imu';counts[kind]+=1
    if kind in last:gaps[kind].append((stamp-last[kind])*1e-9)
    last[kind]=stamp
    if writer:writer.write(connections[stream],stamp,payload)
    if euroc:
     m=ts.deserialize_ros1(payload,'sensor_msgs/msg/Image' if stream==0 else 'sensor_msgs/msg/Imu')
     if stream==0:
      name=str(stamp)+'.png';pixels=m.data.reshape(m.height,m.width)
      if not cv2.imwrite(str(camdir/'data'/name),pixels):raise RuntimeError('Image write failed')
      cw.writerow([str(stamp),name])
     else:iw.writerow([str(stamp),m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z,m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z])
  finally:
   if writer:writer.close()
   if euroc:cf.close();inf.close()
  conn.close()
 if min(counts.values())==0:raise ValueError('No valid measured image/IMU stream')
 report={'source':str(source),'rosbags_version':version('rosbags'),'python_version':sys.version.split()[0],'reader_backend':reader_backend,'allowlisted_topics':[image_topic,imu_topic],'output_topics':['/cam0/image_raw','/imu0'],'timestamp':'original measured header stamp; bag sorted on disk; no interpolation or resampling','orientation':'removed, covariance[0]=-1; measured gyro/acceleration preserved','image_resolution':width_height,'counts':counts,'gaps_s':{k:{'median':float(np.median(v)) if v else None,'p95':float(np.percentile(v,95)) if v else None,'max':max(v) if v else None} for k,v in gaps.items()},'no_ground_truth':True,'units_contract':'source must supply calibrated SI specific force (m/s^2) and angular rate (rad/s), matching calibration IMU frame; no gravity subtraction or implicit axis transform'}
 (output/'export-report.json').write_text(json.dumps(report,indent=2)+'\n');return report

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--image-topic',default='/vio/left/image_raw');p.add_argument('--imu-topic',default='/vio/imu');p.add_argument('--euroc',action='store_true');p.add_argument('--no-ros1',action='store_true');p.add_argument('--reader',choices=['rosbags','native'],default='rosbags');a=p.parse_args()
 if a.no_ros1 and not a.euroc:p.error('Select ROS1 bag and/or EuRoC output')
 print(json.dumps(export(a.source,a.output,a.image_topic,a.imu_topic,a.euroc,not a.no_ros1,a.reader),indent=2))
if __name__=='__main__':main()
