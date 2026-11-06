#!/usr/bin/env python3
import sys,tempfile,unittest,copy,csv,json
from pathlib import Path
import numpy as np
import yaml,cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from kalibr_to_plvins import convert
from export_sensor_bag import export
from audit_sensor_input import audit
from rosbags.rosbag2 import Writer
from rosbags.rosbag2.enums import StoragePlugin
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore,Stores

class Tools(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
  self.cam={'cam0':{'camera_model':'pinhole','distortion_model':'radtan','intrinsics':[700.,710.,640.,400.],'distortion_coeffs':[.01,-.02,.001,-.001],'resolution':[1280,800],'T_imu_cam':[[0.,0.,1.,.1],[-1.,0.,0.,.0375],[0.,-1.,0.,-.02],[0.,0.,0.,1.]]}}
  self.imu={'imu0':{'update_rate':250.,'accelerometer_noise_density':.001,'gyroscope_noise_density':.0001,'accelerometer_random_walk':1e-5,'gyroscope_random_walk':1e-6,'T_i_b':np.eye(4).tolist()}}
  self.cp=self.root/'cam.yaml';self.ip=self.root/'imu.yaml';self.write()
 def tearDown(self):self.temp.cleanup()
 def write(self):
  self.cp.write_text('%YAML:1.0\n'+yaml.safe_dump(self.cam));self.ip.write_text(yaml.safe_dump(self.imu))
 def test_transform_and_noise(self):
  text,meta=convert(self.cp,self.ip);out=self.root/'plvins.yaml';out.write_text(text)
  fs=cv2.FileStorage(str(out),cv2.FILE_STORAGE_READ)
  T=np.asarray(self.cam['cam0']['T_imu_cam']);np.testing.assert_allclose(fs.getNode('extrinsicRotation').mat(),T[:3,:3]);np.testing.assert_allclose(fs.getNode('extrinsicTranslation').mat().ravel(),T[:3,3]);self.assertAlmostEqual(fs.getNode('acc_n').real(),.001*np.sqrt(250));self.assertEqual(fs.getNode('estimate_extrinsic').real(),0);self.assertEqual(fs.getNode('freq').real(),0);self.assertEqual(fs.getNode('loop_closure').real(),0);fs.release()
  self.assertIn('never inverted',meta['T_convention'])
 def test_reject_invalid_and_unsupported_calibrations(self):
  original=copy.deepcopy(self.cam)
  for edit in [('camera_model','omni'),('distortion_model','equidistant'),('timeshift_cam_imu',.01),('intrinsics',[0,10,10,10]),('T_imu_cam',np.zeros((4,4)).tolist())]:
   self.cam=copy.deepcopy(original);self.cam['cam0'][edit[0]]=edit[1];self.write()
   with self.assertRaises(ValueError):convert(self.cp,self.ip)
 def test_nonidentity_imu_calibration_rejected(self):
  self.imu['imu0']['Tw']=(2*np.eye(3)).tolist();self.write()
  with self.assertRaises(ValueError):convert(self.cp,self.ip)
 def test_explicit_legacy_mode(self):
  _,m=convert(self.cp,self.ip,noise_mode='legacy-density');self.assertEqual(m['effective_noise']['acc_n'],.001)
 def make_bag(self,duplicate=False):
  p=self.root/'ros2';ts=get_typestore(Stores.ROS2_JAZZY);T=ts.types
  def header(ns):return T['std_msgs/msg/Header'](T['builtin_interfaces/msg/Time'](ns//10**9,ns%10**9),'measured_sensor')
  with Writer(p,version=9,storage_plugin=StoragePlugin.MCAP) as w:
   image=w.add_connection('/vio/left/image_raw','sensor_msgs/msg/Image',typestore=ts);imu=w.add_connection('/vio/imu','sensor_msgs/msg/Imu',typestore=ts);forbidden=w.add_connection('/truth','nav_msgs/msg/Odometry',typestore=ts)
   # Invalid forbidden payload proves selected deserialization never reads it.
   w.write(forbidden,10**9,b'not a valid odometry payload')
   for i in range(71):
    ns=10**9+i*4000000
    m=T['sensor_msgs/msg/Imu'](header(ns),T['geometry_msgs/msg/Quaternion'](.2,.3,.4,.5),np.zeros(9),T['geometry_msgs/msg/Vector3'](.1,.2,.3),np.arange(9,dtype=float),T['geometry_msgs/msg/Vector3'](1.,2.,9.81),np.arange(9,dtype=float))
    w.write(imu,ns,ts.serialize_cdr(m,m.__msgtype__))
   for i in [2,0,1,3,4,5]+([5] if duplicate else []):
    ns=10**9+i*40000000;pixels=np.arange(4*8,dtype=np.uint8)
    m=T['sensor_msgs/msg/Image'](header(ns),4,6,'mono8',0,8,pixels)
    w.write(image,1400000000+i,ts.serialize_cdr(m,m.__msgtype__))
  meta=yaml.safe_load((p/'metadata.yaml').read_text())['rosbag2_bagfile_information']
  self.assertEqual(meta['version'],9);self.assertEqual(meta['storage_identifier'],'mcap')
  self.assertTrue(any(p.glob('*.mcap')))
  return p
 def test_ros2_to_allowlisted_ros1_and_euroc(self):
  src=self.make_bag();out=self.root/'export';report=export(src,out,'/vio/left/image_raw','/vio/imu',True,True)
  self.assertEqual(report['counts'],{'image':6,'imu':71});self.assertAlmostEqual(report['gaps_s']['image']['max'],.04)
  with AnyReader([out/'sensors.bag']) as reader:
   self.assertEqual({c.topic for c in reader.connections},{'/cam0/image_raw','/imu0'});stamps=[]
   for c,t,raw in reader.messages():
    m=reader.deserialize(raw,c.msgtype);stamps.append(t);self.assertEqual(t,m.header.stamp.sec*10**9+m.header.stamp.nanosec)
    if c.topic=='/imu0':
     self.assertEqual(m.orientation_covariance[0],-1);self.assertEqual(m.angular_velocity.x,.1);self.assertEqual(m.linear_acceleration.z,9.81)
    else:
     self.assertEqual(m.step,6);np.testing.assert_array_equal(m.data.reshape(4,6),np.arange(32,dtype=np.uint8).reshape(4,8)[:,:6])
   self.assertEqual(stamps,sorted(stamps))
  png=out/'mav0/cam0/data/1000000000.png';self.assertEqual(cv2.imread(str(png),0).shape,(4,6))
  with self.assertRaises(ValueError):export(src,out,'/vio/left/image_raw','/vio/imu')
 def test_duplicate_timestamps_rejected(self):
  src=self.make_bag(True)
  with self.assertRaises(ValueError):export(src,self.root/'out','/vio/left/image_raw','/vio/imu')
 def test_input_audit_bracketing_units_and_allowlist(self):
  src=self.make_bag();r=audit(src,self.root/'audit','/vio/left/image_raw','/vio/imu')
  self.assertEqual(r['counts'],{'image':6,'imu':71})
  self.assertEqual(r['strict_imu_bracketing']['images_without_earlier_imu'],1)
  self.assertEqual(r['strict_imu_bracketing']['images_without_later_imu'],0)
  self.assertAlmostEqual(r['sorted_header_interval_s']['image']['median'],.04)
  self.assertEqual(r['invalid_messages'],{'image':0,'imu':0})
  self.assertEqual(r['image_resolutions'],[(6,4)])
if __name__=='__main__':unittest.main(verbosity=2)
