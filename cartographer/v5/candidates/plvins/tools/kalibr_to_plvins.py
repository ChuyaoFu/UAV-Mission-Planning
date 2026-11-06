#!/usr/bin/env python3
"""Convert supplied measured mono-camera/IMU calibration; never invent extrinsics."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import yaml

def load_yaml(path):
 text=Path(path).read_text();return yaml.safe_load('\n'.join(x for x in text.splitlines() if not x.startswith('%YAML')))
def matrix_text(name, value):
 a=np.asarray(value,float);return '%s: !!opencv-matrix\n  rows: %d\n  cols: %d\n  dt: d\n  data: [%s]\n'%(name,a.shape[0],a.shape[1],', '.join(format(v,'.17e') for v in a.flat))
def convert(camera_file,imu_file,camera='cam0',image_topic='/cam0/image_raw',imu_topic='/imu0',g_norm=9.81,noise_mode='discrete',rate=None):
 cam=load_yaml(camera_file)[camera];imu=load_yaml(imu_file)['imu0']
 if cam['camera_model']!='pinhole' or cam['distortion_model']!='radtan':raise ValueError('Only pinhole+radtan supported; do not silently reinterpret another model')
 K=np.asarray(cam['intrinsics'],float);d=np.asarray(cam['distortion_coeffs'],float);T=np.asarray(cam['T_imu_cam'],float)
 if K.shape!=(4,) or d.shape!=(4,) or T.shape!=(4,4) or not all(np.isfinite(x).all() for x in [K,d,T]):raise ValueError('Invalid K/distortion/T')
 if min(K[:2])<=0 or not np.allclose(T[3],[0,0,0,1],atol=1e-9) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(T[:3,:3]),1,atol=1e-6):raise ValueError('Invalid rigid camera-to-IMU transform')
 for name in ['T_i_b','Tw','Ta','R_IMUtoGYRO','R_IMUtoACC']:
  if name in imu:
   a=np.asarray(imu[name],float)
   if a.ndim!=2 or a.shape[0]!=a.shape[1] or not np.allclose(a,np.eye(a.shape[0]),atol=1e-9):raise ValueError(name+' is nonidentity: first calibrate/rotate the measured IMU stream, do not silently drop this calibration')
 if 'Tg' in imu and not np.allclose(imu['Tg'],0,atol=1e-12):raise ValueError('Nonzero IMU g sensitivity needs calibrated input')
 if any(abs(float(x))>1e-12 for x in [cam.get('timeshift_cam_imu',0),imu.get('time_offset',0)]):raise ValueError('PL-VINS has no time-offset calibration here. Align measured exposure/IMU timestamps explicitly before conversion; do not silently discard nonzero offset')
 hz=float(rate if rate is not None else imu['update_rate'])
 if not math.isfinite(hz) or hz<=0:raise ValueError('Positive measured IMU rate required')
 mapping={'acc_n':'accelerometer_noise_density','gyr_n':'gyroscope_noise_density','acc_w':'accelerometer_random_walk','gyr_w':'gyroscope_random_walk'}
 factor=math.sqrt(hz) if noise_mode=='discrete' else 1.
 noises={key:float(imu[name])*factor for key,name in mapping.items()}
 if not all(math.isfinite(v) and v>0 for v in noises.values()):raise ValueError('Positive finite IMU noise required')
 width,height=map(int,cam['resolution'])
 if min(width,height)<=0:raise ValueError('Invalid dimensions')
 fields={'imu_topic':imu_topic,'image_topic':image_topic,'output_path':'/plvins_result.txt','model_type':'PINHOLE','camera_name':camera,'image_width':width,'image_height':height,'distortion_parameters':dict(zip(['k1','k2','p1','p2'],map(float,d))),'projection_parameters':dict(zip(['fx','fy','cx','cy'],map(float,K))),'estimate_extrinsic':0,'ex_calib_result_path':'/extrinsic-unused.yaml','max_cnt':150,'min_dist':30,'freq':0,'F_threshold':1.0,'show_track':0,'equalize':1,'fisheye':0,'max_solver_time':.04,'max_num_iterations':8,'keyframe_parallax':10.,**noises,'g_norm':float(g_norm),'loop_closure':0,'pattern_file':'/unused','voc_file':'/unused','min_loop_num':25,'base_line':0.}
 text='%YAML:1.0\n---\n'+yaml.safe_dump(fields,sort_keys=False)+matrix_text('extrinsicRotation',T[:3,:3])+matrix_text('extrinsicTranslation',T[:3,3:4])
 meta={'camera_file':str(camera_file),'camera_sha256':hashlib.sha256(Path(camera_file).read_bytes()).hexdigest(),'imu_file':str(imu_file),'imu_sha256':hashlib.sha256(Path(imu_file).read_bytes()).hexdigest(),'camera_key':camera,'T_convention':'T_imu_cam maps camera points to the measured IMU frame; copied directly, never inverted','noise_mode':noise_mode,'imu_update_rate_hz':hz,'noise_multiplier':factor,'noise_mapping':mapping,'effective_noise':noises,'noise_note':'IntegrationBase squares these parameters and multiplies noise Jacobians by dt. Discrete mode converts density/random-walk to per-sample std using sqrt(rate). Original midpoint covariance ignores adjacent-sample correlation; not an exact continuous-noise covariance discretization. legacy-density copies values for a separately labelled sensitivity replay.','timestamp_offset_s':0,'input_contract':'mono8 calibrated left camera; measured SI IMU specific force and angular velocity in matching IMU frame; no gravity subtraction','optimization':'fixed measured extrinsics, original point and line gates, no loop closure/GPS/wheel/ToF/depth inputs'}
 return text,meta

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--camera-file',required=True,type=Path);p.add_argument('--imu-file',required=True,type=Path);p.add_argument('--camera',default='cam0');p.add_argument('--output',required=True,type=Path);p.add_argument('--image-topic',default='/cam0/image_raw');p.add_argument('--imu-topic',default='/imu0');p.add_argument('--g-norm',type=float,default=9.81);p.add_argument('--imu-rate',type=float);p.add_argument('--noise-mode',choices=['discrete','legacy-density'],default='discrete');a=p.parse_args()
 text,meta=convert(a.camera_file,a.imu_file,a.camera,a.image_topic,a.imu_topic,a.g_norm,a.noise_mode,a.imu_rate);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text);a.output.with_suffix('.conversion.json').write_text(json.dumps(meta,indent=2)+'\n');print(a.output)
if __name__=='__main__':main()
