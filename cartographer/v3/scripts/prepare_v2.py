#!/usr/bin/python3
from pathlib import Path
import math,copy,json
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
ROOT=Path('/work/cartographer/v3');old=Path('/work/cartographer');pitch_degrees=json.loads((ROOT/'config/model-manifest.json').read_text())['camera_pitch_down_degrees']
# Camera optical-to-IMU rotation includes the physical pitch, IMU stays body-aligned.
f=ROOT/'config/gazebo_vio/kalibr_imucam_chain.yaml';d=yaml.safe_load('\n'.join(f.read_text().splitlines()[1:]));opt=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
r=Rotation.from_euler('y',pitch_degrees,degrees=True).as_matrix()@opt
for i,key in enumerate(['cam0','cam1']):
 t=np.eye(4);t[:3,:3]=r;t[:3,3]=[0,.0375 if i==0 else -.0375,0];d[key]['T_imu_cam']=t.tolist()
lines=['%YAML:1.0','']
for key,camera in d.items():
 lines.append(key+':');lines.append('  T_imu_cam:')
 for row in camera['T_imu_cam']:lines.append('    - '+json.dumps(row))
 for k,v in camera.items():
  if k=='T_imu_cam':continue
  lines.append('  '+k+': '+(json.dumps(v) if isinstance(v,list) else str(v)))
f.write_text('\n'.join(lines)+'\n')
f=ROOT/'config/gazebo_vio/estimator_config.yaml';s=f.read_text().replace('track_frequency: 20.0','track_frequency: 25.0').replace('fast_threshold: 20','fast_threshold: 10').replace('min_px_dist: 10','min_px_dist: 8').replace('num_pts: 200','num_pts: 250').replace('init_max_disparity: 10.0','init_max_disparity: 3.0').replace('init_max_disparity: 0.0','init_max_disparity: 3.0');f.write_text(s)
# Preserve the previous branch, use local v2 scripts/config/logs.
s=(old/'scripts/start_sim.sh').read_text().replace('/work/cartographer/config/carto-model.sdf','/work/cartographer/v3/config/model.sdf').replace('/work/logs/','/work/cartographer/v3/logs/').replace('PX4_PARAM_EKF2_EV_POS_X=0.15','PX4_PARAM_EKF2_EV_POS_X=0.12').replace('PX4_PARAM_EKF2_EV_POS_Z=-0.08','PX4_PARAM_EKF2_EV_POS_Z=-0.055').replace('z: 0.1','z: 0.14')
a=s.index("if [ \"${NAV_WORLD:-dronehacks}\" = 'dronehacks_textured' ]");b=s.index('\nfi',a)+4;s=s[:a]+s[b:];s=s.replace('cd $PX4/build/px4_sitl_default/rootfs &&','cd /work/cartographer/v3/px4-rootfs &&').replace(' ../bin/px4 >',' $PX4/build/px4_sitl_default/bin/px4 -w /work/cartographer/v3/px4-rootfs >')
s=s.replace('exec gz sim -v 4', 'source /opt/ros/jazzy/setup.bash; GZ_CONFIG_PATH=/usr/share/gz GZ_PARTITION=visual_nav_sim exec /usr/bin/gz sim -v 4').replace("'ros2 run ros_gz_bridge","'source /opt/ros/jazzy/setup.bash; ros2 run ros_gz_bridge")
(ROOT/'scripts/start_sim.sh').write_text(s)
s=(old/'scripts/depth_cloud.py').read_text().replace("p=np.stack((z,-(u-(m.width-1)/2)*z/f,-(v-(m.height-1)/2)*z/f),-1)","p=np.stack((z,-(u-320)*z/f,-(v-200)*z/f),-1)\n  angle=np.deg2rad(PITCHVALUE);rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])\n  p=p@rotation.T")
s=s.replace('PITCHVALUE',str(pitch_degrees))
s=s.replace('p=p@rotation.T','p=p@rotation.T\n  if os.environ.get(\"DEPTH_SOURCE\")==\"opencv_passive_stereo\":p[...,1]+=.0375')
(ROOT/'scripts/depth_cloud.py').write_text(s)
s=(old/'scripts/down_range.py').read_text();(ROOT/'scripts/down_range.py').write_text(s)
s=(old/'scripts/stereo_depth.py').read_text();(ROOT/'scripts/stereo_depth.py').write_text(s)
s=(old/'config/oak_3d.lua').read_text().replace('use_odometry = false','use_odometry = true').replace('use_pose_extrapolator = true,','use_pose_extrapolator = true, publish_tracked_pose = true,')
s=s.replace('TRAJECTORY_BUILDER_3D.ceres_scan_matcher.translation_weight = 10.','TRAJECTORY_BUILDER_3D.ceres_scan_matcher.translation_weight = 20.')
(ROOT/'config/oak_3d.lua').write_text(s)
print('v2 camera extrinsics and frame conversions generated')
