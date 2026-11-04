#!/usr/bin/python3
import hashlib,json,os,platform,subprocess
from pathlib import Path
root=Path('/work/cartographer');px=Path('/work/src/PX4-Autopilot/build/px4_sitl_default/bin')
keys=['EKF2_GPS_CTRL','EKF2_EV_CTRL','EKF2_RNG_CTRL','EKF2_HGT_REF','EKF2_EV_POS_X','EKF2_EV_POS_Y','EKF2_EV_POS_Z','EKF2_RNG_POS_Z']
params={k:float(subprocess.check_output([str(px/'px4-param'),'show','-q',k],text=True)) for k in keys}
gps=subprocess.check_output([str(px/'px4-listener'),'sensor_gps','-n','1'],text=True).strip()
r={'packages':subprocess.check_output(['dpkg-query','-W','ros-jazzy-cartographer','ros-jazzy-cartographer-ros'],text=True).splitlines(),'parameters':params,'gnss_sensor':gps,'gnss_input_never_published':gps=='never published','ros_domain_id':os.getenv('ROS_DOMAIN_ID'),'rmw':os.getenv('RMW_IMPLEMENTATION'),'gazebo_partition':'visual_nav_sim','upstream_ros2_repository':'https://github.com/ros2/cartographer_ros/tree/ros2','files_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for d in ['scripts','config'] for f in (root/d).glob('*') if f.is_file()},'scope':'simulation prototype; not a certified event submission; not an Orin performance validation'}
(root/'results/runtime-audit.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='files_sha256'},indent=2))
