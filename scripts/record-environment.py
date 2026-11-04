#!/usr/bin/python3
"""Collect a reproducible environment manifest on the Ubuntu host (sudo)."""
from pathlib import Path
import datetime
import hashlib
import json
import platform
import subprocess

root=Path(__file__).resolve().parents[1]
def output(args):return subprocess.check_output(args,text=True).strip()
def docker(args):return output(['/usr/bin/docker','exec','visual-nav-sim',*args])
repos={}
for name,rel in [('px4','src/PX4-Autopilot'),('px4_msgs','ros_ws/src/px4_msgs'),('openvins','ros_ws/src/open_vins'),('micro_xrce_agent','src/Micro-XRCE-DDS-Agent')]:
    repos[name]=output(['git','-c','safe.directory=*','-C',str(root/rel),'rev-parse','HEAD'])
keys=['EKF2_GPS_CTRL','EKF2_EV_CTRL','EKF2_MAG_TYPE','EKF2_HGT_REF','EKF2_EV_POS_X','EKF2_EV_POS_Y','EKF2_EV_POS_Z','MPC_LAND_SPEED','UXRCE_DDS_SYNCT']
bin='/work/src/PX4-Autopilot/build/px4_sitl_default/bin/'
params={k:float(docker([bin+'px4-param','show','-q',k])) for k in keys}
gnss=docker([bin+'px4-listener','sensor_gps','-n','1'])
(root/'logs/no-gnss-input.log').write_text(gnss+'\n')
flags=docker([bin+'px4-listener','estimator_status_flags','-n','1'])
(root/'logs/vision-estimator-state.log').write_text(flags+'\n')
manifest={
    'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'host':{'architecture':platform.machine(),'os_release':Path('/etc/os-release').read_text(),'gpu':output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'])},
    'container':json.loads(output(['/usr/bin/docker','inspect','visual-nav-sim','--format','{{json .Config.Image}}'])),
    'image_id':output(['/usr/bin/docker','inspect','visual-nav-sim','--format','{{.Image}}']),
    'repos':repos,'runtime_parameters':params,
    'gnss_input_never_published':'never published' in gnss,
    'course_obj_sha256':hashlib.sha256((root/'course_source/dronehacks-course-v3.obj').read_bytes()).hexdigest(),
    'openvins_patch_sha256':hashlib.sha256((root/'config/openvins-jazzy.patch').read_bytes()).hexdigest(),
    'ros_domain_id':42,'agent_udp_port':8889,
    'sensor_model':'generic 640x400 ideal pinhole stereo, 75mm baseline, body-aligned IMU; ideal rendered depth',
    'vehicle_model':'PX4 X500 derivative; not an identified official 7-inch kit',
    'material_profiles':['source flat colors','explicit synthetic regression texture'],
}
(root/'results/environment-manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
print('Recorded environment-manifest.json and actual EKF/GNSS input evidence.')
