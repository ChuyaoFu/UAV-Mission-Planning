#!/usr/bin/env python3
"""Configure a physically rigid OAK/IMU mount; no course inputs or sensor additions.
Simulation-fixture design only. Actual hardware must provide measured calibration.
The existing single ToF remains rigidly downward in the BODY frame.
"""
import argparse,json,math,warnings
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--pitch-deg',type=float,required=True);p.add_argument('--roll-deg',type=float,default=0.,help='Physical rigid-board roll about its forward axis; images are not digitally rotated');p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'config');a=p.parse_args()
assert math.isfinite(a.pitch_deg) and -60<=a.pitch_deg<=60
assert math.isfinite(a.roll_deg) and -180<=a.roll_deg<=180
folder=a.config
robot=json.loads((folder/'robot.json').read_text())
navigation=json.loads((folder/'navigation.json').read_text())
model=ET.parse(folder/'model.sdf');rig=model.find(".//link[@name='vio_rig']")
angles=np.radians([a.roll_deg,a.pitch_deg,0.])
mount=Rotation.from_euler('xyz',angles)
position=np.asarray(robot['imu_in_body_m'])
rig.find('pose').text=' '.join(map(str,[*position,*angles]))
robot['imu_in_body_quaternion_xyzw']=mount.as_quat().tolist()
tof=model.find(".//sensor[@name='down_tof']/pose")
translation=mount.inv().apply(np.asarray(robot['tof_in_body_m'])-position)
with warnings.catch_warnings():
 warnings.filterwarnings('ignore',message='Gimbal lock detected.*')
 # Euler singularity changes only the representation, never the rotation.
 rotation=(mount.inv()*Rotation.from_euler('y',math.pi/2)).as_euler('xyz')
tof.text=' '.join(map(str,[*translation,*rotation]))
robot['profile']=f'Proposed seven-inch SITL fixture: rigid OAK/IMU pitch {a.pitch_deg:g}, roll {a.roll_deg:g} degrees; actual hardware requires measured extrinsics.'
navigation['simulation_mount_pitch_degrees']=a.pitch_deg
navigation['simulation_mount_roll_degrees']=a.roll_deg
navigation['camera_mount_note']=f'Same stereo/IMU, physical rigid pitch {a.pitch_deg:g} and roll {a.roll_deg:g} degrees, body-fixed downward ToF. No digital image rotation, new sensor, texture or ideal depth; hardware mount uncalibrated.'
ET.indent(model);model.write(folder/'model.sdf',encoding='utf-8',xml_declaration=True)
(folder/'robot.json').write_text(json.dumps(robot,indent=2)+'\n')
(folder/'navigation.json').write_text(json.dumps(navigation,indent=2)+'\n')
print(json.dumps({'pitch_down_degrees':a.pitch_deg,'roll_degrees':a.roll_deg,'rig_position_in_body_m':position.tolist(),'tof_kept_body_fixed':True,'environment_modified':False,'hardware_calibration_verified':False},indent=2))
