#!/usr/bin/env python3
"""Configure a physically rigid OAK/IMU mount; no course inputs or sensor additions.
Simulation-fixture design only. Actual hardware must provide measured calibration.
The existing single ToF remains rigidly downward in the BODY frame.
"""
import argparse,json,math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--pitch-deg',type=float,required=True);p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'config');a=p.parse_args();assert math.isfinite(a.pitch_deg) and -10<=a.pitch_deg<=30
folder=a.config;robot=json.loads((folder/'robot.json').read_text());navigation=json.loads((folder/'navigation.json').read_text());model=ET.parse(folder/'model.sdf');rig=model.find(".//link[@name='vio_rig']");mount=Rotation.from_euler('y',a.pitch_deg,degrees=True);position=np.asarray(robot['imu_in_body_m']);rig.find('pose').text=' '.join(map(str,[*position,0.,math.radians(a.pitch_deg),0.]));robot['imu_in_body_quaternion_xyzw']=mount.as_quat().tolist();tof=model.find(".//sensor[@name='down_tof']/pose");translation=mount.inv().apply(np.asarray(robot['tof_in_body_m'])-position);rotation=np.array([0.,math.pi/2-math.radians(a.pitch_deg),0.]);tof.text=' '.join(map(str,[*translation,*rotation]));robot['profile']=f'Proposed seven-inch SITL fixture: rigid OAK/IMU {a.pitch_deg:g}-degree downward pitch; actual hardware requires measured extrinsics.';navigation['simulation_mount_pitch_degrees']=a.pitch_deg;navigation['camera_mount_note']=f'Same stereo+IMU and body-fixed downward ToF. Proposed {a.pitch_deg:g}-degree OAK/IMU mounting, not an identified hardware installation. No new sensor, texture or ideal depth.';ET.indent(model);model.write(folder/'model.sdf',encoding='utf-8',xml_declaration=True);(folder/'robot.json').write_text(json.dumps(robot,indent=2)+'\n');(folder/'navigation.json').write_text(json.dumps(navigation,indent=2)+'\n');print(json.dumps({'pitch_down_degrees':a.pitch_deg,'rig_position_in_body_m':position.tolist(),'tof_kept_body_fixed':True,'environment_modified':False,'hardware_calibration_verified':False},indent=2))
