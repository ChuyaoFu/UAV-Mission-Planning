#!/usr/bin/env python3
"""Set one explicit SITL mounting profile and matching rectified calibration.
This edits simulator files only. Hardware uses its measured calibration files.
"""
import argparse,json,math,xml.etree.ElementTree as ET
from pathlib import Path
import yaml
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('degrees',type=float);a=p.parse_args();assert 0<=a.degrees<=45;root=Path(__file__).resolve().parents[1];theta=math.radians(a.degrees)
p=root/'config/model.sdf';tree=ET.parse(p)
rig=tree.find(".//link[@name='vio_rig']/pose");rv=rig.text.split();rv[4]=str(theta);rig.text=' '.join(rv)
robot=json.loads((root/'config/robot.json').read_text());rv[:3]=[str(x) for x in robot['imu_in_body_m']];rig.text=' '.join(rv);robot['imu_in_body_quaternion_xyzw']=Rotation.from_euler('y',theta).as_quat().tolist();(root/'config/robot.json').write_text(json.dumps(robot,indent=2)+'\n')
for side in ['left','right','depth']:
 e=tree.find(".//sensor[@name='%s']/pose"%side)
 if e is not None:v=e.text.split();v[4]='0';e.text=' '.join(v)
tof=tree.find(".//sensor[@name='down_tof']/pose");t=Rotation.from_euler('y',-theta).apply(np.asarray(robot['tof_in_body_m'])-np.asarray(robot['imu_in_body_m']));tof.text=' '.join(map(str,[*t,0.,math.pi/2-theta,0.]))
ET.indent(tree);tree.write(p,encoding='utf-8',xml_declaration=True)
p=root/'config/gazebo_vio/kalibr_imucam_chain.yaml';d=yaml.safe_load(p.read_text().replace('%YAML:1.0',''));R=[[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]]
for cam in d.values():
 for i in range(3):cam['T_imu_cam'][i][:3]=R[i]
lines=['%YAML:1.0','']
for name,cam in d.items():
 lines.append(name+':')
 for key,value in cam.items():
  if key=='T_imu_cam':lines.append('  T_imu_cam:');lines.extend('    - '+str(row) for row in value)
  else:lines.append('  '+key+': '+str(value))
p.write_text('\n'.join(lines)+'\n')
for name in ['model-manifest.json','navigation.json']:
 p=root/'config'/name;d=json.loads(p.read_text());key='camera_pitch_down_degrees' if name.startswith('model-') else 'simulation_mount_pitch_degrees';d[key]=a.degrees;p.write_text(json.dumps(d,indent=2)+'\n')
print('SITL camera mounting profile:',a.degrees,'degrees; hardware mounting remains unverified')
