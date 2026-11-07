#!/usr/bin/env python3
"""Explicit supported OAK monochrome simulation resolution; no new sensor."""
import argparse,json,math,xml.etree.ElementTree as ET
from pathlib import Path
import yaml
p=argparse.ArgumentParser();p.add_argument('height',type=int,choices=[400,800]);a=p.parse_args();root=Path(__file__).resolve().parents[1];width=a.height*8//5;p=root/'config/model.sdf';tree=ET.parse(p)
for name in ['left','right']:
 camera=tree.find(".//sensor[@name='%s']/camera"%name);camera.find('image/width').text=str(width);camera.find('image/height').text=str(a.height);camera.find('image/format').text='L8'
unused=tree.find(".//sensor[@name='depth']")
if unused is not None:tree.find(".//link[@name='vio_rig']").remove(unused)
ET.indent(tree);tree.write(p,encoding='utf-8',xml_declaration=True)
fov=float(tree.findtext(".//sensor[@name='left']/camera/horizontal_fov"));f=width/2/math.tan(fov/2);p=root/'config/gazebo_vio/kalibr_imucam_chain.yaml';d=yaml.safe_load(p.read_text().replace('%YAML:1.0',''))
for cam in d.values():cam['intrinsics']=[f,f,width/2,a.height/2];cam['resolution']=[width,a.height]
lines=['%YAML:1.0','']
for name,cam in d.items():
 lines.append(name+':')
 for key,value in cam.items():
  if key=='T_imu_cam':lines.append('  T_imu_cam:');lines.extend('    - '+str(row) for row in value)
  else:lines.append('  '+key+': '+str(value))
p.write_text('\n'.join(lines)+'\n');p=root/'config/model-manifest.json';m=json.loads(p.read_text());m['monochrome_stereo_resolution']=[width,a.height];m['rendered_ideal_depth_sensor_present']=False;p.write_text(json.dumps(m,indent=2)+'\n')
print('Monochrome stereo resolution',width,a.height)
