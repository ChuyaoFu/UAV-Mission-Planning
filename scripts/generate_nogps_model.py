#!/usr/bin/env python3
"""Derive PX4's model with the NavSat sensor physically absent."""
from pathlib import Path
import xml.etree.ElementTree as ET
root=Path(__file__).resolve().parents[1]
src=root/'src/PX4-Autopilot/Tools/simulation/gz/models'
dst=root/'sim/models'
def write(tree,name):
    p=dst/name;p.mkdir(parents=True,exist_ok=True)
    tree.getroot().find('model').set('name',name)
    ET.indent(tree);tree.write(p/'model.sdf',encoding='utf-8',xml_declaration=True)
    (p/'model.config').write_text(f'<model><name>{name}</name><version>1</version><sdf version="1.9">model.sdf</sdf></model>\n')
base=ET.parse(src/'x500_base/model.sdf');removed=0
for link in base.getroot().find('model').findall('link'):
    for sensor in list(link.findall('sensor')):
        if sensor.get('type') in ['navsat','gps']:
            link.remove(sensor);removed+=1
    for col in link.findall('collision'):
        sensor=ET.SubElement(link,'sensor',name='monitor_'+col.get('name'),type='contact')
        ET.SubElement(sensor,'always_on').text='true'
        ET.SubElement(sensor,'update_rate').text='100'
        ET.SubElement(sensor,'topic').text=f'/evaluation/contact/{link.get("name")}/{col.get("name")}'
        ET.SubElement(ET.SubElement(sensor,'contact'),'collision').text=col.get('name')
assert removed==1
write(base,'x500_base_nogps')
x500=ET.parse(src/'x500/model.sdf')
x500.getroot().find('model/include/uri').text='model://x500_base_nogps'
write(x500,'x500_nogps')
rig=ET.parse(dst/'x500_vio/model.sdf')
rig.getroot().find('model/include/uri').text='model://x500_nogps'
write(rig,'x500_vio_nogps')
print('Generated no-GNSS vehicle variant: NavSat sensor removed, physics and stereo/IMU unchanged.')
