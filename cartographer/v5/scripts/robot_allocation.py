"""Robot-only SITL mass/rotor calibration; never a course or flight-state input."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


def transform(element):
    a=np.zeros(6) if element is None or not element.text else np.fromstring(element.text,sep=' ')
    if a.shape!=(6,) or not np.isfinite(a).all():raise ValueError('Finite six-coordinate robot pose required')
    T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',a[3:]).as_matrix();T[:3,3]=a[:3]
    return T


def calibration(model_path):
    raw=Path(model_path).read_bytes();model=ET.fromstring(raw).find('model')
    if model is None:raise ValueError('Single robot model required')
    links={x.attrib['name']:x for x in model.findall('link')};cache={'__model__':np.eye(4)}
    def pose(name,visited=()):
        if name in cache:return cache[name]
        if name in visited or name not in links:raise ValueError('Unsupported or cyclic robot reference frame')
        p=links[name].find('pose');parent='__model__' if p is None else p.get('relative_to','__model__')
        cache[name]=pose(parent,visited+(name,))@transform(p);return cache[name]
    body_inverse=np.linalg.inv(pose('base_link'));masses=[]
    for name,link in links.items():
        inertia=link.find('inertial')
        if inertia is None:raise ValueError('Every rigid assembly link needs explicit mass')
        mass=float(inertia.findtext('mass','nan'))
        if not np.isfinite(mass) or mass<=0:raise ValueError('Positive measured/model mass required')
        p=inertia.find('pose');reference=name if p is None else p.get('relative_to',name)
        center=(body_inverse@pose(reference)@transform(p))[:3,3]
        masses.append(dict(link=name,mass_kg=mass,center_body_flu_m=center.tolist()))
    total=sum(x['mass_kg'] for x in masses)
    cg=sum(x['mass_kg']*np.asarray(x['center_body_flu_m']) for x in masses)/total
    rotors=[]
    for plugin in model.findall('plugin'):
        if not plugin.get('filename','').endswith('multicopter-motor-model-system'):continue
        index=int(plugin.findtext('motorNumber'));link=plugin.findtext('linkName')
        position=(body_inverse@pose(link))[:3,3]
        direction=plugin.findtext('turningDirection');km=float(plugin.findtext('momentConstant','nan'))
        if direction not in ('cw','ccw') or not np.isfinite(km) or km<=0:raise ValueError('Explicit rotor direction/moment ratio required')
        rotors.append(dict(index=index,link=link,position_body_flu_m=position.tolist(),
                           position_from_cg_frd_m=((position-cg)*[1,-1,-1]).tolist(),
                           moment_ratio=km if direction=='ccw' else -km,
                           motor_constant=float(plugin.findtext('motorConstant','nan')),
                           max_angular_velocity_radps=float(plugin.findtext('maxRotVelocity','nan'))))
    rotors.sort(key=lambda x:x['index'])
    if [x['index'] for x in rotors]!=list(range(4)):raise ValueError('This integration requires four explicitly indexed rotors')
    maximum=np.array([r['motor_constant']*r['max_angular_velocity_radps']**2 for r in rotors])
    if not np.isfinite(maximum).all() or (maximum<=0).any() or not np.allclose(maximum,maximum[0]):raise ValueError('This ESC calibration requires identical positive maximum rotor thrust')
    hover=total*9.80665/maximum.sum()
    if not .1<hover<.9:raise ValueError('Robot hover thrust outside configured controller domain')
    return dict(source='robot inertial masses and rotor geometry only; no world/state/CAD',
                model_sha256=hashlib.sha256(raw).hexdigest(),mass_kg=total,center_of_mass_body_flu_m=cg.tolist(),
                components=masses,rotors=rotors,hardware_qualified=False,
                actuator_model=dict(thrust_model_factor=1.,esc_velocity_min_radps=0.,
                                    normalized_hover_thrust=float(hover),
                                    assumption='SITL motor thrust=k*omega^2; ESC speed range zero to model maximum; PX4 inverse square-root thrust linearization'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--px4-parameters',action='store_true');a=p.parse_args()
    if os.environ.get('HARDWARE_MODE')=='1':raise PermissionError('Hardware requires physically measured CG and rotor calibration, not SDF masses')
    report=calibration(a.model);Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    for rotor in report['rotors']:
        i=rotor['index'];xyz=rotor['position_from_cg_frd_m']
        if not a.px4_parameters:print(i,*xyz,rotor['moment_ratio']);continue
        for axis,value in zip('XYZ',xyz):print(f'CA_ROTOR{i}_P{axis}',value)
        print(f'CA_ROTOR{i}_KM',rotor['moment_ratio'])
        print(f'SIM_GZ_EC_MIN{i+1}',0)
        print(f'SIM_GZ_EC_MAX{i+1}',rotor['max_angular_velocity_radps'])
    if a.px4_parameters:
        print('THR_MDL_FAC',report['actuator_model']['thrust_model_factor'])
        print('MPC_THR_HOVER',report['actuator_model']['normalized_hover_thrust'])


if __name__=='__main__':main()
