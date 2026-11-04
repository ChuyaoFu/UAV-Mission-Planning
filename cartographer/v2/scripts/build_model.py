#!/usr/bin/python3
"""Explicit seven-inch approximation. Not a measured official airframe."""
import copy,json,math,os
from pathlib import Path
import xml.etree.ElementTree as E
ROOT=Path(__file__).resolve().parents[1]
def sub(e,k,v=None,**kw):
 n=E.SubElement(e,k,**kw)
 if v is not None:n.text=str(v)
 return n
s=E.Element('sdf',version='1.9');m=sub(s,'model',name='seveninch_v2');sub(m,'self_collide','false')
base=sub(m,'link',name='base_link');inert=sub(base,'inertial');sub(inert,'mass',1.10);it=sub(inert,'inertia')
for k,v in dict(ixx=.008,iyy=.008,izz=.014,ixy=0,ixz=0,iyz=0).items():sub(it,k,v)
def shape(link,name,typ,values,pose,color,collision=True):
 for tag in (['visual','collision'] if collision else ['visual']):
  n=sub(link,tag,name=name);sub(n,'pose',pose);g=sub(n,'geometry');sh=sub(g,typ)
  for key,value in values.items():sub(sh,key,value)
  if tag=='visual':mat=sub(n,'material');sub(mat,'ambient',color);sub(mat,'diffuse',color)
  else:
   surf=sub(n,'surface');fr=sub(surf,'friction');ode=sub(fr,'ode');sub(ode,'mu',.8);sub(ode,'mu2',.8)
   c=sub(link,'sensor',name='monitor_'+name,type='contact');sub(c,'always_on','true');sub(c,'update_rate',100);sub(sub(c,'contact'),'collision',name)
shape(base,'body','box',{'size':'.20 .13 .07'},'0 0 0 0 0 0','.12 .25 .65 1')
shape(base,'battery','box',{'size':'.15 .06 .05'},'-.02 0 -.05 0 0 0','.18 .18 .19 1')
for side in [-1,1]:shape(base,'skid_'+str(side),'box',{'size':'.17 .014 .018'},f'0 {side*.065} -.102 0 0 0','.1 .1 .1 1')
for angle in [-math.pi/4,math.pi/4]:shape(base,'arm_'+str(angle),'box',{'size':'.35 .022 .016'},f'0 0 .025 0 0 {angle}','.08 .09 .1 1')
# Reuse exactly the existing noisy base IMU, barometer and magnetometer, no NavSat.
ref=E.parse(ROOT/'config/base-reference.sdf').find("model/link[@name='base_link']")
for sensor in ref.findall('sensor'):
 if sensor.get('type') in ['imu','magnetometer','air_pressure']:base.append(copy.deepcopy(sensor))
positions=[(.115,-.115),(-.115,.115),(.115,.115),(-.115,-.115)]
for i,(x,y) in enumerate(positions):
 link=sub(m,'link',name=f'rotor_{i}');sub(link,'pose',f'{x} {y} .045 0 0 0');inert=sub(link,'inertial');sub(inert,'mass',.004);ii=sub(inert,'inertia')
 for key,val in dict(ixx=8e-6,iyy=8e-6,izz=1.6e-5).items():sub(ii,key,val)
 shape(link,f'rotor_{i}_collision','cylinder',{'radius':.0889,'length':.004},'0 0 0 0 0 0','.08 .08 .08 .65')
 j=sub(m,'joint',name=f'rotor_{i}_joint',type='revolute');sub(j,'parent','base_link');sub(j,'child',f'rotor_{i}');a=sub(j,'axis');sub(a,'xyz','0 0 1');lim=sub(a,'limit');sub(lim,'lower',-1e16);sub(lim,'upper',1e16)
 plugin=sub(m,'plugin',filename='gz-sim-multicopter-motor-model-system',name='gz::sim::systems::MulticopterMotorModel')
 for k,v in dict(jointName=f'rotor_{i}_joint',linkName=f'rotor_{i}',turningDirection='ccw' if i<2 else 'cw',timeConstantUp=.0125,timeConstantDown=.025,maxRotVelocity=1000,motorConstant=5.0e-6,momentConstant=.016,commandSubTopic='command/motor_speed',motorNumber=i,rotorDragCoefficient=8.06428e-5,rollingMomentCoefficient=1e-6,rotorVelocitySlowdownSim=10,motorType='velocity').items():sub(plugin,k,v)
rig=copy.deepcopy(E.parse(ROOT.parent/'cartographer/config/carto-model.sdf').find("model/link[@name='vio_rig']")) if (ROOT.parent/'cartographer/config/carto-model.sdf').exists() else copy.deepcopy(E.parse(ROOT/'config/old-model.sdf').find("model/link[@name='vio_rig']"))
rig.find('pose').text='.12 0 .055 0 0 0';rig.find('inertial/mass').text='.05'
pitch_degrees=float(os.environ.get('CAMERA_PITCH_DEG','30'));pitch=math.radians(pitch_degrees)
for sensor in rig.findall('sensor'):
 if sensor.get('name') in ['left','right','depth']:
  vals=sensor.find('pose').text.split();vals[4]=str(pitch);sensor.find('pose').text=' '.join(vals)
 if sensor.get('name')=='down_tof':
  sensor.find('pose').text='-.12 0 -.080 0 1.57079632679 0';sensor.find('lidar/noise/stddev').text='.005'
m.append(rig);j=sub(m,'joint',name='vio_rig_joint',type='fixed');sub(j,'parent','base_link');sub(j,'child','vio_rig')
plugin=sub(m,'plugin',filename='gz-sim-pose-publisher-system',name='gz::sim::systems::PosePublisher')
for k,v in dict(publish_model_pose='true',publish_link_pose='true',use_pose_vector_msg='true',update_frequency=20).items():sub(plugin,k,v)
E.indent(s);E.ElementTree(s).write(ROOT/'config/model.sdf',encoding='utf-8',xml_declaration=True)
manifest={'model':'seven-inch approximation v2','identified_official_airframe':False,'rotor_diameter_m':.1778,'motor_xy_m':positions,'nominal_total_mass_kg':1.166,'base_inertia_kg_m2':[.008,.008,.014],'motor_constant_N_per_rad_s_squared':5e-6,'max_rot_speed_rad_s':1000,'max_total_thrust_N':20,'camera_pitch_down_degrees':pitch_degrees,'rig_from_base_xyz_m':[.12,0,.055],'down_range_origin_base_z_m':-.025,'collision_envelope_xy_radius_m':.26,'collision_envelope_z_half_m':.12,'safety_margin_m':.06,'added_nonkit_sensors':[]}
(ROOT/'config/model-manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
