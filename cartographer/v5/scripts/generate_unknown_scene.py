#!/usr/bin/python3
"""Evaluation-only unknown furnished room. Never imported by navigation.
Physical boxes/cabinets provide ordinary geometry; no image texture, lidar,
rendered depth, pose topic or navigation target is generated.
"""
import argparse,json,math,random,xml.etree.ElementTree as E
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--world',required=True);p.add_argument('--scenario',required=True);p.add_argument('--floor-material',choices=['matte','wood'],default='matte');a=p.parse_args();r=random.Random(a.seed)
s=E.Element('sdf',version='1.9');w=E.SubElement(s,'world',name='visual_nav');physics=E.SubElement(w,'physics',type='ode')
for k,v in [('max_step_size','.004'),('real_time_factor','1'),('real_time_update_rate','250')]:E.SubElement(physics,k).text=v
E.SubElement(w,'gravity').text='0 0 -9.8';scene=E.SubElement(w,'scene');E.SubElement(scene,'ambient').text='.5 .5 .5 1';E.SubElement(scene,'background').text='.7 .7 .7 1';E.SubElement(scene,'shadows').text='true'
light=E.SubElement(w,'light',name='sun',type='directional');E.SubElement(light,'pose').text='0 0 20 0 0 0';E.SubElement(light,'direction').text='-.2 .4 -1';E.SubElement(light,'diffuse').text='.8 .8 .8 1';E.SubElement(light,'cast_shadows').text='true'
def box(name,xyz,size,color,yaw=0):
 m=E.SubElement(w,'model',name=name);E.SubElement(m,'static').text='true';E.SubElement(m,'pose').text=' '.join(map(str,[*xyz,0,0,yaw]));l=E.SubElement(m,'link',name='body')
 for k in ['collision','visual']:
  v=E.SubElement(l,k,name=k);g=E.SubElement(v,'geometry');b=E.SubElement(g,'box');E.SubElement(b,'size').text=' '.join(map(str,size))
  if k=='visual':
   mat=E.SubElement(v,'material');E.SubElement(mat,'diffuse').text=' '.join(map(str,[*color,1]));E.SubElement(mat,'ambient').text=' '.join(map(str,[*color,1]))
box('floor',[0,0,-.05],[5.4,6.2,.1],[.62,.62,.60])
if a.floor_material=='wood':
 material=w.find("model[@name='floor']/link/visual/material");metal=E.SubElement(E.SubElement(material,'pbr'),'metal');E.SubElement(metal,'albedo_map').text='file:///work/sim/assets/v5-wood-floor.jpg';E.SubElement(metal,'roughness').text='.9';E.SubElement(metal,'metalness').text='0'
for x in [-2.6,2.6]:box('wall_x'+str(x),[x,0,1.2],[.12,6,2.4],[.75,.75,.72])
for y in [-3.,3.]:box('wall_y'+str(y),[0,y,1.2],[5.2,.12,2.4],[.73,.74,.75])
# Independently randomized furniture on each wall, leaving broad circulation.
for side in [-1,1]:
 for row in range(5):
  y=-2.2+row*1.05+r.uniform(-.12,.12);x=side*(2.12+r.uniform(-.08,.08));width=r.uniform(.42,.62);height=r.uniform(1.05,1.9)
  box(f'cabinet_{side}_{row}',[x,y,height/2],[width,.78,height],[.38+r.random()*.1,.26+r.random()*.08,.15+r.random()*.05])
  # Physical door recess outlines, not a printed visual pattern.
  for dz in [.25,.65,1.0]:
   if dz<height:box(f'handle_{side}_{row}_{dz}',[x-side*(width/2+.02),y,dz],[.04,.20,.025],[.15,.15,.15])
for j,(x,y) in enumerate([(r.uniform(-.25,.25),r.uniform(.4,.8)),(r.uniform(-.8,-.5),r.uniform(1.7,2.0))]):
 h=r.uniform(.7,1.2);box(f'crate_{j}',[x,y,h/2],[.45,.45,h],[.43,.32,.21],r.uniform(-.25,.25))
start=[r.uniform(-1.1,-.9),r.uniform(-1.75,-1.5),.14];yaw=r.uniform(-math.pi,math.pi)
Path(a.world).parent.mkdir(parents=True,exist_ok=True);E.indent(s);E.ElementTree(s).write(a.world,encoding='utf-8',xml_declaration=True)
Path(a.scenario).parent.mkdir(parents=True,exist_ok=True);Path(a.scenario).write_text(json.dumps({'label':'unknown-furnished-room-'+str(a.seed)+'-'+a.floor_material,'seed':a.seed,'spawn_xyz':start,'spawn_yaw_rad':yaw,'world':'cadfree_unknown','novel_layout':True,'role':'simulator/evaluator only; navigation has no filesystem access','launch_box_halfwidth_m':.5,'appearance':a.floor_material,'no_added_image_texture':a.floor_material=='matte','appearance_source':None if a.floor_material=='matte' else 'https://polyhaven.com/a/wood_floor','limitation':'Furnished-room generalization only; plain untextured walls remain an observability stress case.'},indent=2))
