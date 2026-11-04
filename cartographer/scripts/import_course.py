#!/usr/bin/env python3
"""Convert the supplied course into static Gazebo meshes, preserving openings.

No convex hull is constructed. Net panels retain their collision boundaries but
are rendered translucent, since the source represents mesh netting as slabs.
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import argparse
import json
import re
import xml.etree.ElementTree as ET

root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--textured',action='store_true');args=ap.parse_args()
model_name='dronehacks_course_textured' if args.textured else 'dronehacks_course'
source=root/'course_source'
dest=root/'sim/models'/model_name
(dest/'meshes').mkdir(parents=True,exist_ok=True)
colors={};materials={};current=None
for line in (source/'dronehacks-course-v3.mtl').read_text().splitlines():
    a=line.split()
    if not a:continue
    if a[0]=='newmtl':current=a[1]
    elif a[0]=='Kd':colors[current]=list(map(float,a[1:4]))
v=[];normals=[];groups={};group=None;material=None
for line in (source/'dronehacks-course-v3.obj').open():
    a=line.split()
    if not a:continue
    if a[0]=='v':v.append(list(map(float,a[1:4])))
    elif a[0]=='vn':normals.append(list(map(float,a[1:4])))
    elif a[0]=='g':group=line[2:].strip();groups[group]=[]
    elif a[0]=='usemtl':material=a[1]
    elif a[0]=='f':groups[group].append(([(int(x.split('/')[0])-1,int(x.split('/')[2])-1) for x in a[1:]],material))

ns='http://www.collada.org/2005/11/COLLADASchema'
def write_dae(path,faces,textured=False):
    # Coordinates remain in the source world frame. Model pose is identity.
    indices=sorted({i for f,_ in faces for i in f});mapping={i:j for j,i in enumerate(indices)}
    used=sorted({m for _,m in faces});mids={m:f'{path.stem}-mat{i}' for i,m in enumerate(used)}
    dae=ET.Element('COLLADA',xmlns=ns,version='1.4.1')
    asset=ET.SubElement(dae,'asset');ET.SubElement(asset,'unit',name='meter',meter='1');ET.SubElement(asset,'up_axis').text='Z_UP'
    if textured:
        images=ET.SubElement(dae,'library_images')
        ET.SubElement(ET.SubElement(images,'image',id='regression-image'),'init_from').text='file:///work/sim/assets/regression-floor.png'
    effects=ET.SubElement(dae,'library_effects');mats=ET.SubElement(dae,'library_materials')
    for m in used:
        mid=mids[m];effect=ET.SubElement(effects,'effect',id=mid+'-effect')
        profile=ET.SubElement(effect,'profile_COMMON')
        if textured:
            surf=ET.SubElement(ET.SubElement(profile,'newparam',sid=mid+'-surface'),'surface',type='2D');ET.SubElement(surf,'init_from').text='regression-image'
            sampler=ET.SubElement(ET.SubElement(profile,'newparam',sid=mid+'-sampler'),'sampler2D');ET.SubElement(sampler,'source').text=mid+'-surface'
        tech=ET.SubElement(profile,'technique',sid='common');shader=ET.SubElement(tech,'lambert')
        if textured:ET.SubElement(ET.SubElement(shader,'diffuse'),'texture',texture=mid+'-sampler',texcoord='UVMap')
        else:ET.SubElement(ET.SubElement(shader,'diffuse'),'color').text=' '.join(map(str,colors[m]+[1]))
        ET.SubElement(ET.SubElement(mats,'material',id=mid,name=mid),'instance_effect',url='#'+mid+'-effect')
    mesh=ET.SubElement(ET.SubElement(ET.SubElement(dae,'library_geometries'),'geometry',id='piece'),'mesh')
    src=ET.SubElement(mesh,'source',id='positions')
    ET.SubElement(src,'float_array',id='positions-array',count=str(len(indices)*3)).text=' '.join(str(c) for i,_ in indices for c in v[i])
    acc=ET.SubElement(ET.SubElement(src,'technique_common'),'accessor',source='#positions-array',count=str(len(indices)),stride='3')
    for c in 'XYZ':ET.SubElement(acc,'param',name=c,type='float')
    nsrc=ET.SubElement(mesh,'source',id='normals')
    ET.SubElement(nsrc,'float_array',id='normals-array',count=str(len(indices)*3)).text=' '.join(str(c) for _,i in indices for c in normals[i])
    nacc=ET.SubElement(ET.SubElement(nsrc,'technique_common'),'accessor',source='#normals-array',count=str(len(indices)),stride='3')
    for c in 'XYZ':ET.SubElement(nacc,'param',name=c,type='float')
    if textured:
        uv=[]
        for vi,ni in indices:
            axis=max(range(3),key=lambda a:abs(normals[ni][a]));axes=[a for a in range(3) if a!=axis]
            uv.extend(v[vi][a]/6.096 for a in axes)
        uvs=ET.SubElement(mesh,'source',id='uv')
        ET.SubElement(uvs,'float_array',id='uv-array',count=str(len(uv))).text=' '.join(map(str,uv))
        uacc=ET.SubElement(ET.SubElement(uvs,'technique_common'),'accessor',source='#uv-array',count=str(len(indices)),stride='2')
        for c in ['S','T']:ET.SubElement(uacc,'param',name=c,type='float')
    verts=ET.SubElement(mesh,'vertices',id='vertices');ET.SubElement(verts,'input',semantic='POSITION',source='#positions')
    for m in used:
        fs=[f for f,mat in faces if mat==m]
        tri=ET.SubElement(mesh,'triangles',count=str(len(fs)),material=mids[m])
        ET.SubElement(tri,'input',semantic='VERTEX',source='#vertices',offset='0')
        ET.SubElement(tri,'input',semantic='NORMAL',source='#normals',offset='1')
        if textured:ET.SubElement(tri,'input',semantic='TEXCOORD',source='#uv',offset='2',set='0')
        ET.SubElement(tri,'p').text=' '.join(str(mapping[i]) for f in fs for i in f for _ in range(3 if textured else 2))
    scenes=ET.SubElement(dae,'library_visual_scenes');scene=ET.SubElement(scenes,'visual_scene',id='scene');node=ET.SubElement(scene,'node',id='piece-node')
    geom=ET.SubElement(node,'instance_geometry',url='#piece')
    bind=ET.SubElement(ET.SubElement(geom,'bind_material'),'technique_common')
    for m in used:
        instance=ET.SubElement(bind,'instance_material',symbol=mids[m],target='#'+mids[m])
        if textured:ET.SubElement(instance,'bind_vertex_input',semantic='UVMap',input_semantic='TEXCOORD',input_set='0')
    ET.SubElement(ET.SubElement(dae,'scene'),'instance_visual_scene',url='#scene')
    ET.ElementTree(dae).write(path,encoding='utf-8',xml_declaration=True)

sdf=ET.Element('sdf',version='1.9');model=ET.SubElement(sdf,'model',name=model_name)
ET.SubElement(model,'static').text='true';link=ET.SubElement(model,'link',name='course')
manifest={'units':'metres','up_axis':'Z','source_sha256':hashlib.sha256((source/'dronehacks-course-v3.obj').read_bytes()).hexdigest(),'pieces':[],'net_visual_transparency':.98,'source_layout':'preliminary','material_profile':'synthetic regression texture' if args.textured else 'source flat colors'}
for n,(name,faces) in enumerate(groups.items()):
    slug=f'{n:02d}_'+re.sub('[^a-z0-9]+','_',name.lower()).strip('_')
    use_texture=args.textured and not any(k in name for k in ['Net','Landing','Launch','Guideline','Approach ring'])
    write_dae(dest/'meshes'/f'{slug}.dae',faces,use_texture)
    for typ in ['visual','collision']:
        elem=ET.SubElement(link,typ,name=slug)
        geom=ET.SubElement(elem,'geometry');mesh=ET.SubElement(geom,'mesh')
        ET.SubElement(mesh,'uri').text=f'model://{model_name}/meshes/{slug}.dae'
        if typ=='visual':
            ET.SubElement(elem,'cast_shadows').text='false' if name.startswith('Net') else 'true'
            if name.startswith('Net'):ET.SubElement(elem,'transparency').text='.98'
    used={i for f,_ in faces for i,_ in f}
    lo=[min(v[i][a] for i in used) for a in range(3)];hi=[max(v[i][a] for i in used) for a in range(3)]
    manifest['pieces'].append({'name':name,'mesh':slug+'.dae','triangles':len(faces),'bounds_m':[lo,hi]})
ET.indent(sdf);ET.ElementTree(sdf).write(dest/'model.sdf',encoding='utf-8',xml_declaration=True)
(dest/'model.config').write_text('<model><name>DroneHacks course v3</name><version>3</version><sdf version="1.9">model.sdf</sdf></model>\n')
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2))

base=root/'src/PX4-Autopilot/Tools/simulation/gz/worlds/default.sdf'
world=ET.parse(base);w=world.getroot().find('world');w.set('name','visual_nav')
# The provided course already contains the floor; avoid double collision planes.
for m in list(w.findall('model')):w.remove(m)
inc=ET.SubElement(w,'include');ET.SubElement(inc,'uri').text=f'model://{model_name}'
world_name='dronehacks_textured.sdf' if args.textured else 'dronehacks.sdf'
ET.indent(world);world.write(root/'sim/worlds'/world_name,encoding='utf-8',xml_declaration=True)
print(f'Imported {len(groups)} pieces, {sum(map(len,groups.values()))} triangles; openings preserved in source meshes.')
