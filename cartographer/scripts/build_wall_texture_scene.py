#!/usr/bin/env python3
"""Independent simulator-only wall appearance variant from actual generated assets.

Copies no navigation state and never regenerates course geometry. The original
model/resources are read-only dependencies; only seven wall visual URIs change.
"""
import argparse
import copy
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as E

NS='http://www.collada.org/2005/11/COLLADASchema';C={'c':NS};E.register_namespace('',NS)
MODEL='dronehacks_course_wall_texture';ORIGINAL='dronehacks_course';WORLD='dronehacks_wall_texture'
TILE_M=1.7
TARGETS={'C2b chicane wall','C1 chicane wall','M1 maze divider','C2a chicane wall (window)','A1 approach wall','M2 maze divider','M3 maze divider'}
TEXTURE_HASH='2a1c07687b6dbb214c4b9213739c6d92d425f1f0fc8ab3ac157f105d78240789'
def q(tag):return '{'+NS+'}'+tag
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def clean(element):
 return (element.tag,tuple(sorted(element.attrib.items())),(element.text or '').strip(),tuple(clean(child) for child in element))
def float_source(mesh,source):
 element=mesh.find('c:source[@id="'+source.lstrip('#')+'"]',C)
 a=element.find('c:float_array',C);stride=int(element.find('c:technique_common/c:accessor',C).get('stride'))
 v=list(map(float,a.text.split()));return [tuple(v[i:i+stride]) for i in range(0,len(v),stride)]
def semantic_faces(tree):
 output=[]
 for mesh in tree.findall('.//c:mesh',C):
  for tri in mesh.findall('c:triangles',C):
   ins={x.get('semantic'):x for x in tri.findall('c:input',C)};stride=max(int(x.get('offset')) for x in ins.values())+1
   p=list(map(int,tri.find('c:p',C).text.split()));vo=int(ins['VERTEX'].get('offset'));no=int(ins['NORMAL'].get('offset'))
   for start in range(0,len(p),stride*3):output.append(tuple((p[start+j*stride+vo],p[start+j*stride+no]) for j in range(3)))
 return Counter(output)

def wall_dae(source,dest):
 tree=E.parse(source);root=tree.getroot();before_faces=semantic_faces(tree)
 before_arrays={x.get('id'):x.text for x in root.findall('.//c:source/c:float_array',C)}
 images=E.SubElement(root,q('library_images'))
 # Put the library before effects, as in the existing importer.
 root.remove(images);root.insert(list(root).index(root.find('c:library_effects',C)),images)
 E.SubElement(E.SubElement(images,q('image'),id='wall-photo-image'),q('init_from')).text='../materials/textures/wood_floor_diff_1k.jpg'
 effects=root.find('c:library_effects',C);materials=root.find('c:library_materials',C)
 bindings=root.findall('.//c:instance_material',C);symbols={}
 for binding in list(bindings):
  symbol=binding.get('symbol');material_id=binding.get('target').lstrip('#')
  original=materials.find('c:material[@id="'+material_id+'"]',C);effect_id=original.find('c:instance_effect',C).get('url').lstrip('#')
  effect=copy.deepcopy(effects.find('c:effect[@id="'+effect_id+'"]',C));new_effect=effect_id+'-wall-photo';effect.set('id',new_effect)
  profile=effect.find('c:profile_COMMON',C);surface_sid=material_id+'-wall-surface';sampler_sid=material_id+'-wall-sampler'
  surface=E.Element(q('newparam'),sid=surface_sid);E.SubElement(E.SubElement(surface,q('surface'),type='2D'),q('init_from')).text='wall-photo-image';profile.insert(0,surface)
  sampler=E.Element(q('newparam'),sid=sampler_sid);E.SubElement(E.SubElement(sampler,q('sampler2D')),q('source')).text=surface_sid;profile.insert(1,sampler)
  shader=profile.find('c:technique/c:lambert',C)
  if shader is None:raise ValueError('Expected original matte Lambert shader')
  diffuse=shader.find('c:diffuse',C);diffuse.clear();E.SubElement(diffuse,q('texture'),texture=sampler_sid,texcoord='UVMap')
  effects.append(effect);material=copy.deepcopy(original);new_material=material_id+'-wall-photo';material.set('id',new_material);material.set('name',new_material);material.find('c:instance_effect',C).set('url','#'+new_effect);materials.append(material)
  new_symbol=symbol+'-wall-photo';symbols[symbol]=new_symbol
  # Bind the textured triangles separately, preserving original cap material.
  parent=next(p for p in root.iter() if binding in list(p))
  new_binding=E.SubElement(parent,q('instance_material'),symbol=new_symbol,target='#'+new_material)
  E.SubElement(new_binding,q('bind_vertex_input'),semantic='UVMap',input_semantic='TEXCOORD',input_set='0')
 textured_count=flat_count=0
 for mesh in root.findall('.//c:mesh',C):
  uv=[];uv_ids={}
  all_positions=float_source(mesh,'#positions');anchor=[min(v[k] for v in all_positions) for k in range(3)]
  for tri in list(mesh.findall('c:triangles',C)):
   inputs={x.get('semantic'):x for x in tri.findall('c:input',C)}
   if set(inputs)!={'VERTEX','NORMAL'}:raise ValueError('Only original untextured course triangles accepted')
   stride=max(int(x.get('offset')) for x in inputs.values())+1;vo=int(inputs['VERTEX'].get('offset'));no=int(inputs['NORMAL'].get('offset'));data=list(map(int,tri.find('c:p',C).text.split()));normals=float_source(mesh,inputs['NORMAL'].get('source'))
   textured=[];flat=[]
   for start in range(0,len(data),stride*3):
    corners=[(data[start+j*stride+vo],data[start+j*stride+no]) for j in range(3)]
    vertical=all(abs(normals[ni][2])<1e-6 for vi,ni in corners)
    if not vertical:flat.extend(data[start:start+stride*3]);flat_count+=1;continue
    textured_count+=1
    for corner_index,(vi,ni) in enumerate(corners):
     key=(vi,ni)
     if key not in uv_ids:
      n=normals[ni];norm=math.hypot(n[0],n[1]);t=(-n[1]/norm,n[0]/norm,0.)
      point=all_positions[vi];u=sum((point[k]-anchor[k])*t[k] for k in range(3))/TILE_M;v=(point[2]-anchor[2])/TILE_M
      uv_ids[key]=len(uv)//2;uv.extend([u,v])
     original_corner=data[start+corner_index*stride:start+(corner_index+1)*stride]
     textured.extend([*original_corner,uv_ids[key]])
   index=list(mesh).index(tri);mesh.remove(tri)
   if flat:
    cap=copy.deepcopy(tri);cap.set('count',str(len(flat)//(stride*3)));cap.find('c:p',C).text=' '.join(map(str,flat));mesh.insert(index,cap);index+=1
   if textured:
    face=copy.deepcopy(tri);face.set('count',str(len(textured)//((stride+1)*3)));face.set('material',symbols[tri.get('material')]);uv_input=E.Element(q('input'),semantic='TEXCOORD',source='#wall-photo-uv',offset=str(stride),set='0');face.insert(list(face).index(face.find('c:p',C)),uv_input);face.find('c:p',C).text=' '.join(map(str,textured));mesh.insert(index,face)
  source_uv=E.Element(q('source'),id='wall-photo-uv');E.SubElement(source_uv,q('float_array'),id='wall-photo-uv-array',count=str(len(uv))).text=' '.join(format(x,'.17g') for x in uv)
  accessor=E.SubElement(E.SubElement(source_uv,q('technique_common')),q('accessor'),source='#wall-photo-uv-array',count=str(len(uv)//2),stride='2')
  for name in ['S','T']:E.SubElement(accessor,q('param'),name=name,type='float')
  mesh.insert(list(mesh).index(mesh.find('c:vertices',C)),source_uv)
 assert semantic_faces(tree)==before_faces
 for element in root.findall('.//c:source/c:float_array',C):
  if element.get('id') in before_arrays:assert element.text==before_arrays[element.get('id')]
 dest.parent.mkdir(parents=True,exist_ok=True);tree.write(dest,encoding='utf-8',xml_declaration=True)
 return {'vertical_textured_triangles':textured_count,'horizontal_original_material_triangles':flat_count,'total_triangles':sum(before_faces.values()),'positions_and_normals_unchanged':True,'oriented_vertex_normal_triangle_multiset_unchanged':True}

def build(source_root,out):
 source=source_root/'sim/models'/ORIGINAL;original_world=source_root/'sim/worlds/dronehacks.sdf';texture=source_root/'sim/assets/v5-wood-floor.jpg'
 assert sha(texture)==TEXTURE_HASH,'Unpinned texture'
 manifest=json.loads((source/'manifest.json').read_text());selected=[p for p in manifest['pieces'] if p['name'] in TARGETS];assert len(selected)==7
 target=out/'sim/models'/MODEL
 if target.exists():raise FileExistsError('Candidate model exists; choose a fresh output root')
 target.mkdir(parents=True);shutil.copyfile(source/'model.config',target/'model.config');texture_dest=target/'materials/textures/wood_floor_diff_1k.jpg';texture_dest.parent.mkdir(parents=True);shutil.copyfile(texture,texture_dest)
 model=E.parse(source/'model.sdf');assert model.find('model').get('name')==ORIGINAL
 before_collisions=[clean(x) for x in model.findall('.//collision')];before_visuals={x.get('name'):clean(x) for x in model.findall('.//visual')};audit=[]
 for piece in selected:
  visual_name=Path(piece['mesh']).stem;visual=model.find('.//visual[@name="'+visual_name+'"]');assert visual is not None
  uri=visual.find('geometry/mesh/uri');original_uri=uri.text;assert original_uri=='model://'+ORIGINAL+'/meshes/'+piece['mesh']
  result=wall_dae(source/'meshes'/piece['mesh'],target/'meshes'/piece['mesh']);uri.text='model://'+MODEL+'/meshes/'+piece['mesh']
  audit.append({'group':piece['name'],'visual':visual_name,'original_visual_uri':original_uri,'candidate_visual_uri':uri.text,'source_sha256':sha(source/'meshes'/piece['mesh']),'candidate_sha256':sha(target/'meshes'/piece['mesh']),**result})
 assert before_collisions==[clean(x) for x in model.findall('.//collision')]
 selected_names={x['visual'] for x in audit}
 assert all(clean(x)==before_visuals[x.get('name')] for x in model.findall('.//visual') if x.get('name') not in selected_names)
 E.indent(model);model.write(target/'model.sdf',encoding='utf-8',xml_declaration=True)
 world=E.parse(original_world);includes=world.findall('.//include');course=[x for x in includes if x.findtext('uri')=='model://'+ORIGINAL];assert len(course)==1
 inc=course[0];inc.find('uri').text='model://'+MODEL
 existing=inc.find('name')
 if existing is None:E.SubElement(inc,'name').text=ORIGINAL
 else:assert existing.text==ORIGINAL
 worlds=out/'sim/worlds';worlds.mkdir(parents=True,exist_ok=True);E.indent(world);world.write(worlds/(WORLD+'.sdf'),encoding='utf-8',xml_declaration=True)
 source_files=sorted([p for p in source.rglob('*') if p.is_file()]+[original_world,texture])
 hashes={str(p.relative_to(source_root)):sha(p) for p in source_files}
 appearance={'condition':'ordinary textured-wall representative condition; not a replica of venue material','world':WORLD,'model_resource':MODEL,'world_instance_name':ORIGINAL,'source_profile':'actual original generated course asset supplied by root','texture':{'asset':'wood_floor','author':'Dimitrios Savva','source_page':'https://polyhaven.com/a/wood_floor','license':'CC0-1.0','license_page':'https://polyhaven.com/license','license_reference':'https://creativecommons.org/publicdomain/zero/1.0/','sha256':TEXTURE_HASH,'original_bitmap_unchanged':True,'nominal_tile_size_m':[TILE_M,TILE_M],'scale_scope':'Published asset width 1.7m used as ordinary UV tiling; representative only, not measured venue scale or tuned against navigation results.','shader':'original Lambert diffuse; no specular, normal, bump or displacement maps','pattern':'photographic irregular wood grain and ordinary plank seams; no symbols, fiducials, coordinates, route code or ideal checkerboard'},'wall_visuals':audit,'invariants':{'collision_count':len(before_collisions),'all_collision_xml_unchanged':True,'collision_resources_original_model':True,'collision_mesh_files_untouched':True,'unchanged_visual_count':len(before_visuals)-len(selected_names),'unchanged_visual_xml_and_uris':True,'all_wall_positions_normals_and_triangle_topology_unchanged':True,'wall_caps_original_material':True,'world_only_changes':'course include resource URI and explicit original instance name','world_physics_lighting_and_other_elements_unchanged':True,'world_original_file_untouched':True,'spawn_and_sensors_not_modified':True},'navigation':{'assets_role':'simulator/evaluator only','no_geometry_or_texture_input_added_to_NAV':True,'no_navigation_code_or_config_changed':True,'existing_allowlist_observation':'competition_landlock_exec.py grants no /work/sim access; root must verify runtime denial on exact candidate paths.','runtime_denial_verified_here':False},'validation_limits':['Offline XML/mesh/provenance checks only; Gazebo render/resource loading and effective material response require root Ubuntu verification.','Not a physical-venue photometric replica; no claim of full-course success or improved estimator/depth quality.'],'source_sha256':hashes}
 (worlds/(WORLD+'.appearance.json')).write_text(json.dumps(appearance,indent=2)+'\n')
 return appearance

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--source-root',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True);a=ap.parse_args();m=build(a.source_root,a.output_root);print(json.dumps({'world':m['world'],'selected_wall_visuals':len(m['wall_visuals']),'textured_triangles':sum(x['vertical_textured_triangles'] for x in m['wall_visuals']),'horizontal_caps_preserved':sum(x['horizontal_original_material_triangles'] for x in m['wall_visuals']),'collision_count':m['invariants']['collision_count'],'unchanged_visuals':m['invariants']['unchanged_visual_count']},indent=2))
