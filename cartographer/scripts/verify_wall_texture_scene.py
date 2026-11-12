#!/usr/bin/env python3
"""Offline invariants for simulator-only wall visual candidate; no Gazebo/ROS."""
import argparse
import copy
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tempfile
import shutil
import xml.etree.ElementTree as E
N={'c':'http://www.collada.org/2005/11/COLLADASchema'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(e):return (e.tag,tuple(sorted(e.attrib.items())),(e.text or '').strip(),tuple(canon(c) for c in e))
def geometry(tree):
 result=Counter();uv_counts=[0,0]
 for mesh in tree.findall('.//c:mesh',N):
  for tri in mesh.findall('c:triangles',N):
   ins={i.get('semantic'):i for i in tri.findall('c:input',N)};stride=max(int(i.get('offset')) for i in ins.values())+1
   p=list(map(int,tri.find('c:p',N).text.split()));assert len(p)==int(tri.get('count'))*3*stride
   assert all(child.tag.endswith('input') for child in list(tri)[:len(ins)]),'input must precede p'
   for offset in range(0,len(p),3*stride):
    result[tuple((p[offset+j*stride+int(ins['VERTEX'].get('offset'))],p[offset+j*stride+int(ins['NORMAL'].get('offset'))]) for j in range(3))]+=1
   uv_counts['TEXCOORD' not in ins]+=int(tri.get('count'))
   if 'TEXCOORD' in ins:
    source=mesh.find('c:source[@id="'+ins['TEXCOORD'].get('source')[1:]+'"]',N);a=list(map(float,source.find('c:float_array',N).text.split()));assert len(a)==int(source.find('c:float_array',N).get('count'));assert all(math.isfinite(v) for v in a)
    assert int(source.find('c:technique_common/c:accessor',N).get('stride'))==2
    indices=p[int(ins['TEXCOORD'].get('offset'))::stride];assert min(indices)>=0 and max(indices)<len(a)//2
 return result,uv_counts

def check(source_root,candidate_root):
 source=source_root/'sim/models/dronehacks_course';target=candidate_root/'sim/models/dronehacks_course_wall_texture';worlds=candidate_root/'sim/worlds';audit=json.loads((worlds/'dronehacks_wall_texture.appearance.json').read_text())
 for name,h in audit['source_sha256'].items():assert sha(source_root/name)==h,'Original source file changed: '+name
 bitmap=target/'materials/textures/wood_floor_diff_1k.jpg';assert sha(bitmap)==audit['texture']['sha256'];assert sha(bitmap)==sha(source_root/'sim/assets/v5-wood-floor.jpg')
 original=E.parse(source/'model.sdf');candidate=E.parse(target/'model.sdf');original_visuals={v.get('name'):v for v in original.findall('.//visual')};candidate_visuals={v.get('name'):v for v in candidate.findall('.//visual')};assert set(original_visuals)==set(candidate_visuals)
 selected={row['visual']:row for row in audit['wall_visuals']};assert len(selected)==10
 for name,visual in candidate_visuals.items():
  if name in selected:
   row=selected[name];assert visual.findtext('geometry/mesh/uri')==row['candidate_visual_uri'];visual.find('geometry/mesh/uri').text=row['original_visual_uri']
  assert canon(visual)==canon(original_visuals[name]),'Unexpected visual change: '+name
 assert canon(original.getroot())==canon(candidate.getroot()),'Model differs beyond selected visual URI'
 collisions=candidate.findall('.//collision');assert len(collisions)==33
 for c in collisions:assert c.findtext('geometry/mesh/uri').startswith('model://dronehacks_course/meshes/')
 original_world=E.parse(source_root/'sim/worlds/dronehacks.sdf');world=E.parse(worlds/'dronehacks_wall_texture.sdf');matches=[i for i in world.findall('.//include') if i.findtext('uri')=='model://dronehacks_course_wall_texture'];assert len(matches)==1;inc=matches[0];assert inc.findtext('name')=='dronehacks_course';inc.find('uri').text='model://dronehacks_course';inc.remove(inc.find('name'));assert canon(original_world.getroot())==canon(world.getroot()),'World changed beyond include/name'
 textured=caps=0
 for row in audit['wall_visuals']:
  name=Path(row['original_visual_uri']).name;a=E.parse(source/'meshes'/name);b=E.parse(target/'meshes'/name)
  ga,counts_a=geometry(a);gb,counts_b=geometry(b);assert ga==gb,'Mesh triangle topology changed';assert sum(ga.values())==row['total_triangles']
  arrays_a={p.get('id'):p.text for p in a.findall('.//c:source/c:float_array',N)};arrays_b={p.get('id'):p.text for p in b.findall('.//c:source/c:float_array',N)}
  assert all(arrays_b[k]==v for k,v in arrays_a.items()),'Original positions/normals changed'
  assert canon(a.find('c:asset',N))==canon(b.find('c:asset',N))
  poses=lambda t:[canon(e) for node in t.findall('.//c:node',N) for e in node if e.tag.rsplit('}',1)[-1] in ['matrix','translate','rotate','scale']]
  assert poses(a)==poses(b),'Mesh transform changed'
  assert counts_b==[row['vertical_textured_triangles'],row['horizontal_original_material_triangles']]
  textured+=counts_b[0];caps+=counts_b[1]
  images=b.findall('.//c:library_images/c:image/c:init_from',N);assert len(images)==1
  assert (target/'meshes'/images[0].text).resolve()==bitmap.resolve()
  assert b.find('.//c:lambert/c:diffuse/c:texture',N) is not None
  assert not b.findall('.//c:phong',N) and not b.findall('.//c:bump',N)
  for tri in b.findall('.//c:triangles',N):
   ins={i.get('semantic'):i for i in tri.findall('c:input',N)};mesh=next(m for m in b.findall('.//c:mesh',N) if tri in list(m));normals=list(map(float,mesh.find('c:source[@id="normals"]/c:float_array',N).text.split()));stride=max(int(i.get('offset')) for i in ins.values())+1;p=list(map(int,tri.find('c:p',N).text.split()));ni=p[int(ins['NORMAL'].get('offset'))::stride]
   if 'TEXCOORD' in ins:assert all(abs(normals[3*i+2])<1e-6 for i in ni),'Horizontal cap textured'
   else:assert all(abs(normals[3*i+2])>.99 for i in ni),'Vertical face lost texture'
 assert textured==96 and caps==44
 assert 'collision' not in str([p.relative_to(target) for p in target.rglob('*') if p.is_file()])
 return {'collision_nodes_identical':33,'unchanged_visuals_identical':len(original_visuals)-len(selected),'textured_vertical_triangles':textured,'preserved_horizontal_triangles':caps,'window_triangles_unchanged':next(row['total_triangles'] for row in audit['wall_visuals'] if 'window' in row['group']),'texture_sha256':sha(bitmap),'source_file_hashes_unchanged':len(audit['source_sha256'])}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--candidate-root',type=Path,required=True);p.add_argument('--negative-controls',action='store_true');a=p.parse_args();print(json.dumps(check(a.source_root,a.candidate_root),indent=2))
 if a.negative_controls:
  mutations=['collision','floor_visual','mesh_positions','world_gravity','bitmap']
  for mutation in mutations:
   with tempfile.TemporaryDirectory() as temp:
    root=Path(temp);shutil.copytree(a.candidate_root,root/'payload');candidate=root/'payload';model=candidate/'sim/models/dronehacks_course_wall_texture';world=candidate/'sim/worlds/dronehacks_wall_texture.sdf'
    if mutation in ['collision','floor_visual']:
     path=model/'model.sdf';t=E.parse(path);e=t.find('.//collision/geometry/mesh/uri') if mutation=='collision' else t.find('.//visual[@name="29_floor_20_x_20_ft"]/geometry/mesh/uri');e.text=e.text.replace('dronehacks_course/','dronehacks_course_wall_texture/');t.write(path)
    elif mutation=='mesh_positions':
     path=model/'meshes/14_c2a_chicane_wall_window.dae';t=E.parse(path);v=t.find('.//c:source[@id="positions"]/c:float_array',N);fields=v.text.split();fields[0]=str(float(fields[0])+.001);v.text=' '.join(fields);t.write(path)
    elif mutation=='world_gravity':
     t=E.parse(world);t.find('.//gravity').text='0 0 -9.81';t.write(world)
    else:
     with (model/'materials/textures/wood_floor_diff_1k.jpg').open('ab') as f:f.write(b'changed')
    try:check(a.source_root,candidate)
    except AssertionError:print('PASS: negative control rejected:',mutation)
    else:raise AssertionError('Negative control unexpectedly passed: '+mutation)
