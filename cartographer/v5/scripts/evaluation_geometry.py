"""Original physical collision geometry, for independent offline scoring only.
Never allowlisted to flight or inference. Does not construct a navigation map.
"""
import hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,trimesh
from scipy.spatial.transform import Rotation

def load_collision_mesh(world_path,model_root):
 meshes=[];hashes={}
 def transform(pose):
  values=np.array([float(x) for x in (pose or '0 0 0 0 0 0').split()]);m=np.eye(4);m[:3,:3]=Rotation.from_euler('xyz',values[3:]).as_matrix();m[:3,3]=values[:3];return m
 def resolve(uri,parent):
  return Path(model_root)/uri[8:] if uri.startswith('model://') else (parent/uri).resolve()
 def load_model(model,world_from_parent,parent):
  world_from_model=world_from_parent@transform(model.findtext('pose'))
  for link in model.findall('link'):
   world_from_link=world_from_model@transform(link.findtext('pose'))
   for collision in link.findall('collision'):
    world_from_shape=world_from_link@transform(collision.findtext('pose'));geometry=collision.find('geometry');mesh=geometry.find('mesh')
    if mesh is not None:
     path=resolve(mesh.findtext('uri'),parent);hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();scene=trimesh.load(str(path),force='scene',process=False);shape=scene.to_geometry();shape.apply_scale([float(x) for x in mesh.findtext('scale','1 1 1').split()])
    elif geometry.find('box') is not None:shape=trimesh.creation.box([float(x) for x in geometry.findtext('box/size').split()])
    elif geometry.find('plane') is not None:
     size=[float(x) for x in geometry.findtext('plane/size').split()];shape=trimesh.Trimesh(vertices=[[-size[0]/2,-size[1]/2,0],[size[0]/2,-size[1]/2,0],[size[0]/2,size[1]/2,0],[-size[0]/2,size[1]/2,0]],faces=[[0,1,2],[0,2,3]],process=False)
    else:raise ValueError('Unsupported physical collision geometry; evaluation rejected')
    shape.apply_transform(world_from_shape);meshes.append(shape)
  for include in model.findall('include'):load_include(include,world_from_model,parent)
  for child in model.findall('model'):load_model(child,world_from_model,parent)
 def load_include(include,world_from_parent,parent):
  path=resolve(include.findtext('uri'),parent);path=path/'model.sdf' if path.is_dir() else path;hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();model=ET.parse(path).getroot().find('model');load_model(model,world_from_parent@transform(include.findtext('pose')),path.parent)
 world=ET.parse(world_path).getroot().find('world')
 for model in world.findall('model'):load_model(model,np.eye(4),Path(world_path).parent)
 for include in world.findall('include'):load_include(include,np.eye(4),Path(world_path).parent)
 scene=trimesh.util.concatenate(meshes)
 return scene,hashes
