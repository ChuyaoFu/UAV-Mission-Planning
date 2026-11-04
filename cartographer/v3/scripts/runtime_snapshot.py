"""Record the actual loaded mapping implementation before arming."""
import os,json,hashlib,subprocess,shutil
from pathlib import Path

def snapshot(output):
 root=Path('/work/cartographer/v3');out=Path(output);binaries={};library=root/'carto_ws/install/cartographer_ros/lib/libcartographer_ros.so';node=root/'carto_ws/install/cartographer_ros/lib/cartographer_ros/cartographer_node'
 for p in [library,node]:binaries[str(p)]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 processes=[]
 for p in Path('/proc').glob('[0-9]*/exe'):
  try:
   if Path(os.readlink(p)).name!='cartographer_node':continue
   env=(p.parent/'environ').read_bytes().split(b'\0')
   if b'ROS_DOMAIN_ID=42' not in env:continue
   maps=(p.parent/'maps').read_text();loaded=sorted(set(x.split()[-1] for x in maps.splitlines() if 'libcartographer_ros.so' in x));processes.append({'pid':int(p.parent.name),'executable':os.readlink(p),'loaded_libraries':loaded})
  except (OSError,ValueError):pass
 if len(processes)!=1 or str(library) not in processes[0]['loaded_libraries']:raise RuntimeError('Validated custom mapper is not loaded in flight domain 42')
 revisions={}
 for package in ['cartographer','cartographer_ros']:
  repo=root/'carto_ws/src'/package;revisions[package]=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip();patch=subprocess.check_output(['git','-C',str(repo),'diff','--binary']);(out/(package+'.patch')).write_bytes(patch)
 for name in ['loop-guard-tests.txt','extrap-after-tests.txt','optimizer-after-tests.txt']:
  shutil.copy2(root/'results'/name,out/name)
 result={'mapping_domain':42,'binaries':binaries,'processes':processes,'revisions':revisions,'source_model_sha256':hashlib.sha256((root/'config/model.sdf').read_bytes()).hexdigest(),'checked_before_arming':True};(out/'runtime-implementation.json').write_text(json.dumps(result,indent=2));return result
