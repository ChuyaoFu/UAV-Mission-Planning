import os
#!/usr/bin/python3
import json,hashlib
from pathlib import Path
root=Path('/work/cartographer/v3');history=[]
for p in sorted((root/'results').glob('course-*/flight.json'),key=lambda p:int(p.parent.name.split('-')[-1])):
 f=json.loads(p.read_text());j=p.parent/'judge.json';history.append({'run':p.parent.name,'controller_complete':f.get('controller_complete',False),'disarmed':f.get('disarmed'),'failure':f.get('failure'),'last_phase':f.get('controller_phases',[{}])[-1].get('stage') if f.get('controller_phases') else None,'independent_judge':json.loads(j.read_text()).get('result') if j.exists() else None})
x=json.loads((root/'results/course-20/source-hashes.json').read_text());y=json.loads((root/'results/course-21/source-hashes.json').read_text());keys=['scripts/'+name for name in ['mission.py','fusion.py','window_localization.py','line_localization.py','cad_registration.py','depth_cloud.py','stereo_depth.py','down_range.py','start.sh','start_sim.sh']]+[k for k in x if k.startswith('config/')]
files={}
for label in ['course-18','course-20','course-21']:
 for path in list((root/'data'/label).glob('*'))+[root/'results'/label/'flight.ulg']:
  if path.is_file():
   h=hashlib.sha256()
   with path.open('rb') as stream:
    for b in iter(lambda:stream.read(1048576),b''):h.update(b)
   files[str(path)]={'bytes':path.stat().st_size,'sha256':h.hexdigest()}
manifest={'host':os.environ.get('UAV_HOST_LABEL','not-recorded'),'host_workspace':os.environ.get('UAV_HOST_WORKSPACE','not-recorded'),'container':'visual-nav-sim','container_workspace':str(root),'final_consecutive_runs':['course-20','course-21'],'identical_control_and_config_inputs':all(x[k]==y[k] for k in keys),'compared_hashes':{k:y[k] for k in keys},'all_course_runs':history,'original_evidence_files':files,'scope':'original source photometry CAD simulator; provisional engineering checkpoints; no official scoring or real-hardware certification','map_source':'actual Cartographer submaps; not the CAD planning prior','openvins_upstream':'69488123ed9362dd44b6f28e7f4680abbff1442b','px4_revision':'d6f12ad1c4f70ad3230afd7d86e971421e02fef4'}
(root/'results/final-manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'identical_control_and_config_inputs':manifest['identical_control_and_config_inputs'],'course_runs':len(history),'evidence_files':len(files)},indent=2))
