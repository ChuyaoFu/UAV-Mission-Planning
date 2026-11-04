#!/usr/bin/python3
"""Post-disarm evidence only. All truth/CAD comparisons stay in this evaluator."""
import sys,json,time,subprocess,hashlib,re,shutil,os
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
root=Path('/work/cartographer/v3');label=sys.argv[1];out=root/'results'/label;deadline=time.monotonic()+240
while not (out/'run.json').exists():
 if time.monotonic()>deadline:raise RuntimeError('Trial did not finish; no simulator reset authorized')
 time.sleep(1)
run=json.loads((out/'run.json').read_text());assert run['disarmed'];px='/work/src/PX4-Autopilot/build/px4_sitl_default/bin/px4-listener';status=subprocess.check_output([px,'vehicle_status','-n','1'],text=True);assert 'arming_state: 1' in status
(out/'postflight-status.txt').write_text(status)
def script(name,*args):subprocess.run(['/usr/bin/python3',str(root/'scripts'/name),*map(str,args)],check=True)
script('query_slam.py',out,'--finish');script('evaluate_bag.py',label);data=json.loads((out/'trajectory.json').read_text());evaluation=json.loads((out/'evaluation-offline.json').read_text());data.update(live_map=evaluation['rows']['map'],local_slam=evaluation['rows']['odom'],fused_control=evaluation['rows']['local']);(out/'trajectory.json').write_text(json.dumps(data));script('compare_slam.py',label,out/'evaluation-offline.json');time.sleep(1);script('capture_structure_map.py',out/'slam');script('export_slam.py',out/'slam');script('make_evidence.py',label)
# Fixed initial gauge alignment only. Never optimized against CAD or truth.
g=np.array(evaluation['rows']['truth']);a=np.array(data['optimized']);a=a[(a[:,0]>=g[0,0])&(a[:,0]<=g[-1,0])];qg=g[np.argmin(abs(g[:,0]-a[0,0])),4:8];r=Rotation.from_quat(qg)*Rotation.from_quat(a[0,4:8]).inv();gp=np.array([np.interp(a[0,0],g[:,0],g[:,i]) for i in [1,2,3]])+Rotation.from_quat(qg).apply([.12,0,.055]);points=np.load(out/'slam/actual-structure-map.npz')['points'];world=r.apply(points-a[0,1:4])+gp;cad=np.load('/work/cartographer/results/cad-grid.npz');indices=np.argwhere(cad['surface']);surface=indices*float(cad['res'])+cad['origin'];dist,_=cKDTree(surface).query(world,workers=2);geometry={'points':len(points),'median_surface_error_m':float(np.median(dist)),'p95_surface_error_m':float(np.percentile(dist,95)),'fraction_farther_than_20cm':float(np.mean(dist>.20)),'method':'fixed initial pose gauge; independent published CAD voxel-surface nearest distance, evaluation only; no trajectory fit','reference_resolution_m':float(cad['res'])};(out/'map-geometry-metrics.json').write_text(json.dumps(geometry,indent=2))
log=root/'logs/v3-slam.log';shutil.copy2(log,out/'slam-final.log');text=log.read_text();loops={'accepted':text.count('STEREO_LOOP_ACCEPT'),'veto_geometry':text.count('STEREO_LOOP_VETO unobservable_geometry'),'veto_innovation':text.count('STEREO_LOOP_VETO innovation'),'policy':'post-refinement displacement <= .25 m, angle <= .07 rad, three independent surface normal directions; detector remains enabled'}
ids=json.loads((out/'videos/screen-source.json').read_text())['window_ids'];env=dict(os.environ,DISPLAY=':0',XAUTHORITY='/tmp/visual-nav-Xauthority')
for key,filename in [('gazebo','gazebo.png'),('slam','slam-live.png')]:subprocess.run(['import','-window',ids[key],str(out/filename)],env=env,check=True)
files={}
for p in list((root/'data'/label).glob('*'))+list((out/'videos').glob('*.mp4'))+[out/'flight.ulg']:
 if p.is_file():
  h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
  files[str(p)]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
summary={'run':run,'judge':json.loads((out/'judge.json').read_text())['result'],'slam':json.loads((out/'slam-metrics.json').read_text()),'geometry':geometry,'loops':loops,'original_file_checksums':files,'truth_use':'post-flight read-only evaluation only; absent from navigation, structural filtering, mapping and display'};(out/'finalized.json').write_text(json.dumps(summary,indent=2));print(json.dumps({'judge':summary['judge'],'slam':summary['slam']['metrics'],'geometry':geometry,'loops':loops},indent=2))
