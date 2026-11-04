#!/usr/bin/python3
"""Compare CAD-only envelopes. Seven-inch proxy is NOT an identified airframe."""
import gc,json,time
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from cad_planner import Grid
root=Path('/work/cartographer/results');data=np.load(root/'cad-grid.npz');surface=data['surface'];origin=data['origin'];res=float(data['res']);data.close()
stages=json.loads(Path('/work/cartographer/config/stages.json').read_text())['stages'];reports=[]
for name,radius,halfheight,y in [('x500-source',.377,.23,.9144),('x500-offset',.377,.23,1.4),('seven-inch-envelope-proxy',.27,.13,.9144)]:
 started=time.monotonic();rx=radius+.06+res*.87;rz=halfheight+.06+res*.87
 dist=distance_transform_edt(~surface,sampling=[res/rx,res/rx,res/rz]);occupied=dist<=1;del dist
 occupied[:,:,int((1.45-origin[2])/res)+1:]=True;grid=Grid(occupied,origin,res)
 if name=='x500-source':
  np.savez_compressed(root/'cad-grid.npz',occupied=occupied,surface=surface,origin=origin,res=res)
  manifest=json.loads((root/'cad-grid.npz.json').read_text());manifest['vehicle_z_half_extent_m']=halfheight;(root/'cad-grid.npz.json').write_text(json.dumps(manifest,indent=2))
 report={'name':name,'vehicle_xy_radius_m':radius,'vehicle_z_half_extent_m':halfheight,'safety_margin_m':.06,'start_y_m':y,'not_identified_real_airframe':name.startswith('seven'),'path_source':'CAD planning only; no flight or truth control','stages':[]}
 start=np.array([-2.5146,y,.8]);allpaths=[]
 for stage in stages:
  try:
   p=grid.plan(start,stage['goal']);report['stages'].append({'name':stage['name'],'path':p.tolist(),'length_m':float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())});allpaths.extend(p.tolist());start=np.array(stage['goal'])
  except (ValueError,RuntimeError) as e:report['stages'].append({'name':stage['name'],'error':str(e)});break
 report['all_paths_found']=len(report['stages'])==len(stages) and all('path' in s for s in report['stages']);report['planning_wall_s']=time.monotonic()-started
 if report['all_paths_found']:
  np.savez_compressed(root/'proxy-cad-grid.npz',occupied=occupied,origin=origin,res=res)
  # Collision-check every output segment in full XYZ grid.
  assert all(grid.segment(a,b) for s in report['stages'] for a,b in zip(s['path'],s['path'][1:]))
  blocker=occupied.copy();i=grid.idx([-2.51,1.75,.8]);blocker[:,i[1]-5:i[1]+5,:]=True;blocked=Grid(blocker,origin,res)
  try:blocked.plan([-2.5146,.9144,.8],[-2.5146,2.58445,.8]);report['blocked_corridor_rejected']=False
  except (ValueError,RuntimeError):report['blocked_corridor_rejected']=True
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(9,8));z=grid.idx([0,0,.8])[2];extent=[origin[0],origin[0]+occupied.shape[0]*res,origin[1],origin[1]+occupied.shape[1]*res]
  ax.imshow(occupied[:,:,z].T,origin='lower',extent=extent,cmap='Greys',alpha=.65)
  for s in report['stages']:
   p=np.array(s['path']);ax.plot(p[:,0],p[:,1],'-o',markersize=3,label=s['name'])
  ax.set(xlabel='x [m]',ylabel='y [m]',title='CAD plan only: 0.27 m radius airframe proxy + 0.06 m margin',aspect='equal');ax.legend();fig.savefig(root/'proxy-cad-plan.png',dpi=160);plt.close(fig)
 reports.append(report);del grid,occupied;gc.collect()
 print(json.dumps({k:v for k,v in report.items() if k!='stages'}),flush=True)
(root/'envelope-plans.json').write_text(json.dumps(reports,indent=2))
