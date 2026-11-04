#!/usr/bin/python3
"""CAD-only planning map with explicit extra horizontal tracking margin."""
import json,time
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from cad_planner import Grid
root=Path('/work/cartographer/v3');data=np.load('/work/cartographer/results/cad-grid.npz');surface=data['surface'];origin=data['origin'];res=float(data['res']);radius=.26;xy_margin=.12;z_half=.12;z_margin=.06
rx=radius+xy_margin+res*.87;rz=z_half+z_margin+res*.87;distance=distance_transform_edt(~surface,sampling=[res/rx,res/rx,res/rz]);occupied=distance<=1;del distance;occupied[:,:,int((1.45-origin[2])/res)+1:]=True
out=root/'config/cad-safe-grid.npz';np.savez_compressed(out,occupied=occupied,origin=origin,res=res);g=Grid(occupied,origin,res)
stages=json.loads((root/'config/stages.json').read_text())['stages'];start=np.array([-2.5146,.9144,.8]);results=[]
for stage in stages:
 try:
  p=g.plan(start,stage['goal']);results.append({'stage':stage['name'],'path':p.tolist(),'length':float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())});start=np.array(stage['goal'])
 except Exception as e:results.append({'stage':stage['name'],'failure':str(e)});break
report={'vehicle_radius_m':radius,'horizontal_tracking_margin_m':xy_margin,'vertical_half_extent_m':z_half,'vertical_margin_m':z_margin,'raster_extra_m':res*.87,'all_stages_planned':len(results)==len(stages) and all('path' in r for r in results),'results':results};(root/'results/safe-grid-plan.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
