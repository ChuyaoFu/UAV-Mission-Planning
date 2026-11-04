#!/usr/bin/python3
"""Post-flight reference overlay in the common course frame. No ROS publishers."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
out=Path(sys.argv[1]);runs=[Path(p) for p in sys.argv[2:]];out.mkdir(exist_ok=True,parents=True)
# Display the published CAD as an EXPLICIT independent gray reference only.
vertices=[];lines=[]
for row in Path('/work/course_source/dronehacks-course-v3.obj').read_text().splitlines():
 a=row.split()
 if not a:continue
 if a[0]=='v':vertices.append(list(map(float,a[1:4])))
 elif a[0]=='f':
  p=np.array([vertices[int(x.split('/')[0])-1] for x in a[1:]]);n=np.cross(p[1]-p[0],p[2]-p[0]);norm=np.linalg.norm(n)
  if norm>1e-8 and abs(n[2])/norm<.25 and p[:,2].max()>.15 and p[:,2].min()<1.3:
   # Only long visible triangle edges, reduced below to a reference silhouette.
   for q,v in zip(p,np.roll(p,-1,axis=0)):
    if np.linalg.norm(v[:2]-q[:2])>.15:lines.append([q[:2],v[:2]])
if lines:
 lines=np.array(lines);keys=np.round(np.sort(lines,axis=1)/.01).astype(int).reshape(-1,4);_,idx=np.unique(keys,axis=0,return_index=True);lines=lines[idx]
fig,axes=plt.subplots(1,len(runs),figsize=(6.5*len(runs),6.5),squeeze=False,layout='constrained');metrics={}
for ax,source in zip(axes[0],runs):
 e=json.load(open(source/'evaluation-offline.json'));g=np.array(e['rows']['truth']);nodes=np.array(json.load(open(source/'trajectory.json'))['optimized']);anchor=nodes[0];qg=g[0,4:8] if anchor[0]<g[0,0] else g[np.argmin(abs(g[:,0]-anchor[0])),4:8];r=Rotation.from_quat(qg)*Rotation.from_quat(anchor[4:8]).inv();gp=np.array([np.interp(anchor[0],g[:,0],g[:,i]) for i in [1,2,3]])+Rotation.from_quat(qg).apply([.12,0,.055]);cloud=np.load(source/'slam/actual-structure-map.npz')['points'];world=r.apply(cloud-anchor[1:4])+gp
 cad=np.load('/work/cartographer/results/cad-grid.npz');indices=np.argwhere(cad['surface']);surf=indices*float(cad['res'])+cad['origin'];dist,_=cKDTree(surf).query(world,workers=2);valid=(nodes[:,0]>=g[0,0])&(nodes[:,0]<=g[-1,0]);a=nodes[valid];estimate=r.apply(a[:,1:4]-anchor[1:4])+gp;truth=np.array([np.interp(a[:,0],g[:,0],g[:,i]) for i in [1,2,3]]).T+Rotation.from_quat(np.array([g[np.argmin(abs(g[:,0]-t)),4:8] for t in a[:,0]])).apply([.12,0,.055]);err=np.linalg.norm(estimate-truth,axis=1);m={'points':len(cloud),'surface_median_m':float(np.median(dist)),'surface_p95_m':float(np.percentile(dist,95)),'surface_over_20cm':float(np.mean(dist>.2)),'node_rmse_m':float(np.sqrt(np.mean(err**2))),'node_max_m':float(err.max()),'gauge':'FIRST initialized node, never reset at airborne observations','geometry_source':'actual measured stereo points transformed by optimized SLAM, never CAD-filled'};metrics[source.name]=m
 ax.add_collection(LineCollection(lines,colors='#8b929a',linewidths=.55,alpha=.35,label='Published CAD reference (evaluation only)'));v=ax.scatter(world[:,0],world[:,1],c=np.minimum(dist,.2),s=2,cmap='viridis',vmin=0,vmax=.2,rasterized=True,label='Actual optimized stereo map');ax.plot(estimate[:,0],estimate[:,1],color='#df6c21',lw=1.2,label='Optimized trajectory');ax.set(aspect='equal',xlim=(-3.15,3.15),ylim=(-3.15,3.15),xlabel='Course x (m)',ylabel='Course y (m)',title=source.name+f'\nNode RMSE {m["node_rmse_m"]*100:.1f} cm | surface P95 {m["surface_p95_m"]*100:.1f} cm');ax.grid(alpha=.15);ax.legend(loc='lower left',fontsize=7);fig.colorbar(v,ax=ax,shrink=.6,label='Independent nearest-surface error (m)')
fig.suptitle('Actual SLAM vs published geometry — single initialization gauge, no trajectory fit\nGray geometry is a labeled post-flight reference, not SLAM output',fontsize=12);fig.savefig(out/'map-vs-reference.png',dpi=150);(out/'corrected-comparison.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
