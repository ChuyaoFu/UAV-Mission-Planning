#!/usr/bin/env python3
"""Standalone offline evaluation figure; never publishes a reference map."""
import sys,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial.transform import Rotation
root=Path(sys.argv[1]);d=np.load(root/'offline-aligned-map.npz');p=d['points'];e=d['surface_distance'];estimated=d['estimated_path'];truth=d['true_path'];fig,(ax,bx)=plt.subplots(1,2,figsize=(12,5.5),layout='constrained');fig.suptitle('Offline validation: online measured map and navigation\nReference geometry/poses were unavailable to navigation',fontsize=13)
for m in ET.parse(root/'evaluation-world.sdf').findall('.//model'):
 if m.get('name')=='floor':continue
 pose=np.array([float(v) for v in m.findtext('pose').split()]);size=np.array([float(v) for v in m.findtext('link/collision/geometry/box/size').split()]);corners=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]])*size/2;corners=Rotation.from_euler('xyz',pose[3:]).apply(corners)+pose[:3];ax.add_patch(Polygon(corners[:,:2],facecolor='#dddddd',edgecolor='#888888',lw=.5))
mask=p[:,2]>.25;sc=ax.scatter(p[mask,0],p[mask,1],c=e[mask],s=7,vmin=0,vmax=.5,cmap='viridis_r',label='Measured occupied voxels above 0.25 m');ax.plot(truth[:,0],truth[:,1],color='black',lw=1.2,label='Independent true flight');ax.plot(estimated[:,0],estimated[:,1],color='#e85f36',lw=1.1,label='Measured flight (initial alignment only)');ax.scatter(truth[0,0],truth[0,1],marker='*',s=120,color='red',label='Initial position');ax.set(xlabel='Reference X (m)',ylabel='Reference Y (m)',title='Actual local occupancy vs reference surfaces');ax.set_aspect('equal');ax.legend(fontsize=7,loc='upper right');fig.colorbar(sc,ax=ax,label='Nearest reference surface distance (m)',shrink=.7)
error=np.linalg.norm(estimated-truth,axis=1);bx.plot(np.linspace(0,1,len(error)),error,color='#e85f36');bx.set(xlabel='Fraction of recorded flight',ylabel='Position error (m)',title='Position error: fixed initial alignment');bx.grid(alpha=.25);fig.savefig(root/'map-and-drift-evaluation.png',dpi=150);fig.savefig(root/'map-and-drift-evaluation.svg');print(root/'map-and-drift-evaluation.png')
