"""Optional aperture avoidance in a navigation cost map only.
Does not modify physical geometry, sensing, SLAM points or judge boundaries.
"""
import numpy as np
from cad_planner import Grid

def avoid_window(grid):
 x=grid.origin[0]+np.arange(grid.occupied.shape[0])*grid.res;y=grid.origin[1]+np.arange(grid.occupied.shape[1])*grid.res;xx,yy=np.meshgrid(x,y,indexing='ij');u=.34201974*xx+.93969277*yy;n=.93969277*xx-.34201974*yy
 mask=(abs(n-.334498)<.44)&(u>-2.75316-.39)&(u<-1.83875+.39)
 occupied=grid.occupied.copy();occupied[mask,:]=True
 return Grid(occupied,grid.origin.copy(),grid.res)
if __name__=='__main__':
 import json
 from cad_planner import load
 from pathlib import Path
 g=avoid_window(load('/work/cartographer/v4/config/cad-safe-grid.npz'));p=g.plan([2.31,-.85,.88],[-1.55,-.8,.8]);out={'route':'Gate B candidate, optional window excluded from planning only','path':p.tolist(),'length_m':float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()),'flight_verified':False};Path('/work/cartographer/v4/results/gate-b-plan.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
