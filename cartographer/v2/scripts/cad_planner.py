#!/usr/bin/python3
"""3-D voxel surface map and A*, from published CAD only, no flown map.
The 4-cm grid conservatively inflates vehicle envelope. Semantic targets are
course annotations, not a replay trajectory; A* determines every intermediate.
"""
import argparse,json,hashlib,heapq,itertools,math
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt,label,generate_binary_structure
class Grid:
 def __init__(self,occupied,origin,res):self.occupied=occupied;self.origin=np.array(origin);self.res=res;self.labels=None
 def idx(self,p):return tuple(np.rint((np.asarray(p)-self.origin)/self.res).astype(int))
 def xyz(self,i):return self.origin+np.array(i)*self.res
 def free(self,i):return all(0<=i[d]<self.occupied.shape[d] for d in range(3)) and not self.occupied[i]
 def segment(self,a,b):
  a=np.asarray(a);b=np.asarray(b);return all(self.free(self.idx(p)) for p in np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/(self.res*.4))+1)))
 def plan(self,start,goal,_flat=False):
  if not _flat:
   if self.segment(start,goal):return np.array([start,goal])
   # Prefer a constant-height 3D collision-checked route; fall back to XYZ search.
   for z in dict.fromkeys([max(start[2],goal[2]),1.05,1.25]):
    a=np.array([start[0],start[1],z]);b=np.array([goal[0],goal[1],z])
    if self.segment(start,a) and self.segment(b,goal):
     try:
      middle=self.plan(a,b,_flat=True)
      return np.vstack([np.asarray(start),middle,np.asarray(goal)])
     except (ValueError,RuntimeError):pass
  s=self.idx(start);g=self.idx(goal)
  if self.labels is None:self.labels=label(~self.occupied,structure=generate_binary_structure(3,1))[0]
  if not self.free(s) or not self.free(g):raise ValueError(f'Endpoint in inflated obstacle: start {start} free={self.free(s)} goal {goal} free={self.free(g)}')
  if self.labels[s]!=self.labels[g]:raise RuntimeError('Endpoints are in disconnected 3D free-space components')
  queue=[(0,0,s)];cost={s:0};prev={};moves=list(itertools.product([-1,0,1],repeat=3));moves.remove((0,0,0));expanded=0
  if _flat:moves=[m for m in moves if m[2]==0]
  while queue:
   _,c,i=heapq.heappop(queue)
   if c>cost[i]+1e-9:continue
   if i==g:
    path=[i]
    while path[-1]!=s:path.append(prev[path[-1]])
    p=[self.xyz(k) for k in path[::-1]];out=[p[0]];k=0
    while k<len(p)-1:
     j=len(p)-1
     while j>k+1 and not self.segment(p[k],p[j]):j-=1
     out.append(p[j]);k=j
    return np.array(out)
   expanded+=1
   for m in moves:
    j=tuple(i[d]+m[d] for d in range(3))
    if not self.free(j):continue
    # Every subset corner must be free; diagonal moves cannot cut obstacles.
    axes=[d for d in range(3) if m[d]]
    if any(not self.free(tuple(i[d]+(m[d] if d in sub else 0) for d in range(3))) for n in range(1,len(axes)) for sub in itertools.combinations(axes,n)):continue
    nc=c+math.sqrt(sum(v*v for v in m))
    if nc<cost.get(j,float('inf')):
     cost[j]=nc;prev[j]=i;h=np.linalg.norm(np.array(j)-g);heapq.heappush(queue,(nc+h,nc,j))
  raise RuntimeError(f'No 3D path after {expanded} expansions')
def build(source,out,res=.015,radius=.377,vertical=.23,margin=.06):
 vertices=[];faces=[];name='';groups={}
 for line in Path(source).read_text().splitlines():
  a=line.split()
  if not a:continue
  if a[0]=='g':name=line[2:].strip();groups[name]=[]
  elif a[0]=='v':vertices.append(list(map(float,a[1:4])));groups[name].append(len(vertices)-1)
  elif a[0]=='f':faces.append((name,[int(s.split('/')[0])-1 for s in a[1:]]))
 v=np.array(vertices);origin=np.array([-3.12,-3.12,-.04]);shape=tuple(np.ceil(np.array([6.28,6.28,3.24])/res).astype(int));surface=np.zeros(shape,bool)
 for name,f in faces:
  a,b,c=v[f];steps=max(1,int(np.ceil(max(np.linalg.norm(a-b),np.linalg.norm(a-c),np.linalg.norm(b-c))/(res*.5))))
  u,w=np.triu_indices(steps+1);u=u/steps;w=(w-u*steps)/steps
  p=a+u[:,None]*(b-a)+w[:,None]*(c-a)
  i=np.rint((p-origin)/res).astype(int);valid=np.all((i>=0)&(i<np.array(shape)),axis=1);i=i[valid];surface[tuple(i.T)]=True
 # EDT in normalized ellipsoid coordinates; half voxel diagonal adds raster error.
 rx=radius+margin+res*.87;rz=vertical+margin+res*.87
 dist=distance_transform_edt(~surface,sampling=[res/rx,res/rx,res/rz]);occ=dist<=1
 # No over-wall shortcut. Grid includes 3D tunnel/window geometry and floor.
 occ[:,:,int((1.45-origin[2])/res)+1:]=True
 np.savez_compressed(out,occupied=occ,surface=surface,origin=origin,res=res)
 bounds={name:[v[ids].min(0).tolist(),v[ids].max(0).tolist()] for name,ids in groups.items() if ids}
 manifest={'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),'resolution_m':res,'vehicle_xy_radius_m':radius,'vehicle_z_half_extent_m':vertical,'safety_margin_m':margin,'raster_extra_m':res*.87,'altitude_cap_m':1.45,'bounds':bounds,'map_source':'published CAD only; no previous run observations'}
 Path(str(out)+'.json').write_text(json.dumps(manifest,indent=2));return Grid(occ,origin,res),manifest
def load(path):
 d=np.load(path);return Grid(d['occupied'],d['origin'],float(d['res']))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--source',default='/work/course_source/dronehacks-course-v3.obj');ap.add_argument('--output',default='/work/cartographer/results/cad-grid.npz');ap.add_argument('--start-y',type=float,default=.9144);args=ap.parse_args()
 if Path(args.output).exists() and Path(args.output+'.json').exists() and json.loads(Path(args.output+'.json').read_text()).get('source_sha256')==hashlib.sha256(Path(args.source).read_bytes()).hexdigest():
  grid=load(args.output);manifest=json.loads(Path(args.output+'.json').read_text())
 else:grid,manifest=build(args.source,args.output)
 # Explicit semantic course annotation; these are not claimed general perception.
 stages=[('first_corridor',[-2.5146,2.58445,.8]),('gate_a',[.90,2.58445,.8]),('tunnel_approach',[2.31,1.8,.88]),('tunnel_exit',[2.31,-.85,.88]),('gate_or_window',[-1.55,-.8,.8]),('landing',[-2.04216,-.79248,.8])]
 start=np.array([-2.5146,args.start_y,.8]);paths=[]
 for name,goal in stages:
  print('Planning '+name,flush=True)
  try:
   p=grid.plan(start,goal);paths.append({'stage':name,'path':p.tolist(),'length_m':float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())});start=np.array(goal)
  except Exception as e:paths.append({'stage':name,'error':str(e)});break
 report={'stages':paths,'all_paths_found':len(paths)==len(stages) and all('path' in p for p in paths),'annotation_limit':'CAD semantic checkpoints manually annotated; no claim of unknown-course semantics','manifest':{k:v for k,v in manifest.items() if k!='bounds'}}
 Path('/work/cartographer/results/cad-plan.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(9,8));z=grid.idx([0,0,.8])[2]
 ax.imshow(grid.occupied[:,:,z].T,origin='lower',extent=[-3.12,3.16,-3.12,3.16],cmap='Greys',alpha=.6)
 for item in paths:
  if 'path' in item:
   p=np.array(item['path']);ax.plot(p[:,0],p[:,1],'-o',label=item['stage'])
 ax.set(xlabel='x [m]',ylabel='y [m]',title='CAD 3D A* plan at 0.8 m slice; inflated X500 envelope',aspect='equal');ax.legend();fig.savefig('/work/cartographer/results/cad-plan.png',dpi=150)
