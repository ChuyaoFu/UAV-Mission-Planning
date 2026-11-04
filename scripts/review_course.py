"""Inspect provided OBJ without changing it; render its actual geometry."""
from pathlib import Path
import json
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

source=Path('/Users/chuyao/Desktop/dronehacks-course-v3-obj')
out=Path('/Users/chuyao/Documents/Codex/2026-10-31/new-chat/outputs')
vertices=[];groups={};counts=Counter();material={};current_material=None
for line in (source/'dronehacks-course-v3.mtl').read_text().splitlines():
    a=line.split()
    if not a:continue
    if a[0]=='newmtl':current_material=a[1]
    if a[0]=='Kd':material[current_material]=list(map(float,a[1:4]))
group=None
for line in (source/'dronehacks-course-v3.obj').open():
    a=line.split()
    if not a:continue
    counts[a[0]]+=1
    if a[0]=='g':
        group=line[2:].strip();groups.setdefault(group,{'faces':[],'colors':[]})
    elif a[0]=='v':vertices.append(list(map(float,a[1:4])))
    elif a[0]=='usemtl':current_material=a[1]
    elif a[0]=='f':
        indices=[int(v.split('/')[0])-1 for v in a[1:]]
        groups[group]['faces'].append(indices)
        groups[group]['colors'].append(material[current_material])
v=np.asarray(vertices)
report={'source_directory':str(source),'units':'metres','up_axis':'Z','counts':dict(counts),'bounds_m':[v.min(axis=0).tolist(),v.max(axis=0).tolist()],'pieces':[],'notes':['The five net pieces are opaque solid thin boxes in the supplied geometry.','There are no texture coordinates or image texture maps.','Source README describes the layout as preliminary.']}
fig=plt.figure(figsize=(14,7.5),facecolor='white')
ax=fig.add_subplot(121);ax3=fig.add_subplot(122,projection='3d')
all_faces=[];all_colors=[]
for name,g in groups.items():
    f=np.asarray(g['faces']);p=v[f];unique=v[np.unique(f)]
    lo=unique.min(axis=0);hi=unique.max(axis=0)
    report['pieces'].append({'name':name,'triangles':len(f),'bounds_m':[lo.tolist(),hi.tolist()]})
    if name.startswith('Net'):continue
    colors=np.asarray(g['colors'])
    if not name.startswith('Floor'):
        # In top view, draw horizontal/sloping faces; side faces have zero area.
        normals=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
        mask=np.abs(normals[:,2])>1e-10
        ax.add_collection(PolyCollection(p[mask,:,:2],facecolors=colors[mask],edgecolors='none'))
    if name.startswith('Floor'):continue
    # Shade using source face geometry; preserves the source material colors.
    normals=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-12)
    light=np.array([-.4,-.6,1.]);light/=np.linalg.norm(light)
    shade=.55+.45*np.abs(np.sum(normals*light,axis=1))
    all_faces.append(p);all_colors.append(np.clip(colors*shade[:,None],0,1))

ax3.add_collection3d(Poly3DCollection(np.concatenate(all_faces),facecolors=np.concatenate(all_colors),edgecolors='none',zsort='average'))

labels={
 'M1':(-1.93,1.35),'M2':(-.81,2.3),'M3':(.30,1.35),
 'C1':(1.25,-.35),'C2a\nwindow':(-.48,-2.27),'C2b':(-.02,-1.0),
 'Barriers':(-.6,.40),'Oval tunnel':(2.30,.45),
 'Launch':(-2.51,.92),'Landing pad':(-2.04,-.79),
}
ax.set_facecolor('#faf9f4')
for label,(x,y) in labels.items():
    ax.text(x,y,label,ha='center',va='center',fontsize=8.5,color='#145b93',bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1.5))
ax.plot([-3.048,3.048,3.048,-3.048,-3.048],[-3.048,-3.048,3.048,3.048,-3.048],'--',color='#555',lw=1)
ax.set(xlim=(-3.3,3.3),ylim=(-3.3,3.3),xlabel='X (m)',ylabel='Y (m)',title='Top view | source layout')
ax.set_aspect('equal');ax.grid(alpha=.15)
ax3.set(xlim=(-3.1,3.1),ylim=(-3.1,3.1),zlim=(0,3.1),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title='Source mesh | oblique view')
ax3.set_box_aspect((6.2,6.2,3.1));ax3.view_init(elev=39,azim=-58)
fig.suptitle('DroneHacks 2026 course v3 — asset inspection',fontsize=17,y=.96)
fig.text(.5,.065,'6.096 m × 6.096 m enclosure; 3.048 m high. Roof and side nets hidden for inspection only.',ha='center',fontsize=10)
fig.text(.5,.035,'Geometry preview only — no simulator, collision model, sensors, or flight validation. Source layout is preliminary.',ha='center',fontsize=9,color='#555')
fig.subplots_adjust(left=.06,right=.98,bottom=.13,top=.88,wspace=.06)
fig.savefig(out/'dronehacks-course-preview.png',dpi=160)
Path('/Users/chuyao/Documents/Codex/2026-10-31/new-chat/work/visual_nav/course-asset-report.json').write_text(json.dumps(report,indent=2))
print('Saved asset preview; source files unchanged.')
