#!/usr/bin/python3
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/work/cartographer/v4/results');out=root/'delivery';out.mkdir(exist_ok=True);p=np.load(root/'slamfix-05/slam/actual-structure-map.npz')['points'];s=json.loads((root/'slamfix-05/slam-comparison.json').read_text());judge=json.loads((root/'slamfix-05/judge.json').read_text())['result'];before=json.loads((root/'slamfix-03/slam-metrics.json').read_text())['metrics']['optimized']['rmse_m'];after=json.loads((root/'joint-imu-vio-03/slam-metrics.json').read_text())['metrics']['optimized']['rmse_m']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False});fig=plt.figure(figsize=(14,8),layout='constrained');g=fig.add_gridspec(2,2,width_ratios=[1.05,1]);ax=fig.add_subplot(g[:,0]);v=ax.scatter(p[:,0],p[:,1],c=p[:,2],s=2,cmap='turbo',rasterized=True);ax.set(xlabel='SLAM map x (m)',ylabel='SLAM map y (m)',title='Actual optimized stereo structure\nNo CAD or truth in reconstruction',aspect='equal');fig.colorbar(v,ax=ax,shrink=.65,label='SLAM map z (m)');ax.grid(alpha=.15)
ax=fig.add_subplot(g[0,1]);ax.bar(['Earlier mapping build','Final mapping build'],[before*100,after*100],color=['#c97950','#167da4'],width=.5)
for i,value in enumerate([before,after]):ax.text(i,value*100+1,f'{value*100:.1f} cm',ha='center',fontweight='bold')
ax.set(ylabel='Optimized-node RMSE (cm)',ylim=(0,36),title='Same recorded episode, independent evaluation\nFixed initial gauge; no scale or trajectory fit');ax.grid(axis='y',alpha=.2)
ax=fig.add_subplot(g[1,1]);start=judge['start_time_s']
for name,label,color,style in [('live_map','Online map-frame pose','#b87445','-'),('optimized','Final optimized structural nodes','#167da4','-')]:
 x=s[name];ax.plot(np.array(x['time'])-start,np.array(x['error'])*100,label=label,color=color,lw=1.2,ls=style)
ax.set(xlabel='Seconds from departure',ylabel='3D error (cm)',title='slamfix-05: live poses and final optimized nodes');ax.legend(fontsize=9);ax.grid(alpha=.2);fig.suptitle('No-GPS mapping repair — real recorded stereo + IMU + downward ToF',fontsize=17,fontweight='bold');fig.savefig(out/'slam-diagnostics.png',dpi=160)
