#!/usr/bin/python3
"""Produce audit/plots from recorded evidence; never used as controller input."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/work/cartographer/results')
def read(name):
 p=root/name;return json.loads(p.read_text()) if p.exists() else None
reports=[]
for label in ['guarded-02','guarded-03','guarded-04']:
 e=read('motion-'+label+'.json');f=read('flight-'+label+'.json');j=read('verdict-'+label+'.json') or read('judge-'+label+'-verdict.json')
 if not j:
  original=read('judge-'+label+'.json');j=original.get('result') if original else None
 if not e or not f:continue
 a=np.array(e['rows']['odom']);g=np.array(e['rows']['truth']);gp=g[:,1:4]+Rotation.from_quat(g[:,4:8]).apply([.15,0,.08]);rot=Rotation.from_quat(g[0,4:8])*Rotation.from_quat(a[0,4:8]).inv();estimate=rot.apply(a[:,1:4]-a[0,1:4])+gp[0]
 truth=np.stack([np.interp(a[:,0],g[:,0],gp[:,i]) for i in range(3)],axis=1);error=np.linalg.norm(estimate-truth,axis=1)
 arm=f['arm_stamp_us']*1e-6;land=f['land_command_stamp_us']*1e-6;end=f['end_stamp_us']*1e-6;mask=(a[:,0]>=arm)&(a[:,0]<=end)
 invalid=[h['stamp'] for h in e['rows'].get('health',[]) if not h['valid'] and arm<=h['stamp']<=land]
 r={'label':label,'navigation_passed':bool(f['passed']),'automatic_disarm_observed':bool(f['disarmed']),'diagnostic_phases_reached':f['reached_waypoints'],'failure':f['failure'],'sample_window_ate_rmse_m':e['summary'].get('ate_rmse_m'),'sample_window_ate_max_m':e['summary'].get('ate_max_m'),'flight_ate_rmse_m':float(np.sqrt(np.mean(error[mask]**2))) if mask.any() else None,'flight_ate_max_m':float(error[mask].max()) if mask.any() else None,'first_invalid_to_land_command_s':float(land-min(invalid)) if invalid else None,'judge':j,'raw_bag_recorded':(root.parent/'data'/label/'metadata.yaml').exists()}
 reports.append(r)
 if label=='guarded-04':
  t0=arm;d=e['rows']['depth'];st=e['rows']['stereo'];health=e['rows'].get('health',[])
  fig,axs=plt.subplots(3,1,figsize=(11,10),sharex=True)
  axs[0].semilogy([v['stamp']-t0 for v in d],[max(v['normal_rank_ratio'],1e-12) for v in d],label='Surface-normal rank ratio');axs[0].axhline(.015,c='r',ls='--',label='Rejection threshold 0.015');axs[0].set_ylabel('Geometry ratio');axs[0].legend(loc='upper right')
  axs[1].plot([v['stamp']-t0 for v in d],[v['valid_fraction'] for v in d],label='Ideal depth valid fraction');axs[1].plot([v['stamp']-t0 for v in st],[v['valid_fraction'] for v in st],label='Passive stereo valid fraction');axs[1].set(ylim=(-.03,1.05),ylabel='Valid fraction');axs[1].legend(loc='lower right')
  axs[2].plot(a[:,0]-t0,error,label='Fixed-initial-alignment position error');axs[2].set(ylabel='3D error [m]',xlabel='Simulation seconds since arm');axs[2].legend(loc='upper right')
  for ax in axs:
   ax.axvline(0,color='green',ls=':');ax.axvline(land-t0,color='red',ls=':',label='Land command');ax.axvline(end-t0,color='black',ls=':');ax.grid(alpha=.25);ax.set_xlim(-2,min(40,a[-1,0]-t0))
  fig.suptitle('Actual no-GPS diagnostic: geometry rejection -> abort -> automatic landing\nNo race checkpoints completed; ideal depth is a simulation baseline',fontsize=13);fig.tight_layout();fig.savefig(root/'flight-diagnostics.png',dpi=160);plt.close(fig)
envelopes=read('envelope-plans.json');proxy=envelopes[-1]
# Determine whether stage four crosses the actual window wall plane.
lo=np.array([-.794496,-3.04709,0]);hi=np.array([-.147379,-1.49671,1.7272]);center=(lo+hi)/2;direction=(hi-lo)[:2];direction/=np.linalg.norm(direction);normal=np.array([direction[1],-direction[0]]);window=False
for stage in proxy['stages']:
 if stage['name']!='gate_or_window':continue
 for pa,pb in zip(stage['path'],stage['path'][1:]):
  pa=np.array(pa);pb=np.array(pb);da=normal@(pa[:2]-center[:2]);db=normal@(pb[:2]-center[:2])
  if da*db<0:
   point=pa+(pb-pa)*da/(da-db);u=direction@(point[:2]-center[:2])
   if abs(u)<np.linalg.norm((hi-lo)[:2])/2 and 0<point[2]<1.7272:window=True
proxy['inferred_stage_four_route']='window' if window else 'gate_b';proxy['bonus_awarded']=False
# Show the z changes so a projection through the window is not mistaken for a wall collision.
npz=np.load(root/'proxy-cad-grid.npz');occ=npz['occupied'];origin=npz['origin'];res=float(npz['res']);z=int(round((.8-origin[2])/res));extent=[origin[0],origin[0]+occ.shape[0]*res,origin[1],origin[1]+occ.shape[1]*res]
fig,(ax,az)=plt.subplots(1,2,figsize=(14,7),gridspec_kw={'width_ratios':[1.25,1]});ax.imshow(occ[:,:,z].T,origin='lower',extent=extent,cmap='Greys',alpha=.65)
s=0
for stage in proxy['stages']:
 p=np.array(stage['path']);line=ax.plot(p[:,0],p[:,1],'-o',ms=3,label=stage['name'])[0];distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]+s;az.plot(distance,p[:,2],'-o',ms=3,color=line.get_color(),label=stage['name']);s=distance[-1]
ax.set(xlabel='x [m]',ylabel='y [m]',aspect='equal',title='Obstacles shown at z = 0.80 m');ax.legend(fontsize=8,loc='lower left');az.set(xlabel='Path distance [m]',ylabel='Planned base height [m]',title='Full 3D collision checks; stage 4 uses window');az.grid(alpha=.25)
fig.suptitle('CAD planning only: 0.27 m radius proxy + 0.06 m margin\nNot an identified official airframe and not a flown trajectory',fontsize=13);fig.tight_layout();fig.savefig(root/'cad-plan-evidence.png',dpi=160);plt.close(fig)
summary={'overall':'NOT COMPLETE: full autonomous course not validated','flight_navigation_passed':False,'full_course_completed':False,'hardware_extra_sensors':'None beyond official downward-ToF proxy; no 360 degree lidar','simulation_trials':reports,'static_ideal':read('static-ideal.json')['summary'],'static_passive_stereo':read('static-source-stereo.json')['summary'],'depth_stop_fault':read('depth-stop-probe.json'),'judge_tests':read('judge-tests.json'),'envelope_plans':envelopes,'runtime':read('runtime-audit.json'),'limitations':['Ideal stereo pinhole and ideal rendered depth are not OAK-D S2 firmware/photometry. OpenCV SGBM is a passive-stereo diagnostic only.','Forward range geometry degenerates; valid depth coverage alone is insufficient. Cartographer covariance is not available; configured measurement noise is not an estimator uncertainty estimate.','X500 is a 2 kg surrogate; the smaller CAD envelope has no identified official airframe dynamics.','Mission controller, online obstacle veto and platform landing sequence are implemented but not validated through the full course.','Judge checkpoint regions are provisional CAD annotations, not official referee boundaries.','No dark-tunnel physical sensing validation, Orin timing validation, window flight or return bonus validation.','No previous-flight map is supplied to localization/planning; ground truth is read only by evaluators and judge.','This is pre-event research; software-development rule 9.1 and final official rules review remain outstanding.','The first guarded-03 rosbag CLI recorder stalled before writing. Its ULog/JSON exist; corrected guarded-04 MCAP is reported separately.']}
(root/'validation-summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False));print(json.dumps({'trials':reports,'proxy_route':proxy['inferred_stage_four_route'],'overall':summary['overall']},indent=2))
