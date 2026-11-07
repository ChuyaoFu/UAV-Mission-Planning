#!/usr/bin/env python3
"""Summarize estimated-only replay, including absent/uninitialized/partial artifacts."""
import argparse,csv,json,re,math
from pathlib import Path
import numpy as np

def distribution(values):
 raw=np.asarray(values,float);v=raw[np.isfinite(raw)]
 out={'n':len(v),'nonfinite_excluded':int(len(raw)-len(v))}
 if len(v):out.update(min=float(v.min()),median=float(np.median(v)),p95=float(np.percentile(v,95)),max=float(v.max()))
 return out

def read_csv(root,name,required,convert,issues):
 path=root/name;rows=[]
 if not path.exists():issues[name]={'state':'missing','invalid_rows':0};return rows
 if path.stat().st_size==0:issues[name]={'state':'empty','invalid_rows':0};return rows
 rejected=0;examples=[]
 with path.open(newline='',errors='replace') as f:
  reader=csv.DictReader(f)
  if not reader.fieldnames or not set(required)<=set(reader.fieldnames):issues[name]={'state':'invalid_header','invalid_rows':0};return rows
  try:
   for row in reader:
    try:
     if None in row or any(row.get(k) in [None,''] for k in required):raise ValueError('incomplete or extra columns')
     parsed=convert(row);parsed['_csv_line']=reader.line_num;rows.append(parsed)
    except (ValueError,TypeError,KeyError,OverflowError) as e:
     rejected+=1
     if len(examples)<5:examples.append({'line':reader.line_num,'error':type(e).__name__})
  except csv.Error as e:rejected+=1;examples.append({'line':reader.line_num,'error':'csv.Error'})
 issues[name]={'state':'records' if rows else 'header_only' if rejected==0 else 'no_valid_rows','valid_rows':len(rows),'invalid_rows':rejected,'examples':examples}
 return rows

def summarize(root):
 root=Path(root);issues={};log_path=root/'plvins.log';log=log_path.read_text(errors='replace') if log_path.exists() else ''
 issues['plvins.log']={'state':'present' if log else 'empty' if log_path.exists() else 'missing'}
 metrics=[]
 pattern=r'PLVINS_METRICS stamp_ns=(\d+) solver=(\d+) points=(\d+) lines=(\d+) eligible_points=(\d+) usable_lines=(\d+) processing_ms=([^\s\x1b]+)'
 names=['stamp_ns','solver','points','lines','eligible_points','usable_lines','processing_ms']
 malformed=0
 for m in re.finditer(pattern,log):
  try:metrics.append(dict(zip(names,[int(x) for x in m.groups()[:-1]]+[float(m.group(7))])))
  except ValueError:malformed+=1
 malformed+=max(0,log.count('PLVINS_METRICS')-len(metrics)-malformed)
 def odom_row(r):
  return {'stamp_ns':int(r['stamp_ns']),**{k:float(r[k]) for k in ['x','y','z','qx','qy','qz','qw']}}
 def resource_row(r):return {'executable':r['executable'],**{k:float(r[k]) for k in ['cpu_percent','rss_bytes','pss_bytes']}}
 def feature_row(r):
  if r['kind'] not in ['points','lines']:raise ValueError('unknown frontend kind')
  return {'kind':r['kind'],'stamp_ns':int(r['stamp_ns']),'count':int(r['count'])}
 odom=read_csv(root,'odometry.csv',['stamp_ns','x','y','z','qx','qy','qz','qw'],odom_row,issues)
 resources=read_csv(root,'resources.csv',['executable','cpu_percent','rss_bytes','pss_bytes'],resource_row,issues)
 frontend=read_csv(root,'frontend.csv',['kind','stamp_ns','count'],feature_row,issues)
 # Subtract integer ns before converting to seconds, preserving close stamps.
 gaps=[(b['stamp_ns']-a['stamp_ns'])*1e-9 for a,b in zip(odom,odom[1:])]
 steps=[];omitted_steps=0;omitted_malformed=0
 for a,b in zip(odom,odom[1:]):
  if b['_csv_line']!=a['_csv_line']+1:omitted_malformed+=1;continue
  p=np.array([a[k] for k in ['x','y','z']]);q=np.array([b[k] for k in ['x','y','z']])
  if np.isfinite(p).all() and np.isfinite(q).all():steps.append(float(np.linalg.norm(q-p)))
  else:omitted_steps+=1
 nonlinear=[m for m in metrics if m['solver']==1];initialization_logs=log.count('Initialization finish!')
 if nonlinear or initialization_logs or odom:
  state='initialization_observed' if odom else 'initialization_observed_without_odometry_samples'
 elif metrics:state='initialization_not_observed'
 elif log or frontend:state='estimator_updates_not_observed'
 else:state='insufficient_artifacts'
 event_strings={'not_enough_features_or_parallax':'Not enough features or parallax','imu_excitation_advisory':'IMU excitation not enouth','solve_gravity_failed':'solve g failed','visual_imu_misalignment':'misalign visual structure with IMU','little_features':'little feature','large_acc_bias':'big IMU acc bias','large_gyro_bias':'big IMU gyr bias','failure':'failure detection!','system_reboot':'system reboot!'}
 quat_norms=[math.hypot(*(r[k] for k in ['qx','qy','qz','qw'])) for r in odom if all(math.isfinite(r[k]) for k in ['qx','qy','qz','qw'])]
 result={'schema':3,'observed_state':state,'artifacts':issues,'processed_frames':len(metrics),'malformed_metric_records':malformed,'nonlinear_frames':len(nonlinear),'first_nonlinear_stamp_ns':nonlinear[0]['stamp_ns'] if nonlinear else None,'initialization_log_events':initialization_logs,'failure_log_events':log.count('failure detection!'),'system_reboot_log_events':log.count('system reboot!'),'log_events':{k:log.count(v) for k,v in event_strings.items()},'process_died_log_lines':sum('process[' in line and 'has died' in line for line in log.splitlines()),'metrics':{name:distribution([m[name] for m in metrics]) for name in names[2:]},'usable_lines_nonzero_frames':sum(m['usable_lines']>0 for m in metrics),'odometry_samples':len(odom),'odometry_gap_s':distribution(gaps),'odometry_step_m':distribution(steps),'odometry_steps_omitted_nonfinite_positions':omitted_steps,'odometry_steps_omitted_malformed_rows':omitted_malformed,'odometry_nonmonotonic_stamps':sum(g<=0 for g in gaps),'odometry_nonfinite_values':sum(not math.isfinite(r[k]) for r in odom for k in ['x','y','z','qx','qy','qz','qw']),'odometry_quaternion_norm':distribution(quat_norms),'frontend':{kind:distribution([r['count'] for r in frontend if r['kind']==kind]) for kind in ['points','lines']},'resources':{exe:{k:distribution([r[k] for r in resources if r['executable']==exe and r[k]>=0]) for k in ['cpu_percent','rss_bytes','pss_bytes']} for exe in sorted({r['executable'] for r in resources})},'interpretation':'Observed state is diagnostic evidence, not successful mission or accuracy. Missing/empty files differ from zero observed odometry. eligible_points/usable_lines are post-update predicates, not exact Ceres residual count. IMU excitation string alone is advisory in the pinned source. Nonfinite values are counted/excluded from distributions; steps do not bridge invalid parsed rows. CPU%100 equals one logical core. No GT input.'}
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('result',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
 if not a.result.is_dir():p.error('Result directory does not exist')
 r=summarize(a.result);out=json.dumps(r,indent=2,allow_nan=False)+'\n';(a.output or a.result/'summary.json').write_text(out);print(out,end='')
if __name__=='__main__':main()
