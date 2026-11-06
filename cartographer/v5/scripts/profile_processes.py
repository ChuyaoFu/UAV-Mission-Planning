#!/usr/bin/python3
"""Read-only process budgets. Desktop measurements never stand in for Orin."""
import os,time,json,argparse,signal
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=180);p.add_argument('--output',required=True);a=p.parse_args();rows=[];previous={};stop=False;hz=os.sysconf('SC_CLK_TCK')
def interrupt(*_):
 global stop
 stop=True
signal.signal(signal.SIGINT,interrupt);start=time.monotonic()
while time.monotonic()-start<a.seconds and not stop:
 now=time.monotonic();processes=[];inventory={};selected=set()
 for path in Path('/proc').glob('[0-9]*'):
  try:
   cmd=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   stat=(path/'stat').read_text().rsplit(')',1)[1].split();pid=int(path.name);inventory[pid]=(path,cmd,int(stat[1]),(int(stat[11])+int(stat[12]))/hz)
   seed=any(x in cmd for x in ['cartographer/v5/scripts/','grabcut-ros2-node/structural_evidence_node.py','ego_planner_node','traj_server','ov_msckf/lib','cartographer_node -configuration_directory /work/cartographer/v5'])
   excluded=any(x in cmd for x in ['profile_processes.py','landlock_exec.py','bash -c','tmux','evaluate_exploration.py','online_view.py','slam_cloud_view.py','inspect_run.py','run_trial.py','run_competition_trial.py','record_competition.py','score_competition.py','capture_map.py','audit_navigation.py'])
   if seed and not excluded:
    domain=os.environ.get('ROS_DOMAIN_ID')
    process_env=(path/'environ').read_bytes().split(b'\0')
    if domain is None or ('ROS_DOMAIN_ID='+domain).encode() in process_env:selected.add(pid)
  except (OSError,ValueError,IndexError):pass
 # Spawned inference workers often have only `python -c ...spawn_main` in
 # cmdline. Include descendants of NAV processes so their memory/CPU counts.
 for _ in range(16):
  children={pid for pid,(_,_,parent,_) in inventory.items() if parent in selected}
  if children<=selected:break
  selected|=children
 for pid in selected:
  try:
   path,cmd,parent,cpu=inventory[pid];old=previous.get(pid);percent=None if old is None or cpu<old[1] else (cpu-old[1])/(now-old[0])*100.;previous[pid]=(now,cpu);pss=0
   for line in (path/'smaps_rollup').read_text().splitlines():
    if line.startswith('Pss:'):pss=int(line.split()[1])*1024
   processes.append({'pid':pid,'command':cmd,'cpu_percent_one_core':percent,'pss_bytes':pss})
  except (OSError,ValueError,IndexError):pass
 rows.append({'wall_elapsed_s':now-start,'processes':processes,'combined_pss_bytes':sum(x['pss_bytes'] for x in processes)});time.sleep(.8)
Path(a.output).write_text(json.dumps({'platform':os.uname().machine,'hardware_timing_validated':False,'includes_navigation_descendants':True,'note':'Development host processes only, including spawned NAV inference workers. GUI, simulator, trial orchestration and independent recorders excluded. Stereo CPU matching is a development proxy; onboard OAK should compute measured depth. Sum PSS avoids shared-library double counting. Target Jetson Orin Nano 8 GB still requires tegrastats, real drivers and thermal/power-mode measurements.','samples':rows},indent=2))
