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
 now=time.monotonic();processes=[]
 for path in Path('/proc').glob('[0-9]*'):
  try:
   cmd=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   if not any(x in cmd for x in ['cartographer/v5/scripts/','ego_planner_node','traj_server','ov_msckf/lib','cartographer_node -configuration_directory /work/cartographer/v5']):continue
   if any(x in cmd for x in ['profile_processes.py','landlock_exec.py','bash -c','tmux','evaluate_exploration.py','online_view.py','slam_cloud_view.py','inspect_run.py','run_trial.py']):continue
   stat=(path/'stat').read_text().split();cpu=(int(stat[13])+int(stat[14]))/hz;pid=int(path.name);old=previous.get(pid);percent=None if old is None else (cpu-old[1])/(now-old[0])*100.;previous[pid]=(now,cpu);pss=0
   for line in (path/'smaps_rollup').read_text().splitlines():
    if line.startswith('Pss:'):pss=int(line.split()[1])*1024
   processes.append({'pid':pid,'command':cmd,'cpu_percent_one_core':percent,'pss_bytes':pss})
  except (OSError,ValueError,IndexError):pass
 rows.append({'wall_elapsed_s':now-start,'processes':processes,'combined_pss_bytes':sum(x['pss_bytes'] for x in processes)});time.sleep(.8)
Path(a.output).write_text(json.dumps({'platform':os.uname().machine,'hardware_timing_validated':False,'note':'Development host processes only. GUI, simulator and recorder excluded. Stereo CPU matching is a development proxy; onboard OAK should compute measured depth. Sum PSS avoids shared-library double counting. Target Jetson Orin Nano 8 GB still requires tegrastats, real drivers and thermal/power-mode measurements.','samples':rows},indent=2))
