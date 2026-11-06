#!/usr/bin/python3
"""Recorded SITL-only exploration trial; truth remains evaluator-only."""
import argparse,atexit,fcntl,hashlib,json,os,re,shutil,signal,subprocess,time
from pathlib import Path
from window_video import WindowVideo
ROOT=Path('/work/cartographer/v5');PX=Path('/work/src/PX4-Autopilot/build/px4_sitl_default/bin')
p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--seconds',type=int,default=180);p.add_argument('--mission',choices=['navigation_check','course_corridor'],default='navigation_check');p.add_argument('--cancel-after',type=float,help='SITL controlled-cancellation regression, wall seconds after navigator launch');a=p.parse_args()
assert Path('/.dockerenv').exists() and os.environ.get('ROS_DOMAIN_ID')=='42'
assert a.label and all(c.isalnum() or c=='-' for c in a.label)
lock=open('/work/data/flight-smoke.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
out=ROOT/'results'/a.label;out.mkdir(parents=True,exist_ok=False);(ROOT/'mission-output').mkdir(exist_ok=True)
setup='source /opt/ros/jazzy/setup.bash; source /work/ros_ws/install/setup.bash; source /work/cartographer/v3/carto_ws/install/setup.bash; source /work/cartographer/v5/ego_ws/install/setup.bash; '
def archive_context():
 # Preserve diagnostics even if preflight or the runner itself exits early.
 for f in (ROOT/'logs').glob('*'):
  if f.is_file():
   try:shutil.copy2(f,out/f.name)
   except OSError:pass
 path=ROOT/'mission-output'/(a.label+'.json')
 if path.exists():shutil.copy2(path,out/'flight.json')
atexit.register(archive_context)

def native_status():return subprocess.check_output([str(PX/'px4-listener'),'vehicle_status','-n','1'],text=True,timeout=8)
assert 'arming_state: 1' in native_status(),'Confirmed disarm required before a trial'
for key in ['EKF2_GPS_CTRL','EKF2_EV_CTRL']:
 text=subprocess.check_output([str(PX/'px4-param'),'show',key],text=True);(out/(key+'.txt')).write_text(text)
 if key=='EKF2_GPS_CTRL':assert re.search(r'EKF2_GPS_CTRL.*?:\s*0\b',re.sub(r'\x1b\[[0-9;]*m','',text),re.S),'GPS fusion must be disabled'
subprocess.run(['python3',str(ROOT/'scripts/landlock_exec.py'),'--probe'],stdout=open(out/'filesystem-isolation.json','w'),check=True)
audit_flags=['--require-guideline'] if a.mission=='course_corridor' else []
subprocess.run(['python3',str(ROOT/'scripts/audit_navigation.py'),str(out/'input-audit.json'),*audit_flags],stdout=open(out/'input-audit.log','w'),check=True)
shutil.copytree(ROOT/'config',out/'config',ignore=shutil.ignore_patterns('._*'));shutil.copytree(ROOT/'scripts',out/'scripts',ignore=shutil.ignore_patterns('__pycache__','._*'))
(out/'source-hashes.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['scripts','config'] for f in (ROOT/folder).rglob('*') if f.is_file() and '__pycache__' not in str(f) and not f.name.startswith('._')},indent=2))
(out/'trial-arguments.json').write_text(json.dumps(vars(a),indent=2))
runtime={}
for folder in [ROOT/'ego_ws/src',ROOT/'ego_ws/build/ego_planner',ROOT/'ego_ws/build/plan_env',ROOT/'ego_ws/build/bspline_opt']:
 for f in folder.rglob('*'):
  if f.is_file() and (f.suffix in ['.cpp','.h','.hpp','.so'] or f.name in ['ego_planner_node','traj_server']):runtime[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
(out/'runtime-source-hashes.json').write_text(json.dumps(runtime,indent=2))
if (ROOT/'evaluation/scenario.json').exists():shutil.copy2(ROOT/'evaluation/scenario.json',out/'scenario-evaluation-only.json')
if (ROOT/'evaluation/active-world.sdf').exists():
 shutil.copy2(ROOT/'evaluation/active-world.sdf',out/'evaluation-world.sdf')
 (out/'evaluation-world-sha256.txt').write_text(hashlib.sha256((out/'evaluation-world.sdf').read_bytes()).hexdigest()+'\n')
wall_budget=a.seconds*3+90;runner_failure=None
jobs=[];handles=[]
def job(log,cmd):
 h=open(out/log,'w');handles.append(h);proc=subprocess.Popen(['bash','-c',setup+'exec '+cmd],stdout=h,stderr=subprocess.STDOUT,start_new_session=True);jobs.append(proc);return proc
bag=ROOT/'data'/a.label;bag.parent.mkdir(exist_ok=True)
topics='/fmu/in/vehicle_command /fmu/out/vehicle_command_ack /grid_map/pipeline_health /fmu/out/vehicle_land_detected /carto/accepted_odom /nav/depth_evidence /nav/planar_health /mapping/global_cloud /trajectory_node_list /nav/guideline_hint /stereo/depth/image /mapping/points /mapping/structural_points /clock /vio/left/image_raw /vio/right/image_raw /vio/imu /stereo/health /carto/depth_health /mapping/health /carto/down_range /carto/localization_health /carto/local_pose /nav/body_odom /nav/camera_pose /nav/exploration_goal /grid_map/knowledge /grid_map/metadata /position_cmd /planning/bspline /carto/mission_state /ov_msckf/odomimu /ov_msckf/visual_update_health /tf /tf_static /tracked_pose /submap_list /constraint_list /fmu/out/vehicle_status_v1 /fmu/out/vehicle_local_position_v1 /fmu/in/vehicle_visual_odometry /fmu/in/trajectory_setpoint'
rec=job('record.log',f'ros2 bag record --storage mcap --storage-preset-profile zstd_fast --use-sim-time --disable-keyboard-controls --max-cache-size 134217728 -o {bag} --topics {topics}')
for _ in range(20):
 if list(bag.glob('*.mcap')):break
 assert rec.poll() is None,'Recorder failed; no arming';time.sleep(.5)
else:raise RuntimeError('MCAP not created; no arming')
video=WindowVideo(out/'videos',a.label);atexit.register(video.stop)
evaluator=job('evaluation.log',f'python3 {ROOT}/scripts/evaluate_exploration.py --seconds {wall_budget+60} --output {out}/evaluation.json --ready-file {out}/evaluation-ready.json')
profile=job('profile.log',f'python3 {ROOT}/scripts/profile_processes.py --seconds {wall_budget+60} --output {out}/process-budget.json')
for _ in range(60):
 if (out/'evaluation-ready.json').exists():break
 assert evaluator.poll() is None,'Evaluator failed before arming';time.sleep(.1)
else:raise RuntimeError('Independent contacts/pose/status not ready; no arming')
mission=ROOT/'mission-output'/(a.label+'.json');flight=job('flight.log',f'env SITL_AUTOMATION_AUTHORIZED=1 python3 {ROOT}/scripts/landlock_exec.py python3 {ROOT}/scripts/navigate.py --output {mission} --mission {a.mission}')
audit=job('flight-audit.log',f'python3 {ROOT}/scripts/audit_navigation.py {out}/flight-input-audit.json --require-navigator {" ".join(audit_flags)}')
try:code=flight.wait(timeout=wall_budget if a.cancel_after is None else min(wall_budget,a.cancel_after))
except subprocess.TimeoutExpired:
 # SIGINT unwinds the navigator through its landing cleanup; never reset an armed simulator.
 runner_failure='Cancellation regression requested' if a.cancel_after is not None else 'Wall-clock watchdog cancellation';os.killpg(flight.pid,signal.SIGINT)
 try:code=flight.wait(timeout=90)
 except subprocess.TimeoutExpired:
  subprocess.run([str(PX/'px4-commander'),'land'],check=True,timeout=8);code=flight.wait(timeout=60)
status=native_status();(out/'final-status.txt').write_text(status)
if 'arming_state: 1' not in status:
 runner_failure=runner_failure or 'Navigator ended while armed; SITL landing fallback used'
 subprocess.run([str(PX/'px4-commander'),'land'],check=True,timeout=8);deadline=time.monotonic()+60
 while time.monotonic()<deadline:
  time.sleep(1);status=native_status()
  if 'arming_state: 1' in status:break
 (out/'final-status.txt').write_text(status)
 if 'arming_state: 1' not in status:raise RuntimeError('Still armed after landing fallback; reset prohibited')
time.sleep(3)
for proc in jobs:
 if proc.poll() is None:os.killpg(proc.pid,signal.SIGINT)
for proc in jobs:
 try:proc.wait(timeout=20)
 except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=5)
video_info=video.stop()
for h in handles:h.close()
subprocess.run(['python3',str(ROOT/'scripts/capture_map.py'),str(out)],stdout=open(out/'map-export.log','w'),stderr=subprocess.STDOUT,check=True)
if mission.exists():shutil.copy2(mission,out/'flight.json')
for f in (ROOT/'logs').glob('*'):
 if f.is_file():shutil.copy2(f,out/f.name)
subprocess.run(['bash','-c',setup+f'python3 {ROOT}/scripts/export_fpv_video.py {bag} {out}/videos/fpv.mp4'],stdout=open(out/'fpv-export.log','w'),stderr=subprocess.STDOUT,check=True)
result=json.loads((out/'evaluation.json').read_text())['result'] if (out/'evaluation.json').exists() else {'engineering_complete':False,'failure':'Independent evaluation unavailable'};controller=json.loads((out/'flight.json').read_text()) if (out/'flight.json').exists() else {'controller_complete':False,'failure':'Navigator ended without a result'};graph=json.loads((out/'flight-input-audit.json').read_text()) if (out/'flight-input-audit.json').exists() else {'valid':False};success=bool(not runner_failure and result['engineering_complete'] and controller['controller_complete'] and graph['valid'])
(out/'run.json').write_text(json.dumps({'label':a.label,'mission_profile':a.mission,'competition_tasks_complete':False,'success':success,'runner_failure':runner_failure,'independent':result,'controller_complete':controller['controller_complete'],'input_audit_valid':graph['valid'],'disarmed':True,'exit_code':code,'bag':str(bag),'videos':{'screen':str(out/'videos/flight-and-slam.mp4'),'fpv':str(out/'videos/fpv.mp4'),'screen_probe':video_info},'scope':'CAD-free exploration/avoidance/return only; not full competition or target-hardware validation'},indent=2));print((out/'run.json').read_text())
