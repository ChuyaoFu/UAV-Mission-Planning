#!/usr/bin/python3
"""Launch reproducible SITL trials, verify recording and final disarm."""
import argparse,datetime,json,os,shutil,subprocess,time,hashlib
from pathlib import Path
root=Path('/work/cartographer/v2');px=Path('/work/src/PX4-Autopilot/build/px4_sitl_default/bin');ap=argparse.ArgumentParser();ap.add_argument('label');ap.add_argument('--mission',action='store_true');ap.add_argument('--seconds',type=int,default=200);a=ap.parse_args()
assert all(c.isalnum() or c=='-' for c in a.label);assert Path('/.dockerenv').exists() and os.environ.get('ROS_DOMAIN_ID')=='42'
out=root/'results'/a.label;out.mkdir(exist_ok=False);duration=a.seconds
shutil.copytree(root/'scripts',out/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
(out/'source-hashes.json').write_text(json.dumps({str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['scripts','config'] for f in (root/folder).rglob('*') if f.is_file() and '__pycache__' not in str(f)},indent=2))
status=subprocess.check_output([str(px/'px4-listener'),'vehicle_status','-n','1'],text=True);assert 'arming_state: 1' in status,'Must be confirmed disarmed'
(root/'data').mkdir(exist_ok=True);bag=root/'data'/a.label
setup='source /opt/ros/jazzy/setup.bash; source /work/ros_ws/install/setup.bash; '
jobs=[];handles=[]
def job(log,command):
 h=open(out/log,'w');handles.append(h);p=subprocess.Popen(['bash','-c',setup+command],stdout=h,stderr=subprocess.STDOUT);jobs.append(p);return p
# Keep depth/stereo quality and actual visual update count along with raw sensor data.
topics='/clock /vio/left/image_raw /vio/right/image_raw /vio/imu /carto/points /carto/depth_health /carto/localization_health /carto/down_range /carto/local_pose /carto/mission_state /cad/localization_health /line/localization_health /window/localization_health /ov_msckf/odomimu /ov_msckf/points_msckf /ov_msckf/visual_update_health /tf /tracked_pose /submap_list /evaluation/poses /fmu/out/vehicle_status_v1 /fmu/out/vehicle_local_position_v1 /fmu/in/vehicle_visual_odometry /fmu/in/vehicle_command /fmu/in/trajectory_setpoint'
rec=job('record.log',f'exec timeout --signal=INT --kill-after=8 {duration+15} ros2 bag record --storage mcap --storage-preset-profile zstd_fast --use-sim-time --disable-keyboard-controls --max-cache-size 134217728 -o {bag} --topics {topics}')
for _ in range(20):
 if list(bag.glob('*.mcap')):break
 assert rec.poll() is None,'Recorder failed before flight';time.sleep(1)
else:raise RuntimeError('Recorder did not create MCAP: did not arm')
job('evaluation.log',f'exec /usr/bin/python3 {root}/scripts/capture_localization.py --seconds {duration} --output {out}/evaluation.json')
job('judge.log',f'export GZ_PARTITION=visual_nav_sim; exec /usr/bin/python3 {root}/scripts/judge.py --record-seconds {duration} --output {out}/judge.json')
command=(f'/usr/bin/python3 {root}/scripts/mission.py --execute --output {out}/flight.json' if a.mission else f'CARTO_FLIGHT_RESULT={out}/flight.json NAV_TEST_REQUIRE_VISION=1 NAV_TEST_BODY_FORWARD=1 /usr/bin/python3 {root}/scripts/flight_diagnostic.py')
f=job('flight.log','exec '+command if a.mission else command);code=f.wait(timeout=duration+45)
status=''
for _ in range(35):
 status=subprocess.check_output([str(px/'px4-listener'),'vehicle_status','-n','1'],text=True)
 if 'arming_state: 1' in status:break
 time.sleep(1)
(out/'final-status.txt').write_text(status)
if 'arming_state: 1' not in status:raise RuntimeError('Still armed: reset prohibited')
# End subscriptions/recorder cleanly; allow judge to capture post-disarm contacts.
time.sleep(3)
for p in jobs:
 if p.poll() is None:p.send_signal(2)
for p in jobs:
 try:p.wait(timeout=15)
 except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=5)
for h in handles:h.close()
# Evaluator currently writes at duration; process interrupt cleanup may omit JSON.
for f in (root/'logs').glob('*'):
 if f.is_file():shutil.copy2(f,out/f.name)
logs=list((root/'px4-rootfs/log').glob('*/*.ulg'))
if logs:shutil.copy2(max(logs,key=lambda p:p.stat().st_mtime),out/'flight.ulg')
shutil.copytree(root/'config',out/'config',dirs_exist_ok=True)
subprocess.run(['bash','-c',setup+f'ros2 bag info {bag} > {out}/bag-info.txt'],check=True)
(out/'run.json').write_text(json.dumps({'label':a.label,'mission':a.mission,'exit_code':code,'disarmed':True,'bag':str(bag),'duration_bound_s':duration},indent=2));print((out/'run.json').read_text())
