#!/usr/bin/env python3
"""Capture one unchanged SITL fixture, including the estimator's raw inputs."""
from pathlib import Path
import json
import shutil
import subprocess
import time

root=Path('/home/fcy/visual_nav')
label='vio-raw-diagnostic-01'
out=root/'results'/label
out.mkdir(exist_ok=False)
def docker(*args,**kw):return subprocess.run(['docker','exec','visual-nav-sim',*args],**kw)
status=docker('/work/src/PX4-Autopilot/build/px4_sitl_default/bin/px4-commander','status',capture_output=True,text=True)
assert 'Disarmed' in status.stdout,status.stdout
docker('bash','/work/scripts/stop-sim.sh',check=True)
docker('env','NAV_WORLD=dronehacks_textured','NAV_ESTIMATOR=vision','bash','/work/scripts/start-sim.sh',check=True)
bag='/work/data/'+label
topics=['/clock','/vio/left/image_raw','/vio/right/image_raw','/vio/imu',
    '/evaluation/poses','/ov_msckf/poseimu','/ov_msckf/odomimu',
    '/ov_msckf/points_msckf','/ov_msckf/points_slam',
    '/fmu/out/vehicle_local_position_v1','/fmu/out/vehicle_status_v1',
    '/fmu/in/vehicle_visual_odometry','/fmu/in/trajectory_setpoint']
handles=[];jobs=[]
def job(name,args):
    h=open(out/name,'w');handles.append(h)
    p=subprocess.Popen(['docker','exec','-e','GZ_PARTITION=visual_nav_sim','visual-nav-sim',*args],stdout=h,stderr=subprocess.STDOUT)
    jobs.append(p)
    return p
rec=job('record.log',['bash','-c','source /opt/ros/jazzy/setup.bash; source /work/ros_ws/install/setup.bash; exec timeout --signal=INT --kill-after=8 68 ros2 bag record --storage mcap --storage-preset-profile zstd_fast --use-sim-time --disable-keyboard-controls --max-cache-size 134217728 -o '+bag+' --topics '+' '.join(topics)])
time.sleep(2)
assert rec.poll() is None,'Recorder exited before flight'
docker('bash','/work/scripts/start-vio.sh',check=True)
docker('tmux','new-session','-d','-s','nav-monitor','exec /opt/px4-venv/bin/python3 /work/scripts/gcs-monitor.py > /work/logs/monitor.log 2>&1',check=True)
docker('tmux','new-session','-d','-s','nav-ev','source /opt/ros/jazzy/setup.bash; source /work/ros_ws/install/setup.bash; exec /usr/bin/python3 /work/scripts/vio-to-px4.py > /work/logs/ev-adapter.log 2>&1',check=True)
time.sleep(8)
for name in ['baseline-flight.json','vehicle-contacts.json']:(root/'results'/name).unlink(missing_ok=True)
shutil.rmtree(root/'results/vio-evaluation',ignore_errors=True)
job('contacts.log',['/usr/bin/python3','/work/scripts/monitor-contacts.py','--seconds','55'])
job('evaluation.log',['bash','-c',f'source /opt/ros/jazzy/setup.bash; exec /usr/bin/python3 /work/scripts/evaluate-vio.py --seconds 55 --label {label}'])
time.sleep(2)
print('Raw recording active; starting unchanged short VIO fixture',flush=True)
with open(out/'flight.log','w') as h:
    flight=docker('bash','-c','source /opt/ros/jazzy/setup.bash; source /work/ros_ws/install/setup.bash; export NAV_TEST_REQUIRE_VISION=1 NAV_TEST_BODY_FORWARD=1; exec /usr/bin/python3 /work/scripts/flight-baseline.py',stdout=h,stderr=subprocess.STDOUT,timeout=100)
print('Flight finished',flight.returncode,flush=True)
for p in jobs:p.wait(timeout=80)
for h in handles:h.close()
for name in ['baseline-flight.json','vehicle-contacts.json']:
    if (root/'results'/name).exists():shutil.copy2(root/'results'/name,out/name)
shutil.copytree(root/'results/vio-evaluation',out/'vio-evaluation',dirs_exist_ok=True)
shutil.copytree(root/'config/gazebo_vio',out/'config',dirs_exist_ok=True)
for name in ['openvins.log','px4-run.log','ev-adapter.log']:shutil.copy2(root/'logs'/name,out/name)
logs=list((root/'src/PX4-Autopilot/build/px4_sitl_default/rootfs/log').glob('*/*.ulg'))
shutil.copy2(max(logs,key=lambda p:p.stat().st_mtime),out/'flight.ulg')
(out/'capture.json').write_text(json.dumps({'label':label,'flight_exit_code':flight.returncode,'bag':bag,'topics':topics,'purpose':'fixed raw stereo and IMU replay; no estimator parameters changed'},indent=2))
print('DONE',out,flush=True)
