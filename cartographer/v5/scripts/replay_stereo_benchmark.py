#!/usr/bin/python3
"""Isolated, bounded replay. No PX4/flight topics are replayed or commanded."""
import argparse,os,subprocess,signal,time,json
from pathlib import Path
assert os.environ.get('ROS_DOMAIN_ID')=='43','Replay is restricted to domain 43'
p=argparse.ArgumentParser();p.add_argument('bag');p.add_argument('output');a=p.parse_args();root=Path(__file__).parent;out=Path(a.output);out.unlink(missing_ok=True);jobs=[];handles=[];saved=Path(a.bag).parents[1]/'results'/Path(a.bag).name/'config';os.environ['STEREO_CALIBRATION']=str(saved/'gazebo_vio/kalibr_imucam_chain.yaml');os.environ['ROBOT_CALIBRATION']=str(saved/'robot.json')
def job(name,args):
 h=open(out.parent/(name+'.log'),'w');handles.append(h);proc=subprocess.Popen(args,stdout=h,stderr=subprocess.STDOUT,start_new_session=True);jobs.append(proc);return proc
try:
 job('rtab-frames',['python3',str(root/'rtab_stereo_frames.py')]);job('rtab-odometry',['python3',str(root/'rtab_odom_launch.py')]);job('rtab-recorder',['python3',str(root/'record_rtab.py'),str(out),'100']);time.sleep(3);player=job('rtab-player',['ros2','bag','play',a.bag,'--clock','100','--disable-keyboard-controls','--topics','/vio/left/image_raw','/vio/right/image_raw','/vio/imu']);player.wait(timeout=95);time.sleep(2)
finally:
 for proc in jobs:
  if proc.poll() is None:os.killpg(proc.pid,signal.SIGINT)
 for proc in jobs:
  try:proc.wait(timeout=12)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
 for h in handles:h.close()
d=json.loads(out.read_text());assert len(d['poses'])>100,'Insufficient measured VO output';print('Recorded pose samples',len(d['poses']))
