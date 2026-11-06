#!/usr/bin/python3
"""Sensor-only OpenVINS comparison in isolated domain 43. Never arms PX4."""
import argparse,json,os,signal,subprocess,tempfile,time,shutil,hashlib
from pathlib import Path
import numpy as np,rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
from std_msgs.msg import Bool,Float64MultiArray
assert os.environ.get('ROS_DOMAIN_ID')=='43'
p=argparse.ArgumentParser();p.add_argument('bag');p.add_argument('output');p.add_argument('--histogram',choices=['HISTOGRAM','CLAHE','NONE'],required=True);p.add_argument('--fast-threshold',type=int,choices=[5,10],default=10);a=p.parse_args();root=Path(__file__).resolve().parents[1];out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
saved=Path(a.bag).parents[1]/'results'/Path(a.bag).name/'config/gazebo_vio';config=Path(tempfile.mkdtemp(prefix='vio-replay-'));shutil.copytree(saved,config,dirs_exist_ok=True);f=config/'estimator_config.yaml';s=f.read_text();import re
s=re.sub(r'histogram_method:.*', 'histogram_method: "'+a.histogram+'"',s);s=re.sub(r'fast_threshold:.*','fast_threshold: '+str(a.fast_threshold),s);f.write_text(s)
rclpy.init();n=Node('measured_vio_replay_monitor',parameter_overrides=[Parameter('use_sim_time',value=True)]);poses=[];health=[];ground=None;moved=False;motion=n.create_publisher(Bool,'/vio/in_motion',10)
def ranging(m):
 global ground,moved
 if not np.isfinite(m.range):return
 if ground is None:ground=m.range
 if m.range>ground+.06:moved=True
 motion.publish(Bool(data=moved))
def odom(m):
 p=m.pose.pose.position;q=m.pose.pose.orientation;poses.append([m.header.stamp.sec+m.header.stamp.nanosec*1e-9,p.x,p.y,p.z,q.x,q.y,q.z,q.w,float(max(m.pose.covariance[::7]))])
subs=[n.create_subscription(Range,'/carto/down_range',ranging,10),n.create_subscription(Odometry,'/ov_msckf/odomimu',odom,20),n.create_subscription(Float64MultiArray,'/ov_msckf/visual_update_health',lambda m:health.append(list(m.data)),20)]
logs=[];jobs=[]
def job(name,cmd):
 h=open(out.with_suffix('.'+name+'.log'),'w');logs.append(h);v=subprocess.Popen(cmd,stdout=h,stderr=subprocess.STDOUT,start_new_session=True);jobs.append(v);return v
try:
 binary=Path('/work/cartographer/v2/ros_ws/install/ov_msckf/lib/ov_msckf/run_subscribe_msckf')
 vio=job('vio',['bash','-c','source /opt/ros/jazzy/setup.bash; source /work/ros_ws/install/setup.bash; source /work/cartographer/v2/ros_ws/install/setup.bash; exec "$@"','vio-replay','python3',str(root/'scripts/landlock_exec.py'),str(binary),'--ros-args','-r','__ns:=/ov_msckf','-p','config_path:='+str(f),'-p','use_sim_time:=true','-p','processing_async:=false']);end=time.monotonic()+3
 while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.01)
 player=job('player',['ros2','bag','play',a.bag,'--clock','100','--disable-keyboard-controls','--topics','/vio/left/image_raw','/vio/right/image_raw','/vio/imu','/carto/down_range']);deadline=time.monotonic()+180
 while player.poll() is None and time.monotonic()<deadline:
  if vio.poll() is not None:raise RuntimeError('VIO process exited; inspect replay VIO log')
  rclpy.spin_once(n,timeout_sec=.01)
 if player.poll() is None:raise RuntimeError('Bounded replay deadline exceeded')
 end=time.monotonic()+1
 while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.01)
finally:
 for v in jobs:
  if v.poll() is None:os.killpg(v.pid,signal.SIGINT)
 for v in jobs:
  try:v.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(v.pid,signal.SIGTERM);v.wait(timeout=5)
 for h in logs:h.close()
 n.destroy_node();rclpy.try_shutdown()
h=np.array(health);moving=h[:,0]>next((p[0] for p in poses if p[3]>.1),h[0,0]) if len(h) else np.array([],bool);age=np.maximum(0,h[:,0]-h[:,1]) if len(h) else np.array([])
assert len(poses)>100 and len(health)>100,'No valid estimator output; replay comparison rejected'
result={'executable':str(binary),'executable_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'histogram':a.histogram,'fast_threshold':a.fast_threshold,'effective_estimator_config':s,'sensor_inputs_only':True,'range_motion_gate_seen':moved,'poses':poses,'health':health,'samples':len(h),'moving_update_age_p50_p95_max_s':np.percentile(age[moving],[50,95,100]).tolist() if moving.any() else [],'moving_visual_update_coverage_fraction':float(np.mean(age[moving]<.5)) if moving.any() else 0.};out.write_text(json.dumps(result));print(json.dumps({k:v for k,v in result.items() if k not in ['poses','health']},indent=2))
