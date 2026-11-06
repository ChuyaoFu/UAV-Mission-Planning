#!/usr/bin/python3
"""Domain-43 same-bag internal DDS/synchronizer A/B. No vehicle commands.

Only replays measured depth and accepted exposure/body poses. The independent
reference/evaluation streams and recorded goals/commands are never played.
"""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile,DurabilityPolicy
from std_msgs.msg import String,Int8MultiArray,Float64MultiArray

assert os.environ.get('ROS_DOMAIN_ID')=='43','Replay is isolated from live flight'
p=argparse.ArgumentParser();p.add_argument('bag');p.add_argument('output');p.add_argument('--offset',type=float,default=35.);p.add_argument('--seconds',type=float,default=15.);p.add_argument('--repeats',type=int,default=1);p.add_argument('--cases',nargs='+',choices=['be20','be4','reliable4'],default=['be4','reliable4']);p.add_argument('--include-inferred',action='store_true');a=p.parse_args()
assert a.seconds>0 and a.offset>=0 and 1<=a.repeats<=5
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
qosfile=out.with_suffix('.qos.yaml')
qosfile.write_text('/stereo/depth/image:\n  reliability: reliable\n  history: keep_last\n  depth: 4\n/nav/depth_evidence:\n  reliability: reliable\n  history: keep_last\n  depth: 2\n')
rows=[]
for repeat in range(a.repeats):
    for case in a.cases:
        rclpy.init();n=Node('measured_map_pipeline_comparison',parameter_overrides=[Parameter('use_sim_time',value=True)])
        health=[];counts={'metadata_received':0,'knowledge_received':0,'max_inferred_free_voxels':0};updates=[]
        latched=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        def receive_health(m):health.append(json.loads(m.data))
        def meta(m):counts['metadata_received']+=1;updates.append([n.get_clock().now().nanoseconds*1e-9,float(m.data[8])])
        def knowledge(m):
            counts['knowledge_received']+=1
            counts['max_inferred_free_voxels']=max(counts['max_inferred_free_voxels'],int((np.asarray(m.data,dtype=np.int8)==-3).sum()))
        subs=[n.create_subscription(String,'/grid_map/pipeline_health',receive_health,4),n.create_subscription(Float64MultiArray,'/grid_map/metadata',meta,latched),n.create_subscription(Int8MultiArray,'/grid_map/knowledge',knowledge,latched)]
        env=dict(os.environ,USE_SIM_TIME='1',DISABLE_VERIFIED_LAUNCH='1',EGO_DEPTH_RELIABLE='1' if case=='reliable4' else '0',EGO_DEPTH_QUEUE='20' if case=='be20' else '4')
        logs=[];jobs=[]
        def job(name,cmd):
            log=open(out.with_suffix(f'.{repeat}-{case}-{name}.log'),'w');logs.append(log)
            proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);jobs.append(proc);return proc
        try:
            planner=job('mapper',['python3',str(Path(__file__).with_name('ego_launch.py'))])
            deadline=time.monotonic()+3
            while time.monotonic()<deadline:
                if planner.poll() is not None:raise RuntimeError('Mapper launch failed')
                rclpy.spin_once(n,timeout_sec=.02)
            player=job('player',['ros2','bag','play',a.bag,'--clock','100','--disable-keyboard-controls','--start-offset',str(a.offset),'--playback-duration',str(a.seconds),'--qos-profile-overrides-path',str(qosfile),'--topics','/stereo/depth/image','/nav/camera_pose','/nav/body_odom',*(['/nav/depth_evidence'] if a.include_inferred else []),'--remap','/stereo/depth/image:=/nav/measured_depth'])
            deadline=time.monotonic()+a.seconds*3+20
            while player.poll() is None and time.monotonic()<deadline:
                if planner.poll() is not None:raise RuntimeError('Mapper exited during replay')
                rclpy.spin_once(n,timeout_sec=.01)
            if player.poll() is None:raise RuntimeError('Bounded replay timeout')
            if player.returncode:raise RuntimeError('Bag player failed: '+str(player.returncode))
            deadline=time.monotonic()+1.5
            while time.monotonic()<deadline:rclpy.spin_once(n,timeout_sec=.02)
            if not health:raise RuntimeError('No pipeline health; instrumentation missing')
            row={'case':case,'repeat':repeat,'observer_counts':counts,'pipeline_final':health[-1],'pipeline_history':health,'received_metadata_times':updates}
            rows.append(row)
            out.write_text(json.dumps({'scope':'Same measured bag; no goals, CAD, truth or vehicle commands; desktop transport test only','bag':a.bag,'includes_recorded_inferred_free':a.include_inferred,'offset_s':a.offset,'duration_s':a.seconds,'rows':rows},indent=2)+'\n')
            print(json.dumps({'case':case,'repeat':repeat,'observer_counts':counts,'pipeline_final':health[-1]}),flush=True)
        finally:
            for proc in jobs:
                if proc.poll() is None:os.killpg(proc.pid,signal.SIGINT)
            for proc in jobs:
                try:proc.wait(timeout=8)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=5)
            for log in logs:log.close()
            n.destroy_node();rclpy.try_shutdown()
