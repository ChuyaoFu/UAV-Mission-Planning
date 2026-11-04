#!/usr/bin/python3
import argparse,json,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from cartographer_ros_msgs.srv import TrajectoryQuery,FinishTrajectory,WriteState
ap=argparse.ArgumentParser();ap.add_argument('output');ap.add_argument('--finish',action='store_true');args=ap.parse_args();out=Path(args.output);out.mkdir(exist_ok=True,parents=True);rclpy.init();n=Node('read_actual_slam_result')
def call(cls,name,request):
 c=n.create_client(cls,name)
 if not c.wait_for_service(timeout_sec=5):raise RuntimeError(name+' unavailable')
 f=c.call_async(request);rclpy.spin_until_future_complete(n,f,timeout_sec=30)
 if not f.done():raise RuntimeError(name+' timed out')
 return f.result()
if args.finish:
 r=FinishTrajectory.Request();r.trajectory_id=0;print(call(FinishTrajectory,'/finish_trajectory',r));time.sleep(3)
r=TrajectoryQuery.Request();r.trajectory_id=0;m=call(TrajectoryQuery,'/trajectory_query',r);rows=[]
for p in m.trajectory:
 q=p.pose.orientation;v=p.pose.position;rows.append([p.header.stamp.sec+p.header.stamp.nanosec*1e-9,v.x,v.y,v.z,q.x,q.y,q.z,q.w])
(out/'trajectory.json').write_text(json.dumps({'optimized':rows}));r=WriteState.Request();r.filename=str(out/'cartographer.pbstream');r.include_unfinished_submaps=True;print(call(WriteState,'/write_state',r));print('optimized nodes',len(rows));n.destroy_node();rclpy.shutdown()
