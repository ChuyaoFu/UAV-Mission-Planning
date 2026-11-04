"""Require an active mapping trajectory before any arming attempt."""
import time,json
from pathlib import Path
import rclpy
from rclpy.node import Node
from cartographer_ros_msgs.srv import GetTrajectoryStates

def verify_mapping(output):
 rclpy.init();n=Node('mapping_preflight');c=n.create_client(GetTrajectoryStates,'/get_trajectory_states')
 try:
  if not c.wait_for_service(timeout_sec=8):raise RuntimeError('Mapping service unavailable; flight refused')
  f=c.call_async(GetTrajectoryStates.Request());rclpy.spin_until_future_complete(n,f,timeout_sec=5)
  if not f.done():raise RuntimeError('Mapping service did not answer; flight refused')
  response=f.result()
  if response.status.code!=0 or 0 not in response.trajectory_states.trajectory_id:raise RuntimeError('No mapping trajectory 0; flight refused')
  index=list(response.trajectory_states.trajectory_id).index(0)
  if response.trajectory_states.trajectory_state[index]!=0:raise RuntimeError('Mapping trajectory is not ACTIVE')
  deadline=time.monotonic()+5
  while n.count_publishers('/mapping/health')<1 and time.monotonic()<deadline:rclpy.spin_once(n,timeout_sec=.1)
  count=n.count_publishers('/mapping/health')
  if count<1:raise RuntimeError('Measured structure filter unavailable; flight refused')
  result={'mapper_service_responding':True,'trajectory_id':0,'trajectory_active':True,'structure_publishers':count,'checked_before_arming':True};Path(output).write_text(json.dumps(result,indent=2));return result
 finally:n.destroy_node();rclpy.shutdown()
if __name__=='__main__':
 import sys
 print(verify_mapping(sys.argv[1]))
