"""Require an active mapping trajectory before any arming attempt."""
import time,json
from pathlib import Path
import rclpy
from rclpy.node import Node
from cartographer_ros_msgs.srv import GetTrajectoryStates,TrajectoryQuery
from rclpy.parameter import Parameter
from std_msgs.msg import String

def verify_mapping(output):
 rclpy.init();n=Node('mapping_preflight',parameter_overrides=[Parameter('use_sim_time',value=True)]);measured={};sub=n.create_subscription(String,'/mapping/health',lambda m:measured.update(json.loads(m.data)),1);c=n.create_client(GetTrajectoryStates,'/get_trajectory_states')
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
  query=n.create_client(TrajectoryQuery,'/trajectory_query')
  if not query.wait_for_service(timeout_sec=3):raise RuntimeError('Mapper trajectory query unavailable')
  request=TrajectoryQuery.Request();request.trajectory_id=0;future=query.call_async(request);rclpy.spin_until_future_complete(n,future,timeout_sec=5)
  if not future.done() or future.result().status.code!=0 or len(future.result().trajectory)<1:raise RuntimeError('No initialized measured SLAM node; flight refused')
  deadline=time.monotonic()+5
  while time.monotonic()<deadline:
   rclpy.spin_once(n,timeout_sec=.05)
   if measured and measured['raw_points']>100 and abs(n.get_clock().now().nanoseconds*1e-9-measured['stamp'])<.25:break
  else:raise RuntimeError('No fresh measured stereo structure/floor health; flight refused')
  result={'initialized_nodes':len(future.result().trajectory),'measured_health':measured,'mapper_service_responding':True,'trajectory_id':0,'trajectory_active':True,'structure_publishers':count,'checked_before_arming':True};Path(output).write_text(json.dumps(result,indent=2));return result
 finally:n.destroy_node();rclpy.shutdown()
if __name__=='__main__':
 import sys
 print(verify_mapping(sys.argv[1]))
