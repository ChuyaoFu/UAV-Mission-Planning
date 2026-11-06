#!/usr/bin/python3
"""Read-only input audit of live navigation and mapping ROS nodes.
Gazebo sensor proxies are permitted; simulator pose/state and ideal depth are not.
Shared /tf is checked by the fusion source's explicit parent/child frame filter.
"""
import argparse,json,time
from pathlib import Path
import rclpy
from rclpy.node import Node
PROTECTED={'carto_course_mission','carto_vio_fusion','measured_structure_cloud','passive_stereo','depth_cloud','run_subscribe_msckf','cartographer_node','slam_live_view','optimized_structure_map_view'}
FORBIDDEN=('/evaluation','/model/','/world/','/gazebo','/vio/depth','gps','ground_truth','model_states')
def audit(output,require_mission=False):
 rclpy.init();node=Node('navigation_input_audit');end=time.monotonic()+(35 if require_mission else 4)
 while time.monotonic()<end:
  rclpy.spin_once(node,timeout_sec=.1)
  if require_mission and any(n=='carto_course_mission' for n,_ in node.get_node_names_and_namespaces()):
   observed={topic for topic,_ in node.get_subscriber_names_and_types_by_node('carto_course_mission','/')}
   if {'/carto/points','/vio/left/image_raw','/carto/local_pose'}<=observed:break
 nodes={};violations=[]
 for name,namespace in node.get_node_names_and_namespaces():
  if name not in PROTECTED:continue
  subscriptions=node.get_subscriber_names_and_types_by_node(name,namespace);nodes[namespace.rstrip('/')+'/'+name]=dict(subscriptions)
  for topic,_ in subscriptions:
   if any(x in topic.lower() for x in FORBIDDEN):violations.append({'node':name,'topic':topic})
 required={'carto_vio_fusion','measured_structure_cloud','cartographer_node'}
 if require_mission:required.add('carto_course_mission')
 missing=sorted(required-{k.split('/')[-1] for k in nodes})
 if require_mission:
  observed=set(nodes.get('/carto_course_mission',{}))
  missing+=sorted({'/carto/points','/vio/left/image_raw','/carto/local_pose'}-observed)
 result={'valid':not violations and not missing,'subscriptions':nodes,'forbidden_inputs':violations,'missing_required_nodes':missing,'scope':'ROS graph only. PX4 GPS parameters/sensor absence and simulator actuator sensors are audited separately. /tf fusion consumes only carto_odom -> vio_imu. CAD/launch priors are declared navigation inputs, not simulator state.'}
 Path(output).write_text(json.dumps(result,indent=2));node.destroy_node();rclpy.shutdown()
 if not result['valid']:raise RuntimeError('Input audit refused: '+str(result))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--require-mission',action='store_true');a=p.parse_args();print(json.dumps(audit(a.output,a.require_mission),indent=2))
