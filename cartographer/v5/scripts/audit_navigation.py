#!/usr/bin/python3
"""Read-only node/topic audit. Filesystem isolation is separately kernel-tested."""
import json,time,sys,argparse
from pathlib import Path
import rclpy
from rclpy.node import Node
PROTECTED={'relative_guideline_perception','cadfree_navigator','measured_navigation_frames','cadfree_ego_planner','cadfree_trajectory_server','carto_vio_fusion','measured_structure_cloud','depth_cloud','passive_stereo','downward_tof','cartographer_node','run_subscribe_msckf','optimized_structure_map_view'}
PROTECTED.update({'planar_free_ray_evidence','home_relative_odometry_adapter'})
PROTECTED.add('structural_free_ray_evidence')
p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--require-navigator',action='store_true');p.add_argument('--require-guideline',action='store_true');args=p.parse_args()
rclpy.init();n=Node('cadfree_input_audit');end=time.monotonic()+5
while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.1)
nodes={};errors=[]
for name,ns in n.get_node_names_and_namespaces():
 if name not in PROTECTED:continue
 subs=n.get_subscriber_names_and_types_by_node(name,ns);nodes[ns.rstrip('/')+'/'+name]=dict(subs)
 for topic,_ in subs:
  if any(v in topic.lower() for v in ['/evaluation','/model/','/world/','gps','/cad/','/line/','/window/','/vio/depth','global_map']):errors.append([name,topic])
required={'cadfree_ego_planner','measured_navigation_frames','carto_vio_fusion','cartographer_node','passive_stereo'};required |= {'cadfree_navigator'} if args.require_navigator else set();missing=required-{k.split('/')[-1] for k in nodes}
frames=nodes.get('/measured_navigation_frames',{});missing |= {'/carto/accepted_odom'} if '/carto/accepted_odom' not in frames else set()
if args.require_guideline:
 required.add('relative_guideline_perception');missing |= required-{k.split('/')[-1] for k in nodes}
 if n.count_publishers('/nav/guideline_hint')!=1:missing.add('single /nav/guideline_hint publisher')
 subs=nodes.get('/relative_guideline_perception',{})
 missing |= {t for t in ['/vio/left/image_raw','/nav/measured_depth','/nav/camera_pose','/nav/body_odom','/carto/down_range'] if t not in subs}
if args.require_navigator:
 subs=nodes.get('/cadfree_navigator',{});missing |= {t for t in ['/nav/body_odom','/carto/localization_health','/position_cmd','/grid_map/knowledge','/fmu/out/vehicle_local_position_v1'] if t not in subs}
planar=nodes.get('/planar_free_ray_evidence')
configuration=json.loads((Path(__file__).resolve().parents[1]/'config/navigation.json').read_text())
if configuration.get('structural_evidence_backend')=='grabcut':
 if '/structural_free_ray_evidence' not in nodes:missing.add('structural_free_ray_evidence')
 else:
  subscriptions=nodes['/structural_free_ray_evidence']
  missing |= {t for t in ['/vio/left/image_raw','/vio/right/image_raw','/vio/imu','/stereo/depth/image','/carto/down_range','/nav/camera_pose','/nav/body_odom','/carto/localization_health'] if t not in subscriptions}
 if n.count_publishers('/nav/depth_evidence')!=1:missing.add('single structural /nav/depth_evidence publisher')
if planar is not None:
 missing |= {t for t in ['/vio/left/image_raw','/vio/right/image_raw','/stereo/depth/image'] if t not in planar}
 if n.count_publishers('/nav/depth_evidence')!=1:missing.add('single /nav/depth_evidence publisher')
result={'valid':not errors and not missing,'protected_node_subscriptions':nodes,'forbidden_inputs':errors,'missing_nodes':sorted(missing),'ideal_depth_publishers':n.count_publishers('/vio/depth/image'),'truth_pose_publishers':n.count_publishers('/evaluation/poses'),'disabled_cloud_publishers':n.count_publishers('/nav/disabled_cloud'),'scope':'Named navigation graph; CAD filesystem denial is a separate kernel-enforced Landlock probe.'};result['valid'] &= result['truth_pose_publishers']==0 and result['ideal_depth_publishers']==0 and result['disabled_cloud_publishers']==0;Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));n.destroy_node();rclpy.shutdown()
if not result['valid']:raise SystemExit(1)
