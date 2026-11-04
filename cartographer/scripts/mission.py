#!/usr/bin/python3
"""SITL course controller: CAD A* + live 3D swept-envelope obstacle veto.
One process handles takeoff, stages, landing and localization-triggered abort.
Never subscribes to truth/GPS. Full-course flight remains UNVALIDATED.
"""
import argparse,fcntl,json,math,os,subprocess,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from cad_planner import load
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2,Range
from std_msgs.msg import String
from px4_msgs.msg import VehicleLocalPosition,VehicleStatus,OffboardControlMode,TrajectorySetpoint,VehicleCommand
D=np.diag([1.,-1.,-1.]);LEVER=np.array([.15,0,.08])
ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--start-y',type=float,default=.9144);ap.add_argument('--output',default='/work/cartographer/results/mission.json');args=ap.parse_args()
assert Path('/.dockerenv').exists() and os.environ.get('ROS_DOMAIN_ID')=='42','SITL only'
lock=open('/work/data/flight-smoke.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
config=json.loads(Path('/work/cartographer/config/stages.json').read_text());grid=load('/work/cartographer/results/cad-grid.npz')
start=np.r_[config['launch_xy'][0],args.start_y,.8];plans=[]
try:
 for s in config['stages']:
  p=grid.plan(start,s['goal']);plans.append((s['name'],p));start=np.array(s['goal'])
except (ValueError,RuntimeError) as error:
 result={'controller_complete':False,'disarmed':None,'arming_attempted':False,'full_course_verified':False,'failure':'CAD preflight refused: '+str(error),'failed_stage':s['name'],'dry_run':not args.execute}
 Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));raise SystemExit(1)
if not args.execute:
 print(json.dumps({'dry_run':True,'plans':[{ 'stage':n,'path':p.tolist()} for n,p in plans],'flight_verified':False}));raise SystemExit(0)
px4=Path('/work/src/PX4-Autopilot/build/px4_sitl_default/bin')
params={k:float(subprocess.check_output([str(px4/'px4-param'),'show','-q',k],text=True)) for k in ['EKF2_GPS_CTRL','EKF2_EV_CTRL']}
assert params=={'EKF2_GPS_CTRL':0.,'EKF2_EV_CTRL':11.},params
assert 'never published' in subprocess.check_output([str(px4/'px4-listener'),'sensor_gps','-n','1'],text=True)
rclpy.init();node=Node('carto_course_mission',parameter_overrides=[Parameter('use_sim_time',value=True)])
state={'pose':None,'px':None,'status':None,'health':None,'range':None,'tree':None,'cloud_stamp':-1}
subs=[]
def sec(h):return h.stamp.sec+h.stamp.nanosec*1e-9
def pose_cb(m):state['pose']=m
for cls,topic,key,qos in [(PoseStamped,'/carto/local_pose','pose',10),(VehicleLocalPosition,'/fmu/out/vehicle_local_position_v1','px',qos_profile_sensor_data),(VehicleStatus,'/fmu/out/vehicle_status_v1','status',qos_profile_sensor_data),(Range,'/carto/down_range','range',10)]:
 subs.append(node.create_subscription(cls,topic,lambda m,k=key:state.update({k:m}),qos))
subs.append(node.create_subscription(String,'/carto/localization_health',lambda m:state.update(health=json.loads(m.data)),10))
mode=node.create_publisher(OffboardControlMode,'/fmu/in/offboard_control_mode',10);traj=node.create_publisher(TrajectorySetpoint,'/fmu/in/trajectory_setpoint',10);cmd=node.create_publisher(VehicleCommand,'/fmu/in/vehicle_command',10)
def stamp():return node.get_clock().now().nanoseconds//1000
def command(c,p1=0.,p2=0.):cmd.publish(VehicleCommand(timestamp=stamp(),command=c,param1=float(p1),param2=float(p2),target_system=1,target_component=1,source_system=250,source_component=1,from_external=True))
def pump(dt=.05):
 end=time.monotonic()+dt
 while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=min(.01,max(0,end-time.monotonic())))
def check():
 now=stamp()*1e-6;h=state['health'];p=state['px'];m=state['pose'];r=state['range']
 if not h or not h['valid'] or now-h['stamp']>.25:raise RuntimeError('Localization unhealthy: '+str(h))
 if p is None or not p.xy_valid or not p.z_valid:raise RuntimeError('PX4 position invalid')
 if not m or now-sec(m.header)>.2:raise RuntimeError('Local odom stale')
 if r is None or now-sec(r.header)>.2:raise RuntimeError('Downward ranging stale')
def local_body():
 m=state['pose'];p=m.pose.position;q=m.pose.orientation;rot=Rotation.from_quat([q.x,q.y,q.z,q.w]);return np.array([p.x,p.y,p.z])-rot.apply(LEVER),rot
end=time.monotonic()+40
while time.monotonic()<end:
 pump()
 try:check();break
 except RuntimeError:pass
else:raise SystemExit('No healthy localization/range; did not arm')
local0,rot0=local_body();px0=state['px'];offset=np.array([px0.x,px0.y,px0.z])-D@local0
world0=np.array([config['launch_xy'][0],args.start_y,state['range'].range+.12]);align=Rotation.from_euler('z',config['launch_yaw_rad'])*rot0.inv();yaw_align=align.as_euler('xyz')[2]
def world_body():return world0+align.apply(local_body()[0]-local0)
def cloud_cb(m):
 pose=state['pose']
 if not pose or abs(sec(pose.header)-sec(m.header))>.08:return
 # Converter publishes tightly packed xyz float32; validate before interpreting.
 if m.point_step!=12 or m.header.frame_id!='vio_imu':return
 points=np.frombuffer(m.data,dtype='<f4').reshape(-1,3)
 q=pose.pose.orientation;p=pose.pose.position;rot=Rotation.from_quat([q.x,q.y,q.z,q.w]);world=world0+align.apply(rot.apply(points)+[p.x,p.y,p.z]-local0)
 state['tree']=cKDTree(world/np.array([.437,.437,.29]));state['cloud_stamp']=sec(m.header)
subs.append(node.create_subscription(PointCloud2,'/carto/points',cloud_cb,qos_profile_sensor_data))
def send(world,yaw):
 local=local0+align.inv().apply(np.asarray(world)-world0);ned=D@local+offset
 mode.publish(OffboardControlMode(timestamp=stamp(),position=True));traj.publish(TrajectorySetpoint(timestamp=stamp(),position=ned.astype(float).tolist(),velocity=[math.nan]*3,acceleration=[math.nan]*3,jerk=[math.nan]*3,yaw=float(-(yaw-yaw_align)),yawspeed=math.nan));pump()
def veto(a,b):
 tree=state['tree']
 if tree is None or stamp()*1e-6-state['cloud_stamp']>.2:raise RuntimeError('No current obstacle cloud')
 samples=np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/.03)+1));distance,_=tree.query(samples/np.array([.437,.437,.29]))
 if np.min(distance)<1.:raise RuntimeError('Live depth blocks swept vehicle envelope')
records=[];armed=False;success=False;failure=None;run_start=stamp();hold=world_body();yaw=config['launch_yaw_rad']
try:
 for _ in range(50):check();send(hold,yaw)
 command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,1,6);command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,1)
 for _ in range(100):
  send(hold,yaw)
  if state['status'] and state['status'].arming_state==VehicleStatus.ARMING_STATE_ARMED:armed=True;break
 if not armed:raise RuntimeError('Arming denied')
 run_start=stamp()
 phases=[('takeoff',np.array([[hold[0],hold[1],.8]]))]+plans
 for stage,path in phases:
  for goal in path:
   reached=None;deadline=stamp()+25000000
   while True:
    check();now=stamp()
    if now-run_start>175000000:raise RuntimeError('Run deadline: land before 180 s')
    if now>deadline:raise RuntimeError('Stage progress timeout '+stage)
    actual=world_body();delta=goal-actual;distance=np.linalg.norm(delta)
    if distance<.06:
     reached=now if reached is None else reached
     if now-reached>400000:break
    else:reached=None
    if np.linalg.norm(delta[:2])>.07:yaw=math.atan2(delta[1],delta[0])
    target=actual+delta*min(1.,.18/max(distance,1e-6))
    if stage!='takeoff':
     if not grid.segment(actual,target):raise RuntimeError('CAD map blocks current swept segment')
     veto(actual,target)
    send(target,yaw)
   records.append({'stage':stage,'stamp_us':stamp(),'position':world_body().tolist()})
  node.get_logger().info('Completed controller phase '+stage)
 # NAV_LAND only after centering over CAD platform. Independent judge checks footprint.
 if np.linalg.norm(world_body()[:2]-config['landing_center_xy'])>.08:raise RuntimeError('Platform centering failed')
 success=True
except Exception as e:failure=str(e);node.get_logger().error(failure)
finally:
 if armed:
  command(VehicleCommand.VEHICLE_CMD_NAV_LAND);deadline=time.monotonic()+35
  while time.monotonic()<deadline:
   pump()
   if state['status'] and state['status'].arming_state!=VehicleStatus.ARMING_STATE_ARMED:break
  else:success=False;failure=(failure or '')+'; disarm timeout'
 disarmed=state['status'] is not None and state['status'].arming_state!=VehicleStatus.ARMING_STATE_ARMED
 Path(args.output).write_text(json.dumps({'controller_phases':records,'failure':failure,'controller_complete':success and disarmed,'disarmed':disarmed,'run_time_s':(stamp()-run_start)*1e-6,'full_course_verified':False,'scope':'provisional CAD annotations; independent judge required','parameters':params},indent=2))
 node.destroy_node();rclpy.shutdown()
if failure:raise SystemExit(failure)
