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
from cad_registration import Registration
from line_localization import LineLocalization
from window_localization import WindowLocalization
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2,Range,Image
from std_msgs.msg import String
from px4_msgs.msg import VehicleLocalPosition,VehicleStatus,OffboardControlMode,TrajectorySetpoint,VehicleCommand
D=np.diag([1.,-1.,-1.]);LEVER=np.array([.12,0,.055])
ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--speed',type=float,default=.22);ap.add_argument('--start-y',type=float,default=.9144);ap.add_argument('--output',default='/work/cartographer/v3/results/mission.json');args=ap.parse_args()
assert Path('/.dockerenv').exists() and os.environ.get('ROS_DOMAIN_ID')=='42','SITL only'
lock=open('/work/data/flight-smoke.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
config=json.loads(Path('/work/cartographer/v3/config/stages.json').read_text());grid=load('/work/cartographer/v3/config/cad-safe-grid.npz')
start=np.r_[config['launch_xy'][0],args.start_y,.8];plans=[]
try:
 for s in config['stages']:
  p=grid.plan(start,s['goal']);clean=[];previous=np.asarray(start)
  for point in p:
   if np.linalg.norm(np.asarray(point)-previous)>.025:clean.append(point);previous=point
  plans.append((s['name'],np.array(clean)));start=np.array(s['goal'])
except (ValueError,RuntimeError) as error:
 result={'controller_complete':False,'disarmed':None,'arming_attempted':False,'full_course_verified':False,'failure':'CAD preflight refused: '+str(error),'failed_stage':s['name'],'dry_run':not args.execute}
 Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));raise SystemExit(1)
if not args.execute:
 print(json.dumps({'dry_run':True,'plans':[{ 'stage':n,'path':p.tolist()} for n,p in plans],'flight_verified':False}));raise SystemExit(0)
px4=Path('/work/src/PX4-Autopilot/build/px4_sitl_default/bin')
params={k:float(subprocess.check_output([str(px4/'px4-param'),'show','-q',k],text=True)) for k in ['EKF2_GPS_CTRL','EKF2_EV_CTRL']}
assert params=={'EKF2_GPS_CTRL':0.,'EKF2_EV_CTRL':13.},params
assert 'never published' in subprocess.check_output([str(px4/'px4-listener'),'sensor_gps','-n','1'],text=True)
registration=Registration();line_localization=LineLocalization();window_localization=WindowLocalization()
rclpy.init();node=Node('carto_course_mission',parameter_overrides=[Parameter('use_sim_time',value=True)])
state={'pose':None,'px':None,'status':None,'health':None,'range':None,'tree':None,'cloud_stamp':-1,'last_observed_position':None,'prior_transitions':0,'clearance_recovery_ticks':0}
subs=[]
def sec(h):return h.stamp.sec+h.stamp.nanosec*1e-9
def pose_cb(m):state['pose']=m
for cls,topic,key,qos in [(PoseStamped,'/carto/local_pose','pose',10),(VehicleLocalPosition,'/fmu/out/vehicle_local_position_v1','px',qos_profile_sensor_data),(VehicleStatus,'/fmu/out/vehicle_status_v1','status',qos_profile_sensor_data),(Range,'/carto/down_range','range',10)]:
 subs.append(node.create_subscription(cls,topic,lambda m,k=key:state.update({k:m}),qos))
subs.append(node.create_subscription(String,'/carto/localization_health',lambda m:state.update(health=json.loads(m.data)),10))
window_health_pub=node.create_publisher(String,'/window/localization_health',10)
line_health_pub=node.create_publisher(String,'/line/localization_health',10)
cad_health_pub=node.create_publisher(String,'/cad/localization_health',10)
phase_pub=node.create_publisher(String,'/carto/mission_state',10)
mode=node.create_publisher(OffboardControlMode,'/fmu/in/offboard_control_mode',10);traj=node.create_publisher(TrajectorySetpoint,'/fmu/in/trajectory_setpoint',10);cmd=node.create_publisher(VehicleCommand,'/fmu/in/vehicle_command',10)
def stamp():return node.get_clock().now().nanoseconds//1000
def command(c,p1=0.,p2=0.):cmd.publish(VehicleCommand(timestamp=stamp(),command=c,param1=float(p1),param2=float(p2),target_system=1,target_component=1,source_system=250,source_component=1,from_external=True))
def pump(dt=.05):
 end=time.monotonic()+dt
 while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=min(.01,max(0,end-time.monotonic())))
def check_once():
 now=stamp()*1e-6;h=state['health'];p=state['px'];m=state['pose'];r=state['range']
 if not h or not h['valid'] or now-h['stamp']>.25:raise RuntimeError('Localization unhealthy: '+str(h))
 if p is None or not p.xy_valid or not p.z_valid:raise RuntimeError('PX4 position invalid')
 if not m or now-sec(m.header)>.2:raise RuntimeError('Local odom stale')
 if r is None or now-sec(r.header)>.2:raise RuntimeError('Downward ranging stale')
def check():
 try:check_once();return
 except RuntimeError as error:
  h=state['health'];status=state['status'];px=state['px']
  if not status or status.arming_state!=VehicleStatus.ARMING_STATE_ARMED or not h or h['reason']!='unobservable' or px is None or not px.xy_valid or not px.z_valid:raise
  node.get_logger().warning('Short localization degradation: brake and wait for observations')
  hold_ned=[float(px.x),float(px.y),float(px.z)];hold_yaw=float(px.heading);start_recovery=stamp();end=start_recovery+2500000
  desired_yaw=float(-(globals().get('yaw',-hold_yaw)-globals().get('yaw_align',0.)))
  turn=math.atan2(math.sin(desired_yaw-hold_yaw),math.cos(desired_yaw-hold_yaw))
  while stamp()<end:
   recovery_yaw=desired_yaw if abs(turn)>.15 else hold_yaw+.30*math.sin(2*math.pi*(stamp()-start_recovery)/2500000.)
   mode.publish(OffboardControlMode(timestamp=stamp(),position=True));traj.publish(TrajectorySetpoint(timestamp=stamp(),position=hold_ned,velocity=[math.nan]*3,acceleration=[math.nan]*3,jerk=[math.nan]*3,yaw=recovery_yaw,yawspeed=math.nan));pump()
   try:check_once();return
   except RuntimeError:pass
  raise RuntimeError('Localization recovery timeout: '+str(error))
def local_body():
 m=state['pose'];p=m.pose.position;q=m.pose.orientation;rot=Rotation.from_quat([q.x,q.y,q.z,q.w]);return np.array([p.x,p.y,p.z])-rot.apply(LEVER),rot
end=time.monotonic()+40
while time.monotonic()<end:
 pump()
 try:check();break
 except RuntimeError:pass
else:raise SystemExit('No healthy localization/range; did not arm')
local0,rot0=local_body();px0=state['px'];offset=np.array([px0.x,px0.y,px0.z])-D@local0
world0=np.array([config['launch_xy'][0],args.start_y,state['range'].range+.025]);align=Rotation.from_euler('z',config['launch_yaw_rad'])*rot0.inv();yaw_align=align.as_euler('xyz')[2];map_shift=np.zeros(3)
height={'z':float(world0[2]),'local_z':float(local0[2]),'rejected':0}
def world_body():
 local=local_body()[0];p=world0+map_shift+align.apply(local-local0)
 predicted=height['z']+local[2]-height['local_z'];measurement=state['range'].range+.025
 if abs(measurement-predicted)<.08:predicted+=.12*(measurement-predicted)
 else:height['rejected']+=1
 height.update(z=float(predicted),local_z=float(local[2]));p[2]=predicted;return p
def apply_correction(correction,pivot):
 global align,map_shift,yaw_align
 rotation=Rotation.from_euler('z',float(correction[2]));translation=np.r_[correction[:2],0.]
 map_shift=pivot+rotation.apply(world0+map_shift-pivot)+translation-world0
 align=rotation*align;yaw_align=align.as_euler('xyz')[2]
def line_cb(m):
 pose=state['pose']
 if not pose or abs(sec(pose.header)-sec(m.header))>.15:return
 body=world_body();rotation=align*local_body()[1];correction=line_localization.update(m,body,rotation)
 if line_localization.health is not None:line_health_pub.publish(String(data=json.dumps(line_localization.health)))
 if correction is not None:apply_correction(correction,body)
 body=world_body();rotation=align*local_body()[1];correction=window_localization.update(m,body,rotation)
 if window_localization.health is not None:window_health_pub.publish(String(data=json.dumps(window_localization.health)))
 if correction is not None:apply_correction(correction,body)
subs.append(node.create_subscription(Image,'/vio/left/image_raw',line_cb,qos_profile_sensor_data))
def cloud_cb(m):
 global align,map_shift,yaw_align
 pose=state['pose']
 if not pose or abs(sec(pose.header)-sec(m.header))>.08:return
 # Converter publishes tightly packed xyz float32; validate before interpreting.
 if m.point_step!=12 or m.header.frame_id!='vio_imu':return
 points=np.frombuffer(m.data,dtype='<f4').reshape(-1,3)
 q=pose.pose.orientation;p=pose.pose.position;rot=Rotation.from_quat([q.x,q.y,q.z,q.w]);world=world0+map_shift+align.apply(rot.apply(points)+[p.x,p.y,p.z]-local0)
 world[:,2]+=world_body()[2]-(world0+map_shift+align.apply(local_body()[0]-local0))[2]
 pivot=world_body();correction=registration.update(world,pivot,sec(m.header))
 if registration.health is not None:cad_health_pub.publish(String(data=json.dumps(registration.health)))
 line_fresh=line_localization.health is not None and sec(m.header)-line_localization.health['stamp']<.4 and line_localization.health['lateral_supported']
 window_fresh=window_localization.health is not None and sec(m.header)-window_localization.health['stamp']<.6 and window_localization.health['supported']
 if correction is not None and not line_fresh and not window_fresh:
  rotation=Rotation.from_euler('z',float(correction[2]));translation=np.r_[correction[:2],0.]
  world=rotation.apply(world-pivot)+pivot+translation
  map_shift=pivot+rotation.apply(world0+map_shift-pivot)+translation-world0
  align=rotation*align;yaw_align=align.as_euler('xyz')[2]
 state['tree']=cKDTree(world/np.array([.32,.32,.18]));state['cloud_stamp']=sec(m.header);state['last_observed_position']=world_body().copy()
subs.append(node.create_subscription(PointCloud2,'/carto/points',cloud_cb,qos_profile_sensor_data))
def send(world,yaw,velocity=None):
 error=np.asarray(world)-world_body();px=state['px'];ned=np.array([px.x,px.y,px.z])
 if velocity is None:ned+=D@align.inv().apply(error)
 mode.publish(OffboardControlMode(timestamp=stamp(),position=True));traj.publish(TrajectorySetpoint(timestamp=stamp(),position=ned.astype(float).tolist(),velocity=([math.nan]*3 if velocity is None else (D@align.inv().apply(velocity)).astype(float).tolist()),acceleration=[math.nan]*3,jerk=[math.nan]*3,yaw=float(-(yaw-yaw_align)),yawspeed=math.nan));pump()
def veto(a,b):
 tree=state['tree']
 age=stamp()*1e-6-state['cloud_stamp']
 if tree is None:raise RuntimeError('No obstacle observations at all')
 if age>.2:
  moved=np.linalg.norm(a-state['last_observed_position'])
  if age>4.0 or moved>.30:raise RuntimeError('Depth recovery budget exceeded: '+str([age,moved]))
  if not state['health'] or not state['health']['valid']:raise RuntimeError('Prior transition requires healthy localization')
  state['prior_transitions']+=1
 samples=np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/.03)+1));distance,_=tree.query(samples/np.array([.32,.32,.18]))
 if np.min(distance)<1.:
  withdrawing=distance[-1]>distance[0]+.025 and np.min(np.diff(distance))>-.005
  if not withdrawing:raise RuntimeError('Live depth blocks swept vehicle envelope')
records=[];trace=[];last_trace=-1;last_log=-1;armed=False;success=False;failure=None;run_start=stamp();hold=world_body();yaw=config['launch_yaw_rad']
try:
 for _ in range(50):check();send(hold,yaw)
 command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,1,6);command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,1)
 for _ in range(100):
  send(hold,yaw)
  if state['status'] and state['status'].arming_state==VehicleStatus.ARMING_STATE_ARMED:armed=True;break
 if not armed:raise RuntimeError('Arming denied')
 run_start=stamp()
 phases=[('takeoff',np.array([[hold[0],hold[1],.8]]))]+plans
 flat_goals=[point for _,path in phases for point in path];goal_index=0
 for stage,path in phases:
  phase_pub.publish(String(data=stage.upper()))
  for goal in path:
   reached=None;blocked_since=None;deadline=stamp()+25000000;preturn=False
   segment_delta=np.asarray(goal)-world_body()
   if stage!='takeoff' and np.linalg.norm(segment_delta[:2])>.10:yaw=math.atan2(segment_delta[1],segment_delta[0])
   while True:
    check();now=stamp()
    if now-run_start>175000000:raise RuntimeError('Run deadline: land before 180 s')
    if now>deadline:raise RuntimeError('Stage progress timeout '+stage)
    actual=world_body();delta=goal-actual;distance=np.linalg.norm(delta)
    if stage!='takeoff' and not preturn and distance<.18 and goal_index+1<len(flat_goals):
     lookahead=np.asarray(flat_goals[goal_index+1])-actual
     if np.linalg.norm(lookahead[:2])>.15:yaw=math.atan2(lookahead[1],lookahead[0]);preturn=True
    if now-last_trace>200000:
     trace.append({'stamp':now*1e-6,'stage':stage,'position':actual.tolist(),'goal':np.asarray(goal).tolist(),'yaw':float((align*local_body()[1]).as_euler('xyz')[2])});last_trace=now
    if now-last_log>2000000:
     node.get_logger().info(stage+' pos='+str(np.round(actual,3).tolist())+' goal='+str(np.round(goal,3).tolist()));last_log=now
    if distance<.06:
     reached=now if reached is None else reached
     if now-reached>400000:break
    else:reached=None
    target=actual+delta*min(1.,.20/max(distance,1e-6))
    velocity=np.zeros(3)
    if stage=='takeoff':target=goal;velocity=None
    if stage!='takeoff':
     world_rot=align*local_body()[1];actual_yaw=world_rot.as_euler('xyz')[2]
     yaw_error=math.atan2(math.sin(yaw-actual_yaw),math.cos(yaw-actual_yaw))
     if abs(yaw_error)>.20:target=actual
     else:
      limit=.50 if stage=='tunnel_exit' else (.10 if stamp()*1e-6-state['cloud_stamp']>.2 else args.speed)
      velocity=delta*.9
      xy=np.linalg.norm(velocity[:2])
      if xy>limit:velocity[:2]*=limit/xy
      velocity[2]=np.clip(velocity[2],-.3,.3)
    if stage!='takeoff':
     if not grid.segment(actual,target):
      try:
       repair=grid.plan(actual,goal);detour=next(point for point in repair[1:] if np.linalg.norm(point-actual)>.025)
       detour_delta=detour-actual;target=actual+detour_delta*min(1.,.15/max(np.linalg.norm(detour_delta),1e-6))
       if not grid.segment(actual,target):raise RuntimeError('Local repair segment still blocked')
       yaw=math.atan2(detour_delta[1],detour_delta[0]);velocity=detour_delta*.7
       if np.linalg.norm(velocity)>.15:velocity*=.15/np.linalg.norm(velocity)
      except (RuntimeError,ValueError,StopIteration):
       if blocked_since is None:blocked_since=now
       if now-blocked_since>5000000:raise RuntimeError('No safe CAD repair from current estimate: '+str(actual.tolist()))
       offsets=np.array([[.025,0,0],[-.025,0,0],[0,.025,0],[0,-.025,0]])
       values=registration.sample(actual+offsets);gradient=np.array([values[0]-values[1],values[2]-values[3],0.])
       if np.linalg.norm(gradient)>.0001:
        gradient/=np.linalg.norm(gradient);target=actual+gradient*.12;velocity=gradient*.08;state['clearance_recovery_ticks']+=1
       else:target=actual;velocity=np.zeros(3)
     try:
      veto(actual,target)
      if grid.free(grid.idx(actual)):blocked_since=None
     except RuntimeError as obstacle:
      if blocked_since is None:blocked_since=now;node.get_logger().warning(str(obstacle)+'; hover and reassess')
      if now-blocked_since>5000000:raise
      target=actual;velocity=np.zeros(3)
    send(target,yaw,velocity)
   records.append({'stage':stage,'stamp_us':stamp(),'position':world_body().tolist()});goal_index+=1
  if stage=='window_observe':
   yaw=math.atan2(window_localization.center[1]-world_body()[1],window_localization.center[0]-world_body()[0]);hold=world_body().copy();until=stamp()+10000000;stable=None
   while stamp()<until:
    check();now=stamp();h=window_localization.health
    good=h is not None and now*1e-6-h['stamp']<.5 and h['supported'] and h['before_px']<5 and np.linalg.norm(h['fit_delta'][:2])<.08
    stable=now if good and stable is None else (stable if good else None)
    if stable is not None and now-stable>700000:break
    send(hold,yaw,np.zeros(3))
   else:raise RuntimeError('Window image registration did not converge; no crossing authorization')
  node.get_logger().info('Completed controller phase '+stage);phase_pub.publish(String(data=stage+' COMPLETE'))
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
 Path(args.output).write_text(json.dumps({'controller_phases':records,'trace':trace,'failure':failure,'controller_complete':success and disarmed,'disarmed':disarmed,'run_time_s':(stamp()-run_start)*1e-6,'full_course_verified':False,'scope':'provisional CAD annotations; independent judge required','parameters':params,'prior_transition_ticks':state['prior_transitions'],'clearance_recovery_ticks':state['clearance_recovery_ticks'],'window_registration':window_localization.events,'range_height_rejections':height['rejected'],'line_registration':line_localization.events,'cad_registration':registration.events,'final_map_shift':map_shift.tolist()},indent=2))
 node.destroy_node();rclpy.shutdown()
if failure:raise SystemExit(failure)
