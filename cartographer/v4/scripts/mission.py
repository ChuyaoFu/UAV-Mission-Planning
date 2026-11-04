#!/usr/bin/python3
"""SITL course controller: CAD A* + live 3D swept-envelope obstacle veto.
One process handles takeoff, stages, landing and localization-triggered abort.
Never subscribes to truth/GPS. Results require the independent post-flight judge.
"""
import argparse,fcntl,json,math,os,subprocess,time
os.environ.setdefault("OPENBLAS_NUM_THREADS","1");os.environ.setdefault("OMP_NUM_THREADS","1")
import cv2
cv2.setNumThreads(1)
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from collections import deque
from scipy.spatial import cKDTree
from cad_planner import load
from cad_registration import Registration
from line_localization import LineLocalization
from window_localization import WindowLocalization
from perception_worker import LatestWorker
from path_tracking import Follower,smooth
from motion_limits import safe_speed,slew_velocity,stopping_distance,visibility_speed_factor
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data,QoSProfile
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2,Range,Image
from std_msgs.msg import String
from px4_msgs.msg import VehicleLocalPosition,VehicleStatus,OffboardControlMode,TrajectorySetpoint,VehicleCommand
D=np.diag([1.,-1.,-1.]);LEVER=np.array([.12,0,.055])
ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--route',choices=['window','gate-b'],default='window');ap.add_argument('--speed',type=float,default=.70);ap.add_argument('--start-y',type=float,default=.9144);ap.add_argument('--output',default='/work/cartographer/v4/results/mission.json');args=ap.parse_args()
assert Path('/.dockerenv').exists() and os.environ.get('ROS_DOMAIN_ID')=='42','SITL only'
lock=open('/work/data/flight-smoke.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
config=json.loads(Path('/work/cartographer/v4/config/stages.json').read_text());grid=load('/work/cartographer/v4/config/cad-safe-grid.npz')
if args.route=='gate-b':
 from route_options import avoid_window
 grid=avoid_window(grid);config['stages']=[s for s in config['stages'] if not s['name'].startswith('window_')]
 for stage in config['stages']:
  if stage['name']=='gate_or_window':stage['name']='gate_b'
start=np.r_[config['launch_xy'][0],args.start_y,.8];plans=[]
try:
 for s in config['stages']:
  p=grid.plan(start,s['goal']);clean=[];previous=np.asarray(start)
  for point in p:
   if np.linalg.norm(np.asarray(point)-previous)>.025:clean.append(point);previous=point
  path=smooth(np.vstack([start,clean]),grid);plans.append((s['name'],path));start=np.array(s['goal'])
except (ValueError,RuntimeError) as error:
 result={'controller_complete':False,'disarmed':None,'arming_attempted':False,'full_course_verified':False,'failure':'CAD preflight refused: '+str(error),'failed_stage':s['name'],'dry_run':not args.execute}
 Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));raise SystemExit(1)
# Semantic milestones stay in order, while tracking remains continuous across
# their vertices. Stop only for takeoff, required image registration and landing.
# Optimize the first corridor-to-maze turn together, so the join has the
# same clearance objective as every interior bend.
first_two=smooth(np.vstack([plans[0][1],plans[1][1][1:]]),grid)
original_plans=[('gate_a',first_two)]+plans[2:]
route_segments={};joined=[]
for group in ([original_plans] if args.route=='gate-b' else [original_plans[:4],original_plans[4:]]):
 combined=[];labels=[];distance=0.
 for name,path in group:
  if name=='gate_a':
   index=int(np.argmin(np.linalg.norm(path[:,:2]-np.asarray(config['stages'][0]['goal'])[:2],axis=1)));labels.append((distance+float(np.linalg.norm(np.diff(path[:index+1],axis=0),axis=1).sum()),'first_corridor'))
  for point in path:
   if combined and np.linalg.norm(point-combined[-1])<.005:continue
   if combined:distance+=float(np.linalg.norm(point-combined[-1]))
   combined.append(point)
  labels.append((distance,name))
 name=group[-1][0];joined.append((name,np.asarray(combined)));route_segments[name]=labels
plans=joined
if not args.execute:
 print(json.dumps({'dry_run':True,'plans':[{ 'stage':n,'path':p.tolist()} for n,p in plans],'flight_verified':False}));raise SystemExit(0)
px4=Path('/work/src/PX4-Autopilot/build/px4_sitl_default/bin')
params={k:float(subprocess.check_output([str(px4/'px4-param'),'show','-q',k],text=True)) for k in ['EKF2_GPS_CTRL','EKF2_EV_CTRL']}
assert params=={'EKF2_GPS_CTRL':0.,'EKF2_EV_CTRL':13.},params
assert 'never published' in subprocess.check_output([str(px4/'px4-listener'),'sensor_gps','-n','1'],text=True)
registration=Registration();line_localization=LineLocalization();window_localization=WindowLocalization()
rclpy.init();node=Node('carto_course_mission',parameter_overrides=[Parameter('use_sim_time',value=True)])
state={'pose':None,'px':None,'status':None,'health':None,'range':None,'tree':None,'cloud_stamp':-1,'last_observed_position':None,'velocity_command':np.zeros(3),'velocity_stamp':None,'motion_limited_ticks':0,'motion_debug':None,'prior_transitions':0,'clearance_recovery_ticks':0}
subs=[];pose_history=deque(maxlen=300)
def sec(h):return h.stamp.sec+h.stamp.nanosec*1e-9
def pose_cb(m):
 state['pose']=m;p=m.pose.position;q=m.pose.orientation;t=sec(m.header)
 if not pose_history or t>pose_history[-1][0]:pose_history.append((t,np.array([p.x,p.y,p.z]),[q.x,q.y,q.z,q.w]))
def exposure_pose(t):
 if not pose_history or t<pose_history[0][0] or t>pose_history[-1][0]+.02:return None
 h=list(pose_history);times=np.array([v[0] for v in h]);query=np.clip(t,times[0],times[-1]);position=np.array([np.interp(query,times,[v[1][i] for v in h]) for i in range(3)]);rotation=Slerp(times,Rotation.from_quat([v[2] for v in h]))([query])[0] if len(h)>1 else Rotation.from_quat(h[0][2]);return position,rotation
for cls,topic,key,qos in [(PoseStamped,'/carto/local_pose','pose',1),(VehicleLocalPosition,'/fmu/out/vehicle_local_position_v1','px',qos_profile_sensor_data),(VehicleStatus,'/fmu/out/vehicle_status_v1','status',qos_profile_sensor_data),(Range,'/carto/down_range','range',1)]:
 subs.append(node.create_subscription(cls,topic,pose_cb if key=='pose' else lambda m,k=key:state.update({k:m}),qos))
subs.append(node.create_subscription(String,'/carto/localization_health',lambda m:state.update(health=json.loads(m.data)),1))
window_health_pub=node.create_publisher(String,'/window/localization_health',10)
line_health_pub=node.create_publisher(String,'/line/localization_health',10)
cad_health_pub=node.create_publisher(String,'/cad/localization_health',10)
phase_pub=node.create_publisher(String,'/carto/mission_state',10)
mode=node.create_publisher(OffboardControlMode,'/fmu/in/offboard_control_mode',10);traj=node.create_publisher(TrajectorySetpoint,'/fmu/in/trajectory_setpoint',10);cmd=node.create_publisher(VehicleCommand,'/fmu/in/vehicle_command',10)
def stamp():return node.get_clock().now().nanoseconds//1000
def command(c,p1=0.,p2=0.):cmd.publish(VehicleCommand(timestamp=stamp(),command=c,param1=float(p1),param2=float(p2),target_system=1,target_component=1,source_system=250,source_component=1,from_external=True))
def pump(dt=.05):
 end=time.monotonic()+dt
 while time.monotonic()<end:
  rclpy.spin_once(node,timeout_sec=min(.01,max(0,end-time.monotonic())))
  if 'perception' in globals():drain_perception()
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
def perception_job(kind,values):
 result=[]
 if kind=='image':
  m,body,rotation=values;stamp_image=sec(m.header);correction=line_localization.update(m,body,rotation)
  if line_localization.health is not None:result.append(('line_health',line_localization.health.copy(),None,stamp_image))
  if correction is not None:
   result.append(('correction',correction,body.copy(),stamp_image));body=body+np.r_[correction[:2],0.];rotation=Rotation.from_euler('z',correction[2])*rotation
  correction=window_localization.update(m,body,rotation)
  if window_localization.health is not None:result.append(('window_health',window_localization.health.copy(),None,stamp_image))
  if correction is not None:result.append(('correction',correction,body.copy(),stamp_image))
 else:
  world,pivot,observed=values;correction=registration.update(world,pivot,observed)
  if registration.health is not None:result.append(('cad_health',registration.health.copy(),None,observed))
  line_fresh=line_localization.health is not None and observed-line_localization.health['stamp']<.4 and line_localization.health['lateral_supported']
  window_fresh=window_localization.health is not None and observed-window_localization.health['stamp']<.6 and window_localization.health['supported']
  if correction is not None and not line_fresh and not window_fresh:result.append(('correction',correction,pivot.copy(),observed))
 return result
perception=LatestWorker(perception_job)
def drain_perception():
 for kind,value,pivot,observed in perception.take():
  if kind=='correction':
   if stamp()*1e-6-observed<.35:apply_correction(value,pivot)
  else:{'line_health':line_health_pub,'window_health':window_health_pub,'cad_health':cad_health_pub}[kind].publish(String(data=json.dumps(value)))
def line_cb(m):
 sample=exposure_pose(sec(m.header))
 if sample is None:return
 position,rotation=sample;local=position-rotation.apply(LEVER);body=world0+map_shift+align.apply(local-local0);body[2]=world_body()[2]+local[2]-local_body()[0][2]
 perception.submit('image',(m,body,align*rotation))
subs.append(node.create_subscription(Image,'/vio/left/image_raw',line_cb,qos_profile_sensor_data))
def cloud_cb(m):
 global align,map_shift,yaw_align
 sample=exposure_pose(sec(m.header))
 if sample is None:return
 # Converter publishes tightly packed xyz float32; validate before interpreting.
 if m.point_step!=12 or m.header.frame_id!='vio_imu':return
 points=np.frombuffer(m.data,dtype='<f4').reshape(-1,3)
 position,rot=sample;world=world0+map_shift+align.apply(rot.apply(points)+position-local0)
 world[:,2]+=world_body()[2]-(world0+map_shift+align.apply(local_body()[0]-local0))[2]
 pivot=world_body();perception.submit('cloud',(world.copy(),pivot.copy(),sec(m.header)))
 state['tree']=cKDTree(world/np.array([.32,.32,.18]));state['cloud_stamp']=sec(m.header);state['last_observed_position']=world_body().copy()
subs.append(node.create_subscription(PointCloud2,'/carto/points',cloud_cb,qos_profile_sensor_data))
def send(world,yaw,velocity=None,reference=None):
 if velocity is not None:
  now=stamp()*1e-6;previous=state['velocity_stamp'];dt=.05 if previous is None else min(.1,max(0.,now-previous));state['velocity_stamp']=now
  # Slew physical velocity commands; stale observations and close obstacles
  # are already checked before this call. Emergency holds bypass the ramp.
  if np.linalg.norm(velocity)>.001:velocity=slew_velocity(state['velocity_command'],velocity,dt,1.)
  state['velocity_command']=np.asarray(velocity).copy()
 error=np.asarray(world)-world_body();px=state['px'];ned=np.array([px.x,px.y,px.z])
 if velocity is None:ned+=D@align.inv().apply(error)
 else:
  # PX4 uses absolute height feedback while XY follows a velocity trajectory.
  # Re-sending measured Z as the target lets observation delay become descent.
  feedback=np.array([0.,0.,error[2]])
  if reference is not None:
   feedback=np.asarray(reference)-world_body();length=np.linalg.norm(feedback[:2])
   if length>.20:feedback[:2]*=.20/length
  ned+=D@align.inv().apply(feedback)
  velocity=np.asarray(velocity).copy();velocity[2]=0.
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
 for phase_index,(stage,path) in enumerate(phases):
  phase_pub.publish(String(data=stage.upper()))
  actual=world_body();points=np.vstack([actual,path]) if np.linalg.norm(path[0]-actual)>.01 else path
  displayed_stage=None;follower=Follower(points);goal=np.asarray(path[-1]);reached=None;blocked_since=None;deadline=stamp()+60000000
  while True:
   check();now=stamp()
   if now-run_start>175000000:raise RuntimeError('Run deadline: land before 180 s')
   if now>deadline:raise RuntimeError('Stage progress timeout '+stage)
   actual=world_body();distance=np.linalg.norm(goal-actual)
   tolerance=.035 if stage=='takeoff' else .075
   if distance<tolerance:
    reached=now if reached is None else reached
    if now-reached>150000:break
   else:reached=None
   reference=None
   if stage=='takeoff':target=goal.copy();velocity=None;tracking={}
   else:
    cruise=.75 if stage=='tunnel_exit' else args.speed
    if stamp()*1e-6-state['cloud_stamp']>.2:cruise=min(cruise,.18)
    target,velocity,tracking=follower.target(actual,cruise);reference=follower.point(follower.progress);reference[2]=target[2]
    # Keep absolute Z position feedback while horizontal speed follows the path.
    lookahead=follower.point(min(follower.s[-1],follower.progress+.55))-actual
    if distance<.35 and phase_index+1<len(phases):
     next_path=phases[phase_index+1][1];next_goal=next((p for p in next_path if np.linalg.norm(p[:2]-goal[:2])>.3),next_path[-1]);lookahead=next_goal-actual
    if np.linalg.norm(velocity[:2])>.03:yaw=math.atan2(velocity[1],velocity[0])
    elif np.linalg.norm(lookahead[:2])>.10:yaw=math.atan2(lookahead[1],lookahead[0])
    actual_yaw=(align*local_body()[1]).as_euler('xyz')[2];yaw_error=math.atan2(math.sin(yaw-actual_yaw),math.cos(yaw-actual_yaw))
    velocity*=visibility_speed_factor(yaw_error,math.atan2(line_localization.resolution[0]/2,line_localization.K[0]))
    if not grid.segment(actual,target):
     reference=None
     try:
      repair=grid.plan(actual,goal);detour=next(point for point in repair[1:] if np.linalg.norm(point-actual)>.025);detour_delta=detour-actual;target=actual+detour_delta*min(1.,.15/max(np.linalg.norm(detour_delta),1e-6))
      if not grid.segment(actual,target):raise RuntimeError('Local repair segment blocked')
      yaw=math.atan2(detour_delta[1],detour_delta[0]);velocity=detour_delta*.7
      if np.linalg.norm(velocity)>.15:velocity*=.15/np.linalg.norm(velocity)
     except (RuntimeError,ValueError,StopIteration):
      if blocked_since is None:blocked_since=now
      if now-blocked_since>5000000:raise RuntimeError('No safe CAD repair from current estimate: '+str(actual.tolist()))
      offsets=np.array([[.025,0,0],[-.025,0,0],[0,.025,0],[0,-.025,0]]);values=registration.sample(actual+offsets);gradient=np.array([values[0]-values[1],values[2]-values[3],0.])
      if np.linalg.norm(gradient)>.0001:gradient/=np.linalg.norm(gradient);target=actual+gradient*.12;velocity=gradient*.08;state['clearance_recovery_ticks']+=1
      else:target=actual;velocity=np.zeros(3)
    vnorm=np.linalg.norm(velocity)
    if vnorm>.01 and grid.free(grid.idx(actual)):
     direction=velocity/vnorm;latency=max(.10,stamp()*1e-6-state['cloud_stamp'])+.08;uncertainty=max(.03,min(.12,state['health'].get('inertial_increment_std_bound_m',.03)));max_probe=min(.65,max(.18,stopping_distance(vnorm,latency,uncertainty=uncertainty)));free_distance=0.
     for probe in np.arange(.03,max_probe+.03,.03):
      endpoint=actual+direction*probe
      if not grid.segment(actual,endpoint):break
      try:veto(actual,endpoint)
      except RuntimeError:break
      free_distance=float(probe)
     cap=safe_speed(free_distance,latency,uncertainty=uncertainty);state['motion_debug']={'clearance_m':free_distance,'latency_s':latency,'uncertainty_m':uncertainty,'requested_mps':float(vnorm),'cap_mps':float(cap)}
     if cap<vnorm:velocity*=cap/vnorm;state['motion_limited_ticks']+=1
    try:
     veto(actual,target)
     if grid.free(grid.idx(actual)):blocked_since=None
    except RuntimeError as obstacle:
     if blocked_since is None:blocked_since=now;node.get_logger().warning(str(obstacle)+'; hover and reassess')
     if now-blocked_since>5000000:raise
     target=actual;velocity=np.zeros(3);reference=None
   active_stage=stage
   if stage in route_segments:
    active_stage=next((label for end,label in route_segments[stage] if follower.progress<end+.01),stage)
   if active_stage!=displayed_stage:phase_pub.publish(String(data=active_stage.upper()));displayed_stage=active_stage
   if now-last_trace>200000:
    trace.append({'stamp':now*1e-6,'stage':active_stage,'position':actual.tolist(),'goal':goal.tolist(),'yaw':float((align*local_body()[1]).as_euler('xyz')[2]),'motion':state['motion_debug'],'tracking':tracking});last_trace=now
   if now-last_log>2000000:node.get_logger().info(active_stage+' pos='+str(np.round(actual,3).tolist())+' target='+str(np.round(target,3).tolist()));last_log=now
   send(target,yaw,velocity,reference)
  records.append({'stage':stage,'stamp_us':stamp(),'position':world_body().tolist()})
  if stage=='window_observe':
   yaw=math.atan2(window_localization.center[1]-world_body()[1],window_localization.center[0]-world_body()[0]);hold=world_body().copy();until=stamp()+10000000;stable=None
   while stamp()<until:
    check();now=stamp();h=window_localization.health
    good=h is not None and now*1e-6-h['stamp']<.5 and h['supported'] and h['before_px']<5 and np.linalg.norm(h['fit_delta'][:2])<.08
    stable=now if good and stable is None else (stable if good else None)
    if stable is not None and now-stable>400000:break
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
 perception.close()
 Path(args.output).write_text(json.dumps({'route':args.route,'perception':{'superseded_jobs':perception.superseded,'compute_ms':{'median':float(np.median(perception.compute_ms)) if perception.compute_ms else None,'p95':float(np.percentile(perception.compute_ms,95)) if perception.compute_ms else None},'error':perception.error},'controller_phases':records,'trace':trace,'failure':failure,'controller_complete':success and disarmed,'disarmed':disarmed,'run_time_s':(stamp()-run_start)*1e-6,'full_course_verified':False,'scope':'provisional CAD annotations; independent judge required','parameters':params,'prior_transition_ticks':state['prior_transitions'],'clearance_recovery_ticks':state['clearance_recovery_ticks'],'motion_limited_ticks':state['motion_limited_ticks'],'window_registration':window_localization.events,'range_height_rejections':height['rejected'],'line_registration':line_localization.events,'cad_registration':registration.events,'final_map_shift':map_shift.tolist()},indent=2))
 node.destroy_node();rclpy.shutdown()
if failure:raise SystemExit(failure)
