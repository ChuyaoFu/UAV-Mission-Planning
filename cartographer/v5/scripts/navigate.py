#!/usr/bin/python3
"""Unknown-layout frontier exploration and measured-map return. SITL arming only.
No scene files, CAD registration, absolute start pose or preset route targets.
"""
import os,json,time,math,signal
from collections import Counter
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy
from std_msgs.msg import String,Int8MultiArray,Float64MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import VehicleStatus,VehicleLocalPosition,VehicleCommand,OffboardControlMode,TrajectorySetpoint
from quadrotor_msgs.msg import PositionCommand
from scipy.spatial.transform import Rotation
from online_map import OnlineMap
from perception_worker import LatestWorker
D=np.diag([1.,-1.,-1.]);ROOT=Path(__file__).resolve().parents[1]
def wrap(a):return math.atan2(math.sin(a),math.cos(a))
class Navigator(Node):
 def __init__(self,output):
  super().__init__('cadfree_navigator',parameter_overrides=[Parameter('use_sim_time',value=os.environ.get('USE_SIM_TIME','1')=='1')]);self.c=json.loads((ROOT/'config/navigation.json').read_text());self.output=Path(output);self.body=None;self.q=None;self.odom_t=-1.;self.health=None;self.px=None;self.status=None;self.reference=None;self.ref_t=-1.;self.meta=None;self.map=None;self.home=None;self.yaw=0.;self.trace=[];self.goals=[];self.phase='INITIALIZING';self.worker=LatestWorker(self.build_map);self.armed=False;self.failure=None;self.complete=False;self.distance=0.;self.previous=None;self.max_excursion=0.;self.safety_vetoes=0;self.veto_reasons=Counter();self.loop_alive=True;self.last_visual_yaw=None;self.last_replan=-1.;self.waiting_trajectory_id=None;self.last_received_trajectory_id=None
  self.brake_hold=None;self.cancelled=False;self.body_velocity=np.zeros(3)
  self.goal_pub=self.create_publisher(PoseStamped,'/nav/exploration_goal',5);self.mode=self.create_publisher(OffboardControlMode,'/fmu/in/offboard_control_mode',10);self.traj=self.create_publisher(TrajectorySetpoint,'/fmu/in/trajectory_setpoint',10);self.command_pub=self.create_publisher(VehicleCommand,'/fmu/in/vehicle_command',10);self.phase_pub=self.create_publisher(String,'/carto/mission_state',10)
  self.subs=[self.create_subscription(Odometry,'/nav/body_odom',self.odom,10),self.create_subscription(String,'/carto/localization_health',lambda m:setattr(self,'health',json.loads(m.data)),5),self.create_subscription(VehicleLocalPosition,'/fmu/out/vehicle_local_position_v1',lambda m:setattr(self,'px',m),qos_profile_sensor_data),self.create_subscription(VehicleStatus,'/fmu/out/vehicle_status_v1',lambda m:setattr(self,'status',m),qos_profile_sensor_data),self.create_subscription(PositionCommand,'/position_cmd',self.trajectory,2)]
  latched=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
  self.subs += [self.create_subscription(Float64MultiArray,'/grid_map/metadata',lambda m:setattr(self,'meta',list(m.data)),latched),self.create_subscription(Int8MultiArray,'/grid_map/knowledge',self.knowledge,latched)]
 def now(self):return self.get_clock().now().nanoseconds*1e-9
 def odom(self,m):
  v=m.twist.twist.linear;p=m.pose.pose.position;q=m.pose.pose.orientation;self.body=np.array([p.x,p.y,p.z]);self.q=Rotation.from_quat([q.x,q.y,q.z,q.w]);self.body_velocity=self.q.apply([v.x,v.y,v.z]);self.odom_t=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
  if self.armed and self.previous is not None:
   delta=float(np.linalg.norm(self.body-self.previous))
   if delta<.2:self.distance+=delta
  self.previous=self.body.copy()
  if self.health and self.health.get('accepted_visual_update_features',0)>=3 and self.health.get('visual_update_age',1.)<.3:self.last_visual_yaw=self.q.as_euler('xyz')[2]
  if self.home is not None:self.max_excursion=max(self.max_excursion,float(np.linalg.norm(self.body[:2]-self.home[:2])))
 def trajectory(self,m):
  if self.last_received_trajectory_id is not None and m.trajectory_id<self.last_received_trajectory_id:return
  self.last_received_trajectory_id=m.trajectory_id
  if self.waiting_trajectory_id is not None and m.trajectory_id<=self.waiting_trajectory_id:return
  self.waiting_trajectory_id=None;self.reference=m;self.ref_t=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
 def knowledge(self,m):
  if self.meta is not None:self.worker.submit('map',(self.meta.copy(),np.asarray(m.data,dtype=np.int8).copy()))
 def build_map(self,kind,data):return [OnlineMap(*data,radius=self.c['vehicle_radius_m']+self.c['xy_margin_m'],z_half=self.c['vertical_half_m']+self.c['vertical_margin_m'])]
 def pump(self,dt=.05):
  end=time.monotonic()+dt
  while time.monotonic()<end:rclpy.spin_once(self,timeout_sec=min(.01,max(0.,end-time.monotonic())))
  for m in self.worker.take():self.map=m
 def phase_set(self,p):self.phase=p;self.phase_pub.publish(String(data=p));self.get_logger().info(p)
 def healthy(self):
  h=self.health
  return self.body is not None and self.px is not None and self.px.xy_valid and self.px.z_valid and h and h['valid'] and self.now()-h['stamp']<.25 and self.now()-self.odom_t<.20
 def command(self,c,p1=0.,p2=0.):self.command_pub.publish(VehicleCommand(timestamp=int(self.now()*1e6),command=c,param1=float(p1),param2=float(p2),target_system=1,target_component=1,source_system=250,source_component=1,from_external=True))
 def send(self,target,yaw,velocity=None):
  error=np.asarray(target)-self.body;px=np.array([self.px.x,self.px.y,self.px.z]);local_yaw=self.q.as_euler('xyz')[2];phi=self.px.heading+local_yaw;frame=Rotation.from_euler('z',phi).as_matrix()@D;ned=px+frame@error;ned_yaw=wrap(self.px.heading-wrap(yaw-local_yaw))
  self.mode.publish(OffboardControlMode(timestamp=int(self.now()*1e6),position=True));self.traj.publish(TrajectorySetpoint(timestamp=int(self.now()*1e6),position=ned.astype(float).tolist(),velocity=[math.nan]*3 if velocity is None else (frame@np.asarray(velocity)).astype(float).tolist(),acceleration=[math.nan]*3,jerk=[math.nan]*3,yaw=float(ned_yaw),yawspeed=math.nan))
  if len(self.trace)==0 or self.now()-self.trace[-1]['stamp']>.2:self.trace.append({'stamp':self.now(),'phase':self.phase,'position':self.body.tolist(),'target':np.asarray(target).tolist(),'yaw':float(local_yaw),'health':self.health['reason'],'map_observation_age_s':None if self.map is None else self.now()-self.map.stamp})
  self.pump()
 def check(self):
  if self.cancelled:raise RuntimeError('Trial cancelled; controlled landing requested')
  if self.armed and hasattr(self,'run_start') and self.now()-self.run_start>self.c['max_mission_seconds']-20:raise RuntimeError('Mission deadline: reserve time for safe landing')
  if self.healthy():return
  if self.body is None or self.px is None:raise RuntimeError('Localization unavailable')
  self.get_logger().warning('Localization degraded: brake and reacquire')
  hold=[float(self.px.x),float(self.px.y),float(self.px.z)];start=self.now();local_yaw=self.q.as_euler('xyz')[2];look=local_yaw if self.last_visual_yaw is None else self.last_visual_yaw;native_yaw=wrap(self.px.heading-wrap(look-local_yaw))
  while self.now()-start<2.5:
   now=int(self.now()*1e6);self.mode.publish(OffboardControlMode(timestamp=now,position=True));self.traj.publish(TrajectorySetpoint(timestamp=now,position=hold,velocity=[math.nan]*3,acceleration=[math.nan]*3,jerk=[math.nan]*3,yaw=float(native_yaw),yawspeed=math.nan));self.pump()
   if self.healthy():return
  raise RuntimeError('Localization recovery budget exhausted')
 def observe(self,seconds=8.,full_sweep=False):
  # Survey a bounded arc; turn back before visual constraints disappear.
  self.phase_set('OBSERVE');hold=self.body.copy();yaw=self.q.as_euler('xyz')[2];start=self.now();direction=1.;target=yaw;turned=False
  while self.now()-start<seconds:
   self.check();h=self.health;age=h.get('visual_update_age',99.);current=self.q.as_euler('xyz')[2]
   if age>.8 and self.last_visual_yaw is not None:
    self.send(hold,self.last_visual_yaw);turned=True
    if abs(wrap(current-self.last_visual_yaw))<.1:break
    continue
   if full_sweep:
    self.send(hold,yaw+2*math.pi*min(1.,(self.now()-start)/seconds));continue
   if abs(wrap(current-yaw))>.75:direction=-1.;turned=True
   target=current+direction*.18
   self.send(hold,target)
   if turned and wrap(current-yaw)<-.6:break
  self.yaw=self.q.as_euler('xyz')[2]
 def turn_to_frontier(self,goal):
  delta=np.asarray(goal)-self.body
  if np.linalg.norm(delta[:2])<.08:return
  bearing=math.atan2(delta[1],delta[0]);hold=self.body.copy();deadline=self.now()+12.;self.phase_set('TURN_TO_OBSERVED_FRONTIER')
  while abs(wrap(bearing-self.q.as_euler('xyz')[2]))>.12:
   self.check();current=self.q.as_euler('xyz')[2];self.send(hold,current+float(np.clip(wrap(bearing-current),-.25,.25)))
   if self.now()>deadline:raise RuntimeError('Observable frontier turn timed out')
  self.yaw=bearing;self.phase_set('EXPLORE')
 def publish_goal(self,p,reason):
  self.brake_hold=self.body.copy()
  self.waiting_trajectory_id=self.last_received_trajectory_id;self.last_replan=self.now();self.reference=None
  m=PoseStamped();m.header.stamp=self.get_clock().now().to_msg();m.header.frame_id='world';m.pose.position.x=float(p[0]);m.pose.position.y=float(p[1]);m.pose.position.z=float(p[2]);m.pose.orientation.w=1.;self.goal_pub.publish(m);self.goals.append({'stamp':self.now(),'position':np.asarray(p).tolist(),'reason':reason});self.get_logger().info('ONLINE GOAL '+str(np.round(p,2).tolist()))
 def replan_if_held(self):
  if self.brake_hold is not None and (np.linalg.norm(self.body_velocity)>.05 or np.linalg.norm(self.body-self.brake_hold)>.05):return
  if self.map is None or self.now()-self.map.stamp>self.c['max_observation_age_s']:return
  if self.goals and self.now()-self.last_replan>1.:
   p=self.goals[-1]['position'];m=PoseStamped();m.header.stamp=self.get_clock().now().to_msg();m.header.frame_id='world';m.pose.position.x=float(p[0]);m.pose.position.y=float(p[1]);m.pose.position.z=float(p[2]);m.pose.orientation.w=1.;self.goal_pub.publish(m);self.last_replan=self.now()
 def brake(self):
  if self.brake_hold is None:
   self.brake_hold=self.body.copy();self.waiting_trajectory_id=self.last_received_trajectory_id;self.reference=None
  self.replan_if_held();self.send(self.brake_hold,self.yaw)
 def follow(self):
  self.check();r=self.reference
  if self.brake_hold is not None and (np.linalg.norm(self.body_velocity)>.05 or np.linalg.norm(self.body-self.brake_hold)>.05):
   self.brake();return
  if r is None or self.now()-self.ref_t>.2 or self.map is None or self.now()-self.map.stamp>self.c['max_observation_age_s']:
   self.safety_vetoes+=1;self.veto_reasons['missing_or_stale_measurements']+=1;self.brake();return
  target=np.array([r.position.x,r.position.y,r.position.z]);v=np.array([r.velocity.x,r.velocity.y,r.velocity.z]);delta=target-self.body
  if np.linalg.norm(delta)>.20:target=self.body+delta*.20/np.linalg.norm(delta)
  speed=np.linalg.norm(v);limit=min(.20,self.c['max_speed_mps']) if self.phase.startswith('RETURN') else self.c['max_speed_mps']
  if speed>limit:v*=limit/speed
  # Keep the observed-viewpoint bearing through acceleration and braking.
  # A negative tracking velocity is a braking/correction command, not a request
  # to yaw 180 degrees and lose the tracked visual scene.
  yaw_error=wrap(self.yaw-self.q.as_euler('xyz')[2])
  end=target+v*.3
  actual_speed=np.linalg.norm(self.body_velocity);stop=self.body+self.body_velocity*(self.c['max_observation_age_s']+actual_speed/(2*self.c['max_acceleration_mps2']))
  reasons=[]
  if abs(yaw_error)>.45:reasons.append('heading')
  if not self.map.segment(self.body,target):reasons.append('body_to_reference_clearance')
  if not self.map.segment(self.body,end):reasons.append('stopping_projection_clearance')
  if not self.map.segment(self.body,stop):reasons.append('measured_velocity_stopping_clearance')
  if reasons:
   self.veto_reasons.update(reasons);self.safety_vetoes+=1;self.brake();return
  self.brake_hold=None;self.send(target,self.yaw,v)
 def run(self):
  start=time.monotonic()
  while time.monotonic()-start<45:
   self.pump()
   if self.healthy() and self.map is not None:break
  else:raise RuntimeError('No healthy odometry/online map; did not arm')
  self.home=self.body.copy();self.yaw=self.q.as_euler('xyz')[2];hold=self.home.copy()
  for _ in range(50):self.check();self.send(hold,self.yaw)
  self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,1.,6.);self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,1.)
  for _ in range(100):
   self.send(hold,self.yaw)
   if self.status is not None and self.status.arming_state==VehicleStatus.ARMING_STATE_ARMED:self.armed=True;break
  if not self.armed:raise RuntimeError('Arming denied')
  self.run_start=self.now();hover=self.home+np.array([0.,0.,self.c['takeoff_height_m']]);self.phase_set('TAKEOFF_OPERATOR_VERIFIED_ZONE')
  survey=self.home+np.array([0.,0.,self.c['takeoff_survey_height_m']]);deadline=self.now()+25
  while np.linalg.norm(self.body-survey)>.06:
   self.check();self.send(survey,self.yaw)
   if self.now()>deadline:raise RuntimeError('Takeoff survey timeout')
  # Observe from the high verified pose before descending: downward stereo
  # needs this viewpoint to measure the upper clearance of cruise corridors.
  self.observe(8.,full_sweep=True)
  self.phase_set('VERTICAL_OBSERVABILITY_SWEEP')
  deadline=self.now()+10
  while np.linalg.norm(self.body-hover)>.05:
   self.check();self.send(hover,self.yaw)
   if self.now()>deadline:raise RuntimeError('Vertical survey descent timeout')
  # The high view measures roof clearance; the low view measures the lower
  # swept volume. A fixed downward camera cannot replace either observation.
  self.observe(8.,full_sweep=True)
  self.phase_set('EXPLORE');explore_start=self.now();goal=None;goal_start=-1.;visited=[self.body.copy()];last_advance=self.now();last_position=self.body.copy()
  while self.now()-explore_start<self.c['explore_seconds']:
   if self.now()-self.run_start>self.c['max_mission_seconds']-35:break
   if np.linalg.norm(self.body-last_position)>.15:last_position=self.body.copy();last_advance=self.now()
   reached=goal is not None and np.linalg.norm(self.body-goal)<.05 and np.linalg.norm(self.body_velocity)<.06
   if goal is None or reached or self.now()-goal_start>10:
    if goal is not None:visited.append(self.body.copy())
    if self.map is None:raise RuntimeError('Online map missing')
    candidate=self.map.frontier(self.body,self.home,visited,self.q.as_euler('xyz')[2],cruise_height=self.c['takeoff_height_m'],max_heading_change=math.pi)
    if candidate is None:
     self.observe(6.);candidate=self.map.frontier(self.body,self.home,visited,self.q.as_euler('xyz')[2],cruise_height=self.c['takeoff_height_m'],max_heading_change=math.pi);self.phase_set('EXPLORE')
    if candidate is None:
     candidate=self.map.frontier(self.body,self.home,visited,self.q.as_euler('xyz')[2],cruise_height=self.c['takeoff_height_m'],max_heading_change=math.pi,allow_altitude_change=True)
    if candidate is None:break
    goal,reason=candidate;self.turn_to_frontier(goal);self.publish_goal(goal,reason);goal_start=self.now();last_advance=self.now();last_position=self.body.copy()
   self.follow()
   if self.now()-last_advance>10:break
   if self.max_excursion>=self.c['minimum_excursion_m'] and self.distance>=self.c['minimum_outbound_distance_m']:break
  self.phase_set('RETURN_ON_OBSERVED_MAP');self.publish_goal(hover,{'source':'own measured initial body pose, not a course coordinate'});deadline=self.now()+40
  while np.linalg.norm(self.body-hover)>.15:
   self.follow()
   if self.now()>deadline:raise RuntimeError('Return timeout')
  self.phase_set('HOME_SETTLE');deadline=self.now()+8.;settled=None
  while True:
   self.check()
   if self.map is None or self.now()-self.map.stamp>self.c['max_observation_age_s']:
    if self.now()>deadline:raise RuntimeError('Home settling fresh observations timed out')
    self.send(self.body.copy(),self.yaw);continue
   if not self.map.segment(self.body,hover):raise RuntimeError('Home settling corridor is not verified free')
   if np.linalg.norm(self.body-hover)<.04:
    if settled is None:settled=self.now()
    if self.now()-settled>.7:break
   else:settled=None
   if self.now()>deadline:raise RuntimeError('Home settling timeout')
   delta=hover-self.body;target=self.body+delta*min(1.,.08/max(np.linalg.norm(delta),1e-9));self.send(target,self.yaw)
  self.complete=self.max_excursion>=self.c['minimum_excursion_m'] and self.distance>=self.c['minimum_outbound_distance_m']
  if not self.complete:raise RuntimeError('Insufficient exploration: stopped safely, not a successful navigation episode')
 def finish(self):
  if self.armed:
   self.phase_set('LAND_HOME' if self.complete else 'ABORT_LAND');self.command(VehicleCommand.VEHICLE_CMD_NAV_LAND);deadline=time.monotonic()+35
   while time.monotonic()<deadline:
    self.pump()
    if self.status is not None and self.status.arming_state!=VehicleStatus.ARMING_STATE_ARMED:break
   else:self.failure=(self.failure or '')+'; disarm timeout';self.complete=False
  disarmed=self.status is not None and self.status.arming_state!=VehicleStatus.ARMING_STATE_ARMED
  if disarmed:self.phase_set('DISARMED_COMPLETE' if self.complete and not self.failure else 'DISARMED_INCOMPLETE')
  self.output.write_text(json.dumps({'controller_complete':bool(self.complete and disarmed and not self.failure),'failure':self.failure,'disarmed':disarmed,'home_measured_body':None if self.home is None else self.home.tolist(),'goals':self.goals,'trace':self.trace,'measured_travel_m':self.distance,'max_measured_horizontal_excursion_m':self.max_excursion,'safety_vetoes':self.safety_vetoes,'safety_veto_reasons':dict(self.veto_reasons),'online_measured_free_cells':0 if self.map is None else int((self.map.state==0).sum()),'online_occupied_cells':0 if self.map is None else int((self.map.state==1).sum()),'scope':'unknown-layout exploration/obstacle avoidance/return baseline; competition tasks not yet enabled','mission_coordinate_source':'own first valid odometry; arbitrary local gauge','preset_waypoints':False,'prior_map_loaded':False,'launch_assumption':self.c['launch_envelope_assumption']},indent=2));self.worker.close()
def install_stop_handlers(n):
 # rclpy must stay alive until finish() has commanded and observed landing.
 signal.signal(signal.SIGINT,lambda *_:setattr(n,'cancelled',True))
 signal.signal(signal.SIGTERM,lambda *_:setattr(n,'cancelled',True))

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();assert Path('/.dockerenv').exists() and os.environ.get('ROS_DOMAIN_ID')=='42' and os.environ.get('SITL_AUTOMATION_AUTHORIZED')=='1','Automatic arming remains SITL-only';from rclpy.signals import SignalHandlerOptions
 rclpy.init(signal_handler_options=SignalHandlerOptions.NO);n=Navigator(a.output);install_stop_handlers(n)
 try:n.run()
 except Exception as e:n.failure=str(e);n.get_logger().error(str(e))
 finally:n.finish();n.destroy_node();rclpy.shutdown()
