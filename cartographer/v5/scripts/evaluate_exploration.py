#!/usr/bin/python3
"""Independent evaluator only: simulator poses/contacts never enter ROS.
Navigation cannot read its output directory under the Landlock profile.
"""
import argparse,json,time,threading,math,signal,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

def score(data):
 r={'engineering_complete':False,'failure':None,'collisions':0,'truth_use':'independent evaluation only'};p=np.array(data['poses']);statuses=data['statuses']
 if not len(p) or not statuses or not data['contact_topics_verified']:r['failure']='Missing independent pose/status/contact evidence';return r
 transitions=[];last=False
 for s in statuses:
  if s['armed']!=last:transitions.append(s);last=s['armed']
 if len(transitions)!=2 or not transitions[0]['armed'] or transitions[1]['armed']:r['failure']='No single complete arm/land cycle';return r
 a,b=transitions[0]['stamp'],transitions[1]['stamp'];start=p[0,1:4];air=np.flatnonzero(p[:,3]>start[2]+.15)
 if not len(air):r['failure']='Never airborne';return r
 depart=np.flatnonzero(np.linalg.norm(p[:,1:4]-start,axis=1)>.015);motion=p[depart[0],0] if len(depart) else a;active=p[(p[:,0]>=a)&(p[:,0]<=b+.2)];end=p[np.argmin(abs(p[:,0]-b))];excursion=float(np.max(np.linalg.norm(active[:,1:3]-start[:2],axis=1)));home=float(np.linalg.norm(end[1:3]-start[:2]));distance=float(np.linalg.norm(np.diff(active[:,1:4],axis=0),axis=1).sum());last_contact={};home_ground=False
 for c in data['contacts']:
  t=c['stamp']
  if t<p[air[0],0] or t>b+.2:continue
  names=' '.join(c['pair']).lower();ground=any(x in names for x in ['floor_20_x_20_ft','guideline','landing_pad','approach_ring','launch_mark','ground_plane','floor'])
  if ground:
   xy=p[np.argmin(abs(p[:,0]-t)),1:3]
   if np.linalg.norm(xy-start[:2])>.24:r['failure']='Ground contact away from home'
   if b-3<t<=b+.2 and np.linalg.norm(xy-start[:2])<.24:home_ground=True
  else:
   obstacle=next((x for x in c['pair'] if 'x500_' not in x and 'drone' not in x),c['pair'][0])
   if t-last_contact.get(obstacle,-math.inf)>.25:r['collisions']+=1
   last_contact[obstacle]=t
 r.update(elapsed_s=float(b-motion),arm_to_disarm_s=float(b-a),max_horizontal_excursion_m=excursion,travel_m=distance,home_center_error_m=home,disarmed=True,ground_at_home=home_ground)
 if r['collisions']:r['failure']=r['failure'] or 'Obstacle contact'
 if abs(end[0]-b)>.2:r['failure']=r['failure'] or 'No fresh independent landing pose'
 if home>.20 or end[3]>start[2]+.12 or not home_ground:r['failure']=r['failure'] or 'Home landing not verified'
 if excursion<1.15 or distance<2.7:r['failure']=r['failure'] or 'Insufficient independent exploration motion'
 r['engineering_complete']=r['failure'] is None;return r

def record(seconds,output,ready_file=None):
 import rclpy
 from rclpy.signals import SignalHandlerOptions
 from rclpy.node import Node
 from rclpy.parameter import Parameter
 from rclpy.qos import qos_profile_sensor_data
 from px4_msgs.msg import VehicleStatus
 from gz.transport13 import Node as GzNode
 from gz.msgs10.pose_v_pb2 import Pose_V
 from gz.msgs10.contacts_pb2 import Contacts
 stop=[False];signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True));rclpy.init(signal_handler_options=SignalHandlerOptions.NO);n=Node('independent_unknown_layout_evaluator',parameter_overrides=[Parameter('use_sim_time',value=True)]);gz=GzNode();data={'poses':[],'statuses':[],'contacts':[],'contact_topics_verified':False};lock=threading.Lock();seen=set();last_seen={}
 def pose(m):
  for p in m.pose:
   if p.name=='x500_vio_0':
    t=p.header.stamp.sec+p.header.stamp.nsec*1e-9
    with lock:data['poses'].append([t,p.position.x,p.position.y,p.position.z,p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w])
 def status(m):data['statuses'].append({'stamp':m.timestamp*1e-6,'armed':m.arming_state==VehicleStatus.ARMING_STATE_ARMED})
 def contact(m):
  t=m.header.stamp.sec+m.header.stamp.nsec*1e-9
  with lock:
   for c in m.contact:
    pair=tuple(sorted([c.collision1.name,c.collision2.name]));seen.add(pair)
    if t-last_seen.get(pair,-math.inf)>.02:data['contacts'].append({'stamp':t,'pair':list(pair)});last_seen[pair]=t
 gz.subscribe(Pose_V,'/model/x500_vio_0/pose',pose);topics=[]
 for link in ET.parse('/work/cartographer/v5/config/model.sdf').findall('.//link'):
  for s in link.findall('sensor'):
   if s.get('type')=='contact':
    topic=f'/world/visual_nav/model/x500_vio_0/link/{link.get("name")}/sensor/{s.get("name")}/contact';topics.append(topic);gz.subscribe(Contacts,topic,contact)
 sub=n.create_subscription(VehicleStatus,'/fmu/out/vehicle_status_v1',status,qos_profile_sensor_data);end=time.monotonic()+seconds
 try:
  while time.monotonic()<end and not stop[0]:
   rclpy.spin_once(n,timeout_sec=.05)
   if ready_file and data['poses'] and data['statuses'] and seen and not Path(ready_file).exists():Path(ready_file).write_text(json.dumps({'native_pose_samples':len(data['poses']),'contact_pairs':len(seen),'status_samples':len(data['statuses'])}))
 except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
 data['contact_topics_verified']=bool(seen)
 Path(output).write_text(json.dumps(data))
 data['result']=score(data);Path(output).write_text(json.dumps(data,indent=2));print(json.dumps(data['result'],indent=2));n.destroy_node();rclpy.try_shutdown()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=180);p.add_argument('--output',required=True);p.add_argument('--ready-file');a=p.parse_args();record(a.seconds,a.output,a.ready_file)
