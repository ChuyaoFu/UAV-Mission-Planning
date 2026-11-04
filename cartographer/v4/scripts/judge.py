#!/usr/bin/python3
"""Independent read-only engineering judge. No flight-control publications.
CAD-derived regions are provisional; result cannot certify official scoring.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
R=.26;ZHALF=.12;PAD=np.array([-2.04216,-.79248]);PADR=.762

def judge(data):
 poses=sorted(data.get('poses',[]),key=lambda r:r[0]);statuses=data.get('statuses',[]);contacts=data.get('contacts',[])
 result={'engineering_complete':False,'official_completion_certified':False,'boundary_authority':'provisional CAD engineering regions','checkpoints':[],'collision_events':0,'penalty_s':0,'failure':None}
 if not poses or not statuses or not data.get('contact_topics_verified',False):result['failure']='Missing pose/status/contact evidence';return result
 p=np.array(poses);basez=p[0,3];air=np.flatnonzero(p[:,3]>basez+.15)
 if not len(air):result['failure']='Never airborne';return result
 t_air=p[air[0],0]
 depart=np.flatnonzero(np.linalg.norm(p[:,1:4]-p[0,1:4],axis=1)>.015);t_start=float(p[depart[0],0])
 result['start_time_s']=t_start
 # Conservative whole-body circular footprint. Provisional semantic regions.
 tests=[('first_corridor',lambda x,y,z: -3.048+R<x<-1.9812-R and 2.1209+R<y<3.048-R),('gate_a',lambda x,y,z:.3556+R<x<1.5748-R and 2.1209+R<y<3.048-R),('tunnel_exit',lambda x,y,z:1.57489+R<x<3.04793-R and y<-.3048-R and z+ZHALF<1.8542),('gate_or_window',lambda x,y,z:x<-1.02616-R and y<-.6223-R)]
 next_stage=0;failure_time=math.inf
 for row in p[air[0]:]:
  t,x,y,z=row[:4]
  if t-t_start>180:result['failure']='180 second limit';failure_time=t;break
  if abs(x)+R>3.048 or abs(y)+R>3.048:result['failure']='Vehicle exits arena';failure_time=t;break
  if z-ZHALF>=1.7272:result['failure']='Over-wall route';failure_time=t;break
  if next_stage<len(tests) and tests[next_stage][1](x,y,z):
   result['checkpoints'].append({'name':tests[next_stage][0],'time_s':float(t-t_start)});next_stage+=1
 # Debounce same continuous contact; all pairs counted, not first N messages.
 last={};nonpad_ground=[]
 for c in sorted(contacts,key=lambda c:c['stamp']):
  t=c['stamp']
  if t<t_air or t>failure_time:continue
  pair=tuple(sorted(c['pair']));names=' '.join(pair).lower()
  ground=any(s in names for s in ['floor_20_x_20_ft','guideline','landing_pad','approach_ring','launch_mark'])
  if ground:
   row=p[np.argmin(abs(p[:,0]-t))];inside=np.linalg.norm(row[1:3]-PAD)+R<=PADR
   if not inside:nonpad_ground.append(t)
  else:
   obstacle=next((name for name in pair if 'x500_' not in name and 'drone' not in name),pair[0])
   if t-last.get(obstacle,-math.inf)>.25:result['collision_events']+=1
   last[obstacle]=t
 if nonpad_ground:
  failure_time=min(failure_time,min(nonpad_ground));result['failure']='Non-platform ground contact'
  result['checkpoints']=[s for s in result['checkpoints'] if t_start+s['time_s']<failure_time]
 result['penalty_s']=result['collision_events']*5
 armed=[s for s in statuses if s['armed']]
 disarmed=[s for s in statuses if not s['armed'] and armed and s['stamp']>armed[0]['stamp']]
 if not disarmed:
  result['failure']=result['failure'] or 'No final disarm evidence';return result
 t_end=disarmed[0]['stamp'];row=p[np.argmin(abs(p[:,0]-t_end))]
 on_pad=np.linalg.norm(row[1:3]-PAD)+R<=PADR and row[3]<basez+.12
 ground_on_pad=any(t_end-3<c['stamp']<=t_end+.2 and any(s in ' '.join(c['pair']).lower() for s in ['floor_20_x_20_ft','landing_pad']) for c in contacts)
 result.update(disarmed=True,on_pad=bool(on_pad),elapsed_s=float(t_end-t_start),penalized_s=float(t_end-t_start+result['penalty_s']))
 if t_end-t_start+result['penalty_s']>180:result['failure']=result['failure'] or '180 second limit including contact penalties'
 if next_stage==4 and on_pad and ground_on_pad and not result['failure']:
  result['checkpoints'].append({'name':'landing','time_s':float(t_end-t_start)});result['engineering_complete']=True
 else:result['failure']=result['failure'] or 'Ordered checkpoints or whole-frame platform landing incomplete'
 return result

def record(seconds,output):
 import threading,time,xml.etree.ElementTree as ET
 import rclpy
 from rclpy.node import Node
 from rclpy.parameter import Parameter
 from rclpy.qos import qos_profile_sensor_data
 from tf2_msgs.msg import TFMessage
 from px4_msgs.msg import VehicleStatus
 from gz.transport13 import Node as GzNode
 from gz.msgs10.contacts_pb2 import Contacts
 rclpy.init();node=Node('independent_course_judge',parameter_overrides=[Parameter('use_sim_time',value=True)]);gz=GzNode();lock=threading.Lock();data={'poses':[],'statuses':[],'contacts':[],'contact_topics_verified':False};seen=set()
 def tf(m):
  for t in m.transforms:
   if t.child_frame_id=='x500_vio_0':
    p=t.transform.translation;data['poses'].append([t.header.stamp.sec+t.header.stamp.nanosec*1e-9,p.x,p.y,p.z])
 def status(m):data['statuses'].append({'stamp':m.timestamp*1e-6,'armed':m.arming_state==VehicleStatus.ARMING_STATE_ARMED})
 last={}
 def contact(m):
  ts=m.header.stamp.sec+m.header.stamp.nsec*1e-9
  with lock:
   for c in m.contact:
    pair=tuple(sorted([c.collision1.name,c.collision2.name]));seen.add(pair)
    if ts-last.get(pair,-1)>.05:data['contacts'].append({'stamp':ts,'pair':list(pair)});last[pair]=ts
 topics=[]
 for link in ET.parse('/work/cartographer/v4/config/model.sdf').findall('.//link'):
  for s in link.findall('sensor'):
   if s.get('type')=='contact':
    topic=f'/world/visual_nav/model/x500_vio_0/link/{link.get("name")}/sensor/{s.get("name")}/contact';topics.append(topic);gz.subscribe(Contacts,topic,contact)
 subs=[node.create_subscription(TFMessage,'/evaluation/poses',tf,100),node.create_subscription(VehicleStatus,'/fmu/out/vehicle_status_v1',status,qos_profile_sensor_data)]
 end=time.monotonic()+seconds
 try:
  while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.03)
 except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
 data['contact_topics_verified']=bool(seen);data['topics']=topics;data['all_collision_pairs']=[list(p) for p in seen];data['result']=judge(data);Path(output).write_text(json.dumps(data,indent=2));print(json.dumps(data['result'],indent=2))
 for topic in topics:gz.unsubscribe(topic)
 time.sleep(.2)
 del gz
 node.destroy_node();rclpy.try_shutdown()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--input');ap.add_argument('--record-seconds',type=float,default=0);ap.add_argument('--output',default='/work/cartographer/v4/results/judge.json');a=ap.parse_args()
 if a.record_seconds:record(a.record_seconds,a.output)
 else:
  result=judge(json.loads(Path(a.input).read_text()));Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
