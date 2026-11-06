#!/usr/bin/env python3
"""Privileged independent recorder only. Never imported by flight/navigation."""
import argparse,json,signal,threading,time,xml.etree.ElementTree as ET
from pathlib import Path


def record(seconds,output,ready_file,model_sdf,model_name):
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from px4_msgs.msg import VehicleStatus,VehicleLandDetected
    from gz.transport13 import Node as GzNode
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.msgs10.contacts_pb2 import Contacts
    stop=[False];signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    n=Node('independent_full_competition_recorder',parameter_overrides=[Parameter('use_sim_time',value=True)])
    gz=GzNode();lock=threading.Lock();seen=set();last_source={};last_contact={}
    data={'poses':[],'statuses':[],'landed':[],'contacts':[],'contact_topics_verified':False,
          'clock_domain':'ros_simulation','pose_reference':'model','model_name':model_name,
          'pose_stamp_source':'native pose header, else native Pose_V header',
          'native_status_stamp_source':'ROS simulation receipt; PX4 boot timestamp retained separately'}
    def now():return n.get_clock().now().nanoseconds*1e-9
    def pose(m):
        for p in m.pose:
            if p.name!=model_name:continue
            header=p.header if p.header.stamp.sec or p.header.stamp.nsec else m.header
            t=header.stamp.sec+header.stamp.nsec*1e-9
            with lock:
                if data['poses'] and t<=data['poses'][-1][0]:continue
                data['poses'].append([t,p.position.x,p.position.y,p.position.z,p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w])
    def native(kind,m):
        source=int(m.timestamp);t=now()
        if source<=last_source.get(kind,-1):return
        last_source[kind]=source
        row={'stamp':t,'native_source_us':source}
        if kind=='statuses':row['armed']=m.arming_state==VehicleStatus.ARMING_STATE_ARMED
        else:row.update(landed=bool(m.landed),freefall=bool(m.freefall))
        if data[kind] and t==data[kind][-1]['stamp'] and all(row.get(k)==data[kind][-1].get(k) for k in row if k not in ('native_source_us',)):return
        data[kind].append(row)
    def contact(m):
        t=m.header.stamp.sec+m.header.stamp.nsec*1e-9
        with lock:
            for c in m.contact:
                pair=tuple(sorted([c.collision1.name,c.collision2.name]));seen.add(pair)
                if t-last_contact.get(pair,-1e9)>.02:data['contacts'].append({'stamp':t,'pair':list(pair)});last_contact[pair]=t
    gz.subscribe(Pose_V,f'/model/{model_name}/pose',pose)
    topics=[]
    for link in ET.parse(model_sdf).findall('.//link'):
        for s in link.findall('sensor'):
            if s.get('type')=='contact':
                topic=f'/world/visual_nav/model/{model_name}/link/{link.get("name")}/sensor/{s.get("name")}/contact'
                topics.append(topic);gz.subscribe(Contacts,topic,contact)
    if not topics:raise RuntimeError('No physical robot contact sensors: refuse independent completion')
    subs=[n.create_subscription(VehicleStatus,'/fmu/out/vehicle_status_v1',lambda m:native('statuses',m),qos_profile_sensor_data),
          n.create_subscription(VehicleLandDetected,'/fmu/out/vehicle_land_detected',lambda m:native('landed',m),qos_profile_sensor_data)]
    data['contact_topics']=topics;end=time.monotonic()+seconds
    try:
        while time.monotonic()<end and not stop[0]:
            rclpy.spin_once(n,timeout_sec=.05)
            if data['poses'] and data['statuses'] and data['landed'] and seen and not Path(ready_file).exists():
                Path(ready_file).write_text(json.dumps({'pose_samples':len(data['poses']),'native_status_samples':len(data['statuses']),'landed_samples':len(data['landed']),'contact_pairs_seen':len(seen)}))
    finally:
        data['contact_topics_verified']=bool(seen and topics)
        with lock:Path(output).write_text(json.dumps(data,indent=2)+'\n')
        n.destroy_node();rclpy.try_shutdown()


def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,required=True);p.add_argument('--output',required=True);p.add_argument('--ready-file',required=True)
    p.add_argument('--model-sdf',default='/work/cartographer/v5/config/model.sdf');p.add_argument('--model-name',default='x500_vio_0');a=p.parse_args()
    record(a.seconds,a.output,a.ready_file,a.model_sdf,a.model_name)

if __name__=='__main__':main()
