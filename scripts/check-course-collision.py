#!/usr/bin/python3
"""Physics integration checks on the imported mesh, with disposable probes."""
import json
import math
import time
from pathlib import Path
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.entity_pb2 import Entity
from gz.msgs10.entity_wrench_pb2 import EntityWrench
from gz.msgs10.boolean_pb2 import Boolean

node=Node();positions={}
def pose_cb(m):
    for p in m.pose:
        if p.name.startswith('collision_probe_'):
            positions[p.name]=(p.position.x,p.position.y,p.position.z)
node.subscribe(Pose_V,'/world/visual_nav/dynamic_pose/info',pose_cb)
pub=node.advertise('/world/visual_nav/wrench/persistent',EntityWrench)
def request(service,msg,typ):
    ok,resp=node.request(service,msg,typ,Boolean,4000)
    if not ok or not resp.data:raise RuntimeError(f'Service failed: {service}')
def wait_position(name):
    for _ in range(50):
        if name in positions:return positions[name]
        time.sleep(.05)
    raise RuntimeError('No probe pose: '+name)
def run(name,start,direction,gravity=False):
    full='collision_probe_'+name
    x,y,z=start
    factory=EntityFactory(name=full,sdf=f'''<sdf version="1.9"><model name="{full}"><pose>{x} {y} {z} 0 0 0</pose><link name="body"><gravity>{str(gravity).lower()}</gravity><inertial><mass>.1</mass><inertia><ixx>.000036</ixx><iyy>.000036</iyy><izz>.000036</izz></inertia></inertial><collision name="ball"><geometry><sphere><radius>.03</radius></sphere></geometry></collision></link></model></sdf>''')
    request('/world/visual_nav/create',factory,EntityFactory)
    try:
        first=wait_position(full)
        if not gravity:
            time.sleep(.3)
            w=EntityWrench();w.entity.name=full;w.entity.type=Entity.MODEL
            w.wrench.force.x=direction[0]*.1;w.wrench.force.y=direction[1]*.1;w.wrench.force.z=direction[2]*.1
            pub.publish(w)
        time.sleep(1.3)
        last=positions[full]
        delta=sum((b-a)*d for a,b,d in zip(first,last,direction))
        return {'start':list(first),'end':list(last),'displacement_along_probe_m':delta}
    finally:
        request('/world/visual_nav/remove',Entity(name=full,type=Entity.MODEL),Entity)

normal=(math.cos(math.radians(20)),-math.sin(math.radians(20)),0)
cx,cy=-.470938,-2.271895
start=(cx-normal[0]*.25,cy-normal[1]*.25,.762)
time.sleep(1)
r={}
r['window_open']=run('window_open',start,normal)
r['window_solid']=run('window_solid',(start[0],start[1],1.45),normal)
r['tunnel_open']=run('tunnel_open',(2.3114,.1,.9398),(0,-1,0))
r['floor']=run('floor',(-2.5,2,.6),(0,0,-1),True)
r['checks']={
    'window_passed':r['window_open']['displacement_along_probe_m']>.4,
    'solid_wall_blocked':r['window_solid']['displacement_along_probe_m']<.25,
    'tunnel_exit_passed':r['tunnel_open']['end'][1]<-.35,
    'floor_supported_probe':.015<r['floor']['end'][2]<.06,
}
Path('/work/results/course-collision.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
if not all(r['checks'].values()):raise SystemExit('Course collision validation failed')
