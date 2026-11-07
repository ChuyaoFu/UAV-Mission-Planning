#!/usr/bin/python3
"""Read-only measured-map coverage diagnosis; no world/reference/goal inputs."""
import argparse,json,time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile,DurabilityPolicy
from nav_msgs.msg import Odometry
from std_msgs.msg import Int8MultiArray,Float64MultiArray,String
from scipy.spatial.transform import Rotation
from online_map import OnlineMap

p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args()
rclpy.init();n=Node('measured_coverage_probe',parameter_overrides=[Parameter('use_sim_time',value=True)])
received={};q=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
subs=[]
for topic,key,kind,qos in [('/grid_map/knowledge','knowledge',Int8MultiArray,q),('/grid_map/metadata','metadata',Float64MultiArray,q),('/nav/body_odom','body',Odometry,5),('/nav/planar_health','planar',String,1),('/carto/localization_health','health',String,1)]:
    subs.append(n.create_subscription(kind,topic,lambda m,k=key:received.__setitem__(k,m),qos))
end=time.monotonic()+5
while time.monotonic()<end and not all(k in received for k in ['knowledge','metadata','body','health']):
    rclpy.spin_once(n,timeout_sec=.05)
end=time.monotonic()+1.1
while time.monotonic()<end and 'planar' not in received:
    rclpy.spin_once(n,timeout_sec=.05)
if not all(k in received for k in ['knowledge','metadata','body']):raise RuntimeError('Missing measured-map/odometry messages')
c=json.loads((Path(__file__).resolve().parents[1]/'config/navigation.json').read_text())
m=OnlineMap(received['metadata'].data,received['knowledge'].data,radius=c['vehicle_radius_m']+c['xy_margin_m'],z_half=c['vertical_half_m']+c['vertical_margin_m'],allow_inferred_free=c.get('geometric_free_rays',False))
o=received['body'];pos=o.pose.pose.position;quat=o.pose.pose.orientation
body=np.array([pos.x,pos.y,pos.z]);rotation=Rotation.from_quat([quat.x,quat.y,quat.z,quat.w]);yaw=rotation.as_euler('xyz')[2]

def volume(center):
    lo=m.idx(center-m.extent);hi=m.idx(center+m.extent)+1
    if np.any(lo<0) or np.any(hi>m.shape):return {'outside_map':True}
    state=m.state[tuple(slice(int(x),int(y)) for x,y in zip(lo,hi))]
    blocked=~((state==0)|(state==-2)|((state==-3)&m.allow_inferred_free))
    indices=np.argwhere(blocked)+lo
    offsets=m.xyz(indices)-center if len(indices) else np.empty((0,3))
    return {'counts':{str(int(v)):int((state==v).sum()) for v in [-3,-2,-1,0,1]},'blocked_relative_xyz_bounds':None if not len(offsets) else [offsets.min(0).tolist(),offsets.max(0).tolist()],'upper_blocked':int((offsets[:,2]>0).sum()),'lower_blocked':int((offsets[:,2]<=0).sum()),'safe':m.traversable(center)}

directions=[]
for angle in np.arange(8)*np.pi/4:
    direction=np.array([np.cos(yaw+angle),np.sin(yaw+angle),0.])
    for distance in [.1,.25,.5]:
        target=body+direction*distance
        directions.append({'relative_bearing_rad':float(angle),'displacement_m':distance,'path_safe':m.segment(body,target),'endpoint':volume(target)})
report={'scope':'Measured occupancy/accepted body pose only; no CAD, simulator pose, preset route or control output','stamp':n.get_clock().now().nanoseconds*1e-9,'raw_map_stamp':m.stamp,'body':body.tolist(),'body_yaw':float(yaw),'body_volume':volume(body),'counts':{str(v):int((m.state==v).sum()) for v in [-3,-2,-1,0,1]},'directions':directions,'health':json.loads(received['health'].data) if 'health' in received else None,'planar_health':json.loads(received['planar'].data) if 'planar' in received else None}
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(out.with_suffix('.npz'),metadata=received['metadata'].data,knowledge=m.state,body=body,body_quaternion=[quat.x,quat.y,quat.z,quat.w])
print(json.dumps({k:report[k] for k in ['body','counts','body_volume','planar_health']},indent=2))
n.destroy_node();rclpy.shutdown()
