#!/usr/bin/env python3
"""Black-box, sensor-only native map experiment on domain43; never arms.

Frozen algorithm predictions are test fixtures, not an online deployment.
No simulator pose, collision mesh or competition geometry is read here.
"""
import argparse,json,os,signal,subprocess,time,copy
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Int8MultiArray,String
from scipy.spatial.transform import Rotation
from map_packet import metadata_from_knowledge
from online_map import OnlineMap


def main():
    p=argparse.ArgumentParser();p.add_argument('bag');p.add_argument('candidates',type=Path);p.add_argument('output',type=Path);p.add_argument('--inject',action='store_true');p.add_argument('--pattern',default='*.npz');p.add_argument('--stream-node',type=Path);p.add_argument('--include-floor',action='store_true');p.add_argument('--seconds',type=float,default=25);a=p.parse_args()
    if os.environ.get('ROS_DOMAIN_ID') not in ('43','44'):raise PermissionError('Sensor-only native map checks require isolated domain43 or44')
    a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];c=json.loads((root/'config/navigation.json').read_text())
    logs=[];jobs=[]
    def job(name,args):
        f=open(a.output/name,'w');logs.append(f);proc=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);jobs.append(proc);return proc
    rclpy.init();n=Node('frozen_structural_volume_probe',parameter_overrides=[Parameter('use_sim_time',value=True)])
    bodies=[];rows=[];snapshots=[];injected=[];pending=[];errors=[];structural_health=[]
    for f in sorted(a.candidates.glob(a.pattern)):
        with np.load(f) as d:
            stamp=int(d['stamp_ns']);surface=d['inferred_depth'].copy();raw=d['original_measured_depth'].copy();K=d['K'].copy();consumer=d['consumer_input_depth'].copy() if 'consumer_input_depth' in d else None
        pending.append((stamp,surface,raw,K,f.name,consumer))
    pending.sort(key=lambda v:v[0])
    pub=n.create_publisher(Image,'/nav/depth_evidence',QoSProfile(depth=2,reliability=ReliabilityPolicy.RELIABLE)) if a.inject else None
    def body(m):
        bodies.append(m)
        if len(bodies)>2:del bodies[1:-1]
    def knowledge(message):
        try:
            meta=metadata_from_knowledge(message);values=np.asarray(message.data,dtype=np.int8)
            if not bodies:return
            m=bodies[-1];point=m.pose.pose.position;q=m.pose.pose.orientation
            position=np.array([point.x,point.y,point.z]);rotation=Rotation.from_quat([q.x,q.y,q.z,q.w])
            home=bodies[0].pose.pose.position;cruise=home.z+c['takeoff_height_m']
            grid=OnlineMap(meta,values,radius=c['vehicle_radius_m']+c['xy_margin_m'],z_half=c['vertical_half_m']+c['vertical_margin_m'],allow_inferred_free=True)
            raw_grid=grid
            if (values==-3).any():
                raw_grid=copy.copy(grid);raw_grid.blocked=np.pad((~((grid.state==0)|(grid.state==-2))).astype(np.int32),((1,0),(1,0),(1,0))).cumsum(0).cumsum(1).cumsum(2)
            v=m.twist.twist.linear;velocity=rotation.apply([v.x,v.y,v.z]);speed=np.linalg.norm(velocity)
            stopping=position+velocity*(max(c['max_observation_age_s'],c.get('inferred_geometry_max_age_s',.6))+speed/(2*c['max_acceleration_mps2']))
            heading=rotation.as_euler('xyz')[2];queries=[]
            for delta in [-np.pi/2,-np.pi/4,0.,np.pi/4,np.pi/2,np.pi]:
                for distance in [.1,.2,.3,.4,.5,.6]:
                    for height_mode,height in [('current',position[2]),('cruise',cruise)]:
                        goal=position.copy();goal[:2]+=distance*np.array([np.cos(heading+delta),np.sin(heading+delta)]);goal[2]=height
                        queries.append({'height_mode':height_mode,'body_relative_bearing_rad':float(delta),'distance_m':distance,'full_segment_free':grid.segment(position,goal),'raw_only_segment_free':raw_grid.segment(position,goal)})
            row={'publish_s':meta[7],'last_raw_exposure_s':meta[8],'body_stamp_ns':int(m.header.stamp.sec)*10**9+int(m.header.stamp.nanosec),
                 'body_position':position.tolist(),'stopping_tail_free':grid.segment(position,stopping),'queries':queries,
                 'measured_free':int((values==0).sum()),'inferred_free':int((values==-3).sum()),'occupied':int((values==1).sum())}
            rows.append(row)
            if any(abs(meta[7]-v[0]*1e-9)<1.5 for v in fixtures) or (a.stream_node and (values==-3).any() and (not snapshots or meta[7]-snapshots[-1]['publish_s']>=.2)):
                name='map-%03d.npz'%len(snapshots);np.savez_compressed(a.output/name,metadata=meta,knowledge=values,body_position=position,body_quaternion=rotation.as_quat())
                snapshots.append({'file':name,'publish_s':meta[7],'last_raw_s':meta[8]})
        except (ValueError,TypeError) as e:errors.append(str(e))
    fixtures=list(pending)
    def structural(message):
        structural_health.append(json.loads(message.data))
        if len(structural_health)>240:del structural_health[:-240]
    subs=[n.create_subscription(String,'/nav/structural_health',structural,10),n.create_subscription(Odometry,'/nav/body_odom',body,100),n.create_subscription(Int8MultiArray,'/grid_map/knowledge',knowledge,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))]
    try:
        mapper=job('mapper.log',['python3',str(root/'scripts/ego_launch.py')]);until=time.monotonic()+3
        while time.monotonic()<until:rclpy.spin_once(n,timeout_sec=.01)
        if a.stream_node:
            node_dir=a.stream_node.resolve();job('structural.log',['bash',str(node_dir/'run_ros2.sh'),str(root/'config/gazebo_vio/kalibr_imucam_chain.yaml'),str(root/'config/robot.json'),str(root/'config/gazebo_vio'),'/work/cartographer/v5/candidates/grabcut-stream-adapter/build/libboundary_lbd.so','cold5_freeze1_seedvalidated_400p','true','ground_ipm','-r','/stereo/depth/image:=/nav/measured_depth','-p','health_topic:=/nav/structural_health','-p','include_floor_completion:='+str(a.include_floor).lower()])
            until=time.monotonic()+3
            while time.monotonic()<until:rclpy.spin_once(n,timeout_sec=.01)
        player=job('player.log',['ros2','bag','play',a.bag,'--clock','100','--disable-keyboard-controls','--playback-duration',str(a.seconds),'--topics','/stereo/depth/image','/nav/camera_pose','/nav/body_odom','/vio/left/image_raw','/vio/right/image_raw','/vio/imu','/carto/down_range','/carto/localization_health','/ov_msckf/joint_clone_states','--remap','/stereo/depth/image:=/nav/measured_depth'])
        deadline=time.monotonic()+a.seconds+30
        while player.poll() is None and time.monotonic()<deadline:
            if mapper.poll() is not None:raise RuntimeError('Native mapper exited')
            rclpy.spin_once(n,timeout_sec=.01);now=n.get_clock().now().nanoseconds
            while pending and now>=pending[0][0]:
                stamp,surface,raw,K,name,consumer=pending.pop(0)
                if not a.inject:continue
                if now-stamp>200000000:errors.append('Frozen exposure expired before injection: '+name);continue
                v,u=np.indices(surface.shape);norm=np.sqrt(1+((u-K[2])/K[0])**2+((v-K[3])/K[1])**2)
                # Actual consumer independently subtracts sqrt(3)*resolution
                # and excludes the terminal voxel. Send only the extra0.10m
                # retreat here; total native retreat remains0.273205m at0.1m.
                lower=consumer if consumer is not None else np.nextafter((surface-.1/norm).astype(np.float32),np.float32(-np.inf));array=np.stack([raw,lower],axis=-1)
                image=Image();image.header.stamp.sec=stamp//10**9;image.header.stamp.nanosec=stamp%10**9;image.header.frame_id='left_optical';image.height=raw.shape[0];image.width=raw.shape[1];image.encoding='32FC2';image.step=image.width*8;image.data=array.astype('<f4',copy=False).tobytes();pub.publish(image)
                injected.append({'file':name,'stamp_ns':stamp,'injection_age_s':(now-stamp)*1e-9})
        until=time.monotonic()+1
        while time.monotonic()<until:rclpy.spin_once(n,timeout_sec=.01)
    finally:
        for proc in jobs:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGINT)
        for proc in jobs:
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=5)
        for f in logs:f.close()
        n.destroy_node();rclpy.try_shutdown()
    report={'scope':'Native isolated-domain sensor replay and frozen inferred-only injection; no flight, truth or CAD inputs; not an online backend qualification',
            'stream_backend':str(a.stream_node) if a.stream_node else None,'include_floor_completion':a.include_floor,'structural_health':structural_health,'injection_requested':a.inject,'injected':injected,'native_consumer_retreat_verified_in_source':True,
            'map_config':c,'snapshots':snapshots,'rows':rows,'errors':errors,'actual_map_states':True}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'snapshots':len(snapshots),'rows':len(rows),'injected':len(injected),'errors':errors[:5],'max_inferred_free':max((r['inferred_free'] for r in rows),default=0)}))
    if not rows:raise SystemExit('No self-contained native map packets captured')


if __name__=='__main__':main()
