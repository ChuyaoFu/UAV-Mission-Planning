#!/usr/bin/python3
"""SITL adapter: OpenVINS local FLU/body FLU -> PX4 arbitrary FRD/body FRD.

OpenVINS odomimu twist is expressed in the IMU/body frame (verified upstream).
No Gazebo pose, GPS, or evaluation topic is subscribed here.
"""
import argparse
import math
import time
import numpy as np
from scipy.spatial.transform import Rotation

D=np.diag([1.,-1.,-1.])
def convert_rotation(q_xyzw):
    return Rotation.from_matrix(D@Rotation.from_quat(q_xyzw).as_matrix()@D).as_quat()
def valid_covariance(c):
    return np.isfinite(c).all() and np.allclose(c,c.T,rtol=1e-5,atol=1e-7) and np.linalg.eigvalsh(c).min()>=-1e-8

ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
if args.self_test:
    assert np.allclose(D@np.array([1,2,3]),[1,-2,-3])
    for axis,sign in [('x',1),('y',-1),('z',-1)]:
        q=Rotation.from_euler(axis,.3).as_quat();expected=Rotation.from_euler(axis,sign*.3).as_matrix()
        assert np.allclose(Rotation.from_quat(convert_rotation(q)).as_matrix(),expected)
    c=np.eye(6)*.001;assert valid_covariance(c)
    bad=c.copy();bad[0,1]=bad[1,0]=.01;assert not valid_covariance(bad)
    bad=c.copy();bad[0,0]=float('nan');assert not valid_covariance(bad)
    print('Frame conversion and invalid/indefinite covariance rejection tests passed.')
    raise SystemExit(0)

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from px4_msgs.msg import VehicleOdometry
rclpy.init()
node=Node('vio_to_px4',parameter_overrides=[Parameter('use_sim_time',value=True)])
pub=node.create_publisher(VehicleOdometry,'/fmu/in/vehicle_visual_odometry',10)
state={'camera_stamp':-1.,'last_sent':-1.,'last_warn':0.,'sent':0,'rejected':0}
def timestamp(h):return h.stamp.sec+h.stamp.nanosec*1e-9
def pose_cb(m):state['camera_stamp']=timestamp(m.header)
def callback(m):
    ts=timestamp(m.header);now=node.get_clock().now().nanoseconds*1e-9
    if ts-state['last_sent']<.019:return
    pc=np.asarray(m.pose.covariance).reshape(6,6);vc=np.asarray(m.twist.covariance).reshape(6,6)
    valid=(m.header.frame_id=='global' and m.child_frame_id=='imu' and -.05<now-ts<.2 and -.05<ts-state['camera_stamp']<.2 and valid_covariance(pc) and valid_covariance(vc) and np.max(np.diag(pc)[:3])<.09 and np.max(np.diag(pc)[3:])<.04)
    p=m.pose.pose.position;q=m.pose.pose.orientation;vel=m.twist.twist.linear;omega=m.twist.twist.angular
    a=np.array([p.x,p.y,p.z]);quat=np.array([q.x,q.y,q.z,q.w]);v=np.array([vel.x,vel.y,vel.z]);w=np.array([omega.x,omega.y,omega.z])
    valid=valid and np.isfinite(np.r_[a,quat,v,w]).all() and abs(np.linalg.norm(quat)-1)<.01
    if not valid:
        state['rejected']+=1
        if time.monotonic()-state['last_warn']>5:
            node.get_logger().warning(f'Rejecting EV: odometry age {now-ts:.3f}s, visual-pose age {ts-state["camera_stamp"]:.3f}s, max position variance {max(np.diag(pc)[:3]):.4f}')
            state['last_warn']=time.monotonic()
        return
    qr=convert_rotation(quat)
    out=VehicleOdometry(timestamp=int(now*1e6),timestamp_sample=int(ts*1e6),pose_frame=VehicleOdometry.POSE_FRAME_FRD,position=(D@a).astype(np.float32).tolist(),q=[float(qr[3]),float(qr[0]),float(qr[1]),float(qr[2])],velocity_frame=VehicleOdometry.VELOCITY_FRAME_BODY_FRD,velocity=(D@v).astype(np.float32).tolist(),angular_velocity=(D@w).astype(np.float32).tolist(),position_variance=np.maximum(np.diag(pc)[:3],.0004).astype(np.float32).tolist(),orientation_variance=np.maximum(np.diag(pc)[3:],.0004).astype(np.float32).tolist(),velocity_variance=np.maximum(np.diag(vc)[:3],.0025).astype(np.float32).tolist(),quality=1)
    # quality=1 signals a valid gated sample; it is not a probability or score.
    pub.publish(out);state['last_sent']=ts;state['sent']+=1

subs=[node.create_subscription(PoseWithCovarianceStamped,'/ov_msckf/poseimu',pose_cb,10),node.create_subscription(Odometry,'/ov_msckf/odomimu',callback,20)]
node.get_logger().info('EV adapter ready: arbitrary FRD pose; BODY_FRD velocity; IMU lever arm configured in PX4.')
try:rclpy.spin(node)
finally:print(state);node.destroy_node();rclpy.shutdown()
