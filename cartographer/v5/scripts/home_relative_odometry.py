#!/usr/bin/env python3
"""Optional measured home-relative adapter; no commands, truth or resets.

Requires the optional retained-home-clone OpenVINS build. It preserves the
published joint-posterior pose, moves its reference point once, and joins an
actual recent native IMU twist under an explicit bounded acceleration model.
Default launch configuration does not select this output.
"""
import copy,json,re
from collections import OrderedDict,deque
import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray,String
from robot_geometry import IMU_IN_BODY,BODY_FROM_IMU
from home_relative_geometry import body_pose_to_native_imu,align_native_twist,covariance,validate_reference,ReferenceChanged


def ns(message):return int(message.header.stamp.sec)*1000000000+int(message.header.stamp.nanosec)


def rotation(pose):
    q=pose.orientation;a=np.array([q.x,q.y,q.z,q.w])
    if not np.isfinite(a).all() or abs(np.linalg.norm(a)-1)>1e-6:raise ValueError('Invalid actual native rotation')
    return Rotation.from_quat(a)


class Adapter(Node):
    def __init__(self):
        import os
        super().__init__('home_relative_odometry_adapter',parameter_overrides=[Parameter('use_sim_time',value=os.environ.get('USE_SIM_TIME','1')=='1')])
        self.raw_poses=OrderedDict();self.pending=OrderedDict();self.twists=deque(maxlen=256)
        self.reference=None;self.fixed_reference=None;self.epoch=None;self.fault=None;self.stream_stamps={};self.producers_verified=False;self.producer_counts={};self.fast_time_verified=os.environ.get('NATIVE_FAST_IMU_EPOCH_VERIFIED')=='1';self.last=-1;self.accepted=0;self.rejected=0;self.reason='waiting for calibrated native reference';self.provenance={}
        self.pub=self.create_publisher(Odometry,'/nav/home_relative_odomimu',20)
        self.health=self.create_publisher(String,'/nav/home_relative_adapter_health',5)
        self.subs=[self.create_subscription(PoseWithCovarianceStamped,'/ov_msckf/home_relative_body_pose',self.home,20),
                   self.create_subscription(PoseWithCovarianceStamped,'/ov_msckf/poseimu',self.native_pose,20),
                   self.create_subscription(Odometry,'/ov_msckf/odomimu',self.twist,100),
                   self.create_subscription(Float64MultiArray,'/ov_msckf/home_relative_body_reference',self.ref,10)]
        self.create_timer(.5,self.report)

    def now_ns(self):return self.get_clock().now().nanoseconds

    def ref(self,message):
        try:
            a,epoch=validate_reference(message.data,IMU_IN_BODY,BODY_FROM_IMU,self.fixed_reference,self.fast_time_verified)
            if not self.check_stream('reference',int(round(a[4]*1e9))):return
            self.reference=a;self.fixed_reference=a.copy();self.epoch=epoch;self.drain()
        except ReferenceChanged as error:self.fail_session(str(error))
        except ValueError as error:self.reference=None;self.reason=str(error);self.rejected+=1

    def fail_session(self,reason):
        self.fault=reason;self.reference=None;self.raw_poses.clear();self.pending.clear();self.twists.clear()

    def check_stream(self,name,stamp):
        if stamp<self.stream_stamps.get(name,-1):
            self.fail_session('Native '+name+' clock regressed; explicit session restart required');return False
        self.stream_stamps[name]=stamp;return True

    def remember(self,cache,message,limit):
        if not self.check_stream('home' if cache is self.pending else 'poseimu',ns(message)):return
        cache[ns(message)]=message
        while len(cache)>limit:cache.popitem(last=False)

    def home(self,message):self.remember(self.pending,message,8);self.drain()
    def native_pose(self,message):self.remember(self.raw_poses,message,64);self.drain()

    def twist(self,message):
        if not self.check_stream('twist',ns(message)):return
        if not self.twists or ns(message)>ns(self.twists[-1]):self.twists.append(message)
        self.drain()

    def drain(self):
        if self.fault or self.reference is None or not self.producers_verified:return
        for stamp,message in list(self.pending.items()):
            if stamp<=self.last:
                del self.pending[stamp];continue
            age=(self.now_ns()-stamp)*1e-9
            if age>.20:
                del self.pending[stamp];self.reason='Current posterior pose expired';self.rejected+=1;continue
            if age<-.005 or stamp not in self.raw_poses or not self.twists or ns(self.twists[-1])<stamp:continue
            causal=[m for m in self.twists if ns(m)<=stamp]
            if not causal:continue
            source=causal[-1];twist_age=(stamp-ns(source))*1e-9
            try:
                match=re.fullmatch(r'vio_home_yaw_bodyorigin_([0-9]+)',message.header.frame_id)
                if match is None or int(match[1])!=self.epoch:raise ValueError('Pose does not match verified actual home epoch')
                raw=self.raw_poses[stamp]
                if raw.header.frame_id!='global' or source.header.frame_id!='global' or source.child_frame_id!='imu':raise ValueError('Unexpected native pose/twist frame contract')
                rp=raw.pose.pose;body=message.pose.pose;p=body.position;q=body.orientation
                position,quaternion,pc=body_pose_to_native_imu([p.x,p.y,p.z],[q.x,q.y,q.z,q.w],message.pose.covariance,IMU_IN_BODY,BODY_FROM_IMU)
                v=source.twist.twist.linear;w=source.twist.twist.angular
                linear,angular,vc=align_native_twist([v.x,v.y,v.z],[w.x,w.y,w.z],source.twist.covariance,
                    rotation(source.pose.pose),rotation(rp),twist_age,source.pose.covariance,raw.pose.covariance)
                output=Odometry();output.header=copy.deepcopy(message.header);output.child_frame_id='vio_imu'
                output.pose.pose.position.x=float(position[0]);output.pose.pose.position.y=float(position[1]);output.pose.pose.position.z=float(position[2])
                for name,value in zip(('x','y','z','w'),quaternion):setattr(output.pose.pose.orientation,name,float(value))
                output.pose.covariance=pc.ravel().tolist();output.twist.covariance=vc.ravel().tolist()
                for name,value in zip(('x','y','z'),linear):setattr(output.twist.twist.linear,name,float(value))
                for name,value in zip(('x','y','z'),angular):setattr(output.twist.twist.angular,name,float(value))
                self.pub.publish(output);self.last=stamp;self.accepted+=1;self.reason='actual joint home-relative pose and bounded-age native twist'
                self.provenance={'pose_stamp_ns':stamp,'native_twist_stamp_ns':ns(source),'twist_age_s':twist_age,
                    'twist_motion_model':'constant world velocity, acceleration bound 2m/s2 and angular acceleration bound 4rad/s2; actual attitude uncertainty added',
                    'pose_covariance_chart':'position HOME, local IMU angle','home_epoch_ns':self.epoch,
                    'pose_covariance_clipped':False,'position_prior_changed':False,'calibration_uncertainty_in_posterior':False,'native_twist_covariance_complete':False,'twist_enclosure_scope':'first-order enclosure relative to supplied native covariance approximation','native_fast_imu_epoch_verified':self.fast_time_verified}
            except ValueError as error:self.reason=str(error);self.rejected+=1
            del self.pending[stamp]

    def report(self):
        self.producer_counts={topic:self.count_publishers(topic) for topic in ('/ov_msckf/poseimu','/ov_msckf/odomimu','/ov_msckf/home_relative_body_pose','/ov_msckf/home_relative_body_reference')}
        if any(v>1 for v in self.producer_counts.values()):self.fail_session('Multiple native producers; session identity ambiguous')
        self.producers_verified=all(v==1 for v in self.producer_counts.values())
        self.drain()
        self.health.publish(String(data=json.dumps({'stamp':self.now_ns()*1e-9,'valid':not self.fault and self.reference is not None and self.producers_verified and self.last>=0 and 0<=self.now_ns()-self.last<200000000,
            'reason':self.fault or self.reason,'accepted':self.accepted,'rejected':self.rejected,'pending':len(self.pending),'producer_counts':self.producer_counts,**self.provenance})))


if __name__=='__main__':
    rclpy.init();node=Adapter()
    try:rclpy.spin(node)
    finally:node.destroy_node();rclpy.try_shutdown()
