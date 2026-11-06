"""Real Fusion callbacks with NO-DDS stubs; never imported by production."""
import json
import sys
from types import SimpleNamespace as NS
import unittest
import numpy as np
from competition_ros_stubs import install,message_class

msgs=install()
sys.modules['px4_msgs.msg'].VehicleOdometry=message_class('VehicleOdometry',
    constants={'POSE_FRAME_FRD':2,'VELOCITY_FRAME_BODY_FRD':3,'VELOCITY_FRAME_UNKNOWN':0})
sys.modules['sensor_msgs.msg'].PointCloud2=message_class('PointCloud2')
sys.modules['rclpy.node'].Node.create_timer=lambda self,*a,**kw:NS()
from fusion import Fusion


def node():
    n=Fusion();n.test_time=10.;n.last_update=9.95
    return n


def message(stamp):
    m=msgs.Odometry();ns=round(stamp*1e9)
    m.header.stamp.sec=ns//10**9;m.header.stamp.nanosec=ns%10**9
    m.header.frame_id='vio_home_yaw_bodyorigin_1000000000';m.child_frame_id='vio_imu'
    m.pose.covariance=(np.eye(6)*1e-4).ravel().tolist()
    m.twist.covariance=(np.eye(6)*1e-4).ravel().tolist()
    m.twist.twist.linear.x=.02
    return m


def published(n):
    return json.dumps({k:v.messages for k,v in n.publishers.items()},default=vars,sort_keys=True)


class TimestampCases(unittest.TestCase):
    def test_rejected_future_does_not_suppress_valid_stream(self):
        n=node();n.vio_cb(message(110.))
        self.assertEqual(n.last_processed,-1.)
        self.assertEqual(n.reason,'invalid_vio')
        self.assertFalse(n.ev.messages)
        n.vio_cb(message(9.95))
        self.assertEqual(n.accepted,1)
        self.assertEqual(len(n.ev.messages),1)
        self.assertEqual(n.last_processed,9.95)

    def test_invalid_pose_allows_valid_replacement_at_same_stamp(self):
        for kind in ('quaternion','covariance'):
            with self.subTest(kind=kind):
                n=node();bad=message(9.95)
                if kind=='quaternion':bad.pose.pose.orientation.w=2.
                else:bad.pose.covariance[0]=float('nan')
                n.vio_cb(bad)
                self.assertEqual(n.last_processed,-1.)
                n.vio_cb(message(9.95))
                self.assertEqual(n.accepted,1)

    def test_valid_stream_is_unchanged_by_rejected_packets_or_duplicates(self):
        clean,noisy=node(),node()
        for i in range(12):
            t=9.95+.05*i
            for n in (clean,noisy):n.test_time=t+.05;n.last_update=t
            noisy.vio_cb(message(t+100.))
            clean.vio_cb(message(t));noisy.vio_cb(message(t))
            noisy.vio_cb(message(t))
        self.assertEqual(clean.accepted,12)
        self.assertEqual(noisy.accepted,12)
        self.assertEqual(published(clean),published(noisy))
        self.assertEqual(clean.last_processed,noisy.last_processed)


if __name__=='__main__':unittest.main()
