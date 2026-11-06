"""Actual display callbacks with no DDS or flight controller."""
import ast
import copy
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS
from scipy.spatial.transform import Rotation
from frames import PoseHistory


class AlignmentTests(unittest.TestCase):
    def test_future_slam_pose_waits_for_measured_bracket(self):
        path=Path(__file__).with_name('sensor_frames.py');tree=ast.parse(path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Frames')
        methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('map_pose','drain_slam_pose')]
        def transform():return NS(header=None,transform=NS(translation=NS(),rotation=NS()))
        env=dict(copy=copy,Rotation=Rotation,TransformStamped=transform,stamp=lambda m:m.header.stamp.sec+m.header.stamp.nanosec*1e-9)
        exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),str(path),'exec'),env)
        class View:pass
        for name in ('map_pose','drain_slam_pose'):setattr(View,name,env[name])
        view=View();view.gauge=False;view.pending_slam=deque(maxlen=16);view.history=PoseHistory();sent=[];view.tf=NS(sendTransform=sent.append)
        view.history.append(1.,[2,0,0],[0,0,0,1]);view.history.append(1.05,[2,0,0],[0,0,0,1])
        m=NS(header=NS(stamp=NS(sec=1,nanosec=100000000),frame_id='map'),pose=NS(position=NS(x=5,y=0,z=0),orientation=NS(x=0,y=0,z=0,w=1)))
        view.map_pose(m);self.assertFalse(view.gauge);self.assertFalse(sent)
        view.history.append(1.1,[2,0,0],[0,0,0,1]);view.drain_slam_pose()
        self.assertTrue(view.gauge);self.assertEqual(len(sent),1);self.assertAlmostEqual(sent[0].transform.translation.x,-3.)
        self.assertEqual(sent[0].header.stamp.nanosec,100000000);self.assertEqual(m.header.frame_id,'map')
        view.map_pose(m);self.assertEqual(len(sent),1)


if __name__=='__main__':unittest.main()
