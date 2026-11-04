import unittest,numpy as np
from path_tracking import Follower,smooth
from cad_planner import Grid
class Tracking(unittest.TestCase):
 def test_parallel_leg_no_jump(self):
  f=Follower([[0,0,0],[2,0,0],[2,.15,0],[0,.15,0]]);f.project([.1,.15,0]);self.assertLess(f.progress,.6)
 def test_curvature_slows(self):
  straight=Follower([[0,0,0],[2,0,0]]);turn=Follower([[0,0,0],[.25,0,0],[.25,1,0]])
  self.assertLess(np.linalg.norm(turn.target(np.zeros(3))[1]),np.linalg.norm(straight.target(np.zeros(3))[1]))
 def test_smoothing_does_not_cut_obstacle(self):
  occupied=np.zeros((100,100,3),bool);occupied[40:60,40:60,:]=True;g=Grid(occupied,[0,0,0],.02);path=np.array([[.2,1,.02],[.7,1.4,.02],[1.3,1.4,.02],[1.8,1,.02]]);p=smooth(path,g)
  self.assertTrue(all(g.segment(a,b) for a,b in zip(p,p[1:])))
if __name__=='__main__':unittest.main()
