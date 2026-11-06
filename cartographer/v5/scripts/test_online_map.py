import unittest
import numpy as np
from online_map import OnlineMap
class MapTests(unittest.TestCase):
 def make(self,a):return OnlineMap([0.,0.,0.,.1,*a.shape,1.,1.],a,radius=.1,z_half=.1)
 def test_unseen_is_not_free(self):
  m=self.make(np.full((20,20,20),-1,dtype=np.int8));self.assertFalse(m.traversable([1.,1.,1.]))
 def test_body_cannot_clip_unknown(self):
  a=np.zeros((20,20,20),np.int8);a[10:,:,:]=-1;m=self.make(a);self.assertTrue(m.traversable([.75,1.,1.]));self.assertFalse(m.traversable([.95,1.,1.]))
 def test_obstacle_blocks_whole_segment(self):
  a=np.zeros((20,20,20),np.int8);a[10,:,:]=1;m=self.make(a);self.assertFalse(m.segment([.4,1.,1.],[1.6,1.,1.]))
 def test_no_diagonal_unknown_shortcut(self):
  a=np.zeros((20,20,20),np.int8);a[10:,:,5:]=-1;m=self.make(a);self.assertFalse(m.segment([.5,.5,.5],[1.5,1.5,1.5]))
 def test_frontier_comes_from_connected_measured_map(self):
  a=np.full((40,40,20),-1,np.int8);a[5:35,5:35,1:18]=0;m=self.make(a);v=m.frontier(np.array([2.,2.,.7]),np.array([2.,2.,0.]),[],0.);self.assertIsNotNone(v);self.assertTrue(m.traversable(v[0]));self.assertTrue(m.segment(np.array([2.,2.,.7]),v[0]));self.assertLessEqual(np.linalg.norm(v[0]-np.array([2.,2.,.7])),.45)
 def test_exploration_goal_preserves_forward_view(self):
  a=np.full((40,40,20),-1,np.int8);a[5:35,5:35,1:18]=0;m=self.make(a);position=np.array([2.,2.,.7]);v=m.frontier(position,np.array([2.,2.,0.]),[],0.);delta=v[0]-position
  if np.linalg.norm(delta[:2])>.08:self.assertLessEqual(abs(np.arctan2(delta[1],delta[0])),.5)
 def test_forward_launch_candidate_survives_arbitrary_local_grid_offset(self):
  home=np.array([0.,.12,-.055]);origin=np.array([-4.,-4.,-1.]);shape=(80,80,30);index=np.indices(shape).reshape(3,-1).T;xyz=origin+(index+.5)*.1;a=np.full(shape,-1,np.int8);inside=(np.abs(xyz[:,0]-home[0])<.5)&(np.abs(xyz[:,1]-home[1])<.5)&(xyz[:,2]>home[2]+.12)&(xyz[:,2]<home[2]+1.05);a.ravel()[inside]=-2;m=OnlineMap([*origin,.1,*shape,1.,1.],a,radius=.32,z_half=.16);position=home+np.array([0.,0.,.55]);candidate=m.frontier(position,home,[position],-np.pi/2)
  self.assertIsNotNone(candidate);delta=candidate[0]-position;self.assertLess(delta[1],-.12);self.assertLess(abs(delta[2]),.12);self.assertTrue(m.segment(position,candidate[0]))
if __name__=='__main__':unittest.main()
