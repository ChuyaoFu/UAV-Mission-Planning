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
  home=np.array([0.,.12,-.055]);origin=np.array([-4.,-4.,-1.]);shape=(80,80,30);index=np.indices(shape).reshape(3,-1).T;xyz=origin+(index+.5)*.1;a=np.full(shape,-1,np.int8);inside=(np.abs(xyz[:,0]-home[0])<.5)&(np.abs(xyz[:,1]-home[1])<.5)&(xyz[:,2]>home[2]+.12)&(xyz[:,2]<home[2]+1.05);a.ravel()[inside]=-2;observed=(np.abs(xyz[:,0]-home[0])<.7)&(xyz[:,1]>home[1]-1.2)&(xyz[:,1]<home[1]+.3)&(xyz[:,2]>home[2]+.12)&(xyz[:,2]<home[2]+1.05);a.ravel()[observed]=0;m=OnlineMap([*origin,.1,*shape,1.,1.],a,radius=.32,z_half=.16);position=home+np.array([0.,0.,.55]);candidate=m.frontier(position,home,[position],-np.pi/2)
  self.assertIsNotNone(candidate);delta=candidate[0]-position;self.assertLess(delta[1],-.12);self.assertLess(abs(delta[2]),.12);self.assertTrue(m.segment(position,candidate[0],extra_margin=.05))
 def test_altitude_viewpoint_requires_connected_measured_free_volume(self):
  a=np.full((40,40,20),-1,np.int8);a[10:30,10:30,2:12]=0;m=self.make(a);position=np.array([2.,2.,.55]);home=np.array([2.,2.,0.]);visited=[np.array([x,y,.55]) for x in np.arange(1.,3.,.1) for y in np.arange(1.,3.,.1)]
  self.assertIsNone(m.frontier(position,home,visited,0.,max_heading_change=np.pi))
  candidate=m.frontier(position,home,visited,0.,max_heading_change=np.pi,allow_altitude_change=True);self.assertIsNotNone(candidate);self.assertTrue(m.segment(position,candidate[0],extra_margin=.05));self.assertTrue(candidate[1]['observed_free_altitude_maneuver'])
 def test_guideline_cannot_open_unknown_or_cross_wall(self):
  a=np.full((40,40,20),-1,np.int8);a[5:35,5:35,1:18]=0;a[23,:,:]=1;m=self.make(a);position=np.array([2.,2.,.7]);hint=[{'endpoints':[[2.,2.,0.],[3.5,2.,0.]]}];before=m.state.copy();v=m.frontier(position,np.array([2.,2.,0.]),[],0.,max_heading_change=np.pi,guideline_segments=hint)
  self.assertIsNotNone(v);self.assertTrue(m.segment(position,v[0]));self.assertLess(v[0][0],2.3);np.testing.assert_array_equal(m.state,before)
  unknown=self.make(np.full((40,40,20),-1,np.int8));self.assertIsNone(unknown.frontier(position,np.array([2.,2.,0.]),[],0.,guideline_segments=hint))
 def test_return_routes_around_observed_concavity(self):
  a=np.full((40,40,20),-1,np.int8);a[3:37,3:37,2:17]=0;a[18:22,3:27,:]=-1;m=self.make(a);p=np.array([1.45,2.35,.75]);home=np.array([2.55,2.35,.75]);self.assertFalse(m.segment(p,home))
  for _ in range(30):
   result=m.route_step(p,home);self.assertIsNotNone(result);goal,reason=result;self.assertTrue(m.segment(p,goal));self.assertLessEqual(np.linalg.norm(goal-p),.35001);p=goal
   if np.linalg.norm(p-home)<.03:break
  self.assertLess(np.linalg.norm(p-home),.03)
 def test_return_does_not_cross_unobserved_partition(self):
  a=np.zeros((40,40,20),np.int8);a[18:22,:,:]=-1;m=self.make(a);self.assertIsNone(m.route_step(np.array([1.,2.,.75]),np.array([3.,2.,.75])))
 def test_return_rechecks_exact_body_clearance(self):
  a=np.zeros((40,40,20),np.int8);a[10:,:,:]=-1;m=self.make(a);self.assertIsNone(m.route_step(np.array([.95,1.,.7]),np.array([.5,1.,.7])))
 def incremental_fixture(self):
  origin=np.array([-2.,-2.,-1.]);shape=(40,40,25);xyz=origin+(np.indices(shape).reshape(3,-1).T+.5)*.1;a=np.full(shape,-1,np.int8);free=(xyz[:,0]>-.5)&(xyz[:,0]<.6)&(np.abs(xyz[:,1])<.5)&(xyz[:,2]>.12)&(xyz[:,2]<1.05);a.ravel()[free]=0
  m=OnlineMap([*origin,.1,*shape,1.,1.],a);position=np.array([.15,.05,.55]);visited=[np.array([x,y,.55]) for x in np.arange(-.15,.151,.1) for y in np.arange(-.15,.151,.1)];return m,position,visited
 def test_one_newly_observed_voxel_can_extend_exploration(self):
  m,position,visited=self.incremental_fixture();result=m.frontier(position,np.zeros(3),visited,0.,max_heading_change=np.pi);self.assertIsNotNone(result);goal,reason=result;self.assertTrue(reason['fine_observed_step']);self.assertGreater(goal[0],position[0]+.08);self.assertLessEqual(np.linalg.norm(goal-position),.16);self.assertTrue(m.segment(position,goal));self.assertGreaterEqual(min(np.linalg.norm(goal[:2]-p[:2]) for p in visited),.08)
 def test_fine_step_cannot_repeat_already_visited_grid_increment(self):
  m,position,visited=self.incremental_fixture();visited.extend([np.array([.25,y,.55]) for y in np.arange(-.15,.151,.1)]);self.assertIsNone(m.frontier(position,np.zeros(3),visited,0.,max_heading_change=np.pi))
 def test_fine_step_does_not_open_unknown_body_clearance(self):
  m,position,visited=self.incremental_fixture();a=m.state.copy();a[25,:,:]=-1;blocked=OnlineMap([*m.origin,m.res,*m.shape,1.,1.],a);self.assertIsNone(blocked.frontier(position,np.zeros(3),visited,0.,max_heading_change=np.pi))
 def test_inferred_free_requires_explicit_opt_in_and_keeps_obstacles(self):
  a=np.full((20,20,20),-3,np.int8);metadata=[0.,0.,0.,.1,*a.shape,1.,1.];default=OnlineMap(metadata,a,radius=.1,z_half=.1);self.assertFalse(default.traversable([1.,1.,1.]));enabled=OnlineMap(metadata,a,radius=.1,z_half=.1,allow_inferred_free=True);self.assertTrue(enabled.traversable([1.,1.,1.]));a[10,:,:]=1;blocked=OnlineMap(metadata,a,radius=.1,z_half=.1,allow_inferred_free=True);self.assertFalse(blocked.segment([.5,1.,1.],[1.5,1.,1.]))
if __name__=='__main__':unittest.main()
