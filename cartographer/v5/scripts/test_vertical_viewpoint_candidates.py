import unittest
import numpy as np
from online_map import OnlineMap
from task_native_contact import observed_vertical_frontier

class VerticalCandidatesTests(unittest.TestCase):
    def world(self,*,ceiling=None,barrier=False,inferred=False,enabled=False):
        shape=(20,20,20);origin=np.array([-1.,-1.,-.5]);res=.1
        xyz=origin+(np.indices(shape).reshape(3,-1).T+.5)*res
        keep=(xyz[:,0]>-.3)&(xyz[:,0]<.4)&(xyz[:,1]>-.3)&(xyz[:,1]<.4)&(xyz[:,2]>.2)&(xyz[:,2]<.9)
        state=np.full(shape,-1,np.int8);state.ravel()[keep]=-3 if inferred else 0
        if ceiling is not None:state[:,:,np.arange(shape[2])*.1+origin[2]>=ceiling]=-1
        if barrier:state[:,:,10]=1
        return OnlineMap([*origin,res,*shape,0,2.],state,.32,.16,allow_inferred_free=enabled),np.array([.037,.043,.596])
    def test_offgrid_vertical_candidates_preserve_measured_xy_and_resolution(self):
        m,p=self.world();q,indices=m.vertical_viewpoint_candidates(p)
        self.assertGreater(len(q),0);np.testing.assert_array_equal(q[:,:2],np.repeat(p[None,:2],len(q),axis=0))
        steps=(q[:,2]-p[2])/m.res;np.testing.assert_allclose(steps,np.round(steps))
        self.assertTrue((m.components[tuple(indices.T)]==m.component(p)).all())
    def test_existing_3d_helper_recovers_ascending_viewpoint_missing_from_grid_centers(self):
        m,p=self.world();grid=m.xyz(np.argwhere((m.components==m.component(p))&m.safe))
        self.assertFalse(((grid[:,2]-p[2])>=m.res*.8).any())
        result=observed_vertical_frontier(m,p);self.assertIsNotNone(result)
        q,reason=result;np.testing.assert_array_equal(q[:2],p[:2]);self.assertGreater(q[2],p[2]);self.assertAlmostEqual(q[2]-p[2],m.res)
        self.assertTrue(m.segment(p,q));self.assertTrue(reason['observed_free_altitude_maneuver'])
    def test_altitude_frontier_uses_same_score_and_exact_full_body_path(self):
        m,p=self.world();before=m.state.copy()
        result=m.frontier(p,np.zeros(3),[p],0.,max_heading_change=np.pi,cruise_height=p[2],allow_altitude_change=True)
        self.assertIsNotNone(result);q,reason=result;np.testing.assert_array_equal(q[:2],p[:2]);self.assertTrue(m.segment(p,q));self.assertTrue(reason['fine_observed_step']);np.testing.assert_array_equal(m.state,before)
    def test_no_altitude_mode_does_not_generate_new_axis_candidates(self):
        m,p=self.world()
        from unittest.mock import patch
        with patch.object(m,'vertical_viewpoint_candidates',side_effect=AssertionError('disabled branch')):
            m.frontier(p,np.zeros(3),[p],0.,cruise_height=p[2],allow_altitude_change=False)
    def test_unknown_ceiling_or_full_body_shell_cannot_be_opened(self):
        m,p=self.world(ceiling=.8)
        self.assertFalse(m.segment(p,p+[0,0,.1]));self.assertIsNone(observed_vertical_frontier(m,p))
        m,p=self.world();state=m.state.copy();state[7,10,11]=-1
        blocked=OnlineMap([*m.origin,m.res,*m.shape,0,m.stamp],state,.32,.16)
        result=observed_vertical_frontier(blocked,p)
        if result is not None:self.assertTrue(blocked.segment(p,result[0]))
        self.assertFalse(blocked.segment(p,p+[0,0,.1]))
    def test_new_candidates_do_not_bypass_novelty_or_extra_clearance(self):
        m,p=self.world();visited=np.vstack([m.xyz(np.argwhere(m.safe)),m.vertical_viewpoint_candidates(p)[0]])
        self.assertIsNone(m.frontier(p,np.zeros(3),visited,0.,cruise_height=p[2],allow_altitude_change=True))
        self.assertIsNone(observed_vertical_frontier(m,p,extra=.1))
    def test_inferred_space_requires_existing_opt_in(self):
        m,p=self.world(inferred=True);self.assertIsNone(observed_vertical_frontier(m,p))
        m,p=self.world(inferred=True,enabled=True);result=observed_vertical_frontier(m,p);self.assertIsNotNone(result);self.assertTrue(m.segment(p,result[0]))
    def test_goal_directed_descent_and_blocked_component_stay_constrained(self):
        m,p=self.world();result=observed_vertical_frontier(m,p,goal=p-[0,0,.3]);self.assertIsNotNone(result)
        q,_=result;self.assertLess(q[2],p[2]);self.assertTrue(m.segment(p,q))
        m,p=self.world(barrier=True);self.assertIsNone(observed_vertical_frontier(m,p))
if __name__=='__main__':unittest.main()
