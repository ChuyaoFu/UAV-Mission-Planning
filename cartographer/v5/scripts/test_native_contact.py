"""Synthetic declared-error/contact protocol tests. No actual flight evidence."""
import sys,unittest,math
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from task_native_contact import *
from competition_inputs import BodyHistory,NativeState
from task_landing import RangePose
from task_safety import MeasuredMapGuard
from task_uncertainty import UncertaintyPolicy
from online_map import OnlineMap


def history(t=2.,p=(0,0,.411),error=.003):
    h=BodyHistory(uncertainty_policy=UncertaintyPolicy(admission_limit_m=.5));cov=np.diag([(error/3)**2]*3+[1e-8]*3)
    h.add(int(t*1e9),p,[0,0,0,1],[0,0,0],[0,0,0],cov,'world');return h


def fixture(now=2.,p=(0,0,.411),pose_error=.003,age=1.,radius=.9):
    h=history(now,p,pose_error);src=history(now-age,(0,-1,1),pose_error);source=pose_envelope(src,int((now-age)*1e9))
    a=np.linspace(0,2*np.pi,48,endpoint=False);boundary=np.c_[np.cos(a),np.sin(a)]*radius
    xy=np.array([[x,y] for x in np.linspace(-1.2,1.2,9) for y in np.linspace(-1.2,1.2,9)]);A=np.c_[xy,np.ones(len(xy))]
    support=RetainedSupport('pad','dock','cross',now-age,'e','rig',(0.,0.,0.),(0.,0.,1.),boundary,convex_xy(xy),np.zeros(2),np.zeros(3),np.linalg.pinv(A),np.full(len(xy),.005),source,'a'*64,len(xy),.25,.01,0.,.01)
    anchor=dict(position_local_m=[0,0,.111],sensor_epoch='e',calibration_id='rig',position_three_sigma_proxy_m=.003,pose_covariance=np.eye(6).tolist())
    foot=MechanicalStartup(.111,.011753499813681907,'e','rig','b'*64,anchor,30,0.,1.2)
    shape=(50,50,35);values=np.zeros(shape,np.int8);values[:,:,5]=1
    m=OnlineMap([-2.5,-2.5,-.5,.1,*shape,0,now],values,.32,.16)
    guard=MeasuredMapGuard(m,'e',.32,.16,extra_collision_margin_m=0.,uncertainty={'body_three_sigma_proxy_m':pose_error+.001})
    rs=RangePose(now,now,now,p[2]-.025,np.array(p),np.eye(3),np.zeros(3),np.zeros(3),'e',False,True,True,True,pose_error)
    native=NativeState();native.update('status',NS(timestamp=int(now*1e6),arming_state=2),now);native.update('land',NS(timestamp=int(now*1e6),landed=False,freefall=False,at_rest=False),now)
    adapter=NativeMechanicalContact([0,0,-.025],[0,0,-1],'rig').configure_tof(.02,.005)
    return h,support,foot,guard,rs,native,adapter


class NativeContactTests(unittest.TestCase):
    def test_retained_support_survives_current_camera_occlusion_with_actual_pose_p(self):
        h,s,f,g,r,n,a=fixture(age=12.)
        cert=a.certify(g,s,f,r,2.,body_history=h,native=n)
        self.assertIsNotNone(cert,a.last_rejection);self.assertFalse(cert.grounded)
        self.assertFalse(cert.observation['current_camera_visibility_required'])
        self.assertFalse(cert.observation['global_covariance_subtracted'])
        self.assertTrue(cert.covers_motion(r.position_local_m,r.position_local_m,2.05,'e'))

    def test_dynamic_boundary_accepts_non4cm_pose_and_rejects_insufficient_pad(self):
        h,s,f,g,r,n,a=fixture(pose_error=.05,radius=1.3)
        cert=a.certify(g,s,f,r,2.,body_history=h,native=n);self.assertIsNotNone(cert,a.last_rejection)
        self.assertEqual(r.position_error_m,.05)
        h,s,f,g,r,n,a=fixture(pose_error=.15,radius=.6)
        self.assertIsNone(a.certify(g,s,f,r,2.,body_history=h,native=n));self.assertIn('boundary',a.last_rejection)

    def test_native_landed_at_rest_is_required_for_grounding_not_range_only(self):
        h,s,f,g,r,n,a=fixture(p=(0,0,.111));cert=a.certify(g,s,f,r,2.,body_history=h,native=n)
        self.assertIsNotNone(cert,a.last_rejection);self.assertFalse(cert.grounded)
        n.update('land',NS(timestamp=2000001,landed=True,freefall=False,at_rest=True),2.);r=__import__('dataclasses').replace(r,landed=True)
        self.assertTrue(a.certify(g,s,f,r,2.,body_history=h,native=n).grounded)

    def test_unknown_upper_or_side_voxel_veto_and_no_map_mutation(self):
        h,s,f,g,r,n,a=fixture();idx=g.map.idx([.2,0,.3]);g.map.state[tuple(idx)]=-1;before=g.map.state.copy()
        self.assertIsNone(a.certify(g,s,f,r,2.,body_history=h,native=n));self.assertIn('Unknown',a.last_rejection)
        np.testing.assert_array_equal(g.map.state,before)

    def test_tof_conflict_epoch_age_and_covariance_growth_veto(self):
        from dataclasses import replace
        h,s,f,g,r,n,a=fixture()
        for ss,rr in [(s,replace(r,range_m=1.5)),(replace(s,epoch='other'),r),(replace(s,stamp=-200),r)]:
            self.assertIsNone(a.certify(g,ss,f,rr,2.,body_history=h,native=n))
        h2=history(2.,(0,0,.411),.3)
        self.assertIsNone(a.certify(g,s,f,r,2.,body_history=h2,native=n))

    def test_full_actual_covariance_retained_and_no_joint_assumption(self):
        h=history();e=pose_envelope(h,2000000000);P=h.items[0].pose_covariance.copy()
        self.assertGreater(e.point_error([1,0,0]),0);np.testing.assert_array_equal(e.marginals[0],P)
        self.assertFalse(e.as_dict()['crosscov_available']);self.assertFalse(e.as_dict()['covariance_subtracted'])

    def test_vertical_frontier_requires_observed_full_body_upper_sweep(self):
        h,s,f,g,r,n,a=fixture();p=np.array([0,0,.8]);q=observed_vertical_frontier(g.map,p,extra=0.)
        self.assertIsNotNone(q);self.assertGreater(q[0][2],p[2])
        values=g.map.state.copy();values[:,:,g.map.idx(p+[0,0,.25])[2]]=-1
        m=OnlineMap([-2.5,-2.5,-.5,.1,*g.map.shape,0,2.],values,.32,.16)
        self.assertIsNone(observed_vertical_frontier(m,p,goal=p+[0,0,1.]))

if __name__=='__main__':unittest.main()
