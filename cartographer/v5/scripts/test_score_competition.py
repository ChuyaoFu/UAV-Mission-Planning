"""Synthetic truth-domain scorer tests. No real course/flight evidence."""
import copy,unittest
import numpy as np
from score_competition import score,REQUIRED


def fixture_and_data():
    def gate(name,x,n=(1,0,0)):
        return {'id':name,'center_m':[x,0,.6],'normal_forward':list(n),'axis_u':[0,1,0],'axis_v':[0,0,1],'half_width_m':1.,'half_height_m':.5}
    gates={'tunnel_entry':gate('entry',1),'tunnel_exit':gate('exit',2),'window':gate('window',3),'wall':gate('wall',2,(-1,0,0))}
    gates['wall']['minimum_body_bottom_z_m']=.35
    f={'schema_version':1,'qualification':'private-independent-fixture','robot':{'radius_m':.24,'half_height_m':.12,'resting_body_height_m':.1},'gates':gates,
       'dock':{'center_surface_m':[4,0,0],'axis_u':[1,0,0],'axis_v':[0,1,0],'half_width_m':.6,'half_height_m':.6,'contact_tokens':['dock_surface']},'home_contact_tokens':['home_floor']}
    knots=np.array([[0,0,0,.1],[1,0,0,.1],[2,0,0,.6],[3,1.4,0,.6],[4,2.4,0,.6],[5,3.4,0,.6],[6,4,0,.6],[7,4,0,.1],[8.2,4,0,.1],[9,4,0,.8],[11,0,0,.8],[12,0,0,.1],[12.5,0,0,.1]])
    ts=np.round(np.arange(0,12.51,.05),6)
    poses=[[float(t),*[float(np.interp(t,knots[:,0],knots[:,k])) for k in (1,2,3)],0,0,0,1] for t in ts]
    statuses=[{'stamp':float(t),'armed':bool(1<=t<7.2 or 8<=t<12.2)} for t in np.round(np.arange(0,12.51,.1),6)]
    lands=[{'stamp':float(t),'landed':bool(t<=1 or 7<=t<=8.2 or t>=12),'freefall':False} for t in ts]
    contacts=[{'stamp':t,'pair':['drone_foot',token]} for t,token in [(.9,'home_floor'),(7.1,'dock_surface'),(7.2,'dock_surface'),(8.,'dock_surface'),(12.1,'home_floor'),(12.2,'home_floor')]]
    d={'poses':poses,'statuses':statuses,'landed':lands,'contacts':contacts,'contact_topics_verified':True,'clock_domain':'ros_simulation','pose_reference':'body','synthetic':True}
    c={'competition_tasks_complete':True,'completed':{k:{'verified':True} for k in REQUIRED},'observe_only':False}
    return f,d,c


class ScoreTests(unittest.TestCase):
    def test_model_pose_requires_explicit_actual_body_calibration(self):
        f,d,c=fixture_and_data();d['pose_reference']='model'
        self.assertFalse(score(d,f,c)['competition_tasks_complete'])
        offset=np.array([.1,-.2,.3]);f['robot']['model_from_body_position_quaternion']=[*offset,0,0,0,1]
        for p in d['poses']:p[1:4]=(np.asarray(p[1:4])-offset).tolist()
        self.assertTrue(score(d,f,c)['competition_tasks_complete'])

    def test_synthetic_full_order_native_cycles_contacts_and_overwall_pass(self):
        f,d,c=fixture_and_data();r=score(d,f,c)
        self.assertTrue(r['competition_tasks_complete'],r)
        self.assertEqual(set(r['events']),set(REQUIRED))

    def test_missing_distinct_tunnel_exit_refuses(self):
        f,d,c=fixture_and_data();f['gates']['tunnel_exit']=copy.deepcopy(f['gates']['tunnel_entry'])
        self.assertFalse(score(d,f,c)['competition_tasks_complete'])

    def test_narrow_window_full_body_does_not_fit_even_center_crossing(self):
        f,d,c=fixture_and_data();f['gates']['window']['half_width_m']=.2
        self.assertFalse(score(d,f,c)['competition_tasks_complete'])

    def test_wall_bottom_height_and_permitted_span_must_both_fit(self):
        f,d,c=fixture_and_data();f['gates']['wall']['minimum_body_bottom_z_m']=.75
        self.assertFalse(score(d,f,c)['competition_tasks_complete'])
        f,d,c=fixture_and_data();f['gates']['wall']['center_m'][1]=2.
        self.assertFalse(score(d,f,c)['competition_tasks_complete'])

    def test_native_rearm_missing_refuses(self):
        f,d,c=fixture_and_data()
        for s in d['statuses']:
            if 8<=s['stamp']<12.2:s['armed']=False
        self.assertFalse(score(d,f,c)['competition_tasks_complete'])

    def test_landed_and_physical_contact_both_required(self):
        for kind in ('landed','contact'):
            f,d,c=fixture_and_data()
            if kind=='landed':d['landed']=[]
            else:d['contacts']=[x for x in d['contacts'] if 'dock_surface' not in x['pair']]
            self.assertFalse(score(d,f,c)['competition_tasks_complete'])

    def test_obstacle_contact_or_truth_gap_refuses(self):
        f,d,c=fixture_and_data();d['contacts'].append({'stamp':4.,'pair':['drone','window_rim']})
        self.assertFalse(score(d,f,c)['competition_tasks_complete'])
        f,d,c=fixture_and_data();d['poses']=[p for p in d['poses'] if not 4<p[0]<5]
        self.assertFalse(score(d,f,c)['competition_tasks_complete'])

    def test_controller_self_report_cannot_replace_independent_score_and_vice_versa(self):
        f,d,c=fixture_and_data();c['competition_tasks_complete']=False
        r=score(d,f,c);self.assertTrue(r['independent_competition_complete']);self.assertFalse(r['competition_tasks_complete'])
        f,d,c=fixture_and_data();d['contacts']=[]
        self.assertFalse(score(d,f,c)['independent_competition_complete'])

if __name__=='__main__':unittest.main()
