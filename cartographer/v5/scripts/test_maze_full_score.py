"""Private actual geometry with SYNTHETIC trajectories, never actual flight evidence."""
import copy,json,unittest,os
from pathlib import Path
import numpy as np
from score_competition import score,REQUIRED,ordered_maze_gates,gate_crossing
F=json.loads(Path(os.environ.get('PRIVATE_GRADER_FIXTURE',str(Path(__file__).resolve().parents[1]/'tests/evaluation-fixtures/original-course-v3.json'))).read_text())


def complete_data():
    f=copy.deepcopy(F);g=f['gates'];home=np.array([-2.7,2.58445,.111]);tube=np.array(g['tunnel_entry']['center_m']);w=np.array(g['window']['center_m']);n=np.array(g['window']['normal_forward']);dock=np.array(f['dock']['center_surface_m'])+[0,0,.111];wall=np.array(g['wall']['center_m'])
    knots=np.array([[0,*home],[1,*home],[2,*[home[0],home[1],.8636]],
        [5,-1.5,2.58445,.8636],[8,-1.45,.97155,.8636],[11,-.4,.97155,.8636],
        [14,-.35,2.58445,.8636],[17,.8,2.58445,.8636],
        [20,*[tube[0],1.8,tube[2]]],[23,*[tube[0],.7,tube[2]]],[26,*[tube[0],-.9,tube[2]]],
        [29,*(w-.6*n)],[32,*(w+.6*n)],[35,*[dock[0],dock[1],.8]],[38,*dock],[40,*dock],
        [42,*[dock[0],dock[1],2.]],[45,*(wall+[0,-.6,.25])],[48,*(wall+[0,.6,.25])],
        [51,*[home[0],home[1],2.]],[54,*home],[55,*home]])
    ts=np.round(np.arange(0,55.001,.05),6)
    poses=[[float(t),*[float(np.interp(t,knots[:,0],knots[:,k])) for k in (1,2,3)],0,0,0,1] for t in ts]
    statuses=[{'stamp':float(t),'armed':bool(1<=t<38.2 or 39.5<=t<54.5)} for t in np.round(np.arange(0,55.001,.1),6)]
    lands=[{'stamp':float(t),'landed':bool(t<=1 or 38<=t<=40 or t>=54),'freefall':False} for t in ts]
    contacts=[{'stamp':t,'pair':['drone_foot',token]} for t,token in [(.9,'20_launch_mark'),(38.2,'32_landing_pad_5_ft'),(39.5,'32_landing_pad_5_ft'),(54.5,'20_launch_mark')]]
    d=dict(poses=poses,statuses=statuses,landed=lands,contacts=contacts,contact_topics_verified=True,clock_domain='ros_simulation',pose_reference='body',synthetic=True)
    c=dict(competition_tasks_complete=True,completed={k:{'verified':True} for k in REQUIRED},observe_only=False)
    return f,d,c


class FullMazeScore(unittest.TestCase):
    def test_all_three_then_tunnel_full_synthetic_course_passes(self):
        f,d,c=complete_data();r=score(d,f,c)
        self.assertTrue(r['competition_tasks_complete'],r);self.assertTrue(r['maze_traversal_complete'])
        self.assertEqual(len(r['events']),11)
        times=[r['events'][k]['stamp'] for k in ['MAZE_1','MAZE_2','MAZE_3','TUNNEL_ENTRY','TUNNEL_EXIT','WINDOW']]
        self.assertTrue(np.all(np.diff(times)>0),times)
        for i in range(1,4):self.assertTrue(r['events'][f'MAZE_{i}']['whole_body_crossed'])

    def test_any_one_missing_or_over_top_refuses_full(self):
        for g in F['maze_gates']:
            f,d,c=complete_data()
            for p in d['poses']:
                if p[0]<20 and abs(p[1]-g['center_m'][0])<.7:p[3]=3.
            r=score(d,f,c);self.assertFalse(r['competition_tasks_complete'],g['id']);self.assertFalse(r['independent_competition_complete'])
            self.assertIn('No full-body directed crossing',r['failure'])

    def test_missing_duplicate_or_malformed_gate_metadata_refuses(self):
        for kind in ['missing','duplicate','reverse_normal','raise_floor','erase_all']:
            f,d,c=complete_data()
            if kind=='missing':f['maze_gates'].pop()
            elif kind=='duplicate':f['maze_gates'][1]=copy.deepcopy(f['maze_gates'][0])
            elif kind=='reverse_normal':f['maze_gates'][0]['normal_forward']=[-1,0,0]
            elif kind=='raise_floor':f['maze_gates'][0]['channel_bounds_z_m'][0]=.2
            else:f.pop('maze_gates')
            self.assertFalse(score(d,f,c)['competition_tasks_complete'],kind)

    def test_fixture_list_order_does_not_replace_physical_order(self):
        f,d,c=complete_data();f['maze_gates']=f['maze_gates'][::-1];r=score(d,f,c)
        self.assertTrue(r['competition_tasks_complete'],r)
        self.assertEqual([r['events'][f'MAZE_{i}']['gate_id'] for i in range(1,4)],[g['id'] for g in F['maze_gates']])

    def test_tunnel_before_maze_does_not_supply_required_order(self):
        f,d,c=complete_data()
        # Excise all maze traversals; retain complete tube/window/dock/native/return evidence.
        for p in d['poses']:
            if p[0]<20:p[1:4]=[-2.7,1.8,.9]
        r=score(d,f,c);self.assertFalse(r['competition_tasks_complete']);self.assertIn('maze-gap-M1',r['failure'])

    def test_reverse_or_incomplete_slab_crossing_refuses(self):
        g=F['maze_gates'][0];c=np.array(g['center_m']);n=np.array(g['normal_forward']);robot=F['robot']
        for distances in [np.linspace(.6,-.6,21),np.linspace(-.6,.28,21)]:
            poses=np.array([[i*.05,*(c+x*n),0,0,0,1] for i,x in enumerate(distances)])
            with self.assertRaises(ValueError):gate_crossing(poses,g,robot,0)

if __name__=='__main__':unittest.main()
