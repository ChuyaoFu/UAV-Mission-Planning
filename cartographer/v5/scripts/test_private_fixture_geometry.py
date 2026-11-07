"""PRIVATE source-geometry tests using SYNTHETIC poses only; no flight evidence."""
import copy,json,unittest,os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from score_competition import gate_crossing,on_pad,convex_polygon,polygon_fits,score
from test_score_competition import fixture_and_data
F=json.loads(Path(os.environ.get('PRIVATE_GRADER_FIXTURE',str(Path(__file__).resolve().parents[1]/'tests/evaluation-fixtures/original-course-v3.json'))).read_text())
R=F['robot']

def crossing(g,u=0.,v=0.,extent=.6):
    c=np.asarray(g['center_m']);n=np.asarray(g['normal_forward']);a=np.asarray(g['axis_u']);b=np.asarray(g['axis_v'])
    return np.array([[float(t),*(c+d*n+u*a+v*b),0,0,0,1] for t,d in zip(np.arange(25)*.05,np.linspace(-extent,extent,25))])

def pose(x,y,tilt=0.):
    p=F['dock'];c=np.asarray(p['center_surface_m']);c[:2]+=[x,y];c[2]+=R['resting_body_height_m']
    return np.r_[0,c,Rotation.from_euler('y',tilt).as_quat()]

class PrivateGeometry(unittest.TestCase):
    def test_missing_maze_traversal_prevents_full_course_claim(self):
        f,d,c=fixture_and_data();f['maze_gates']=F['maze_gates']
        r=score(d,f,c)
        self.assertFalse(r['independent_competition_complete']);self.assertFalse(r['competition_tasks_complete'])
        self.assertIn('No full-body directed crossing',r['failure'])

    def test_source_identity_and_names(self):
        self.assertTrue(F['private_evaluation_only']);self.assertTrue(F['navigation_input_forbidden'])
        self.assertEqual(F['robot']['model_from_body_position_quaternion'],[0,0,0,0,0,0,1])
        self.assertIn('32_landing_pad_5_ft',F['dock']['contact_tokens'])
        self.assertEqual(len(F['maze_gates']),3)
        self.assertTrue(F['maze_gates_current_scorer_consumes'])

    def test_actual_inner_oval_not_outer_box(self):
        g=F['gates']['tunnel_entry'];self.assertEqual(len(g['aperture_polygon_uv']),194)
        self.assertTrue(gate_crossing(crossing(g,.3,0),g,R,0)['whole_body_crossed'])
        fallback=copy.deepcopy(g);fallback.pop('aperture_polygon_uv')
        with self.assertRaises(ValueError):gate_crossing(crossing(g,.3,0),fallback,R,0)
        # Fits outer x/z bounding box, but the upper oval corner is solid.
        self.assertLess(.4+R['radius_m'],g['bounding_half_extents_m'][0])
        self.assertLess(.64+R['half_height_m'],g['bounding_half_extents_m'][1])
        with self.assertRaises(ValueError):gate_crossing(crossing(g,.4,.64),g,R,0)

    def test_actual_window_both_faces_and_whole_body(self):
        g=F['gates']['window'];self.assertEqual(len(g['hole_vertices_world_m']),8)
        self.assertEqual([len(x) for x in g['aperture_face_polygons_uv']],[4,4])
        self.assertTrue(gate_crossing(crossing(g),g,R,0)['whole_body_crossed'])
        with self.assertRaises(ValueError):gate_crossing(crossing(g,.25,0),g,R,0)
        # A center crossing does not suffice when the far side of the BODY remains in the slab.
        with self.assertRaises(ValueError):gate_crossing(crossing(g,extent=.28),g,R,0)

    def test_actual_round_pad_body_support(self):
        pad=F['dock'];self.assertTrue(on_pad(pose(.4,0),pad,R))
        fallback=copy.deepcopy(pad);fallback.pop('support_polygon_uv')
        self.assertFalse(on_pad(pose(.4,0),fallback,R))
        self.assertFalse(on_pad(pose(.4,.4),pad,R))
        self.assertFalse(on_pad(pose(0,0,tilt=1.2),pad,{**R,'radius_m':.8}))

    def test_invalid_polygon_is_refused_not_convexified(self):
        for p in ([[0,0],[1,0],[.2,.2],[1,1],[0,1]],[[0,0],[1,0],[0,0]],[[0,0],[1,1],[0,1],[1,0]]):
            with self.assertRaises(ValueError):convex_polygon(p)
        p=F['dock']['support_polygon_uv'];a=convex_polygon(p);b=convex_polygon(p[::-1]);self.assertEqual(len(a[0]),len(b[0]))
        self.assertTrue(polygon_fits(pose(0,0),np.zeros(3),np.array([1,0,0]),np.array([0,1,0]),b,R))

    def test_wall_actual_top_and_slab(self):
        g=F['gates']['wall'];self.assertAlmostEqual(g['minimum_body_bottom_z_m'],1.7272)
        self.assertTrue(gate_crossing(crossing(g,v=.13),g,R,0)['whole_body_crossed'])
        with self.assertRaises(ValueError):gate_crossing(crossing(g,v=.11),g,R,0)

    def test_three_maze_source_channels_each_require_below_wall(self):
        for g in F['maze_gates']:
            self.assertAlmostEqual(2*g['half_width_m'],.9271)
            self.assertTrue(gate_crossing(crossing(g),g,R,0)['whole_body_crossed'])
            with self.assertRaises(ValueError):gate_crossing(crossing(g,v=1.),g,R,0)

if __name__=='__main__':unittest.main()
