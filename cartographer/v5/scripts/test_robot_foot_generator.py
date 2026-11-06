"""Robot-model extraction only. No field/CAD/world inputs are opened."""
import sys,json,unittest,os,tempfile
from pathlib import Path
from generate_robot_foot_geometry import generate
from task_mechanical_foot import MechanicalFootGeometry,RelativeContractError
DEFAULT=str(Path(__file__).resolve().parents[1]/'config/model.sdf')
MODEL=Path(os.environ.get('ROBOT_MODEL_SDF',DEFAULT))

class RobotFootGenerator(unittest.TestCase):
    @unittest.skipUnless(MODEL.exists(),'Set ROBOT_MODEL_SDF to actual robot model on this host')
    def test_actual_model_reproduces_foot_geometry_and_nonfoot_audit(self):
        record=generate(MODEL);g=MechanicalFootGeometry.from_dict(record);points,h,error=g.validate('sitl',record['payload']['source_model_sha256'])
        self.assertEqual(len(points),8);self.assertAlmostEqual(h,.111);self.assertLess(error,.012)
        self.assertEqual(len(record['payload']['other_collision_support_audit']),9)
        self.assertLess(record['payload']['other_collision_downward_support_upper_m'],.09)

    def test_static_field_and_unknown_shapes_are_refused(self):
        examples=['<sdf><model name="static_course"><static>true</static><link name="base_link"/></model></sdf>',
            '<sdf><model name="robot"><link name="base_link"><collision name="mystery"><geometry><sphere><radius>1</radius></sphere></geometry></collision></link></model></sdf>']
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'synthetic.sdf'
            for xml in examples:
                p.write_text(xml)
                with self.assertRaises(ValueError):generate(p)

    def test_missing_model_or_invalid_geometry_budget_refuses(self):
        with self.assertRaises(ValueError):generate('/never-opened',point_error_m=0.)
        with self.assertRaises(ValueError):generate('/never-opened',tilt_limit_deg=6.)

if __name__=='__main__':unittest.main()
