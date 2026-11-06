import tempfile
import unittest
from pathlib import Path
import numpy as np
from robot_allocation import calibration


class AllocationTests(unittest.TestCase):
    model=Path(__file__).resolve().parents[1]/'config/model.sdf'

    def test_payload_mass_and_cg_change_real_hover_moments(self):
        report=calibration(self.model)
        self.assertAlmostEqual(report['mass_kg'],1.166)
        self.assertAlmostEqual(report['center_of_mass_body_flu_m'][0],.05*.22/1.166)
        rotors=report['rotors'];xyz=np.array([r['position_from_cg_frd_m'] for r in rotors])
        A=np.vstack([np.ones(4),-xyz[:,1],xyz[:,0],[r['moment_ratio'] for r in rotors]])
        thrust=np.linalg.solve(A,np.array([1.,0.,0.,0.]))
        self.assertTrue((thrust>0).all());self.assertLess(thrust.max(),.3)
        self.assertTrue(np.allclose(A@thrust,[1,0,0,0],atol=1e-12))
        # Old symmetric allocation produces a real pitch moment around the
        # displaced CG under collective thrust; this is not an estimator test.
        self.assertGreater(abs((A@np.full(4,.25))[2]),.009)

    def test_centered_payload_restores_symmetric_hover(self):
        raw=self.model.read_text();import xml.etree.ElementTree as ET
        tree=ET.fromstring(raw);tree.find("model/link[@name='vio_rig']/pose").text='0 0 .055 0 0 0'
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'robot.sdf';p.write_text(ET.tostring(tree,encoding='unicode'));report=calibration(p)
        self.assertAlmostEqual(report['center_of_mass_body_flu_m'][0],0.)
        self.assertAlmostEqual(sum(r['position_from_cg_frd_m'][0] for r in report['rotors']),0.)


if __name__=='__main__':unittest.main()
