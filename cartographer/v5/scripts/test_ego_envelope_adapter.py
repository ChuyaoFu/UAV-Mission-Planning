"""Isolate ROS stubs from other unit tests; never creates a DDS participant."""
import os,sys,subprocess,tempfile,unittest
from pathlib import Path

class EnvelopeAdapterTests(unittest.TestCase):
    def test_actual_competition_adapter(self):
        source=Path(__file__).resolve().parent;config=source.parent/'config'
        with tempfile.TemporaryDirectory(dir=os.environ.get('COMPETITION_TEST_WORKDIR')) as td:
            env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(source),
                ROBOT_CALIBRATION=str(config/'robot.json'),STEREO_CALIBRATION=str(config/'gazebo_vio/kalibr_imucam_chain.yaml'),
                COMPETITION_STUB_RESULT=str(Path(td)/'result.json'))
            r=subprocess.run([sys.executable,str(source/'ros_cases_ego_collision_envelope.py')],env=env,capture_output=True,text=True,timeout=45)
        if os.environ.get('EGO_ENVELOPE_TEST_LOG'):Path(os.environ['EGO_ENVELOPE_TEST_LOG']).write_text(r.stdout+r.stderr)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)

if __name__=='__main__':unittest.main()
