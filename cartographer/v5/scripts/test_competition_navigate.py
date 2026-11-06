"""Keep ROS stubs isolated from the rest of the project's unit-test process."""
import os,sys,unittest,subprocess,tempfile
from pathlib import Path


class ROSAdapterSubprocessTests(unittest.TestCase):
    def test_real_navigator_adapter_with_explicit_no_dds_ros_px4_stubs(self):
        source=Path(__file__).resolve().parent
        candidates=[source/'reference/v5/scripts',source.parent/'ros_snapshot/v5/scripts',source]
        reference=next((p for p in candidates if (p/'navigate.py').exists()),None)
        self.assertIsNotNone(reference,'Provide real navigate.py and calibrated config snapshot for interface tests')
        config=reference.parent/'config'
        root=os.environ.get('COMPETITION_TEST_WORKDIR')
        if root:Path(root).mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='competition-no-dds-',dir=root) as tmp:
            env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
            env['PYTHONPATH']=str(source)+os.pathsep+str(reference)
            env['ROBOT_CALIBRATION']=str(config/'robot.json')
            env['STEREO_CALIBRATION']=str(config/'gazebo_vio/kalibr_imucam_chain.yaml')
            env['COMPETITION_STUB_RESULT']=str(Path(tmp)/'result.json')
            result=subprocess.run([sys.executable,str(source/'ros_cases_competition.py')],env=env,
                                   capture_output=True,text=True,timeout=45)
        if os.environ.get('COMPETITION_STUB_LOG'):
            Path(os.environ['COMPETITION_STUB_LOG']).write_text(result.stdout+result.stderr)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

if __name__=='__main__':unittest.main()
