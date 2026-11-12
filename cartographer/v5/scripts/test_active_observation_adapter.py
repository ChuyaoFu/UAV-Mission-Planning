"""Subprocess boundary keeps ROS/PX4 stubs out of other test modules."""
import os,subprocess,sys,tempfile,unittest
from pathlib import Path

class ActiveObservationAdapterTests(unittest.TestCase):
    def test_active_observation_callbacks_in_no_dds_process(self):
        source=Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=os.environ.get('COMPETITION_TEST_WORKDIR')) as td:
            env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
            env['PYTHONPATH']=str(source)+os.pathsep+env.get('PYTHONPATH','')
            env.setdefault('ROBOT_CALIBRATION',str(source.parent/'config/robot.json'))
            env.setdefault('STEREO_CALIBRATION',str(source.parent/'config/gazebo_vio/kalibr_imucam_chain.yaml'))
            env['COMPETITION_STUB_RESULT']=str(Path(td)/'result.json')
            boot='import os,runpy,sys;from pathlib import Path;from competition_ros_stubs import install;install();import navigate;navigate.ROOT=Path(os.environ["ROBOT_CALIBRATION"]).parents[1];path=sys.argv[1];sys.argv=[path];runpy.run_path(path,run_name="__main__")'
            r=subprocess.run([sys.executable,'-c',boot,str(source/'ros_cases_active_observation.py')],env=env,capture_output=True,text=True,timeout=45)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
if __name__=='__main__':unittest.main()
