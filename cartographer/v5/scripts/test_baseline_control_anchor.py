"""Run actual navigation-only callbacks in an isolated NO-DDS process."""
import os,subprocess,sys,unittest
from pathlib import Path


class BaselineAnchorRegression(unittest.TestCase):
    def test_actual_navigation_startup_qualification(self):
        scripts=Path(__file__).resolve().parent
        env=dict(os.environ,ROBOT_CALIBRATION=str(scripts.parent/'config/robot.json'),
                 STEREO_CALIBRATION=str(scripts.parent/'config/gazebo_vio/kalibr_imucam_chain.yaml'))
        result=subprocess.run([sys.executable,str(scripts/'ros_cases_baseline_control_anchor.py')],env=env,text=True,capture_output=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
