"""Run actual Navigator bridge callbacks with isolated NO-DDS stubs."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class NativeExecutionBridgeTests(unittest.TestCase):
    def test_original_ego_sample_and_stable_control_frame(self):
        scripts=Path(__file__).resolve().parent
        env=os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE='1',
                   ROBOT_CALIBRATION=str(scripts.parent/'config/robot.json'),
                   STEREO_CALIBRATION=str(scripts.parent/'config/gazebo_vio/kalibr_imucam_chain.yaml'))
        result=subprocess.run([sys.executable,str(scripts/'ros_cases_native_execution_bridge.py')],
                              env=env,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
