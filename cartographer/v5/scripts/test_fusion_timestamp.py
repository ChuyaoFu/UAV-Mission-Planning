"""Run actual Fusion callbacks in an isolated, explicitly no-DDS process."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class FusionTimestampTests(unittest.TestCase):
    def test_rejected_samples_do_not_poison_the_stream(self):
        scripts=Path(__file__).resolve().parent
        env=os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE='1',NAV_HOME_RELATIVE_ODOMETRY='1',
                   ROBOT_CALIBRATION=str(scripts.parent/'config/robot.json'))
        result=subprocess.run([sys.executable,str(scripts/'ros_cases_fusion_timestamp.py')],
                              env=env,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
