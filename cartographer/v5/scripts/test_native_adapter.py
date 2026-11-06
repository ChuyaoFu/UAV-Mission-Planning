"""Real adapter protocol in an isolated no-DDS subprocess."""
import os,sys,unittest,tempfile,subprocess,shutil
from pathlib import Path

class NativeAdapterTests(unittest.TestCase):
    def test_complete_typed_adapter_protocol(self):
        source=Path(__file__).resolve().parent;reference=source/'reference/v5/scripts'
        if not (reference/'navigate.py').exists():reference=source
        config=reference.parent/'config'
        with tempfile.TemporaryDirectory(prefix='typed-native-no-dds-',dir=os.environ.get('COMPETITION_TEST_WORKDIR')) as td:
            env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(source)+os.pathsep+str(reference),
                ROBOT_CALIBRATION=str(config/'robot.json'),STEREO_CALIBRATION=str(config/'gazebo_vio/kalibr_imucam_chain.yaml'),
                COMPETITION_STUB_RESULT=str(Path(td)/'result.json'))
            result=subprocess.run([sys.executable,str(source/'ros_cases_native_mechanical.py')],env=env,capture_output=True,text=True,timeout=60)
            if os.environ.get('NATIVE_TEST_EVIDENCE_PATH') and (Path(td)/'typed-native-evidence.json').exists():shutil.copy2(Path(td)/'typed-native-evidence.json',os.environ['NATIVE_TEST_EVIDENCE_PATH'])
            if os.environ.get('NATIVE_TEST_LOG_PATH'):Path(os.environ['NATIVE_TEST_LOG_PATH']).write_text(result.stdout+result.stderr)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

if __name__=='__main__':unittest.main()
