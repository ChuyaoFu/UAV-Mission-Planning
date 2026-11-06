"""Runner command routing only; never starts ROS, PX4 or simulator."""
import contextlib,io,json,unittest
from unittest.mock import patch
from run_competition_trial import main,parser,commands

ARGS=['dry-test','--seconds','600','--fixture','/work/cartographer/v5/evaluation/private.json','--tof-error-m','.003','--tof-extrinsic-error-m','.001','--stereo-error-m','.02','--map-pose-error-m','.20']

class RunnerTests(unittest.TestCase):
    def test_dry_run_cannot_launch_processes_and_fixture_never_enters_flight_command(self):
        stream=io.StringIO()
        with patch('subprocess.Popen',side_effect=AssertionError('No process permitted')),contextlib.redirect_stdout(stream):
            self.assertEqual(main(ARGS+['--dry-run']),0)
        data=json.loads(stream.getvalue());flight=data['commands']['flight']
        self.assertNotIn('/work/cartographer/v5/evaluation/private.json',flight)
        self.assertIn('/work/cartographer/v5/scripts/competition_navigate.py',flight)
        self.assertIn('--map-pose-error-m',flight)
        self.assertIn('--controller',data['commands']['scorer'])

    def test_observe_only_routes_to_competition_bridge_with_actuation_disabled(self):
        a=parser().parse_args(ARGS+['--observe-only']);c=commands(a)
        self.assertIn('--observe-only',c['flight'])
        self.assertIn('/nav/measured_depth',c['record'])
        self.assertNotIn('/model/x500_vio_0/pose',c['record'])

    def test_invalid_sensor_or_policy_domains_fail_even_dry_run(self):
        for extra in (['--uncertainty-admission-m','.6'],['--map-pose-error-m','-.1'],['--stereo-error-m','.06']):
            with self.assertRaises(ValueError):main(ARGS+extra+['--dry-run'])

if __name__=='__main__':unittest.main()
