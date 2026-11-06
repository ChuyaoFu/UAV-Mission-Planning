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

    def test_native_envelope_is_explicit_and_records_both_sides(self):
        default=commands(parser().parse_args(ARGS))
        enabled=commands(parser().parse_args(ARGS+['--require-ego-envelope']))
        self.assertNotIn('--require-ego-envelope',default['flight'])
        self.assertIn('--require-ego-envelope',enabled['flight'])
        for topic in ['/nav/ego_collision_envelope','/nav/ego_collision_envelope_ack']:
            self.assertIn(topic,enabled['record'])

    def test_explicit_mechanical_geometry_only_routes_to_observe_inside_robot_config(self):
        a=parser().parse_args(ARGS+['--observe-only','--foot-geometry','robot-foot-geometry.json']);c=commands(a)
        self.assertIn('--foot-geometry',c['flight']);self.assertIn('/work/cartographer/v5/config/robot-foot-geometry.json',c['flight'])
        self.assertIn('/nav/home_relative_adapter_health',c['record'])
        for extra in (['--foot-geometry','robot-foot-geometry.json'],['--observe-only','--foot-geometry','/work/cartographer/v5/evaluation/private.json']):
            with self.assertRaises(ValueError):commands(parser().parse_args(ARGS+extra))

    def test_invalid_sensor_or_policy_domains_fail_even_dry_run(self):
        for extra in (['--uncertainty-admission-m','.6'],['--map-pose-error-m','-.1'],['--stereo-error-m','.06']):
            with self.assertRaises(ValueError):main(ARGS+extra+['--dry-run'])

if __name__=='__main__':unittest.main()
