"""Default-off typed launch routing; no process, ROS or flight is started."""
import unittest
from run_competition_trial import parser,commands
A=['synthetic-native','--tof-error-m','.02','--tof-extrinsic-error-m','.005','--stereo-error-m','.02','--map-pose-error-m','.03']

class NativeRunner(unittest.TestCase):
    def test_default_off_and_explicit_complete_task_flags(self):
        self.assertFalse(parser().parse_args(A).complete_task)
        c=commands(parser().parse_args(A+['--complete-task','--foot-geometry','robot-foot-geometry.json','--contact-rig-error-m','.005']))
        self.assertIn('--complete-task',c['flight']);self.assertIn('--contact-rig-error-m',c['flight'])
        self.assertIn('/work/cartographer/v5/config/robot-foot-geometry.json',c['flight'])

    def test_conditional_model_is_explicit(self):
        self.assertFalse(parser().parse_args(A).allow_conditional_free)
        self.assertNotIn('--allow-conditional-free',commands(parser().parse_args(A))['flight'])
        self.assertIn('--allow-conditional-free',commands(parser().parse_args(A+['--allow-conditional-free']))['flight'])

    def test_no_implicit_geometry_or_rig_error(self):
        for flags in (['--complete-task'],['--complete-task','--foot-geometry','robot-foot-geometry.json']):
            with self.assertRaises(ValueError):commands(parser().parse_args(A+flags))

    def test_private_fixture_not_passed_as_robot_geometry(self):
        with self.assertRaises(ValueError):commands(parser().parse_args(A+['--complete-task','--foot-geometry','/work/cartographer/v5/evaluation/private.json','--contact-rig-error-m','.005']))

if __name__=='__main__':unittest.main()
