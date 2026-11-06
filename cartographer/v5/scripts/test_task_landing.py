"""Sensor-domain calibration tests: no simulator state/CAD input."""
import dataclasses
import unittest
import numpy as np
from task_landing import RestingFootCalibrator,RangePose,surface_relative_observation

MOUNT=np.array([0.,0.,-.023]);AXIS=[0,0,-1]

def sample(stamp,range_m=.101,position=(0,0,.124),landed=True,armed=False,**kwargs):
    values=dict(stamp=stamp,range_stamp=stamp,pose_stamp=stamp,range_m=range_m,
                position_local_m=position,local_from_body_rotation=np.eye(3),velocity_local_mps=[0,0,0],
                angular_velocity_body_radps=[0,0,0],epoch='own-local',landed=landed,armed=armed,
                range_valid=True,tracking_valid=True)
    values.update(kwargs);return RangePose(**values)


def calibrated_foot():
    calibrator=RestingFootCalibrator(MOUNT,AXIS,'actual-kit',.003,.001)
    for i in range(30):
        s=sample(i*.05,range_m=.101+.0004*np.sin(i))
        assert calibrator.ingest(s,s.stamp)
    result=calibrator.result(1.45)
    assert result is not None
    return result


class FootCalibrationTests(unittest.TestCase):
    def test_actual_resting_range_and_extrinsic_set_offset_not_a_test_default(self):
        for raw,mountz in ((.101,-.023),(.157,-.036)):
            c=RestingFootCalibrator([0,0,mountz],AXIS,'rig',.003,.001)
            for i in range(25):self.assertTrue(c.ingest(sample(i*.05,raw),i*.05))
            result=c.result(1.2)
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result.offset_m,raw-mountz)
            self.assertEqual(result.sample_count,25)
            self.assertAlmostEqual(result.resting_ground_z_local_m,.124-result.offset_m)

    def test_no_measurement_short_window_or_stale_window_never_calibrates(self):
        c=RestingFootCalibrator(MOUNT,AXIS,'rig',.003,.001)
        self.assertIsNone(c.result(0.))
        for i in range(10):c.ingest(sample(i*.05),i*.05)
        self.assertIsNone(c.result(.45))
        for i in range(10,25):c.ingest(sample(i*.05),i*.05)
        self.assertIsNone(c.result(1.5))

    def test_bad_sensor_status_or_vehicle_state_clears_continuous_window(self):
        bads=[dict(landed=False),dict(armed=True),dict(range_valid=False),dict(tracking_valid=False),
              dict(range_m=np.nan),dict(range_m=0.),dict(range_stamp=-1.),dict(pose_stamp=-1.),
              dict(velocity_local_mps=[.1,0,0]),dict(angular_velocity_body_radps=[0,.1,0])]
        for bad in bads:
            c=RestingFootCalibrator(MOUNT,AXIS,'rig',.003,.001)
            for i in range(24):c.ingest(sample(i*.05),i*.05)
            self.assertFalse(c.ingest(sample(1.2,**bad),1.2),bad)
            self.assertIsNone(c.result(1.2))
            self.assertEqual(len(c.samples),0)

    def test_tilt_motion_discontinuity_and_unstable_range_rejected(self):
        angle=np.deg2rad(4);R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        for bad in (dict(local_from_body_rotation=R),dict(position=(.02,0,.124)),dict(range_m=.14)):
            c=RestingFootCalibrator(MOUNT,AXIS,'rig',.003,.001)
            c.ingest(sample(0.),0.)
            s=sample(.05,**bad)
            self.assertFalse(c.ingest(s,.05))
        c=RestingFootCalibrator(MOUNT,AXIS,'rig',.003,.001)
        c.ingest(sample(0.),0.)
        self.assertFalse(c.ingest(sample(.3),.3))

    def test_duplicate_or_reset_samples_cannot_reuse_previous_dwell(self):
        c=RestingFootCalibrator(MOUNT,AXIS,'rig',.003,.001)
        for i in range(24):c.ingest(sample(i*.05),i*.05)
        self.assertFalse(c.ingest(sample(1.15),1.15))
        for i in range(24):c.ingest(sample(2+i*.05),2+i*.05)
        self.assertTrue(c.ingest(sample(3.2,epoch='new-gauge'),3.2))
        self.assertIsNone(c.result(3.2))
        self.assertEqual(len(c.samples),1)

    def test_preserves_vio_gauge_and_reports_raised_platform_relative_to_start(self):
        cal=calibrated_foot();platform=.63;bodyz=platform+cal.offset_m+.20
        raw=bodyz+MOUNT[2]-platform
        s=sample(2.,raw,position=(0,0,bodyz),landed=False,armed=True)
        observation=surface_relative_observation(s,2.,MOUNT,AXIS,cal,[0,0,platform],[0,0,1],'observed-platform')
        self.assertAlmostEqual(observation['foot_clearance_m'],.20,places=6)
        self.assertAlmostEqual(observation['surface_height_above_start_ground_m'],platform-cal.resting_ground_z_local_m)
        self.assertAlmostEqual(observation['plane_residual_m'],0.)
        self.assertFalse(observation['vio_global_z_correction_applied'])
        np.testing.assert_allclose(s.position_local_m,[0,0,bodyz])

if __name__=='__main__':unittest.main()
