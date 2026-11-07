"""Resting foot calibration regressions; synthetic measurements only."""
import sys,unittest
from pathlib import Path
import numpy as np
from task_landing import RestingFootCalibrator,RangePose
from task_relative_contact import existing_foot_calibrator_diagnostics


class LegacyFootGaugeTests(unittest.TestCase):
    def test_legacy_offset_never_rejects_154_or_418_global_error_or_constant_origin(self):
        for error,shift in [(0.154,[0,0,0]),(.418,[1e6,-2e6,3e6])]:
            c=RestingFootCalibrator([0,0,-.023],[0,0,-1],'rig',.003,.001)
            for i in range(30):
                t=i*.05;p=np.array([0,0,.124])+shift
                s=RangePose(t,t,t,.101,p,np.eye(3),[0,0,0],[0,0,0],'e',True,False,True,True,error)
                self.assertTrue(c.ingest(s,t),c.last_rejection)
            cal=c.result(t);self.assertIsNotNone(cal);self.assertAlmostEqual(cal.offset_m,.124)
            self.assertTrue(existing_foot_calibrator_diagnostics(c,t)['ready'])
            np.testing.assert_allclose(cal.resting_body_local_m,np.array([0,0,.124])+shift)

    def test_declared_base_error_at_cap_plus_real_offset_variation_stays_refused(self):
        c=RestingFootCalibrator([0,0,-.023],[0,0,-1],'rig',.02,.005)
        for i in range(30):
            t=i*.05;s=RangePose(t,t,t,.101+.0001*np.sin(i),[0,0,.124],np.eye(3),[0,0,0],[0,0,0],'e',True,False,True,True,.154)
            self.assertTrue(c.ingest(s,t))
        self.assertIsNone(c.result(t));d=existing_foot_calibrator_diagnostics(c,t)
        self.assertEqual(d['declared_error_sum_m'],.025);self.assertEqual(d['admission_error_cap_m'],.025)
        self.assertGreater(d['computed_offset_error_m'],.025);self.assertGreater(d['offset_spread_m'],0.)
        self.assertIsNotNone(d['diagnostic_only_median_offset_m']);self.assertTrue(d['diagnostic_median_not_accepted'])
        self.assertFalse(d['ready']);self.assertFalse(d['diagnostic_median_is_calibration']);self.assertFalse(d['numeric_admission_is_sensor_accuracy_qualification'])
        self.assertEqual(c.last_rejection,None)
        self.assertIn('25 mm',d['reason'])

    def test_diagnostics_expose_continuity_staleness_and_real_local_rejection(self):
        c=RestingFootCalibrator([0,0,-.023],[0,0,-1],'rig',.003,.001)
        self.assertIn('no synchronized',existing_foot_calibrator_diagnostics(c,0.)['reason'])
        for i in range(30):
            t=i*.05;s=RangePose(t,t,t,.101,[0,0,.124],np.eye(3),[0,0,0],[0,0,0],'e',True,False,True,True,.154)
            c.ingest(s,t)
        self.assertIn('stale',existing_foot_calibrator_diagnostics(c,2.)['reason'])
        s=RangePose(2.,2.,2.,.101,[0,0,.124],np.eye(3),[.1,0,0],[0,0,0],'e',True,False,True,True,.154)
        self.assertFalse(c.ingest(s,2.))
        d=existing_foot_calibrator_diagnostics(c,2.)
        self.assertEqual(d['sample_count'],0);self.assertIn('stationary',d['reason'])

if __name__=='__main__':unittest.main()
