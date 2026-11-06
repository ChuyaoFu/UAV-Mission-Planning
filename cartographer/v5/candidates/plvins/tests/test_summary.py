import unittest,tempfile,csv,json,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from summarize_replay import summarize
class Summary(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def csv(self,name,header,rows=()):
  with (self.root/name).open('w',newline='') as f:
   w=csv.writer(f);w.writerow(header);w.writerows(rows)
 def empty_files(self):
  self.csv('odometry.csv',['stamp_ns','x','y','z','qx','qy','qz','qw'])
  self.csv('resources.csv',['executable','cpu_percent','rss_bytes','pss_bytes'])
  self.csv('frontend.csv',['kind','stamp_ns','count'])
 def metric(self,solver=0,stamp=1000000000,cpu='3.2'):
  return 'PLVINS_METRICS stamp_ns=%d solver=%d points=12 lines=8 eligible_points=5 usable_lines=0 processing_ms=%s\n'%(stamp,solver,cpu)
 def assert_json(self,result):json.loads(json.dumps(result,allow_nan=False))
 def test_header_only_uninitialized(self):
  self.empty_files();(self.root/'plvins.log').write_text(self.metric()+'Not enough features or parallax; Move device around\nIMU excitation not enouth!\n')
  r=summarize(self.root);self.assertEqual(r['observed_state'],'initialization_not_observed');self.assertEqual(r['odometry_samples'],0);self.assertIsNone(r['first_nonlinear_stamp_ns']);self.assertEqual(r['artifacts']['odometry.csv']['state'],'header_only');self.assertEqual(r['log_events']['imu_excitation_advisory'],1);self.assertEqual(r['odometry_gap_s']['n'],0);self.assert_json(r)
 def test_missing_artifacts_not_false_uninitialized(self):
  r=summarize(self.root);self.assertEqual(r['observed_state'],'insufficient_artifacts');self.assertEqual(r['artifacts']['odometry.csv']['state'],'missing');self.assert_json(r)
 def test_zero_byte_and_invalid_headers(self):
  (self.root/'odometry.csv').touch();(self.root/'frontend.csv').write_text('garbled\n');r=summarize(self.root);self.assertEqual(r['artifacts']['odometry.csv']['state'],'empty');self.assertEqual(r['artifacts']['frontend.csv']['state'],'invalid_header');self.assert_json(r)
 def test_nonlinear_but_observer_no_odom(self):
  self.empty_files();(self.root/'plvins.log').write_text(self.metric(1));r=summarize(self.root);self.assertEqual(r['observed_state'],'initialization_observed_without_odometry_samples');self.assertEqual(r['nonlinear_frames'],1);self.assert_json(r)
 def test_partial_and_malformed_rows(self):
  self.empty_files();self.csv('odometry.csv',['stamp_ns','x','y','z','qx','qy','qz','qw'],[[10,0,0,0,0,0,0,1],['broken',1,0,0,0,0,0,1],[30,3,0,0,0,0,0,1],[40,'partial']]);r=summarize(self.root);self.assertEqual(r['odometry_samples'],2);self.assertEqual(r['artifacts']['odometry.csv']['invalid_rows'],2);self.assertEqual(r['odometry_steps_omitted_malformed_rows'],1);self.assertEqual(r['odometry_step_m']['n'],0);self.assert_json(r)
 def test_nan_inf_no_invalid_json(self):
  self.empty_files();(self.root/'plvins.log').write_text(self.metric(1,cpu='nan')+self.metric(1,2,cpu='inf'))
  self.csv('odometry.csv',['stamp_ns','x','y','z','qx','qy','qz','qw'],[[10,0,0,0,0,0,0,1],[20,'nan',0,0,0,0,0,1],[30,2,0,0,0,0,0,1]])
  self.csv('resources.csv',['executable','cpu_percent','rss_bytes','pss_bytes'],[['plvins_estimator',-1,100,-1],['plvins_estimator',20,200,100]])
  r=summarize(self.root);self.assertEqual(r['odometry_nonfinite_values'],1);self.assertEqual(r['odometry_steps_omitted_nonfinite_positions'],2);self.assertEqual(r['metrics']['processing_ms']['nonfinite_excluded'],2);self.assertEqual(r['resources']['plvins_estimator']['cpu_percent']['n'],1);self.assert_json(r)
 def test_nanosecond_integer_precision_and_nonmonotonic(self):
  self.empty_files();n=1700000000000000000
  self.csv('odometry.csv',['stamp_ns','x','y','z','qx','qy','qz','qw'],[[n,0,0,0,0,0,0,1],[n+1,1,0,0,0,0,0,1],[n,2,0,0,0,0,0,1]])
  r=summarize(self.root);self.assertEqual(r['odometry_nonmonotonic_stamps'],1);self.assertAlmostEqual(r['odometry_gap_s']['max'],1e-9);self.assertAlmostEqual(r['odometry_gap_s']['min'],-1e-9);self.assert_json(r)
 def test_failure_logs_without_estimator_metrics(self):
  (self.root/'plvins.log').write_text('process[plvins_estimator-3] has died\nsystem reboot!\nPLVINS_METRICS corrupt partial\n')
  r=summarize(self.root);self.assertEqual(r['process_died_log_lines'],1);self.assertEqual(r['malformed_metric_records'],1);self.assertEqual(r['observed_state'],'estimator_updates_not_observed');self.assert_json(r)
if __name__=='__main__':unittest.main(verbosity=2)
