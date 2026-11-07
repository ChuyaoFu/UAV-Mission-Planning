"""Transport/guard checks only. Actual causal snapshots, no ROS publication.
Synthetic optical frame_id and controlled clocks are labelled test-only; they
are never claimed as physical provenance or fed into flight/navigation.
"""
import unittest,time,json,sys,copy,os
from pathlib import Path
import numpy as np,cv2,yaml
from stream_adapter import GrabCutLatestAdapter
from bench_stream import fixture
BASE=Path(__file__).resolve().parent
RIG=yaml.safe_load('\n'.join(l for l in (BASE/'replay-calibration/kalibr_imucam_chain.yaml').read_text().splitlines() if not l.startswith('%YAML')))
TIC=np.array(RIG['cam0']['T_imu_cam']);M=[cv2.imread(str(BASE/f'replay-calibration/self-mask{i}.png'),0)>0 for i in (0,1)]
LIB=sys.argv[1] if len(sys.argv)>1 and Path(sys.argv[1]).is_file() else None
if LIB:sys.argv.pop(1)
class Guards(unittest.TestCase):
 def setUp(self):self.job,self.expected=fixture(Path(os.environ.get('SENSOR_CONTRACT_FIXTURE',BASE/'replay-fixtures/job-18700000000.npz')));self.adapters=[]
 def tearDown(self):
  for a in self.adapters:a.close()
 def adapter(self,**kw):
  a=GrabCutLatestAdapter(TIC,*M,LIB or 'NOT_STARTED',**kw);self.adapters.append(a);return a
 def offered(self,f):
  f=dict(f,left_frame_id='TRANSPORT_TEST_ONLY',depth_frame_id='TRANSPORT_TEST_ONLY')
  for k in ('left','right','measured_stereo'):f[k].setflags(write=False)
  return f
 def test_default_disabled_has_no_job_or_cache(self):
  a=self.adapter();self.assertFalse(a.start());self.assertFalse(a.offer_exposure({}));self.assertEqual(len(a.frames),0);self.assertIsNone(a.process)
 def test_exact_header_and_immutable_input_are_required(self):
  a=self.adapter(enabled=True);f=self.offered(self.job['frames'][0]);bad=dict(f,right_stamp_ns=int(f['stamp_ns'])+1);self.assertFalse(a.offer_exposure(bad));f['left']=f['left'].copy()
  with self.assertRaises(ValueError):a.offer_exposure(f)
 def test_cache_bounded_nonwaiting_and_not_a_batch_dependency(self):
  a=self.adapter(enabled=True,max_cached_frames=3);elapsed=[]
  for frame in self.job['frames']:
   f=self.offered(frame);t=time.perf_counter();self.assertTrue(a.offer_exposure(f));elapsed.append(time.perf_counter()-t)
  self.assertEqual(len(a.frames),3);self.assertEqual(next(reversed(a.frames)),int(self.job['frames'][-1]['stamp_ns']));print('admission seconds',elapsed,flush=True)
 def test_raw_nan_bits_survive_and_stale_never_refreshes(self):
  s=int(self.expected['expected_stamp_ns']);clock=[s+100_000_000];a=self.adapter(enabled=True,permit_conditional_free=True,clock_ns=lambda:clock[0],health_ok_now=lambda:True);raw=self.job['frames'][-1]['measured_stereo'].copy();raw.view(np.uint32)[0,0]=0x7fc00123;packet=dict(stamp_ns=s,original_measured_depth=raw,K=self.job['frames'][-1]['K'],inferred_depth=self.expected['expected_surface']);r=dict(packet=packet,eligible_for_inferred_free=True,stats=dict(job_input_ros_age_s=.1),submitted_monotonic=time.monotonic());e=a.evidence_32fc2(r);np.testing.assert_array_equal(e[...,0].view(np.uint32),raw.view(np.uint32));self.assertGreater(np.isfinite(e[...,1]).sum(),0);clock[0]=s+600_000_000;e=a.evidence_32fc2(r);self.assertFalse(np.isfinite(e[...,1]).any());np.testing.assert_array_equal(e[...,0].view(np.uint32),raw.view(np.uint32))
 def test_consumer_quantization_credit_requires_verified_contract(self):
  with self.assertRaises(ValueError):self.adapter(consumer_reserve_credit_m=.17)
 def test_missing_current_health_forces_raw_only(self):
  s=int(self.expected['expected_stamp_ns']);a=self.adapter(enabled=True,permit_conditional_free=True,clock_ns=lambda:s);packet=dict(stamp_ns=s,original_measured_depth=self.job['frames'][-1]['measured_stereo'],K=self.job['frames'][-1]['K'],inferred_depth=self.expected['expected_surface']);e=a.evidence_32fc2(dict(packet=packet,eligible_for_inferred_free=True));self.assertFalse(np.isfinite(e[...,1]).any())
 @unittest.skipUnless(LIB,'built native library required')
 def test_native_process_death_does_not_block_raw_admission(self):
  s=int(self.expected['expected_stamp_ns']);a=self.adapter(enabled=True,clock_ns=lambda:s);self.assertTrue(a.start());a.process.terminate();a.process.join(timeout=1);t=time.perf_counter();self.assertTrue(a.offer_exposure(self.offered(self.job['frames'][-1])));self.assertLess(time.perf_counter()-t,.05);self.assertFalse(a.process.is_alive())
class BodyClockCases(unittest.TestCase):
 def test_20hz_poses_quantized_to_4ms_accept_a_52ms_bracket(self):
  from live_inputs import body_at,transform
  from scipy.spatial.transform import Rotation
  poses={1000000000:transform([0,0,0],[0,0,0,1]),1052000000:transform([.0052,0,0],Rotation.from_rotvec([0,0,.0104]).as_quat())}
  value,metadata=body_at(1033000000,poses)
  np.testing.assert_allclose(value[:3,3],[.0033,0,0],atol=1e-14)
  np.testing.assert_allclose(value[:3,:3],Rotation.from_rotvec([0,0,.0066]).as_matrix(),atol=1e-14)
  self.assertGreater(metadata['position_interpolation_bound_m'],0)
  self.assertEqual(body_at(1033000000,{1000000000:poses[1000000000],1061000000:poses[1052000000]}),(None,None))
if __name__=='__main__':unittest.main()
