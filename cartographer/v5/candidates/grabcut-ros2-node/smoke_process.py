"""Actual snapshot process/deadline smoke; controlled test transport only.
No health is invented, no ROS publisher exists and conditional-free is disabled.
"""
import sys,time,json
from pathlib import Path
import numpy as np,cv2,yaml
from bench_stream import fixture
from stream_adapter import GrabCutLatestAdapter

def main():
 b=Path(__file__).resolve().parent;library=sys.argv[1];out=Path(sys.argv[2]);r=yaml.safe_load('\n'.join(l for l in (b/'replay-calibration/kalibr_imucam_chain.yaml').read_text().splitlines() if not l.startswith('%YAML')));job,d=fixture(b/'replay-fixtures/job-18700000000.npz');s=int(d['expected_stamp_ns']);a=GrabCutLatestAdapter(np.array(r['cam0']['T_imu_cam']),*[cv2.imread(str(b/f'replay-calibration/self-mask{i}.png'),0)>0 for i in (0,1)],library,enabled=True,permit_conditional_free=False,clock_ns=lambda:s+10_000_000);a.start();lat=[]
 try:
  for f in job['frames']:
   f=dict(f,left_frame_id='TRANSPORT_TEST_ONLY_NOT_ROS_PROVENANCE',depth_frame_id='TRANSPORT_TEST_ONLY_NOT_ROS_PROVENANCE')
   for k in ('left','right','measured_stereo'):f[k].setflags(write=False)
   t=time.perf_counter();a.offer_exposure(f,pose_health_ok=None);lat.append(time.perf_counter()-t)
  job['joint'].setflags(write=False);a.offer_native_joint(job['joint']);deadline=time.monotonic()+8;result=None
  while time.monotonic()<deadline:
   result=a.poll()
   if result is not None:break
   time.sleep(.01)
  if result is None:raise RuntimeError('no bounded worker result')
  info=dict(scope='Actual causal frame data; controlled test-only clock/frame_id, no pose health/ROS publication/conditional-free permission. Measures transport/process behavior, not physical validation.',admission_s=lat,maximum_admission_s=max(lat),result_status=result['stats']['status'],stats=result['stats'],eligible_for_inferred_free=result.get('eligible_for_inferred_free',False),raw_process_alive=True);assert not info['eligible_for_inferred_free'];out.write_text(json.dumps(info,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n');print(json.dumps(info,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()))
 finally:a.close()
if __name__=='__main__':main()
