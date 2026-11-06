"""No live processes, ROS, FCU or video; state/ownership boundary regressions."""
import os,signal,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
from run_competition_trial import retain_until_safe,request_capture_flush,native_status_fields,commands,parser
from record_competition import recording_keepalive_fresh


class RetentionTests(unittest.TestCase):
    def test_live_or_armed_or_unreadable_state_keeps_waiting_and_heartbeats(self):
        now=[0.];beats=[]
        flight=NS(pid=777,poll=lambda:None if now[0]<20 else 0)
        def native():
            if now[0]<5:raise TimeoutError('native unavailable')
            return f'timestamp: {int((now[0]+1)*1e6)}\narming_state: {2 if now[0]<40 else 1}'
        with patch('os.killpg',side_effect=AssertionError('wait loop cannot signal anything')):
            result=retain_until_safe(flight,native,lambda s:beats.append((now[0],s.copy())),sleep=lambda dt:now.__setitem__(0,now[0]+dt),clock=lambda:now[0])
        self.assertTrue(result['resolved']);self.assertEqual(now[0],41.)
        self.assertTrue(beats[0][1]['native_read_error']);self.assertFalse(any(s['cleanup_allowed'] for _,s in beats[:-1]))
        self.assertTrue(all(b[0]-a[0]<=16 for a,b in zip(beats,beats[1:])))

    def test_stale_disarm_sample_does_not_resolve_intervention(self):
        now=[0.];flight=NS(pid=5,poll=lambda:None if now[0]<1 else 0)
        def native():return f'timestamp: {100 if now[0]<3 else 101}\narming_state: 1'
        r=retain_until_safe(flight,native,lambda s:None,sleep=lambda dt:now.__setitem__(0,now[0]+dt),clock=lambda:now[0])
        self.assertTrue(r['resolved']);self.assertEqual(now[0],3.)

    def test_normal_exited_disarmed_path_has_no_intervention_or_capture_change(self):
        heartbeat=Mock();flight=NS(pid=5,poll=lambda:0)
        r=retain_until_safe(flight,lambda:'timestamp: 100\narming_state: 1',heartbeat)
        self.assertFalse(r['entered']);heartbeat.assert_not_called()

    def test_disk_flush_never_signals_flight_or_external_slam_fcu(self):
        flight=NS(pid=1,poll=lambda:None);rec=NS(pid=2,poll=lambda:None);dead=NS(pid=3,poll=lambda:0)
        video=NS(p=NS(pid=4,poll=lambda:None,send_signal=Mock()))
        with patch('os.killpg') as kill:request_capture_flush([flight,rec,dead],flight,video)
        kill.assert_called_once_with(2,signal.SIGINT);video.p.send_signal.assert_called_once_with(signal.SIGINT)

    def test_independent_recorder_leash_requires_fresh_existing_heartbeat(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'intervention-needed.json'
            self.assertFalse(recording_keepalive_fresh(path,lambda:100.))
            path.write_text('{}');os.utime(path,(100.,100.))
            self.assertTrue(recording_keepalive_fresh(path,lambda:120.))
            self.assertFalse(recording_keepalive_fresh(path,lambda:161.))
            self.assertFalse(recording_keepalive_fresh(path,lambda:99.))

    def test_native_status_parse_and_independent_command_keepalive_are_explicit(self):
        self.assertEqual(native_status_fields('timestamp: 123\narming_state: 2'),(2,123))
        self.assertEqual(native_status_fields('bad'),(None,None))
        a=parser().parse_args(['x','--tof-error-m','.003','--tof-extrinsic-error-m','.001','--stereo-error-m','.02','--map-pose-error-m','.42'])
        c=commands(a);self.assertIn('--keepalive-file',c['independent']);self.assertNotIn('--keepalive-file',c['flight'])
        self.assertEqual(int(c['independent'][c['independent'].index('--seconds')+1]),a.seconds*3+600)

if __name__=='__main__':unittest.main()
