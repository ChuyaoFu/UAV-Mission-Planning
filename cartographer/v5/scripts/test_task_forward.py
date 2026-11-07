import dataclasses
import unittest
import numpy as np
from task_observations import measured_forward_evidence
from task_safety import MeasuredMapGuard
from online_map import OnlineMap
from test_task_observations import exposure
import test_task_executive as fixtures
from test_task_executive import target,measured_map,guard_at


class ForwardProofTests(unittest.TestCase):
    def crossing_fixture(self):
        fixture=fixtures.ExecutiveTests();fixture.setUp();t=target('tunnel_entry')
        for _ in range(80):
            result=fixture.tick([t] if fixture.exec.phase!='VERIFY' else [])
            fixture.follow(result)
            if fixture.exec.phase=='VERIFY':break
        self.assertEqual(fixture.exec.phase,'VERIFY')
        return fixture

    def evidence(self,f):
        return measured_forward_evidence(np.full((300,400),1.,np.float32),[300,300,200,150],
                                         exposure(f.now,f.p,epoch='own-local'),f.now)

    def test_forward_only_camera_completes_crossing_without_turning_back(self):
        f=self.crossing_fixture();f.now+=.15;e=self.evidence(f)
        self.assertIsNotNone(e)
        result=f.exec.tick(f.feedback(),[],guard_at(f.map,f.now),f.now,forward_evidence=e)
        self.assertEqual(f.exec.stage,'TUNNEL_EXIT',result.as_dict())
        record=f.exec.completed['TUNNEL_ENTRY']
        self.assertEqual(record['evidence_mode'],'continuous_vio_prior_paths_and_new_forward_measurements')
        self.assertEqual(record['far_side_observation_stamps'],[])
        self.assertGreaterEqual(record['forward_proof']['far_side_direct_points'],48)
        self.assertTrue(record['forward_proof']['pose_epoch_unchanged'])
        self.last_report=f.exec.report()

    def test_wrong_side_reset_stale_depth_or_no_new_map_cannot_replace_reobservation(self):
        for kind in ('near_side_origin','reset','old_depth','old_map'):
            f=self.crossing_fixture();cross=f.exec.cross_stamp;f.now+=.15;e=self.evidence(f)
            g=guard_at(f.map,f.now)
            if kind=='near_side_origin':e=dataclasses.replace(e,camera_origin_local_m=(-1,0,.7))
            if kind=='reset':e=dataclasses.replace(e,epoch='reset')
            if kind=='old_depth':e=dataclasses.replace(e,depth_stamp=cross-.1)
            if kind=='old_map':g=guard_at(f.map,cross)
            result=f.exec.tick(f.feedback(),[],g,f.now,forward_evidence=e)
            self.assertEqual(result.action,'observe',kind)
            self.assertNotIn('TUNNEL_ENTRY',f.exec.completed)

    def test_inferred_missing_or_narrow_patch_depth_never_supplies_forward_proof(self):
        d=np.full((300,400),1.,np.float32);K=[300,300,200,150];exp=exposure()
        self.assertIsNone(measured_forward_evidence(d,K,exp,1.,np.zeros(d.shape,bool)))
        d[:]=np.nan;d[10:20,10:20]=1.
        self.assertIsNone(measured_forward_evidence(d,K,exp,1.))

    def test_existing_collision_padding_is_not_implicitly_added_twice(self):
        m=measured_map();state=m.state.copy()
        state[:,m.idx([0,.35,0])[1]:,:]=1
        m=OnlineMap([*m.origin,m.res,*m.shape,0,1.],state,.25,.12)
        a=[0,.03,.7];b=[.1,.03,.7]
        regular=MeasuredMapGuard(m,'own-local',.25,.12,tracking_margin_m=.04)
        self.assertIsNotNone(regular.certify([a,b],1.,'own-local','existing padded body'))
        extra=MeasuredMapGuard(m,'own-local',.25,.12,extra_collision_margin_m=.04)
        self.assertIsNone(extra.certify([a,b],1.,'own-local','explicit extra padding'))
        self.assertEqual(regular.certify([a,b],1.,'own-local','test').as_dict()['extra_collision_margin_m'],0.)

if __name__=='__main__':unittest.main()
