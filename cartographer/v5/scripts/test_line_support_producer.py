import copy,math,unittest
from pathlib import Path
import line_support_producer as lsp
from line_support_producer import LineSupportProducer,ProducerConfig,parse

NOW=10_000_000_000
EXPOSURE=NOW-60_000_000
TOF=EXPOSURE-20_000_000


def config(**kw):
    values=dict(enabled=True,epoch='home-123',frame='world',
                declaration='SYNTHETIC_TEST_DOMAIN_NOT_FOR_FLIGHT',input_age_ns=500_000_000)
    values.update(kw);return ProducerConfig(**values)


def hint(exposure=EXPOSURE,tof=TOF,frame='world',floor=-.0125):
    seg=lambda a,b,n,src=lsp.MEASURED_SOURCE:{'endpoints':[[*a,floor],[*b,floor]],'support_points':n,'source':src}
    return {'stamp':exposure*1e-9,'floor_z_measured':floor,
            'source':'relative image/stereo/ToF geometry only; not a pose or free-space correction',
            'segments':[seg((.30,.52),(1.10,.55),40),   # 0: long forward edge
                        seg((.35,.60),(.90,.61),50),    # 1: shorter, more support (length*support 27.5 < 32.0)
                        seg((.40,-.30),(.42,.40),50)],  # 2: transverse edge
            'conditional_support':{'schema':lsp.HINT_SCHEMA,'frame_id':frame,
                                   'exposure_stamp_ns':exposure,'tof_stamp_ns':tof}}


def emit(p,h=None,indices=(0,1),now=NOW,allowed=True,home=7):
    return p.record(hint() if h is None else h,indices,now_ns=now,allowed=allowed,home_epoch_ns=home)


class ProducerTests(unittest.TestCase):
    def test_default_is_off_and_needs_no_dimensions(self):
        p=LineSupportProducer(ProducerConfig())
        r=emit(p)
        self.assertIsNone(r.wire);self.assertEqual(r.reason,'disabled');self.assertEqual(p.sequence,0)
        self.assertFalse(any('halfwidth' in f or 'z_m' in f for f in ProducerConfig.__dataclass_fields__))

    def test_enabled_requires_explicit_identifiers_and_native_input_age(self):
        for bad in [dict(epoch=''),dict(frame=''),dict(declaration=''),dict(epoch='has space'),
                    dict(declaration='x'*129),dict(input_age_ns=0),dict(input_age_ns=500_000_001),
                    dict(input_age_ns=.25),dict(enabled=1)]:
            with self.assertRaises(ValueError,msg=bad):config(**bad)

    def test_emits_one_exact_measured_segment_with_original_stamps(self):
        p=LineSupportProducer(config());h=hint()
        r=emit(p,h)
        self.assertEqual(r.reason,'emitted')
        t=r.wire.split();self.assertEqual(len(t),12)
        self.assertEqual(t[:7],['EGO_LINE_SUPPORT_V1','home-123','world','SYNTHETIC_TEST_DOMAIN_NOT_FOR_FLIGHT','1',str(EXPOSURE),str(TOF)])
        d=parse(r.wire)
        a,b=h['segments'][0]['endpoints']
        self.assertEqual((d['ax'],d['ay'],d['bx'],d['by'],d['floor_z']),(a[0],a[1],b[0],b[1],h['floor_z_measured']))
        self.assertEqual(r.metadata['chosen_segment_index'],0)
        self.assertFalse(r.metadata['extrapolated'])
        self.assertEqual(h,hint(),'hint must not be mutated')

    def test_does_not_choose_unselected_segments(self):
        p=LineSupportProducer(config())
        r=emit(p,indices=[2])  # caller chose only the transverse one; producer does not second-guess
        self.assertEqual(r.metadata['chosen_segment_index'],2)
        self.assertIsNone(emit(LineSupportProducer(config()),indices=[]).wire)

    def test_caller_gate_and_home_epoch(self):
        p=LineSupportProducer(config())
        self.assertEqual(emit(p,allowed=False).reason,'caller gate closed')
        self.assertEqual(emit(p,allowed=1).reason,'caller gate closed')
        self.assertIn('home epoch unavailable',emit(p,home=None).reason)
        self.assertIsNotNone(emit(p,home=7).wire)
        r=emit(p,hint(exposure=EXPOSURE+1,tof=TOF+1),home=8)
        self.assertIn('home epoch changed',r.reason)
        # Fault is permanent for this producer instance.
        self.assertIsNone(emit(p,hint(exposure=EXPOSURE+2,tof=TOF+2),home=7).wire)

    def test_current_unversioned_hint_produces_nothing(self):
        h=hint();del h['conditional_support']
        r=emit(LineSupportProducer(config()),h)
        self.assertIsNone(r.wire);self.assertIn('guideline_node extension required',r.reason)
        h=hint();h['conditional_support']['schema']='other'
        self.assertIsNone(emit(LineSupportProducer(config()),h).wire)

    def test_frame_and_stamp_consistency(self):
        self.assertIn('frame',emit(LineSupportProducer(config()),hint(frame='vio_global')).reason)
        h=hint();h['stamp']+=1e-3
        self.assertIn('does not match',emit(LineSupportProducer(config()),h).reason)
        for bad in [-1,1<<63,1.5,True,None]:
            h=hint();h['conditional_support']['tof_stamp_ns']=bad
            self.assertIn('invalid raw source',emit(LineSupportProducer(config()),h).reason,bad)

    def test_freshness_and_tof_skew_use_existing_bounds(self):
        p=LineSupportProducer(config())
        self.assertIn('later than native clock',emit(p,now=EXPOSURE-1).reason)
        self.assertIn('older than native input age',emit(p,now=TOF+500_000_001).reason)
        self.assertIsNotNone(emit(LineSupportProducer(config()),now=TOF+500_000_000).wire)
        self.assertIn('80ms',emit(p,hint(tof=EXPOSURE-80_000_001)).reason)
        self.assertIsNotNone(emit(LineSupportProducer(config()),hint(tof=EXPOSURE-80_000_000)).wire)
        self.assertIsNotNone(emit(LineSupportProducer(config()),hint(tof=EXPOSURE+20_000_000),now=EXPOSURE+20_000_000).wire)

    def test_order_only_advances_on_success_and_never_refreshes(self):
        p=LineSupportProducer(config())
        self.assertIsNotNone(emit(p).wire)
        self.assertIn('not newer',emit(p).reason)  # same exposure resent: refused, not re-stamped
        self.assertEqual(p.sequence,1)
        self.assertIsNone(emit(p,hint(exposure=EXPOSURE+5,tof=TOF+5),allowed=False).wire)
        r=emit(p,hint(exposure=EXPOSURE+5,tof=TOF+5))
        self.assertEqual(parse(r.wire)['sequence'],2)
        q=LineSupportProducer(config(),sequence_start=41)
        self.assertEqual(parse(emit(q).wire)['sequence'],42)

    def test_selected_segment_must_be_measured_floor_geometry(self):
        cases=[]
        h=hint();h['segments'][0]['source']='floor+ToF structural geometry';cases.append((h,'not a measured'))
        h=hint();h['segments'][0]['endpoints'][1][2]+=.01;cases.append((h,'not on the measured floor'))
        h=hint();h['segments'][0]['support_points']=7;cases.append((h,'below existing minimum'))
        h=hint();h['segments'][0]['support_points']=8.;cases.append((h,'below existing minimum'))
        h=hint();h['segments'][0]['endpoints'][1][:2]=h['segments'][0]['endpoints'][0][:2];cases.append((h,'no horizontal extent'))
        h=hint();h['segments'][0]['endpoints'][0][0]=math.nan;cases.append((h,'endpoints invalid'))
        h=hint();h['segments'][0]['endpoints']=h['segments'][0]['endpoints'][:1];cases.append((h,'endpoints invalid'))
        h=hint();h['floor_z_measured']=math.inf;cases.append((h,'floor_z'))
        for h,reason in cases:
            self.assertIn(reason,emit(LineSupportProducer(config()),h,indices=[0]).reason)
        for idx in ([3],[-1],[0,0],[True],['0']):
            self.assertIn('indices invalid',emit(LineSupportProducer(config()),indices=idx).reason,idx)

    def test_wire_roundtrips_exact_doubles(self):
        h=hint(floor=-0.012345678901234567)
        h['segments'][0]['endpoints']=[[0.1+0.2,1/3,h['floor_z_measured']],[2/3,-1e-7,h['floor_z_measured']]]
        d=parse(emit(LineSupportProducer(config()),h,indices=[0]).wire)
        self.assertEqual((d['ax'],d['ay'],d['bx'],d['by']),(0.1+0.2,1/3,2/3,-1e-7))
        self.assertEqual(d['floor_z'],-0.012345678901234567)

    def test_parse_rejects_malformed(self):
        good=emit(LineSupportProducer(config())).wire
        for bad in [good+' EXTRA',good.replace('EGO_LINE_SUPPORT_V1','EGO_LINE_SUPPORT_V2'),
                    good.replace(' 1 ',' 0 ',1),good.replace('world','wor ld'),'x'*1025]:
            with self.assertRaises(ValueError):parse(bad)

    def test_measured_source_literal_matches_detector(self):
        source=(Path(__file__).resolve().parent/'relative_guideline.py').read_text()
        self.assertIn(repr(lsp.MEASURED_SOURCE),source)


if __name__=='__main__':
    unittest.main()
