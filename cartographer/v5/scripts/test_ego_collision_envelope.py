import unittest,math
from dataclasses import FrozenInstanceError
from ego_collision_envelope import Envelope,EnvelopeSender,floor_ns

class EnvelopeTests(unittest.TestCase):
    def test_full_payload_and_immutable_record(self):
        s=EnvelopeSender('body-test');r=s.issue([.1,.2,.3],100,100000100,100)
        self.assertEqual(len(r.wire().split()),8)
        self.assertEqual(Envelope.parse(r.wire()),r)
        with self.assertRaises(FrozenInstanceError):r.valid_until_ns=999
        for field,value in [('epoch','other'),('sequence','2'),('issued','101'),('until','99999999'),('extra','.4')]:
            t=r.wire().split();i={'epoch':1,'sequence':2,'issued':3,'until':4,'extra':7}[field];t[i]=value
            self.assertFalse(s.accept_ack(' '.join(t),102,102))
        self.assertTrue(s.accept_ack(r.wire(),103,103))
        self.assertFalse(s.accept_ack(r.wire(),104,104))

    def test_malformed_native_contracts(self):
        for wire in ['','X body 1 0 100 .1 .1 .1','EGO_ENVELOPE_V1 body 0 0 100 .1 .1 .1',
                     'EGO_ENVELOPE_V1 body 1 0 100000001 .1 .1 .1',
                     'EGO_ENVELOPE_V1 body 1 0 100 nan .1 .1',
                     'EGO_ENVELOPE_V1 body 1 0 100 .1 .1 1.1',
                     'EGO_ENVELOPE_V1 b/ody 1 0 100 .1 .1 .1',
                     'EGO_ENVELOPE_V1 body +1 0 100 .1 .1 .1']:
            with self.subTest(wire=wire),self.assertRaises(ValueError):Envelope.parse(wire)

    def test_twenty_hz_updates_reuse_live_ack_without_sequence_deadlock(self):
        s=EnvelopeSender('body')
        for i in range(20):
            t=i*50000000;r=s.issue((.1,.2,.3),t,t+100000000,t)
            if i:self.assertTrue(s.ready((.1,.2,.3),t,t))
            self.assertTrue(s.accept_ack(r.wire(),t+1000000,t+1000000))
            self.assertTrue(s.ready((.1,.2,.3),t+2000000,t+2000000))
        self.assertEqual(s.counts['accepted'],20)

    def test_expired_or_delayed_ack_and_steady_clock_never_extend(self):
        s=EnvelopeSender('body');r=s.issue(.1,0,100000000,0)
        self.assertFalse(s.accept_ack(r.wire(),100000000,100000000))
        s=EnvelopeSender('body');r=s.issue(.1,0,100000000,0)
        self.assertTrue(s.accept_ack(r.wire(),1,1))
        self.assertFalse(s.ready(.1,1,100000000))
        self.assertEqual(s.counts['expired'],1)

    def test_later_unknown_native_record_must_also_cover_current_guard(self):
        s=EnvelopeSender('body');a=s.issue(.2,0,100000000,0);s.accept_ack(a.wire(),1,1)
        b=s.issue(.1,100,100000100,100)
        self.assertFalse(s.ready(.15,101,101)) # b could already be active
        self.assertTrue(s.ready(.1,102,102))
        self.assertTrue(s.accept_ack(b.wire(),103,103))
        self.assertFalse(s.ready(.15,104,104))

    def test_later_earlier_expiry_cannot_fall_back_to_old_ack(self):
        s=EnvelopeSender('body');a=s.issue(.2,0,100000000,0);s.accept_ack(a.wire(),1,1)
        s.issue(.2,100,1000,100)
        self.assertFalse(s.ready(.1,1000,1000))
        c=s.issue(.2,1001,100001001,1001);s.accept_ack(c.wire(),1002,1002)
        self.assertTrue(s.ready(.1,1003,1003))

    def test_clock_regression_latches_and_float_deadline_rounds_earlier(self):
        s=EnvelopeSender('body');r=s.issue(.1,100,100000100,100)
        self.assertFalse(s.accept_ack(r.wire(),99,101))
        self.assertFalse(s.ready(.1,102,102))
        value=math.nextafter(2.1,0.)
        n,d=value.as_integer_ratio()
        self.assertEqual(floor_ns(value),n*10**9//d)
        self.assertLessEqual(floor_ns(value)*d,n*10**9)

if __name__=='__main__':unittest.main()
