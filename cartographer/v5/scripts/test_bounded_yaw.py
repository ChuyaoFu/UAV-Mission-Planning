import math,unittest
from bounded_yaw import YawRequestLimiter,wrap


class YawRequests(unittest.TestCase):
    def test_half_turn_cannot_be_sent_as_one_step(self):
        limiter=YawRequestLimiter(math.radians(20),math.radians(10))
        self.assertEqual(limiter.update(math.pi,0.,1.),0.)
        value=limiter.update(math.pi,0.,1.05)
        self.assertAlmostEqual(value,math.radians(1),places=10)

    def test_wrap_uses_short_path(self):
        limiter=YawRequestLimiter(.4,.2);a=math.pi-.01
        limiter.update(-math.pi+.01,a,1.)
        value=limiter.update(-math.pi+.01,a,1.1)
        self.assertGreater(wrap(value-a),0.)
        self.assertAlmostEqual(wrap(value-(-math.pi+.01)),0.)

    def test_clock_pause_or_long_gap_cannot_accumulate_a_large_turn(self):
        limiter=YawRequestLimiter(.4,.2);limiter.update(2.,0.,1.)
        self.assertEqual(limiter.update(2.,0.,1.),0.)
        self.assertLessEqual(limiter.update(2.,0.,100.),.080000001)
        self.assertEqual(limiter.update(2.,.3,.5),.3)

    def test_native_heading_jump_cannot_keep_old_large_error(self):
        limiter=YawRequestLimiter(.4,.2);limiter.update(0.,0.,1.)
        value=limiter.update(0.,1.5,1.1)
        self.assertLessEqual(abs(wrap(value-1.5)),.20000001)

    def test_invalid_attitude_is_rejected(self):
        with self.assertRaises(ValueError):YawRequestLimiter(.4,.2).update(0.,math.nan,1.)


if __name__=='__main__':unittest.main()
