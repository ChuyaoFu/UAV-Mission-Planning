import unittest
from conditional_memory import policy


class ConditionalMemoryTests(unittest.TestCase):
    def test_legacy_policy_is_unchanged(self):
        c={'max_observation_age_s':.35}
        self.assertEqual(policy(c),dict(enabled=False,lifetime_s=.6,withdraw_before_expiry_s=0.,reaction_s=.6))

    def test_static_support_withdraws_before_the_last_snapshot_can_expire(self):
        c={'conditional_static_memory':True,'max_observation_age_s':.35,
           'max_acceleration_mps2':.5,'max_speed_mps':.3,
           'inferred_geometry_input_max_age_s':.5}
        p=policy(c)
        self.assertAlmostEqual(p['withdraw_before_expiry_s'],2.7)
        # Last advertised safe support survives the oldest accepted map packet,
        # reaction and full braking at the declared 1m/s speed bound.
        last_safe_age=p['lifetime_s']-p['withdraw_before_expiry_s']
        self.assertLessEqual(last_safe_age+.35+p['reaction_s']+1./.5,p['lifetime_s'])
        c['conditional_support_lifetime_s']=3.
        with self.assertRaises(ValueError):policy(c)


if __name__=='__main__':unittest.main()
