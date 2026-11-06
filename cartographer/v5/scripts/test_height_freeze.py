"""Raised surfaces cannot change the VIO gauge after semantic height freeze."""
import unittest
from types import SimpleNamespace as S
from height_odometry import HeightObserver


def odometry(t, z):
    return S(header=S(stamp=S(sec=int(t), nanosec=round((t-int(t))*1e9))),
             pose=S(pose=S(position=S(z=z), orientation=S(x=0., y=0., z=0., w=1.))))


class HeightFreezeTest(unittest.TestCase):
    def test_platform_range_preserves_correction_and_continuous_history(self):
        observer = HeightObserver()
        observer.correction = .037
        observer.odometry(odometry(1., .2))
        observer.pending = (1., .1)
        observer.freeze()
        self.assertIsNone(observer.pending)
        for t, z, surface_range in [(1.1, .5, .4), (1.2, .7, .08), (1.3, .71, .07)]:
            observer.pending = (t, surface_range)
            self.assertEqual(observer.odometry(odometry(t, z)), .037)
        observer.freeze()
        self.assertTrue(observer.frozen)
        self.assertEqual(observer.history[-1][1], .71)
        self.assertEqual(observer.accepted, 0)


if __name__ == '__main__':
    unittest.main()
