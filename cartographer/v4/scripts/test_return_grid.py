"""Check the shipped return grid preserves obstacles and outbound policy."""
import unittest
from pathlib import Path
import numpy as np
from cad_planner import load
ROOT=Path(__file__).resolve().parents[1]
class ReturnGrid(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.outbound=load(ROOT/'config/cad-safe-grid.npz');cls.return_grid=load(ROOT/'config/cad-return-grid.npz')
 def test_wall_is_retained(self):
  self.assertFalse(self.return_grid.segment([-2.04216,-.79248,.8],[-2.5146,1.2644,.8]))
  self.assertTrue(self.return_grid.segment([-2.04216,-.79248,2.15],[-2.5146,1.2644,2.15]))
 def test_caps_are_separate(self):
  self.assertFalse(self.outbound.segment([-2.04216,-.79248,.8],[-2.04216,-.79248,2.15]))
  self.assertTrue(self.return_grid.segment([-2.04216,-.79248,.8],[-2.04216,-.79248,2.15]))
 def test_no_obstacles_removed_below_outbound_cap(self):
  z=self.outbound.idx([0,0,1.45])[2]+1
  np.testing.assert_array_equal(self.outbound.occupied[:,:,:z],self.return_grid.occupied[:,:,:z])
 def test_descent_standoff_and_arena_boundary(self):
  self.assertTrue(self.return_grid.segment([-2.5146,1.2644,2.15],[-2.5146,1.2644,.8]))
  self.assertFalse(self.return_grid.segment([-2.5146,1.2644,2.15],[-3.2,1.2644,2.15]))
if __name__=='__main__':unittest.main()
