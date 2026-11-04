import unittest
from types import SimpleNamespace as S
from height_odometry import HeightObserver

def odom(t,z):return S(header=S(stamp=S(sec=int(t),nanosec=round((t-int(t))*1e9))),pose=S(pose=S(position=S(z=z),orientation=S(x=0.,y=0.,z=0.,w=1.))))
def ranging(t,z):return S(header=S(stamp=S(sec=int(t),nanosec=round((t-int(t))*1e9))),range=z-.08,min_range=.05,max_range=4.)
class Height(unittest.TestCase):
 def test_drift_and_once(self):
  f=HeightObserver()
  for i in range(101):
   t=i*.05;f.ranging(ranging(t,.8));f.odometry(odom(t,i*.001))
  self.assertLess(abs(f.correction+.1),.01);old=f.correction
  for j in range(1,10):f.odometry(odom(5+j*.001,.1))
  self.assertEqual(f.correction,old)
 def test_obstacle_top(self):
  f=HeightObserver();f.ranging(ranging(0,.8));f.odometry(odom(0,0));f.ranging(ranging(.1,.5));f.odometry(odom(.1,0));self.assertEqual(f.correction,0);self.assertEqual(f.rejected,1)

class Reacquisition(unittest.TestCase):
 def test_only_wide_observed_floor_can_reacquire(self):
  import numpy as np
  def cloud(t,span):
   a,b=np.meshgrid(np.linspace(-span,span,15),np.linspace(-span,span,15));p=np.c_[a.ravel(),b.ravel(),np.full(a.size,-.8)].astype('<f4')
   return S(header=S(stamp=S(sec=int(t),nanosec=round((t-int(t))*1e9))),point_step=12,data=p.tobytes())
  for span,should_correct in [(.1,False),(.8,True)]:
   f=HeightObserver();f.ranging(ranging(0,.8));f.odometry(odom(0,0))
   for i in range(1,81):
    t=i*.05;f.ranging(ranging(t,.8));f.odometry(odom(t,.18));f.cloud(cloud(t,span))
   if should_correct:self.assertLess(abs(f.correction+.18),.02);self.assertGreater(f.reacquired,0)
   else:self.assertEqual(f.correction,0.)

if __name__=='__main__':unittest.main()
