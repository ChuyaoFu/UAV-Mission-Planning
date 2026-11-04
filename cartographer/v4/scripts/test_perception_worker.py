import unittest,threading,time
from perception_worker import LatestWorker
class Worker(unittest.TestCase):
 def test_latest_frame_coalescing(self):
  entered=threading.Event();release=threading.Event();seen=[]
  def handler(key,value):
   seen.append(value)
   if value==0:entered.set();release.wait(1)
   return [(key,value,None,0)]
  w=LatestWorker(handler);w.submit('image',0);self.assertTrue(entered.wait(1))
  for i in range(1,1001):w.submit('image',i)
  self.assertEqual(len(w.jobs),1);release.set();deadline=time.monotonic()+1
  while len(seen)<2 and time.monotonic()<deadline:time.sleep(.005)
  w.close();self.assertEqual(seen,[0,1000]);self.assertIsNone(w.error)
if __name__=='__main__':unittest.main()
