"""Single bounded worker: expensive perception cannot queue unbounded frames.
No ROS or simulator dependencies. The control thread applies returned results.
"""
import threading,time
from collections import deque
class LatestWorker:
 def __init__(self,handler):
  self.handler=handler;self.jobs={};self.results=deque(maxlen=32);self.condition=threading.Condition();self.closed=False;self.superseded=0;self.compute_ms=deque(maxlen=1024);self.error=None;self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
 def submit(self,key,value):
  with self.condition:
   if key in self.jobs:self.superseded+=1
   self.jobs[key]=(time.monotonic(),value);self.condition.notify()
 def run(self):
  while True:
   with self.condition:
    while not self.jobs and not self.closed:self.condition.wait()
    if self.closed:return
    key=min(self.jobs,key=lambda k:self.jobs[k][0]);_,value=self.jobs.pop(key)
   start=time.perf_counter()
   try:result=self.handler(key,value)
   except Exception as error:self.error=repr(error);result=[]
   with self.condition:self.results.extend(result);self.compute_ms.append((time.perf_counter()-start)*1000)
 def take(self):
  with self.condition:result=list(self.results);self.results.clear();return result
 def close(self):
  with self.condition:self.closed=True;self.condition.notify_all()
  self.thread.join(timeout=2)
