#!/usr/bin/env python
"""ROS1/Python2-compatible observer. Records estimated outputs and process usage only."""
from __future__ import print_function
import argparse,csv,os,json,time,threading,signal,sys
import rospy
import psutil
from sensor_msgs.msg import PointCloud
from nav_msgs.msg import Odometry

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 if not os.path.isdir(a.output):os.makedirs(a.output)
 files={};writers={};lock=threading.Lock();stop=[False]
 def open_csv(name,columns):
  f=open(os.path.join(a.output,name+'.csv'),'w');files[name]=f;writers[name]=csv.writer(f);writers[name].writerow(columns)
 open_csv('frontend',['kind','stamp_ns','count','ids_shared_with_previous_message'])
 open_csv('odometry',['stamp_ns','x','y','z','qx','qy','qz','qw','vx','vy','vz','receipt_wall_s'])
 open_csv('resources',['wall_s','pid','executable','cpu_percent','rss_bytes','pss_bytes'])
 # Persist headers even when initialization never produces odometry, or node startup fails.
 for f in files.values():f.flush()
 last={}
 def features(msg,kind):
  ids=set(int(v+.5) for v in msg.channels[0].values) if msg.channels else set()
  with lock:
   writers['frontend'].writerow([kind,msg.header.stamp.to_nsec(),len(msg.points),len(ids.intersection(last.get(kind,set())))])
   last[kind]=ids
 def odom(msg):
  q=msg.pose.pose.orientation;t=msg.pose.pose.position;v=msg.twist.twist.linear
  with lock:writers['odometry'].writerow([msg.header.stamp.to_nsec(),t.x,t.y,t.z,q.x,q.y,q.z,q.w,v.x,v.y,v.z,time.time()])
 rospy.init_node('plvins_replay_observer',anonymous=True,disable_signals=True)
 subs=[rospy.Subscriber('/feature_tracker/feature',PointCloud,lambda m:features(m,'points'),queue_size=100),rospy.Subscriber('/linefeature_tracker/linefeature',PointCloud,lambda m:features(m,'lines'),queue_size=100),rospy.Subscriber('/plvins_estimator/odometry',Odometry,odom,queue_size=100)]
 def shutdown(*unused):stop[0]=True
 signal.signal(signal.SIGINT,shutdown);signal.signal(signal.SIGTERM,shutdown)
 cache={}
 while not stop[0] and not rospy.is_shutdown():
  for proc in psutil.process_iter():
   try:
    exe=os.path.basename(proc.exe())
    if exe not in ('feature_tracker','LineFeature_tracker','plvins_estimator'):continue
    if proc.pid not in cache:cache[proc.pid]=proc;proc.cpu_percent(None)
    else:proc=cache[proc.pid]
    pss=-1
    try:
     with open('/proc/%d/smaps_rollup'%proc.pid) as f:
      for line in f:
       if line.startswith('Pss:'):pss=int(line.split()[1])*1024;break
    except (IOError,ValueError):pass
    with lock:writers['resources'].writerow([time.time(),proc.pid,exe,proc.cpu_percent(None),proc.memory_info().rss,pss])
   except (psutil.NoSuchProcess,psutil.AccessDenied):continue
  with lock:
   for f in files.values():f.flush()
  time.sleep(.5)
 for sub in subs:sub.unregister()
 with lock:
  for f in files.values():f.close()
 rospy.signal_shutdown('replay observer stopped')
if __name__=='__main__':main()
