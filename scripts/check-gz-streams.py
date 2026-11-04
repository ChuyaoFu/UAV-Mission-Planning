#!/usr/bin/python3
import json
import time
from collections import defaultdict
from pathlib import Path
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.imu_pb2 import IMU
node=Node();counts=defaultdict(int);stamps=defaultdict(list)
def cb(k,m):
    counts[k]+=1
    stamps[k].append(m.header.stamp.sec+m.header.stamp.nsec*1e-9)
for topic,typ in [('/vio/left/image',Image),('/vio/right/image',Image),('/vio/depth/image',Image),('/vio/imu',IMU)]:
    node.subscribe(typ,topic,lambda m,k=topic:cb(k,m))
time.sleep(8)
r={'counts':dict(counts),'rate_sim_hz':{k:(len(s)-1)/(s[-1]-s[0]) for k,s in stamps.items() if len(s)>1}}
left=stamps['/vio/left/image'];right=set(stamps['/vio/right/image'])
r['stereo_exact_fraction']=sum(x in right for x in left)/max(1,len(left))
Path('/work/results/gz-streams.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
