#!/usr/bin/python3
"""Observe vehicle contacts; ground contact is expected in this smoke fixture."""
import argparse
import json
import time
from pathlib import Path
import xml.etree.ElementTree as ET
from gz.transport13 import Node
from gz.msgs10.contacts_pb2 import Contacts
ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=float,default=55);args=ap.parse_args()
root=Path('/work');node=Node();counts={'messages':0,'floor_contacts':0,'nonfloor_contacts':0};events=[];times=[];pairs={}
topics=[]
for link in ET.parse(root/'sim/models/x500_base_nogps/model.sdf').findall('.//link'):
    for s in link.findall('sensor'):
        if s.get('type')=='contact':
            topics.append(f'/world/visual_nav/model/x500_vio_0/link/{link.get("name")}/sensor/{s.get("name")}/contact')
def cb(msg):
    counts['messages']+=1
    stamp=msg.header.stamp.sec+msg.header.stamp.nsec*1e-9
    if not times:times.append(stamp)
    if len(times)==1:times.append(stamp)
    else:times[1]=stamp
    for c in msg.contact:
        pair=[c.collision1.name,c.collision2.name]
        key=tuple(sorted(pair))
        if key not in pairs:
            pairs[key]={'collisions':list(key),'count':0,'first_stamp_s':stamp,'last_stamp_s':stamp}
        pairs[key]['count']+=1
        pairs[key]['last_stamp_s']=stamp
        if any('floor_20_x_20_ft' in n for n in pair):counts['floor_contacts']+=1
        else:
            counts['nonfloor_contacts']+=1
            if len(events)<20:
                events.append({'stamp':msg.header.stamp.sec+msg.header.stamp.nsec*1e-9,'collisions':pair})
for topic in topics:node.subscribe(Contacts,topic,cb)
time.sleep(args.seconds)
r={**counts,'message_time_interval_s':times,'topics':topics,'collision_pairs':list(pairs.values()),'nonfloor_events':events,'passed':counts['messages']>0 and counts['floor_contacts']>0 and counts['nonfloor_contacts']==0}
(root/'results/vehicle-contacts.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if not r['passed']:raise SystemExit('Contact monitoring failed or a non-floor collision occurred')
