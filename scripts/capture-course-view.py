#!/usr/bin/python3
"""Render a north-up inspection camera; it is not an autonomy sensor."""
from pathlib import Path
import time
import cv2
import numpy as np
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.entity_pb2 import Entity
from gz.msgs10.boolean_pb2 import Boolean
node=Node();frames=[]
def cb(m):
    if not frames:
        a=np.frombuffer(m.data,dtype=np.uint8).reshape(m.height,m.width,3)
        frames.append(a.copy())
node.subscribe(Image,'/inspection/overhead',cb)
sdf='''<sdf version="1.9"><model name="inspection_camera"><static>true</static><pose>0 0 6.2 0 1.570796326795 1.570796326795</pose><link name="body"><sensor name="overhead" type="camera"><topic>/inspection/overhead</topic><always_on>true</always_on><update_rate>1</update_rate><camera><horizontal_fov>1.0</horizontal_fov><image><width>1024</width><height>1024</height><format>R8G8B8</format></image><clip><near>.05</near><far>50</far></clip></camera></sensor></link></model></sdf>'''
ok,res=node.request('/world/visual_nav/create',EntityFactory(name='inspection_camera',sdf=sdf),EntityFactory,Boolean,5000)
assert ok and res.data
try:
    for _ in range(150):
        if frames:break
        time.sleep(.1)
    if not frames:raise RuntimeError('Inspection camera produced no image')
    cv2.imwrite('/work/results/course-overhead.png',cv2.cvtColor(frames[0],cv2.COLOR_RGB2BGR))
    print('Saved actual Gazebo overhead render: /work/results/course-overhead.png')
finally:node.request('/world/visual_nav/remove',Entity(name='inspection_camera',type=Entity.MODEL),Entity,Boolean,5000)
