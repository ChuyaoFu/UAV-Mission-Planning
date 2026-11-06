#!/usr/bin/python3
"""Simulator-only start pose. This metadata is denied to navigation processes."""
import json,math
from pathlib import Path
p=Path('/work/cartographer/v5/evaluation/scenario.json');c=json.loads(p.read_text());x,y,z=c['spawn_xyz'];yaw=c['spawn_yaw_rad']
print('sdf_filename: "/work/cartographer/v5/config/model.sdf", name: "x500_vio_0", pose: {position: {x: %s, y: %s, z: %s}, orientation: {w: %s, z: %s}}' % (x,y,z,math.cos(yaw/2),math.sin(yaw/2)))
