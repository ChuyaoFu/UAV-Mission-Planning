#!/usr/bin/env python3
"""Exercise candidate launch generation with ROS launch classes stubbed.

Usage: python3 test_launch_opt_in.py CANDIDATE_EGO_LAUNCH AUDITED_REPOSITORY
Only opt-in wiring/remaps are under test; no ROS processes are started.
"""
from pathlib import Path
import importlib.util
import os
import sys
from types import ModuleType


class Node:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class LaunchDescription:
    def __init__(self, entities):
        self.entities = entities


launch = ModuleType('launch')
launch.LaunchDescription = LaunchDescription
launch_ros = ModuleType('launch_ros')
actions = ModuleType('launch_ros.actions')
actions.Node = Node
camera = ModuleType('camera_geometry')
camera.load_rig = lambda: ([1., 1., 1., 1.], None, None, None)
memory = ModuleType('conditional_memory')
memory.policy = lambda c: {'lifetime_s': .6, 'withdraw_before_expiry_s': 0., 'enabled': False}
sys.modules.update({'launch': launch, 'launch_ros': launch_ros, 'launch_ros.actions': actions,
                    'camera_geometry': camera, 'conditional_memory': memory, 'yaml': ModuleType('yaml')})
spec = importlib.util.spec_from_file_location('candidate_launch', Path(sys.argv[1]))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.root = Path(sys.argv[2]) / 'cartographer/v5'
original = os.environ.get('EGO_REQUIRE_COLLISION_ENVELOPE')
try:
    for value, expected in [(None, False), ('0', False), ('1', True)]:
        if value is None:
            os.environ.pop('EGO_REQUIRE_COLLISION_ENVELOPE', None)
        else:
            os.environ['EGO_REQUIRE_COLLISION_ENVELOPE'] = value
        planner, server = module.generate_launch_description().entities
        p = planner.parameters[0]
        assert p['grid_map/require_collision_envelope'] is expected
        assert p['grid_map/strict_unknown'] is True
        assert p['grid_map/obstacles_inflation'] == .32 and p['grid_map/vertical_inflation'] == .16
        assert ('grid_map/collision_envelope', '/nav/ego_collision_envelope') in planner.remappings
        assert ('grid_map/collision_envelope_ack', '/nav/ego_collision_envelope_ack') in planner.remappings
        assert server.parameters[0]['traj_server/time_forward'] == .4
finally:
    if original is None:
        os.environ.pop('EGO_REQUIRE_COLLISION_ENVELOPE', None)
    else:
        os.environ['EGO_REQUIRE_COLLISION_ENVELOPE'] = original
print('PASS: default off, explicit opt-in, topics, unchanged base envelope and yaw lookahead')
