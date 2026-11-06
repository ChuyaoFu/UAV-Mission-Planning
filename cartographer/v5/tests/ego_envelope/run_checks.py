#!/usr/bin/env python3
"""Run candidate offline checks. Args: PATCHED_UPSTREAM AUDITED_REPOSITORY.

On this Mac, set SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk.
Ubuntu needs only a C++17 compiler and Python3, with no ROS launch required.
"""
from pathlib import Path
import subprocess
import sys
import tempfile

tests = Path(__file__).resolve().parent
root, repo = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
with tempfile.TemporaryDirectory() as temp:
    exe = str(Path(temp)/'test_collision_envelope')
    subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                    '-I', str(root/'src/planner/plan_env/include'),
                    str(tests/'test_collision_envelope.cpp'), '-o', exe], check=True)
    subprocess.run([exe], check=True)
for script in ['test_native_hooks.py', 'test_native_depth_timeout.py']:
    subprocess.run([sys.executable, str(tests/script), str(root)], check=True)
launch = repo/'cartographer/v5/scripts/ego_launch.py'
subprocess.run([sys.executable, str(tests/'test_launch_opt_in.py'), str(launch), str(repo)], check=True)
print('PASS: candidate offline suite; no ROS processes or flight started')
