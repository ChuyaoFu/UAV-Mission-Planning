#!/usr/bin/env python3
"""Launch-time ROBOT calibration selection; no state, world or flight inputs."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def select(text, robot, enabled):
    text = re.sub(r'^(retain_home_clone|home_relative_imu_from_body):.*\n?', '', text,
                  flags=re.MULTILINE)
    if not enabled:
        return text.rstrip() + '\nretain_home_clone: false\n'
    if not re.search(r'^calib_cam_timeoffset:\s*false(?:\s|$)', text, re.MULTILINE):
        raise ValueError('Retained reference requires explicitly fixed camera/IMU clock calibration')
    p = np.asarray(robot['imu_in_body_m'], dtype=float)
    q = np.asarray(robot['imu_in_body_quaternion_xyzw'], dtype=float)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all() or abs(np.linalg.norm(q)-1.) > 1e-6:
        raise ValueError('Complete finite, unit ROBOT IMU-to-BODY calibration required')
    inverse = Rotation.from_quat(q).inv()
    transform = np.r_[inverse.apply(-p), inverse.as_quat()]
    return (text.rstrip() + '\nretain_home_clone: true\nhome_relative_imu_from_body: ['
            + ', '.join(format(v, '.17e') for v in transform) + ']\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('config', type=Path)
    p.add_argument('robot', type=Path)
    p.add_argument('--enable-home-relative', action='store_true')
    a = p.parse_args()
    result = select(a.config.read_text(), json.loads(a.robot.read_text()), a.enable_home_relative)
    temporary = a.config.with_suffix(a.config.suffix + '.reference.tmp')
    temporary.write_text(result)
    temporary.replace(a.config)


if __name__ == '__main__': main()
