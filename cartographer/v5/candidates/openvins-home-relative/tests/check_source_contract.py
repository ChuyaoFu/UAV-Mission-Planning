#!/usr/bin/env python3
from pathlib import Path
import sys,re
base,src=map(Path,sys.argv[1:]);old=(base/'ov_msckf/src/state/StateHelper.cpp').read_text();new=(src/'ov_msckf/src/state/StateHelper.cpp').read_text()
# All numerical EKF, augmentation and removal machinery outside the two explicit guards remains byte-identical.
retain='''
  // Keep only this real correlated variable. Callers STILL erase its recent
  // clone map entry, preserving feature cleanup/reanchoring/window behavior.
  // Covers ordinary/init catch-up and the explicit-ZUPT deletion path.
  if (state->_options.retain_home_clone && state->_home_pose == marg) return;
'''
capture='''
  // Register AFTER the ordinary clone and optional timing covariance update.
  // No extra clone, measurement, prior or covariance write is performed.
  if (state->_options.retain_home_clone && !state->_home_pose) {
    state->_home_pose = pose;
    state->_home_camera_time = state->_timestamp;
  }
'''
assert new.replace(retain,'',1).replace(capture,'\n',1)==old
for rel in ['ov_msckf/src/core/VioManager.cpp','ov_msckf/src/core/VioManagerHelper.cpp','ov_msckf/src/update/UpdaterSLAM.cpp','ov_msckf/src/update/UpdaterMSCKF.cpp','ov_msckf/src/update/UpdaterZeroVelocity.cpp','ov_msckf/src/state/Propagator.cpp','ov_core/src/track/TrackKLT.cpp','ov_init/src/init/StaticInitializer.cpp']:
 if (base/rel).exists():assert (base/rel).read_bytes()==(src/rel).read_bytes(),rel
options=(src/'ov_msckf/src/state/StateOptions.h').read_text();assert 'bool retain_home_clone = false;' in options and 'int max_clone_size = 11;' in options
pub=(src/'ov_msckf/src/ros/ROS2Visualizer.cpp').read_text();a=pub.index('  // Same CURRENT IMU posterior');b=pub.index('  // Read-only joint posterior',a)
assert pub.count('home_yaw_relative_body_pose(')==1
assert 'message.header=poseIinM.header' in pub[a:b]
assert 'state->_imu->Rot().transpose()' in pub[a:b] and 'state->_imu->pos()' in pub[a:b]
assert '_clones_IMU' not in pub[a:b]
assert 'home_relative_body_pose' in pub and 'home_relative_body_reference' in pub
assert 'std::exception' in pub[a:b]
assert (src/'ov_msckf/src/state/HomeRelativePose.h').read_text()==Path('outputs/home-relative-msckf-v1/tests/home_relative_pose.h').read_text()
assert (src/'ov_msckf/src/state/HomeYawRelativePose.h').read_text()==Path('outputs/home-relative-msckf-v1/tests/home_yaw_relative_pose.h').read_text().replace('"home_relative_pose.h"','"HomeRelativePose.h"')
print('PASS source contract: only capture+retention guard in native StateHelper; all EKF math and recent feature/anchor/SLAM/MSCKF/ZUPT/propagation/frontend unchanged; defaultoff; recent11; same-update IMU pose; BODY yaw pose-only chart; no extra old clone in recent map')
