# Relative geometry and native estimator telemetry

The default flight estimator remains the existing OpenVINS build. The optional
`patches/openvins-joint-clone-telemetry.patch` adds read-only ROS telemetry to
that baseline. It changes neither filter priors nor measurements. Publication
requires a subscriber; the bounded payload contains at most ten clone pairs.
Build it in an isolated workspace before selecting a different flight binary.

`/ov_msckf/joint_clone_states` is a Float64MultiArray. Schema 1 starts with
`[1, pair_count, camera_to_imu_time_offset_seconds]`, followed by 160 doubles
per pair:

```
camera_time0, camera_time1,
position_G0[3], quaternion_JPL_GtoI0_xyzw[4],
position_G1[3], quaternion_JPL_GtoI1_xyzw[4],
joint_covariance_row_major[12,12]
```

The covariance order is `[theta_I0, position_G0, theta_I1, position_G1]`.
It is obtained from the actual retained-clone posterior, including cross terms.
Native camera timestamps are doubles: check their residual to original image
headers instead of treating them as newly measured integer exposure stamps.
The quaternion's numeric xyzw values equal the Hamilton IMU-to-world rotation;
do not invert them a second time.

`scripts/relative_clone_covariance.py` checks the schema and propagates the full
joint posterior through a fixed calibrated camera-to-IMU transform. Its result
is `T_camera0_from_camera1` with a 6x6 covariance in the chart
`[additive_translation_camera0, right_rotation_camera1]`. For a point
`y = R X_camera1 + t`, the corresponding Jacobian is `[I, -R skew(X_camera1)]`.
It is not an ordinary ROS pose covariance with both blocks in one frame.
Camera calibration and clock uncertainty remain separate inputs to consumers.

Shared global translation cancels algebraically through the relative Jacobian
and the actual cross-covariance. No guessed gauge variance is subtracted, and
independent position, orientation and lever-arm errors remain. Do not combine
this posterior silently with ToF-height-corrected navigation poses: the native
clone means precede that correction.

`replay_vio.py --save-joint-clones` writes a sequence of NumPy arrays, one
`numpy.save` record per native message, to `.joint-clones.npyseq`. Read the
completed file with repeated `numpy.load(..., allow_pickle=False)` until EOF.
The JSON records the executable/library hashes, effective configuration and
message count. Sensor replay is isolated on ROS domain 43 and never arms PX4.

The actual source24 baseline replay produced 2,444 joint messages. Its moving
visual-update coverage was 0.9661, with update-age p95 0.252 s and maximum
1.700 s. These are replay diagnostics, not physical accuracy qualification.
Independent checks against native PoseJPL updates covered 207 analytic
covariance cases and three 60,000-sample Monte Carlo cases. They validate the
coordinate convention and propagation, not the sensor noise model.

The separate `task_relative_contact.py` module provides same-exposure local
support diagnostics. Its local geometry is independent of arbitrary world
translation. Historical reuse still requires relative-motion uncertainty;
global map registration and unknown-space vetoes remain independent. It is
not connected to the native landing controller and does not authorize flight.
