# Optional retained estimated home clone + gravity-aligned BODY pose

SOURCE/AUDIT CANDIDATE ONLY. No PX4 connection, root fusion/sensor_frames/map/controller edits, prior/noise/gate changes or automatic promotion. PRIVATE fixture is root-owned; no CAD/fixture geometry is used here. Tested math/StateHelper evidence does not establish physical error bounds.

## Apply to the isolated native source

Upstream pin69488123ed9362dd44b6f28e7f4680abbff1442b, after main v2 openvins.patch and root openvins-joint-clone-telemetry.patch. Optional GFTT can coexist because its files do not overlap this patch's edited regions, but keep the v2 baseline for equivalence testing. Primary patch: patches/0001-optional-retained-home-yaw-pose.patch. Seven files. `python3 tools/install_patch.py SOURCE --check` verifies touched-file baseline hashes and clean application; omit --check to apply. It does not write configs, build, deploy or connect anything. Unknown edits are rejected, not overwritten. Existing default outputs remain unchanged.

Feature defaults: retain_home_clone:false. When enabled, the SAME first real estimated clone is retained as a six-dimensional State variable with its original full covariance/cross terms. It is removed from the recent clone map at the normal time, so max_clones11 and all feature cleanup/SLAM reanchoring/window logic stay unchanged. At most one extra6D variable; no unbounded clone history. Home mean/P continue ordinary EKF updates. No added home constraint, zero prior, process noise reduction or covariance clipping.

Use an explicit fixed body calibration. tools/make_body_transform.py reads actual robot.json T_BODY_IMU and writes an isolated fragment only, with T_IMU_BODY=inverse(T_BODY_IMU). It refuses overwrite. Example config/source24-home-fragment.yaml was derived from supplied actual source24 SITL robot.json (20deg rig), hashes recorded. Do not reuse across mounts. The supplied robot profile remains a SITL approximation, not measured hardware qualification. Add the generated keys exactly once to an isolated estimator YAML:

```
retain_home_clone: true
home_relative_imu_from_body: [tx, ty, tz, qx, qy, qz, qw]
```

Translation meters, quaternion numeric Hamilton, camera time calibration fixed. Missing/nonfinite/nonunit transform or online time calibration rejects this optional mode. Original configuration/feature budgets remain unchanged. T_IMU_BODY is not T_BODY_IMU, and neither is the camera-to-IMU transform.

## Explicit output interface

`/ov_msckf/home_relative_body_pose`: geometry_msgs/PoseWithCovarianceStamped. BODY position and BODY orientation in a gravity-aligned HOME-YAW frame whose origin is the initial estimated BODY position. R_W_H=Rz(yaw(R_W_Ihome*R_I_BODY)), never the pitched IMU axes or full initial BODY rotation. World Z remains gravity-up; initial BODY need not be level. Covariance row/column order[x_H,y_H,z_H,thetaX_H,thetaY_H,thetaZ_H], with LEFT small-angle orientation in HOME. Full6x6, including cross blocks. The old mixed right-current-frame covariance is not published as ROS pose covariance. MATH_AND_LIFECYCLE.md provides analytic Jacobian and perturbation proof.

This is POSE ONLY. No odometry twist or zero twist covariance is fabricated. Root explicitly owns a future adapter joining real native twist with timestamp/age checks and converting BODY-origin/HOME-left covariance into the current IMU-origin/local-angle interface. Do not apply the existing IMU-to-BODY lever subtraction again to this BODY pose, or feed its standard HOME angular covariance straight into a native local-angle input. No root adapter/frames/fusion code changed here.

Header is copied exactly from existing `/ov_msckf/poseimu`: current actual FILTER UPDATE epoch `state->_timestamp` in camera seconds plus fixed camera-to-IMU offset, converted by original ROSVisualizerHelper round-to-ns routine. It is NOT fast IMU-propagated time, wall receipt time or last clone time. Current IMU pose/full joint covariance is read, including accepted ZUPT epochs with no new clone. Pose publication is not evidence of a visual update; retain root visual health/gap gates. No interpolation or restamping.

`/ov_msckf/home_relative_body_reference`: Float64MultiArray, emitted when both it and primary pose have subscribers. Data12: [schema1, home_camera_s, home_imu_s, current_camera_s, current_imu_s, T_IMU_BODY tx,ty,tz,qx,qy,qz,qw]. Frame id `vio_home_yaw_bodyorigin_<rounded_native_anchor_imu_ns>` identifies the native reference epoch, not a recovered original integer sensor timestamp. Original anchor means are estimated/refined, not frozen numeric truth; don't join sessions by frame id alone.

Time association rule: preserve the actual camera integer header and compare it against reported native current_camera_s, recording residual `Decimal.from_float(t_camera)*1e9 - integer_camera_ns`. In this SITL time domain require <=1ns (source24 native clone residuals were far smaller). For large real epoch seconds, double seconds can lose submicrosecond bits: explicitly account for a precision-only bound such as max(1ns,2*ulp(t_camera)*1e9), record that residual and reject outside it. Do not use a wide age tolerance to pretend different images have the same exposure identity. Independently require pose header to equal original helper's conversion of current_imu_s, and current_imu_s=current_camera_s+fixed offset. Same-source camera CSV was not supplied for this new producer, so no end-to-end integer-header association test is claimed. Native source24 joint clock audit is separately frozen. Do not interpret ROS receipt time minus sim header as a latency unless both use the same clock.

Both new publishers are absent by default. When enabled without pose subscribers, retention still maintains the requested joint posterior, but no pose-covariance math/message is computed. Invalid new telemetry is skipped/logged; original state/output is untouched. Single serialized filter-update/publication context is the existing ROS2 assumption; no extra mutating thread/subscription is added. Home is first ESTIMATED clone, not guaranteed physical launch/dock truth: operational binding to the operator's ground-home observation is a downstream task.

## Native evidence and remaining root checks

Actual StateHelper EKFPropagation/EKFUpdate/get_marginal_covariance/marginalize/clone/initialize_invertible/augment_clone/marginalize_old_clone bodies compiled under ASan/UBSan with original IMU/PoseJPL/Vec and a State registry shim.240cycles+14initializations+11deletions: active mean and full active marginal covariance differences both0; recent11; home only+6 after eviction, updated home mean/nonzero cross, unchanged FEJ/PSD. This is not a full State/VioManager/ROS2 build.

200native PoseJPL central-difference checks for old pure relative diagnostic and200 for NEW yaw output pass, including nonzero fixed lever, arbitrary home roll/pitch and pitchedIMU20deg. New yaw J maxabs6.02e-9; at perfectly correlated initial home/current, position cancels while actual roll/pitch angular covariance trace0.00503635rad² remains. No output P clipping. Source contract proves only capture+retention guard change StateHelper; all numerical EKF bodies, frontend, VioManager, MSCKF, SLAM, ZUPT, Propagator and initializer are byte-unchanged; ROS2 patch adds separate optional publication only.

Root must clean-build ROS2, replay identical sensor-only bags with fixed options/serialization and compare DEFAULT-OFF and enabled ORIGINAL outputs/posterior diagnostics against baseline. Require active means/covariance marginal equality to roundoff, same accepted visual updates/clone epochs/features and no unexplained data drops; independently grade physical trajectory and CPU/PSS. Reject instead of changing priors/noise/gates if equivalence fails. Then validate BODY origin, gravity-up, integer time residuals, covariance chart and adapter separately. No flight or navigation admission is claimed by this package.
