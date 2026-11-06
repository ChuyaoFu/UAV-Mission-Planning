# CAD-free navigation baseline

This is an experimental replacement for the scene-dependent v4 mission stack. It builds an online occupancy map, selects observed-free frontier viewpoints, and returns to its own initial odometry position. It does not yet implement the complete competition task sequence. Recorded results, failed counterexamples and resource measurements are documented in [VALIDATION.md](VALIDATION.md). The per-challenge competition plan is in [COURSE_PLAN.md](COURSE_PLAN.md).

Navigation receives rectified stereo images, IMU, a single downward range measurement, continuous visual-inertial odometry, and PX4 estimated local state. It never receives a simulator pose, rendered depth image, GPS, course CAD, absolute start coordinates, or preset route targets. Simulator spawn metadata and independent scoring live outside the navigation filesystem allowlist.

The current simulator uses the same approximate seven-inch aircraft, 1280×800 monochrome stereo, and a **20-degree downward rigid camera/IMU mounting profile**. A proposed forward mounting at `[0.22, 0, 0.055]` m clears the propeller sweep. This is a feasible simulation fixture, not an identified official mounting. Native 1280×800 images remain the VIO input. The CPU stereo proxy matches 640×400 image pairs, scales the focal length and left/right disparity tolerance together, and nearest-resamples measured/invalid values to the native ray grid. This does not fill missing depth or reproduce the OAK firmware. The actual OAK mounting is unknown. This profile is an explicit test configuration, not a verified hardware calibration. Hardware mode requires explicitly supplied camera calibration, full IMU-to-body translation/rotation, and self-occlusion geometry. The PX4 external-vision message reports the body origin, with angular lever velocity removed; PX4 EV position offsets must therefore be zero.

## Architecture

- OpenVINS provides continuous stereo-inertial odometry. A downward-ToF observer supports height; Cartographer consumes this odometry for mapping only. Neither local scan matches nor optimized loop-map poses feed back into the flight-control frame. Sparse-scan correction was rejected after the recorded estimator ablation increased position error.
- The pinned upstream [EGO-Swarm ROS 2 implementation](https://github.com/ZJU-FAST-Lab/ego-planner-swarm/tree/ros2_version) performs local trajectory optimization. The patch and upstream GPL license are included under `patches/` and `upstream/`.
- The occupancy mapper uses exposure-synchronized measured stereo depth. Missing, NaN, low-texture, or out-of-range depth stays unknown. Positive but insufficient occupancy evidence also remains blocked as unknown; occupied visualization requires probability above 0.8. Unknown and occupied cells block the complete vehicle clearance box, including vertical clearance.
- Frontier goals come from the connected observed-free component, exploration history, and a bounded information-gain score. A candidate outside the current forward view first causes an in-place measured-state turn. Translation is gated by current heading and complete-vehicle clearance; quality-gated panoramas from higher and lower operator-verified survey poses supply upper and lower clearance observations. Exploration first uses a narrow relative cruise band. If those viewpoints are exhausted, a height change may be selected only from the connected observed-free volume; an unseen ceiling does not justify a climb. A connected-component grid path supplies short local targets to EGO, with exact full-body segment checks. Home is the first measured body pose in an arbitrary local coordinate frame.
- A separate native-Gazebo evaluator records poses and contacts without publishing them into ROS. Both controller completion and independent evidence are required to accept a recorded episode.

An operator must verify a local **1 m × 1 m takeoff box**, up to **1.05 m above the resting body**, before starting. This local operating precondition is represented as state `-2`, separately from sensor-measured free space (`0`). It is not a reconstructed map. No other unknown space is cleared by this assumption. A forward camera cannot verify its own initial blind volume. An initial vertical sweep inside this operator-verified box observes the lower cruise volume from a higher measured viewpoint; it is a generic active-perception maneuver, not a field waypoint.

## Resource limits and Orin Nano 8 GB

The map contains 192,000 voxels (8 × 8 × 3 m at 0.1 m resolution). It is a resource window around the estimator's local origin, not a map of the field. Outside this window is blocked. Stereo processing is limited to 10 Hz, with a latest-frame worker; OpenVINS uses an 11-clone window and at most 50 persistent SLAM features. Failed EGO planning attempts are limited to 5 Hz. Visualization history and worker timing buffers are bounded. RViz, Gazebo, evaluation, and video encoding belong on the development/offboard computer.

Desktop OpenCV stereo matching is a development proxy. Hardware should use the OAK's measured, confidence-filtered and synchronized depth output with its actual rectified calibration. No artificial depth completion is assumed. The preinstalled JetPack version and power mode are unknown; the present implementation does not require CUDA or a large neural model.

`collect_target_profile.sh OUTPUT_DIRECTORY [SECONDS]` reads JetPack identification, power mode, and `tegrastats` without modifying the device. Desktop process CPU/PSS results are not Orin performance validation. A hardware deployment still requires camera/IMU timing and extrinsic calibration, real-driver tests, and sustained memory/latency/thermal measurements. Short unobstructed motions use upstream cubic B-splines only after each four-control-point hull is certified known-free for the complete vehicle and derivative control points certify speed/acceleration/jerk bounds. Measured velocity supplies an additional latency/braking-clearance check. ROS body odometry uses the standard child-frame twist, and the planner adapter explicitly rotates that velocity into the world frame. A braking hold latches a fixed position, waits for settling, and rejects the previous trajectory across goal changes. Tracking holds trigger replanning from measured state, so a clock-advanced trajectory is not treated as the actual vehicle state.

Automatic arming is currently restricted to the existing SITL container/domain.

## Development commands

The current integration uses the existing Ubuntu ROS 2 Jazzy simulation container, `/work/ros_ws`, OpenVINS under v2, and the patched Cartographer workspace under v3. This is not yet a standalone hardware installer. Use the ROS distribution compatible with the actual JetPack host; do not upgrade the preinstalled stack based on an assumption.

```bash
bash /work/cartographer/v5/scripts/bootstrap_ego.sh
bash /work/cartographer/v5/scripts/start.sh source
bash /work/cartographer/v5/scripts/open_windows.sh
python3 /work/cartographer/v5/scripts/run_trial.py UNIQUE_LABEL --seconds 150
# Only after confirmed disarm:
bash /work/cartographer/v5/scripts/stop.sh
```

The trial runner requires both visible flight/SLAM windows, a recording, GPS fusion disabled, verified native evaluation subscriptions, and an input/filesystem audit before arming. Results are saved under `/work/cartographer/v5/results/UNIQUE_LABEL/`; sensor bags are under `data/`; screen and first-person videos are under `results/UNIQUE_LABEL/videos/`.

Independent unknown-room tests are generated with `generate_unknown_scene.py`. Their world and start metadata must remain in the evaluation/simulator directories; the navigation configuration must not be changed between seeds. Furnished-room success does not establish observability on textureless walls.

## Checks

```bash
python3 -m unittest discover -s cartographer/v5/scripts -p 'test_*.py'
ROS_DOMAIN_ID=43 python3 cartographer/v5/scripts/verify_mapper_runtime.py
ROS_DOMAIN_ID=43 python3 cartographer/v5/scripts/verify_planner_runtime.py
ROS_DOMAIN_ID=43 python3 cartographer/v5/scripts/verify_cancel_runtime.py
ROS_DOMAIN_ID=43 python3 cartographer/v5/scripts/verify_brake_runtime.py
```

The runtime tests use synthetic measurements in isolated domain 43 and never arm a vehicle. Domain 42 is reserved for the actual flight test. The file isolation probe and ROS input audit cover separate parts of the no-prior/no-truth contract. They do not establish complete application security.

## Current limits

The present return baseline assumes a static environment. It may translate backward through already observed free space while retaining a tracked camera heading. The forward-only camera cannot detect newly introduced obstacles behind it. This limitation must be resolved or explicitly accepted for a real operating envelope.

Plain walls and missing stereo can prevent safe expansion of the map. In that case the vehicle must hold, reacquire observable views, return, or land; an unknown volume is not converted into free space for completion. The saved local home position does not by itself constitute a verified visual loop closure. Long-distance global relocalization, larger-map management, complete competition semantics, and deployment on the actual Orin remain separate work.

## Optional stereo-odometry benchmark

The RTAB-Map ROS 2 stereo/IMU adapter and isolated replay runner are included as an independent comparison, not enabled flight inputs. The installed development package is `ros-jazzy-rtabmap-odom` 0.23.13. On the first recorded furnished-room episode, the tested settings lost tracking in 98 of 625 diagnostic frames and produced 0.480 m fixed-initial-alignment position RMSE (0.228 m in XY). They were therefore rejected as a flight-control replacement. These results are specific to this configuration and recording, not a general ranking of the estimators.

## Offline completion experiment

PromptDA Small is benchmarked separately on monochrome-left images and measured stereo anchors. Its output is saved as inferred depth and is not a navigation input. Zero-filled and nearest-interpolated prompts are labeled separately. The prediction process uses the same filesystem isolation and cannot access independent geometry/poses. The current test does not justify treating model predictions as observed free space or claiming Orin timing.
