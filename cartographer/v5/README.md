# CAD-free navigation baseline

This is an experimental replacement for the scene-dependent v4 mission stack. It builds an online occupancy map, selects observed-free frontier viewpoints, and returns to its own initial odometry position. An experimental full-task adapter is available, but the original competition course has not been completed. Recorded results, failed counterexamples and resource measurements are documented in [VALIDATION.md](VALIDATION.md). The per-challenge competition plan is in [COURSE_PLAN.md](COURSE_PLAN.md). Current structural-model integration, conditional assumptions and actual November 6 evidence are recorded in [STRUCTURAL_INTEGRATION.md](STRUCTURAL_INTEGRATION.md).

Navigation receives rectified stereo images, IMU, a single downward range measurement, continuous visual-inertial odometry, and PX4 estimated local state. It never receives a simulator pose, rendered depth image, GPS, course CAD, absolute start coordinates, or preset route targets. Simulator spawn metadata and independent scoring live outside the navigation filesystem allowlist.

The selected original-course integration test uses the same approximate seven-inch aircraft, 1280×800 monochrome stereo, and a **forward rigid camera/IMU mounting profile tilted 20 degrees downward**, with zero mounting roll. The checked-in default profile remains 15 degrees downward for a later controlled comparison. A proposed forward mounting at `[0.22, 0, 0.055]` m clears the propeller sweep. The mounting change preserves the rigid stereo/IMU transforms and counter-rotates the existing ToF to keep it body-fixed. At the calibrated 800p intrinsics the vertical half-FOV is about 27.8 degrees; a 20-degree downward mount leaves only 7.8 degrees above horizontal, making near upper-body clearance difficult to observe. The forward fixture tests a general visibility correction without changing course appearance or adding sensors. This is a proposed simulation fixture, not an identified official mounting. Native 1280×800 images remain the VIO input. The CPU stereo proxy matches 640×400 image pairs, scales the focal length and left/right disparity tolerance together, and nearest-resamples measured/invalid values to the native ray grid. This does not fill missing depth or reproduce the OAK firmware. The user has authorized testing a proposed mounting before adapting its physical installation. The actual hardware mount remains uncalibrated. This profile is an explicit test configuration, not a verified hardware calibration. Hardware mode requires explicitly supplied camera calibration, full IMU-to-body translation/rotation, and self-occlusion geometry. The PX4 external-vision message reports the body origin, with angular lever velocity removed; PX4 EV position offsets must therefore be zero.

## Reused components and custom work

| Area | Provenance | Current qualification |
|---|---|---|
| Flight controller | PX4 | Upstream; simulator parameters are not hardware tuning. |
| Stereo-inertial estimator | OpenVINS | Upstream plus integration/health patches; original-course dropouts remain. |
| Point/line inertial candidate | PL-VINS | Pinned upstream in an isolated Noetic container; Ubuntu build passes, first original-course replay did not initialize. Not a flight input. |
| Mapping graph | Cartographer | Upstream plus repairs; mapping only, no pose jump into control. |
| Local trajectory optimization | EGO-Swarm ROS 2 | Pinned upstream plus strict-clearance/runtime repairs. |
| Sensor frames, ToF height support and validity gates | Custom integration | Regression-tested; hardware calibration/timing unverified. |
| Frontier selection, measured-view memory and task executive | Custom logic | Experimental; not a drop-in mature exploration framework. |
| Relative guideline, opening geometry and planar stereo interpolation | Custom prototypes using OpenCV primitives | Limited synthetic/recorded validation; not established SOTA modules. |

Unit tests do not establish module maturity, complete-course success or real-aircraft readiness. Optional geometric visibility remains disabled by default and separate from measured depth/occupied surfaces.

## Architecture

- OpenVINS provides continuous stereo-inertial odometry. A downward-ToF observer supports height; Cartographer consumes this odometry for mapping only. Neither local scan matches nor optimized loop-map poses feed back into the flight-control frame. Sparse-scan correction was rejected after the recorded estimator ablation increased position error.
- The pinned upstream [EGO-Swarm ROS 2 implementation](https://github.com/ZJU-FAST-Lab/ego-planner-swarm/tree/ros2_version) performs local trajectory optimization. The patch and upstream GPL license are included under `patches/` and `upstream/`.
- The occupancy mapper uses exposure-synchronized measured stereo depth. Missing, NaN, low-texture, or out-of-range depth stays unknown. Positive but insufficient occupancy evidence also remains blocked as unknown; occupied visualization requires probability above 0.8. Unknown and occupied cells block the complete vehicle clearance box, including vertical clearance.
- Frontier goals come from the connected observed-free component, exploration history, and a bounded information-gain score. A bounded memory of recently tracked views selects a useful camera heading before translation; it is tied to measured position, height and visual-update quality. A known-free translation may retain a sideways or backward textured view. Translation is gated by current heading and complete-vehicle clearance; quality-gated panoramas from higher and lower operator-verified survey poses supply upper and lower clearance observations. Exploration first uses a narrow relative cruise band. If those viewpoints are exhausted, a height change may be selected only from the connected observed-free volume; an unseen ceiling does not justify a climb. A connected-component grid path supplies short local targets to EGO, with exact full-body segment checks. Home is the first measured body pose in an arbitrary local coordinate frame.
- A separate native-Gazebo evaluator records poses and contacts without publishing them into ROS. Both controller completion and independent evidence are required to accept a recorded episode.

An operator must verify a local **1 m × 1 m takeoff box**, up to **1.05 m above the resting body**, before starting. This local operating precondition is represented as state `-2`, separately from sensor-measured free space (`0`). It is not a reconstructed map. No other unknown space is cleared by this assumption. A forward camera cannot verify its own initial blind volume. An initial vertical sweep inside this operator-verified box observes the lower cruise volume from a higher measured viewpoint; it is a generic active-perception maneuver, not a field waypoint.

## Resource limits and Orin Nano 8 GB

The map contains 1,536,000 voxels (16 × 16 × 6 m at 0.1 m resolution). It is a resource window around the estimator's local origin, not a map of the field. Outside this window is blocked. Relative grayscale/stereo/ToF guideline detection is limited to 5 Hz and 12 line segments; it biases observed-free frontier scores without correcting pose or clearing voxels. Stereo processing is limited to 10 Hz, with a latest-frame worker; OpenVINS uses an 11-clone window and at most 50 persistent SLAM features. Failed EGO planning attempts are limited to 5 Hz. Visualization history and worker timing buffers are bounded. RViz, Gazebo, evaluation, and video encoding belong on the development/offboard computer.

Desktop OpenCV stereo matching is a development proxy. Hardware should use the OAK's measured, confidence-filtered and synchronized depth output with its actual rectified calibration. No artificial depth completion is assumed. The preinstalled JetPack version and power mode are unknown; the present implementation does not require CUDA or a large neural model.

`collect_target_profile.sh OUTPUT_DIRECTORY [SECONDS]` reads JetPack identification, power mode, and `tegrastats` without modifying the device. Desktop process CPU/PSS results are not Orin performance validation. A hardware deployment still requires camera/IMU timing and extrinsic calibration, real-driver tests, and sustained memory/latency/thermal measurements. Short unobstructed motions use upstream cubic B-splines only after each four-control-point hull is certified known-free for the complete vehicle and derivative control points certify speed/acceleration/jerk bounds. Measured velocity supplies an additional latency/braking-clearance check. ROS body odometry uses the standard child-frame twist, and the planner adapter explicitly rotates that velocity into the world frame. A braking hold latches a fixed position, waits for settling, and rejects the previous trajectory across goal changes. Tracking holds trigger replanning from measured state, so a clock-advanced trajectory is not treated as the actual vehicle state.

The current bounded map window is 16×16×6 m at 0.1 m resolution (1,536,000 voxels), selected as a generic resource profile rather than from course dimensions. Native mapper tests verify this actual grid shape and preserve unknown space. Actual Orin performance still requires measurement.

Automatic arming is currently restricted to the existing SITL container/domain.

## Development commands

The current integration uses the existing Ubuntu ROS 2 Jazzy simulation container, `/work/ros_ws`, OpenVINS under v2, and the patched Cartographer workspace under v3. This is not yet a standalone hardware installer. Use the ROS distribution compatible with the actual JetPack host; do not upgrade the preinstalled stack based on an assumption.

```bash
bash /work/cartographer/v5/scripts/bootstrap_ego.sh
bash /work/cartographer/v5/scripts/start.sh source
bash /work/cartographer/v5/scripts/open_windows.sh
python3 /work/cartographer/v5/scripts/run_trial.py UNIQUE_LABEL --mission course_corridor --seconds 180
# Only after confirmed disarm:
bash /work/cartographer/v5/scripts/stop.sh
```

The `course_corridor` profile is the first-stage integration harness, not completed corridor recognition or competition scoring. It requires the relative-guideline node and its measured inputs before arming. The short `navigation_check` remains available as a separate profile.

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

The full CAD-free task executive has a SITL-only ROS/PX4 adapter in `competition_navigate.py`; see [COMPETITION_NAVIGATE.md](COMPETITION_NAVIGATE.md). Its native interface, task perception and complete mission still require original-course validation. The first observe-only run issued no control requests and rejected body poses above its 0.15 m three-sigma budget. Do not equate this interface exercise with task completion or qualified contact sensing.

An independently installed [PL-VINS candidate](candidates/plvins/README.md) uses a pinned Noetic/Focal container, preserving the main Jazzy/OpenVINS installation. It consumes only original left-image and IMU measurements during isolated replay. This is a packaging and estimator comparison path, not an automatic estimator replacement.

## Current limits

The present return baseline assumes a static environment. It may translate backward through already observed free space while retaining a tracked camera heading. The forward-only camera cannot detect newly introduced obstacles behind it. This limitation must be resolved or explicitly accepted for a real operating envelope.

Plain walls and missing stereo can prevent safe expansion of the map. In that case the vehicle must hold, reacquire observable views, return, or land; an unknown volume is not converted into free space for completion. The saved local home position does not by itself constitute a verified visual loop closure. Long-distance global relocalization, larger-map management, complete competition semantics, and deployment on the actual Orin remain separate work.

## Optional stereo-odometry benchmark

The RTAB-Map ROS 2 stereo/IMU adapter and isolated replay runner are included as an independent comparison, not enabled flight inputs. The installed development package is `ros-jazzy-rtabmap-odom` 0.23.13. On the first recorded furnished-room episode, the tested settings lost tracking in 98 of 625 diagnostic frames and produced 0.480 m fixed-initial-alignment position RMSE (0.228 m in XY). They were therefore rejected as a flight-control replacement. These results are specific to this configuration and recording, not a general ranking of the estimators.

## Offline completion experiment

PromptDA Small is benchmarked separately on monochrome-left images and measured stereo anchors. Its output is saved as inferred depth and is not a navigation input. Zero-filled and nearest-interpolated prompts are labeled separately. The prediction process uses the same filesystem isolation and cannot access independent geometry/poses. The current test does not justify treating model predictions as observed free space or claiming Orin timing.

## Original-course localization and stereo investigations

Primary integration tests use the unchanged original course. OpenVINS now uses CLAHE, selected by replaying exactly the same original-course stereo/IMU recording through the pinned patched executable in isolated domain 43. `replay_vio.py` replays only recorded sensor inputs; `evaluate_vio_replay.py` applies independent first-pose-aligned scoring afterwards. No estimator receives the reference trajectory. Earlier replays using an unpatched installation were rejected and must not be mixed into the comparison. See [VALIDATION.md](VALIDATION.md).

`extract_stereo_samples.py` exports only raw monochrome stereo, its rectified calibration, self-occlusion mask, and classical measured-stereo baseline. The optional `benchmark_learned_stereo.py` evaluates pinned LAS2-S in a separate `--offline-stereo` filesystem profile. It cannot read the course, poses, evaluator, or episode results. Predictions are saved as inferred depth, not published to navigation. `evaluate_source_stereo.py` separately ray-scores frozen predictions against the original physical collision meshes and native evaluator poses. Left/right and photometric consistency alone do not establish safe depth on plain surfaces. No learned candidate is enabled for flight by these tools. CPU timing is not Orin timing.
