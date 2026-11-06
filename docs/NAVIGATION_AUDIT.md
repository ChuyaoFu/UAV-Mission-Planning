# Navigation input and hardware portability audit

The active implementation is `cartographer/v4`. It is a **known-course, CAD-aided GPS-denied controller**, not an unknown-environment autonomous system or a hardware-ready flight package. Earlier v1/v2/v3 and root simulation scripts remain historical experiments and must not be mistaken for the current launch path.

## Simulator state and truth

The mission consumes fused visual/inertial pose, measured passive-stereo clouds, downward range, localization health, and PX4 estimated local position/status. `VehicleLocalPosition` is the flight controller's estimate, not Gazebo's model pose. The control conversion uses that estimate to express a navigation error in PX4's local frame.

OpenVINS receives stereo images, IMU and the motion-initialization signal. Cartographer receives measured stereo structure, IMU and VIO odometry. Fusion filters shared TF messages to exactly `carto_odom -> vio_imu`; global loop-closure poses do not enter PX4. Source inspection found no evaluator/Gazebo pose/GPS subscription in this chain.

Gazebo still publishes model/link pose for `/evaluation/poses`. The independent judge, recorder and offline evaluators read it. Recording truth in the same MCAP does not make it a navigation input. `audit_runtime_inputs.py` saves the actual protected nodes' subscriptions before flight and again while the mission runs. This audit checks named ROS nodes and topic names; it is not a security boundary or a proof about every native PX4/Gazebo internal implementation. PX4's own simulated IMU/barometer/actuator interfaces are simulator sensor/plant interfaces, not a navigation truth-position shortcut.

Before execution, `mission.py` checks `EKF2_GPS_CTRL=0`, `EKF2_EV_CTRL=13`, and that `sensor_gps` has never been published. The v4 launch now rejects the historical `ideal` depth option. Its depth input is `/stereo/depth/image`, computed from the actual left/right images. Historical resource generation still includes an unused rendered-depth camera; the live subscription audit must show no protected consumer of `/vio/depth/image`.

## Hard-coded assumptions and remaining migration work

| Assumption | Current use | Hardware implication |
| --- | --- | --- |
| Published CAD geometry and manually annotated stage coordinates | A*, local collision checks, CAD distance-field corrections | Requires an independently surveyed course prior; this is not online-only SLAM navigation. Changes to the real course require updating the prior. |
| Launch position/yaw | Initial VIO-to-course alignment in `stages.json` and `--start-y` | Aircraft must be physically placed/aligned at the surveyed start, or initial registration must estimate this transform. It is a prior, not a measured global position. |
| Window dimensions/orientation and red floor guideline | Passive image-to-prior registration | Specific to this course and appearance. Window and line detections require real-camera validation; they are not general semantic recognition. |
| IMU/body/ToF offsets | Lever-arm compensation and height observation | Shared in `config/robot.json`; `HARDWARE_MODE=1` requires an explicit measured `ROBOT_CALIBRATION`. General nonparallel body/IMU axes are not implemented by this translation-only profile. |
| Rectified intrinsics, stereo baseline and camera/IMU transform | Projection, disparity and registration | Shared rectified calibration; hardware mode requires explicit `STEREO_CALIBRATION`. Real cameras must supply rectified images at the calibrated resolution and correct exposure timestamps. |
| Body radius, tracking and vertical margins | Inflated planning grid and live obstacle envelope | Planning values are declared in `robot.json`; the historical live envelope remains conservative numerical values. Measure the actual rotor envelope and estimator uncertainty. |
| Cruise/braking/latency thresholds | Speed governor and recovery budgets | Engineering assumptions validated on this SITL model, not measured limits for an official aircraft or an onboard computer. |
| Downward ToF proxy | Minimum of a bounded 25-ray cone, exposed as a single range | A simulator proxy for the kit sensor, not an added mapping lidar. It omits real reflectance, crosstalk, multipath and firmware failure modes. Use the real sensor driver and validate invalid readings. |
| Low-noise pinhole/global-shutter images and synchronized timestamps | Simulator sensor inputs | Blur, vibration, exposure changes, timestamp offset, texture loss and real sensor noise remain to be tested on recordings. |
| Flat continuous floor and conservative surface-transition rejection | Height observer | Wall tops/obstacles must not redefine the global floor. Ramps and wide elevated surfaces remain ambiguous without additional evidence. |
| Fixed ROS domain, paths, node names, PX4 IDs and simulated clock | Prepared-container launch and SITL-only arming guard | A separate hardware launch, clock synchronization and measured flight-controller configuration are required. The guard remains in place. |
| Rotor allocation, hover thrust and controller parameters in `start.sh` | This simulator plant | Never copy these actuator settings to a physical aircraft. Use the target's calibrated mixer, actuator geometry and flight tuning. |
| Blind upward region with forward/downward cameras | Takeoff and return climb | The return climb requires a surveyed clear vertical corridor and roof prior. It cannot detect a newly introduced obstacle above the camera field of view. This sensor layout does not justify unknown-overhead obstacle avoidance. |

## Repairs made for the return mission

- Replaced repeated application of the latest downward range inside `world_body()` with the timestamped `HeightObserver`: each sample is applied once; tilt, inconsistent surfaces and unsupported floor reacquisition are rejected.
- Centralized body/IMU/ToF translation calibration and added a hardware-profile requirement. The default values match the existing model; the physical model and sensor geometry were not changed.
- Preserved the outbound 1.45 m planning cap. A separate return grid retains the same CAD surfaces and vehicle margins, with a 2.50 m policy cap below the 3.048 m roof.
- Moved passive-stereo matching out of the ROS image callbacks into a bounded latest-pair worker, with depth-one input queues. The matcher quality checks remain unchanged; obsolete work is dropped instead of accumulating camera latency. `STEREO_THREADS` configures the CPU matcher (default 4 for the measured development host); the target computer must be benchmarked separately.
- Bounded return climb/descent position commands and reduced them further on stale depth, braking before the existing observation-displacement veto. No observation-age/displacement threshold was increased.
- Added an explicit surveyed descent-view landmark selected by name from CAD bounds. The aircraft faces the observed course corner before descending; the prior selects camera direction and does not supply a pose measurement. This remains course-specific active-perception planning.
- Added actual destination landing/disarm, rearm without resetting localization, return climb, wall crossing, descent and home landing/disarm.
- Extended the independent judge to require both landing cycles, a measured crossing of the specified wall above its top, final launch-area ground contact, arena/roof clearance and collision penalties. It still rejects outbound over-wall shortcuts.
- Added synthetic regression cases for skipped destination landing, missing rearm, missing final contact, insufficient wall clearance and return contacts. Synthetic tests are not flight evidence.

The return configuration is an engineering task extension requested by the user. It does not assert official competition scoring or permission for hardware flight.

`docs/source-manifest.json` retains original import provenance, records the import hash separately for edited files, and updates current hashes for repository development. Episode source hashes and mapper binary hashes are captured independently before each run.
