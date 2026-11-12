# Official DroneHacks Lab sim — what an Ubuntu Gazebo replica must match (2026-11-06)

Goal: `node/mission_node.py` + `tools/flight_run.py` run UNCHANGED on Ubuntu Gazebo and behave like the
official lab (world "DroneHacks prepared", aircraft Sky Beta). The official sim uses NVIDIA PhysX
(Isaac-Sim-class), so identical physics is impossible; match the airframe, sensors, interfaces and
estimator behaviour. Sources: official guides (AI_CONTEXT.md, SIM_SENSORS.md), live topic inspection on
2026-11-07 (see memory "official-lab-facts"), mission_node code.

## Verified facts (match exactly)

| item | official value |
|---|---|
| aircraft | Sky Beta, 1.25 kg, 7 in props, OAK-D S2; planner half span 0.20 m incl. props (`common.geometry.SKY_BETA_HALF_SPAN_M`) |
| stereo | `/stereo/left/image`, `/stereo/right/image`: `sensor_msgs/Image` mono8 640x400, 15 Hz (modelled losses possible) |
| left intrinsics | fx = fy ≈ 421, cx 319.2, cy 200.7 (`common.geometry.OFFICIAL_K`); HFOV = 2·atan(320/421) ≈ 74.5° |
| baseline | OAK-D S2 stereo, ~7.5 cm (verify: oak_right TF) |
| depth | `/camera/depth/image_raw` aligned to the LEFT camera (same K), 15 Hz; `/camera/depth/camera_info`; encoding 32FC1 m or 16UC1 mm (node accepts both); invalid = 0/NaN |
| camera mount | horizontal, optical axis = body forward (0° pitch); TF `base_link -> oak_left` t = (0.116, 0.037, 0.029) m, R_base_cam (FLU←optical) = [[0,0,1],[-1,0,0],[0,-1,0]] |
| TF frames | `base_link<-oak_left`, `base_link<-oak_right`, `base_link<-oak_rgb`, `oak_left<-oak_imu`, `odom<-base_link` |
| IMU | 480 Hz (OAK IMU, frame oak_imu); physics 240 Hz |
| down range | `/fmu/out/distance_sensor` (px4_msgs DistanceSensor, PX4-side sensor) — node checks `min_distance <= current_distance <= max_distance and signal_quality != 0` |
| estimator | cuVSLAM (stereo+IMU) -> `onboard/vio/ev_bridge.py` -> `/fmu/in/vehicle_visual_odometry`; `/estimator/healthy` (`std_msgs/Bool`, ~>1 Hz, node requires fresh <1 s and true) |
| PX4 | uXRCE-DDS, px4_msgs with `_v1` outputs: `/fmu/out/vehicle_local_position_v1`, `/fmu/out/vehicle_status_v1`, plus `/fmu/out/vehicle_attitude`, `/fmu/out/vehicle_land_detected`; inputs `/fmu/in/offboard_control_mode`, `/fmu/in/trajectory_setpoint`, `/fmu/in/vehicle_command`; QoS BEST_EFFORT + TRANSIENT_LOCAL depth 1 |
| control | Offboard position (+vel/acc feed-forward) setpoints at 20 Hz; SET_MODE 176 (1, 6) -> nav_state 14; ARM 400; LAND 21 -> nav_state 18; stamps = system clock µs; mission logic on `/clock` sim time |
| observed behaviour | hover settles 8–16 cm below setpoint; RTF ≈ 0.86; sim battery drains over flights (battery_unhealthy blocks arming until Reset world) |
| no other topics | no RGB topic for navigation (oak_rgb TF exists); world observer is display-only |

## Unknown — capture at the next official GPU session (one script, save to flights/spec/)
- `param show` for MPC_*, EKF2_*, COM_*, NAV_*, SENS_*, CA_* (airframe, max speeds/tilt, EV fusion, range fusion).
- Full `/stereo/left/camera_info` (D, P), right camera_info, depth encoding and valid range, `ros2 topic hz` of each.
- `/fmu/out/distance_sensor` min/max/fov/orientation/rate; IMU topic name and noise.
- `ros2 run tf2_tools view_frames` (full tree with numbers).
- `onboard/config/*`, `onboard/services.py`, `onboard/vio/ev_bridge.py` health rules, cuVSLAM params.
Until then, the Ubuntu replica uses the values above and documents every assumption.

## Captured from the official lab (2026-11-07 09:47 PST; full files on the team disk: `claude_dev/flights/spec-20261107/`)
- Left and right camera_info: 640x400, plumb_bob with D = 0, K fx = fy = 421.016, cx 319.248, cy 200.659; right P[3] = -31.866 → baseline **0.0757 m**.
- TF base_link→oak_left t (0.116, 0.037, 0.029); base_link→oak_right t (0.116, -0.038, 0.029); both R = optical (quat xyzw 0.5, -0.5, 0.5, -0.5).
- TF oak_left→oak_imu t (0.075, -0.002, 0.000), roll 180°. IMU topic `/stereo/imu`.
- `/fmu/out/distance_sensor`: min 0.0, max 4.0, variance 0.0255, signal_quality **-1** (not 0 → the node accepts it), orientation 25 (down); reads 0.05 m on the ground.
- Topics also present: `/estimator/healthy`, `/refnav/odom`, `/visual_slam/*` (status, processing_status), `/fmu/out/estimator_status_flags`, `/fmu/out/battery_status_v1`.
- The onboard config is copied there too (`onboard_config/`, `ev_bridge.py`, `services.py`).

## The REAL drone differs (skybeta-05, 2026-11-07): see `outputs/claude-official/realdrone-REPORT.md`
ROS 2 Humble; contract `/visual_slam/tracking/odometry` (ENU) + `/skybeta/state` + `/skybeta/setpoint/pose|twist`; no depth topic; no distance_sensor topic; `/fmu/out/vehicle_local_position` (no `_v1`); camera mount x 0.1053, y 0, z 0.0304.
