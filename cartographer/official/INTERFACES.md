# Official-lab stack: module contracts (2026-11-07)

Goal: fly the REAL DroneHacks venue (photos: `/private/tmp/claude-501/-Users-chuyao-Documents-Codex-2026-10-31-cartographer-dronehacks/2198afae-c72b-4034-8353-abe0a35bdca0/images/1.webp` … `7.webp`) and the official sim with one Python stack. No floor guideline exists in the real venue. Sensors: OAK-D stereo (horizontal, 640x400, fx≈421), depth aligned to the left camera, IMU, downward ToF (`/fmu/out/distance_sensor`), cuVSLAM VIO → PX4. No GPS, no CAD; coarse stage positions measured on site (±0.5 m) are allowed as a prior.

Course (venue photos): 1 corridor between printed sci-fi banner panels on wooden frames; 2 tunnel = brass/gold hoops + black mesh, rock-print fabric on top (stereo depth sees the black mesh poorly); 3 window = tan board with a rectangular opening, black strip below; real 3-D rock cones (~0.2–0.4 m) AND flat painted rocks on a glossy orange floor; final landing target = large white rock/stone block; then disarm, re-arm, climb over a wall, return to the measured start, land. Arena = black net cage ~6.1 x 6.1 x 3.0 m (net is nearly invisible to depth: geofence it).

Frames and helpers: `common/geometry.py` (`Frame`, `camera_pose_ned`, NED world, FRD body, optical camera). All modules are PURE (numpy/OpenCV/scipy only, no ROS, no clock reads); each ships `test_*.py` runnable with the repo `.venv` (`/Users/chuyao/Documents/Codex/2026-10-31/cartographer-dronehacks/.venv/bin/python`, Python 3.9, numpy 1.26, OpenCV 4.11) and must also run on Python 3.12 (official container).

## perception/floor.py (+ landing target)
- `fit_floor(frame, prior_floor_d=None) -> FloorFit(found, d, normal_ned, inlier_frac)`: dominant floor plane from depth (RANSAC).
- `obstacles(frame, floor: FloorFit, min_height_m=0.05, max_range=4.0) -> ObstacleCloud(points_ned (N,3), height_above_floor (N,))`: only 3-D structure; painted rocks / floor print must be rejected.
- `landing_target(frame, floor) -> LandingTarget(found, center_ned (3,), top_d, extent_m, confidence)`: large bright, low-texture white/grey block raised above the floor with a flat top.

## perception/tunnel.py
- `detect(frame, prior_entry_ned=None) -> TunnelEstimate(found, entry_ned (3,), axis_ned (2,) unit, radius_m, confidence)`: tunnel mouth from the bright brass hoop arcs (ellipse fit) + any depth on the fabric top; do NOT rely on depth of the black mesh.

## perception/window.py
- `detect(frame, half_span_m, half_height_m, prior_center_ned=None) -> WindowEstimate(found, center_ned (3,), normal_ned (3,), width_m, height_m, clearance_ok, confidence)`: board plane + rectangular region whose depth is clearly beyond the board.

## mission/core.py
- Pure state machine, same style as the official `refnav/core.py` (step(now, vehicle, perception) -> Output(setpoint NED+yaw, commands, events)).
- Phases: wait → arm → takeoff → corridor (coarse goals + obstacle-aware carrot) → tunnel_align → tunnel_traverse → window_align → window_traverse → land_target → disarm → rearm → climb_over_wall → return_home → land_home → done; any phase can abort to land.
- Obstacle interface it consumes: `ObstacleMap.clearance(p_from_ned, p_to_ned, radius_m) -> metres free along the segment (inf if clear)` and `ObstacleMap.blocked(p_ned, radius_m) -> bool`.
- Course prior: JSON of coarse stage goals relative to the measured start (forward/left/up metres + tolerance); placeholder values until the team measures the venue.

## node (main session)
ROS 2 shell in the official workspace: PX4 topics (`*_v1`), Offboard setpoints at 20 Hz, TF/CameraInfo, depth → `ObstacleMap`, calls the perception modules, runs `mission.core`.
