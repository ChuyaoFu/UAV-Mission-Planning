# Competition Navigator v3: reviewable ROS/PX4 adapter

`competition_navigate.py` connects the v2 complete task executive to the existing real `navigate.Navigator`. It is a separate entry point; it does not run the navigation-check-only `Navigator.run()`. This delivery was developed only in an independent copy. **No SSH, DDS, PX4, simulator, deployment, or flight was operated by this chat.** Tests use explicit isolated ROS/PX4 stubs and synthetic sensor/perception/controller fixtures.

`integration-from-v2.patch` adds the adapter/input helpers/tests and small v2 integration fixes. It does NOT replace `navigate.py`, `camera_geometry.py`, `view_memory.py`, `bounded_yaw.py`, map dimensions, robot mounting, fusion policy, or any active configuration. The reference source/config under `reference/v5/` is an unchanged read-only snapshot for reproducible NO-DDS tests, not a deployment overlay.

## Dependencies on the primary chat's current baseline

The subclass integrates the baseline's `vehicle_status` and `command_ack` callbacks: one status/ack subscription feeds BOTH the inherited native handshake fields and the task's ordered native-state cache. Do not add a second independent status callback that loses one of these states. The reference snapshot includes the primary chat's native yaw limiter, preflight-aware handshake fields, calibrated `bearing_visible`, and `ViewMemory.choose(..., allowed=...)` support.

Ordinary translation reuses `publish_goal` / actual EGO `PositionCommand` / `follow` / `brake` / `send`. The original health, visual-observability, accepted velocity, frame conversion, complete-body clearance and stopping gates remain active. Aligned short goals are held until reached instead of cancelling EGO whenever the executive recomputes a receding target. The adapter uses the map budget configured by the primary chat and does not enlarge it.

The adapter's proof map defaults to direct measured free plus the explicitly declared initial launch region. It sets its own `geometric_free_rays=False`; conditional `-3` cells are not silently counted as unconditional competition proof. This does not change the baseline's separate geometric-free diagnostic configuration or global map data. If the primary chat wants conditional-geometry competition control, that needs an explicit provenance-aware integration review rather than changing a hidden default.

## Sensor and native interfaces

| Input | Use |
| --- | --- |
| `/nav/body_odom` | Accepted BODY local ENU pose/covariance, body-frame twist, body gyro. Inherited conversion expresses velocity in local ENU. No pose differentiation. |
| `/vio/left/image_raw` | Calibrated mono8 left image, native grid. |
| `/nav/measured_depth` | Raw direct metric 32FC1 stereo, `left_optical`; no ideal/completed depth. |
| `/nav/camera_pose` | Exposure-time calibrated left-optical pose in accepted `world` frame. |
| `/carto/down_range` | Actual valid Range with original exposure stamp/min/max; no rendering/truth API. |
| `/fmu/out/vehicle_status_v1` | Actual armed/navigation/preflight state. |
| `/fmu/out/vehicle_land_detected` | Actual `landed`, excluding freefall; no inference from disarmed or low z. |
| `/fmu/out/vehicle_local_position_v1` | Existing native setpoint-frame conversion and emergency hold. |
| `/fmu/out/vehicle_command_ack` | Owned command result; absence is timeout/unknown, not an explicit rejection. |
| `/grid_map/metadata`, `/grid_map/knowledge` | Existing Navigator/OnlineMap interface. No course file or prior map is loaded. |

Left/depth/camera pose are joined by EXACT integer nanosecond exposure keys. Body pose must bracket the same exposure; Slerp/translation/twist interpolation never extrapolates and rejects >100 ms pose gaps. The camera pose is cross-checked against the accepted body pose and actual calibrated body-to-optical mounting. Image encoding/grid/stride/endian are checked; invalid direct depth stays invalid.

ToF is independently paired to bracketing accepted BODY pose at its exposure time. Duplicate range stamps are ignored instead of being counted twice. Native PX4 boot microseconds are used for native-source ordering only; native freshness uses recorded ROS receipt time. Native status is required within 500 ms, landed/local state within 350 ms, ToF within 150 ms. Exact-topic/version/QoS/rate behavior still needs the primary chat's real ROS validation.

Camera/IMU mounting is ALWAYS read from the actual robot/stereo calibration. No 5-degree/15-degree constant is embedded. Exploration calls the primary chat's full calibrated `bearing_visible` with measured body tilt and filters remembered headings through `ViewMemory.allowed`. A high-texture backwards yaw cannot override the actual frontier view cone. Return translation may retain its useful view, as in the baseline.

## Startup and first-launch authorization

Automatic control still requires ALL of: authorized SITL container, ROS domain 42, `SITL_AUTOMATION_AUTHORIZED=1`, simulation clock, and no hardware mode. The check occurs before Node construction. A newly started controller refuses an already-armed vehicle and does not claim/control it. `--observe-only` performs sensor diagnostics and emits no control commands/setpoints.

Before arming, actual stationary upright disarmed PX4-landed ToF/body samples must yield `RestingFootCalibrator.result`. There is no guessed foot offset. The own measured resting body pose defines home. Native OFFBOARD is primed/confirmed before independently requesting ARM; `pre_flight_checks_pass` must be true. ARM retries are separated by at least 0.5 s and bounded by a 15 s bootstrap phase. Actual native armed state, not command emission, confirms arming. Explicit denied/unsupported/failed/cancelled acknowledgements stop the request; no force-arm parameter exists. Missing acknowledgement is not labelled a native denial.

The FIRST launch alone uses the user's verified 1 m x 1 m local box up to 1.05 m above the measured resting body. It includes the padded robot envelope and measured stopping projection. Initial underside every-pixel stereo is not required: a forward camera cannot see it. The startup certificate is explicitly `declared_local_launch_only`, does not clear the map, and cannot be reused for dock/home/rearming. Startup follows the current baseline's configured survey/cruise heights and quality-gated bounded arcs, with the inherited native yaw limiter. No full 360-degree sweep is commanded.

## Complete task loop and proof

The executive is ticked with NEW matched ToF/body feedback. It covers corridor/frontier exploration, tunnel entry AND distinct exit, window/allowed alternative, terminal dock landing/disarm, confirmed rearming/liftoff, observed wall return and own-home landing/disarm. Time limits only abort; no time or distance completes a task.

`PriorControlLedger` records certificates for ACTUAL published control or continued native LAND. Planned executive intents alone are never recorded as flown authority. Delayed accepted feedback is split at body samples AND command issue times; each subinterval requires a certificate issued beforehand. Later commands/maps cannot retrospectively bless already-flown motion. Queues/history/ledger/workers are bounded. Pose time/frame jumps stop the audit and trigger native hold.

Unknown-layout exploration calls the existing online-map frontier, at the current measured altitude and adjacent voxel bands, without surveyed points/turns. The camera view cone is checked before turning/following. Every actual reference and measured stopping projection is rechecked in the current complete-body map. The primary chat still owns full 16x16x4 map and Orin resource budgeting.

Before dock approach, the adapter requires a FRESH strict dense support patch acquired AFTER the height-freeze acknowledgement. Missing support causes further measured-frontier work rather than first overflying the pad and discovering a blind underside view. Support checking is NOT relaxed: every covered native pixel must be direct, no holes/edges/protrusions are filled, and actual independent matching-grid spacing is used. A nearest-upsampled half-resolution depth image cannot pretend its copied pixels are independent measurements.

A front/down camera may lose the marker beneath the body. A still-valid original dense patch PLUS fresh calibrated ToF/accepted pose/current-map contact certificate can then authorize LAND; a fresh underside marker image is not falsely required. This is the existing static measured-support contract, not new free-space inference. Calibrated home/dock alignment is 2.5 cm before entering the contact mode whose lateral drift limit is 4 cm. A contact veto keeps valid body-feedback continuity but advances no task evidence.

## Native mode ownership

`land` sends mature PX4 LAND, maintains the offboard hold only while OFFBOARD is still the observed native mode, and stops ALL conflicting offboard heartbeat/position publication after actual LAND confirmation. Context is refreshed through the strict v2 monitor. A veto requests actual native AUTO LOITER (PX4 custom main/submode 4/3) and confirms native navigation state; merely sending an ignored offboard point is not treated as an abort. Direct emergency hold yaw also passes through the SAME inherited native-heading rate/lead limiter. This uses the official [PX4 mode definitions](https://github.com/PX4/PX4-Autopilot/blob/main/src/modules/commander/px4_custom_mode.h) and [native state/command result definitions](https://github.com/PX4/px4_msgs/blob/main/msg/VehicleStatus.msg).

Dock disarm/rearm needs actual native feedback and the calibrated contact/plane/ToF dwell. Rearming first establishes a new offboard prime, preflight readiness and its own ARM request/status confirmation. `takeoff` uses the contact-constrained monitored liftoff before returning to normal mapped flight. It creates no new launch-box exception.

On failure the adapter does not call the baseline's unconditional `finish()->LAND`. It requests and confirms native HOLD, or safely disarms only with fresh actual landed/near-zero calibrated ToF/static feedback. It does not exit an owned armed OFFBOARD vehicle on an unconfirmed handoff: it remains alive, writes an INCOMPLETE result and waits for native confirmation or primary/operator takeover. The primary chat must validate these mode transitions and abort behavior on real DDS before relying on this entry point. The controller does not automatically resume a broken strict VIO evidence chain.

## Height observer API

The adapter publishes only `true` on `/nav/freeze_global_height` before platform approach and waits for a fresh `/carto/localization_health.global_height_frozen=true`. It never sends false/unlocks through dock/disarm/rearming/return. Root has already installed the monotonic process-lifetime latch; see `FUSION_FREEZE_API.md` for the minimal contract. Raw range/plane-relative height remains observable, while the accepted VIO gauge and current accumulated correction stay fixed. No platform is treated as global z drift.

## Concrete primary integration blockers

1. **Landing speed mismatch in source:** `start.sh` currently sets `MPC_LAND_SPEED .35`; the v2 monitor defaults to <=.25 m/s. `start_sim.sh` seeds .2, but the later startup script can override it. Verify the ACTUAL running parameter. The separate proposed `px4-land-speed-0p20.patch` aligns the source to .20; it was not applied here. Do not silently relax the monitor to make a test pass.
2. **Strict support retention stays <=10 seconds.** Slow approach + LAND + dwell + mode prime + rearm/liftoff may exceed it. Acquire genuinely new direct support evidence or refuse; copying a timestamp is prohibited. The all-task protocol fixture supplies synthetic support snapshots to isolate controller wiring and does NOT demonstrate that a fixed real camera can refresh them at every pose.
3. **Perception is still a bounded geometry/class hypothesis.** Actual original-course detection recall, tunnel exit visibility/topology, true wall-top versus aperture-lip identity and platform semantics need recorded frames and independent scoring. Generic successful controller evidence is not official competition validation.
4. **No real DDS/Orin/flight validation here.** Verify actual message version/QoS/rates, preflight/ACK handshakes, mode cancellation, VIO continuity across disarm, complete-body map transport consistency, depth provenance/error budgets, native descent speed and scheduling. `--observe-only` is the first integration gate, not a flight.

## Native interface and runner

The explicit uncertainty policy retains the original BODY covariance and uses a configurable admission domain (default0.20m). Each motion/stopping certificate includes the complete current and retained map-reference uncertainty. Admission alone never permits motion. `--map-pose-error-m` is required and must cover retained contributing observations; it is a declaration requiring qualification, not an automatically measured bound. Local contact precision is checked separately. No covariance subtraction or gauge cancellation is assumed.

The actual Ubuntu observe-only integration joined exposure data with a400-sample BODY cache and zero rejected joins. Its approximately0.154m three-sigma covariance was preserved. It sent no control commands. This validates an interface, not startup/contact qualification or competition completion.

`run_competition_trial.py` provides a separate full-mission runner. It requires actual disarm, exclusive flight lock, disabled GPS, input/filesystem audits, both visible windows and video recording before launching control. It records the original camera stream, Gazebo/SLAM windows and independent pose/native-status/physical-contact evidence. Private task geometry under `evaluation/` is read only by the offline scorer after controller output is frozen. Successful completion requires the ordered tasks, both native arm/disarm cycles, whole-body crossings, zero unexpected contacts, validated graph and preserved videos.

## Primary-run examples

Run from the actual installed scripts directory after the primary chat applies/reviews the patch, supplies actual calibrated error budgets and verifies the native profile. Variables below must be explicit measured/validated budgets, not synthetic fixture values.

```sh
python3 competition_navigate.py --observe-only --output /work/competition-observe.json \
  --tof-error-m "$TOF_ERROR_M" --tof-extrinsic-error-m "$TOF_EXTRINSIC_ERROR_M" \
  --stereo-error-m "$DIRECT_STEREO_ERROR_M" \
  --uncertainty-admission-m 0.20 --map-pose-error-m "$RETAINED_MAP_POSE_ERROR_M"

python3 competition_navigate.py --output /work/competition-result.json \
  --tof-error-m "$TOF_ERROR_M" --tof-extrinsic-error-m "$TOF_EXTRINSIC_ERROR_M" \
  --stereo-error-m "$DIRECT_STEREO_ERROR_M" --mission-budget-s 600 \
  --uncertainty-admission-m 0.20 --map-pose-error-m "$RETAINED_MAP_POSE_ERROR_M"
```

CLI authorization, native state, calibration and live-map gates remain mandatory. The examples do not start/modify a simulator or grant hardware permission.

## NO-DDS verification

```sh
PYTHONPATH=reference/v5/scripts:. python -m unittest discover -s . -p 'test_*.py'
```

The isolated subprocess imports the real copied `navigate.Navigator`, with explicit `rclpy`/message stubs. It never creates DDS, sockets, a simulator or a PX4 process. Tests exercise actual inherited frame conversion/follow/brake/publication, exact exposure/body joins, native mode/arm/preflight/ACK confirmation, readonly diagnostics, first-launch scope, yaw/view constraints, rejected unknown/stopping volume, native LAND ownership, real HOLD confirmation, own callback integration, no retroactive motion authorization, and a complete eight-task protocol trace.

The complete protocol test stubs perception/support patches and a bounded vehicle-response fixture. It is not a real sensor image test, native controller performance test or flight. `ros-interface-evidence.json` marks these limits explicitly. Existing strict support/hole/edge/provenance negative tests remain in the ordinary unit suite. All output success flags in the synthetic fixture are unit outcomes only.

Frozen verification: **100 parent-suite tests passed (includes the isolated-subprocess wrapper), and all 32 explicit NO-DDS ROS/PX4 cases passed**. See `unit-results.txt`, `ros-stub-results.txt`, and `ros-interface-evidence.json`. Do not add the subprocess wrapper and its contained cases as if they were independent flights.

Apply `integration-from-v2.patch` ONCE for the complete frozen bridge and required v2 fixes. For separate review, the same changes are split into `bridge-only.patch` and `v2-adapter-fixes.patch`; `task-executive-v2-fixes.patch` is only the executive portion of the latter, so do not apply both copies. `px4-land-speed-0p20.patch` is a separate optional source-profile recommendation, not a live vehicle parameter change. The fusion latch is already in the primary baseline and is documented rather than duplicated.
