# Structural mapping and complete-task integration — 6 November 2026

This checkpoint has **not completed the original competition course**. No hardware flight or Jetson Orin timing has been validated. Ubuntu development tests use the original course, GPS disabled, the existing stereo/IMU and downward ToF, and estimated odometry. Course geometry and simulator poses are confined to independent evaluation.

## Selected mounting and reusable components

The current trial uses a rigid 20-degree downward camera/IMU mounting with zero roll, selected before adapting the physical fixture. The checked-in general profile remains 15 degrees downward for a controlled comparison. Physical camera/IMU/body/ToF calibration is still required. OpenVINS, PX4, EGO-Planner and Cartographer remain the reused estimator, controller, local planner and visualization components. OpenCV GrabCut and line detection provide conditional scene geometry; their integration and task executive are project-specific code.

## Conditional geometry model

On 6 November the user explicitly confirmed that the local floor is continuous
and flat and the walls are vertical, and authorized those structural conditions.
This does not authorize assumed wall coordinates, ceiling clearance, absence of
hidden obstacles, or conversion of inferred support into measured depth.
The user separately supplied the enclosing net dimensions: a 20 by 20 ft floor
(6.096 by 6.096 m) and a 10 ft net height (3.048 m). The launch location and
heading within that enclosure were not supplied. These dimensions do not define
wall locations, waypoints or observed free voxels.

`bootstrap_structural_evidence.sh` builds the source-only native OpenCV helper. Configure `structural_evidence_backend: grabcut`, variant `cold5_freeze1_seedvalidated_240p`, and explicitly enable geometric free rays to select it. The task adapter additionally requires `--allow-conditional-free`. Both option and configuration must enable consumption. The default task policy remains measured-only.

Segmentation runs at 384×240; images, masks, intrinsics and output ray geometry retain the original 1280×800 grid. Stereo-measured floor fits are checked against gravity and the actual ToF. Floor continuation and vertical termination at observed boundaries are **conditional static-scene assumptions**, not directly measured wall depth. They create no occupied endpoints. Raw occupancy and ambiguity veto inference. The inference surface is retreated by 0.10 m, and the native mapper independently retreats a further voxel diagonal (0.173205 m at 0.1 m resolution).

Static support retains its original exposure timestamp for at most 10 s. It is withdrawn from navigation 2.7 s before expiry, reserving map/packet age, control reaction and braking under the declared 1 m/s speed and 0.5 m/s² braking bounds. Raw freshness is unchanged. Exceeding the motion bound or encountering contrary raw evidence rejects its use. This assumption must be reconsidered for moving obstacles or unsupported scene structures.

## Actual evidence

- Independent collision-mesh evaluation of 27 actual source frames found worst surface positive-error p99 of 4.42 cm. The retreated ray prefixes had no positive p99 or >25 cm overrun in these samples. This is a sampled simulator evaluation, not a hardware accuracy bound.
- A 65 s sensor-only ROS replay through the actual native mapper produced 571 map packets, 35 accepted inference publications, and 33 map packets with newly usable full-body paths compared with measured-only queries on the **same packet**. The 202 improved queries overlap in time and direction; they are not 202 flights.
- The native runtime test verified the real 160×160×60 grid, inferred-only cells, terminal retreat, early expiry withdrawal, and preservation of a newly measured occluder.
- Full trial 31 stopped in ascent when a prior certificate did not reserve subsequent covariance growth. The causal forecast reserves actual native velocity uncertainty plus an explicit bounded-acceleration enclosure before command publication and charges it to the geometry. The current control certificate lasts at most 0.10 s, refreshed by the 20 Hz control loop; its forecast also includes actual pose age. Path and contact certificates cannot outlive this reserve. Future covariance is never substituted into an old certificate.
- Trial 33 passed 185 prior-motion audit segments, then rejected a current startup envelope. Native LAND was commanded by the primary operator and disarm confirmed. It is an unsuccessful autonomous episode. Videos and the intervention record were retained.
- Trial 34 passed 271 motion-audit segments. Its recorded stopping projection exceeded the unchanged launch half-width after charging a 0.35 s forecast horizon. Native LAND and disarm were recorded. Trial 35 tests the shorter immutable certificate lifetime and the complete certified local-box startup target.
- Trial 32 never armed because the packaged CLI initially lacked the new explicit conditional flag. That packaging error was corrected before trial 33; the failed artifact is retained.

The original-course video archive is `/home/fcy/Videos/UAV-Mission-Planning/original-course/` on Ubuntu, with desktop shortcut `UAV-Mission-Recordings`. Each episode retains screen flight/SLAM and FPV video; successful episodes must not be overwritten.

## Startup and native contact

Typed mechanical startup and LAND/disarm/rearm/home contact remain explicit `--complete-task` functionality. Robot skid geometry comes from the aircraft model, not course CAD. Startup uses actual native status, ToF and BODY history. Cancelling startup ZUPT after an ordered native landed=true→false transition changes only the estimator mode; it does not claim measured ascent or inject position, velocity or covariance. The actual ToF departure detector stays separate.

Native HOLD was not confirmed in the recorded no-GPS failures: PX4 rejected AUTO_LOITER without the necessary global state. The adapter keeps recording while armed. Do not treat a HOLD request or a terminated flight process as confirmed safe disarm, and do not silently substitute POSCTL without valid manual input.

## Resource measurements

Trial 31 NAV processes, including the spawned inference worker, peaked at 1,997.83 MiB combined PSS and averaged 358.17% of one desktop CPU core. These figures exclude simulator, GUI and independent recorders. They are development-host measurements, not Orin Nano 8 GB qualification. Real OAK measured depth, synchronized drivers, JetPack compatibility, power mode, thermal behavior and sustained latency still need board-side validation.

## Conditional startup-floor domain

With the explicit conditional-free option and configuration both enabled, the
startup lower support enclosure assumes a static, continuous, near-horizontal
supporting floor throughout the already declared local launch box. The actual
ToF, mechanical robot geometry, upright attitude and home-plane consistency
still must pass. This is additional model knowledge, not a measured floor hull
or course CAD. The allowable center domain is the launch box eroded by the full
vehicle and the forecast uncertainty; increasing uncertainty shrinks this domain.
The earlier radius `error + reference_error + tracking` incorrectly made support
area grow with uncertainty and was rejected by actual trial 35. Default task mode
retains the original restricted support behavior. This condition must not be
used over holes, stairs, flexible ground or an unverified launch support area.

## Native land-detector ZUPT notification experiment

Trials 34–36 enabled early cancellation of startup ZUPT after native
landed=true→false. The native detector declared not-landed before ToF confirmed
physical ascent. Trial 36 stopped at a measured stopping projection outside the
unchanged launch box. This does not prove its estimated lateral velocity was
false, but it does not establish an advantage for early mode release.
`NATIVE_LAND_ZUPT_RELEASE=1` now retains this experiment as explicit opt-in; the
default returns to the existing actual ToF departure notification. Trial 37
retains the full certified survey target, 100 ms forecast and corrected support
domain while testing this single mode-release difference.

The requested 10AM Los Angeles full-course acceptance deadline was not met.
No current episode is represented as an autonomous full-course success.

## Robot center of gravity and PX4 allocation

The 1.1 kg base, four 4 g rotors and 50 g forward camera/IMU assembly sum to
1.166 kg. The 0.22 m forward mounting displaces total CG by about 9.434 mm.
The old start script supplied symmetric ±0.115 m rotor arms around the body
origin. Pinned PX4 `control_allocator/module.yaml` defines CA_ROTOR positions
relative to **center of gravity**, so these were the wrong physical lever arms.
`robot_allocation.py` now resolves only robot link/inertial frames, sums masses
and sets FRD rotor positions relative to that CG. It preserves motor numbering
and physical spin/moment ratios. No aircraft mass or course geometry is changed.
A static force/moment test confirms positive torque-balanced hover allocation
and the pitch moment produced by the old symmetric collective allocation.
Trial 38 is the first flight verification of this correction; the analytical
check alone does not establish flight success. Hardware must supply measured
masses/CG and rotor geometry, rather than using the SDF mass assumptions.

## Matching the actuator thrust curve

Trial 38 changed the drift direction after correcting the CG lever arms. The
remaining interface mismatch is that PX4 allocates normalized thrust while the
SITL ESC previously mapped it linearly to 150–1000 rad/s (`THR_MDL_FAC=0`), with
physical force proportional to angular speed squared. Correcting geometric
lever arms without that thrust curve overcompensates asymmetric rotor loading.
For this explicit SITL motor model, the integration now uses PX4's existing
inverse-square-root thrust shaping (`THR_MDL_FAC=1`) with a zero-to-model-maximum
ESC speed range. `MPC_THR_HOVER` follows total robot weight divided by total
modeled maximum rotor thrust (0.571728 for this assembly). The controller's
armed minimum thrust remains separate. A physical force/moment test includes
the actual quadratic motor curve and rejects the old nonlinear torque balance.
These are model/driver calibration parameters. Hardware requires a measured
ESC/motor/propeller thrust curve and hover calibration; these SITL values are
not hardware tuning or a navigation shortcut. Trial 39 tests the corrected
interface without changing the aircraft, sensors or original course.

## First verified startup and tracked-heading frontier policy

Trial 39 completed the vertical bootstrap and entered TUNNEL_ENTRY_SEARCH.
At estimated height 0.527 m its actual downward range was 0.615 m, with 12
accepted visual constraints and a 48 ms update age. It issued a sensor/map-derived
first frontier `[0.05, -0.05, 0.45]`, then failed localization during a turn.
Recorded yaw moved from about -0.56 to -1.40 rad while accepted constraints fell
from 12 to zero; health became unobservable and native LAND intervention was
recorded. The episode is incomplete.

The complete-task frontier selector now retains the current constrained view
when a fresh full-body/uncertainty path certificate already covers the entire
short translation. It requires at least four accepted constraints and update
age at most 0.20 s. Unknown paths do not acquire this exception and still need
the calibrated observation-facing policy. This reuses the baseline known-space
translation policy and avoids an unnecessary pure turn toward a weak view.
Both known and unknown map cases are tested through actual adapter methods.
Trial 40 verifies this change on the original course.

Trial 40 confirmed a second policy-order bug: a rejected frontier candidate
changed the held yaw before its short motion received a certificate. The
controller could therefore turn toward a candidate that it never flew. Trial
41 moves heading selection after authorization and applies the known-space
view policy to the actual certified short step. A callback regression proves
that a rejected candidate cannot change the hold heading.

Trial 41 reached the first online frontier and retained visual tracking through
2,824 audited motion segments. It remained near the start and ultimately
rejected an unavailable current reference/stopping volume; native LAND was
commanded and disarm confirmed. It did not traverse the maze. Its backend
recorded 260 conditional publications, so absence of all inference was excluded.
The planning worker now uses a bounded clearance-eroded copy of the actual
map, matching the final uncertainty margins before selecting candidates. If
horizontal frontiers are exhausted, it reuses the existing connected, known-free
3D viewpoint fallback. Sensor-derived guideline segments only bias scores.
Unknown ceilings and uncertified moves remain blocked.

The display alignment callback now queues up to 16 original SLAM pose stamps
that arrive ahead of accepted odometry, and publishes its single static gauge
only once actual estimates bracket one of them. Previously every future pose
could be dropped, leaving RViz unable to transform the optimized SLAM map. A
no-DDS callback test verifies original-stamp preservation, deferred publication
and lack of gauge changes. This TF alignment remains visualization-only.

## Native EGO execution audit

The parallel audit found two reproducible integration defects. Fusion advanced
its input timestamp watermark before rejecting an invalid or future-stamped
message; normal subsequent measurements could then be suppressed. The watermark
now advances only after the existing input checks pass. A no-DDS regression
executes the actual callback and checks recovery and unchanged valid outputs.

The pinned EGO collision callback entered EMERGENCY_STOP on depth timeout but
continued into stale-map replanning, which could overwrite the emergency state.
An immediate return preserves EGO's native emergency handling. The test in
`scripts/verify_ego_depth_timeout.py` compiles the actual callback body and checks
eight cases; removing only the new return makes its negative control fail.
The updated Python suite passes 297 tests. These are code-level results, not a
new flight or course completion.

The audit also confirmed that `traj_server/time_forward` affects yaw lookahead
only; position, velocity and acceleration share the current trajectory time.
The native planner's base body envelope and the task adapter's additional
uncertainty margins differed at this checkpoint. The opt-in integration below
passes the task adapter's actual extra XYZ to native collision queries while
preserving unknown-space and actual-velocity stopping checks.

The user also approved switching the PX4 primary height reference to the
existing visual/inertial estimate, retaining the downward ToF for near-ground
and contact observations. The v5 startup profile now sets EKF2_EV_CTRL=15,
EKF2_HGT_REF=3 and EKF2_RNG_CTRL=0. The competition runner verifies these actual
values before arming. Existing barometer configuration is unchanged. This
avoids treating a new supporting surface as the local-height datum, but does
not remove the need to monitor VIO drift or validate hardware time alignment.

## Fixed execution frame and native collision envelope

The EGO-to-PX4 bridge now establishes one rigid BODY/world-to-native-NED
transform after the existing measured stationary, disarmed ground qualification.
It transforms EGO position, velocity and acceleration consistently. It no longer
reanchors on every command, clips EGO position independently, or scales velocity
without the corresponding trajectory. Ordered native XY/Z reset deltas translate
the bridge once. Heading/velocity resets, missed resets, source clock restarts
and unannounced jumps stop execution and require requalification. Holding a
rejected frontier retains a fixed brake target instead of following each newly
estimated BODY position.

An actual disarmed Ubuntu ground probe qualified this bridge and retained it
for three seconds without an alignment fault. It sent no ARM or setpoint. This
checks integration and continuity at rest; it is not a physical accuracy bound
or an in-flight result. The navigation-only entry also requires measured
stationary disarmed/landed evidence before priming; the complete-task entry
continues to use its mechanical startup observer.

`EGO_REQUIRE_COLLISION_ENVELOPE=1` enables the native gate at stack startup.
The corresponding task/runner flag is `--require-ego-envelope`. Both are
explicit and default off. The runner verifies the actual native ROS parameter
before launching flight and records input and acknowledgment topics. The atomic
protocol carries session, sequence, issue/deadline nanoseconds and actual
`guard.extra` XYZ. Its deadline cannot exceed the existing map/causal-forecast
deadline or 100 ms. Native collision queries add this extra exactly once to
the existing body envelope. Expired results are not published and unpublished
trajectory state is rolled back. Missing/expired input invokes native emergency
handling. The existing depth-timeout fault remains latched until a separate
recovery; receipt of an envelope alone does not clear it.

Offline checks compile actual native query/replanning/collision callback bodies,
exercise protocol rejection and strict ROS/steady deadlines, and verify negative
controls. Ubuntu builds all six required native packages. The native live map
and the task's map snapshot are still independent: this integration unifies body
volume semantics, not map versions. The final task guard remains required.
Actual planning latency and in-flight behavior require a recorded original-course
trial. No new sensor, geometry truth or course waypoint enters these interfaces.

The actual disarmed transport probe sent 187 records from the live sensor/map
guard and received 187 matching ACKs with no rejection. ROS-clock ACK latency
was 4 ms at p95 and 12 ms maximum; this is not hardware or wall-clock latency.
The probe prohibited actuation and route methods. Only 93 of 187 immediately
following goal checks passed because current uncertainty often exceeded the
previous ACK slightly. New goals and resends therefore wait for a covering ACK;
following an existing trajectory retains the existing current-map execution
certificate instead of clearing the trajectory on every ACK fluctuation.
A regression verifies that an obstacle inside only the additional body margin
still prevents a new setpoint. The integrated Python suite passed 318 tests;
the subsequent follow-path correction passed its isolated adapter regression.

Trial 43 enabled this interface on the original course. Startup completed,
two online goals were issued, and native EGO published three trajectories before
emergency braking. The fixed bridge retained continuity through 866 audited
segments. The episode still failed near the start and required operator native
LAND; actual landed/disarmed status was confirmed. Both videos were retained.

The exact recorded atomic map at observation time 275.500 s, published at
275.584 s, reproduces the controller's decision at 275.684 s. Current BODY and
actual stopping volume are free. Only the next reference segment fails: its
full additional margin intersects 63 unknown cells in one lateral column.
The other two checked paths pass. A native emergency-stop reference also appears
in the recording. This exposes an adapter recovery defect: rejection of a future
reference ends the episode even when a certified current-body brake is still
available. It does not justify reducing the robot size, uncertainty margins or
unknown-space veto. The targeted fix now recertifies the current body and its
actual stopping path before using the existing fixed brake. It drops the rejected
reference and waits for a new native trajectory ID. The fallback's published
setpoint receives its own current certificate; recursive failure or an unsafe
current body/stopping path retains the fatal handoff. Six new callback cases and
the existing adapter/envelope/mechanical regressions pass. The unchanged recorded
map reproduces the candidate-only rejection and safe-brake branch conditions.

Trial 44 completed startup and issued two online goals, then rejected an unsafe
actual stopping volume. In this episode the three checks were `[true, true,
false]`: candidate position and command-velocity lookahead passed, actual
measured-velocity stopping did not. The new candidate-only fallback therefore
correctly remained unavailable. Native LAND intervention was recorded and actual
landed/disarmed status confirmed. Input isolation passed; no truth was fed to
navigation. The archive now holds 44 episodes and zero full-course successes.

The frozen trial-43 packet contains 779 measured-free, 541 inferred-free and
759 declared-launch voxels. Under that packet's complete body/uncertainty volume,
only 12 grid centers are safe. The continuous center region is also confined to
approximately X/Y +/-0.06789 m and Z 0.3921--0.7079 m. Inference does not expand
this component. This is a coverage limitation, not a claim of physical collision
or a reason to mark unknown space free. A same-XY upward observation candidate
is geometrically available but omitted by the grid-center sampler; its usefulness
for acquiring new depth still needs runtime validation.

The user subsequently confirmed that the real walls have clearly visible
texture. The `source-walls` simulator profile is a separate representative
textured-wall condition. It reuses a pinned CC0 material and preserves the
original course collision geometry, openings, floor, task markings and sensors.
It is not a photometric reproduction of the real walls. Passive stereo still
computes depth from the left/right images. Appearance provenance is recorded
only in the independent evaluation area and copied into episode artifacts;
navigation cannot read it. Trials 43 and 44 retain the original weak-texture
condition for comparison.

Generate the separate appearance condition from the existing original assets:

```bash
python3 /work/cartographer/scripts/build_wall_texture_scene.py \
  --source-root /work --output-root /work/wall-texture-candidate
python3 /work/cartographer/scripts/verify_wall_texture_scene.py \
  --source-root /work --candidate-root /work/wall-texture-candidate
```

The source root must contain the original course model, world and pinned
`sim/assets/v5-wood-floor.jpg` asset. Install only the generated new model and
world resources into `sim/`; do not replace the original course. Select
`start.sh source-walls` after confirmed disarm. The photographic material uses
the asset's published 1.7 m width as a representative UV tile size. It adds no
fiducials and does not establish real-venue texture scale or depth accuracy.
