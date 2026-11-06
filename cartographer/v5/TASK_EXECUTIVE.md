# CAD-free task executive v2: measured foot calibration and terminal contact

This is an independent source/patch delivery. It adds physically executable landing/liftoff and forward-only crossing evidence to v1. It neither controls PX4/a simulator nor edits the primary checkout or active deployment. All recorded examples are synthetic unit fixtures. No original-course completion, real sensor recording, Orin timing, or hardware readiness is claimed.

Apply `integration-v1-to-v2.patch` to an unchanged v1 integration. `integration-from-snapshot.patch` is provided for a clean pre-v1 snapshot. The primary chat owns `navigate.py` wiring, native mode acknowledgement, flight, global-height-observer policy and scoring. Neither patch changes `navigate.py` or `online_map.py`.

## What changes

- `task_landing.py`: `RestingFootCalibrator` consumes actual range, calibrated ToF mounting/ray, synchronized accepted body pose/attitude, stationary velocity/gyro and confirmed disarmed PX4 landed status. It requires at least 20 continuous samples spanning one second, rejects unstable/moving/tilted/stale/reset data, and supplies no fallback offset. Exactly upright/down-looking geometry gives `foot_offset = range - tof_in_body.z`.
- `surface_relative_observation` reports actual ToF hit, plane residual, foot clearance and platform height above the own observed start ground. It NEVER changes VIO position/height or treats a platform as a global-z correction.
- `task_contact.py`: a `SupportPatch` from dense direct native stereo and a `ContactLandingAdapter` separate measured support contact from ordinary free flight. Native LAND and initial liftoff receive short-lived, continuously monitored certificates. Every global map array remains unchanged.
- `task_surfaces.py`: explicit H, cross, and general closed-mark categories; an unmarked own launch surface can be reobserved with `HomePlaneDetector`. Generic mark shape/plane consistency remains a heuristic, not a semantic guarantee of the competition endpoint.
- `task_safety.py`: existing padded `OnlineMap.extent` is used once. `tracking_margin_m=.04` is ONLY an actual-motion audit tube. Extra geometric inflation is separately named `extra_collision_margin_m` and defaults to zero.
- `task_observations.py`: fresh direct-stereo `ForwardEvidence`. A crossed rim need not be seen by a nonexistent rear camera. Continuous unchanged-gauge measured VIO, prior complete-body map certificates, signed full-body crossing with pose/plane errors, and fresh far-side forward depth PLUS a newer map provide the alternative evidence branch.
- `task_executive.py`: additive optional contact/forward arguments, explicit native LAND/takeoff intents, measured calibration installation, platform-height staging before horizontal approach, retained contact/arming/liftoff proof, and unmarked-home perception role. Measured rim boundary axes avoid incorrectly requiring a window's vertical clearance to equal the rotor diameter; the existing complete-body map remains authoritative.

## Startup calibration: no guessed foot offset

The ToF origin/ray and their error budgets MUST come from actual robot/sensor calibration. The calibration input position is BODY local ENU, not the IMU lever-arm position. Input stamps share one clock domain. Do not use simulator truth or a synthetic fixture's offset.

```python
from task_landing import RestingFootCalibrator, RangePose
from task_executive import TaskExecutive, MissionLimits
from task_perception_helper import TaskPerceptionHelper

calibrator = RestingFootCalibrator(
    measured_tof_in_body_m, measured_tof_axis_body, actual_robot_calibration_id,
    range_error_m=validated_tof_error_m,
    extrinsic_error_m=validated_tof_extrinsic_error_m)

# On each NEW actual ToF sample, with body pose interpolated to its timestamp:
range_pose = RangePose(
    stamp=tof_stamp, range_stamp=tof_stamp, pose_stamp=interpolated_body_stamp,
    range_m=actual_range, position_local_m=accepted_body_local_enu,
    local_from_body_rotation=accepted_body_rotation,
    velocity_local_mps=accepted_body_velocity_enu,
    angular_velocity_body_radps=accepted_body_gyro,
    epoch=preserved_vio_epoch, landed=confirmed_px4_landed,
    armed=confirmed_armed, range_valid=actual_valid_range_status,
    tracking_valid=accepted_tracking)
calibrator.ingest(range_pose, now_same_clock)
foot = calibrator.result(now_same_clock)  # None means measurement not ready.

# Keep the own initial body pose and gauge through landing/disarm/rearm.
mission = TaskExecutive(MissionLimits(map_padded_radius, map_padded_halfheight),
                        own_recorded_resting_body_pose, preserved_vio_epoch)
if foot is not None:
    mission.install_foot_calibration(foot)
perception = TaskPerceptionHelper(foot_calibration=foot)
```

`ingest` requires valid tracking/range, disarmed+landed, upright within two degrees, low measured linear/angular velocity, exposure skew <=20 ms, fresh samples and no movement/reset through the continuous window. Rejection clears the window. `result` expires if sampling stops. The immutable accepted physical calibration remains usable through rearming; it is not recalibrated on an unknown platform or reset to a different VIO gauge.

`foot.resting_ground_z_local_m` comes from the own measured resting body pose and actual range. It is not a surveyed ground height. The body-frame ground offset may be smaller than the map's padded half-height; v2 handles that through a contact primitive instead of rejecting correct robot calibration.

## Capture and freeze a measured support patch before native LAND

First observe a stable horizontal-plane track from synchronized mono/direct stereo/local poses. Task role `dock` accepts H/cross/closed-mark proposals by default; the caller may narrow the policy when the actual target class is known. `home` uses the own launch measurement rather than requiring a marker. `TaskPerceptionHelper(foot).observe(..., role='home', ...)` fits CURRENT direct stereo near the own measured start and rejects a raised surface or mismatched gauge.

Before moving over the platform, capture the ENTIRE required support patch while it is in view. A front/down camera may lose that patch below the body, so do not wait until touchdown to acquire it.

```python
from task_contact import build_support_patch, ContactLandingAdapter, LandingContext

adapter = ContactLandingAdapter(measured_tof_in_body_m, measured_tof_axis_body,
                               actual_robot_calibration_id)
required_halfwidth = (latest_online_map.extent[0] + guard.extra
                      + latest_online_map.res + adapter.max_lateral_error)
support = build_support_patch(
    direct_measured_depth, matching_rectified_K, exposure_packet,
    stable_measured_landing_track, now_same_clock,
    required_half_width_m=required_halfwidth,
    calibrated_depth_error_m=validated_direct_stereo_error_m,
    measured_mask=strict_direct_provenance_mask)
# None is a refusal; do not substitute a hull, inferred depth, or older target.
```

The plane comes from measured local geometry, not a level/height/size preset. `build_support_patch` projects an enlarged square through the actual exposure transform and requires direct valid depth at EVERY covered native pixel, max residual within the supplied calibrated error budget, adequate native ray spacing, stable observations, upright plane and known calibration. A single missing/inferred pixel, foreground protrusion, far-background hole, edge, inadequate image coverage or low confidence rejects the patch. This strict checker may refuse legitimate low-texture platforms. That is an operational limitation requiring actual frames/viewpoint work, not permission to fill holes with completion depth.

Only two native ray pixels are added at the raster boundary. The caller's required width already includes existing robot/pose/tracking padding; another hidden pose-margin copy is not added.

## Native LAND and monitored contact liftoff

Feed new accepted RangePose/Feedback in the same VIO gauge at >=10 Hz. Interpolate body pose to the ToF stamp. `Feedback.stamp`/body pose must match the supplied RangePose within 25 ms/2 cm; do not repeatedly attach one old range to newer moving poses. Existing native control and stopping gates continue running at their own normal rates.

```python
context = LandingContext(adapter, support, foot, latest_synchronized_range_pose)
intent = mission.tick(
    measured_feedback, latest_tracks, guard, now_same_clock,
    forward_evidence=latest_frame.get('forward_evidence'),
    landing_context=context)
```

During `DOCK`/`HOME` landing or `REARM`, a missing/invalid contact context causes hold. Accepted context derives calibrated foot clearance and body-up from the actual synchronized range/attitude packet; it never alters body position. Supplied foot clearance inconsistent with that measurement is rejected. The active target/home plane must match the support proof.

| Intent | Primary navigator responsibility |
| --- | --- |
| `land` | Request the existing mature PX4 LAND mode. After native mode confirmation, stop conflicting offboard position setpoints. Refresh the contact monitor every accepted range/pose tick. |
| `takeoff` | After confirmed rearming, execute the existing controlled/native takeoff using its liftoff certificate and normal speed/attitude gates. Do not blindly fly through unknown overhead volume. |
| `hold`/monitor rejection | Abort/brake LAND or takeoff using the primary navigator's actual mode-aware safe control path. Merely publishing an ignored offboard setpoint while PX4 remains in LAND is not an abort. |
| `arm`/`disarm` | Use the existing command/acknowledgement path. Subsequent actual feedback must confirm the result. |
| `move`/`observe`/`explore` | Keep v1's ordinary map/stopping/external-step audit contracts. No contact exemption is used for corridor, tunnel, window, wall or horizontal transit. |

A descent certificate expires after at most 250 ms and needs a fresh map; support geometry can be retained for at most 10 seconds in the existing STATIC-environment assumption. The stamp is never refreshed by copying old data. If approach/landing/disarm/rearm outlasts that window, acquire new measured support evidence or hold. There is no time-based success fallback. A moving obstacle/newly opened hole under an occluded body is not covered by this static-patch assumption.

Monitors reject missing/stale/invalid ToF, gauge change, support disagreement, excessive tilt/gyro, lateral drift, excessive horizontal/descent/ascent speed, excessive altitude/contact penetration, wrong calibration, unknown queried body cells, unrelated occupied cells, unsupported edges, or unconfirmed arming. Stable actual landed status, measured speed, upright attitude, calibrated near-zero foot clearance and plane/body agreement must persist for the dwell before disarm. Native commands alone never establish landing or rearming.

### Exact scope of the contact exception

The map already contains a padded body envelope. Ordinary AABB queries reject a legal final contact because their artificial lower inflation enters floor cells. The adapter locally clips that lower inflation to the specifically observed support/contact band and preserves the existing padded sides and top (including any explicitly requested extra collision margin). It permits an occupied voxel only if that exact voxel intersects the densely measured support-plane interval, and requires its whole XY footprint inside the verified patch. ALL queried unknown or non-support occupied cells reject. No `state`, `free`, integral-map, launch-box, plane, or confidence array is modified.

The returned certificate says `full_body_free=False`: known supporting contact is not falsely called free space. It records `free_upper_volume_and_measured_contact_verified=True`, the support proof, actual foot calibration, exempted support-cell count, map/time/epoch and vertical goal. Dense support proof is what rejects a small protrusion hidden in the same coarse voxel as the floor. These are bounded sensor/error-model checks, not formally validated material/load-bearing guarantees or subpixel-hole detection.

The same measured local primitive permits initial liftoff after confirmed rearming; its overhead/side volume is still map-checked. Above the contact region, normal free-flight map checks apply. Landing and liftoff never turn a floor region into a new launch-box exception.

## Forward-only post-crossing evidence

```python
# In the EXISTING LatestWorker; return a LIST containing the frame result.
frame = perception.frame(mono8, direct_measured_depth, K, exposure,
                         mission.desired_role, now_same_clock, direct_mask)
# handler return example: [frame]
# On control thread:
intent = mission.tick(feedback, frame['tracks'], guard, now_same_clock,
                      forward_evidence=frame['forward_evidence'],
                      landing_context=current_contact_context_or_none)
```

`frame` keeps the old detached-track contract and adds bounded direct 3-D measurements with exposure stamps/calibration/epoch. The forward branch completes a passage only after the same prior full-body crossing certificate and continuous no-reset accepted motion chain used by v1, a whole-body signed-plane crossing including pose/plane error, a NEW far-side measured camera view with distributed direct points beyond the plane, and a map newer than the crossing. Unknown/inferred/narrow-patch depth, old map, wrong-side camera origin, stale/skewed/reset exposure or lost motion continuity cannot satisfy it. Reobserving the same rim remains an optional equivalent branch; no blind 180-degree turn is requested.

Tunnel EXIT still requires a distinct measured exit plane. Crossing a single entry and collecting forward depth does not declare tunnel completion. No fixed tunnel length, turns, travel threshold or timer is used.

## Height-observer and coordinate contract

Freeze any ToF-derived GLOBAL z correction before platform approach/overflight, through landing/disarm/rearm/liftoff as the primary navigator's semantic-stage policy requires. Continue VIO/local-body tracking across disarm. These modules only report surface-relative observations and do not implement or override the global observer. A platform's height above the initial ground is meaningful; it must not be interpreted as VIO drift. Root owns the observer policy and its actual ROS/FCU wiring.

`surface_relative_observation(..., plane_center_local_m, plane_normal_local, target_id)` explicitly reports `observed_surface_z_local_m`, `surface_height_above_start_ground_m`, `foot_clearance_m`, and `plane_residual_m`, with `vio_global_z_correction_applied=False`.

## Evidence and verification

- **80 isolated tests pass**; **126 combined snapshot/candidate tests pass**. No old source tests were removed. The shared source snapshot has 54 regressions; the package has 72 added tests beyond the original eight opening tests.
- A REAL `OnlineMap` with occupied floor and unknown below it refuses ordinary padded free-space touchdown, accepts the measured contact primitive, and is byte-for-byte unchanged. Native LAND -> landed dwell -> disarm confirmation -> rearm confirmation -> contact-constrained liftoff is tested with actual-range-domain synthetic samples, preserved VIO gauge and retained proof records.
- Negative tests cover missing/inferred support pixels, holes/edges, a protrusion inside the same coarse floor voxel, unknown rotor periphery, other occupied body cells, overhead obstacles during takeoff, ToF/plane mismatch, stale map/support/range, drift/tilt/velocity, invalid landed/armed sequence, wrong robot calibration, gauge reset and unsupported landing semantics.
- A forward-only-camera trace passes without any rear rim observation; reset/wrong-side/old-map/inferred-depth alternatives fail. Existing padding is tested without automatic duplication. Rotated cross/H markers and an unmarked measured home plane are tested.
- `v2-evidence.json` contains SYNTHETIC contact/liftoff and forward-proof traces plus desktop timing, explicitly marked `original_course_flight_validated=false`, `real_sensor_recordings_used=false`, `orin_validated=false`. No actual competition success is claimed.

Standalone extracted package:

```sh
PYTHONPATH=reference:. python -m unittest discover -s . -p 'test_*.py'
PYTHONPATH=reference:. python benchmark_task_landing_v2.py --iterations 100 --output v2-evidence.json
```

`reference/online_map.py` is an unchanged testing dependency, absent from both patches. The primary integration must still validate actual sensor coverage, exposure timing, mode acknowledgement/abort behavior and original-scene flights. Orin measurements remain outstanding.
