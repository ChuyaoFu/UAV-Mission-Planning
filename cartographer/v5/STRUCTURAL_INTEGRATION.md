# Structural mapping and complete-task integration — 6 November 2026

This checkpoint has **not completed the original competition course**. No hardware flight or Jetson Orin timing has been validated. Ubuntu development tests use the original course, GPS disabled, the existing stereo/IMU and downward ToF, and estimated odometry. Course geometry and simulator poses are confined to independent evaluation.

## Selected mounting and reusable components

The current trial uses a rigid 20-degree downward camera/IMU mounting with zero roll, selected before adapting the physical fixture. The checked-in general profile remains 15 degrees downward for a controlled comparison. Physical camera/IMU/body/ToF calibration is still required. OpenVINS, PX4, EGO-Planner and Cartographer remain the reused estimator, controller, local planner and visualization components. OpenCV GrabCut and line detection provide conditional scene geometry; their integration and task executive are project-specific code.

## Conditional geometry model

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
