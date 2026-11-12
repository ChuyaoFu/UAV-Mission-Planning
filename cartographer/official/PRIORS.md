# Priors used by the official-lab stack (audit, 2026-11-07)

Rule: no GPS, no CAD, no simulator truth in the flight loop. Allowed: coarse stage positions measured on
site relative to the start pad (about ±0.5 m), vehicle and sensor calibration, generic object-shape ranges.

## 1. Course prior JSON (the only place where stage positions live)
`mission/course_prior.*.json`, start frame (forward/left/up from the start pad), loaded at runtime:
fence box, cruise height, corridor waypoints, tunnel entry, axis and length, window centre and through
direction, landing-rock centre and top height, wall crossing points and top height, each with a tolerance.
- `course_prior.official_sim.json`: corridor and first turn estimated from the official follow camera;
  later stages are placeholders.
- `ubuntu_compat/course_prior.replica.json`: generated from `replica/layout.py`, which is the replica's
  TRUTH. Use it only as a labelled reference run. Normal sim runs use `perturb_prior.py` seeds
  (xy ±0.4 m, z ±0.1 m, ±8°) so tests never get a better prior than the team can measure.
- Real venue: the team measures these on site and writes a new JSON; no code change.

## 2. Measured at runtime (not priors)
Home position and yaw at arming; start-floor height (base_link at rest); camera K from CameraInfo;
camera mount from TF; ToF range; VIO pose; wall top from the depth map (it can only RAISE the prior).

## 3. Vehicle constants
`half_span_m` 0.20 (Sky Beta incl. props), clearance margins, speed and accel limits (`mission/core.py` Params).

## 4. Generic shape ranges in detectors (no positions)
- Tunnel: hoop radius 0.25–0.90 m. A measured radius or hoop spacing is used only if provided
  (default None). `bottom_gap_m` 0.08 is a guess from the photo.
- Window: opening 0.20–1.6 m, aspect ≤ 3, sill ≥ 0.15 m above the floor. A known size is optional.
- White rock: raised flat top 0.10–1.0 m high, extent 0.25–2.0 m, brighter than the floor
  (or the soft brightness mode).
- Floor: fit near the measured start floor, fused with the ToF.

## 5. Tuned on the photo-estimated replica (overfitting risk: re-check with real data)
`tunnel_standoff_m` 1.8 (now auto-shrinks to fit the fence, min 1.2), `observe_above_top_m` 0.55,
detector confidence thresholds, the window-size comment "~0.75 x 0.6 m", tunnel `bottom_gap_m`.
Replica dimensions are eyeballed from 7 photos.

## 6. Sim-only truth (never in the flight loop)
Ubuntu Gazebo truth is recorded in a side bag for analysis only. PX4 `SIM_GZ_EN_ODOM=0` keeps truth
out of EKF2, and `check_compat.py` FAILs if it leaks. `--ev-source gt` is a debug-only switch and is
never used for a reported run.
