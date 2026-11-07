# CAD-free competition implementation plan

## Scope and permitted prior information

The user accepts the explicitly declared local takeoff clearance and the task order/target categories. Preserve the original competition world's geometry and appearance. Navigation must not receive course CAD, absolute spawn coordinates/yaw, surveyed task coordinates, wall/window dimensions, a turn sequence, or evaluator state. Home is the first measured body pose. Simulator geometry, spawn placement and native truth remain simulator/evaluator inputs only.

The current aircraft and 20-degree camera/IMU mounting are simulation approximations. Actual mounting, rectified intrinsics, full extrinsics, timing, mass/inertia and braking performance remain unverified. Sensor types alone do not establish equivalence to the real aircraft. Keep one calibrated robot/sensor configuration for planning and validation rather than tuning the scene to the controller.

## Shared architecture

- Continuous OpenVINS/IMU odometry and measured range support a control frame separate from globally optimized mapping.
- Confidence-filtered stereo and multi-view observations build occupancy. Unknown and occupied space remain distinct; a semantic detection never declares an opening collision-free.
- A rolling local collision map and a persistent keyframe/topological graph replace the current fixed 8 m window as the complete-course representation. Fine local maps resolve narrow openings without refining the entire world.
- EGO supplies trajectories. Full-body clearance, latency/braking bounds, observability, fixed braking anchors and fresh-trajectory gating constrain execution.
- Task transitions use measured target-relative geometry and PX4 estimated flight/landing state. Independent simulator evidence scores the run and never triggers a navigation transition.

## Challenges

| Challenge | Observation and action | Online completion evidence |
|---|---|---|
| Takeoff | Initialize stereo/IMU, establish the measured home frame, and use only the declared local launch clearance. Choose observation maneuvers that retain real image features. | Healthy measured pose and stable hover; no imported world alignment. |
| Corridor and obstacles | Estimate visible wall/obstacle boundaries and traversable space from stereo. Use observed guidelines as relative direction cues. Plan inside measured clearance and choose camera headings that retain useful features. | Measured traversal of the observed corridor and acquisition of the next task's entrance. |
| Maze | Build junction/corridor connectivity online, retain explored branches, distinguish a measured dead end from insufficient observation, and select useful unexplored branches. Backtrack through the observed graph when needed. | Reach an observed exit/next-stage target through a connected traversed route, not an annotated maze coordinate. |
| Tunnel | Detect the rim/opening, estimate its plane, axis and cross-section using multiple stereo views, align the whole vehicle, and plan through verified free space. Maintain entrance/exit keyframes and bound inertial propagation when texture drops. | Cross the observed exit plane with the complete vehicle and reacquire a stable exterior view. |
| Window or allowed alternate gate | Detect candidate openings, triangulate their boundaries, and compare measured clearance/uncertainty against the calibrated vehicle envelope. Refine only the opening neighborhood. Choose an allowed alternate route if the window cannot be verified passable. | Whole-body passage through the measured opening plane; no fixed center/height/width or hidden alternate-gate coordinate. |
| Destination landing | Estimate the target plane and center from its observed markings/edges. Use visible target edges and neighboring landmarks during descent. Treat range-to-platform separately from the persistent altitude reference. | PX4 estimated landed state followed by actual disarm; hovering near a target is insufficient. |
| Rearm and over-wall return | Preserve home, VIO and the observed graph across disarm/rearm. Identify the relevant wall from observations, measure its upper boundary and the crossing/ascent clearance, and plan a feasible crossing toward home. | Entire vehicle clears the observed wall boundary and reaches the home-side observed region. No stored wall height or surveyed climb corridor. |
| Final home landing | Combine the saved measured home position with geometrically verified visual relocalization against own keyframes; align using launch-area observations and descend with range support. | PX4 estimated landing and disarm at the measured home target; independent scoring checks physical landing error. |

## Required failure behavior and visibility limits

A blank image or missing stereo is insufficient evidence for a wall or free space. Improve the view, use additional measured viewpoints, backtrack or abort within the localization/clearance budget. Repetitive corridors require visual/geometric verification before accepting loop closures. Global map changes must not directly jump the control frame.

The existing forward/downward camera cannot be assumed to see directly above or below the aircraft. Over-wall ascent requires a real observable clearance path, potentially using observation positions and feasible vehicle-attitude maneuvers. The actual camera mounting must be checked before claiming this is achievable. No ideal upward sensor, invisible gimbal, known roof height or blind vertical ascent may substitute for that check. Landing likewise requires a verified final-approach visibility envelope.

## Orin Nano 8 GB

Use the OAK's actual calibrated/confidence-filtered stereo output on hardware. Keep image/feature/keyframe queues bounded, schedule task perception at a bounded lower rate than control, and keep simulator/RViz/recording offboard. The tested CPU stereo proxy retains native imagery for VIO and supports reduced-resolution matching with calibrated focal/disparity scaling. Desktop PSS/CPU measurements are development evidence only; JetPack compatibility and sustained Orin latency/thermal/memory measurements remain required.

## Implementation and validation order

1. Work in the unchanged original competition maze. Repair its currently observed VIO dropouts and measure stereo coverage before extending motion. The first current source-course run aborted on localization recovery exhaustion and is not a successful course run.
2. Add relative guideline/entrance perception and online maze connectivity, then demonstrate corridor/maze traversal without imported coordinates.
3. Add tunnel and opening geometry, including local map refinement and whole-body passage checks.
4. Add target-relative landing and persistent state across disarm/rearm.
5. Validate observable wall crossing and visually corrected home return.
6. Repeat the complete sequence, retain failures and successful recordings, then optimize end-to-end time. Short-room smoke tests do not establish competition completion.

The current v5 implements the navigation foundation and short exploration/return tests. Most task-specific perception, full-course state transitions, verified visual relocalization and over-wall clearance handling remain to be implemented and validated.
