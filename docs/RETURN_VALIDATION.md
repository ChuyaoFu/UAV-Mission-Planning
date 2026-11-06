# Destination landing and over-wall return validation

The task is an engineering extension: complete the outbound course, land and disarm on the destination platform, rearm without resetting VIO/SLAM, fly over Barrier 1 and land/disarm at the surveyed launch area. This does not certify official competition scoring.

## Recorded episodes

| Episode | Controller | Independent judge | Contacts | First motion to final disarm |
| --- | --- | --- | --- | --- |
| return-01 | Aborted during descent; safely landed | Failed | 0 | 100.124 s |
| return-02 | Aborted after slow depth triggered continuous conservative speed | Failed; outbound incomplete | See retained raw evidence | Not a complete task |
| return-03 | Complete, no controller failure | Complete, both landings and wall crossing verified | 0 | 168.540 s |
| return-04 | Aborted during fast descent when observation displacement exceeded its budget | Retained as a failed controller run | See retained evidence | Not a successful controller run |
| return-05 | Aborted when low-texture descent view became unobservable | Retained as a failed controller run | See retained evidence | Not a successful controller run |
| return-06 | Complete with final four-thread, bounded-vertical, corner-view configuration | Complete, both landings and wall crossing verified | 0 | 118.548 s |

`return-01` exposed a descent too close to the wall's inflated margin; the return now descends 0.35 m farther inside the start side before a low-altitude alignment leg. The physical wall and planning margins were retained. Its original judge additionally rejected ground contact during second-arm motor spin-up; the judge now uses measured second takeoff time instead of a one-second guess, with a regression case. That correction does not change the controller failure or turn this episode into a successful control run.

`return-02` measured stereo receipt age of 252 ms median and 356 ms P95. The original synchronous matching callback accumulated old image frames, repeatedly forcing 0.18 m/s conservative cruise. It reached a per-stage timeout before completing the outbound route. Stereo was moved to a bounded latest-pair worker with depth-one image subscriptions. Stage time budgets account for the conservative speed; the overall 175 s control deadline and 180 s engineering scoring bound remain.

`return-03` used one matcher thread. It completed the task in 168.540 s, with final center 0.0376 m from the launch center and at least 0.3005 m vertical clearance over the wall across the recorded whole-body crossing region. The continuous flight-plus-SLAM recording is 205.0 s and includes preparation, both landings and post-disarm footage without acceleration. The mission's arm-confirmed-to-final-disarm runtime was 171.632 s.

Its optimized SLAM node RMSE was 0.1539 m. The measured structural cloud contained 7,206 points, with 0.0551 m median and 0.1857 m P95 distance to the independent CAD reference. These are sparse observed surfaces, not complete reconstruction or proof that unobserved space is free. No loop closures were accepted in this episode.

`return-04` used the faster matcher and exposed a distant vertical position target that could move more than 0.30 m during a brief stereo dropout. Return vertical commands are now limited to 0.18 m steps, reduced to at most 0.06 m on stale depth, with a 0.25 m displacement brake reserve. The existing 0.30 m/4 s observation veto remains unchanged.

`return-05` exposed a separate observability failure: its recorded descent camera pointed predominantly at a blank surface, with zero accepted visual features and no stereo structure. The controller correctly refused to coast past the localization uncertainty bound. Descent now first faces a declared surveyed corner (M1 divider endpoint, derived from CAD bounds by name) and holds altitude until the heading is acquired. This changes camera orientation, not the sensor model, texture, or pose source.

## Final implementation: return-06

The final configuration completed the entire task in **118.548 s**, with **zero contacts** and both actual landing/disarm cycles verified. Arm confirmation to final disarm took 121.684 s. The final body center was 0.01725 m from the launch center. Across the recorded whole-body wall-crossing region, the minimum vertical clearance above the wall was 0.30076 m. The actual synchronized window recording is 141.4 s including preparation and postflight, without acceleration.

Stereo receipt age was 60 ms median and 76 ms P95, measured from the original MCAP reception timestamps. This is a desktop measurement, not a target-hardware benchmark.

Optimized-node RMSE was 0.05950 m, while full-recording online fused-control odometry RMSE was 0.27253 m. The CAD/observed-line/window-corrected navigation body pose used by the mission had 0.05951 m RMSE against independent truth. These are distinct quantities; the navigation frame is CAD-aided, while the SLAM display is reconstructed from measured points. The actual structural map contained 7,773 points, with 0.03368 m median and 0.11974 m P95 distance to the independent CAD reference.

The episode source hashes match the current implementation files exactly. See [the machine-readable summary](validation/return-mission-summary.json). The 26 core tests and 10 existing synthetic acceptance checks passed. A single flight with the final configuration has been completed; repeated identical-configuration reliability and completion within 60 seconds have not been established.

## CPU scheduling comparison

Three actual recorded stereo pairs were processed with the identical SGBM matcher and the same left/right, photometric and depth validity thresholds. Median compute time was 156.15 ms with one thread, 96.16 ms with two threads and 61.83 ms with four threads. All three tested depth outputs were identical, including invalid pixels. These small controlled measurements motivated configurable `STEREO_THREADS` (default 4 on this development host); they are not an Orin benchmark or a statistical claim over the whole flight.

The real-time subscription audit and native PX4 input audit recorded no protected consumer of evaluator pose or ideal depth, `EKF2_GPS_CTRL=0`, `EKF2_EV_CTRL=13`, and no published `sensor_gps`. The independent judge/evaluator and MCAP recorder retain truth solely for read-only verification.

## Evidence locations

Original Ubuntu evidence is under `/home/fcy/visual_nav/cartographer/v4/results/<episode>/`, with original MCAP data under `/home/fcy/visual_nav/cartographer/v4/data/<episode>/`. Successful videos and selected reports are also retained locally under `outputs/return-home/`. Video/raw sensor files remain excluded from Git.

A real-airframe deployment has not been validated. See [the navigation audit](NAVIGATION_AUDIT.md) for surveyed-start/CAD assumptions, blind overhead regions, calibration requirements, sensor proxy limitations and remaining hardware work.
