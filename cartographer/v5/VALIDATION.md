# Experimental CAD-free navigation validation

This milestone tests online measured-stereo mapping, local obstacle avoidance, and return to the first measured body pose. It does not test the competition task sequence or demonstrate hardware readiness. All navigation runs disable GPS fusion and exclude simulator pose/depth, field CAD, and preset route targets. Native simulator poses and contacts are collected by a separate evaluator only.

## Unchanged original-course localization and first-stage tests

Primary integration now uses the original source-course geometry and materials. These episodes do not complete the maze or competition tasks:

| Episode | Result | Motion to disarm | Contacts | Independent excursion | Home error |
|---|---|---:|---:|---:|---:|
| course-cadfree-01 | Localization recovery exhausted | 37.936 s | 0 | 0.206 m | 0.128 m |
| course-view-02 | CLAHE and measured-view memory; insufficient expansion | 71.236 s | 0 | 0.257 m | 0.025 m |
| course-guideline-03 | Relative guideline integrated; insufficient expansion | 116.932 s | 0 | 0.416 m | 0.059 m |
| course-survey-04 | Two-sided yaw experiment regressed; rejected | 31.748 s | 0 | 0.151 m | 0.083 m |
| course-parallax-05 | Small transverse-view experiment; return timed out; rejected | 127.976 s | 0 | 0.278 m | 0.109 m |

`course-view-02` and `course-guideline-03` returned and disarmed without exhausting localization recovery, but they remained near the launch envelope. The third episode supplied relative line hints to five online goals without changing map occupancy or pose. Depth coverage at low-texture views was approximately 2–4%, leaving insufficient verified full-body clearance to enter the maze. The two-sided yaw experiment subsequently exhausted recovery during its low-height survey; it landed without contact and that policy was removed. Localization is therefore improved on some runs, not resolved across the original scene. A bounded transverse-view experiment inside the declared launch box did not improve exploration and later timed out returning; it also landed without contact. Its extra motion was removed from the active controller. The complete original-course task sequence and a 60 s result remain unachieved. Screen/SLAM and first-person recordings of these failed episodes are retained as well as earlier successful-room videos.

On the **same recorded original-course sensors** from `course-cadfree-01`, the patched OpenVINS executable (`sha256:2313d892b466fbf959fa4014f7aa33ed5293a03594c15febc2001766542d9402`) produced:

| Image preprocessing | First-pose-aligned 3D position RMSE | Visual age < 0.5 s coverage | Visual age p95 / max |
|---|---:|---:|---:|
| HISTOGRAM | 0.08314 m | 88.19% | 0.848 / 1.800 s |
| NONE | 0.07314 m | 86.91% | 1.100 / 1.948 s |
| CLAHE | 0.04690 m | 89.73% | 0.752 / 1.800 s |

All three comparisons contain 11,025 poses and 883 health samples. CLAHE was selected without relaxing the flight observability gate. A bounded memory of measured feature-rich views avoids forcing the camera towards a blank goal bearing; it is not a surveyed heading or route. This is one recording and does not establish general localization robustness. An earlier replay accidentally resolved an unpatched executable; its missing-health results were rejected and excluded above.

For `course-guideline-03`, navigation peak PSS was **1,233.56 MiB**, and combined desktop CPU was **242.8% median / 272.4% p95** (100% = one CPU core), excluding Gazebo/RViz/video/evaluation. This includes the new bounded guideline module. It is not an Orin Nano latency, GPU-memory or thermal measurement.

## Recorded results

The acceptance check requires one arm/disarm cycle, no obstacle contacts, at least 1.15 m independent horizontal excursion and 2.7 m travel, and landing within 0.20 m of the launch center. The controller independently requires 1.2 m excursion and 2.5 m measured outbound travel. These are short navigation checks, not competition scoring.

| Episode | Result | Motion to disarm | Contacts | Excursion | Home error |
|---|---|---:|---:|---:|---:|
| unknown-17-01 | Earlier prototype passed; superseded | 47.940 s | 0 | 1.486 m | 0.185 m |
| oak800-wood-17-01 | Insufficient exploration | 41.744 s | 0 | 0.249 m | 0.074 m |
| raycoverage-wood-17-01 | Insufficient exploration | 43.352 s | 0 | 0.149 m | 0.019 m |
| surveygrid-wood-17-01 | Strict measured-map version passed | 62.720 s | 0 | 1.309 m | 0.057 m |
| surveygrid-wood-23-01 | Failed: clearance/return stall, then watchdog cleanup fault | — | Incomplete evaluation | — | — |
| cancelverify-wood-23 | Expected cancellation; controlled landing/disarm verified | 18.232 s | 0 | 0.147 m | 0.066 m |
| heading-wood-23-02 | Earlier heading fix passed | 68.568 s | 0 | 1.308 m | 0.0049 m |
| heading-wood-17-01 | Regression failed: return stall | 77.140 s | 0 | 0.297 m | 0.254 m |
| brakepoint-wood-17-01 | Insufficient exploration | 38.160 s | 0 | 0.147 m | 0.011 m |
| latched-wood-17-01 | Fixed braking-anchor version passed | 75.788 s | 0 | 1.274 m | 0.023 m |
| cpu400-wood-17-01 | Historical CPU400 profile passed | 82.164 s | 0 | 1.288 m | 0.023 m |
| cpu400-wood-23-01 | Historical CPU400 profile passed | 85.448 s | 0 | 1.299 m | 0.021 m |
| cpu400-matte-17-01 | Failed: localization recovery exhausted | 72.192 s | 0 | 0.390 m | 0.246 m |


**Historical generated-room comparison:** the three `cpu400` episodes have identical source/configuration SHA-256 manifests.** The two successful rooms change the furniture and initial heading; the matte episode removes the wood appearance without changing navigation. The matte counterexample fails the task and is retained. Generalization across arbitrary materials, the original course and full competition tasks has not been demonstrated. The current 82–85 s short-navigation runs do not meet a 60 s target. Lower stereo compute latency did not produce a shorter total mission in this comparison.

The early prototype used a weaker stereo confidence gate and an incorrect independently tilted camera/IMU simulation mounting. It is retained as evidence of that prototype only. It does not establish success of the stricter physical mounting and confidence configuration.

The `wood` tests use an independently generated furnished room and a separately fetched CC0 photographic wood-floor appearance. They are a different appearance benchmark from both the matte-room stress test and the original unchanged course. Their results must not be substituted for original-course acceptance. Seed 23 changes the room furniture and initial heading without changing the navigation configuration.

The failed seed-23 run exposed a real harness defect: default ROS SIGINT handling invalidated the context before the navigator could send its landing command. PX4 subsequently disarmed, but that run is rejected and its evaluator did not cover a complete cycle. Cooperative SIGINT/SIGTERM handling and a larger bounded wall-clock watchdog were added. A separate actual SITL cancellation test then landed with no contacts and confirmed disarm, without using the runner's landing fallback.

## Map and estimator evidence

For `surveygrid-wood-17-01`, alignment uses only the first simultaneous measured/independent body pose. No fitted map or trajectory alignment is used:

- Online navigation position RMSE: **0.02720 m**; maximum **0.04026 m**.
- Measured occupied voxels: nearest reference-surface distance median **0.00716 m**, 90th percentile **0.05765 m**.
- Occupied voxels above 0.25 m: median **0.03364 m**, 90th percentile **0.07882 m**.
- Actual optimized structural point cloud: median **0.06271 m**, 90th percentile **0.12403 m**.
- **178,587 of 192,000 voxels remained unknown.** These surface statistics measure agreement where observations exist; they do not establish full coverage or certify unobserved free space.

A same-recording estimator ablation on `unknown-17-01`, evaluated at common 20 Hz timestamps, gave raw OpenVINS body-position RMSE **0.03780 m**, versus **0.11686 m** with the earlier sparse Cartographer correction. That feedback path was removed. Cartographer scan/loop poses are mapping-only; return currently relies on continuous VIO/ToF odometry and the observed local occupancy map, not a validated visual relocalization loop.

The upstream mapper's endpoint/traversal deduplication also truncated distinct measured voxel paths. After replacing it with complete traversal and once-per-frame hit-priority evidence, an isolated 1280×800 depth-plane test verified all **204** sampled ray-interior voxels. Measured free cells increased from **237** to **730** for the same input; zero/NaN/out-of-range measurements still leave unknown cells unchanged.

## Control and observability repairs

A fixed downward camera needs both high and low measured viewpoints to observe the upper and lower swept volumes. The survey uses two bounded panoramas inside the same declared launch box. Subsequent height maneuvers require an already observed connected free volume; no additional free space is declared.

Safety braking now latches the first braking position instead of moving the hold target with every drifting pose. Replanning waits for low measured velocity and recovery of the hold position. The previous trajectory ID remains rejected across a goal change even if its reference was cleared by braking. An isolated domain-43 command regression verifies the anchor under a synthetic 15 cm pose drift, settling before replanning, and rejection of old commands. No vehicle is armed by that regression.

The observed-depth pipeline has a bounded 350 ms age budget. The measured-velocity stopping check includes that full latency. CPU matching uses 640×400 pairs while VIO retains native 1280×800 imagery; focal length and LR pixel-error limits scale together, and nearest resampling preserves invalid depth. Known-disparity and constant-image regressions verify metric scale and unknown preservation. Real OAK firmware confidence still needs hardware validation.

## Resource evidence

The successful strict run's development-PC navigation processes used a maximum summed PSS of **1,185.55 MiB**. Combined CPU utilization was **336% median / 374% 95th percentile**, where 100% denotes one desktop CPU core. GUI, simulator, recorder and evaluator are excluded. CPU stereo matching is a development proxy for measured depth produced by the OAK hardware. These numbers do not establish Orin Nano latency, memory headroom under JetPack, power, or thermal behavior.

For `cpu400-wood-17-01`, peak navigation PSS was **1,117.60 MiB** and CPU was **210.5% median / 237.2% 95th percentile** on the same development host. Stereo compute logs report approximately **28–44 ms**, with **60–68 ms** observation age at the sampled log points. These are development-host measurements, not Orin measurements or a controlled same-trajectory performance comparison.

The map, feature counts, queues and keyframe history are bounded. Actual Orin Nano 8 GB profiling remains required. JetPack version, power mode and camera mounting have not been identified.

## Reproduction and artifacts

See [README.md](README.md) for setup and launch commands. Every accepted trial records source/configuration and runtime-source hashes, ROS input audits, filesystem-isolation evidence, native evaluation, MCAP data, and both screen/SLAM and first-person videos. On the development Ubuntu host, the strict successful recordings are:

```text
/work/cartographer/v5/results/surveygrid-wood-17-01/videos/flight-and-slam.mp4
/work/cartographer/v5/results/surveygrid-wood-17-01/videos/fpv.mp4
```

Videos and raw sensor bags are excluded from Git. The strict success recording was also copied to the local workspace under `outputs/cadfree-navigation/surveygrid-wood-17-01/videos/`.

Current checks include the pure Python regression suite and isolated ROS-domain-43 mapper, planner and cooperative-cancellation regressions. Domain 43 synthetic tests never arm PX4. Live flights and recordings use domain 42.

The current pure-Python suite has **152 passing tests** (2026-11-06). Isolated ROS regressions additionally verify measured ray coverage, goal heights, rotated body-frame velocity, cooperative cancellation, fixed braking anchors and stale-trajectory rejection. Synthetic task checks do not demonstrate original-course perception or flight completion.

## Original-course learned-stereo counterexample

Pinned LAS2-S was evaluated on four frozen monochrome pairs from `course-view-02`, using its real rectified stereo baseline rather than example camera parameters. Inference was kernel-isolated from source meshes, native poses, and episode results; predictions were frozen before a separate ray evaluator read physical collision meshes. This evaluator includes the source course's nearly transparent collision nets.

| Frame time | Classical coverage | LAS2-S LR/photo-filtered coverage | Filtered overestimate > 0.25 m fraction | Filtered positive-error p99 |
|---|---:|---:|---:|---:|
| 23.8 s | 0.00% | 0.92% | 0.00% | 0.206 m |
| 38.8 s | 4.47% | 23.81% | 0.00% | 0.160 m |
| 53.8 s | 2.68% | 20.33% | 13.89% | 0.471 m |
| 68.8 s | 4.26% | 31.84% | 3.22% | 0.423 m |

Coverage is against sampled physical reference rays in the 0.25–8 m interval, not full-volume knowledge. Filtering uses two directional predictions, the same 0.75-native-pixel LR tolerance, image photometric consistency, and self masking. The classical accepted rays in these samples had no >0.25 m overestimates, but their coverage was inadequate. LAS2-S remains **rejected as a flight free-space input** despite greater density. Four samples are not a broad model benchmark. CPU two-direction inference/validation took 0.364–0.430 s with four desktop threads and peak RSS about 825 MiB; this is not Orin GPU/TensorRT performance and does not satisfy the current 350 ms pipeline budget.

## Original-course continuation, November 6

No CAD-free complete competition episode has succeeded. The navigation baseline still has not demonstrated corridor completion. All results below retain the unchanged original competition scene; no simulator pose, ideal depth, course coordinates or route list enters navigation.

| Episode | Change under test | Maximum independent excursion | Outcome |
|---|---|---:|---|
| course-return-06 | Bounded observed-free A* return | 0.228 m | Fresh-observation settling timeout |
| course-homeview-07 | Retained initial observable headings and fixed settling anchor | 0.327 m | Returned/disarmed, insufficient exploration |
| course-twist-fine-08 | Actual estimated twist and one-voxel measured exploration step | 0.411 m | Returned/disarmed, insufficient exploration |
| course-bounded-09 | Bounded upstream planner and exact exposure synchronization | 0.512 m | Returned/disarmed, insufficient exploration |
| course-forward-10 | Horizontal rigid stereo/IMU mount | 0.151 m | Localization recovery exhausted during takeoff |
| course-mount5-11 | Five-degree downward rigid mount | 0.266 m | Localization recovery exhausted; ground landing outside home tolerance |
| course-planar5-12 | Separate conditional planar free-ray stream, five-degree mount | 0.154 m | Localization recovery exhausted; disarmed near home |

All these independent evaluations reported zero obstacle contacts. Episode 09 completed before the later host reboot. The recorded EGO memory kill occurred at 04:05:50 UTC in the previous boot, before the bounded planner patch; it is not evidence that episode 09 exhausted memory.

The current proposed rigid mount is 15 degrees downward. Earlier 0/5-degree camera recordings frequently show almost only a blank wall during ascent, with too few measured corners; five-degree tracking loss begins before the first observation rotation. The latest diagnostic also reduces the relative survey height to 0.70 m, replaces prescribed full rotations with brief quality-gated arcs, and brakes toward recent observable headings before the inertial bridge expires. These are robot visibility/control policies, not course coordinates. Actual hardware extrinsic calibration and validation remain outstanding.

### Frozen native-ray planar candidate

Candidate SHA-256 `ecf966753a587b357dc6717c2e65781d57797e0a5879983078523e125a9602bd` was independently re-evaluated on all 23 previously exported original-course stereo frames. Predictions were produced under the offline Landlock allowlist, frozen with SHA-256 manifests, and then scored separately against physical source collision meshes and evaluator poses. Both merged outputs and inferred-only ray tails were scored. Across 21,973 sampled inferred-only rays there were no overestimates above 0.25 m, and every scored frame had a zero positive-error p99; blank frames remained blank. This one-ray-per-eight-pixels sample is not an arbitrary-scene safety certificate. The model assumes locally planar surfaces and can miss unobservable same-appearance foreground objects.

Linux ROS-domain-43 checks pass for both independent input streams: raw zero/NaN/out-of-range measurements remain unknown; 204 measured ray-interior voxels are verified; 519 inferred-free voxels are marked separately; no inferred occupied/measured hit is inserted; terminal-voxel reserve, 0.6 s expiration and measured-occluder preservation pass. Five valid planner trajectories verify height/body-twist handling and repeated goal recovery. The native six-package rebuild succeeds; the checked-in patch applies to the pinned upstream revision.

The raw 10 Hz measured stream is independent of the optional model's 5 Hz output. Inferred evidence cannot refresh raw observation time or overwrite measured obstacles. The optional stream is disabled in the repository default. In episode 12 the 150 ms model budget was frequently exceeded under the complete live stack (371/544 outputs at one snapshot); 486 outputs used measured-only fallback. More density alone has therefore not resolved the bottleneck. Do not represent offline throughput as live or Orin-qualified performance.

The v2 task executive/perception/contact modules are integrated as pure modules, including measured resting-foot calibration, contact-aware LAND/liftoff certificates and forward-only passage evidence. Their synthetic regression checks pass; the native competition controller bridge and actual original-course task recognition remain separate validation requirements. Global ToF height correction now has a one-way freeze interface so platform contact cannot redefine the original floor gauge. No task completion is inferred from timers, commanded positions or synthetic test traces.

Original-course diagnostic videos are retained on Ubuntu at `/home/fcy/Videos/UAV-Mission-Planning/original-course/`, with per-episode outcomes. These must not be labeled successful complete-course recordings. Successful generated-room recordings remain separate historical smoke tests.

### Subsequent integration checks, November 6

There is still no successful complete original-course CAD-free episode. Episodes 13–19 did not escape the launch neighborhood. Episode 13 reached a 0.264 m independently measured horizontal excursion, returned to 0.010 m from home and disarmed without obstacle contacts. Episode 17 did not arm: its original one-shot mode/arm exchange did not receive an arm acknowledgement. The subsequent finite, status/acknowledgement-driven handshake worked in episode 18, which later lost visual updates and disarmed. A requested-heading slew limiter reduced the observed peak native estimated yaw rate from 118 deg/s in episode 16 to 26.76 deg/s in episode 18; it is a request limit, not a guaranteed physical rate. A physical 45-degree stereo/IMU roll in episode 19 also failed to maintain exploration tracking.

The frozen planar candidate was accelerated without changing its five output arrays or plane diagnostics: 30 actual sensor frames were bitwise equal on Linux. A live snapshot recorded 224 publications and zero stale/budget/deadline failures, with the last model computation taking 93.85 ms. This improves delivery, not observability. Four source15 packets still supplied zero newly unknown map voxels after existing raw/map overlap; their inferred endpoints were about 0.67–0.68 m below the body. Dense inferred image coverage did not clear the limiting upper/lateral body volume. Original invalid pixels and separate inference provenance remain preserved.

EGO now exposes bounded pipeline counters. Sensor-only domain43 transport comparisons showed reliable delivery was more consistent under load: one 12 s replay integrated 120/120 offered frames with reliable queue4, versus 1/120 with best-effort queue20; a reordered repeat delivered 74/80 with the latter. This is a load-sensitive transport finding, not a fixed best-effort loss rate. Reliable internal output is selected; native camera input remains sensor QoS. Exact exposure matching and unknown-volume rejection are unchanged.

An isolated OpenVINS LK-before-RANSAC repair was built but not promoted. Source19 replay coverage fell from 0.906 to 0.881 and independent position RMSE rose from 0.0172 to 0.0214 m. Source16 coverage improved from 0.866 to 0.908, while RMSE rose from 0.0523 to 0.0913 m. More accepted visual updates did not establish better position accuracy. These short launch motions are not full-course estimator validation.

PL-VINS commit `4292f04fd7c0d350befcf43342ae27339ee3f38d` compiled and passed its native modified-LSD smoke check in a separate Noetic/Focal image. The original source19 bag exported 883 images and 11,031 IMU samples with original header times, pixels and SI measurements. A 1x replay processed 881 frames but produced zero nonlinear estimator frames and zero odometry samples. Point observations had median18/max47; line observations median8/max14; usable triangulated lines remained zero. The original point/parallax initialization conditions were retained. Image size4.01GB is disk usage; uninitialized estimator CPU is not an operating-load measurement. No output was connected to PX4.

The complete task adapter's first 45-wall-second observe-only run issued no control requests. It rejected871 body updates for the explicit0.15m three-sigma position budget, leaving no joined frames. Later analysis of the extended recorder showed a steady three-sigma floor of about0.154m and a startup maximum0.418m. The recorder continued after the observer stopped; its9526 samples are not all from the45s observation window. This mismatch needs an explicit task uncertainty policy; covariance must not be reduced to make the adapter pass.

The current local regression suite passes180 tests. Its wrapper includes32 ROS-stub adapter scenarios; those are not32 additional flights. None of these desktop results establishes Orin Nano8GB performance.

The longer PL-VINS source15 replay subsequently initialized once and produced 810 odometry samples, then failed/rebooted. Independent fixed-first-body-pose scoring without scale/trajectory fitting found 3D RMSE 10.658 m and maximum error 15.431 m; only 36.352–76.800 s of the recording had estimated output. This configuration is rejected as a flight estimator. A cross-version matrix-serialization issue was repaired separately: Python OpenCV4.11 misread integer-looking values in `dt: d` matrices, while the actual estimator container's C++ OpenCV4.2 read the same earlier calibration correctly. Seven sensor-export/calibration tests pass on Ubuntu after emitting explicit floating-point literals. The large native4.2 trajectory error cannot be attributed to the separate4.11 parsing bug.
