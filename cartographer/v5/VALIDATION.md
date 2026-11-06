# Experimental CAD-free navigation validation

This milestone tests online measured-stereo mapping, local obstacle avoidance, and return to the first measured body pose. It does not test the competition task sequence or demonstrate hardware readiness. All navigation runs disable GPS fusion and exclude simulator pose/depth, field CAD, and preset route targets. Native simulator poses and contacts are collected by a separate evaluator only.

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

## Resource evidence

The successful strict run's development-PC navigation processes used a maximum summed PSS of **1,185.55 MiB**. Combined CPU utilization was **336% median / 374% 95th percentile**, where 100% denotes one desktop CPU core. GUI, simulator, recorder and evaluator are excluded. CPU stereo matching is a development proxy for measured depth produced by the OAK hardware. These numbers do not establish Orin Nano latency, memory headroom under JetPack, power, or thermal behavior.

The map, feature counts, queues and keyframe history are bounded. Actual Orin Nano 8 GB profiling remains required. JetPack version, power mode and camera mounting have not been identified.

## Reproduction and artifacts

See [README.md](README.md) for setup and launch commands. Every accepted trial records source/configuration and runtime-source hashes, ROS input audits, filesystem-isolation evidence, native evaluation, MCAP data, and both screen/SLAM and first-person videos. On the development Ubuntu host, the strict successful recordings are:

```text
/work/cartographer/v5/results/surveygrid-wood-17-01/videos/flight-and-slam.mp4
/work/cartographer/v5/results/surveygrid-wood-17-01/videos/fpv.mp4
```

Videos and raw sensor bags are excluded from Git. The strict success recording was also copied to the local workspace under `outputs/cadfree-navigation/surveygrid-wood-17-01/videos/`.

Current checks include the pure Python regression suite and isolated ROS-domain-43 mapper, planner and cooperative-cancellation regressions. Domain 43 synthetic tests never arm PX4. Live flights and recordings use domain 42.
