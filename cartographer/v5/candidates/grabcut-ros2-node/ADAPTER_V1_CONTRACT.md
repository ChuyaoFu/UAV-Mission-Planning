# Default-OFF bounded structural worker (reviewable stream adapter)

This is a nonpublishing Python adapter backed by native OpenCV GrabCut/LSD and the source-built PL-VINS LBD C ABI. It consumes only bounded exact-exposure snapshots plus the actual current native joint-clone posterior. No episode batch, future poses, model download, GT/CAD/commands or occupied inference. Root must wire it into the sole existing evidence publisher, replacing the old planar inference backend; this package does not create a second publisher.

## Linux benchmark: ready to run

From this directory on the existing Ubuntu/Orin host with system OpenCV4.6, NumPy, SciPy, PyYAML and CMake:

```sh
cmake -S native -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
python3 bench_stream.py --fixtures replay-fixtures --rig replay-calibration/kalibr_imucam_chain.yaml --mask-dir replay-calibration --lbd-library build/libboundary_lbd.so --unbounded-diagnostic --output benchmark-linux-full
python3 bench_stream.py --fixtures replay-fixtures --rig replay-calibration/kalibr_imucam_chain.yaml --mask-dir replay-calibration --lbd-library build/libboundary_lbd.so --max-compute-s .4 --output benchmark-linux-budget
```

The first command measures full cold/warm computation and compares all four outputs against the frozen400p predictions. The second enforces the runtime budget and records drops. Both are offline diagnostic executables without ROS publication. benchmark.json records the actual platform, OpenCV versions, full/native timing, cache hits and peak RSS (LinuxKiB normalized to bytes). Cross-version outputs are saved for independent grading if any pixels differ. A Linux result must actually say linux_measured=true; this chat has only a macOS host and no Docker/Podman/Colima, so no Linux result is fabricated.

The replay fixtures contain only the exact actual supporting sensor exposures, original ToF/gravity metadata and each exposure's actual same-time native posterior. They are small causal snapshots for verification, not runtime input dependencies. Original ROS header objects/frame_id were not exported, so replay results are diagnostic only and cannot be published. Actual native calibration/self masks are bundled. No nav world pose is supplied to inference.

## Runtime integration

`GrabCutLatestAdapter` in stream_adapter.py is default enabled=false and permit_conditional_free=false. Construction requires actual internal T_imu_from_camera, actual native left/right bool self masks and a Linux-built LBD library. `start()` creates one spawned isolated process and a lightweight scheduler only after explicitly enabled.

`offer_exposure(frame, original_header=raw.header, pose_health_ok=actual_health)` admits caller-owned immutable arrays without an image copy or a waiting lock. `offer_native_joint(actual_record)` admits the original schema1 native joint record; full PSD/convention validation runs inside the isolated worker. Missing observations/health are never invented. Raw10Hz proceeds independently even if admission fails, native inference overruns or its process crashes. The caller must retain original image array ownership and not mutate buffers after transfer.

Required frame fields: original left/right mono8, original measured_stereo float32, K, baseline, stamp_ns, left_stamp_ns/right_stamp_ns/depth_stamp_ns (all exactly equal), actual left_frame_id/depth_frame_id (equal), actual gravity_up_optical/gravity_stamp_ns, actual tof_range_m/tof_stamp_ns/tof_valid and calibrated actual tof_origin_optical_m/tof_direction_optical for this exposure. Do not substitute nav/cameraPose for native clone means. Internal calibration is constructor-fixed; a rig change requires a new instance.

Pass an actual matching ROS clock function and a current estimator-health callback (`clock_ns=...`, `health_ok_now=...`). Source health and current health must both be explicitly true before conditional-free output. `poll()` returns at most the latest completed result, preserving actual exposure stamps and original raw header. `evidence_32fc2(result)` performs final current ROS AND wall-age checks; channel0 is original raw, channel1 is conditional INFERRED free-only axialZ. ros_worker_hook.completed_to_message returns an original-header32FC2 message and never publishes it. Keep consumer raw dedup/freshness, exact pose sync, terminal quantization, occupied/ambiguous vetoes, whole-body/stopping checks and the full actual age/TTL budget.

Defaults: max2Hz, compute budget0.4s, input/final ROS and wall age0.5s,24frame cache,1.5s cache horizon, one job total in flight, one latest result, up to6 metric-distinct supporting views. No inference job backlog. Time resets clear caches; an exact current-posterior exposure is mandatory. Native OpenCV calls are nonpreemptible, so a call may overrun the budget inside its isolated process; its result is discarded and raw callbacks remain independent. Computed immutable segmentation features can be retained to warm the bounded cache even after a deadline drop.

The model is still the scored400p, five-iteration, RNG0 measured-seed GrabCut. No silent240/320p, one-iteration or warm-GMM algorithm substitution. Per-image segmentation is cached; metric lines are reconstructed anew with native means from the same current posterior. LBD describes only the selected native-image segment and records Hamming diagnostics; the frozen longest-valid-cut-boundary association remains governed by full multiview geometry, not a newly introduced LBD gate.

The native C ABI has a small defensive indexing fix: retained KeyLine class IDs are contiguous and mapped back to original input IDs after invalid/border-line rejection. A mixed valid/invalid/valid descriptor call previously could crash the old wrapper. The stream worker is isolated regardless. Original frozen LBD library and all four frozen predictions remain unchanged.

No range/raw/mask/photometric/foreground/component/native2px/8cm gates are relaxed. Actual positive raw pixels are never replaced; model rays do not create occupied endpoints. A generic Euclidean reserve sqrt(3)*voxel+0.1m is applied. max ray length remains2.5m; output float32 lower values round downward. Optional consumer reserve credit defaults0 and requires an explicit verified actual mapper subtraction; terminal-cell/footprint handling remains in the consumer. The geometric reserve is conditional policy, not a physical error certificate.

## Status and remaining evidence

No certified wall/free lower bound or whole-body clearance is asserted. Output stays conditional INFERRED; navigation metadata remains disabled until root's independent ray/volume/runtime decisions. Source24 actual unknown-volume/query export is still needed for actual upper/side whole-body gain; pixels/spans are not a substitute. No flight or deployment was performed here.

## Review/verification snapshot

The frozen adapter version accepts the actual1280x800 native grid only;400p is an internal segmentation copy and native rays/K/pixels never change. A new resolution policy must change the coordinate conversion explicitly and be independently frozen/graded. `ExactExposureAssembler` is the optional bounded nonpublishing joiner for left/right/depth plus actual original-stamped floor metadata; it invents no missing ToF/body-camera transform.

Seven local contract checks passed, including process death isolation and raw NaN bit preservation. Actual process smoke recorded callback admission below0.1ms, a real400p native call overrunning0.4s, and a deadline_drop with no eligible inference. Controlled test clock/frame IDs are labelled transport-only; no missing physical health was set true.

Local full snapshot replay returns identical finite masks for all four frozen candidates.18.7s/126.7s depth is float32-bit identical;18.6s/126.6s earlier-prefix depths differ by at most0.24/2.03micrometres because the original prefix batch reconstructed serialized line coordinates as float64 while the canonical stream extractor retains native LSD float32 coordinates. No original frozen arrays changed. Linux/OpenCV-version numerical changes are reported and actual outputs saved for regrading.

Mac CPU full replay: cold3.01/3.38s, cached0.58/1.38s; these values do NOT establish Linux viability or justify rejecting Linux. The default0.4s policy actually drops such overruns. Do not deploy on the strength of a Mac replay; run the included Linux benchmark and retain actual exposure age/consumerTTL in stopping checks.
