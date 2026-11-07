# ROS2 node v3: explicit240p conditional IPM+floor policy, default inference OFF

This is a new source-only release. Frozen400p models/v2archives are unchanged.
Only segmentation is resized to384x240; original images, intrinsics, masks,
raw depths, ray geometry and32FC2output remain1280x800. `cut_lines_240.py` uses
actual pixel-centre conversion `(seg+.5)/.3-.5`; minimum line80px, opposite
side offset6px and native border4px remain in ORIGINAL pixel units.

## Build and launch

Use the existing sourced ROS2 environment with system OpenCV4.6/cv_bridge,
NumPy/SciPy/PyYAML. Build the native helper if not already built:

```sh
cmake -S native -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
bash run_ros2.sh "$PWD/replay-calibration/kalibr_imucam_chain.yaml" "$PWD/replay-calibration/tof-origin-body.json" "$PWD/replay-calibration" "$PWD/build/libboundary_lbd.so" cold5_freeze1_seedvalidated_240p false ground_ipm
```

Use actual live calibration/masks if different from this actual shared rig.
After root's independent physical grade/timing/full-volume checks, changing the
sixth argument totrue enables conditional inference. The original authoritative
raw topic stays independent10Hz. Only ONE completed-inference publisher owns
`/nav/depth_evidence`; stop the old planar evidence owner first. Node enforces
process lease and actual graph ownership and publishes no raw-only packet on
that FREE synchronizer (which would consume its exact pose match).

Default model=ground_ipm, include_floor_completion=true, variant=
`cold5_freeze1_seedvalidated_240p`, enable_inferred_free=false. The cold GMM uses
five iterations, warm uses one GC_EVAL_FREEZE_MODEL iteration. Current measured
floor/nonfloor hard seeds each must pass95% learned-class coverage within3sigma
(standard deviation floor1DN); failed current fit retrains/rechecks, failed
historical fit rejects. Appearance validation does not refresh range, pose,
evidence timestamp or mapperTTL. IPM uses an actual stereo floor fit validated
with actual ToF/gravity, not a temporal baseline or ideal camera/fixture pose.

Fitted-floor completion and vertical termination share ONE32FC2packet. Raw bits
are unchanged inchannel0; channel1 is surface−0.10m/raynorm, downward-rounded.
Native must independently apply sqrt(3)*0.1=0.173205m diagonal retreat, occupied/
ambiguous vetoes, terminal/footprint quantization, exact0.005s pose matching and
full body/stopping queries. No inference creates occupied endpoints.

## Timing and queues

Max2.5Hz, workercompute0.4s, actual input/final ROS+wall age0.5s, unchangedraw
freshness0.25s. No age/TTL/range relaxation. SourceSGBM input age~0.228s is
observed SITL latency; no OAK hardware latency claim is made. Native original
ray limit2.5m/axial0.3..2.4m and existing body margins remain unchanged.

One isolated latest worker, one job in flight, bounded caches, completed valid
GMM/LBD stages retained before an expired packet is dropped. IMU queue200,
body/range/camera-pose queues20, image queues1; all bounded and reported. Health
also reports actual IMU sample count/support endpoints/maxgap, source-health
attempt rejections, complete/partial cache counts, and true queue/compute/
submit-to-finish times. The previous input_to_worker_delay metric sampled after
compute; it now samples BEFORE evaluate. Age rejection always used the real
parent submission+ROS exposure times and is unchanged.

Native ray norms/axes are cached and native plane dot products use equivalent
separable arithmetic, avoiding full native3D tensors. Four runtime-vs-reference
checks have identical common float32depths; one frame omits3 additional boundary
pixels, others have identical masks. The new model still needs its own Linux
output grade. No raw/pose timestamp is restamped to mask latency.

## Actual frozen inputs and Linux commands

Shared `depth/deliverables/grabcut-ipm240/` contains four new frozen source24/
source25high/cruise predictions, originalscore-inputs, hashes/integrity, and24
consecutive10Hzinputs+causalwarmoutputs. Reference24sequence:23valid/1rejected
(source25t121.452 failed actualfloor normal/ToF/support); retain raw/unknown on
rejection. LocalMacwarmcompute~0.135–0.238s is NOT a Linux or ROS result.

```sh
python3 bench_ipm_stream.py --inputs /path/to/grabcut-ipm240/score-inputs --mask-dir replay-calibration --variant cold5_freeze1_seedvalidated_240p --include-floor-completion --unbounded-diagnostic --output ipm240-linux-four
python3 bench_ipm_stream.py --inputs /path/to/grabcut-ipm240/cruise-sequence-inputs --mask-dir replay-calibration --variant cold5_freeze1_seedvalidated_240p --include-floor-completion --max-compute-s .4 --output ipm240-linux-sequence
```

Live ROS node uses actualleft/right/raw/IMU/Range/bodyOdom/exact cameraPose and
stamped localizationhealth. Native joint telemetry is supported/cached for the
separate temporal mode; IPM deliberately has no parallax/clone prerequisite.
World frame checked=world for these sources; configured mount quaternion is
never read. Actual ToF body translation and actual estimated transforms are
used. DefaultOFF, source delivery only: no SSH, ROS run, arming or deployment
was performed here. Root owns actual Linuxinflow/physical grade/fullquery test.

## Primary integration validation

The primary workspace subsequently built and ran this backend in Ubuntu ROS 2.
The native mapper, 27 sampled independent geometric grades and a 65 s real
sensor replay are described in [STRUCTURAL_INTEGRATION.md](../../STRUCTURAL_INTEGRATION.md).
No original-course completion or Orin hardware validation is claimed.
The integration accepts a measured BODY bracket up to 60 ms because the actual
20 Hz stream has 48/52 ms quantized intervals. It retains both source stamps and
records bounded interpolation terms; it never changes the image exposure time.
The source-release statements above describe the parallel delivery before
primary integration. The regenerated SHA256 manifest covers this integrated tree.
