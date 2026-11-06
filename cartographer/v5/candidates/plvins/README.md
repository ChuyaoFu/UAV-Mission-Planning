# PL-VINS isolated replay candidate

Official `cnqiangfu/PL-VINS`, GPL-3.0, pinned **4292f04fd7c0d350befcf43342ae27339ee3f38d**. This is an isolated estimator comparison, **not a navigation input**. Source subset with its upstream `LICENCE`, pin/checksums, two reviewable patches, Docker recipes, calibration converter, sensor-only exporter, replay/resource observer and result summary are included.

On November 6 the Noetic/Focal recipe built successfully on the Ubuntu development computer, including its actual OpenCV4.2 native modified-LSD smoke check. Image `uav-plvins-candidate:noetic` has ID `28b291352d04` and uses 4.01 GB on disk. An isolated, network-disabled source19 replay exported 883 original images/11,031 IMU samples and processed 881 frames at 1x playback, but produced **zero initialized estimator frames and zero odometry**. Points had median18/max47; lines median8/max14; usable triangulated lines remained zero. Initialization conditions were not relaxed. The Jazzy/OpenVINS flight installation was preserved. An installed image or detected 2D lines do not establish a working estimator. These desktop results do not qualify Orin Nano 8 GB runtime.

Use `container/Dockerfile.noetic`. It builds on Focal's one system OpenCV4.2/Ceres1.14. A public-API-only, separately namespaced copy of OpenCV4.5.5's restored LSD implementation fills the missing4.2 implementation; it neither replaces system symbols nor loads a second OpenCV. `cv_bridge` remains on the same system OpenCV4.2. The namespaced LSD and modified-LSD/LBD sources passed compilation and native smoke execution in the Ubuntu image. The container isolates legacy ROS1 dependencies from Jazzy and the eventual preinstalled JetPack host. Both CPU architectures have pinned base manifests; the complete image has only been built/tested on the development x86_64 host.

A longer source15 replay processed 2,166 frames, initialized once, produced 810 odometry samples and then reported one estimator failure/reboot. Only 36.352–76.800 s of the 9.136–117.868 s reference interval was covered. Independent fixed-first-body-pose scoring, without scale/trajectory fitting, found 3D RMSE 10.658 m and maximum position error 15.431 m. This configuration is **rejected as a flight estimator**. Initialization and line residuals alone did not establish a correct trajectory.

The converter emits explicit floating-point scientific notation for OpenCV matrix elements. The previous integer-looking literals in a `dt: d` matrix round-tripped correctly in the actual C++ OpenCV4.2 estimator container, but returned corrupt matrices through the analysis environment's Python OpenCV4.11 FileStorage. Seven sensor-tool tests pass on Ubuntu after the serialization repair. Geometry values and transform direction are unchanged.

The alternative `container/Dockerfile` uses original-era Melodic/Bionic/OpenCV3.2/Ceres1.13 and only patch0001. Both base multiarchitecture image indexes were checked against the registry and pinned; amd64 and arm64 manifests are included. Apt package revisions are recorded inside a successful image at `/opt/build-packages.txt`; apt repositories remain external build dependencies.

## Root build/run

At this extracted bundle root:

```sh
bash build.sh uav-plvins-candidate:noetic
```

The build uses only camera_model, feature_tracker and plvins_estimator; no wheel/GPS/CAD/depth/ToF/ground-truth/simulation publisher or pose-graph process is built/launched. ModifiedLSD, LBD and matcher compile from source; the repository's prebuilt liblinedesc.so is excluded. After catkin builds, the image also compiles/runs the native line detector/matcher/blank-frame smoke test against its **actual** architecture and OpenCV version. An apt/build/API error should be handed back with the first error block; no filter rewrite is needed for this packaging path.

Prepare a host Python3 environment and export only the two measured topics:

```sh
python3 -m venv /path/to/scratch/plvins-tools
/path/to/scratch/plvins-tools/bin/pip install -r tools/requirements.txt
/path/to/scratch/plvins-tools/bin/python tools/export_sensor_bag.py /path/to/source19_bag --output /path/to/plvins-data --image-topic /vio/left/image_raw --imu-topic /vio/imu --euroc
/path/to/scratch/plvins-tools/bin/python tools/kalibr_to_plvins.py --camera-file /path/to/ACTUAL_RUN_kalibr_imucam_chain.yaml --imu-file /path/to/ACTUAL_RUN_kalibr_imu_chain.yaml --output /path/to/plvins-data/plvins.yaml
```

Use the **actual source19 diagonal45/pitch15 calibration**, not `config/default-rig-example.yaml`: the included example is explicitly the default local unrotated rig and demonstrates parsing only. Converter preserves `T_imu_cam` direction directly (camera→IMU), K/distortion, native resolution, SI measurement/noise fields; fixes measured extrinsics and disables loop closure. It rejects unsupported models, nonrigid transforms, nonidentity unhandled IMU calibration and nonzero unaligned time offsets. No time/axis transform is guessed. PL-VINS consumes gyro and specific force; do not subtract gravity or provide odometry as IMU. IMU orientation is stripped/marked unavailable by exporter.

The exporter reads/deserializes only selected Image/Imu connections, spools on disk, sorts by exact **header** timestamps, removes image row padding, preserves pixels/gyro/acceleration and rejects duplicate stamps. Output ROS1 wire messages are Melodic-compatible. No resampling, interpolation, truth/pose/depth export or fallback coordinates. EuRoC image/IMU tables are optional; no state_groundtruth file is created.

```sh
docker run --rm --network none --cpus 4 --memory 4g --shm-size 512m -v /path/to/plvins-data:/data uav-plvins-candidate:noetic /opt/run_replay.sh /data/plvins.yaml /data/sensors.bag /data/results-1x 1.0
/path/to/scratch/plvins-tools/bin/python tools/summarize_replay.py /path/to/plvins-data/results-1x
```

One isolated container owns its own local ROS master. Outputs and logs land in a new result directory; existing results are not overwritten. Estimator publishes only inside the container; no bridge to PX4/navigation is included. Repeat the exact same measured source segment and1x playback on the same Ubuntu hardware for comparison with OpenVINS; a0.5x run can separately diagnose processing lag, but is not a real-time success. Sensor quality/trajectory accuracy cannot be inferred from a successful build.

The source emits `PLVINS_METRICS` for exposure stamp, INITIAL/NON_LINEAR state, incoming2D point/line observations, post-update eligible points and usable triangulated lines (original>=5-observation/start-frame/triangulation predicate), plus processing time. The observer records estimated odometry and each process's CPU%, RSS/PSS. The summary reports initialization/reboot events, usable-line counts, processing time and estimated-trajectory gaps/nonfinite values/steps. These are continuity/resource measurements, not ground-truth accuracy. CPU100% denotes one logical core; RSS/PSS and local runtime do not prove Orin suitability.

## Patch scope and limitations

Patch0001 removes absolute paths/prebuilt-library linking, supplies OpenCV includes, guards empty line descriptors and zero-line timing division, joins independent point/line outputs by **exact exposure stamp**, routes trajectory files into the mounted output directory and adds read-only post-update metrics. The original FIFO mismatch was reproduced using the actual getMeasurements function; seven actual-function ASan/UBSan cases pass after the boundary fix. Solver, point/line residuals, triangulation, feature thresholds and initialization requirements are unchanged.

Patch0002 updates legacy OpenCV names/includes and adds the namespaced LSD backport. Its provenance JSON records source URL/hash, archive hash and mechanical adaptations; source license notices and OpenCV license are retained. No algorithm change to restored LSD is intended. A16-frame synthetic ARM comparison matches the normal4.5.5 implementation's feature/match/reset counts exactly. The Linux ROS source build and its actual4.2 native smoke test also pass.

**PL-VINS is not line-only initialization.** Its existing `relativePose()` still requires more than20 point correspondences and normalized parallax equivalent to>30 pixels at its fixed460 focal normalization. Lines become usable only with>=5 observations and valid triangulation. Blank gray views remain unobservable; no point gate was lowered, and no line midpoint is presented as a3D point.

Noise conversion: default `discrete` mode multiplies Kalibr densities/random-walk strengths by sqrt(IMU rate), based on the pinned IntegrationBase's dt-scaled noise Jacobians; all source values/effective values/formulas are recorded. The original midpoint covariance omits adjacent-sample noise correlation, so this is not an exact continuous-time covariance discretization. `--noise-mode legacy-density` provides a clearly labelled sensitivity replay that copies original values. Keep the selected mode identical in controlled point/line comparisons and inspect initialization/bias behavior; no sensor noise values were tuned to make a run pass.

## Local completed evidence

- Source and Docker base pins frozen; both patches cleanly apply in sequence and reproduce manifest hashes.
- Seven converter/ROS2→ROS1+EuRoC tests pass on Ubuntu with Python3.12 and rosbags0.10.11, including Jazzy version9 MCAP, transform direction, noise scaling, unsupported calibration rejection, preservation of measured SI values/timestamps and pixel padding removal. An invalid forbidden-topic payload is present in the test bag and never deserialized/exported. Eight summary robustness tests also pass.
- Seven actual getMeasurements C++ cases pass ASan/UBSan; old mismatched pairing is reproduced.
- Actual modifiedLSD/LBD/matcher C++ source built and ran on Apple arm64. Native smoke, blank recovery, gap reset and16-frame normal/backport parity pass. `cv2.setNumThreads(1)`/C++setNumThreads(1) reports10 on this macOS GCD backend: **not a single-core benchmark**.
- Seven real mount5 snapshots: line counts7/5/0/2/0/8/8; corresponding point detector candidates30/3/25/4/21/26/25 include CLAHE/quantization responses, not accepted estimator features. Both near-uniform images still have zero qualified long lines. Probe wall time median11.31ms/max25.16ms; CPU median11.31ms. Four real source15 packets have6/9/10/10 lines and median10.40ms probe wall time. These times are local full probe work, not PL-VINS estimator cost or Orin measurements.
- Snapshot gaps exceed250ms, so tracking resets on each gap. No initialization, usable triangulated-line or trajectory-continuity result is claimed from snapshots. The separate actual contiguous source15/source19 replay results above take precedence over snapshot feature counts.

Fresh local tests:

```sh
python3 tests/test_sensor_tools.py
python3 tests/native_smoke.py /path/to/compiled/plvins_lsd_probe
cmake -S standalone -B /path/to/scratch/lsd-build -DPLVINS_SOURCE=/path/to/patched/PL-VINS
cmake --build /path/to/scratch/lsd-build --parallel 2
python3 tools/run_lsd_sequence.py --probe /path/to/scratch/lsd-build/plvins_lsd_probe --euroc /path/to/plvins-data --output /path/to/probe-result
```

The independent2D probe retains all qualifying lines rather than the official tracker's35+35 replenishment policy; its consecutive-age candidates are not the estimator's triangulated usable lines. It accepts no pose, IMU orientation, truth, depth or3D point input.

Primary references: [official PL-VINS README](https://github.com/cnqiangfu/PL-VINS), [pinned estimator](https://github.com/cnqiangfu/PL-VINS/blob/4292f04fd7c0d350befcf43342ae27339ee3f38d/vins_estimator/src/estimator.cpp), [pinned IMU integration](https://github.com/cnqiangfu/PL-VINS/blob/4292f04fd7c0d350befcf43342ae27339ee3f38d/vins_estimator/src/factor/integration_base.h), [Kalibr noise model](https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model), [OpenCV LSD availability](https://docs.opencv.org/4.x/db/d73/classcv_1_1LineSegmentDetector.html), [restored LSD source](https://github.com/opencv/opencv/blob/4.5.5/modules/imgproc/src/lsd.cpp).
