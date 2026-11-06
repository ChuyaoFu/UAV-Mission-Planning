# Optional standard GFTT corner proposals for OpenVINS KLT

Experimental isolated source patch. No flight/deployment or automatic promotion. Root owns full native ROS2 build and independent measured source21/source15/unseen-view replay.

## Apply and configure

Pinned upstream: 69488123ed9362dd44b6f28e7f4680abbff1442b.

Primary `patches/0001-optional-klt-gftt.patch` applies AFTER current cartographer `patches/openvins.patch`, whose hash and exact baseline are recorded in reports/BASELINE.json. Use `python3 tools/install_patch.py SOURCE --check` for a read-only hash/clean-apply check, omit --check to apply to the isolated tree. It refuses unknown/already edited sources. It does not build, write config, deploy or replay. For pristine upstream only, use --baseline upstream; the separate upstream patch is supplied. Do not apply both variants. Six source files change. Changed files supplied alongside are the primary v2-relative variant.

Set in an ISOLATED actual estimator config:

```
use_klt: true
klt_detector: GFTT
klt_detector_timing: true
```

Default klt_detector is FAST; default timing is false. Both new keys are optional and support existing parser ROS overrides. Only uppercase FAST/GFTT accepted; GFTT with use_klt=false rejects instead of silently ignoring the option. No frontend CLI is duplicated. Existing KLT constructor callers remain source-compatible via default arguments. Options -> VioManager -> TrackKLT -> active Grider_GRID -> shared Grider_FAST cell proposal helper is connected for mono and both stereo cameras.

Timing=true emits INFO `[DETECTOR-KLT] method=... width=... height=... cells=... candidates=... elapsed_ms=...` for each grid invocation, for either detector. This includes grid detection/subpixel/mask work, excludes preprocessing, LK, estimator and logging. Existing broader TIME-KLT timers remain. Disable detailed logging for operational resource measurement after the audit if its I/O matters.

## Important actual baseline distinction

The current main v2 patch already adds an unconditional Shi-Tomasi fallback in sparse FAST cells: quality0.005, hard-coded distance8, synthetic equal response1. This candidate removes that hidden mixed behavior and explicitly selects pure upstream FAST OR standard GFTT quality0.01 with the configured min_px_dist. Thus new FAST default matches pristine upstream FAST, not the historical v2 hybrid. The independently compiled upstream/default parity test proves 80 ordered refined points/responses byte-identical on a nontrivial fixed noisy image. Do not label historical source21/source15 v2 results as pure FAST or attribute all differences to discovering GFTT. Preserve the existing v2 hybrid binary for root's actual baseline comparison.

The v2 initializer8, accepted visual-update health instrumentation, calibration/noise, feature budgets, grid quotas/valid cells, occupancy rejection, LK stereo/temporal tracking, F-RANSAC and MSCKF code remain untouched. Source contract checks exact tracking/matching function equality and untouched initializer/descriptor/updater bytes. The old per-grid quota floor(num_features/(grid_x*grid_y))+1 is retained, including its original small raw-candidate oversubscription; no extra feature budget or denser grid is introduced. Grider candidates are not equivalent to retained tracks or accepted measurements.

## Detector and mask behavior

GFTT calls actual cv::goodFeaturesToTrack with maxCorners=original cell quota, qualityLevel0.01, minDistance=original configured min_px_dist, blockSize3, useHarrisDetector=false, k0.04. No parameter search, negative FAST threshold, temporal scene switch, line midpoint or invented depth/3D point. Native GFTT quality order is preserved; no invented response score is assigned or used for ranking. Subpixel refinement is the existing cornerSubPix.

OpenVINS mask >127 means EXCLUDE. GFTT mask nonzero means ALLOW. Candidate converts with compare(excluded_mask,127,CMP_LE), retaining the >127 exclusion boundary exactly; post-refinement GFTT points are again checked for finite/in-bounds/unmasked coordinates. A mask128/255 excludes,127 allows, fully masked and blank inputs produce zero. No environment/mask is disabled.

Inherited stereo replenishment starts mask1_updated from mask0. GFTT mode starts from mask1 itself, retaining each camera's self-occlusion mask and original current-track exclusion rectangles. FAST mode keeps original behavior for upstream parity. Native actual stereo detection tests use opposite nontrivial left/right masks and exercise this ownership fix. Original projected right points are still subject to the original right-mask cleanup, with the original LK status checks unchanged.

## Verification

Actual modified grider/helper functions compiled and executed under ASan/UBSan with native OpenCV4.5.5/Eigen3.4. Tests cover exact standard GFTT output/order/parameters, real low-contrast image corners, min distance, mask polarity/boundary, post-subpixel masking, original quotas, blank0/empty cells and FAST raw parity. Independently compiled pristine upstream FAST grid yields byte-identical ordered refined keypoints.

Actual source `perform_detection_monocular`, `perform_detection_stereo` and `extract_grid` methods are extracted without editing their bodies and compiled with an estimator-free TrackKLT member shim. They pass masks, right ownership, low-contrast proposals, real unchanged left-to-right LK/shared IDs and blank0 under ASan/UBSan. This does not constitute full TrackKLT class/ROS2/MSCKF compilation; root's native build is required. CMake tests require OV_SOURCE, EIGEN_SOURCE and a compatible OpenCV_DIR; set ACTUAL_DETECTION_TEST to the C++ file generated by tests/build_actual_detection_test.py. ENABLE_SANITIZERS=ON enables the checks.

Both source patch variants clean-check. Hash manifests and test logs included. Full GPL3 LICENSE included. No root source modified.

## Measured image probes and limits

Read only left pixels: four supplied source15 NPZ left arrays and seven source21 left-image MCAP messages, never pose/truth/depth. Pixel/hash provenance included. Source21 samples are pixel-identical (one independent image), source15 four are unique. Native preprocessing follows actual CLAHE clip10/tile8x8, grid5x5, budget250, FASTthreshold10, min_px_dist8. Full-grid cold proposals/refinement are measured with two warm-ups plus ten repeats, not full VO.

Source15 pure FAST candidates14–61 versus GFTT123–204; median of per-frame medians0.4515ms versus7.5578ms. Source21 unique view33 versus218; about0.4023ms versus8.0106ms. These raw counts are not physical constraints or accepted updates. Local arm64 static OpenCV4.5.5 reports10 GCD threads despite setNumThreads1; caller uses sanitizers, OpenCV library does not. Neither Orin nor one-core performance is established. Increased subsequent LK/BA work is not included. Source15 probe lacks a same-exposure mask and explicitly uses zero; source21 supplied actual portrait90 mask0 is all zero. Nontrivial mask correctness comes from actual-function synthetic tests, not a claim these probe frames exercise self-occlusion.

Root should compare the existing v2 hybrid baseline with this explicit GFTT candidate on21/15 and unseen view. Preserve fixed calibration, noise, budget and all gates; grade accepted-update continuity, independent physical trajectory accuracy and full CPU/PSS. More proposed corners alone is insufficient. Blank/dynamic-shadow views remain a limitation. No promotion based on the probe.
