# Optional native core received-input ledger

DIAGNOSTIC SOURCE ONLY, default off. No ROS bag/player framework, estimator/state/prior/noise/gate changes or PX4. Purpose: distinguish actual received sensor/order differences from the source24 ON/OFF21.2s posterior jump. It does not normalize/align outputs to force equivalence. Root owns native build/replay.

Primary patches/0001-optional-received-input-ledger.patch edits five files against the recognized GFTT+joint+home source. The separate one-line fast-time patch touches no overlapping file; both can be built together. Use tools/install_patch.py SOURCE --check (omit --check to apply). Unknown source hashes are rejected. Changed files, hashes, actual logger header tests and a source contract supplied. Original numerical/filter source is byte-identical after removal of explicit logger hooks/options; StateHelper, Propagator, update/feature/anchor code unchanged.

Set ONE optional field in each isolated replay's effective estimator config (or its existing ROS parser override), with a UNIQUE new writable path:

```
sensor_input_audit_path: /tmp/source24-home-ON-input.csv
```

Use a distinct OFF path; keep all other options/data/subscriptions/load equal. Empty default constructs no logger and performs no payload hashing/IO. Explicit path must be writable and absent; existing output is refused, never overwritten. This patch does not edit root replay_vio.py. Root can pass/append its field when producing temporary replay configs. No source/scene/episode time gate is introduced.

## What actual events are recorded

Each row begins kind, monotonic audit sequence, total received core-entry IMU count, latest received IMU time, rolling IMU FNV1a64. All original double fields are emitted at full roundtrip precision. Sequence is actual synchronized core entry/log call order, not intended bag file order.

- IMU: actual message timestamp, wm3 and am3 BEFORE normal feed handling. The cumulative hash covers all seven native double values as same-machine bytes. No measured value changes.
- CAM: CameraData timestamp/count, then for each actual image: sensor ID, rows,cols,type, image FNV64 and mask FNV64. The prefix gives IMU waterline/count/hash at actual feed entry, before normal tracking. Pixel hashing handles noncontinuous Mat row layout; masks remain untouched. This audits core data, not original stereo right-header identity, which CameraData does not preserve. If matching physical exposure provenance is separately needed, root should also log original left/right integer headers in its sync callback; no claim that a pixel hash reconstructs that lost header.
- MOTION_NOTICE: actual calls of existing notify_external_motion, before its unchanged has_moved flag assignment. Prefix records actual IMU context. It is a notice event, not a fabricated ToF source timestamp. The next CAM in the same serialized context shows which frame saw it.
- INIT: actual successful initializer timestamp, original IMU value matrix dimensions/data, and returned initializer covariance dimensions/row-major data. Hook is after unchanged set_initial_covariance and before normal catch-up. Root static initialization gives IMU16 plus15x15P, exposing v/bg/ba as well as pose. It is actual native initialization, never a forced initial state or guessed zero bias.
- END: graceful ledger destructor footer, final count/watermark/hash. Missing END means incomplete trace, not success.

Same-machine FNV64 is a fast diagnostic fingerprint, not a cryptographic proof of payload equality; all IMU doubles are also present. Camera/mask bytes are not copied into the log. No truth/CAD/depth is read. Header doubles are native values; retain original integer sensor header association separately when required. Camera entries are at VioManager API feed, not a guarantee every bag message reached DDS or was accepted by an upstream sync/frequency gate.

## Comparison and interpretation

```
python3 tools/compare_ledgers.py ON.csv OFF.csv --output comparison.json
```

It rejects incomplete/nonmonotonic audit files and emits only compact counts/first differences: raw IMU values, camera timestamp/image/mask hashes, per-camera IMU watermarks, motion/init records and whole core input event order. No multi-MB output. If IMU values match but watermarks/interleaving differ, a static initializer window or propagation data availability can differ without packet loss. If input ledger and actual init snapshots match but outputs jump, inspect accepted feature IDs/SLAM anchors/Jacobians/chi2 decisions at the first divergent update; count-only health cannot identify that branch. Do not loosen thresholds.

Logging and full-frame image/mask hashing add CPU/IO and can perturb scheduling. Enable the SAME audit in both same-binary controls, inspect observed input delivery instead of assuming the instrument is timing-neutral, and do not use audited resource numbers as production overhead. A direct serialized VioManager replay using existing sensor exports/actual feed trace is a separate small option only if DDS remains inconclusive; no large C++ bag framework supplied here.

Native ASan/UBSan logger tests pass raw IMU fidelity, counts/waterline, matching ROI/continuous pixel hashes, changed pixel detection, mask hashing, unchanged inputs/initial P, monotonic sequence/END and no overwrite. Comparator self-check and clean patch check pass. This does not compile full ROS/VioManager here; root's native build is required. Existing home numerical core0 test remains valid at its recorded source; this observer does not establish actual ROS equivalence. Defaultoff flight behavior remains unchanged.
