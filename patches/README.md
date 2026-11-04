# Pinned upstreams and patches

These are the revisions recorded for the previously prepared Linux runtime:

| Project | Revision |
|---|---|
| PX4 | `d6f12ad1c4f70ad3230afd7d86e971421e02fef4` (v1.17.0) |
| OpenVINS | `69488123ed9362dd44b6f28e7f4680abbff1442b` |
| Cartographer, ros2 branch | `1489fc933b6e0b1da936ca08af462abe559dd02e` |
| Cartographer ROS, ros2 branch | `743345256b43a7fd95765d40c7332b6f31ab66bb` |

Micro XRCE-DDS Agent used v2.4.3. ROS 2 Jazzy / Gazebo Harmonic and compatible `px4_msgs` are required. The repository is not a captured package image or complete dependency lock.

Apply `cartographer.patch` and `cartographer_ros.patch` once against their listed revisions. The patches include the loop-geometry guard, sparse-scan extrapolation, joint IMU/VIO optimization regression and Jazzy build fixes. Apply `openvins.patch` against the listed OpenVINS revision.

The v3/v4 `patch_*.py` and migration helpers document intermediate edits; do not run all of them after applying these canonical patches, as that can duplicate modifications.

The known prepared layout built Cartographer under `/work/cartographer/v3/carto_ws` and reused it from v4 via a symlink. OpenVINS was built under `/work/cartographer/v2/ros_ws`. Source build/install directories are intentionally not versioned.

Recorded C++ regression output is under `docs/validation/`: five loop-guard, nine extrapolation and two optimizer cases. These files are prior test evidence, not fresh CI results.
