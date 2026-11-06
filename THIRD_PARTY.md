# Third-party components and assets

The repository contains integration code and patches for separately maintained projects. It does not replace their licenses or redistribute their compiled installations.

- PX4 Autopilot: https://github.com/PX4/PX4-Autopilot — BSD-3-Clause upstream.
- EGO-Swarm ROS 2: https://github.com/ZJU-FAST-Lab/ego-planner-swarm — GPL-3.0 upstream. The pinned revision, local patch, and license are in `cartographer/v5/upstream/` and `cartographer/v5/patches/`. Only measured-sensor planning packages are integrated.
- RTAB-Map / rtabmap_ros: https://github.com/introlab/rtabmap and https://github.com/introlab/rtabmap_ros — BSD-3-Clause upstream; optional diagnostic comparison only.
- PromptDA: https://github.com/DepthAnything/PromptDA — Apache-2.0 upstream; optional offline experiment only, with license in `cartographer/v5/upstream/PromptDA-LICENSE`. Weights and predictions are not flight inputs or committed assets.
- Poly Haven `wood_floor`: https://polyhaven.com/a/wood_floor — CC0 appearance-only benchmark asset, fetched separately; URL and checksum are recorded in `cartographer/v5/upstream/room-material.json`. This benchmark does not replace the unchanged source-course test.
- OpenVINS: https://github.com/rpng/open_vins — GPL-3.0 upstream. Its patch must be used in accordance with that license.
- Cartographer / ROS 2 maintenance branches: https://github.com/ros2/cartographer and https://github.com/ros2/cartographer_ros — Apache-2.0 upstream.
- ROS 2, Gazebo and Micro XRCE-DDS retain their respective upstream licenses.
- `course_source/` contains user-supplied preliminary DroneHacks course assets. No separate license grant for those assets is inferred by this repository.
- Reference SDF/configuration files include material derived from the upstream simulator stack. Keep their upstream notices when redistributing derivatives.

No project-wide license has been selected for the integration code. Public repository visibility alone does not establish an additional license grant.
