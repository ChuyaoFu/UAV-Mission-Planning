# Runtime layout and execution

The full flight stack was developed in a prepared Ubuntu 24.04 container running on an Ubuntu host with NVIDIA graphics. Core tests are portable; full ROS/PX4/Gazebo execution is Linux-specific. A fresh end-to-end image rebuild from this checkout has not yet been verified.

## Expected layout

Mount this repository as `/work` in the container. Runtime scripts expect:

```text
/work/src/PX4-Autopilot/
/work/src/Micro-XRCE-DDS-Agent/
/work/ros_ws/install/                         px4_msgs and base ROS workspace
/work/cartographer/v2/ros_ws/install/         patched OpenVINS overlay
/work/cartographer/v3/carto_ws/install/       patched Cartographer overlay
/work/cartographer/v4/carto_ws -> ../v3/carto_ws
/work/sim/                                   generated Gazebo resources
```

Pinned upstream revisions and canonical patches are listed in [patches/README.md](../patches/README.md). Source clones, ROS build/install trees, simulator logs and PX4 root filesystems are not committed.

The archived `docker/Dockerfile.prepared-runtime` expects a `px4-setup/` build context taken from the matching PX4 checkout's `Tools/setup/`. It is an environment reference, not an independently verified one-command installer. The base build needs ROS/Gazebo, Ceres/Boost/OpenCV, Lua/protobuf, Cartographer dependencies, MCAP ROS bag support and Gazebo transport Python bindings. Video export also requires ffmpeg, ImageMagick, wmctrl and an authenticated display session.

## Prepare generated directories

Inside the prepared `/work` container:

```bash
mkdir -p cartographer/{logs,data,results}
mkdir -p cartographer/v2/{logs,data,results}
mkdir -p cartographer/v3/{logs,data,results}
mkdir -p cartographer/v4/{logs,data,results}
cp docs/validation/*.txt cartographer/v3/results/
cp docs/validation/*.txt cartographer/v4/results/
# Create this only after the v3 Cartographer workspace exists:
ln -s ../v3/carto_ws cartographer/v4/carto_ws
```

Do not replace an existing workspace with the symlink. `runtime_snapshot.py` checks the actual mapped shared library before arming.

Generate the base simulator resources from the matching PX4 checkout, then import the original course materials. `scripts/generate_sim.py` includes a historical textured reference scene; the validated mission selects the original-material `source` world and the passive `stereo` front end. The compiled CAD grid under `cartographer/results/` is a navigation resource derived from the supplied mesh, not a previous-flight SLAM map.

The generated original-material course importer is `scripts/import_course.py`. The `cartographer/scripts/import_course.py` copy comes from an earlier packaged layout and resolves a different root; use the root `scripts/` entry in the `/work` layout.

## Existing prepared-runtime mission

```bash
source /opt/ros/jazzy/setup.bash
source /work/ros_ws/install/setup.bash
source /work/cartographer/v4/carto_ws/install/setup.bash
export ROS_DOMAIN_ID=42 GZ_PARTITION=visual_nav_sim

bash /work/cartographer/v4/scripts/stop.sh
bash /work/cartographer/v4/scripts/start.sh source stereo
bash /work/cartographer/v4/scripts/open_windows.sh
python3 /work/cartographer/v4/scripts/run_trial.py unique-label --mission --route window --seconds 140
# Only after disarm and the run.json is complete:
python3 /work/cartographer/v4/scripts/finalize_episode.py unique-label
```

`start.sh` starts the simulator and estimators. `run_trial.py --mission` commands the SITL flight after mapping/recording checks. A new label is mandatory because existing evidence directories are not overwritten. Reset/stop refuses an armed vehicle. The experiment records the actual Gazebo and RViz windows, so a valid authenticated X11/Xwayland display is required. The historical native display layout assumes a large desktop and may need resizing.

`--route gate-b` is an experimental alternative. The known 59.328 s trial had one contact and a 5 s engineering penalty; it is not a demonstrated faster clean route. It excludes the optional aperture in the planning cost map only, without changing Gazebo geometry or SLAM input.

## Read-only analysis

`evaluate_bag.py`, `geometry_audit.py`, `height_audit.py`, `map_comparison.py` and `portability_check.py` process original recordings. Truth/CAD overlays belong to independent evaluation; do not connect evaluator truth to the control or mapping chain.

`run_mapping_ablation.sh` / `replay_recorded_mapping.py` require ROS domain **43**, isolated from live-flight domain **42**. A replay instance check prevents multiple mapping servers in the replay domain. Startup can be regenerated from a small number of actual recorded floor observations; it is not a synthetic ideal plane.

Experimental `weight-*.lua`, `no-loops.lua` and `imu-extrapolator.lua` files are ablation configurations, not the validated default. Historical `prepare_v2.py` and patch migration helpers can overwrite configuration; use the canonical patches and existing generated configs for reproduction.
