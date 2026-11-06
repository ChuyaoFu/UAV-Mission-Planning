# English Gazebo handoff: reproduce on another Ubuntu computer

This is the installation and handoff entry point for a collaborator and her
Codex. It reconstructs the existing `/work` runtime from the source files in
this repository. It does not need the original Jetson, SSH passwords, a copied
Docker disk, or a previous flight recording.

**Start with `CODEX_START_HERE.md` in this directory.** The current mission
implementation is v4. The v3 and v4 historical results, including their different
configurations and limits, remain in `docs/VALIDATION.md`.

## What is included

- Preliminary DroneHacks OBJ/MTL course, original-material importer, fixed
  seven-inch approximate model, camera/IMU configuration and CAD planning grids.
- ROS 2 Jazzy / Gazebo Harmonic / PX4 SITL / passive stereo / OpenVINS /
  customized Cartographer / mission and engineering judge source.
- Immutable Git revisions, the three canonical source patches and a pinned
  multi-architecture ROS base-image digest in `upstreams.lock.json`.
- Dependency image, source fetcher, build scripts, desktop launch, environment
  checks, and a **disarmed** sensor/mapper smoke test.
- Fresh C++ mapping tests and provenance files, generated during installation.

Git excludes the multi-GB container image, downloaded upstreams, build trees,
sensor recordings, videos and logs. Those are created or downloaded locally.
Internet is required during initial installation. The original environment's
apt/pip snapshot was not fully frozen: exact installed versions are recorded
in `results/reproduction/build-manifest.json` after each new build. Source and
configuration equivalence do not imply byte-identical binaries or identical
flight timings across computers.

## Target machine

Use **Ubuntu Desktop 24.04 or 22.04, x86-64**, running an Ubuntu 24.04 container.
A native Linux Docker Engine and Git are prerequisites. The full GUI/flight
workflow is not validated on macOS Docker Desktop, Windows or ARM. The original
host used an NVIDIA GPU. An NVIDIA Linux driver plus Container Toolkit is the
recommended path for comparable rendering performance. Software rendering is
available for installation and connectivity checks; it is not a flight-speed
benchmark.

Allow at least 16 GB host RAM (24-32 GB preferred), swap, and about **40 GB of
free disk** for dependencies and build results. Use two build jobs initially.
The first installation can take tens of minutes or longer depending on CPU,
network and package mirrors; it downloads far more than this small Git checkout.

Install Docker using its [official Ubuntu instructions](https://docs.docker.com/engine/install/ubuntu/).
For NVIDIA, follow the [official Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
These are host setup tasks; the repository does not change host GPU drivers or
automatically restart the Docker daemon. `ctl.sh` uses `sudo docker` when the
current user does not have direct Docker access.

## 1. Clone and inspect

```bash
git clone https://github.com/ChuyaoFu/UAV-Mission-Planning.git
cd UAV-Mission-Planning
cat repro/CODEX_START_HERE.md
cat repro/upstreams.lock.json
```

Keep a dedicated copy of the checkout. Generated source/build directories are
ignored by Git and can become large. The container name defaults to
`uav-gazebo`; it is separate from the original developer's `visual-nav-sim`.
The runtime still uses ROS domain 42, Gazebo partition `visual_nav_sim` and
Agent UDP 8889. Stop another instance using those values before launching this
one. Do not attach physical USB/serial flight controllers to this container.

## 2. Build the environment image and the pinned software

```bash
bash repro/ctl.sh build-image
bash repro/ctl.sh up
BUILD_JOBS=2 bash repro/ctl.sh build
bash repro/ctl.sh check
```

`build-image` installs the actual ROS/Gazebo and native compiler dependencies.
It obtains the matching PX4 setup scripts by revision and verifies their
SHA-256. `build` fetches all six pinned repositories, initializes PX4
submodules, applies canonical patches once, compiles PX4, the XRCE Agent,
px4_msgs, OpenVINS and Cartographer, creates the expected overlay/symlink
layout, and generates the world and mesh resources.

It also compiles and executes the loop-guard, pose-extrapolation and IMU/VIO
optimization tests. Their **new** outputs are required by the environment check
and copied to the runtime locations consumed by `runtime_snapshot.py`.
Historical test files are not substituted for new build tests.

Build logs are `logs/repro-*.log`. `fetch_sources.py` is idempotent for the
expected patches and refuses to switch a modified dependency checkout to a
different revision. Do not run the historical `prepare_v2.py` or `patch_*.py`
migration helpers after the canonical patches.

## 3. Check the sensor and mapper chain without flying

If a different simulation is active, confirm it is disarmed and stop it first.
For DDS images, raise the host socket limits for the current boot:

```bash
sudo sysctl -w net.core.rmem_max=8388608 net.core.wmem_max=8388608
bash repro/ctl.sh smoke
```

The smoke test starts the **source-material, passive-stereo** stack, then
checks the progressing simulation clock, left/right images and timestamp
pairing, IMU, downward range, OpenVINS odometry, PX4 status and a responding
Cartographer trajectory service. It verifies that GNSS was never published
and the aircraft remained disarmed. A pass stops the simulation cleanly.

The output is `results/reproduction/smoke.json`. This is an installation and
connectivity result, not a full course or flight-accuracy certification.

For a host without NVIDIA, create the container using:

```bash
UAV_GPU=software bash repro/ctl.sh up
```

The rendering mode is fixed at container creation. A failed NVIDIA Container
Toolkit setup is not automatically hidden by a software fallback when NVIDIA
was explicitly selected. Inspect the error and fix the host setup.

## 4. Open Gazebo and RViz on her Ubuntu desktop

Use a terminal in the logged-in graphical desktop, with `DISPLAY` and its
X11/Xwayland authentication cookie available:

```bash
bash repro/ctl.sh start
bash repro/ctl.sh gui
```

`start` leaves the vehicle **disarmed**. `gui` starts Gazebo and RViz using the
existing desktop cookie. It accepts the current `DISPLAY`, including `:1`;
it does not use `xhost +`. Desktop window sizes can be adjusted manually.
Gazebo/RViz logs are in `cartographer/v4/logs/`.

If the cookie cannot be discovered, set `XAUTHORITY` to the readable file used
by that desktop before running `gui`. A plain SSH shell usually does not have
the local desktop's display variables. For Wayland, an Xwayland session must
be available. The full recorded mission needs both actual windows because the
trial runner verifies and records them before issuing flight commands.

## 5. Explicit simulated flight and result export

After checking the scene and successful sensor/mapper health, the collaborator
may explicitly run a **simulation-only** course trial:

```bash
bash repro/ctl.sh trial trial-001
bash repro/ctl.sh finalize trial-001
```

This arms PX4 SITL and commands the mission. Use a new label every time.
It records MCAP, actual GUI video and camera video, saves active implementation
hashes, checks final disarm and evaluates the engineering checkpoint regions.
Outputs live in `cartographer/v4/results/trial-001/` and the corresponding
`data/` folder. Each recording can be hundreds of MB, so leave them out of Git.

Reset between trials only after confirmed disarm:

```bash
bash repro/ctl.sh stop
bash repro/ctl.sh start
bash repro/ctl.sh gui
```

`stop` preserves source and data and refuses an armed reset. The handoff does
not require re-running flights merely to inspect the environment.

## Common failures

| Symptom | What to check |
|---|---|
| Docker access denied | Start Docker; allow `sudo docker` or configure access on that host. |
| NVIDIA runtime / `--gpus` fails | Verify the host driver and Container Toolkit; create a separate software container for a smoke test. |
| Compilation killed | Lower `BUILD_JOBS=1`; check host memory and swap. |
| ROS Python import fails | Use the container's `/usr/bin/python3` and source the named ROS overlays; avoid host Conda. |
| Stereo image loss | Check DDS socket limits, simulation clock/rate and host load. |
| GUI cannot connect | Check the logged-in desktop's `DISPLAY` and `XAUTHORITY`; do not disable X authentication. |
| Mapper isn't loaded | Read v4 logs and the source/patch lock; don't bypass the runtime implementation check. |
| Existing tmux simulation | Finish/stop the existing disarmed instance; do not run two PX4s on UDP 8889/domain 42. |
| A patch no longer applies | Verify the exact SHA and a clean dependency checkout; do not apply migration helpers over it. |

## Validation and boundaries

The original v3/v4 episode results remain historical evidence. A new-machine
build/check/smoke pass is separate evidence. Consult the repository's
**Reproduce Linux runtime** GitHub Actions run for the fresh build status and
its downloadable logs. A green core-Python workflow alone does not establish
that Gazebo or PX4 compiled.

The seven-inch vehicle is an approximation, not an identified competition
airframe. Images and downward ToF remain simulation models; actual OAK firmware,
lighting, vibration, calibration, Orin load and physical Pixhawk integration
need separate validation. Published CAD may supply the declared navigation
prior; live simulator truth is confined to independent evaluation. See
`THIRD_PARTY.md` for component and course-asset licensing provenance.

### Where the files are

```text
repro/                         this English handoff, source lock, build/check/launch tools
docker/Dockerfile.repro        base dependency image built from the source lock
patches/                       canonical OpenVINS/Cartographer patches
course_source/                 small original OBJ/MTL course assets
scripts/                       resource generators and original diagnostics
config/                        ROS/Gazebo bridge
cartographer/v4/               current mission, sensors, fusion and evaluation
results/reproduction/          locally generated build/check/smoke provenance
```

Questions about how to use Codex are covered by the prompt in
`CODEX_START_HERE.md`; this repository is the actual source/environment handoff,
in addition to the earlier conversational context package.
