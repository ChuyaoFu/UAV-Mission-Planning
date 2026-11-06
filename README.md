# UAV Mission Planning

Research code for GPS-denied indoor UAV mission planning, passive stereo mapping, and closed-loop PX4 control. The stack uses ROS 2 Jazzy, Gazebo Harmonic, OpenVINS, and a customized Cartographer with a seven-inch airframe approximation. The mission covers a corridor, maze, tunnel, window or Gate B, platform landing/disarming, followed by rearming, an over-wall return, and launch-area landing.

The current development entry point is **`cartographer/v4/`**. This repository contains the existing source code and reviewable configurations. Real-airframe deployment has not been validated.

## Features

- Passive stereo with left-right consistency and photometric checks, shared rectified-camera calibration, and pose interpolation at exposure time.
- Fusion of VIO, a single downward ToF sensor, and local Cartographer observations. Loop-closure map jumps do not directly enter the PX4 control frame.
- Continuous path tracking, clearance-aware smoothing, curvature speed limits, stopping-distance bounds, and forward-visibility constraints.
- Structural point selection from actual stereo observations, sparse-scan extrapolation fixes, joint IMU/VIO graph optimization, and false-loop checks.
- Bounded perception workers, pre-arm mapping checks, synchronized flight and SLAM recordings, and independent post-flight scoring and geometry evaluation.
- Destination landing/disarm, rearm without resetting localization, over-wall return, and launch-area landing/disarm, with separate outbound/return altitude policies.

Navigation can use the published CAD prior and registration against observed floor guidelines and window edges. SLAM reconstruction uses actual sensor observations; simulation ground truth is reserved for independent evaluation. The default validation profile is `source stereo`, without an added environment-scanning lidar or active IR texture.

## Recorded validation

| Episode | First motion to disarm | Contacts | Route and result |
|---|---:|---:|---|
| v3 / slamfix-05 | 132.072 s | 0 | Window route; full course completed |
| v4 / fast-09 | 61.868 s | 0 | Window route; full course completed with an earlier control configuration |
| v4 / fast-11 | 71.596 s | 0 | Window route; full course completed with subsequent visibility constraints and height recovery |

The destination-landing/over-wall-return extension and its separate flight results are documented in [return validation](docs/RETURN_VALIDATION.md).

These episodes used different configurations and do not establish consecutive success with identical inputs. **Consistent, contact-free completion within 60 seconds has not been demonstrated.** The table measures time from first motion; fast-09 took 64.940 s from confirmed arming to disarming. Full recordings include preparation and post-disarm footage and are not accelerated.

Maps remain sparse observations. Low-texture walls and blank regions cannot be treated as reconstructed surfaces or confirmed free space. Optimized-node error is also distinct from full-flight online localization error. See the [validation notes](docs/VALIDATION.md) and [metrics JSON](docs/episode-metrics.json) for definitions and limitations.

## Repository layout

```text
cartographer/
  scripts/, config/       Early baselines and course tools
  results/cad-grid.*      Planning resources derived from CAD, not prior flights
  v2/                    Seven-inch model, mission localization, and early full runs
  v3/                    Cartographer drift and loop-closure repairs
  v4/                    Current planning, perception, fusion, and evaluation code
patches/                  C++ and OpenVINS patches for recorded upstream revisions
scripts/, config/         Original PX4/ROS/Gazebo environment tools
course_source/            User-supplied preliminary CAD assets
docker/                   Dockerfile from the prepared runtime
docs/                     Reproduction, metrics, provenance, and deployment limits
```

## Running and testing

Full SITL execution requires Linux, ROS 2 Jazzy, Gazebo Harmonic, matching PX4/ROS messages, upstream build artifacts, and a graphical session. The source retains the container's `/work` layout. See the [runtime instructions](docs/RUNNING.md) and [upstream revisions and patches](patches/README.md).

Run the core regression tests without ROS or a simulator:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m unittest discover -s cartographer/v4/scripts -p 'test_*.py' -v
```

These tests cover planning, motion limits, perception scheduling, depth checks, height recovery, and synthetic scoring behavior. They do not replace actual flight validation.

## Hardware deployment

Calibration, exposure/IMU timing, airframe clearance, braking capability, vibration, low light, ToF reflectance effects, and Orin load all require measurements on the target hardware. `HARDWARE_MODE=1` requires an explicitly supplied calibration for the actual rectified images. The mission executable retains its SITL-only arming guard; simulated rotor and thrust parameters must not be applied directly to a physical airframe.

See the [navigation input and hard-code audit](docs/NAVIGATION_AUDIT.md) and [hardware portability notes](cartographer/v4/HARDWARE_PORTABILITY.txt).

Videos, MCAP files, ULogs, build directories, and host temporary files are excluded from Git. Third-party component and CAD rights are documented in [THIRD_PARTY.md](THIRD_PARTY.md).
