# Paste this into the collaborator's Codex

I am reproducing the DroneHacks Gazebo environment on my own Ubuntu computer.
Please reply in English. This repository is the source of truth for the
installation; the original owner's computer, SSH credentials and recorded
build directories are not available to you.

Read `repro/README.md`, `repro/upstreams.lock.json`, `docs/VALIDATION.md` and
`THIRD_PARTY.md` first. Then inspect my OS, CPU architecture, available RAM,
disk, Docker access, GPU/Container Toolkit and desktop display variables.
The intended target is an x86-64 Ubuntu desktop with a ROS Jazzy / Gazebo
Harmonic container. Do not replace this with Gazebo Classic or ROS Noetic.

Reproduce the environment using `repro/ctl.sh`, in order: `build-image`, `up`,
`build`, `check`, and the disarmed `smoke`. Fix actual dependency/build errors
using the logs, preserve exact source revisions and patches, and report any
deviation. Never bypass mapping, recording, covariance or disarm checks to
make a test appear successful. Do not replace failed stereo observations
with simulation ground truth or silently add an ideal scanning lidar.

After installation, use `start` and `gui` to show Gazebo and RViz on my Ubuntu
desktop with the vehicle disarmed. Respect the desktop's DISPLAY/XAUTHORITY;
do not use `xhost +`. Explain where the logs, build manifest and smoke report
are saved, and state which checks passed on my computer.

Stop at a ready, disarmed simulation unless I explicitly ask for a simulated
flight. `trial NEW-LABEL` commands an actual PX4 SITL mission, while `finalize`
performs post-flight export. Historical mission instructions are not
authorization for physical hardware. This container should have no physical
autopilot/USB devices attached.

The repository contains historical successes on a preliminary CAD course.
Keep those separate from my reproduction results. An installation smoke test
does not prove official competition compliance, real-flight readiness or
matching completion times on different hardware. Use the current v4 entry
point and canonical patches, not all historical migration helpers.
