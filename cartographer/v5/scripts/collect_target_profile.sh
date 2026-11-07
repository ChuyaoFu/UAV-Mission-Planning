#!/usr/bin/env bash
# Read-only identification/profiling. Never changes JetPack, clocks or power mode.
set -eu
out=${1:?output directory required};seconds=${2:-60}
mkdir -p "$out"
uname -a > "$out/kernel.txt"
cat /etc/os-release > "$out/os-release.txt"
if [ -f /etc/nv_tegra_release ];then cat /etc/nv_tegra_release > "$out/nv-tegra-release.txt";fi
dpkg-query -W nvidia-jetpack nvidia-l4t-core 2>&1 > "$out/jetpack-packages.txt" || true
if command -v nvpmodel >/dev/null;then nvpmodel -q > "$out/power-mode.txt" 2>&1 || true;fi
if command -v tegrastats >/dev/null;then timeout --signal=INT "$seconds" tegrastats --interval 1000 > "$out/tegrastats.txt" 2>&1 || true;fi
printf '%s\n' 'These measurements describe this host only. Use the real stereo/IMU driver and flight-control workload for target acceptance.' > "$out/measurement-scope.txt"
