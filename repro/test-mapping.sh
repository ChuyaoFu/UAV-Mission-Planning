#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /work/cartographer/v3/carto_ws/install/setup.bash
build=/work/cartographer/v3/carto_ws/build/cartographer
results=/work/cartographer/v3/results
mkdir -p "$results" /work/cartographer/v4/results
declare -a targets=(
  cartographer.mapping.internal.constraints.stereo_loop_guard_test
  cartographer.mapping.pose_extrapolator_test
  cartographer.mapping.internal.optimization.optimization_problem_3d_test
)
cmake --build "$build" --parallel "${BUILD_JOBS:-2}" --target "${targets[@]}"
"$build/${targets[0]}" > "$results/loop-guard-tests.txt" 2>&1
"$build/${targets[1]}" > "$results/extrap-after-tests.txt" 2>&1
"$build/${targets[2]}" > "$results/optimizer-after-tests.txt" 2>&1
# A nonexistent gtest filter exits successfully with zero tests; reject that.
for file in loop-guard-tests.txt extrap-after-tests.txt optimizer-after-tests.txt; do
 grep -Eq '\[  PASSED  \] [1-9][0-9]* test' "$results/$file"
 cp "$results/$file" /work/cartographer/v4/results/
 tail -4 "$results/$file"
done
