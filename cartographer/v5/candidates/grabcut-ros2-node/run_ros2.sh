#!/usr/bin/env bash
set -euo pipefail
node_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ $# -lt 4 ]]; then
  echo 'Usage: run_ros2.sh ACTUAL_RIG_YAML ACTUAL_ROBOT_JSON ACTUAL_MASK_DIR LINUX_LBD_SO [ENGINE_VARIANT] [ENABLE=false] [MODEL=temporal] [extra ROS args...]' >&2
  exit 2
fi
rig_path="$1"; robot_path="$2"; mask_dir="$3"; lbd_path="$4"
variant="${5:-cold5_freeze1_seedvalidated_240p}"; enabled="${6:-false}"; model="${7:-ground_ipm}"
shift "$(( $# < 7 ? $# : 7 ))"
exec env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 "$node_dir/structural_evidence_node.py" --ros-args -p use_sim_time:=true -p "rig_path:=$rig_path" -p "robot_path:=$robot_path" -p "mask_dir:=$mask_dir" -p "lbd_library:=$lbd_path" -p "engine_variant:=$variant" -p "enable_inferred_free:=$enabled" -p "model_kind:=$model" -p native_joint_topic:=auto -p include_floor_completion:=true -p native_diagonal_retreat_verified:=true -p inferred_input_max_age_s:=0.5 -p raw_input_max_age_s:=0.25 -p compute_budget_s:=0.4 -p max_hz:=2.5 -p voxel_resolution_m:=0.1 "$@"
