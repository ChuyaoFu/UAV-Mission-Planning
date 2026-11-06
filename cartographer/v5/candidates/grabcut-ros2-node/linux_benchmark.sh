#!/usr/bin/env bash
set -euo pipefail
adapter_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Linux benchmark requires a real Linux host; no macOS result will be labelled Linux.' >&2
  exit 2
fi
cmake -S "$adapter_dir/native" -B "$adapter_dir/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$adapter_dir/build" -j2
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 "$adapter_dir/bench_stream.py" --fixtures "$adapter_dir/replay-fixtures" --rig "$adapter_dir/replay-calibration/kalibr_imucam_chain.yaml" --mask-dir "$adapter_dir/replay-calibration" --lbd-library "$adapter_dir/build/libboundary_lbd.so" --unbounded-diagnostic --output "$adapter_dir/benchmark-linux-full" > "$adapter_dir/benchmark-linux-full.log" 2>&1
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 "$adapter_dir/bench_stream.py" --fixtures "$adapter_dir/replay-fixtures" --rig "$adapter_dir/replay-calibration/kalibr_imucam_chain.yaml" --mask-dir "$adapter_dir/replay-calibration" --lbd-library "$adapter_dir/build/libboundary_lbd.so" --max-compute-s .4 --output "$adapter_dir/benchmark-linux-budget" > "$adapter_dir/benchmark-linux-budget.log" 2>&1
python3 - "$adapter_dir" <<'PY'
import json,sys
from pathlib import Path
for mode in ('full','budget'):
 r=json.load(open(Path(sys.argv[1])/('benchmark-linux-'+mode)/'benchmark.json'))
 print(mode,r['platform'],[(x['file'],round(x['wall_s'],4),x['stats']['status'],x.get('different_surface_mask_pixels')) for x in r['rows']])
PY
