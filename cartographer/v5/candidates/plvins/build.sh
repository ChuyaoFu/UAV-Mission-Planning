#!/usr/bin/env bash
# Verify the source archive before the isolated source build.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python3 - "$root" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
pin=json.loads((root/'PIN.json').read_text())
actual=hashlib.sha256((root/'upstream.tar.gz').read_bytes()).hexdigest()
if actual != pin['upstream_tar_sha256']:
    raise SystemExit('Pinned PL-VINS source archive checksum mismatch')
print('Verified PL-VINS source',pin['commit'],actual)
PY
exec docker build -f "$root/container/Dockerfile.noetic" --build-arg BUILD_JOBS=2 -t "${1:-uav-plvins-candidate:noetic}" "$root"
