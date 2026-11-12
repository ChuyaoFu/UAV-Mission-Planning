#!/usr/bin/env python3
"""Smoke test of the shared official code as synced into the profile (run in the container from the profile root).
  python3 ubuntu_compat/smoke_test.py      -> imports, default params that matter for the Ubuntu runs, code sha
"""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mission.core import CoursePrior, Mission, Params  # noqa: E402
from node.obstacle_map import ObstacleMap  # noqa: E402
from perception import floor, tunnel, window  # noqa: E402,F401

p = Params()
Mission(CoursePrior.load(str(ROOT / 'ubuntu_compat/course_prior.replica.json')), p)
om = ObstacleMap()
keys = ['v_touch', 'push_m', 'tunnel_standoff_m', 'detect_min_conf', 'side_max_m', 'fence_margin_m']
out = {k: getattr(p, k, None) for k in keys}
out.update(map_min_hits=om.min_hits, map_carve=getattr(om, 'carve', None), map_carve_margin=getattr(om, 'carve_margin', None))
dig = {}
for d in ('common', 'node', 'mission', 'perception', 'tools'):
    for f in sorted((ROOT / d).rglob('*.py')):
        dig[str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
out['code_sha'] = hashlib.sha256(json.dumps(dig, sort_keys=True).encode()).hexdigest()[:16]
print(json.dumps(out))
