#!/usr/bin/env python3
"""Make independent scorer views of frozen outputs; reads no scene or truth.

Legacy source scorer expects inferred_depth/raw_inferred_depth. This exports
free-only/surface-only arrays to avoid diluting inferred-ray errors with raw hits.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('frozen_predictions')
    parser.add_argument('free_only_output')
    args = parser.parse_args()
    src, dst = Path(args.frozen_predictions), Path(args.free_only_output)
    if dst.exists():
        raise ValueError('Use a new output directory; never overwrite a frozen scorer view')
    dst.mkdir(parents=True)
    rows = []
    for path in sorted(src.glob('frame-*.npz')):
        with np.load(path, allow_pickle=False) as data:
            target = dst / path.name
            np.savez_compressed(target, stamp=data['stamp'], inferred_depth=data['free_ray_lower_bound'], raw_inferred_depth=data['plane_only_inferred_depth'])
            rows.append(dict(frame=path.name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), scorer_view_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    if not rows:
        raise ValueError('No frozen predictions found')
    manifest = json.loads((src / 'benchmark.json').read_text())
    (dst / 'scorer-view.json').write_text(json.dumps(dict(source_model_sha256=manifest['source_sha256'], truth_used=False, field_meaning=dict(inferred_depth='inferred free-only lower optical-axis Z; not occupied hit', raw_inferred_depth='inferred planar surface only; not measured'), frames=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
