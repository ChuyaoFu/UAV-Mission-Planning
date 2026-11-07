#!/usr/bin/env python3
"""Frozen sensor-only planar candidate runner. Never arms or publishes."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from planar_stereo import PlanarStereo, PlanePolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs')
    parser.add_argument('output')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    model, rows = PlanarStereo(), []
    for path in sorted(Path(args.inputs).glob('frame-*.npz')):
        with np.load(path, allow_pickle=False) as data:
            right_mask = data['right_self_mask'] if 'right_self_mask' in data else None
            start = time.perf_counter()
            result = model.compute(data['left'], data['measured_stereo'], data['K'], float(data['baseline']), data['self_mask'], right=data['right'], right_self_mask=right_mask)
            elapsed = time.perf_counter() - start
            extra, measured = result['inferred_depth'], result['measured_depth']
            merged = np.where(np.isfinite(measured), measured, result['free_ray_lower_bound'])
            # Legacy evaluator field names retained only for independent scoring.
            np.savez_compressed(out / path.name, stamp=data['stamp'], measured_depth=measured, inferred_depth=merged, raw_inferred_depth=np.where(np.isfinite(measured), measured, extra), plane_only_inferred_depth=extra, free_ray_lower_bound=result['free_ray_lower_bound'], depth_interval_m=result['depth_interval_m'], plane_id=result['plane_id'])
            rows.append(dict(frame=path.name, stamp=float(data['stamp']), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), compute_s=elapsed, measured_fraction=float(np.isfinite(measured).mean()), inferred_fraction=float(np.isfinite(extra).mean()), combined_fraction=float(np.isfinite(merged).mean()), native_measured_overlap_count=int((np.isfinite(extra) & ~np.isnan(measured)).sum()), right_self_mask_available=right_mask is not None, planes=result['planes']))
            print(json.dumps(rows[-1]), flush=True)
    if not rows:
        raise ValueError('No stereo samples; refusing an empty benchmark')
    report = dict(algorithm='Bounded sensor-supported planar disparity interpolation', safety_revision='native_ray_veto_v1', source_sha256=hashlib.sha256(Path(__file__).with_name('planar_stereo.py').read_bytes()).hexdigest(), policy=vars(PlanePolicy()), navigation_enabled=False, target_hardware_qualified=False, requested_opencv_threads=1, reported_opencv_threads=cv2.getNumThreads(), geometric_prior='Local planar interpolation inside eroded inlier hulls and .15m actual native-ray 3D support distance; native foreground, edges, photo and projected right occlusion vetoes. Conditional geometric inference, not certified free space.', input='Synchronized mono stereo, LR/photo-filtered metric stereo and available self masks only; no world/pose/CAD/floor input', free_ray_warning='Lower optical-axis Z endpoints are free-only evidence, never inferred occupied hits. Legacy merged fields are offline scorer-only and must not be published as camera depth.', mask_warning='A missing right self mask is reported per frame; live adapter requires both masks.', frames=rows)
    (out / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
