#!/usr/bin/env python3
"""Synthetic sensor-array resource probe. No ROS, simulator, flight or GT input.

Run each case in its own process so peak RSS is attributable to that workload.
The timing is the machine running this script; never rename it Orin evidence.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
import cv2
import numpy as np
from planar_stereo import PlanarStereo, PlanePolicy
from planar_evidence_runtime import pack_evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['blank', 'sparse-plane', 'three-planes', 'dense-random', 'fully-measured', 'fragmented-24-regions'], required=True)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--output', required=True)
    parser.add_argument('--budget-s', type=float, default=None)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError('At least one repeat is required')
    cv2.setNumThreads(1)
    shape = (800, 1280)
    k = np.array([759.8347725610749, 759.8347725610749, 640., 400.])
    left = np.full(shape, 170, np.uint8)
    raw = np.full(shape, np.nan, np.float32)
    rng = np.random.default_rng(31)
    if args.case == 'sparse-plane':
        raw[32:768:8, 64:1216:8] = 2.
    elif args.case == 'three-planes':
        for i, depth in enumerate((1., 2., 3.)):
            raw[32 + i * 240:240 + i * 240:6, 64:1216:6] = depth
    elif args.case == 'dense-random':
        valid = rng.random(shape) < .5
        raw[valid] = rng.uniform(.3, 7.9, int(valid.sum())).astype(np.float32)
    elif args.case == 'fully-measured':
        raw[:] = 2.
    right = left.copy()
    if args.case == 'fragmented-24-regions':
        left[:] = right[:] = 0
        for row in range(6):
            for col in range(4):
                y, x = 16 + row * 128, 16 + col * 312
                depth = 1. + row % 3
                tone = 140 + 40 * ((row + col) % 2)
                left[y:y+112, x:x+292] = tone
                shift = int(round(k[0] * .075 / depth))
                right[y:y+112, max(0, x-shift):max(0, x+292-shift)] = tone
                raw[y+8:y+104:6, x+8:x+284:6] = depth
    mask = np.zeros(shape, bool)
    model = PlanarStereo()
    timings, cpu, pack_times, fractions, planes, regions, budget_exhausted = [], [], [], [], [], [], []
    for _ in range(args.repeats + 1):
        start, cpu_start = time.perf_counter(), time.process_time()
        result = model.compute(left, raw, k, .075, mask, right=right, right_self_mask=mask, deadline_monotonic=None if args.budget_s is None else start + args.budget_s)
        elapsed, cpu_elapsed = time.perf_counter() - start, time.process_time() - cpu_start
        start = time.perf_counter()
        evidence = pack_evidence(raw, result, enabled=True)
        pack_elapsed = time.perf_counter() - start
        np.testing.assert_array_equal(evidence[..., 0], raw)
        assert not np.isfinite(evidence[..., 1][np.isfinite(raw)]).any()
        timings.append(elapsed)
        cpu.append(cpu_elapsed)
        pack_times.append(pack_elapsed)
        fractions.append(float(np.isfinite(evidence[..., 1]).mean()))
        planes.append(len(result['planes']))
        regions.append(sum(row['supported_regions'] for row in result['planes']))
        budget_exhausted.append(bool(result.get('budget_exhausted', False)))
    # Exclude exactly one warm-up from the reported latency distribution.
    timings, cpu, pack_times = timings[1:], cpu[1:], pack_times[1:]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == 'darwin' else peak * 1024
    report = dict(case=args.case, repeats=args.repeats, resolution=[1280, 800], machine=dict(system=platform.system(), release=platform.release(), machine=platform.machine(), python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__), target_hardware_qualified=False, source_sha256=hashlib.sha256(Path(__file__).with_name('planar_stereo.py').read_bytes()).hexdigest(), policy=vars(PlanePolicy()), requested_opencv_threads=1, reported_opencv_threads=cv2.getNumThreads(), opencv_parallel_framework=next((line.strip() for line in cv2.getBuildInformation().splitlines() if 'Parallel framework:' in line), 'unknown'), blas_env={name: os.environ[name] for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}, compute_p50_s=float(np.percentile(timings, 50)), compute_p95_s=float(np.percentile(timings, 95)), compute_max_s=max(timings), process_cpu_mean_s=float(np.mean(cpu)), evidence_pack_p95_s=float(np.percentile(pack_times, 95)), peak_process_rss_mib=peak_bytes / 2**20, inferred_fraction_min=min(fractions), inferred_fraction_max=max(fractions), fitted_planes_max=max(planes), supported_regions_max=max(regions), cooperative_budget_s=args.budget_s, budget_exhausted_frames=sum(budget_exhausted[1:]), limits='RSS is an observed desktop process high-water mark including arrays/results, not a proven upper bound or ROS transport/RMW measurement. Native workload and loop counts are bounded; latency gate is soft, not preemption.')
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
