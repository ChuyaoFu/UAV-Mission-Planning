#!/usr/bin/env python3
"""Validate frozen planar output provenance/NaNs without any geometry truth."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from planar_evidence_runtime import pack_evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sensor_inputs')
    parser.add_argument('predictions')
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    rows = []
    manifest = json.loads((Path(args.predictions) / 'benchmark.json').read_text())
    for path in sorted(Path(args.sensor_inputs).glob('frame-*.npz')):
        prediction = Path(args.predictions) / path.name
        with np.load(path, allow_pickle=False) as raw, np.load(prediction, allow_pickle=False) as out:
            measured = raw['measured_stereo']
            preserved = out['measured_depth']
            assert measured.dtype == preserved.dtype == np.float32
            assert np.array_equal(measured.view(np.uint32), preserved.view(np.uint32)), 'Raw measurement bits changed'
            assert np.array_equal(raw['stamp'], out['stamp']), 'Exposure stamp changed'
            surface = out['plane_only_inferred_depth']
            lower = out['free_ray_lower_bound']
            interval = out['depth_interval_m']
            ids = out['plane_id']
            support = np.isfinite(surface)
            for array in (surface, lower, interval, ids):
                assert array.shape == measured.shape, 'Native shape changed'
            for array in (surface, lower, interval):
                assert array.dtype == np.float32
                assert np.isnan(array[~support]).all(), 'Unsupported output must remain NaN'
            assert np.array_equal(np.isfinite(lower), support)
            assert np.array_equal(np.isfinite(interval), support)
            assert np.array_equal(ids >= 0, support)
            assert (ids[~support] == -1).all()
            assert not (support & ~np.isnan(measured)).any(), 'Inference overwrote an original pixel'
            assert (lower[support] > .25).all() and (surface[support] < 8.).all()
            assert (lower[support] < surface[support]).all()
            assert (interval[support] <= .12 * surface[support] + 1e-6).all()
            np.testing.assert_allclose(interval[support], surface[support] - lower[support], atol=2e-6)
            result = dict(inferred_depth=surface, free_ray_lower_bound=lower, depth_interval_m=interval, plane_id=ids)
            evidence = pack_evidence(measured, result, enabled=True)
            assert np.array_equal(evidence[..., 0].view(np.uint32), measured.view(np.uint32))
            assert np.array_equal(np.isfinite(evidence[..., 1]), support)
            rows.append(dict(frame=path.name, input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), prediction_sha256=hashlib.sha256(prediction.read_bytes()).hexdigest(), native_inferred_pixels=int(support.sum()), contract_passed=True))
    assert rows, 'Refuse empty verification'
    report = dict(source_sha256=manifest['source_sha256'], frame_count=len(rows), contract_passed=True, truth_used=False, scope='Provenance/NaN/interval/wire invariants only; not geometric safety or hardware timing qualification', frames=rows)
    Path(args.report).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
