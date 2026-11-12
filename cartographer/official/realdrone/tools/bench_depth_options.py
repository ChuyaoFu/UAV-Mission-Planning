#!/usr/bin/env python3
"""Offline benchmark of the real-drone depth options (no ROS needed).

Timing: the REAL captured skybeta-05 pairs (capture_pair.py npz: 1024x640 mono + camera_info + TF), full pipeline
per option = downscale + rectify (+ band crop) + matcher + the sim's post-processing (+ upsample to 640x400), wall and
CPU per frame at a given cv thread count. Also the Humble Image.data publish cost (bytes vs array.array).
Accuracy: the venue-replica frames (outputs/venue-replica/frames/*.npz: noisy 640x400 OAK-like pair + depth_ideal
truth), each option's depth at the obstacle map's sampling (stride 4 on 640x400) vs truth in 0.3-6 m.

Usage: python3 bench_depth_options.py --pairs pair1.npz [--replica DIR --n 60] [--threads 1 2] [--json out.json]
"""
import argparse, array, ast, glob, json, math, sys, time
from collections.abc import Sequence
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stereo_depth_node import RectifyModel, DepthPipeline, OPTIONS, quat_xyzw_to_mat   # noqa: E402


def load_pairs(path):
    z = np.load(path, allow_pickle=True)
    info = ast.literal_eval(str(z['info'])); tf = ast.literal_eval(str(z['tf']))
    t = tf['oak_left_camera_optical_frame->oak_right_camera_optical_frame']
    il, ir = info['left'], info['right']
    return dict(L=z['left'], R=z['right'], Kl=np.array(il['K']).reshape(3, 3), Kr=np.array(ir['K']).reshape(3, 3),
                size=(il['w'], il['h']), R_lr=quat_xyzw_to_mat(*t[3:]), t_lr=np.array(t[:3]))


def humble_setter(value):
    """What rosidl_generator_py (Humble) runs when a non-array.array is assigned to a uint8[] field."""
    assert (isinstance(value, Sequence) and not isinstance(value, str) and all(isinstance(v, int) for v in value)
            and all(0 <= v < 256 for v in value))
    return array.array('B', value)


def time_options(P, names, threads, reps):
    out = {}
    for name in names:
        cv2.setNumThreads(threads)
        pipe = DepthPipeline(OPTIONS[name], P['Kl'], np.zeros(5), P['Kr'], np.zeros(5), P['size'], P['R_lr'], P['t_lr'])
        n = len(P['L']); pipe.run(P['L'][0], P['R'][0])
        w = c = 0.0; tr = tm = 0.0; d = None
        for i in range(reps):
            k = i % n
            t0 = time.perf_counter(); c0 = time.process_time()
            d, _, tim = pipe.run(P['L'][k], P['R'][k])
            w += time.perf_counter() - t0; c += time.process_time() - c0; tr += tim['rect']; tm += tim['match']
        v = d[d > 0]
        out[name] = dict(threads=threads, wall_ms=round(1e3 * w / reps, 2), cpu_ms=round(1e3 * c / reps, 2),
                         rect_ms=round(1e3 * tr / reps, 2), match_ms=round(1e3 * tm / reps, 2),
                         work=f"{pipe.work_size[0]}x{pipe.work_size[1]}", min_depth_m=round(pipe.min_depth, 3),
                         valid=round(float(np.mean(d > 0)), 3), median_m=round(float(np.median(v)), 3) if v.size else None)
    return out


def accuracy(names, replica_dir, n_frames, stride=4, zmin=0.3, zmax=6.0, truth='clean'):
    """Per option vs replica truth ('clean': depth_full where it equals depth_ideal within 1 %, i.e. no cage net / tunnel
    mesh in front, so the truth is unambiguous for stereo; or a npz key such as 'depth_full'),
    sampled like the obstacle map (stride 4 at 640x400). Relative error buckets by true range; `phantom_near`: returns
    closer than 0.8x truth and < 2 m (false obstacles, counted only where the truth is defined) per frame."""
    files = sorted(glob.glob(str(Path(replica_dir) / '*.npz')))
    files = [f for f in files if not Path(f).name.startswith('compare_photo')]
    if n_frames and len(files) > n_frames:
        files = [files[i] for i in np.linspace(0, len(files) - 1, n_frames).astype(int)]
    B = ((0.3, 2.0), (2.0, 4.0), (4.0, 6.0))
    acc = {k: dict(err={b: [] for b in B}, n_truth={b: 0 for b in B}, phantom=0,
                   rows=dict(top=[0, 0], mid=[0, 0], bot=[0, 0])) for k in names}
    for f in files:
        z = np.load(f, allow_pickle=True)
        L, R = z['image'], z['image_right']
        if truth == 'clean':
            gf, gi = z['depth_full'].astype(np.float32), z['depth_ideal'].astype(np.float32)
            gt = np.where(np.abs(gf - gi) <= 0.01 * gf, gf, np.nan).astype(np.float32)
        else:
            gt = z[truth].astype(np.float32)
        k = np.asarray(z['K'], float).ravel()
        K = k.reshape(3, 3) if k.size == 9 else np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1.0]])
        h, w = L.shape
        sub = (slice(0, h, stride), slice(0, w, stride))
        g = gt[sub]; fin = np.isfinite(g); ok = fin & (g >= zmin) & (g <= zmax)
        rr = np.arange(0, h, stride)[:, None] * np.ones((1, g.shape[1]))
        for k in names:
            pipe = DepthPipeline(OPTIONS[k], K, np.zeros(5), K, np.zeros(5), (w, h), np.eye(3), np.array([0.075, 0, 0]),
                                 rectified_input=True)
            d = pipe.run(L, R)[0][sub]
            a = acc[k]
            for b in B:
                m = ok & (g >= b[0]) & (g < b[1]); vv = m & (d > 0)
                a['n_truth'][b] += int(m.sum()); a['err'][b].extend((np.abs(d[vv] - g[vv]) / g[vv]).tolist())
            a['phantom'] += int((fin & (d > 0) & (d < 2.0) & (d < 0.8 * np.where(fin, g, 1e9))).sum())
            for nm, lo, hi in (('top', 0, h * 0.2), ('mid', h * 0.2, h * 0.8), ('bot', h * 0.8, h)):
                m = ok & (rr >= lo) & (rr < hi)
                a['rows'][nm][0] += int(m.sum()); a['rows'][nm][1] += int((m & (d > 0)).sum())
    out = {}
    for k, a in acc.items():
        r = dict(frames=len(files), truth=truth)
        for b in B:
            e = np.asarray(a['err'][b]); key = f'{b[0]:g}-{b[1]:g}m'
            r[key] = dict(valid=round(e.size / max(a['n_truth'][b], 1), 3),
                          med=round(float(np.median(e)), 4) if e.size else None,
                          p90=round(float(np.percentile(e, 90)), 3) if e.size else None,
                          out10=round(float(np.mean(e > 0.10)), 3) if e.size else None)
        r['phantom_near_per_frame'] = round(a['phantom'] / max(len(files), 1), 1)
        r['valid_by_rows'] = {nm: round(v[1] / max(v[0], 1), 3) for nm, v in a['rows'].items()}
        out[k] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', required=True); ap.add_argument('--replica'); ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--threads', type=int, nargs='+', default=[1, 2]); ap.add_argument('--reps', type=int, default=12)
    ap.add_argument('--options', nargs='+', default=list(OPTIONS)); ap.add_argument('--json')
    a = ap.parse_args()
    P = load_pairs(a.pairs)
    rep = dict(host=dict(machine=__import__('platform').machine(), cv=cv2.__version__), timing={}, publish={})
    for th in a.threads:
        rep['timing'][th] = time_options(P, a.options, th, a.reps)
        for k, v in rep['timing'][th].items():
            print(f'[time t{th}] {k:14s} {v}', flush=True)
    for nm, buf in (('depth_640x400_f32', np.zeros((400, 640), np.float32).tobytes()),
                    ('left_640x400_u8', np.zeros((400, 640), np.uint8).tobytes())):
        t0 = time.perf_counter(); humble_setter(buf); t1 = time.perf_counter(); array.array('B', buf); t2 = time.perf_counter()
        rep['publish'][nm] = dict(humble_bytes_ms=round(1e3 * (t1 - t0), 2), array_array_ms=round(1e3 * (t2 - t1), 3))
    print('[publish]', rep['publish'], flush=True)
    if a.replica:
        rep['accuracy'] = accuracy(a.options, a.replica, a.n)
        for k, v in rep['accuracy'].items():
            print(f'[acc] {k:14s} {v}', flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
