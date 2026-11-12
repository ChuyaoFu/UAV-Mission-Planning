#!/usr/bin/env python3
"""Sanity-check the packaged replica set against its own ground truth.

Usage: python check_dataset.py [--root outputs/venue-replica]
Checks per frame: dtypes/shapes/K; stereo `depth` NaN or within the device range
(fx*B/96 .. fx*B/1 m); the two ground-truth passes agree (depth_ideal >= depth_full);
stereo depth on well-textured structure (boards, banners, rings, fabric, rocks, blocks, pad; 0.33..6 m,
>= 2000 px) has median relative error < 5 %;
and geometric consistency of labels vs ground-truth depth (`depth_ideal`):
  - landing target (flat blue pad) top centre: back-projected depth point within 2 cm of the label
  - window centre (board facing camera): depth through the hole is >= 0.3 m beyond the board
  - floor: back-projected depth on the start pad region has height ~0
Prints a summary; exit code 1 on any failure.
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset import load_frame                                     # noqa: E402
sys.path.insert(0, os.path.dirname(HERE))
from common.geometry import OFFICIAL_K                              # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))


def pix_depth(fr, uv, r=1):
    u, v = int(round(uv[0])), int(round(uv[1]))
    d = fr.depth[max(v - r, 0):v + r + 1, max(u - r, 0):u + r + 1]
    d = d[np.isfinite(d)]
    return float(np.median(d)) if d.size else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join(REPO, 'outputs', 'venue-replica'))
    a = ap.parse_args()
    idx = json.load(open(os.path.join(a.root, 'index.json')))
    fails, stats = [], {'landing_top_err_cm': [], 'window_gap_m': [], 'floor_h_err_cm': [],
                        'stereo_rel_err_p50_pct': [], 'stereo_fill_pct': []}
    for f in idx['frames']:
        fr, lab = load_frame(os.path.join(a.root, f['file']), depth_key='depth_ideal')
        npz = np.load(os.path.join(a.root, f['file']))
        full, stereo = npz['depth_full'], npz['depth']
        if not (npz['image_right'].shape == (400, 640) and npz['surface'].shape == (400, 640) and stereo.dtype == np.float32):
            fails.append((f['id'], 'stereo fields')); continue
        sv = stereo[np.isfinite(stereo)]
        zmin, zmax = fr.K[0] * 0.075 / 96, fr.K[0] * 0.075 / 1.0
        if sv.size and (sv.min() < zmin - 1e-3 or sv.max() > zmax + 1e-3):
            fails.append((f['id'], f'stereo depth out of device range [{sv.min():.2f}, {sv.max():.2f}]'))
        e = f.get('stereo_rel_err_p50_solid')
        if e is not None and f['group'] != 'compare':          # compare views look through the net from outside
            stats['stereo_rel_err_p50_pct'].append(100 * e)
            if e > 0.05:
                fails.append((f['id'], f'stereo median rel err {e:.3f} on solid surfaces'))
        stats['stereo_fill_pct'].append(100 * f['depth_valid_frac'])
        ok = (fr.image.dtype == np.uint8 and fr.image.shape == (400, 640) and fr.depth.dtype == np.float32
              and fr.depth.shape == (400, 640) and np.allclose(fr.K, OFFICIAL_K)
              and np.allclose(fr.R_nc @ fr.R_nc.T, np.eye(3), atol=1e-6))
        if not ok:
            fails.append((f['id'], 'shape/dtype/K/R')); continue
        d = fr.depth
        if np.any(d[np.isfinite(d)] <= 0):
            fails.append((f['id'], 'non-positive depth'))
        both = np.isfinite(d) & np.isfinite(full)
        if np.mean(d[both] < full[both] - 0.01) > 0.001:
            fails.append((f['id'], 'depth_ideal < depth_full'))
        fl, W = lab['frame'], lab['world']
        st = fl['landing_top_center']
        uu, vv = (int(round(x)) for x in st['px'][0]) if st['in_fov'][0] else (0, 0)
        pad_patch = npz['surface'][max(vv - 1, 0):vv + 2, max(uu - 1, 0):uu + 2]
        pad_id = idx['surface_names'].index('landing_pad') if 'landing_pad' in idx['surface_names'] else -1
        if st['in_fov'][0] and st['unoccluded'][0] and st['z'][0] < 4 and np.all(pad_patch == pad_id):   # not through a slit
            # sample a few px *inside* the top face (centre)
            z = pix_depth(fr, st['px'][0])
            u, v = st['px'][0]
            fx, fy, cx, cy = fr.K
            p = fr.R_nc @ np.array([(u - cx) / fx * z, (v - cy) / fy * z, z]) + fr.cam_ned
            err = abs(p[2] - W['landing_target']['top_d'])
            stats['landing_top_err_cm'].append(100 * err)
            if err > 0.02:
                fails.append((f['id'], f'landing target top err {err:.3f} m'))
        wc = fl['window_center']
        if f['group'] == 'window' and wc['in_fov'][0]:
            z_hole = pix_depth(fr, wc['px'][0], r=3)
            stats['window_gap_m'].append(z_hole - wc['z'][0])
            if not z_hole > wc['z'][0] + 0.3:
                fails.append((f['id'], f'window hole depth {z_hole:.2f} vs board {wc["z"][0]:.2f}'))
        if f['group'] == 'start':
            pts, vu = fr.points_ned(stride=4, max_range=1.0)
            near = pts[np.abs(pts[:, 0]) < 0.25]
            if len(near):
                e = float(np.median(np.abs(near[:, 2])))
                stats['floor_h_err_cm'].append(100 * e)
                if e > 0.01:
                    fails.append((f['id'], f'floor height err {e:.3f}'))
    summ = {k: (len(v), round(float(np.max(v)), 3) if v else None) for k, v in stats.items()}
    print(f"{len(idx['frames'])} frames checked; (n, max) per check: {summ}")
    print('FAILURES:', fails if fails else 'none')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
