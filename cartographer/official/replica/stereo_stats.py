#!/usr/bin/env python3
"""Per-surface hole fraction and stereo depth error of the packaged replica set.

Usage: python stereo_stats.py [--root outputs/venue-replica]
For every non-compare frame and every surface type (npz `surface`), over pixels whose true
range is inside the device range (0.33 m .. 6 m) and right of the 96-px left band (no
matches possible there), reports: fill = stereo returns / pixels; error of returns vs the
truth (`depth_ideal`; for cage_net / black_mesh the nearest truth is `depth_full`), as median
absolute (cm) / median relative / p90 relative / share > 10 % off. For net and mesh pixels it
also splits returns into 'on the net/mesh', 'on what is behind it', 'neither'.
Writes ROOT/stereo_stats.json and prints a Markdown table.
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stereo_depth as SD    # noqa: E402
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
ORDER = ['floor', 'banner_panel', 'brass_ring', 'tunnel_fabric', 'black_mesh', 'cage_net', 'tan_board',
         'landing_pad', 'cinder_block', 'floor_tape', 'rock', 'hall', 'other']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join(REPO, 'outputs', 'venue-replica'))
    a = ap.parse_args()
    idx = json.load(open(os.path.join(a.root, 'index.json')))
    names = idx['surface_names']
    fx = idx['K'][0]
    zmin = fx * SD.BASELINE_M / (SD.SGBM['numDisparities'] - 1)
    band = SD.SGBM['numDisparities']
    acc = {n: dict(px=0, px_all=0, ret=0, ret_all=0, abs=[], rel=[], on_near=0, on_far=0, n_frames=0) for n in names}
    for f in idx['frames']:
        if f['group'] == 'compare':
            continue
        d = np.load(os.path.join(a.root, f['file']))
        st, di, df, s = d['depth'], d['depth_ideal'], d['depth_full'], d['surface']
        inr = np.isfinite(df) & (df >= zmin) & (df < 6.0)
        cols = np.zeros_like(inr); cols[:, band:] = True
        for k, n in enumerate(names):
            m_all = (s == k) & inr
            if not m_all.any():
                continue
            A = acc[n]; A['n_frames'] += 1
            A['px_all'] += int(m_all.sum()); A['ret_all'] += int((m_all & np.isfinite(st)).sum())
            m = m_all & cols
            r = m & np.isfinite(st)
            A['px'] += int(m.sum()); A['ret'] += int(r.sum())
            if not r.any():
                continue
            ref = df if n in ('cage_net', 'black_mesh') else di
            ok = r & np.isfinite(ref)
            e = st[ok] - ref[ok]
            A['abs'] += list(np.abs(e)[::5]); A['rel'] += list((np.abs(e) / ref[ok])[::5])
            if n in ('cage_net', 'black_mesh'):
                near = np.abs(st - df) < 0.05 * df
                far = np.isfinite(di) & (np.abs(st - di) < 0.05 * di)
                A['on_near'] += int((r & near).sum()); A['on_far'] += int((r & far & ~near).sum())
    out = {}
    for n in ORDER:
        A = acc[n]
        if A['px'] < 5000:
            continue
        rel = np.array(A['rel']); ab = np.array(A['abs'])
        o = {'frames': A['n_frames'], 'pixels': A['px'],
             'hole_frac': round(1 - A['ret'] / A['px'], 3),
             'hole_frac_incl_left_band': round(1 - A['ret_all'] / A['px_all'], 3),
             'abs_err_p50_cm': round(float(np.median(ab)) * 100, 1) if ab.size else None,
             'rel_err_p50_pct': round(float(np.median(rel)) * 100, 2) if rel.size else None,
             'rel_err_p90_pct': round(float(np.percentile(rel, 90)) * 100, 1) if rel.size else None,
             'bad_gt10pct': round(float(np.mean(rel > 0.10)), 3) if rel.size else None}
        if n in ('cage_net', 'black_mesh') and A['ret']:
            o['returns_on_net_or_mesh'] = round(A['on_near'] / A['ret'], 3)
            o['returns_on_background'] = round(A['on_far'] / A['ret'], 3)
            o['returns_neither'] = round(1 - (A['on_near'] + A['on_far']) / A['ret'], 3)
        out[n] = o
    json.dump({'range_m': [round(zmin, 3), 6.0], 'left_band_px': band, 'params': SD.params(), 'surfaces': out},
              open(os.path.join(a.root, 'stereo_stats.json'), 'w'), indent=1)
    print('| surface | frames | holes | holes incl. left band | |err| p50 | rel p50 | rel p90 | >10 % off | net/mesh split (on / behind / neither) |')
    print('|---|---|---|---|---|---|---|---|---|')
    for n, o in out.items():
        split = (f"{o['returns_on_net_or_mesh']:.0%} / {o['returns_on_background']:.0%} / {o['returns_neither']:.0%}"
                 if 'returns_on_net_or_mesh' in o else '')
        print(f"| {n} | {o['frames']} | {o['hole_frac']:.0%} | {o['hole_frac_incl_left_band']:.0%} | {o['abs_err_p50_cm']} cm | "
              f"{o['rel_err_p50_pct']} % | {o['rel_err_p90_pct']} % | {o['bad_gt10pct']:.0%} | {split} |")


if __name__ == '__main__':
    main()
