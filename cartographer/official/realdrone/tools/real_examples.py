#!/usr/bin/env python3
"""Artifact types + example PNGs for real_eval.py results (its --save-npz).
Per mission-relevant near return labelled phantom (see-through in other views) or real, image-side causes at its pixel:
  periodic   NetMask periodicity score >= 0.25 (net-like texture)        glare     5x5 max of the left image >= 245
  textureless local std (7x7, blur 1) < 3 DN                             blur      frame angular rate > 60 deg/s or
  edge       within 2 px of a depth step > 30 %                                     Laplacian variance < 0.5 x median
  below_floor more than 0.15 m under the floor
(a return can have several). Renders a few frames: rectified left with near returns by label | OFF depth |
model+noise depth | what the filters removed.
Usage: python3 real_examples.py NPZ OUT_DIR TAG [--frames i,j,...]"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from net_mask import NetMask  # noqa: E402


def rot_rate(Rw, t):
    w = np.zeros(len(t))
    for i in range(1, len(t)):
        dR = Rw[i - 1].T @ Rw[i]
        ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        w[i] = ang / max(t[i] - t[i - 1], 1e-3)
    w[0] = w[1] if len(t) > 1 else 0
    return w


def colorize(z, zmax=5.0):
    zz = np.clip(z / zmax, 0, 1)
    c = cv2.applyColorMap((255 * (1 - zz)).astype(np.uint8), cv2.COLORMAP_TURBO)
    c[z <= 0] = 0
    return c


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('npz'); ap.add_argument('out'); ap.add_argument('tag')
    ap.add_argument('--frames', default=''); ap.add_argument('--on', default='mask_noise')
    a = ap.parse_args()
    Z = np.load(a.npz)
    D, lab, hgt, Lw, t, Rw = Z['Doff'], Z['lab'], Z['hgt'], Z['Lw'], Z['t'], Z['Rw']
    fx, fy, cx, cy = Z['K']; n, H, W = D.shape
    keeps = {k[5:]: Z[k] for k in Z.files if k.startswith('keep_')}
    v, u = np.mgrid[0:H, 0:W].astype(np.float32); rx, ry = (u - cx) / fx, (v - cy) / fy
    rngf = np.sqrt(1 + rx * rx + ry * ry)
    w = rot_rate(Rw, t)
    nm = NetMask()
    sharp = np.array([cv2.Laplacian(Lw[i], cv2.CV_32F).var() for i in range(n)])
    blur_f = (w > 60) | (sharp < 0.5 * np.median(sharp))
    cats = ('periodic', 'glare', 'textureless', 'blur', 'edge', 'below_floor', 'none_of_these')
    cnt = {g: {c: 0 for c in cats} for g in ('phantom_off', 'real_off', 'phantom_left_on', 'removed_on_phantom', 'removed_on_real')}
    tot = {g: 0 for g in cnt}
    per_frame = []
    on = keeps[a.on]
    for i in range(n):
        z = D[i].astype(np.float32); h = hgt[i].astype(np.float32)
        mis = (lab[i] > 0) & (h > 0.10) & (z * rngf >= 0.4)
        bf = (lab[i] > 0) & (h < -0.15)
        sel = {'phantom_off': (mis & (lab[i] == 1)) | bf, 'real_off': mis & (lab[i] == 2)}
        sel['phantom_left_on'] = sel['phantom_off'] & on[i]
        sel['removed_on_phantom'] = sel['phantom_off'] & ~on[i]
        sel['removed_on_real'] = sel['real_off'] & ~on[i]
        any_ = sel['phantom_off'] | sel['real_off']
        nm_net = 0
        if any_.any():
            score, _per = nm.periodicity(Lw[i])
            if score.shape != (H, W):
                score = cv2.resize(score.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            per_m = score >= nm.score_min
            glare = cv2.dilate(Lw[i], np.ones((5, 5), np.uint8)) >= 245
            b = cv2.GaussianBlur(Lw[i].astype(np.float32), (0, 0), 1.0)
            sd = np.sqrt(np.maximum(cv2.blur(b * b, (7, 7)) - cv2.blur(b, (7, 7)) ** 2, 0))
            tl = sd < 3.0
            zz = np.where(z > 0, z, np.nan)
            mx = cv2.dilate(np.nan_to_num(zz, nan=0), np.ones((5, 5), np.uint8))
            mn = cv2.erode(np.nan_to_num(zz, nan=1e4), np.ones((5, 5), np.uint8))
            edge = (mx > 1.3 * np.where(mn < 1e4, mn, mx)) & (z > 0)
            feats = dict(periodic=per_m, glare=glare, textureless=tl, blur=np.full((H, W), bool(blur_f[i])), edge=edge,
                         below_floor=(h < -0.15) & (lab[i] > 0))
            for g, m in sel.items():
                k = int(m.sum()); tot[g] += k
                if not k: continue
                hit = np.zeros(k, bool)
                for c, f in feats.items():
                    x = f[m]; cnt[g][c] += int(x.sum()); hit |= x
                cnt[g]['none_of_these'] += int((~hit).sum())
            nm_net = int((sel['phantom_off'] & per_m).sum())
        per_frame.append(dict(i=i, t=float(t[i]), ph=int(sel['phantom_off'].sum()), ph_on=int(sel['phantom_left_on'].sum()),
                              real=int(sel['real_off'].sum()), per_ph=nm_net, rate=round(float(w[i]), 1), blur=bool(blur_f[i])))
    res = dict(tag=a.tag, n=n, on=a.on, totals=tot,
               fractions={g: {c: round(cnt[g][c] / max(tot[g], 1), 3) for c in cats} for g in cnt},
               blur_frames=int(blur_f.sum()), rot_rate_p90=round(float(np.percentile(w, 90)), 1))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    # example frames: most phantom returns (>= 5 s apart), most periodic-texture phantoms
    pf = sorted(per_frame, key=lambda r: -r['ph']); chosen = []
    for r in pf:
        if all(abs(r['t'] - per_frame[c]['t']) > 5 for c in chosen): chosen.append(r['i'])
        if len(chosen) >= 3: break
    pp = sorted(per_frame, key=lambda r: -r['per_ph'])
    if pp and pp[0]['per_ph'] > 0 and pp[0]['i'] not in chosen: chosen.append(pp[0]['i'])
    if a.frames: chosen = [int(x) for x in a.frames.split(',')]
    res['examples'] = []
    for i in chosen:
        z = D[i].astype(np.float32)
        g = cv2.cvtColor(Lw[i], cv2.COLOR_GRAY2BGR)
        A = (g * 0.6).astype(np.uint8)
        for L_, col in ((1, (0, 0, 255)), (2, (0, 220, 0)), (3, (0, 220, 255))):
            A[lab[i] == L_] = col
        Bz = cv2.addWeighted(g, 0.35, colorize(z), 0.65, 0)
        zon = np.where(on[i], z, 0)
        Cz = cv2.addWeighted(g, 0.35, colorize(zon), 0.65, 0)
        Dm = (g * 0.6).astype(np.uint8); rem = (z > 0) & ~on[i]
        Dm[rem & (lab[i] == 1)] = (0, 0, 255); Dm[rem & (lab[i] == 2)] = (0, 220, 0); Dm[rem & ~np.isin(lab[i], (1, 2))] = (255, 0, 255)
        up = lambda x: cv2.resize(x, (2 * W, 2 * H), interpolation=cv2.INTER_NEAREST)
        P = [up(x) for x in (A, Bz, Cz, Dm)]
        for p_, txt in zip(P, (f'{a.tag} #{i} t={t[i]:.1f}s  near<2m: red=free-space (seen through) green=confirmed yellow=unknown',
                               'depth, filters OFF (turbo: near=red, 5 m=blue)', f'depth, {a.on} ON',
                               'removed by filters: red=free-space green=confirmed magenta=unlabelled/far')):
            cv2.rectangle(p_, (0, 0), (2 * W, 22), (0, 0, 0), -1)
            cv2.putText(p_, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        img = np.vstack([np.hstack(P[:2]), np.hstack(P[2:])])
        fn = Path(a.out) / f'{a.tag}_f{i:03d}.png'; cv2.imwrite(str(fn), img)
        res['examples'].append(dict(file=fn.name, **per_frame[i]))
    Path(a.out, f'{a.tag}_artifacts.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ('tag', 'totals', 'fractions', 'blur_frames', 'rot_rate_p90', 'examples')}, indent=1))


if __name__ == '__main__':
    main()
