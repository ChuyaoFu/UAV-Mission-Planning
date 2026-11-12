#!/usr/bin/env python3
"""Offline check of line_ipm on extracted frames. Usage: test_frames.py FRAMES_DIR IDX [IDX ...]"""
import json, math, sys
import numpy as np, cv2
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from line_ipm import Detector, merge, quat_wxyz_to_mat

d = sys.argv[1]; z = np.load(d + '/frames.npz', allow_pickle=True)
meta = json.loads(str(z['meta'])); K = z['K'].tolist(); tfs = json.loads(str(z['tfs']))
t_bc, q_bc = tfs['base_link<-oak_left']; R_bc = quat_wxyz_to_mat(*q_bc)
home_d = meta[0]['ned'][2]; d_floor = home_d + 0.04
det = Detector()
for idx in map(int, sys.argv[2:]):
    m = meta[idx]; im = z['images'][idx]
    dep = z['depths'][idx] if 'depths' in z.files else None
    dK = z['dK'].tolist() if 'dK' in z.files and np.isfinite(z['dK']).all() else None
    top, valid = det.ipm(im, K, R_bc, t_bc, m['q'], m['ned'], m['heading'], d_floor, depth=None if dep is None or not np.isfinite(dep).any() else dep, dK=dK)
    segs, mask = det.detect(top, valid); lines = merge(segs)
    vis = cv2.cvtColor(top, cv2.COLOR_GRAY2BGR); vis[mask > 0] = (0, 0, 255)
    for L in lines[:4]:
        a = L['point'] + L['dir']*L['s_min']; b = L['point'] + L['dir']*L['s_max']
        pa = (int((a[1]+det.yh)/det.res), int((det.x1-a[0])/det.res)); pb = (int((b[1]+det.yh)/det.res), int((det.x1-b[0])/det.res))
        cv2.line(vis, pa, pb, (0, 255, 0), 2)
    cv2.imwrite(f'{d}/ipm_{idx:03d}.png', np.hstack([cv2.cvtColor(cv2.resize(im, (400, 250)), cv2.COLOR_GRAY2BGR), cv2.resize(vis, (250, 250))]))
    print(idx, 'alt', round(d_floor - m['ned'][2], 2), 'valid', round(float(valid.mean()), 2), 'segs', len(segs),
          'lines', [(round(math.degrees(math.atan2(L['dir'][1], L['dir'][0])), 1), round(float(L['point'] @ np.array([-L['dir'][1], L['dir'][0]])), 2),
                     round(L['length'], 2), round(L['s_min'], 2), round(L['s_max'], 2)) for L in lines[:4]])
