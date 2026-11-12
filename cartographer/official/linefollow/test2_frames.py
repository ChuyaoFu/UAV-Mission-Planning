#!/usr/bin/env python3
"""Offline check of the existing stereo-supported GuidelineDetector on official-lab frames.
Usage: test2_frames.py FRAMES_DIR IDX [IDX ...]  (writes det_IDX.png)"""
import json, math, sys
import numpy as np, cv2
from scipy.spatial.transform import Rotation
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from line_ipm import quat_wxyz_to_mat, F_FLU_FRD
from relative_guideline import GuidelineDetector

ENU_FROM_NED = np.array([[0., 1, 0], [1, 0, 0], [0, 0, -1]])
d = sys.argv[1]; z = np.load(d + '/frames.npz', allow_pickle=True)
meta = json.loads(str(z['meta'])); K = z['K']; tfs = json.loads(str(z['tfs']))
t_bc, q_bc = tfs['base_link<-oak_left']; R_bc = quat_wxyz_to_mat(*q_bc)
home_d = meta[0]['ned'][2]; d_floor = home_d + 0.04
det = GuidelineDetector()
for idx in map(int, sys.argv[2:]):
    m = meta[idx]; im = z['images'][idx]; dep = z['depths'][idx]
    R_nb = quat_wxyz_to_mat(*m['q']); R_nc = R_nb @ F_FLU_FRD @ R_bc
    cam_ned = np.array(m['ned']) + R_nb @ (F_FLU_FRD @ np.array(t_bc))
    R_wc = ENU_FROM_NED @ R_nc; cam_w = ENU_FROM_NED @ cam_ned; floor_up = -d_floor
    segs = det.detect(im, dep, K, cam_w, Rotation.from_matrix(R_wc), floor_up)
    vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR); out = []
    body_w = ENU_FROM_NED @ np.array(m['ned']); hd = m['heading']
    for s in segs:
        a, b = np.array(s['endpoints'])
        for p0, p1 in [(a, b)]:
            pc = [R_wc.T @ (p - cam_w) for p in (p0, p1)]
            if min(p[2] for p in pc) > .05:
                uv = [(int(K[0]*p[0]/p[2]+K[2]), int(K[1]*p[1]/p[2]+K[3])) for p in pc]
                cv2.line(vis, uv[0], uv[1], (0, 0, 255), 2)
        # express in the body level frame: x forward, y right (NED heading)
        def loc(p):
            n, e = p[1]-body_w[1], p[0]-body_w[0]
            return round(n*math.cos(hd)+e*math.sin(hd), 2), round(-n*math.sin(hd)+e*math.cos(hd), 2)
        ang = math.degrees(math.atan2(*(np.array(loc(b))-np.array(loc(a)))[::-1]))
        out.append((loc(a), loc(b), round(ang, 1), s['support_points']))
    cv2.imwrite(f'{d}/det_{idx:03d}.png', vis)
    print(idx, 'alt', round(d_floor-m['ned'][2], 2), 'segs', out)
