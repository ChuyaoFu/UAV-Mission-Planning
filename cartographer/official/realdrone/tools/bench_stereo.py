#!/usr/bin/env python3
"""Offline: rectify + SGBM a captured pair (capture_pair.py .npz); timing per mode/threads + rectification residual.
Usage: python3 bench_stereo.py pair.npz [out_prefix]"""
import ast, sys, time
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stereo_depth_node import RectifyModel, Sgbm, quat_xyzw_to_mat   # noqa: E402

z = np.load(sys.argv[1]); out = sys.argv[2] if len(sys.argv) > 2 else None
info = ast.literal_eval(str(z['info'])); tf = ast.literal_eval(str(z['tf']))
t = tf['oak_left_camera_optical_frame->oak_right_camera_optical_frame']
il, ir = info['left'], info['right']
m = RectifyModel(np.array(il['K']).reshape(3, 3), np.zeros(5), np.array(ir['K']).reshape(3, 3), np.zeros(5),
                 (il['w'], il['h']), quat_xyzw_to_mat(*t[3:]), np.array(t[:3]), width=int(640))
print('model', m.describe())
L, R = z['left'][0], z['right'][0]
Lr, Rr = m.apply(L, R)
# sub-pixel vertical residual: LK tracking of corners left -> right
pts = cv2.goodFeaturesToTrack(Lr, 800, 0.01, 6)
if pts is not None:
    guess = pts.copy(); guess[:, 0, 0] -= 40
    p2, st, err = cv2.calcOpticalFlowPyrLK(Lr, Rr, pts, guess, winSize=(21, 21), maxLevel=4,
                                           flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    ok = st[:, 0] == 1
    dy = (pts[ok, 0, 1] - p2[ok, 0, 1]); dx = pts[ok, 0, 0] - p2[ok, 0, 0]
    good = (dx > 0) & (np.abs(dy) < 4)
    print('LK rect residual: n=%d dy median %.3f px, |dy| p90 %.3f px, disparity median %.1f px -> %.3f m' % (
        good.sum(), np.median(dy[good]), np.percentile(np.abs(dy[good]), 90), np.median(dx[good]), m.fx * m.baseline / np.median(dx[good])))
for mode in ('SGBM', 'SGBM_3WAY', 'HH4'):
    for th in (1, 2, 4):
        cv2.setNumThreads(th); s = Sgbm(mode=mode)
        s.depth(Lr, Rr, m.fx, m.baseline)
        t0 = time.perf_counter(); c0 = time.process_time(); n = 5
        for _ in range(n): d = s.depth(Lr, Rr, m.fx, m.baseline)
        wall = (time.perf_counter() - t0) / n; cpu = (time.process_time() - c0) / n
        print('%-10s threads %d: %.1f ms wall, %.1f ms cpu, valid %.3f, median %.3f m' % (mode, th, 1e3 * wall, 1e3 * cpu, np.mean(d > 0), np.median(d[d > 0]) if np.any(d > 0) else -1))
t0 = time.perf_counter()
for _ in range(10): m.apply(L, R)
print('rectify+resize %.1f ms' % (1e3 * (time.perf_counter() - t0) / 10))
if out:
    cv2.setNumThreads(2); d = Sgbm(mode='SGBM_3WAY').depth(Lr, Rr, m.fx, m.baseline)
    vis = cv2.applyColorMap(np.clip(255 * (1 - d / 4.0), 0, 255).astype(np.uint8), cv2.COLORMAP_JET); vis[d <= 0] = 0
    cv2.imwrite(out + '_depth.png', np.hstack([cv2.cvtColor(Lr, cv2.COLOR_GRAY2BGR), vis]))
    cv2.imwrite(out + '_rect.png', np.hstack([Lr, Rr]))
