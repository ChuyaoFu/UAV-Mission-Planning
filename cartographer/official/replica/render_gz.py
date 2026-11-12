#!/usr/bin/env python3
"""Render the pose list in a running headless Gazebo (inside the one-shot container).

Usage: python3 render_gz.py --pass full|depth --poses world/poses.json --out OUT [--limit N]
For every pose: set_pose on model `cam` -> wait until /world/replica/pose/info reports
the new pose (sim stamp s) -> take the first frame(s) with stamp >= s + 1 sensor period
(RGB and depth must share the same stamp) -> OUT/raw_<pass>/<id>.npz.
pass full : rgb + rgb_right (640x400, area-averaged from the 2x supersampled 1280x800, 7.5 cm baseline) + depth_full
pass depth: depth (scene without cage net / tunnel mesh)
"""
import argparse, json, os, threading, time
import numpy as np
import cv2
from gz.transport13 import Node
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.image_pb2 import Image
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.pose_v_pb2 import Pose_V

WORLD = 'replica'
PERIOD = 0.05


def stamp(m):
    return m.header.stamp.sec + m.header.stamp.nsec * 1e-9


class Grabber:
    def __init__(self, topics):
        self.node = Node()
        self.lock = threading.Lock()
        self.hist = {t: [] for t in topics}
        self.cam = None
        for t in topics:
            self.node.subscribe(Image, t, self._img_cb(t))
        self.node.subscribe(Pose_V, f'/world/{WORLD}/pose/info', self._pose_cb)

    def _img_cb(self, t):
        def cb(msg):
            with self.lock:
                h = self.hist[t]; h.append((round(stamp(msg), 4), msg)); del h[:-8]
        return cb

    def _pose_cb(self, msg):
        for p in msg.pose:
            if p.name == 'cam':
                q = p.orientation
                with self.lock:
                    self.cam = (stamp(msg), np.array([p.position.x, p.position.y, p.position.z]), np.array([q.w, q.x, q.y, q.z]))

    def set_pose(self, xyz, q):
        """Send set_pose. The Python binding sometimes never sees the reply (esp. under load)
        although the command is applied, so the return value is advisory: the caller
        confirms the move through pose/info."""
        req = Pose(); req.name = 'cam'
        req.position.x, req.position.y, req.position.z = xyz
        req.orientation.w, req.orientation.x, req.orientation.y, req.orientation.z = q
        ok, rep = self.node.request(f'/world/{WORLD}/set_pose', req, Pose, Boolean, 1000)
        return bool(ok and rep.data)

    def wait_pose(self, xyz, q, timeout=6.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                c = self.cam
            if c is not None and np.abs(c[1] - xyz).max() < 2e-4 and abs(abs(np.dot(c[2], q)) - 1) < 1e-6:
                return c[0]
            time.sleep(0.005)
        return None

    def wait_frames(self, after, timeout=6.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                sets = [{s: m for s, m in h if s >= after - 1e-6} for h in self.hist.values()]
            common = set.intersection(*[set(d) for d in sets]) if sets else set()
            if common:
                s = min(common)
                return s, [d[s] for d in sets]
            time.sleep(0.005)
        return None, None


def to_rgb(m):
    a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width * 3].reshape(m.height, m.width, 3)
    return a


def to_depth(m):
    return np.frombuffer(m.data, np.float32).reshape(m.height, m.width).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pass', dest='pss', choices=('full', 'depth'), required=True)
    ap.add_argument('--poses', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--ids', default='')
    a = ap.parse_args()
    poses = json.load(open(a.poses))['poses']
    if a.ids:
        keep = set(a.ids.split(','))
        poses = [p for p in poses if p['id'] in keep]
    if a.limit:
        poses = poses[:a.limit]
    out = os.path.join(a.out, f'raw_{a.pss}')
    done = {f[:-4] for f in os.listdir(out)} if os.path.isdir(out) else set()
    poses = [p for p in poses if p['id'] not in done]          # resumable: only render what is missing
    topics = ['/replica/rgb', '/replica/rgb_right', '/replica/depth'] if a.pss == 'full' else ['/replica/depth']
    os.makedirs(out, exist_ok=True)
    g = Grabber(topics)
    t0 = time.time()
    while time.time() - t0 < 120:                         # wait for the server + first frames
        with g.lock:
            ready = all(len(h) for h in g.hist.values()) and g.cam is not None
        if ready:
            break
        time.sleep(0.2)
    else:
        raise SystemExit('no frames from gz server')
    print(f'[{a.pss}] server ready after {time.time() - t0:.1f}s; {len(poses)} poses to render ({len(done)} already on disk)', flush=True)
    log = []
    t_start = time.time()
    consecutive_fail = 0
    for k, p in enumerate(poses):
        xyz, q = np.array(p['gz_xyz']), np.array(p['gz_quat_wxyz'])
        stage = ''
        for attempt in range(6):
            replied = g.set_pose(xyz, q)
            s_pose = g.wait_pose(xyz, q, timeout=3.0)
            if s_pose is None:
                stage = f'wait_pose (set_pose replied={replied})'; continue
            s_img, msgs = g.wait_frames(s_pose + PERIOD)
            if s_img is not None:
                break
            stage = f'wait_frames(after {s_pose + PERIOD:.2f})'
        else:
            with g.lock:
                last = {t: (h[-1][0] if h else None) for t, h in g.hist.items()}
                cam_s = g.cam[0] if g.cam else None
            print('FAILED', p['id'], 'stage', stage, 'last image stamps', last, 'last pose/info stamp', cam_s, flush=True)
            log.append({'id': p['id'], 'ok': False, 'stage': stage})
            consecutive_fail += 1
            if consecutive_fail >= 5:                        # server stalled: stop instead of looping
                json.dump(log, open(os.path.join(a.out, f'render_log_{a.pss}.json'), 'w'), indent=0)
                raise SystemExit(f'[{a.pss}] aborting after 5 consecutive failures')
            continue
        consecutive_fail = 0
        rec = {'id': p['id'], 'ok': True, 'pose_stamp': s_pose, 'img_stamp': s_img}
        if a.pss == 'full':
            down = lambda m: cv2.resize(to_rgb(m), (m.width // 2, m.height // 2), interpolation=cv2.INTER_AREA)
            np.savez_compressed(os.path.join(out, p['id'] + '.npz'), rgb=down(msgs[0]), rgb_right=down(msgs[1]),
                                depth_full=to_depth(msgs[2]), stamp=s_img)
        else:
            np.savez_compressed(os.path.join(out, p['id'] + '.npz'), depth=to_depth(msgs[0]), stamp=s_img)
        log.append(rec)
        if k % 20 == 0:
            print(f'[{a.pss}] {k + 1}/{len(poses)} {p["id"]} stamp {s_img:.2f} ({time.time() - t_start:.0f}s)', flush=True)
    json.dump(log, open(os.path.join(a.out, f'render_log_{a.pss}.json'), 'w'), indent=0)
    print(f'[{a.pss}] done {sum(r["ok"] for r in log)}/{len(poses)} in {time.time() - t_start:.0f}s', flush=True)
    os._exit(0)          # skip the gz-transport Python teardown (it can segfault at interpreter exit)


if __name__ == '__main__':
    main()
