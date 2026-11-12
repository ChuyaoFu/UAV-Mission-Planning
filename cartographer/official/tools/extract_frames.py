#!/usr/bin/env python3
"""Extract left-camera frames + matched PX4 state from a recorded official-lab bag.

Usage: python3 extract_frames.py BAG OUT_DIR [--every 0.5]
Writes OUT_DIR/frames.npz (images, stamps, NED pos, q_wxyz FRD->NED, range,
K, R/t base_link<-camera) and a few PNGs for viewing. Offline only.
"""
import argparse, json, math, os
import numpy as np
import cv2
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

p = argparse.ArgumentParser(); p.add_argument('bag'); p.add_argument('out'); p.add_argument('--every', type=float, default=.5)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=a.bag, storage_id=''), rosbag2_py.ConverterOptions('', ''))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
msg = lambda topic, data: deserialize_message(data, get_message(types[topic]))
imgs, pos, att, rng, tfs, depths = [], [], [], [], {}, []
K = None; frame_id = None; last_img = -1e9
while r.has_next():
    topic, data, t = r.read_next(); t *= 1e-9
    if topic == '/stereo/left/camera_info' and K is None:
        m = msg(topic, data); K = [m.k[0], m.k[4], m.k[2], m.k[5]]; frame_id = m.header.frame_id; size = (m.width, m.height)
    elif topic == '/stereo/left/image' and t - last_img >= a.every:
        m = msg(topic, data); last_img = t
        im = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.step)[:, :m.width * (1 if m.encoding in ('mono8', '8UC1') else 3)]
        if m.encoding not in ('mono8', '8UC1'):
            im = cv2.cvtColor(im.reshape(m.height, m.width, 3), cv2.COLOR_RGB2GRAY)
        item = (t, m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, im.copy(), m.header.frame_id)
        near_d = [d_ for td, d_ in depths if abs(td-t) < .1]
        imgs.append(item + ((near_d[-1].copy(),) if near_d else ()))
    elif topic == '/camera/depth/camera_info' and 'dK' not in globals():
        m = msg(topic, data); dK = [m.k[0], m.k[4], m.k[2], m.k[5]]; dframe = m.header.frame_id
    elif topic == '/camera/depth/image_raw':
        m = msg(topic, data); dt_ = np.float32 if m.encoding == '32FC1' else np.uint16
        dep = np.frombuffer(bytes(m.data), dt_).reshape(m.height, -1)[:, :m.width].astype(np.float32)
        if m.encoding != '32FC1': dep = dep/1000.
        depths.append((t, dep))
        if len(depths) > 6: depths.pop(0)
        for k, item in enumerate(imgs):
            if len(item) == 4 and abs(item[0]-t) < .1: imgs[k] = item + (dep.copy(),)
    elif topic == '/fmu/out/vehicle_local_position_v1':
        m = msg(topic, data); pos.append((t, m.x, m.y, m.z, m.heading))
    elif topic == '/fmu/out/vehicle_attitude':
        m = msg(topic, data); att.append((t, *[float(c) for c in m.q]))
    elif topic == '/fmu/out/distance_sensor':
        m = msg(topic, data); rng.append((t, m.current_distance))
    elif topic in ('/tf_static', '/tf'):
        for tr in msg(topic, data).transforms:
            tfs[(tr.header.frame_id, tr.child_frame_id)] = ([tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z],
                                                            [tr.transform.rotation.w, tr.transform.rotation.x, tr.transform.rotation.y, tr.transform.rotation.z])
pos, att, rng = np.array(pos), np.array(att), np.array(rng) if rng else np.zeros((0, 2))
near = lambda arr, t: arr[np.argmin(np.abs(arr[:, 0] - t))] if len(arr) else None
rows = []
for i, item in enumerate(imgs):
    t, stamp, im, fid = item[:4]
    P = near(pos, t); Q = near(att, t); R = near(rng, t)
    rows.append(dict(t=t, stamp=stamp, ned=P[1:4].tolist(), heading=float(P[4]), q=Q[1:5].tolist(), range=None if R is None else float(R[1])))
    cv2.imwrite(f'{a.out}/f{i:03d}_z{-P[3]:.2f}.png', im)
dshape = next((x[4].shape for x in imgs if len(x) > 4), (1, 1))
np.savez_compressed(f'{a.out}/frames.npz', images=np.stack([x[2] for x in imgs]), meta=json.dumps(rows), K=np.array(K), frame_id=frame_id,
                    depths=np.stack([x[4] if len(x) > 4 else np.full(dshape, np.nan, np.float32) for x in imgs]),
                    dK=np.array(globals().get('dK', [np.nan]*4)), dframe=globals().get('dframe', ''),
                    tfs=json.dumps({f'{k[0]}<-{k[1]}': v for k, v in tfs.items()}))
print('frames', len(imgs), 'K', K, 'frame', frame_id, 'size', size, 'depth', globals().get('dframe'), globals().get('dK'), dshape, 'with_depth', sum(len(x) > 4 for x in imgs))
print('tf keys', [f'{k[0]}<-{k[1]}' for k in tfs][:20])
for i, row in enumerate(rows): print(i, round(row['t'], 2), 'ned', [round(x, 2) for x in row['ned']], 'range', row['range'])
