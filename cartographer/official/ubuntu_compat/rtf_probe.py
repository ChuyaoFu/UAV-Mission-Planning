#!/usr/bin/env python3
"""Timing of N flights flown AT THE SAME TIME (rtf_test.sh): per flight the real-time factor and the sensor rates in
SIM time, over the wall-clock window in which ALL N flights' bags were recording (so every number is at full load).

  python3 rtf_probe.py FLIGHT_DIR [...]      (container: rosbag2_py)   -> markdown table + JSON on stderr

RTF = d(sim clock) / d(wall) from /clock vs bag receive time. Rates = messages received in the window / sim seconds
of the window (no deserialisation, only /clock is decoded): depth image, stereo left, VIO into PX4
(/visual_slam/tracking/odometry), OpenVINS camera update (/ov_msckf/poseimu, side bag), OpenVINS IMU propagation
(/ov_msckf/odomimu), down ToF scan.
"""
import json, sys
from pathlib import Path
import numpy as np
from rclpy.serialization import deserialize_message
from rosgraph_msgs.msg import Clock
import rosbag2_py

MAIN = ['/clock', '/camera/depth/image_raw', '/stereo/left/image', '/visual_slam/tracking/odometry']
SIDE = ['/ov_msckf/poseimu', '/ov_msckf/odomimu', '/gz/down_tof/scan']


def stamps(bag, topics):
    out = {t: [] for t in topics}; clk = []
    if not Path(bag).exists():
        return out, clk
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    have = {t.name for t in r.get_all_topics_and_types()}
    r.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in have]))
    while r.has_next():
        t, data, rt = r.read_next()
        if t == '/clock':
            m = deserialize_message(data, Clock); clk.append((rt * 1e-9, m.clock.sec + m.clock.nanosec * 1e-9))
        else:
            out[t].append(rt * 1e-9)
    return out, clk


def main():
    fl = []
    for f in sys.argv[1:]:
        F = Path(f); a, clk = stamps(F / 'bag', MAIN); s, _ = stamps(F / 'sim' / 'sidebag', SIDE)
        if len(clk) < 10:
            print(f'# {F.name}: no /clock'); continue
        fl.append((F, {**a, **s}, np.array(clk)))
    w0 = max(c[0, 0] for _, _, c in fl); w1 = min(c[-1, 0] for _, _, c in fl)
    print(f'common window {w1 - w0:.1f} s wall, {len(fl)} flights at once')
    print('| flight | RTF | depth Hz | stereo L Hz | VIO→PX4 Hz | OpenVINS cam Hz | OpenVINS IMU Hz | ToF Hz |')
    print('|---|---|---|---|---|---|---|---|')
    res = []
    for F, tp, clk in fl:
        s0, s1 = np.interp([w0, w1], clk[:, 0], clk[:, 1]); ds = s1 - s0; rtf = ds / (w1 - w0)
        hz = {t: (sum(1 for x in v if w0 <= x <= w1) / ds if ds > 0 else 0.0) for t, v in tp.items()}
        res.append(dict(flight=F.name, window_wall_s=round(w1 - w0, 1), rtf=round(rtf, 3), **{k: round(v, 1) for k, v in hz.items()}))
        print(f"| {F.name} | {rtf:.3f} | {hz['/camera/depth/image_raw']:.1f} | {hz['/stereo/left/image']:.1f} | {hz['/visual_slam/tracking/odometry']:.1f} | "
              f"{hz['/ov_msckf/poseimu']:.1f} | {hz['/ov_msckf/odomimu']:.0f} | {hz['/gz/down_tof/scan']:.1f} |")
    print(json.dumps(res), file=sys.stderr)


if __name__ == '__main__':
    main()
