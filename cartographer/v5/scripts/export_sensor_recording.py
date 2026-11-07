#!/usr/bin/env python3
"""Offline, lossless CDR subset for perception/localization analysis.

No commands, exploration goals, simulator poses, contacts or CAD are exported.
Estimated odometry/native posterior telemetry is labelled as estimator output.
"""
import argparse
import json
from pathlib import Path
import rosbag2_py

TOPICS = ['/clock', '/vio/left/image_raw', '/vio/right/image_raw', '/vio/imu',
          '/stereo/depth/image', '/carto/down_range', '/nav/camera_pose',
          '/nav/body_odom', '/ov_msckf/poseimu', '/ov_msckf/home_relative_body_pose',
          '/ov_msckf/home_relative_body_reference', '/ov_msckf/joint_clone_states',
          '/nav/home_relative_adapter_health', '/ov_msckf/visual_update_health',
          '/carto/localization_health']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError('Existing exports cannot be overwritten')
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(a.source), storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    selected = [m for m in reader.get_all_topics_and_types() if m.name in TOPICS]
    reader.set_filter(rosbag2_py.StorageFilter(topics=[m.name for m in selected]))
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(a.output), storage_id='mcap',
                                         storage_preset_profile='zstd_fast'),
                rosbag2_py.ConverterOptions('', ''))
    counts = {m.name: 0 for m in selected}
    for metadata in selected: writer.create_topic(metadata)
    while reader.has_next():
        topic, payload, stamp = reader.read_next()
        if topic not in counts: raise RuntimeError('Reader returned a forbidden topic')
        writer.write(topic, payload, stamp)
        counts[topic] += 1
    del writer
    (a.output/'sensor-export.json').write_text(json.dumps({
        'source': str(a.source), 'counts': counts, 'cdr_payloads_unchanged': True,
        'storage_and_sensor_timestamps_unchanged': True,
        'odometry_and_joint_covariance_are_estimates': True,
        'truth_commands_goals_or_CAD_included': False}, indent=2)+'\n')
    print(json.dumps(counts))


if __name__ == '__main__': main()
