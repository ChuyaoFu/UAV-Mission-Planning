#!/usr/bin/env python3
"""Offline actual-frame tests of the measured-line short reference (trial45).

Adapted from the frozen line-reference-v1 candidate tests, but run with the
settings the flight caller (competition_navigate.explore) actually uses, read
from its source. Starts no ROS or flight. Packets are the 8 strictly matched
trial45 frames (image, passive depth, K, camera/BODY pose, ToF floor_z).
Set LINE_REPLAY_DIR to override the default packet directory; the tests skip
when the packets are absent (outputs/ is not tracked by Git).
"""
import copy
import math
import os
import re
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from measured_line_reference import short_forward_reference, camera_horizontal_half_fov_rad
from relative_guideline import GuidelineDetector

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PACKETS = Path(os.environ.get('LINE_REPLAY_DIR',
                              REPO/'outputs/cadfree-navigation/course-structural20-full-45/line-replay'))
CALLER = (HERE/'competition_navigate.py').read_text()


def flight_settings():
    call = re.search(r'short_forward_reference\(self\.guideline,self\.body,forward,now=now,\s*'
                     r'lookahead_m=([0-9.]+),max_step_m=([0-9.]+),max_axis_turn_rad=self\.line_axis_turn_bound\)', CALLER)
    bound = re.search(r'self\.line_axis_turn_bound=camera_horizontal_half_fov_rad\(self\.K,w\)', CALLER)
    if call is None or bound is None:
        raise AssertionError('competition_navigate no longer uses the half-FOV line axis window; update these tests')
    return float(call.group(1)), float(call.group(2))


LOOKAHEAD_M, MAX_STEP_M = flight_settings()
frames = []
if PACKETS.is_dir():
    detector = GuidelineDetector()
    for path in sorted(PACKETS.glob('*.npz')):
        a = np.load(path)
        rotation = Rotation.from_quat(a['camera_quaternion_xyzw'])
        segments = detector.detect(a['image'], a['depth'], a['K'], a['camera_position'], rotation, float(a['floor_z']))
        stamp = int(a['stamp_ns'])*1e-9
        kw = dict(now=stamp, lookahead_m=LOOKAHEAD_M, max_step_m=MAX_STEP_M,
                  max_axis_turn_rad=camera_horizontal_half_fov_rad(a['K'], a['image'].shape[1]))
        hint = {'stamp': stamp, 'segments': segments}
        body = a['body_position'].copy()
        forward = rotation.apply([0, 0, 1])[:2]  # Same optical-axis preference as the flight caller.
        frames.append((hint, body, forward, kw, short_forward_reference(hint, body, forward, **kw)))


@unittest.skipUnless(len(frames) == 8, f'trial45 line-replay packets not found at {PACKETS}')
class ActualFrameFlightSettingsTests(unittest.TestCase):
    def test_settings_come_from_flight_caller(self):
        self.assertEqual((LOOKAHEAD_M, MAX_STEP_M), (.30, .20))
        for *_, kw, _ in frames:
            self.assertAlmostEqual(math.degrees(kw['max_axis_turn_rad']), 40.1, places=1)

    def test_actual_counts_preserved(self):
        self.assertEqual([len(f[0]['segments']) for f in frames], [2, 3, 4, 5, 4, 4, 5, 5])

    def test_transverse_first_frame_does_not_become_forward_route(self):
        self.assertIsNone(frames[0][4])
        self.assertTrue(all(f[4] is not None for f in frames[1:]))

    def test_former_85_degree_window_admitted_transverse_edges(self):
        # Why the window changed: 85 deg accepted the 62.0 s transverse edge and
        # blended transverse edges into later axes (about -30..-47 deg off).
        wide = [short_forward_reference(hint, body, forward, **{**kw, 'max_axis_turn_rad': math.radians(85)})
                for hint, body, forward, kw, _ in frames]
        self.assertIsNotNone(wide[0])  # 62.0 s: transverse light edge taken as the route.
        self.assertIsNone(frames[0][4])
        for index in range(2, 8):  # 63.0-65.5 s: extra transverse segments mixed into the axis.
            flight = frames[index][4]
            self.assertLess(len(flight['source_segment_indices']), len(wide[index]['source_segment_indices']), index)
            cosine = float(np.array(flight['axis_xy_unit']) @ np.array(wide[index]['axis_xy_unit']))
            self.assertLess(cosine, math.cos(math.radians(20)), index)

    def test_same_altitude_bounded_derived_references(self):
        for hint, body, forward, kw, r in frames:
            if r is None:
                continue
            target = np.array(r['reference'])
            self.assertAlmostEqual(target[2], body[2])
            self.assertLessEqual(np.linalg.norm(target - body), MAX_STEP_M + 1e-12)
            self.assertFalse(r['collision_or_free_space_authorization'])
            self.assertFalse(r['reference_is_direct_stereo_measurement'])
            self.assertFalse(r['body_to_reference_is_a_measured_line_segment'])
            self.assertTrue(r['approach_and_axis_extension_require_map_validation'])
            self.assertGreater(r['body_to_nearest_measured_segment_xy_m'], 0.)
            step = target[:2] - body[:2]
            self.assertAlmostEqual(r['reference_motion_bearing_rad'], math.atan2(step[1], step[0]))
            self.assertEqual(r['reference_kind'], 'supporting_axis_extension_with_lateral_approach')

    def test_axis_consistency_across_actual_frames(self):
        axes = [np.array(f[4]['axis_xy_unit']) for f in frames if f[4] is not None]
        for a, b in zip(axes, axes[1:]):
            self.assertGreater(float(a @ b), math.cos(math.radians(3)))

    def test_endpoint_order_does_not_flip_reference(self):
        for hint, body, forward, kw, _ in frames:
            other = copy.deepcopy(hint)
            for segment in other['segments']:
                segment['endpoints'].reverse()
            a = short_forward_reference(hint, body, forward, **kw)
            b = short_forward_reference(other, body, forward, **kw)
            if a is None:
                self.assertIsNone(b)
            else:
                np.testing.assert_allclose(a['reference'], b['reference'], atol=1e-12)
                np.testing.assert_allclose(a['axis_xy_unit'], b['axis_xy_unit'], atol=1e-12)

    def test_stale_and_future_observations_not_used(self):
        hint, body, forward, kw, _ = frames[1]
        for delta in [.4, .5, -.001]:
            self.assertIsNone(short_forward_reference(hint, body, forward, **{**kw, 'now': hint['stamp'] + delta}))

    def test_structural_or_look_only_not_admitted_as_measured(self):
        hint, body, forward, kw, _ = frames[1]
        for source in ['image_only_look', 'floor+ToF structural prior geometry']:
            other = copy.deepcopy(hint)
            for segment in other['segments']:
                segment['source'] = source
            self.assertIsNone(short_forward_reference(other, body, forward, **kw))

    def test_lateral_approach_is_explicit_and_not_cleared(self):
        hint, body, forward, kw, r = frames[3]
        moved = body.copy()
        moved[:2] += np.array(r['normal_xy_unit'])*.10
        altered = short_forward_reference(hint, moved, forward, **kw)
        self.assertTrue(altered['motion_components']['has_lateral_approach'])
        self.assertGreater(abs(altered['motion_components']['toward_supporting_axis_m']), .01)
        self.assertFalse(altered['collision_or_free_space_authorization'])
        self.assertTrue(altered['approach_and_axis_extension_require_map_validation'])


class HalfFovTests(unittest.TestCase):
    def test_matches_frozen_expression_and_rejects_bad_calibration(self):
        self.assertAlmostEqual(camera_horizontal_half_fov_rad([759.83477256, 759.83477256, 640., 400.], 1280),
                               math.atan(640/759.83477256))
        self.assertAlmostEqual(camera_horizontal_half_fov_rad([500., 500., 300., 200.], 640), math.atan(300/500))
        for K, width in [([0., 1., 320., 200.], 640), ([500., 500., 0., 200.], 640), ([500., 500., 640., 200.], 640),
                         ([math.nan, 500., 320., 200.], 640)]:
            with self.assertRaises(ValueError):
                camera_horizontal_half_fov_rad(K, width)


if __name__ == '__main__':
    unittest.main(verbosity=2)
