"""Short local references derived from existing stereo-supported floor segments.

No planner, map fill, depth generation or pose correction. The caller owns
session/freshness admission, conditional corridor semantics and collision/stop
certification. A reference generally extends a measured line's supporting axis;
it is explicitly marked as derived, not a directly measured endpoint.
"""
import math
import numpy as np

MEASURED_SOURCE = 'grayscale line with measured stereo floor support'


def camera_horizontal_half_fov_rad(K, width_px):
    """Axis window used for the forward line filter: the actual camera's
    horizontal half field of view, min(cx, width-cx)/fx, from the calibration in
    use. Same expression as the frozen actual-frame tests (trial45 replay), so
    their initial-transverse-edge rejection applies to the flight caller too.
    """
    fx, _, cx, _ = (float(v) for v in K)
    width = float(width_px)
    if not all(math.isfinite(v) for v in (fx, cx, width)) or fx <= 0 or not 0 < cx < width:
        raise ValueError('Finite positive fx and principal point inside the image required')
    return math.atan(min(cx, width - cx) / fx)


def short_forward_reference(hint, body_position, forward_xy, *, now,
                            lookahead_m, max_step_m, max_axis_turn_rad,
                            max_age_s=.4):
    """Return a small same-height reference or None if no forward measured axis.

    max_age_s=.4 is the current guideline consumer's existing age limit.
    All motion/heading scales are caller-supplied routine planning settings,
    not physical corridor width or measurement-error bounds.
    """
    body = np.asarray(body_position, dtype=float)
    forward = np.asarray(forward_xy, dtype=float)
    limits = [now, lookahead_m, max_step_m, max_axis_turn_rad, max_age_s]
    if body.shape != (3,) or forward.shape != (2,) or not np.isfinite(body).all() or not np.isfinite(forward).all():
        raise ValueError('Finite BODY position and horizontal preferred direction required')
    if not np.isfinite(limits).all() or min(lookahead_m, max_step_m, max_age_s) <= 0 or not 0 < max_axis_turn_rad < math.pi/2:
        raise ValueError('Invalid short-reference settings')
    norm = np.linalg.norm(forward)
    if norm <= 1e-9:
        raise ValueError('Preferred direction must be nonzero')
    forward = forward / norm
    if not hint or not math.isfinite(float(hint.get('stamp', math.nan))):
        return None
    stamp = float(hint['stamp'])
    age = now - stamp
    if not stamp <= now < stamp + max_age_s:
        return None
    observations = []
    for index, segment in enumerate(hint.get('segments', [])):
        # Keep image-only / floor+ToF structural geometry out of this measured API.
        if segment.get('source') != MEASURED_SOURCE:
            continue
        endpoints = np.asarray(segment.get('endpoints'), dtype=float)
        support = float(segment.get('support_points', 0))
        if endpoints.shape != (2, 3) or not np.isfinite(endpoints).all() or not math.isfinite(support) or support < 8:
            continue
        vector = endpoints[1, :2] - endpoints[0, :2]
        length = float(np.linalg.norm(vector))
        if length <= 1e-9:
            continue
        axis = vector / length
        if axis @ forward < 0:
            axis = -axis
        alignment = float(np.clip(axis @ forward, -1., 1.))
        if alignment < math.cos(max_axis_turn_rad):
            continue
        observations.append((index, endpoints, axis, length, support))
    if not observations:
        return None
    # A single continuous local guideline: align its observed edge axes, then use
    # the median normal intercept (two edges give their middle). This is a line
    # reference estimate, not a calibrated measurement of painted center/width.
    weights = np.array([item[3]*item[4] for item in observations])
    axis = sum(item[2]*weight for item, weight in zip(observations, weights))
    axis /= np.linalg.norm(axis)
    normal = np.array([-axis[1], axis[0]])
    intercepts = np.array([np.mean(item[1][:, :2], axis=0) @ normal for item in observations])
    offset = float(np.median(intercepts))
    lateral_error = offset - float(body[:2] @ normal)
    anchor = body[:2] + normal*lateral_error
    aim = anchor + axis*lookahead_m
    displacement = aim - body[:2]
    distance = float(np.linalg.norm(displacement))
    reference = body.copy()
    reference[:2] += displacement*min(1., max_step_m/max(distance, 1e-12))
    intervals = [(float(min(item[1][:, :2] @ axis)), float(max(item[1][:, :2] @ axis))) for item in observations]
    reference_coordinate = float(reference[:2] @ axis)
    extension = min(max(low-reference_coordinate, reference_coordinate-high, 0.) for low, high in intervals)
    nearest_observed_distance = min(float(np.linalg.norm(body[:2] - (item[1][0,:2] +
        np.clip(np.dot(body[:2]-item[1][0,:2], item[1][1,:2]-item[1][0,:2]) /
                max(item[3]**2, 1e-12), 0., 1.)*(item[1][1,:2]-item[1][0,:2]))))
        for item in observations)
    step = reference[:2]-body[:2]
    lateral_step = float(step @ normal)
    return {
        'stamp': float(hint['stamp']),
        'age_s': age,
        'reference': reference.tolist(),
        'axis_xy_unit': axis.tolist(),
        'heading_rad': float(math.atan2(axis[1], axis[0])),
        'reference_motion_bearing_rad': float(math.atan2(step[1], step[0])),
        'reference_kind': 'supporting_axis_extension_with_lateral_approach',
        'normal_xy_unit': normal.tolist(),
        'lateral_reference_error_m': lateral_error,
        'source_segment_indices': [item[0] for item in observations],
        'observed_axis_intervals_m': [list(interval) for interval in intervals],
        'supporting_axis_extension_m': float(extension),
        'body_to_nearest_measured_segment_xy_m': nearest_observed_distance,
        'motion_components': {
            'parallel_to_observed_axis_m': float(step @ axis),
            'toward_supporting_axis_m': lateral_step,
            'has_lateral_approach': abs(lateral_step)>1e-9,
            'lateral_offset_is_from_supporting_axis_extension': True,
        },
        'body_to_reference_is_a_measured_line_segment': False,
        'approach_and_axis_extension_require_map_validation': True,
        'source': 'short reference derived from measured stereo-supported floor-line axes',
        'reference_is_direct_stereo_measurement': False,
        'uses_single_continuous_guideline_prior': True,
        'painted_center_or_width_is_directly_measured': False,
        'collision_or_free_space_authorization': False,
        'altitude_source': 'unchanged current BODY altitude',
    }
