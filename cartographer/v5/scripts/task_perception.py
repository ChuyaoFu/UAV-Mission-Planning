"""Bounded generic opening-rim measurements from synchronized mono/stereo.

The API accepts no scene pose, dimensions, CAD, task coordinates or world map.
It fits a planar rim from DIRECT measured depth, then intersects observed image
boundaries with that plane. Interior background evidence is reported separately.
An opening candidate is never a declaration that its swept volume is free.
"""
from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class OpeningLimits:
    max_image_width: int = 640
    max_contours: int = 80
    max_candidates: int = 6
    max_rim_points: int = 400
    plane_trials: int = 48
    min_rim_points: int = 24
    min_rim_fraction: float = 0.55
    plane_tolerance_m: float = 0.02
    minimum_depth_m: float = 0.25
    maximum_depth_m: float = 8.0


def _pixels_to_points(uv, depth, K):
    uv = np.asarray(uv, dtype=float)
    rays = np.column_stack(((uv[:, 0] - K[2]) / K[0],
                            (uv[:, 1] - K[3]) / K[1], np.ones(len(uv))))
    return rays * np.asarray(depth)[:, None]


def _quadrants(uv, center):
    angle = np.arctan2(uv[:, 1] - center[1], uv[:, 0] - center[0])
    return np.floor((angle + math.pi) * (2.0 / math.pi)).astype(int) % 4


def _fit_plane(points, limits):
    """RANSAC plus SVD; rejects line-like support and mixed-depth boundaries."""
    if len(points) < limits.min_rim_points:
        return None
    rng = np.random.default_rng(0)
    best = None
    tolerance = limits.plane_tolerance_m
    for _ in range(limits.plane_trials):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-7:
            continue
        normal /= norm
        distance = -float(normal @ a)
        mask = np.abs(points @ normal + distance) < tolerance
        if best is None or mask.sum() > best.sum():
            best = mask
    if best is None or best.sum() < limits.min_rim_points or best.mean() < limits.min_rim_fraction:
        return None
    for _ in range(2):
        support = points[best]
        center = support.mean(0)
        _, singular, axes = np.linalg.svd(support - center, full_matrices=False)
        if len(singular) < 3 or singular[1] / math.sqrt(len(support)) < .025:
            return None
        normal = axes[-1]
        if normal @ center > 0:
            normal = -normal
        distance = -float(normal @ center)
        best = np.abs(points @ normal + distance) < tolerance
        if best.sum() < limits.min_rim_points or best.mean() < limits.min_rim_fraction:
            return None
    residual = np.abs(points[best] @ normal + distance)
    return normal, distance, best, float(np.percentile(residual, 90))


def _intersect_plane(uv, K, normal, distance):
    rays = _pixels_to_points(uv, np.ones(len(uv)), K)
    denominator = rays @ normal
    if np.any(np.abs(denominator) < .05):
        return None
    ranges = -distance / denominator
    if not np.isfinite(ranges).all() or (ranges <= 0).any():
        return None
    return rays * ranges[:, None]


class OpeningDetector:
    """Candidate rectangles and ellipses; semantic task identity is external.

    `measured_depth` must contain only direct stereo measurements. Pass the
    planar completion candidate's `measured_depth` channel, NEVER inferred_depth.
    `measured_mask` can explicitly exclude any non-measurement provenance.
    Coordinates and normals are in the calibrated LEFT OPTICAL frame.
    """
    def __init__(self, limits=None):
        self.limits = limits or OpeningLimits()

    def _image_candidates(self, gray):
        limits = self.limits
        scale = min(1., limits.max_image_width / gray.shape[1])
        image = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(image, (3, 3), .7)
        edges = cv2.Canny(blurred, 30, 90)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        image_area = image.size
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:limits.max_contours]
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not .004 * image_area <= area <= .75 * image_area or len(contour) < 24:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter < 50:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if min(w, h) < 18 or x < 2 or y < 2 or x + w >= image.shape[1] - 2 or y + h >= image.shape[0] - 2:
                continue
            poly = cv2.approxPolyDP(contour, .02 * perimeter, True)
            if len(poly) == 4 and cv2.isContourConvex(poly):
                shape = 'rectangle'
                boundary = poly.reshape(-1, 2).astype(float)
            else:
                try:
                    (cx, cy), axes, rotation = cv2.fitEllipse(contour)
                except cv2.error:
                    continue
                if min(axes) < 18 or max(axes) / min(axes) > 6:
                    continue
                xy = contour.reshape(-1, 2).astype(float) - [cx, cy]
                theta = math.radians(rotation)
                R = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
                normalized = (xy @ R.T) / (np.asarray(axes) / 2)
                if np.percentile(np.abs(np.linalg.norm(normalized, axis=1) - 1), 90) > .12:
                    continue
                shape = 'ellipse'
                theta = np.linspace(0, 2 * math.pi, 48, endpoint=False)
                local = np.column_stack((axes[0] / 2 * np.cos(theta), axes[1] / 2 * np.sin(theta)))
                boundary = local @ R + [cx, cy]
            native = (boundary + .5) / scale - .5
            native_contour = (contour.reshape(-1, 2).astype(float) + .5) / scale - .5
            candidates.append({'kind': shape, 'boundary': native, 'contour': native_contour,
                               'center': native.mean(0), 'area': area / scale**2})
        # Inner boundaries have conservative aperture dimensions. Retain them
        # ahead of a co-centered, enclosing outer frame boundary.
        selected = []
        for candidate in sorted(candidates, key=lambda c: c['area']):
            duplicate = False
            for old in selected:
                radius = math.sqrt(old['area'] / math.pi)
                if old['kind'] == candidate['kind'] and np.linalg.norm(old['center'] - candidate['center']) < max(4., .12 * radius):
                    ratio = candidate['area'] / old['area']
                    if ratio < 1.05:
                        duplicate = True
                        break
            if not duplicate:
                selected.append(candidate)
        return sorted(selected, key=lambda c: c['area'], reverse=True)[:limits.max_candidates * 2]

    def _geometry(self, candidate, depth, K, direct):
        limits = self.limits
        result = {'kind': candidate['kind'], 'frame': 'left_optical',
                  'image_center_uv': candidate['center'].tolist(),
                  'boundary_uv': candidate['boundary'].tolist(),
                  'status': 'rim_depth_unverified', 'confidence': 0.,
                  'confidence_is_calibrated_probability': False,
                  'center_camera_m': None, 'normal_camera': None,
                  'clearance_lower_m': None, 'passage_clearance_verified': False,
                  'source': 'observed image boundary and direct measured stereo only'}
        filled = np.zeros(depth.shape, np.uint8)
        contour = np.rint(candidate['contour']).astype(np.int32).reshape(-1, 1, 2)
        cv2.drawContours(filled, [contour], -1, 1, cv2.FILLED)
        outside = (cv2.dilate(filled, np.ones((7, 7), np.uint8)) != 0) & (filled == 0)
        valid = outside & direct
        yy, xx = np.nonzero(valid)
        if len(xx) < limits.min_rim_points:
            return result
        ids = np.linspace(0, len(xx) - 1, min(len(xx), limits.max_rim_points)).astype(int)
        uv = np.column_stack((xx[ids], yy[ids]))
        points = _pixels_to_points(uv, depth[yy[ids], xx[ids]], K)
        fit = _fit_plane(points, limits)
        if fit is None:
            return result
        normal, distance, inliers, residual = fit
        quadrants = np.unique(_quadrants(uv[inliers], candidate['center']))
        if len(quadrants) < 3:
            return result
        boundary3 = _intersect_plane(candidate['boundary'], K, normal, distance)
        if boundary3 is None:
            return result
        # A projected camera-horizontal basis makes dimensions reproducible;
        # the fitted aperture's own axes carry any in-plane rotation.
        e1 = np.array([1., 0., 0.]) - normal[0] * normal
        if np.linalg.norm(e1) < .1:
            return result
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        base = boundary3.mean(0)
        xy = np.column_stack(((boundary3 - base) @ e1, (boundary3 - base) @ e2))
        boundary_pixel_margin = float(np.median(points[inliers, 2]) / max(K[0], 1.)) * 2
        margin = max(.015, 2 * residual + boundary_pixel_margin)
        if candidate['kind'] == 'rectangle':
            sides = np.roll(boundary3, -1, axis=0) - boundary3
            lengths = np.linalg.norm(sides, axis=1)
            unit = sides / np.maximum(lengths[:, None], 1e-9)
            if min(lengths) < .1 or max(abs(np.dot(unit[i], unit[(i+1) % 4])) for i in range(4)) > .25:
                return result
            dimensions = np.array([min(lengths[0], lengths[2]), min(lengths[1], lengths[3])])
            center = base
            result['dimension_names'] = ['first_edge_width', 'second_edge_height']
            result['rim_corners_camera_m'] = boundary3.tolist()
        else:
            try:
                center2, axes, angle = cv2.fitEllipse((xy * 1000).astype(np.float32).reshape(-1, 1, 2))
            except cv2.error:
                return result
            dimensions = np.asarray(axes) / 1000
            center = base + e1 * (center2[0] / 1000) + e2 * (center2[1] / 1000)
            result['dimension_names'] = ['minor_diameter', 'major_diameter']
            dimensions.sort()
            result['in_plane_rotation_degrees'] = float(angle)
        result.update({'status': 'rim_geometry_only', 'center_camera_m': center.tolist(),
                       'normal_camera': normal.tolist(), 'plane_camera': [*normal.tolist(), distance],
                       'rim_dimensions_m': dimensions.tolist(), 'rim_boundary_camera_m': boundary3.tolist(),
                       'clearance_lower_m': np.maximum(0., dimensions - 2 * margin).tolist(),
                       'geometry_margin_m': margin, 'clearance_bound_is_statistically_validated': False, 'rim_measured_support': int(inliers.sum()),
                       'rim_support_fraction': float(inliers.mean()), 'rim_quadrants': len(quadrants),
                       'rim_residual_p90_m': residual})
        # Test only a central region, explicitly retaining unknown counts.
        # This is evidence behind the plane, never a certified swept opening.
        central = np.rint(candidate['center'] + .4 * (candidate['boundary'] - candidate['center'])).astype(np.int32)
        zone = np.zeros(depth.shape, np.uint8)
        cv2.fillPoly(zone, [central], 1)
        yy, xx = np.nonzero(zone)
        ids = np.linspace(0, len(xx) - 1, min(len(xx), 128)).astype(int)
        uv2 = np.column_stack((xx[ids], yy[ids]))
        plane_points = _intersect_plane(uv2, K, normal, distance)
        if plane_points is None:
            return result
        observed = direct[yy[ids], xx[ids]]
        z = depth[yy[ids], xx[ids]]
        farther = observed & (z > plane_points[:, 2] + max(.12, 3 * margin))
        blocking = observed & (z <= plane_points[:, 2] + margin)
        same_surface = observed & (np.abs(z - plane_points[:, 2]) <= margin)
        foreground = observed & (z < plane_points[:, 2] - margin)
        background_quadrants = len(np.unique(_quadrants(uv2[farther], candidate['center']))) if farther.any() else 0
        background = int(farther.sum()) >= 8 and background_quadrants >= 3 and int(blocking.sum()) == 0
        result['center_evidence'] = {'sampled_pixels': len(ids), 'observed_pixels': int(observed.sum()),
                                     'unknown_fraction': float(1 - observed.mean()),
                                     'farther_pixels': int(farther.sum()), 'blocking_pixels': int(blocking.sum()),
                                     'same_surface_pixels': int(same_surface.sum()), 'foreground_pixels': int(foreground.sum()),
                                     'background_quadrants': background_quadrants,
                                     'background_observed': bool(background)}
        if blocking.sum() >= 3:
            result['status'] = 'interior_obstructed_or_same_surface'
        elif background:
            result['status'] = 'rim_and_background_observed'
        support_score = min(1., inliers.sum() / 100) * inliers.mean() * max(0., 1 - residual / limits.plane_tolerance_m)
        result['confidence'] = float(support_score * (1. if background else .55))
        return result

    def detect(self, gray, measured_depth, K, measured_mask=None):
        gray = np.asarray(gray)
        depth = np.asarray(measured_depth)
        K = np.asarray(K, dtype=float)
        if gray.ndim != 2 or gray.dtype != np.uint8 or depth.shape != gray.shape:
            raise ValueError('Require synchronized mono8 and same-grid metric depth')
        if K.shape != (4,) or not np.isfinite(K).all() or min(K[:2]) <= 0:
            raise ValueError('K must be finite rectified [fx, fy, cx, cy]')
        if max(gray.shape) > 1920 or min(gray.shape) < 32:
            raise ValueError('Image dimensions exceed the bounded detector profile')
        direct = np.isfinite(depth) & (depth >= self.limits.minimum_depth_m) & (depth <= self.limits.maximum_depth_m)
        if measured_mask is not None:
            if np.asarray(measured_mask).shape != depth.shape:
                raise ValueError('Measurement-provenance mask mismatch')
            direct &= np.asarray(measured_mask, dtype=bool)
        candidates = self._image_candidates(gray)
        paired = [(candidate, self._geometry(candidate, depth, K, direct)) for candidate in candidates]
        retained = []
        for candidate, result in sorted(paired, key=lambda item: item[0]['area']):
            enclosing = False
            for inner, previous in retained:
                if candidate['kind'] != inner['kind'] or candidate['area'] <= inner['area']:
                    continue
                if np.linalg.norm(candidate['center'] - inner['center']) > max(4., .12 * math.sqrt(inner['area'] / math.pi)):
                    continue
                polygon = candidate['boundary'].astype(np.float32)
                if not all(cv2.pointPolygonTest(polygon, tuple(map(float, point)), False) >= 0 for point in inner['boundary']):
                    continue
                if previous['center_camera_m'] is None:
                    # A visible inner edge without metric support prevents using
                    # its larger outer frame as a verified aperture estimate.
                    enclosing = True
                elif result['center_camera_m'] is not None:
                    n1 = np.asarray(previous['normal_camera']);n2 = np.asarray(result['normal_camera'])
                    c1 = np.asarray(previous['center_camera_m']);c2 = np.asarray(result['center_camera_m'])
                    if abs(float(n1 @ n2)) > .98 and abs(float(n1 @ (c1 - c2))) < .04:
                        enclosing = True
                if enclosing:
                    break
            if not enclosing:
                retained.append((candidate, result))
        return [result for _, result in sorted(retained, key=lambda item: item[0]['area'], reverse=True)[:self.limits.max_candidates]]
