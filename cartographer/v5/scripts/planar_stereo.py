"""Bounded plane interpolation of confidence-filtered stereo measurements.

This is geometric inference under a local planar-surface assumption, not new
sensor measurements. No CAD, surveyed floor, pose or learned depth is used.
The result separates measured depth, inferred surface, and conservative free-ray
endpoints. Unknown pixels remain NaN outside supported hulls/regions. Endpoints
are not occupied-surface measurements and must not be inserted as such.
"""
from dataclasses import dataclass
import time
import cv2
import numpy as np


def _expired(deadline):
    return deadline is not None and time.monotonic() >= deadline


@dataclass(frozen=True)
class PlanePolicy:
    max_working_pixels: int = 256000  # native 1280x800 maximum
    max_planes: int = 3
    max_anchors: int = 2400
    max_regions_per_plane: int = 8
    ransac_iterations: int = 96
    disparity_inlier_px: float = .75  # native-image pixels, shared stereo tolerance
    min_plane_inliers: int = 40
    min_plane_fraction: float = .12
    min_region_inliers: int = 12
    min_region_hull_pixels: int = 180  # at the half-resolution working grid
    max_support_distance_m: float = .15
    minimum_model_depth_margin_m: float = .03
    max_relative_interval: float = .12
    canny_low: float = 8.
    canny_high: float = 16.


def _hull(points, shape):
    mask = np.zeros(shape, np.uint8)
    if len(points) < 3:
        return mask
    uv = np.asarray(points, np.float32)
    singular = np.linalg.svd(uv - uv.mean(0), compute_uv=False)
    # Near-collinear stereo support does not certify a two-dimensional patch.
    if len(singular) < 2 or singular[1] < 4. or singular[1] / max(singular[0], 1e-6) < .025:
        return mask
    hull = cv2.convexHull(uv)
    cv2.fillConvexPoly(mask, np.rint(hull).astype(np.int32), 1)
    return mask


class PlanarStereo:
    def __init__(self, policy=None):
        self.policy = policy or PlanePolicy()
        self.rng = np.random.default_rng(2026)  # repeatable generic RANSAC, not a field seed

    def compute(self, left, measured_depth, K, baseline, self_mask=None, right=None, right_self_mask=None, deadline_monotonic=None):
        p = self.policy
        left = np.asarray(left)
        measured = np.asarray(measured_depth)
        K = np.asarray(K, dtype=np.float64)
        if left.ndim != 2 or left.dtype != np.uint8:
            raise ValueError('Expected rectified mono8 image')
        if measured.dtype != np.float32:
            raise ValueError('Expected metric float32 measured depth')
        if K.shape != (4,) or not np.isfinite(K).all() or min(K[:2]) <= 0:
            raise ValueError('Expected finite [fx, fy, cx, cy] calibration')
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError('Expected positive rectified stereo baseline')
        for mask in (self_mask, right_self_mask):
            if mask is not None and (np.asarray(mask).shape != left.shape or not np.isfinite(mask).all()):
                raise ValueError('Self masks must be finite and use the native image grid')
        if left.shape != measured.shape:
            raise ValueError('Image and measured stereo grids must agree')
        height, width = left.shape
        if height % 2 or width % 2:
            raise ValueError('Even native dimensions are required')
        if height * width // 4 > p.max_working_pixels or min(height, width) < 8:
            raise ValueError('Image exceeds bounded working grid or is too small')
        if right is not None and (np.asarray(right).shape != left.shape or np.asarray(right).dtype != np.uint8):
            raise ValueError('Expected matching rectified right mono8 image')
        size = (width // 2, height // 2)
        if _expired(deadline_monotonic):
            return self._budget_unknown(measured)
        gray = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
        depth = cv2.resize(measured, size, interpolation=cv2.INTER_NEAREST)
        blocked = np.zeros_like(gray, bool) if self_mask is None else cv2.resize(np.asarray(self_mask, bool).astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0
        valid = np.isfinite(depth) & (depth > .25) & (depth < 8.) & ~blocked
        vv, uu = np.indices(depth.shape)
        # Fit native-pixel disparity as an affine function of normalized ray.
        rays = np.stack([(uu * 2 - K[2]) / K[0], (vv * 2 - K[3]) / K[1], np.ones_like(uu)], -1)
        ys, xs = np.nonzero(valid)
        disparity = K[0] * baseline / depth[ys, xs]
        design = rays[ys, xs]
        inferred = np.full_like(depth, np.nan)
        free_end = np.full_like(depth, np.nan)
        interval = np.full_like(depth, np.nan)
        plane_id = np.full(depth.shape, -1, np.int16)
        if len(xs) < p.min_plane_inliers or right is None or not np.isnan(measured).any():
            return self._native(measured, inferred, free_end, interval, plane_id, [])

        if np.asarray(right).shape != left.shape:
            raise ValueError('Stereo grids must agree')
        right_gray = cv2.resize(np.asarray(right, np.uint8), size, interpolation=cv2.INTER_AREA)
        right_blocked = np.zeros_like(gray, bool) if right_self_mask is None else cv2.resize(np.asarray(right_self_mask, bool).astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0
        minimum_support = max(p.min_plane_inliers, int(np.ceil(p.min_plane_fraction * len(xs))))

        # Image edges are barriers, never merely smooth over a foreground rim.
        smooth = cv2.GaussianBlur(gray, (3, 3), .65)
        edges = cv2.Canny(smooth, p.canny_low, p.canny_high)
        barriers = cv2.dilate(edges, np.ones((3, 3), np.uint8)) > 0
        barriers |= blocked
        _, labels = cv2.connectedComponents((~barriers).astype(np.uint8), connectivity=4)
        anchor_labels = labels[ys, xs]
        remaining = np.ones(len(xs), bool)
        rows = []
        native_anchors = {}
        for index in range(p.max_planes):
            if _expired(deadline_monotonic):
                return self._budget_unknown(measured)
            available = np.flatnonzero(remaining)
            if len(available) < minimum_support:
                break
            sample_ids = available if len(available) <= p.max_anchors else self.rng.choice(available, p.max_anchors, replace=False)
            A = design[sample_ids]
            values = disparity[sample_ids]
            best = None
            best_count = 0
            for iteration in range(p.ransac_iterations):
                if iteration % 8 == 0 and _expired(deadline_monotonic):
                    return self._budget_unknown(measured)
                three = self.rng.choice(len(sample_ids), 3, replace=False)
                model = A[three]
                if abs(np.linalg.det(model)) < 1e-5:
                    continue
                coefficients = np.linalg.solve(model, values[three])
                residual = np.abs(A @ coefficients - values)
                inliers = residual < p.disparity_inlier_px
                count = int(inliers.sum())
                if count > best_count:
                    best_count = count
                    best = coefficients
            if best is None or best_count < max(p.min_plane_inliers, int(np.ceil(p.min_plane_fraction * len(sample_ids)))):
                break
            inliers = remaining & (np.abs(design @ best - disparity) < p.disparity_inlier_px)
            for _ in range(2):
                coefficients = np.linalg.lstsq(design[inliers], disparity[inliers], rcond=None)[0]
                inliers = remaining & (np.abs(design @ coefficients - disparity) < p.disparity_inlier_px)
            if inliers.sum() < minimum_support:
                break
            # Right-image projection must preserve a nondegenerate horizontal
            # patch: u_right = u_left - disparity(u_left, v).
            if 1. - coefficients[0] / K[0] < .05:
                remaining[inliers] = False
                continue
            remaining[inliers] = False
            residuals = design[inliers] @ coefficients - disparity[inliers]
            sigma = max(.125, float(1.4826 * np.median(np.abs(residuals - np.median(residuals)))))
            covariance = np.linalg.pinv(design[inliers].T @ design[inliers]) * sigma ** 2
            predicted_disparity = rays @ coefficients
            predicted = (K[0] * baseline / np.maximum(predicted_disparity, 1e-6)).astype(np.float32)
            disparity_std = np.sqrt(np.maximum(np.einsum('...i,ij,...j->...', rays, covariance, rays), 0.))
            # Upper disparity bound corresponds to a nearer conservative surface.
            disparity_margin = p.disparity_inlier_px + 3. * disparity_std
            lower = K[0] * baseline / np.maximum(predicted_disparity + disparity_margin, 1e-6)
            lower -= p.minimum_model_depth_margin_m
            uncertainty = predicted - lower
            support = np.zeros_like(valid)
            supported_regions = 0
            region_labels, region_counts = np.unique(anchor_labels[inliers], return_counts=True)
            order = np.argsort(region_counts)[::-1][:p.max_regions_per_plane]
            for label in region_labels[order]:
                if _expired(deadline_monotonic):
                    return self._budget_unknown(measured)
                if label == 0:
                    continue
                region_inliers = inliers & (anchor_labels == label)
                if region_inliers.sum() < p.min_region_inliers:
                    continue
                hull = _hull(np.column_stack([xs[region_inliers], ys[region_inliers]]), depth.shape)
                # Raster hull boundaries alone can round outward. A working-grid
                # erosion keeps upsampled native pixels inside the true hull.
                hull = cv2.erode(hull, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
                candidate = hull.astype(bool) & (labels == label)
                if candidate.sum() < p.min_region_hull_pixels:
                    continue
                # No extrapolation beyond the inlier hull. Additionally cap local
                # interpolation in metric space, even inside large sparse hulls.
                zeros = np.ones(depth.shape, np.uint8)
                zeros[ys[region_inliers], xs[region_inliers]] = 0
                _, nearest = cv2.distanceTransformWithLabels(zeros, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
                anchor_y, anchor_x = np.nonzero(zeros == 0)
                anchor_points = rays[anchor_y, anchor_x] * depth[anchor_y, anchor_x, None]
                nearest_points = anchor_points[np.maximum(nearest - 1, 0)]
                difference = np.linalg.norm(rays * predicted[..., None] - nearest_points, axis=2)
                candidate &= difference <= p.max_support_distance_m
                support |= candidate
                supported_regions += 1
            # Never clear in front of measured nearer surfaces, including another
            # plane. A dilated foreground mask protects depth discontinuities.
            foreground = valid & (depth + np.maximum(.03, .03 * depth) < predicted)
            foreground = cv2.dilate(foreground.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            xr = (uu - predicted_disparity * .5).astype(np.float32)
            yr = vv.astype(np.float32)
            warped = cv2.remap(right_gray, xr, yr, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            photometric = cv2.boxFilter(np.abs(gray.astype(np.float32) - warped.astype(np.float32)), cv2.CV_32F, (5, 5))
            right_mask = cv2.remap(right_blocked.astype(np.uint8), xr, yr, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1) > 0
            photometric_support = (xr >= 1) & (xr < size[0] - 2) & (photometric < 5.) & ~right_mask
            accepted = support & photometric_support & ~barriers & ~foreground & ~valid & (predicted_disparity > 0) & (predicted > .25) & (predicted < 8.) & (lower > .25) & (uncertainty <= p.max_relative_interval * predicted)
            # Competing models are not resolved by accepting the farther surface.
            overwrite = accepted & (~np.isfinite(inferred) | (predicted < inferred))
            inferred[overwrite] = predicted[overwrite]
            # Preserve the shortest bound from every accepted competing model,
            # even if the selected surface is from a different nearer model.
            old_lower = free_end[accepted]
            free_end[accepted] = np.where(np.isfinite(old_lower), np.minimum(old_lower, lower[accepted]), lower[accepted])
            interval[accepted] = inferred[accepted] - free_end[accepted]
            plane_id[overwrite] = index
            native_anchors[index] = (ys[inliers] * 2, xs[inliers] * 2, depth[ys[inliers], xs[inliers]])
            rows.append({'index':index,'inlier_count':int(inliers.sum()),'native_disparity_sigma_px':sigma,'supported_regions':supported_regions,'inferred_working_pixels':int(overwrite.sum()),'coefficients_native_disparity':coefficients.tolist(),'covariance':covariance.tolist()})
        result = self._native(measured, inferred, free_end, interval, plane_id, rows)
        return self._native_veto(result, left, right, K, baseline, self_mask, right_self_mask, native_anchors, deadline_monotonic)

    def _native_veto(self, result, left, right, K, baseline, self_mask, right_self_mask, native_anchors, deadline_monotonic):
        """Recheck each native ray; half-grid interpolation cannot certify neighbors.

        This veto does not prove an unseen obstacle absent. In particular the
        right photo test is not an independent right disparity/LR measurement.
        """
        if _expired(deadline_monotonic):
            return self._budget_unknown(result['measured_depth'])
        if not np.isfinite(result['inferred_depth']).any():
            return result
        measured = result['measured_depth']
        h, w = measured.shape
        v, u = np.indices((h, w), dtype=np.float32)
        blocked = np.zeros((h, w), bool) if self_mask is None else np.asarray(self_mask, bool)
        right_blocked = np.zeros((h, w), np.uint8) if right_self_mask is None else np.asarray(right_self_mask, bool).astype(np.uint8)
        right_blocked = cv2.dilate(right_blocked, np.ones((3, 3), np.uint8))
        # Any native measured hit, including one discarded by half-grid sampling,
        # protects both its left neighborhood and its right-image projection.
        valid = np.isfinite(measured) & (measured > 0) & ~blocked
        near_left = np.where(valid, measured, np.inf).astype(np.float32)
        near_left = cv2.erode(near_left, np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=float('inf'))
        z_right = np.full((h, w), np.inf, np.float32)
        yy, xx = np.nonzero(valid)
        projected_x = xx - K[0] * baseline / measured[yy, xx]
        for rounded in (np.floor(projected_x), np.ceil(projected_x)):
            keep = (rounded >= 0) & (rounded < w)
            np.minimum.at(z_right, (yy[keep], rounded[keep].astype(np.int32)), measured[yy[keep], xx[keep]])
        z_right = cv2.erode(z_right, np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=float('inf'))
        if _expired(deadline_monotonic):
            return self._budget_unknown(measured)
        # Native edges preserve thin foreground that area reduction can erase.
        edge = cv2.Canny(left, self.policy.canny_low, self.policy.canny_high)
        native_barrier = cv2.dilate(edge, np.ones((3, 3), np.uint8)) > 0
        right_edges = cv2.dilate(cv2.Canny(right, self.policy.canny_low, self.policy.canny_high), np.ones((3, 3), np.uint8))
        nx, ny = (u - K[2]) / K[0], (v - K[3]) / K[1]
        left_float = left.astype(np.float32)
        for row in result['planes']:
            if _expired(deadline_monotonic):
                return self._budget_unknown(measured)
            index = row['index']
            selected = result['plane_id'] == index
            if not selected.any():
                continue
            # Remove the upsampling fringe: the half-grid hull must contain all
            # neighbors of a native ray, not just its nearest sample.
            selected = cv2.erode(selected.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
            coeff = np.asarray(row['coefficients_native_disparity'])
            disparity = coeff[0] * nx + coeff[1] * ny + coeff[2]
            surface = K[0] * baseline / np.maximum(disparity, 1e-6)
            cov = np.asarray(row['covariance'])
            variance = cov[0, 0] * nx * nx + cov[1, 1] * ny * ny + cov[2, 2] + 2. * (cov[0, 1] * nx * ny + cov[0, 2] * nx + cov[1, 2] * ny)
            sigma = np.sqrt(np.maximum(variance, 0.))
            lower = K[0] * baseline / np.maximum(disparity + self.policy.disparity_inlier_px + 3. * sigma, 1e-6) - self.policy.minimum_model_depth_margin_m
            # Never increase the previous free-ray claim while fixing native rays.
            lower = np.minimum(lower, result['free_ray_lower_bound'])
            xr = (u - disparity).astype(np.float32)
            warped = cv2.remap(right, xr, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            photo = cv2.boxFilter(np.abs(left_float - warped.astype(np.float32)), cv2.CV_32F, (5, 5))
            rb = cv2.remap(right_blocked, xr, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1) > 0
            re = cv2.remap(right_edges, xr, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255) > 0
            rz = cv2.remap(z_right, xr, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            tolerance = np.maximum(.03, .03 * surface)
            selected &= np.isnan(measured) & ~blocked & ~native_barrier & ~rb & ~re
            selected &= (xr >= 1) & (xr < w - 2) & (photo < 5.)
            selected &= (near_left + tolerance >= surface) & (rz + tolerance >= surface)
            selected &= (near_left >= lower) & (rz >= lower)
            selected &= (disparity > 0) & (surface > .25) & (surface < 8.) & (lower > .25)
            selected &= (surface - lower <= self.policy.max_relative_interval * surface)
            # Enforce the metric radius again on each actual native ray. A
            # half-grid neighbor is not the same 3D point on a tilted surface.
            ay, ax, az = native_anchors[index]
            zeros = np.ones((h, w), np.uint8)
            zeros[ay, ax] = 0
            _, nearest = cv2.distanceTransformWithLabels(zeros, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            sy, sx = np.nonzero(selected)
            nearest_ids = nearest[sy, sx] - 1
            near_z = az[nearest_ids]
            dx = nx[sy, sx] * surface[sy, sx] - ((ax[nearest_ids] - K[2]) / K[0]) * near_z
            dy = ny[sy, sx] * surface[sy, sx] - ((ay[nearest_ids] - K[3]) / K[1]) * near_z
            dz = surface[sy, sx] - near_z
            selected[sy, sx] &= dx * dx + dy * dy + dz * dz <= self.policy.max_support_distance_m ** 2
            reject = (result['plane_id'] == index) & ~selected
            for name in ('inferred_depth', 'free_ray_lower_bound', 'depth_interval_m'):
                result[name][reject] = np.nan
            result['plane_id'][reject] = -1
            result['inferred_depth'][selected] = surface[selected]
            result['free_ray_lower_bound'][selected] = lower[selected]
            result['depth_interval_m'][selected] = surface[selected] - lower[selected]
            row['accepted_native_pixels'] = int(selected.sum())
        if _expired(deadline_monotonic):
            return self._budget_unknown(measured)
        return result

    @staticmethod
    def _budget_unknown(measured):
        shape = (measured.shape[0] // 2, measured.shape[1] // 2)
        blank = np.full(shape, np.nan, np.float32)
        result = PlanarStereo._native(measured, blank, blank, blank, np.full(shape, -1, np.int16), [])
        result['budget_exhausted'] = True
        return result

    @staticmethod
    def _native(measured, inferred, lower, interval, plane_id, rows):
        size = (measured.shape[1], measured.shape[0])
        resize = lambda a: cv2.resize(a, size, interpolation=cv2.INTER_NEAREST)
        return {'measured_depth':measured.copy(),'inferred_depth':resize(inferred),'free_ray_lower_bound':resize(lower),'depth_interval_m':resize(interval),'plane_id':resize(plane_id),'planes':rows,'source':'local planar interpolation of confidence-filtered stereo; inferred, not a direct measurement','navigation_enabled':False}
