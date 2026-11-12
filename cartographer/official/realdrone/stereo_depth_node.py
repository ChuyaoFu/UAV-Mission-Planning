#!/usr/bin/env python3
"""Real-drone stereo depth for the skybeta OAK-D-S2 (no depth topic on the drone).

/oak/{left,right}/image_mono (1024x640 mono8, 30 Hz, already undistorted on the device: camera_info D = 0,
R = I, but K differs per eye: cy 305 vs 316 px -> 11 px raw vertical disparity) + camera_info + the TF between
the two optical frames (from the device calibration; 75.3 mm baseline, 0.33 deg relative yaw)
  -> exact 2x INTER_AREA downscale where it applies, then ONE remap = scaling + cv2.stereoRectify (alpha 0)
  -> matcher at the preset's working size (default `half320`: 320x200, StereoSGBM 3-way, 48 disparities = same
     0.33 m minimum depth as 640/96), the replica / official-compat post-processing (LR check 1 px, uniqueness 10,
     1/8 px subpixel rounding, local-texture gate on the left image, median on disparity)
  -> optional repetitive-texture (cage net) phantom mask, `net_mask:=true` (net_mask.py; default off on the drone),
     or the model-based variant `net_model:=true` (net_model.py: prior fence + VIO pose, Barrois et al. 2010)
  -> depth nearest-upsampled to `out_width` (640x400), so the mission sees the same geometry as before.
  -> optional planar scan for the maze planner, `depth_scan:=true` (depth_scan.py; default off): per column the nearest
     return whose height above the camera is in [depth_scan_h_min_m, depth_scan_h_max_m], <= depth_scan_range_max_m,
     from the final (filtered) working-resolution depth.
Publishes (all on the RECTIFIED left camera, frame `rect_frame`, header stamp = the source image stamp):
  /camera/depth/image_raw     32FC1 metres along the optical z axis, invalid = 0
  /camera/depth/camera_info   K = P(rect) at the output size, D = 0
  /stereo/left/image          the rectified left mono8 the depth is aligned to (output size)
  /stereo/left/camera_info    same intrinsics
  /tf_static                  <left optical frame> -> rect_frame (pure rotation R1^T), so base_link resolves.
  /team13/depth_scan          (depth_scan:=true) sensor_msgs/LaserScan in `depth_scan_frame` = rect_frame's origin
                              with x forward / y left / z up (static TF rect_frame -> depth_scan_frame), image stamp;
                              finite = nearest in-band return, +inf = free to range_max, NaN = no in-band data;
                              intensities = support pixels (output-image px). See depth_scan.py.
Presets / costs / accuracy: OPTIONS below and tools/bench_depth_options.py. Images arrive as raw CDR (no Python message
objects); outgoing Image.data is an array.array (Humble's per-element field check on bytes cost ~0.2 s per frame on the
Orin and was why the 2026-11-07 bench ran at 1.1 Hz). SIGINT/SIGTERM exit within 1.5 s.

Pure functions (RectifyModel, DepthPipeline, parse_image_cdr) need only numpy + cv2 and are used by the offline
tests/benchmarks.
Usage (inside the skybeta container): python3 stereo_depth_node.py --ros-args -p rate_hz:=10.0 [-p preset:=half320]
"""
import array
import json
import math
import os
import resource
import signal
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from net_mask import NetMask, DEFAULTS as NET_DEFAULTS   # noqa: E402
from net_model import NetModel, OdomPose, load_fence, quat_xyzw_to_R   # noqa: E402
from depth_filters import min_support, TemporalCheck, declare_typed   # noqa: E402
from depth_scan import DepthScan, Q_OPTICAL_SCAN_XYZW   # noqa: E402
try:                                   # same SGBM / post-processing numbers as the sims
    from replica.stereo_depth import SGBM as SIM_SGBM, POST as SIM_POST   # noqa: E402
except Exception:                      # vendored copy without replica/: identical literals
    SIM_SGBM = dict(minDisparity=0, numDisparities=96, blockSize=5, P1=8 * 5 * 5, P2=32 * 5 * 5, disp12MaxDiff=1,
                    preFilterCap=31, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, mode='MODE_SGBM')
    SIM_POST = dict(subpixel_bits=3, texture_win=7, texture_blur_sigma=1.0, texture_min_dn=1.2, median_ksize=5,
                    min_disp_px=1.0)


def quat_xyzw_to_mat(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w); x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat_xyzw(R):
    R = np.asarray(R, float); t = np.trace(R)
    if t > 0:
        s = 2.0 * math.sqrt(1.0 + t); w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]); w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]); w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]); w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return (x, y, z, w)


def scale_K(K, s):
    """Pixel-centre-correct intrinsics for an image resized by factor s."""
    K = np.asarray(K, float).reshape(3, 3).copy()
    K[0, 0] *= s; K[1, 1] *= s
    K[0, 2] = (K[0, 2] + 0.5) * s - 0.5; K[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    return K


class RectifyModel:
    """Downscale + stereo rectification maps built from the two CameraInfos and the left->right extrinsic.

    R_lr, t_lr: pose of the RIGHT optical frame in the LEFT optical frame (what TF gives for
    lookup_transform(left, right)): p_left = R_lr @ p_right + t_lr.
    """

    def __init__(self, K_l, D_l, K_r, D_r, src_size, R_lr, t_lr, width=640, alpha=0.0):
        w0, h0 = src_size
        self.s = float(width) / float(w0)
        self.size = (int(round(w0 * self.s)), int(round(h0 * self.s)))
        Kl, Kr = scale_K(K_l, self.s), scale_K(K_r, self.s)
        Dl = np.asarray(D_l if D_l is not None and len(D_l) else np.zeros(5), float)
        Dr = np.asarray(D_r if D_r is not None and len(D_r) else np.zeros(5), float)
        R_lr = np.asarray(R_lr, float); t_lr = np.asarray(t_lr, float)
        R = R_lr.T; T = -R_lr.T @ t_lr                     # OpenCV: X_r = R X_l + T
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            Kl, Dl, Kr, Dr, self.size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
        self.map_l = cv2.initUndistortRectifyMap(Kl, Dl, self.R1, self.P1, self.size, cv2.CV_16SC2)
        self.map_r = cv2.initUndistortRectifyMap(Kr, Dr, self.R2, self.P2, self.size, cv2.CV_16SC2)
        self.fx = float(self.P1[0, 0]); self.fy = float(self.P1[1, 1])
        self.cx = float(self.P1[0, 2]); self.cy = float(self.P1[1, 2])
        self.baseline = float(-self.P2[0, 3] / self.P2[0, 0])
        self.rect_angle_deg = float(np.degrees(np.linalg.norm(cv2.Rodrigues(self.R1)[0])))

    def K(self):
        return (self.fx, self.fy, self.cx, self.cy)

    def rect_in_left_quat(self):
        """Rotation of the rectified left frame in the original left optical frame (TF parent=left, child=rect)."""
        return mat_to_quat_xyzw(self.R1.T)

    def apply(self, left, right):
        if left.shape[1] != self.size[0]:
            left = cv2.resize(left, self.size, interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, self.size, interpolation=cv2.INTER_AREA)
        return (cv2.remap(left, *self.map_l, cv2.INTER_LINEAR),
                cv2.remap(right, *self.map_r, cv2.INTER_LINEAR))

    def describe(self):
        return dict(size=list(self.size), scale=round(self.s, 4), fx=round(self.fx, 2), fy=round(self.fy, 2),
                    cx=round(self.cx, 2), cy=round(self.cy, 2), baseline_m=round(self.baseline, 5),
                    rect_rot_deg=round(self.rect_angle_deg, 3),
                    min_depth_m=round(self.fx * self.baseline / SIM_SGBM['numDisparities'], 3))


class Sgbm:
    """StereoSGBM + the sim's post-processing; depth float32, invalid = 0."""

    def __init__(self, mode='SGBM_3WAY', num_disp=None, texture_min_dn=None, median_ksize=None, max_depth_m=0.0):
        q = dict(SIM_SGBM); q.pop('mode')
        if num_disp: q['numDisparities'] = int(num_disp)
        self.mode_name = str(mode)
        self.m = cv2.StereoSGBM_create(mode=getattr(cv2, 'STEREO_SGBM_MODE_' + self.mode_name), **q)
        self.post = dict(SIM_POST)
        if texture_min_dn is not None: self.post['texture_min_dn'] = float(texture_min_dn)
        if median_ksize is not None: self.post['median_ksize'] = int(median_ksize)
        self.max_depth = float(max_depth_m)
        self.params = dict(sgbm=dict(q, mode=self.mode_name), post=self.post, max_depth_m=self.max_depth)

    def depth(self, left, right, fx, baseline):
        p = self.post
        d16 = self.m.compute(left, right)                          # int16, 4 fractional bits, <0 invalid
        valid = d16 > 0
        disp = d16.astype(np.float32) * (1.0 / 16.0)
        qn = float(2 ** p['subpixel_bits'])
        disp = np.round(disp * qn) / qn
        b = cv2.GaussianBlur(left.astype(np.float32), (0, 0), p['texture_blur_sigma'])
        w = (p['texture_win'], p['texture_win'])
        std = np.sqrt(np.maximum(cv2.blur(b * b, w) - cv2.blur(b, w) ** 2, 0))
        valid &= std >= p['texture_min_dn']
        disp[~valid] = 0.0
        if p['median_ksize'] > 1:
            disp = cv2.medianBlur(disp, p['median_ksize'])          # holes stay holes (device-like)
        valid &= disp >= p['min_disp_px']
        depth = np.zeros(disp.shape, np.float32)
        depth[valid] = (fx * baseline) / disp[valid]
        if self.max_depth > 0:
            depth[depth > self.max_depth] = 0.0
        return depth


# Depth presets. `width` is the working (matching) width of the 1024x640 source; the published depth / left image are
# always `out_width` wide (nearest-neighbour upsampling of depth, so the mission sees the same 640x400 geometry and its
# stride-4 sampling is unchanged). `band`: only rows [top, bottom) (fractions) are matched, the rest are 0 (unknown).
OPTIONS = {
    'legacy640': dict(width=640, matcher='SGBM_3WAY', num_disp=96, block=5, median=5, texture_win=7, legacy=True),
    'ref640': dict(width=640, matcher='SGBM_3WAY', num_disp=96, block=5, median=5, texture_win=7),
    '640_nd64': dict(width=640, matcher='SGBM_3WAY', num_disp=64, block=5, median=5, texture_win=7),
    'band640': dict(width=640, matcher='SGBM_3WAY', num_disp=96, block=5, median=5, texture_win=7, band=(0.2, 0.8)),
    'mid512': dict(width=512, matcher='SGBM_3WAY', num_disp=80, block=5, median=5, texture_win=5),
    'half320': dict(width=320, matcher='SGBM_3WAY', num_disp=48, block=5, median=3, texture_win=5),
    'half320_b3': dict(width=320, matcher='SGBM_3WAY', num_disp=48, block=3, median=3, texture_win=5),
    'half320_band': dict(width=320, matcher='SGBM_3WAY', num_disp=48, block=5, median=3, texture_win=5,
                         band=(0.2, 0.8)),
    'bm320': dict(width=320, matcher='BM', num_disp=48, block=9, median=3, texture_win=5),
}
DEFAULT_PRESET = 'half320'


class DepthPipeline:
    """source pair -> (depth float32 [out size, invalid 0], rectified left uint8 [out size], timings).

    Not rectified_input: 1024x640 source + CameraInfo K/D + left->right extrinsic (see RectifyModel).
    rectified_input=True: the pair is already rectified at src_size with intrinsics K_l (replica / sim frames); t_lr[0]
    is the baseline."""

    def __init__(self, opt, K_l, D_l, K_r, D_r, src_size, R_lr, t_lr, out_width=640, rectified_input=False,
                 max_depth_m=10.0, texture_min_dn=None, net_mask=None, net_model=None, speckle=None, minsup=None,
                 temporal=None, scan=None):
        """speckle (W, R): SGBM speckleWindowSize / speckleRange at the 640 scale (default 100, 2; scaled to the work
        size). minsup (N, D): depth_filters.min_support on the final disparity, N px / D px at the 640 scale. temporal
        (tol, frames, require): depth_filters.TemporalCheck (needs the pose and stamp in run()). scan: dict of
        depth_scan.DepthScan keyword arguments (h_min, h_max, range_max, support_tol) -> self.scan after each run()."""
        o = dict(opt); self.opt = o
        self.net_mask = net_mask                         # NetMask instance or None (model-free texture mask)
        self.net_model = net_model                       # NetModel instance or None (fence-model filter, needs a pose)
        self.minsup = minsup; self.temporal_cfg = temporal; self.tcheck = None
        self.mask_frac = 0.0; self.model_info = {}
        w0, h0 = src_size
        self.rectified_input = bool(rectified_input)
        if self.rectified_input:
            s = o['width'] / float(w0)
            self.work_size = (int(round(w0 * s)), int(round(h0 * s)))
            Kw = scale_K(K_l, s); self.fx, self.baseline = float(Kw[0, 0]), float(abs(t_lr[0]))
            self.Kw = (float(Kw[0, 0]), float(Kw[1, 1]), float(Kw[0, 2]), float(Kw[1, 2]))
            self.model = None
        else:
            self.model = RectifyModel(K_l, D_l, K_r, D_r, src_size, R_lr, t_lr, width=o['width'])
            self.work_size = self.model.size; self.fx, self.baseline = self.model.fx, self.model.baseline
            self.Kw = self.model.K()
        ww, hw = self.work_size
        self.up = float(out_width) / ww
        self.out_size = (int(round(ww * self.up)), int(round(hw * self.up)))
        fx, fy, cx, cy = self.Kw
        Ko = scale_K(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), self.up)
        self.K_out = (float(Ko[0, 0]), float(Ko[1, 1]), float(Ko[0, 2]), float(Ko[1, 2]))
        self.left_map = None
        if self.model is not None and abs(self.up - 1.0) > 1e-9:
            # rectified left at the OUTPUT size straight from the full-res source (one remap; R1 is scale-free)
            P = np.array([[Ko[0, 0], 0, Ko[0, 2], 0], [0, Ko[1, 1], Ko[1, 2], 0], [0, 0, 1.0, 0]])
            Dl = np.asarray(D_l if D_l is not None and len(D_l) else np.zeros(5), float)
            self.left_map = cv2.initUndistortRectifyMap(np.asarray(K_l, float).reshape(3, 3), Dl, self.model.R1, P,
                                                        self.out_size, cv2.CV_16SC2)
        b = o.get('band')
        self.rows = (0, hw) if not b else (int(round(hw * b[0])), int(round(hw * b[1])))
        self.pre_k = 1
        if self.model is not None:
            # Downscale by an exact integer factor first (INTER_AREA's fast box path; a non-integer INTER_AREA ratio such
            # as 1024->640 costs ~4 ms per image on a fast x86, ~20 ms on the Orin), then ONE remap does the rest of the
            # scaling + rectification. 1024x640 -> 512x320 -> remap 320x200; for 640 the remap samples the source.
            for k in (8, 4, 2):
                if w0 % k == 0 and h0 % k == 0 and w0 // k >= ww:
                    self.pre_k = k; break
            k = self.pre_k; pre = (w0 // k, h0 // k)
            Dl = np.asarray(D_l if D_l is not None and len(D_l) else np.zeros(5), float)
            Dr = np.asarray(D_r if D_r is not None and len(D_r) else np.zeros(5), float)
            ml = cv2.initUndistortRectifyMap(scale_K(K_l, 1.0 / k), Dl, self.model.R1, self.model.P1, self.work_size, cv2.CV_16SC2)
            mr = cv2.initUndistortRectifyMap(scale_K(K_r, 1.0 / k), Dr, self.model.R2, self.model.P2, self.work_size, cv2.CV_16SC2)
            self.pre_size = pre
            r0, r1 = self.rows
            self.maps = tuple((m[0][r0:r1], m[1][r0:r1]) for m in (ml, mr))
            self.full_left_map = ml
        sc = ww / 640.0
        nd = int(o['num_disp']); blk = int(o['block'])
        if o['matcher'] == 'BM':
            m = cv2.StereoBM_create(numDisparities=nd, blockSize=blk)
            m.setPreFilterCap(31); m.setTextureThreshold(10); m.setUniquenessRatio(10)
            m.setSpeckleWindowSize(int(round(100 * sc * sc))); m.setSpeckleRange(max(1, int(round(2 * sc))))
            m.setDisp12MaxDiff(1)
        else:
            q = dict(SIM_SGBM); q.pop('mode')
            if speckle:
                q['speckleWindowSize'], q['speckleRange'] = int(speckle[0]), int(speckle[1])
            q.update(numDisparities=nd, blockSize=blk, P1=8 * blk * blk, P2=32 * blk * blk,
                     speckleWindowSize=int(round(q['speckleWindowSize'] * sc * sc)),
                     speckleRange=max(1, int(round(q['speckleRange'] * sc))))
            m = cv2.StereoSGBM_create(mode=getattr(cv2, 'STEREO_SGBM_MODE_' + o['matcher']), **q)
        self.matcher = m
        self.post = dict(SIM_POST, median_ksize=int(o['median']), texture_win=int(o['texture_win']))
        if texture_min_dn is not None: self.post['texture_min_dn'] = float(texture_min_dn)
        self.max_depth = float(max_depth_m)
        self.min_depth = self.fx * self.baseline / nd
        self.scanner = None; self.scan = None
        if scan is not None:                     # planar scan from the final working-resolution depth (depth_scan.py)
            self.scanner = DepthScan(self.Kw, self.work_size, range_min=self.min_depth, px_scale=self.up, **scan)

    def describe(self):
        return dict(opt=self.opt, work=list(self.work_size), out=list(self.out_size), rows=list(self.rows),
                    net_mask=None if self.net_mask is None else self.net_mask.params(),
                    fx_work=round(self.fx, 2), baseline_m=round(self.baseline, 5), min_depth_m=round(self.min_depth, 3),
                    K_out=[round(v, 2) for v in self.K_out],
                    rect_rot_deg=None if self.model is None else round(self.model.rect_angle_deg, 3))

    def _prepare(self, left, right):
        ww, hw = self.work_size; r0, r1 = self.rows
        if self.model is None:
            if left.shape[1] != ww:
                left = cv2.resize(left, self.work_size, interpolation=cv2.INTER_AREA)
                right = cv2.resize(right, self.work_size, interpolation=cv2.INTER_AREA)
            return left, right, left[r0:r1], right[r0:r1]
        if self.opt.get('legacy'):                       # the 2026-11-07 bench pipeline, kept for comparison only
            Lb, Rb = self.model.apply(left, right)
            return Lb, None, Lb, Rb
        if self.pre_k > 1:
            ls = cv2.resize(left, self.pre_size, interpolation=cv2.INTER_AREA)
            rs = cv2.resize(right, self.pre_size, interpolation=cv2.INTER_AREA)
        else:
            ls, rs = left, right
        (ml, ml2), (mr, mr2) = self.maps
        Lb = cv2.remap(ls, ml, ml2, cv2.INTER_LINEAR); Rb = cv2.remap(rs, mr, mr2, cv2.INTER_LINEAR)
        return ls, rs, Lb, Rb

    def run(self, left, right, pose=None, stamp=None):
        """pose: (R_nc, cam_ned) optical camera -> prior start-frame NED at the image stamp, for net_model (else skipped)"""
        t0 = time.perf_counter()
        ls, _, Lb, Rb = self._prepare(left, right)
        if self.left_map is not None:
            left_out = cv2.remap(left, *self.left_map, cv2.INTER_LINEAR)
        elif self.model is None:
            left_out = ls if abs(self.up - 1.0) < 1e-9 else cv2.resize(ls, self.out_size, interpolation=cv2.INTER_AREA)
        elif self.rows == (0, self.work_size[1]):
            left_out = Lb
        else:
            left_out = cv2.remap(ls, *self.full_left_map, cv2.INTER_LINEAR)
        t1 = time.perf_counter()
        p = self.post
        d16 = self.matcher.compute(Lb, Rb)
        valid = d16 > 0
        disp = d16.astype(np.float32) * (1.0 / 16.0)
        qn = float(2 ** p['subpixel_bits'])
        disp = np.round(disp * qn) / qn
        b = cv2.GaussianBlur(Lb.astype(np.float32), (0, 0), p['texture_blur_sigma'])
        w = (p['texture_win'], p['texture_win'])
        std = np.sqrt(np.maximum(cv2.blur(b * b, w) - cv2.blur(b, w) ** 2, 0))
        valid &= std >= p['texture_min_dn']
        disp[~valid] = 0.0
        if p['median_ksize'] > 1:
            disp = cv2.medianBlur(disp, p['median_ksize'])
        valid &= disp >= p['min_disp_px']
        tm = 0.0
        if self.net_mask is not None:            # repetitive-texture (cage net) aliases -> invalid (net_mask.py)
            tm = time.perf_counter()
            bad = self.net_mask(Lb, np.where(valid, disp, np.float32(0)), Rb)
            self.mask_frac = float(np.count_nonzero(bad)) / max(int(np.count_nonzero(valid)), 1)
            valid &= ~bad
            tm = time.perf_counter() - tm
        if self.net_model is not None and pose is not None and self.rows == (0, self.work_size[1]):
            t_ = time.perf_counter()               # fence-model filter (net_model.py), work resolution, FFT at out size
            bad, self.model_info = self.net_model(left_out, np.where(valid, disp, np.float32(0)), Rb, self.Kw,
                                                  self.baseline, pose[0], pose[1], left_work=Lb)
            self.mask_frac = float(np.count_nonzero(bad)) / max(int(np.count_nonzero(valid)), 1)
            valid &= ~bad
            tm += time.perf_counter() - t_
        if self.minsup or (self.temporal_cfg and pose is not None and stamp is not None):
            t_ = time.perf_counter()                 # free-space noise filters (depth_filters.py), work resolution
            sc_ = self.work_size[0] / 640.0
            if self.minsup:
                valid &= min_support(np.where(valid, disp, np.float32(0)), max(1, int(round(self.minsup[0] * sc_ * sc_))),
                                     self.minsup[1] * sc_)
            if self.temporal_cfg and pose is not None and stamp is not None and self.rows == (0, self.work_size[1]):
                if self.tcheck is None:
                    tol, nfr, req = self.temporal_cfg
                    self.tcheck = TemporalCheck(self.Kw, tol=tol, nprev=int(nfr), require=bool(req))
                z = np.where(valid, np.float32(self.fx * self.baseline) / np.maximum(disp, 1e-6), 0).astype(np.float32)
                valid &= self.tcheck(z, pose[0], pose[1], stamp)
            tm += time.perf_counter() - t_
        band = np.zeros(disp.shape, np.float32)
        band[valid] = (self.fx * self.baseline) / disp[valid]
        if self.max_depth > 0:
            band[band > self.max_depth] = 0.0
        r0, r1 = self.rows
        if (r0, r1) == (0, self.work_size[1]):
            depth = band
        else:
            depth = np.zeros((self.work_size[1], self.work_size[0]), np.float32); depth[r0:r1] = band
        t2 = time.perf_counter()
        depth_work = depth
        if abs(self.up - 1.0) > 1e-9:
            depth = cv2.resize(depth, self.out_size, interpolation=cv2.INTER_NEAREST)
        t3 = time.perf_counter()
        ts = 0.0
        if self.scanner is not None:             # planar scan of the final (filtered) working-resolution depth
            self.scan = self.scanner(depth_work)
            ts = time.perf_counter() - t3
        return depth, left_out, dict(rect=t1 - t0, match=t2 - t1 - tm, mask=tm, up=t3 - t2, scan=ts)


def image_msg_data(arr):
    """Image.data payload. MUST be array.array('B'): on Humble, assigning bytes/ndarray makes rosidl run a per-element
    Python type/range check over the whole buffer (~48 ms per 640x400 float frame on a 5 GHz x86, ~4x that on the Orin)."""
    return array.array('B', np.ascontiguousarray(arr).tobytes())




def parse_image_cdr(b):
    """sensor_msgs/Image straight from its serialized CDR (rclpy raw=True subscription), without building the Python
    message (halves the receive cost of the 2 x 30 Hz 1024x640 streams). Returns
    (sec, nanosec, frame_id, height, width, encoding, step, data memoryview); raises ValueError if malformed."""
    try:
        return _parse_image_cdr(memoryview(b))
    except struct.error as e:
        raise ValueError(f'truncated Image CDR: {e}') from None


def _parse_image_cdr(mv):
    if len(mv) < 40 or mv[0] != 0 or mv[1] not in (0, 1):
        raise ValueError('not a CDR payload')
    e = '<' if mv[1] == 1 else '>'

    def align4(o):
        return o + ((4 - (o - 4) % 4) % 4)                     # alignment is relative to the body (offset 4)
    o = 4
    sec, nsec, ln = struct.unpack_from(e + 'iII', mv, o); o += 12
    frame_id = bytes(mv[o:o + max(ln - 1, 0)]).decode('utf-8', 'replace'); o += ln
    o = align4(o)
    h, w, ln = struct.unpack_from(e + 'III', mv, o); o += 12
    enc = bytes(mv[o:o + max(ln - 1, 0)]).decode('ascii', 'replace'); o += ln
    o += 1                                                     # is_bigendian
    o = align4(o)
    step, n = struct.unpack_from(e + 'II', mv, o); o += 8
    if n != step * h or o + n > len(mv) or step < w:
        raise ValueError(f'inconsistent Image CDR (h {h} step {step} n {n} len {len(mv)})')
    return sec, nsec, frame_id, h, w, enc, step, mv[o:o + n]


def to_gray(enc, h, w, step, data):
    if enc in ('mono8', '8UC1'):
        return np.frombuffer(data, np.uint8).reshape(h, step)[:, :w]
    if enc in ('bgr8', 'rgb8'):
        img = np.frombuffer(data, np.uint8).reshape(h, step)[:, :w * 3].reshape(h, w, 3)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if enc == 'bgr8' else cv2.COLOR_RGB2GRAY)
    return None


def resolve_options(preset, width=0, num_disp=0, matcher='', block=0, median=-1, texture_win=0, band=(0.0, 1.0)):
    """Preset + explicit overrides (0 / '' / -1 = keep the preset). A width override without num_disp keeps the
    preset's minimum depth (~0.33 m): num_disp scales with the width, rounded up to a multiple of 16."""
    if preset not in OPTIONS:
        raise ValueError(f'unknown stereo preset {preset!r}; one of {sorted(OPTIONS)}')
    o = dict(OPTIONS[preset])
    if width:
        if not num_disp:
            num_disp = max(16, int(math.ceil(o['num_disp'] * float(width) / o['width'] / 16.0)) * 16)
        o['width'] = int(width); o.pop('legacy', None)
    if num_disp: o['num_disp'] = int(num_disp)
    if matcher: o['matcher'] = str(matcher)
    if block: o['block'] = int(block)
    if median >= 0: o['median'] = int(median)
    if texture_win: o['texture_win'] = int(texture_win)
    if band and tuple(band) != (0.0, 1.0): o['band'] = (float(band[0]), float(band[1]))
    return o


# ---------------------------------------------------------------------------------------------- ROS node
def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CameraInfo, Image, LaserScan
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import TransformStamped
    from tf2_ros import Buffer, TransformListener
    from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
    try:
        from rclpy.executors import ExternalShutdownException
    except ImportError:                                        # older rclpy
        class ExternalShutdownException(Exception):
            pass

    class StereoDepthNode(Node):
        def __init__(self):
            super().__init__('team13_stereo_depth')
            P = lambda n, v: declare_typed(self, n, v)          # any override type -> the default's type
            self.topics = dict(left=P('left_topic', '/oak/left/image_mono'), right=P('right_topic', '/oak/right/image_mono'),
                               left_info=P('left_info_topic', '/oak/left/camera_info'),
                               right_info=P('right_info_topic', '/oak/right/camera_info'))
            self.preset = str(P('preset', DEFAULT_PRESET))
            self.opt = resolve_options(self.preset, width=int(P('width', 0)), num_disp=int(P('num_disparities', 0)),
                                       matcher=str(P('sgbm_mode', '')), block=int(P('block_size', 0)),
                                       median=int(P('median_ksize', -1)), texture_win=int(P('texture_win', 0)),
                                       band=(float(P('band_top', 0.0)), float(P('band_bottom', 1.0))))
            self.out_width = int(P('out_width', 640))
            self.texture_min_dn = float(P('texture_min_dn', SIM_POST['texture_min_dn']))
            self.max_depth = float(P('max_depth_m', 10.0))
            self.net_mask = bool(P('net_mask', False))
            nm_over = {k: P('net_mask_' + k, v) for k, v in (('score_min', NET_DEFAULTS['score_min']),
                                                               ('d_margin', NET_DEFAULTS['d_margin']),
                                                               ('pfrac', NET_DEFAULTS['pfrac']))}
            self.net_mask_kw = {k: float(v) for k, v in nm_over.items()}
            self.net_mask_kw['lag_thin'] = bool(P('net_mask_lag_thin', False))     # Orin budget fallback (-~20 % cost)
            # model-based variant: fence from the course prior + camera pose from VIO odometry and TF
            self.net_model_on = bool(P('net_model', False))
            fence_path = str(P('net_fence_json', 'mission/course_prior.site.json'))
            self.net_base_frame = str(P('net_base_frame', 'base_link'))
            odom_topic = str(P('net_odom_topic', '/visual_slam/tracking/odometry'))
            self.odom_pose = OdomPose(anchor=str(P('net_anchor', 'first')), base_height_m=float(P('net_base_height_m', 0.05)),
                                      max_dt=float(P('net_odom_max_dt_s', 0.1)))
            self.fence = None; self.R_bc = None; self.t_bc = None; self.model_ok = 0
            # free-space noise filters (depth_filters.py; all off by default)
            spk = (int(P('sgbm_speckle_window', 0)), int(P('sgbm_speckle_range', 0)))
            self.speckle = spk if spk[0] > 0 else None
            ms = (int(P('noise_minsup_px', 0)), float(P('noise_minsup_diff_px', 1.0)))
            self.minsup = ms if ms[0] > 0 else None
            tt = (float(P('noise_temporal_tol', 0.0)), int(P('noise_temporal_frames', 2)), bool(P('noise_temporal_require', False)))
            self.temporal = tt if tt[0] > 0 else None
            # planar scan for the maze planner's 2-D grid (depth_scan.py; off by default)
            self.depth_scan = bool(P('depth_scan', False))
            self.scan_kw = dict(h_min=float(P('depth_scan_h_min_m', -0.25)), h_max=float(P('depth_scan_h_max_m', 0.35)),
                                range_max=float(P('depth_scan_range_max_m', 5.0)),
                                support_tol=float(P('depth_scan_support_tol', 0.10)))
            self.scan_frame = str(P('depth_scan_frame', 'team13_depth_scan'))
            scan_topic = str(P('depth_scan_topic', '/team13/depth_scan'))
            if self.net_model_on:
                fp = Path(fence_path) if Path(fence_path).is_absolute() else ROOT / fence_path
                try:
                    self.fence = load_fence(str(fp))
                except Exception as e:  # noqa: BLE001
                    self.get_logger().error(f'net_model: cannot read the fence from {fp}: {e!r}; model filter OFF')
                    self.net_model_on = False
            self.rate_hz = float(P('rate_hz', 10.0))
            self.sync_tol = float(P('sync_tol_s', 0.004))
            self.rect_frame = P('rect_frame', 'team13_left_rect_optical_frame')
            self.use_tf = bool(P('use_tf_extrinsics', True))
            self.raw_rx = bool(P('raw_rx', True))
            depth = int(P('image_queue_depth', 2))
            nthreads = int(P('cv_threads', 2))
            if nthreads > 0: cv2.setNumThreads(nthreads)
            self.pub_depth = self.create_publisher(Image, P('depth_topic', '/camera/depth/image_raw'), qos_profile_sensor_data)
            self.pub_dinfo = self.create_publisher(CameraInfo, P('depth_info_topic', '/camera/depth/camera_info'), qos_profile_sensor_data)
            self.pub_left = self.create_publisher(Image, P('left_out_topic', '/stereo/left/image'), qos_profile_sensor_data)
            self.pub_linfo = self.create_publisher(CameraInfo, P('left_out_info_topic', '/stereo/left/camera_info'), qos_profile_sensor_data)
            # reliable (small message): matches both reliable and best-effort / sensor-data subscribers
            self.pub_scan = (self.create_publisher(LaserScan, scan_topic, QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=5)) if self.depth_scan else None)
            self.info = {}; self.pipe = None; self.frames = {'l': {}, 'r': {}}
            self.lock = threading.Lock(); self.evt = threading.Event(); self.stop = threading.Event()
            self.tf_buffer = Buffer(); self.tf_listener = TransformListener(self.tf_buffer, self)
            self.static_tf = StaticTransformBroadcaster(self)
            self.create_subscription(CameraInfo, self.topics['left_info'], lambda m: self.info.__setitem__('l', m), qos_profile_sensor_data)
            if self.net_model_on or self.temporal:
                self.create_subscription(Odometry, odom_topic, self.on_odom, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, self.topics['right_info'], lambda m: self.info.__setitem__('r', m), qos_profile_sensor_data)
            # newest frames only: a deep queue just adds latency when the matcher falls behind
            img_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=max(depth, 1))
            for side, key in (('l', 'left'), ('r', 'right')):
                if self.raw_rx:
                    self.create_subscription(Image, self.topics[key], lambda b, s=side: self.on_raw(s, b), img_qos, raw=True)
                else:
                    self.create_subscription(Image, self.topics[key], lambda m, s=side: self.on_msg(s, m), img_qos)
            self.stats = self._zero_stats(); self.stats_t = time.monotonic(); self.cpu0 = time.process_time()
            self.t_start = time.monotonic()
            self.create_timer(1.0, self.try_build)
            self.create_timer(5.0, self.report)
            self.worker_thread = threading.Thread(target=self.worker, daemon=True); self.worker_thread.start()
            self.get_logger().info(f'team13 stereo depth: {self.topics} preset {self.preset} -> {json.dumps(self.opt)}, '
                                   f'out width {self.out_width}, <= {self.rate_hz} Hz, raw_rx {self.raw_rx}, '
                                   f'cv threads {nthreads}, net_mask {self.net_mask} {self.net_mask_kw if self.net_mask else ""}, '
                                   f'net_model {self.net_model_on} fence {self.fence} odom {odom_topic} anchor {self.odom_pose.anchor}, '
                                   f'speckle {self.speckle} minsup {self.minsup} temporal {self.temporal}, '
                                   f'depth_scan {self.depth_scan} {self.scan_kw if self.depth_scan else ""}'
                                   f'{(" -> " + scan_topic + " in " + self.scan_frame) if self.depth_scan else ""}')

        @staticmethod
        def _zero_stats():
            return dict(n=0, t_rect=0.0, t_match=0.0, t_mask=0.0, t_up=0.0, t_pub=0.0, valid=0.0, rx=0, raw_fallback=0,
                        lat=0.0, mask_frac=0.0, t_scan=0.0, scan_hit=0.0, scan_free=0.0)

        def _store(self, side, sec, nsec, img):
            t = sec + nsec * 1e-9
            with self.lock:
                self.stats['rx'] += 1
                d = self.frames[side]; d[t] = ((sec, nsec), img)
                for k in sorted(d)[:-4]: d.pop(k)
            self.evt.set()

        def on_raw(self, side, b):
            try:
                sec, nsec, _, h, w, enc, step, data = parse_image_cdr(b)
            except ValueError:
                self.stats['raw_fallback'] += 1
                return self.on_msg(side, deserialize_message(b, Image))
            img = to_gray(enc, h, w, step, data)
            if img is not None:
                self._store(side, sec, nsec, img)

        def on_msg(self, side, m):
            img = to_gray(m.encoding, m.height, m.width, m.step, m.data)
            if img is not None:
                self._store(side, m.header.stamp.sec, m.header.stamp.nanosec, img)

        def try_build(self):
            if self.pipe is not None or 'l' not in self.info or 'r' not in self.info:
                return
            il, ir = self.info['l'], self.info['r']
            R_lr, t_lr, src = np.eye(3), None, 'camera_info'
            if self.use_tf:
                try:
                    from rclpy.time import Time
                    x = self.tf_buffer.lookup_transform(il.header.frame_id, ir.header.frame_id, Time())
                    tr, q = x.transform.translation, x.transform.rotation
                    R_lr = quat_xyzw_to_mat(q.x, q.y, q.z, q.w); t_lr = np.array([tr.x, tr.y, tr.z]); src = 'tf'
                except Exception as e:  # noqa: BLE001
                    if time.monotonic() - self.t_start < 10.0:
                        self.get_logger().info(f'waiting for TF {il.header.frame_id}->{ir.header.frame_id}: {e}')
                        return
                    self.get_logger().warn('no left->right TF after 10 s: falling back to camera_info P baseline, R = I')
            if t_lr is None:
                B = max(abs(il.p[3]), abs(ir.p[3])) / max(il.p[0], 1e-6)
                t_lr = np.array([B, 0.0, 0.0])
            K_l = np.array(il.k).reshape(3, 3); K_r = np.array(ir.k).reshape(3, 3)
            D_l = np.array(il.d[:8] if len(il.d) >= 8 else il.d, float); D_r = np.array(ir.d[:8] if len(ir.d) >= 8 else ir.d, float)
            if not np.any(D_l): D_l = np.zeros(5)
            if not np.any(D_r): D_r = np.zeros(5)
            if il.width != ir.width or il.height != ir.height:
                self.get_logger().error('left/right camera_info sizes differ'); return
            self.pipe = DepthPipeline(self.opt, K_l, D_l, K_r, D_r, (il.width, il.height), R_lr, t_lr,
                                      out_width=self.out_width, max_depth_m=self.max_depth, texture_min_dn=self.texture_min_dn,
                                      net_mask=NetMask(**self.net_mask_kw) if self.net_mask and not self.net_model_on else None,
                                      net_model=NetModel(self.fence, mask_kw=self.net_mask_kw) if self.net_model_on else None,
                                      speckle=self.speckle, minsup=self.minsup, temporal=self.temporal,
                                      scan=self.scan_kw if self.depth_scan else None)
            self.left_frame = il.header.frame_id
            st = TransformStamped(); st.header.stamp = self.get_clock().now().to_msg()
            st.header.frame_id = self.left_frame; st.child_frame_id = self.rect_frame
            qx, qy, qz, qw = self.pipe.model.rect_in_left_quat()
            st.transform.rotation.x, st.transform.rotation.y, st.transform.rotation.z, st.transform.rotation.w = qx, qy, qz, qw
            tfs = [st]
            if self.depth_scan:                  # scan frame: same origin, x forward / y left / z up
                ss = TransformStamped(); ss.header.stamp = st.header.stamp
                ss.header.frame_id = self.rect_frame; ss.child_frame_id = self.scan_frame
                (ss.transform.rotation.x, ss.transform.rotation.y, ss.transform.rotation.z,
                 ss.transform.rotation.w) = Q_OPTICAL_SCAN_XYZW
                tfs.append(ss)
                self.scan_msg = self.make_scan()
            self.static_tf.sendTransform(tfs)
            self.cinfo = self.make_info()
            self.get_logger().info(f'rectification ready (extrinsic from {src}, t_lr {np.round(t_lr, 5).tolist()}): '
                                   f'{json.dumps(self.pipe.describe())}; static TF {self.left_frame} -> {self.rect_frame}')

        def make_info(self):
            m = CameraInfo(); w, h = self.pipe.out_size
            m.width, m.height = w, h; m.distortion_model = 'plumb_bob'; m.d = [0.0] * 5
            fx, fy, cx, cy = self.pipe.K_out
            m.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            m.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            m.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            m.header.frame_id = self.rect_frame
            return m

        def make_scan(self):
            sc = self.pipe.scanner; m = LaserScan(); m.header.frame_id = self.scan_frame
            m.angle_min, m.angle_max, m.angle_increment = sc.angle_min, sc.angle_max, sc.angle_increment
            m.time_increment = 0.0; m.scan_time = 1.0 / max(self.rate_hz, 0.1)
            m.range_min, m.range_max = float(sc.range_min), float(sc.range_max)
            return m

        def take_pair(self):
            with self.lock:
                L, R = self.frames['l'], self.frames['r']
                best = None
                for tl in sorted(L, reverse=True):
                    for tr in R:
                        if abs(tl - tr) <= self.sync_tol:
                            best = (tl, tr); break
                    if best: break
                if best is None:
                    return None
                stamp, il = L[best[0]]; _, ir = R[best[1]]
                for d, t in ((L, best[0]), (R, best[1])):
                    for k in [k for k in d if k <= t]: d.pop(k)
            return stamp, il, ir

        def worker(self):
            period = 1.0 / max(self.rate_hz, 0.1); last = 0.0
            while not self.stop.is_set():
                self.evt.wait(0.2); self.evt.clear()
                if self.pipe is None or self.stop.is_set(): continue
                wait = last + period - time.monotonic()
                if wait > 0 and self.stop.wait(wait): break
                item = self.take_pair()
                if item is None: continue
                last = time.monotonic()
                try:
                    self.process(*item)
                except Exception as e:  # noqa: BLE001 - keep the node alive
                    if self.stop.is_set(): break
                    self.get_logger().error(f'stereo processing failed: {e!r}')

        def on_odom(self, m):
            p, q = m.pose.pose.position, m.pose.pose.orientation
            self.odom_pose.add(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w))

        def model_pose(self, stamp):
            """(R_nc, cam_ned) of the rectified camera in the prior's start frame at the image stamp, or None"""
            if self.R_bc is None:
                try:
                    tf = self.tf_buffer.lookup_transform(self.net_base_frame, self.rect_frame, rclpy.time.Time())
                except Exception:  # noqa: BLE001 - TF not up yet
                    return None
                r, t = tf.transform.rotation, tf.transform.translation
                self.R_bc = quat_xyzw_to_R(r.x, r.y, r.z, r.w); self.t_bc = (t.x, t.y, t.z)
                self.get_logger().info(f'net_model: {self.net_base_frame} <- {self.rect_frame} t={np.round(self.t_bc, 3).tolist()}')
            return self.odom_pose.camera(stamp[0] + stamp[1] * 1e-9, self.R_bc, self.t_bc)

        def process(self, stamp, left, right):
            pose = self.model_pose(stamp) if (self.net_model_on or self.temporal) else None
            depth, lrect, tim = self.pipe.run(left, right, pose, stamp[0] + stamp[1] * 1e-9)
            if self.net_model_on:
                self.model_ok += int(pose is not None and bool(self.pipe.model_info.get('model')))
            t2 = time.perf_counter()
            h, w = depth.shape
            dm = Image(); dm.header.stamp.sec, dm.header.stamp.nanosec = stamp; dm.header.frame_id = self.rect_frame
            dm.height, dm.width, dm.encoding, dm.is_bigendian, dm.step = h, w, '32FC1', 0, w * 4
            dm.data = image_msg_data(depth)
            lm = Image(); lm.header = dm.header
            lm.height, lm.width, lm.encoding, lm.is_bigendian, lm.step = h, w, 'mono8', 0, w
            lm.data = image_msg_data(lrect)
            ci = self.cinfo; ci.header.stamp = dm.header.stamp
            self.pub_left.publish(lm); self.pub_linfo.publish(ci)
            self.pub_depth.publish(dm); self.pub_dinfo.publish(ci)
            if self.pub_scan is not None and self.pipe.scan is not None:
                rng, inten = self.pipe.scan
                sm = self.scan_msg; sm.header.stamp = dm.header.stamp
                sm.ranges = array.array('f', rng.tobytes()); sm.intensities = array.array('f', inten.tobytes())
                self.pub_scan.publish(sm)
                self.stats['scan_hit'] += float(np.count_nonzero(np.isfinite(rng))) / rng.size
                self.stats['scan_free'] += float(np.count_nonzero(np.isposinf(rng))) / rng.size
            t3 = time.perf_counter()
            s = self.stats; s['n'] += 1
            s['t_rect'] += tim['rect']; s['t_match'] += tim['match']; s['t_up'] += tim['up']; s['t_pub'] += t3 - t2
            s['t_mask'] += tim['mask']; s['mask_frac'] += self.pipe.mask_frac; s['t_scan'] += tim['scan']
            s['valid'] += float(np.count_nonzero(depth)) / depth.size
            c = depth[h // 3:2 * h // 3, w // 3:2 * w // 3]; c = c[c > 0]
            s['center_med'] = float(np.median(c)) if c.size > 50 else None
            s['lat'] += self.get_clock().now().nanoseconds * 1e-9 - (stamp[0] + stamp[1] * 1e-9)

        def report(self):
            s = self.stats; now = time.monotonic(); dt = now - self.stats_t
            cpu = time.process_time() - self.cpu0
            n = max(s['n'], 1)
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            self.get_logger().info(json.dumps(dict(
                hz=round(s['n'] / dt, 2), rx_hz=round(s['rx'] / dt, 1), cpu_pct=round(100 * cpu / dt, 1),
                ms_rect=round(1e3 * s['t_rect'] / n, 1), ms_match=round(1e3 * s['t_match'] / n, 1),
                ms_up=round(1e3 * s['t_up'] / n, 1), ms_pub=round(1e3 * s['t_pub'] / n, 1),
                ms_mask=round(1e3 * s['t_mask'] / n, 1), net_mask_frac=round(s['mask_frac'] / n, 4),
                ms_scan=round(1e3 * s['t_scan'] / n, 2) if self.depth_scan else None,
                scan_hit=round(s['scan_hit'] / n, 3) if self.depth_scan else None,
                scan_free=round(s['scan_free'] / n, 3) if self.depth_scan else None,
                net_model_frames=self.model_ok if self.net_model_on else None,
                net_model=self.pipe.model_info if (self.net_model_on and self.pipe is not None) else None,
                lat_ms=round(1e3 * s['lat'] / n, 1) if s['n'] else None,
                valid=round(s['valid'] / n, 3), center_med_m=None if s.get('center_med') is None else round(s['center_med'], 3),
                raw_fallback=s['raw_fallback'], maxrss_mb=round(rusage.ru_maxrss / 1024, 1), ready=self.pipe is not None,
                preset=self.preset)))
            self.stats = self._zero_stats(); self.stats_t = now; self.cpu0 = time.process_time()

    # Shutdown: own SIGINT/SIGTERM handling (not rclpy's) so the node always exits promptly: stop the worker, spin out,
    # destroy, and a hard os._exit deadline 1.5 s after the first signal (a second signal exits at once).
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        rclpy.init()
    stopping = threading.Event()

    def on_signal(signum, _frame):
        if stopping.is_set():
            os._exit(128 + signum)
        stopping.set()
        t = threading.Timer(1.5, os._exit, (0,)); t.daemon = True; t.start()
    signal.signal(signal.SIGINT, on_signal); signal.signal(signal.SIGTERM, on_signal)
    n = StereoDepthNode()
    ex = SingleThreadedExecutor(); ex.add_node(n)
    try:
        while not stopping.is_set() and rclpy.ok():
            ex.spin_once(timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        n.stop.set(); n.evt.set()
        n.worker_thread.join(timeout=0.8)
        for f in (lambda: n.get_logger().info('team13 stereo depth: shutdown'), n.destroy_node,
                  lambda: rclpy.ok() and rclpy.shutdown(), sys.stdout.flush, sys.stderr.flush):
            try:
                f()
            except Exception:  # noqa: BLE001 - never block the exit
                pass
        os._exit(0)


if __name__ == '__main__':
    main()
