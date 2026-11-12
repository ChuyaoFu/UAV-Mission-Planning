"""OAK-D-S2-like stereo depth for the venue replica: sensor model + SGBM.

All parameters are in SENSOR / SGBM / POST below and are written into index.json.
This is an approximation of the device pipeline, not a calibrated model:
  * the device uses a census-transform SGM in hardware; OpenCV StereoSGBM uses a
    Birchfield-Tomasi cost on x-Sobel pre-filtered images (also insensitive to
    small per-eye gain differences, but not identical);
  * the device "confidence threshold" is cost-based; here it is emulated by SGBM's
    uniqueness ratio + a local-texture threshold measured on the (noisy) left image.
"""
import zlib
import numpy as np
import cv2

BASELINE_M = 0.075

SENSOR = dict(
    gain_log_sigma=0.08, gain_clip=(0.85, 1.18),   # per-frame auto-exposure/gain wobble (shared by both eyes)
    eye_gain_sigma=0.015,                         # per-eye gain mismatch
    black_level=2.0,                              # DN
    read_noise=1.2, shot_coeff=0.06,              # noise sigma = sqrt(read^2 + shot_coeff * I) DN
    blur_sigma=0.4,                               # optics, px
)

SGBM = dict(
    minDisparity=0, numDisparities=96, blockSize=5,
    P1=8 * 5 * 5, P2=32 * 5 * 5,
    disp12MaxDiff=1,                              # left-right consistency check (px)
    preFilterCap=31, uniquenessRatio=10,
    speckleWindowSize=100, speckleRange=2,
    mode='MODE_SGBM',
)

POST = dict(
    subpixel_bits=3,                              # OAK-D subpixel: 1/8 px
    texture_win=7, texture_blur_sigma=1.0,        # local std of the blurred left image ...
    texture_min_dn=1.2,                           # ... below this -> no return (low-texture holes); tuned so
                                                  #     plain boards keep ~40 % fill, see REPORT section 9
    median_ksize=5,                               # device median filter 5x5 (on disparity)
    min_disp_px=1.0,                              # -> max depth = fx*B/1 ~ 31.6 m
)


def frame_seed(frame_id):
    return zlib.crc32(frame_id.encode())


def sensor_model(gray_l, gray_r, seed):
    """Apply gain wobble, per-eye gain mismatch, blur and read+shot noise -> uint8 pair."""
    rng = np.random.default_rng(seed)
    p = SENSOR
    g = float(np.clip(np.exp(rng.normal(0, p['gain_log_sigma'])), *p['gain_clip']))
    out = []
    for img in (gray_l, gray_r):
        e = 1.0 + rng.normal(0, p['eye_gain_sigma'])
        I = img.astype(np.float32) * g * e + p['black_level']
        if p['blur_sigma'] > 0:
            I = cv2.GaussianBlur(I, (0, 0), p['blur_sigma'])
        sig = np.sqrt(p['read_noise'] ** 2 + p['shot_coeff'] * np.clip(I, 0, None))
        I = I + rng.normal(0, 1, I.shape).astype(np.float32) * sig
        out.append(np.clip(np.round(I), 0, 255).astype(np.uint8))
    return out[0], out[1], g


def _matcher():
    q = dict(SGBM)
    mode = getattr(cv2, 'STEREO_SGBM_' + q.pop('mode'))
    return cv2.StereoSGBM_create(mode=mode, **q)


def sgbm_depth(left, right, fx, baseline=BASELINE_M):
    """uint8 left/right (rectified, same K) -> (depth float32 m aligned to left, NaN = no return,
    disparity float32 px, valid mask)."""
    d16 = _matcher().compute(left, right)                       # int16, 4 fractional bits, <0 invalid
    valid = d16 > 0
    disp = d16.astype(np.float32) / 16.0
    q = 2 ** POST['subpixel_bits']
    disp = np.round(disp * q) / q
    disp[~valid] = 0.0
    # confidence: low local texture -> no return
    b = cv2.GaussianBlur(left.astype(np.float32), (0, 0), POST['texture_blur_sigma'])
    w = (POST['texture_win'], POST['texture_win'])
    std = np.sqrt(np.maximum(cv2.blur(b * b, w) - cv2.blur(b, w) ** 2, 0))
    valid &= std >= POST['texture_min_dn']
    disp[~valid] = 0.0
    # device median filter on the disparity map (holes stay holes)
    disp = cv2.medianBlur(disp, POST['median_ksize'])
    valid &= disp >= POST['min_disp_px']
    depth = np.full(disp.shape, np.nan, np.float32)
    depth[valid] = fx * baseline / disp[valid]
    return depth, disp, valid


def params():
    return {'baseline_m': BASELINE_M, 'sensor': SENSOR, 'sgbm': SGBM, 'post': POST,
            'note': 'OpenCV StereoSGBM approximation of the OAK-D S2 census-SGM pipeline; see stereo_depth.py'}
