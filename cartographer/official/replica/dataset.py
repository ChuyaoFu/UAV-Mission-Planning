"""Load the venue-replica test frames (outputs/venue-replica/) as common.geometry.Frame.

    from replica.dataset import load_frame, iter_frames
    frame, labels = load_frame('outputs/venue-replica/frames/tunnel_d100_l+000_h045.npz')
    for frame, labels, meta in iter_frames('outputs/venue-replica', group='window'): ...

Frame.depth is the OAK-D-like STEREO depth (SGBM on the noisy left/right pair, holes = NaN);
Frame.image is the noisy left mono image it was computed from. Ground truth:
depth_key='depth_ideal' (no cage net / tunnel mesh) or 'depth_full' (every surface).
`labels` is ground truth for OFFLINE evaluation only (estimated venue geometry):
labels['world']['tunnel']['entry_ned'], labels['world']['window']['center_ned'], ...,
labels['frame'][...] per-frame projections/visibility. Never import this from flight code.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.geometry import Frame   # noqa: E402


def load_frame(path, depth_key='depth'):
    d = np.load(path, allow_pickle=False)
    fr = Frame(image=d['image'], depth=d[depth_key].astype(np.float32), K=tuple(float(x) for x in d['K']),
               R_nc=d['R_nc'], cam_ned=d['cam_ned'], stamp=float(d['stamp']), body_ned=d['body_ned'],
               heading=float(d['heading']))
    return fr, json.loads(str(d['labels']))


def stereo_like(depth, image, fx=421.0, baseline=0.075, seed=0, sub_px=1 / 8, noise_px=0.08,
                tex_thr=1.5, min_z=0.30, max_z=12.0):
    """(Superseded by the stereo `depth` field / stereo_depth.py; kept for quick experiments.)
    Rough OAK-D-style degradation of the ideal replica depth (an approximation, not a
    calibrated model): disparity quantisation (sub_px), disparity noise (noise_px, i.e.
    sigma_z ~ z^2 * noise_px / (fx * B)), holes where the 7x7 grey-level std < tex_thr
    (low texture: white stone top, plain boards), and no depth below min_z / above max_z."""
    import cv2
    rng = np.random.default_rng(seed)
    z = depth.astype(np.float32).copy()
    ok = np.isfinite(z) & (z > 0)
    disp = np.where(ok, fx * baseline / np.where(ok, z, 1), 0)
    disp = np.round((disp + rng.normal(0, noise_px, z.shape)) / sub_px) * sub_px
    out = np.where(ok & (disp > 0), fx * baseline / np.maximum(disp, 1e-6), np.nan).astype(np.float32)
    g = image.astype(np.float32)
    std = np.sqrt(np.maximum(cv2.blur(g * g, (7, 7)) - cv2.blur(g, (7, 7)) ** 2, 0))
    out[(std < tex_thr) | (out < min_z) | (out > max_z)] = np.nan
    return out


def iter_frames(root, group=None, depth_key='depth'):
    idx = json.load(open(os.path.join(root, 'index.json')))
    for f in idx['frames']:
        if group is None or f['group'] == group:
            fr, lab = load_frame(os.path.join(root, f['file']), depth_key)
            yield fr, lab, f
