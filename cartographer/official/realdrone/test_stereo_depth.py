#!/usr/bin/env python3
"""Offline tests of the real-drone stereo depth pipeline (numpy + cv2; the CDR test also needs rclpy + sensor_msgs).
Usage: python3 test_stereo_depth.py [pair.npz]"""
import array
import math
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stereo_depth_node import (OPTIONS, DEFAULT_PRESET, DepthPipeline, RectifyModel, image_msg_data,  # noqa: E402
                               parse_image_cdr, resolve_options, scale_K, to_gray)
from net_mask import NetMask  # noqa: E402
from net_model import NetModel, OdomPose, quat_xyzw_to_R  # noqa: E402

FENCE = dict(forward_min=-3.96, forward_max=2.13, left_min=-5.57, left_max=0.53, up_max=3.0)
R_BASE_OPT = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])      # base_link FLU <- optical (x right, y down, z fwd)

K = np.array([[661.0, 0, 483.0], [0, 661.0, 305.0], [0, 0, 1.0]])
KR = np.array([[661.0, 0, 510.0], [0, 661.0, 316.0], [0, 0, 1.0]])
YAW = math.radians(0.33)
R_LR = np.array([[math.cos(YAW), 0, math.sin(YAW)], [0, 1, 0], [-math.sin(YAW), 0, math.cos(YAW)]])
T_LR = np.array([0.07529, -0.00053, 0.0002])


def synthetic_pair(z=2.0, seed=0):
    """A textured fronto-parallel plane at depth z seen by the two (unrectified, different-K) source cameras."""
    rng = np.random.default_rng(seed)
    tex = cv2.resize(rng.integers(0, 255, (80, 128)).astype(np.uint8), (2048, 1280), interpolation=cv2.INTER_CUBIC)
    tex = cv2.GaussianBlur(tex, (0, 0), 1.2)
    span = 2.0 * z  # plane size in metres covered by the texture (centred)

    def render(Kc, R_wc, t_wc):
        u, v = np.meshgrid(np.arange(1024, dtype=np.float64), np.arange(640, dtype=np.float64))
        rays = np.stack([(u - Kc[0, 2]) / Kc[0, 0], (v - Kc[1, 2]) / Kc[1, 1], np.ones_like(u)], -1) @ R_wc.T
        s = (z - t_wc[2]) / rays[..., 2]
        X = t_wc[0] + s * rays[..., 0]; Y = t_wc[1] + s * rays[..., 1]
        mx = ((X / span + 0.5) * 2047).astype(np.float32); my = ((Y / span + 0.5) * 1279).astype(np.float32)
        return cv2.remap(tex, mx, my, cv2.INTER_LINEAR)
    return render(K, np.eye(3), np.zeros(3)), render(KR, R_LR, T_LR)


def test_presets_resolve():
    for name in OPTIONS:
        o = resolve_options(name)
        assert o['num_disp'] % 16 == 0
    assert DEFAULT_PRESET in OPTIONS
    o = resolve_options('half320', width=640)
    assert o['width'] == 640 and o['num_disp'] == 96, o           # keeps the ~0.33 m minimum depth
    o = resolve_options('ref640', width=512)
    assert o['num_disp'] == 80, o
    o = resolve_options('half320', band=(0.2, 0.8), matcher='BM', median=0)
    assert o['band'] == (0.2, 0.8) and o['matcher'] == 'BM' and o['median'] == 0
    try:
        resolve_options('nope'); raise AssertionError('unknown preset accepted')
    except ValueError:
        pass


def test_geometry_and_depth():
    """Each preset: output 640x400, K_out = the 640 rectified K (same camera as before), and a plane at 2 m reads 2 m."""
    L, R = synthetic_pair(2.0)
    ref = RectifyModel(K, np.zeros(5), KR, np.zeros(5), (1024, 640), R_LR, T_LR, width=640)
    for name in OPTIONS:
        p = DepthPipeline(OPTIONS[name], K, np.zeros(5), KR, np.zeros(5), (1024, 640), R_LR, T_LR)
        d, left, tim = p.run(L, R)
        assert d.shape == (400, 640) and d.dtype == np.float32 and left.shape == (400, 640) and left.dtype == np.uint8, name
        assert np.allclose(p.K_out[:2], ref.K()[:2], rtol=0.01), (name, p.K_out, ref.K())
        assert abs(p.K_out[2] - ref.cx) < 1.5 and abs(p.K_out[3] - ref.cy) < 1.5, (name, p.K_out, ref.K())
        v = d[d > 0]
        assert v.size > 0.2 * d.size * (0.6 if 'band' in name else 1.0), (name, v.size / d.size)
        med = float(np.median(v))
        assert abs(med - 2.0) < 0.08, (name, med)
        if 'band' in name:
            r0, r1 = int(400 * 0.2), int(400 * 0.8)
            assert not np.any(d[:r0 - 2] > 0) and not np.any(d[r1 + 2:] > 0), name
        assert set(tim) == {'rect', 'match', 'mask', 'up', 'scan'} and tim['mask'] == 0.0 and tim['scan'] == 0.0
        assert p.scan is None


def test_left_and_depth_aligned():
    """The published left image and the depth share the 640x400 rectified geometry (half320 vs ref640 left images)."""
    L, R = synthetic_pair(2.0, seed=3)
    a = DepthPipeline(OPTIONS['ref640'], K, np.zeros(5), KR, np.zeros(5), (1024, 640), R_LR, T_LR).run(L, R)[1]
    b = DepthPipeline(OPTIONS['half320'], K, np.zeros(5), KR, np.zeros(5), (1024, 640), R_LR, T_LR).run(L, R)[1]
    sh, _ = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32))
    assert abs(sh[0]) < 0.6 and abs(sh[1]) < 0.6, sh


def net_scene(seed=0, period=8, H=200, W=320):
    """320x200 rectified-left-like image: left half a net (thin dark vertical cords every `period` px over a smooth,
    weakly textured background), right half a random-texture board."""
    rng = np.random.default_rng(seed)
    bg = cv2.GaussianBlur(rng.normal(120, 25, (H, W)).astype(np.float32), (0, 0), 6) + rng.normal(0, 1.2, (H, W))
    img = bg.copy()
    x = np.arange(W)
    cords = (np.abs(((x + 0.3) % period) - period / 2) > period / 2 - 0.8)          # ~1.5 px cords
    img[:, :W // 2][:, cords[:W // 2]] -= 60
    board = cv2.GaussianBlur(rng.normal(128, 50, (H, W)).astype(np.float32), (0, 0), 1.0)
    img[:, W // 2:] = board[:, W // 2:]
    return np.clip(img, 0, 255).astype(np.uint8)


def test_net_mask_periodicity():
    img = net_scene()
    m = NetMask()
    sc, per = m.periodicity(img)
    net, board = (slice(40, 160), slice(40, 120)), (slice(40, 160), slice(200, 280))
    assert np.median(sc[net]) > 0.5, np.median(sc[net])
    assert abs(np.median(per[net]) - 8) <= 1, np.median(per[net])
    # random texture: the score's noise floor (median ~0.2); the D and cost checks keep such boards (see below)
    assert np.median(sc[board]) < 0.25 and np.percentile(sc[board], 90) < 0.4, np.percentile(sc[board], [50, 90])


def layered_right(img, base_disp, near):
    """right image of a scene at `base_disp` everywhere plus nearer rectangles (r0, r1, c0, c1, disp): R(x - d) = L(x)"""
    R = np.zeros_like(img); W = img.shape[1]
    R[:, :W - base_disp] = img[:, base_disp:]
    rng = np.random.default_rng(99)
    for r0, r1, c0, c1, dd in near:
        # the background hidden behind the near rectangle in the left view: an independent texture (not the rect's own)
        R[r0:r1, c0 - base_disp:c1 - base_disp] = cv2.GaussianBlur(
            rng.normal(128, 40, (r1 - r0, c1 - c0)).astype(np.float32), (0, 0), 1.0).clip(0, 255).astype(np.uint8)
        R[r0:r1, c0 - dd:c1 - dd] = img[r0:r1, c0:c1]
    return R


def test_net_mask_aliases_only():
    """net and board at disparity 10; an alias patch on the net reported one period (8 px) nearer: removed. A real near
    box on the textured board and the board's edge strip next to the net (both at 18) and the correctly matched net
    stay. Without the right image (no cost check) the edge strip next to the net is the weak case."""
    for seed in (1, 5):
        img = net_scene(seed)
        near = [(80, 100, 220, 260, 18), (0, 200, 160, 176, 18)]
        right = layered_right(img, 10, near)
        d = np.full(img.shape, 10.0, np.float32)
        for r0, r1, c0, c1, dd in near:
            d[r0:r1, c0:c1] = dd
        d[80:100, 50:90] = 18.0                     # alias patch on the net
        bad = NetMask()(img, d, right)
        assert bad[84:96, 56:84].mean() > 0.9, bad[84:96, 56:84].mean()
        assert bad[:, :160][d[:, :160] == 10].mean() < 0.01
        assert bad[80:100, 220:260].mean() < 0.02, bad[80:100, 220:260].mean()
        assert bad[:, 160:176].mean() < 0.12, bad[:, 160:176].mean()
        d0 = d.copy(); d0[85:95, 60:80] = 0
        assert not NetMask()(img, d0, right)[85:95, 60:80].any()        # invalid stays invalid (not "removed")


def test_net_mask_full_res_and_pipeline():
    img = cv2.resize(net_scene(2), (640, 400), interpolation=cv2.INTER_LINEAR)
    fb = 421.0 * 0.075                                         # net at disparity 16 px (640), alias patch at 32
    z = np.full(img.shape, fb / 16, np.float32); z[160:200, 100:180] = fb / 32
    right = cv2.resize(layered_right(net_scene(2), 8, []), (640, 400), interpolation=cv2.INTER_LINEAR)
    out, bad = NetMask().apply_depth(img, z, 421.0, 0.075, right)
    assert out.shape == z.shape and bad.shape == z.shape and bad[170:190, 110:170].mean() > 0.9
    assert np.all(out[bad] == 0) and np.all(out[~bad] == z[~bad])
    L, R = synthetic_pair(2.0, seed=4)                       # random texture: the mask must cost ~nothing
    p0 = DepthPipeline(OPTIONS['half320'], K, np.zeros(5), KR, np.zeros(5), (1024, 640), R_LR, T_LR)
    p1 = DepthPipeline(OPTIONS['half320'], K, np.zeros(5), KR, np.zeros(5), (1024, 640), R_LR, T_LR, net_mask=NetMask())
    d0, _, _ = p0.run(L, R); d1, _, tim = p1.run(L, R)
    assert tim['mask'] > 0 and np.all((d1 > 0) <= (d0 > 0))
    assert (np.count_nonzero(d0) - np.count_nonzero(d1)) < 0.01 * np.count_nonzero(d0), p1.mask_frac
    assert p1.describe()['net_mask']['score_min'] == NetMask().score_min


def cam_pose(f=0.3, l=-0.2, up=1.0, yaw_deg=10.0, pitch_deg=5.0):
    """optical camera -> start-frame NED for a body at (f, l, up) yawed left by yaw, pitched down by pitch"""
    y, pt = math.radians(yaw_deg), math.radians(pitch_deg)
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1.0]])
    Ry = np.array([[math.cos(pt), 0, math.sin(pt)], [0, 1.0, 0], [-math.sin(pt), 0, math.cos(pt)]])
    R_sb = Rz @ Ry
    F = np.diag([1.0, -1.0, -1.0])
    return F @ R_sb @ R_BASE_OPT, F @ np.array([f, l, up])


def test_net_model_geometry():
    """model depth back-projects onto the fence box faces; floor pixels carry no net; refinement recovers a shifted face"""
    K4 = (421.0, 421.0, 319.2, 200.7); fb = 421.0 * 0.075
    m = NetModel(FENCE)
    R_nc, cam = cam_pose()
    d_net, fid, planes = m.model_disparity((400, 640), K4, fb, R_nc, cam)
    v, u = np.mgrid[0:400, 0:640]
    ok = fid >= 0
    assert 0.3 < ok.mean() < 1.0
    Z = np.where(ok, fb / np.maximum(d_net, 1e-6), 0)
    P = np.stack([(u - K4[2]) / K4[0] * Z, (v - K4[3]) / K4[1] * Z, Z], -1) @ R_nc.T + cam
    f_, l_, up_ = P[..., 0], -P[..., 1], -P[..., 2]
    dist = np.minimum.reduce([np.abs(f_ - FENCE['forward_min']), np.abs(f_ - FENCE['forward_max']),
                              np.abs(l_ - FENCE['left_min']), np.abs(l_ - FENCE['left_max']), np.abs(up_ - FENCE['up_max'])])
    assert dist[ok].max() < 1e-6, dist[ok].max()
    assert np.all(up_[ok] > -1e-6)                                   # nothing under the floor is called net
    # measured disparities of the forward face 0.2 m farther than the prior -> the fit moves the face by ~+0.2 m
    shifted = dict(FENCE, forward_max=FENCE['forward_max'] + 0.2)
    d_true, fid_t, _ = NetModel(shifted).model_disparity((400, 640), K4, fb, R_nc, cam)
    rng = np.random.default_rng(0)
    meas = np.where(fid_t >= 0, d_true + rng.normal(0, 0.15, d_true.shape), 0).astype(np.float32)
    scale, sig = m.refine(meas, fid, planes)
    i = 1                                                            # forward_max
    h = planes[i][3]
    assert abs(h * (1 / scale[i] - 1) - 0.2) < 0.03, (h, scale[i])
    assert sig[i] is not None and sig[i] < 0.3


def test_odom_pose():
    """anchor 'first': a vehicle started yawed in an arbitrary odom frame reproduces the start-frame pose"""
    R_bc, t_bc = R_BASE_OPT, np.array([0.116, 0.037, 0.029])
    yaw0 = math.radians(37.0); p0 = np.array([1.5, -2.0, 0.3])
    q0 = (0.0, 0.0, math.sin(yaw0 / 2), math.cos(yaw0 / 2))
    op = OdomPose(anchor='first', base_height_m=0.05)
    op.add(10.0, p0, q0)
    # 1 m forward and 0.5 m up in the start frame, same heading
    c, s_ = math.cos(yaw0), math.sin(yaw0)
    op.add(11.0, p0 + np.array([c, s_, 0.5]), q0)
    R_nc, cam = op.camera(11.02, R_bc, t_bc)
    F = np.diag([1.0, -1.0, -1.0])
    assert np.allclose(cam, F @ (np.array([1.0, 0, 0.55]) + t_bc), atol=1e-9), cam
    assert np.allclose(R_nc, F @ R_bc, atol=1e-9)
    assert op.camera(12.0, R_bc, t_bc) is None                     # no odometry near that stamp
    o2 = OdomPose(anchor='origin', base_height_m=0.05); o2.add(0.0, (0.0, 0, 0), (0, 0, 0, 1.0))
    assert np.allclose(o2.camera(0.0, R_bc, t_bc)[1], F @ (np.array([0, 0, 0.05]) + t_bc))
    assert np.allclose(quat_xyzw_to_R(0, 0, 0, 1), np.eye(3))


def test_net_model_decision():
    """model mode on the synthetic net: alias removed, the net at the model disparity and a real near box kept"""
    img = net_scene(3)
    near = [(80, 100, 220, 260, 18), (0, 200, 160, 176, 18)]
    right = layered_right(img, 10, near)
    d = np.full(img.shape, 10.0, np.float32)
    for r0, r1, c0, c1, dd in near:
        d[r0:r1, c0:c1] = dd
    d[80:100, 50:90] = 18.0
    m = NetModel(FENCE)
    bad = m.mask(img, d, right, model_disp=np.full(img.shape, 10.0, np.float32), model_tol=1.5)
    assert bad[84:96, 56:84].mean() > 0.9
    assert bad[d == 10].mean() < 0.01 and bad[80:100, 220:260].mean() < 0.02 and bad[:, 160:176].mean() < 0.12
    # a model saying the net is NEARER than the alias patch (wrong fence): no candidates there -> nothing removed
    assert not m.mask(img, d, right, model_disp=np.full(img.shape, 20.0, np.float32))[84:96, 56:84].any()


def test_depth_filters():
    """min_support drops a small isolated blob but keeps a wall; TemporalCheck drops a one-frame speck in front of a wall
    seen in the previous frame, keeps the wall, and keeps everything on the first frame"""
    from depth_filters import min_support, TemporalCheck
    disp = np.zeros((200, 320), np.float32)
    disp[50:150, 20:120] = 12.0 + 0.01 * np.arange(100)[None, :]          # wall (smoothly varying)
    disp[60:66, 200:206] = 25.0                                            # 36-px floating blob
    keep = min_support(disp, 100, 1.0)
    assert keep[50:150, 20:120].all() and not keep[60:66, 200:206].any()
    K4 = (210.0, 210.0, 159.5, 99.5)
    tc = TemporalCheck(K4, tol=0.05, nprev=1)
    R = np.eye(3); c = np.zeros(3)
    z = np.full((200, 320), 2.0, np.float32)
    assert tc(z, R, c, 0.0).all()                                          # first frame: no history
    z2 = z.copy(); z2[100:104, 150:154] = 0.8                              # speck in front of the wall
    k2 = tc(z2, R, c, 0.066)
    assert not k2[100:104, 150:154].any() and k2[:90].all()
    assert tc(z, R, c, 1.0).all()                                          # history too old -> keep


def test_param_types():
    """integer-looking / float-looking / string overrides must not crash a typed declaration (compat 2026-11-08:
    `-p noise_minsup_diff_px:=1` for a double default -> InvalidParameterTypeException, no depth)"""
    from depth_filters import coerce, declare_typed
    assert coerce(1, 1.0) == 1.0 and isinstance(coerce(1, 1.0), float)
    assert coerce(150.0, 0) == 150 and isinstance(coerce('150', 0), int)
    assert coerce('true', False) is True and coerce(0, True) is False and coerce('False', True) is False
    assert coerce(5, '') == '5'
    try:
        import rclpy
        from rclpy.node import Node
    except ImportError:
        print('  (ROS part skipped: no rclpy)'); return
    ov = ['--ros-args', '-p', 'noise_minsup_diff_px:=1', '-p', 'noise_minsup_px:=150.0', '-p', 'sgbm_speckle_window:=800',
          '-p', 'sgbm_speckle_range:=2.0', '-p', 'rate_hz:=10', '-p', 'net_mask:=true', '-p', 'noise_temporal_tol:=0',
          '-p', 'net_fence_json:=mission/course_prior.example.json']
    ctx = rclpy.Context(); rclpy.init(args=ov, context=ctx)
    try:
        n = Node('team13_param_test', context=ctx)
        got = {k: declare_typed(n, k, d) for k, d in (('noise_minsup_diff_px', 1.0), ('noise_minsup_px', 0), ('sgbm_speckle_window', 0),
                                                     ('sgbm_speckle_range', 0), ('rate_hz', 10.0), ('net_mask', False),
                                                     ('noise_temporal_tol', 0.0), ('net_fence_json', ''), ('untouched', 0.25))}
        assert got == dict(noise_minsup_diff_px=1.0, noise_minsup_px=150, sgbm_speckle_window=800, sgbm_speckle_range=2,
                           rate_hz=10.0, net_mask=True, noise_temporal_tol=0.0, net_fence_json='mission/course_prior.example.json',
                           untouched=0.25), got
        assert isinstance(got['noise_minsup_diff_px'], float) and isinstance(got['noise_minsup_px'], int)
        n.destroy_node()
    finally:
        rclpy.shutdown(context=ctx)


def render_flu(Kc, size, plates, floor_h=None):
    """depth image (optical z, 0 = none) of axis-aligned plates x = X, y in [y0, y1] (left), h in [h0, h1] (up), in the
    camera's x forward / y left / z up frame; optional floor at height floor_h"""
    fx, fy, cx, cy = Kc; W, H = size
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    ty, th = -(u - cx) / fx, -(v - cy) / fy                     # left and up per metre forward
    z = np.full((H, W), np.inf)
    for X, y0, y1, h0, h1 in plates:
        hit = (ty * X >= y0) & (ty * X <= y1) & (th * X >= h0) & (th * X <= h1)
        z = np.where(hit, np.minimum(z, X), z)
    if floor_h is not None:
        with np.errstate(divide='ignore'):
            zf = np.where(th < 0, floor_h / th, np.inf)
        z = np.minimum(z, zf)
    return np.where(np.isfinite(z), z, 0).astype(np.float32)


def test_depth_scan():
    """band, nearest return, angles (LaserScan convention), +inf / NaN semantics, support, cost; pipeline output"""
    from depth_scan import DepthScan, scan_points, R_OPTICAL_SCAN, Q_OPTICAL_SCAN_XYZW
    from stereo_depth_node import quat_xyzw_to_mat
    assert np.allclose(quat_xyzw_to_mat(*Q_OPTICAL_SCAN_XYZW), R_OPTICAL_SCAN)
    assert np.allclose(R_OPTICAL_SCAN @ [1, 0, 0], [0, 0, 1]) and np.allclose(R_OPTICAL_SCAN @ [0, 1, 0], [-1, 0, 0])
    Kc, size = (210.5, 210.5, 159.35, 100.1), (320, 200)      # the real node's half320 working camera
    plates = [(1.5, 0.3, 0.6, -0.1, 0.1),                      # obstacle left, in band
              (1.0, -5, 5, 0.38, 0.45),                        # overhead beam just above the band (in view): ignored
              (2.0, -0.6, -0.3, 0.25, 0.30),                   # top of the band, right
              (1.2, -0.6, -0.3, -0.40, -0.30),                 # below the band, right: ignored
              (3.0, -9, 9, -9, 9)]                             # back wall
    z = render_flu(Kc, size, plates, floor_h=-1.0)
    z[:, :10] = 0.0                                            # texture-less strip at the image's left edge
    sc = DepthScan(Kc, size, h_min=-0.25, h_max=0.35, range_max=2.5, range_min=0.33, px_scale=2.0)
    rng, inten = sc(z)
    m = sc.meta(); assert m['n'] == rng.size == inten.size and abs(m['angle_increment'] - 1 / 210.5) < 1e-12
    a = sc.angle_min + sc.angle_increment * np.arange(rng.size); t = np.tan(a)
    phi_col = -np.arctan((np.arange(320) - 159.35) / 210.5)
    nan_side = a > phi_col[10] + 1.5 * sc.angle_increment     # bins fed only by the invalid columns
    clear = ~nan_side & (a < phi_col[10] - 1.5 * sc.angle_increment)
    inner = lambda lo, hi: (t > lo + 0.01) & (t < hi - 0.01)
    assert np.all(np.isnan(rng[nan_side])), rng[nan_side]
    box_l, box_r = inner(0.3 / 1.5, 0.6 / 1.5), inner(-0.6 / 2.0, -0.3 / 2.0)
    assert np.allclose(rng[box_l], 1.5 / np.cos(a[box_l]), rtol=2e-3), rng[box_l]
    assert np.allclose(rng[box_r], 2.0 / np.cos(a[box_r]), rtol=2e-3), rng[box_r]
    other = clear & ~inner(0.3 / 1.5 - 0.03, 0.6 / 1.5 + 0.03) & ~inner(-0.6 / 2.0 - 0.03, -0.3 / 2.0 + 0.03)
    assert np.all(np.isposinf(rng[other])), rng[other]       # wall at 3 m > range_max: free to 2.5 m
    fin = np.isfinite(rng)
    assert np.all(inten[fin] > 0) and np.all(inten[~fin] == 0)
    # support on the left box: its face spans 0.2 m of height at 1.5 m = 28 rows x 2^2 output px per column
    assert 100 <= np.median(inten[box_l]) <= 2 * 120, np.median(inten[box_l])
    P = scan_points(rng, sc.angle_min, sc.angle_increment)
    assert np.all(P[:, 0] > 1.49) and np.all(P[:, 0] < 2.01)  # no beam (1.0 m) / low plate (1.2 m) returns
    left = P[P[:, 1] > 0]; right = P[P[:, 1] < 0]
    assert np.allclose(left[:, 0], 1.5, atol=0.01) and left[:, 1].min() > 0.29 and left[:, 1].max() < 0.61
    assert np.allclose(right[:, 0], 2.0, atol=0.012) and right[:, 1].min() > -0.61 and right[:, 1].max() < -0.29
    rng5, _ = DepthScan(Kc, size, h_min=-0.25, h_max=0.35, range_max=5.0)(z)   # wall within range now
    assert np.allclose(rng5[other], 3.0 / np.cos(a[other]), rtol=2e-3)
    # a band that excludes the camera height (e.g. only overhead): the beam appears, the obstacles do not
    rb, _ = DepthScan(Kc, size, h_min=0.37, h_max=0.6, range_max=5.0)(z)
    assert np.allclose(rb[clear], 1.0 / np.cos(a[clear]), rtol=2e-3)
    # cost at the compat resolution (640x400) and the real node's (320x200)
    import time as _t
    for Kx, sz in ((Kc, size), ((421.0, 421.0, 319.2, 200.7), (640, 400))):
        s2 = DepthScan(Kx, sz); zz = render_flu(Kx, sz, plates, floor_h=-1.0); s2(zz)
        t0 = _t.perf_counter()
        for _ in range(50):
            s2(zz)
        print(f'  depth scan {sz[0]}x{sz[1]}: {(_t.perf_counter() - t0) / 50 * 1e3:.2f} ms, {s2.n} bins')
    # in the pipeline: scan of the final depth at the working resolution, timed separately
    L, R = synthetic_pair(2.0, seed=6)
    pp = DepthPipeline(OPTIONS['half320'], K, np.zeros(5), KR, np.zeros(5), (1024, 640), R_LR, T_LR,
                       scan=dict(h_min=-0.25, h_max=0.35, range_max=5.0, support_tol=0.1))
    d, _, tim = pp.run(L, R)
    r2, i2 = pp.scan
    assert tim['scan'] > 0 and r2.size == pp.scanner.n and pp.scanner.size == pp.work_size
    assert abs(pp.scanner.range_min - pp.min_depth) < 1e-9
    c = r2[np.abs(np.arange(r2.size) - r2.size // 2) < r2.size // 4]
    c = c[np.isfinite(c)]; ac = np.abs(np.arange(r2.size) - r2.size // 2) < r2.size // 4
    assert c.size > 0.8 * ac.sum(), c.size
    aa = (pp.scanner.angle_min + pp.scanner.angle_increment * np.arange(r2.size))[ac][np.isfinite(r2[ac])]
    assert abs(float(np.median(c * np.cos(aa))) - 2.0) < 0.08, np.median(c * np.cos(aa))   # plane at 2 m forward


def test_image_msg_data_is_array():
    d = np.arange(12, dtype=np.float32).reshape(3, 4)
    a = image_msg_data(d)
    assert isinstance(a, array.array) and a.typecode == 'B' and a.tobytes() == d.tobytes()
    s = image_msg_data(np.zeros((4, 6), np.uint8)[:, ::2])   # non-contiguous input
    assert len(s) == 12


def test_to_gray():
    g = to_gray('mono8', 2, 3, 4, bytes(range(8)))
    assert g.shape == (2, 3) and g[1, 0] == 4
    c = to_gray('bgr8', 1, 2, 6, bytes([0, 0, 255, 255, 0, 0]))
    assert c.shape == (1, 2) and c[0, 0] > c[0, 1]
    assert to_gray('16UC1', 1, 1, 2, b'\0\0') is None


def test_cdr_parse_roundtrip():
    try:
        from rclpy.serialization import serialize_message
        from sensor_msgs.msg import Image
    except ImportError:
        print('  (skipped: no rclpy)'); return
    for fid, enc, w, h, step in (('oak_left_camera_optical_frame', 'mono8', 1024, 640, 1024), ('', 'mono8', 5, 3, 8),
                                 ('abc', 'bgr8', 4, 2, 12), ('x' * 7, '8UC1', 3, 1, 3)):
        m = Image(); m.header.stamp.sec, m.header.stamp.nanosec = 1794086123, 41601000; m.header.frame_id = fid
        m.height, m.width, m.encoding, m.step = h, w, enc, step
        m.data = array.array('B', (np.arange(h * step) % 251).astype(np.uint8).tobytes())
        b = serialize_message(m)
        sec, nsec, f, hh, ww, e, st, data = parse_image_cdr(b)
        assert (sec, nsec, f, hh, ww, e, st) == (1794086123, 41601000, fid, h, w, enc, step), (sec, nsec, f, hh, ww, e, st)
        assert bytes(data) == m.data.tobytes()
    try:
        parse_image_cdr(b[:-5]); raise AssertionError('truncated payload accepted')
    except ValueError:
        pass


def test_real_pair(path):
    import ast
    z = np.load(path, allow_pickle=True)
    info = ast.literal_eval(str(z['info'])); tf = ast.literal_eval(str(z['tf']))
    from stereo_depth_node import quat_xyzw_to_mat
    t = tf['oak_left_camera_optical_frame->oak_right_camera_optical_frame']
    il, ir = info['left'], info['right']
    for name in ('ref640', DEFAULT_PRESET):
        p = DepthPipeline(OPTIONS[name], np.array(il['K']).reshape(3, 3), np.zeros(5), np.array(ir['K']).reshape(3, 3),
                          np.zeros(5), (il['w'], il['h']), quat_xyzw_to_mat(*t[3:]), np.array(t[:3]))
        d = p.run(z['left'][0], z['right'][0])[0]
        print(f'  real pair {name}: valid {np.mean(d > 0):.3f}, median {np.median(d[d > 0]):.3f} m')


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and k != 'test_real_pair']
    for k, f in tests:
        f(); print('ok', k)
    if len(sys.argv) > 1:
        test_real_pair(sys.argv[1])
    print(f'{len(tests)}/{len(tests)} passed')
