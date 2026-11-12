#!/usr/bin/env python3
"""Sim-side OAK-D S2 'driver' for the official-compat profile.

gz (bridged to /gz/oak/*)  ->  official ROS interface:
  /stereo/left/camera_info, /stereo/right/camera_info   spec K, zero distortion, right P with Tx = -fx*B
  /camera/depth/image_raw  32FC1 metres aligned to oak_left, invalid (inf/NaN/out of [min,max]) -> 0.0
  /camera/depth/camera_info  same K as the left camera
  /tf_static  base_link->oak_left/oak_right/oak_rgb (optical, horizontal), oak_left->oak_imu
Stamps are copied from gz (sim time), so image/camera_info/depth pairs match exactly.

Modes (start_compat.sh --images / --depth; recorded in run/logs/stack.json and each flight's sim/):
  image_model:=gz    stereo images bridged straight from gz (gz camera noise sigma 0.006)  [default; all flights before 2026-11-07]
  image_model:=oak   gz cameras render CLEAN (gen_model --image-model oak); this node applies the venue-replica
                     OAK-D S2 sensor model (oak_stereo_model.py = verbatim copy of replica/stereo_depth.py: shared
                     per-frame gain wobble, per-eye gain, blur 0.4 px, read+shot noise; seed = CRC32 of the stamp)
                     and publishes the noisy pair as /stereo/{left,right}/image
  depth_mode:=ideal  /camera/depth/image_raw from the gz depth camera (net + tunnel mesh invisible)   [default]
                     (+ depth_noise_k: Gaussian sigma = k * d^2 m; depth_dropout: fraction of pixels set invalid, i.i.d.)
  depth_mode:=stereo /camera/depth/image_raw = replica SGBM on the noisy pair (96 disp, LR check 1 px,
                     1/8 px, texture gate 1.2 DN, 5x5 median), aligned to left, NaN -> 0 (needs image_model oak)
Option -p depth_noise_k:=K (ideal mode only) adds sigma = K*z^2 noise (default 0).
Option -p depth_scan:=true (start_compat.sh --depth-scan): /team13/depth_scan sensor_msgs/LaserScan from the published
depth (either mode, after the net / noise filters): per column the nearest return whose height above the camera is in
[depth_scan_h_min_m, depth_scan_h_max_m] (default -0.25 / 0.35), <= depth_scan_range_max_m (5); frame
team13_depth_scan = oak_left's origin with x forward / y left / z up (static TF oak_left -> team13_depth_scan); stamp
= the depth stamp. depth_scan.py = verbatim copy of realdrone/depth_scan.py (the real drone's stereo node).
"""
import array, hashlib, json, math, threading, time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from nav_msgs.msg import Odometry
from depth_filters import declare_typed        # any override type -> the default's type (no InvalidParameterType crash)
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

P = json.loads((Path(__file__).resolve().parent / 'profile.json').read_text())


def mat_to_quat_xyzw(R):
    R = np.asarray(R, float); t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2; w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return x, y, z, w


class OakSimDriver(Node):
    def __init__(self):
        super().__init__('oak_sim_driver')
        c, d = P['camera'], P['depth']
        self.noise_k = float(self.declare_parameter('depth_noise_k', 0.0).value)
        self.dropout = float(self.declare_parameter('depth_dropout', 0.0).value)   # ideal depth: i.i.d. pixel dropouts (holes, never phantoms)
        self.image_model = self.declare_parameter('image_model', 'gz').value
        self.depth_mode = self.declare_parameter('depth_mode', 'ideal').value
        # cage-net phantom mask (net_mask.py = verbatim copy of realdrone/net_mask.py, the real drone's stereo node)
        T = lambda n, v: declare_typed(self, n, v)
        self.net_mask = T('net_mask', False)
        # model-based variant (net_model.py, Barrois et al. 2010): the flight's prior fence + the VIO pose
        self.net_model = T('net_model', False)
        self.net_fence = T('net_fence_json', '')
        self.nm = None; self.nmod = None; self.op = None; self.nm_frac = []; self.nm_model_n = 0
        # free-space noise filters (depth_filters.py = verbatim copy of realdrone/depth_filters.py); all off by default
        self.spk = (T('sgbm_speckle_window', 0), T('sgbm_speckle_range', 0))
        self.tex_min = T('texture_min_dn', 0.0)
        self.minsup = (T('noise_minsup_px', 0), T('noise_minsup_diff_px', 1.0))
        self.ttol = T('noise_temporal_tol', 0.0)
        self.tfr = T('noise_temporal_frames', 2)
        self.treq = T('noise_temporal_require', False)
        self.tcheck = None; self.noise_frac = []
        # planar scan for the maze planner (depth_scan.py = verbatim copy of realdrone/depth_scan.py); off by default
        self.scan_on = T('depth_scan', False)
        self.scan_kw = dict(h_min=T('depth_scan_h_min_m', -0.25), h_max=T('depth_scan_h_max_m', 0.35),
                            range_max=T('depth_scan_range_max_m', 5.0), support_tol=T('depth_scan_support_tol', 0.10))
        self.scan_frame = T('depth_scan_frame', 'team13_depth_scan')
        self.scan_topic = T('depth_scan_topic', '/team13/depth_scan')
        self.scanner = None; self.scan_st = dict(n=0, ms=0.0, hit=0.0, t=time.monotonic())
        assert self.image_model in ('gz', 'oak') and self.depth_mode in ('ideal', 'stereo'), (self.image_model, self.depth_mode)
        assert not (self.depth_mode == 'stereo' and self.image_model != 'oak'), 'stereo depth needs image_model oak'
        self.rng = np.random.default_rng(1)
        self.W, self.H = c['width'], c['height']; self.K = c['K']; self.B = c['baseline_m']
        self.dmin, self.dmax = d['min_m'], d['max_m']
        out = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)
        self.pub_li = self.create_publisher(CameraInfo, '/stereo/left/camera_info', out)
        self.pub_ri = self.create_publisher(CameraInfo, '/stereo/right/camera_info', out)
        self.pub_d = self.create_publisher(Image, '/camera/depth/image_raw', out)
        self.pub_di = self.create_publisher(CameraInfo, '/camera/depth/camera_info', out)
        self.create_subscription(CameraInfo, '/gz/oak/left/camera_info', lambda m: self.on_info(m, 'oak_left', 0.0, self.pub_li), qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/gz/oak/right/camera_info', lambda m: self.on_info(m, 'oak_right', -self.K[0] * self.B, self.pub_ri), qos_profile_sensor_data)
        if self.depth_mode == 'ideal':
            self.create_subscription(Image, '/gz/oak/depth/image', self.on_depth, qos_profile_sensor_data)
        if self.image_model == 'oak':
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import oak_stereo_model as SM
            self.SM = SM
            sha = hashlib.sha256((Path(__file__).resolve().parent / 'oak_stereo_model.py').read_bytes()).hexdigest()[:16]
            self.get_logger().info(f'image_model=oak depth_mode={self.depth_mode} stereo model sha256[:16]={sha} '
                                   f'params={json.dumps(SM.params())}')
            if self.net_mask and self.depth_mode == 'stereo':
                import net_mask as NMM
                self.nm = NMM.NetMask()
                nsha = hashlib.sha256((Path(__file__).resolve().parent / 'net_mask.py').read_bytes()).hexdigest()[:16]
                self.get_logger().info(f'net_mask ON sha256[:16]={nsha} params={json.dumps(self.nm.params())}')
            if self.depth_mode == 'stereo' and (self.spk[0] > 0 or self.tex_min > 0 or self.minsup[0] > 0 or self.ttol > 0):
                import depth_filters as DF
                self.DF = DF
                if self.spk[0] > 0:
                    SM.SGBM['speckleWindowSize'], SM.SGBM['speckleRange'] = self.spk[0], max(1, self.spk[1])
                if self.tex_min > 0:
                    SM.POST['texture_min_dn'] = self.tex_min
                if self.ttol > 0:
                    self.tcheck = DF.TemporalCheck(tuple(self.K), tol=self.ttol, nprev=self.tfr, require=self.treq)
                dsha = hashlib.sha256((Path(__file__).resolve().parent / 'depth_filters.py').read_bytes()).hexdigest()[:16]
                self.get_logger().info(f'noise filters ON sha256[:16]={dsha} speckle {self.spk} texture_min {self.tex_min} '
                                       f'minsup {self.minsup} temporal {(self.ttol, self.tfr, self.treq)} -> SGBM {SM.SGBM} POST {SM.POST}')
            if (self.net_model or self.ttol > 0) and self.depth_mode == 'stereo':
                import net_model as NMOD
                if self.net_model:
                    fence = NMOD.load_fence(self.net_fence)
                    self.nmod = NMOD.NetModel(fence)
                else:
                    fence = None
                self.op = NMOD.OdomPose(anchor='origin', base_height_m=0.05)    # ev_bridge odom: base_link on the pad
                self.R_bc = np.asarray(c['R_base_optical'], float); self.t_bc = np.asarray(c['left_in_base_m'], float)
                self.create_subscription(Odometry, '/visual_slam/tracking/odometry', self.on_odom, qos_profile_sensor_data)
                msha = hashlib.sha256((Path(__file__).resolve().parent / 'net_model.py').read_bytes()).hexdigest()[:16]
                self.get_logger().info(f'net_model ON sha256[:16]={msha} fence {fence} from {self.net_fence}')
            self.pub_l = self.create_publisher(Image, '/stereo/left/image', out)
            self.pub_r = self.create_publisher(Image, '/stereo/right/image', out)
            self.pend = {}; self.lock = threading.Lock(); self.stat = dict(n=0, ms=[], t=time.monotonic())
            self.create_subscription(Image, '/gz/oak/left/image', lambda m: self.on_raw(m, 0), qos_profile_sensor_data)
            self.create_subscription(Image, '/gz/oak/right/image', lambda m: self.on_raw(m, 1), qos_profile_sensor_data)
        self.gz_k_logged = set()
        if self.scan_on:
            import depth_scan as DS
            self.DS = DS
            rmin = self.K[0] * self.B / float(self.SM.SGBM['numDisparities']) if self.depth_mode == 'stereo' else self.dmin
            self.scanner = DS.DepthScan(tuple(self.K), (self.W, self.H), range_min=rmin, **self.scan_kw)
            self.pub_scan = self.create_publisher(LaserScan, self.scan_topic, out)
            ssha = hashlib.sha256((Path(__file__).resolve().parent / 'depth_scan.py').read_bytes()).hexdigest()[:16]
            self.get_logger().info(f'depth_scan ON sha256[:16]={ssha} -> {self.scan_topic} in {self.scan_frame} '
                                   f'(child of {P["depth"]["frame_id"]}) {json.dumps(self.scanner.meta())}')
        self.static_tf()

    @staticmethod
    def gray(m):
        return np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.step)[:, :m.width]

    def mono(self, header, frame, img):
        o = Image(); o.header = header; o.header.frame_id = frame
        o.height, o.width = img.shape; o.encoding = 'mono8'; o.is_bigendian = 0; o.step = img.shape[1]
        o.data = np.ascontiguousarray(img).tobytes()
        return o

    def on_raw(self, m, eye):
        """pair clean gz left/right by stamp -> OAK sensor model -> publish noisy pair (+ SGBM depth)"""
        if m.encoding not in ('mono8', 'L8', '8UC1'):
            self.get_logger().warn(f'unexpected gz image encoding {m.encoding}', throttle_duration_sec=5.0); return
        key = (m.header.stamp.sec, m.header.stamp.nanosec)
        with self.lock:
            slot = self.pend.setdefault(key, [None, None]); slot[eye] = m
            if slot[0] is None or slot[1] is None:
                old = sorted(k for k in self.pend if k < key)
                for k in old[:-4]:
                    del self.pend[k]                      # drop unmatched old halves
                return
            del self.pend[key]
        t0 = time.perf_counter()
        L, R = slot
        seed = self.SM.frame_seed(f'{key[0]}.{key[1]:09d}')
        nl, nr, _g = self.SM.sensor_model(self.gray(L), self.gray(R), seed)
        self.pub_l.publish(self.mono(L.header, 'oak_left', nl))   # left first: the node pairs depth with the newest left
        self.pub_r.publish(self.mono(R.header, 'oak_right', nr))
        if self.depth_mode == 'stereo':
            depth, _disp, _valid = self.SM.sgbm_depth(nl, nr, self.K[0], baseline=self.B)
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            pose = None
            if self.op is not None:
                pose = self.op.camera(key[0] + key[1] * 1e-9, self.R_bc, self.t_bc)
            if pose is not None and self.nmod is not None:  # model-based: fence + VIO pose (falls back below if no pose)
                disp = np.where(depth > 0, np.float32(self.K[0] * self.B) / np.maximum(depth, 1e-6), 0).astype(np.float32)
                bad, _info = self.nmod(nl, disp, nr, tuple(self.K), self.B, pose[0], pose[1])
                depth = depth.copy(); depth[bad] = 0.0; self.nm_model_n += 1
                self.nm_frac.append(float(np.count_nonzero(bad)) / max(1, int(np.count_nonzero(depth)) + int(np.count_nonzero(bad))))
            elif self.nm is not None:                     # repetitive-texture (net) aliases -> 0
                depth, bad = self.nm.apply_depth(nl, depth, self.K[0], self.B, nr)
                self.nm_frac.append(float(np.count_nonzero(bad)) / max(1, int(np.count_nonzero(depth)) + int(np.count_nonzero(bad))))
            if self.minsup[0] > 0 or self.tcheck is not None:   # free-space noise filters
                n0 = int(np.count_nonzero(depth)); depth = depth.copy()
                if self.minsup[0] > 0:
                    disp = np.where(depth > 0, np.float32(self.K[0] * self.B) / np.maximum(depth, 1e-6), 0).astype(np.float32)
                    depth[~self.DF.min_support(disp, self.minsup[0], self.minsup[1])] = 0.0
                if self.tcheck is not None and pose is not None:
                    depth[~self.tcheck(depth, pose[0], pose[1], key[0] + key[1] * 1e-9)] = 0.0
                self.noise_frac.append(1.0 - int(np.count_nonzero(depth)) / max(n0, 1))
            o = Image(); o.header = L.header; o.header.frame_id = P['depth']['frame_id']
            o.height, o.width, o.encoding, o.is_bigendian = depth.shape[0], depth.shape[1], '32FC1', 0
            o.step = depth.shape[1] * 4; o.data = depth.tobytes()
            self.pub_d.publish(o)
            self.pub_di.publish(self.info(L.header, P['depth']['frame_id'], 0.0))
            if self.scanner is not None:
                self.publish_scan(L.header, depth)
        st = self.stat; st['n'] += 1; st['ms'].append(1e3 * (time.perf_counter() - t0))
        if time.monotonic() - st['t'] > 10.0:
            ms = np.asarray(st['ms'])
            self.get_logger().info(f'oak pipeline: {st["n"]} pairs in {time.monotonic() - st["t"]:.1f}s wall, '
                                   f'{ms.mean():.1f} ms mean / {ms.max():.1f} ms max per pair, pending {len(self.pend)}'
                                   + (f', net filter removed {100 * np.mean(self.nm_frac):.2f} % of returns' if self.nm_frac else '')
                                   + (f', net_model frames {self.nm_model_n}' if self.nmod is not None else '')
                                   + (f', noise filters removed {100 * np.mean(self.noise_frac):.2f} %' if self.noise_frac else ''))
            st.update(n=0, ms=[], t=time.monotonic()); self.nm_frac = []; self.noise_frac = []

    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.op.add(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w))

    def info(self, header, frame, tx):
        fx, fy, cx, cy = self.K
        m = CameraInfo(); m.header = header; m.header.frame_id = frame
        m.width, m.height = self.W, self.H; m.distortion_model = P['camera']['distortion_model']
        m.d = list(P['camera']['D']); m.k = [fx, 0., cx, 0., fy, cy, 0., 0., 1.]
        m.r = [1., 0., 0., 0., 1., 0., 0., 0., 1.]; m.p = [fx, 0., cx, tx, 0., fy, cy, 0., 0., 0., 1., 0.]
        return m

    def on_info(self, m, frame, tx, pub):
        if frame not in self.gz_k_logged:                       # what the renderer itself reports
            self.gz_k_logged.add(frame)
            self.get_logger().info(f'gz camera_info {frame}: K={list(np.round(m.k, 3))} {m.width}x{m.height}')
        pub.publish(self.info(m.header, frame, tx))

    def on_depth(self, m):
        if m.encoding not in ('32FC1',):
            self.get_logger().warn(f'unexpected gz depth encoding {m.encoding}', throttle_duration_sec=5.0); return
        d = np.frombuffer(bytes(m.data), np.float32).reshape(m.height, -1)[:, :m.width].copy()
        bad = ~np.isfinite(d) | (d < self.dmin) | (d > self.dmax)
        if self.noise_k > 0:
            d = d + self.rng.standard_normal(d.shape).astype(np.float32) * (self.noise_k * d * d)
        if self.dropout > 0:
            bad |= self.rng.random(d.shape) < self.dropout
        d[bad] = 0.0
        o = Image(); o.header = m.header; o.header.frame_id = P['depth']['frame_id']
        o.height, o.width, o.encoding, o.is_bigendian = m.height, m.width, '32FC1', 0
        o.step = m.width * 4; o.data = d.astype(np.float32).tobytes()
        self.pub_d.publish(o)
        self.pub_di.publish(self.info(m.header, P['depth']['frame_id'], 0.0))
        if self.scanner is not None:
            self.publish_scan(m.header, d.astype(np.float32))

    def publish_scan(self, header, depth):
        """final depth (after the net / noise filters) -> /team13/depth_scan LaserScan, stamp = the depth stamp"""
        t0 = time.perf_counter(); sc = self.scanner
        if depth.shape != (sc.size[1], sc.size[0]):
            self.get_logger().warn(f'depth_scan: depth {depth.shape} != {sc.size[::-1]}, no scan', throttle_duration_sec=5.0)
            return
        rng, inten = sc(depth)
        m = LaserScan(); m.header.stamp = header.stamp; m.header.frame_id = self.scan_frame
        m.angle_min, m.angle_max, m.angle_increment = sc.angle_min, sc.angle_max, sc.angle_increment
        m.time_increment = 0.0; m.scan_time = 1.0 / float(P['depth']['rate_hz'])
        m.range_min, m.range_max = float(sc.range_min), float(sc.range_max)
        m.ranges = array.array('f', rng.tobytes()); m.intensities = array.array('f', inten.tobytes())
        self.pub_scan.publish(m)
        st = self.scan_st; st['n'] += 1; st['ms'] += 1e3 * (time.perf_counter() - t0)
        st['hit'] += float(np.count_nonzero(np.isfinite(rng))) / rng.size
        if time.monotonic() - st['t'] > 10.0:
            self.get_logger().info(f'depth_scan: {st["n"]} scans, {st["ms"] / st["n"]:.2f} ms mean (incl. publish), '
                                   f'{100 * st["hit"] / st["n"]:.1f} % of bins with a return')
            self.scan_st = dict(n=0, ms=0.0, hit=0.0, t=time.monotonic())

    def static_tf(self):
        c, imu = P['camera'], P['imu']
        R_bc = np.asarray(c['R_base_optical'], float)
        q_bc = mat_to_quat_xyzw(R_bc)
        tfs = []
        for child, t in (('oak_left', c['left_in_base_m']), ('oak_right', c['right_in_base_m']), ('oak_rgb', c['rgb_in_base_m'])):
            tfs.append(self.tf('base_link', child, t, q_bc))
        # oak_imu axes = body FLU, expressed in the oak_left optical frame
        t_li = R_bc.T @ (np.asarray(imu['in_base_m']) - np.asarray(c['left_in_base_m']))
        tfs.append(self.tf('oak_left', 'oak_imu', t_li, mat_to_quat_xyzw(R_bc.T)))
        if self.scanner is not None:             # scan frame: the depth camera's origin, x forward / y left / z up
            tfs.append(self.tf(P['depth']['frame_id'], self.scan_frame, (0.0, 0.0, 0.0), self.DS.Q_OPTICAL_SCAN_XYZW))
        self.stb = StaticTransformBroadcaster(self); self.stb.sendTransform(tfs)

    def tf(self, parent, child, t, q):
        m = TransformStamped(); m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id, m.child_frame_id = parent, child
        m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = (float(v) for v in t)
        m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z, m.transform.rotation.w = (float(v) for v in q)
        return m


def main():
    rclpy.init(); n = OakSimDriver()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
