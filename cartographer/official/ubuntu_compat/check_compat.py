#!/usr/bin/env python3
"""Non-flying check of the running official-compat sim against OFFICIAL_SIM_SPEC.md.

  source ubuntu_compat/env.sh; python3 ubuntu_compat/check_compat.py [--secs 10] [--out check.json]

Run with the vehicle disarmed on the start pad. Prints PASS / FAIL / INFO per item
(INFO = no official value to compare against yet) and writes the same as JSON.
Exit code = number of FAILs.
"""
import argparse, json, math, subprocess, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / 'profile.json').read_text())
PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
SENS_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                      history=HistoryPolicy.KEEP_LAST, depth=50)
K_SPEC = P['camera']['K']
try:                                   # which image/depth mode the running stack uses (start_compat.sh)
    STACK = json.loads(Path('/compat/run/logs/stack.json').read_text())
except Exception:  # noqa: BLE001
    STACK = {}
STEREO = STACK.get('depth') == 'stereo'


def q_to_R(x, y, z, w):
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def imp(t):
    pkg, _, name = t.split('/')
    return getattr(__import__(f'{pkg}.msg', fromlist=[name]), name)


TOPICS = {   # topic: (type, qos, keep_last_n)
    '/clock': ('rosgraph_msgs/msg/Clock', SENS_QOS, 1),
    '/stereo/left/image': ('sensor_msgs/msg/Image', SENS_QOS, 40),
    '/stereo/right/image': ('sensor_msgs/msg/Image', SENS_QOS, 40),
    '/stereo/left/camera_info': ('sensor_msgs/msg/CameraInfo', SENS_QOS, 1),
    '/stereo/right/camera_info': ('sensor_msgs/msg/CameraInfo', SENS_QOS, 1),
    '/camera/depth/image_raw': ('sensor_msgs/msg/Image', SENS_QOS, 40),
    '/camera/depth/camera_info': ('sensor_msgs/msg/CameraInfo', SENS_QOS, 1),
    '/gz/oak/left/camera_info': ('sensor_msgs/msg/CameraInfo', SENS_QOS, 1),
    P['imu']['topic']: ('sensor_msgs/msg/Imu', SENS_QOS, 200),
    '/estimator/healthy': ('std_msgs/msg/Bool', QoSProfile(depth=10), 5),
    '/fmu/out/vehicle_local_position_v1': ('px4_msgs/msg/VehicleLocalPosition', PX4_QOS, 1),
    '/fmu/out/vehicle_status_v1': ('px4_msgs/msg/VehicleStatus', PX4_QOS, 1),
    '/fmu/out/vehicle_attitude': ('px4_msgs/msg/VehicleAttitude', PX4_QOS, 1),
    '/fmu/out/vehicle_land_detected': ('px4_msgs/msg/VehicleLandDetected', PX4_QOS, 1),
    '/fmu/out/distance_sensor': ('px4_msgs/msg/DistanceSensor', PX4_QOS, 20),
    '/fmu/out/vehicle_odometry': ('px4_msgs/msg/VehicleOdometry', PX4_QOS, 1),
    '/gz/gt/odometry': ('nav_msgs/msg/Odometry', SENS_QOS, 1),
}
PX4_IN = ['/fmu/in/offboard_control_mode', '/fmu/in/trajectory_setpoint', '/fmu/in/vehicle_command',
          '/fmu/in/vehicle_visual_odometry']


class Checker(Node):
    def __init__(self):
        super().__init__('check_compat')
        from tf2_ros import Buffer, TransformListener
        self.tfb = Buffer(); self.tfl = TransformListener(self.tfb, self)
        self.last = defaultdict(list); self.stamps = defaultdict(list); self.recv = defaultdict(list)
        for t, (ty, q, n) in TOPICS.items():
            self.create_subscription(imp(ty), t, lambda m, t=t, n=n: self.on(t, m, n), q)

    def on(self, t, m, n):
        self.recv[t].append((time.monotonic(), self.get_clock().now().nanoseconds * 1e-9))
        if hasattr(m, 'header'):
            self.stamps[t].append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        elif hasattr(m, 'timestamp'):
            self.stamps[t].append(m.timestamp * 1e-6)
        elif hasattr(m, 'clock'):
            self.stamps[t].append(m.clock.sec + m.clock.nanosec * 1e-9)
        L = self.last[t]; L.append(m)
        if len(L) > n:
            del L[0]


R = []


def item(status, name, measured, expected=''):
    R.append(dict(status=status, item=name, measured=measured, expected=expected))
    print(f'{status:4s}  {name:46s} {str(measured):48s} {expected}', flush=True)


def check(cond, name, measured, expected):
    item('PASS' if cond else 'FAIL', name, measured, expected)


def rate(st):
    u = sorted(set(st))
    return (len(u) - 1) / (u[-1] - u[0]) if len(u) > 2 and u[-1] > u[0] else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--secs', type=float, default=10.0); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    rclpy.init(); n = Checker()
    n.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
    t_end = time.monotonic() + a.secs
    while time.monotonic() < t_end:
        rclpy.spin_once(n, timeout_sec=0.05)
    names = dict(n.get_topic_names_and_types())

    item('INFO', 'stack modes (images / depth / ev source)', f"{STACK.get('images', 'gz')} / {STACK.get('depth', 'ideal')} / {STACK.get('ev_source')}", '')
    # ---------------- graph: presence + type
    for t, (ty, _, _) in TOPICS.items():
        if t.startswith('/gz/'):
            continue
        check(ty in names.get(t, []), f'topic {t}', names.get(t, 'missing'), ty)
    for t in PX4_IN:
        subs = n.get_subscriptions_info_by_topic(t)
        check(len(subs) > 0, f'PX4 subscribes {t}', f'{len(subs)} reader(s)', '>=1 (uXRCE agent)')
    pubs = [p_.node_name for p_ in n.get_publishers_info_by_topic('/fmu/in/vehicle_visual_odometry')]
    check(pubs == ['ev_bridge'], 'EV publisher = ev_bridge only', pubs, "['ev_bridge']")
    for t in ('/tf', '/tf_static'):
        check(t in names, f'topic {t}', 'present' if t in names else 'missing', 'tf2_msgs/msg/TFMessage')

    # ---------------- rates (sim time from stamps)
    c = P['camera']
    exp_rates = {'/stereo/left/image': (c['rate_hz'], 0.12), '/stereo/right/image': (c['rate_hz'], 0.12),
                 '/camera/depth/image_raw': (P['depth']['rate_hz'], 0.12), '/stereo/left/camera_info': (c['rate_hz'], 0.12),
                 '/camera/depth/camera_info': (P['depth']['rate_hz'], 0.12), P['imu']['topic']: (P['imu']['rate_hz'], 0.06)}
    for t, (hz, tol) in exp_rates.items():
        r = rate(n.stamps[t]); check(abs(r - hz) <= tol * hz, f'rate {t}', f'{r:.1f} Hz (sim)', f'{hz} Hz +-{int(tol*100)}%')
    hb = n.recv['/estimator/healthy']
    hz_h = (len(hb) - 1) / (hb[-1][0] - hb[0][0]) if len(hb) > 2 and hb[-1][0] > hb[0][0] else 0.0
    check(hz_h >= 5.0, 'rate /estimator/healthy', f'{hz_h:.1f} Hz (wall)', '>= 5 Hz (node: fresh < 1 s)')
    for t, lo in (('/fmu/out/vehicle_local_position_v1', 20), ('/fmu/out/vehicle_attitude', 20),
                  ('/fmu/out/distance_sensor', 10), ('/fmu/out/vehicle_status_v1', 0.5), ('/fmu/out/vehicle_land_detected', 0.2)):
        r = rate(n.stamps[t]); check(r >= lo, f'rate {t}', f'{r:.1f} Hz (PX4 time)', f'>= {lo} Hz')
    clk = n.recv['/clock']; st = n.stamps['/clock']
    if len(clk) > 2:
        rtf = (st[-1] - st[0]) / max(1e-6, clk[-1][0] - clk[0][0])
        item('INFO', 'real-time factor', f'{rtf:.2f}', 'official ~0.86')

    # ---------------- images
    for t, frame in (('/stereo/left/image', 'oak_left'), ('/stereo/right/image', 'oak_right')):
        m = (n.last[t] or [None])[-1]
        if m is None:
            check(False, f'{t} content', 'no message', ''); continue
        check(m.encoding == c['encoding'] and (m.width, m.height) == (c['width'], c['height']),
              f'{t} encoding/size', f'{m.encoding} {m.width}x{m.height}', f'{c["encoding"]} {c["width"]}x{c["height"]}')
        check(m.header.frame_id == frame, f'{t} frame_id', m.header.frame_id, frame)
        img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.step)[:, :m.width]
        item('INFO', f'{t} brightness mean/std', f'{img.mean():.1f} / {img.std():.1f}', 'textured (std > 10)')
    for t, frame, tx in (('/stereo/left/camera_info', 'oak_left', 0.0), ('/stereo/right/camera_info', 'oak_right', -K_SPEC[0] * c['baseline_m']),
                         ('/camera/depth/camera_info', 'oak_left', 0.0)):
        m = (n.last[t] or [None])[-1]
        if m is None:
            check(False, f'{t}', 'no message', ''); continue
        K = (m.k[0], m.k[4], m.k[2], m.k[5])
        check(max(abs(x - y) for x, y in zip(K, K_SPEC)) < 0.5, f'{t} K (fx fy cx cy)', [round(x, 2) for x in K], K_SPEC)
        check(m.header.frame_id == frame, f'{t} frame_id', m.header.frame_id, frame)
        if 'right' in t:
            item('INFO', f'{t} P Tx (=-fx*B)', round(m.p[3], 2), f'{tx:.2f} (official value not captured)')
    g = (n.last['/gz/oak/left/camera_info'] or [None])[-1]
    if g is not None:
        Kg = (g.k[0], g.k[4], g.k[2], g.k[5])
        check(max(abs(x - y) for x, y in zip(Kg, K_SPEC)) < 0.5, 'renderer (gz) left K', [round(x, 2) for x in Kg], K_SPEC)

    # ---------------- depth
    dm = (n.last['/camera/depth/image_raw'] or [None])[-1]
    if dm is not None:
        check(dm.encoding in ('32FC1', '16UC1') and (dm.width, dm.height) == (c['width'], c['height']),
              'depth encoding/size', f'{dm.encoding} {dm.width}x{dm.height}', '32FC1 m | 16UC1 mm, 640x400')
        check(dm.header.frame_id == 'oak_left', 'depth frame_id (aligned to left)', dm.header.frame_id, 'oak_left')
        d = np.frombuffer(bytes(dm.data), np.float32 if dm.encoding == '32FC1' else np.uint16).reshape(dm.height, -1)[:, :dm.width].astype(np.float32)
        if dm.encoding == '16UC1':
            d = d / 1000.0
        bad = np.isinf(d).sum() + (d < 0).sum()
        check(bad == 0, 'depth invalid encoded as 0/NaN only', f'{int(bad)} inf/negative px', '0')
        v = d[(d > 0) & np.isfinite(d)]
        item('INFO', 'depth valid fraction / range', f'{len(v) / d.size:.2f} / {v.min() if len(v) else 0:.2f}-{v.max() if len(v) else 0:.2f} m', '')
        # renderer intrinsics: horizontal camera at rest sees the floor at z = fy*h / (v - cy)
        rows = np.arange(215, 361); zs = []
        for r_ in rows:
            seg = d[r_, 300:340]; seg = seg[seg > 0]
            zs.append(np.median(seg) if len(seg) > 10 else np.nan)
        zs = np.asarray(zs); ok = np.isfinite(zs)
        if ok.sum() > 50:
            A = np.c_[np.ones(ok.sum()), zs[ok]]; sol, *_ = np.linalg.lstsq(A, zs[ok] * rows[ok], rcond=None)
            fyh, cy = sol; h = P['airframe']['base_link_rest_height_m'] + c['left_in_base_m'][2]
            gt = (n.last['/gz/gt/odometry'] or [None])[-1]
            if gt is not None:                         # sim-side truth of the camera height above the visual floor (z = 0)
                h = gt.pose.pose.position.z + c['left_in_base_m'][2]
                item('INFO', 'base_link height at rest (gz truth)', f'{gt.pose.pose.position.z:.4f} m', '0.05 m (node assumes floor = base - 0.05)')
            if STEREO:                                 # stereo depth: holes + matching noise, not a renderer-K test
                item('INFO', 'depth floor fit (stereo depth): cy / fy', f'{cy:.2f} / {fyh / h:.1f}', f'{K_SPEC[3]} / {K_SPEC[1]}')
            else:
                check(abs(cy - K_SPEC[3]) < 2.0, 'depth floor fit: cy', f'{cy:.2f}', f'{K_SPEC[3]} +-2')
                check(abs(fyh / h - K_SPEC[1]) < 0.04 * K_SPEC[1], f'depth floor fit: fy (camera h {h:.4f} m)', f'{fyh / h:.1f}', f'{K_SPEC[1]} +-4%')
        # depth/left stamp pairing (node pairs within 0.04 s)
        ls = set(round(s, 4) for s in n.stamps['/stereo/left/image']); ds = [round(s, 4) for s in n.stamps['/camera/depth/image_raw']]
        frac = sum(1 for s in ds if s in ls) / max(1, len(ds))
        check(frac > 0.9, 'depth stamps == left image stamps', f'{frac:.2f}', '> 0.9')
    # stereo baseline from disparity * depth / fx
    try:
        import cv2
        lm = n.last['/stereo/left/image']; rm = n.last['/stereo/right/image']
        rs = {round(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, 4): m for m in rm}
        dd = {round(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, 4): m for m in n.last['/camera/depth/image_raw']}
        pair = next(((l, rs[k], dd[k]) for l in reversed(lm) for k in [round(l.header.stamp.sec + l.header.stamp.nanosec * 1e-9, 4)]
                     if k in rs and k in dd), None)
        if pair:
            L_, R_, D_ = pair
            li = np.frombuffer(bytes(L_.data), np.uint8).reshape(L_.height, L_.step)[:, :L_.width]
            ri = np.frombuffer(bytes(R_.data), np.uint8).reshape(R_.height, R_.step)[:, :R_.width]
            dz = np.frombuffer(bytes(D_.data), np.float32).reshape(D_.height, -1)[:, :D_.width]
            sg = cv2.StereoSGBM_create(minDisparity=0, numDisparities=96, blockSize=7, P1=8 * 49, P2=32 * 49, uniquenessRatio=10)
            disp = sg.compute(li, ri).astype(np.float32) / 16.0
            msk = (disp > 2) & (dz > 0.4) & (dz < 5.0)
            if msk.sum() > 500:
                B = float(np.median(disp[msk] * dz[msk] / K_SPEC[0]))
                check(abs(B - c['baseline_m']) < 0.006, 'stereo baseline (SGBM disparity x depth / fx)', f'{B:.4f} m ({int(msk.sum())} px)', f'{c["baseline_m"]} m')
    except Exception as e:  # noqa: BLE001
        item('INFO', 'stereo baseline check skipped', repr(e)[:60], '')

    # ---------------- IMU
    im = n.last[P['imu']['topic']]
    if im:
        acc = np.array([[m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z] for m in im]).mean(0)
        gyr = np.array([[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z] for m in im]).std(0)
        check(im[-1].header.frame_id == P['imu']['frame_id'], 'IMU frame_id', im[-1].header.frame_id, P['imu']['frame_id'])
        check(abs(np.linalg.norm(acc) - 9.81) < 0.2, 'IMU |acc| at rest', f'{np.linalg.norm(acc):.3f} m/s2 {np.round(acc, 2).tolist()}', '9.81 (+z up in FLU)')
        item('INFO', 'IMU gyro noise std (rad/s)', np.round(gyr, 4).tolist(), 'official noise not captured')

    # ---------------- TF
    def lookup(p, ch):
        try:
            return n.tfb.lookup_transform(p, ch, Time())
        except Exception:
            return None
    Rbc = np.asarray(c['R_base_optical'], float)
    for ch, t_exp in (('oak_left', c['left_in_base_m']), ('oak_right', c['right_in_base_m']), ('oak_rgb', c['rgb_in_base_m'])):
        tf = lookup('base_link', ch)
        if tf is None:
            check(False, f'TF base_link->{ch}', 'missing', ''); continue
        tr, q = tf.transform.translation, tf.transform.rotation
        t_ = np.array([tr.x, tr.y, tr.z])
        Rm = q_to_R(q.x, q.y, q.z, q.w)
        check(np.abs(t_ - t_exp).max() < 1e-3, f'TF base_link->{ch} t', np.round(t_, 4).tolist(), t_exp)
        check(np.abs(Rm - Rbc).max() < 1e-3, f'TF base_link->{ch} R (FLU<-optical)', np.round(Rm, 3).astype(int).tolist() if np.abs(Rm - np.round(Rm)).max() < 1e-3 else np.round(Rm, 3).tolist(), c['R_base_optical'])
    tl, tr_ = lookup('base_link', 'oak_left'), lookup('base_link', 'oak_right')
    if tl and tr_:
        B = math.dist((tl.transform.translation.x, tl.transform.translation.y, tl.transform.translation.z),
                      (tr_.transform.translation.x, tr_.transform.translation.y, tr_.transform.translation.z))
        check(abs(B - c['baseline_m']) < 0.002, 'TF baseline |oak_left - oak_right|', f'{B:.4f} m', '~0.075 m')
    ti = lookup('oak_left', 'oak_imu')
    check(ti is not None, 'TF oak_left->oak_imu', 'present' if ti else 'missing', 'present')
    to = lookup('odom', 'base_link')
    if to is not None:
        age = n.get_clock().now().nanoseconds * 1e-9 - (to.header.stamp.sec + to.header.stamp.nanosec * 1e-9)
        check(age < 0.5, 'TF odom->base_link fresh', f'age {age:.2f} s, t={np.round([to.transform.translation.x, to.transform.translation.y, to.transform.translation.z], 3).tolist()}', '< 0.5 s')
    else:
        check(False, 'TF odom->base_link', 'missing', 'present')

    # ---------------- PX4 state
    lp = (n.last['/fmu/out/vehicle_local_position_v1'] or [None])[-1]
    if lp:
        check(bool(lp.xy_valid and lp.z_valid and math.isfinite(lp.heading)), 'PX4 local position valid (node pose_ok)', f'xy {lp.xy_valid} z {lp.z_valid} heading {lp.heading:.3f}', 'all valid')
        item('INFO', 'PX4 local position at rest', f'({lp.x:.3f}, {lp.y:.3f}, {lp.z:.3f}) hdg {math.degrees(lp.heading):.1f} deg', 'start-anchored ~0')
    vs = (n.last['/fmu/out/vehicle_status_v1'] or [None])[-1]
    if vs:
        check(vs.arming_state == 1, 'PX4 disarmed on the pad', vs.arming_state, '1 (disarmed)')
        item('INFO', 'PX4 pre_flight_checks_pass (current mode)', f'{bool(vs.pre_flight_checks_pass)} nav_state {vs.nav_state}',
             'Hold needs global position (none, GPS-denied); Offboard arming needs local position only')
    ld = (n.last['/fmu/out/vehicle_land_detected'] or [None])[-1]
    if ld:
        check(bool(ld.landed), 'PX4 land detected at rest', bool(ld.landed), 'True')
    ds = n.last['/fmu/out/distance_sensor']
    if ds:
        m = ds[-1]; ok = m.min_distance <= m.current_distance <= m.max_distance and m.signal_quality != 0
        check(ok, 'distance_sensor valid (node rule)', f'cur {m.current_distance:.3f} [{m.min_distance}, {m.max_distance}] q {m.signal_quality}', 'min<=cur<=max, q!=0')
        item('PASS' if m.orientation == 25 else 'INFO', 'distance_sensor orientation', m.orientation, '25 = downward (official not captured)')
    hb = n.last['/estimator/healthy']
    check(bool(hb) and bool(hb[-1].data), '/estimator/healthy', hb[-1].data if hb else 'none', 'True while VIO fresh')

    # ---------------- QoS
    for t in ('/fmu/out/vehicle_local_position_v1', '/fmu/out/vehicle_status_v1', '/fmu/out/distance_sensor', '/fmu/out/vehicle_attitude'):
        pubs = n.get_publishers_info_by_topic(t)
        if pubs:
            q = pubs[0].qos_profile
            check(q.reliability == ReliabilityPolicy.BEST_EFFORT and q.durability == DurabilityPolicy.TRANSIENT_LOCAL,
                  f'QoS {t}', f'{q.reliability.name} {q.durability.name}', 'BEST_EFFORT TRANSIENT_LOCAL')
    for t in ('/stereo/left/image', '/camera/depth/image_raw', '/estimator/healthy'):
        pubs = n.get_publishers_info_by_topic(t)
        if pubs:
            q = pubs[0].qos_profile
            item('INFO', f'QoS {t}', f'{q.reliability.name} {q.durability.name} depth {q.depth}', 'node subscribes sensor_data (BEST_EFFORT): compatible')

    # ---------------- airframe (generated model) + PX4 params
    try:
        import xml.etree.ElementTree as ET
        sdf = ET.parse('/compat/run/models/skybeta_oak/model.sdf').getroot()
        mass = sum(float(l.find('inertial/mass').text) for l in sdf.iter('link'))
        check(abs(mass - P['airframe']['mass_kg']) < 0.01, 'model total mass', f'{mass:.3f} kg', '1.25 kg (Sky Beta)')
        rot = [tuple(float(x) for x in l.find('pose').text.split()[:2]) for l in sdf.iter('link') if l.get('name', '').startswith('rotor_')]
        span = max(max(abs(x), abs(y)) for x, y in rot) + 0.0889
        item('INFO', 'rotor layout / axis half extent incl. 7in props', f'{rot} / {span:.3f} m', 'planner half span 0.20 m')
    except Exception as e:  # noqa: BLE001
        item('INFO', 'model check skipped', repr(e)[:60], '')
    try:
        names_ = ['EKF2_EV_CTRL', 'EKF2_HGT_REF', 'EKF2_GPS_CTRL', 'EKF2_RNG_CTRL', 'EKF2_EV_DELAY', 'MPC_THR_HOVER', 'UXRCE_DDS_SYNCT', 'SIM_GZ_EN_LIDAR']
        vals = {}
        for k in names_ + ['SIM_GZ_EN_ODOM', 'SIM_BAT_ENABLE']:
            out = subprocess.run(['/work/src/PX4-Autopilot/build/px4_sitl_default/bin/px4-param', 'show', k], cwd='/compat/run/px4-rootfs',
                                 capture_output=True, text=True, timeout=10).stdout
            vals[k] = next((ln.split(':')[-1].strip() for ln in out.splitlines() if f' {k} ' in ln), '?')
        check(vals.get('SIM_GZ_EN_ODOM') == '0', 'PX4 SIM_GZ_EN_ODOM (no gz truth into EKF2)', vals.get('SIM_GZ_EN_ODOM'), '0')
        item('INFO', 'PX4 params (official not captured)', vals, '')
    except Exception as e:  # noqa: BLE001
        item('INFO', 'PX4 param read skipped', repr(e)[:60], '')

    fails = sum(1 for r in R if r['status'] == 'FAIL')
    print(f'\n{sum(1 for r in R if r["status"] == "PASS")} PASS, {fails} FAIL, {sum(1 for r in R if r["status"] == "INFO")} INFO', flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(dict(when=time.strftime('%Y-%m-%d %H:%M:%S'), results=R, fails=fails), indent=1, default=str))
    n.destroy_node(); rclpy.shutdown()
    return fails


if __name__ == '__main__':
    raise SystemExit(main())
