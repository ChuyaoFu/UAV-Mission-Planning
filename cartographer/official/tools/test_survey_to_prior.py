#!/usr/bin/env python3
"""Tests for tools/survey_to_prior.py and tools/survey_bag.py (Python 3.9+, numpy, OpenCV).

    python3 tools/test_survey_to_prior.py            # synthetic tests (no ROS, no bag files needed)
    SURVEY_TEST_BAG=<flight bag dir> python3 tools/test_survey_to_prior.py   # + one real bag smoke test

The synthetic walk is written as a real (uncompressed) mcap with ros2msg schemas and CDR payloads, so the
built-in reader and the CDR decoder are exercised end to end; the `mcap` package reader too when installed.
"""
import json
import math
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import survey_bag as sb                      # noqa: E402
import survey_to_prior as stp                # noqa: E402
from mission.core import CoursePrior, StartFrame   # noqa: E402

# ------------------------------------------------------------------ minimal CDR encoder + mcap writer
SCHEMAS = {
    'px4_msgs/msg/VehicleLocalPosition': ('uint64 timestamp\nbool xy_valid\nbool z_valid\nfloat32 x\nfloat32 y\nfloat32 z\n'
                                          'float32[2] delta_xy\nfloat32 vx\nfloat32 vy\nfloat32 vz\nfloat32 heading\n'
                                          'uint32 MESSAGE_VERSION = 1\n'),
    'px4_msgs/msg/VehicleAttitude': 'uint64 timestamp\nfloat32[4] q  # w x y z\n',
    'px4_msgs/msg/DistanceSensor': ('uint64 timestamp\nfloat32 min_distance\nfloat32 max_distance\nfloat32 current_distance\n'
                                    'int8 signal_quality\nuint8 ROTATION_DOWNWARD_FACING = 25\n'),
    'tf2_msgs/msg/TFMessage': ('geometry_msgs/TransformStamped[] transforms\n'
                               '================================================================================\n'
                               'MSG: geometry_msgs/TransformStamped\nstd_msgs/Header header\nstring child_frame_id\nTransform transform\n'
                               '================================================================================\n'
                               'MSG: geometry_msgs/Transform\nVector3 translation\nQuaternion rotation\n'
                               '================================================================================\n'
                               'MSG: geometry_msgs/Quaternion\nfloat64 x 0\nfloat64 y 0\nfloat64 z 0\nfloat64 w 1\n'
                               '================================================================================\n'
                               'MSG: geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z\n'
                               '================================================================================\n'
                               'MSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n'
                               '================================================================================\n'
                               'MSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n'),
    'sensor_msgs/msg/Image': ('std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\nuint8 is_bigendian\n'
                              'uint32 step\nuint8[] data\n'
                              '================================================================================\n'
                              'MSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n'
                              '================================================================================\n'
                              'MSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n'),
}


class Cdr:
    def __init__(self):
        self.b = bytearray(b'\x00\x01\x00\x00')

    def al(self, n):
        while (len(self.b) - 4) % n:
            self.b.append(0)

    def p(self, fmt, v):
        n = struct.calcsize(fmt); self.al(n); self.b += struct.pack('<' + fmt, v); return self

    def s(self, txt):
        e = txt.encode() + b'\x00'; self.p('I', len(e)); self.b += e; return self

    def arr(self, fmt, vals, seq=False):
        if seq:
            self.p('I', len(vals))
        n = struct.calcsize(fmt)
        if len(vals):
            self.al(n)
        for v in vals:
            self.b += struct.pack('<' + fmt, v)
        return self


def enc_lpos(t, p, v, hd):
    c = Cdr().p('Q', int(t * 1e6)).p('?', True).p('?', True)
    for x in p:
        c.p('f', x)
    c.arr('f', [0.0, 0.0])
    for x in v:
        c.p('f', x)
    return bytes(c.p('f', hd).b)


def enc_att(t, q):
    return bytes(Cdr().p('Q', int(t * 1e6)).arr('f', list(q)).b)


def enc_tof(t, d):
    return bytes(Cdr().p('Q', int(t * 1e6)).p('f', 0.02).p('f', 4.0).p('f', d).p('b', -1).b)


def enc_tf(chain):
    c = Cdr().p('I', len(chain))
    for par, ch, t, q in chain:
        c.p('i', 0).p('I', 0).s(par).s(ch)
        for x in t:
            c.p('d', x)
        for x in q:                                                   # x y z w
            c.p('d', x)
    return bytes(c.b)


def enc_image(stamp, img, enc='mono8'):
    h, w = img.shape[:2]
    c = Cdr().p('i', int(stamp)).p('I', int(round((stamp - int(stamp)) * 1e9))).s('oak_left')
    c.p('I', h).p('I', w).s(enc).p('B', 0).p('I', img.strides[0])
    return bytes(c.arr('B', list(img.tobytes()), seq=True).b)


def write_mcap(path, msgs):
    """msgs: list of (topic, schema name, log_time_s, cdr bytes). No chunks, no compression, no summary."""
    def rec(op, body):
        return bytes([op]) + struct.pack('<Q', len(body)) + body

    def s_(x):
        e = x.encode(); return struct.pack('<I', len(e)) + e
    out = bytearray(sb.MCAP_MAGIC)
    out += rec(0x01, s_('ros2') + s_('test'))
    sid, cid = {}, {}
    for topic, sname, _, _ in msgs:
        if sname not in sid:
            sid[sname] = len(sid) + 1
            data = SCHEMAS[sname].encode()
            out += rec(0x03, struct.pack('<H', sid[sname]) + s_(sname) + s_('ros2msg') + struct.pack('<I', len(data)) + data)
        if topic not in cid:
            cid[topic] = len(cid) + 1
            out += rec(0x04, struct.pack('<HH', cid[topic], sid[sname]) + s_(topic) + s_('cdr') + struct.pack('<I', 0))
    for k, (topic, _, t, data) in enumerate(msgs):
        ns = int(round(t * 1e9))
        out += rec(0x05, struct.pack('<HIQQ', cid[topic], k, ns, ns) + data)
    out += rec(0x0F, struct.pack('<I', 0))
    out += rec(0x02, struct.pack('<QQI', 0, 0, 0))
    out += sb.MCAP_MAGIC
    Path(path).write_bytes(bytes(out))


# ------------------------------------------------------------------ synthetic protocol walk (replica truth)
TRUTH = json.loads((HERE.parent / 'ubuntu_compat' / 'course_prior.replica.json').read_text())
REST_H, TOF_REST = 0.05, 0.049
STOPS = [  # name, f, l, base up, yaw (deg CCW from forward), surface up under the drone
    ('start', 0.0, 0.0, 0.0, 0.0, -0.05),
    ('c1', 1.0, -0.05, 0.8, 0.0, None), ('c2', 2.4, 0.0, 0.8, 0.0, None), ('c3', 3.7, 0.15, 0.8, 0.0, None),
    ('tunnel', 4.4, 0.6, 0.57, -90.0, -0.05),
    ('w0', 4.4, 0.6, 1.16, 180.0, None), ('w1', 5.0, -2.4, 1.16, 180.0, None),
    ('window', 4.5, -3.75, 1.16, 180.0, -0.05),
    ('fence_front', 5.15, -3.1, 1.3, 0.0, -0.05),
    ('fence_right', 1.75, -5.0, 1.3, -90.0, -0.05),
    ('rock_view', 2.5, -3.75, 0.70, 180.0, -0.05),
    ('rock_top', 0.95, -3.75, 0.40, 180.0, 0.20),              # held 15 cm above resting: ToF corrects
    ('wall_from', 0.3, -3.1, 1.0, 90.0, -0.05),
    ('wall_top', 0.3, -2.25, 1.70, 90.0, -0.05),              # skids on the 2 cm edge, ToF sees the floor
    ('wall_to', 0.3, -1.4, 1.0, -90.0, -0.05),
    ('fence_back', -0.55, -0.6, 1.3, 180.0, -0.05),
    ('fence_left', 2.6, 0.7, 1.3, 90.0, -0.05),
    ('end', 0.0, 0.0, 0.0, 0.0, -0.05),
]
HOME_NED, HOME_YAW = np.array([1.5, -2.0, -0.3]), 0.6


def synth_walk(path, stops=STOPS, hold=4.0, speed=0.5, rate=50.0, seed=0, drift=None, with_tf=True, runaway=False):
    rng = np.random.default_rng(seed)
    sf = StartFrame(HOME_NED, HOME_YAW)
    msgs, t = [], 1000.0
    keys = []                                                         # (t, flu, yaw_deg, surface or None, still)
    prev = None
    for name, f, l, u, yaw, surf in stops:
        p = np.array([f, l, u])
        if prev is not None:
            d = np.linalg.norm(p - prev[0]); n = max(2, int(d / speed * rate))
            y0, y1 = prev[1], yaw
            dy = (y1 - y0 + 180) % 360 - 180
            for k in range(1, n):
                s = 0.5 - 0.5 * math.cos(math.pi * k / n)
                keys.append((t, prev[0] + (p - prev[0]) * s, y0 + dy * s, None)); t += 1 / rate
        is_stop = surf is not None
        for k in range(int((6.0 if name in ('start', 'end') else hold) * rate) if is_stop else 1):
            keys.append((t, p, yaw, surf)); t += 1 / rate
        prev = (p, yaw)
    T = np.array([k[0] for k in keys]); FLU = np.array([k[1] for k in keys])
    if drift is not None:
        FLU = FLU + np.outer((T - T[0]) / (T[-1] - T[0]), drift)
    if runaway:                                                       # VIO lost after the walk: pose flies off
        n = int(5 * rate)
        T = np.r_[T, T[-1] + np.arange(1, n + 1) / rate]
        FLU = np.r_[FLU, FLU[-1] + np.outer(np.arange(1, n + 1) / rate, [400.0, -150.0, 5.0])]
        keys = keys + [(t_, FLU[len(keys) + k], 0.0, None) for k, t_ in enumerate(T[len(keys):])]
    V = np.gradient(np.array([sf.to_ned(x) for x in FLU]), T, axis=0) + rng.normal(0, 0.005, FLU.shape)
    NED = np.array([sf.to_ned(x) for x in FLU + rng.normal(0, 0.002, FLU.shape)])
    for k, (tt, flu, yaw, surf) in enumerate(keys):
        hd = HOME_YAW - math.radians(yaw)
        msgs.append(('/fmu/out/vehicle_local_position_v1', 'px4_msgs/msg/VehicleLocalPosition', tt,
                     enc_lpos(tt, NED[k], V[k], (hd + math.pi) % (2 * math.pi) - math.pi)))
        if k % 2 == 0:
            msgs.append(('/fmu/out/vehicle_attitude', 'px4_msgs/msg/VehicleAttitude', tt, enc_att(tt, (math.cos(hd / 2), 0, 0, math.sin(hd / 2)))))
        if k % 5 == 0:
            s_up = -0.05 if surf is None else surf
            tof = (flu[2] - s_up) - REST_H + TOF_REST + rng.normal(0, 0.003)
            msgs.append(('/fmu/out/distance_sensor', 'px4_msgs/msg/DistanceSensor', tt, enc_tof(tt, tof)))
    if with_tf:
        msgs.insert(0, ('/tf_static', 'tf2_msgs/msg/TFMessage', 999.0, enc_tf([
            ('base_link', 'oak', (0.1, 0.03, 0.02), (0, 0, 0, 1)),
            ('oak', 'oak_left', (0.016, 0.007, 0.009), (-0.5, 0.5, -0.5, 0.5))])))
    tiny = np.zeros((4, 6), np.uint8)                                 # one paired frame: gives the depth frame_id
    msgs.insert(1, ('/camera/depth/image_raw', 'sensor_msgs/msg/Image', 1001.0, enc_image(50.0, tiny)))
    msgs.insert(2, ('/stereo/left/image', 'sensor_msgs/msg/Image', 1001.0, enc_image(50.01, tiny)))
    write_mcap(path, msgs)
    return sf


def run(argv):
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sv = stp.main(argv)
    sv._stdout = buf.getvalue()
    return sv


class TestReader(unittest.TestCase):
    def test_cdr_roundtrip_and_tf_chain(self):
        with tempfile.TemporaryDirectory() as d:
            img = (np.arange(30 * 40) % 251).astype(np.uint8).reshape(30, 40)
            write_mcap(Path(d) / 'b.mcap', [
                ('/tf_static', 'tf2_msgs/msg/TFMessage', 1.0, enc_tf([
                    ('base_link', 'oak', (0.1, 0.03, 0.02), (0, 0, 0, 1)),
                    ('oak', 'oak_left', (0.016, 0.007, 0.009), (-0.5, 0.5, -0.5, 0.5))])),
                ('/fmu/out/distance_sensor', 'px4_msgs/msg/DistanceSensor', 2.0, enc_tof(2.0, 1.234)),
                ('/fmu/out/vehicle_local_position_v1', 'px4_msgs/msg/VehicleLocalPosition', 2.5,
                 enc_lpos(2.5, (1.0, -2.0, -0.5), (0.1, 0.2, 0.3), 1.25)),
                ('/stereo/left/image', 'sensor_msgs/msg/Image', 3.0, enc_image(12.345, img))])
            readers = ['builtin'] + (['mcap'] if sb.resolve_reader('auto') == 'mcap' else [])
            for rd in readers:
                got = list(sb.open_bag(Path(d), None, rd))
                self.assertEqual([g[0] for g in got], ['/tf_static', '/fmu/out/distance_sensor',
                                                       '/fmu/out/vehicle_local_position_v1', '/stereo/left/image'], rd)
                tof, lp, im = got[1][2], got[2][2], got[3][2]
                self.assertAlmostEqual(tof['current_distance'], 1.234, 5)
                self.assertEqual(tof['signal_quality'], -1)
                self.assertAlmostEqual(lp['y'], -2.0, 6); self.assertAlmostEqual(lp['vz'], 0.3, 6)
                self.assertAlmostEqual(lp['heading'], 1.25, 6); self.assertNotIn('MESSAGE_VERSION', lp)
                self.assertAlmostEqual(got[2][1], 2.5, 9)
                self.assertTrue(np.array_equal(sb.image_array(im), img))
                self.assertAlmostEqual(sb.stamp_s(im), 12.345, 6)
                tf = sb.TfTree(); tf.add_msg(got[0][2])
                R, t = tf.lookup('base_link', 'oak_left')
                # optical z (forward) -> FLU x; optical x (right) -> FLU -y
                self.assertTrue(np.allclose(R @ [0, 0, 1], [1, 0, 0], atol=1e-9))
                self.assertTrue(np.allclose(R @ [1, 0, 0], [0, -1, 0], atol=1e-9))
                self.assertTrue(np.allclose(t, [0.116, 0.037, 0.029]))
                self.assertIsNone(tf.lookup('base_link', 'nowhere'))


class TestSyntheticWalk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.bag = Path(cls.tmp.name) / 'walk'
        cls.bag.mkdir()
        synth_walk(cls.bag / 'walk_0.mcap')
        cls.out = Path(cls.tmp.name) / 'out'
        (Path(cls.tmp.name) / 'stops.txt').write_text('1 tunnel   # no detectors in this test: name the tunnel stop\n')
        cls.sv = run([str(cls.bag), '--name', 'syn', '--out-dir', str(cls.out), '--no-detect', '--reader', 'builtin',
                      '--stops', str(Path(cls.tmp.name) / 'stops.txt'),
                      '--truth', str(HERE.parent / 'ubuntu_compat' / 'course_prior.replica.json')])
        cls.js = json.loads((cls.out / 'course_prior.syn.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_stops_and_start_frame(self):
        S = self.sv.stops
        stops_expected = [s for s in STOPS if s[5] is not None]
        self.assertEqual(len(S), len(stops_expected))
        for s, e in zip(S, stops_expected):
            self.assertLess(np.hypot(s.flu[0] - e[1], s.flu[1] - e[2]), 0.02, (s.idx, e[0]))
            self.assertLess(abs(wrap(s.yaw_deg - e[4])), 1.0, (s.idx, e[0]))
        self.assertEqual(S[0].label, 'start'); self.assertEqual(S[-1].label, 'end')
        self.assertTrue(np.allclose(self.sv.sf.o, HOME_NED, atol=0.01))
        self.assertAlmostEqual(self.sv.sf.yaw, HOME_YAW, 2)

    def test_geometry_labels(self):
        got = {s.label for s in self.sv.stops}
        for lab in ('rock_top', 'wall_from', 'wall_top', 'wall_to', 'fence_front', 'fence_back', 'fence_left', 'fence_right'):
            self.assertIn(lab, got)
        by = {s.label: s for s in self.sv.stops}
        self.assertLess(abs(by['wall_from'].flu[1] + 3.1), 0.02)      # rock side
        self.assertLess(abs(by['wall_to'].flu[1] + 1.4), 0.02)        # home side

    def test_prior_values_vs_truth(self):
        js, T = self.js, TRUTH
        for k in ('forward_min', 'forward_max', 'left_min', 'left_max'):
            self.assertLess(abs(js['fence'][k] - T['fence'][k]), 0.03, k)
        self.assertLess(abs(js['wall']['top_up'] - T['wall']['top_up']), 0.02)
        self.assertLess(abs(js['landing_target']['top_up'] - T['landing_target']['top_up']), 0.02)
        self.assertLess(abs(js['wall']['cross_from']['left'] + 3.1), 0.02)
        self.assertLess(abs(js['wall']['cross_to']['left'] + 1.4), 0.02)
        # corridor: every truth corridor goal within 0.25 m of start -> waypoints -> tunnel stop polyline
        poly = [(0.0, 0.0)] + [(c['forward'], c['left']) for c in js['corridor']] + [(4.4, 0.6)]
        for c in T['corridor']:
            self.assertLess(min(seg_dist((c['forward'], c['left']), poly[i], poly[i + 1]) for i in range(len(poly) - 1)), 0.25)

    def test_no_invention(self):
        js = self.js
        self.assertIsNone(js['tunnel']); self.assertIsNone(js['window'])
        self.assertFalse(self.sv.loads)
        w = ' '.join(self.sv.warn)
        self.assertIn('tunnel: no confident detection', w); self.assertIn('window: no confident detection', w)
        self.assertEqual(self.sv.stops[1].label, 'tunnel'); self.assertTrue(self.sv.stops[1].user)
        prov = {k: v['source'] for k, v in self.sv.src.items()}
        self.assertIn('rock_top', prov['landing_target.center'])
        self.assertIn('pose+tof', prov['landing_target.top_up'])
        self.assertIn('net touch', prov['fence.forward_max'])
        self.assertTrue(any(l.startswith('fence.forward_max:') for l in js['_comment']))

    def test_png_and_report(self):
        self.assertTrue((self.out / 'course_prior.syn.png').stat().st_size > 5000)
        rep = json.loads((self.out / 'course_prior.syn.report.json').read_text())
        self.assertTrue(rep['truth']['fence.left_min']['pass'])
        self.assertEqual(len(rep['stops']), len(self.sv.stops))

    def test_fallback_fills_and_marks(self):
        out = Path(self.tmp.name) / 'fb'
        sv = run([str(self.bag), '--name', 'fb', '--out-dir', str(out), '--no-detect', '--reader', 'builtin',
                  '--fallback', str(HERE.parent / 'ubuntu_compat' / 'course_prior.replica.json')])
        self.assertTrue(sv.loads)
        self.assertIn('FALLBACK', sv.src['tunnel']['source'])
        self.assertIn('FALLBACK', sv.src['cruise_up_m']['source'])         # no tunnel stop -> no corridor height
        js = json.loads((out / 'course_prior.fb.json').read_text())
        CoursePrior.from_dict(js)
        self.assertLess(abs(js['wall']['top_up'] - 1.65), 0.02)     # measured fields are not overwritten


class TestAssignment(unittest.TestCase):
    def test_missing_touch_is_not_invented(self):
        with tempfile.TemporaryDirectory() as d:
            st = [s for s in STOPS if s[0] != 'fence_front']
            synth_walk(Path(d) / 'w.mcap', stops=st)
            sv = run([str(Path(d) / 'w.mcap'), '--out-dir', d, '--no-detect', '--reader', 'builtin'])
            js = json.loads((Path(d) / 'course_prior.survey.json').read_text())
            self.assertNotIn('forward_max', js['fence'])
            self.assertTrue(any('fence_front' in w for w in sv.warn))
            self.assertNotIn('fence_front', [s.label for s in sv.stops])

    def test_stops_file_overrides(self):
        with tempfile.TemporaryDirectory() as d:
            synth_walk(Path(d) / 'w.mcap')
            n = len([s for s in STOPS if s[5] is not None])
            lab = {str(i): 'skip' for i in range(n)}
            lab.update({'0': 'start', str(n - 1): 'end', '1': 'tunnel', '2': 'window'})
            Path(d, 'stops.json').write_text(json.dumps(lab))
            sv = run([str(Path(d) / 'w.mcap'), '--out-dir', d, '--no-detect', '--reader', 'builtin', '--stops', str(Path(d, 'stops.json'))])
            self.assertEqual(sv.stops[1].label, 'tunnel'); self.assertTrue(sv.stops[1].user)
            self.assertFalse(any(s.label.startswith('fence') or s.label.startswith('wall') for s in sv.stops))
            # the corridor ends at the user-labelled tunnel stop
            self.assertEqual(sv.src['corridor']['source'].split(':')[0], 'path start->tunnel stop')
            Path(d, 'stops.txt').write_text('1 tunnel\n2 bogus\n')
            with self.assertRaises(SystemExit):
                run([str(Path(d) / 'w.mcap'), '--out-dir', d, '--no-detect', '--reader', 'builtin', '--stops', str(Path(d, 'stops.txt'))])

    def test_closure_warning_on_drift(self):
        with tempfile.TemporaryDirectory() as d:
            synth_walk(Path(d) / 'w.mcap', drift=np.array([0.3, 0.0, 0.0]))
            sv = run([str(Path(d) / 'w.mcap'), '--out-dir', d, '--no-detect', '--reader', 'builtin'])
            # the drifted end stop is no longer on the pad -> not 'end'; either way a warning is raised
            self.assertTrue(any('closure' in w or 'did not end back on the pad' in w for w in sv.warn))

    def test_closure_correct_removes_linear_drift(self):
        with tempfile.TemporaryDirectory() as d:
            synth_walk(Path(d) / 'w.mcap', drift=np.array([0.25, -0.15, 0.0]))
            err = {}
            for flag in ([], ['--closure-correct']):
                sv = run([str(Path(d) / 'w.mcap'), '--out-dir', d, '--no-detect', '--reader', 'builtin'] + flag)
                js = json.loads((Path(d) / 'course_prior.survey.json').read_text())
                err[bool(flag)] = max(abs(js['fence'][k] - TRUTH['fence'][k]) if k in js['fence'] else 9.0   # refused = large
                                      for k in ('forward_min', 'forward_max', 'left_min', 'left_max'))
            self.assertGreater(err[False], 0.1)
            self.assertLess(err[True], 0.04)

    def test_pose_divergence_is_cut(self):
        with tempfile.TemporaryDirectory() as d:
            synth_walk(Path(d) / 'w.mcap', runaway=True)
            sv = run([str(Path(d) / 'w.mcap'), '--out-dir', d, '--no-detect', '--reader', 'builtin'])
            self.assertTrue(any('pose diverges' in w for w in sv.warn))
            self.assertEqual(sv.stops[-1].label, 'end')
            self.assertLess((Path(d) / 'course_prior.survey.png').stat().st_size, 3e6)

    def test_tf_from_bag_or_default(self):
        with tempfile.TemporaryDirectory() as d:
            synth_walk(Path(d) / 'w.mcap')
            sv = run([str(Path(d) / 'w.mcap'), '--out-dir', d, '--no-detect', '--reader', 'builtin', '--list-stops'])
            self.assertEqual(sv.tr.tf_source.split(';')[0], 'tf_static base_link<-oak_left')
            self.assertTrue(np.allclose(sv.tr.t_bc, [0.116, 0.037, 0.029]))
            self.assertEqual(sv.tr.pair, {0: 0})
            synth_walk(Path(d) / 'w.mcap', with_tf=False)
            sv = run([str(Path(d) / 'w.mcap'), '--out-dir', d, '--no-detect', '--reader', 'builtin', '--list-stops'])
            self.assertTrue(any('extrinsic' in w for w in sv.warn))


class TestHelpers(unittest.TestCase):
    def test_rdp_and_smooth(self):
        P = np.array([[0, 0], [1, 0.01], [2, 0], [2.01, 1], [2, 2]], float)
        self.assertEqual(stp.rdp(P, 0.1), [0, 2, 4])
        self.assertEqual(len(stp.smooth(np.arange(10.0), 3)), 10)

    def test_cluster_prefers_support(self):
        d = [dict(pos=np.array([0, 0, 0.]) + 0.01 * k, conf=0.6) for k in range(5)] + [dict(pos=np.array([2, 0, 0.]), conf=0.99)]
        cl = stp.cluster(d, 0.3)
        self.assertEqual(len(cl[0]['m']), 5)

    def test_stop_label_formats(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'a.json').write_text('["start", null, "tunnel"]')
            self.assertEqual(stp.load_stop_labels(Path(d, 'a.json'), 3), {0: 'start', 2: 'tunnel'})
            Path(d, 'b.txt').write_text('# idx label\n4 wall_top\n')
            self.assertEqual(stp.load_stop_labels(Path(d, 'b.txt'), 5), {4: 'wall_top'})


class TestRealBag(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('SURVEY_TEST_BAG'), 'set SURVEY_TEST_BAG to a flight bag dir')
    def test_list_stops(self):
        sv = run([os.environ['SURVEY_TEST_BAG'], '--list-stops'])
        self.assertGreaterEqual(len(sv.stops), 2)
        self.assertEqual(sv.stops[0].label, 'start')


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def seg_dist(p, a, b):
    p, a, b = (np.asarray(x, float) for x in (p, a, b))
    ab = b - a; L2 = float(ab @ ab)
    s = 0.0 if L2 < 1e-12 else float(np.clip((p - a) @ ab / L2, 0, 1))
    return float(np.linalg.norm(p - a - s * ab))


if __name__ == '__main__':
    unittest.main(verbosity=2)
