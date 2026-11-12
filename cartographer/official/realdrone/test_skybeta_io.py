"""Pure tests for realdrone/skybeta_io.py (no ROS). Cross-checks the organizers' frames.py when
SKYBETA_FRAMES points at a copy of skybeta_flight_interface/frames.py (or it is importable).
Run: python3 -m pytest -q realdrone/test_skybeta_io.py   (or python3 realdrone/test_skybeta_io.py)"""
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realdrone import skybeta_io as io   # noqa: E402
from common.geometry import camera_pose_ned, quat_wxyz_to_mat   # noqa: E402


def _frames():
    p = os.environ.get('SKYBETA_FRAMES')
    if p and Path(p).exists():
        spec = importlib.util.spec_from_file_location('sky_frames', p); m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m); return m
    try:
        from skybeta_flight_interface import frames
        return frames
    except ImportError:
        return None


def q_yaw_xyzw(yaw, roll=0.0, pitch=0.0):
    cr, sr, cp, sp, cy, sy = (math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2),
                              math.cos(yaw / 2), math.sin(yaw / 2))
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def test_position_and_heading():
    for yaw_enu in np.radians([0, 30, 90, -120, 179]):
        pos, q, hdg, vel = io.odom_to_ned((1.0, 2.0, 3.0), q_yaw_xyzw(yaw_enu), (1.0, 0.0, 0.5))
        assert pos == (2.0, 1.0, -3.0)
        assert abs(io.wrap_pi(hdg - (math.pi / 2 - yaw_enu))) < 1e-9
        # body forward 1 m/s at ENU yaw -> NED velocity along the NED heading; up 0.5 -> down -0.5
        assert np.allclose(vel, (math.cos(hdg), math.sin(hdg), -0.5), atol=1e-9)


def test_matches_organizer_frames():
    fr = _frames()
    if fr is None:
        print('skip: no skybeta frames.py'); return
    rng = np.random.default_rng(0)
    for _ in range(50):
        r, p, y = rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(-math.pi, math.pi)
        qx = q_yaw_xyzw(y, r, p)
        mine = io.q_enu_flu_to_ned_frd(qx); theirs = fr.quat_enu_flu_to_ned_frd(qx)
        assert min(np.abs(np.subtract(mine, theirs)).max(), np.abs(np.add(mine, theirs)).max()) < 1e-12
        assert abs(io.wrap_pi(io.yaw_of_wxyz(mine) - fr.yaw_ned_from_quat_xyzw(qx))) < 1e-9
        v = rng.normal(size=3)
        assert np.allclose(io.enu_to_ned(v), fr.enu_to_ned(v))
        # setpoint path: our ENU quaternion must give the interface the NED yaw we asked for
        yaw_ned = rng.uniform(-math.pi, math.pi)
        _, q = io.setpoint_enu((0, 0, -1), yaw_ned)
        assert abs(io.wrap_pi(fr.yaw_ned_from_quat_xyzw(q) - yaw_ned)) < 1e-9
        # position round trip through the interface's own conversion (no anchoring)
        sp = rng.normal(size=3); p_enu, _ = io.setpoint_enu(sp, None)
        assert np.allclose(fr.setpoint_position_enu_to_ned(p_enu), sp)


def test_setpoint_no_yaw_is_zero_quat():
    p, q = io.setpoint_enu((1.0, 2.0, -1.5), None)
    assert p == (2.0, 1.0, 1.5) and q == (0.0, 0.0, 0.0, 0.0)
    _, q = io.setpoint_enu((0, 0, 0), float('nan'))
    assert q == (0.0, 0.0, 0.0, 0.0)


def test_camera_pose_consistent_with_tf_chain():
    """camera_pose_ned(q from odometry) must equal composing the ENU TF chain odom->base_link->camera, then permuting."""
    rng = np.random.default_rng(1)
    R_bc = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]]); t_bc = np.array([0.105, 0.037, 0.030])
    P_NED_ENU = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.]])
    for _ in range(20):
        qx = q_yaw_xyzw(rng.uniform(-3, 3), rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)); p = rng.normal(size=3)
        pos, q, _, _ = io.odom_to_ned(p, qx, (0, 0, 0))
        R_nc, cam = camera_pose_ned(q, pos, R_bc, t_bc)
        R_ob = quat_wxyz_to_mat(qx[3], qx[0], qx[1], qx[2])                  # ENU <- FLU
        cam_enu = p + R_ob @ t_bc; R_oc = R_ob @ R_bc
        assert np.allclose(cam, P_NED_ENU @ cam_enu, atol=1e-9)
        assert np.allclose(R_nc, P_NED_ENU @ R_oc, atol=1e-9)


def test_limit_lead():
    sp, c = io.limit_lead((3.0, 0, 0), (0, 0, 0), 1.5)
    assert c and np.allclose(sp, (1.5, 0, 0))
    sp, c = io.limit_lead((0.5, 0.5, -0.3), (0, 0, 0), 1.5)
    assert not c and np.allclose(sp, (0.5, 0.5, -0.3))


def test_pose_history():
    h = io.PoseHistory()
    for i in range(10):
        h.add(100.0 + i / 30.0, (i, 0, 0), (1, 0, 0, 0))
    p, q = h.at(100.0 + 3 / 30.0); assert np.allclose(p, (3, 0, 0))
    p, q = h.at(100.0 + 3.5 / 30.0); assert np.allclose(p, (3.5, 0, 0))
    assert h.at(99.0) is None and h.at(101.0) is None
    assert h.at(100.0 + 9 / 30.0 + 0.02) is not None                       # just after the newest pose



# ------------------------------------------------------------------ adapter logic with stubbed ROS
def _stub_ros():
    import types
    class Msg:
        def __init__(self): pass
    class Obj:  # attribute bag
        def __getattr__(self, k):
            v = Obj(); object.__setattr__(self, k, v); return v
    def mk(name, **attrs):
        m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m
    class PoseStamped:
        def __init__(self): self.header = Obj(); self.pose = Obj()
    for n in ('rclpy', 'geometry_msgs', 'nav_msgs', 'std_msgs', 'px4_msgs'):
        mk(n)
    mk('rclpy.qos', DurabilityPolicy=Obj(), HistoryPolicy=Obj(), QoSProfile=lambda **k: None, ReliabilityPolicy=Obj())
    mk('geometry_msgs.msg', PoseStamped=PoseStamped); mk('nav_msgs.msg', Odometry=Msg); mk('std_msgs.msg', String=Msg)
    mk('px4_msgs.msg', VehicleLandDetected=Msg, VehicleLocalPosition=Msg, VehicleStatus=Msg)

    class Clock:
        def __init__(self, n): self.n = n
        def now(self):
            c = self; o = Obj(); o.nanoseconds = int(c.n.t * 1e9); o.to_msg = lambda: c.n.t; return o
    class Node:
        def __init__(self): self.t = 1000.0; self.pubs = []; self.subs = {}; self.params = {}
        def declare_parameter(self, n, v):
            o = Obj(); o.value = self.params.get(n, v); return o
        def create_publisher(self, typ, topic, q):
            node = self
            class P:
                def publish(self, m): node.pubs.append((topic, m))
            return P()
        def create_subscription(self, typ, topic, cb, q): self.subs[topic] = cb
        def get_clock(self): return Clock(self)
    return Node, Obj


def test_adapter_publish_logic():
    import json as _json
    from types import SimpleNamespace as NS
    Node, Obj = _stub_ros()
    import importlib
    sio = importlib.reload(io)
    from mission.core import Vehicle
    node, veh, events = Node(), Vehicle(), []
    a = sio.SkybetaIO(node, veh, events.append)
    sub = node.subs
    def odom(x, y, z, yaw=0.0):
        m = Obj(); m.header.stamp.sec = int(node.t); m.header.stamp.nanosec = int((node.t % 1) * 1e9)
        q = q_yaw_xyzw(yaw); p = m.pose.pose.position; p.x, p.y, p.z = x, y, z
        o = m.pose.pose.orientation; o.x, o.y, o.z, o.w = q; t = m.twist.twist.linear; t.x = t.y = t.z = 0.0
        sub['/visual_slam/tracking/odometry'](m)
    def status(nav, arm):
        m = Obj(); m.nav_state, m.arming_state = nav, arm; sub[a.topics['vehicle_status']](m)
    def state(**kw):
        m = Obj(); m.data = _json.dumps(dict(dict(origin_enu=[0, 0, -0.01], landed=True), **kw)); sub['/skybeta/state'](m)
    vo = Obj(); vo.vo_state = 1
    a.vo_available = True
    sub.setdefault('/visual_slam/status', a.on_vo)
    # 1) waiting: stream the CURRENT pose (ENU), zero quaternion, even if the mission asks for something else
    odom(1.0, 2.0, 0.05); a.on_vo(vo); status(4, 1); state(); fresh = a.update(node.t)
    assert fresh and not veh.offboard_active and veh.pose_ok and abs(veh.clamp_origin_d - 0.01) < 1e-9
    sent = a.publish(NS(pos_ned=np.array([5.0, 5.0, -2.0]), yaw=0.3, vel_ned=None, acc_ned=None), node.t, fresh)
    topic, m = node.pubs[-1]
    assert topic == '/skybeta/setpoint/pose' and (m.pose.position.x, m.pose.position.y, m.pose.position.z) == (1.0, 2.0, 0.05)
    assert (m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w) == (0.0, 0.0, 0.0, 0.0)
    # 2) offboard: the mission setpoint, lead-limited to 1.5 m, with yaw
    node.t += 0.05; odom(1.0, 2.0, 0.05); a.on_vo(vo); status(14, 2); fresh = a.update(node.t)
    assert veh.offboard_active and veh.armed and any(e['type'] == 'sky_offboard' for e in events)
    a.publish(NS(pos_ned=np.array([2.0, 1.0, -0.6]), yaw=math.pi / 2, vel_ned=np.zeros(3), acc_ned=None), node.t, fresh)
    _, m = node.pubs[-1]
    assert np.allclose((m.pose.position.x, m.pose.position.y, m.pose.position.z), (1.0, 2.0, 0.6))
    assert abs(m.pose.orientation.w - 1.0) < 1e-9                       # NED yaw 90 deg (east) = ENU yaw 0
    assert any(e['type'] == 'sky_feedforward_dropped' for e in events)
    a.publish(NS(pos_ned=np.array([2.0, 10.0, -0.6]), yaw=None, vel_ned=None, acc_ned=None), node.t, fresh)
    _, m = node.pubs[-1]; d = math.dist((m.pose.position.x, m.pose.position.y, m.pose.position.z), (1.0, 2.0, 0.05))
    assert abs(d - 1.5) < 1e-6 and any(e['type'] == 'sky_lead_clipped' for e in events)
    # 3) offboard + mission None -> publish nothing (the interface keeps holding our last pose)
    n0 = len(node.pubs); assert a.publish(NS(pos_ned=None, yaw=None, vel_ned=None, acc_ned=None), node.t, fresh) is None
    assert len(node.pubs) == n0
    # 4) stale odometry -> nothing at all, pose_ok false
    node.t += 0.5; status(14, 2); fresh = a.update(node.t)
    assert not fresh and not veh.pose_ok and a.publish(NS(pos_ned=np.zeros(3), yaw=None, vel_ned=None, acc_ned=None), node.t, fresh) is None
    # 5) VehicleCommands are logged, never sent
    a.log_commands([NS(name='land', command=21, param1=0.0, param2=0.0)], node.t)
    assert any(e['type'] == 'sky_command_not_sent' for e in events) and all(t == '/skybeta/setpoint/pose' for t, _ in node.pubs)
    # 6) pose history at the image stamp
    assert a.pose_at(1000.05) is not None
    for k in [k for k in list(sys.modules) if k.split('.')[0] in ('rclpy', 'geometry_msgs', 'nav_msgs', 'std_msgs', 'px4_msgs')]:
        sys.modules.pop(k)


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k)
