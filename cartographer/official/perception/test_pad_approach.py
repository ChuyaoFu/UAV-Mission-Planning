"""Tests for perception/pad_approach.py. Run: python3 perception/test_pad_approach.py

Closed loop: a kinematic vehicle (first-order lag on the body setpoint, 20 Hz) + camera frames at
10 Hz. Planner logic runs with a fake detector that sees the pad only where the level camera can
(floor from h / tan(25.3 deg) ahead, +-37 deg horizontally, <= 4 m), with range-proportional noise.
One end-to-end run renders frames with test_floor's ray caster and uses the real detector.
"""
import math
import os
import sys
import time
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # cartographer/official
from perception import pad_approach as PA            # noqa: E402

CAM_FWD, CAM_UP = 0.116, 0.029
ORDER = ['goto_standoff', 'descend_observe', 'acquire', 'approach', 'settle', 'blind', 'final_descent', 'touchdown']


def fake_frame(body, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    cam = np.array([body[0] + CAM_FWD * c, body[1] + CAM_FWD * s, body[2] - CAM_UP])
    return types.SimpleNamespace(cam_ned=cam, body_ned=np.asarray(body, float).copy(), heading=yaw)


def fake_detector(pad_xy, pad_yaw=0.35, rng=None, false_rate=0.0, blackout=None, clock=None, floor_z=0.0):
    rng = rng or np.random.default_rng(0)

    def det(frame, fz, prior_xy, radius):
        if blackout is not None and blackout[0] <= clock[0] < blackout[1]:
            return None, floor_z
        cam = frame.cam_ned; yaw = frame.heading
        h = floor_z - cam[2]
        d = np.asarray(pad_xy) - cam[:2]
        a = d[0] * math.cos(yaw) + d[1] * math.sin(yaw); b = -d[0] * math.sin(yaw) + d[1] * math.cos(yaw)
        if false_rate and rng.random() < false_rate:              # a confident hit somewhere else
            xy = np.asarray(prior_xy) + rng.uniform(-1, 1, 2) * radius
            if np.hypot(*(xy - np.asarray(pad_xy))) > 0.4:
                return types.SimpleNamespace(found=True, center_ned=np.r_[xy, floor_z], yaw_ned=0.1, confidence=0.6,
                                             range_m=2.0, factors={'visible': 1.0}), floor_z
        if not (a >= h / PA.TAN_HALF_VFOV + 0.25 and a <= 4.0 and abs(math.atan2(b, a)) <= math.radians(37)):
            return None, floor_z
        if prior_xy is not None and np.hypot(*(np.asarray(pad_xy) - np.asarray(prior_xy))) > radius:
            return None, floor_z
        xy = np.asarray(pad_xy) + rng.normal(0, 0.012 * a, 2)
        return types.SimpleNamespace(found=True, center_ned=np.r_[xy, floor_z], yaw_ned=pad_yaw + rng.normal(0, 0.02),
                                     confidence=0.8, range_m=float(math.hypot(a, b)), factors={'visible': 1.0}), floor_z
    return det


def run(appr, body0, yaw0, t_max=150.0, frame_fn=fake_frame, clock=None, tau=0.4, dt=0.05, frame_every=2):
    body = np.array(body0, float); yaw = float(yaw0)
    phases, events, outs = [], [], []
    k = 0; t = 0.0
    while t < t_max:
        if clock is not None:
            clock[0] = t
        fr = frame_fn(body, yaw) if k % frame_every == 0 else None
        out = appr.update(t, frame=fr, body_ned=body, yaw=yaw)
        outs.append(out); events += out.events
        phases += [e[1]['phase'] for e in out.events if e[0] == 'pad_phase' and (not phases or phases[-1] != e[1]['phase'])]
        if out.land:
            break
        if out.handoff and len([o for o in outs if o.handoff]) > 40:
            break
        a = 1.0 - math.exp(-dt / tau)
        body = body + a * (out.setpoint_ned - body)
        yaw = PA._wrap(yaw + a * PA._wrap(out.yaw - yaw))
        t += dt; k += 1
    return body, yaw, phases, events, outs


def _ev(events, name):
    return [e[1] for e in events if e[0] == name]


# ============================================================================ planner logic
def test_full_sequence_lands_on_the_pad():
    """Prior 0.3 m off; start 3 m away at 1 m. Phases in order, final xy error <= 5 cm vs the TRUE
    pad, blind segment < 1 m, camera never below the observe height before the blind leg."""
    pad = np.array([0.0, 0.0]); prior = pad + np.array([0.22, -0.2])
    clock = [0.0]
    appr = PA.PadApproach(prior, floor_z=0.0, detector=fake_detector(pad, clock=clock))
    body, yaw, phases, ev, outs = run(appr, (-3.0, 0.6, -1.0), 0.2, clock=clock)
    assert phases == ORDER, phases
    err = float(np.hypot(*(body[:2] - pad)))
    assert outs[-1].land and err <= 0.05, (err, body)
    blind = _ev(ev, 'pad_phase'); bl = [e for e in blind if e['phase'] == 'blind'][0]
    assert bl['dist_m'] < 1.0, bl
    lk = _ev(ev, 'pad_locked')[0]
    assert not _ev(ev, 'pad_lost_hold'), 'no hold when the pad leaves the bottom of the image'
    assert np.hypot(*(np.array(lk['center']) - pad)) < 0.10, lk
    obs_z = [o.setpoint_ned[2] for o in outs if o.phase in ('acquire', 'approach', 'settle', 'blind')]
    assert abs(np.median(obs_z) - (-(0.40 - CAM_UP))) < 0.02, np.median(obs_z)


def test_external_detection_path():
    """mission-core style: the node runs the detector with search() and feeds observe(); update()
    gets no frame. Same landing as the frame path."""
    pad = np.array([0.5, -0.3]); clock = [0.0]
    det = fake_detector(pad, clock=clock)
    appr = PA.PadApproach(pad + np.array([0.2, 0.2]), floor_z=0.0, detector=lambda *a: (_ for _ in ()).throw(AssertionError('unused')))
    body = np.array([-2.0, 0.4, -0.9]); yaw = 0.0; t = 0.0; phases = []
    for k in range(4000):
        clock[0] = t
        if k % 2 == 0:
            sr = appr.search()
            if sr is not None:
                d, fz = det(fake_frame(body, yaw), appr.floor_z, sr[0], sr[1])
                appr.observe(t, d, fz)
        out = appr.update(t, frame=None, body_ned=body, yaw=yaw)
        phases += [e[1]['phase'] for e in out.events if e[0] == 'pad_phase']
        if out.land:
            break
        a = 1.0 - math.exp(-0.05 / 0.4)
        body = body + a * (out.setpoint_ned - body); yaw = PA._wrap(yaw + a * PA._wrap(out.yaw - yaw)); t += 0.05
    assert out.land and np.hypot(*(body[:2] - pad)) <= 0.05, (phases, body)
    assert appr.search() is None                      # pad below the image now


def test_not_found_scans_then_hands_off():
    """No pad: acquire -> slow yaw scan (+-35 deg) -> hold position and hand off to the pilot."""
    clock = [0.0]
    appr = PA.PadApproach((0.0, 0.0), detector=lambda *a: (None, 0.0))
    body, yaw, phases, ev, outs = run(appr, (-2.0, 0.0, -0.8), 0.0, clock=clock, t_max=90)
    assert phases[:4] == ['goto_standoff', 'descend_observe', 'acquire', 'scan'] and phases[-1] == 'handoff', phases
    yaws = [math.degrees(o.yaw) for o in outs if o.phase == 'scan']
    assert max(yaws) > 30 and min(yaws) < -30, (max(yaws), min(yaws))
    ho = [o for o in outs if o.handoff]
    assert ho and np.ptp([o.setpoint_ned[0] for o in ho]) < 1e-6 and _ev(ev, 'pad_phase')[-1]['reason'] == 'pad_not_found'


def test_scan_finds_pad_outside_the_first_view():
    """The pad is 1.15 m to the side of the prior: outside the camera's view from the standoff and
    outside the acquire search radius. The yaw scan (wider radius) finds and locks it."""
    pad = np.array([0.4, 1.15]); prior = np.array([0.0, 0.0])
    clock = [0.0]
    appr = PA.PadApproach(prior, detector=fake_detector(pad, clock=clock))
    body, yaw, phases, ev, outs = run(appr, (-2.5, 0.0, -0.8), 0.0, clock=clock)
    assert 'scan' in phases and phases[-1] == 'touchdown', phases
    assert _ev(ev, 'pad_locked')[0]['phase'] == 'scan'
    assert np.hypot(*(body[:2] - pad)) <= 0.05, body


def test_false_detections_do_not_move_the_lock():
    """30 % confident hits elsewhere in the search area: lock and landing stay on the pad."""
    pad = np.array([1.0, -0.5]); clock = [0.0]
    appr = PA.PadApproach(pad + 0.15, detector=fake_detector(pad, false_rate=0.3, rng=np.random.default_rng(3), clock=clock))
    body, yaw, phases, ev, outs = run(appr, (-1.5, -0.5, -0.9), 0.0, clock=clock)
    assert phases[-1] == 'touchdown', phases
    assert np.hypot(*(body[:2] - pad)) <= 0.05, body
    assert len(_ev(ev, 'pad_detection_rejected')) > 0


def test_detections_drop_during_approach():
    """2.5 s without detections while the whole pad should be in view: hold (carrot frozen), then
    re-acquire and land. Near the end, missing detections are expected and cause no hold."""
    pad = np.array([0.0, 0.0]); clock = [0.0]
    P = PA.ApproachParams(standoff_m=2.2)              # room for a loss while the whole pad is in view
    appr = PA.PadApproach(pad, P, detector=fake_detector(pad, clock=clock, blackout=None))
    # find when the approach starts, then rerun with a blackout there
    _, _, _, ev0, outs0 = run(appr, (-2.0, 0.0, -0.8), 0.0, clock=clock)
    t_app = [i for i, o in enumerate(outs0) if o.phase == 'approach'][0] * 0.05
    clock = [0.0]
    appr = PA.PadApproach(pad, PA.ApproachParams(standoff_m=2.2), detector=fake_detector(pad, clock=clock, blackout=(t_app, t_app + 2.5)))
    body, yaw, phases, ev, outs = run(appr, (-2.0, 0.0, -0.8), 0.0, clock=clock)
    assert _ev(ev, 'pad_lost_hold') and _ev(ev, 'pad_reacquired'), [e[0] for e in ev]
    assert phases[-1] == 'touchdown' and np.hypot(*(body[:2] - pad)) <= 0.05


def test_stop_distance_and_blind_length_follow_the_fov():
    """Camera-to-centre stop = h / tan(bottom-edge angle): 0.85 m at 0.40 m (official), 0.80 m (real)."""
    a = PA.PadApproach((0, 0))
    assert abs(a.stop_cam_dist() - 0.40 / PA.TAN_HALF_VFOV) < 1e-9 and 0.83 < a.stop_cam_dist() < 0.86
    a.p.tan_half_vfov = (400 - 193.87) / 414.35                 # real skybeta rectified 640x400
    assert 0.79 < a.stop_cam_dist() < 0.82
    a = PA.PadApproach((0, 0), PA.ApproachParams(observe_cam_h_m=0.30))
    assert a.stop_cam_dist() < 0.65


# ============================================================================ rendered, real detector
def test_rendered_closed_loop_real_detector():
    """test_floor's ray caster renders every camera frame (pad on the floor, a cinder block and a
    painted rock beside it, black tape); fit_floor + landing_target drive the planner. Starts at the
    standoff, 0.7 m high, prior 0.25 m off. Lands within 5 cm of the true centre."""
    import test_floor as TF
    pol = TF.PAD_LUMA
    pad_xy = np.array([2.0, 0.3])
    scene = [TF.floor_grey(pol['floor']), TF.pad(pad_xy[0], pad_xy[1], yaw=0.4, body=pol['body'], dark=pol['dark'], white=pol['white']),
             TF._cinder(pad_xy[0] + 0.3, pad_xy[1] + 0.85), TF.paint(pad_xy[0] - 0.2, pad_xy[1] - 0.9, 0.22, 0.16, 0.3, 55),
             TF.tape(pad_xy[0] + 0.1, pad_xy[1], 2.6, 2.2, 0.0)]
    seed = [0]

    def render(body, yaw):
        seed[0] += 1
        cam_h = -body[2] + CAM_UP
        return TF.make_frame(scene, cam_h=cam_h, yaw=yaw, xy=(body[0], body[1]), floor_dropout=0.3, seed=500 + seed[0])

    prior = pad_xy + np.array([-0.2, 0.15])
    appr = PA.PadApproach(prior, floor_z=0.0)
    t0 = time.time()
    body, yaw, phases, ev, outs = run(appr, (pad_xy[0] - 1.5, pad_xy[1], -(0.7 - CAM_UP)), 0.0, frame_fn=render, t_max=60.0, frame_every=4)
    assert phases[-1] == 'touchdown', (phases, [e for e in ev if e[0] != 'pad_detection_rejected'][:12])
    err = float(np.hypot(*(body[:2] - pad_xy)))
    assert err <= 0.05, (err, _ev(ev, 'pad_locked'), outs[-1].pad)
    print(f'    rendered run: {len(outs) * 0.05:.1f} s sim, {time.time() - t0:.1f} s wall, err {100 * err:.1f} cm, '
          f'lock {_ev(ev, "pad_locked")[0]}')


def main():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    fails = 0
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            print(f'PASS {name} ({time.perf_counter() - t0:.1f}s)')
        except Exception as e:  # noqa: BLE001
            fails += 1
            import traceback
            print(f'FAIL {name}: {type(e).__name__}: {e}')
            traceback.print_exc(limit=3)
    print(f'{len(tests) - fails}/{len(tests)} passed')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.path.insert(0, _HERE)
    sys.exit(main())
