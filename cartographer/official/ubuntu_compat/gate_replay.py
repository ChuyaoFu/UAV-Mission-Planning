#!/usr/bin/env python3
"""Did the mission's prior gate reject CORRECT detections? (sim-side; needs a non-light bag)

  python3 gate_replay.py FLIGHT_DIR        -> FLIGHT_DIR/sim/gate_replay.json + summary line per detector

The node logs detections only on change / every 2 s and the Latch counts gate rejections without an event, so the
detectors are re-run offline: every 2nd recorded (left, depth) pair inside each detect window (status hint
`detector`), through the node's OWN MissionNode._detect from the flight's code.tar.gz, with the mission's NED priors
(StartFrame(home event) applied to the flown prior.json, exactly as Mission._record_home). Each detection with
conf >= the mission's min_conf is classified against replica truth (ubuntu_compat/course_prior.replica.json through
the same start frame):
  tunnel  correct = radial (perpendicular to the true axis) < 0.15 m and |axial| < 0.30 m (2nd hoop = wrong)
  window  correct = |centre - truth| < 0.20 m
  landing correct = xy < 0.25 m and |top - truth top| < 0.10 m
and tested against the Latch gate: |centre - prior| (3-D) <= stage tol + detect_prior_gate_m.
Reported: correct / correct-but-REJECTED / false / false-but-ACCEPTED counts, the gate value that would accept every
correct detection (max distance correct -> prior), and how many false detections that gate would also accept.
Landing frames the detector itself dropped as `outside_prior` (its own prior_radius_m) are re-run without a prior to
show whether that inner gate removed the real stone or only stray blobs.
"""
import json, math, sys, tarfile, tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_flight import read, stamp  # noqa: E402


def main():
    F = Path(sys.argv[1])
    tmp = Path(tempfile.mkdtemp(prefix='gate_'))
    with tarfile.open(F / 'code.tar.gz') as tf:
        tf.extractall(tmp)
    sys.path.insert(0, str(tmp))
    from common.geometry import Frame, camera_pose_ned
    from mission.core import CoursePrior, Params, StartFrame
    from node.mission_node import MissionNode
    meta = json.loads((F / 'meta.json').read_text())
    params = Params()
    for k, v in (meta.get('params') or {}).items():
        setattr(params, k, type(getattr(params, k))(v))
    P = json.loads((HERE / 'profile.json').read_text())
    c = P['camera']; K = tuple(c['K']); R_bc = np.asarray(c['R_base_optical'], float); t_bc = tuple(c['left_in_base_m'])
    prior = CoursePrior.from_dict(json.loads((F / 'prior.json').read_text()))
    truth = CoursePrior.from_dict(json.loads((HERE / 'course_prior.replica.json').read_text()))
    home, yaw, status = None, None, []
    for ln in (F / 'mission.log').read_text().splitlines():
        if not ln.startswith('{'):
            continue
        j = json.loads(ln.replace('NaN', 'null'))
        if 'event' in j and j['event'].get('kind') == 'home':
            home = np.asarray(j['event']['home_ned'], float); yaw = math.radians(j['event']['yaw_deg'])
        elif 'phase' in j and 'event' not in j:
            status.append((j['t'], (j.get('hints') or {}).get('detector'), j['phase']))
    if home is None:
        print('no home event'); return 1
    sf = StartFrame(home, yaw)
    g = dict(tunnel_entry=sf.to_ned(prior.tunnel_entry), tunnel_axis=sf.direction(prior.tunnel_axis_deg),
             window_center=sf.to_ned(prior.window_center), window_dir=sf.direction(prior.window_through_deg),
             rock=sf.to_ned(prior.landing_center))
    tr = dict(tunnel=sf.to_ned(truth.tunnel_entry), tunnel_axis=sf.direction(truth.tunnel_axis_deg),
              window=sf.to_ned(truth.window_center), landing=sf.to_ned(truth.landing_center))
    pri = dict(tunnel=g['tunnel_entry'], window=g['window_center'], landing=g['rock'])
    gate = dict(tunnel=prior.tunnel_tol_m + params.detect_prior_gate_m, window=prior.window_tol_m + params.detect_prior_gate_m,
                landing=prior.landing_tol_m + params.detect_prior_gate_m)
    min_conf = dict(tunnel=params.tunnel_min_conf, window=params.detect_min_conf,
                    landing=getattr(params, 'landing_min_conf', params.detect_min_conf))
    st_t = np.array([s[0] for s in status])
    A = read(F / 'bag', ['/stereo/left/image', '/camera/depth/image_raw', '/fmu/out/vehicle_local_position_v1', '/fmu/out/vehicle_attitude'])
    lp = A['/fmu/out/vehicle_local_position_v1']; att = A['/fmu/out/vehicle_attitude']
    lp_t = np.array([r for r, _ in lp]); at_t = np.array([r for r, _ in att])
    floor_d = float(lp[0][1].z) + 0.05
    lefts = {round(stamp(m), 3): m for _, m in A['/stereo/left/image']}
    dummy = SimpleNamespace(mission=SimpleNamespace(g=g), floor_d=floor_d, per=SimpleNamespace(floor=None, floor_t=None),
                            half_span=params.half_span_m, half_height=0.10)
    rows, k = [], 0
    for rt, m in A['/camera/depth/image_raw']:
        s = stamp(m)
        i = int(np.searchsorted(st_t, s)) - 1
        if i < 0 or not status[i][1] or status[i][1] not in ('tunnel', 'window', 'landing'):
            continue
        k += 1
        if k % 2:
            continue
        kind = status[i][1]; L = lefts.get(round(s, 3))
        if L is None:
            continue
        a_ = int(np.searchsorted(lp_t, rt)) - 1; b_ = int(np.searchsorted(at_t, rt)) - 1
        if a_ < 0 or b_ < 0:
            continue
        lpm = lp[a_][1]; pos = np.array([lpm.x, lpm.y, lpm.z]); spd = float(np.linalg.norm([lpm.vx, lpm.vy, lpm.vz]))
        R_nc, cam = camera_pose_ned(tuple(float(x) for x in att[b_][1].q), pos, R_bc, t_bc)
        img = np.frombuffer(bytes(L.data), np.uint8).reshape(L.height, L.step)[:, :L.width]
        d = np.frombuffer(bytes(m.data), np.float32).reshape(m.height, -1)[:, :m.width]
        fr = Frame(image=img, depth=d, K=K, R_nc=R_nc, cam_ned=cam, stamp=s)
        try:
            est = MissionNode._detect(dummy, kind, fr)
        except Exception as e:  # noqa: BLE001
            rows.append(dict(t=round(s, 2), kind=kind, err=repr(e)[:80])); continue
        found = bool(getattr(est, 'found', False)); conf = getattr(est, 'confidence', None)
        ctr = getattr(est, 'entry_ned', None) if kind == 'tunnel' else getattr(est, 'center_ned', None)
        r = dict(t=round(s, 2), kind=kind, phase=status[i][2], found=found, conf=None if conf is None else round(float(conf), 2),
                 speed=round(spd, 2), why=getattr(est, 'reason', None))
        if found and conf is not None and conf >= min_conf[kind] and ctr is not None and np.all(np.isfinite(ctr)):
            ctr = np.asarray(ctr, float)
            if kind == 'tunnel':
                ax = np.r_[tr['tunnel_axis'][:2], 0.0]; ax /= np.linalg.norm(ax); dv = ctr - tr['tunnel']
                axial = float(dv @ ax); radial = float(np.linalg.norm(dv - axial * ax))
                ok = radial < 0.15 and abs(axial) < 0.30; r.update(radial=round(radial, 3), axial=round(axial, 3))
            elif kind == 'window':
                e_ = float(np.linalg.norm(ctr - tr['window'])); ok = e_ < 0.20; r['err'] = round(e_, 3)
            else:
                exy = float(np.linalg.norm(ctr[:2] - tr['landing'][:2]))
                top = getattr(est, 'top_d', None); ez = None if top is None else float(top) - float(tr['landing'][2])
                ok = exy < 0.25 and (ez is None or abs(ez) < 0.10); r.update(err=round(exy, 3), top_err=None if ez is None else round(ez, 3))
            dg = float(np.linalg.norm(ctr - pri[kind]))
            r.update(cand=True, correct=bool(ok), d_prior=round(dg, 3), in_gate=dg <= gate[kind])
        if kind == 'landing' and r.get('why') == 'outside_prior':
            # the landing detector's OWN prior gate (prior_radius_m) dropped every candidate: re-run without a prior
            dummy.mission.g = dict(g, rock=None)
            try:
                e2 = MissionNode._detect(dummy, kind, fr)
            finally:
                dummy.mission.g = g
            c2 = getattr(e2, 'center_ned', None)
            if getattr(e2, 'found', False) and c2 is not None and np.all(np.isfinite(c2)):
                c2 = np.asarray(c2, float); exy = float(np.linalg.norm(c2[:2] - tr['landing'][:2]))
                r.update(noprior_found=True, noprior_err=round(exy, 3), noprior_correct=exy < 0.25,
                         noprior_d_prior_xy=round(float(np.linalg.norm(c2[:2] - pri['landing'][:2])), 3),
                         noprior_conf=round(float(getattr(e2, 'confidence', 0) or 0), 2))
        rows.append(r)
    summ = {}
    for kind in ('tunnel', 'window', 'landing'):
        R = [r for r in rows if r['kind'] == kind]
        C = [r for r in R if r.get('cand')]
        cor = [r for r in C if r['correct']]; fal = [r for r in C if not r['correct']]
        need = max((r['d_prior'] for r in cor), default=None)
        summ[kind] = dict(frames=len(R), candidates=len(C), gate_m=round(gate[kind], 2),
                          prior_err_m=round(float(np.linalg.norm(pri[kind] - tr[kind])), 3),
                          correct=len(cor), correct_rejected=sum(1 for r in cor if not r['in_gate']),
                          false=len(fal), false_accepted=sum(1 for r in fal if r['in_gate']),
                          gate_needed_for_all_correct_m=None if need is None else round(need, 3),
                          false_inside_needed_gate=None if need is None else sum(1 for r in fal if r['d_prior'] <= need),
                          false_min_d_prior_m=min((r['d_prior'] for r in fal), default=None),
                          reasons={w: sum(1 for r in R if r.get('why') == w) for w in sorted({str(r.get('why')) for r in R if not r.get('cand')})})
        if kind == 'landing':
            NP = [r for r in R if r.get('noprior_found')]
            summ[kind]['detector_gate_outside_prior'] = dict(
                rerun_found=len(NP), correct=sum(1 for r in NP if r['noprior_correct']),
                correct_d_prior_xy_max=max((r['noprior_d_prior_xy'] for r in NP if r['noprior_correct']), default=None),
                false_d_prior_xy_min=min((r['noprior_d_prior_xy'] for r in NP if not r['noprior_correct']), default=None))
    out = dict(flight=F.name, gate_rule='|centre - prior| <= tol + detect_prior_gate_m', summary=summ, rows=rows)
    (F / 'sim' / 'gate_replay.json').write_text(json.dumps(out, indent=1, default=str))
    for kind, s_ in summ.items():
        print(F.name, kind, json.dumps({k_: v for k_, v in s_.items() if k_ != 'reasons'}), 'not-candidate reasons', s_['reasons'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
