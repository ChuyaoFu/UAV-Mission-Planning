#!/usr/bin/env python3
"""Run ONE recorded test flight of node/mission_node.py in the official lab.

Every flight gets its own folder on the persistent team disk
(<claude_dev>/flights/<YYYYmmdd-HHMMSS>_<name>/, survives a GPU release; /tmp does not):
  meta.json    what ran: name, notes, wall start, prior + params, command, topics, code sha256
  code.tar.gz  snapshot of common/ node/ mission/ perception/ tools/ as flown
  prior.json   the course prior as flown
  mission.log  node stdout/stderr (2 Hz status JSON lines + events)
  bag/         rosbag2 (mcap, zstd) of PX4 in/out, estimator, VIO, TF, and (unless --light) stereo-left + depth
  bag.log      recorder output
  result.json  exit code, final phase/sub, durations, events, max displacement, bag size
and one row in <claude_dev>/flights/INDEX.md.

Terminal:   python3 tools/flight_run.py --name corridor_v08 --params '{"v_cruise": 0.8}' --notes "after Reset world"
Web "Run saved file" (managed lab recording + grade): edit flights/next_flight.json, then Run this file.
Ctrl-C / SIGTERM / "Stop program": the node is asked to LAND, then the bag is closed cleanly.
"""
import argparse, datetime as dt, hashlib, io, json, math, os, queue, signal, subprocess, sys, tarfile, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                 # claude_dev/ (or cartographer/official locally)
FLIGHTS = Path(os.environ.get('FLIGHTS_DIR', ROOT / 'flights'))
CODE_DIRS = ('common', 'node', 'mission', 'perception', 'tools')
TOPICS_CORE = [
    '/clock', '/tf', '/tf_static',
    '/fmu/out/vehicle_local_position_v1', '/fmu/out/vehicle_attitude', '/fmu/out/vehicle_status_v1',
    '/fmu/out/vehicle_land_detected', '/fmu/out/distance_sensor', '/fmu/out/vehicle_odometry',
    '/fmu/out/failsafe_flags', '/fmu/out/battery_status',
    '/fmu/in/trajectory_setpoint', '/fmu/in/offboard_control_mode', '/fmu/in/vehicle_command',
    '/fmu/in/vehicle_visual_odometry',
    '/estimator/healthy', '/visual_slam/tracking/odometry', '/visual_slam/status',
]
TOPICS_IMAGES = ['/stereo/left/image', '/stereo/left/camera_info', '/camera/depth/image_raw', '/camera/depth/camera_info']


def code_snapshot(dest: Path, dirs=CODE_DIRS):
    digests = {}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for d in dirs:
            for f in sorted((ROOT / d).rglob('*')) if (ROOT / d).is_dir() else []:
                if f.is_file() and '__pycache__' not in f.parts and f.suffix in ('.py', '.json', '.md', '.yaml', '.sh'):
                    rel = str(f.relative_to(ROOT))
                    digests[rel] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
                    tar.add(f, arcname=rel)
    dest.write_bytes(buf.getvalue())
    total = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()[:16]
    return total, digests


def stop(proc, first=signal.SIGINT, wait_s=15.0):
    """Signal a child's process group, escalating INT -> TERM -> KILL."""
    if proc is None or proc.poll() is not None:
        return proc.returncode if proc else None
    for sig, w in ((first, wait_s), (signal.SIGTERM, 5.0), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            return proc.wait(timeout=w)
        except subprocess.TimeoutExpired:
            continue
    return proc.poll()


def child_env():
    # Background shells ignore SIGINT and children inherit that; reset it so ros2 bag closes cleanly on INT.
    def pre():
        signal.signal(signal.SIGINT, signal.SIG_DFL); signal.signal(signal.SIGTERM, signal.SIG_DFL)
    return dict(start_new_session=True, preexec_fn=pre)


def du(p: Path):
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) if p.exists() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name')
    ap.add_argument('--prior', default=None, help='relative to claude_dev/, default mission/course_prior.official_sim.json')
    ap.add_argument('--params', default=None, help='JSON overrides for mission.core Params')
    ap.add_argument('--timeout', type=float, default=None, help='wall seconds before a commanded LAND')
    ap.add_argument('--light', action='store_true', help='no camera topics in the bag')
    ap.add_argument('--notes', default=None)
    ap.add_argument('--real', action='store_true', help='real aircraft: wall clock (use_sim_time false), no /clock in the bag')
    ap.add_argument('--config', default=str(FLIGHTS / 'next_flight.json'), help='used when --name is not given')
    ap.add_argument('--io', choices=('px4', 'skybeta'), default='px4',
                    help='skybeta: real drone through /skybeta/* (implies --real); needs flight_interface + stereo depth running')
    ap.add_argument('--dry-run', action='store_true',
                    help='skybeta bench: setpoints go to /team13/dry_run/setpoint_pose, never /skybeta/setpoint/pose')
    a = ap.parse_args()
    cfg = {}
    if a.name is None and Path(a.config).exists():
        cfg = json.loads(Path(a.config).read_text())
    name = a.name or cfg.get('name', 'flight')
    prior = a.prior or cfg.get('prior', 'mission/course_prior.official_sim.json')
    params = a.params if a.params is not None else json.dumps(cfg.get('params', {}))
    timeout = a.timeout if a.timeout is not None else float(cfg.get('timeout_s', 240))
    light = a.light or bool(cfg.get('light', False))
    notes = a.notes if a.notes is not None else cfg.get('notes', '')
    real = a.real or bool(cfg.get('real', False)) or a.io == 'skybeta'
    pv = json.loads(params)                                              # fail before flying on bad JSON
    if isinstance(pv, str):                                                # tolerate the old '"{...}"' workaround
        pv = json.loads(pv)
    if not isinstance(pv, dict):
        raise SystemExit(f'--params must be a JSON object, got {params!r}')
    params = json.dumps(pv)

    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    out = FLIGHTS / f'{stamp}_{"".join(c if c.isalnum() or c in "-_." else "_" for c in name)}'
    out.mkdir(parents=True, exist_ok=False)
    code_sha, digests = code_snapshot(out / 'code.tar.gz', CODE_DIRS + (('realdrone',) if a.io == 'skybeta' else ()))
    (out / 'prior.json').write_text((ROOT / prior).read_text())
    topics = [t for t in TOPICS_CORE if not (real and t == '/clock')] + ([] if light else TOPICS_IMAGES)
    node_cmd = ['python3', '-u', 'node/mission_node.py', '--ros-args', '-p', f'use_sim_time:={"false" if real else "true"}',
                '-p', f'prior:={prior}', '-p', 'params_json:=' + json.dumps(params)]   # YAML-quoted: rcl parses a bare {...} as a map and exits
    if a.io == 'skybeta':                                                  # real drone, organizers' contract
        sys.path.insert(0, str(ROOT))
        from realdrone.skybeta_io import BAG_TOPICS, BAG_IMAGE_TOPICS
        topics = BAG_TOPICS + ([] if light else BAG_IMAGE_TOPICS)
        node_cmd += ['-p', 'io:=skybeta']
        if a.dry_run:
            node_cmd += ['-p', 'sky_setpoint_topic:=/team13/dry_run/setpoint_pose']
            topics = topics + ['/team13/dry_run/setpoint_pose']
    meta = dict(name=name, notes=notes, stamp=stamp, wall_start=time.time(), prior=prior, params=pv, io=a.io, dry_run=a.dry_run,
                timeout_s=timeout, real=real, topics=topics, node_cmd=node_cmd, code_sha=code_sha, code_files=digests,
                host=os.uname().nodename, ros_domain=os.environ.get('ROS_DOMAIN_ID'))
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    print(f'[flight_run] {out}', flush=True)

    stop_req = {'sig': None}
    def on_sig(sig, _frm):
        stop_req['sig'] = sig
    signal.signal(signal.SIGINT, on_sig); signal.signal(signal.SIGTERM, on_sig)

    bag_log = open(out / 'bag.log', 'w')
    backstop = ['timeout', '-s', 'INT', '-k', '20', str(int(timeout + 90))]   # if this runner is SIGKILLed
    bag = subprocess.Popen(backstop + ['ros2', 'bag', 'record', '-o', str(out / 'bag'), '-s', 'mcap',
                                       *([] if os.environ.get('ROS_DISTRO') == 'humble' else ['--storage-preset-profile', 'zstd_fast']), *topics],   # positional topics: Humble has no --topics
                           stdout=bag_log, stderr=subprocess.STDOUT, **child_env())
    time.sleep(3.0)
    log = open(out / 'mission.log', 'w')
    t0 = time.time()
    node = subprocess.Popen(['timeout', '-s', 'INT', '-k', '40', str(int(timeout + 30))] + node_cmd, cwd=str(ROOT),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, **child_env())
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l) for l in node.stdout] and None, daemon=True).start()
    events, last, finished, home, max_xy, reason = [], None, None, None, 0.0, 'node_exit'
    while True:
        try:
            line = lines.get(timeout=0.05)
        except queue.Empty:
            line = None
        if line:
            log.write(line); log.flush(); sys.stdout.write(line); sys.stdout.flush()
            s = line.strip()
            if s.startswith('{'):
                try:
                    j = json.loads(s)
                except ValueError:
                    j = None
                if j and 'event' in j:
                    events.append(j)
                elif j and 'phase' in j:
                    last = j
                    p = j.get('pos')
                    if p and all(isinstance(c, (int, float)) and math.isfinite(c) for c in p):
                        home = home or p
                        max_xy = max(max_xy, math.hypot(p[0] - home[0], p[1] - home[1]))
            elif 'finished:' in s:
                finished = s.split('finished:', 1)[1].strip()
            continue
        if node.poll() is not None and lines.empty():
            break
        if stop_req['sig'] is not None and reason == 'node_exit':
            reason = f'signal_{stop_req["sig"]}'; os.killpg(node.pid, signal.SIGINT)   # node commands LAND on INT
        if time.time() - t0 > timeout and reason == 'node_exit':
            reason = 'timeout'; os.killpg(node.pid, signal.SIGINT)
    log.close()
    node_rc = node.wait()
    time.sleep(3.0)                                                        # keep the landing tail in the bag
    bag_rc = stop(bag, signal.SIGINT); bag_log.close()
    res = dict(name=name, stamp=stamp, reason=reason, node_rc=node_rc, bag_rc=bag_rc, finished=finished,
               final_phase=(last or {}).get('phase'), final_sub=(last or {}).get('sub'), final_pos=(last or {}).get('pos'),
               sim_t_end=(last or {}).get('t'), wall_s=round(time.time() - t0, 1), max_xy_from_start_m=round(max_xy, 2),
               events=events, bag_bytes=du(out / 'bag'))
    (out / 'result.json').write_text(json.dumps(res, indent=1, default=str))
    idx = FLIGHTS / 'INDEX.md'
    if not idx.exists():
        idx.write_text('| folder | name | result | wall s | max xy m | params | notes |\n|---|---|---|---|---|---|---|\n')
    with open(idx, 'a') as f:
        f.write(f'| {out.name} | {name} | {finished or res["final_phase"]} ({reason}) | {res["wall_s"]} | '
                f'{res["max_xy_from_start_m"]} | `{params}` | {notes} |\n')
    print(f'[flight_run] done: {finished or res["final_phase"]} reason={reason} bag={res["bag_bytes"]/1e6:.1f} MB -> {out}',
          flush=True)
    return 0 if node_rc == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
