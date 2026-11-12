#!/usr/bin/env python3
"""Team 13 recorded flight for the web "Run saved file" button (copy to the workspace root).

Runs claude_dev/tools/flight_run.py with claude_dev/flights/next_flight.json, so one click gives
the lab's managed Run (code, evidence, recording, grade) AND our own flights/<stamp>_<name>/ record.
A managed Run executes a READ-ONLY snapshot of the project, so flight records go to the live
workspace (or another writable place), never into the snapshot.
"""
import os, runpy, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.join(HERE, 'claude_dev') if os.path.isdir(os.path.join(HERE, 'claude_dev')) else os.path.dirname(HERE)


def writable_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, '.write_probe')
        with open(probe, 'w') as f:
            f.write('ok')
        os.remove(probe)
        return True
    except OSError:
        return False


if 'FLIGHTS_DIR' not in os.environ:
    for cand in ('/var/lib/flyx-lab/workspace/claude_dev/flights', os.path.join(DEV, 'flights'),
                 '/var/lib/flyx-lab/companion-output/claude_flights', '/tmp/claude_flights'):
        if writable_dir(cand):
            os.environ['FLIGHTS_DIR'] = cand
            break
cfg = os.path.join(os.environ.get('FLIGHTS_DIR', ''), 'next_flight.json')
os.chdir(DEV)
sys.argv = [os.path.join(DEV, 'tools', 'flight_run.py')] + (['--config', cfg] if os.path.isfile(cfg) else [])
print(f'[claude_flight] code {DEV}  records -> {os.environ.get("FLIGHTS_DIR")}  config {sys.argv[1:]}', flush=True)
runpy.run_path(sys.argv[0], run_name='__main__')
