#!/usr/bin/env python3
"""Fetch exact source revisions and apply canonical patches once, without reset."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / 'repro/upstreams.lock.json').read_text())

def git(path, *args, capture=False):
    return subprocess.run(['git', '-C', str(path), *args], check=True,
                          text=True, stdout=subprocess.PIPE if capture else None)

def fetch(name, spec, submodules=True):
    path = ROOT / spec['path']
    if not path.exists():
        path.mkdir(parents=True)
        git(path, 'init', '-q')
        git(path, 'remote', 'add', 'origin', spec['url'])
    if not (path / '.git').is_dir():
        raise RuntimeError(f'{path} exists but is not a source checkout; preserve it and choose a fresh repository folder')
    # These are intentionally fetched dependency trees, never the project repo.
    subprocess.run(['git', 'config', '--global', '--add', 'safe.directory', str(path)], check=True)
    current = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                             capture_output=True, text=True)
    if current.returncode or current.stdout.strip() != spec['revision']:
        if git(path, 'status', '--porcelain', capture=True).stdout.strip():
            raise RuntimeError(f'Refusing to change a modified dependency checkout: {path}')
        git(path, 'fetch', '--depth=1', 'origin', spec['revision'])
        git(path, 'checkout', '--detach', spec['revision'])
    actual = git(path, 'rev-parse', 'HEAD', capture=True).stdout.strip()
    if actual != spec['revision']:
        raise RuntimeError(f'{name}: wrong revision {actual}')
    if spec.get('submodules') and submodules:
        git(path, 'submodule', 'update', '--init', '--recursive', '--depth=1')
    if spec.get('patch'):
        patch = ROOT / spec['patch']
        forward = subprocess.run(['git', '-C', str(path), 'apply', '--check', str(patch)],
                                 capture_output=True, text=True)
        if forward.returncode == 0:
            git(path, 'apply', str(patch))
            state = 'applied'
        else:
            reverse = subprocess.run(['git', '-C', str(path), 'apply', '--reverse', '--check', str(patch)],
                                     capture_output=True, text=True)
            if reverse.returncode:
                raise RuntimeError(f'{name}: canonical patch neither applies nor matches existing changes:\n{forward.stderr}')
            state = 'already applied'
        print(f'{name}: {actual}; patch {state}', flush=True)
    else:
        print(f'{name}: {actual}', flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only', nargs='+', choices=list(LOCK['repositories']))
    ap.add_argument('--no-submodules', action='store_true', help='source/patch audit only; insufficient for a PX4 build')
    args = ap.parse_args()
    for name in args.only or LOCK['repositories']:
        fetch(name, LOCK['repositories'][name], not args.no_submodules)
