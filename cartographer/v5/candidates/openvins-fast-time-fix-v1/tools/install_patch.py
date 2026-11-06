#!/usr/bin/env python3
from pathlib import Path
import argparse,json,hashlib,subprocess
p=argparse.ArgumentParser(description='Apply one-file read-only fast-prediction time fix to recognized source; no state/config/build/deployment');p.add_argument('source',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args();bundle=Path(__file__).resolve().parents[1];meta=json.loads((bundle/'reports/source-hashes.json').read_text());root=a.source.resolve();f=root/meta['path']
if hashlib.sha256(f.read_bytes()).hexdigest()!=meta['baseline_sha256']:raise SystemExit('Unknown or already modified Propagator.cpp; refusing overwrite')
patch=bundle/'patches/0001-fast-prediction-imu-time-endpoint.patch';subprocess.run(['patch','--batch','--dry-run','-p1','-i',str(patch)],cwd=root,check=True)
if not a.check:
 subprocess.run(['patch','--batch','-p1','-i',str(patch)],cwd=root,check=True);assert hashlib.sha256(f.read_bytes()).hexdigest()==meta['candidate_sha256']
print('PASS recognized one-file '+('read-only check' if a.check else 'source patch applied; no build/config/state/flight changes'))
