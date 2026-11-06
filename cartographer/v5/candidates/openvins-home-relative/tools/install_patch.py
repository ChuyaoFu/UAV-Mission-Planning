#!/usr/bin/env python3
from pathlib import Path
import argparse,json,hashlib,subprocess
p=argparse.ArgumentParser(description='Recognized isolated v2+joint source only; no config/build/deployment');p.add_argument('source',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args();root=a.source.resolve();bundle=Path(__file__).resolve().parents[1];rows=json.loads((bundle/'reports/source-manifest.json').read_text());patch=bundle/'patches/0001-optional-retained-home-yaw-pose.patch'
for row in rows:
 f=root/row['path'];h=hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None
 if h!=row['baseline_sha256']:raise SystemExit('Unknown/already-modified file: '+str(f))
subprocess.run(['patch','--batch','--dry-run','-p1','-i',str(patch)],cwd=root,check=True)
if not a.check:
 subprocess.run(['patch','--batch','-p1','-i',str(patch)],cwd=root,check=True)
 for row in rows:assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['changed_sha256']
print('PASS recognized baseline '+('read-only check' if a.check else 'applied; no config/build/deployment/PX4 changes'))
