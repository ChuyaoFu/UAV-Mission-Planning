#!/usr/bin/env python3
"""Hash-checked isolated source patch; no builds, config writes or deployment."""
from pathlib import Path
import argparse,json,hashlib,subprocess
p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('--baseline',choices=['v2','upstream'],default='v2');p.add_argument('--check',action='store_true');a=p.parse_args();r=Path(__file__).resolve().parents[1];root=a.source.resolve()
manifest=r/'reports'/('source-manifest.json' if a.baseline=='v2' else 'upstream-source-manifest.json');patch=r/'patches'/('0001-optional-klt-gftt.patch' if a.baseline=='v2' else '0001-optional-klt-gftt-upstream.patch')
for row in json.loads(manifest.read_text()):
 f=root/row['path'];got=hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None
 if got!=row['baseline_sha256']:raise SystemExit('Refusing unknown or already edited source: '+str(f))
subprocess.run(['patch','--batch','--dry-run','-p1','-i',str(patch)],cwd=root,check=True)
if not a.check:
 subprocess.run(['patch','--batch','-p1','-i',str(patch)],cwd=root,check=True)
 for row in json.loads(manifest.read_text()):assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['changed_sha256']
print('PASS '+a.baseline+' recognized source '+('read-only check' if a.check else 'applied; no build/deployment'))
