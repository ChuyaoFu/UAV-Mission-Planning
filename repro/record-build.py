#!/usr/bin/env python3
"""Save actual dependency versions and generated-resource hashes after build."""
from pathlib import Path
import hashlib,json,subprocess,platform
root=Path('/work')
def output(*args):return subprocess.check_output(args,text=True).strip()
lock=json.loads((root/'repro/upstreams.lock.json').read_text())
revisions={k:output('git','-C',str(root/v['path']),'rev-parse','HEAD') for k,v in lock['repositories'].items()}
files={}
for pattern in ['patches/*.patch','sim/worlds/*.sdf','sim/server.config','cartographer/v4/config/*','config/bridge.yaml']:
 for p in root.glob(pattern):
  if p.is_file():files[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'platform':platform.platform(),'integration_revision':output('git','-C',str(root),'rev-parse','HEAD'),
 'integration_working_tree':output('git','-C',str(root),'status','--short'),
 'upstream_revisions':revisions,'source_lock':lock,
 'system_packages':output('dpkg-query','-W','-f=${Package}\t${Version}\n'),
 'px4_python_packages':output('/opt/px4-venv/bin/python3','-m','pip','freeze'),
 'resource_sha256':files,'clean_machine_flight_certified':False}
out=root/'results/reproduction';out.mkdir(parents=True,exist_ok=True)
(out/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Saved results/reproduction/build-manifest.json')
