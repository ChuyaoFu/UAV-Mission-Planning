#!/usr/bin/env python3
"""Simulator-only, pinned CC0 indoor surface appearance benchmark asset."""
import hashlib,json,sys,urllib.request
from pathlib import Path
manifest=json.loads((Path(__file__).resolve().parents[1]/'upstream/room-material.json').read_text());out=Path(sys.argv[1]);out.parent.mkdir(parents=True,exist_ok=True)
if not out.exists() or hashlib.sha256(out.read_bytes()).hexdigest()!=manifest['sha256']:
 data=urllib.request.urlopen(manifest['url'],timeout=30).read();assert hashlib.sha256(data).hexdigest()==manifest['sha256'],'Material content changed';out.write_bytes(data)
print(json.dumps({'asset':manifest['asset'],'path':str(out),'sha256':manifest['sha256']}))
