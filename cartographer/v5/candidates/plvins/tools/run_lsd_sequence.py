#!/usr/bin/env python3
"""Run the compiled pinned modified-LSD/LBD probe on EuRoC images or sensor NPZ snapshots."""
import argparse,csv,json,subprocess,tempfile
from pathlib import Path
import cv2
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',required=True,type=Path);p.add_argument('--euroc',type=Path);p.add_argument('--npz-dir',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
if bool(a.euroc)==bool(a.npz_dir):p.error('Choose exactly one measured input')
a.output.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix='lsd-probe-') as temp:
 temp=Path(temp);entries=[]
 if a.euroc:
  cam=a.euroc/'mav0/cam0'
  with (cam/'data.csv').open() as f:
   for row in csv.reader(f):
    if not row or row[0].startswith('#'):continue
    im=cv2.imread(str(cam/'data'/row[1]),cv2.IMREAD_GRAYSCALE)
    if im is None:raise ValueError('Missing measured image')
    entries.append((int(row[0]),im))
 else:
  for path in sorted(a.npz_dir.glob('*.npz')):
   with np.load(path,allow_pickle=False) as z:
    # Explicit allowlist; never touch pose, map, depth, inferred-free or truth fields.
    im=z['left'];stamp=int(z['stamp_ns']) if 'stamp_ns' in z.files else int(round(float(z['stamp'])*1e9))
    if im.ndim!=2 or im.dtype!=np.uint8:raise ValueError('Expected measured mono8 left')
    entries.append((stamp,im.copy()))
 entries.sort(key=lambda x:x[0]);manifest=temp/'images.csv'
 with manifest.open('w') as f:
  for i,(stamp,im) in enumerate(entries):
   path=temp/('%06d.pgm'%i)
   if not cv2.imwrite(str(path),im):raise RuntimeError('PGM write failed')
   f.write(str(stamp)+','+str(path)+'\n')
 result=subprocess.run([str(a.probe.resolve()),str(manifest)],capture_output=True,text=True,check=True)
 (a.output/'features.csv').write_text(result.stdout);(a.output/'runtime.txt').write_text(result.stderr)
 rows=list(csv.DictReader(result.stdout.splitlines()));cols=['point_candidates','line_candidates','matched_lines','line_tracks_ge3','total_ms','cpu_ms','orientation_bins']
 report={'input':str(a.euroc or a.npz_dir),'frames':len(rows),'source':'actual pinned C++ modifiedLSD/LBD and matching gates; independent probe keeps all qualifying lines instead of official35+35 replenishment policy','metrics':{c:{'min':min(float(r[c]) for r in rows),'median':float(np.median([float(r[c]) for r in rows])),'p95':float(np.percentile([float(r[c]) for r in rows],95)),'max':max(float(r[c]) for r in rows)} for c in cols} if rows else {},'gap_reset_frames':sum(int(r['gap_reset']) for r in rows),'not_estimator':True,'initialization':None,'trajectory_continuity':None,'limitation':'2D line counts and consecutive-age candidates are not triangulated usable estimator constraints; no fabricated 3D points or navigation output','runtime':result.stderr.strip()}
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
