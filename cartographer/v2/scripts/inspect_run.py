#!/usr/bin/python3
import json,sys
from pathlib import Path
root=Path('/work/cartographer/v2/results')/sys.argv[1]
for name in ['flight.json','run.json']:
 p=root/name
 if p.exists():
  d=json.loads(p.read_text());print(name,{k:v for k,v in d.items() if k not in ['rows','trace','cad_registration','line_registration','window_registration']})
p=root/'evaluation.json'
if p.exists():
 d=json.loads(p.read_text());print('evaluation',d['summary']);g=d['rows']['truth'];print('truth_xyz_span',[max(v[i] for v in g)-min(v[i] for v in g) for i in [1,2,3]])
 h=d['rows']['health'];reasons={v['reason']:sum(x['reason']==v['reason'] for x in h) for v in h};print('health',reasons)
 v=d['rows'].get('visual_update',[]);print('actual_update_counts',len(v),max((x[2] for x in v),default=0))
p=root/'judge.json'
if p.exists():print('judge',json.loads(p.read_text())['result'])
