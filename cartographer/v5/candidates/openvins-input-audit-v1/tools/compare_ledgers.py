#!/usr/bin/env python3
"""Compare real native input ledgers without printing large payload arrays."""
import argparse,csv,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('left',type=Path);p.add_argument('right',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def load(path):
 with path.open() as f:r=[x for x in csv.reader(f) if x and not x[0].startswith('#')]
 if not r or r[-1][0]!='END':raise ValueError('Incomplete ledger: missing graceful END '+str(path))
 if [int(x[1]) for x in r]!=list(range(len(r))):raise ValueError('Missing/regressing ledger sequence')
 return r
def firstdiff(left,right,view=lambda x:x):
 for i in range(max(len(left),len(right))):
  x=left[i] if i<len(left) else None;y=right[i] if i<len(right) else None
  if x is None or y is None or view(x)!=view(y):return {'index':i,'left_prefix':None if x is None else x[:8],'right_prefix':None if y is None else y[:8]}
 return None
L,R=load(a.left),load(a.right);report={'inputs':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [a.left,a.right]},'kinds':{}}
for kind in ['IMU','CAM','MOTION_NOTICE','INIT']:
 l=[x for x in L if x[0]==kind];r=[x for x in R if x[0]==kind];entry={'counts':[len(l),len(r)],'first_exact_record_difference_excluding_seq':firstdiff(l,r,lambda x:x[:1]+x[2:])}
 if kind=='IMU':entry['first_raw_header_or_payload_difference']=firstdiff(l,r,lambda x:x[5:])
 if kind=='CAM':
  entry['first_timestamp_image_or_mask_difference']=firstdiff(l,r,lambda x:x[5:]);entry['first_imu_watermark_difference']=firstdiff(l,r,lambda x:x[2:5])
 report['kinds'][kind]=entry
l=[x for x in L if x[0] in ['IMU','CAM','MOTION_NOTICE']];r=[x for x in R if x[0] in ['IMU','CAM','MOTION_NOTICE']];report['first_core_input_order_or_value_difference']=firstdiff(l,r,lambda x:x[:1]+x[2:]);report['footer_prefixes']=[L[-1][:5],R[-1][:5]]
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
