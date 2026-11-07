"""Actual high/cruise IPM stream-policy benchmark, no ROS/publication/GT.
New optimized outputs are separate files requiring independent grading.
"""
import argparse,json,time,platform
from pathlib import Path
import numpy as np,cv2
from ipm_engine import IPMEngine

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--mask-dir',required=True);p.add_argument('--variant',default='cold5_freeze1_seedvalidated_240p');p.add_argument('--include-floor-completion',action='store_true');p.add_argument('--max-compute-s',type=float,default=.4);p.add_argument('--unbounded-diagnostic',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);m=[cv2.imread(str(Path(a.mask_dir)/f'self-mask{i}.png'),0)>0 for i in (0,1)];rows=[];current_dataset=None;engine=None
 for path in sorted(Path(a.inputs).glob('*input-*.npz'),key=lambda p:(p.stem.split('-')[0],int(p.stem.rsplit('-',1)[1]))):
  dataset=path.stem.split('-')[0]
  if dataset!=current_dataset:engine=IPMEngine(np.eye(4),*m,'not_required_for_IPM',variant=a.variant,include_floor=a.include_floor_completion);current_dataset=dataset
  d=np.load(path);f={k:d[k] for k in d.files};s=int(f['stamp_ns'])
  for k in ('left_stamp_ns','right_stamp_ns','depth_stamp_ns'):f[k]=s
  packet,meta=engine.evaluate(dict(frames=[f],joint=None),None if a.unbounded_diagnostic else a.max_compute_s);row=dict(input=str(path.resolve()),dataset=dataset,stamp_ns=s,result=meta,budget_s=a.max_compute_s,within_requested_budget=meta['compute_s']<=a.max_compute_s)
  if packet is not None:
   output=out/(f'{dataset}-stream-ipm-{s}.npz');np.savez_compressed(output,**packet);row['output']=str(output.resolve());row['surface_pixels']=int(np.isfinite(packet['inferred_depth']).sum())
  rows.append(row);print(json.dumps(row,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()),flush=True)
 (out/'benchmark.json').write_text(json.dumps(dict(platform=platform.platform(),linux_measured=platform.system()=='Linux',policy=a.variant,model='NEW single-exposure conditional fitted-ground IPM',deadline_enforced=not a.unbounded_diagnostic,rows=rows),indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
if __name__=='__main__':main()
