"""Run on Linux/Orin for genuine compute/peakRSS results. No ROS/publication.
Fixtures are bounded causal snapshots, not runtime episode dependencies.
"""
import argparse,json,time,platform,hashlib
from pathlib import Path
import numpy as np,cv2,yaml
from engine import Engine

def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
def fixture(path):
 d=np.load(path);frames=[]
 for i,s in enumerate(d['stamps']):
  prefix=f'f{i}_';frames.append({k[len(prefix):]:d[k] for k in d.files if k.startswith(prefix)})
 return dict(frames=frames,joint=d['joint']),d

def main():
 p=argparse.ArgumentParser();p.add_argument('--fixtures',required=True);p.add_argument('--rig',required=True);p.add_argument('--mask-dir',required=True);p.add_argument('--lbd-library',required=True);p.add_argument('--output',required=True);p.add_argument('--engine-variant',choices=['frozen400p','cold5_freeze1_400p','cold5_freeze1_seedvalidated_400p'],default='frozen400p');p.add_argument('--max-compute-s',type=float,default=.4);p.add_argument('--unbounded-diagnostic',action='store_true');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rig=yaml.safe_load('\n'.join(l for l in Path(a.rig).read_text().splitlines() if not l.startswith('%YAML')));TIC=np.array(rig['cam0']['T_imu_cam']);m0=cv2.imread(str(Path(a.mask_dir)/'self-mask0.png'),0)>0;m1=cv2.imread(str(Path(a.mask_dir)/'self-mask1.png'),0)>0;engine=Engine(TIC,m0,m1,a.lbd_library,variant=a.engine_variant);rows=[]
 for path in sorted(Path(a.fixtures).glob('job-*.npz'),key=lambda p:int(p.stem.split('-')[1])):
  job,d=fixture(path);t=time.monotonic();packet,stats=engine.evaluate(job,None if a.unbounded_diagnostic else a.max_compute_s);elapsed=time.monotonic()-t;row=dict(file=path.name,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),wall_s=elapsed,stats=stats,budget_s=a.max_compute_s,within_requested_budget=elapsed<=a.max_compute_s)
  if packet is not None:
   z=packet['inferred_depth'];expected=d['expected_surface'];finite=np.isfinite(z);oldfinite=np.isfinite(expected);both=finite&oldfinite;row.update(surface_pixels=int(finite.sum()),different_surface_mask_pixels=int(np.count_nonzero(finite!=oldfinite)),different_surface_float32_bits=int(np.count_nonzero(z.view(np.uint32)!=expected.view(np.uint32))),max_surface_abs_difference_m=float(abs(z[both]-expected[both]).max()) if both.any() else None,current_posterior_stamp_ns=packet['same_posterior_latest_stamp_ns']);np.savez_compressed(out/(path.stem+'-actual-output.npz'),**packet)
  rows.append(row);print(json.dumps(row,default=serial),flush=True)
 report=dict(platform=platform.platform(),system=platform.system(),linux_measured=platform.system()=='Linux',opencv=cv2.__version__,numpy=np.__version__,model=a.engine_variant+' native LSD/LBD actual same-posterior geometry',deadline_enforced=not a.unbounded_diagnostic,diagnostic_no_ROS_or_free_publication=True,rows=rows);(out/'benchmark.json').write_text(json.dumps(report,indent=2,default=serial)+'\n');(out/'opencv-build.txt').write_text(cv2.getBuildInformation())
if __name__=='__main__':main()
