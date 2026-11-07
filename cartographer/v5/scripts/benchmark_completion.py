#!/usr/bin/python3
"""Offline learned depth-completion benchmark. Never publishes flight input.
Only camera images and measured stereo enter the model. Inferred values are
saved separately and are not certified observed free space.
"""
import os,sys,time,json,resource,hashlib,argparse
from pathlib import Path
import numpy as np,cv2,torch
from scipy.ndimage import distance_transform_edt
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'upstream/PromptDA'))
from promptda.promptda import PromptDA
p=argparse.ArgumentParser();p.add_argument('inputs');p.add_argument('output');p.add_argument('--prompt-mode',choices=['sparse','nearest'],default='sparse');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1);checkpoint=root/'models/promptda-vits.ckpt';model=PromptDA(encoder='vits',ckpt_path=str(checkpoint)).eval();rows=[]
for path in sorted(Path(a.inputs).glob('frame-*.npz')):
 d=np.load(path);image=d['image'];measured=d['measured_stereo'];valid=np.isfinite(measured)&(measured>=.3)&(measured<=2.4);values=np.where(valid,measured,0.).astype(np.float32);weights=cv2.resize(valid.astype(np.float32),(128,80),interpolation=cv2.INTER_AREA);prompt=cv2.resize(values,(128,80),interpolation=cv2.INTER_AREA)/np.maximum(weights,1e-6);prompt[weights<.25]=0.
 if a.prompt_mode=='nearest' and (prompt>0).any():
  nearest=distance_transform_edt(prompt<=0,return_distances=False,return_indices=True);prompt=prompt[tuple(nearest)]
 if np.count_nonzero(prompt)<80:rows.append({'file':path.name,'rejected':'Insufficient actual stereo anchors'});continue
 x=torch.from_numpy(cv2.resize(image,(448,280)).astype(np.float32)/255.).permute(2,0,1).unsqueeze(0);z=torch.from_numpy(prompt).unsqueeze(0).unsqueeze(0);start=time.monotonic()
 with torch.inference_mode():prediction=model.predict(x,z);prediction=torch.nn.functional.interpolate(prediction,size=measured.shape,mode='bilinear',align_corners=False)[0,0].cpu().numpy()
 elapsed=time.monotonic()-start;error=np.abs(prediction[valid]-measured[valid]);np.savez_compressed(out/path.name,inferred_depth=prediction,stamp=float(d['stamp']));rows.append({'file':path.name,'stamp':float(d['stamp']),'cpu_seconds':elapsed,'measured_anchor_pixels':int(valid.sum()),'measured_anchor_abs_error_median_m':float(np.median(error)),'measured_anchor_abs_error_p90_m':float(np.percentile(error,90)),'inferred_min_m':float(prediction.min()),'inferred_max_m':float(prediction.max())});print(json.dumps(rows[-1]),flush=True)
summary={'model':'PromptDA Small','model_parameters':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'input_size':[448,280],'prompt_size':[128,80],'threads':2,'prompt_mode':a.prompt_mode,'device':'CPU development host; not Orin timing','maximum_process_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,'navigation_enabled':False,'interpretation':'Model inference only; predicted depth is not physical range evidence or a certified free volume','frames':rows};(out/'completion-benchmark.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
