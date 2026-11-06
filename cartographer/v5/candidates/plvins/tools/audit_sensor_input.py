#!/usr/bin/env python3
"""Read only specified measured Image/Imu topics; describe timing/texture without changing data."""
import argparse,bisect,csv,json,math,time
from pathlib import Path
from importlib.metadata import version
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore,Stores

def stats(x):
 a=np.asarray(x,float)
 return {'n':len(a),'min':float(a.min()),'median':float(np.median(a)),'p95':float(np.percentile(a,95)),'max':float(a.max())} if len(a) else {'n':0}
def image_quality(image):
 hist=np.bincount(image.reshape(-1),minlength=256);p=hist[hist>0]/image.size
 small=image[::4,::4].astype(np.int16)
 return {'gray_mean':float(image.mean()),'gray_std':float(image.std()),'dominant_gray_fraction':float(hist.max()/image.size),'entropy_bits':float(-np.sum(p*np.log2(p))),'sampled_gradient_ge2_fraction':float((np.abs(np.diff(small,axis=1))>=2).mean()) if small.shape[1]>1 else 0.}
def audit(source,output,image_topic='/cam0/image_raw',imu_topic='/imu0'):
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 stamps={'image':[],'imu':[]};ages={'image':[],'imu':[]};frames={'image':set(),'imu':set()};reversed_arrivals={'image':0,'imu':0};invalid={'image':0,'imu':0};rows=[];acc=[];gyro=[];imu_first=None;initial_a=[];initial_w=[];dimensions=set()
 with AnyReader([Path(source)],default_typestore=get_typestore(Stores.ROS2_JAZZY)) as reader:
  connections=[c for c in reader.connections if c.topic in [image_topic,imu_topic]]
  if {c.topic for c in connections}!={image_topic,imu_topic}:raise ValueError('Both explicit measured sensor topics required')
  for c,record_ns,raw in reader.messages(connections=connections):
   m=reader.deserialize(raw,c.msgtype);kind='image' if c.topic==image_topic else 'imu';ns=int(m.header.stamp.sec)*10**9+int(m.header.stamp.nanosec)
   if stamps[kind] and ns<=stamps[kind][-1]:reversed_arrivals[kind]+=1
   stamps[kind].append(ns);ages[kind].append((record_ns-ns)*1e-9);frames[kind].add(m.header.frame_id)
   if kind=='image':
    if m.encoding!='mono8' or m.step<m.width or len(m.data)<m.step*m.height or m.width<=0 or m.height<=0:invalid[kind]+=1;continue
    dimensions.add((m.width,m.height));image=np.asarray(m.data,dtype=np.uint8)[:m.step*m.height].reshape(m.height,m.step)[:,:m.width]
    rows.append({'stamp_ns':ns,**image_quality(image)})
   else:
    a=np.array([m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z],float);w=np.array([m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z],float)
    if not np.isfinite(a).all() or not np.isfinite(w).all():invalid[kind]+=1;continue
    acc.append(float(np.linalg.norm(a)));gyro.append(float(np.linalg.norm(w)))
    if imu_first is None:imu_first=ns
    if ns-imu_first<=2*10**9:initial_a.append(a.tolist());initial_w.append(w.tolist())
 if rows:
  with (output/'image-quality.csv').open('w') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 ordered={k:sorted(s) for k,s in stamps.items()};imu=ordered['imu'];image=ordered['image'];brackets=[]
 for ns in image:
  left=bisect.bisect_left(imu,ns)-1;right=bisect.bisect_right(imu,ns)
  brackets.append((ns,((ns-imu[left])*1e-9 if left>=0 else None),((imu[right]-ns)*1e-9 if right<len(imu) else None)))
 interval={k:stats(np.diff(s)*1e-9) for k,s in ordered.items()}
 report={'source':str(source),'rosbags_version':version('rosbags'),'allowed_topics':[image_topic,imu_topic],'counts':{k:len(s) for k,s in stamps.items()},'frame_ids':{k:sorted(v) for k,v in frames.items()},'image_resolutions':sorted(dimensions),'invalid_messages':invalid,'header_nonincreasing_in_record_order':reversed_arrivals,'duplicate_header_stamps':{k:len(s)-len(set(s)) for k,s in stamps.items()},'first_last_header_ns':{k:[s[0],s[-1]] if s else [] for k,s in ordered.items()},'sorted_header_interval_s':interval,'record_minus_header_s':{k:stats(v) for k,v in ages.items()},'strict_imu_bracketing':{'images_without_earlier_imu':sum(a is None for _,a,_ in brackets),'images_without_later_imu':sum(b is None for _,_,b in brackets),'earlier_distance_s':stats([a for _,a,_ in brackets if a is not None]),'later_distance_s':stats([b for _,_,b in brackets if b is not None])},'imu_acc_norm_m_s2':stats(acc),'imu_gyro_norm_rad_s':stats(gyro),'initial_2s_measured_medians':{'acc_vector_m_s2':np.median(initial_a,axis=0).tolist() if initial_a else [],'gyro_vector_rad_s':np.median(initial_w,axis=0).tolist() if initial_w else [],'interpretation':'Not assumed stationary; no expected axis/gravity direction or truth alignment inferred'},'image_quality':{k:stats([r[k] for r in rows]) for k in ['gray_std','dominant_gray_fraction','entropy_bits','sampled_gradient_ge2_fraction']},'near_uniform_image_stamps_ns':[r['stamp_ns'] for r in rows if r['dominant_gray_fraction']>=.999 and r['gray_std']<1.],'interpretation':'Texture statistics are diagnostics, not accepted features or a navigation gate. Sorted-header output bags have record_minus_header=0 by construction; this is not original transport latency. Strict IMU bracketing matches PLVINS image-admission boundary; edge frames may be unbracketed. No axis/unit conversion, resampling, pose/depth/truth input or gate change.'}
 (output/'input-quality.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--image-topic',default='/cam0/image_raw');p.add_argument('--imu-topic',default='/imu0');a=p.parse_args();print(json.dumps(audit(a.source,a.output,a.image_topic,a.imu_topic),indent=2))
