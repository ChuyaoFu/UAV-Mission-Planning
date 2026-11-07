import cv2,numpy as np
def extract(res):
 im=(res['label400']==1).astype(np.uint8)*255;a=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(im)[0];lines=[]
 if a is None:return lines
 for line in a.reshape(-1,4):
  p,q=line.reshape(2,2);length=np.linalg.norm(q-p)
  if length<40:continue
  t=(q-p)/length;n=np.array([-t[1],t[0]]);sides=[]
  for sign in (-1,1):
   xy=np.rint(p[None,:]+np.linspace(.05,.95,61)[:,None]*(q-p)+sign*3*n).astype(int);valid=(xy[:,0]>=2)&(xy[:,0]<im.shape[1]-2)&(xy[:,1]>=2)&(xy[:,1]<im.shape[0]-2);v=res['label400'][xy[valid,1],xy[valid,0]];sides.append((float((v==1).mean()) if len(v) else 0.,float((v==2).mean()) if len(v) else 0.))
  support=max(min(sides[0][0],sides[1][1]),min(sides[1][0],sides[0][1]))
  if support>=.8:lines.append(dict(line=(line+.5)*2-.5,length=float(length*2),side_support=support))
 return sorted(lines,key=lambda x:-x['length'])[:8]
