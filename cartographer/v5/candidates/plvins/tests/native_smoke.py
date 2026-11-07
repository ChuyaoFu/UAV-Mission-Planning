#!/usr/bin/env python
from __future__ import print_function
import os,tempfile,subprocess,csv,sys,shutil
probe=sys.argv[1];root=tempfile.mkdtemp(prefix='plvins-lsd-smoke-')
try:
 manifest=os.path.join(root,'images.csv')
 with open(manifest,'w') as f:
  for i in range(7):
   width,height=320,240;data=bytearray([175]*(width*height))
   if i<5:
    for x in [40,100,160,220,280]:
     for y in range(30,210):
      for dx in [-1,0,1]:data[y*width+x+dx]=20
   path=os.path.join(root,str(i)+'.pgm')
   with open(path,'wb') as image:image.write(('P5\n%d %d\n255\n'%(width,height)).encode('ascii'));image.write(data)
   f.write(str(1000000000+i*50000000)+','+path+'\n')
 out=subprocess.check_output([probe,manifest]);out=out.decode('ascii')
 rows=list(csv.DictReader(out.splitlines()))
 assert len(rows)==7 and int(rows[0]['line_candidates'])>0
 assert int(rows[5]['line_candidates'])==0 and int(rows[6]['line_candidates'])==0
 assert any(int(r['matched_lines'])>0 for r in rows[1:5])
 print('PASS compiled modifiedLSD/LBD detect, match and blank-frame recovery on this container architecture')
finally:shutil.rmtree(root)
