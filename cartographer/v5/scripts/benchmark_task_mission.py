"""Desktop-only synthetic bounded workload; never an Orin/flight claim."""
import argparse,json,platform,time
import cv2
import numpy as np
from task_perception import OpeningDetector
from task_surfaces import LandingPlaneDetector,WallTopDetector
from test_task_surfaces import landing_image,down_exposure,forward_exposure
from test_task_perception import OpeningTests
from test_task_executive import ExecutiveTests,target


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--iterations',type=int,default=100)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    if not 1<=args.iterations<=10000: raise ValueError('Bounded iteration count required')
    image,depth,K=OpeningTests().rectangle()
    limage,ldepth,lK=landing_image()
    wimage=np.full((300,400),150,np.uint8);wdepth=np.full(wimage.shape,2.,np.float32);wdepth[:150]=4.
    od,ld,wd=OpeningDetector(),LandingPlaneDetector(),WallTopDetector()
    jobs={'opening_400x300':lambda:od.detect(image,depth,K),
          'landing_400x300':lambda:ld.detect(limage,ldepth,lK,down_exposure(),1.),
          'wall_400x300':lambda:wd.detect(wimage,wdepth,[300,300,200,150],forward_exposure(),1.)}
    result={'synthetic_fixture':True,'original_course_flight_validated':False,
            'orin_validated':False,'platform':platform.platform(),'python':platform.python_version(),
            'opencv':cv2.__version__,'iterations':args.iterations,'timing_ms':{}}
    for name,job in jobs.items():
        for _ in range(3):job()
        samples=[]
        for _ in range(args.iterations):
            start=time.perf_counter();job();samples.append((time.perf_counter()-start)*1000)
        result['timing_ms'][name]={'median':float(np.median(samples)),'p95':float(np.percentile(samples,95)),'max':max(samples)}
    fixture=ExecutiveTests();fixture.setUp()
    centers={'tunnel_entry':(1,0,.7),'tunnel_exit':(2.2,0,.7),'window':(3.6,0,.7),'dock':(5.,0,0),'wall':(2.5,0,.8)}
    intents=[]
    for _ in range(500):
        role=fixture.exec.desired_role
        tracks=[] if role is None or role=='home' else [target(role,center=centers[role],normal=(1,0,0) if role=='wall' else (-1,0,0),origin=fixture.p)]
        intent=fixture.tick(tracks);fixture.follow(intent)
        intents.append({'stamp':fixture.now,**intent.as_dict()})
        if intent.action=='complete': break
    result['synthetic_trace']={'report':fixture.exec.report(),'intents':intents}
    with open(args.output,'w') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='synthetic_trace'},indent=2))
    if not result['synthetic_trace']['report']['competition_tasks_complete']: raise RuntimeError('Synthetic contract trace failed')

if __name__=='__main__':main()
