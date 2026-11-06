"""Synthetic sensor-domain/contact/forward-proof evidence, never a flight."""
import argparse,json,time,platform
import numpy as np
from test_task_contact import ContactTests,NativeLandingExecutiveTests
from test_task_forward import ForwardProofTests


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--iterations',type=int,default=100);args=parser.parse_args()
    if not 1<=args.iterations<=10000:raise ValueError('Bounded iterations required')
    native=NativeLandingExecutiveTests('test_occupied_floor_native_land_disarm_rearm_liftoff_with_preserved_vio')
    native.setUp();native.test_occupied_floor_native_land_disarm_rearm_liftoff_with_preserved_vio()
    forward=ForwardProofTests('test_forward_only_camera_completes_crossing_without_turning_back')
    forward.setUp();forward.test_forward_only_camera_completes_crossing_without_turning_back()
    fixture=ContactTests();fixture.setUp();times=[]
    for i in range(args.iterations+3):
        start=time.perf_counter();cert=fixture.adapter.certify(fixture.guard,fixture.patch,fixture.cal,fixture.s,2.)
        if cert is None:raise RuntimeError(fixture.adapter.last_rejection)
        if i>=3:times.append((time.perf_counter()-start)*1000)
    result={'synthetic_fixture':True,'real_sensor_recordings_used':False,'original_course_flight_validated':False,
            'orin_validated':False,'platform':platform.platform(),
            'native_land_contact_and_liftoff':{'foot_calibration_from_range_samples':native.calibration,
                                            'report':native.last_report,'intents':native.intents},
            'forward_only_crossing':forward.last_report,
            'desktop_contact_refresh_ms':{'iterations':args.iterations,'median':float(np.median(times)),
                                          'p95':float(np.percentile(times,95)),'max':max(times)}}
    with open(args.output,'w') as f:json.dump(result,f,indent=2)
    print(json.dumps(result['desktop_contact_refresh_ms'],indent=2))

if __name__=='__main__':main()
