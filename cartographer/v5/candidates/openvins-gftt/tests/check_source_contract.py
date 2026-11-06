from pathlib import Path
import sys
base,changed=map(Path,sys.argv[1:]);s=(changed/'ov_core/src/track/TrackKLT.cpp').read_text();old=(base/'ov_core/src/track/TrackKLT.cpp').read_text()
a='void TrackKLT::perform_matching';assert s[s.index(a):s.index('void TrackKLT::extract_grid')].strip()==old[old.index(a):].strip()
a='void TrackKLT::feed_new_camera';b='void TrackKLT::perform_detection_monocular';assert s[s.index(a):s.index(b)]==old[old.index(a):old.index(b)]
for f in ['ov_init/src/init/InertialInitializer.cpp','ov_core/src/track/TrackDescriptor.cpp','ov_msckf/src/update/UpdaterMSCKF.cpp']:
 assert (base/f).read_bytes()==(changed/f).read_bytes(),f
assert s.count('extract_grid(')==4 and '(use_gftt ? mask1 : mask0).clone()' in s
opt=(changed/'ov_msckf/src/core/VioManagerOptions.h').read_text();assert 'klt_detector = "FAST"' in opt and 'klt_detector_timing = false' in opt
assert 'parse_config("klt_detector", klt_detector, false)' in opt
assert '(!use_klt && klt_detector != "FAST")' in opt
assert '0.005' not in (changed/'ov_core/src/track/Grider_GRID.h').read_text()
print('PASS exact original tracking/perform_matching/LK/F-RANSAC; v2 initializer8, descriptors, MSCKF unchanged; 3 detection entrypoints; optional FAST default; no hidden old GFTT fallback')
