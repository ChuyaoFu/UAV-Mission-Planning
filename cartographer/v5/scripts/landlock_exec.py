#!/usr/bin/python3
"""Kernel-enforced filesystem allowlist for navigation, inherited by children.
The simulator/evaluator are outside this profile. No scene/CAD/result mounts.
"""
import ctypes,os,sys
from pathlib import Path
libc=ctypes.CDLL(None,use_errno=True)
class Ruleset(ctypes.Structure):_fields_=[('handled_access_fs',ctypes.c_uint64)]
class PathRule(ctypes.Structure):_pack_=1;_fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int32)]
READ=(1<<0)|(1<<2)|(1<<3);ALL=(1<<13)-1
abi=libc.syscall(444,0,0,1)
if abi<1:raise SystemExit('Landlock unavailable: refuse unisolated navigation')
rs=Ruleset(ALL);fd=libc.syscall(444,ctypes.byref(rs),ctypes.sizeof(rs),0)
if fd<0:raise OSError(ctypes.get_errno(),'landlock_create_ruleset')
offline_stereo=sys.argv[1:2]==['--offline-stereo']
if offline_stereo:sys.argv.pop(1)
offline_model=sys.argv[1:2]==['--offline-model']
if offline_model:sys.argv.pop(1)
paths=['/usr','/opt','/lib','/lib64','/etc','/proc','/sys','/dev','/work/ros_ws','/work/cartographer/v2/ros_ws','/work/cartographer/v3/carto_ws','/work/cartographer/v5/ego_ws','/work/cartographer/v5/config']
flight_modules=['navigate','view_memory','relative_guideline','guideline_node','online_map','perception_worker','fusion','height_odometry','frames','robot_geometry','camera_geometry','stereo_depth','self_mask','stereo_quality','depth_cloud','mapping_cloud','down_range','sensor_frames','slam_cloud_view','ego_launch','gcs-monitor']
flight_modules += ['competition_navigate','competition_inputs','task_executive','task_observations','task_perception','task_perception_helper','task_surfaces','task_landing','task_contact','task_safety']
flight_modules += ['planar_stereo','planar_evidence_runtime','planar_evidence_node','bounded_yaw']
paths += ['/work/cartographer/v5/scripts/'+name+'.py' for name in flight_modules]
for key in ['STEREO_CALIBRATION','ROBOT_CALIBRATION']:
 if os.environ.get(key):
  calibration=Path(os.environ[key]).resolve()
  if not calibration.is_file():raise ValueError('Calibration must be an existing file: '+key)
  paths.append(str(calibration))
if offline_model:paths += ['/work/cartographer/v5/inference-venv','/work/cartographer/v5/models','/work/cartographer/v5/upstream/PromptDA','/work/cartographer/v5/scripts/benchmark_completion.py','/work/cartographer/v5/mission-output/completion-inputs']
if offline_stereo:paths += ['/work/cartographer/v5/inference-venv','/work/cartographer/v5/models/LAS2_S.pth','/work/cartographer/v5/upstream/LiteAnyStereo','/work/cartographer/v5/scripts/benchmark_learned_stereo.py','/work/cartographer/v5/mission-output/stereo-inputs']
for path in paths+['/work/cartographer/v5/scripts','/tmp','/root/.ros','/work/cartographer/v5/mission-output']:
 if not Path(path).exists():continue
 access=ALL if path in ['/tmp','/root/.ros','/dev'] else READ
 if path=='/work/cartographer/v5/mission-output':access=ALL & ~READ
 if path=='/work/cartographer/v5/scripts':access=1<<3
 if not Path(path).is_dir():access &= ~(1<<3)
 f=os.open(path,os.O_PATH|os.O_CLOEXEC);rule=PathRule(access,f)
 if libc.syscall(445,fd,1,ctypes.byref(rule),0)<0:raise OSError(ctypes.get_errno(),'landlock_add_rule '+path)
 os.close(f)
if libc.prctl(38,1,0,0,0)<0 or libc.syscall(446,fd,0)<0:raise OSError(ctypes.get_errno(),'landlock_restrict_self')
os.close(fd)
if sys.argv[1:] == ['--probe']:
 import json
 denied={}
 for path in ['/work/course_source/dronehacks-course-v3.obj','/work/cartographer/results/cad-grid.npz','/work/cartographer/v4/config/stages.json','/work/cartographer/v5/evaluation/scenario.json','/work/cartographer/v5/scripts/generate_unknown_scene.py','/work/cartographer/v5/mission-output/episode-index.json','/work/cartographer/v5/scripts/evaluation_geometry.py','/work/cartographer/v5/models/LAS2_S.pth','/work/cartographer/v5/mission-output/las2-source/benchmark.json']:
  try:open(path,'rb');denied[path]=False
  except PermissionError:denied[path]=True
  except FileNotFoundError:denied[path]='absent'
 print(json.dumps({'landlock_abi':abi,'scene_reads_denied':denied}));raise SystemExit(0 if all(denied.values()) else 1)
if len(sys.argv)<2:raise SystemExit('command required')
os.execvp(sys.argv[1],sys.argv[1:])
