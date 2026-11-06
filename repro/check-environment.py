#!/usr/bin/python3
"""Check installed files and source provenance; this never arms or starts SITL."""
from pathlib import Path
import importlib,json,subprocess,xml.etree.ElementTree as ET
root=Path('/work');fail=[];checks={}
def check(name,fn):
 try:checks[name]=fn()
 except Exception as e:checks[name]={'error':str(e)};fail.append(name)
def need(path):
 p=root/path
 if not p.is_file():raise FileNotFoundError(str(p))
 return str(path)
for f in ['src/PX4-Autopilot/build/px4_sitl_default/bin/px4',
 'ros_ws/install/setup.bash','cartographer/v2/ros_ws/install/setup.bash',
 'cartographer/v4/carto_ws/install/setup.bash',
 'cartographer/v4/carto_ws/install/cartographer_ros/lib/libcartographer_ros.so',
 'sim/worlds/dronehacks.sdf','sim/server.config',
 'cartographer/v4/config/cad-safe-grid.npz','cartographer/results/cad-grid.npz',
 'cartographer/v4/results/loop-guard-tests.txt',
 'cartographer/v4/results/extrap-after-tests.txt',
 'cartographer/v4/results/optimizer-after-tests.txt']:
 check(f,lambda f=f:need(f))
for module in ['rclpy','cv2','numpy','scipy','rosbag2_py','gz.transport13','gz.msgs10.contacts_pb2','px4_msgs.msg','cartographer_ros_msgs.srv']:
 check(module,lambda module=module:importlib.import_module(module).__name__)
def no_gps():
 tree=ET.parse(root/'cartographer/v4/config/model.sdf')
 assert not tree.findall('.//include'),'Expected self-contained seven-inch model'
 assert not [s for s in tree.findall('.//sensor') if s.get('type') in ['gps','navsat']]
 return 'no NavSat sensor in self-contained model'
check('vehicle_navsat_absent',no_gps)
def revisions():
 lock=json.loads((root/'repro/upstreams.lock.json').read_text());result={}
 for k,v in lock['repositories'].items():
  actual=subprocess.check_output(['git','-C',str(root/v['path']),'rev-parse','HEAD'],text=True).strip()
  assert actual==v['revision'],f'{k}: {actual}'
  result[k]=actual
 return result
check('source_revisions',revisions)
check('agent',lambda:subprocess.check_output(['bash','-c','command -v MicroXRCEAgent'],text=True).strip())
result={'passed':not fail,'failed_checks':fail,'checks':checks,'flight_validated':False}
out=root/'results/reproduction';out.mkdir(parents=True,exist_ok=True)
(out/'environment-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
raise SystemExit(bool(fail))
