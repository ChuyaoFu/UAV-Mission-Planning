#!/usr/bin/env python3
"""Repeat the frozen short SITL fixture; stop immediately on any failed flight."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

root=Path('/home/fcy/visual_nav')
config=root/'config/gazebo_vio/estimator_config.yaml'
frozen=hashlib.sha256(config.read_bytes()).hexdigest()
summary=[]
def docker(*args, **kw):
    return subprocess.run(['docker','exec','visual-nav-sim',*args],**kw)
def archive(label):
    out=root/'results'/label
    out.mkdir(exist_ok=True)
    for name in ['baseline-flight.json','vehicle-contacts.json','contact-timeline.json']:
        p=root/'results'/name
        if p.exists(): shutil.copy2(p,out/name)
    shutil.copytree(root/'results/vio-evaluation',out/'vio-evaluation',dirs_exist_ok=True)
    shutil.copy2(config,out/'estimator_config.yaml')
    for name in ['openvins.log','px4-run.log','ev-adapter.log','repeat-flight.log','repeat-contacts.log','repeat-vio.log']:
        p=root/'logs'/name
        if p.exists(): shutil.copy2(p,out/name)
    logs=list((root/'src/PX4-Autopilot/build/px4_sitl_default/rootfs/log').glob('*/*.ulg'))
    if logs: shutil.copy2(max(logs,key=lambda p:p.stat().st_mtime),out/'flight.ulg')
    return out

# Preserve the completed first run before resetting its simulator.
archive('imu-zupt-clearance-cycle1')
for cycle in [2,3]:
    label=f'imu-zupt-clearance-cycle{cycle}'
    print('START',label,flush=True)
    assert hashlib.sha256(config.read_bytes()).hexdigest()==frozen
    status=docker('/work/src/PX4-Autopilot/build/px4_sitl_default/bin/px4-commander','status',capture_output=True,text=True)
    if 'Disarmed' not in status.stdout:
        raise RuntimeError('Refusing restart: vehicle is not confirmed disarmed')
    docker('bash','/work/scripts/stop-sim.sh',check=True)
    subprocess.run(['bash',str(root/'scripts/launch-environment.sh'),'textured','vision'],check=True)
    time.sleep(8)
    # Remove completed outputs so a failed run can never reuse an earlier pass.
    for name in ['baseline-flight.json','vehicle-contacts.json','contact-timeline.json']:
        (root/'results'/name).unlink(missing_ok=True)
    shutil.rmtree(root/'results/vio-evaluation',ignore_errors=True)
    procs=[]
    handles=[]
    for name,args in [
        ('repeat-contacts.log',['/usr/bin/python3','/work/scripts/monitor-contacts.py','--seconds','55']),
        ('repeat-vio.log',['bash','-c',f'source /opt/ros/jazzy/setup.bash; exec /usr/bin/python3 /work/scripts/evaluate-vio.py --seconds 55 --label {label}'])]:
        h=open(root/'logs'/name,'w');handles.append(h)
        procs.append(subprocess.Popen(['docker','exec','-e','GZ_PARTITION=visual_nav_sim','visual-nav-sim',*args],stdout=h,stderr=subprocess.STDOUT))
    time.sleep(2)
    with open(root/'logs/repeat-flight.log','w') as h:
        flight=docker('bash','-c','source /opt/ros/jazzy/setup.bash; source /work/ros_ws/install/setup.bash; export NAV_TEST_REQUIRE_VISION=1 NAV_TEST_BODY_FORWARD=1; exec /usr/bin/python3 /work/scripts/flight-baseline.py',stdout=h,stderr=subprocess.STDOUT,timeout=100)
    for p in procs:p.wait(timeout=70)
    for h in handles:h.close()
    out=archive(label)
    f=json.loads((out/'baseline-flight.json').read_text())
    m=json.loads((out/'vio-evaluation/metrics.json').read_text())
    c=json.loads((out/'vehicle-contacts.json').read_text())
    # Keep the strict monitor result unchanged. Separately report known 1.27 mm
    # floor marking contacts after NAV_LAND; never waive an airborne collision.
    unexpected=[]
    marking=[]
    for pair in c.get('collision_pairs',[]):
        names=pair['collisions']
        if any('floor_20_x_20_ft' in n for n in names):continue
        if (any(n.endswith('::11_guideline_zone_1') for n in names)
            and pair['first_stamp_s']>=f['land_command_stamp_us']*1e-6):
            marking.append(pair)
        else:unexpected.append(pair)
    result={'label':label,'config_sha256':frozen,'flight_exit_code':flight.returncode,
        'flight_passed':f['passed'],'vio_passed':m['passed'],
        'ate_rmse_m':m.get('ate_rmse_m'),'max_error_m':m.get('max_error_m'),
        'contact_monitor_passed':c['passed'],'landing_floor_marking_contacts':marking,
        'unexpected_contacts':unexpected,
        'fixture_checks_passed':flight.returncode==0 and f['passed'] and m['passed'] and c['messages']>0 and not unexpected}
    summary.append(result)
    (out/'repeat-check.json').write_text(json.dumps(result,indent=2))
    (root/'results/frozen-vio-repeats.json').write_text(json.dumps(summary,indent=2))
    print('RESULT',json.dumps(result),flush=True)
    if not result['fixture_checks_passed']:
        print('STOP: investigate this result before any further flight',flush=True)
        break
print('DONE',flush=True)
