#!/usr/bin/env python3
"""Independent SITL competition runner. No reset, no navigation truth inputs."""
import argparse,fcntl,hashlib,json,os,re,shlex,shutil,signal,subprocess,sys,time
from pathlib import Path
from task_uncertainty import UncertaintyPolicy
from window_video import WindowVideo

SETUPS=['/opt/ros/jazzy/setup.bash','/work/ros_ws/install/setup.bash','/work/cartographer/v3/carto_ws/install/setup.bash','/work/cartographer/v5/ego_ws/install/setup.bash']
TOPICS=['/clock','/vio/left/image_raw','/vio/right/image_raw','/vio/imu','/nav/measured_depth','/nav/camera_pose','/nav/body_odom','/carto/accepted_odom','/carto/down_range','/carto/localization_health','/grid_map/metadata','/grid_map/knowledge','/grid_map/pipeline_health','/stereo/health','/carto/depth_health','/mapping/health','/position_cmd','/planning/bspline','/carto/mission_state','/ov_msckf/odomimu','/ov_msckf/visual_update_health','/fmu/out/vehicle_status_v1','/fmu/out/vehicle_land_detected','/fmu/out/vehicle_command_ack','/fmu/out/vehicle_local_position_v1','/fmu/in/vehicle_command','/fmu/in/trajectory_setpoint','/fmu/in/vehicle_visual_odometry','/nav/freeze_global_height','/nav/home_relative_adapter_health','/nav/home_relative_odomimu','/ov_msckf/home_relative_body_reference']
TOPICS += ['/vio/in_motion','/stereo/depth/image','/nav/depth_evidence','/nav/structural_health','/nav/planar_health','/ov_msckf/poseimu','/ov_msckf/home_relative_body_pose','/ov_msckf/joint_clone_states']


def parser():
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--seconds',type=int,default=600);p.add_argument('--root',default='/work/cartographer/v5')
    p.add_argument('--fixture',help='Private independent fixture under ROOT/evaluation; never passed to flight')
    p.add_argument('--tof-error-m',type=float,required=True);p.add_argument('--tof-extrinsic-error-m',type=float,required=True);p.add_argument('--stereo-error-m',type=float,required=True)
    p.add_argument('--uncertainty-admission-m',type=float,default=.20);p.add_argument('--map-pose-error-m',type=float,required=True)
    p.add_argument('--allow-conditional-free',action='store_true')
    p.add_argument('--complete-task',action='store_true',help='Default off: typed mechanical complete native task')
    p.add_argument('--contact-rig-error-m',type=float)
    p.add_argument('--retained-support-max-age-s',type=float,default=120.)
    p.add_argument('--foot-geometry',help='ROBOT geometry JSON inside config, observe-only or explicit complete-task')
    p.add_argument('--observe-only',action='store_true');p.add_argument('--dry-run',action='store_true');return p


def commands(a):
    root=Path(a.root);scripts=root/'scripts';out=root/'results'/a.label;mission=root/'mission-output'/(a.label+'.json');bag=root/'data'/a.label
    flight=['env','SITL_AUTOMATION_AUTHORIZED=1','PYTHONDONTWRITEBYTECODE=1','python3',str(scripts/'competition_landlock_exec.py'),'python3',str(scripts/'competition_navigate.py'),'--output',str(mission),'--mission-budget-s',str(a.seconds),'--tof-error-m',str(a.tof_error_m),'--tof-extrinsic-error-m',str(a.tof_extrinsic_error_m),'--stereo-error-m',str(a.stereo_error_m),'--uncertainty-admission-m',str(a.uncertainty_admission_m),'--map-pose-error-m',str(a.map_pose_error_m)]
    if a.observe_only:flight.append('--observe-only')
    if a.allow_conditional_free:flight.append('--allow-conditional-free')
    if a.foot_geometry:
        if not (a.observe_only or a.complete_task):raise ValueError('Mechanical foot native mode requires explicit complete-task')
        config=(root/'config').resolve();geometry=Path(a.foot_geometry)
        if not geometry.is_absolute():geometry=config/geometry
        geometry=geometry.resolve()
        if config not in geometry.parents:raise ValueError('Only ROBOT foot geometry under config may enter flight observer')
        flight+=['--foot-geometry',str(geometry)]
    if a.complete_task:
        if not a.foot_geometry or a.contact_rig_error_m is None or not 0<=a.contact_rig_error_m<=.02:raise ValueError('Complete task requires explicit ROBOT geometry and complete rig-point error')
        flight+=['--complete-task','--contact-rig-error-m',str(a.contact_rig_error_m),'--retained-support-max-age-s',str(a.retained_support_max_age_s)]
    return {'flight':flight,'record':['ros2','bag','record','--storage','mcap','--storage-preset-profile','zstd_fast','--use-sim-time','--disable-keyboard-controls','--max-cache-size','134217728','-o',str(bag),'--topics',*TOPICS],
            'independent':['python3',str(scripts/'record_competition.py'),'--seconds',str(a.seconds*3+600),'--keepalive-file',str(out/'intervention-needed.json'),'--output',str(out/'independent.json'),'--ready-file',str(out/'independent-ready.json'),'--model-sdf',str(root/'config/model.sdf')],
            'scorer':['python3',str(scripts/'score_competition.py'),'--recording',str(out/'independent.json'),'--fixture',str(a.fixture),'--controller',str(out/'frozen-controller.json'),'--output',str(out/'competition-score.json')]}



def native_status_fields(text):
    text=re.sub(r'\x1b\[[0-9;]*m','',text)
    armed=re.search(r'\barming_state:\s*(\d+)\b',text);stamp=re.search(r'\btimestamp:\s*(\d+)\b',text)
    return (None if armed is None else int(armed[1]),None if stamp is None else int(stamp[1]))



def request_capture_flush(jobs,flight,video):
    """Disk emergency: flush owned recorders only; never flight or FCU."""
    for proc in jobs:
        if proc is not flight and proc.poll() is None:
            try:os.killpg(proc.pid,signal.SIGINT)
            except ProcessLookupError:pass
    if video is not None and video.p.poll() is None:
        try:video.p.send_signal(signal.SIGINT)
        except ProcessLookupError:pass


def retain_until_safe(flight,status_reader,heartbeat,sleep=time.sleep,clock=time.monotonic):
    """Recording stays live; no control, cancellation, cleanup or native action."""
    entered=False;previous_disarm=None;last_emit=-float('inf')
    while True:
        try:
            text=status_reader();arming,stamp=native_status_fields(text);error=None
        except Exception as e:arming=stamp=None;error=str(e)
        live=flight.poll() is None
        if not entered and not live and arming==1 and stamp is not None:return {'entered':False,'resolved':True}
        resolved=bool(entered and not live and arming==1 and stamp is not None and previous_disarm is not None and stamp>previous_disarm)
        state={'entered':True,'resolved':resolved,'intervention_needed':not resolved,'cleanup_allowed':resolved,
               'flight_alive':live,'flight_pid':flight.pid,'native_arming_state':arming,'native_stamp_us':stamp,
               'native_read_error':error,'reset_permitted':False,'control_ownership_changed':False}
        if resolved or clock()-last_emit>=15.:
            heartbeat(state);last_emit=clock()
        if resolved:return state
        entered=True;previous_disarm=stamp if not live and arming==1 and stamp is not None else None
        sleep(1.)


def main(argv=None):
    a=parser().parse_args(argv)
    if not a.label or not re.fullmatch('[A-Za-z0-9-]+',a.label):raise ValueError('Bounded alphanumeric/hyphen label required')
    if not 60<=a.seconds<=1800:raise ValueError('Mission seconds outside bounded domain')
    UncertaintyPolicy(admission_limit_m=a.uncertainty_admission_m,map_pose_error_m=a.map_pose_error_m)
    if not 0<a.tof_error_m<=.02 or not 0<=a.tof_extrinsic_error_m<=.02 or not 0<a.stereo_error_m<=.02:
        raise ValueError('Explicit sensor error domain invalid; numerical admission is not accuracy qualification')
    cmd=commands(a)
    if a.dry_run:print(json.dumps({'commands':cmd,'truth_path_passed_to_flight':False,'executes_nothing':True},indent=2));return 0
    if not (Path('/.dockerenv').exists() and os.environ.get('ROS_DOMAIN_ID')=='42' and os.environ.get('SITL_AUTOMATION_AUTHORIZED')=='1' and os.environ.get('HARDWARE_MODE','0')!='1' and os.environ.get('USE_SIM_TIME','1')=='1'):
        raise PermissionError('Explicit authorized container SITL/simulation clock required')
    root=Path(a.root).resolve();out=root/'results'/a.label;mission=root/'mission-output'/(a.label+'.json');bag=root/'data'/a.label;scripts=root/'scripts'
    if root!=Path('/work/cartographer/v5'):raise ValueError('Actual run must match the kernel isolation profile root /work/cartographer/v5')
    if not a.observe_only:
        if not a.fixture:raise ValueError('Full competition requires private independent fixture')
        fixture=Path(a.fixture).resolve(strict=True)
        if root/'evaluation' not in fixture.parents:raise ValueError('Private fixture must be outside navigation allowlist under ROOT/evaluation')
    lock=open('/work/data/flight-smoke.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir(parents=True,exist_ok=False);mission.parent.mkdir(exist_ok=True);bag.parent.mkdir(exist_ok=True)
    if mission.exists() or bag.exists():raise ValueError('Existing trial artifacts cannot be overwritten')
    px=Path('/work/src/PX4-Autopilot/build/px4_sitl_default/bin')
    def native_status():return subprocess.check_output([str(px/'px4-listener'),'vehicle_status','-n','1'],text=True,timeout=8)
    def disarmed():return bool(re.search(r'arming_state:\s*1\b',native_status()))
    if not disarmed():raise RuntimeError('Actual disarm required; runner never resets or takes over an armed vehicle')
    for key in ['EKF2_GPS_CTRL','EKF2_EV_CTRL','MPC_LAND_SPEED']:
        raw=subprocess.check_output([str(px/'px4-param'),'show',key],text=True,timeout=8);(out/(key+'.txt')).write_text(raw)
        text=re.sub(r'\x1b\[[0-9;]*m','',raw);match=re.search(re.escape(key)+r'.*?:\s*([-+0-9.eE]+)',text,re.S)
        if match is None:raise RuntimeError('Cannot verify actual parameter '+key)
        value=float(match[1])
        if key=='EKF2_GPS_CTRL' and value!=0:raise RuntimeError('GPS fusion must be disabled')
        if key=='EKF2_EV_CTRL' and value==0:raise RuntimeError('Actual external vision fusion required')
        if key=='MPC_LAND_SPEED' and not a.observe_only and not 0<value<=.25:raise RuntimeError('Actual landing speed exceeds unchanged contact monitor; no threshold relaxation')
    jobs=[];handles=[];flight=None;failure=None;video=None;video_info=None;fpv=None;intervention={};capture_stopped_for_disk=False
    setup='set -e; '+''.join('source '+shlex.quote(s)+'; ' for s in SETUPS)
    def job(name,args):
        handle=open(out/name,'w');handles.append(handle)
        proc=subprocess.Popen(['bash','-c',setup+'exec '+shlex.join(args)],stdout=handle,stderr=subprocess.STDOUT,start_new_session=True);jobs.append(proc);return proc
    def runlog(name,args):
        with open(out/name,'w') as handle:return subprocess.run(['bash','-c',setup+'exec '+shlex.join(args)],stdout=handle,stderr=subprocess.STDOUT,timeout=30,check=True)
    def stop(proc):
        if proc.poll() is None:os.killpg(proc.pid,signal.SIGINT)
        try:proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=10)
    try:
        runlog('filesystem-isolation.json',['python3',str(scripts/'competition_landlock_exec.py'),'--probe'])
        runlog('input-audit.log',['python3',str(scripts/'audit_navigation.py'),str(out/'input-audit.json')])
        shutil.copytree(root/'config',out/'config',ignore=shutil.ignore_patterns('._*'))
        shutil.copytree(scripts,out/'scripts',ignore=shutil.ignore_patterns('__pycache__','._*'))
        hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['scripts','config'] for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p) and not p.name.startswith('._')}
        (out/'source-hashes.json').write_text(json.dumps(hashes,indent=2));(out/'arguments.json').write_text(json.dumps(vars(a),indent=2))
        rec=job('record.log',cmd['record'])
        deadline=time.monotonic()+15
        while not list(bag.glob('*.mcap')):
            if rec.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Recorder unavailable; no arming')
            time.sleep(.1)
        video=WindowVideo(out/'videos',a.label)
        independent=job('independent.log',cmd['independent']);deadline=time.monotonic()+20
        while not (out/'independent-ready.json').exists():
            if independent.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Independent pose/native landed/contact streams not ready; no arming')
            time.sleep(.1)
        flight=job('flight.log',cmd['flight'])
        def cancel(*_):
            nonlocal failure
            failure='Runner cancellation requested'
            if flight.poll() is None:os.killpg(flight.pid,signal.SIGINT)
        signal.signal(signal.SIGINT,cancel);signal.signal(signal.SIGTERM,cancel)
        runlog('flight-input-audit.log',['python3',str(scripts/'audit_navigation.py'),str(out/'flight-input-audit.json'),'--require-navigator'])
        try:code=flight.wait(timeout=a.seconds*3+90)
        except subprocess.TimeoutExpired:
            failure='Wall-clock watchdog requested monitored native handoff';os.killpg(flight.pid,signal.SIGINT)
            try:code=flight.wait(timeout=120)
            except subprocess.TimeoutExpired:
                failure+='; adapter remains alive for primary intervention';code=None
        time.sleep(.5)
    except (Exception,KeyboardInterrupt) as e:
        failure=str(e);code=None
        if flight is not None and flight.poll() is None:
            os.killpg(flight.pid,signal.SIGINT)
            try:code=flight.wait(timeout=120)
            except subprocess.TimeoutExpired:failure+='; adapter remains alive for primary intervention'
    finally:
        # Preserve ALL capture until the adapter exited AND actual disarm is confirmed.
        if flight is not None:
            def intervention_heartbeat(state):
                nonlocal failure,capture_stopped_for_disk
                if state['intervention_needed']:failure=failure or 'Primary intervention required before capture cleanup'
                try:free=shutil.disk_usage(out).free
                except OSError as e:
                    free=None;failure=(failure or '')+'; CRITICAL disk status unavailable: '+str(e)
                if free is not None and free<2147483648 and not capture_stopped_for_disk:
                    failure=(failure or '')+'; CRITICAL disk reserve below2GiB; capture-only stop requested, original files preserved'
                    capture_stopped_for_disk=True
                    # Controlled recording flush only. Never signal flight/SLAM/FCU.
                    request_capture_flush(jobs,flight,video)
                state={**state,'wall_time_unix_s':time.time(),'runner_failure':failure,'disk_free_bytes':free,
                       'capture_stopped_for_disk':capture_stopped_for_disk,
                       'recorders':[{'pid':p.pid,'returncode':p.poll()} for p in jobs if p is not flight],
                       'video_pid':None if video is None else video.p.pid,
                       'video_returncode':None if video is None else video.p.poll()}
                try:
                    target=out/'intervention-needed.json';temporary=target.with_suffix('.tmp')
                    temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(target)
                except OSError as e:print('CRITICAL INTERVENTION HEARTBEAT WRITE FAILED: '+str(e),flush=True)
                print('INTERVENTION_NEEDED '+json.dumps(state),flush=True)
            saved={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
            def acknowledge_signal(*_):
                print('INTERVENTION_NEEDED operator signal acknowledged; captures retained until flight exit and native disarm; no native command issued',flush=True)
            for sig in saved:signal.signal(sig,acknowledge_signal)
            try:intervention=retain_until_safe(flight,native_status,intervention_heartbeat)
            finally:
                for sig,handler in saved.items():signal.signal(sig,handler)
        # Never kill a live flight process or blindly LAND/reset at a task failure.
        for proc in jobs:
            if proc is not flight:
                try:stop(proc)
                except Exception as e:failure=(failure or '')+'; auxiliary cleanup: '+str(e)
        for handle in handles:handle.close()
        if video is not None:
            try:video_info=video.stop()
            except Exception as e:failure=(failure or '')+'; flight/SLAM video finalization: '+str(e)
        if bag.exists() and video is not None:
            try:
                fpv=out/'videos/fpv.mp4'
                runlog('fpv-export.log',['python3',str(scripts/'export_fpv_video.py'),str(bag),str(fpv)])
            except Exception as e:
                fpv=None;failure=(failure or '')+'; original camera video export: '+str(e)
        if mission.exists():shutil.copy2(mission,out/'frozen-controller.json')
        try:
            final_status=native_status();(out/'final-status.txt').write_text(final_status)
            is_disarmed=bool(re.search(r'arming_state:\s*1\b',final_status))
        except Exception as e:
            is_disarmed=False;failure=(failure or '')+'; final native state unavailable: '+str(e)
        if not is_disarmed:failure=(failure or '')+'; actual vehicle armed: primary intervention required, reset prohibited'
        independent_complete=False
        if not a.observe_only and (out/'independent.json').exists() and (out/'frozen-controller.json').exists():
            try:
                result=subprocess.run(['bash','-c',setup+'exec '+shlex.join(cmd['scorer'])],capture_output=True,text=True,timeout=60)
                (out/'scoring.log').write_text(result.stdout+result.stderr)
                if (out/'competition-score.json').exists():independent_complete=bool(json.loads((out/'competition-score.json').read_text())['competition_tasks_complete'])
            except Exception as e:failure=(failure or '')+'; independent scoring unavailable: '+str(e)
        try:audit=json.loads((out/'flight-input-audit.json').read_text()) if (out/'flight-input-audit.json').exists() else {}
        except (ValueError,OSError) as e:
            audit={};failure=(failure or '')+'; input audit unavailable: '+str(e)
        complete=bool(not failure and code==0 and is_disarmed and independent_complete and audit.get('valid') and video_info and fpv and not a.observe_only)
        report={'mission_profile':'competition','competition_tasks_complete':complete,'success':complete,'observe_only':a.observe_only,'runner_failure':failure,'disarmed':is_disarmed,'exit_code':code,'flight_process_alive':flight is not None and flight.poll() is None,'flight_pid':None if flight is None else flight.pid,'bag':str(bag),'input_audit_valid':bool(audit.get('valid')),'independent_score_file':str(out/'competition-score.json'),'truth_fed_to_navigation':False,'reset_performed':False,'hardware_or_orin_validated':False}
        report['intervention']=intervention
        report['capture_stopped_for_disk']=capture_stopped_for_disk
        report['videos']={'screen':str(out/'videos/flight-and-slam.mp4'),'fpv':None if fpv is None else str(fpv),'screen_probe':video_info}
        (out/'run.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if complete or a.observe_only and not failure and code==0 and is_disarmed else 2

if __name__=='__main__':raise SystemExit(main())
