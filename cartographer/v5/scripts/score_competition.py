#!/usr/bin/env python3
"""Independent, offline full-task score. Private fixture never enters navigation."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

REQUIRED=('TUNNEL_ENTRY','TUNNEL_EXIT','WINDOW','DOCK','REARM','RETAKEOFF','WALL','HOME')


def unit(x):
    a=np.asarray(x,float)
    if a.shape!=(3,) or not np.isfinite(a).all() or abs(np.linalg.norm(a)-1)>.001:raise ValueError('Fixture axes must be finite unit vectors')
    return a


def pose_at(poses,t):
    i=int(np.searchsorted(poses[:,0],t))
    if i<len(poses) and abs(poses[i,0]-t)<1e-9:return poses[i]
    if i==0 or i==len(poses) or poses[i,0]-poses[i-1,0]>.25:raise ValueError('Missing bracketing independent pose')
    a,b=poses[i-1],poses[i];u=(t-a[0])/(b[0]-a[0]);q=Slerp([0.,1.],Rotation.from_quat([a[4:8],b[4:8]]))([u])[0].as_quat()
    return np.r_[t,a[1:4]*(1-u)+b[1:4]*u,q]


def support(pose,axis,robot):
    v=Rotation.from_quat(pose[4:8]).inv().apply(axis)
    return robot['radius_m']*np.linalg.norm(v[:2])+robot['half_height_m']*abs(v[2])


def gate_crossing(poses,gate,robot,after):
    c=np.asarray(gate['center_m'],float);n=unit(gate['normal_forward']);u=unit(gate['axis_u']);v=unit(gate['axis_v'])
    if c.shape!=(3,) or not np.isfinite(c).all() or not np.allclose(np.array([n,u,v])@np.array([n,u,v]).T,np.eye(3),atol=.001):raise ValueError('Invalid private gate basis')
    for key in (('half_width_m',) if 'minimum_body_bottom_z_m' in gate else ('half_width_m','half_height_m')):
        if key not in gate or not math.isfinite(gate[key]) or gate[key]<=0:raise ValueError('Private gate size required')
    active=poses[poses[:,0]>=after];start=None
    for p in active:
        d=float((p[1:4]-c)@n);s=support(p,n,robot)
        if d<=-s:start=p
        if start is None or d<s:continue
        times=np.unique(np.r_[np.linspace(start[0],p[0],max(9,int((p[0]-start[0])/.02)+1)),active[(active[:,0]>=start[0])&(active[:,0]<=p[0]),0]])
        valid=True
        for t in times:
            q=pose_at(poses,float(t));delta=q[1:4]-c
            if abs(delta@n)>support(q,n,robot)+.005:continue
            if abs(delta@u)+support(q,u,robot)>gate['half_width_m']:valid=False;break
            if 'minimum_body_bottom_z_m' in gate:
                if q[3]-support(q,np.array([0.,0.,1.]),robot)<gate['minimum_body_bottom_z_m']:valid=False;break
            elif abs(delta@v)+support(q,v,robot)>gate['half_height_m']:valid=False;break
        if valid:return {'stamp':float(p[0]),'before_stamp':float(start[0]),'gate_id':gate['id'],'whole_body_crossed':True}
        start=None
    raise ValueError('No full-body directed crossing of '+gate['id'])


def on_pad(p,pad,robot):
    c=np.asarray(pad['center_surface_m'],float);u=unit(pad['axis_u']);v=unit(pad['axis_v'])
    if abs(u[2])>.001 or abs(v[2])>.001 or abs(u@v)>.001:raise ValueError('Pad must have horizontal orthogonal axes')
    delta=p[1:4]-c
    return bool(abs(delta@u)+support(p,u,robot)<=pad['half_width_m'] and
                abs(delta@v)+support(p,v,robot)<=pad['half_height_m'] and
                abs(p[3]-c[2]-robot['resting_body_height_m'])<=.06)


def landed_at(records,t):
    candidates=[r for r in records if 0<=t-r['stamp']<=.35]
    return bool(candidates and candidates[-1]['landed'] and not candidates[-1].get('freefall',False))


def score(data,fixture,controller):
    r={'competition_tasks_complete':False,'independent_competition_complete':False,'failure':None,'events':{},
       'truth_use':'offline independent scorer only','controller_complete':bool(controller.get('competition_tasks_complete'))}
    try:
        if fixture.get('schema_version')!=1 or fixture.get('qualification')!='private-independent-fixture':raise ValueError('Explicit private fixture v1 required')
        if data.get('clock_domain')!='ros_simulation' or not data.get('contact_topics_verified'):raise ValueError('Missing common clock/contact acquisition evidence')
        poses=np.asarray(data['poses'],float)
        if poses.ndim!=2 or poses.shape[1]!=8 or len(poses)<2 or not np.isfinite(poses).all() or (np.diff(poses[:,0])<=0).any():raise ValueError('Invalid independent pose stream')
        if np.max(np.diff(poses[:,0]))>.25:raise ValueError('Independent pose gap exceeds 250 ms')
        if (abs(np.linalg.norm(poses[:,4:8],axis=1)-1)>.001).any():raise ValueError('Invalid independent orientation')
        robot=fixture['robot']
        if not all(0<float(robot[k])<1 for k in ('radius_m','half_height_m','resting_body_height_m')):raise ValueError('Actual robot bounds required')
        if data.get('pose_reference')=='model':
            transform=np.asarray(robot['model_from_body_position_quaternion'],float)
            if transform.shape!=(7,) or not np.isfinite(transform).all() or abs(np.linalg.norm(transform[3:])-1)>.001:raise ValueError('Actual model-to-body calibration required')
            rotations=Rotation.from_quat(poses[:,4:8]);poses=poses.copy()
            poses[:,1:4]+=rotations.apply(transform[:3]);poses[:,4:8]=(rotations*Rotation.from_quat(transform[3:])).as_quat()
        elif data.get('pose_reference')!='body':raise ValueError('Independent pose reference must be explicit')
        statuses=data['statuses'];lands=data['landed'];transitions=[]
        if not statuses or statuses[0]['armed']:raise ValueError('Initial native disarm missing')
        if any(b['stamp']<=a['stamp'] or b['stamp']-a['stamp']>1. for a,b in zip(statuses,statuses[1:])):raise ValueError('Native status continuity lost')
        last=False
        for s in statuses:
            if s['armed']!=last:transitions.append(s);last=s['armed']
        if [x['armed'] for x in transitions]!=[True,False,True,False]:raise ValueError('Need exactly launch arm, dock disarm, rearm, home disarm')
        a,d,b,h=[float(x['stamp']) for x in transitions]
        home=pose_at(poses,a)[1:4]
        if not landed_at(lands,d) or not landed_at(lands,h):raise ValueError('Native landed confirmation missing at disarm')
        gates=fixture['gates'];entry=gates['tunnel_entry'];exit=gates['tunnel_exit']
        if entry['id']==exit['id'] or np.linalg.norm(np.asarray(entry['center_m'])-exit['center_m'])<=2*robot['radius_m']:raise ValueError('Tunnel entry/exit must be distinct physical gates')
        event=gate_crossing(poses,entry,robot,a);r['events']['TUNNEL_ENTRY']=event
        event=gate_crossing(poses,exit,robot,event['stamp']);r['events']['TUNNEL_EXIT']=event
        alternatives=gates.get('window_alternatives',[gates.get('window')]);options=[]
        for alternative in alternatives:
            if alternative is None:continue
            try:options.append(gate_crossing(poses,alternative,robot,event['stamp']))
            except ValueError:pass
        if not options:raise ValueError('No full-body window/permitted-door crossing')
        event=min(options,key=lambda x:x['stamp']);r['events']['WINDOW']=event
        if event['stamp']>=d:raise ValueError('Dock disarm precedes required passages')
        dock=fixture['dock'];dock_pose=pose_at(poses,d)
        if not on_pad(dock_pose,dock,robot):raise ValueError('Dock disarm outside whole-body pad/resting-height bounds')
        if np.linalg.norm(dock_pose[1:3]-home[:2])<2*robot['radius_m']:raise ValueError('Dock is not distinct from home')
        r['events']['DOCK']={'stamp':d,'landed':True,'armed':False}
        if not on_pad(pose_at(poses,b),dock,robot) or not landed_at(lands,b):raise ValueError('Rearm must occur while landed on same dock')
        r['events']['REARM']={'stamp':b,'armed':True}
        clear=float(dock['center_surface_m'][2])+robot['resting_body_height_m']+.20
        take=next((p for p in poses if b<p[0]<h and p[3]>=clear),None)
        if take is None:raise ValueError('No measured retakeoff above dock')
        r['events']['RETAKEOFF']={'stamp':float(take[0])}
        wall=gate_crossing(poses,gates['wall'],robot,float(take[0]));r['events']['WALL']=wall
        if 'minimum_body_bottom_z_m' not in gates['wall']:raise ValueError('Independent wall-top height required')
        end=pose_at(poses,h)
        if wall['stamp']>=h or np.linalg.norm(end[1:3]-home[:2])>.20 or abs(end[3]-home[2])>.06:raise ValueError('Home landing after wall crossing not verified')
        r['events']['HOME']={'stamp':h,'landed':True,'armed':False,'horizontal_error_m':float(np.linalg.norm(end[1:3]-home[:2]))}
        home_tokens=fixture['home_contact_tokens'];dock_tokens=dock['contact_tokens']
        if not home_tokens or not dock_tokens or any(not isinstance(x,str) or len(x)<3 for x in home_tokens+dock_tokens):raise ValueError('Explicit private support collision names required')
        dock_contact=home_contact=False;collisions=[]
        for contact in data['contacts']:
            t=float(contact['stamp'])
            if not a<=t<=h+.2:continue
            p=pose_at(poses,min(max(t,poses[0,0]),poses[-1,0]));names=' '.join(contact['pair'])
            at_home=np.linalg.norm(p[1:3]-home[:2])<=.24 and abs(p[3]-home[2])<=.08
            at_dock=on_pad(p,dock,robot)
            home_allowed=at_home and (t<r['events']['TUNNEL_ENTRY']['stamp'] or t>wall['stamp']) and any(x in names for x in home_tokens)
            dock_allowed=at_dock and event['stamp']<t<take[0] and any(x in names for x in dock_tokens)
            if home_allowed:
                if h-2<=t<=h+.2:home_contact=True
            elif dock_allowed:
                if d-2<=t<=b+.2:dock_contact=True
            else:collisions.append(contact)
        r['unexpected_contacts']=collisions
        if collisions:raise ValueError('Unexpected independent obstacle/ground contact')
        if not dock_contact or not home_contact:raise ValueError('Independent dock/home physical support contact missing')
        r['independent_competition_complete']=True
        completed=controller.get('completed',{})
        if set(completed)!=set(REQUIRED) or not all(completed[k].get('verified') for k in REQUIRED):raise ValueError('Controller lacks eight verified task records')
        if not r['controller_complete'] or controller.get('adapter_failure') or controller.get('observe_only'):raise ValueError('Controller did not confirm full competition completion')
        r['competition_tasks_complete']=True
    except (ValueError,KeyError,TypeError,IndexError) as e:r['failure']=str(e)
    return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--recording',required=True);p.add_argument('--fixture',required=True);p.add_argument('--controller',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    # Freeze predictions BEFORE reading private fixture. Navigation is separately
    # kernel-isolated from both paths by competition_landlock_exec.py.
    control_bytes=Path(a.controller).read_bytes();controller=json.loads(control_bytes)
    fixture_bytes=Path(a.fixture).read_bytes();recording_bytes=Path(a.recording).read_bytes()
    result=score(json.loads(recording_bytes),json.loads(fixture_bytes),controller)
    result['frozen_input_sha256']={k:hashlib.sha256(v).hexdigest() for k,v in [('controller',control_bytes),('fixture',fixture_bytes),('recording',recording_bytes)]}
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return 0 if result['competition_tasks_complete'] else 2

if __name__=='__main__':raise SystemExit(main())
