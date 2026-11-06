"""Actual candidate callback bodies, isolated from ROS; no estimator spoofing."""
from pathlib import Path
import ast,math
from types import SimpleNamespace as NS

candidate=Path(__file__).resolve().with_name('fusion.py')
tree=ast.parse(candidate.read_text());klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Fusion')
methods={n.name:n for n in klass.body if isinstance(n,ast.FunctionDef)}
module=ast.Module(body=[methods[n] for n in ['land_cb','range_cb','status_cb']],type_ignores=[])
env={'math':math,'Bool':lambda **kw:NS(**kw),'VehicleStatus':NS(ARMING_STATE_ARMED=2)}
exec(compile(ast.fix_missing_locations(module),str(candidate),'exec'),env)
class Node:
 now_value=10.
 def now(self):return self.now_value
for name in ['land_cb','range_cb','status_cb']:setattr(Node,name,env[name])
def node():
 n=Node();n.native_land_source=-1;n.native_land_received=-1.;n.native_landed_seen=False
 n.startup_zupt_released=False;n.startup_zupt_release_reason=None;n.departed=False;n.depart_start=-1.;n.armed=False;n.ground=.08
 n.messages=[];n.motion=NS(publish=lambda m:n.messages.append(m.data));n.height_observer=NS(ranging=lambda m:None)
 return n
def land(n,t,v):n.land_cb(NS(timestamp=t,landed=v))
n=node();land(n,0,True);land(n,1,False);assert not n.startup_zupt_released and not n.messages
land(n,2,True);assert n.native_landed_seen and not n.messages
n.status_cb(NS(arming_state=2));assert n.armed and not n.startup_zupt_released and not n.messages
land(n,1,False);land(n,2,False);assert not n.startup_zupt_released # stale/duplicate ignored
land(n,3,False);assert n.startup_zupt_released and n.messages==[True]
assert not n.departed and n.depart_start==-1. and n.ground==.08 # no ascent claim/ToF correction
n.range_cb(NS(range=.08));assert n.messages[-1] is True and not n.departed # no deadlock on +6cm
n.status_cb(NS(arming_state=1));assert not n.departed and n.startup_zupt_released
land(n,4,True);assert n.messages[-1] is True and n.startup_zupt_released # one-way until restart
n.now_value=9.;land(n,5,False);assert n.native_land_source==4 # receipt clock regression rejected
n=node();n.armed=True;n.range_cb(NS(range=.15));assert n.departed and n.messages==[True] # original ToF path preserved
# Actual report starts with repeated true, so a missed first Bool is retried.
statement=methods['report'].body[0]
f=ast.FunctionDef(name='repeat',args=methods['report'].args,body=[statement],decorator_list=[])
exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),str(candidate),'exec'),env)
n=node();env['repeat'](n);assert n.messages==[]
n.startup_zupt_released=True;env['repeat'](n);assert n.messages==[True]
print('PASS actual callbacks: no ARM/command release; prior landed=true required; positive ordered native timestamps; true->false releases only ZUPT mode; no departed/ground change; ToF path unchanged; disarm/land do not reenable; repeat true; clock regression rejected')
