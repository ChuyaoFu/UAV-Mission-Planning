#!/usr/bin/python3
"""Safety/scoring behavior tests, synthetic data never claimed as flight evidence."""
import copy,json
from judge import judge

def trace():
 return {'contact_topics_verified':True,'poses':[[0,-2.5146,1.4,.22],[1,-2.5146,1.4,.8],[2,-2.5146,2.58,.8],[3,.9,2.58,.8],[4,2.31,-.85,.88],[5,-1.55,-1.1,.8],[6,-2.04216,-.79248,.8],[7,-2.04216,-.79248,.22]],'statuses':[{'stamp':.1,'armed':True},{'stamp':7.1,'armed':False}],'contacts':[{'stamp':7,'pair':['course::floor_20_x_20_ft','drone::body']}]}
checks=[]
def run(name,data,pred):
 r=judge(data);assert pred(r),(name,r);checks.append(name)
run('Complete synthetic sequence stays uncertified',trace(),lambda r:r['engineering_complete'] and not r['official_completion_certified'])
a=trace();a['poses'].pop(2);run('Skipping first checkpoint cannot count later checkpoints',a,lambda r:not r['engineering_complete'] and not r['checkpoints'])
a=trace();a['poses'].insert(2,[1.5,-2.5,1.5,.22]);a['contacts'].append({'stamp':1.5,'pair':['course::floor_20_x_20_ft','drone::body']});run('Non-pad ground contact terminates run',a,lambda r:r['failure']=='Non-platform ground contact' and not r['engineering_complete'])
a=trace();a['statuses']=[{'stamp':.1,'armed':True}];run('No disarm means no completion',a,lambda r:not r['engineering_complete'])
a=trace();a['contact_topics_verified']=False;run('Empty/unverified contact subscription cannot pass',a,lambda r:not r['engineering_complete'])
a=trace();a['poses'][-1][1]+=.51;run('Whole frame must fit landing ring',a,lambda r:not r['engineering_complete'])
a=trace();a['poses'][3][3]=2.;run('Over-wall route fails strict baseline',a,lambda r:r['failure']=='Over-wall route')
a=trace();a['statuses'][-1]['stamp']=181.;run('Time limit enforced',a,lambda r:not r['engineering_complete'])
a=trace();a['contacts']+=[{'stamp':4.1,'pair':['wall','drone']},{'stamp':4.15,'pair':['wall','drone']},{'stamp':4.6,'pair':['wall','drone']}];run('Debounce continuous touch; separate touch adds five seconds',a,lambda r:r['collision_events']==2 and r['penalty_s']==10)
a=trace();a['contacts']+=[{'stamp':4.1,'pair':['wall','drone::base']},{'stamp':4.12,'pair':['wall','drone::rotor']}];run('Same wall impact across multiple body sensors counts once',a,lambda r:r['collision_events']==1)
print(json.dumps({'test_kind':'synthetic judge acceptance, not flight validation','passed':len(checks),'cases':checks},indent=2))
