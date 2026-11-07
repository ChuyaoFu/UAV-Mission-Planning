"""Synthetic return evidence tests; these do not establish flight success."""
import copy,unittest
from judge import judge

def sample():
 p=[[0,-2.5146,.9144,.14],[1,-2.5146,.9144,.8],[2,-2.5146,2.58,.8],[3,.9,2.58,.8],[4,2.31,-.85,.88],[5,-1.55,-1.1,.8],[6,-2.04216,-.79248,.8],[7,-2.04216,-.79248,.14],[8.1,-2.04216,-.79248,.14],[9,-2.04216,-.79248,.8],[10,-2.04216,-.79248,2.15],[11,-2.4,.4,2.15],[11.1,-2.42,.5,2.15],[12,-2.5146,.9144,2.15],[13,-2.5146,.9144,.8],[14,-2.5146,.9144,.14]]
 return {'require_return_home':True,'contact_topics_verified':True,'poses':p,'statuses':[{'stamp':.1,'armed':True},{'stamp':7.1,'armed':False},{'stamp':8,'armed':True},{'stamp':14.1,'armed':False}], 'contacts':[{'stamp':7,'pair':['course::floor_20_x_20_ft','drone::body']},{'stamp':14,'pair':['course::floor_20_x_20_ft','drone::body']}]}
class ReturnJudge(unittest.TestCase):
 def test_ordered_return(self):
  result=judge(sample());self.assertTrue(result['engineering_complete'],result);self.assertTrue(result['over_wall_verified']);self.assertFalse(result['official_completion_certified'])
 def test_no_rearm(self):
  data=sample();data['statuses']=data['statuses'][:2];self.assertFalse(judge(data)['engineering_complete'])
 def test_cannot_skip_destination(self):
  data=sample();data['contacts']=data['contacts'][1:];self.assertFalse(judge(data)['engineering_complete'])
 def test_cannot_claim_wall_from_altitude_only(self):
  data=sample()
  for p in data['poses'][10:12]:p[3]=1.
  self.assertFalse(judge(data)['engineering_complete'])
 def test_final_contact_required(self):
  data=sample();data['contacts']=data['contacts'][:1];self.assertFalse(judge(data)['engineering_complete'])
 def test_outbound_shortcut_still_rejected(self):
  data=sample();data['poses'][3][3]=2.15;self.assertFalse(judge(data)['engineering_complete'])
 def test_return_wrong_ground_contact(self):
  data=sample();data['contacts'].append({'stamp':11,'pair':['floor_20_x_20_ft','drone']});self.assertFalse(judge(data)['engineering_complete'])
 def test_dock_ground_during_rearm_spinup(self):
  data=sample();data['statuses'][2]['stamp']=7.2;data['contacts'].append({'stamp':8.1,'pair':['floor_20_x_20_ft','drone']});self.assertTrue(judge(data)['engineering_complete'])
 def test_return_wall_contact_penalized(self):
  data=sample();data['contacts'].append({'stamp':11,'pair':['Barrier 1','drone']});r=judge(data);self.assertEqual(r['collision_events'],1);self.assertEqual(r['penalty_s'],5)
if __name__=='__main__':unittest.main()
