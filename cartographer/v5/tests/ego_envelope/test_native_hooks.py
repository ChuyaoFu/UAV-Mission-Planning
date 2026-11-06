#!/usr/bin/env python3
"""Compile actual native query + publication functions with small ROS/Eigen doubles.

Usage: python3 test_native_hooks.py /path/to/candidate/ego-planner-swarm
This verifies changed C++ control flow and volume arithmetic, not ROS transport
or native planning speed. No function body under test is rewritten into Python.
"""
from pathlib import Path
import subprocess
import sys
import tempfile


def function(text, signature):
    start = text.index(signature)
    opening = text.index('{', start)
    depth, end = 1, opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


STUBS = r'''
#include <plan_env/collision_envelope.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#define RCLCPP_WARN(...) ((void)0)
using std::cout;using std::endl;
namespace Eigen {
struct Vector3d {
  std::array<double,3> a;
  Vector3d(double x=0,double y=0,double z=0):a{{x,y,z}}{}
  double x()const{return a[0];} double y()const{return a[1];} double z()const{return a[2];}
  Vector3d operator+(const Vector3d& b)const{return {a[0]+b.a[0],a[1]+b.a[1],a[2]+b.a[2]};}
  Vector3d operator-(const Vector3d& b)const{return {a[0]-b.a[0],a[1]-b.a[1],a[2]-b.a[2]};}
  Vector3d& operator+=(const Vector3d& b){*this=*this+b;return *this;}
};
struct Vector3i {
  std::array<int,3> a;
  Vector3i(int x=0,int y=0,int z=0):a{{x,y,z}}{}
  int x()const{return a[0];} int y()const{return a[1];} int z()const{return a[2];}
  static Vector3i Ones(){return {1,1,1};}
  Vector3i& operator+=(const Vector3i& b){for(int k=0;k<3;++k)a[k]+=b.a[k];return *this;}
};
struct MatrixXd {
  std::vector<Vector3d> c;
  int cols()const{return int(c.size());}
  bool allFinite()const{for(auto& v:c)for(double x:v.a)if(!std::isfinite(x))return false;return true;}
  MatrixXd middleCols(int i,int n)const{return {{c.begin()+i,c.begin()+i+n}};}
  const MatrixXd& rowwise()const{return *this;}
  Vector3d minCoeff()const{auto v=c[0];for(auto& p:c)for(int k=0;k<3;++k)v.a[k]=std::min(v.a[k],p.a[k]);return v;}
  Vector3d maxCoeff()const{auto v=c[0];for(auto& p:c)for(int k=0;k<3;++k)v.a[k]=std::max(v.a[k],p.a[k]);return v;}
  double operator()(int i,int j)const{return c[j].a[i];}
};
struct VectorXd {std::vector<double> v;int rows()const{return int(v.size());}double operator()(int i)const{return v[i];}};
}
struct Time {std::int64_t ns;std::int64_t nanoseconds()const{return ns;}double seconds()const{return ns*1e-9;}};
struct Node {std::int64_t ns=1000000000;Time now()const{return {ns};}};
class GridMap {
public:
  Node* node_;
  plan_env::CollisionEnvelope collision_envelope_;
  struct {double obstacles_inflation_=.32;} mp_;
  double vertical_inflation_=.16;
  bool strict_unknown_=true;
  Eigen::Vector3i safety_shape_{31,31,31};
  std::vector<int> safety_prefix_;
  struct {std::vector<int> occupancy_buffer_inflate_=std::vector<int>(27000,0);} md_;
  GridMap(Node* n,bool required):node_(n),collision_envelope_(required){buildPrefix(false);}
  bool collisionEnvelopeReady()const;
  bool collisionEnvelopeCurrent(std::uint64_t)const;
  std::uint64_t collisionEnvelopeSequence()const{return collision_envelope_.sequence();}
  bool collisionExtent(Eigen::Vector3d&)const;
  bool sweptKnownFree(const Eigen::Vector3d&);
  bool controlHullKnownFree(const Eigen::MatrixXd&);
  bool boxKnownFree(const Eigen::Vector3d&,const Eigen::Vector3d&);
  int getInflateOccupancy(Eigen::Vector3d);
  void posToIndex(const Eigen::Vector3d& p,Eigen::Vector3i& i)const{
    for(int k=0;k<3;++k)i.a[k]=int(std::floor((p.a[k]+1.)/.1));
  }
  bool isInMap(const Eigen::Vector3i& i)const{for(int v:i.a)if(v<0||v>=30)return false;return true;}
  bool isInMap(const Eigen::Vector3d& p)const{Eigen::Vector3i i;posToIndex(p,i);return isInMap(i);}
  int toAddress(const Eigen::Vector3i& i)const{return (i.x()*30+i.y())*30+i.z();}
  void buildPrefix(bool unknown){
    safety_prefix_.assign(31*31*31,0);
    auto at=[&](int x,int y,int z)->int&{return safety_prefix_[(x*31+y)*31+z];};
    for(int x=1;x<=30;++x)for(int y=1;y<=30;++y)for(int z=1;z<=30;++z){
      int blocked=unknown&&x==11&&y==16&&z==17;
      at(x,y,z)=blocked+at(x-1,y,z)+at(x,y-1,z)+at(x,y,z-1)-at(x-1,y-1,z)-at(x-1,y,z-1)-at(x,y-1,z-1)+at(x-1,y-1,z-1);
    }
  }
};
namespace geometry_msgs {namespace msg {struct Point{double x,y,z;};}}
namespace traj_utils {namespace msg {struct Bspline{int order,traj_id;Time start_time;std::vector<geometry_msgs::msg::Point> pos_pts;std::vector<double> knots;};}}
struct PositionTrajectory {
  Eigen::MatrixXd getControlPoint()const{return {{{0,0,.65},{.1,0,.65},{.2,0,.65},{.3,0,.65}}};}
  Eigen::MatrixXd get_control_points()const{return getControlPoint();}
  Eigen::VectorXd getKnot()const{return {{0,1}};}
};
struct LocalTrajData {int traj_id_=7;Time start_time_{1000000000};PositionTrajectory position_traj_;};
struct Manager {
  LocalTrajData local_data_;GridMap* grid_map_;Node* node_;
  int calls=0;std::int64_t cost_ns=0;
  bool reboundReplan(Eigen::Vector3d,Eigen::Vector3d,Eigen::Vector3d,Eigen::Vector3d,Eigen::Vector3d,bool,bool){
    ++calls;node_->ns+=cost_ns;local_data_.traj_id_=42;return true;
  }
};
struct Publisher {int count=0;void publish(const traj_utils::msg::Bspline&){++count;}};
struct Visualization {void displayOptimalList(const Eigen::MatrixXd&,int){}};
class EGOReplanFSM {
public:
  Node* node_;Manager* planner_manager_;Publisher* bspline_pub_;Visualization* visualization_;
  double last_planning_attempt_s_=-1.;bool have_new_target_=true;
  Eigen::Vector3d start_pt_,start_vel_,start_acc_,local_target_pt_,local_target_vel_;
  void getLocalTarget(){}
  bool callReboundReplan(bool,bool);
};
void require(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
void accept(GridMap& g,std::int64_t now=1000000000){
  require(g.collision_envelope_.accept("EGO_ENVELOPE_V1 body-1 1 1000000000 1100000000 .10787 .10787 .12787",
          now,plan_env::CollisionEnvelope::steadyNow()),"grant fixture setup");
}
'''

MAIN = r'''
int main(){try{
  const Eigen::Vector3d body(-.02365,.01256,.65343),stop(-.06712,.07665,.66989);
  Node node;GridMap base(&node,false);base.buildPrefix(true);
  require(base.getInflateOccupancy(stop)==0,"default-off base volume changed");
  GridMap expanded(&node,true);expanded.buildPrefix(true);
  require(expanded.getInflateOccupancy(body)==1,"missing grant should block");
  accept(expanded);
  require(expanded.getInflateOccupancy(body)==0,"known expanded current body rejected");
  require(expanded.getInflateOccupancy(stop)==1,"expanded stop unknown voxel missed");
  Eigen::MatrixXd stationary{{body,body,body,body}},stopHull{{stop,stop,stop,stop}};
  require(expanded.controlHullKnownFree(stationary),"known full-body hull rejected");
  require(!expanded.controlHullKnownFree(stopHull),"hull omitted same extra");
  node.ns=1100000000;
  require(expanded.getInflateOccupancy(body)==1&&!expanded.controlHullKnownFree(stationary),"expired query/hull accepted");
  for(const auto& scenario:std::vector<std::array<std::int64_t,4>>{
      {1,1000000000,30000000,1},  // within actual remaining lifetime
      {1,1000000000,100000000,0}, // exact deadline
      {1,1000000000,110000000,0}, // native allowed budget can exceed grant
      {1,1090000000,20000000,0},  // ACK/queue age consumes the lifetime
      {0,1000000000,200000000,1}  // opt-out remains unchanged
  }){
    Node n;n.ns=scenario[1];GridMap g(&n,scenario[0]);if(scenario[0])accept(g,n.ns);
    Manager manager;manager.grid_map_=&g;manager.node_=&n;manager.cost_ns=scenario[2];
    Publisher pub;Visualization viz;EGOReplanFSM f;f.node_=&n;f.planner_manager_=&manager;f.bspline_pub_=&pub;f.visualization_=&viz;
    bool result=f.callReboundReplan(false,false);
    require(result==bool(scenario[3])&&pub.count==scenario[3],"actual publication gate failed");
    require(manager.local_data_.traj_id_==(result?42:7),"discarded result did not restore prior trajectory");
  }
  {Node n;GridMap g(&n,true);Manager m;m.grid_map_=&g;m.node_=&n;Publisher p;Visualization v;
    EGOReplanFSM f;f.node_=&n;f.planner_manager_=&m;f.bspline_pub_=&p;f.visualization_=&v;
    require(!f.callReboundReplan(false,false)&&m.calls==0&&p.count==0,"missing grant entered planner");}
  std::cout<<"PASS: actual native swept/hull/occupancy queries use the same extra and fail closed\n";
  std::cout<<"PASS: actual FSM publication gate, 5 timing/opt-out cases, prior-trajectory rollback\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
'''


def build_run(code, root, directory, label):
    cpp, exe = directory / (label+'.cpp'), directory / label
    cpp.write_text(STUBS + code + MAIN)
    subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                    '-I', str(root/'src/planner/plan_env/include'), str(cpp), '-o', str(exe)], check=True)
    return subprocess.run([str(exe)], capture_output=True, text=True)


if __name__ == '__main__':
    root = Path(sys.argv[1]).resolve()
    grid = (root/'src/planner/plan_env/src/grid_map.cpp').read_text()
    header = (root/'src/planner/plan_env/include/plan_env/grid_map.h').read_text()
    fsm = (root/'src/planner/plan_manage/src/ego_replan_fsm.cpp').read_text()
    signatures = ['bool GridMap::collisionEnvelopeReady()', 'bool GridMap::collisionEnvelopeCurrent(',
                  'bool GridMap::collisionExtent(', 'bool GridMap::sweptKnownFree(',
                  'bool GridMap::controlHullKnownFree(', 'bool GridMap::boxKnownFree(']
    code = '\n'.join(function(grid, name) for name in signatures)
    code += '\n' + function(header, 'inline int GridMap::getInflateOccupancy(')
    code += '\n' + function(fsm, 'bool EGOReplanFSM::callReboundReplan(')
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        result = build_run(code, root, directory, 'candidate')
        if result.returncode:
            raise SystemExit(result.stdout+result.stderr)
        print(result.stdout.strip())
        # Both negative controls alter actual function text, not a Python model.
        without_extra = code.replace('extent+=Eigen::Vector3d(extra[0],extra[1],extra[2]);',
                                     '(void)extra;')
        negative = build_run(without_extra, root, directory, 'negative_extra')
        if not negative.returncode:
            raise SystemExit('Missing-extra negative control unexpectedly passed')
        print('PASS: missing-extra negative control fails:', negative.stderr.strip())
        without_deadline = code.replace('if(!planner_manager_->grid_map_->collisionEnvelopeCurrent(envelope_sequence)){',
                                        'if(false){')
        negative = build_run(without_deadline, root, directory, 'negative_deadline')
        if not negative.returncode:
            raise SystemExit('Missing-publication-gate negative control unexpectedly passed')
        print('PASS: missing-publication-gate negative control fails:', negative.stderr.strip())
