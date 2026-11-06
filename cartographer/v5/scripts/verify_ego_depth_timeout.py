#!/usr/bin/env python3
"""Compile and execute the actual patched FSM callback with small ROS test doubles.

Usage: python3 test_native_depth_timeout.py /path/to/patched/ego-planner-swarm
Needs a C++17 compiler; no ROS/Eigen installation. This checks callback control
flow only, not ROS scheduling, emergency trajectory dynamics or hardware.
"""
from pathlib import Path
import subprocess
import sys
import tempfile

STUBS = r'''
#include <cmath>
#include <iostream>
#include <vector>
#include <stdexcept>
#define RCLCPP_ERROR(...) ((void)0)
#define RCLCPP_WARN(...) ((void)0)
namespace Eigen { struct Vector3d {
  double x=0;
  Vector3d operator-(Vector3d other) const { return {x-other.x}; }
  double norm() const { return std::abs(x); }
}; }
struct Time {
  double t;
  double seconds() const { return t; }
  Time operator-(Time other) const { return {t-other.t}; }
};
struct Node { Time now() {return {10.};} };
struct Trajectory { Eigen::Vector3d evaluateDeBoorT(double) {return {};} };
struct LocalTrajData { Time start_time_{9.9}; double duration_=1.; Trajectory position_traj_; };
struct GridMap {
  bool timeout=false; int occupied=0, queries=0;
  bool getOdomDepthTimeout() { return timeout; }
  int getInflateOccupancy(Eigen::Vector3d) { ++queries; return occupied; }
};
struct SwarmTraj { int drone_id=0; Time start_time_{9.9}; Trajectory position_traj_; };
struct Manager {
  LocalTrajData local_data_; GridMap grid; GridMap* grid_map_=&grid;
  std::vector<SwarmTraj> swarm_trajs_buf_;
  struct {int drone_id=0;} pp_;
  double getSwarmClearance() {return .4;}
};
class EGOReplanFSM {
public:
  enum State {WAIT_TARGET, EMERGENCY_STOP, EXEC_TRAJ, REPLAN_TRAJ};
  State exec_state_=EXEC_TRAJ;
  bool enable_fail_safe_=true, replan_result=true;
  int replans=0, publications=0;
  double emergency_time_=.7;
  Manager manager; Manager* planner_manager_=&manager;
  Node node; Node* node_=&node;
  void checkCollisionCallback();
  void changeFSMExecState(State s,const char*) {exec_state_=s;}
  bool planFromCurrentTraj() {++replans;return replan_result;}
  void publishSwarmTrajs(bool) {++publications;}
};
'''

MAIN = r'''
void require(bool ok,const char* text) {if(!ok) throw std::runtime_error(text);}
int main() {
  try {
    for(int occupied : {0,1}) for(bool replan : {false,true}) {
      EGOReplanFSM f;f.manager.grid.timeout=true;
      f.manager.grid.occupied=occupied;f.replan_result=replan;
      f.checkCollisionCallback();
      require(f.exec_state_==EGOReplanFSM::EMERGENCY_STOP,"timeout emergency overwritten");
      require(!f.enable_fail_safe_,"timeout fail-safe flag not cleared");
      require(f.manager.grid.queries==0 && f.replans==0 && f.publications==0,
              "timeout entered stale-map collision/replanning branch");
    }
    {EGOReplanFSM f;f.checkCollisionCallback();
      require(f.exec_state_==EGOReplanFSM::EXEC_TRAJ && f.manager.grid.queries>0 && f.replans==0,
              "fresh free-map execution changed");}
    {EGOReplanFSM f;f.manager.grid.occupied=1;f.checkCollisionCallback();
      require(f.exec_state_==EGOReplanFSM::EXEC_TRAJ && f.replans==1 && f.publications==1,
              "fresh-map collision replanning changed");}
    {EGOReplanFSM f;f.manager.grid.occupied=1;f.replan_result=false;f.checkCollisionCallback();
      require(f.exec_state_==EGOReplanFSM::EMERGENCY_STOP && f.replans==1,
              "fresh-map imminent collision emergency changed");}
    {EGOReplanFSM f;f.exec_state_=EGOReplanFSM::WAIT_TARGET;
      f.manager.grid.timeout=true;f.checkCollisionCallback();
      require(f.exec_state_==EGOReplanFSM::WAIT_TARGET && f.manager.grid.queries==0,
              "WAIT_TARGET early exit changed");}
    std::cout << "PASS: 8 actual-callback control-flow cases\n";return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
'''


def callback(source):
    start = source.index('void EGOReplanFSM::checkCollisionCallback()')
    opening = source.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def run(text, directory, label):
    cpp = directory / (label + '.cpp')
    exe = directory / label
    cpp.write_text(STUBS + text + MAIN)
    subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', str(cpp), '-o', str(exe)], check=True)
    return subprocess.run([str(exe)], capture_output=True, text=True)


if __name__ == '__main__':
    source = Path(sys.argv[1]) / 'src/planner/plan_manage/src/ego_replan_fsm.cpp'
    actual = callback(source.read_text())
    marker = 'changeFSMExecState(EMERGENCY_STOP, "SAFETY");'
    beginning, tail = actual.split(marker, 1)
    if 'return;' not in tail.split('/* ---------- check trajectory ---------- */', 1)[0]:
        raise SystemExit('Fix missing: timeout branch does not return')
    # Negative control removes only the new early return from the actual code.
    old = beginning + marker + tail.replace('return;', '', 1)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        result = run(actual, directory, 'fixed')
        if result.returncode:
            raise SystemExit(result.stderr)
        print(result.stdout.strip())
        old_result = run(old, directory, 'negative_control')
        if old_result.returncode == 0:
            raise SystemExit('Negative control unexpectedly passed')
        print('PASS: unpatched negative control fails:', old_result.stderr.strip())
