from pathlib import Path
p=Path('/work/cartographer/v3/carto_ws/src/cartographer/cartographer/mapping/internal/optimization/optimization_problem_3d_test.cc');s=p.read_text();needle='}  // namespace\n}  // namespace optimization';test='''TEST_F(OptimizationProblem3DTest, UsesOdometryWhileKeepingFullImuConstraints) {
  auto options = CreateOptions();
  options.set_fix_z_in_3d(false);
  options.set_use_online_imu_extrinsics_in_3d(false);
  options.set_odometry_translation_weight(1000.);
  options.set_odometry_rotation_weight(1000.);
  options.set_local_slam_pose_translation_weight(.01);
  OptimizationProblem3D problem(options);
  const common::Time start = common::FromUniversal(0);
  for (int i = 0; i <= 200; ++i) {
    problem.AddImuData(0, {start + common::FromSeconds(.01 * i), Eigen::Vector3d(0,0,9.8), Eigen::Vector3d::Zero()});
  }
  std::vector<OptimizationProblem3D::Constraint> constraints;
  for (int i = 0; i < 3; ++i) {
    const auto time = start + common::FromSeconds(i);
    const auto local = transform::Rigid3d::Translation(Eigen::Vector3d(1.1 * i,0,0));
    problem.AddTrajectoryNode(0, NodeSpec3D{time, local, local});
    problem.AddOdometryData(0, {time, transform::Rigid3d::Translation(Eigen::Vector3d(i,0,0))});
    constraints.push_back({SubmapId{0,0}, NodeId{0,i}, {local,1.,1.}, OptimizationProblem3D::Constraint::INTRA_SUBMAP});
  }
  problem.AddSubmap(0, transform::Rigid3d::Identity());
  problem.Solve(constraints, {{0, PoseGraphInterface::TrajectoryState::ACTIVE}}, {});
  EXPECT_NEAR(2., problem.node_data().at(NodeId{0,2}).global_pose.translation().x(), .01);
}
''';assert needle in s;s=s.replace(needle,test+needle);p.write_text(s)
