from pathlib import Path
p=Path('/work/cartographer/v4/carto_ws/src/cartographer/cartographer/mapping/pose_extrapolator.cc');s=p.read_text();old='  return extrapolation_delta * linear_velocity_from_odometry_;\n}'
new='''  // Integrate the observed odometry poses over the scan gap. A single
  // velocity fitted over that gap is not a valid predictor past its end,
  // especially when the vehicle turns or brakes while stereo is sparse.
  if (odometry_data_.front().time <= newest_timed_pose.time &&
      newest_timed_pose.time <= odometry_data_.back().time) {
    const auto sample = [this](const common::Time query) {
      auto right = std::lower_bound(
          odometry_data_.begin(), odometry_data_.end(), query,
          [](const sensor::OdometryData& data, const common::Time& t) {
            return data.time < t;
          });
      if (right == odometry_data_.begin()) return right->pose;
      if (right == odometry_data_.end()) right = std::prev(right);
      const auto left = std::prev(right);
      const double interval = common::ToSeconds(right->time - left->time);
      if (interval <= 1e-6) return right->pose;
      const double u = common::ToSeconds(query - left->time) / interval;
      return transform::Rigid3d(
          (1. - u) * left->pose.translation() + u * right->pose.translation(),
          left->pose.rotation().slerp(u, right->pose.rotation()));
    };
    const transform::Rigid3d at_scan = sample(newest_timed_pose.time);
    const transform::Rigid3d at_query = sample(time);
    return newest_timed_pose.pose.rotation() * at_scan.rotation().inverse() *
           (at_query.translation() - at_scan.translation());
  }
  return extrapolation_delta * linear_velocity_from_odometry_;
}'''
assert old in s;s=s.replace(old,new,1);p.write_text(s)
p=Path('/work/cartographer/v4/carto_ws/src/cartographer/cartographer/mapping/pose_extrapolator_test.cc');s=p.read_text();needle='}  // namespace\n}  // namespace mapping';extra='''TEST(PoseExtrapolatorTest, SparseScansInterpolateOdometryAtScanTime) {
  PoseExtrapolator e(common::FromSeconds(.5), .1);
  const common::Time t = common::FromUniversal(123);
  e.AddOdometryData({t, transform::Rigid3d::Identity()});
  e.AddOdometryData({t + common::FromSeconds(1), transform::Rigid3d::Translation(Eigen::Vector3d(1,0,0))});
  e.AddPose(t + common::FromSeconds(.5), transform::Rigid3d::Translation(Eigen::Vector3d(.5,0,0)));
  e.AddOdometryData({t + common::FromSeconds(2), transform::Rigid3d::Translation(Eigen::Vector3d(1,1,0))});
  EXPECT_THAT(e.ExtrapolatePose(t + common::FromSeconds(2)), transform::IsNearly(transform::Rigid3d::Translation(Eigen::Vector3d(1,1,0)), 1e-8));
}
''';s=s.replace(needle,extra+needle);p.write_text(s)
