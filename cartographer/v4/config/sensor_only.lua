include "map_builder.lua"
include "trajectory_builder.lua"
options = {
 map_builder = MAP_BUILDER,
 trajectory_builder = TRAJECTORY_BUILDER,
 map_frame = "map", tracking_frame = "vio_imu", published_frame = "vio_imu",
 odom_frame = "carto_odom", provide_odom_frame = true,
 publish_frame_projected_to_2d = false, use_pose_extrapolator = true, publish_tracked_pose = true,
 use_odometry = false, use_nav_sat = false, use_landmarks = false,
 num_laser_scans = 0, num_multi_echo_laser_scans = 0, num_subdivisions_per_laser_scan = 1,
 num_point_clouds = 1, lookup_transform_timeout_sec = 0.1,
 submap_publish_period_sec = 0.5, pose_publish_period_sec = 0.02,
 trajectory_publish_period_sec = 0.1, rangefinder_sampling_ratio = 1.,
 odometry_sampling_ratio = 1., fixed_frame_pose_sampling_ratio = 1.,
 imu_sampling_ratio = 1., landmarks_sampling_ratio = 1.,
}
MAP_BUILDER.use_trajectory_builder_3d = true
MAP_BUILDER.num_background_threads = 4
TRAJECTORY_BUILDER_3D.min_range = 0.25
TRAJECTORY_BUILDER_3D.max_range = 8.
TRAJECTORY_BUILDER_3D.num_accumulated_range_data = 1
TRAJECTORY_BUILDER_3D.voxel_filter_size = 0.05
TRAJECTORY_BUILDER_3D.high_resolution_adaptive_voxel_filter.max_length = 0.2
TRAJECTORY_BUILDER_3D.high_resolution_adaptive_voxel_filter.min_num_points = 150
TRAJECTORY_BUILDER_3D.low_resolution_adaptive_voxel_filter.min_num_points = 200
TRAJECTORY_BUILDER_3D.submaps.high_resolution = 0.08
TRAJECTORY_BUILDER_3D.submaps.low_resolution = 0.3
TRAJECTORY_BUILDER_3D.submaps.num_range_data = 60
TRAJECTORY_BUILDER_3D.ceres_scan_matcher.translation_weight = 20.
TRAJECTORY_BUILDER_3D.ceres_scan_matcher.rotation_weight = 40.
POSE_GRAPH.optimize_every_n_nodes = 30
POSE_GRAPH.constraint_builder.sampling_ratio = 0.1
return options
