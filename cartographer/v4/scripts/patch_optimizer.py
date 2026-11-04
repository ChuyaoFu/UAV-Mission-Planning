from pathlib import Path
p=Path('/work/cartographer/v4/carto_ws/src/cartographer/cartographer/mapping/internal/optimization/optimization_problem_3d.cc');s=p.read_text();old='''  if (options_.fix_z_in_3d()) {
    // Add penalties for violating odometry (if available) and changes between''';new='''  {
    // Stereo/VIO supplies relative poses in full 3D as well as fixed-height
    // operation. Preserve these factors without disabling inertial gravity.
    // Add penalties for violating odometry (if available) and changes between''';assert old in s;s=s.replace(old,new,1);p.write_text(s)
