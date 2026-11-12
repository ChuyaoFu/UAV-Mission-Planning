"""Team 13 on the skybeta contract: the organizers' flight_interface + our stereo depth + our mission.

Modelled on competitor_example/launch/competitor.launch.py. The ``flight_interface`` node below is EXACTLY the
example's (same executable, name and parameters, auto_offboard_on_arm defaults to true): it owns the 10 Hz
OffboardControlMode heartbeat, the altitude/speed envelope and the one Offboard request after the pilot arms and
switches to Position. Our nodes never publish to /fmu/in/*.

    ros2 launch team13_nav team13.launch.py prior:=mission/course_prior.site.json
    ros2 launch team13_nav team13.launch.py mission:=false            # interface + stereo depth only (bench)
    ros2 launch team13_nav team13.launch.py mission:=false stereo:=false   # interface only
    ros2 launch team13_nav team13.launch.py params_json:='{"pilot_mode": true, "stop_after": "takeoff"}'

Every launch with record:=true writes <flights_dir>/<stamp>_<name>/: meta.json, mission.log, stereo_depth.log and
bag/ (mcap; record_images:=true adds the rectified left image + depth).
"""
import datetime
import hashlib
import json
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# name -> (default, help). The first block is competitor.launch.py's flight-interface block, unchanged.
ARGUMENTS = [
    ('max_altitude_m', '3.0', 'ceiling above the odometry origin, metres'),
    ('min_altitude_m', '0.3', 'floor above the odometry origin, metres'),
    ('max_speed_mps', '1.5', 'cap on the norm of a twist setpoint, m/s'),
    ('setpoint_rate_hz', '10.0',
     'OffboardControlMode heartbeat rate -- PX4 needs > 2 Hz'),
    ('allow_mode_request', 'false',
     'enable /skybeta/request_offboard. SIM ONLY: on a real vehicle Offboard '
     'follows the pilot arming, then switching to Position'),
    ('auto_offboard_on_arm', 'true',
     'enter Offboard once when the pilot arms, then switches to Position. false: the vehicle '
     'stays in Position under the pilot'),
    ('odometry_topic', '/visual_slam/tracking/odometry',
     'source of the ENU pose the altitude band is measured against'),
    # --- team 13 ---
    ('mission', 'true', 'start the team13 mission node (io:=skybeta)'),
    ('stereo', 'true', 'start the team13 stereo depth node'),
    ('prior', 'mission/course_prior.site.json', 'course prior, relative to the vendored official/ tree'),
    ('params_json', '{"pilot_mode": true, "pilot_pos_gain": 0.95}',
     'mission.core Params overrides (JSON object, JSON true/false); 0.95 = fleet MPC_XY_P'),
    ('stereo_rate_hz', '10.0', 'max stereo depth rate'),
    ('stereo_preset', 'half320', 'depth preset (stereo_depth_node.OPTIONS): half320 (default, 320x200 matching) | '
     'mid512 | ref640 | half320_band | bm320 ...; output is always 640x400'),
    ('stereo_width', '0', 'override the preset working width (0 = preset)'),
    ('sgbm_mode', '', 'override the preset matcher: SGBM_3WAY | SGBM | HH4 | BM (empty = preset)'),
    ('cv_threads', '2', 'OpenCV threads for the stereo node'),
    ('net_mask', 'false', 'repetitive-texture (cage net) phantom mask in the stereo node (net_mask.py); off until '
     'validated on real images'),
    ('net_mask_score_min', '0.25', 'net mask periodicity threshold (higher = fewer pixels removed)'),
    ('net_model', 'false', 'model-based net filter (net_model.py: prior fence + VIO pose; implies the net mask decision); '
     'off until validated on real images'),
    ('net_fence', '', 'fence JSON for net_model (course prior or bare fence), relative to official/; empty = the prior arg'),
    ('net_anchor', 'first', 'start frame for net_model: first = the first VIO pose (launch on the takeoff spot) | '
     'origin = the odometry frame itself'),
    ('depth_scan', 'false', 'planar scan for the maze planner (depth_scan.py): /team13/depth_scan LaserScan in frame '
     'team13_depth_scan (rect camera origin, x forward / y left / z up), nearest in-band return per column'),
    ('depth_scan_h_min', '-0.25', 'scan band bottom, metres above the camera centre (camera up)'),
    ('depth_scan_h_max', '0.35', 'scan band top, metres above the camera centre (camera up)'),
    ('depth_scan_range_max', '5.0', 'scan max range, metres (nearest in-band return beyond it -> +inf)'),
    ('noise_minsup', '0', 'free-space noise filter (depth_filters.min_support): drop disparity components <= N px '
     '(640 scale) on the final depth; 0 = off. Real flights 2026-11-08: 150 (with sgbm_speckle 800)'),
    ('noise_minsup_diff', '1.0', 'min-support: max disparity step inside a component, px (640 scale)'),
    ('sgbm_speckle_window', '0', 'SGBM speckleWindowSize at the 640 scale; 0 = node default (100). Real flights: 800'),
    ('sgbm_speckle_range', '2', 'SGBM speckleRange at the 640 scale (used only with sgbm_speckle_window > 0)'),
    ('record', 'true', 'bag + logs for this launch under flights_dir'),
    ('record_images', 'false', 'also bag /stereo/left/image + /camera/depth/image_raw (~10 MB/s)'),
    ('flights_dir', '/root/team_ws/team13_flights', 'where flights are kept (bind-mounted team workspace)'),
    ('name', 'launch', 'flight name used in the folder name'),
]


def _code_sha(root):
    h = hashlib.sha256()
    for f in sorted(Path(root).rglob('*.py')):
        if '__pycache__' not in f.parts:
            h.update(str(f.relative_to(root)).encode()); h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _team13(context):
    from team13_nav.entry import bag_topics, official_root
    a = {k: LaunchConfiguration(k).perform(context) for k, _, _ in ARGUMENTS}
    root = official_root()
    actions, env = [], {}
    if a['record'].lower() == 'true':
        stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        out = Path(a['flights_dir']) / f'{stamp}_{"".join(c if c.isalnum() or c in "-_." else "_" for c in a["name"])}'
        out.mkdir(parents=True, exist_ok=True)
        topics = bag_topics(a['record_images'].lower() == 'true')
        (out / 'meta.json').write_text(json.dumps(dict(args=a, stamp=stamp, official_root=str(root), code_sha=_code_sha(root),
                                                       topics=topics, host=os.uname().nodename,
                                                       ros_domain=os.environ.get('ROS_DOMAIN_ID')), indent=1))
        prior = root / a['prior']
        if prior.is_file():
            (out / 'prior.json').write_text(prior.read_text())
        env = {'TEAM13_LOG_DIR': str(out)}
        actions.append(ExecuteProcess(cmd=['ros2', 'bag', 'record', '-o', str(out / 'bag'), '-s', 'mcap', *topics],  # positional topics: Humble has no --topics
                                      output='log', name='team13_bag'))
        actions.append(LogInfo(msg=f'team13: recording this launch to {out}'))
    if a['stereo'].lower() == 'true':
        actions.append(Node(package='team13_nav', executable='team13_stereo_depth', name='team13_stereo_depth',
                            output='screen', emulate_tty=True, additional_env=env,
                            parameters=[{'rate_hz': float(a['stereo_rate_hz']), 'preset': a['stereo_preset'],
                                         'width': int(a['stereo_width']), 'sgbm_mode': a['sgbm_mode'],
                                         'cv_threads': int(a['cv_threads']),
                                         'net_mask': a['net_mask'].lower() == 'true',
                                         'net_mask_score_min': float(a['net_mask_score_min']),
                                         'net_model': a['net_model'].lower() == 'true',
                                         'net_fence_json': a['net_fence'] or a['prior'], 'net_anchor': a['net_anchor'],
                                         'depth_scan': a['depth_scan'].lower() == 'true',
                                         'depth_scan_h_min_m': float(a['depth_scan_h_min']),
                                         'depth_scan_h_max_m': float(a['depth_scan_h_max']),
                                         'depth_scan_range_max_m': float(a['depth_scan_range_max']),
                                         'noise_minsup_px': int(float(a['noise_minsup'])),
                                         'noise_minsup_diff_px': float(a['noise_minsup_diff']),
                                         'sgbm_speckle_window': int(float(a['sgbm_speckle_window'])),
                                         'sgbm_speckle_range': int(float(a['sgbm_speckle_range']))}]))
    if a['mission'].lower() == 'true':
        actions.append(Node(package='team13_nav', executable='team13_mission', name='official_mission',
                            output='screen', emulate_tty=True, additional_env=env,
                            parameters=[{'io': 'skybeta', 'prior': a['prior'],
                                         'params_json': ParameterValue(a['params_json'], value_type=str)}]))
    return actions


def generate_launch_description():
    declared = [
        DeclareLaunchArgument(name, default_value=default, description=help_)
        for name, default, help_ in ARGUMENTS
    ]

    flight_interface = Node(
        package='skybeta_flight_interface',
        executable='flight_interface',
        name='flight_interface',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'max_altitude_m': LaunchConfiguration('max_altitude_m'),
            'min_altitude_m': LaunchConfiguration('min_altitude_m'),
            'max_speed_mps': LaunchConfiguration('max_speed_mps'),
            'setpoint_rate_hz': LaunchConfiguration('setpoint_rate_hz'),
            'allow_mode_request': LaunchConfiguration('allow_mode_request'),
            'auto_offboard_on_arm': LaunchConfiguration('auto_offboard_on_arm'),
            'odometry_topic': LaunchConfiguration('odometry_topic'),
        }],
    )

    return LaunchDescription(declared + [flight_interface, OpaqueFunction(function=_team13)])
