#!/usr/bin/env python3
"""Generate the official-compatible Sky Beta + OAK-D S2 Gazebo Harmonic model.

  python3 gen_model.py OUT_DIR [--image-model gz|oak]   -> OUT_DIR/skybeta_oak/{model.sdf,model.config}

Values come from profile.json (spec values + documented assumptions). PX4 sensor
names (imu_sensor, air_pressure_sensor, magnetometer_sensor on base_link, the
lidar_sensor_link/lidar down range) follow PX4 v1.17 GZBridge so PX4 itself
receives them. OAK streams use their own gz topics (/gz/oak/...), bridged and
relayed to the official ROS names by bridge.yaml + oak_sim_driver.py.
"""
import json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / 'profile.json').read_text())


def noise(stddev, bias_std=0.0, dyn=0.0, corr=400.0):
    extra = (f'<bias_mean>0</bias_mean><bias_stddev>{bias_std}</bias_stddev>'
             f'<dynamic_bias_stddev>{dyn}</dynamic_bias_stddev><dynamic_bias_correlation_time>{corr}</dynamic_bias_correlation_time>'
             if bias_std or dyn else '')
    return f'<noise type="gaussian"><mean>0</mean><stddev>{stddev}</stddev>{extra}</noise>'


def imu_block(gyro, acc, gyro_bias=0.0, acc_bias=0.0, gyro_dyn=0.0, acc_dyn=0.0):
    g = ''.join(f'<{a}>{noise(gyro, gyro_bias, gyro_dyn)}</{a}>' for a in 'xyz')
    l = ''.join(f'<{a}>{noise(acc, acc_bias, acc_dyn)}</{a}>' for a in 'xyz')
    return f'<imu><angular_velocity>{g}</angular_velocity><linear_acceleration>{l}</linear_acceleration></imu>'


def box(name, pose, size, rgba, collide=True, mu=0.9, yaw=0.0):
    s = ' '.join(f'{v:.4f}' for v in size); p = ' '.join(f'{v:.4f}' for v in pose) + f' 0 0 {yaw:.7f}'
    out = (f'<visual name="{name}"><pose>{p}</pose><geometry><box><size>{s}</size></box></geometry>'
           f'<material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material></visual>')
    if collide:
        out += (f'<collision name="{name}"><pose>{p}</pose><geometry><box><size>{s}</size></box></geometry>'
                f'<surface><friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction></surface></collision>')
    return out


def camera(name, kind, topic, frame, pose, fmt, near, far, extra=''):
    c = P['camera']; W, H = c['width'], c['height']; fx, fy, cx, cy = c['K']
    hfov = 2.0 * math.atan(W / (2.0 * fx))
    p = ' '.join(f'{v:.4f}' for v in pose)
    return f'''
      <sensor name="{name}" type="{kind}">
        <pose>{p} 0 0 0</pose>
        <topic>{topic}</topic>
        <gz_frame_id>{frame}</gz_frame_id>
        <always_on>1</always_on>
        <update_rate>{c["rate_hz"]}</update_rate>
        <camera name="{name}">
          <horizontal_fov>{hfov:.6f}</horizontal_fov>
          <image><width>{W}</width><height>{H}</height><format>{fmt}</format></image>
          <clip><near>{near}</near><far>{far}</far></clip>
          <lens><intrinsics><fx>{fx}</fx><fy>{fy}</fy><cx>{cx}</cx><cy>{cy}</cy><s>0</s></intrinsics></lens>
          <camera_info_topic>{topic.rsplit("/", 1)[0]}/camera_info</camera_info_topic>
          {extra}
        </camera>
      </sensor>'''


def rotor(i, x, y, direction):
    return f'''
    <link name="rotor_{i}">
      <pose>{x} {y} 0.030 0 0 0</pose>
      <inertial><mass>0.008</mass><inertia><ixx>1.6e-05</ixx><iyy>1.6e-05</iyy><izz>3.2e-05</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
      <visual name="prop"><geometry><cylinder><radius>0.0889</radius><length>0.004</length></cylinder></geometry>
        <material><ambient>.08 .08 .08 .6</ambient><diffuse>.08 .08 .08 .6</diffuse></material><transparency>0.4</transparency></visual>
      <collision name="prop"><geometry><cylinder><radius>0.0889</radius><length>0.004</length></cylinder></geometry></collision>
    </link>
    <joint name="rotor_{i}_joint" type="revolute">
      <parent>base_link</parent><child>rotor_{i}</child>
      <axis><xyz>0 0 1</xyz><limit><lower>-1e+16</lower><upper>1e+16</upper></limit></axis>
    </joint>
    <plugin filename="gz-sim-multicopter-motor-model-system" name="gz::sim::systems::MulticopterMotorModel">
      <jointName>rotor_{i}_joint</jointName><linkName>rotor_{i}</linkName>
      <turningDirection>{direction}</turningDirection>
      <timeConstantUp>0.0125</timeConstantUp><timeConstantDown>0.025</timeConstantDown>
      <maxRotVelocity>1000.0</maxRotVelocity><motorConstant>8.54858e-06</motorConstant><momentConstant>0.016</momentConstant>
      <commandSubTopic>command/motor_speed</commandSubTopic><motorNumber>{i}</motorNumber>
      <rotorDragCoefficient>8.06428e-05</rotorDragCoefficient><rollingMomentCoefficient>1e-06</rollingMomentCoefficient>
      <rotorVelocitySlowdownSim>10</rotorVelocitySlowdownSim><motorType>velocity</motorType>
    </plugin>'''


def build(image_model='gz'):
    a, c, d, imu, rng = P['airframe'], P['camera'], P['depth'], P['imu'], P['range']
    # gz: renderer adds sigma 0.006 noise; oak: clean render, oak_sim_driver applies the replica OAK sensor model
    cam_noise = '<noise><type>gaussian</type><mean>0</mean><stddev>0.006</stddev></noise>' if image_model == 'gz' else ''
    r = a['rotor_xy_m']; m_body = a['mass_kg'] - 4 * 0.008
    zb = -a['base_link_rest_height_m']                          # skid bottom (base_link 5 cm above floor at rest)
    vis = ''.join([
        box('body', (0, 0, 0.0), (0.16, 0.09, 0.045), '.15 .15 .17 1'),
        box('battery', (-0.01, 0, 0.033), (0.13, 0.045, 0.03), '.2 .2 .22 1'),
        box('skid_l', (0, 0.045, zb + 0.006), (0.14, 0.012, 0.012), '.1 .1 .1 1'),
        box('skid_r', (0, -0.045, zb + 0.006), (0.14, 0.012, 0.012), '.1 .1 .1 1'),
        box('leg_l', (0, 0.045, zb + 0.020), (0.012, 0.012, 0.022), '.1 .1 .1 1', collide=False),
        box('leg_r', (0, -0.045, zb + 0.020), (0.012, 0.012, 0.022), '.1 .1 .1 1', collide=False),
        box('arm_a', (0, 0, 0.022), (2 * r * math.sqrt(2) + 0.03, 0.02, 0.01), '.08 .08 .09 1', collide=False, yaw=math.pi / 4),
        box('arm_b', (0, 0, 0.022), (2 * r * math.sqrt(2) + 0.03, 0.02, 0.01), '.08 .08 .09 1', collide=False, yaw=-math.pi / 4),
        box('oak_housing', (0.10, -0.0005, 0.029), (0.022, 0.11, 0.025), '.05 .05 .05 1', collide=False),
    ])
    # depth must not see the arena net / tunnel black mesh (visibility flag bit 1 set by build_world.py)
    depth_extra = '<visibility_mask>4294967293</visibility_mask>'
    sensors = ''.join([
        camera('oak_left', 'camera', '/gz/oak/left/image', 'oak_left', c['left_in_base_m'], 'L8', 0.10, 40.0, cam_noise),
        camera('oak_right', 'camera', '/gz/oak/right/image', 'oak_right', c['right_in_base_m'], 'L8', 0.10, 40.0, cam_noise),
        camera('oak_depth', 'depth_camera', '/gz/oak/depth/image', 'oak_left', c['left_in_base_m'], 'R_FLOAT32',
               d['min_m'], d['max_m'], depth_extra),
    ])
    gs = imu['gyro_noise_density'] * math.sqrt(imu['rate_hz']); as_ = imu['accel_noise_density'] * math.sqrt(imu['rate_hz'])
    ip = ' '.join(f'{v:.4f}' for v in imu['in_base_m'])
    lp = ' '.join(f'{v:.4f}' for v in rng['in_base_m'])
    sdf = f'''<?xml version="1.0"?>
<!-- GENERATED by ubuntu_compat/gen_model.py from profile.json - do not edit -->
<sdf version="1.9">
  <model name="skybeta_oak">
    <self_collide>false</self_collide>
    <link name="base_link">
      <inertial><mass>{m_body:.4f}</mass>
        <inertia><ixx>0.0090</ixx><iyy>0.0095</iyy><izz>0.0165</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
      {vis}
      <sensor name="air_pressure_sensor" type="air_pressure">
        <gz_frame_id>base_link</gz_frame_id><always_on>1</always_on><update_rate>50</update_rate>
        <air_pressure><pressure>{noise(3)}</pressure></air_pressure>
      </sensor>
      <sensor name="magnetometer_sensor" type="magnetometer">
        <gz_frame_id>base_link</gz_frame_id><always_on>1</always_on><update_rate>100</update_rate>
        <magnetometer><x>{noise(0.0001)}</x><y>{noise(0.0001)}</y><z>{noise(0.0001)}</z></magnetometer>
      </sensor>
      <sensor name="imu_sensor" type="imu">
        <gz_frame_id>base_link</gz_frame_id><always_on>1</always_on><update_rate>240</update_rate>
        {imu_block(0.0008726646, 0.00637)}
      </sensor>
      <sensor name="oak_imu" type="imu">
        <pose>{ip} 0 0 0</pose>
        <topic>/gz/oak/imu</topic><gz_frame_id>{imu["frame_id"]}</gz_frame_id>
        <always_on>1</always_on><update_rate>{imu["rate_hz"]}</update_rate>
        {imu_block(round(gs, 6), round(as_, 6), gyro_bias=0.0005, acc_bias=0.01)}
      </sensor>
      {sensors}
    </link>
    <link name="lidar_sensor_link">
      <pose>{lp} 0 0 0</pose>
      <inertial><mass>0.001</mass><inertia><ixx>1e-07</ixx><iyy>1e-07</iyy><izz>1e-07</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
      <sensor name="lidar" type="gpu_lidar">
        <pose>0 0 0 3.1415927 0 0</pose>
        <gz_frame_id>down_tof</gz_frame_id><always_on>1</always_on><update_rate>{rng["rate_hz"]}</update_rate>
        <lidar>
          <scan><horizontal><samples>1</samples><resolution>1</resolution><min_angle>0</min_angle><max_angle>0</max_angle></horizontal>
                <vertical><samples>1</samples><resolution>1</resolution><min_angle>1.5707963</min_angle><max_angle>1.5707963</max_angle></vertical></scan>
          <range><min>{rng["min_m"]}</min><max>{rng["max_m"]}</max><resolution>0.001</resolution></range>
          <noise><type>gaussian</type><mean>0</mean><stddev>0.005</stddev></noise>
        </lidar>
        <visualize>false</visualize>
      </sensor>
    </link>
    <joint name="lidar_joint" type="fixed"><parent>base_link</parent><child>lidar_sensor_link</child></joint>
    {rotor(0, r, -r, 'ccw')}
    {rotor(1, -r, r, 'ccw')}
    {rotor(2, r, r, 'cw')}
    {rotor(3, -r, -r, 'cw')}
    <plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
      <dimensions>3</dimensions><odom_frame>gt_world</odom_frame><robot_base_frame>gt_base_link</robot_base_frame>
      <odom_publish_frequency>50</odom_publish_frequency><odom_topic>/gz/gt/odometry</odom_topic>
      <odom_covariance_topic>/gz/gt/odometry_with_covariance</odom_covariance_topic>
      <tf_topic>/gz/gt/tf_unused</tf_topic>
    </plugin>
  </model>
</sdf>
'''
    return sdf


def main():
    image_model = sys.argv[sys.argv.index('--image-model') + 1] if '--image-model' in sys.argv else 'gz'
    assert image_model in ('gz', 'oak'), image_model
    out = Path(sys.argv[1]) / 'skybeta_oak'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'model.sdf').write_text(build(image_model))
    (out / 'model.config').write_text('<?xml version="1.0"?><model><name>skybeta_oak</name><version>1.0</version>'
                                      '<sdf version="1.9">model.sdf</sdf><description>official-compat Sky Beta + OAK-D S2 (approx.)</description></model>\n')
    print(out / 'model.sdf')


if __name__ == '__main__':
    main()
