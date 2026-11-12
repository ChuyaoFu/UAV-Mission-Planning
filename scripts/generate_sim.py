#!/usr/bin/env python3
"""Generate a reproducible stereo/IMU rig and textured static-obstacle world.

Camera noise is intentionally ideal in this first baseline. The depth sensor is
Gazebo's ideal depth camera, not a model of OAK stereo matching failure modes.
No simulated pose is sent to the visual estimator.
"""
from pathlib import Path
import math
import random
import re
import shutil
import struct
import xml.etree.ElementTree as ET
import zlib

root = Path(__file__).resolve().parents[1]
sim = root / 'sim'
models = sim / 'models'
rig = models / 'x500_vio'
rig.mkdir(parents=True, exist_ok=True)
(sim / 'worlds').mkdir(parents=True, exist_ok=True)
(sim / 'assets').mkdir(parents=True, exist_ok=True)
src = root / 'src/PX4-Autopilot/Tools/simulation/gz'
server_config = ET.parse(src / 'server.config')
plugins = server_config.getroot().find('plugins')
for plugin in list(plugins):
    if plugin.get('filename') in ['libOpticalFlowSystem.so', 'libGstCameraSystem.so']:
        plugins.remove(plugin)
server_config.write(sim / 'server.config', encoding='utf-8', xml_declaration=True)

model = ET.Element('sdf', version='1.9')
m = ET.SubElement(model, 'model', name='x500_vio')
inc = ET.SubElement(m, 'include', merge='true')
ET.SubElement(inc, 'uri').text = 'model://x500'
# Extend the merged base link instead of adding a moving camera joint.
link = ET.SubElement(m, 'link', name='vio_rig')
ET.SubElement(link, 'pose', relative_to='base_link').text = '.15 0 .08 0 0 0'
inertial = ET.SubElement(link, 'inertial')
ET.SubElement(inertial, 'mass').text = '.05'
inertia = ET.SubElement(inertial, 'inertia')
for k, v in [('ixx', '.0001'), ('iyy', '.0001'), ('izz', '.0001')]:
    ET.SubElement(inertia, k).text = v
joint = ET.SubElement(m, 'joint', name='vio_rig_joint', type='fixed')
ET.SubElement(joint, 'parent').text = 'base_link'
ET.SubElement(joint, 'child').text = 'vio_rig'

hfov = 1.4
for name, y, typ in [('left', .0375, 'camera'), ('right', -.0375, 'camera'), ('depth', 0, 'depth_camera')]:
    s = ET.SubElement(link, 'sensor', name=name, type=typ)
    ET.SubElement(s, 'pose').text = f'0 {y} 0 0 0 0'
    ET.SubElement(s, 'topic').text = f'/vio/{name}/image'
    ET.SubElement(s, 'gz_frame_id').text = f'{name}_optical'
    ET.SubElement(s, 'always_on').text = 'true'
    ET.SubElement(s, 'update_rate').text = '20'
    c = ET.SubElement(s, 'camera')
    ET.SubElement(c, 'horizontal_fov').text = str(hfov)
    im = ET.SubElement(c, 'image')
    ET.SubElement(im, 'width').text = '640'
    ET.SubElement(im, 'height').text = '400'
    ET.SubElement(im, 'format').text = 'R8G8B8'
    clip = ET.SubElement(c, 'clip')
    ET.SubElement(clip, 'near').text = '.15'
    ET.SubElement(clip, 'far').text = '30'

s = ET.SubElement(link, 'sensor', name='imu', type='imu')
ET.SubElement(s, 'topic').text = '/vio/imu'
ET.SubElement(s, 'gz_frame_id').text = 'vio_imu'
ET.SubElement(s, 'always_on').text = 'true'
ET.SubElement(s, 'update_rate').text = '250'
imu = ET.SubElement(s, 'imu')
for kind, sigma in [('angular_velocity', .0002), ('linear_acceleration', .02)]:
    val = ET.SubElement(imu, kind)
    for axis in 'xyz':
        noise = ET.SubElement(ET.SubElement(val, axis), 'noise', type='gaussian')
        ET.SubElement(noise, 'mean').text = '0'
        ET.SubElement(noise, 'stddev').text = str(sigma)
pose = ET.SubElement(m, 'plugin', filename='gz-sim-pose-publisher-system', name='gz::sim::systems::PosePublisher')
ET.SubElement(pose, 'publish_model_pose').text = 'true'
ET.SubElement(pose, 'publish_link_pose').text = 'true'
ET.SubElement(pose, 'use_pose_vector_msg').text = 'true'
ET.SubElement(pose, 'update_frequency').text = '20'
ET.indent(model)
ET.ElementTree(model).write(rig / 'model.sdf', encoding='utf-8', xml_declaration=True)
(rig / 'model.config').write_text('<model><name>x500_vio</name><version>1.0</version><sdf version="1.9">model.sdf</sdf></model>\n')

# Texture: random checker tiles give persistent corners at different scales.
rng = random.Random(42)
w = 512
tiles = [[rng.randrange(35, 230) for _ in range(32)] for _ in range(32)]
pixels = b''.join(b'\x00' + bytes(tiles[y//16][x//16] for x in range(w) for _ in range(3)) for y in range(w))
def chunk(t, d):
    return struct.pack('!I', len(d)) + t + d + struct.pack('!I', zlib.crc32(t+d) & 0xffffffff)
png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', w, w, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b'')
(sim / 'assets/texture.png').write_bytes(png)
world = ET.parse(src / 'worlds/default.sdf')
wo = world.getroot().find('world')
wo.set('name', 'visual_nav')
def textured(material):
    pbr = ET.SubElement(material, 'pbr')
    metal = ET.SubElement(pbr, 'metal')
    ET.SubElement(metal, 'albedo_map').text = 'file:///work/sim/assets/texture.png'
    ET.SubElement(metal, 'roughness').text = '.95'
    ET.SubElement(metal, 'metalness').text = '0'
ground = wo.find("model[@name='ground_plane']/link/visual")
ground.find('geometry/plane/size').text = '30 24'
textured(ground.find('material'))
def box(name, xyz, size, color):
    mo = ET.SubElement(wo, 'model', name=name)
    ET.SubElement(mo, 'static').text = 'true'
    ET.SubElement(mo, 'pose').text = ' '.join(map(str, xyz)) + ' 0 0 0'
    li = ET.SubElement(mo, 'link', name='body')
    for typ in ['collision', 'visual']:
        v = ET.SubElement(li, typ, name=typ)
        geom = ET.SubElement(v, 'geometry')
        ET.SubElement(ET.SubElement(geom, 'box'), 'size').text = ' '.join(map(str, size))
        if typ == 'visual':
            mat = ET.SubElement(v, 'material')
            ET.SubElement(mat, 'ambient').text = color
            ET.SubElement(mat, 'diffuse').text = color
            textured(mat)
box('obstacle', (5, 0, 1.5), (1, 2.0, 3), '.8 .5 .3 1')
box('wall_front', (12, 0, 2), (.15, 18, 4), '.8 .8 .8 1')
box('wall_left', (4, 8, 2), (16, .15, 4), '.8 .8 .8 1')
box('wall_right', (4, -8, 2), (16, .15, 4), '.8 .8 .8 1')
for i in range(12):
    box(f'landmark_{i}', (2+i*.75, (-1 if i%2 else 1)*5, .4+(i%3)*.4), (.35, .35, .8+(i%3)*.8), '.7 .8 .9 1')
ET.indent(world)
world.write(sim / 'worlds/visual_nav.sdf', encoding='utf-8', xml_declaration=True)

config = root / 'config/gazebo_vio'
config.mkdir(parents=True, exist_ok=True)
baseline = root / 'ros_ws/src/open_vins/config/euroc_mav'
if not baseline.is_dir():
    baseline = root / 'cartographer/v2/ros_ws/src/open_vins/config/euroc_mav'
if not baseline.is_dir():
    raise FileNotFoundError('Fetch the pinned OpenVINS source before generating simulator resources')
est = (baseline / 'estimator_config.yaml').read_text()
for a, b in [('calib_cam_extrinsics: true', 'calib_cam_extrinsics: false'),
             ('calib_cam_intrinsics: true', 'calib_cam_intrinsics: false'),
             ('calib_cam_timeoffset: true', 'calib_cam_timeoffset: false'),
             ('gravity_mag: 9.81', 'gravity_mag: 9.8'),
             ('try_zupt: false', 'try_zupt: true'),
             ('zupt_chi2_multipler: 0', 'zupt_chi2_multipler: 1.0'),
             ('zupt_max_disparity: 0.5', 'zupt_max_disparity: 0.0'),
             ('zupt_only_at_beginning: false', 'zupt_only_at_beginning: true'),
             ('track_frequency: 21.0', 'track_frequency: 20.0')]:
    est = est.replace(a, b)
(config / 'estimator_config.yaml').write_text(est)
imucfg = (baseline / 'kalibr_imu_chain.yaml').read_text().replace('rostopic: /imu0', 'rostopic: /vio/imu')
# Gazebo publishes independent sample noise at the physics rate (250 Hz).
# Match its continuous-time noise density; the simulated biases are constant.
for key,value in [('accelerometer_noise_density',.02/math.sqrt(250)),
                  ('gyroscope_noise_density',.0002/math.sqrt(250)),
                  ('accelerometer_random_walk',1e-5),('gyroscope_random_walk',1e-6),
                  ('update_rate',250.0)]:
    imucfg=re.sub(r'('+key+r':)\s*\S+',lambda m:m.group(1)+' '+str(value),imucfg)
(config / 'kalibr_imu_chain.yaml').write_text(imucfg)
fx = 320 / math.tan(hfov/2)
chain = '%YAML:1.0\n'
for i, y in [(0, .0375), (1, -.0375)]:
    chain += f'''\ncam{i}:
  T_imu_cam:
    - [0., 0., 1., 0.]
    - [-1., 0., 0., {y}]
    - [0., -1., 0., 0.]
    - [0., 0., 0., 1.]
  cam_overlaps: [{1-i}]
  camera_model: pinhole
  distortion_coeffs: [0., 0., 0., 0.]
  distortion_model: radtan
  intrinsics: [{fx}, {fx}, 320., 200.]
  resolution: [640, 400]
  rostopic: /vio/{['left','right'][i]}/image_raw
'''
(config / 'kalibr_imucam_chain.yaml').write_text(chain)
print('Generated stereo model, world, and simulation-only VIO configuration.')
