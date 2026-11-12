#!/usr/bin/env python3
"""Flyable world from the venue replica (render-only SDF) for the official-compat sim.

  python3 build_world.py REPLICA_DIR OUT_SDF [STEP_S [MODEL_DIR MODEL_NAME [SPAWN_F,L,YAW_DEG]]]
  (SPAWN: vehicle start pose in the course start frame = gz world; default the launch, heading north)

REPLICA_DIR = cartographer/official/replica (with world/ built by the venue-replica
session). Changes vs. world/venue_full.sdf:
  * physics step STEP_S (default 1/480 s) and the PX4 system plugins (imu, baro, mag,
    contact, navsat, sensors/ogre2);
  * the dataset camera model "cam" removed;
  * mesh/texture URIs made absolute (world can live elsewhere);
  * arena net + tunnel black mesh get visibility flag 0x2 -> the depth camera (mask
    0xFFFFFFFD) does not see them, like OAK stereo (venue_depth.sdf convention);
  * a static "venue_collision" model of primitives (the replica has visuals only): floor, every course solid
    at its physical size (course_solids.py: maze boards / barrier, window board with its hole, cinder blocks, boulders),
    tunnel hoops + shell, cage net walls and roof.
Sim-side only: flight code never sees layout.py.
"""
import math, re, sys
from pathlib import Path

rep = Path(sys.argv[1]).resolve(); out = Path(sys.argv[2]); step = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0 / 480.0
sys.path.insert(0, str(rep))
import layout as L  # noqa: E402

src = (rep / 'world' / 'venue_full.sdf').read_text()
wdir = str(rep / 'world')

PLUGINS = '''
  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
  <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
  <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
  <plugin filename="gz-sim-air-pressure-system" name="gz::sim::systems::AirPressure"/>
  <plugin filename="gz-sim-magnetometer-system" name="gz::sim::systems::Magnetometer"/>
  <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"/>
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
  <spherical_coordinates><surface_model>EARTH_WGS84</surface_model><world_frame_orientation>ENU</world_frame_orientation>
    <latitude_deg>37.4275</latitude_deg><longitude_deg>-122.1697</longitude_deg><elevation>30</elevation><heading_deg>0</heading_deg></spherical_coordinates>
'''

s = re.sub(r'<physics[^>]*>.*?</physics>',
           f'<physics name="compat" type="dart"><max_step_size>{step:.10f}</max_step_size><real_time_factor>1.0</real_time_factor>'
           f'<real_time_update_rate>0</real_time_update_rate></physics>', src, count=1, flags=re.S)
s = re.sub(r'\s*<plugin filename="gz-sim-[^"]*"[^>]*?(/>|>.*?</plugin>)', '', s, flags=re.S)   # drop old plugin set
s = s.replace('</physics>', '</physics>' + PLUGINS, 1)
s = re.sub(r'<model name="cam">.*?</model>', '', s, flags=re.S)
s = s.replace('<uri>meshes/', f'<uri>{wdir}/meshes/')
s = re.sub(r'<(albedo_map|emissive_map|normal_map|roughness_map)>textures/', lambda m: f'<{m.group(1)}>{wdir}/textures/', s)
for vis in ('net', 'tunnel_mesh'):
    s, n = re.subn(rf'(<visual name="{vis}">)', r'\1<visibility_flags>2</visibility_flags>', s)
    assert n == 1, vis
assert '<model name="cam">' not in s


# ------------------------------------------------------------------ collision primitives (gz frame)
def gz_xy(a, b):
    return a - L.START[0], -(b - L.START[1])


cols = []


def add_box(name, a, b, h0, h1, len_n, len_e, yaw_deg=0.0):
    """box centred at cage (a, b), extent len_n along the yawed north axis, len_e across."""
    x, y = gz_xy(a, b)
    cols.append(f'<collision name="{name}"><pose>{x:.4f} {y:.4f} {(h0 + h1) / 2:.4f} 0 0 {-math.radians(yaw_deg):.6f}</pose>'
                f'<geometry><box><size>{len_n:.4f} {len_e:.4f} {h1 - h0:.4f}</size></box></geometry></collision>')


def add_cyl(name, a, b, r, h):
    x, y = gz_xy(a, b)
    cols.append(f'<collision name="{name}"><pose>{x:.4f} {y:.4f} {h / 2:.4f} 0 0 0</pose>'
                f'<geometry><cylinder><radius>{r:.4f}</radius><length>{h:.4f}</length></cylinder></geometry></collision>')


cols.append('<collision name="floor"><pose>0 0 -0.05 0 0 0</pose><geometry><box><size>40 40 0.1</size></box></geometry>'
            '<surface><friction><ode><mu>0.9</mu><mu2>0.9</mu2></ode></friction></surface></collision>')
# every course solid with its PHYSICAL size (course_solids.py: maze boards, window board slabs around the hole, wall,
# cinder blocks, astronaut, rocks; layout._solids() adds +0.02 m per board face for pose rejection, not used here);
# the tunnel is built below in more detail than layout's placeholder
sys.path.insert(0, str(Path(__file__).resolve().parent))
import course_solids  # noqa: E402
for name, kind, q in course_solids.solids(L):
    if kind == 'obox':
        ca, cb, hl, hp, yaw, h0, h1 = q
        add_box(name, ca, cb, h0, h1, 2 * hl, 2 * hp, yaw)
    else:
        ca, cb, r, h0, h1 = q
        x, y = gz_xy(ca, cb)
        cols.append(f'<collision name="{name}"><pose>{x:.4f} {y:.4f} {(h0 + h1) / 2:.4f} 0 0 0</pose>'
                    f'<geometry><cylinder><radius>{r:.4f}</radius><length>{h1 - h0:.4f}</length></cylinder></geometry></collision>')
tg = course_solids.tunnel_geom(L); n_seg = 24                # hoops + shell around the tunnel axis (any horizontal axis)
(ea, eb), (ua, ub), ln = tg['entry'], tg['axis'], tg['length']
hoops = [tuple(h) for h in L.tunnel_hoops()] if 'axis_ab' in L.TUNNEL else [(L.TUNNEL['a'], hb) for hb in L.tunnel_hoops()]
# box local frame: x = chord (tangent), y = tunnel axis, z = radial. pose yaw psi turns local y onto the axis in gz
# (gz x = north, y = west): axis_gz = (ua, -ub)  ->  psi = atan2(-ua, -ub); local x after yaw = lateral (cos psi, sin psi)
psi = math.atan2(-ua, -ub); lat = (math.cos(psi), math.sin(psi))
mid_x, mid_y = gz_xy(ea + ua * ln / 2, eb + ub * ln / 2)
for j in range(n_seg):
    ang = 2 * math.pi * (j + 0.5) / n_seg
    chord = 2 * tg['radius'] * math.sin(math.pi / n_seg) + 0.01
    for k, (ha, hb) in enumerate(hoops):
        hx, hy = gz_xy(ha, hb); r = tg['radius']
        x, y, z = hx + r * math.sin(ang) * lat[0], hy + r * math.sin(ang) * lat[1], tg['centre_h'] + r * math.cos(ang)
        cols.append(f'<collision name="hoop{k}_{j}"><pose>{x:.4f} {y:.4f} {z:.4f} 0 {ang:.5f} {psi:.5f}</pose>'
                    f'<geometry><box><size>{chord:.4f} 0.02 0.02</size></box></geometry></collision>')
    rs = tg['mesh_r'] + 0.012
    xs, ys, zs = mid_x + rs * math.sin(ang) * lat[0], mid_y + rs * math.sin(ang) * lat[1], tg['centre_h'] + rs * math.cos(ang)
    if zs > 0.03:
        cols.append(f'<collision name="shell_{j}"><pose>{xs:.4f} {ys:.4f} {zs:.4f} 0 {ang:.5f} {psi:.5f}</pose>'
                    f'<geometry><box><size>{chord:.4f} {ln:.4f} 0.01</size></box></geometry></collision>')
C, H, off = L.CAGE, L.CAGE_H, L.NET_OFFSET
add_box('net_s', -off, C / 2, 0.0, H, 0.02, C + 0.1)
add_box('net_n', C + off, C / 2, 0.0, H, 0.02, C + 0.1)
add_box('net_w', C / 2, -off, 0.0, H, C + 0.1, 0.02)
add_box('net_e', C / 2, C + off, 0.0, H, C + 0.1, 0.02)
add_box('net_roof', C / 2, C / 2, H, H + 0.02, C + 0.1, C + 0.1)

coll_model = ('<model name="venue_collision"><static>true</static><link name="link">\n' + '\n'.join(cols) +
              '\n</link></model>\n')
inc = ''
if len(sys.argv) > 5:                                # the vehicle: at the launch facing +x (north), or at SPAWN
    sf, sl, sy = (float(v) for v in sys.argv[6].split(',')) if len(sys.argv) > 6 and sys.argv[6] else (0.0, 0.0, 0.0)
    inc = (f'<include><uri>{Path(sys.argv[4]).resolve()}</uri><name>{sys.argv[5]}</name>'   # gz x = f, y = l (course start frame)
           f'<pose>{sf:.4f} {sl:.4f} 0.052 0 0 {math.radians(sy):.6f}</pose></include>\n')   # skids 2 mm above the floor
s = s.replace('</world>', coll_model + inc + '</world>', 1)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(s)
print(f'{out}: {len(cols)} collision primitives, step {step:.6f} s')
