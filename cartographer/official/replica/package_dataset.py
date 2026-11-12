#!/usr/bin/env python3
"""Turn the raw Gazebo renders into the perception test set.

Usage: python package_dataset.py --raw RAW_DIR [--out outputs/venue-replica]
RAW_DIR holds raw_full/<id>.npz (rgb, rgb_right, depth_full) and raw_depth/<id>.npz (depth
of the scene without net / tunnel mesh). Writes OUT/frames/<id>.npz, OUT/index.json, OUT/qa/.

Per-frame npz (common/geometry.py `Frame` conventions: NED world, optical camera):
  image       uint8 (400,640)  left mono camera as the stack sees it: RGB->gray (2x supersampled)
                               + sensor model (gain wobble, read+shot noise, slight blur)
  image_right uint8            right eye, same K, 7.5 cm to the right, same sensor model
  depth       float32          PRIMARY: OAK-D-like stereo depth from (image, image_right) via
                               stereo_depth.sgbm_depth; metres along optical z, NaN = no return
  depth_ideal float32          ground truth without cage net / tunnel mesh (old `depth`)
  depth_full  float32          ground truth of every surface incl. net + tunnel mesh
  surface     uint8            per-pixel surface type (surfaces.NAMES, index.json 'surface_names')
  rgb         uint8 (400,640,3) clean colour left image (diagnostics only)
  gain        float            per-frame exposure/gain factor used by the sensor model
  K (4,) fx fy cx cy = OFFICIAL_K; R_nc (3,3) optical->NED; cam_ned (3,); body_ned (3,)
  heading (rad, NED yaw), stamp (sim s), labels (JSON string: ground truth, OFFLINE ONLY)
Load with replica.dataset.load_frame(path) -> (Frame, labels dict).
"""
import argparse, json, math, os, sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import layout as L                                         # noqa: E402
import stereo_depth as SD                                   # noqa: E402
import surfaces as SF                                       # noqa: E402
from common.geometry import Frame, OFFICIAL_K, OFFICIAL_SIZE   # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))


def visible(frame, depth_full, P, tol=0.06):
    """Project NED points; returns uv (N,2), z (N,), in_fov (N,), unoccluded (N,)."""
    uv, z = frame.project(P)
    W, H = OFFICIAL_SIZE
    inf = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
    vis = np.zeros(len(P), bool)
    for i in np.nonzero(inf)[0]:
        u, v = int(round(uv[i, 0])), int(round(uv[i, 1]))
        u0, u1, v0, v1 = max(u - 1, 0), min(u + 2, W), max(v - 1, 0), min(v + 2, H)
        d = depth_full[v0:v1, u0:u1]
        d = d[np.isfinite(d)]
        vis[i] = d.size > 0 and d.max() >= z[i] - (tol + 0.02 * z[i])
    return uv, z, inf, vis


def ring_points(center, radius, axis_ne, n=36):
    ax = np.array([axis_ne[0], axis_ne[1], 0.0]); ax /= np.linalg.norm(ax)
    side = np.array([-ax[1], ax[0], 0.0]); up = np.array([0, 0, -1.0])
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.asarray(center) + radius * (np.cos(t)[:, None] * side + np.sin(t)[:, None] * up)


def frame_labels(frame, depth_full, W):
    out = {}
    def pack(name, P, extra=None, need_all=False):
        uv, z, inf, vis = visible(frame, depth_full, np.asarray(P, float))
        rec = {'px': uv.round(2).tolist(), 'z': z.round(4).tolist(), 'in_fov': inf.tolist(), 'unoccluded': vis.tolist(),
               'visible_frac': float(vis.mean()), 'fully_in_fov': bool(inf.all())}
        if extra:
            rec.update(extra)
        out[name] = rec
    t = W['tunnel']
    pack('tunnel_entry_center', [t['entry_ned']])
    pack('tunnel_entry_ring', ring_points(t['entry_ned'], t['radius_m'], t['axis_ned']))
    w = W['window']
    c = np.array(w['center_ned']); n = np.array(w['normal_ned']); side = np.cross(n, [0, 0, 1.0]); up = np.array([0, 0, -1.0])
    hw, hh = w['width_m'] / 2, w['height_m'] / 2
    pack('window_center', [c])
    pack('window_hole_corners', [c + side * sx * hw + up * sy * hh for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1))])
    s = W['landing_target']
    tc = np.array(s['top_center_ned']); ln, le = s['extent_m'][0] / 2, s['extent_m'][1] / 2
    pack('landing_top_center', [tc])
    corners = s.get('corners_ned') or [tc + [sn * ln, se * le, 0] for sn, se in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    pack('landing_top_corners', corners)
    wl = W['wall']
    wc = np.array(wl['center_ned']); hwid = wl['width_m'] / 2
    pack('wall_top_edge', wl.get('top_edge_ned') or
         [wc + [sx * hwid, 0, -(wl['top_h'] - (wl['top_h'] + wl['bottom_h']) / 2)] for sx in (-1, 1)])
    rocks = []
    for k, r in enumerate(W['rocks']):
        P = [r['apex_ned'], (np.array(r['base_ned']) + np.array(r['apex_ned'])) / 2]
        uv, z, inf, vis = visible(frame, depth_full, np.asarray(P, float))
        if inf.any():
            rocks.append({'index': k, 'apex_px': uv[0].round(2).tolist(), 'mid_px': uv[1].round(2).tolist(),
                          'range_m': float(z[1]), 'unoccluded': bool(vis.any())})
    out['rocks_in_view'] = rocks
    flat = []
    for k, r in enumerate(W.get('painted_rocks_flat', [])):
        uv, z, inf, vis = visible(frame, depth_full, np.asarray([r['center_ned']], float))
        if inf[0] and vis[0]:
            flat.append({'index': k, 'px': uv[0].round(2).tolist(), 'range_m': float(z[0])})
    out['painted_rocks_in_view'] = flat
    return out


def draw_overlay(img, lab):
    """Label overlay for QA: bright = unoccluded, dark = behind something."""
    o = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    def pts(rec, only_vis=False):
        return [(int(round(u)), int(round(v))) for (u, v), f, s in zip(rec['px'], rec['in_fov'], rec['unoccluded'])
                if f and (s or not only_vis)]
    dim = lambda c, ok: c if ok else tuple(x // 3 for x in c)
    r = lab['tunnel_entry_ring']
    for (u, v), f, s in zip(r['px'], r['in_fov'], r['unoccluded']):
        if f:
            cv2.circle(o, (int(round(u)), int(round(v))), 2, dim((0, 255, 255), s), -1)
    for name, col in (('tunnel_entry_center', (0, 255, 255)), ('window_center', (255, 0, 255))):
        for p, s in zip(pts(lab[name]), lab[name]['unoccluded']):
            cv2.drawMarker(o, p, dim(col, lab[name]['visible_frac'] > 0 or name == 'tunnel_entry_center'), cv2.MARKER_CROSS, 14, 2)
    for name, col in (('window_hole_corners', (255, 0, 255)), ('landing_top_corners', (0, 255, 0))):
        c = lab[name]
        if all(c['in_fov']):
            cv2.polylines(o, [np.array(pts(c), np.int32)], True, dim(col, c['visible_frac'] >= 0.5), 2)
    for p in pts(lab['wall_top_edge'], only_vis=True):
        cv2.circle(o, p, 4, (255, 128, 0), -1)
    for r in lab['rocks_in_view']:
        cv2.drawMarker(o, tuple(int(round(x)) for x in r['apex_px']), dim((0, 0, 255), r['unoccluded']),
                       cv2.MARKER_TRIANGLE_UP, 10, 2)
    for r in lab['painted_rocks_in_view']:
        cv2.drawMarker(o, tuple(int(round(x)) for x in r['px']), (255, 255, 0), cv2.MARKER_DIAMOND, 8, 1)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True)
    ap.add_argument('--poses', default=os.path.join(HERE, 'world', 'poses.json'))
    ap.add_argument('--textures-json', default=os.path.join(HERE, 'assets', 'textures', 'textures.json'))
    ap.add_argument('--out', default=os.path.join(REPO, 'outputs', 'venue-replica'))
    a = ap.parse_args()
    poses = json.load(open(a.poses))['poses']
    painted = json.load(open(a.textures_json))['painted_rocks']
    W = L.labels_world(painted)
    os.makedirs(os.path.join(a.out, 'frames'), exist_ok=True)
    os.makedirs(os.path.join(a.out, 'qa'), exist_ok=True)
    index, bad = [], []
    for p in poses:
        fa = os.path.join(a.raw, 'raw_full', p['id'] + '.npz'); fb = os.path.join(a.raw, 'raw_depth', p['id'] + '.npz')
        if not (os.path.exists(fa) and os.path.exists(fb)):
            bad.append((p['id'], 'missing')); continue
        A, B = np.load(fa), np.load(fb)
        rgb = A['rgb']; depth_full = A['depth_full'].astype(np.float32); depth_ideal = B['depth'].astype(np.float32)
        for d in (depth_ideal, depth_full):
            d[~np.isfinite(d) | (d <= 0)] = np.nan
        gray, gray_r, gain = SD.sensor_model(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                                             cv2.cvtColor(A['rgb_right'], cv2.COLOR_RGB2GRAY), SD.frame_seed(p['id']))
        depth, _, _ = SD.sgbm_depth(gray, gray_r, OFFICIAL_K[0])
        R_nc = np.array(p['R_nc']); cam = np.array(p['cam_ned'])
        fr = Frame(image=gray, depth=depth, K=OFFICIAL_K, R_nc=R_nc, cam_ned=cam, stamp=float(A['stamp']),
                   body_ned=np.array(p['body_ned']), heading=p['heading'])
        surface = SF.classify(depth_ideal, depth_full, OFFICIAL_K, R_nc, cam)
        # pass consistency: removing net/mesh can only make depth larger or equal
        both = np.isfinite(depth_ideal) & np.isfinite(depth_full)
        closer = float(np.mean(depth_ideal[both] < depth_full[both] - 0.01)) if both.any() else 1.0
        zmin = OFFICIAL_K[0] * SD.BASELINE_M / (SD.SGBM['numDisparities'] - 1)   # device minimum range
        # stereo sanity statistic over well-textured structure only (floor is glossy and seen at grazing
        # angles, net / mesh / hall / pipes are expected to be wrong): catches a broken stereo geometry
        textured = [SF.ID[k] for k in ('banner_panel', 'brass_ring', 'tunnel_fabric', 'tan_board', 'landing_pad',
                                        'cinder_block', 'rock')]
        solid = (np.isfinite(depth) & np.isfinite(depth_ideal) & (depth_ideal >= zmin) & (depth_ideal < 6)
                 & np.isin(surface, textured))
        rel = np.abs(depth[solid] - depth_ideal[solid]) / depth_ideal[solid]
        lab = {'world': W, 'frame': frame_labels(fr, depth_full, W),
               'pose': {'group': p['group'], 'height_m': -cam[2], 'roll_deg': p['roll_deg'], 'pitch_deg': p['pitch_deg']},
               'warning': 'ground truth for offline evaluation only; never feed into flight code'}
        np.savez_compressed(os.path.join(a.out, 'frames', p['id'] + '.npz'), image=gray, image_right=gray_r,
                            depth=depth, depth_ideal=depth_ideal, depth_full=depth_full, surface=surface,
                            rgb=rgb, gain=np.float64(gain), K=np.array(OFFICIAL_K, np.float64), R_nc=R_nc, cam_ned=cam,
                            body_ned=np.array(p['body_ned']), heading=np.float64(p['heading']),
                            stamp=np.float64(A['stamp']), labels=np.array(json.dumps(lab)))
        fl = lab['frame']
        vis = {'tunnel_entry': bool(fl['tunnel_entry_center']['in_fov'][0] and fl['tunnel_entry_ring']['visible_frac'] > 0.3),
               'window': bool(fl['window_hole_corners']['fully_in_fov'] and fl['window_center']['unoccluded'][0]
                              and fl['window_hole_corners']['visible_frac'] >= 0.5),
               'landing_target': bool(fl['landing_top_corners']['fully_in_fov'] and fl['landing_top_center']['unoccluded'][0]),
               'wall': bool(any(fl['wall_top_edge']['unoccluded'])),
               'rocks': [r['index'] for r in fl['rocks_in_view'] if r['unoccluded']],
               'painted_rocks': [r['index'] for r in fl['painted_rocks_in_view']]}
        index.append({'file': f'frames/{p["id"]}.npz', 'id': p['id'], 'group': p['group'], 'cam_ned': p['cam_ned'],
                      'height_m': round(-cam[2], 3), 'heading_deg': round(math.degrees(p['heading']), 2),
                      'roll_deg': p['roll_deg'], 'pitch_deg': p['pitch_deg'], 'visible': vis,
                      'depth_valid_frac': round(float(np.isfinite(depth).mean()), 4),
                      'depth_ideal_valid_frac': round(float(np.isfinite(depth_ideal).mean()), 4),
                      'stereo_rel_err_p50_solid': round(float(np.median(rel)), 4) if rel.size >= 2000 else None,
                      'stereo_bad10_frac_solid': round(float(np.mean(rel > 0.1)), 4) if rel.size >= 2000 else None,
                      'gain': round(float(gain), 3),
                      'pass_mismatch_frac': round(closer, 5)})
        if closer > 0.001:
            bad.append((p['id'], f'depth passes disagree on {closer:.3%} of pixels'))
        cv2.imwrite(os.path.join(a.out, 'qa', p['id'] + '.jpg'), draw_overlay(gray, lab['frame']), [cv2.IMWRITE_JPEG_QUALITY, 85])
    from collections import Counter
    meta = {
        'name': 'DroneHacks 2026 venue replica (Gazebo Harmonic, headless ogre2)',
        'created': '2026-11-06', 'n_frames': len(index), 'groups': dict(Counter(f['group'] for f in index)),
        'conventions': 'common/geometry.py Frame: NED world (origin = start pad centre on the floor), optical camera '
                       '(x right, y down, z forward); depth = metres along optical z, NaN = no return',
        'K': list(OFFICIAL_K), 'size': list(OFFICIAL_SIZE),
        'fields': ['image', 'image_right', 'depth', 'depth_ideal', 'depth_full', 'surface', 'rgb', 'gain', 'K', 'R_nc',
                   'cam_ned', 'body_ned', 'heading', 'stamp', 'labels'],
        'depth_note': '`depth` = OAK-D-like stereo depth (SGBM on the noisy left/right pair, see stereo_params), NaN = no return; '
                      'the stereo pair SEES the cage net and the tunnel black mesh. `depth_ideal` = noise-free truth without '
                      'net/mesh; `depth_full` = noise-free truth of every surface.',
        'stereo_params': SD.params(),
        'surface_names': SF.NAMES,
        'labels_note': 'json.loads(str(npz["labels"])): world ground truth (estimated venue geometry) + per-frame projections. '
                       'OFFLINE EVALUATION ONLY.',
        'world_labels': W,
        'frames': index,
        'problems': bad,
    }
    json.dump(meta, open(os.path.join(a.out, 'index.json'), 'w'), indent=1)
    print(len(index), 'frames;', meta['groups'], '; problems:', bad[:10])


if __name__ == '__main__':
    main()
