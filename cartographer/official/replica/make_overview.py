#!/usr/bin/env python3
"""Overview mosaic + photo comparison for the packaged replica set.

Usage: python make_overview.py PHOTO_DIR [--root outputs/venue-replica]
Writes ROOT/overview.png (top-down map + sample frames with labels + depth +
photo/render pairs) and ROOT/compare_photos.png (larger photo vs render pairs).
"""
import argparse, json, math, os, sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import layout as L                 # noqa: E402
import poses as PZ                 # noqa: E402
from make_textures import load     # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
GROUP_COL = {'route': (200, 200, 200), 'tunnel': (0, 220, 255), 'window': (255, 0, 255), 'pad': (0, 255, 0), 'return': (180, 120, 255), 'gateb': (255, 255, 0),
             'wall': (255, 160, 0), 'start': (0, 0, 255), 'compare': (128, 128, 128)}


def put(img, text, org, scale=0.5, col=(255, 255, 255), th=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def topdown(index, size=900):
    """Map, north up, east right; cage coords a (north), b (east)."""
    m = 60
    s = (size - 2 * m) / (L.CAGE + 1.4)
    img = np.full((size, size, 3), 30, np.uint8)
    P = lambda a, b: (int(m + (b + 0.7) * s), int(size - m - (a + 0.7) * s))
    cv2.rectangle(img, P(-L.TARP, -L.TARP), P(L.CAGE + L.TARP, L.CAGE + L.TARP), (40, 80, 200), -1)
    cv2.rectangle(img, P(0, 0), P(L.CAGE, L.CAGE), (20, 20, 20), 2)
    tex = json.load(open(os.path.join(HERE, 'assets', 'textures', 'textures.json')))
    for r in tex['painted_rocks']:
        n = L.floor_uv_to_ned(r['u_m'], r['v_m'])
        cv2.circle(img, P(n[0] + L.START[0], n[1] + L.START[1]), int(r['size_m'] / 2 * s * 0.7), (60, 60, 80), -1)
    for name, pts in L.TAPE:                               # black floor tape (start box + maze path)
        Q = np.array([P(f + L.START[0], -l + L.START[1]) for f, l in pts], np.int32)
        cv2.polylines(img, [Q], False, (15, 15, 15), 3, cv2.LINE_AA)
    put(img, 'start', P(L.START[0] - 0.62, L.START[1] - 0.35), 0.45)
    for p in L.PANELS:
        y = math.radians(p['yaw']); da, db = math.cos(y) * p['w'] / 2, math.sin(y) * p['w'] / 2
        col = (220, 220, 230) if p['skin'] == 'banner' else (120, 170, 210)       # banner / plywood
        cv2.line(img, P(p['a'] - da, p['b'] - db), P(p['a'] + da, p['b'] + db), col, 6)
    T = L.TUNNEL; hb = L.tunnel_hoops(); ra, rb = L.tunnel_right(); R0 = T['radius']
    box = [(hb[0][0] + k * R0 * ra, hb[0][1] + k * R0 * rb) for k in (-1, 1)] + \
          [(hb[-1][0] + k * R0 * ra, hb[-1][1] + k * R0 * rb) for k in (1, -1)]
    cv2.fillPoly(img, [np.array([P(*q) for q in box], np.int32)], (60, 90, 160))
    for (ha, hb_) in hb:
        cv2.line(img, P(ha - R0 * ra, hb_ - R0 * rb), P(ha + R0 * ra, hb_ + R0 * rb), (60, 190, 230), 3)
    put(img, 'tunnel (in N, out S)', P(hb[0][0] + 0.2, hb[0][1] - 0.9), 0.45, (60, 190, 230))
    W = L.WINDOW; y = np.radians(W['yaw']); da, db = np.cos(y), np.sin(y)
    cv2.line(img, P(W['a'] - da * W['width'] / 2, W['b'] - db * W['width'] / 2),
             P(W['a'] + da * W['width'] / 2, W['b'] + db * W['width'] / 2), (120, 170, 210), 6)
    cv2.line(img, P(W['a'] - da * W['hole_w'] / 2, W['b'] - db * W['hole_w'] / 2),
             P(W['a'] + da * W['hole_w'] / 2, W['b'] + db * W['hole_w'] / 2), (255, 0, 255), 3)
    put(img, 'window', P(W['a'] - 0.1, W['b'] + 0.25), 0.5, (255, 0, 255))
    pr = L.PAD
    cv2.circle(img, P(pr['n'] + L.START[0], pr['e'] + L.START[1]), int(pr['ring_r'] * s), (235, 235, 235), 2)
    put(img, 'barrier (fly over on return)', P(L.WALL['y'] * L.FT + 0.08, 0.2), 0.42, (255, 160, 0))
    pc = L.pad_corners_ned()
    cv2.fillPoly(img, [np.array([P(c[0] + L.START[0], c[1] + L.START[1]) for c in pc], np.int32)], (215, 165, 70))
    put(img, 'landing pad', P(L.PAD['n'] + L.START[0] - 0.55, L.PAD['e'] + L.START[1] - 0.45), 0.5, (0, 255, 0))
    for (bn, be, yaw) in L.CINDER_BLOCKS:
        cv2.circle(img, P(bn + L.START[0], be + L.START[1]), 5, (170, 170, 170), -1)
    for (a, b, h, r) in L.ROCKS:
        cv2.circle(img, P(a, b), max(3, int(r * s)), (40, 60, 120), -1)
        cv2.circle(img, P(a, b), max(3, int(r * s)), (0, 0, 255), 1)
    D, _ = PZ.smooth_route(PZ.ROUTE)
    cv2.polylines(img, [np.array([P(a, b) for a, b in D], np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
    for f in index:
        n, e = f['cam_ned'][0], f['cam_ned'][1]
        a, b = n + L.START[0], e + L.START[1]
        hd = math.radians(f['heading_deg'])
        p0 = P(a, b); p1 = P(a + 0.18 * math.cos(hd), b + 0.18 * math.sin(hd))
        col = GROUP_COL.get(f['group'], (255, 255, 255))
        cv2.circle(img, p0, 3, col, -1); cv2.line(img, p0, p1, col, 1, cv2.LINE_AA)
    put(img, 'Top-down (north up, east right), COURSE_MAP.md layout, 20 x 20 ft (6.10 m) cage', (10, 22), 0.55)
    put(img, 'dark discs = craters printed on the tarp, red rim = boulders, grey dots = cinder blocks', (10, 42), 0.45)
    y = 64
    for g, c in GROUP_COL.items():
        put(img, g, (12, y), 0.45, c); y += 18
    return img


def depth_vis(d, dmax=7.0):
    v = np.clip(1 - np.nan_to_num(d, nan=dmax) / dmax, 0, 1)
    im = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    im[~np.isfinite(d)] = 0
    return im


def tile(root, f, w=400):
    h = int(w * 0.625)
    qa = cv2.imread(os.path.join(root, 'qa', f['id'] + '.jpg'))
    d = np.load(os.path.join(root, f['file']))['depth']
    a = cv2.resize(qa, (w, h), interpolation=cv2.INTER_AREA)
    b = cv2.resize(depth_vis(d), (w, h), interpolation=cv2.INTER_NEAREST)
    put(a, f"{f['id']}", (6, 16), 0.42, GROUP_COL.get(f['group'], (255, 255, 255)))
    put(b, f"stereo depth (0-7 m, black = hole), h={f['height_m']:.2f} m", (6, 16), 0.42)
    return np.concatenate([a, b], 1)


def compare_pairs(root, photo_dir, w=640):
    pairs = [(10, 'compare_photo10_start_maze'), (12, 'compare_photo12_corridor'), (13, 'compare_photo13_maze_lane'),
             (16, 'compare_photo16_tunnel'), (17, 'compare_photo17_window_pad')]
    rows = []
    for i, fid in pairs:
        ph = load(photo_dir, i)
        ph = cv2.resize(ph, (int(w * 4 / 3 * 0.625), int(w * 0.625)), interpolation=cv2.INTER_AREA)
        r = np.load(os.path.join(root, 'frames', fid + '.npz'))
        rgb = cv2.cvtColor(r['rgb'], cv2.COLOR_RGB2BGR)
        rgb = cv2.resize(rgb, (w, int(w * 0.625)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(cv2.resize(r['image'], (w, int(w * 0.625))), cv2.COLOR_GRAY2BGR)
        put(ph, f'venue photo {i}', (8, 22), 0.6); put(rgb, f'replica render ({fid})', (8, 22), 0.6)
        put(gray, 'replica `image` (gray, what perception gets)', (8, 22), 0.6)
        rows.append(np.concatenate([ph, rgb, gray], 1))
    return np.concatenate(rows, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('photo_dir')
    ap.add_argument('--root', default=os.path.join(REPO, 'outputs', 'venue-replica'))
    a = ap.parse_args()
    idx = json.load(open(os.path.join(a.root, 'index.json')))
    frames = [f for f in idx['frames'] if f['group'] != 'compare']
    pick = []
    for h in ('h040', 'h080', 'h120'):
        r = [f for f in frames if f['id'].startswith(f'route_{h}')]
        pick += [r[int(k)] for k in np.linspace(0, len(r) - 1, 4)]
    for g, ids in (('tunnel', ['tunnel_d100_l+000_h045', 'tunnel_d060_l-035_h080']),
                   ('window', ['window_d140_l+000_h130', 'window_d080_l+045_h100']),
                   ('pad', ['pad_d120_az090_h045', 'padf_d155_az150_h045']),
                   ('wall', ['wall_climb_h040', 'wall_over_k04']), ('start', ['start_pad_yaw+00'])):
        pick += [f for f in frames if f['id'] in ids]
    tiles = [tile(a.root, f) for f in pick]
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)
    td = topdown(idx['frames'], size=grid.shape[1] // 2)
    cmp_ = compare_pairs(a.root, a.photo_dir, w=400)
    cmp_ = cv2.resize(cmp_, (grid.shape[1] - td.shape[1], int(cmp_.shape[0] * (grid.shape[1] - td.shape[1]) / cmp_.shape[1])))
    top_h = max(td.shape[0], cmp_.shape[0])
    pad = lambda im: np.pad(im, ((0, top_h - im.shape[0]), (0, 0), (0, 0)))
    top = np.concatenate([pad(td), pad(cmp_)], 1)
    head = np.full((50, grid.shape[1], 3), 20, np.uint8)
    put(head, f"DroneHacks 2026 venue replica - {len(frames)} frames (+{len(idx['frames']) - len(frames)} photo-compare views), "
              "Gazebo Harmonic headless, OFFICIAL_K 640x400 horizontal; overlays: yellow=tunnel entry, magenta=window hole, "
              "green=landing-pad outline, orange=barrier top edge, red=boulder tops, cyan=printed craters", (10, 32), 0.62)
    out = np.concatenate([head, top, grid], 0)
    cv2.imwrite(os.path.join(a.root, 'overview.png'), out)
    cv2.imwrite(os.path.join(a.root, 'compare_photos.png'), compare_pairs(a.root, a.photo_dir, w=640))
    print('overview', out.shape, '->', a.root)


if __name__ == '__main__':
    main()
