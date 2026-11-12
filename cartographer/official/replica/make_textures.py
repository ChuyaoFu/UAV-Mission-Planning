#!/usr/bin/env python3
"""Build the replica's textures from the 2026-11-06 venue photos (offline, Mac/Linux).

Usage:  python make_textures.py PHOTO_DIR [--out assets/textures]
PHOTO_DIR holds 1.webp … 7.webp (or .jpg). Every crop is taken through the cage
net, so the thin dark net cords are removed with a black-hat mask + inpainting;
details the inpainting destroys (panel vents/seams) are redrawn on top. The hall,
plywood and helipad textures are procedural (colours sampled from the photos).
Writes JPEG/PNG textures and `textures.json` (painted-rock decal positions on
the floor, needed for the ground-truth labels).
"""
import argparse, json, os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
FLOOR_PX = 4096          # floor texture side (covers layout.FLOOR_SIZE metres)


def load(photo_dir, i):
    for ext in ('jpg', 'png', 'webp'):
        p = os.path.join(photo_dir, f'{i}.{ext}')
        if os.path.exists(p):
            im = cv2.imread(p)
            if im is None and ext == 'webp':
                from PIL import Image
                im = cv2.cvtColor(np.array(Image.open(p).convert('RGB')), cv2.COLOR_RGB2BGR)
            if im is not None:
                return im
    raise FileNotFoundError(f'photo {i} not in {photo_dir}')


def denet(img, k=11, thr=30, extra=None, radius=4):
    """Remove thin dark net cords: black-hat > thr -> inpaint."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = (bh > thr).astype(np.uint8) * 255
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    if extra is not None:
        m = np.maximum(m, extra)
    return cv2.inpaint(img, m, radius, cv2.INPAINT_TELEA)


def flatten_light(img, sigma=60, strength=0.85):
    """Divide out baked low-frequency lighting (keeps the local texture)."""
    f = img.astype(np.float32) + 1
    low = cv2.GaussianBlur(f, (0, 0), sigma)
    mean = f.reshape(-1, 3).mean(0)
    out = f / low * mean
    return np.clip(strength * out + (1 - strength) * f, 0, 255).astype(np.uint8)


def mirror_tile(tile, h, w):
    t = np.concatenate([tile, tile[:, ::-1]], 1)
    t = np.concatenate([t, t[::-1]], 0)
    reps = (h // t.shape[0] + 1, w // t.shape[1] + 1, 1)
    return np.tile(t, reps)[:h, :w]


def value_noise(h, w, cell, rng):
    g = rng.random((h // cell + 2, w // cell + 2)).astype(np.float32)
    return cv2.resize(g, (w + 2 * cell, h + 2 * cell), interpolation=cv2.INTER_CUBIC)[cell:cell + h, cell:cell + w]


# ----------------------------------------------------------------------------- floor
def _sample(img, cond):
    px = img[cond]
    return np.median(px, axis=0) if len(px) > 50 else None


def floor_texture(ph, rng, floor_size_m):
    """Orange printed floor. Colours and the fine print/grain come from de-netted
    crops of photos 4/5; the large painted rocks and pebbles are redrawn in the
    photo's style (the net cannot be cleanly removed over the dark print)."""
    N = FLOOR_PX
    ppm = N / floor_size_m
    p5 = denet(ph[5][560:880, 60:1240], k=13, thr=28)
    hsv = cv2.cvtColor(p5, cv2.COLOR_BGR2HSV)
    orange = _sample(p5, (hsv[..., 1] > 120) & (hsv[..., 2] > 120))
    p4 = ph[4]
    rock4 = p4[800:960, 930:1180]                      # big painted rock (photo 4)
    v4 = rock4.max(2)
    dark = _sample(rock4, (v4 > 45) & (v4 < 110))
    pal = dict(orange=orange, dark=dark if dark is not None else np.array((58, 62, 80.)),
               light=np.array((70, 102, 178.)), outline=np.array((30, 35, 50.)),
               shadow=orange * 0.72, pebble=np.array((150, 190, 228.)), crater=orange * 0.8)
    # base: photo colour x low-frequency mottling + fine high-pass detail of photo 5
    g = p5.astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 5)
    hp = cv2.resize(hp, (1150, 760), interpolation=cv2.INTER_CUBIC)
    tex = (orange[None, None] * (0.93 + 0.14 * value_noise(N, N, 512, rng))[..., None]
           * (0.97 + 0.06 * value_noise(N, N, 128, rng))[..., None])
    tex += 0.45 * mirror_tile(hp, N, N)
    tex = np.clip(tex, 0, 255)
    col = lambda c: tuple(float(x) for x in c)
    # craters / pebbles (photo 4/5 style)
    for _ in range(int(16 * floor_size_m ** 2)):
        x, y = rng.uniform(0, N, 2).astype(int)
        r = int(rng.uniform(0.006, 0.02) * ppm)
        cv2.ellipse(tex, (x, y), (r, max(2, int(r * 0.7))), 0, 0, 360, col(pal['crater']), -1, cv2.LINE_AA)
    for _ in range(int(9 * floor_size_m ** 2)):
        x, y = rng.uniform(0, N, 2).astype(int)
        rx = int(rng.uniform(0.012, 0.03) * ppm); ry = max(3, int(rx * rng.uniform(0.45, 0.7)))
        ang = float(rng.uniform(-20, 20))
        cv2.ellipse(tex, (x + 3, y + 4), (rx, ry), ang, 0, 360, col(pal['shadow']), -1, cv2.LINE_AA)
        cv2.ellipse(tex, (x, y), (rx, ry), ang, 0, 360, col(pal['pebble']), -1, cv2.LINE_AA)
        cv2.ellipse(tex, (x, y), (rx, ry), ang, 0, 360, col(pal['outline']), 2, cv2.LINE_AA)
    placed = []
    for (u, v, size) in DECALS:
        cx, cy = u * ppm, v * ppm
        k = int(rng.integers(7, 11))
        ang = np.sort(rng.uniform(0, 2 * np.pi, k))
        el = rng.uniform(0.55, 0.9)
        rot = rng.uniform(0, np.pi)
        rad = size / 2 * ppm * rng.uniform(0.65, 1.0, k)
        xs, ys = rad * np.cos(ang), rad * np.sin(ang) * el
        pts = np.stack([xs * np.cos(rot) - ys * np.sin(rot) + cx, xs * np.sin(rot) + ys * np.cos(rot) + cy], -1)
        poly = pts.astype(np.int32)
        sh = (poly + np.array([0.04 * ppm, 0.05 * ppm])).astype(np.int32)
        over = tex.copy()
        cv2.fillPoly(over, [sh], col(pal['shadow']), cv2.LINE_AA)
        tex = 0.5 * tex + 0.5 * over
        cv2.fillPoly(tex, [poly], col(pal['dark']), cv2.LINE_AA)
        c0 = poly.mean(0)
        sub = (c0 + (poly - c0) * 0.55 + np.array([0.08, -0.06]) * size * ppm).astype(np.int32)
        cv2.fillPoly(tex, [sub[: max(4, k - 2)]], col(pal['light'] * rng.uniform(0.85, 1.05)), cv2.LINE_AA)
        cv2.polylines(tex, [poly], True, col(pal['outline']), 5, cv2.LINE_AA)
        placed.append({'u_m': u, 'v_m': v, 'size_m': size})
    return np.clip(tex, 0, 255).astype(np.uint8), placed


# painted (flat) rock layout on the floor texture: metres from the texture's
# north-west corner, +u = east, +v = south (see layout.floor_uv_to_ned); size = m.
DECALS = [(1.6, 1.0, 0.85), (3.6, 2.4, 0.55), (2.2, 4.4, 0.9), (4.6, 4.8, 0.7), (0.8, 3.2, 0.6),
          (5.4, 1.4, 0.6), (3.0, 0.6, 0.5), (5.6, 3.2, 0.75), (2.6, 3.0, 0.45), (4.4, 0.9, 0.5)]


def landing_pad_texture(size=1024):
    """Blue folding landing pad (photos 17, 18): light blue square with rounded corners, dark H,
    white ring + dashed inner ring, 4 dark bars along the edges, faint fold creases. BGR."""
    blue, dark, white = (205, 158, 62), (62, 66, 72), (238, 240, 240)
    t = np.zeros((size, size, 3), np.uint8)
    t[:] = (40, 70, 190)                                   # outside the rounded corners: floor-ish
    r = int(0.05 * size)
    cv2.rectangle(t, (r, 0), (size - r, size), blue, -1)
    cv2.rectangle(t, (0, r), (size, size - r), blue, -1)
    for cx, cy in ((r, r), (size - r, r), (r, size - r), (size - r, size - r)):
        cv2.circle(t, (cx, cy), r, blue, -1, cv2.LINE_AA)
    c = size // 2
    crease = tuple(int(x * 0.9) for x in blue)
    cv2.line(t, (c, 0), (c, size), crease, max(2, size // 300), cv2.LINE_AA)
    cv2.line(t, (0, c), (size, c), crease, max(2, size // 300), cv2.LINE_AA)
    cv2.circle(t, (c, c), int(0.36 * size), white, max(3, size // 160), cv2.LINE_AA)
    for k in range(64):                                    # dashed inner ring
        if k % 2 == 0:
            cv2.ellipse(t, (c, c), (int(0.33 * size),) * 2, 0, k * 360 / 64, (k + 1) * 360 / 64, white, max(1, size // 500), cv2.LINE_AA)
    for a0 in (20, 110, 200, 290):                         # the four white arcs around the H
        cv2.ellipse(t, (c, c), (int(0.25 * size),) * 2, 0, a0, a0 + 50, white, max(3, size // 120), cv2.LINE_AA)
    w = int(0.045 * size); hh = int(0.16 * size)
    cv2.rectangle(t, (c - hh, c - hh), (c - hh + 2 * w, c + hh), dark, -1)      # H legs
    cv2.rectangle(t, (c + hh - 2 * w, c - hh), (c + hh, c + hh), dark, -1)
    cv2.rectangle(t, (c - hh, c - w), (c + hh, c + w), dark, -1)                # H bar
    bl, bw, inset = int(0.30 * size), int(0.055 * size), int(0.07 * size)
    cv2.rectangle(t, (c - bl // 2, inset), (c + bl // 2, inset + bw), dark, -1)                     # 4 edge bars
    cv2.rectangle(t, (c - bl // 2, size - inset - bw), (c + bl // 2, size - inset), dark, -1)
    cv2.rectangle(t, (inset, c - bl // 2), (inset + bw, c + bl // 2), dark, -1)
    cv2.rectangle(t, (size - inset - bw, c - bl // 2), (size - inset, c + bl // 2), dark, -1)
    return t


def helipad(size=600):
    """Red start pad with white ring + H (photo 4)."""
    t = np.zeros((size, size, 3), np.uint8)
    t[:] = (60, 52, 200)                                  # BGR red, warm-lit
    c = size // 2
    cv2.circle(t, (c, c), int(size * 0.36), (225, 225, 235), max(3, size // 80), cv2.LINE_AA)
    for a in range(0, 360, 45):                          # dashed outer ring ticks
        r0, r1 = size * 0.40, size * 0.46
        p0 = (int(c + r0 * np.cos(np.radians(a))), int(c + r0 * np.sin(np.radians(a))))
        p1 = (int(c + r1 * np.cos(np.radians(a))), int(c + r1 * np.sin(np.radians(a))))
        cv2.line(t, p0, p1, (225, 225, 235), max(3, size // 80), cv2.LINE_AA)
    w = size // 14
    cv2.rectangle(t, (c - 4 * w, c - 4 * w), (c - 2 * w, c + 4 * w), (235, 235, 240), -1)
    cv2.rectangle(t, (c + 2 * w, c - 4 * w), (c + 4 * w, c + 4 * w), (235, 235, 240), -1)
    cv2.rectangle(t, (c - 2 * w, c - w), (c + 2 * w, c + w), (235, 235, 240), -1)
    return t


# ----------------------------------------------------------------------------- panels
def panel_texture(ph):
    """Printed sci-fi banner (photo 6 centre panel), de-netted, with the fine
    vent/seam/rivet details redrawn crisp on top."""
    c = ph[6][450:1100, 400:1105].copy()
    h, w = c.shape[:2]
    pipes = np.zeros((h, w), np.uint8)
    pipes[:, 96:140] = 255                               # vertical cage pipe
    pipes[55:90, :445] = 255                             # horizontal cage pipe
    pipes[600:, :160] = 255                              # timestamp overlay corner
    base = denet(c, k=11, thr=30, extra=pipes, radius=6)
    base = cv2.bilateralFilter(base, 7, 25, 7)
    ink = (88, 70, 62)                                   # dark blue-grey line colour (BGR)
    def slats(x0, x1, y0, y1, n):
        for k in range(n):
            y = int(y0 + (k + 0.5) * (y1 - y0) / n)
            cv2.line(base, (x0, y), (x1, y), ink, 6, cv2.LINE_AA)
    cv2.rectangle(base, (360, 108), (472, 282), ink, 3, cv2.LINE_AA)
    slats(375, 458, 118, 272, 11)
    for (y0, y1) in ((428, 482), (512, 566)):
        cv2.rectangle(base, (6, y0 - 6), (128, y1 + 6), ink, 3, cv2.LINE_AA)
        slats(10, 120, y0, y1, 5)
    for x in (338, 520):
        cv2.line(base, (x, 82), (x, 330), ink, 2, cv2.LINE_AA)
    cv2.line(base, (0, 82), (w, 82), ink, 2, cv2.LINE_AA)
    cv2.line(base, (0, 400), (w, 400), ink, 2, cv2.LINE_AA)
    cv2.rectangle(base, (175, 402), (515, 600), ink, 2, cv2.LINE_AA)
    cv2.rectangle(base, (520, 402), (700, 600), ink, 2, cv2.LINE_AA)
    for (x, y) in ((190, 418), (500, 418), (190, 585), (500, 585), (350, 300), (510, 300),
                   (535, 418), (685, 585), (20, 300), (330, 95), (530, 95), (690, 300)):
        cv2.circle(base, (x, y), 5, ink, 2, cv2.LINE_AA)
    return base


# ----------------------------------------------------------------------------- misc
def fabric_texture(ph):
    """Rock-print fabric over the tunnel (photo 2)."""
    c = ph[2][200:320, 560:900]
    c = denet(c, k=13, thr=26, radius=5)
    c = flatten_light(c, sigma=40, strength=0.6)
    return mirror_tile(cv2.resize(c, (680, 240), interpolation=cv2.INTER_CUBIC), 480, 1024)


def rock_texture(ph):
    """3-D rock cones (photo 4, left cone body)."""
    c = ph[4][448:535, 174:242]
    c = denet(c, k=11, thr=26, radius=4)
    t = mirror_tile(cv2.resize(c, (204, 261), interpolation=cv2.INTER_CUBIC), 512, 512)
    return t


def stone_texture(ph):
    """White/grey landing stone (photo 7 pavers, outside the net)."""
    c = ph[7][1045:1130, 1232:1340]
    t = cv2.resize(c, (512, 413), interpolation=cv2.INTER_CUBIC)
    t = flatten_light(t, sigma=60, strength=0.8)
    t = cv2.convertScaleAbs(t, alpha=1.12, beta=18)      # the real block reads whiter
    return mirror_tile(t, 1024, 1024)


def board_texture(rng, w_px, h_px, strip_frac, hole=None):
    """Tan plywood/card board (photo 3/7) with the black strip at the bottom."""
    base = np.array((128, 168, 205), np.float32)          # BGR tan under warm light
    n1 = value_noise(h_px, w_px, 64, rng)
    grain = value_noise(h_px, w_px * 8, 24, rng)[:, ::8]
    t = base[None, None] * (0.90 + 0.12 * n1[..., None] + 0.05 * grain[..., None])
    t = np.clip(t, 0, 255).astype(np.uint8)
    s = int(h_px * (1 - strip_frac))
    t[s:] = (22, 22, 24)
    return t


def hall_texture(rng):
    """Procedural hall wall: dark furniture zone, wood, warm cove band, slats."""
    W, H = 2048, 512                                      # 26 m x 6.5 m strip
    t = np.zeros((H, W, 3), np.float32)
    yy = np.arange(H)[:, None]
    z = 6.5 * (1 - yy / H)                                # metres above floor
    wood = np.array((70, 120, 170), np.float32)
    t[:] = wood * 0.8
    t[(z < 1.0)[:, 0]] = (35, 38, 45)
    for x0 in range(0, W, 23):                            # vertical slats upper wall
        t[(z > 2.9)[:, 0], x0:x0 + 3] *= 0.7
    band = (z > 2.4) & (z < 2.85)
    t[band[:, 0]] = (150, 210, 255)
    glow = np.exp(-((z - 2.62) / 0.6) ** 2)
    t += (glow * 60)[..., None] * np.array((0.5, 0.8, 1.0))
    n = value_noise(H, W, 32, rng)
    t *= (0.9 + 0.2 * n)[..., None]
    for x0 in range(40, W, 160):                          # dark table/chair blobs
        t[(z < 0.9)[:, 0], x0:x0 + 70] *= 0.6
    return np.clip(t, 0, 255).astype(np.uint8)


def hall_emissive(tex):
    g = cv2.cvtColor(tex, cv2.COLOR_BGR2GRAY)
    e = np.where((g > 180)[..., None], tex, 0).astype(np.uint8)
    return cv2.GaussianBlur(e, (0, 0), 4)


def carpet_texture(rng):
    n = value_noise(512, 512, 4, rng) * 0.6 + value_noise(512, 512, 64, rng) * 0.4
    t = np.stack([60 + 30 * n, 52 + 26 * n, 45 + 22 * n], -1)
    for k in range(0, 512, 128):                          # carpet-tile grid
        t[k:k + 2] *= 0.8; t[:, k:k + 2] *= 0.8
    return np.clip(t, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('photo_dir')
    ap.add_argument('--out', default=os.path.join(HERE, 'assets', 'textures'))
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    import sys
    sys.path.insert(0, HERE)
    import layout
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    ph = {i: load(a.photo_dir, i) for i in range(1, 8)}
    floor, placed = floor_texture(ph, rng, layout.FLOOR_SIZE)
    # start pad (helipad) printed into the floor
    # (2026-11-07) no red H pad any more: the drone starts in a black-tape square (photos 10, 11);
    # the tape is a separate floor mesh (build_world.build_tape). helipad() is kept for reference.
    W = lambda name, im, q=92: cv2.imwrite(os.path.join(a.out, name), im, [cv2.IMWRITE_JPEG_QUALITY, q])
    W('floor.jpg', floor, 90)
    W('panel.jpg', panel_texture(ph))
    W('fabric.jpg', fabric_texture(ph))
    W('rock.jpg', rock_texture(ph))
    W('stone.jpg', stone_texture(ph))
    W('board_window.jpg', board_texture(rng, 1024, 922, layout.WINDOW['strip_h'] / (layout.WINDOW['top'] - layout.WINDOW['bottom'])))
    W('board_wall.jpg', board_texture(rng, 1024, 896, layout.WALL['strip_h'] / (layout.WALL['top'] - layout.WALL['bottom'])))
    hall = hall_texture(rng)
    W('hall.jpg', hall)
    W('hall_emissive.jpg', hall_emissive(hall))
    W('carpet.jpg', carpet_texture(rng))
    W('landing_pad.jpg', landing_pad_texture(), 95)
    with open(os.path.join(a.out, 'textures.json'), 'w') as f:
        json.dump({'floor_px': FLOOR_PX, 'floor_size_m': layout.FLOOR_SIZE, 'painted_rocks': placed,
                   'source': 'venue photos 2026-11-06 (2,4,5,6,7) + procedural'}, f, indent=1)
    print('textures ->', a.out)


if __name__ == '__main__':
    main()
