#!/usr/bin/env python3
"""Offline evaluation of perception/{floor,tunnel,window}.py on the venue-replica set.

Usage: python eval_perception.py [--root outputs/venue-replica] [--out ROOT/eval]
Runs the modules UNCHANGED on every non-compare frame under 4 conditions:
  depth : 'ideal'  = `depth_ideal` (noise-free truth, no net / tunnel mesh)
          'stereo' = `depth` (OAK-D-like SGBM stereo from the noisy left/right pair, sees net + mesh)
  Both conditions get the same noisy left `image`.
  prior : 'prior'  = coarse site-measured priors = truth + N(0, 0.2 m) clipped to 0.5 m
                     (tunnel entry, window centre, stone centre), ToF floor d = 0 + N(0, 2 cm) (floor, tunnel,
                     window), measured sizes (hoop radius / gap / spacing / count, window size)
          'none'   = no priors, module defaults
Ground truth (labels) is used only for scoring. Writes OUT/records.json, OUT/summary.json
and prints the summary tables.
"""
import argparse, json, math, os, sys, time, zlib
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, OFFICIAL)
from dataset import load_frame                                   # noqa: E402
import layout as L                                               # noqa: E402
from common.geometry import SKY_BETA_HALF_SPAN_M                  # noqa: E402
from perception import floor as FL, tunnel as TU, window as WI    # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
HALF_H = 0.10
# measured cage box (site geofence, NED n_min, n_max, e_min, e_max): known in both prior conditions
KEEP_XY = (-L.START[0], L.CAGE - L.START[0], -L.START[1], L.CAGE - L.START[1])


def noisy(p, rng, sigma=0.2, clip=0.5, dims=3):
    p = np.asarray(p, float).copy()
    e = np.zeros(3); e[:dims] = rng.normal(0, sigma, dims)
    n = np.linalg.norm(e)
    if n > clip:
        e *= clip / n
    return p + e


def ang_deg(a, b):
    a = np.asarray(a, float)[:2]; b = np.asarray(b, float)[:2]
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(math.degrees(math.acos(np.clip(c, -1, 1))))


def truth_heights(fr_ideal, stride=2):
    """True height above the (flat, d=0) floor for every stride-2 pixel, from ideal depth;
    NaN outside the geofenced cage box (KEEP_XY shrunk by 5 cm), which obstacles() drops by design."""
    P, vu = fr_ideal.points_ned(stride=stride, max_range=4.0)
    n0, n1, e0, e1 = KEEP_XY
    inside = (P[:, 0] > n0 + 0.05) & (P[:, 0] < n1 - 0.05) & (P[:, 1] > e0 + 0.05) & (P[:, 1] < e1 - 0.05)
    H = np.full(fr_ideal.depth.shape, np.nan, np.float32)
    H[vu[inside, 0], vu[inside, 1]] = -P[inside, 2]
    return H


def run_frame(path, meta, cond_depth, cond_prior, rng):
    ideal, lab = load_frame(path, depth_key='depth_ideal')
    fr = load_frame(path, depth_key='depth')[0] if cond_depth == 'stereo' else ideal
    W, F = lab['world'], lab['frame']
    pri = cond_prior == 'prior'
    rec = {'id': meta['id'], 'group': meta['group'], 'depth': cond_depth, 'prior': cond_prior,
           'height': meta['height_m'], 'vis': meta['visible']}
    # ---------------- floor + obstacles + landing
    t0 = time.perf_counter()
    ff = FL.fit_floor(fr, prior_floor_d=float(rng.normal(0, 0.02)) if pri else None, keep_xy=KEEP_XY)
    ob = FL.obstacles(fr, ff, keep_xy=KEEP_XY)
    lp = noisy(W['landing_target']['top_center_ned'], rng) if pri else None
    lt = FL.landing_target(fr, ff, prior_center_ned=lp, keep_xy=KEEP_XY)
    rec['t_floor_ms'] = 1000 * (time.perf_counter() - t0)
    rec['floor'] = {'found': bool(ff.found), 'd_err': abs(ff.d) if np.isfinite(ff.d) else None,
                    'tilt_err_deg': float(math.degrees(math.acos(np.clip(abs(ff.normal_ned[2]), -1, 1))))
                    if np.all(np.isfinite(ff.normal_ned)) else None, 'source': ff.source, 'reason': ff.reason}
    Ht = truth_heights(ideal)
    # how much IN-CAGE floor the camera really sees (stride 4, within 5 m): frames without it can only
    # be solved with a floor prior (keep_xy removes the hall floor seen through the net by design)
    Pf, vuf = ideal.points_ned(stride=4, max_range=5.0)
    n0, n1, e0, e1 = KEEP_XY
    surf_f = np.load(path)['surface'][vuf[:, 0], vuf[:, 1]]
    rec['floor_inview_pts'] = int(np.sum((surf_f == 1) & (Pf[:, 0] > n0) & (Pf[:, 0] < n1) & (Pf[:, 1] > e0) & (Pf[:, 1] < e1)))
    valid_c = np.isfinite(fr.depth)
    true_raised = np.isfinite(Ht) & (Ht >= 0.10) & valid_c
    det = np.zeros_like(true_raised)
    fp_pts = painted_fp = n_real = n_net = n_phantom = n_phantom_corr = 0
    if len(ob.pixels_vu):
        v, u = ob.pixels_vu[:, 0], ob.pixels_vu[:, 1]
        det[v, u] = True
        th = Ht[v, u]
        false = np.isfinite(th) & (th < 0.03)
        fp_pts = int(false.sum())
        # where does each obstacle point really sit?  on the ideal surface / on net-mesh / nowhere (phantom)
        z = fr.depth[v, u]
        di = np.load(path)['depth_ideal'][v, u]; dfl = np.load(path)['depth_full'][v, u]
        tol = np.maximum(0.05, 0.05 * z)
        on_ideal = np.isfinite(di) & (np.abs(z - di) < tol)
        on_net = ~on_ideal & np.isfinite(dfl) & (np.abs(z - dfl) < tol)
        # stereo windows straddling the sub-pixel net cords return the NET PLANE even where this
        # pixel's own ray passes between cords -> count points near a net plane / the tunnel mesh as net
        P0 = ob.points_ned
        a_, b_, h_ = P0[:, 0] + L.START[0], P0[:, 1] + L.START[1], -P0[:, 2]
        C, off = L.CAGE, L.NET_OFFSET
        near_wall = np.minimum.reduce([np.abs(a_ + off), np.abs(a_ - C - off), np.abs(b_ + off), np.abs(b_ - C - off),
                                       np.abs(h_ - L.CAGE_H - off)]) < 0.15
        T = L.TUNNEL
        (ea, eb), (ua, ub), (ra, rb) = T['entry_ab'], T['axis_ab'], L.tunnel_right()
        s_ = (a_ - ea) * ua + (b_ - eb) * ub; l_ = (a_ - ea) * ra + (b_ - eb) * rb    # along / right of the axis
        near_mesh = (np.abs(np.hypot(l_, h_ - T['centre_h']) - T['mesh_r']) < 0.10) & (s_ > -0.15) & (s_ < T['length'] + 0.15)
        on_net |= ~on_ideal & (near_wall | near_mesh)
        phantom = ~on_ideal & ~on_net
        n_real, n_net, n_phantom = int(on_ideal.sum()), int(on_net.sum()), int(phantom.sum())
        P = ob.points_ned
        rng_h = np.linalg.norm(P[:, :2] - fr.cam_ned[:2], axis=1)
        # flyable corridor: >= 0.4 m inside the net, 0.1-1.6 m high, within 2.5 m of the camera
        inside = (a_ > 0.4) & (a_ < C - 0.4) & (b_ > 0.4) & (b_ < C - 0.4)
        corr = phantom & inside & (rng_h < 2.5) & (h_ > 0.1) & (h_ < 1.6)
        n_phantom_corr = int(corr.sum())
        surf = np.load(path)['surface'][v, u]
        flo = phantom & (surf == 1)                          # phantom whose pixel really sees the floor print
        for r in W.get('painted_rocks_flat', []):
            c = np.array(r['center_ned'])
            painted_fp += int(np.sum(np.linalg.norm(P[flo][:, :2] - c[:2], axis=1) < r['size_m'] / 2))
    rec['obstacles'] = {'n': int(len(ob.pixels_vu)), 'floor_used': bool(ob.floor_used),
                        'true_raised_px': int(true_raised.sum()),
                        'recall': float(det[true_raised].mean()) if true_raised.sum() >= 20 else None,
                        'false_raised_pts': fp_pts, 'painted_fp_pts': painted_fp,
                        'pts_on_surface': n_real, 'pts_on_net_or_mesh': n_net, 'pts_phantom': n_phantom,
                        'pts_phantom_in_corridor': n_phantom_corr}
    tc = np.array(W['landing_target']['top_center_ned'])
    rec['landing'] = {'found': bool(lt.found), 'conf': float(lt.confidence),
                      'xy_err': float(np.linalg.norm(lt.center_ned[:2] - tc[:2])) if lt.found else None,
                      'top_err': float(abs(lt.top_d - W['landing_target']['top_d'])) if lt.found else None,
                      'range': float(np.linalg.norm(tc[:2] - fr.cam_ned[:2])), 'reason': lt.reason,
                      'est': lt.center_ned.round(3).tolist() if lt.found else None}
    # ---------------- tunnel
    T = W['tunnel']
    hs = T['length_m'] / (len(T['hoop_centers_ned']) - 1)                  # measured hoop spacing / count
    params = TU.TunnelParams(radius_m=T['radius_m'], bottom_gap_m=-T['entry_ned'][2] - T['radius_m'],
                             hoop_spacing_m=hs, n_hoops=len(T['hoop_centers_ned'])) if pri else None
    t0 = time.perf_counter()
    te = TU.detect(fr, noisy(T['entry_ned'], rng) if pri else None, floor_z_ned=float(rng.normal(0, 0.02)) if pri else None,
                   params=params)
    rec['t_tunnel_ms'] = 1000 * (time.perf_counter() - t0)
    # the module reports the hoop NEAREST the camera as "entry" and the axis pointing away
    # from the camera, so score against whichever tunnel end is nearer
    e0, e1 = np.array(T['entry_ned']), np.array(T['exit_ned'])
    e, other = (e0, e1) if np.linalg.norm(e0 - fr.cam_ned) <= np.linalg.norm(e1 - fr.cam_ned) else (e1, e0)
    ax = (other - e)[:2] / np.linalg.norm((other - e)[:2])
    rec['tunnel'] = {'found': bool(te.found), 'conf': float(te.confidence),
                     'dist': float(np.linalg.norm(e - fr.cam_ned)), 'end': 'entry' if e is e0 else 'exit',
                     'entry_err': float(np.linalg.norm(te.entry_ned - e)) if te.found else None,
                     'axis_err_deg': ang_deg(te.axis_ned, ax) if te.found else None,
                     'radius_err': float(abs(te.radius_m - T['radius_m'])) if te.found else None,
                     'est': te.entry_ned.round(3).tolist() if te.found else None,
                     'est_axis': np.asarray(te.axis_ned).round(3).tolist() if te.found else None,
                     'n_rings': len(te.rings), 'align_ok': bool(getattr(te, 'align_ok', False)),
                     # what matters for flying through: offset from the tunnel axis line, and depth along it
                     'axis_perp_err': float(np.linalg.norm((te.entry_ned - e0) - np.dot((te.entry_ned - e0)[:2], T['axis_ned'])
                                                           * np.array([T['axis_ned'][0], T['axis_ned'][1], 0.0]))) if te.found else None,
                     'along_err': float(np.dot((te.entry_ned - e)[:2], ax)) if te.found else None,
                     'nearest_hoop_err': float(min(np.linalg.norm(te.entry_ned - np.array(h)) for h in T['hoop_centers_ned'])) if te.found else None}
    # ---------------- window
    Wn = W['window']
    t0 = time.perf_counter()
    we = WI.detect(fr, SKY_BETA_HALF_SPAN_M, HALF_H, prior_center_ned=noisy(Wn['center_ned'], rng) if pri else None,
                   known_size_m=(Wn['width_m'], Wn['height_m']) if pri else None,
                   floor_z_ned=float(rng.normal(0, 0.02)) if pri else None)
    rec['t_window_ms'] = 1000 * (time.perf_counter() - t0)
    c = np.array(Wn['center_ned'])
    n_exp = np.array(Wn['normal_ned']) * (1.0 if np.dot(fr.cam_ned - c, Wn['normal_ned']) >= 0 else -1.0)  # towards camera
    rec['window'] = {'found': bool(we.found), 'conf': float(we.confidence), 'clear': bool(we.clearance_ok),
                     'dist': float(np.linalg.norm(c - fr.cam_ned)), 'reason': we.reason, 'source': we.source,
                     'center_err': float(np.linalg.norm(we.center_ned - c)) if we.found else None,
                     'normal_err_deg': ang_deg(we.normal_ned, n_exp) if we.found else None,
                     'w_err': float(we.width_m - Wn['width_m']) if we.found else None,
                     'h_err': float(we.height_m - Wn['height_m']) if we.found else None,
                     'est': we.center_ned.round(3).tolist() if we.found else None,
                     'est_wh': [round(float(we.width_m), 3), round(float(we.height_m), 3)] if we.found else None}
    return rec


def pct(x, q):
    x = [v for v in x if v is not None]
    return round(float(np.percentile(x, q)), 3) if x else None


def summarize(R):
    out = {}
    by = defaultdict(list)
    for r in R:
        by[(r['depth'], r['prior'])].append(r)
    for key, rs in sorted(by.items()):
        k = f'{key[0]}/{key[1]}'
        s = {}
        s['floor'] = {'found_rate': round(np.mean([r['floor']['found'] for r in rs]), 3),
                      'd_err_p50_cm': pct([100 * r['floor']['d_err'] if r['floor']['d_err'] is not None else None for r in rs], 50),
                      'd_err_p95_cm': pct([100 * r['floor']['d_err'] if r['floor']['d_err'] is not None else None for r in rs], 95),
                      'tilt_err_p95_deg': pct([r['floor']['tilt_err_deg'] for r in rs], 95),
                      'obst_recall_p50': pct([r['obstacles']['recall'] for r in rs], 50),
                      'obst_recall_p10': pct([r['obstacles']['recall'] for r in rs], 10),
                      'frames_with_false_raised': int(sum(r['obstacles']['false_raised_pts'] > 0 for r in rs)),
                      'false_raised_pts_total(raised over true floor, incl. net hits)': int(sum(r['obstacles']['false_raised_pts'] for r in rs)),
                      'painted_rock_fp_pts_total': int(sum(r['obstacles']['painted_fp_pts'] for r in rs)),
                      'obst_pts_on_surface_frac': round(sum(r['obstacles']['pts_on_surface'] for r in rs) / max(1, sum(r['obstacles']['n'] for r in rs)), 3),
                      'obst_pts_on_net_mesh_frac': round(sum(r['obstacles']['pts_on_net_or_mesh'] for r in rs) / max(1, sum(r['obstacles']['n'] for r in rs)), 3),
                      'obst_pts_phantom_frac': round(sum(r['obstacles']['pts_phantom'] for r in rs) / max(1, sum(r['obstacles']['n'] for r in rs)), 3),
                      'frames_with_corridor_phantoms(>=20pts)': int(sum(r['obstacles']['pts_phantom_in_corridor'] >= 20 for r in rs)),
                      'corridor_phantom_pts_p90': pct([r['obstacles']['pts_phantom_in_corridor'] for r in rs], 90),
                      'ms_p50': pct([r['t_floor_ms'] for r in rs], 50)}
        vis_f = [r for r in rs if r['floor_inview_pts'] >= 100]
        nov_f = [r for r in rs if r['floor_inview_pts'] < 100]
        s['floor']['with_incage_floor(>=100pts)'] = {
            'n': len(vis_f), 'found_rate': round(np.mean([r['floor']['found'] for r in vis_f]), 3),
            'not_found': [r['id'] for r in vis_f if not r['floor']['found']],
            'd_err_gt5cm': [(r['id'], round(r['floor']['d_err'], 3)) for r in vis_f if r['floor']['d_err'] is not None and r['floor']['d_err'] > 0.05],
            'd_err_p95_cm': pct([100 * r['floor']['d_err'] if r['floor']['d_err'] is not None else None for r in vis_f], 95),
            'obst_recall_p10': pct([r['obstacles']['recall'] for r in vis_f], 10)}
        s['floor']['without_incage_floor'] = {
            'n': len(nov_f), 'found_wrong_gt10cm': [(r['id'], round(r['floor']['d_err'], 2)) for r in nov_f
                                                    if r['floor']['found'] and r['floor']['d_err'] is not None and r['floor']['d_err'] > 0.10],
            'd_err_p95_cm': pct([100 * r['floor']['d_err'] if r['floor']['d_err'] is not None else None for r in nov_f], 95)}
        pos = [r for r in rs if r['vis']['landing_target']]
        neg = [r for r in rs if not r['vis']['landing_target']]
        s['landing_by_view'] = {f'{name}': [sum(r['landing']['found'] for r in b), len(b)] for name, b in (
            ('range<=3.5m & cam>=0.35m above top', [r for r in pos if r['landing']['range'] <= 3.5 and r['height'] - L.PAD['thick'] >= 0.35]),
            ('range<=3.5m & cam<0.35m above top', [r for r in pos if r['landing']['range'] <= 3.5 and r['height'] - L.PAD['thick'] < 0.35]),
            ('range>3.5m', [r for r in pos if r['landing']['range'] > 3.5]))}
        s['landing'] = {'n_pos': len(pos), 'found_rate_pos': round(np.mean([r['landing']['found'] for r in pos]), 3) if pos else None,
                        'xy_err_p50': pct([r['landing']['xy_err'] for r in pos], 50), 'xy_err_max': pct([r['landing']['xy_err'] for r in pos], 100),
                        'top_err_max': pct([r['landing']['top_err'] for r in pos], 100),
                        'false_found_neg': int(sum(r['landing']['found'] and (r['landing']['xy_err'] or 0) > 0.3 for r in neg)),
                        'missed_ids': [r['id'] for r in pos if not r['landing']['found']]}
        pos = [r for r in rs if r['vis']['tunnel_entry']]
        neg = [r for r in rs if not r['vis']['tunnel_entry']]
        s['tunnel'] = {'n_pos': len(pos)}
        for lo, hi in ((0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 9)):
            b = [r for r in pos if lo <= r['tunnel']['dist'] < hi]
            if b:
                s['tunnel'][f'{lo:.0f}-{hi:.0f}m'] = {
                    'n': len(b), 'found': round(np.mean([r['tunnel']['found'] for r in b]), 3),
                    'good(<0.15m)': round(np.mean([r['tunnel']['found'] and r['tunnel']['entry_err'] < 0.15 for r in b]), 3),
                    'entry_err_p50': pct([r['tunnel']['entry_err'] for r in b], 50),
                    'axis_err_p50_deg': pct([r['tunnel']['axis_err_deg'] for r in b], 50)}
        g = [r for r in rs if r['tunnel']['found'] and r['tunnel']['conf'] >= 0.6]
        s['tunnel']['gated(conf>=0.6)'] = {
            'n_accepted': len(g), 'n_accepted_on_visible': sum(r['vis']['tunnel_entry'] for r in g),
            'off_axis_max_m': pct([r['tunnel']['axis_perp_err'] for r in g], 100),
            'off_axis_p50_m': pct([r['tunnel']['axis_perp_err'] for r in g], 50),
            'along_axis_jump_frames(>0.3m)': int(sum(abs(r['tunnel']['along_err']) > 0.3 for r in g)),
            'recall_visible_<=2m': round(np.mean([r['tunnel']['found'] and r['tunnel']['conf'] >= 0.6 for r in pos if r['tunnel']['dist'] <= 2.0]), 3),
            'recall_visible_2-3m': round(np.mean([r['tunnel']['found'] and r['tunnel']['conf'] >= 0.6 for r in pos if 2.0 < r['tunnel']['dist'] <= 3.0]), 3)}
        # tunnel session's recommendation: with prior conf >= 0.4 AND align_ok; without prior conf >= 0.6 AND align_ok
        thr = 0.4 if key[1] == 'prior' else 0.6
        ga = [r for r in rs if r['tunnel']['found'] and r['tunnel']['conf'] >= thr and r['tunnel'].get('align_ok')]
        s['tunnel']['gated_align_ok'] = {
            'conf_thr': thr, 'n_accepted': len(ga),
            'off_axis_ge_0.10m_accepted': int(sum(r['tunnel']['axis_perp_err'] >= 0.10 for r in ga)),
            'recall_visible_<=2m': round(np.mean([any(g['id'] == p_['id'] for g in ga) for p_ in pos if p_['tunnel']['dist'] <= 2.0]), 3),
            'recall_visible_2-3m': round(np.mean([any(g['id'] == p_['id'] for g in ga) for p_ in pos if 2.0 < p_['tunnel']['dist'] <= 3.0]), 3)}
        s['tunnel']['off_axis_gt_0.15m_ungated'] = int(sum(r['tunnel']['found'] and r['tunnel']['axis_perp_err'] > 0.15 for r in rs))
        s['tunnel']['wrong_found_pos(>0.3m)'] = int(sum(r['tunnel']['found'] and r['tunnel']['entry_err'] > 0.3 for r in pos))
        s['tunnel']['false_found_neg(>0.5m)'] = int(sum(r['tunnel']['found'] and r['tunnel']['entry_err'] > 0.5 for r in neg))
        s['tunnel']['ms_p50'] = pct([r['t_tunnel_ms'] for r in rs], 50)
        pos = [r for r in rs if r['vis']['window']]
        neg = [r for r in rs if not r['vis']['window']]
        s['window'] = {'n_pos': len(pos), 'found_rate_pos': round(np.mean([r['window']['found'] for r in pos]), 3) if pos else None,
                       'center_err_p50': pct([r['window']['center_err'] for r in pos], 50),
                       'center_err_max': pct([r['window']['center_err'] for r in pos], 100),
                       'normal_err_max_deg': pct([r['window']['normal_err_deg'] for r in pos], 100),
                       'w_err_p50': pct([r['window']['w_err'] for r in pos], 50), 'h_err_p50': pct([r['window']['h_err'] for r in pos], 50),
                       'clear_rate_found_le3.2m': round(np.mean([r['window']['clear'] for r in pos if r['window']['found'] and r['window']['dist'] <= 3.2]), 3)
                       if any(r['window']['found'] and r['window']['dist'] <= 3.2 for r in pos) else None,
                       'false_found_neg(>0.3m)': int(sum(r['window']['found'] and r['window']['center_err'] > 0.3 for r in neg)),
                       'found_neg_ok(<0.1m)': int(sum(r['window']['found'] and r['window']['center_err'] < 0.1 for r in neg)),
                       'found_by_dist': {f'{lo}-{hi}m': [sum(r['window']['found'] for r in pos if lo <= r['window']['dist'] < hi),
                                                          sum(lo <= r['window']['dist'] < hi for r in pos)]
                                         for lo, hi in ((0, 2), (2, 3.2), (3.2, 4.5), (4.5, 9))},
                       'missed': [(r['id'], r['window']['reason'], round(r['window']['dist'], 2)) for r in pos if not r['window']['found']],
                       'ms_p50': pct([r['t_window_ms'] for r in rs], 50)}
        out[k] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join(REPO, 'outputs', 'venue-replica'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    out = a.out or os.path.join(a.root, 'eval')
    os.makedirs(out, exist_ok=True)
    idx = json.load(open(os.path.join(a.root, 'index.json')))
    frames = [f for f in idx['frames'] if f['group'] != 'compare']
    if a.limit:
        frames = frames[:a.limit]
    R = []
    t0 = time.time()
    for cd in ('ideal', 'stereo'):
        for cp in ('prior', 'none'):
            rng = np.random.default_rng(2026)
            for f in frames:
                R.append(run_frame(os.path.join(a.root, f['file']), f, cd, cp, rng))
            print(f'{cd}/{cp} done ({time.time() - t0:.0f}s)', flush=True)
    json.dump(R, open(os.path.join(out, 'records.json'), 'w'), indent=0, default=float)
    S = summarize(R)
    json.dump(S, open(os.path.join(out, 'summary.json'), 'w'), indent=1, default=float)
    print(json.dumps(S, indent=1, default=float))


if __name__ == '__main__':
    main()
