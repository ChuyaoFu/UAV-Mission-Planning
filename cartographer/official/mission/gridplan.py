"""Grid planning on an OccGrid (maze-planner-design.md, design v1): cost map, A*, string pulling, frontiers.

Pure Python + numpy, no clock reads.  Cells are (i, j) indices of the OccGrid (i along u, j along v).

* C-space (strict, the default): a cell is blocked when it is closer than ``plan_clear_m`` (0.38 = official reach
  0.28 + 0.1) to CONFIRMED occupied cells, or closer than ``suspect_clear_m`` (0.28) to suspect cells.  Relaxed
  (``hard_clear_m`` 0.30, the navigator's first no-path step): cells 0.30-0.38 m from confirmed structure are
  *narrow* (``narrow_cost`` per cell, own class), so a gap narrower than 2 x 0.38 m (official first corner 0.73 m,
  maze gaps 0.8 m) is flown along its centre when it is the only way.  Within
  ``plan_clear_m`` of suspect cells it costs ``suspect_cost`` (15) extra per cell: squeezing past an unconfirmed
  return is used only when every other way is several metres longer.
* Cell cost: 1 for known free, ``unknown_cost`` for unknown (optimistic planning: unknown space is passable but
  dearer, the navigator looks before it flies there), + ``w_clear * max(0, pref_clear_m - clearance) /
  pref_clear_m`` (prefers lane centres: replaces vertex centring in map mode).
* ``prune`` (string pulling) never shortcuts closer to structure than the A* path itself was, and never through
  a worse cell class (free < unknown < suspect-near): every waypoint segment keeps >= ``plan_clear_m`` from
  confirmed structure by construction.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .occgrid import edt

SQ2 = math.sqrt(2.0)


@dataclass
class CostMap:
    res: float
    blocked: np.ndarray        # bool (C-space)
    cost: np.ndarray           # per-cell traversal cost (inf where blocked)
    clear: np.ndarray          # m to confirmed occupied (+ fence)
    clear_s: np.ndarray        # m to suspect
    unknown: np.ndarray        # bool
    free: np.ndarray           # bool: known free
    cls: np.ndarray            # int8: 0 free, 1 unknown, 2 narrow, 3 narrow + unknown, 4 near suspect, 5 blocked
    plan_clear_m: float
    hard_clear_m: float = 0.30


def cost_map(grid, plan_clear_m=0.38, pref_clear_m=0.55, unknown_cost=2.0, suspect_cost=15.0, suspect_clear_m=0.28,
             w_clear=3.0, hard_clear_m=None, narrow_cost=6.0, fence_clear_m=None, suspect_block=True,
             suspect_cell_cost=50.0, suspect_small_cells=8, prior_cost=15.0, prior_cell_cost=50.0,
             extra_block=None) -> CostMap:
    """hard_clear_m None = plan_clear_m (strict: blocked < 0.38 m).  fence_clear_m: the fence (grid border) gets its
    own hard limit (e.g. the mission's fence_margin_m 0.40) and counts like a wall (fence_clear_m - plan_clear_m)
    nearer in the narrow / centring terms; None = the fence is ordinary occupied structure.  suspect_block False
    (relaxed): SMALL suspect components (<= suspect_small_cells, a stereo-phantom-sized blob) only cost
    (suspect_cost within plan_clear_m, suspect_cell_cost on one); larger ones (a wall seen from one place) still block
    within suspect_clear_m.  The follower's observe / nudge / re-confirm handles the blob on the way, and a moving
    camera carves a stereo phantom.  The grid's soft prior layer (prior wall lines not yet seen through) never
    blocks: prior_cost per cell within plan_clear_m of it, prior_cell_cost on it (and its own class, like suspect)."""
    occ = grid.occupied()
    hard = plan_clear_m if hard_clear_m is None else min(hard_clear_m, plan_clear_m)
    if fence_clear_m is None:
        clear = grid.clearance('occupied')
        fence_block = np.zeros_like(occ)
    else:
        d_o = edt(occ & ~grid.fence) * grid.res
        d_f = edt(grid.fence) * grid.res
        b = 0.5 * grid.res
        d_o, d_f = np.maximum(d_o - b, 0.0), np.maximum(d_f - b, 0.0)
        fence_block = d_f < fence_clear_m
        clear = np.minimum(d_o, d_f - (fence_clear_m - plan_clear_m))
    clear_s = grid.clearance('suspect')
    unknown = grid.unknown()
    free = grid.known_free()
    sus = grid.suspect()
    if suspect_block:
        sus_block = (clear_s < suspect_clear_m) | sus
    else:
        big = sus & ~_small_components(sus, suspect_small_cells)
        sus_block = big if not big.any() else (big | (edt(big) * grid.res - 0.5 * grid.res < suspect_clear_m))
    blocked = (clear < hard) | fence_block | occ | sus_block
    if extra_block is not None:
        blocked = blocked | extra_block
    narrow = clear < plan_clear_m
    near_s = clear_s < plan_clear_m
    cost = np.where(unknown, float(unknown_cost), 1.0)
    cost = cost + np.where(narrow, float(narrow_cost), 0.0)
    cost = cost + np.where(near_s, float(suspect_cost), 0.0)
    pl = grid.prior_live() if hasattr(grid, 'prior_live') else np.zeros_like(occ)
    near_p = np.zeros_like(occ)
    if pl.any():
        near_p = edt(pl) * grid.res - 0.5 * grid.res < plan_clear_m
        cost = cost + np.where(near_p, float(prior_cost), 0.0) + np.where(pl, float(prior_cell_cost), 0.0)
    if not suspect_block:
        cost = cost + np.where(sus, float(suspect_cell_cost), 0.0)
    if pref_clear_m > 0 and w_clear > 0:
        cost = cost + w_clear * np.maximum(0.0, pref_clear_m - np.minimum(clear, clear_s)) / pref_clear_m
    cost = np.where(blocked, np.inf, cost)
    cls = (np.where(unknown, 1, 0) + np.where(narrow, 2, 0)).astype(np.int8)
    cls[near_s | near_p] = 4
    cls[blocked] = 5
    blocked[0, :] = blocked[-1, :] = blocked[:, 0] = blocked[:, -1] = True   # (A* never steps off the grid)
    cost[0, :] = cost[-1, :] = cost[:, 0] = cost[:, -1] = np.inf
    return CostMap(res=grid.res, blocked=blocked, cost=cost, clear=clear, clear_s=clear_s, unknown=unknown,
                   free=free, cls=cls, plan_clear_m=float(plan_clear_m), hard_clear_m=float(hard))


def _small_components(mask, max_cells):
    """Cells of 8-connected components of mask with <= max_cells cells."""
    out = np.zeros_like(mask)
    cells = set(zip(*np.nonzero(mask)))
    while cells:
        seed = cells.pop()
        comp, stack = [seed], [seed]
        while stack:
            i, j = stack.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    q = (i + di, j + dj)
                    if q in cells:
                        cells.remove(q)
                        comp.append(q)
                        stack.append(q)
        if len(comp) <= max_cells:
            C = np.asarray(comp)
            out[C[:, 0], C[:, 1]] = True
    return out


def nearest_unblocked(cm: CostMap, ij, max_m=0.4) -> Optional[Tuple[int, int]]:
    """The start may sit inside C-space (VIO, a wall just measured): the nearest unblocked cell within max_m, ties
    -> larger clearance.  None if there is none."""
    i, j = int(ij[0]), int(ij[1])
    H, W = cm.blocked.shape
    if 0 <= i < H and 0 <= j < W and not cm.blocked[i, j]:
        return (i, j)
    R = int(math.ceil(max_m / cm.res))
    a0, a1, b0, b1 = max(0, i - R), min(H, i + R + 1), max(0, j - R), min(W, j + R + 1)
    if a0 >= a1 or b0 >= b1:
        return None
    ii, jj = np.mgrid[a0:a1, b0:b1]
    d = np.hypot(ii - i, jj - j) * cm.res
    ok = ~cm.blocked[a0:a1, b0:b1] & (d <= max_m)
    if not ok.any():
        return None
    key = np.where(ok, d - 0.01 * np.minimum(cm.clear[a0:a1, b0:b1], 1.0), np.inf)
    k = int(np.argmin(key))
    return (int(ii.ravel()[k]), int(jj.ravel()[k]))


def astar(cm: CostMap, start, goal_mask: np.ndarray, goal_center=None, goal_radius_cells=0.0,
          max_expand: Optional[int] = None) -> Optional[Tuple[List[Tuple[int, int]], float]]:
    """8-connected A* from ``start`` (i, j) to any cell of ``goal_mask``.  Edge cost = step length (cells) x the
    destination cell's cost.  Heuristic: octile-free Euclidean distance to the goal disc (admissible: min cost 1).
    Returns (path cells incl. start, total cost) or None."""
    H, W = cm.cost.shape
    s = int(start[0]) * W + int(start[1])
    goal = goal_mask.ravel()
    if not goal.any():
        return None
    cost = cm.cost.ravel().tolist()
    if goal_center is None:
        gi, gj = np.nonzero(goal_mask)
        gc = (float(gi.mean()), float(gj.mean()))
        rad = float(np.max(np.hypot(gi - gc[0], gj - gc[1])))
    else:
        gc, rad = (float(goal_center[0]), float(goal_center[1])), float(goal_radius_cells)
    ii, jj = np.mgrid[0:H, 0:W]
    h = np.maximum(np.hypot(ii - gc[0], jj - gc[1]) - rad, 0.0).ravel().tolist()
    goal_l = goal.tolist()
    g = {s: 0.0}
    parent = {s: -1}
    closed = set()
    heap = [(h[s], 0.0, s)]
    nb = ((-W, 1.0), (W, 1.0), (-1, 1.0), (1, 1.0), (-W - 1, SQ2), (-W + 1, SQ2), (W - 1, SQ2), (W + 1, SQ2))
    n_exp = 0
    hpush, hpop = heapq.heappush, heapq.heappop
    while heap:
        f, gs, n = hpop(heap)
        if n in closed:
            continue
        if goal_l[n]:
            path = []
            while n != -1:
                path.append((n // W, n % W))
                n = parent[n]
            return path[::-1], gs
        closed.add(n)
        n_exp += 1
        if max_expand is not None and n_exp > max_expand:
            return None
        for off, step in nb:
            m = n + off
            c = cost[m]
            if c == math.inf or m in closed:
                continue
            ng = gs + step * c
            if ng < g.get(m, math.inf):
                g[m] = ng
                parent[m] = n
                hpush(heap, (ng + h[m], ng, m))
    return None


def dijkstra(cm: CostMap, start, max_cost=math.inf) -> np.ndarray:
    """Cost-to-reach from ``start`` for every cell (inf = unreachable); used to rank frontiers."""
    H, W = cm.cost.shape
    s = int(start[0]) * W + int(start[1])
    cost = cm.cost.ravel().tolist()
    dist = [math.inf] * (H * W)
    dist[s] = 0.0
    heap = [(0.0, s)]
    nb = ((-W, 1.0), (W, 1.0), (-1, 1.0), (1, 1.0), (-W - 1, SQ2), (-W + 1, SQ2), (W - 1, SQ2), (W + 1, SQ2))
    hpush, hpop = heapq.heappush, heapq.heappop
    while heap:
        d, n = hpop(heap)
        if d > dist[n] or d > max_cost:
            continue
        for off, step in nb:
            m = n + off
            c = cost[m]
            if c == math.inf:
                continue
            nd = d + step * c
            if nd < dist[m]:
                dist[m] = nd
                hpush(heap, (nd, m))
    return np.asarray(dist).reshape(H, W)


def _seg_cells(a, b, res_frac=0.5):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = max(2, int(math.ceil(float(np.hypot(*(b - a))) / res_frac)) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    q = np.rint(a[None, :] + t * (b - a)[None, :]).astype(int)
    return q


def los_ok(cm: CostMap, a, b, min_clear=None, max_cls=0, tol_m=0.04) -> bool:
    """Straight segment a -> b (cells) passes no blocked cell, no cell of a worse class than ``max_cls`` and keeps
    >= ``min_clear`` (m, - tol) from confirmed structure (default hard_clear_m).  tol 0.04 ~ half a cell diagonal:
    sample points on cell corners round either way.  Blocked cells (C-space) are refused regardless."""
    q = _seg_cells(a, b)
    H, W = cm.blocked.shape
    if (q[:, 0] < 0).any() or (q[:, 0] >= H).any() or (q[:, 1] < 0).any() or (q[:, 1] >= W).any():
        return False
    if cm.blocked[q[:, 0], q[:, 1]].any():
        return False
    if int(cm.cls[q[:, 0], q[:, 1]].max()) > max_cls:
        return False
    mc = cm.hard_clear_m if min_clear is None else min_clear
    if mc < cm.plan_clear_m:                  # inside a narrow gap: never closer than the path itself was
        tol_m = 0.0
    return float(np.minimum(cm.clear, cm.clear_s)[q[:, 0], q[:, 1]].min()) >= mc - tol_m


def prune(cm: CostMap, path: Sequence[Tuple[int, int]], tol_m=0.04) -> List[Tuple[int, int]]:
    """Greedy string pulling: from each anchor, the furthest path cell reachable in a straight line that is never
    closer to structure than the original path between them (- tol) nor through a worse cell class."""
    if len(path) <= 2:
        return list(path)
    P = np.asarray(path, int)
    clr = np.minimum(cm.clear, cm.clear_s)[P[:, 0], P[:, 1]]
    cls = cm.cls[P[:, 0], P[:, 1]].astype(int)
    out = [tuple(P[0])]
    a = 0
    while a < len(P) - 1:
        k = a + 1
        mc, mx = float(clr[a:k + 1].min()), int(cls[a:k + 1].max())
        best = k
        while k + 1 < len(P):
            k2 = k + 1
            mc2, mx2 = min(mc, float(clr[k2])), max(mx, int(cls[k2]))
            if los_ok(cm, P[a], P[k2], min_clear=mc2, max_cls=mx2, tol_m=tol_m):
                k, mc, mx, best = k2, mc2, mx2, k2
            else:
                break
        out.append(tuple(P[best]))
        a = best
    return out


def frontiers(cm: CostMap, min_size=3) -> List[dict]:
    """Known-free, unblocked cells with a 4-neighbour unknown cell, grouped (8-connected).  Each: dict(cell (the
    member nearest the group's centroid), size, cells (k, 2))."""
    F = cm.free & ~cm.blocked
    U = cm.unknown
    nbr_u = np.zeros_like(U)
    nbr_u[1:, :] |= U[:-1, :]
    nbr_u[:-1, :] |= U[1:, :]
    nbr_u[:, 1:] |= U[:, :-1]
    nbr_u[:, :-1] |= U[:, 1:]
    fr = F & nbr_u
    cells = set(zip(*np.nonzero(fr)))
    out = []
    while cells:
        seed = cells.pop()
        comp = [seed]
        stack = [seed]
        while stack:
            i, j = stack.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    q = (i + di, j + dj)
                    if q in cells:
                        cells.remove(q)
                        comp.append(q)
                        stack.append(q)
        if len(comp) < min_size:
            continue
        C = np.asarray(comp, int)
        c = C.mean(axis=0)
        k = int(np.argmin(np.hypot(C[:, 0] - c[0], C[:, 1] - c[1])))
        out.append(dict(cell=(int(C[k, 0]), int(C[k, 1])), size=len(comp), cells=C))
    return out


def choose_frontier(cm: CostMap, start, goal_cell, fronts: List[dict], lam=1.0, dist=None) -> Optional[dict]:
    """The frontier minimising (cost to reach it) + lam * (distance frontier -> goal, cells).  ``dist``: a
    dijkstra() result from ``start`` (computed if None).  Adds 'reach' and 'score' to the chosen dict."""
    if not fronts:
        return None
    D = dijkstra(cm, start) if dist is None else dist
    best, bs = None, math.inf
    for fr in fronts:
        C = fr['cells']
        r = D[C[:, 0], C[:, 1]]
        k = int(np.argmin(r))
        if not math.isfinite(float(r[k])):
            continue
        cell = (int(C[k, 0]), int(C[k, 1]))
        sc = float(r[k]) + lam * float(math.hypot(cell[0] - goal_cell[0], cell[1] - goal_cell[1]))
        if sc < bs:
            bs, best = sc, dict(fr, cell=cell, reach=float(r[k]), score=sc)
    return best


def waypoints(grid, cm: CostMap, cells: Sequence[Tuple[int, int]], merge_m=0.3, max_seg_m=1.5) -> np.ndarray:
    """Pruned cells -> grid-frame (u, v) waypoints (start cell dropped): segments shorter than ``merge_m`` merged
    where the straight line stays valid, segments longer than ``max_seg_m`` split."""
    C = [tuple(c) for c in cells]
    if len(C) < 2:
        return np.zeros((0, 2))
    keep = [C[0]]
    for k in range(1, len(C)):
        c = C[k]
        last = k == len(C) - 1
        if not last and math.hypot(c[0] - keep[-1][0], c[1] - keep[-1][1]) * cm.res < merge_m:
            nxt = C[k + 1]
            q = np.vstack([_seg_cells(keep[-1], c), _seg_cells(c, nxt)])     # never closer than the two segments
            cl = float(np.minimum(cm.clear, cm.clear_s)[q[:, 0], q[:, 1]].min())
            if los_ok(cm, keep[-1], nxt, min_clear=cl, max_cls=int(cm.cls[q[:, 0], q[:, 1]].max())):
                continue
            if len(keep) >= 2:                                               # or drop the previous point instead
                q = np.vstack([_seg_cells(keep[-2], keep[-1]), _seg_cells(keep[-1], c)])
                cl = float(np.minimum(cm.clear, cm.clear_s)[q[:, 0], q[:, 1]].min())
                if los_ok(cm, keep[-2], c, min_clear=cl, max_cls=int(cm.cls[q[:, 0], q[:, 1]].max())):
                    keep[-1] = c
                    continue
        keep.append(c)
    uv = grid.uv_of_cell(np.asarray(keep, float))
    out = []
    for k in range(1, len(uv)):
        a, b = uv[k - 1], uv[k]
        n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / max_seg_m)))
        for t in range(1, n + 1):
            out.append(a + (b - a) * (t / n))
    return np.asarray(out)


@dataclass
class Plan:
    cells: List[Tuple[int, int]]          # A* path (incl. start)
    pruned: List[Tuple[int, int]]
    uv: np.ndarray                        # waypoints (grid frame), start excluded
    cost: float
    unknown_share: float                  # fraction of path cells that are unknown
    start_moved: bool                     # the start was inside C-space; planned from the nearest free cell


def plan_to_region(grid, cm: CostMap, start_uv, center_uv, radius_m, max_expand=None, merge_m=0.3,
                   max_seg_m=1.5) -> Optional[Plan]:
    """A* from start (grid-frame uv) to the disc (center_uv, radius_m), prune, waypoints."""
    s0 = tuple(int(x) for x in grid.cell_of_uv(np.asarray(start_uv, float)))
    s = nearest_unblocked(cm, s0)
    if s is None:
        return None
    ii, jj = np.mgrid[0:cm.blocked.shape[0], 0:cm.blocked.shape[1]]
    U = grid.uv_of_cell(np.stack([ii, jj], axis=-1))
    gm = (np.hypot(U[..., 0] - center_uv[0], U[..., 1] - center_uv[1]) <= radius_m) & ~cm.blocked
    gc = grid.cell_of_uv(np.asarray(center_uv, float)).astype(float)
    r = astar(cm, s, gm, goal_center=gc, goal_radius_cells=radius_m / grid.res, max_expand=max_expand)
    if r is None:
        return None
    cells, c = r
    pr = prune(cm, cells)
    P = np.asarray(cells, int)
    unk = float(cm.unknown[P[:, 0], P[:, 1]].mean())
    return Plan(cells=cells, pruned=pr, uv=waypoints(grid, cm, pr, merge_m=merge_m, max_seg_m=max_seg_m), cost=c,
                unknown_share=unk, start_moved=s != s0)
