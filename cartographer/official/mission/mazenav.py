"""Map-mode maze navigation (maze-planner-design.md v1, phase 2 / plan A): checkpoint stages, replanning.

Pure Python + numpy, no ROS, no clock reads (the caller passes ``now``), no core.py import.  Everything here is in the
grid frame (u, v) of an occgrid.GridFrame; the mission converts to NED and builds the Follower.

Stages (the checkpoint ORDER is fixed, positions are coarse):
* ``gate``  : a gap to pass through (maze wall end, Gate A).  Goal = cells beyond the gate line (0.25 ``exit_m`` ..
  ``exit_m + 1.1`` along ``through``) within ``search_m`` sideways of the prior centre, so A* finds the real gap
  wherever it is in that strip.  Passed when the vehicle crosses the gate line within ``search_m`` (+ 0.5).
* ``region``: reach the checkpoint area (a coarse Gate A / maze-exit coordinate).  Goal = the unblocked cell nearest
  the prior centre within ``search_m``.  Passed on entering ``arrive_m`` (default = ``search_m``, 1 m) of the centre,
  or within 0.3 m of the goal cell.
* ``final`` : an exact end point (the tunnel standoff).  Goal = that cell, or the unblocked cell nearest it within
  ``search_m`` (event).  Passed when the mission's follower arrives (the mission calls ``finish()``).

Planning: A* on the cost map (optimistic unknown), through the current stage and chained on to the next one, so the
vehicle never stops at a gate.  Strict first (confirmed structure blocks < 0.38 m); when that finds no path, relaxed
(``relax_kw``: confirmed blocks < 0.30 m with 0.30-0.38 m 'narrow' cells; suspect cells as strict) -- a gap
narrower than 2 x 0.38 m is flown along its centre.  The grid's soft prior layer (prior wall lines) only costs.  Replans at ``replan_period_s`` and on demand; the current path is
kept unless it became blocked (C-space), a new one is >= ``replan_gain`` cheaper, or the stage changed (no jitter).
Nothing even relaxed: ``nopath`` -- the mission re-observes (one yaw scan), then flies the prior route from where it
is (or lands).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .gridplan import CostMap, Plan, astar, cost_map, nearest_unblocked, prune, waypoints, _seg_cells


@dataclass
class Stage:
    name: str
    kind: str                              # 'gate' | 'region' | 'final'
    center: np.ndarray                     # grid frame (u, v)
    search_m: float = 1.0
    through: Optional[np.ndarray] = None   # gate: unit crossing direction (u, v)
    exit_m: float = 0.4
    arrive_m: Optional[float] = None      # region: passed within this of the centre (None = search_m)
    goal: Optional[np.ndarray] = None      # region / final: the goal cell centre last planned to (u, v)


@dataclass
class NavCmd:
    kind: str                              # 'path' | 'nopath' | 'turn' (face goals[0..] first: the side mask)
    goals: np.ndarray                      # (k, 2) grid-frame waypoints, start excluded
    final: bool                            # the last goal is the final stage's end point
    stage: int
    reason: str
    info: dict = field(default_factory=dict)


class MazeNavigator:
    def __init__(self, grid, stages: List[Stage], plan_clear_m=0.38, pref_clear_m=0.55, unknown_cost=2.0,
                 suspect_cost=15.0, suspect_clear_m=0.28, replan_period_s=0.67, replan_gain=0.15, merge_m=0.3,
                 max_seg_m=1.5, chain_next=True, near_skip_m=0.35, fence_clear_m=None, relax_clear_m=0.30,
                 narrow_cost=6.0, prior_cost=15.0, prior_cell_cost=50.0):
        self.grid = grid
        self.stages = list(stages)
        self.cm_kw = dict(plan_clear_m=plan_clear_m, pref_clear_m=pref_clear_m, unknown_cost=unknown_cost,
                          suspect_cost=suspect_cost, suspect_clear_m=suspect_clear_m, hard_clear_m=None,
                          narrow_cost=narrow_cost, fence_clear_m=fence_clear_m, prior_cost=prior_cost,
                          prior_cell_cost=prior_cell_cost)
        self.relax_kw = dict(self.cm_kw, hard_clear_m=relax_clear_m)
        self.replan_period_s, self.replan_gain = replan_period_s, replan_gain
        self.merge_m, self.max_seg_m, self.chain_next, self.near_skip_m = merge_m, max_seg_m, chain_next, near_skip_m
        self.cheaper_hold_s = 3.0                 # at most one cost-only replacement per this long
        self.k = 0
        self.path: Optional[np.ndarray] = None    # current path polyline (u, v) incl. its start
        self.path_kind = None
        self.path_mode = None                     # 'strict' | 'relaxed'
        self.t_plan = -math.inf
        self.cm: Optional[CostMap] = None
        self.passed: List[str] = []
        self._stage_changed = True
        self.last_info: dict = {}

    # ---- stages
    @property
    def stage(self) -> Optional[Stage]:
        return self.stages[self.k] if self.k < len(self.stages) else None

    def done(self) -> bool:
        return self.k >= len(self.stages)

    def finish(self):
        """The mission's follower arrived at the final end point."""
        if self.stage is not None:
            self.passed.append(self.stage.name)
            self.k += 1
            self._stage_changed = True

    def progress(self, prev_uv, pos_uv) -> List[str]:
        """Advance past every stage the move prev -> pos completed; returns their names."""
        out = []
        while self.stage is not None and self.stage.kind != 'final':
            st = self.stage
            if st.kind == 'gate' and st.through is not None:
                if prev_uv is None:
                    break
                d = st.through
                s0 = float((np.asarray(prev_uv) - st.center) @ d)
                s1 = float((np.asarray(pos_uv) - st.center) @ d)
                lat = float(abs((np.asarray(pos_uv) - st.center) @ np.array([-d[1], d[0]])))
                if not (s0 < 0.0 <= s1 and lat <= st.search_m + 0.5):
                    break
            else:
                ar = st.search_m if st.arrive_m is None else st.arrive_m
                d_c = float(np.linalg.norm(np.asarray(pos_uv) - st.center))
                d_g = math.inf if st.goal is None else float(np.linalg.norm(np.asarray(pos_uv) - st.goal))
                if d_c > ar and d_g > 0.3:
                    break
            out.append(st.name)
            self.passed.append(st.name)
            self.k += 1
            self._stage_changed = True
        return out

    # ---- goals
    def _cell_uv(self):
        if getattr(self, '_U', None) is None:
            g = self.grid
            ii, jj = np.mgrid[0:g.nu, 0:g.nv]
            self._U = g.uv_of_cell(np.stack([ii, jj], axis=-1))
        return self._U

    def goal_mask(self, cm: CostMap, st: Stage):
        """(mask, centre cell, radius in cells) of the stage goal; mask may be empty."""
        g, U = self.grid, self._cell_uv()
        free = ~cm.blocked
        if st.kind == 'gate' and st.through is not None:
            d = st.through
            rel = U - st.center
            s = rel @ d
            lat = np.abs(rel @ np.array([-d[1], d[0]]))
            lo, hi = 0.25 * st.exit_m, st.exit_m + 1.1           # (the wall line itself is uncertain by +-0.5 m:
            m = (s >= lo) & (s <= hi) & (lat <= st.search_m) & free  # a narrow next lane may end 0.6 m beyond it)
            c = st.center + d * (0.5 * (lo + hi))
            return m, g.cell_of_uv(c).astype(float), (st.search_m + 0.5 * (hi - lo)) / g.res
        dist = np.hypot(U[..., 0] - st.center[0], U[..., 1] - st.center[1])
        cand = free & (dist <= st.search_m)
        m = np.zeros_like(free)
        if cand.any():
            kc = tuple(int(x) for x in g.cell_of_uv(st.center))
            if st.kind == 'final' and free[kc]:
                k, st.goal = kc, np.asarray(st.center, float).copy()      # the exact end point
            else:                                   # nearest to the centre, ties -> more clearance
                key = np.where(cand, dist - 0.01 * np.minimum(cm.clear, 1.0), np.inf)
                k = np.unravel_index(int(np.argmin(key)), key.shape)
                st.goal = U[k].copy()
            m[k] = True
        return m, g.cell_of_uv(st.center).astype(float), 0.0

    # ---- planning
    def _astar_to(self, cm, start_cell, st):
        m, gc, rc = self.goal_mask(cm, st)
        if not m.any():
            return None
        return astar(cm, start_cell, m, goal_center=gc, goal_radius_cells=rc)

    def plan(self, cm: CostMap, start_uv) -> Optional[Plan]:
        g = self.grid
        s0 = tuple(int(x) for x in g.cell_of_uv(np.asarray(start_uv, float)))
        s = nearest_unblocked(cm, s0)
        st = self.stage
        if s is None or st is None:
            return None
        r = self._astar_to(cm, s, st)
        if r is None:
            return None
        cells, cost = r
        final = st.kind == 'final'
        if self.chain_next and not final and self.k + 1 < len(self.stages):
            r2 = self._astar_to(cm, cells[-1], self.stages[self.k + 1])
            if r2 is not None:
                cells = cells + r2[0][1:]
                cost += r2[1]
                final = self.stages[self.k + 1].kind == 'final'
        pr = prune(cm, cells)
        uv = waypoints(g, cm, pr, merge_m=self.merge_m, max_seg_m=self.max_seg_m)
        if final and len(uv):
            fs = self.stages[self.k] if st.kind == 'final' else self.stages[self.k + 1]
            uv[-1] = fs.goal if fs.goal is not None else uv[-1]       # the exact end point (cell centre otherwise)
        P = np.asarray(cells, int)
        p = Plan(cells=cells, pruned=pr, uv=uv, cost=cost, unknown_share=float(cm.unknown[P[:, 0], P[:, 1]].mean()),
                 start_moved=s != s0)
        p.final = final
        return p

    def path_status(self, cm: CostMap, pos_uv):
        """(valid, remaining cost) of the current path from the point nearest pos on: blocked C-space cells beyond
        near_skip_m of the vehicle make it invalid (the vehicle itself may sit inside C-space at a corner)."""
        if self.path is None or len(self.path) < 2:
            return False, math.inf
        P = self.path
        pos = np.asarray(pos_uv, float)
        best, kb, tb = math.inf, 0, 0.0
        for k in range(len(P) - 1):
            a, b = P[k], P[k + 1]
            e = b - a
            t = float(np.clip((pos - a) @ e / max(float(e @ e), 1e-12), 0.0, 1.0))
            d = float(np.linalg.norm(a + t * e - pos))
            if d < best:
                best, kb, tb = d, k, t
        pts = [P[kb] + tb * (P[kb + 1] - P[kb])] + [P[k] for k in range(kb + 1, len(P))]
        return self.poly_cost(cm, pts, pos)

    def _same_path(self, uv, pos_uv, tol=0.3):
        """Every new waypoint lies within tol of the current path polyline (and vice versa for its end)."""
        P = self.path
        def dist(q, Q):
            best = math.inf
            for a, b in zip(Q[:-1], Q[1:]):
                e = b - a
                t = float(np.clip((q - a) @ e / max(float(e @ e), 1e-12), 0.0, 1.0))
                best = min(best, float(np.linalg.norm(a + t * e - q)))
            return best
        Q = np.vstack([np.asarray(pos_uv, float)[None, :], np.asarray(uv, float)])
        return all(dist(q, P) <= tol for q in uv) and dist(P[-1], Q) <= tol

    def poly_cost(self, cm: CostMap, pts, pos):
        """(valid, cost) of a polyline: cost = mean cell cost x length / res per segment (the same metric for the
        current and a new path); blocked cells beyond near_skip_m of pos make it invalid."""
        pos = np.asarray(pos, float)
        g = self.grid
        cost = 0.0
        valid = True
        for a, b in zip(pts[:-1], pts[1:]):
            q = _seg_cells(g.cell_of_uv(a), g.cell_of_uv(b))
            if not len(q):
                continue
            H, W = cm.blocked.shape
            ok = (q[:, 0] >= 0) & (q[:, 0] < H) & (q[:, 1] >= 0) & (q[:, 1] < W)
            if not ok.all():
                return False, math.inf
            uvq = g.uv_of_cell(q)
            far = np.hypot(uvq[:, 0] - pos[0], uvq[:, 1] - pos[1]) > self.near_skip_m
            if (cm.blocked[q[:, 0], q[:, 1]] & far).any():
                valid = False
            c = cm.cost[q[:, 0], q[:, 1]]
            c = np.where(np.isfinite(c), c, 50.0)
            cost += float(c.mean()) * float(np.linalg.norm(b - a)) / g.res
        return valid, cost

    def step(self, now, pos_uv, force=False, can_replace=True, force_if_new=False, block=None) -> Optional[NavCmd]:
        """Replan if due (or forced).  Returns a command when the follower should get a new path ('path'), or when
        nothing connects to the stage goal even relaxed ('nopath'); else None.  force_if_new: a forced replan (the
        follower is blocked) only returns a path that differs from the current one by > 0.3 m somewhere -- the same
        path again would only reset the follower's observe / nudge / re-confirm."""
        if self.done():
            return None
        period = 0.2 if self._stage_changed else self.replan_period_s   # (stage change: as soon as replaceable)
        if not (force or now - self.t_plan >= period):
            return None
        self.t_plan = now
        st = self.stage
        cm = cost_map(self.grid, extra_block=block, **self.cm_kw)
        p, mode = self.plan(cm, pos_uv), 'strict'
        if p is None:                             # relaxed: narrow (0.30-0.38 m) gaps
            cmr = cost_map(self.grid, extra_block=block, **self.relax_kw)
            p = self.plan(cmr, pos_uv)
            if p is not None:
                cm, mode = cmr, 'relaxed'
        if p is None and block is not None:      # blocked only by the caller's side mask (cells the camera cannot
            for kw in (self.cm_kw, self.relax_kw):  # have seen beside it): turn along that plan first, then replan
                pn = self.plan(cost_map(self.grid, **kw), pos_uv)
                if pn is not None and len(pn.uv):
                    info = dict(stage=st.name, stage_kind=st.kind, mode='turn')
                    self.last_info = info
                    return NavCmd('turn', pn.uv, False, self.k, 'side_unknown', info)
        self.cm = cm
        info = dict(stage=st.name, stage_kind=st.kind, mode=mode)
        if p is None:
            self.last_info = info
            return NavCmd('nopath', np.zeros((0, 2)), False, self.k, 'no_path', info)
        if not len(p.uv):                         # already inside the goal
            if st.kind == 'final':
                p.uv = np.asarray(st.goal if st.goal is not None else st.center, float)[None, :]
                p.final = True
            else:
                self.passed.append(st.name)
                self.k += 1
                self._stage_changed = True
                info['passed_inside'] = st.name
                return self.step(now, pos_uv, force=True, can_replace=can_replace) if not self.done() else None
        valid, rem = self.path_status(cm, pos_uv)
        info.update(cost=round(p.cost, 1), n=len(p.uv), unknown=round(p.unknown_share, 2),
                    old_cost=None if not math.isfinite(rem) else round(rem, 1), old_valid=valid)
        why = None
        if self._stage_changed:
            why = 'stage'
        elif self.path is None:
            why = 'first'
        elif not valid:
            why = 'blocked'
        elif force:
            why = 'forced'
        elif mode != self.path_mode:
            why = 'mode'
        elif now - getattr(self, '_t_cheaper', -math.inf) >= self.cheaper_hold_s:
            new_cost = self.poly_cost(cm, [np.asarray(pos_uv, float)] + list(p.uv), pos_uv)[1]
            info['new_poly_cost'] = round(new_cost, 1)
            if new_cost < (1.0 - self.replan_gain) * rem and rem - new_cost >= 5.0:
                why = 'cheaper'
                self._t_cheaper = now
        if why == 'cheaper' and not can_replace:
            self._t_cheaper = -math.inf               # (not applied: may try again next period)
        if why == 'forced' and force_if_new and self.path is not None and self._same_path(p.uv, pos_uv):
            self.last_info = info
            return None
        if why is None or not (can_replace or why in ('blocked', 'forced')):
            self.last_info = info
            return None
        self._stage_changed = False
        start = np.asarray(pos_uv, float)
        self.path = np.vstack([start[None, :], p.uv]) if len(p.uv) else None
        self.path_kind, self.path_mode = 'path', mode
        self.last_info = info
        if p.start_moved:
            info['start_moved'] = True
        return NavCmd('path', p.uv, bool(getattr(p, 'final', False)), self.k, why, info)


def gate_stage(name, center_uv, through_uv, search_m=1.0, exit_m=0.4) -> Stage:
    d = np.asarray(through_uv, float)
    return Stage(name=name, kind='gate', center=np.asarray(center_uv, float), search_m=search_m,
                 through=d / float(np.linalg.norm(d)), exit_m=exit_m)


def _seg_x(a, b, c, d) -> bool:
    """2-D segments a-b and c-d intersect (proper or touching)."""
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 * o2 <= 0) and (o3 * o4 <= 0)


def infer_gates(corridor_uv, walls_uv, search_m=1.0, exit_m=0.4, beyond_m=0.15, max_gap_m=1.3) -> List[Stage]:
    """Old priors without `checkpoints`: every corridor leg that crosses a prior wall LINE beyond that wall's end (the
    path goes round the end) is a gate at the crossing, through = the leg direction -- only if the crossing is within
    max_gap_m of that end and no other prior wall lies between the end and the crossing (W4 / C1 beyond the barrier
    are not maze gaps)."""
    out = []
    C = [np.asarray(c, float)[:2] for c in corridor_uv]
    for i in range(len(C) - 1):
        a, b = C[i], C[i + 1]
        e = b - a
        for name, (p0, p1) in walls_uv:
            p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
            w = p1 - p0
            den = e[0] * (-w[1]) - e[1] * (-w[0])
            if abs(den) < 1e-9:
                continue
            r = p0 - a
            t = (r[0] * (-w[1]) - r[1] * (-w[0])) / den        # along the leg
            s = (e[0] * r[1] - e[1] * r[0]) / den              # along the wall line (0..1 = on the wall)
            Lw = float(np.linalg.norm(w))
            if not (0.0 <= t <= 1.0) or not (s < -beyond_m / Lw or s > 1.0 + beyond_m / Lw):
                continue
            x = a + t * e
            end = p0 if s < 0.0 else p1
            if float(np.linalg.norm(x - end)) > max_gap_m:      # far beyond the end: not a gap round it
                continue
            if any(_seg_x(end, x, np.asarray(q0, float), np.asarray(q1, float))
                   for nm, (q0, q1) in walls_uv if nm != name):
                continue
            if any(o[1].name == 'gate_%s' % name and float(np.linalg.norm(o[1].center - x)) < 0.3 for o in out):
                continue                                       # (a vertex ON the line: both legs cross it once)
            out.append((i + t, gate_stage('gate_%s' % name, x, e, search_m=search_m, exit_m=exit_m)))
    out.sort(key=lambda z: z[0])
    return [s for _, s in out]
