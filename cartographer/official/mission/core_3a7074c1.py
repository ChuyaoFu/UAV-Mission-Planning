"""Mission state machine for the DroneHacks official lab and the real venue.

Same style as the official ``refnav/core.py``: pure Python + numpy, the caller
passes every timestamp (nothing here reads a clock), no ROS, no I/O except
``CoursePrior.load``.  The ROS node calls ``Mission.step(now, vehicle, perception)``
at ~20 Hz and forwards the ``Output``:

* ``Output.pos_ned is None``  -> publish NOTHING (neither TrajectorySetpoint nor the
  OffboardControlMode heartbeat).  This is the case once we let go (PX4 left
  Offboard / pilot took over), as soon as PX4 is in AUTO_LAND, and in the
  terminal phases ``done`` / ``aborted`` / ``released``.
* ``Output.commands``  -> px4_msgs VehicleCommand (fields ``command``, ``param1``,
  ``param2``).  Disarm is always the NORMAL disarm (param2 = 0), never forced.
* ``Output.events``    -> log them (dicts: t, phase, kind, ...).
* ``Output.hints``     -> which detector to run now and the priors to pass to it.

Frames: PX4 local NED world (x north, y east, z down, metres), yaw = PX4 NED
heading (rad).  The course prior is written in the START frame: origin = the
vehicle position at the moment PX4 first reports ``armed``, forward = the heading
at that moment, then left and up (metres).

Perception inputs are duck-typed against ``official/INTERFACES.md`` so this file
does not import the (separately developed) perception modules:

* TunnelEstimate(found, entry_ned (3,), axis_ned (2,), radius_m, confidence)
* WindowEstimate(found, center_ned (3,), normal_ned (3,), width_m, height_m,
  clearance_ok, confidence)
* LandingTarget(found, center_ned (3,), top_d, extent_m, confidence)
* FloorFit(found, d, normal_ned, inlier_frac)    (carried, not used for control)
* ObstacleMap.clearance(p_from, p_to, radius_m) -> metres free (inf if clear),
  ObstacleMap.blocked(p, radius_m) -> bool
"""
from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------- PX4 constants
NAV_MANUAL = 0
NAV_POSCTL = 2
NAV_AUTO_LOITER = 4
NAV_OFFBOARD = 14
NAV_AUTO_LAND = 18
VEHICLE_CMD_NAV_LAND = 21
VEHICLE_CMD_DO_SET_MODE = 176
VEHICLE_CMD_COMPONENT_ARM_DISARM = 400
PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6.0

PHASES = ('wait', 'arm', 'takeoff', 'corridor', 'tunnel_align', 'tunnel_traverse',
          'window_align', 'window_traverse', 'land_target', 'disarm', 'rearm',
          'climb_over_wall', 'return_home', 'land_home', 'done')
ABORT_PHASE = 'abort_land'
TERMINAL_PHASES = ('done', 'aborted', 'released')
# detector the node should run (or warm up) in each phase; 'wall' = depth measurement of the wall top
DETECTOR_FOR_PHASE = {'corridor': 'tunnel', 'tunnel_align': 'tunnel', 'window_align': 'window',
                      'window_traverse': 'landing', 'land_target': 'landing', 'rearm': 'wall',
                      'climb_over_wall': 'wall', 'gate_pass': 'landing', 'landing_route': 'landing'}
# optional stages from the official course map (prior keys chicane / gate_b / landing_route):
#   tunnel_traverse -> [chicane] -> window_align -> window_traverse | gate_pass -> [landing_route] -> land_target
OPTIONAL_PHASES = ('chicane', 'gate_pass', 'landing_route')
# phases in which we own an armed, flying vehicle in Offboard
FLIGHT_PHASES = ('takeoff', 'corridor', 'tunnel_align', 'tunnel_traverse', 'window_align',
                 'window_traverse', 'land_target', 'climb_over_wall', 'return_home', 'land_home', 'staged_stop',
                 'chicane', 'gate_pass', 'landing_route')
# staged tests: Params.stop_after = one of these -> when it completes normally: hover, land in place, disarm, done
STOP_AFTER_PHASES = ('takeoff', 'corridor', 'tunnel_align', 'tunnel_traverse', 'window_align', 'window_traverse',
                     'land_target', 'chicane', 'gate_pass', 'landing_route')
NEXT_PHASE = {a: b for a, b in zip(PHASES, PHASES[1:])}


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def heading_of(v) -> float:
    return math.atan2(float(v[1]), float(v[0]))


class _Hit(float):
    """A distance along a ray that also carries the map point it was measured to (.pt)."""

    def __new__(cls, d, pt):
        o = float.__new__(cls, d)
        o.pt = np.asarray(pt, float)
        return o


def _min_dist(X, B):
    """(M,) distance from each 2-D point of X to the nearest of B (inf when B is empty)."""
    if not len(B):
        return np.full(len(X), math.inf)
    out = np.empty(len(X))
    for i in range(0, len(X), 256):                  # (bounded memory)
        d = X[i:i + 256, None, :] - B[None, :, :]
        out[i:i + 256] = np.sqrt(np.min(np.einsum('ijk,ijk->ij', d, d), axis=1))
    return out


def _seg_min_dist(A, C, B, trim_a=0.0, trim_c=0.0, rev=False):
    """(M,) for every candidate end C[m]: distance from the segment A -> C[m] (or C[m] -> A when rev) to the
    nearest of the 2-D points B, ignoring the first trim_a metres at A and the last trim_c at C[m]."""
    C = np.asarray(C, float).reshape(-1, 2)
    if not len(B):
        return np.full(len(C), math.inf)
    A = np.asarray(A, float)[:2]
    d = C - A                                        # (M,2)
    L = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
    ua, ub = np.minimum(trim_a / L, 0.5), np.maximum(1.0 - trim_c / L, 0.5)
    out = np.empty(len(C))
    rel = B - A                                      # (N,2)
    for i in range(0, len(C), 64):
        dd, LL = d[i:i + 64], L[i:i + 64]
        u = (rel @ dd.T) / (LL * LL)[None, :]       # (N,m)
        u = np.clip(u, ua[i:i + 64][None, :], ub[i:i + 64][None, :])
        px = rel[:, 0:1] - u * dd[:, 0][None, :]
        py = rel[:, 1:2] - u * dd[:, 1][None, :]
        out[i:i + 64] = np.sqrt(np.min(px * px + py * py, axis=0))
    return out


def _dilate(m):
    """8-neighbour binary dilation of a 2-D bool grid (pure numpy)."""
    d = m.copy()
    d[1:, :] |= m[:-1, :]
    d[:-1, :] |= m[1:, :]
    e = d.copy()
    e[:, 1:] |= d[:, :-1]
    e[:, :-1] |= d[:, 1:]
    return e


def _tallest_column(Q, cell, gap):
    """Largest vertical extent of contiguous points (z gaps <= gap) inside one cell x cell xy cell of the NED points Q
    (two grids, the second shifted by half a cell: a thin column on a cell boundary is not split)."""
    if not len(Q):
        return 0.0
    return max(_tallest_column_grid(Q, cell, gap, 0.0), _tallest_column_grid(Q, cell, gap, 0.5 * cell))


def _tallest_column_grid(Q, cell, gap, shift):
    k = np.floor((Q[:, :2] + shift) / cell).astype(np.int64)
    order = np.lexsort((Q[:, 2], k[:, 1], k[:, 0]))
    k, z = k[order], Q[order, 2]
    brk = np.r_[True, np.any(k[1:] != k[:-1], axis=1) | (np.diff(z) > gap)]
    run = np.cumsum(brk) - 1
    lo = np.full(run[-1] + 1, np.inf)
    hi = np.full(run[-1] + 1, -np.inf)
    np.minimum.at(lo, run, z)
    np.maximum.at(hi, run, z)
    return float((hi - lo).max())


def _dilate3(m):
    """26-neighbour binary dilation of a 3-D bool grid (pure numpy)."""
    d = m.copy()
    for ax in range(3):
        e = d.copy()
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3
        sl_a[ax], sl_b[ax] = slice(1, None), slice(None, -1)
        e[tuple(sl_a)] |= d[tuple(sl_b)]
        e[tuple(sl_b)] |= d[tuple(sl_a)]
        d = e
    return d


def _runs_along(off):
    """Offsets along a fence face of the points at one depth: a board running ALONG the net fills >= 3 of the five
    0.1 m bins 0.45-0.95 m away on one side (a board crossing it, or a phantom blob, fills at most one)."""
    off = np.asarray(off, float)
    for sd in (1.0, -1.0):
        o = sd * off
        bins = [int(np.count_nonzero((o >= 0.45 + 0.1 * k) & (o < 0.55 + 0.1 * k))) >= 3 for k in range(5)]
        if sum(bins) >= 3:
            return True
    return False


# --------------------------------------------------------------------- params
@dataclass
class Params:
    """Every tunable number of the mission, in one place (SI units, NED)."""
    # vehicle size -> ObstacleMap radius = half_span + clearance_margin
    half_span_m: float = 0.28          # official lab grades contact at "airframe reach" 0.28 m (was 0.20 = the
    clearance_margin_m: float = 0.07   # SKY_BETA half span): probe radius 0.35, still < half the official first-
    hard_margin_m: float = 0.03        # corner gap (0.73 m).  Hard radius: the segment leaving the vehicle
    # speeds: fast where it is open, slow only where it must be
    v_cruise: float = 1.8              # corridor / transits (m/s)
    a_max: float = 1.5                 # carrot acceleration (m/s^2)
    a_brake: float = 2.5               # carrot deceleration (obstacle, leash)
    v_up: float = 1.0                  # takeoff and liftoff climb
    v_align: float = 0.8               # last move onto the tunnel/window axis
    v_tunnel: float = 1.0              # constant traverse speed in the tunnel
    v_window: float = 0.8              # constant traverse speed through the window (official-compat round 4: at
                                       # 1.2 m/s lateral offsets were 0.6-16.3 cm with a 1.4-7.8 cm latch error)
    v_descent: float = 0.5             # descent onto a landing surface
    v_touch: float = 0.35              # last decimetres before touchdown. PX4 (1.17 MulticopterLandDetector) only
                                       # sets ground_contact while trajectory_setpoint.velocity[2] >= 1.1 *
                                       # LNDMC_Z_VEL_MAX, so this is ALSO the feed-forward kept through touchdown
    v_no_map: float = 0.4              # speed cap while the obstacle map is missing / stale
    yaw_rate: float = 1.6              # yaw setpoint slew (rad/s) in the open
    yaw_rate_near: float = 0.4         # ... within yaw_near_m of a mapped obstacle or the net (official runs 002-004:
    yaw_near_m: float = 1.0            # a 1.6 rad/s 90 deg yaw facing the wall at 0.4 m -> 0.3 m hold oscillation;
                                       # run_005 at 0.4 rad/s: the first corner settles)
    corner_min_frac: float = 0.3       # corner speed = v * max(frac, (1 + cos(turn)) / 2)
    # VIO protection (official lab run_003: flying straight at the corridor end wall, < ~0.6 m camera-to-wall, starved
    # cuVSLAM; the estimate drifted 0.3 m and the drone hit the wall)
    vio_wall_standoff_m: float = 0.8   # camera -> measured obstacle ahead in view: the carrot never gets closer;
                                       # a transit waypoint the prior put closer than this is clipped (turn early)
    corner_turn_deg: float = 20.0      # a path vertex turning more than this is a corner
    corner_yaw_ahead_m: float = 1.0    # within this of a corner: turning on purpose (no yaw-slowdown)
    yaw_lookahead_m: float = 1.2       # obstacle-checked paths face the path point this far ahead of the carrot: on a
                                       # straight leg the leg itself; approaching a corner the camera swings around it
                                       # (looks through the gap behind the board end, not at the board beside the lane)
    near_waypoint_m: float = 1.0       # within this of a corner or the end of an obstacle-checked path ...
    v_near_waypoint: float = 0.3       # ... speed is capped at this (official sim: ~0.3 m overshoot from 0.5 m/s)
    near_waypoint_lag_m: float = 0.25  # the carrot reaches that speed this much earlier (the vehicle lags ~0.2 s)
    nudge_max_speed: float = 0.15      # parallax nudges start only once (nearly) stopped, centred on the vehicle
    vio_view_radius_m: float = 0.2     # "straight ahead": a thin probe along the camera heading from the carrot
    vio_hold_margin_m: float = 0.1     # the carrot already stops this much before the standoff (tracking overshoot)
    # tight corners (official-compat maze v3: 0.43 m from the corner point to the wall ahead; the carrot rounded it at
    # 0.47 m/s, PX4 (MPC_XY_P 0.95, MPC_XY_VEL_P_ACC 1.8) decelerates at ~0.7 m/s^2 -> 0.17-0.21 m overshoot, props
    # (0.23 m) hit the wall): stop ON the corner point, settle, face the next leg, then go
    tight_corner_space_m: float = 0.6  # corner point -> measured obstacle beyond it (incoming direction) below this
    a_corner_stop: float = 0.35        # carrot (= feed-forward) deceleration into a tight corner (m/s^2)
    a_stop_plan: float = 0.5           # planned stops on obstacle-checked paths (path end, corners, obstacles):
                                       # never brake the carrot harder than the vehicle can follow
    veh_brake_mps2: float = 0.6        # what the vehicle really manages: the carrot is never left behind the
    veh_lag_s: float = 0.15            # vehicle's own stop point (proj + v^2 / (2 * this) + v * lag); 0 = off
    acc_ff_max: float = 0.6            # |acceleration feed-forward| cap on obstacle-checked paths (run_005: -1.69)
    corner_yaw_early: bool = True      # stop corners: turn to the next leg during the last corner_yaw_ahead_m,
                                       # slowly enough to finish before the corner (no 90 deg yaw in place)
    corner_clear_target_m: float = 0.6 # stop-corner points move (if the corridor allows) to this from boards / net
    vertex_seen_only: bool = True      # (v4J eyeball s2: with W1 not yet measured, centring pushed the first corner
                                       # 0.6 m off the west net, past W1's real face; the first leg ran into W1)
                                       # with no measured board around the vertex, it moves only to points seen
                                       # free (camera view / own track)
    vertex_fence_clear_m: float = 0.55 # (v4J: centring took the first corner to 0.40 m from the north / west net; a
                                       # 0.15 m overshoot or 0.23 m VIO error then broke 0.28) vertex moves keep this
                                       # from the fence, or at least no closer than the vertex already is
    carrot_back_v: float = 0.15        # the carrot moves back (VIO standoff) at most this fast, never behind the vehicle
    hold_drift_m: float = 0.15         # holding still but |pos - sp| grows past this -> back away from the nearest
    hold_drift_back_m: float = 0.15    # wall by this (official run_003: +-0.3 m hold oscillation after the yaw)
    hold_drift_wait_s: float = 2.0     # ... and hold there this long before going on
    corner_creep_m: float = 0.15       # the last metres into a tight corner at v_corner_creep (robust without the
    v_corner_creep: float = 0.15       # acceleration feed-forward: the PX4 lag error scales with the deceleration)
    corner_settle_tol_m: float = 0.08  # settled = vehicle this close to the corner point (horizontal) ...
    corner_settle_v: float = 0.10      # ... and slower than this
    corner_yaw_tol_deg: float = 12.0   # then go once the heading is this close to the next leg
    corner_settle_max_s: float = 6.0   # never wait longer than this at a corner (within 2x the settle tolerance)
    stop_corner_deg: float = 45.0      # corners turning >= this also stop on the point (overshoot <= 0.1 m everywhere);
                                       # smaller turns (prior noise, e.g. a 0.47 m leg + 69 deg -> after centring
                                       # < 45 deg) are rounded and never trigger the corner hold
    # narrow lanes (official-compat seed 6: a tape-level prior point 0.2 m from a board -> 7 prop strikes): corridor
    # vertices are pushed onto the MEASURED lane centre, >= lane_clear_m from mapped boards (and inside the fence)
    lane_clear_m: float = 0.40         # prop diagonal 0.23 + margin
    snap_max_m: float = 0.45           # never move a prior vertex further than this
    snap_range_m: float = 3.0          # vertices this far ahead of the carrot are (re)centred as the map fills in
    snap_band_m: float = 0.40          # map points within +- this of the flight height count as boards
    snap_period_s: float = 0.25
    fence_phantom_m: float = 0.5       # stereo phantoms in front of the net: an obstacle this close to a fence face
                                       # does not hold / block a corner after the nudge; we go on slowly (v_minor).
    # ... and ONLY there (official-compat v4 round B: the generic small-cluster rule ignored thin board ends -- the
    # W1 north / W2 south ends seen edge-on are 6-11 point clusters -> 7-10 prop strikes, two crashes).  A cluster
    # is discounted only if it lies within fence_phantom_m of a fence face AND is not continuous with structure:
    # map points confirmed from two viewpoints that reach phantom_inward_m past the band (a board end at the net)
    # or run along the face, or a prior wall line (optional prior key `walls`) within wall_line_keep_m.
    phantom_inward_m: float = 0.25     # structure reaching fence_phantom_m + this from the face = a board, not a blob
    phantom_gap_m: float = 0.2         # map points this close together are one structure (connected component)
    net_layer_m: float = 0.20          # points this close to the face are the net itself (the geofence keeps us off);
                                       # they are never 'structure' and never vertical support (v4E: north/west-net
                                       # phantoms in the first corridor were 'supported' through the net; the compat
                                       # analysis also classes <= 0.20 m of a cage plane as net)
    confirm_horizontal: bool = True    # (v4C: the vertical takeoff climb 'confirmed' a west-net phantom -> blocked at
                                       # the start 10/10; climbing does not carve net phantoms: count horizontal only)
    confirm_baseline_m: float = 0.25   # "seen from two viewpoints": camera moved this far (or confirm_angle_deg
    confirm_angle_deg: float = 10.0    # in bearing) while the map point stayed in view and was not carved
    confirm_period_s: float = 0.25
    confirm_memory_s: float = 30.0     # map cells seen in view are remembered this long: lane centring keeps away
                                       # from boards the node's 4 s map already forgot (v4 offset seed 5: a vertex
                                       # re-centred straight toward W1's forgotten end)
    wall_line_keep_m: float = 0.5      # cluster this close (+ its radius + the wall's prior uncertainty) to a prior
                                       # wall segment or end: always real
    board_open_min_pts: int = 4        # lane centring away from the net: >= this many points per 0.3 m block
                                       # (an edge-on thin board end; inside the net band: minor_cluster_n)
    # chicane / gate line-up (round B demo 230517: the straight line chicane end -> Gate B line-up passed 0.05 m
    # from C1's south end): chicane vertices (and the chicane END, judged with the next stage's line-up point as
    # its next leg) snap to the CENTRE of the measured gap, legs checked within gap_leg_m of the vertex
    wall_relative: bool = True         # corridor / maze / landing route: the prior polyline is only the TOPOLOGY
                                       # (turn signs, order); vertices go to the centre of the MEASURED lane / turn
                                       # gap (<= gap_snap_max_m, legs <= 1.5 x prior length); prior xy = fallback
    gap_snap_max_m: float = 0.7
    gap_leg_m: float = 1.2
    gap_clear_cap_m: float = 0.6       # centring objective: min(board clearance, fence distance, this)
    gate_standoff_min_m: float = 0.45  # the Gate B line-up point moves toward the gate (not below this) until it
                                       # clears mapped boards by lane_clear_m
    vio_hold_max_s: float = 4.0        # camera-ahead hold with a clear path probe: creep on after this (no deadlock)
    unseen_check_m: float = 0.6        # the path this far ahead of the carrot must have been in the camera view;
    unseen_look_max_s: float = 3.0     # else stop and look at it first (a leg 52 deg off the camera hit an unseen
                                       # board end in seed 6); go on anyway after this
    # contact (prop strike): IMU spike or velocity jump -> hover, back off along where we came from, replan
    contact_accel_mps2: float = 15.0   # |raw IMU specific force| above this (gravity ~9.8 included)
    contact_dv_mps2: float = 12.0      # |d vel_ned / dt| above this (no IMU field)
    contact_hover_s: float = 0.5
    contact_backoff_m: float = 0.20
    contact_backoff_v: float = 0.3
    contact_max: int = 4               # contacts per phase before stop + land
    # (v4J eyeball s3, v4K map s1, the real drone's 38 m VIO jump after an impact: the mission kept flying on a broken
    # estimate; contact recovery / replanning / centring went back into the same wall.)  VIO-fault guard: an impact,
    # a pose-health drop, or an estimate jump > vio_jump_m in one step that the velocity does not explain -> hover
    # with the setpoint ON the current estimate (never chase a jumped estimate), follower and centring frozen.  Once
    # VIO is consistent for vio_ok_s (and >= vio_fault_hold_s): an impact backs off along the flown track
    # (<= contact_backoff_m), then hold vio_quarantine_s (the node map's memory: points recorded during the fault
    # expire) and replan.  Not consistent within vio_fault_max_s or total jump > vio_jump_abort_m -> land in place.
    vio_guard: bool = True
    vio_jump_m: float = 0.3
    vio_jump_abort_m: float = 0.5
    vio_fault_hold_s: float = 1.5
    vio_ok_s: float = 1.0
    vio_fault_max_s: float = 4.0
    vio_quarantine_s: float = 4.0
    vio_fault_repeat_n: int = 0        # > 0: this many faults within vio_fault_repeat_s -> land (off: contact_max)
    vio_fault_repeat_s: float = 20.0
    prop_radius_m: float = 0.28        # airframe reach (official lab contact radius); feed-forward toward a wall is
    ff_stop_decel: float = 0.4         # capped so that v^2 / (2 * this) + v * veh_lag_s + prop_radius_m +
                                       # ff_wall_margin_m <= distance ahead
    ff_wall_margin_m: float = 0.10
    leash_lo_m: float = 0.5            # carrot this far ahead of the vehicle: start holding it back
    leash_hi_m: float = 1.2            # ...this far: carrot stops
    yaw_slow_lo_deg: float = 15.0      # |heading - travel dir| below this: full speed
    yaw_slow_hi_deg: float = 50.0      # above this: yaw_slow_min_frac of the speed
    yaw_slow_min_frac: float = 0.2
    yaw_end_dist_m: float = 1.0        # last metre of an approach: already face the stage
    # obstacle probe and side step
    probe_min_m: float = 1.2           # probe length ahead of the carrot (>= stopping distance + margin)
    probe_margin_m: float = 0.5
    stop_margin_m: float = 0.3         # carrot stops this far before the first blocked point
    side_step_m: float = 0.15          # candidate lateral offsets: k * side_step
    side_max_m: float = 1.2
    v_side: float = 0.8                # lateral offset slew (m/s)
    return_margin_m: float = 0.10      # extra radius before moving back toward the nominal path
    avoid_period_s: float = 0.2        # search rate limit
    blocked_timeout_s: float = 6.0     # hover this long with no free side step -> land
    # stereo phantoms (Ubuntu official-compat stereo runs: 3-7 phantom voxels triggered 1.05-1.2 m side steps into
    # never-observed panels): small clusters only slow / stop us, and side steps stay out of unobserved space
    minor_cluster_n: int = 12          # fewer map points than this near the first blocked point = "minor" cluster
    cluster_radius_extra_m: float = 0.10   # count points within radius + this of the first blocked probe sample
    v_minor: float = 0.5               # speed cap while a minor cluster is in the probe (carving gets time to act)
    minor_wait_s: float = 1.0          # minor cluster still there after this (or we had to stop) -> parallax nudge;
                                       # still minor after the nudge -> ignored (excluded locally), passed at v_minor.
                                       # Minor clusters never block and never cause a side step (round 5b seed 7).
    nudge_lat_m: float = 0.15          # parallax nudge: one sine period sideways +-this ...
    nudge_up_m: float = 0.10           # ... plus an up-bump, so view-locked stereo mismatches get carved
    nudge_period_s: float = 2.0        # done once per blockage, before anything is declared blocked
    observe_scan_deg: float = 20.0     # when the nudge cannot go sideways (sides not seen free): an observation
    observe_scan_period_s: float = 6.0 # manoeuvre instead -- yaw scan +-this in place (0.37 rad/s peak) and a
    observe_up_m: float = 0.2          # climb of this, so the node can carve a view-locked phantom (v4C start)
    # a SMALL cluster that is kept (not a net phantom) and still blocks after the nudge (v4D: 7/10 stuck at the start
    # behind a 7-10 point mismatch 0.55 m ahead in a lane too narrow to pass it): observe -> sidestep -> re-confirm.
    # Yaw up to reconfirm_scan_deg toward a side to see it free, sidestep 0.2-0.3 m into seen-free space for a
    # horizontal baseline, watch it; gone (carved) within reconfirm_watch_s (> the node's 4 s map memory) = phantom
    # (ignored for this path), still there = real (normal blocked handling)
    # vertical support (user-approved): every real obstacle on this course stands on the floor (boards on stands,
    # walls, edge-on board ends = floor-to-top columns, boulders, the tunnel, gates).  A SMALL cluster whose connected
    # map points (gaps <= phantom_gap_m, in 3-D) span < floating_span_m vertically is a floating phantom candidate:
    # discounted like a net phantom (slow pass), re-checked while it is ignored; prior wall lines keep it real
    floating_span_m: float = 0.3
    # (v4G: all 44 kept clusters were 'supported'; the phantoms' support ran down to the floor, h 0.00-0.13 m.)  Floor
    # points (below support_floor_up_m, start-frame up ~ 0.2 m above the floor) are never support, and support is a
    # vertical column: the connected points must include one support_cell_m (xy) cell whose contiguous points span
    # >= floating_span_m (a board end, a wall face, a post).  A blob tied to the floor, or to a wall only through a
    # sloped noise chain, is floating.  Clusters near prior boulders / tunnel / gate / window stay real (boulders
    # are low: their points sit mostly in the floor band).
    support_floor_up_m: float = 0.15
    # (v4J: blind2 priors carry wall lines 0.3 m off; the band r + wall_line_keep_m + tol is ~1 m, so every cluster in
    # a 1.1 m lane was 'prior_wall_line' = real and the column check never ran.)  Near a prior wall line the column
    # check decides too (prior_wall_line is logged); True = the old shortcut.
    prior_line_real: bool = False
    support_gap_m: float = 0.15        # points this close are connected for the support check (phantom_gap_m was
                                       # 0.2: a phantom 0.3 m off a wall face joined the wall and its columns)
    support_cell_m: float = 0.15
    support_prior_m: float = 0.3
    never_block_n: int = 3             # (v4E) a cluster of fewer map points never blocks on its own: slow pass + nudge
    board_support_filter: bool = True  # (v4E) lane centring / gap vias use only floor-standing in-field points (no net
                                       # layer, footprint spans >= floating_span_m, no ignored clusters): net ghosts in
                                       # the first corridor pulled the turn vertex toward W1's north end
    # (v4E: the start observations held at the take-off point, 0.36-0.44 m in front of the never-seen barrier; hold
    # drift during the yaw scans took it to 0.24-0.27 m.)  Every observation / nudge / sidestep / backoff target keeps
    # move_clear_m (= prop_radius_m + 0.1) from measured map points and the fence, its airframe edge that far out is
    # seen free, and behind a hover point that much must be KNOWN free (own track or camera view: the rear is never
    # seen).  (Prior obstacles: see prior_keepout below -- never a hard limit, they can be 0.5 m off.)  A hover
    # point that fails it moves forward along the path (<= hold_advance_max_m, the move itself clear) to the best
    # point first.
    move_clear_m: float = 0.38
    hold_advance_max_m: float = 0.3
    # (v4F: 2 flights 0.264-0.267 m from the barrier behind the start, 1 more at 0.31 m.)  The barrier is the prior
    # `wall` entry (cross_from / cross_to), not a `walls` line or the fence: the keep-out never saw it, nor the
    # boulders.  Now the keep-out of every observation / nudge / sidestep / return / backoff target and of the carrot
    # pull-back counts every prior obstacle: the fence, the barrier (a plane across the cross_from -> cross_to line,
    # wall_keep_to_m from cross_to (<= half way), only within wall_keep_half_m along it -- the sim priors cross along
    # the pad -> launch line, ~15 deg off the wall normal -- on the side the vehicle is on; nothing while climbing
    # over it), the boulders, prior wall lines, and the never-seen space behind the take-off point (start heading,
    # within start_rear_half_m sideways, takeoff / corridor phases; the own track covers trace_free_m of it).  Prior
    # positions can be 0.3-0.6 m off and the take-off point itself may be closer than move_clear_m (w1: 0.355 m), so
    # for these the rule is: keep move_clear_m, or -- already closer -- get no closer (tan 10 deg of the move length
    # of slack: sideways is sideways); prior points the camera saw free (or our own track) do not count there.
    # Returns onto the path, pull-backs and backoffs may go back onto our own track.  Measured map points stay a
    # hard move_clear_m.  (v4F eyeball s2: the re-confirm sidestep along a diagonal first leg went 0.11 m back.
    # Hovering ON the take-off point -- map s1, eyeball s1 / s4: 0.25-0.27 m after 0.09-0.10 m of drift -- is not a
    # move; no keep-out can fix that.)
    prior_keepout: bool = True
    # (v4H: eyeball s1 moved back right after the start.)  Final clamp on every commanded position in the takeoff /
    # corridor phases: never more than start_back_max_m behind the take-off point (start heading) while within
    # start_rear_half_m sideways of it -- whatever move produced it.  Returns / pull-backs / backoffs "onto our own
    # track" only count within on_track_m of where the vehicle actually was.
    start_back_max_m: float = 0.05
    # (v4H map s4: the hover point had moved 0.2 m forward, the stop-distance pull-back took it 0.09 m back, VIO
    # error 0.18 m did the rest: 0.25 m from the barrier.)  Until the vehicle first gets start_zone_m away from the
    # take-off point, commanded positions in that lane never go back below the most forward one so far (up to
    # start_ratchet_m): forward progress off the barrier is kept.
    start_ratchet_m: float = 0.3
    start_zone_m: float = 1.0
    # (v4J, coordinator's call) Start advance: the venue launch spot is ~0.46 m in front of the barrier; with 0.28 m
    # reach + 0.1-0.2 m of VIO drift a hover on the take-off point is unsafe (v4H eyeball s1: hovered there, drifted
    # 0.14 m back, hit the barrier).  The first observation hover in the start zone moves forward to f = min(
    # start_advance_m, nearest map point ahead - start_advance_clear_m, seen-free path - start_advance_clear_m) at
    # <= hold_advance_v -- closer to a kept cluster than the camera-distance rule allows -- unless something
    # confirmed (a big cluster, or one re-confirmed real) is within start_confirmed_m ahead.
    start_advance_m: float = 0.35
    # ... and a kept cluster the advance brings within start_sticky_release_m of the camera (its stereo blind zone:
    # the map forgets it in 4 s -- a real board end 0.67 m ahead was hit without this) stays an obstacle in every
    # probe until it is seen absent again from >= start_sticky_release_m (events start_sticky / _blocked / _cleared).
    start_sticky_release_m: float = 0.5
    start_advance_clear_m: float = 0.35
    start_confirmed_m: float = 0.7
    hold_advance_v: float = 0.2
    on_track_m: float = 0.08
    wall_keep_to_m: float = 0.35
    wall_keep_half_m: float = 0.6
    start_rear_keep: bool = True
    start_rear_half_m: float = 0.8
    reconfirm_scan_deg: float = 80.0   # (+-60 is not enough: the camera sits 0.116 m ahead of the body, so the airframe
                                       # edge abreast is ~98 deg off the original heading)
    reconfirm_step_m: float = 0.3
    reconfirm_min_step_m: float = 0.15   # (v4F: 0.2 -> 0.15 -- with move_clear_m the 1.10 m start lane leaves
                                         # ~0.18 m toward W1's face; the largest step that fits is taken)
    reconfirm_watch_s: float = 4.5
    ignore_slow_dist_m: float = 1.0    # stay at v_minor while within this of an ignored cluster
    corridor_side_max_m: float = 0.6   # corridor side steps never exceed this (no routing into the next bay)
    # serpentine maze (stage 1 = 3 lanes, two 180 deg U-turns through ~1.1 m gaps): a corridor vertex where the path
    # reverses by >= uturn_min_deg and the next leg is offset sideways by >= uturn_min_shift_m gets a crossing waypoint
    # at the lane end (lane -> perpendicular crossing -> next lane), so no leg cuts diagonally through a board
    uturn_min_deg: float = 120.0
    uturn_min_shift_m: float = 0.4
    side_unseen_max_m: float = 0.3     # lateral side steps beyond this need the whole new path seen recently...
    # "move only into seen-free space" (props hit a board end while nudging / hovering sideways): any translation
    # sideways or backward needs the swept region (incl. the airframe edge) seen free in the last seen_free_s
    # (camera FOV cone), or the vehicle's own track (trace_free_m, last trace_s), or -- in the stereo blind zone
    # < seen_near_m from the camera -- within the map memory seen_near_memory_s; else look first, then move
    seen_free_s: float = 2.0
    seen_near_m: float = 0.5
    seen_near_memory_s: float = 4.0
    trace_free_m: float = 0.2
    trace_s: float = 8.0
    seen_check_ahead_m: float = 0.6    # "new path" = from abreast of the vehicle to this far ahead on the offset path;
                                       # otherwise hold and yaw to look at it first (further ahead is re-probed while
                                       # flying forward with the camera on the path)
    seen_memory_s: float = 3.0         # camera view history used for "seen recently"
    seen_range_m: float = 3.5
    seen_min_range_m: float = 0.4
    seen_half_fov_deg: float = 33.0    # OAK-D left: +-37 deg horizontal, keep a margin
    seen_half_vfov_deg: float = 23.0   # +-25 deg vertical
    cam_forward_m: float = 0.116       # camera ahead of base_link
    map_max_age_s: float = 1.0
    # arrival / settle
    arrive_tol_m: float = 0.25
    pass_tol_m: float = 0.15
    run_out_m: float = 1.0             # carrot keeps going past a pass-through goal
    settle_tol_m: float = 0.10
    settle_speed: float = 0.20
    settle_yaw_deg: float = 8.0
    # detection latch (N consecutive fresh, consistent frames)
    detect_n: int = 5
    detect_max_age_s: float = 0.5
    # slow detectors (round B: new perception/tunnel.py 0.7-1.1 s per frame, one result every ~2 s, stamped at the
    # image -> every frame was older than detect_max_age_s on arrival and silently dropped: no latch, timeout).
    # Hover-gated frames (tunnel / window) may be this old if the vehicle hovered (speed, zone, yaw) the whole time
    # since the image; the latch then needs fewer frames, spanning >= detect_span_s, never fewer than detect_n_min.
    detect_max_latency_s: float = 2.5
    detect_span_s: float = 3.0
    detect_n_min: int = 3
    detect_min_conf: float = 0.5
    landing_min_conf: float = 0.35     # stereo-image runs: correct rock hits scored 0.36-0.53
    landing_window: int = 8            # landing latch: detect_n consistent hits among the last landing_window frames
    tunnel_min_conf: float = 0.4       # replica eval: with a 0.2-0.5 m prior error correct tunnel hits score ~0.5;
                                       # 0.6 kept only 62 % recall at <= 2 m, 0.4 kept 90-100 %
    detect_pos_tol_m: float = 0.15
    detect_ang_tol_deg: float = 12.0
    detect_prior_gate_m: float = 0.5   # accepted distance from the prior = prior tol + this
    detect_prior_ang_deg: float = 35.0
    detect_speed_max: float = 0.35     # tunnel/window frames only count while (nearly) hovering
    detect_yaw_tol_deg: float = 15.0   # ...and facing the stage
    detect_zone_m: float = 0.6         # ...and within this of the standoff (hover) point
    detect_ahead_m: float = 0.1        # ...and no more than this closer to the stage than the standoff
    # takeoff
    takeoff_tol_m: float = 0.20
    min_alt_m: float = 0.35            # lowest traverse height above the start floor
    # tunnel
    tunnel_standoff_m: float = 1.8     # hover/detect this far before the prior entry. Replica eval: <= 2 m 100 %
                                       # detected, 2-3 m 46-77 %, and < ~1 m camera distance the detector reports
                                       # the SECOND hoop (+0.6 m along the axis): with a 0.5 m prior error the
                                       # camera stays >= ~1.1 m away
    tunnel_align_m: float = 0.6        # aligned pre-entry point this far before the entry
    # depth-only tunnel (official sim: a tiled arch, NO brass hoops -> the hoop detector never fires; on the real
    # drone the hoops can be missed too): no latch after this long hovering at the standoff -> the opening's free
    # cross-section is measured in the obstacle map (side walls, arch, floor) just inside the entry, prior axis,
    # traverse as a narrow wall-relative corridor ('tunnel_mode' depth_fallback)
    tunnel_depth_fallback_s: float = 8.0
    tunnel_depth_slab_m: float = 1.0   # cross-section = map points 0..this behind the prior entry plane
    tunnel_depth_min_pts: int = 6      # per side wall
    standoff_wall_clear_m: float = 0.6 # stage standoff points shrink toward the stage (not below standoff_wall_min_m)
    standoff_wall_min_m: float = 0.9   # until >= this from MEASURED walls (official: tunnel standoff vs north wall)
    yaw_end_match: bool = True         # the turn to face the stage at a path end is done on the way in (speed matched)
    yaw_open_side: bool = True         # big yaws (> 100 deg) turn the way whose headings are clear of near walls
    tunnel_exit_m: float = 0.5         # traverse ends this far past the exit
    tunnel_min_radius_m: float = 0.30
    standoff_min_m: float = 1.2        # tunnel/window standoff shrinks (5 cm steps) down to this to fit the fence
    # re-acquire (round 6 eyeball priors: the prior put the target out of view -> cut_by_fov / not_enclosed /
    # no_flat_top until the timeout): after reacquire_n consecutive missed frames while hovering, back off
    reacquire_n: int = 15              # consecutive counted frames without a usable detection
    reacquire_max: int = 2             # attempts per stage
    reacquire_step_m: float = 0.5      # standoff / observe distance added per attempt
    window_standoff_max_m: float = 2.6
    board_min_points: int = 40         # window board estimate from map points around the prior (median + PCA normal)
    board_search_lat_m: float = 1.6    # board half width + eyeball prior error
    board_search_along_m: float = 0.6
    standoff_fence_clear_m: float = 0.5    # hover points and corridor goals keep this from every fence side (the cage
                                           # frame is a real obstacle at the net; > probe radius 0.35 + tracking)
    # commit gate at the align point, then a straight traverse on the latched axis with lateral correction
    commit_lat_tol_m: float = 0.04     # horizontal cross-axis error
    commit_vert_tol_m: float = 0.04    # vertical error
    commit_lat_vel: float = 0.10       # velocity perpendicular to the axis (3-D)
    commit_along_tol_m: float = 0.15   # along-axis distance from the align point
    commit_hold_s: float = 0.5         # all of the above, continuously
    commit_relax_after_s: float = 3.0  # aligning longer than this: tolerances x commit_relax_factor (VIO jitter)
    commit_relax_factor: float = 1.5
    traverse_lat_gain: float = 1.0     # extra velocity feed-forward toward the axis (1/s) during the traverse
    traverse_lat_vmax: float = 0.15
    # after a latch, map points inside the opening are ignored by the align probe (stereo phantoms seen through mesh/hole)
    tunnel_exclude_margin_m: float = 0.10  # exclusion cylinder radius = detected tunnel radius - this
    tunnel_exclude_front_m: float = 0.70   # ...starting this far in front of the entry plane (covers the align point)
    window_exclude_margin_m: float = 0.05  # radius = min(width, height) / 2 - this
    window_exclude_half_len_m: float = 0.30   # behind the board; in front it reaches window_exclude_front_m
    window_exclude_front_m: float = 0.70
    exclude_to_standoff: bool = True   # after a latch the exclusion reaches back to where we hover (round 5b seed 6:
                                       # a view-locked phantom 1.1 m before the tunnel entry blocked the align)
    window_exclude_approach_r_m: float = 0.40   # radius of the window exclusion in front of the board
    # window
    window_standoff_m: float = 1.5
    window_align_m: float = 0.6
    window_exit_m: float = 0.6
    # landing: target = blue H pad (~0.6 m) lying flat on the floor (LandingTarget.top_d = the floor); and home
    observe_dist_m: float = 1.6        # horizontal camera: look at the pad from here (lower bound) before flying over it
    observe_max_depression_deg: float = 20.0   # observe distance >= above_top / tan(this) + landing tol, so the pad
                                               # top stays in the +-25 deg vertical FOV despite the prior error
    observe_dist_max_m: float = 3.0
    cam_half_vfov_down_deg: float = 25.3   # horizontal camera: image bottom edge below the horizon (OAK, 400 px rows)
    pad_half_m: float = 0.30           # half the pad size: the WHOLE pad (near edge) must be inside the image
    observe_view_margin_m: float = 0.05    # -> base_link observe distance >= h_cam / tan(25.3 deg) + 0.30 + 0.05 +
                                           # cam_forward_m 0.116 (1.63 m at 0.55 m camera height)
    observe_bearing_step_deg: float = 15.0  # observe pose search around the prior pad centre
    v_backoff: float = 0.4             # backing off / sideways with the camera kept on the pad (not looking where we go)
    # checkpoint 4 of the official map: Gate B (open gap, no detector) or the window; '' = prior 'checkpoint4', else
    # gate_b when the prior has a gate_b section, else window
    checkpoint4: str = ''
    gate_standoff_m: float = 0.8       # line up this far before the gate centre ...
    gate_exit_m: float = 0.8           # ... and leave it this far beyond
    v_gate: float = 0.6
    gate_side_margin_m: float = 0.10   # side steps inside the gap keep props + this from its edges
    wall_side_margin_m: float = 0.3    # observe poses stay this far on the pad side of the wall prior's mid-plane
    landing_gate_m: float = 1.2        # landing prior gate (tunnel/window stay at tol + detect_prior_gate_m = 1.0)
    observe_above_top_m: float = 0.55  # camera height above the target top (pad: the floor) while observing; floor-
                                       # landing session to confirm for the pad (rock replica eval: 0.4-0.65 m
                                       # works; < 0.35 m the low-texture top is a depth hole, 3/14 found)
    land_settle_tol_m: float = 0.06    # xy settle over the target before descending (pad ~0.6 m square)
    over_up_m: float = 0.5             # hover this high above the target top (pad: floor) before descending
    slow_above_m: float = 0.3          # switch to v_touch this close to the expected surface
    tof_slow_m: float = 0.35           # ...or when the ToF reads less than this
    tof_touch_m: float = 0.15          # ToF "near zero". Real Sky Beta: tof_m = EKF dist_bottom, reads 0.100 m when
                                       # LANDED (skybeta-05 bench) -> contact must be above that
    tof_max_age_s: float = 0.3
    touch_hold_s: float = 0.4          # ToF + PX4 landed must hold this long before disarm
    descend_timeout_s: float = 10.0    # descent without a confirmed touchdown -> abort (PX4 AUTO_LAND)
    push_m: float = 0.25               # descent setpoint stays at most this far below the vehicle
    push_contact_m: float = 0.5        # ...once the ToF reads contact (<= tof_touch_m): press harder
    px4_lndmc_z_vel_max: float = 0.25  # PX4 LNDMC_Z_VEL_MAX; Mission refuses v_touch < 1.2x this
    contact_handoff_s: Optional[float] = 4.0   # ToF contact this long but PX4 still not landed -> send LAND and
                                       # let PX4 finish touchdown + auto-disarm (None = never hand off)
    disarmed_dwell_s: float = 1.0
    # wall
    liftoff_m: float = 0.5             # straight up off the landing target before heading for the wall
    wall_margin_m: float = 0.5         # over-wall height = wall top + this
    wall_measure_max_age_s: float = 1.0
    # return home
    return_up_m: float = 1.0
    return_high: bool = True           # fly home at the higher of cruise and the wall-crossing height, then descend
                                       # over home only (round 6 seed 7: descending while translating clipped an
                                       # edge-on board that stereo cannot see)
    # safety
    max_mission_s: float = 240.0       # since arming
    pose_bad_land_s: float = 1.0
    fence_margin_m: float = 0.4        # setpoints are clipped this far inside the fence
    fence_inset_m: float = 0.25        # vehicle centre this close to the fence (or outside) -> stop + land; never
                                       # less than half_span_m + fence_trip_margin_m (the props reach the net first)
    fence_trip_margin_m: float = 0.0   # (trip at the airframe reach itself: 0.28 m from the net)
    brake_s: float = 0.5
    land_clear_m: float = 0.45         # (v4K: a corridor_blocked abort landed 0.26 m from a cinder block at W1's end)
                                       # a non-emergency abort landing first slides back along the flown path (<= 1 m)
                                       # until this far from prior wall lines / measured boards (A-frame feet, blocks)
    cmd_retry_s: float = 1.0
    pre_stream_s: float = 1.0          # setpoints streamed before asking for Offboard
    wait_pose_ok_s: float = 1.0
    dt_max_s: float = 0.2
    # staged real-drone tests (params_json: {"stop_after": "tunnel_align"}): '' = full mission
    stop_after: str = ''               # one of STOP_AFTER_PHASES; tunnel_align / window_align stop latched + aligned,
                                       # before entering; land_target stops on the rock (already down: just disarm)
    stop_hover_s: float = 5.0          # hold where the phase ended, then land in place and disarm
    stop_land_px4: bool = False        # True: land with PX4 LAND instead of the mission's ToF-gated descent
    # real drone, organizers' skybeta contract (params_json {"pilot_mode": true}). The pilot arms (Altitude) and
    # switches to Position; the flight interface enters Offboard once per arming (Vehicle.offboard_active). No
    # arm/offboard/land/disarm commands exist; setpoints are POSE ONLY (no feed-forward: speed ~ gain x carrot lead),
    # clamped to pilot_min_up_m .. pilot_max_up_m above the cuVSLAM origin (Vehicle.clamp_origin_d). Every landing
    # descends to the clamp and asks the pilot to land ('pilot_land_requested'); a re-arm continues the course.
    pilot_mode: bool = False
    pilot_min_up_m: float = 0.3
    pilot_max_up_m: float = 3.0
    pilot_pos_gain: float = 0.95       # PX4 MPC_XY_P (fleet 0.95; MPC_Z_P 1.0) (1/s): speed ~ gain x setpoint lead
    pilot_max_lead_m: float = 1.5      # the skybeta adapter clips any pose to this far from the vehicle
                                       # (sky_lead_clipped): our carrot never asks for more
    pilot_floor_settle_m: float = 0.08 # within this of the floor clamp and not moving vertically = descent done
    pilot_hold_descend_v: float = 0.10 # pilot_hold: the setpoint only ever goes DOWN, >= this fast, to the clamp
                                       # (real skybeta-05 2026-11-08: PX4 local vs cuVSLAM odom 0.46 m apart, the
                                       # vehicle sat above its setpoint, 'pos + push' ratcheted up -> 2.63 m)
    climb_guard_m: float = 0.5         # pilot_mode: > this above the highest altitude the prior / phase intends and
    climb_guard_v: float = 0.1         # still rising (> this up) -> stop publishing ('climb_guard'): pilot has it
    pilot_rearm_offboard_wait_s: float = 3.0   # armed after a re-arm but no Offboard this long -> pilot owns it
    phase_timeout_s: Dict[str, float] = field(default_factory=lambda: dict(
        arm=10.0, takeoff=8.0, corridor=150.0, tunnel_align=40.0, tunnel_traverse=8.0,
        window_align=40.0, window_traverse=8.0, land_target=60.0, disarm=6.0, rearm=10.0,
        climb_over_wall=30.0, return_home=40.0, land_home=40.0, staged_stop=25.0,
        chicane=90.0, gate_pass=50.0, landing_route=80.0))   # staged_stop: + stop_hover_s
    # (official lab: 0.3 m/s near corners and 0.4 rad/s yaw near walls roughly double the phase times)

    @property
    def radius(self) -> float:
        return self.half_span_m + self.clearance_margin_m

    @property
    def hard_radius(self) -> float:
        return self.half_span_m + self.hard_margin_m


# ------------------------------------------------------------- I/O structures
@dataclass
class Vehicle:
    """What PX4 tells us (node fills it from vehicle_local_position, vehicle_attitude,
    vehicle_status, vehicle_land_detected, distance_sensor)."""
    pos_ned: Any = (0.0, 0.0, 0.0)
    heading: float = 0.0
    vel_ned: Any = (0.0, 0.0, 0.0)
    q_wxyz: Any = (1.0, 0.0, 0.0, 0.0)
    armed: bool = False
    nav_state: int = NAV_MANUAL
    landed: bool = True
    pose_ok: bool = True               # VIO/local position valid (xy_valid & z_valid & VIO healthy)
    tof_m: Optional[float] = None      # downward ToF distance (m); None/NaN = no valid reading
    tof_t: Optional[float] = None      # stamp of tof_m (None = current)
    offboard_active: Optional[bool] = None   # skybeta /skybeta/state; pilot_mode uses it instead of nav_state
    clamp_origin_d: float = 0.0        # NED z of the skybeta pose-clamp origin (= -origin_enu.z of cuVSLAM odom)
    accel: Any = None                  # raw IMU specific force (3, m/s^2, any frame; only |accel| is used) or None
    odom_d: Optional[float] = None     # optional: the vehicle's NED z per cuVSLAM odom (e.g. sky_vio_ekf_diverge);
                                       # pilot_mode logs PX4-local minus odom z at arming (not compensated)


@dataclass
class Perception:
    """Latest output of every perception module with its stamp (same clock as `now`)."""
    tunnel: Any = None
    tunnel_t: Optional[float] = None
    window: Any = None
    window_t: Optional[float] = None
    landing: Any = None
    landing_t: Optional[float] = None
    floor: Any = None
    floor_t: Optional[float] = None
    obstacles: Any = None              # ObstacleMap
    obstacles_t: Optional[float] = None
    wall_top_d: Optional[float] = None  # measured wall top (NED z, down); main session wires depth here
    wall_t: Optional[float] = None


@dataclass(frozen=True)
class Command:
    name: str                          # 'offboard' | 'arm' | 'disarm' | 'land'
    command: int                       # px4_msgs VehicleCommand.command
    param1: float = 0.0
    param2: float = 0.0


def cmd_offboard() -> Command:
    return Command('offboard', VEHICLE_CMD_DO_SET_MODE, 1.0, PX4_CUSTOM_MAIN_MODE_OFFBOARD)


def cmd_arm() -> Command:
    return Command('arm', VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, 0.0)


def cmd_disarm() -> Command:
    return Command('disarm', VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0, 0.0)   # param2=0: never forced


def cmd_land() -> Command:
    return Command('land', VEHICLE_CMD_NAV_LAND)


@dataclass
class Output:
    phase: str
    sub: str = ''
    pos_ned: Optional[np.ndarray] = None   # None -> publish nothing
    vel_ned: Optional[np.ndarray] = None   # velocity feed-forward
    acc_ned: Optional[np.ndarray] = None   # acceleration feed-forward (may be ignored)
    yaw: Optional[float] = None
    commands: List[Command] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)
    hints: Dict[str, Any] = field(default_factory=dict)

    @property
    def offboard(self) -> bool:
        return self.pos_ned is not None


# --------------------------------------------------------------- course prior
def _flu(d, key, need_up=True):
    try:
        f, l = float(d['forward']), float(d['left'])
        u = float(d['up']) if need_up else float(d.get('up', 0.0))
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError('course prior: %s needs numeric forward/left%s' % (key, '/up' if need_up else '')) from e
    return np.array([f, l, u])


@dataclass
class CoursePrior:
    """Coarse stage positions in the START frame (forward/left/up metres).  Angles are
    degrees counter-clockwise from start-forward (90 = left)."""
    fence: Dict[str, float]            # forward_min/max, left_min/max, up_max
    corridor: List[np.ndarray]
    tunnel_entry: np.ndarray           # centre of the entry mouth (f, l, up)
    tunnel_axis_deg: float
    tunnel_length_m: float
    tunnel_tol_m: float
    window_center: np.ndarray
    window_through_deg: float          # direction of travel through the window
    window_tol_m: float
    landing_center: np.ndarray         # (f, l, top_up): centre of the target top (H pad on the floor: top_up 0)
    landing_tol_m: float
    wall_cross_from: np.ndarray        # (f, l, 0): ground point before the wall
    wall_cross_to: np.ndarray          # (f, l, 0): ground point after the wall
    wall_top_up: float
    wall_tol_m: float
    cruise_up_m: float = 1.0
    tolerance_m: float = 0.5
    # optional (official course map): tunnel exit -> chicane waypoints -> checkpoint 4 -> landing route
    chicane: List[np.ndarray] = field(default_factory=list)
    gate_b_center: Optional[np.ndarray] = None
    gate_b_through_deg: float = 0.0
    gate_b_width_m: float = 0.0
    gate_b_standoff_m: Optional[float] = None   # line-up / exit distances (default Params.gate_standoff_m / exit_m)
    gate_b_exit_m: Optional[float] = None
    checkpoint4: str = ''              # 'gate_b' | 'window' | '' (default: gate_b if given)
    landing_route: List[np.ndarray] = field(default_factory=list)          # after either checkpoint
    landing_route_gate_b: List[np.ndarray] = field(default_factory=list)   # gate_b.after (overrides)
    landing_route_window: List[np.ndarray] = field(default_factory=list)   # window.after (overrides)
    walls: List[tuple] = field(default_factory=list)   # optional board lines ((f0, l0), (f1, l1), tol_m): clusters
                                                       # on / near them are never discounted as phantoms
    boulders: List[tuple] = field(default_factory=list)   # optional (f, l, height_m, radius_m): move keep-out only

    @classmethod
    def from_dict(cls, d: dict) -> 'CoursePrior':
        tol = float(d.get('tolerance_m', 0.5))
        fz = d.get('fence') or {}
        try:
            fence = {k: float(fz[k]) for k in ('forward_min', 'forward_max', 'left_min', 'left_max', 'up_max')}
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError('course prior: fence needs forward_min/max, left_min/max, up_max') from e
        if not (fence['forward_min'] < 0.0 < fence['forward_max'] and fence['left_min'] < 0.0 < fence['left_max']
                and fence['up_max'] > 0.5):
            raise ValueError('course prior: the start must lie strictly inside the fence')
        t, w, lt, wl = d.get('tunnel'), d.get('window'), d.get('landing_target'), d.get('wall')
        if not (t and w and lt and wl):
            raise ValueError('course prior: needs tunnel, window, landing_target and wall sections')
        lc = _flu(lt['center'], 'landing_target.center', need_up=False)
        lc[2] = float(lt['top_up'])
        p = cls(
            fence=fence,
            corridor=[_flu(c, 'corridor[%d]' % i) for i, c in enumerate(d.get('corridor', []))],
            tunnel_entry=_flu(t['entry'], 'tunnel.entry'),
            tunnel_axis_deg=float(t['axis_deg']), tunnel_length_m=float(t['length_m']),
            tunnel_tol_m=float(t.get('tol_m', tol)),
            window_center=_flu(w['center'], 'window.center'),
            window_through_deg=float(w['through_deg']), window_tol_m=float(w.get('tol_m', tol)),
            landing_center=lc, landing_tol_m=float(lt.get('tol_m', tol)),
            wall_cross_from=_flu(wl['cross_from'], 'wall.cross_from', need_up=False),
            wall_cross_to=_flu(wl['cross_to'], 'wall.cross_to', need_up=False),
            wall_top_up=float(wl['top_up']), wall_tol_m=float(wl.get('tol_m', tol)),
            cruise_up_m=float(d.get('cruise_up_m', 1.0)), tolerance_m=tol)
        if p.tunnel_length_m <= 0.0 or p.cruise_up_m <= 0.0:
            raise ValueError('course prior: tunnel.length_m and cruise_up_m must be positive')
        cu = p.cruise_up_m

        def pts(lst, name):
            out = []
            for i, c in enumerate(lst or []):
                q = _flu(dict(c, up=c.get('up', cu)), '%s[%d]' % (name, i))
                out.append(q)
            return out
        p.chicane = pts(d.get('chicane'), 'chicane')
        gb = d.get('gate_b')
        if gb:
            p.gate_b_center = _flu(dict(gb['center'], up=gb['center'].get('up', cu)), 'gate_b.center')
            p.gate_b_through_deg = float(gb['through_deg'])
            p.gate_b_width_m = float(gb.get('width_m', 1.0))
            p.gate_b_standoff_m = gb.get('standoff_m')
            p.gate_b_exit_m = gb.get('exit_m')
            p.landing_route_gate_b = pts(gb.get('after'), 'gate_b.after')
        p.landing_route_window = pts(w.get('after'), 'window.after')
        p.landing_route = pts(d.get('landing_route'), 'landing_route')
        cp = str(d.get('checkpoint4', '') or '')
        if cp not in ('', 'gate_b', 'window'):
            raise ValueError("course prior: checkpoint4 must be 'gate_b' or 'window'")
        if cp == 'gate_b' and p.gate_b_center is None:
            raise ValueError('course prior: checkpoint4 gate_b needs a gate_b section')
        p.checkpoint4 = cp
        wtol = float(d.get('walls_tol_m', tol))       # (default: the prior's own tolerance_m)
        for i, wd in enumerate(d.get('walls') or []):
            try:
                a, b = (wd['from'], wd['to']) if isinstance(wd, dict) else (wd[0], wd[1])
                fa = (float(a['forward']), float(a['left'])) if isinstance(a, dict) else (float(a[0]), float(a[1]))
                fb = (float(b['forward']), float(b['left'])) if isinstance(b, dict) else (float(b[0]), float(b[1]))
                wt = float(wd.get('tol_m', wtol)) if isinstance(wd, dict) else wtol
            except (KeyError, TypeError, ValueError, IndexError) as e:
                raise ValueError('course prior: walls[%d] needs from/to {forward, left}' % i) from e
            p.walls.append((fa, fb, wt))
        for i, bd in enumerate(d.get('boulders') or []):
            try:                                      # (defaults: map boulder, 18 in diameter, 0.35 m high)
                p.boulders.append((float(bd['forward']), float(bd['left']), float(bd.get('height_m', 0.35)),
                                   float(bd.get('radius_m', 0.23))))
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                raise ValueError('course prior: boulders[%d] needs forward/left' % i) from e
        return p

    @classmethod
    def load(cls, path: str) -> 'CoursePrior':
        with open(path) as f:
            return cls.from_dict(json.load(f))


class StartFrame:
    """START frame (forward, left, up) <-> PX4 NED, anchored at the arming pose."""

    def __init__(self, home_ned, yaw: float):
        self.o = np.asarray(home_ned, float).reshape(3).copy()
        self.yaw = float(yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        self.fwd = np.array([c, s, 0.0])
        self.left = np.array([s, -c, 0.0])

    def to_ned(self, flu) -> np.ndarray:
        f, l, u = (float(x) for x in flu)
        return self.o + f * self.fwd + l * self.left + np.array([0.0, 0.0, -u])

    def from_ned(self, p) -> np.ndarray:
        d = np.asarray(p, float).reshape(3) - self.o
        return np.array([d @ self.fwd, d @ self.left, -d[2]])

    def direction(self, deg_left: float) -> np.ndarray:
        a = math.radians(deg_left)
        return math.cos(a) * self.fwd + math.sin(a) * self.left


# ---------------------------------------------------------- detection latch
class Latch:
    """N consecutive consistent detections -> one latched estimate.

    A frame is consistent when it lies within the prior gate, its direction is within
    the prior angle, it is within pos/ang tolerance of the running mean, and its `key`
    (e.g. window clearance_ok) equals the buffer's key.  Any inconsistent frame restarts
    the streak from itself; a missing / low-confidence frame clears it."""

    def __init__(self, n, pos_tol, ang_tol_deg, prior=None, gate=None, prior_dir=None, prior_ang_deg=35.0,
                 window=None):
        self.n, self.pos_tol, self.ang_tol = int(n), float(pos_tol), math.radians(ang_tol_deg)
        self.window = max(self.n, int(window or n))   # == n: N consecutive; > n: N of the last `window` frames
        self.frame = 0
        self.prior = None if prior is None else np.asarray(prior, float).reshape(3)
        self.gate = gate
        self.prior_dir = None if prior_dir is None else np.asarray(prior_dir, float)[:2] / np.linalg.norm(np.asarray(prior_dir, float)[:2])
        self.prior_ang = math.radians(prior_ang_deg)
        self.buf: List[tuple] = []
        self.key = None
        self.rejected = 0

    def set_n(self, n):
        """Frames needed (slow detector: fewer frames, same time span).  Consecutive mode stays consecutive."""
        n = max(1, int(n))
        if self.window == self.n:
            self.window = n
        self.n = n
        self.buf = self.buf[-self.n:]

    def miss(self):
        if self.window == self.n:
            self.buf = []
        else:                                         # N-of-M: a miss only ages the buffer
            self.frame += 1
            self.buf = [b for b in self.buf if b[3] > self.frame - self.window]

    def _mean(self):
        pos = np.mean([b[0] for b in self.buf], axis=0)
        dirs = [b[1] for b in self.buf if b[1] is not None]
        d = None
        if dirs:
            d = np.sum(dirs, axis=0)
            d = d / max(np.linalg.norm(d), 1e-9)
        return pos, d

    def feed(self, pos, direction=None, key=None, extra=None) -> bool:
        pos = np.asarray(pos, float).reshape(3)
        if not np.all(np.isfinite(pos)):
            self.miss(); return False
        if self.prior is not None and self.gate is not None and np.linalg.norm(pos - self.prior) > self.gate:
            self.rejected += 1; self.miss(); return False
        d = None
        if direction is not None:
            d = np.asarray(direction, float).reshape(-1)[:2]
            nd = float(np.linalg.norm(d))
            if not math.isfinite(nd) or nd < 1e-6:
                self.miss(); return False
            d = d / nd
            if self.prior_dir is not None:
                if d @ self.prior_dir < 0.0:
                    d = -d                      # axis / normal sign: always along the travel direction
                if math.acos(min(1.0, float(d @ self.prior_dir))) > self.prior_ang:
                    self.rejected += 1; self.miss(); return False
        if self.buf:
            mp, md = self._mean()
            bad = key != self.key or np.linalg.norm(pos - mp) > self.pos_tol
            if not bad and d is not None and md is not None:
                bad = math.acos(max(-1.0, min(1.0, float(d @ md)))) > self.ang_tol
            if bad:
                self.buf = []                         # inconsistent = unstable: always restart
        self.key = key
        self.frame += 1
        self.buf.append((pos, d, dict(extra or {}), self.frame))
        self.buf = [b for b in self.buf if b[3] > self.frame - self.window][-self.n:]
        return self.stable

    @property
    def stable(self) -> bool:
        return len(self.buf) >= self.n

    def value(self):
        pos, d = self._mean()
        keys = set().union(*[b[2].keys() for b in self.buf]) if self.buf else set()
        extra = {}
        for k in keys:
            vals = [float(b[2][k]) for b in self.buf if b[2].get(k) is not None]
            extra[k] = float(np.mean(vals)) if vals else None
        return pos, d, extra


# ------------------------------------------------------- carrot path follower
class Step:
    __slots__ = ('sp', 'status', 'free')

    def __init__(self, sp, status, free):
        self.sp, self.status, self.free = sp, status, free


class _FixedLatch:
    """A latch-like stand-in for an estimate that did not come from the detector (depth-only tunnel)."""

    def __init__(self, pos, direction, extra):
        self._v = (np.asarray(pos, float).copy(), np.asarray(direction, float).copy(), dict(extra))

    def value(self):
        return self._v


class Follower:
    """Carrot on a polyline with a speed profile (accel-limited, corner and end limits,
    leash to the vehicle) giving position + velocity/acceleration feed-forward.

    avoid=None      : no obstacle checks (takeoff, traverse, descent).
    avoid='stop'    : probe ahead and brake for obstacles, never deviate (final alignment).
    avoid='lateral' : probe ahead; if blocked search horizontal offsets perpendicular to the path.
    avoid='up'      : same, offsets straight up only (wall crossing)."""

    def __init__(self, start, goals, pr: Params, v_max, v0=0.0, avoid=None, yaw=None, yaw_end=None,
                 run_out=0.0, tol=None, side_max=None, leash=None, gap_snap=False, next_after=None):
        pts = [np.asarray(start, float).reshape(3).copy()]
        for g in goals:
            g = np.asarray(g, float).reshape(3)
            if np.linalg.norm(g - pts[-1]) > 1e-3:
                pts.append(g.copy())
        n_goal = len(pts)
        if run_out > 0.0 and len(pts) >= 2:
            d = pts[-1] - pts[-2]
            pts.append(pts[-1] + d / np.linalg.norm(d) * run_out)
        if len(pts) == 1:
            pts.append(pts[0].copy())
        self.pr = pr
        self.pass_through = run_out > 0.0
        self.v_max, self.a = float(v_max), pr.a_max
        self.avoid, self.yaw_fixed, self.yaw_end = avoid, yaw, yaw_end
        self._set_path(pts, n_goal)
        self.tol = pr.arrive_tol_m if tol is None else tol
        self.s, self.v = 0.0, min(max(float(v0), 0.0), self.v_max)
        self.off = self.off_tgt = 0.0
        self.blocked_since = None
        self.side_max = pr.side_max_m if side_max is None else float(side_max)
        self.leash_lo, self.leash_hi = (pr.leash_lo_m, pr.leash_hi_m) if leash is None else leash
        self.ignored: List[tuple] = []        # exclusion cylinders around ignored minor clusters
        self.flags: List[tuple] = []          # (kind, data) notices for the mission's event log
        self._episode_t0 = None
        self._nudge_t0 = None
        self._nudged = False
        self._last_lat = None
        self._minor_since = None
        self._corner_hold = False
        self._corner_hold_k = None
        self._tight: Dict[int, float] = {}    # tight corner vertex -> measured space beyond it (m)
        self._settled: set = set()            # tight corners already settled at
        self._settle = None                   # dict(k, t0, phase) while stopped at a tight corner
        self.fence_ahead = None               # (p, unit dir) -> distance to the net (set by the mission)
        self.fence_dist = None                # p -> horizontal distance to the nearest fence face
        self.fence_face = None                # p -> (distance, outward unit normal NED) of the nearest fence face
        self.fence_phantom = None             # (q, direction) -> q looks like a phantom in front of the net ahead
        self.phantom_ok = None                # (centre, radius) -> a cluster that may be discounted (net band only)
        self.floating_ok = None               # (centre, radius) -> a small cluster with no vertical support
        self.seen_free = None                 # q -> q was seen free recently / is on our own track (lateral moves)
        self._nudge_lat = True
        self._nudge_h0 = self._nudge_yaw = None
        self.count_near = None                # (centre, radius) -> map points there (re-confirmation)
        self.hold_score = None                # (q, back unit or None, ignored) -> min(map / prior / rear clearance)
        self.map_dist = None                  # (q, cap, ignored) -> distance to the nearest map point
        self.keep_ok = None                   # (q, ref or [refs]) -> prior keep-out: >= move_clear_m or no closer
        self.on_track = None                  # q -> on our own recent track (known free, already flown there)
        self.start_f = None                   # q -> forward offset from the take-off point while in the start zone
        self._sticky: List[dict] = []         # kept clusters the start advance took into the camera's blind zone
        self._sticky_flagged = False
        self._rc_real: List[np.ndarray] = []  # cluster centres re-confirmed real
        self._pb_flagged = False
        self.float_info = None               # (centre, radius) -> why the vertical-support check said what it said
        self._hold_adv = None                 # dict(s, t0, done): hover point moved forward before an observation
        self._rc = None                       # observe -> sidestep -> re-confirm state
        self._rc_done: List[np.ndarray] = []  # cluster centres already re-confirmed (real or cleared)
        self._ignored_keep: List[np.ndarray] = []   # ignored as re-confirmed phantoms: no revalidation
        self.gap_snap = bool(gap_snap)        # chicane: snap vertices to the CENTRE of the measured gap
        self.next_after = None if next_after is None else np.asarray(next_after, float).reshape(3).copy()
                                              # (the next stage's line-up point: the END vertex is snapped too)
        self._kept_flagged = False
        self._t_revalid = -math.inf
        self.clear_xy = None                  # (M,3) points, z -> (M,) horizontal clearance to mapped boards
        self.boards_xy = None                 # (lo, hi, z) -> (N,2) board points (filtered once per update)
        self._snap_next = {}                  # vertex -> earliest next search after a fruitless one
        self.snap = False                     # centre the vertices on the measured lane (corridor-like paths)
        self.valid_pt = None                  # p -> inside the fence margin
        self._t_snap = -math.inf
        self.P0 = self.P.copy()               # prior vertices (before centring)
        self._clip = None                     # (S, goal) after an end clip (survives re-centring)
        self._vio_hold_since = None
        self.vio_check = True                 # camera-ahead hold / end clip (off for the short observe-pose moves)
        self._look_unseen = None              # heading to look at the unseen path ahead before going there
        self._look_unseen_t0 = None
        self.look_off = None
        self._look_heading = None
        self._t_search = -math.inf
        self._last_normal = None
        self._last_heading = None

    def _set_path(self, pts, n_goal):
        pr = self.pr
        self.P = np.array(pts, float)
        seg = np.diff(self.P, axis=0)
        self.L = np.linalg.norm(seg, axis=1)
        self.D = seg / np.maximum(self.L, 1e-9)[:, None]
        self.cum = np.concatenate([[0.0], np.cumsum(self.L)])
        self.S = float(self.cum[-1])
        self.n_goal = n_goal
        self.S_goal = float(self.cum[n_goal - 1])
        self.goal = self.P[n_goal - 1].copy()
        self.vc, self.corners = [], []
        self.turn = {}
        for i in range(1, len(self.P) - 1):
            c = float(np.clip(self.D[i - 1] @ self.D[i], -1.0, 1.0))
            self.vc.append((float(self.cum[i]), self.v_max * max(pr.corner_min_frac, 0.5 * (1.0 + c))))
            a_, b_ = self.D[i - 1][:2], self.D[i][:2]          # corner = a HORIZONTAL heading change (camera view)
            na, nb = float(np.linalg.norm(a_)), float(np.linalg.norm(b_))
            if i < n_goal - 1 and na > 0.5 and nb > 0.5:
                ch = float(np.clip((a_ @ b_) / (na * nb), -1.0, 1.0))
                ang = math.degrees(math.acos(ch))
                self.turn[i] = ang
                if ang > pr.corner_turn_deg:
                    self.corners.append((float(self.cum[i]), i))

    def _next_corner(self, s, within):
        for sc, i in self.corners:
            if 0.0 <= sc - s <= within:
                return i
        return None

    def _next_stop(self, s):
        """(vertex index, distance) of the next corner or (non pass-through) path end."""
        best = None
        for sc, i in self.corners:
            if sc >= s and (best is None or sc - s < best[1]):
                best = (i, sc - s)
        if not self.pass_through and (best is None or self.S_goal - s < best[1]):
            best = (self.n_goal - 1, max(0.0, self.S_goal - s))
        return best

    def _clip_end(self, reason, dist):
        """The PATH END sits inside the VIO wall standoff: end here.  Corners are never clipped (no diagonal shortcut
        through a board end): there the carrot holds while the look-ahead yaw turns the camera to the next leg, and
        continues to the true corner once the wall is no longer straight ahead of the camera."""
        here = self.ref(self.s)
        original = self.goal.copy()
        self.S = self.S_goal = self.s
        self.goal = here.copy()
        self._clip = (self.s, here.copy())
        self.flags.append(('waypoint_clipped', {'original': original, 'clipped': here, 'reason': reason,
                                                'camera_to_obstacle_m': round(float(dist), 3)}))

    def _along(self, chain, free):
        run = 0.0
        for a_, b_ in zip(chain[:-1], chain[1:]):
            seg = float(np.linalg.norm(b_ - a_))
            if run + seg >= free:
                return a_ + (b_ - a_) * ((free - run) / seg if seg > 1e-9 else 0.0)
            run += seg
        return np.asarray(chain[-1], float)

    def _in_view(self, pos, heading, q):
        if heading is None:
            return True
        pr = self.pr
        cam = pos + pr.cam_forward_m * np.array([math.cos(heading), math.sin(heading), 0.0])
        d = np.asarray(q, float) - cam
        if math.hypot(d[0], d[1]) < 0.05:
            return True
        return abs(wrap(math.atan2(d[1], d[0]) - heading)) <= math.radians(pr.seen_half_fov_deg)

    # geometry
    def pos(self, s):
        s = min(max(s, 0.0), self.S)
        i = min(int(np.searchsorted(self.cum, s, side='right')) - 1, len(self.L) - 1)
        i = max(i, 0)
        return self.P[i] + self.D[i] * (s - self.cum[i])

    def tangent(self, s, h=0.3):
        t = self.pos(s + h) - self.pos(s - h)
        n = float(np.linalg.norm(t))
        if n < 1e-6:
            i = max(min(int(np.searchsorted(self.cum, s, side='right')) - 1, len(self.L) - 1), 0)
            t, n = self.D[i], float(np.linalg.norm(self.D[i]))
        return t / n if n > 1e-9 else np.zeros(3)

    def normal(self, s):
        if self.avoid == 'up':
            return np.array([0.0, 0.0, -1.0])
        t = self.tangent(s)
        h = math.hypot(t[0], t[1])
        if h < 0.2:
            return self._last_normal if self._last_normal is not None else np.array([0.0, 1.0, 0.0])
        self._last_normal = np.array([-t[1] / h, t[0] / h, 0.0])
        return self._last_normal

    def ref(self, s, off=None):
        off = self.off if off is None else off
        p = self.pos(s)
        return p + off * self.normal(s) if off != 0.0 else p

    def project(self, p):
        p = np.asarray(p, float)
        best_s, best_d = 0.0, math.inf
        for i in range(len(self.L)):
            si = 0.0 if self.L[i] < 1e-9 else float(np.clip((p - self.P[i]) @ self.D[i], 0.0, self.L[i]))
            d = float(np.linalg.norm(p - (self.P[i] + self.D[i] * si)))
            if d < best_d - 1e-9:
                best_d, best_s = d, float(self.cum[i]) + si
        return best_s

    def v_limit(self, s):
        ad = min(self.a, self.pr.a_stop_plan) if self.avoid in ('lateral', 'stop') else self.a   # (planned decel)
        v = min(self.v_max, math.sqrt(2.0 * ad * max(0.0, self.S - s)))
        for sc, vc in self.vc:
            if sc >= s:
                v = min(v, math.sqrt(vc * vc + 2.0 * ad * (sc - s)))
        if self.avoid in ('lateral', 'stop'):     # obstacle-checked paths near walls: slow at corners and the end
                                                  # (not the wall crossing: its waypoints are above the wall)
            pr = self.pr
            vn2 = pr.v_near_waypoint ** 2
            stops = [sc for sc, _ in self.corners if sc >= s]
            if not self.pass_through:
                stops.append(self.S_goal)
            for sc in stops:
                v = min(v, math.sqrt(vn2 + 2.0 * ad * max(0.0, sc - s - pr.near_waypoint_m - pr.near_waypoint_lag_m)))
            for sc, _ in self.corners:                # ... and leaving it: within near_waypoint_m on both sides
                if 0.0 <= s - sc <= pr.near_waypoint_m:
                    v = min(v, pr.v_near_waypoint)
            ac, vc2 = pr.a_corner_stop, pr.v_corner_creep ** 2
            for k in self._tight:                     # tight corner: ramp to a creep, then to 0 ON the corner point
                if k in self._settled:
                    continue
                sc = float(self.cum[k])
                if sc >= s - 1e-6:
                    v = min(v, math.sqrt(2.0 * ac * max(0.0, sc - s)),
                            math.sqrt(vc2 + 2.0 * ac * max(0.0, sc - s - pr.corner_creep_m)))
        return v

    def _tight_stop(self, s):
        """Arc length of the next unsettled tight corner at or after s (None: none)."""
        best = None
        for k in self._tight:
            sc = float(self.cum[k])
            if k not in self._settled and sc >= s - 1e-6 and (best is None or sc < best[0]):
                best = (sc, k)
        return best

    def _update_tight(self, s, ahead):
        """Mark corners whose measured space beyond the corner point (incoming direction) is < tight_corner_space_m."""
        pr = self.pr
        for sc, k in self.corners:
            if k in self._tight or k in self._settled or not (-1e-6 <= sc - s <= 2.5):
                continue
            d = self.D[k - 1].copy()
            d[2] = 0.0
            n = float(np.linalg.norm(d))
            if n < 0.3:
                continue
            sp = ahead(self.P[k], d / n, pr.prop_radius_m, self.ignored) if ahead is not None else None
            if self.fence_ahead is not None:      # the net counts as a wall (depth rarely sees it)
                sf = self.fence_ahead(self.P[k], d / n)
                sp = sf if sp is None else min(sp, sf)
            ang = self.turn.get(k, 0.0)
            if sp is not None and sp < pr.tight_corner_space_m:
                self._tight[k] = float(sp)
                self.flags.append(('tight_corner', {'corner': self.P[k].copy(), 'space_ahead_m': round(float(sp), 3),
                                                    'turn_deg': round(ang, 1)}))
            elif ang >= pr.stop_corner_deg:       # a real (wide) corner: same ramp + stop, overshoot <= 0.1 m
                self._tight[k] = float(sp) if sp is not None and math.isfinite(sp) else math.inf
                self.flags.append(('stop_corner', {'corner': self.P[k].copy(), 'turn_deg': round(ang, 1),
                                                   'space_ahead_m': None if sp is None or not math.isfinite(sp)
                                                   else round(float(sp), 3)}))

    def _off_path_ahead(self, q, s, look=1.5):
        """Horizontal distance from q to the path ahead of s (look metres, plus 0.6 m beyond the end)."""
        ss = np.linspace(s, min(self.S, s + look), 16)
        P = np.array([self.ref(x) for x in ss])
        if s + look > self.S and self.S > 1e-6:
            d = self.D[-1]
            P = np.vstack([P, self.ref(self.S) + np.outer(np.linspace(0.1, 0.6, 6), d)])
        return float(np.min(np.hypot(P[:, 0] - q[0], P[:, 1] - q[1])))

    def _leg_clear(self, prv, v, nxt, z):
        """Smallest board clearance along prv -> v -> nxt (sampled every 0.1 m, the end points excluded)."""
        pts = []
        for a, b in ((prv, v), (v, nxt)):
            L = float(np.linalg.norm((b - a)[:2]))
            n = max(2, int(L / 0.1))
            for u in np.linspace(0.0, 1.0, n + 1)[1:-1]:
                pts.append(a + (b - a) * u)
        pts.append(v)
        B = getattr(self, '_B', None)
        if B is None:
            c = self.clear_xy(np.array(pts), z) if self.clear_xy is not None else None
        else:
            c = _min_dist(np.array(pts)[:, :2], B)
        return float(c.min()) if c is not None and len(c) else math.inf

    def _floating(self, c, r=None):
        """A small cluster at c without vertical support (no floor-standing structure): may be discounted."""
        return self.floating_ok is not None and c is not None and bool(self.floating_ok(c, r))

    def _phantom(self, c, r=None):
        """The cluster at c may be discounted (stereo phantom in front of the net, not structure)."""
        return self.phantom_ok is not None and c is not None and bool(self.phantom_ok(c, r))

    def _toward_net(self, q, d):
        """q is within fence_phantom_m of the fence face we are flying INTO (direction d): stereo phantoms in front
        of the net there never hold / block a corner after the nudge.  (A board beside the lane near a side net
        does not qualify -- the lane centring keeps us off it.)"""
        return self.fence_phantom is not None and self.pr.fence_phantom_m > 0 and bool(self.fence_phantom(q, d))

    @staticmethod
    def _grid_for(r, step=0.05):
        cache = Follower.__dict__.get('_grids')
        if cache is None:
            cache = {}
            Follower._grids = cache
        key = (round(float(r), 3), round(float(step), 3))
        if key not in cache:
            g = np.arange(-r, r + 1e-9, step)
            gx, gy = np.meshgrid(g, g)
            m = np.hypot(gx, gy) <= r + 1e-9
            cache[key] = np.stack([gx[m], gy[m]], axis=1)
        return cache[key]

    def _snap_vertices(self, now, s):
        """Push the vertices ahead (prior points) onto the measured lane centre: >= lane_clear_m from mapped boards,
        inside the fence, moved as little as possible (<= snap_max_m).  Re-run as the map fills in.
        gap_snap (chicane): every vertex, and the END vertex judged against next_after (the next stage's line-up
        point), goes to the CENTRE of the measured gap (largest min clearance to boards / net, <= gap_snap_max_m)
        with both legs clear within gap_leg_m -- round B demo: the line chicane end -> Gate B passed 0.05 m from
        C1's south end."""
        pr = self.pr
        if not self.snap or self.boards_xy is None or now - self._t_snap < pr.snap_period_s:
            return
        if self._rc is not None:
            return                                # (re-confirmation: the camera looks sideways / at the cluster, the
                                                  # map ahead is partial -- an unseen board end looks free)
        self._t_snap = now
        gap = self.gap_snap
        smax = pr.gap_snap_max_m if gap else pr.snap_max_m
        grid = Follower._grid_for(smax)
        changed = False
        ks = list(range(1, self.n_goal - 1))       # interior vertices (the end is a stage standoff) ...
        if self.next_after is not None and self._clip is None and self.n_goal >= 2:
            ks.append(self.n_goal - 1)             # ... unless the next leg is known (chicane end -> line-up)
        for k in ks:
            end = k == self.n_goal - 1
            ds = float(self.cum[k]) - s
            frozen = 0.4 if (end or k in self._tight or self.turn.get(k, 0.0) >= pr.stop_corner_deg) else 0.05
            if ds < frozen or ds > pr.snap_range_m or k in self._settled or (self._settle and self._settle['k'] == k):
                continue                          # (a stop corner stays put while the carrot creeps into it)
            if now < self._snap_next.get(k, -math.inf):
                continue
            base = self.P0[k]
            prv, nxt = self.P[k - 1], (self.next_after if end else self.P[k + 1])
            box = np.vstack([prv[:2], nxt[:2], base[:2] - smax, base[:2] + smax])
            B = self.boards_xy(box.min(axis=0) - 0.8, box.max(axis=0) + 0.8, float(base[2]))
            if B is None:
                continue
            self._B = B
            if gap:                               # wall-relative (corridor / maze / chicane): the gap centre
                changed |= self._snap_gap_vertex(k, now, ds, B, prv, nxt, base, end)
                continue
            cur_clr = _min_dist(self.P[k][None, :2], B)
            corner = (not gap) and self.turn.get(k, 0.0) >= pr.stop_corner_deg

            def legs(C, corner=corner, prv=prv, nxt=nxt, B=B):
                """Leg clearance near an L corner (the last / first metre of the legs): turning around a board end
                (seed 6 v4: the next leg passed W1's end at 0.31 m).  Straight-through vertices: the side step
                handles obstacles ON a leg."""
                if not corner:
                    return np.full(len(C), math.inf)
                la = np.maximum(np.hypot(*(C - prv[:2]).T), 1e-6)
                lb = np.maximum(np.hypot(*(C - nxt[:2]).T), 1e-6)
                da = _seg_min_dist(prv, C, B, trim_a=float(max(0.15, la.min() - 1.0)))
                db = _seg_min_dist(nxt, C, B, trim_a=float(max(0.15, lb.min() - 1.0)))
                return np.minimum(da, db)

            def full_legs(C, prv=prv, nxt=nxt, B=B, end=end):
                """Chicane: whole legs (a move must never make one worse -- no jumping across a board; legs still
                too close get vias in _gap_vias)."""
                da = _seg_min_dist(prv, C, B, trim_a=0.15)
                db = _seg_min_dist(nxt, C, B, trim_a=0.25 if end else 0.15)
                return np.minimum(da, db)
            leg_now = float(legs(self.P[k][None, :2])[0])
            now_min = min(float(cur_clr[0]), leg_now)
            want = pr.corner_clear_target_m if corner else pr.lane_clear_m   # stop corners: 0.6 m if it fits
            fd_now = math.inf                     # (measured walls only: the net is the fence margin's business)
            fd_k = float(self.fence_dist(self.P[k])) if self.fence_dist is not None else math.inf
            if corner:
                fd_now = fd_k                     # (v4K) a corner also wants the net at corner_clear_target_m
            if gap:                               # chicane: re-centre (vias too) while the gap fills in on the map
                want = pr.gap_clear_cap_m
            if now_min >= pr.lane_clear_m and (not (corner or gap) or float(cur_clr[0]) >= want) and \
                    (not corner or fd_k >= min(want, pr.vertex_fence_clear_m) - 0.01):
                continue                          # sticky: never drift back toward the prior when the map forgets
            c = self.P[k][None, :] + np.c_[grid, np.zeros(len(grid))]
            within = np.linalg.norm((c - base)[:, :2], axis=1) <= smax + 1e-9
            c = c[within]
            clr = _min_dist(c[:, :2], B)
            a0, b0 = (base - prv)[:2], (nxt - base)[:2]
            a1, b1 = (c - prv)[:, :2], (nxt - c)[:, :2]
            la, lb = np.linalg.norm(a1, axis=1), np.linalg.norm(b1, axis=1)
            na, nb = float(np.linalg.norm(a0)), float(np.linalg.norm(b0))
            if na < 1e-3 or nb < 1e-3:
                continue
            t0 = math.degrees(math.acos(float(np.clip((a0 @ b0) / (na * nb), -1.0, 1.0))))
            t1 = np.degrees(np.arccos(np.clip(np.sum(a1 * b1, axis=1) / np.maximum(la * lb, 1e-9), -1.0, 1.0)))
            keep = ((a1 @ a0) > 0) & ((b1 @ b0) > 0) & (la >= min(0.3, 0.8 * na)) & (lb >= min(0.3, 0.8 * nb)) & \
                (t1 <= t0 + 20.0)                 # never reverse / collapse a leg or make a new sharp corner
            if self.valid_pt is not None:
                keep &= np.array([bool(self.valid_pt(q)) for q in c])
            if self.seen_free is not None and pr.vertex_seen_only and keep.any() and not len(B):   # (v4K) nothing
                keep[keep] = [bool(self.seen_free(q)) for q in c[keep]]                         # measured: seen only
            fd_c = np.full(len(c), math.inf)
            if self.fence_dist is not None and np.isfinite(fd_k):
                fd_c[keep] = [float(self.fence_dist(q)) for q in c[keep]]
                keep &= fd_c >= min(pr.vertex_fence_clear_m, fd_k) - 0.01   # (v4K) never toward the net
            if not keep.any():
                continue
            sh = np.hypot(*(c - self.P[k])[:, :2].T)
            lg = np.full(len(c), -np.inf)
            lg[keep] = legs(c[keep][:, :2])
            need = pr.lane_clear_m if leg_now < pr.lane_clear_m else min(pr.lane_clear_m, leg_now) - 0.02
            ok = keep & (clr >= pr.lane_clear_m) & (lg >= need)
            if gap and ok.any():
                fl_now = float(full_legs(self.P[k][None, :2])[0])
                fl = np.full(len(c), -np.inf)
                fl[ok] = full_legs(c[ok][:, :2])
                ok &= fl >= min(fl_now, pr.lane_clear_m) - 0.02
            if ok.any() and gap:                  # the centre of the gap: largest min clearance (boards, legs, net)
                fd = np.full(len(c), math.inf)
                if self.fence_dist is not None:
                    fd[ok] = [float(self.fence_dist(q)) for q in c[ok]]
                score = np.minimum(np.minimum(clr, lg), np.minimum(fd, pr.gap_clear_cap_m)) - 0.05 * sh
                j = int(np.argmax(np.where(ok, score, -np.inf)))
            elif ok.any() and corner and min(float(cur_clr[0]), fd_k) < want:   # stop corner: corner_clear_target_m from
                sc = np.minimum(np.minimum(clr, want), fd_c)   # the measured walls and the net if the corridor allows
                score = sc - 0.05 * sh
                okc = ok & (lg >= min(pr.lane_clear_m, leg_now))   # (never trade leg clearance for it: no flip-flop)
                if not okc.any():
                    self._snap_next[k] = now + 1.0
                    continue
                j = int(np.argmax(np.where(okc, score, -np.inf)))
                if now_min >= pr.lane_clear_m and sc[j] < min(float(cur_clr[0]), fd_now, want) + 0.03:
                    self._snap_next[k] = now + 1.0
                    continue                      # no better place: stay
            elif ok.any():                        # the smallest move that clears the vertex AND both legs
                j = int(np.argmin(np.where(ok, sh - 1e-3 * np.minimum(clr, 2.0), np.inf)))
            else:                                 # cannot reach lane_clear_m everywhere: the best min clearance
                score = np.where(keep, np.minimum(clr, lg), -np.inf)
                j = int(np.argmax(score))
                if score[j] < now_min + 0.05:
                    self._snap_next[k] = now + 1.0
                    continue                      # nothing better within reach: leave it (look again in 1 s)
            new = c[j]
            mv = new - self.P[k]
            nm = float(np.linalg.norm(mv))
            if nm > 0.1 and ds < 1.0:             # close ahead (seen late): migrate smoothly, <= 0.1 m per update
                new = self.P[k] + mv * (0.1 / nm)
            if float(np.linalg.norm(new - self.P[k])) > 0.03:
                before = float(cur_clr[0])
                self.flags.append(('waypoint_centred', {'index': k, 'prior': base.copy(), 'from': self.P[k].copy(),
                                                        'to': new.copy(), 'clearance_before_m': round(before, 3),
                                                        'clearance_m': round(float(_min_dist(new[None, :2], B)[0]), 3),
                                                        'legs_m': round(float(legs(new[None, :2])[0]), 3),
                                                        'gap': gap, 'end': end}))
                self.P[k] = new
                self._tight.pop(k, None)          # re-judge the corner at its new place
                changed = True
        if changed:
            s_keep = self.s
            self._set_path(self.P, self.n_goal)
            if self._clip is not None:
                self.S = self.S_goal = min(self.S, self._clip[0])
                self.goal = self._clip[1].copy()
            self.s = min(s_keep, self.S)
        if gap:
            self._gap_vias(now, s)

    def _snap_gap_vertex(self, k, now, ds, B, prv, nxt, base, end):
        """Wall-relative vertex: move it (<= gap_snap_max_m from the prior) to the point with the largest clearance
        min(boards, net, gap_clear_cap_m): the centre between the MEASURED walls of the lane, and along the lane the
        centre of the turn gap (board end <-> front wall).  No leg may get worse (never jump across a board), legs
        stay <= 1.5 x their prior length, no reversal / new sharp corner.  Coarse 0.1 m then fine 0.05 m search;
        moves only for >= 0.03 m more clearance (sticky).  Legs still too close get vias (_gap_vias)."""
        pr = self.pr
        cap, smax = pr.gap_clear_cap_m, pr.gap_snap_max_m
        cur = self.P[k]
        L0a = float(np.linalg.norm((self.P0[k] - self.P0[k - 1])[:2]))
        L0b = float(np.linalg.norm(((self.next_after if end else self.P0[k + 1]) - self.P0[k])[:2]))
        a0, b0 = (base - prv)[:2], (nxt - base)[:2]
        na, nb = float(np.linalg.norm(a0)), float(np.linalg.norm(b0))
        if na < 1e-3 or nb < 1e-3:
            return False
        t0 = math.degrees(math.acos(float(np.clip((a0 @ b0) / (na * nb), -1.0, 1.0))))
        tc = 0.25 if end else 0.15

        def fdist(C):
            if self.fence_dist is None:
                return np.full(len(C), math.inf)
            return np.array([float(self.fence_dist(q)) for q in C])


        def legs(C):                              # (non-worsening: every leg, the first one too)
            return np.minimum(_seg_min_dist(prv, C[:, :2], B, trim_a=0.15), _seg_min_dist(nxt, C[:, :2], B, trim_a=tc))

        cur2 = cur[None, :2]                      # a leg with something ON its line now (< 0.10 m) is the side
        thru_a = float(_seg_min_dist(prv, cur2, B, trim_a=0.15)[0]) < 0.10   # step's business, not the vertex's: out of
        thru_b = float(_seg_min_dist(nxt, cur2, B, trim_a=tc)[0]) < 0.10     # the objective (tight legs stay in it)

        def legs_near(C):                         # the legs within gap_leg_m of the vertex (turning round an end)
            out = np.full(len(C), math.inf)
            if not thru_a:
                la = np.hypot(*(C[:, :2] - prv[:2]).T)
                out = np.minimum(out, _seg_min_dist(prv, C[:, :2], B, trim_a=float(max(0.15, la.min() - pr.gap_leg_m))))
            if not thru_b:
                lb = np.hypot(*(C[:, :2] - nxt[:2]).T)
                out = np.minimum(out, _seg_min_dist(nxt, C[:, :2], B, trim_a=float(max(tc, lb.min() - pr.gap_leg_m))))
            return out
        fd_cur = float(fdist(cur[None, :])[0])
        cur_raw = float(min(_min_dist(cur[None, :2], B)[0], fd_cur, legs_near(cur[None, :])[0], cap))
        fl_now = float(legs(cur[None, :])[0])
        if cur_raw >= cap - 1e-6:
            return False                          # plenty of room: leave the prior point alone
        need = min(fl_now, pr.lane_clear_m) - 0.02

        def evaluate(C):
            a1, b1 = (C - prv)[:, :2], (nxt - C)[:, :2]
            la, lb = np.linalg.norm(a1, axis=1), np.linalg.norm(b1, axis=1)
            t1 = np.degrees(np.arccos(np.clip(np.sum(a1 * b1, axis=1) / np.maximum(la * lb, 1e-9), -1.0, 1.0)))
            keep = ((a1 @ a0) > 0) & ((b1 @ b0) > 0) & (la >= min(0.3, 0.8 * na)) & (lb >= min(0.3, 0.8 * nb)) & \
                (t1 <= t0 + 30.0) & (la <= 1.5 * L0a + 0.3) & (lb <= 1.5 * L0b + 0.3) & \
                (np.linalg.norm((C - base)[:, :2], axis=1) <= smax + 1e-9)
            if self.valid_pt is not None and keep.any():
                keep[keep] = [bool(self.valid_pt(q)) for q in C[keep]]
            if keep.any() and self.seen_free is not None and pr.vertex_seen_only and not len(B):   # (v4K) nothing
                keep[keep] = [bool(self.seen_free(q)) for q in C[keep]]   # measured: only the net pushes -> only into
                                                                          # space seen free (eyeball s2: off the net
                                                                          # into the unseen W1)
            fdk = np.full(len(C), math.inf)
            if keep.any() and np.isfinite(fd_cur):     # (v4K) never toward the net: >= vertex_fence_clear_m
                fdk[keep] = fdist(C[keep])              # or no closer than the vertex already is
                keep &= fdk >= min(pr.vertex_fence_clear_m, fd_cur) - 0.01
            raw = np.full(len(C), -np.inf)
            if keep.any():
                ck = C[keep]
                fl = legs(ck)
                r = np.minimum(np.minimum(np.minimum(_min_dist(ck[:, :2], B), fdk[keep]), legs_near(ck)), cap)
                r[fl < need] = -np.inf             # a leg would get worse: not this one
                raw[keep] = r
            return raw
        g1 = Follower._grid_for(smax, 0.1)
        C1 = cur[None, :] + np.c_[g1, np.zeros(len(g1))]
        r1 = evaluate(C1)
        if not np.isfinite(r1).any():
            self._snap_next[k] = now + 1.0
            return False
        sh1 = np.hypot(*(C1 - cur)[:, :2].T)
        j = int(np.argmax(r1 - 0.05 * sh1))
        g2 = Follower._grid_for(0.1, 0.05)
        C2 = C1[j][None, :] + np.c_[g2, np.zeros(len(g2))]
        r2 = evaluate(C2)
        sh2 = np.hypot(*(C2 - cur)[:, :2].T)
        j2 = int(np.argmax(r2 - 0.05 * sh2))
        best, braw = (C2[j2], float(r2[j2])) if r2[j2] - 0.05 * sh2[j2] >= r1[j] - 0.05 * sh1[j] else (C1[j], float(r1[j]))
        if braw < cur_raw + 0.03:
            self._snap_next[k] = now + 1.0
            return False
        new = best.copy()
        mv = new - cur
        nm = float(np.linalg.norm(mv))
        if nm > 0.1 and ds < 1.0:                 # close ahead (seen late): migrate smoothly
            new = cur + mv * (0.1 / nm)
        if float(np.linalg.norm(new - cur)) <= 0.03:
            return False
        self.flags.append(('waypoint_centred', {'index': k, 'prior': base.copy(), 'from': cur.copy(), 'to': new.copy(),
                                                'clearance_before_m': round(cur_raw, 3), 'clearance_m': round(braw, 3),
                                                'legs_m': round(float(legs(new[None, :])[0]), 3), 'gap': True,
                                                'end': end}))
        self.P[k] = new
        self._tight.pop(k, None)
        return True

    def _gap_vias(self, now, s):
        """Chicane legs ahead (and the virtual leg END -> next_after) that pass a mapped board closer than
        lane_clear_m get via points: q = the board point nearest the leg, w = from q toward the leg; the via goes to
        q + w * R, R = half the gap to the next board along w (the GAP CENTRE, e.g. C1's south end <-> C2) or
        lane_clear_m + 0.05 when open -- repeated on the new legs, so a leg grazing a board end is rounded around it."""
        pr = self.pr
        if self._clip is not None or self.pass_through:
            return
        k0 = int(np.searchsorted(self.cum, s, side='right')) - 1
        R_open = pr.lane_clear_m + 0.05
        added = 0
        k = max(1, k0 + 1)
        while k <= self.n_goal - 1 and added < 4:
            virtual = k == self.n_goal - 1
            if virtual and self.next_after is None:
                break
            a = self.P[k]
            b = self.next_after if virtual else self.P[k + 1]
            if float(self.cum[k]) - s > pr.snap_range_m:
                break
            lo = np.minimum(a[:2], b[:2]) - 1.6
            hi = np.maximum(a[:2], b[:2]) + 1.6
            B = self.boards_xy(lo, hi, float(a[2]))
            if B is None or not len(B):
                k += 1
                continue
            trim_c = 0.25 if virtual else 0.0     # (the line-up point itself: the next stage's business)
            d_old = float(_seg_min_dist(a, b[None, :2], B, trim_c=trim_c)[0])
            if d_old >= pr.lane_clear_m:
                k += 1
                continue
            ab = (b - a)[:2]
            L = float(np.linalg.norm(ab))
            if L < 0.3:
                k += 1
                continue
            u = ab / L
            rel = B - a[:2]
            t = np.clip(rel @ u, 0.0, L - trim_c)
            near = a[:2] + t[:, None] * u[None, :]
            dist = np.hypot(*(near - B).T)
            i = int(np.argmin(dist))
            q, cp = B[i], near[i]
            w = cp - q
            if float(np.linalg.norm(w)) < 1e-3:   # the leg runs through it: go round on the roomier side
                n_ = np.array([-u[1], u[0]])
                side = [float(_min_dist((q + sd * R_open * n_)[None, :], B)[0]) for sd in (1.0, -1.0)]
                w = n_ if side[0] >= side[1] else -n_
            w = w / float(np.linalg.norm(w))
            r_ = B - q
            tw = r_ @ w
            lat = np.abs(r_ @ np.array([-w[1], w[0]]))
            opp = (tw > 0.15) & (tw < 2.0 * pr.gap_snap_max_m) & (lat < 0.15)
            R = 0.5 * float(tw[opp].min()) if opp.any() else R_open   # gap centre (C1 end <-> C2), else round it
            z = float(a[2] + (b[2] - a[2]) * (float(t[i]) / L))
            via = np.array([q[0] + w[0] * R, q[1] + w[1] * R, z])
            if float(np.linalg.norm((via - a)[:2])) < 0.2 or float(np.linalg.norm((via - b)[:2])) < 0.2 or \
                    (self.valid_pt is not None and not self.valid_pt(via)):
                k += 1
                continue
            d1 = float(_seg_min_dist(a, via[None, :2], B)[0])
            d2 = float(_seg_min_dist(via, b[None, :2], B, trim_c=trim_c)[0])
            if min(d1, d2) < d_old + 0.03:
                k += 1                            # no better: leave it (the probe / side step handle it)
                continue
            self.P = np.insert(self.P, k + 1, via, axis=0)
            self.P0 = np.insert(self.P0, k + 1, via, axis=0)
            self.n_goal += 1
            sh = lambda dct: {(j + 1 if j > k else j): v for j, v in dct.items()}   # noqa: E731
            self._tight, self._snap_next = sh(self._tight), sh(self._snap_next)
            self._settled = {(j + 1 if j > k else j) for j in self._settled}
            if self._settle and self._settle['k'] > k:
                self._settle['k'] += 1
            if self._corner_hold_k is not None and self._corner_hold_k > k:
                self._corner_hold_k += 1
            self.flags.append(('gap_via', {'after': a.copy(), 'via': via.copy(), 'board_pt': np.r_[q, z],
                                           'leg_clear_before_m': round(d_old, 3),
                                           'leg_clear_m': round(min(d1, d2), 3), 'radius_m': round(R, 3),
                                           'gap_centre': bool(opp.any()), 'end': virtual}))
            s_keep = self.s
            self._set_path(self.P, self.n_goal)
            self.s = min(s_keep, self.S)
            added += 1                            # re-check the first new leg (k -> via) next

    def _pull_back(self, s_target, pos, vel, dt):
        """Move the carrot back toward s_target -- gradually (<= carrot_back_v) and never behind the vehicle while it
        still moves forward (official run_002/003: setpoints jumped behind the moving vehicle -> overshoot and a slow
        hold oscillation into the wall)."""
        pr = self.pr
        sp_ = self.project(pos)
        s_target = max(s_target, self._s_floor())  # (never back toward the unseen rear past a safe hover point)
        if vel is not None:
            va = float(np.asarray(vel, float)[:2] @ self.tangent(sp_)[:2])
            if va > 0.03:
                s_target = max(s_target, sp_)
        if s_target < self.s - 1e-9 and self.keep_ok is not None:
            refs = [np.asarray(pos, float), self.ref(self.s, self.off)]   # (v4G) never back into new space closer
            x = s_ok = self.s                                               # to a prior obstacle / the never-seen
            while x > s_target + 1e-9:                                      # rear (our own track: allowed)
                x = max(s_target, x - 0.05)
                q = self.ref(x, self.off)
                if not ((self.on_track is not None and self.on_track(q)) or self.keep_ok(q, refs)):
                    break
                s_ok = x
            if s_ok > s_target + 0.02 and not self._pb_flagged:
                self._pb_flagged = True
                self.flags.append(('pull_back_limited', {'wanted_m': round(self.s - s_target, 3),
                                                         'kept_m': round(self.s - s_ok, 3),
                                                         'carrot': self.ref(s_ok, self.off).copy()}))
            s_target = s_ok
        if s_target < self.s:
            self.s = max(s_target, self.s - pr.carrot_back_v * max(dt, 0.0))
        self.v = 0.0

    def _early_corner(self, s):
        """(k, distance, next-leg heading) of the next unsettled stop corner within corner_yaw_ahead_m."""
        pr = self.pr
        if not pr.corner_yaw_early or self.avoid not in ('lateral', 'stop') or self.yaw_fixed is not None:
            return None
        best = None
        for k in self._tight:
            if k in self._settled or self.turn.get(k, 0.0) < pr.stop_corner_deg or k >= len(self.D):
                continue
            dist = float(self.cum[k]) - s
            if -1e-6 <= dist <= pr.corner_yaw_ahead_m and (best is None or dist < best[1]):
                sc_ = getattr(self, '_seen_cb', None)
                din = self.D[k - 1]
                if sc_ is not None and dist > 0.15 and math.hypot(din[0], din[1]) > 0.3 and \
                        not sc_(self.P[k] + 0.3 * din):
                    continue                      # first SEE the corner (and what is beyond it), then turn
                dk = self.D[k]
                if math.hypot(dk[0], dk[1]) > 0.3:
                    best = (k, max(0.0, dist), math.atan2(dk[1], dk[0]))
        return best

    def _yaw_end_eff(self):
        """yaw_end, or (next_after known) the heading of the next leg: look where we will go next."""
        if self.next_after is not None:
            d = (self.next_after - self.goal)[:2]
            if float(np.linalg.norm(d)) > 0.3:
                return math.atan2(d[1], d[0])
        return self.yaw_end

    def heading_cmd(self):
        if self._rc is not None and self._rc.get('yaw') is not None:
            return self._rc['yaw']                    # re-confirmation: look at a side / at the cluster
        if self._nudge_t0 is not None and self._nudge_yaw is not None:
            return self._nudge_yaw                    # observation manoeuvre: small yaw scan in place
        if self._look_unseen is not None:
            return self._look_unseen                  # look at the unseen path ahead before flying into it
        if self.yaw_fixed is not None:
            return self.yaw_fixed
        if self.look_off is not None and self._look_heading is not None:
            return self._look_heading                 # look at the side before stepping into it
        if self._look_unseen is not None:
            return self._look_unseen                  # look at the unseen path ahead before flying into it
        if self._settle is not None:              # stopped on a tight corner: face the next leg, then go
            dk = self.D[self._settle['k']]
            if math.hypot(dk[0], dk[1]) > 0.3:
                self._last_heading = math.atan2(dk[1], dk[0])
                return self._last_heading
        ye = self._yaw_end_eff()
        if ye is not None and self.S_goal - self.s <= self.pr.yaw_end_dist_m:
            return ye
        ec = self._early_corner(self.s)
        if ec is not None:                        # stop corner ahead: already turn to the next leg (slowly)
            self._last_heading = ec[2]
            return ec[2]
        if self._corner_hold and self._corner_hold_k is not None:   # held at a corner: look straight down the next leg
            dk = self.D[self._corner_hold_k]
            if math.hypot(dk[0], dk[1]) > 0.3:
                self._last_heading = math.atan2(dk[1], dk[0])
                return self._last_heading
        if self.avoid is not None:                # look at where we are going: the path point yaw_lookahead_m ahead
            d = self.pos(min(self.S_goal, self.s + self.pr.yaw_lookahead_m)) - self.pos(self.s)
            if math.hypot(d[0], d[1]) > 0.3:
                self._last_heading = math.atan2(d[1], d[0])
                return self._last_heading
        t = self.tangent(min(self.S, self.s + 0.4), h=0.4)
        if math.hypot(t[0], t[1]) > 0.3:
            self._last_heading = math.atan2(t[1], t[0])
        return self._last_heading

    def _chain(self, pos, s0, s1, off, lead_pt=None):
        n = max(1, int(math.ceil(max(s1 - s0, 0.0) / 0.3)))
        pts = [np.asarray(pos, float)]
        if lead_pt is not None:
            pts.append(lead_pt)
        pts += [self.ref(s0 + (s1 - s0) * k / n, off) for k in range(n + 1)]
        return pts

    def _lat_normal(self, s):
        """Horizontal right-hand normal of the path (for nudges, whatever the avoid axis)."""
        t = self.tangent(s)
        h = math.hypot(t[0], t[1])
        if h < 0.2:
            return self._last_lat if self._last_lat is not None else np.array([0.0, 1.0, 0.0])
        self._last_lat = np.array([-t[1] / h, t[0] / h, 0.0])
        return self._last_lat

    def _rc_offset(self, pos, side, probe, valid):
        """Largest sidestep (reconfirm_step_m .. reconfirm_min_step_m) toward `side` that is inside the fence,
        clear of the map (hard radius) and seen free (incl. the airframe edge, abreast and 0.3 m ahead)."""
        pr = self.pr
        s_v = self.project(pos)
        n = self._lat_normal(s_v)
        t = self.tangent(s_v)
        t = np.array([t[0], t[1], 0.0]) / max(1e-6, math.hypot(t[0], t[1]))
        for o in np.arange(pr.reconfirm_step_m, pr.reconfirm_min_step_m - 1e-9, -0.025):
            q = pos + side * o * n
            if valid is not None and not valid(q):
                continue
            if probe is not None and probe([pos, q], extra=self.ignored, radius=pr.hard_radius) != math.inf:
                continue
            if self.count_near is not None and (self.count_near(q, pr.hard_radius) or 0) > 0:
                continue                          # (the target itself: a hit at the very end of the probe segment)
            if not self._move_ok(q, pos):         # move_clear_m from map points; prior obstacles: no closer (v4G)
                continue
            edge = [q + side * pr.move_clear_m * n + da * t for da in (0.0, 0.3)]
            if all(self.seen_free(e) for e in edge):
                return float(o)
        return None

    def _move_ok(self, q, cur=None):
        """A move / hover target keeps move_clear_m from measured map points (ignored clusters excepted), and from
        the prior obstacles (fence, barrier, boulders, wall lines, the never-seen rear of the start) move_clear_m
        -- or, if the vehicle at `cur` is already closer, it gets no closer."""
        m = self.pr.move_clear_m
        if self.keep_ok is not None and self.map_dist is not None and cur is not None:
            return self.map_dist(q, m, self.ignored) >= m - 1e-6 and self.keep_ok(q, cur)
        if self.hold_score is None:
            return True
        return self.hold_score(q, None, self.ignored) >= m - 1e-6

    def _keep_ok(self, q, cur, track_ok=False):
        """Prior keep-out only: see _move_ok.  track_ok (returns onto the path, pull-backs): a point on our own track
        passes too -- back where we already were is no new risk (and staying beside a real cluster leaves the next
        small sidestep inside the camera's 0.4 m blind range: never seen free, a dead-lock)."""
        if self.keep_ok is None or cur is None:
            return True
        if track_ok and self.on_track is not None and self.on_track(q):
            return True
        return self.keep_ok(q, cur)

    def _return_off(self, pos, o_t=0.0):
        """Offset to return to after a sidestep: o_t, else the one nearest to it (toward the current offset) whose
        carrot keeps the prior keep-out or lies on our own track (never into new space closer to the barrier / the
        never-seen rear).  The periodic return retries later."""
        if self.keep_ok is None or abs(self.off - o_t) < 1e-6:
            return o_t
        n = max(1, int(math.ceil(abs(self.off - o_t) / 0.025)))
        for k in range(n + 1):
            o = o_t + (self.off - o_t) * k / n
            if self._keep_ok(self.ref(self.s, o), pos, track_ok=True):
                if k:
                    self.flags.append(('return_limited', {'wanted_m': round(o_t, 3), 'kept_m': round(o, 3)}))
                return float(o)
        self.flags.append(('return_limited', {'wanted_m': round(o_t, 3), 'kept_m': round(self.off, 3)}))
        return float(self.off)

    def _s_floor(self):
        """While an episode's hover point was moved forward (rear clearance), the carrot is not pulled back behind it."""
        a = self._hold_adv
        return a['s'] if a is not None else -math.inf

    def _sticky_probe(self, probe):
        """probe() that also treats the sticky clusters (centre, radius) as solid."""
        pr = self.pr

        def wrapped(chain, strict=False, extra=None, radius=None):
            free = probe(chain, strict=strict, extra=extra, radius=radius)
            done = 0.0
            for i in range(len(chain) - 1):
                if done >= free:
                    break
                a, b = np.asarray(chain[i], float), np.asarray(chain[i + 1], float)
                d = b - a
                seg = float(np.linalg.norm(d))
                if seg < 1e-6:
                    continue
                rad = radius if radius is not None else \
                    ((pr.radius + (pr.return_margin_m if strict else 0.0)) if (strict or i > 0) else pr.hard_radius)
                for st in self._sticky:
                    R = st['r'] + rad
                    w = a - st['c']
                    A, B, C = float(d @ d), 2.0 * float(w @ d), float(w @ w) - R * R
                    if C <= 0.0:
                        free = min(free, done)
                        continue
                    disc = B * B - 4.0 * A * C
                    if disc < 0.0:
                        continue
                    t = (-B - math.sqrt(disc)) / (2.0 * A)
                    if 0.0 <= t <= 1.0:
                        free = min(free, done + t * seg)
                done += seg
            return free
        return wrapped

    def _sticky_release(self, now, pos, heading):
        """Drop a sticky cluster once it is seen absent (< 2 map points) for 0.5 s from >= start_sticky_release_m,
        inside the camera FOV."""
        pr = self.pr
        if heading is None or self.count_near is None:
            return
        cam = pos + pr.cam_forward_m * np.array([math.cos(heading), math.sin(heading), 0.0])
        keep = []
        for st in self._sticky:
            dv = (st['c'] - cam)[:2]
            vis = float(np.linalg.norm(dv)) >= pr.start_sticky_release_m and \
                abs(wrap(math.atan2(dv[1], dv[0]) - heading)) <= math.radians(pr.seen_half_fov_deg - 3.0)
            n = self.count_near(st['c'], st['r'] + 0.1) if vis else None
            if vis and n is not None and n < 2:
                st['gone_t'] = st.get('gone_t') or now
                if now - st['gone_t'] >= 0.5:
                    self.flags.append(('start_sticky_cleared', {'center': st['c'].copy(), 'n': st['n']}))
                    continue
            else:
                st['gone_t'] = None
            keep.append(st)
        self._sticky = keep

    def _safe_hold(self, now, pos, probe, heading=None, spd=0.0, confirmed_ahead=False, cluster=None):
        """Before hovering for an observation (parallax nudge / yaw scan / re-confirmation): the hover point keeps
        move_clear_m from map points and the fence, with that much KNOWN free behind it (the rear is
        never seen: v4E drifted to 0.24 m from the barrier behind the take-off point during the scans).  Else move
        the carrot forward along the path (<= hold_advance_max_m, the move itself clear) to the best point.  True
        once the hover point is final (reached, or nothing better)."""
        pr = self.pr
        if self.hold_score is None or self.avoid != 'lateral':
            return True
        adv = self._hold_adv
        if adv is None:
            s0 = max(self.project(pos), min(self.s, self.project(pos) + 0.05))
            t = self.tangent(s0)
            h = math.hypot(t[0], t[1])
            back = -np.array([t[0] / h, t[1] / h, 0.0]) if h > 0.2 else None
            q0 = self.ref(s0, self.off)
            best, sc0 = s0, self.hold_score(q0, back, self.ignored)
            best_sc = sc0
            start = (self.start_f is not None and self.start_f(q0) is not None and not confirmed_ahead
                     and pr.start_advance_m > 0.0)
            if start:                             # (v4J) off the take-off point, whatever the camera sees ahead
                cl = pr.start_advance_clear_m
                for s1 in np.arange(s0 + 0.05, min(self.S, s0 + pr.start_advance_m + 0.5) + 1e-9, 0.05):
                    q = self.ref(float(s1), self.off)
                    fq = self.start_f(q)
                    if fq is None or fq > pr.start_advance_m + 1e-6:
                        break
                    if probe is not None and probe([pos, q], extra=self.ignored, radius=pr.hard_radius) != math.inf:
                        break
                    if self.map_dist is not None and self.map_dist(q, cl, self.ignored) < cl - 1e-6:
                        break
                    far = self.ref(min(self.S, float(s1) + cl), self.off)   # seen free (where the camera can
                    hd = self._last_heading if heading is None else heading   # see at all: not its 0.4 m blind zone)
                    cam = pos + (pr.cam_forward_m * np.array([math.cos(hd), math.sin(hd), 0.0]) if hd is not None
                                 else 0.0)
                    if self.seen_free is not None and float(math.hypot(*(far - cam)[:2])) >= pr.seen_min_range_m \
                            and not self.seen_free(far):
                        break
                    best = float(s1)
                best_sc = self.hold_score(self.ref(best, self.off), back, self.ignored)
                if cluster is not None and best > s0 + 1e-6 and cluster[0] is not None:
                    c_k, r_k = np.asarray(cluster[0], float), float(cluster[1] or 0.15)
                    tt = self.tangent(best)
                    tt = np.array([tt[0], tt[1], 0.0]) / max(1e-6, math.hypot(tt[0], tt[1]))
                    cam_t = self.ref(best, self.off) + pr.cam_forward_m * tt
                    cam_m = float(np.linalg.norm((c_k - cam_t)[:2])) - r_k
                    if cam_m < pr.start_sticky_release_m:       # (the map will forget it: keep it ourselves)
                        self._sticky.append(dict(c=c_k.copy(), r=r_k, n=cluster[2], support=cluster[3]))
                        self.flags.append(('start_sticky', {'center': c_k.copy(), 'radius_m': round(r_k, 3),
                                                            'n': cluster[2], 'support': cluster[3],
                                                            'camera_m': round(cam_m, 3)}))
            elif sc0 < pr.move_clear_m - 1e-6:
                s_end = min(self.S, s0 + pr.hold_advance_max_m)
                hd = self._last_heading if heading is None else heading
                cam_d = (lambda q_: self.map_dist(q_ + pr.cam_forward_m * np.array([math.cos(hd), math.sin(hd), 0.0]),
                                                  cam_need, self.ignored)) if (self.map_dist is not None and hd is not None) \
                    else None
                cam_need = pr.seen_min_range_m + 0.1  # the camera stays outside the stereo blind zone of what it
                cam_lim = min(cam_need, cam_d(q0)) if cam_d is not None else 0.0   # watches (inside it the map
                for s1 in np.arange(s0 + 0.05, s_end + 1e-9, 0.05):                 # forgets it: 'floating')
                    q = self.ref(float(s1), self.off)
                    if probe is not None and probe([pos, q], extra=self.ignored, radius=pr.hard_radius) != math.inf:
                        break                     # (the move itself: never into the obstacle ahead)
                    if self.seen_free is not None and not self.seen_free(q):
                        break
                    if cam_d is not None and cam_d(q) < cam_lim - 1e-6:
                        break
                    sc = self.hold_score(q, back, self.ignored)
                    if sc > best_sc + 0.01:
                        best, best_sc = float(s1), sc
                    if sc >= pr.move_clear_m - 1e-6:
                        break
            adv = self._hold_adv = dict(s=best, s0=s0, t0=now, done=best <= s0 + 1e-6)
            if not adv['done']:
                self.flags.append(('start_advance' if start else 'hold_advanced',
                                   {'from': q0.copy(), 'to': self.ref(best, self.off).copy(),
                                    'clear_before_m': round(sc0, 3), 'clear_m': round(best_sc, 3)}))
        if adv['done']:
            return True
        ramp = adv['s0'] + pr.hold_advance_v * (now - adv['t0'])          # (<= hold_advance_v)
        self.s, self.v = max(self.s, min(adv['s'], ramp)), 0.0
        if (float(np.linalg.norm((pos - self.ref(adv['s'], self.off))[:2])) < 0.03 and spd <= 0.05) or \
                now - adv['t0'] > 5.0 + (adv['s'] - adv['s0']) / max(pr.hold_advance_v, 1e-3):
            adv['done'] = True
            return True
        return False

    def _reconfirm_step(self, now, pos, heading, probe, valid):
        """scan (look toward a side) -> step (sidestep there, face the cluster) -> watch.  Gone = phantom."""
        pr, rc = self.pr, self._rc
        n_now = self.count_near(rc['c'], rc['r'] + 0.1)

        def finish(kind):
            self.flags.append((kind, {'center': rc['c'].copy(), 'stage': rc['phase'], 's': round(now - rc['t0'], 2),
                                      'step_m': rc.get('off')}))
            self._rc_done.append(rc['c'].copy())
            if kind == 'cluster_real':
                self._rc_real.append(rc['c'].copy())
            if kind == 'phantom_cleared':
                e = (rc['c'].copy(), np.array([0.0, 0.0, 1.0]), rc['r'] + 0.05, rc['r'] + 0.05)
                self.ignored.append(e)
                self._ignored_keep.append(e[0])
                self.off_tgt = self._return_off(pos)   # back onto the path (our own track) -- never back
            else:                                 # real / no room: back onto the path (our own track) and a fresh
                self.off_tgt = self._return_off(pos)   # blockage episode for the normal handling (side-step search, its
                self.blocked_since = None         # own look-first, its own timeout)
                self._t_search = -math.inf
                self.look_off = None
                self._look_unseen = self._look_unseen_t0 = None
            self._rc = None
        in_view = False                           # 'gone' only counts while the camera LOOKS at it (out of view the
        if heading is not None:                   # map just forgets it after its 4 s memory -- not carving)
            cam = pos + pr.cam_forward_m * np.array([math.cos(heading), math.sin(heading), 0.0])
            dv = (rc['c'] - cam)[:2]
            in_view = float(np.linalg.norm(dv)) >= 0.45 and \
                abs(wrap(math.atan2(dv[1], dv[0]) - heading)) <= math.radians(pr.seen_half_fov_deg - 3.0)
        if rc['phase'] in ('step', 'watch') and in_view:
            rc['seen_t'] = rc.get('seen_t', now)
            if n_now is not None and n_now < 2 and now - rc['seen_t'] >= 0.5:
                finish('phantom_cleared')         # carved from the new viewpoint (looked at for >= 0.5 s)
                return
        if now - rc['t0'] > 25.0:
            finish('reconfirm_timeout')
            return
        if rc['phase'] == 'scan':
            if not rc['sides']:
                finish('reconfirm_no_room')
                return
            sd = rc['sides'][0]
            n = self._lat_normal(self.project(pos))
            ys = math.atan2(sd * n[1], sd * n[0])
            rc['yaw'] = wrap(rc['h0'] + float(np.clip(wrap(ys - rc['h0']), -math.radians(pr.reconfirm_scan_deg),
                                                      math.radians(pr.reconfirm_scan_deg))))
            if heading is not None and abs(wrap(rc['yaw'] - heading)) <= math.radians(3.0):
                if rc.get('dwell') is None:
                    rc['dwell'] = now
                elif now - rc['dwell'] >= 0.5:
                    o = self._rc_offset(pos, sd, probe, valid)
                    rc['dwell'] = None
                    if o is None:
                        rc['sides'].pop(0)
                    else:
                        rc.update(phase='step', side=sd, off=round(o, 2), t1=now)
                        self.off_tgt = self.off + sd * o
                        self.flags.append(('reconfirm_sidestep', {'offset_m': round(sd * o, 2)}))
            return
        d = (rc['c'] - pos)[:2]
        rc['yaw'] = math.atan2(d[1], d[0])         # watch the cluster from the new viewpoint
        if rc['phase'] == 'step':
            if (abs(self.off - self.off_tgt) < 0.02 and
                    float(np.linalg.norm((pos - self.ref(self.s))[:2])) < 0.08) or now - rc['t1'] > 6.0:
                rc.update(phase='watch', t2=now)
            return
        if now - rc['t2'] >= pr.reconfirm_watch_s and rc.get('seen_t') is not None:
            finish('cluster_real')                # still there after > the map memory: being re-measured

    def _nudge_period(self):
        return self.pr.nudge_period_s if self._nudge_lat else self.pr.observe_scan_period_s

    def _nudge(self, t_rel, s):
        pr = self.pr
        w = 2.0 * math.pi / self._nudge_period()
        n, up = self._lat_normal(s), np.array([0.0, 0.0, -1.0])
        lat = pr.nudge_lat_m if self._nudge_lat else 0.0
        upm = pr.nudge_up_m if self._nudge_lat else pr.observe_up_m   # (observation manoeuvre: climb + yaw scan)
        if not self._nudge_lat and self._nudge_h0 is not None:
            self._nudge_yaw = wrap(self._nudge_h0 + math.radians(pr.observe_scan_deg) * math.sin(w * t_rel))
        d = lat * math.sin(w * t_rel) * n + upm * 0.5 * (1.0 - math.cos(w * t_rel)) * up
        v = lat * w * math.cos(w * t_rel) * n + upm * 0.5 * w * math.sin(w * t_rel) * up
        return d, v

    def _nudge_side_seen(self, pos):
        """Both sideways nudge extremes (+ the airframe edge, abreast and 0.3 m either way) seen free."""
        if self.seen_free is None:
            return True
        pr, n = self.pr, self._lat_normal(self.s)
        t = self.tangent(self.s)
        t = np.array([t[0], t[1], 0.0]) / max(1e-6, math.hypot(t[0], t[1]))
        r = pr.nudge_lat_m + pr.prop_radius_m
        pts = [pos + sg * r * n + da * t for sg in (1.0, -1.0) for da in (-0.3, 0.0, 0.3)]
        return all(self.seen_free(q) for q in pts)

    def _nudge_side_clear(self, pos):
        """Both sideways nudge extremes keep move_clear_m from map points and the fence."""
        n = self._lat_normal(self.s)
        return all(self._move_ok(pos + sg * self.pr.nudge_lat_m * n, pos) for sg in (1.0, -1.0))

    def _tiny(self, c, r):
        """Fewer than never_block_n map points there: never blocks on its own."""
        if self.count_near is None or c is None:
            return False
        n = self.count_near(c, float(r) + self.pr.cluster_radius_extra_m)
        return n is not None and n < self.pr.never_block_n

    def _nudge_safe(self, pos, probe):
        pr, n = self.pr, self._lat_normal(self.s)
        ends = [pos + pr.nudge_lat_m * n, pos - pr.nudge_lat_m * n, pos + np.array([0.0, 0.0, -pr.nudge_up_m])]
        return all(probe([pos, q]) == math.inf for q in ends)

    def _search(self, pos, L, probe, valid, seen=None):
        """Nearest free offset.  Returns (offset or None, nearest free-but-unseen offset or None)."""
        pr = self.pr
        unseen = None
        s_v = self.project(pos)
        k_max = int(round(self.side_max / pr.side_step_m))
        cands = []
        for k in range(-k_max, k_max + 1):
            o = k * pr.side_step_m
            if self.avoid == 'up' and o < 0.0:
                continue
            if abs(o - self.off) < 1e-9:
                continue
            cands.append(o)
        cands.sort(key=lambda o: (abs(o - self.off), abs(o)))
        for o in cands:
            chain = self._chain(pos, s_v, min(self.S, max(s_v, self.s) + L), o)
            if valid is not None and not all(valid(q) for q in chain[1:]):
                continue
            if probe(chain) != math.inf:
                continue
            if not self._move_ok(self.ref(s_v, o), pos):   # the sidestep target: map move_clear_m, priors no closer
                continue
            sf_ = self.seen_free
            if self.avoid == 'lateral' and ((sf_ is not None and abs(o - self.off) > 1e-9) or
                                            (seen is not None and abs(o) > pr.side_unseen_max_m + 1e-9)):
                side = self._chain(pos, s_v, min(self.S, s_v + pr.seen_check_ahead_m), o)[1:]
                chk = sf_ if sf_ is not None else seen
                if sf_ is not None:               # + the airframe edge (+ 0.1 m) on the side we move to
                    sg = 1.0 if o > self.off else -1.0
                    side += [self.ref(min(self.S, s_v + da), o + sg * pr.move_clear_m) for da in (0.0, 0.3)]
                missing = [q for q in side if not chk(q)]
                if missing:
                    if unseen is None:
                        unseen = (o, missing)
                    continue
            return o, None
        return None, unseen

    def _try_return(self, pos, L, probe, valid, seen=None):
        """Move the offset back toward the nominal path.  The ACTUAL lateral motion (vehicle -> new carrot) must be
        clear, and the swept region must have been seen recently (round 5b seed 7: the
        return from 1.2 to 0.3 m slid into a never-measured panel side).  Returns (offset or None, unseen or None)."""
        pr = self.pr
        s_v = self.project(pos)
        o_t = self.off_tgt
        unseen = None
        k = int(round(abs(o_t) / pr.side_step_m))
        for j in range(0, k):                         # nearest to the nominal path first
            o = math.copysign(j * pr.side_step_m, o_t)
            chain = self._chain(pos, s_v, min(self.S, max(s_v, self.s) + L), o)
            if valid is not None and not all(valid(q) for q in chain[1:]):
                continue
            if probe(chain, strict=True) != math.inf:
                continue
            if probe([pos, self.ref(max(self.s, s_v), o)], strict=True) != math.inf:   # the sideways sweep itself
                continue
            if not self._keep_ok(self.ref(max(self.s, s_v), o), pos, track_ok=True):   # (v4G) no closer to a
                continue                                  # prior obstacle / the never-seen rear, unless on our track
            if seen is not None and self.avoid == 'lateral' and abs(self.off - o) > 1e-9:
                # every return step: no walking back into unseen space 0.3 m at a time
                n = max(2, int(math.ceil(abs(self.off - o) / pr.side_step_m)) + 1)
                sweep = [self.ref(s_v, x) for x in np.linspace(self.off, o, n)[1:]]
                sweep += self._chain(pos, s_v, min(self.S, s_v + pr.seen_check_ahead_m), o)[1:]
                chk = self.seen_free if self.seen_free is not None else seen
                if self.seen_free is not None:
                    sg = 1.0 if o > self.off else -1.0
                    sweep += [self.ref(s_v, o + sg * pr.move_clear_m)]
                missing = [q for q in sweep if not chk(q)]
                if missing:
                    if unseen is None:
                        unseen = (o, missing)
                    continue
            return o, None
        return None, unseen

    def _set_look(self, pos, unseen):
        self.look_off, missing = unseen               # look at the middle of what has not been seen
        b = [math.atan2(q[1] - pos[1], q[0] - pos[0]) for q in missing]
        rel = [wrap(x - b[0]) for x in b]
        self._look_heading = wrap(b[0] + 0.5 * (min(rel) + max(rel)))

    def step(self, dt, now, pos, heading, probe=None, valid=None, cap=math.inf, seen=None, cluster=None, vel=None,
             ahead=None):
        pr, a = self.pr, self.a
        pos = np.asarray(pos, float)
        self._seen_cb = seen
        if self._sticky:
            self._sticky_release(now, pos, heading)
        raw_probe = probe
        if self._sticky and probe is not None:
            probe = self._sticky_probe(probe)     # (v4K) kept start clusters the camera can no longer see
        if self.avoid in ('lateral', 'stop'):
            self._snap_vertices(now, self.s)
            self._update_tight(self.s, ahead)
        ts = self._tight_stop(self.s)
        if ts is not None and self.s >= ts[0] - 1e-3:  # carrot ON a tight corner: settle, face the next leg, go
            sc, k = ts
            self.s, self.v = sc, 0.0
            if self._settle is None or self._settle['k'] != k:
                self._settle = dict(k=k, t0=now, phase='settle')
                self.flags.append(('corner_settle', {'corner': self.P[k].copy(), 'carrot': self.ref(sc).copy(),
                                                     'space_ahead_m': round(self._tight[k], 3)}))
            st = self._settle
            dxy = float(np.linalg.norm((pos - self.ref(sc))[:2]))   # (the carrot may carry a side-step offset)
            spd = float(np.linalg.norm(vel)) if vel is not None else 0.0
            late = now - st['t0'] > pr.corner_settle_max_s and dxy <= 2.0 * pr.corner_settle_tol_m
            if st['phase'] == 'settle' and ((dxy <= pr.corner_settle_tol_m and spd <= pr.corner_settle_v) or late):
                st['phase'] = 'yaw'
            dk = self.D[k]
            aligned = heading is None or math.hypot(dk[0], dk[1]) < 0.3 or \
                abs(wrap(math.atan2(dk[1], dk[0]) - heading)) <= math.radians(pr.corner_yaw_tol_deg)
            if st['phase'] == 'yaw' and (aligned or now - st['t0'] > 2.0 * pr.corner_settle_max_s):
                self._settled.add(k)
                self._settle = None
                self.flags.append(('corner_go', {'corner': self.P[k].copy(), 'wait_s': round(now - st['t0'], 2),
                                                 'settle_err_m': round(dxy, 3)}))
            else:
                ref = self.ref(self.s)
                return Step((ref, np.zeros(3), np.zeros(3), self.heading_cmd()), 'moving', math.inf)
        ref = self.ref(self.s)
        lead = float(np.linalg.norm(ref - pos))
        v_t = min(self.v_limit(min(self.S, self.s + self.v * dt)), cap)   # limit where the carrot will be
        if lead > self.leash_lo:
            v_t *= min(1.0, max(0.0, (self.leash_hi - lead) / (self.leash_hi - self.leash_lo)))
        t = self.tangent(self.s)
        turning = (self._next_corner(self.s, pr.corner_yaw_ahead_m) is not None or
                   (self.yaw_end is not None and self.S_goal - self.s <= pr.yaw_end_dist_m))   # heading differs on
        if self.avoid is not None and heading is not None and math.hypot(t[0], t[1]) > 0.5 and not turning \
                and self.yaw_fixed is None:       # (a fixed yaw = flying sideways / backward on purpose)
            err = abs(wrap(math.atan2(t[1], t[0]) - heading))
            lo, hi = math.radians(pr.yaw_slow_lo_deg), math.radians(pr.yaw_slow_hi_deg)
            if err > lo:
                v_t *= max(pr.yaw_slow_min_frac, 1.0 - (1.0 - pr.yaw_slow_min_frac) * (err - lo) / (hi - lo))
        if self.avoid == 'lateral' and seen is not None and heading is not None:   # (fixed yaw too: look, then back off)
            q = self.ref(min(self.S, self.s + pr.unseen_check_m))
            dq = (q - pos)[:2]
            chk = self.seen_free if (self.yaw_fixed is not None and self.seen_free is not None) else seen
            if float(np.linalg.norm(dq)) > 0.25 and not chk(q):  # (fixed yaw = backing / sideways: stricter)
                if self._look_unseen_t0 is None:
                    self._look_unseen_t0 = now
                    self.flags.append(('look_unseen_path', {'point': q.copy()}))
                if now - self._look_unseen_t0 <= pr.unseen_look_max_s:
                    self._look_unseen = math.atan2(dq[1], dq[0])
                    v_t = 0.0
                else:
                    self._look_unseen = None
            else:
                self._look_unseen = self._look_unseen_t0 = None
        ye_ = self._yaw_end_eff()
        if pr.yaw_end_match and ye_ is not None and heading is not None and self.yaw_fixed is None and \
                self.avoid in ('lateral', 'stop') and not self.pass_through:
            d_end = self.S_goal - self.s              # the turn to face the stage: done on the way in, slowly
            rem = abs(wrap(ye_ - heading))
            if d_end <= pr.yaw_end_dist_m and rem > math.radians(pr.corner_yaw_tol_deg):
                v_t = min(v_t, max(0.05, d_end / (rem / max(pr.yaw_rate_near, 1e-3))))
        ec = self._early_corner(self.s)
        if ec is not None and heading is not None:   # stop corner ahead: turn to the next leg on the way in, and
            k, dist, want = ec                       # arrive no sooner than the (slow, near-wall) yaw is done
            rem = abs(wrap(want - heading))
            if rem > math.radians(pr.corner_yaw_tol_deg):
                t_need = rem / max(pr.yaw_rate_near, 1e-3)
                v_t = min(v_t, max(0.05, dist / t_need))
        free, blocked = math.inf, False
        nudge_d, nudge_v = np.zeros(3), np.zeros(3)
        vio_clip = None
        if self.avoid == 'lateral' and probe is not None and heading is not None and pr.vio_wall_standoff_m > 0 \
                and self.vio_check:
            # (not the wall crossing: it climbs while approaching the wall on purpose; its 'up' search handles it)
            # VIO: the camera (at the carrot, looking along the actual heading) must stay >= vio_wall_standoff_m from
            # a measured obstacle straight ahead -- also beyond the end of the path
            hd = np.array([math.cos(heading), math.sin(heading), 0.0])
            cam = ref + pr.cam_forward_m * hd
            hold_at = pr.vio_wall_standoff_m + pr.vio_hold_margin_m
            ray = [cam, cam + hd * (hold_at + 0.6)]
            fv = ahead(cam, hd, pr.vio_view_radius_m, self.ignored) if ahead is not None else None
            if fv is None:                              # map without .points: estimate from the probe
                fv = probe(ray, extra=self.ignored, radius=pr.vio_view_radius_m)
                fv = fv + pr.vio_view_radius_m if math.isfinite(fv) else fv
            if fv < hold_at and cluster is not None:
                n_v, c_v, r_v = cluster(ray, max(0.0, fv - pr.vio_view_radius_m), self.ignored)
                cv_ = c_v if c_v is not None else getattr(fv, 'pt', None)
                if n_v is not None and n_v < pr.minor_cluster_n and (n_v < pr.never_block_n or self._phantom(cv_, r_v)
                                                                     or self._floating(cv_, r_v)):
                    fv = math.inf                       # a few phantom voxels in front of the net do not starve VIO
                                                        # (elsewhere a small cluster may be a thin board end)
            if fv < hold_at and self._toward_net(cam + hd * fv, hd):
                fv = math.inf                           # in front of the net: stereo phantoms; the fence prior and
                v_t = min(v_t, pr.v_minor)              # the path probe keep us off the net itself -> go on slowly
            toward, facing = 0.0, 0.0
            if fv < hold_at:
                tt = self.tangent(self.s)[:2]
                facing = float(tt @ hd[:2])             # would we fly toward what the camera sees?
                hit = cam + hd * fv
                dh = (hit - ref)[:2]
                nd = float(np.linalg.norm(dh))
                toward = float(tt @ dh / nd) if nd > 1e-6 else 1.0   # is the obstacle itself ahead along the path?
            obs_ahead = toward > math.cos(math.radians(45.0))
            holding = fv < hold_at and (obs_ahead or facing > math.cos(math.radians(70.0)))
            if holding and self._off_path_ahead(getattr(fv, 'pt', cam + hd * fv), self.s) > pr.radius + 0.05:
                holding = False                         # beside the path (seen obliquely), not in our way
            if holding:
                if self._vio_hold_since is None:
                    self._vio_hold_since = now
            else:
                self._vio_hold_since = None
            self._vio_stuck = False
            if holding and now - self._vio_hold_since > pr.vio_hold_max_s and probe is not None and \
                    probe(self._chain(pos, self.s, min(self.S, self.s + pr.probe_min_m), self.off, lead_pt=ref)) \
                    == math.inf:
                holding = False                         # the path itself is clear: creep on (never dead-lock)
                v_t = min(v_t, pr.v_minor * 0.5)
                if not getattr(self, '_vio_timeout_flagged', False):
                    self._vio_timeout_flagged = True
                    self.flags.append(('vio_hold_timeout', {'camera_to_obstacle_m': round(fv, 3)}))
            elif holding and now - self._vio_hold_since > pr.vio_hold_max_s:
                self._vio_stuck = True                  # held, and the path is not clear either: an observation /
                                                        # blocked verdict must follow (no endless hover)
            if holding:
                v_t = 0.0                               # never fly toward a wall < 0.8 m straight ahead of the camera
                if obs_ahead and fv < pr.vio_wall_standoff_m - 0.05:   # carrot inside: pull back, never behind the vehicle
                    self._pull_back(max(self.project(pos), self.s - (pr.vio_wall_standoff_m - fv) / toward), pos, vel, dt)
                nxt = self._next_stop(self.s)
                if nxt is not None and nxt[1] <= pr.near_waypoint_m:
                    if nxt[0] >= self.n_goal - 1:
                        vio_clip = ('vio_wall_ahead_of_path_end', fv)
                    elif not self._corner_hold and (nxt[0] in self._tight or
                                                    self.turn.get(nxt[0], 0.0) >= pr.stop_corner_deg):
                        self._corner_hold = True        # (a real corner; prior-noise kinks never hold)
                        self._corner_hold_k = nxt[0]
                        self.flags.append(('vio_corner_hold', {'corner': self.P[nxt[0]].copy(),
                                                               'carrot': ref.copy(), 'camera_to_obstacle_m': round(fv, 3)}))
            elif self._corner_hold and (fv >= hold_at + 0.05 or not math.isfinite(fv)):
                self._corner_hold = False             # (hysteresis: no hold / release chatter)
                self.flags.append(('vio_corner_release', {'carrot': ref.copy()}))
        if self.ignored and self.phantom_ok is not None and now - self._t_revalid >= 0.5:
            self._t_revalid = now                 # an ignored cluster that turned out to be structure (seen from a
            keep = []                             # second viewpoint: a board end) is real again
            for e in self.ignored:
                if self.phantom_ok(e[0], e[2]) or self._floating(e[0], e[2]) or self._tiny(e[0], e[2]) or \
                        any(float(np.linalg.norm(e[0] - c)) < 1e-6 for c in self._ignored_keep):
                    keep.append(e)
                else:
                    self.flags.append(('ignore_revoked', {'center': np.asarray(e[0], float).copy(),
                                                          'radius_m': round(float(e[2]), 3)}))
            self.ignored = keep
        if self._rc is not None:                  # observe -> sidestep -> re-confirm: hold, nothing else decides
            v_t = 0.0
            self._reconfirm_step(now, pos, heading, probe, valid)
        if self.avoid is not None and probe is not None and self._rc is None:
            P = lambda ch, strict=False: probe(ch, strict=strict, extra=self.ignored)   # noqa: E731
            L = max(pr.probe_min_m, self.v * self.v / (2.0 * min(a, pr.a_stop_plan)) + pr.probe_margin_m)
            chain = self._chain(pos, self.s, min(self.S, self.s + L), self.off, lead_pt=ref)
            free = P(chain)
            if any(np.linalg.norm(pos - c[0]) < pr.ignore_slow_dist_m for c in self.ignored):
                v_t = min(v_t, pr.v_minor)
            if free < math.inf:
                n_cl, c_cl, r_cl = cluster(chain, free, self.ignored) if cluster is not None else (None, None, None)
                small = n_cl is not None and n_cl < pr.minor_cluster_n
                tiny = n_cl is not None and n_cl < pr.never_block_n     # (v4E: 1-2 points never block on their own)
                minor = small and (tiny or self._phantom(c_cl, r_cl) or self._floating(c_cl, r_cl))   # net band, or
                if self._sticky and probe is not raw_probe and \
                        free < raw_probe(chain, extra=self.ignored) - 1e-6:   # (v4K) a sticky start cluster binds:
                    small = minor = tiny = False                            # a real obstacle, never minor
                if minor and tiny and not getattr(self, '_tiny_flagged', False):                  # no floor support
                    self._tiny_flagged = True
                    self.flags.append(('tiny_cluster', {'n': n_cl, 'center': None if c_cl is None else
                                                        np.asarray(c_cl, float)}))
                elif minor and not self._phantom(c_cl, r_cl) and not getattr(self, '_float_flagged', False):
                    self._float_flagged = True
                    self.flags.append(('floating_cluster', {'n': n_cl, 'center': np.asarray(c_cl, float),
                                                            'radius_m': None if r_cl is None else round(float(r_cl), 3),
                                                            'support': self.float_info(c_cl, r_cl) if
                                                            self.float_info is not None else None}))
                if small and not minor and not self._kept_flagged:   # can be an edge-on thin board end (round B)
                    self._kept_flagged = True
                    info = self.float_info(c_cl, r_cl) if (self.float_info is not None and c_cl is not None) else None
                    self.flags.append(('small_cluster_kept', {'n': n_cl, 'free_m': round(free, 3),
                                                              'center': None if c_cl is None else np.asarray(c_cl, float),
                                                              'radius_m': None if r_cl is None else round(float(r_cl), 3),
                                                              'support': info}))
                if not minor and c_cl is not None and self._toward_net(c_cl, self.tangent(self.s)):
                    minor = True                    # in front of the net: nudge, then ignore it and go on slowly
                    if not getattr(self, '_fence_cl_flagged', False):
                        self._fence_cl_flagged = True
                        self.flags.append(('fence_cluster', {'n': n_cl, 'center': np.asarray(c_cl, float),
                                                             'fence_m': round(float(self.fence_dist(c_cl)), 3)}))
                if self._episode_t0 is None:
                    self._episode_t0 = now
                if minor:
                    if self._minor_since is None:
                        self._minor_since = now
                    v_t = min(v_t, pr.v_minor)
                else:
                    self._minor_since = None
                # 0) VIO: a real obstacle ahead in the camera's view keeps the camera >= vio_wall_standoff_m away
                stop_eff = pr.stop_margin_m
                in_view = False
                if not minor:
                    in_view = self._in_view(pos, heading, self._along(chain, free))
                    if in_view:
                        stop_eff = max(stop_eff, pr.vio_wall_standoff_m + pr.cam_forward_m - pr.radius)
                stopped = free - lead < stop_eff + 0.1 or getattr(self, '_vio_stuck', False)   # (the camera-ahead
                # hold stops us a few cm earlier for an obstacle off the axis and cannot creep: without this, no nudge
                # / observation / blocked verdict ever followed -- a hover until the stage timeout)
                clipping = False
                if in_view and stopped and self.avoid == 'lateral':
                    nxt = self._next_stop(self.s)
                    if nxt is not None and nxt[1] <= pr.near_waypoint_m and nxt[0] >= self.n_goal - 1:
                        clipping = True             # the path END is too close to that obstacle (corners: hold)
                spd = float(np.linalg.norm(vel)) if vel is not None else 0.0
                # 1) parallax nudge, once per blockage, before anything is declared blocked; only once (nearly)
                #    stopped and centred on the vehicle (never toward the obstacle ahead)
                if (self._nudge_t0 is None and not self._nudged and not clipping
                        and (stopped or (minor and now - self._episode_t0 >= pr.minor_wait_s))):
                    confirmed = free < pr.start_confirmed_m and c_cl is not None and (
                        not small or any(float(np.linalg.norm((np.asarray(c_cl, float) - c)[:2])) < 0.3
                                         for c in self._rc_real))
                    kept_cl = None if (minor or c_cl is None) else (c_cl, r_cl, n_cl, self.float_info(c_cl, r_cl)
                                                                     if self.float_info is not None else None)
                    ready = self._safe_hold(now, pos, probe, heading, spd, confirmed, kept_cl)   # (v4F) hover point clear of the rear
                    self.s = min(self.s, max(self.project(pos), self._s_floor()))
                    self.v, v_t = 0.0, 0.0
                    if spd <= pr.nudge_max_speed and ready:
                        if self._nudge_safe(pos, P):
                            self._nudge_lat = self._nudge_side_seen(pos) and self._nudge_side_clear(pos)
                            # (sideways only into seen-free space, move_clear_m from map / fence)
                            self._nudge_h0, self._nudge_yaw = heading, None
                            if not self._nudge_lat:
                                self.flags.append(('nudge_vertical_only', {'scan_deg': pr.observe_scan_deg,
                                                                           'up_m': pr.observe_up_m}))
                            self._nudge_t0 = now
                            self.flags.append(('parallax_nudge', {'minor': minor, 'n': n_cl, 'free_m': round(free, 3)}))
                        else:
                            self._nudged = True
                            self.flags.append(('nudge_skipped', {'free_m': round(free, 3)}))
                if self._nudge_t0 is not None:
                    tr = now - self._nudge_t0
                    if tr >= self._nudge_period():
                        self._nudge_t0, self._nudged = None, True
                        self._nudge_yaw = None
                    else:
                        nudge_d, nudge_v = self._nudge(tr, self.s)
                        v_t = 0.0
                nudging = self._nudge_t0 is not None
                found, searched = None, False
                if minor:
                    # 2) tiny clusters never block and never cause a side step; after the nudge they are ignored
                    if self._nudged and not nudging and c_cl is not None:
                        self.ignored.append((np.asarray(c_cl, float), np.array([0.0, 0.0, 1.0]), r_cl, r_cl))
                        self.flags.append(('minor_ignored', {'n': n_cl, 'center': np.asarray(c_cl, float),
                                                             'radius_m': round(r_cl, 3)}))
                    self.look_off = None
                elif clipping:
                    pass                            # handled after the braking below
                elif (small and self._nudged and not nudging and c_cl is not None and self.count_near is not None
                      and self.avoid == 'lateral' and heading is not None and self.seen_free is not None
                      and not any(float(np.linalg.norm((c_cl - c)[:2])) < 0.3 for c in self._rc_done)):
                    self._rc = dict(phase='scan', t0=now, c=np.asarray(c_cl, float).copy(),   # a kept small
                                    r=float(r_cl or 0.15), h0=float(heading), sides=[1.0, -1.0], yaw=None)   # cluster
                    self.flags.append(('reconfirm_start', {'center': self._rc['c'].copy(), 'n': n_cl,
                                                           'free_m': round(free, 3)}))
                    v_t = 0.0
                else:
                    if self.avoid in ('lateral', 'up') and not nudging and now - self._t_search >= pr.avoid_period_s:
                        self._t_search, searched = now, True
                        found, unseen = self._search(pos, L, P, valid, seen)
                        if found is not None:
                            self.off_tgt = found
                        self.look_off = None
                        if found is None and unseen is not None:
                            self._set_look(pos, unseen)
                    shifting = abs(self.off_tgt - self.off) > 1e-6
                    if self.avoid == 'stop' or searched:
                        blocked = found is None and not shifting
                    else:                           # no search this step: keep the previous verdict
                        blocked = self.blocked_since is not None and not shifting
                    blocked = blocked and self._nudged and not nudging   # 3) only after the parallax nudge
                free_ref = free - lead
                v_t = min(v_t, math.sqrt(2.0 * pr.a_stop_plan * max(0.0, free_ref - stop_eff)))
                if in_view and free_ref < stop_eff - 0.05:   # carrot inside the VIO standoff: pull it back out
                    self._pull_back(max(0.0, self.s - (stop_eff - free_ref)), pos, vel, dt)
                elif free_ref < 0.0:                # obstacle appeared between vehicle and carrot
                    self.s = min(self.s, self.project(pos))
                    self.v = 0.0
                if clipping:
                    self._clip_end('in_view_obstacle_at_path_end', free - lead + pr.radius)
                    blocked = False
            else:                                   # clear: the blockage episode is over
                self._minor_since = self._episode_t0 = self._nudge_t0 = None
                self._nudged = False
                self._kept_flagged = False
                self._pb_flagged = False
                if self._hold_adv is not None and self.s > self._hold_adv['s'] + 0.05:
                    self._hold_adv = None
                if abs(self.off_tgt) < 1e-6:
                    self.look_off = None
            if free == math.inf and abs(self.off_tgt) > 1e-6 and now - self._t_search >= pr.avoid_period_s \
                    and self._rc is None:
                self._t_search = now
                back, unseen = self._try_return(pos, L, P, valid, seen)
                self.look_off = None
                if back is not None:
                    self.off_tgt = back
                elif unseen is not None:
                    self._set_look(pos, unseen)
        if vio_clip is not None and not blocked and self.S_goal > self.s + 1e-6:
            self._clip_end(*vio_clip)
        if blocked:
            if self.blocked_since is None:
                self.blocked_since = now
        else:
            self.blocked_since = None
        # lateral offset slew
        d_off = float(np.clip(self.off_tgt - self.off, -pr.v_side * dt, pr.v_side * dt))
        self.off += d_off
        # carrot integration
        dv = float(np.clip(v_t - self.v, -pr.a_brake * dt, a * dt))
        acc = dv / dt if dt > 0.0 else 0.0
        self.v = max(0.0, self.v + dv)
        s_new = min(self.S, self.s + self.v * dt)
        ts = self._tight_stop(self.s)
        if ts is not None and s_new >= ts[0]:       # never past an unsettled tight corner
            s_new, self.v, acc = ts[0], 0.0, 0.0
        self.s = s_new
        if self.s >= self.S - 1e-9:
            self.s, self.v, acc = self.S, 0.0, 0.0
        if pr.veh_brake_mps2 > 0 and vel is not None and self.avoid is not None:
            # never leave the carrot behind where the vehicle itself will stop (official run_002: a hold set v 0 at
            # f 1.20 while the vehicle flew 0.55 m/s -> 0.5 m overshoot into the wall, then oscillation)
            sp_ = self.project(pos)
            va = float(np.asarray(vel, float)[:2] @ self.tangent(sp_)[:2])
            if va > 0.1 and v_t <= 0.05 and self.v < va - 0.1:   # stopping harder than the vehicle can follow
                s_veh = sp_ + va * va / (2.0 * pr.veh_brake_mps2) + va * pr.veh_lag_s
                s_car = self.s + self.v * self.v / (2.0 * pr.veh_brake_mps2)
                if s_veh > s_car + 0.05:
                    lim = self.S
                    ts2 = self._tight_stop(sp_)
                    if ts2 is not None:
                        lim = min(lim, ts2[0])
                    if math.isfinite(free):           # (free: probe distance from the vehicle)
                        lim = min(lim, sp_ + max(0.0, free - pr.stop_margin_m))
                    s_fl = max(sp_, min(s_veh, lim))
                    if s_fl > self.s + 1e-6:
                        if not getattr(self, '_floor_flagged', False):
                            self._floor_flagged = True
                            self.flags.append(('carrot_floor', {'vehicle_v': round(va, 2),
                                                                'moved_m': round(s_fl - self.s, 3)}))
                        self.s, self.v, acc = s_fl, 0.0, 0.0
        ref = self.ref(self.s)
        t = self.tangent(self.s)
        for k in self._tight:                       # around a tight corner: feed-forward along the leg itself (the
            sc = float(self.cum[k])                 # smoothed tangent would push toward the wall beyond the corner)
            if sc - 0.3 <= self.s < sc - 1e-9 and k not in self._settled:
                t = self.D[k - 1]
            elif sc - 1e-9 <= self.s <= sc + 0.3 and k in self._settled:
                t = self.D[k]
        vel = self.v * t + (d_off / dt if dt > 0.0 else 0.0) * self.normal(self.s)
        if self.avoid in ('lateral', 'stop') and ahead is not None and pr.ff_stop_decel > 0:
            sv = float(np.linalg.norm(vel))
            if sv > 0.05:                           # never feed forward toward a wall closer than stop distance + props
                dw = ahead(pos, vel / sv, pr.prop_radius_m, self.ignored)
                if dw is not None and math.isfinite(dw):
                    room = max(0.0, dw - pr.prop_radius_m - pr.ff_wall_margin_m)
                    a_, lg = pr.ff_stop_decel, pr.veh_lag_s      # v^2/(2a) + v*lag <= room
                    vmax = max(0.0, -a_ * lg + math.sqrt((a_ * lg) ** 2 + 2.0 * a_ * room))
                    if sv > vmax:
                        vel = vel * (vmax / sv)
                        self.v = min(self.v, vmax)
                        acc = min(acc, 0.0)
        if self.avoid in ('lateral', 'stop') and pr.acc_ff_max > 0:
            acc = float(np.clip(acc, -pr.acc_ff_max, pr.acc_ff_max))
        sp = (ref + nudge_d, vel + nudge_v, acc * t, self.heading_cmd())
        if self.blocked_since is not None:
            status = 'stuck' if now - self.blocked_since > pr.blocked_timeout_s else 'blocked'
            if self._sticky and not self._sticky_flagged:
                self._sticky_flagged = True
                st = self._sticky[0]
                self.flags.append(('start_sticky_blocked', {'center': st['c'].copy(), 'radius_m': round(st['r'], 3),
                                                            'n': st['n'], 'support': st['support']}))
        elif self.pass_through and self.project(pos) >= self.S_goal - pr.pass_tol_m:
            status = 'passed'
        elif self.s >= self.S - 1e-9 and np.linalg.norm(pos - ref) <= self.tol:
            status = 'arrived'
        else:
            status = 'moving'
        return Step(sp, status, free)

    @property
    def end_ref(self):
        return self.ref(self.S)


# --------------------------------------------------------------- slow descent
class Descent:
    """Slow descent at a fixed xy; touchdown = ToF near zero AND PX4 landed, held."""

    def __init__(self, xy, z0, surface_z=None, pose_only_gain=None):
        self.xy = np.asarray(xy, float).reshape(-1)[:2].copy()
        self.z_sp = float(z0)
        self.surface_z = surface_z
        self.touch_since = None
        self.contact_since = None              # ToF says we are on the surface (PX4 may not agree yet)
        self.pose_only_gain = pose_only_gain   # pilot_mode: no feed-forward, the setpoint lead sets the speed

    def step(self, dt, now, pos, landed, tof, pr: Params):
        contact = tof is not None and tof <= pr.tof_touch_m
        near = (tof is not None and tof < pr.tof_slow_m) or \
               (self.surface_z is not None and pos[2] > self.surface_z - pr.slow_above_m)
        rate = pr.v_touch if near else pr.v_descent
        z_next = self.z_sp + rate * dt
        # never wind the setpoint far below the vehicle, but keep the velocity feed-forward at `rate` even when
        # capped: PX4's land detector needs a commanded descent >= 1.1 * LNDMC_Z_VEL_MAX to see ground contact
        push = pr.push_contact_m if contact else pr.push_m
        if self.pose_only_gain:                # pose only: lead the vehicle by rate / gain so it really moves at rate
            push = max(push, rate / self.pose_only_gain)
        cap = float(pos[2]) + push
        vz = rate
        self.z_sp = max(self.z_sp, min(z_next, cap))   # never RISES (a vehicle sitting above its setpoint must not
                                                       # drag 'pos + push' upward)
        if contact:
            if self.contact_since is None:
                self.contact_since = now
        else:
            self.contact_since = None
        touch = contact and bool(landed)
        if touch:
            if self.touch_since is None:
                self.touch_since = now
        else:
            self.touch_since = None
        done = touch and now - self.touch_since >= pr.touch_hold_s
        sp = (np.array([self.xy[0], self.xy[1], self.z_sp]), np.array([0.0, 0.0, vz]), np.zeros(3), None)
        return sp, done


# ------------------------------------------------------------------- mission
class Mission:
    def __init__(self, prior: CoursePrior, params: Optional[Params] = None, autostart: bool = True):
        self.prior = prior
        self.p = params or Params()
        self.started = bool(autostart)
        if self.p.v_touch < 1.2 * self.p.px4_lndmc_z_vel_max:
            raise ValueError('Params.v_touch=%.2f must be >= 1.2 * px4_lndmc_z_vel_max (%.2f): PX4 only detects '
                             'ground contact while the commanded descent is >= 1.1 * LNDMC_Z_VEL_MAX'
                             % (self.p.v_touch, 1.2 * self.p.px4_lndmc_z_vel_max))
        if self.p.stop_after not in ('',) + STOP_AFTER_PHASES:
            raise ValueError('Params.stop_after=%r: use one of %s (or "" for the full mission)'
                             % (self.p.stop_after, ', '.join(STOP_AFTER_PHASES)))
        if self.p.fence_margin_m < self._trip_inset() + 0.1:
            raise ValueError('Params.fence_margin_m=%.2f must be >= the net trip line (%.2f) + 0.1'
                             % (self.p.fence_margin_m, self._trip_inset()))
        F, m = prior.fence, self.p.fence_margin_m
        if min(-F['forward_min'], F['forward_max'], -F['left_min'], F['left_max']) < m + 0.05:
            raise ValueError('course prior: the start is closer than fence_margin_m + 0.05 = %.2f m to the fence; '
                             'move the start or reduce Params.fence_margin_m' % (m + 0.05))
        self.phase, self.sub = 'wait', ''
        self.t_phase = None
        self.t_last = None
        self.t_arm = None
        self.sf: Optional[StartFrame] = None
        self.home = None
        self.home_yaw = None
        self.g: Dict[str, Any] = {}
        self.yaw_sp = None
        self.history: List[tuple] = []     # (t, phase, reason)
        self.abort_reason = None
        self.release_reason = None
        self._cmd_t: Dict[str, float] = {}
        self._pose_bad_since = None
        self._pose_ok_since = None
        self._stream_since = None
        self._seen_offboard = False
        self._nav_at_disarm = None
        self._t_disarmed = None
        self._hold = None
        self._last_sp = None
        self._fol: Optional[Follower] = None
        self._latch: Optional[Latch] = None
        self._desc: Optional[Descent] = None
        self._stamps: Dict[str, float] = {}
        self._latched: Dict[str, dict] = {}
        self._once = set()
        self._face = None
        self._z_over = None
        self._t_land_req = None
        self._t_desc = None
        self._handoff_land = False
        self._stopped_after = None
        self._rearm_armed_since = None
        self._miss_run = 0
        self._miss_reasons: Dict[str, int] = {}
        self._reacq = {}                         # stage -> attempts
        self._contacts = {}                      # phase -> prop strikes detected
        self._contact = None                     # recovery state after a strike
        self._stage_dir: Dict[str, np.ndarray] = {}   # stage -> approach direction after a re-acquire
        self._feed_outcome = None
        self._t_align = None
        self._commit_since = None
        self._plane_logged = False
        self._standoffs: Dict[str, np.ndarray] = {}
        self._views: List[tuple] = []           # (t, camera_ned, heading) for "seen recently"
        self._view_arr = None
        self._motion: List[tuple] = []          # (t, pos, speed, heading): hover check for slow detections
        self._det_stamps: Dict[str, List[float]] = {}   # recent fresh detection stamps per kind (detector rate)
        self._t_conf = -math.inf                # map points confirmed from two viewpoints (cell keys, sorted)
        self._conf_k = np.zeros(0, np.int64)
        self._conf_c = np.zeros((0, 3))
        self._conf_h = np.zeros(0)
        self._conf_t = np.zeros(0)
        self._conf_m = np.zeros(0, bool)
        self._phantom_memo: Dict[tuple, tuple] = {}
        self._float_info: Dict[tuple, dict] = {}
        self._exclude: Dict[str, list] = {}     # latched opening interiors, per stage
        self._excl_ok: Dict[type, bool] = {}

    def start(self):
        """Operator 'go' when constructed with autostart=False."""
        self.started = True

    # -------------------------------------------------------------- plumbing
    def step(self, now: float, vehicle: Vehicle, perception: Optional[Perception] = None) -> Output:
        p = self.p
        now = float(now)
        self.dt = 0.0 if self.t_last is None else min(max(now - self.t_last, 0.0), p.dt_max_s)
        self.t_last = now
        if self.t_phase is None:
            self.t_phase = now
            self.history.append((now, self.phase, 'start'))
        self.now, self.veh = now, vehicle
        self.per = perception if perception is not None else Perception()
        self.pos = np.asarray(vehicle.pos_ned, float).reshape(3)
        self.vel = np.asarray(vehicle.vel_ned, float).reshape(3)
        self._cmds: List[Command] = []
        self._events: List[dict] = []
        self._hints: Dict[str, Any] = {}
        self._record_view()
        self._record_motion()
        self._confirm_update()
        sp = None
        if self.phase not in TERMINAL_PHASES:
            handled, sp = self._guard()
            if not handled and self.phase not in TERMINAL_PHASES:
                sp = getattr(self, '_ph_' + self.phase)()
        out = Output(phase=self.phase, sub=self.sub, commands=self._cmds, events=self._events)
        if sp is not None and self.phase not in TERMINAL_PHASES:
            pos_sp, vel_sp, acc_sp, yaw_t = sp
            pos_sp = np.asarray(pos_sp, float).reshape(3).copy()
            vel_sp = np.asarray(vel_sp, float).reshape(3).copy()
            acc_sp = np.asarray(acc_sp, float).reshape(3).copy()
            if self.sf is not None and (self.phase in FLIGHT_PHASES or self.phase == ABORT_PHASE):
                pos_sp, vel_sp, acc_sp = self._clip_fence(pos_sp, vel_sp, acc_sp)
            if self.sf is not None and self.phase in ('takeoff', 'corridor'):
                pos_sp, vel_sp, acc_sp = self._clip_start_rear(pos_sp, vel_sp, acc_sp)
            if self.p.pilot_mode:                 # the skybeta pose clamp, so our state matches what PX4 gets
                pos_sp[2] = min(max(pos_sp[2], self._pilot_ceiling_z()), self._pilot_floor_z())
            out.pos_ned, out.vel_ned, out.acc_ned = pos_sp, vel_sp, acc_sp
            out.yaw = self._slew_yaw(yaw_t)
            self._last_sp = pos_sp
        else:
            self.yaw_sp = None
        out.phase, out.sub = self.phase, self.sub
        out.hints = self._make_hints()
        return out

    def _start_f(self, q):
        """Forward offset of q from the take-off point (start heading) while the vehicle is still in the start zone
        of the corridor phase and q is in the start lane (start_rear_half_m sideways), else None."""
        p, sf = self.p, self.sf
        if sf is None or self.phase != 'corridor' or getattr(self, '_left_start', False):
            return None
        x = np.asarray(q, float)[:2] - sf.o[:2]
        if abs(float(x @ sf.left[:2])) > p.start_rear_half_m:
            return None
        return float(x @ sf.fwd[:2])

    def _clip_start_rear(self, pos, vel, acc):
        """(v4H) Never command a position more than start_back_max_m behind the take-off point (start heading)
        within start_rear_half_m sideways of it (the barrier behind the start is never seen), and -- until the vehicle
        first leaves start_zone_m -- never back below the most forward commanded position (<= start_ratchet_m)."""
        p, sf = self.p, self.sf
        if p.start_back_max_m is None:
            return pos, vel, acc
        if float(math.hypot(*(self.pos[:2] - sf.o[:2]))) > p.start_zone_m:
            self._left_start = True
        x = pos[:2] - sf.o[:2]
        f, l = float(x @ sf.fwd[:2]), float(x @ sf.left[:2])
        if abs(l) > p.start_rear_half_m:
            return pos, vel, acc
        floor = -p.start_back_max_m
        if p.start_ratchet_m and not getattr(self, '_left_start', False):
            fm = getattr(self, '_start_fmax', 0.0)
            floor = max(floor, min(fm, p.start_ratchet_m))
            self._start_fmax = max(fm, min(f, p.start_ratchet_m))
        if f >= floor - 1e-6:
            return pos, vel, acc
        pos = pos + (floor - f) * sf.fwd
        for v in (vel, acc):
            b = float(v[:2] @ sf.fwd[:2])
            if b < 0.0:
                v -= b * sf.fwd
        if not getattr(self, '_rear_clip_flagged', False):
            self._rear_clip_flagged = True
            self._event('start_rear_clip', wanted_f=round(f, 3), held_f=round(floor, 3))
        return pos, vel, acc

    def _event(self, kind, **kw):
        e = {'t': self.now, 'phase': self.phase, 'kind': kind}
        for k, v in kw.items():
            e[k] = np.round(np.asarray(v, float), 3).tolist() if isinstance(v, np.ndarray) else v
        self._events.append(e)

    def _event_once(self, key, kind, **kw):
        if key not in self._once:
            self._once.add(key)
            self._event(kind, **kw)

    def _cmd(self, c: Command):
        if self.p.pilot_mode:                     # skybeta: no command channel exists
            return
        last = self._cmd_t.get(c.name)
        if last is None or self.now - last >= self.p.cmd_retry_s:
            self._cmd_t[c.name] = self.now
            self._cmds.append(c)

    def _set_phase(self, name, reason=''):
        stop = self.p.stop_after
        if stop and self.phase == stop and name != stop and name not in (ABORT_PHASE,) + TERMINAL_PHASES:
            return self._enter_staged_stop(reason)   # (left the stage normally, whichever stage comes next)
        self._event('phase', to=name, reason=reason)
        self.history.append((self.now, name, reason))
        self.phase, self.sub, self.t_phase = name, '', self.now
        self._fol = None
        self._desc = None
        self._handoff_land = False

    def _enter_staged_stop(self, reason):
        prev = self.phase
        self._stopped_after = prev
        self._event('phase', to='staged_stop', reason='stop_after_%s (%s)' % (prev, reason))
        self.history.append((self.now, 'staged_stop', 'stop_after_' + prev))
        self.phase, self.t_phase = 'staged_stop', self.now
        self._fol = None
        self._desc = None
        self._handoff_land = False
        self.sub = 'disarm' if prev == 'land_target' else 'hover'   # on the rock already (ToF + landed confirmed)
        self._hold = self.pos.copy()
        self._event('staged_stop', after=prev, hover_s=self.p.stop_hover_s if self.sub == 'hover' else 0.0,
                    land='px4' if self.p.stop_land_px4 else 'mission')

    def _finish_staged_stop(self):
        prev = self._stopped_after
        self._set_phase('done', 'stopped_after_' + prev)
        self.sub = 'stopped_after_' + prev
        self._event('staged_stop_complete', after=prev, t_flight=round(self.now - self.t_arm, 2), pos=self.pos)

    def _ph_staged_stop(self):
        p, v = self.p, self.veh
        if self.sub == 'hover':
            if self.now - self.t_phase < p.stop_hover_s:
                return self._hold_sp(self._hold)
            if p.stop_land_px4 and not p.pilot_mode:
                self.sub = 'px4_land'
                self._handoff_land = True
            else:
                self.sub = 'descend'
                self._desc = Descent(self.pos[:2], float(self.pos[2]), surface_z=float(self.home[2]),
                                     pose_only_gain=p.pilot_pos_gain if p.pilot_mode else None)
                self._t_desc = self.now
        if self.sub == 'px4_land':                # until PX4 switches; then the guard keeps us silent
            self._cmd(cmd_land())
            return self._hold_sp(self._hold)
        if p.pilot_mode and self.sub in ('descend', 'pilot_hold'):
            return self._pilot_landing_step(self._desc.xy)
        if self.sub == 'descend':
            if self.now - self._t_desc > p.descend_timeout_s:
                return self._abort('no_touchdown_confirmation_staged_stop', tof_m=self._tof(), landed=bool(v.landed))
            sp, touched = self._desc.step(self.dt, self.now, self.pos, v.landed, self._tof(), p)
            if not touched:
                self._maybe_handoff()
                return sp
            self.sub = 'disarm'
            self._event('touchdown_staged_stop', pos=self.pos, tof_m=self._tof())
        if v.armed:
            self._cmd(cmd_disarm())
            return self._press_sp()
        self._finish_staged_stop()
        return None

    def _press_sp(self, contact=True):
        """On a surface and still armed: keep pressing down with a commanded descent so PX4 stays 'landed'."""
        push = self.p.push_contact_m if contact else self.p.push_m
        return (self.pos + np.array([0.0, 0.0, push]), np.array([0.0, 0.0, self.p.v_touch]), np.zeros(3), None)

    def _maybe_handoff(self):
        """ToF has shown contact for contact_handoff_s but PX4 never reported landed: ask PX4 LAND to finish."""
        p, d = self.p, self._desc
        if p.contact_handoff_s is None or d is None or d.contact_since is None or self.veh.landed:
            return
        if self.now - d.contact_since >= p.contact_handoff_s:
            if not self._handoff_land:
                self._handoff_land = True
                self._event('touchdown_handoff_to_px4', tof_m=self._tof(), contact_s=round(self.now - d.contact_since, 2))
            self._cmd(cmd_land())

    def _hold_sp(self, pos, yaw=None):
        return (np.asarray(pos, float).copy(), np.zeros(3), np.zeros(3), yaw)

    def _slew_yaw(self, target):
        if self.yaw_sp is None:
            self.yaw_sp = float(self.veh.heading)
        if target is None:
            return self.yaw_sp
        if self.phase in ('wait', 'arm', 'disarm', 'rearm'):
            self.yaw_sp = wrap(float(target))
        else:
            rate = self.p.yaw_rate_near if self._near_obstacle() else self.p.yaw_rate
            r = min(self.p.yaw_rate, rate) * self.dt
            err = wrap(float(target) - self.yaw_sp)
            sgn = self._yaw_side(float(target), err)
            if sgn is not None and (sgn > 0) != (err > 0):
                err = err + sgn * 2.0 * math.pi         # the long way round, through open space
            self.yaw_sp = wrap(self.yaw_sp + float(np.clip(err, -r, r)))
        return self.yaw_sp

    def _yaw_side(self, target, err):
        """Big yaw (> 100 deg): the turn direction whose headings keep the camera off near walls (official run_009:
        VIO drifted 0.7 m turning toward the tunnel next to the north wall).  Decided once per target."""
        p = self.p
        if self._fol is not None and getattr(self._fol, '_rc', None) is not None:
            return None                           # (re-confirmation scan: the short way, past the cluster)
        if not p.yaw_open_side or abs(err) <= math.radians(100.0) or self.sf is None or \
                self.phase not in FLIGHT_PHASES:
            if abs(err) <= math.radians(20.0):
                self._yaw_dir = None
            return getattr(self, '_yaw_dir', (None, None))[1] if getattr(self, '_yaw_dir', None) and \
                abs(wrap(self._yaw_dir[0] - target)) < 0.05 else None
        cur = getattr(self, '_yaw_dir', None)
        if cur is not None and abs(wrap(cur[0] - target)) < 0.05:
            return cur[1]
        best = None
        for sg in (1.0, -1.0):
            sweep = abs(err) if (sg > 0) == (err > 0) else 2.0 * math.pi - abs(err)
            worst = math.inf
            for k in range(1, int(sweep / math.radians(15.0)) + 1):
                h = self.yaw_sp + sg * k * math.radians(15.0)
                hd = np.array([math.cos(h), math.sin(h), 0.0])
                f = self._ahead(self.pos + p.cam_forward_m * hd, hd, 0.25)
                f = math.inf if f is None else float(f)
                if self.sf is not None:
                    f = min(f, self._fence_ahead(self.pos, hd))
                worst = min(worst, f)
            sc = (min(worst, 1.5), -sweep)         # clear of walls first (up to 1.5 m), then the shorter turn
            if best is None or sc > best[0]:
                best = (sc, sg)
        self._yaw_dir = (target, best[1])
        if (best[1] > 0) != (err > 0):
            self._event('yaw_long_way', target_deg=round(math.degrees(target), 1), clear_m=round(best[0][0], 2))
        return best[1]

    def _near_obstacle(self):
        """A mapped obstacle (flight band) or the net within yaw_near_m of the vehicle (cached 0.2 s)."""
        p = self.p
        if self.now - getattr(self, '_t_near', -math.inf) < 0.2:
            return self._near
        self._t_near = self.now
        near = False
        if self.sf is not None and self.phase in FLIGHT_PHASES:
            near = self._fence_dist(self.pos) < p.yaw_near_m
            om = self.per.obstacles
            pts = getattr(om, 'points', None) if om is not None else None
            if not near and isinstance(pts, np.ndarray) and pts.ndim == 2 and len(pts):
                x, y = pts[:, 0] - self.pos[0], pts[:, 1] - self.pos[1]
                sel = (np.abs(x) < p.yaw_near_m) & (np.abs(y) < p.yaw_near_m) & \
                    (np.abs(pts[:, 2] - self.pos[2]) < p.snap_band_m)
                near = bool(sel.any() and float(np.min(x[sel] ** 2 + y[sel] ** 2)) < p.yaw_near_m ** 2)
        self._near = near
        return near

    def _make_hints(self):
        h = dict(self._hints)
        h.setdefault('detector', DETECTOR_FOR_PHASE.get(self.phase))
        if self.g:
            h['home_ned'] = self.home.copy()
            h['tunnel_prior_entry_ned'] = self.g['tunnel_entry'].copy()
            h['window_prior_center_ned'] = self.g['window_center'].copy()
            h['landing_prior_ned'] = self.g['rock'].copy()
            h['wall_prior_top_d'] = float(self.home[2] - self.prior.wall_top_up)
        return h

    # -------------------------------------------------------------- geometry
    def _z_up(self, up):
        return float(self.home[2] - up)

    def _ceiling_z(self):
        return self._z_up(self.prior.fence['up_max'] - self.p.fence_margin_m)

    def _clamp_alt(self, z):
        return float(min(max(z, self._ceiling_z()), self._z_up(self.p.min_alt_m)))

    def _inside(self, pos, inset, inset_up=None):
        f, l, u = self.sf.from_ned(pos)
        F = self.prior.fence
        iu = inset if inset_up is None else inset_up
        return (F['forward_min'] + inset <= f <= F['forward_max'] - inset and
                F['left_min'] + inset <= l <= F['left_max'] - inset and u <= F['up_max'] - iu)

    def _clip_pos(self, pos):
        f, l, u = self.sf.from_ned(pos)
        F, m = self.prior.fence, self.p.fence_margin_m
        f2 = min(max(f, F['forward_min'] + m), F['forward_max'] - m)
        l2 = min(max(l, F['left_min'] + m), F['left_max'] - m)
        u2 = min(u, F['up_max'] - m)
        changed = abs(f2 - f) + abs(l2 - l) + abs(u2 - u) > 1e-6
        return (self.sf.to_ned((f2, l2, u2)) if changed else np.asarray(pos, float)), changed

    def _clip_fence(self, pos, vel, acc):
        q, changed = self._clip_pos(pos)
        if changed:
            self._event_once(('fence_clip', self.phase), 'fence_clip', wanted=pos)
            return q, np.zeros(3), np.zeros(3)
        return pos, vel, acc

    def _in_control(self):
        v = self.veh
        if self.p.pilot_mode and v.offboard_active is not None:
            return bool(v.offboard_active)
        return v.nav_state == NAV_OFFBOARD

    def _pilot_floor_z(self):
        return float(self.veh.clamp_origin_d) - self.p.pilot_min_up_m

    def _pilot_ceiling_z(self):
        return float(self.veh.clamp_origin_d) - self.p.pilot_max_up_m

    def _pilot_landing_step(self, xy):
        """pilot_mode landing: descend (ToF-gated speeds) but never below the pose clamp; once in ToF contact or
        settled at the clamp, hold there and ask the pilot to land.  PX4 / the pilot disarms; the guard reacts."""
        p, v = self.p, self.veh
        z_floor = self._pilot_floor_z()
        if self.sub == 'descend':
            sp, _ = self._desc.step(self.dt, self.now, self.pos, v.landed, self._tof(), p)
            self._desc.z_sp = min(self._desc.z_sp, z_floor)
            at_floor = self.pos[2] >= z_floor - p.pilot_floor_settle_m and abs(float(self.vel[2])) < 0.1
            d = self._desc
            contact = d.contact_since is not None and self.now - d.contact_since >= p.touch_hold_s
            timed_out = self.now - self._t_desc > p.descend_timeout_s
            if not (at_floor or contact or timed_out):
                pos_sp = sp[0].copy()
                pos_sp[2] = min(pos_sp[2], z_floor)
                return (pos_sp, sp[1], sp[2], None)
            self.sub = 'pilot_hold'
            d.hold_z = min(float(d.z_sp), z_floor)  # continue from the descent setpoint: no step up at the hand-over
            self._event('pilot_land_requested', why='contact' if contact else 'at_clamp' if at_floor else 'timeout',
                        tof_m=self._tof(), up_from_origin=round(float(v.clamp_origin_d - self.pos[2]), 3),
                        above_sp_m=round(float(d.z_sp - self.pos[2]), 3))
        d = self._desc
        z_new = min(z_floor, float(self.pos[2]) + p.push_m)   # press toward the clamp, never a far setpoint ...
        prev = getattr(d, 'hold_z', None)                     # ... and NEVER up: ramp down >= pilot_hold_descend_v
        z_hold = z_new if prev is None else min(z_floor, max(prev + p.pilot_hold_descend_v * self.dt, z_new))
        d.hold_z = z_hold
        return (np.array([xy[0], xy[1], z_hold]), np.zeros(3), np.zeros(3), None)

    def _trip_inset(self):
        return max(self.p.fence_inset_m, self.p.half_span_m + self.p.fence_trip_margin_m)

    def _clear_of_fence(self, q):
        return self._inside(q, max(self.p.standoff_fence_clear_m, self.p.fence_margin_m) - 1e-6)

    def _pull_in(self, q):
        """Move q (start-frame forward/left/up) into the box standoff_fence_clear_m inside the fence."""
        f, l, u = self.sf.from_ned(q)
        F, m = self.prior.fence, max(self.p.standoff_fence_clear_m, self.p.fence_margin_m)
        f = min(max(f, F['forward_min'] + m), F['forward_max'] - m)
        l = min(max(l, F['left_min'] + m), F['left_max'] - m)
        u = min(u, F['up_max'] - self.p.fence_margin_m)
        return self.sf.to_ned((f, l, u))

    def _valid(self, q):
        return self._inside(q, self.p.fence_margin_m - 1e-6)

    # ---- camera view history ("has the camera seen this point recently?")
    def _record_view(self):
        p, h = self.p, float(self.veh.heading)
        if self._views and self.now - self._views[-1][0] < 0.1:
            return
        cam = self.pos + p.cam_forward_m * np.array([math.cos(h), math.sin(h), 0.0])
        self._views.append((self.now, cam, h))
        keep = max(p.seen_memory_s, p.seen_near_memory_s, p.seen_free_s)
        self._views = [v for v in self._views if self.now - v[0] <= keep]
        self._view_arr = (np.array([v[1] for v in self._views]), np.array([v[2] for v in self._views]),
                          np.array([v[0] for v in self._views]))
        if self.phase in FLIGHT_PHASES or self.phase == ABORT_PHASE:
            tr = getattr(self, '_trace', None)
            if tr is None:
                tr = self._trace = []
            if not tr or self.now - tr[-1][0] >= 0.1:
                tr.append((self.now, self.pos.copy()))
                while tr and self.now - tr[0][0] > p.trace_s:
                    tr.pop(0)
                self._trace_arr = np.array([x[1] for x in tr])

    def _count_near(self, c, r):
        """Map points within r of c (active exclusions applied); None if the map has no .points."""
        om = self.per.obstacles
        pts = getattr(om, 'points', None) if om is not None else None
        if not (isinstance(pts, np.ndarray) and pts.ndim == 2 and pts.shape[1] == 3):
            return None
        c = np.asarray(c, float)
        sel = (np.abs(pts[:, 0] - c[0]) <= r) & (np.abs(pts[:, 1] - c[1]) <= r) & (np.abs(pts[:, 2] - c[2]) <= r)
        Q = pts[sel]
        ex = self._active_exclude()
        if ex and len(Q):
            Q = Q[self._outside(Q, ex)]
        return int(np.count_nonzero(np.linalg.norm(Q - c, axis=1) <= r)) if len(Q) else 0

    def _on_track(self, q):
        """q is where the vehicle itself recently was (on_track_m sideways, 0.3 m in height)."""
        T = getattr(self, '_trace_arr', None)
        if T is None or not len(T):
            return False
        d = T - np.asarray(q, float)
        return bool(np.any((d[:, 0] ** 2 + d[:, 1] ** 2 < self.p.on_track_m ** 2) & (np.abs(d[:, 2]) < 0.3)))

    def _seen_free(self, q):
        """q may be entered: seen in the camera FOV cone within seen_free_s, on our own recent track, or (stereo
        blind zone near the camera) seen within the map memory."""
        p = self.p
        q = np.asarray(q, float)
        T = getattr(self, '_trace_arr', None)
        if T is not None and len(T):
            d = T - q
            if bool(np.any((d[:, 0] ** 2 + d[:, 1] ** 2 < p.trace_free_m ** 2) & (np.abs(d[:, 2]) < 0.3))):
                return True
        if self._seen(q, p.seen_free_s):
            return True
        h = float(self.veh.heading)
        cam = self.pos + p.cam_forward_m * np.array([math.cos(h), math.sin(h), 0.0])
        return math.hypot(*(q - cam)[:2]) < p.seen_near_m and self._seen(q, p.seen_near_memory_s)

    def _seen(self, q, memory=None):
        if self._view_arr is None:
            return False
        p = self.p
        C, H, T = self._view_arr
        m = self.now - T <= (p.seen_memory_s if memory is None else memory)
        if not m.any():
            return False
        C, H = C[m], H[m]
        d = np.asarray(q, float)[None, :] - C
        hd = np.hypot(d[:, 0], d[:, 1])
        bearing = np.arctan2(d[:, 1], d[:, 0]) - H
        bearing = (bearing + math.pi) % (2.0 * math.pi) - math.pi
        elev = np.arctan2(-d[:, 2], np.maximum(hd, 1e-6))
        ok = ((hd >= p.seen_min_range_m) & (hd <= p.seen_range_m) & (np.abs(bearing) <= math.radians(p.seen_half_fov_deg))
              & (np.abs(elev) <= math.radians(p.seen_half_vfov_deg)))
        return bool(ok.any())

    def _record_motion(self):
        v = float(np.linalg.norm(self.vel))
        self._motion.append((self.now, self.pos.copy(), v if math.isfinite(v) else math.inf, float(self.veh.heading)))
        t0 = self.now - self.p.detect_max_latency_s - 0.5
        while self._motion and self._motion[0][0] < t0:
            self._motion.pop(0)

    def _hovered_since(self, t0, face_yaw=None, zone=None):
        """Hover gate over [t0, now]: slow, inside the detect zone and facing the stage the whole time (a slow
        detector's frame is still about where we are now)."""
        p, M = self.p, self._motion
        if not M or M[0][0] > t0 + 0.1:
            return False                          # no history back to the image
        for t, q, v, h in M:
            if t < t0 - 0.1:
                continue
            if v > p.detect_speed_max:
                return False
            if zone is not None:
                z_pt, z_dir = zone
                if np.linalg.norm(q - z_pt) > p.detect_zone_m or (q - z_pt) @ z_dir > p.detect_ahead_m:
                    return False
            if face_yaw is not None and abs(wrap(face_yaw - h)) > math.radians(p.detect_yaw_tol_deg):
                return False
        return True

    # ---- map points confirmed from two viewpoints (the node carves what a new viewpoint sees through)
    @staticmethod
    def _cell_keys(q, cell=0.1):
        ijk = np.floor(np.asarray(q, float) / cell).astype(np.int64) + (1 << 20)
        return (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]

    def _confirm_update(self):
        """Every confirm_period_s: map cells in the camera view now; a cell still in the map when the camera has
        moved >= confirm_baseline_m (or turned the bearing to it >= confirm_angle_deg) since it was first in view
        was seen from two viewpoints without being carved."""
        p = self.p
        if self.now - self._t_conf < p.confirm_period_s:
            return
        self._t_conf = self.now
        if len(self._conf_t):
            keep = self.now - self._conf_t <= p.confirm_memory_s
            if not keep.all():
                self._conf_k, self._conf_c = self._conf_k[keep], self._conf_c[keep]
                self._conf_h, self._conf_t, self._conf_m = self._conf_h[keep], self._conf_t[keep], self._conf_m[keep]
        om = self.per.obstacles
        pts = getattr(om, 'points', None) if om is not None else None
        h = float(self.veh.heading)
        if not (isinstance(pts, np.ndarray) and pts.ndim == 2 and pts.shape[1] == 3 and len(pts)) or \
                not math.isfinite(h):
            return
        ch, sh = math.cos(h), math.sin(h)
        cam = self.pos + p.cam_forward_m * np.array([ch, sh, 0.0])
        R = p.seen_range_m                        # (box prefilter, comparisons only: cheap on a 60k-point map)
        x, y = pts[:, 0] - cam[0], pts[:, 1] - cam[1]
        box = (np.abs(x) <= R) & (np.abs(y) <= R)
        if not box.any():
            return
        x, y, zz, q = x[box], y[box], pts[box, 2] - cam[2], pts[box]
        hd = np.hypot(x, y)
        vis = (hd >= p.seen_min_range_m) & (hd <= R) & \
            (x * ch + y * sh >= math.cos(math.radians(p.seen_half_fov_deg)) * hd) & \
            (np.abs(zz) <= math.tan(math.radians(p.seen_half_vfov_deg)) * hd)
        if not vis.any():
            return
        q = q[vis]
        uk, idx = np.unique(self._cell_keys(q), return_index=True)
        bu = np.arctan2(q[idx, 1] - cam[1], q[idx, 0] - cam[0])
        K = self._conf_k
        if len(K):
            j = np.minimum(np.searchsorted(K, uk), len(K) - 1)
            found = K[j] == uk
        else:
            j, found = np.zeros(len(uk), int), np.zeros(len(uk), bool)
        if found.any():
            fi = j[found]
            dc = self._conf_c[fi] - cam
            moved = np.linalg.norm(dc[:, :2] if p.confirm_horizontal else dc, axis=1) >= p.confirm_baseline_m
            turned = np.abs((bu[found] - self._conf_h[fi] + math.pi) % (2.0 * math.pi) - math.pi) >= \
                math.radians(p.confirm_angle_deg)
            self._conf_m[fi] |= moved | turned
            self._conf_t[fi] = self.now
        new = ~found
        if new.any():
            n = int(new.sum())
            K2 = np.concatenate([K, uk[new]])
            order = np.argsort(K2, kind='stable')
            self._conf_k = K2[order]
            self._conf_c = np.vstack([self._conf_c, np.repeat(cam[None, :], n, axis=0)])[order]
            self._conf_h = np.concatenate([self._conf_h, bu[new]])[order]
            self._conf_t = np.concatenate([self._conf_t, np.full(n, self.now)])[order]
            self._conf_m = np.concatenate([self._conf_m, np.zeros(n, bool)])[order]

    def _remembered(self, lo, hi, z, band=None):
        """(N,3) centres of map cells seen in view within confirm_memory_s, in the xy box lo..hi and the flight band
        around z (+- band, default snap_band_m; lane centring only; the obstacle probe uses the live map)."""
        K = self._conf_k
        if not len(K):
            return np.zeros((0, 3))
        if getattr(self, '_rem_t', None) != self._t_conf or len(self._rem_c) != len(K):
            off = 1 << 20
            i = (K >> 42) - off
            j = ((K >> 21) & ((1 << 21) - 1)) - off
            kk = (K & ((1 << 21) - 1)) - off
            self._rem_c = (np.stack([i, j, kk], axis=1).astype(float) + 0.5) * 0.1
            self._rem_t = self._t_conf
        C = self._rem_c
        sel = (C[:, 0] >= lo[0]) & (C[:, 0] <= hi[0]) & (C[:, 1] >= lo[1]) & (C[:, 1] <= hi[1]) & \
            (np.abs(C[:, 2] - z) <= (self.p.snap_band_m if band is None else band))
        return C[sel]

    def _confirmed(self, pts):
        """Bool mask: map points whose cell was seen from two viewpoints."""
        K = self._conf_k
        if not len(K) or not len(pts):
            return np.zeros(len(pts), bool)
        k = self._cell_keys(pts)
        j = np.minimum(np.searchsorted(K, k), len(K) - 1)
        return (K[j] == k) & self._conf_m[j]

    def _face_dists(self, pts):
        """(N, 4) horizontal distances of NED points to the fence faces forward_min, forward_max, left_min, left_max."""
        F, sf = self.prior.fence, self.sf
        P = np.asarray(pts, float).reshape(-1, 3)
        x, y = P[:, 0] - sf.o[0], P[:, 1] - sf.o[1]  # (elementwise: no BLAS call, which stalls under CPU load)
        ff, ll = x * sf.fwd[0] + y * sf.fwd[1], x * sf.left[0] + y * sf.left[1]
        return np.stack([ff - F['forward_min'], F['forward_max'] - ff, ll - F['left_min'], F['left_max'] - ll], axis=1)

    def _phantom_ok(self, c, r=None):
        """May the obstacle cluster at c (radius r) be discounted as a stereo phantom in front of the net?  Only
        within fence_phantom_m of a fence face, never near a prior wall line, never continuous with structure
        (see _structure_near).  Anywhere else every cluster is real, however small (thin board ends)."""
        p = self.p
        if p.fence_phantom_m <= 0 or c is None or self.sf is None:
            return False
        c = np.asarray(c, float).reshape(3)
        if not np.all(np.isfinite(c)):
            return False
        r = float(r) if r is not None and math.isfinite(float(r)) else 0.1
        dd = self._face_dists(c[None, :])[0]
        face = int(np.argmin(dd))
        if dd[face] >= p.fence_phantom_m:
            return False
        for a, b, wt in self.g.get('walls', ()):
            if float(_seg_min_dist(a, np.asarray(b, float)[None, :], c[None, :2])[0]) < r + p.wall_line_keep_m + wt:
                return False                      # on / near a prior board line or board end
        key = (face, int(math.floor(c[0] / 0.1)), int(math.floor(c[1] / 0.1)), int(math.floor(c[2] / 0.1)))
        memo = self._phantom_memo.get(key)
        if memo is not None and self.now - memo[0] < p.confirm_period_s:
            return memo[1]
        why = self._structure_near(c, r, face)
        ok = why is None
        if len(self._phantom_memo) > 512:
            self._phantom_memo.clear()
        self._phantom_memo[key] = (self.now, ok)
        if not ok:
            self._event_once(('net_structure',) + key[1:3], 'net_cluster_is_structure', center=c, why=why,
                             fence_m=round(float(dd[face]), 3))
        return ok

    def _floating_ok(self, c, r=None):
        """The small cluster at c has no vertical support: its connected map points (3-D, gaps <= phantom_gap_m,
        within 1.5 m) span < floating_span_m vertically -- not a board / board end / wall / boulder / tunnel, which
        all stand on the floor (edge-on board ends are floor-to-top columns).  Never near a prior wall line.  (Searched
        within 1 m around it and +-1.5 m in height.)  The net layer (net_layer_m from a fence face) is no support:
        only in-field structure is (v4E: first-corridor phantoms were 'supported' through the north / west net)."""
        p = self.p
        if p.floating_span_m <= 0 or c is None or self.sf is None:
            return False
        c = np.asarray(c, float).reshape(3)
        if not np.all(np.isfinite(c)):
            return False
        r = float(r) if r is not None and math.isfinite(float(r)) else 0.1
        key = ('float', int(math.floor(c[0] / 0.1)), int(math.floor(c[1] / 0.1)), int(math.floor(c[2] / 0.1)))
        near_line = False
        for a, b, wt in self.g.get('walls', ()):
            if float(_seg_min_dist(a, np.asarray(b, float)[None, :], c[None, :2])[0]) < r + p.wall_line_keep_m + wt:
                near_line = True
                if p.prior_line_real:
                    self._float_info[key] = dict(why='prior_wall_line')
                    return False                  # (20.1: on / near a prior board line or end: real)
        near = self._near_prior_structure(c, r)
        if near is not None:
            self._float_info[key] = dict(why='prior_' + near)
            return False                          # (boulders are low: their points are mostly floor band)
        memo = self._phantom_memo.get(key)
        if memo is not None and self.now - memo[0] < p.confirm_period_s:
            return memo[1]
        om = self.per.obstacles
        pts = getattr(om, 'points', None) if om is not None else None
        ok = False
        info = dict(why='no_points')
        if isinstance(pts, np.ndarray) and pts.ndim == 2 and pts.shape[1] == 3 and len(pts):
            R = r + 1.0
            sel = np.all(np.abs(pts - c) <= np.array([R, R, 1.5]), axis=1)
            P = pts[sel]
            ex = self._active_exclude()
            if ex and len(P):
                P = P[self._outside(P, ex)]
            n_floor = 0
            if len(P):
                dfc = self._face_dists(P).min(axis=1)
                n_net = int(np.count_nonzero(dfc < p.net_layer_m))
                P, dfc = P[dfc >= p.net_layer_m], dfc[dfc >= p.net_layer_m]   # the net itself: never support
                info = dict(why='net_layer_only', net_pts=n_net)
            if len(P) and p.support_floor_up_m is not None:
                above = P[:, 2] < self._z_up(p.support_floor_up_m)            # (v4J) the floor is never support
                n_floor = int(np.count_nonzero(~above))
                P, dfc = P[above], dfc[above]
                if not len(P):
                    info = dict(why='floor_only', net_pts_dropped=n_net, floor_pts_dropped=n_floor)
            if len(P):
                cell = 0.5 * p.support_gap_m          # (v4K: 0.2 tied a blob 0.32 m off a wall face to the wall)
                o = c - np.array([R, R, 1.5])
                n = (np.ceil(2.0 * np.array([R, R, 1.5]) / cell) + 1).astype(int)
                ijk = np.clip(np.floor((P - o) / cell).astype(int), 0, n - 1)
                occ = np.zeros(tuple(n), bool)
                occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
                conn = _dilate3(occ)
                g = [(np.arange(n[i]) + 0.5) * cell + o[i] for i in range(3)]
                d2 = (g[0][:, None, None] - c[0]) ** 2 + (g[1][None, :, None] - c[1]) ** 2 + \
                    (g[2][None, None, :] - c[2]) ** 2
                comp = occ & (d2 <= (r + 0.1) ** 2)
                if comp.any():
                    for _ in range(int(n.max()) * 2):
                        nxt = _dilate3(comp) & conn
                        if np.array_equal(nxt, comp):
                            break
                        comp = nxt
                    inc = comp[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
                    z = P[inc, 2]
                    col = (_tallest_column(P[inc], p.support_cell_m, p.phantom_gap_m) if p.support_cell_m
                           else float(z.max() - z.min())) if len(z) else 0.0
                    ok = len(z) > 0 and col < p.floating_span_m            # (v4J) support = a vertical column
                    if len(z):
                        Q, dq = P[inc], dfc[inc]
                        lo, hi = int(np.argmin(z)), int(np.argmax(z))
                        info = dict(why='floating' if ok else 'supported', span_m=round(float(z.max() - z.min()), 3),
                                    column_m=round(col, 3), comp_n=int(len(z)), net_pts_dropped=n_net,
                                    floor_pts_dropped=n_floor,
                                    fence_m=[round(float(dq.min()), 3), round(float(dq.max()), 3)],
                                    low=np.round(Q[lo], 3).tolist(), high=np.round(Q[hi], 3).tolist())
                    else:
                        info = dict(why='no_component', net_pts_dropped=n_net, floor_pts_dropped=n_floor)
        if len(self._phantom_memo) > 512:
            self._phantom_memo.clear()
        if len(self._float_info) > 512:
            self._float_info.clear()
        self._phantom_memo[key] = (self.now, ok)
        if near_line:
            info = dict(info, prior_wall_line=True)
        self._float_info[key] = info
        return ok

    def _near_prior_structure(self, c, r):
        """Which prior structure (boulder / tunnel / gate / window) is within its tolerance + r + support_prior_m of c
        (horizontally), else None."""
        p, g, P = self.p, getattr(self, 'g', None), self.prior
        if not g:
            return None
        q = np.asarray(c, float)[:2]
        m = float(r) + p.support_prior_m
        for b, h, rb in g.get('boulders', ()):
            if math.hypot(*(q - np.asarray(b, float)[:2])) < rb + P.tolerance_m + m:
                return 'boulder'
        te, ta = g.get('tunnel_entry'), g.get('tunnel_axis')
        if te is not None and ta is not None:
            a = np.asarray(te, float)[:2]
            b = a + np.asarray(ta, float)[:2] * P.tunnel_length_m
            if float(_seg_min_dist(a, b[None, :], q[None, :])[0]) < 0.6 + P.tunnel_tol_m + m:
                return 'tunnel'
        gc, gd = g.get('gate_center'), g.get('gate_dir')
        if gc is not None and gd is not None:
            n = np.array([-gd[1], gd[0]])
            for sg in (1.0, -1.0):
                post = np.asarray(gc, float)[:2] + sg * 0.5 * float(g.get('gate_width') or 1.0) * n
                if math.hypot(*(q - post)) < P.tolerance_m + m:
                    return 'gate'
        wc, wd = g.get('window_center'), g.get('window_dir')
        if wc is not None and wd is not None:
            n = np.array([-wd[1], wd[0]])
            a, b = np.asarray(wc, float)[:2] - 0.75 * n, np.asarray(wc, float)[:2] + 0.75 * n
            if float(_seg_min_dist(a, b[None, :], q[None, :])[0]) < P.window_tol_m + m:
                return 'window'
        return None

    def _float_info_at(self, c, r=None):
        """Diagnostics of the last vertical-support verdict at c (for the small_cluster_kept event)."""
        if c is None:
            return None
        c = np.asarray(c, float).reshape(3)
        key = ('float', int(math.floor(c[0] / 0.1)), int(math.floor(c[1] / 0.1)), int(math.floor(c[2] / 0.1)))
        return self._float_info.get(key)

    def _static_clear(self, q):
        """Clearance from q to the PRIOR obstacles: fence faces and (prior_keepout) the barrier, boulders, prior wall
        lines and the never-seen space behind the take-off point.  Priors can be 0.3-0.6 m off: callers apply it as
        'keep move_clear_m, or no closer than now' (_keep_ok), never as a hard limit."""
        q = np.asarray(q, float).reshape(3)
        if self.sf is None:
            return math.inf
        d = float(self._face_dists(q[None, :])[0].min())
        if not self.p.prior_keepout:
            return d
        return min(d, self._barrier_clear(q), self._boulder_clear(q), self._walls_clear(q), self._rear_clear(q))

    def _keep_ok(self, q, ref):
        """Prior keep-out of a move target q: >= move_clear_m from the prior obstacles, or no closer than `ref`
        (a position or a list of them: the best one counts) already is -- up to tan 10 deg of the move length (a
        sideways step along a path tangent a few degrees off still counts as sideways; v4F eyeball s2's re-confirm
        step, 0.11 m back for 0.2 m, does not)."""
        m = self.p.move_clear_m
        q = np.asarray(q, float).reshape(3)
        refs = [np.asarray(r, float).reshape(3) for r in (ref if isinstance(ref, (list, tuple)) else [ref])]
        sc = [self._static_clear(r) for r in refs]
        k = int(np.argmax(sc))
        slack = 0.005 + math.tan(math.radians(10.0)) * float(math.hypot(*(q - refs[k])[:2]))
        return min(m, self._static_clear(q)) >= min(m, sc[k]) - slack

    _FAR_SIDE = ('chicane', 'gate_pass', 'window_align', 'window_traverse', 'landing_route', 'land_target',
                 'disarm', 'rearm')
    _HOME_SIDE = ('wait', 'arm', 'takeoff', 'corridor', 'tunnel_align', 'tunnel_traverse', 'return_home', 'land_home')

    def _wall_side(self):
        """+1: the vehicle is on the cross_to (start) side of the barrier, -1: on the cross_from side (after the
        tunnel), 0: flying over it (or after a stop there)."""
        ph = self.phase
        if ph == 'climb_over_wall':
            s = 0
        elif ph in self._FAR_SIDE:
            s = -1
        elif ph in self._HOME_SIDE:
            s = 1
        else:                                     # staged stop / abort landing: where it was
            s = getattr(self, '_wside', 1)
        self._wside = s
        return s

    def _seg_unseen_dist(self, a, b, q, cap):
        """Horizontal distance from q to the nearest point of the prior segment a-b (2-D, at q's height) that was NOT
        seen free -- in the camera view within seen_free_s or on our own track: a prior obstacle point we have seen
        free is wrong there (priors can be 0.5 m off; the measured map has the real one).  All seen: inf.  The
        distance itself when it is >= cap (no check needed)."""
        a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
        q = np.asarray(q, float).reshape(3)
        ab = b - a
        L2 = float(ab @ ab)
        u0 = float(np.clip((q[:2] - a) @ ab / L2, 0.0, 1.0)) if L2 > 1e-12 else 0.0
        c0 = a + u0 * ab
        d0 = float(math.hypot(*(q[:2] - c0)))
        if d0 >= cap:
            return d0
        L = math.sqrt(L2)
        n = int(math.ceil(math.sqrt(max(cap * cap - d0 * d0, 0.0)) / 0.05)) if L > 1e-6 else 0
        for k in sorted(range(-n, n + 1), key=abs):      # nearest first
            u = u0 + k * 0.05 / L if L > 1e-6 else u0
            if u < -1e-9 or u > 1.0 + 1e-9:
                continue
            c = a + u * ab
            d = float(math.hypot(*(q[:2] - c)))
            if d < cap and not self._seen_free(np.array([c[0], c[1], q[2]])):
                return d
        return math.inf

    def _barrier_clear(self, q):
        """Clearance from q to the barrier prior (the wall flown over on the return, prior `wall`): signed distance
        to a plane across the cross_from -> cross_to line, wall_keep_to_m from cross_to (<= half way), positive on
        the vehicle's side; only within wall_keep_half_m along the plane (the crossing need not be the wall normal);
        plane points seen free do not count (_seg_unseen_dist); above the prior top (+ tol) the height over it
        counts."""
        p, g = self.p, getattr(self, 'g', None)
        if not g:
            return math.inf
        a, b = g.get('wall_from'), g.get('wall_to')
        side = self._wall_side()
        if a is None or b is None or side == 0:
            return math.inf
        q = np.asarray(q, float).reshape(3)
        b2 = np.asarray(b, float)[:2]
        u = np.asarray(a, float)[:2] - b2
        L = float(math.hypot(u[0], u[1]))
        if L < 1e-6:
            return math.inf
        u = u / L                                 # cross_to -> cross_from
        w = b2 + min(p.wall_keep_to_m, 0.5 * L) * u
        t = np.array([-u[1], u[0]])
        d = q[:2] - w
        if abs(float(d @ t)) > p.wall_keep_half_m:
            return math.inf
        s = -float(d @ u) * side
        if 0.0 <= s < p.move_clear_m:
            h = p.wall_keep_half_m
            s = max(s, self._seg_unseen_dist(w - h * t, w + h * t, q, p.move_clear_m))
        top_d = self._z_up(self.prior.wall_top_up + self.prior.wall_tol_m)
        return max(s, top_d - float(q[2]))

    def _boulder_clear(self, q):
        """Clearance from q to the prior boulders (upright cylinders; above the top the height over it counts; a
        boulder whose near side was seen free does not count)."""
        g = getattr(self, 'g', None)
        q = np.asarray(q, float).reshape(3)
        best = math.inf
        for c, h, r in (g.get('boulders', ()) if g else ()):
            v = q[:2] - np.asarray(c, float)[:2]
            nv = float(math.hypot(v[0], v[1]))
            hd = nv - r
            if hd < self.p.move_clear_m:
                e = np.asarray(c, float)[:2] + (v / nv if nv > 1e-6 else np.array([1.0, 0.0])) * r
                if self._seen_free(np.array([e[0], e[1], q[2]])):
                    continue
            best = min(best, max(hd, self._z_up(h) - float(q[2])))
        return best

    def _walls_clear(self, q):
        """Horizontal distance from q to the prior wall lines (`walls`; line points seen free do not count)."""
        g = getattr(self, 'g', None)
        best = math.inf
        for a, b, wt in (g.get('walls', ()) if g else ()):
            best = min(best, self._seg_unseen_dist(a, b, q, self.p.move_clear_m))
        return best

    def _rear_clear(self, q):
        """Distance from q to the never-seen space behind the take-off point (start heading; the own track covers
        trace_free_m of it), within start_rear_half_m sideways of it, in the takeoff / corridor phases (the camera
        never looks back there; the barrier stands 0.36-0.44 m behind the start in the venue)."""
        p, sf = self.p, self.sf
        if not p.start_rear_keep or sf is None or self.phase not in ('takeoff', 'corridor'):
            return math.inf
        x = np.asarray(q, float)[:2] - sf.o[:2]
        if abs(float(x @ sf.left[:2])) > p.start_rear_half_m:
            return math.inf
        return float(x @ sf.fwd[:2]) + p.trace_free_m

    def _map_dist(self, q, r, ignored=()):
        """3-D distance from q to the nearest map point (active exclusions and `ignored` zones excepted), capped at r."""
        q = np.asarray(q, float).reshape(3)
        om = self.per.obstacles
        pts = getattr(om, 'points', None) if om is not None else None
        if not (isinstance(pts, np.ndarray) and pts.ndim == 2 and pts.shape[1] == 3 and len(pts)):
            return float(r)
        Q = pts[np.all(np.abs(pts - q) <= r, axis=1)]
        ex = list(self._active_exclude() or []) + list(ignored or [])
        if ex and len(Q):
            Q = Q[self._outside(Q, ex)]
        return min(float(r), float(np.sqrt(((Q - q) ** 2).sum(axis=1)).min())) if len(Q) else float(r)

    def _hold_score(self, q, back=None, ignored=()):
        """min(clearance to map points (3-D, ignored clusters excepted), to the prior obstacles (_static_clear),
        known-free extent behind q along `back` (own track / camera view: the rear is never seen)), capped at
        move_clear_m."""
        p = self.p
        m = p.move_clear_m
        q = np.asarray(q, float).reshape(3)
        sc = min(m, self._static_clear(q), self._map_dist(q, m, ignored))
        if back is not None and sc > 0.0:
            b = np.asarray(back, float).reshape(3)
            ext = 0.0
            for x in np.linspace(m / 8.0, m, 8):
                if not self._seen_free(q + b * x):
                    break
                ext = float(x)
            sc = min(sc, ext)
        return sc

    def _structure_near(self, c, r, face):
        """None if the cluster at c is an isolated blob in front of fence face `face`; else why it is structure:
        the connected map points (gaps <= phantom_gap_m, net layer excluded) that were seen from two viewpoints
        reach phantom_inward_m past the phantom band (a board ending at the net) or run >= 0.9 m along the face
        (a board beside the net).  No map points to judge by: structure (never discount blind)."""
        p, om = self.p, self.per.obstacles
        pts = getattr(om, 'points', None) if om is not None else None
        if not (isinstance(pts, np.ndarray) and pts.ndim == 2 and pts.shape[1] == 3):
            return 'no_points'
        R = r + 1.6
        sel = (np.abs(pts[:, 0] - c[0]) <= R) & (np.abs(pts[:, 1] - c[1]) <= R) & \
            (np.abs(pts[:, 2] - c[2]) <= p.snap_band_m)
        P = pts[sel]
        ex = self._active_exclude()
        if ex and len(P):
            P = P[self._outside(P, ex)]
        if not len(P):
            return None
        dd = self._face_dists(P)
        P, dface = P[dd.min(axis=1) >= p.net_layer_m], dd[dd.min(axis=1) >= p.net_layer_m, face]
        if not len(P):
            return None
        cell = 0.5 * p.phantom_gap_m              # occupancy grid around c; connected = gaps <= phantom_gap_m
        o = c[:2] - R
        nx = int(math.ceil(2.0 * R / cell)) + 1
        ij = np.clip(np.floor((P[:, :2] - o) / cell).astype(int), 0, nx - 1)
        occ = np.zeros((nx, nx), bool)
        occ[ij[:, 0], ij[:, 1]] = True
        conn = _dilate(occ)
        g = (np.arange(nx) + 0.5) * cell + o[0], (np.arange(nx) + 0.5) * cell + o[1]
        comp = occ & (np.hypot(g[0][:, None] - c[0], g[1][None, :] - c[1]) <= r + 0.1)
        if not comp.any():
            return None
        for _ in range(4 * nx):                   # flood fill (vectorised)
            nxt = _dilate(comp) & conn
            if np.array_equal(nxt, comp):
                break
            comp = nxt
        idx = np.nonzero(comp[ij[:, 0], ij[:, 1]])[0]
        Q, dq = P[idx], dface[idx]
        conf = self._confirmed(Q)
        if not conf.any():
            return None
        Q, dq = Q[conf], dq[conf]
        if float(dq.max()) >= p.fence_phantom_m + p.phantom_inward_m:
            return 'reaches_inward'
        tg = self.sf.left[:2] if face < 2 else self.sf.fwd[:2]
        along = Q[:, 0] * tg[0] + Q[:, 1] * tg[1]
        if float(along.max() - along.min()) >= 0.9:
            return 'runs_along_net'
        return None

    # ---- obstacle map access (exclusion zones, duck-typed)
    def _active_exclude(self):
        for kind in ('tunnel', 'window'):
            if self.phase.startswith(kind) and kind in self._exclude:
                return self._exclude[kind]
        return None

    def _supports_exclude(self, fn):
        key = (type(getattr(fn, '__self__', fn)), getattr(fn, '__name__', ''))
        if key not in self._excl_ok:
            try:
                ps = inspect.signature(fn).parameters.values()
                self._excl_ok[key] = any(q.name == 'exclude' or q.kind == q.VAR_KEYWORD for q in ps)
            except (TypeError, ValueError):
                self._excl_ok[key] = False
        return self._excl_ok[key]

    def _clearance(self, om, a, b, r, extra=None):
        ex = (self._active_exclude() or []) + list(extra or [])
        if ex and self._supports_exclude(om.clearance):
            return om.clearance(a, b, r, exclude=ex)
        return om.clearance(a, b, r)

    @staticmethod
    def _outside(points, exclude):
        keep = np.ones(len(points), bool)
        for c, ax, r, hl in exclude or ():
            ax = np.asarray(ax, float)
            n = float(np.linalg.norm(ax))
            if n < 1e-9:
                continue
            ax = ax / n
            rel = points - np.asarray(c, float)
            along = rel @ ax
            radial = np.linalg.norm(rel - along[:, None] * ax[None, :], axis=1)
            keep &= ~((np.abs(along) <= hl) & (radial <= r))
        return keep

    def _cluster(self, chain, free, extra=None):
        """(number of map points near the first blocked probe sample or None if the map cannot tell, cluster centre
        or None if it cannot be excluded, exclusion radius)."""
        om = self.per.obstacles
        if om is None or not math.isfinite(free):
            return None, None, None
        q, run = np.asarray(chain[-1], float), 0.0
        for a, b in zip(chain[:-1], chain[1:]):
            seg = float(np.linalg.norm(b - a))
            if run + seg >= free:
                q = a + (b - a) * ((free - run) / seg if seg > 1e-9 else 0.0)
                break
            run += seg
        r = self.p.radius + self.p.cluster_radius_extra_m
        ex = (self._active_exclude() or []) + list(extra or [])
        pts = getattr(om, 'points', None)
        near = None
        if isinstance(pts, np.ndarray) and pts.ndim == 2 and pts.shape[1] == 3:
            if ex and len(pts):
                pts = pts[self._outside(pts, ex)]
            near = pts[np.linalg.norm(pts - q, axis=1) <= r] if len(pts) else pts
        if hasattr(om, 'count'):
            n = om.count(q, r, exclude=ex) if (ex and self._supports_exclude(om.count)) else om.count(q, r)
            n = None if n is None else int(n)
        elif near is not None:
            n = len(near)
        else:
            return None, None, None
        centre, rad = None, None
        if near is not None and len(near) and self._supports_exclude(om.clearance):
            centre = near.mean(axis=0)
            rad = float(min(0.3, max(0.1, np.linalg.norm(near - centre, axis=1).max() + 0.08)))
        return n, centre, rad

    def _ahead(self, cam, h, r, extra=None):
        """Distance along the unit ray h from cam to the nearest map point within r of the ray (inf: none); None when
        the map has no .points (the caller then estimates it with clearance())."""
        om = self.per.obstacles
        pts = getattr(om, 'points', None) if om is not None else None
        if not (isinstance(pts, np.ndarray) and pts.ndim == 2 and pts.shape[1] == 3):
            return None
        ex = (self._active_exclude() or []) + list(extra or [])
        if ex and len(pts):
            pts = pts[self._outside(pts, ex)]
        if not len(pts):
            return math.inf
        d = pts - np.asarray(cam, float)
        along = d @ h
        lat = np.linalg.norm(d - along[:, None] * h[None, :], axis=1)
        sel = (along > 0.0) & (lat < r)
        if not sel.any():
            return math.inf
        i = int(np.argmin(np.where(sel, along, np.inf)))
        return _Hit(float(along[i]), pts[i])

    def _probe(self, chain, strict=False, extra=None, radius=None):
        om = self.per.obstacles
        if om is None:
            return math.inf
        r_soft = self.p.radius + (self.p.return_margin_m if strict else 0.0)
        free = 0.0
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            seg = float(np.linalg.norm(b - a))
            if seg < 1e-4:
                continue
            r = radius if radius is not None else (r_soft if (strict or i > 0) else self.p.hard_radius)
            c = self._clearance(om, a, b, r, extra)
            c = math.inf if c is None else float(c)
            if c < seg:
                return free + max(c, 0.0)
            free += seg
        return math.inf

    def _speed_cap(self):
        per = self.per
        if per.obstacles is None or (per.obstacles_t is not None and self.now - per.obstacles_t > self.p.map_max_age_s):
            self._event_once(('no_map', self.phase), 'obstacle_map_missing_or_stale')
            return self.p.v_no_map
        return math.inf

    def _new_follower(self, goals, v_max, avoid=None, yaw=None, yaw_end=None, run_out=0.0, tol=None, start=None,
                      side_max=None, gap_snap=False, next_after=None):
        if start is not None:
            pass
        elif (self._last_sp is not None and self.phase in FLIGHT_PHASES
                and np.linalg.norm(self._last_sp - self.pos) < self.p.leash_lo_m):
            start = self._last_sp                 # continue from the carrot: no setpoint jump at hand-offs
        else:
            start = self.pos
        d0 = np.asarray(goals[0], float) - start
        n0 = float(np.linalg.norm(d0))
        v0 = float(self.vel @ d0 / n0) if n0 > 1e-3 else 0.0
        leash = None
        if self.p.pilot_mode:                     # no feed-forward: steady lead = v / gain, so allow it (and no more)
            hi = min(float(v_max) / self.p.pilot_pos_gain + 0.65, self.p.pilot_max_lead_m)
            leash = (max(0.2, hi - 0.5), hi)
        self._fol_kw = dict(v_max=v_max, avoid=avoid, yaw=yaw, yaw_end=yaw_end, run_out=run_out, tol=tol,
                            side_max=side_max, leash=leash, gap_snap=gap_snap, next_after=next_after)
        self._fol = self._make_follower(start, goals, v0=v0, **self._fol_kw)
        return self._fol

    def _make_follower(self, start, goals, v0=0.0, **kw):
        f = Follower(start, goals, self.p, kw.pop('v_max'), v0=v0, **kw)
        f.fence_ahead = self._fence_ahead
        f.fence_dist = self._fence_dist
        f.fence_face = self._fence_face
        f.fence_phantom = self._fence_phantom
        f.phantom_ok = self._phantom_ok
        f.floating_ok = self._floating_ok
        f.float_info = self._float_info_at
        f.hold_score = self._hold_score
        f.map_dist = self._map_dist
        f.keep_ok = self._keep_ok
        f.on_track = self._on_track
        f.start_f = self._start_f
        f.seen_free = self._seen_free
        f.count_near = self._count_near
        f.clear_xy = self._clear_xy
        f.boards_xy = self._boards_xy
        f.valid_pt = self._valid
        f.snap = f.avoid == 'lateral'
        return f

    def _fence_face(self, p):
        F = self.prior.fence
        f, l, _ = self.sf.from_ned(p)
        faces = ((f - F['forward_min'], -self.sf.fwd), (F['forward_max'] - f, self.sf.fwd),
                 (l - F['left_min'], -self.sf.left), (F['left_max'] - l, self.sf.left))
        d, n = min(faces, key=lambda x: x[0])
        return float(d), np.asarray(n, float)

    def _fence_phantom(self, q, d):
        """q (an obstacle we are flying toward along d) is within fence_phantom_m of THAT fence face and is a blob,
        not a wall running along the net (a board parallel to the net is real: map points 0.45-0.9 m along the face
        on either side at the same depth)."""
        p = self.p
        dist, n = self._fence_face(q)
        d = np.asarray(d, float)[:2]
        nd = float(np.linalg.norm(d))
        if dist >= p.fence_phantom_m or nd < 1e-6 or float(d @ n[:2]) / nd <= math.cos(math.radians(45.0)):
            return False
        om = self.per.obstacles
        pts = getattr(om, 'points', None) if om is not None else None
        if isinstance(pts, np.ndarray) and pts.ndim == 2 and len(pts):
            q = np.asarray(q, float)
            rel = pts - q
            tg = np.array([-n[1], n[0]])
            dep = np.abs(rel[:, :2] @ n[:2])
            along = rel[:, :2] @ tg
            band = (dep < 0.12) & (np.abs(rel[:, 2]) < p.snap_band_m)
            if _runs_along(along[band]):
                return False                      # continues along the net: a wall, keep treating it as real
        return self._phantom_ok(q)                # (and not a board END at the net / on a prior wall line)

    def _net_blobs(self, pts):
        """Bool mask: points within fence_phantom_m of a fence face that do NOT continue along that face (0.45-0.9 m
        to either side at the same depth) -- blobs in front of the net, not boards running along it."""
        p, F = self.p, self.prior.fence
        rel = pts - self.sf.o
        ff, ll = rel @ self.sf.fwd, rel @ self.sf.left
        dd = np.stack([ff - F['forward_min'], F['forward_max'] - ff, ll - F['left_min'], F['left_max'] - ll], axis=1)
        face = np.argmin(dd, axis=1)
        near = dd[np.arange(len(pts)), face] < p.fence_phantom_m
        out = np.zeros(len(pts), bool)
        idx = np.where(near)[0]
        if not len(idx):
            return out
        cell = np.c_[face[idx], np.floor(ff[idx] / 0.1), np.floor(ll[idx] / 0.1)].astype(int)   # judge per 0.1 m cell
        keys, inv = np.unique(cell, axis=0, return_inverse=True)
        verdict = np.zeros(len(keys), bool)
        for k, (fc, cf, cl) in enumerate(keys):
            dep, alo = (ff, ll) if fc < 2 else (ll, ff)
            d0 = (cf + 0.5) * 0.1 if fc < 2 else (cl + 0.5) * 0.1
            a0 = (cl + 0.5) * 0.1 if fc < 2 else (cf + 0.5) * 0.1
            band = np.abs(dep - d0) < 0.12
            verdict[k] = not _runs_along((alo - a0)[band])
            if verdict[k]:                         # a board END at the net continues inward past the band
                dk = float(dd[idx[inv.reshape(-1) == k], fc].min())
                dc = np.sort(dd[np.abs(alo - a0) < 0.12, fc])
                dc = dc[dc >= dk - 0.1]
                if len(dc):
                    seq = np.r_[dk, dc]
                    br = np.nonzero(np.diff(seq) > p.phantom_gap_m)[0]
                    reach = float(seq[br[0]] if len(br) else seq[-1])
                    if reach >= p.fence_phantom_m + p.phantom_inward_m:
                        verdict[k] = False
        out[idx] = verdict[inv.reshape(-1)]
        return out

    def _fence_dist(self, p):
        """Horizontal distance from p to the nearest fence (net) face."""
        F = self.prior.fence
        f, l, _ = self.sf.from_ned(p)
        return float(min(f - F['forward_min'], F['forward_max'] - f, l - F['left_min'], F['left_max'] - l))

    def _clear_xy(self, cands, z):
        """(M,) horizontal distance from each candidate (M,3 NED) to the nearest mapped BOARD point within
        snap_band_m of height z; None when the map has no .points."""
        cands = np.asarray(cands, float).reshape(-1, 3)
        B = self._boards_xy(cands[:, :2].min(axis=0) - 1.0, cands[:, :2].max(axis=0) + 1.0, z)
        return None if B is None else _min_dist(cands[:, :2], B)

    def _boards_xy(self, lo, hi, z):
        """Board points (N,2 NED xy) in the box lo..hi within snap_band_m of height z: active exclusions applied,
        >= minor_cluster_n points per 0.3 m block, stereo blobs in front of the net dropped.  None: no .points."""
        om = self.per.obstacles
        pts = getattr(om, 'points', None) if om is not None else None
        if not (isinstance(pts, np.ndarray) and pts.ndim == 2 and pts.shape[1] == 3):
            return None
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        if not len(pts):
            pts = np.zeros((0, 3))
        sel = (np.abs(pts[:, 2] - z) <= self.p.snap_band_m) & np.all(pts[:, :2] >= lo, axis=1) & \
            np.all(pts[:, :2] <= hi, axis=1)
        pts = pts[sel]
        mem = self._remembered(lo, hi, z)          # + what the node's short map memory already forgot
        if len(mem):
            pts = np.vstack([pts, mem]) if len(pts) else mem
        ex = list(self._active_exclude() or []) + list(getattr(self._fol, 'ignored', None) or [])
        if ex and len(pts):                        # (+ the follower's ignored phantom clusters)
            pts = pts[self._outside(pts, ex)]
        if len(pts):                               # boards: >= board_open_min_pts points in the 0.3 m cell block
            cell = 0.1                             # (an edge-on thin board end is a few points); in the band in
            ij = np.floor((pts[:, :2] - lo) / cell).astype(int)   # front of the net >= minor_cluster_n
            nx, ny = int(ij[:, 0].max()) + 3, int(ij[:, 1].max()) + 3
            g = np.zeros((nx, ny), int)
            np.add.at(g, (ij[:, 0] + 1, ij[:, 1] + 1), 1)
            s = sum(np.roll(np.roll(g, dx, 0), dy, 1) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
            sn = s[ij[:, 0] + 1, ij[:, 1] + 1]
            near = self._face_dists(pts).min(axis=1) < self.p.fence_phantom_m if self.p.fence_phantom_m > 0 \
                else np.zeros(len(pts), bool)
            pts = pts[np.where(near, sn >= self.p.minor_cluster_n, sn >= self.p.board_open_min_pts)]
        if len(pts) and self.p.fence_phantom_m > 0:
            pts = pts[~self._net_blobs(pts)]       # stereo phantoms in front of the net never move the path
        if len(pts) and self.p.floating_span_m > 0 and self.p.board_support_filter:
            pts = pts[self._floor_standing(pts, lo, hi, z, ex)]   # (v4F) same criteria as blocking
        return pts[:, :2].copy()

    def _floor_standing(self, pts, lo, hi, z, ex=None):
        """Bool mask over flight-band points: they belong to floor-standing in-field structure -- not the net layer
        (net_layer_m from a fence face), and the map points at their footprint (3x3 cells of 0.1 m: live map + the
        confirm memory, +-1.5 m in height, net layer excluded) span >= floating_span_m vertically (the vertical-
        support rule of the blocking decision; board faces and edge-on board ends are floor-to-top columns).  Near a
        prior wall line: always kept (20.1).  v4F: net ghosts in the first corridor no longer pull / push vertices."""
        p = self.p
        P = np.asarray(pts, float).reshape(-1, 3)
        keep = self._face_dists(P).min(axis=1) >= p.net_layer_m
        om = self.per.obstacles
        live = getattr(om, 'points', None) if om is not None else None
        lo2, hi2 = np.asarray(lo, float)[:2] - 0.15, np.asarray(hi, float)[:2] + 0.15
        F = [np.zeros((0, 3))]
        if isinstance(live, np.ndarray) and live.ndim == 2 and live.shape[1] == 3 and len(live):
            F.append(live[(np.abs(live[:, 2] - z) <= 1.5) & np.all(live[:, :2] >= lo2, axis=1) &
                          np.all(live[:, :2] <= hi2, axis=1)])
        F.append(self._remembered(lo2, hi2, z, band=1.5))
        F = np.vstack(F)
        if ex and len(F):
            F = F[self._outside(F, ex)]
        if len(F):
            F = F[self._face_dists(F).min(axis=1) >= p.net_layer_m]
        sup = np.zeros(len(P), bool)
        if len(F):
            cell = 0.1
            ij = np.floor((F[:, :2] - lo2) / cell).astype(int)
            nx, ny = int(ij[:, 0].max()) + 3, int(ij[:, 1].max()) + 3
            zmin, zmax = np.full((nx, ny), np.inf), np.full((nx, ny), -np.inf)
            np.minimum.at(zmin, (ij[:, 0] + 1, ij[:, 1] + 1), F[:, 2])
            np.maximum.at(zmax, (ij[:, 0] + 1, ij[:, 1] + 1), F[:, 2])
            lo3 = np.min([np.roll(np.roll(zmin, dx, 0), dy, 1) for dx in (-1, 0, 1) for dy in (-1, 0, 1)], axis=0)
            hi3 = np.max([np.roll(np.roll(zmax, dx, 0), dy, 1) for dx in (-1, 0, 1) for dy in (-1, 0, 1)], axis=0)
            pj = np.floor((P[:, :2] - lo2) / cell).astype(int) + 1
            ok = (pj[:, 0] >= 0) & (pj[:, 0] < nx) & (pj[:, 1] >= 0) & (pj[:, 1] < ny)
            pjc = np.clip(pj, 0, [nx - 1, ny - 1])
            span = hi3[pjc[:, 0], pjc[:, 1]] - lo3[pjc[:, 0], pjc[:, 1]]
            sup = ok & np.isfinite(span) & (span >= p.floating_span_m)
        walls = self.g.get('walls', ()) if isinstance(self.g, dict) else ()
        if walls and len(P):
            near = np.zeros(len(P), bool)
            for a, b, wt in walls:
                a2, b2 = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
                d = b2 - a2
                L2 = max(float(d @ d), 1e-12)
                u = np.clip(((P[:, :2] - a2) @ d) / L2, 0.0, 1.0)
                dist = np.hypot(*(P[:, :2] - (a2 + u[:, None] * d)).T)
                near |= dist < p.wall_line_keep_m + wt
            sup |= near
        ok = keep & sup
        net = ~keep                                # the net layer: only where in-field structure runs into it (a
        if net.any() and ok.any():                 # board ending at the net, W2's north end) -- never the net itself
            ok[net] = _min_dist(P[net][:, :2], P[ok][:, :2]) < p.phantom_gap_m
        return ok

    def _contact_check(self):
        """Prop strike: |IMU| spike or a velocity jump.  Returns the measured value or None."""
        p, v = self.p, self.veh
        a = getattr(v, 'accel', None)
        val = None
        if a is not None:
            an = float(np.linalg.norm(np.asarray(a, float)))
            if math.isfinite(an) and an > p.contact_accel_mps2:
                val = ('imu', an)
        pv, pt = getattr(self, '_cv_prev', (None, None))
        if val is None and pv is not None and self.now - pt > 1e-3:
            dvr = float(np.linalg.norm(self.vel - pv)) / (self.now - pt)
            if dvr > p.contact_dv_mps2:
                val = ('dv', dvr)
        self._cv_prev = (self.vel.copy(), self.now)
        return val

    def _hold_drift_check(self):
        """Holding still (carrot stopped) but the vehicle drifts away from the setpoint by > hold_drift_m and the
        error is growing: back away from the nearest wall now instead of waiting (official run_003: +-0.3 m, ~6 s
        hold oscillation after the corner yaw facing the wall -> contact).  Reuses the contact recovery."""
        p, f = self.p, self._fol
        sp = self._last_sp
        if sp is None or f.v > 0.02 or p.hold_drift_m <= 0:
            self._drift_prev = None
            return
        err = float(np.linalg.norm((self.pos - sp)[:2]))
        prev = getattr(self, '_drift_prev', None)
        self._drift_prev = err
        if prev is None or err <= prev + 1e-4 or float(np.linalg.norm(self.vel[:2])) < 0.03 or \
                float((self.pos - sp)[:2] @ self.vel[:2]) <= 0.0:
            return                                 # (only while moving AWAY from the held setpoint)
        if err <= p.hold_drift_m:
            return
        away = self._away_from_wall()
        if away is None:
            away = (sp - self.pos).copy()
            away[2] = 0.0
            nn = float(np.linalg.norm(away))
            away = away / nn if nn > 1e-6 else np.zeros(3)
        q = self.pos + away * p.hold_drift_back_m
        if not self._valid(q) or self._probe([self.pos, q], radius=p.half_span_m) != math.inf or \
                not self._seen_free(q + away * p.move_clear_m) or not self._keep_ok(q, self.pos):
            back = (sp - self.pos).copy()          # not seen free: at least back toward the setpoint (our track)
            back[2] = 0.0
            nb = float(np.linalg.norm(back))
            away = back / nb if nb > 1e-6 else np.zeros(3)
            q = self.pos + away * p.hold_drift_back_m
            if not self._valid(q) or self._probe([self.pos, q], radius=p.half_span_m) != math.inf or \
                    not (self._keep_ok(q, self.pos) or self._on_track(q)):   # (v4G: no new space closer to a
                away = np.zeros(3)                                         # prior obstacle / the unseen rear)
        self._event('hold_drift', err_m=round(err, 3), pos=self.pos.copy(), backoff_dir=away.copy())
        self._contact = dict(t0=self.now - p.contact_hover_s, p0=self.pos.copy(), dir=away, yaw=float(self.veh.heading),
                             back_m=p.hold_drift_back_m if float(np.linalg.norm(away)) > 0 else 0.0, fol=f,
                             min_hold=p.hold_drift_wait_s)

    def _away_from_wall(self, with_dist=False):
        """Unit horizontal direction away from the nearest map point (flight band, within 1 m) or the nearest
        fence face if that is closer; None if nothing is near.  with_dist: (direction, distance)."""
        p = self.p
        best, d_best = None, 1.0
        om = self.per.obstacles
        pts = getattr(om, 'points', None) if om is not None else None
        if isinstance(pts, np.ndarray) and pts.ndim == 2 and len(pts):
            x, y = pts[:, 0] - self.pos[0], pts[:, 1] - self.pos[1]
            sel = (np.abs(x) < 1.0) & (np.abs(y) < 1.0) & (np.abs(pts[:, 2] - self.pos[2]) < p.snap_band_m)
            if sel.any():
                d2 = x[sel] ** 2 + y[sel] ** 2
                i = int(np.argmin(d2))
                if d2[i] < d_best ** 2 and d2[i] > 1e-6:
                    d_best = math.sqrt(float(d2[i]))
                    best = -np.array([x[sel][i], y[sel][i], 0.0]) / d_best
        if self.sf is not None:
            fd, n = self._fence_face(self.pos)
            if fd < d_best:
                best = -np.asarray(n, float)
                best[2] = 0.0
                d_best = fd
        return (best, d_best) if with_dist else best

    def _contact_recover(self):
        """Hover, back off along where we came from, then replan the rest of the follower path."""
        p, c = self.p, self._contact
        el = self.now - c['t0']
        if el < p.contact_hover_s:
            return Step((c['p0'].copy(), np.zeros(3), np.zeros(3), c['yaw']), 'moving', math.inf)
        tb = el - p.contact_hover_s
        dist = min(c['back_m'], tb * p.contact_backoff_v)
        q = c['p0'] + c['dir'] * dist
        if dist < c['back_m'] - 1e-6:
            return Step((q, c['dir'] * p.contact_backoff_v, np.zeros(3), c['yaw']), 'moving', math.inf)
        t_back = c['back_m'] / p.contact_backoff_v
        if (np.linalg.norm(self.pos - q) > 0.08 and tb < t_back + 2.0) or tb < t_back + c.get('min_hold', 0.0):
            return Step((q, np.zeros(3), np.zeros(3), c['yaw']), 'moving', math.inf)
        old = c['fol']                             # replan from here: the remaining vertices of the old path
        goals = [old.P[i].copy() for i in range(1, old.n_goal) if float(old.cum[i]) > old.s + 0.05]
        if not goals:
            goals = [old.goal.copy()]
        kw = dict(self._fol_kw)
        if old.pass_through:
            kw['run_out'] = max(kw.get('run_out') or 0.0, 0.3)
        self._fol = self._make_follower(self.pos.copy(), goals, **kw)
        self._fol._settled = set()
        self._event('contact_replan', goals=len(goals), pos=self.pos.copy())
        self._contact = None
        return None

    def _fence_ahead(self, p, d):
        """Horizontal distance from p along the unit direction d (NED) to the fence (net) wall (inf: never)."""
        F = self.prior.fence
        q = self.sf.from_ned(p)
        e = np.array([np.asarray(d, float) @ self.sf.fwd, np.asarray(d, float) @ self.sf.left])
        best = math.inf
        for i, (lo, hi) in enumerate(((F['forward_min'], F['forward_max']), (F['left_min'], F['left_max']))):
            if e[i] > 1e-6:
                best = min(best, (hi - q[i]) / e[i])
            elif e[i] < -1e-6:
                best = min(best, (lo - q[i]) / e[i])
        return max(0.0, best)

    def _vio_fault_start(self, why, jump, back=None, back_m=0.0):
        p, now = self.p, self.now
        hist = [t for t in getattr(self, '_vf_hist', []) if now - t < p.vio_fault_repeat_s] + [now]
        self._vf_hist = hist
        self._event('vio_fault', why=why, jump_m=round(float(jump), 3), recent=len(hist))
        if p.vio_fault_repeat_n > 0 and len(hist) >= p.vio_fault_repeat_n:
            self._vf = None
            self._abort('vio_fault_repeated', brake=False)
            return
        self._vf = dict(t0=now, why=why, jump=float(jump), ok_since=None, yaw=float(self.veh.heading),
                        back=None if back is None else np.asarray(back, float).copy(), back_m=float(back_m))

    def _vio_guard(self):
        """VIO-fault guard (see Params.vio_guard).  Returns a Step while it holds, None otherwise."""
        p, now, pos, vel = self.p, self.now, self.pos, self.vel
        prev = getattr(self, '_vf_prev', None)
        self._vf_prev = (now, pos.copy(), vel.copy())
        jump = 0.0
        if prev is not None and 1e-3 < now - prev[0] <= 0.25:
            jump = float(np.linalg.norm((pos - prev[1] - 0.5 * (prev[2] + vel) * (now - prev[0]))[:2]))
        drop = getattr(self, '_vf_pose_drop', False)
        self._vf_pose_drop = False
        hold = Step((pos.copy(), np.zeros(3), np.zeros(3), float(self.veh.heading)), 'moving', math.inf)
        if not p.vio_guard or self._fol is None or not self._fol.avoid or getattr(self, '_contact', None) is not None:
            return None
        f = getattr(self, '_vf', None)
        if f is None:
            if jump > p.vio_jump_m:
                self._vio_fault_start('jump', jump)
            elif drop:
                self._vio_fault_start('pose', 0.0)
            else:
                return None
            f = self._vf
            if f is None:                         # (a repeated fault: landing)
                return hold
        elif jump > p.vio_jump_m:
            f['jump'] += jump
        el = now - f['t0']
        if f['jump'] > p.vio_jump_abort_m:
            self._vf = None
            self._abort('vio_fault_jump', brake=False, jump_m=round(f['jump'], 3))
            return hold
        ok = bool(self.veh.pose_ok) and jump <= p.vio_jump_m
        f['ok_since'] = (f['ok_since'] if f['ok_since'] is not None else now) if ok else None
        if el >= p.vio_fault_hold_s and f['ok_since'] is not None and now - f['ok_since'] >= p.vio_ok_s:
            self._vf = None                       # recovered: (impact) back off along the flown track, then hold
            self._event('vio_recovered', why=f['why'], after_s=round(el, 2))   # the quarantine, then replan
            back = f['back'] if f['back'] is not None else np.zeros(3)
            self._contact = dict(t0=now - p.contact_hover_s, p0=pos.copy(), dir=back, yaw=f['yaw'],
                                 back_m=f['back_m'] if float(np.linalg.norm(back)) > 0 else 0.0, fol=self._fol,
                                 min_hold=p.vio_quarantine_s)
            return None
        if el > p.vio_fault_max_s:
            self._vf = None
            self._abort('vio_fault', brake=False, why=f['why'])
            return hold
        self._fol.v = 0.0                         # hover ON the estimate: nothing advances, nothing is re-centred
        return Step((pos.copy(), np.zeros(3), np.zeros(3), f['yaw']), 'moving', math.inf)

    def _fstep(self):
        vf = self._vio_guard() if self.phase in FLIGHT_PHASES else None
        if vf is not None:
            return vf
        hit = self._contact_check() if self.phase in FLIGHT_PHASES and self._fol.avoid else None
        if getattr(self, '_contact', None) is None and hit is not None:
            pv = getattr(self, '_v_before', None)
            vb = (pv if pv is not None else -self.vel).copy()
            vb[2] = 0.0
            nb = float(np.linalg.norm(vb))
            back = -vb / nb if nb > 0.05 else np.zeros(3)
            back_m = 0.0
            for bm in (self.p.contact_backoff_m, 0.5 * self.p.contact_backoff_m):   # (v4F) the backed-off airframe
                q = self.pos + back * bm            # keeps move_clear_m behind it known free and the prior keep-out
                if nb > 0.05 and self._seen_free(q + back * self.p.move_clear_m) and \
                        (self._keep_ok(q, self.pos) or self._on_track(q)):
                    back_m = bm                     # (v4G: barrier, unseen rear ...) -- else shorter, else hover
                    break
            if back_m <= 0.0:
                back = np.zeros(3)
            q = self.pos + back * back_m
            if nb <= 0.05 or not self._valid(q) or self._probe([self.pos, q], radius=0.15) != math.inf \
                    or not self._seen_free(q):          # (just hit something: probe the body core only)
                back = np.zeros(3)                  # no known (seen) free direction: just hover
            n = self._contacts.get(self.phase, 0) + 1
            self._contacts[self.phase] = n
            self._event('contact', source=hit[0], value=round(hit[1], 2), pos=self.pos.copy(), count=n,
                        backoff_dir=back.copy())
            if n > self.p.contact_max:
                self._abort('repeated_contact')
            elif self.p.vio_guard:                # (v4K) an impact disturbs VIO: hover on the estimate until it is
                self._vio_fault_start('contact', 0.0, back=back,          # consistent, then back off + quarantine
                                      back_m=back_m if float(np.linalg.norm(back)) > 0 else 0.0)
                if getattr(self, '_vf', None) is not None:
                    return Step((self.pos.copy(), np.zeros(3), np.zeros(3), float(self.veh.heading)), 'moving',
                                math.inf)
            else:
                self._contact = dict(t0=self.now, p0=self.pos.copy(), dir=back, yaw=float(self.veh.heading),
                                     back_m=back_m if float(np.linalg.norm(back)) > 0 else 0.0,
                                     fol=self._fol)
        if getattr(self, '_contact', None) is None and self.phase in FLIGHT_PHASES and self._fol.avoid:
            self._hold_drift_check()
        if getattr(self, '_contact', None) is not None:
            r = self._contact_recover()
            if r is not None:
                return r
        if self._fol is None:                      # aborted above (repeated contact): hold here this step
            return Step((self.pos.copy(), np.zeros(3), np.zeros(3), float(self.veh.heading)), 'moving', math.inf)
        if float(np.linalg.norm(self.vel[:2])) > 0.1:
            self._v_before = self.vel.copy()       # the last free direction of travel (before any contact)
        f = self._fol
        look_before = f.look_off
        r = f.step(self.dt, self.now, self.pos, float(self.veh.heading),
                   probe=self._probe if f.avoid else None, valid=self._valid if f.avoid else None,
                   cap=self._speed_cap() if f.avoid else math.inf,
                   seen=self._seen if f.avoid else None, cluster=self._cluster if f.avoid else None,
                   vel=self.vel, ahead=self._ahead if f.avoid else None)
        for kind, data in f.flags:
            self._event(kind, **data)
        f.flags = []
        if f._minor_since is not None and not getattr(f, '_reported_minor', False):
            f._reported_minor = True
            self._event('minor_obstacle', free_m=round(r.free, 3))
        elif f._minor_since is None:
            f._reported_minor = False
        if f.look_off is not None and f.look_off != look_before:
            self._event('look_before_side_step', offset_m=round(f.look_off, 3))
        if f.off_tgt != getattr(f, '_reported_off', 0.0):
            f._reported_off = f.off_tgt
            self._event('avoid', offset_m=round(f.off_tgt, 3), axis=f.avoid)
        if r.status == 'blocked' and not getattr(f, '_reported_block', False):
            f._reported_block = True
            self._event('blocked', free_m=round(r.free, 3))
        elif r.status != 'blocked' and r.status != 'stuck':
            f._reported_block = False
        return r

    def _tof(self):
        v = self.veh
        if v.tof_m is None:
            return None
        t = float(v.tof_m)
        if not math.isfinite(t) or t < 0.0:
            return None
        if v.tof_t is not None and self.now - float(v.tof_t) > self.p.tof_max_age_s:
            return None
        return t

    def _settled(self, target, yaw=None, tol=None, xy_only=False):
        p = self.p
        d = (self.pos - target)[:2] if xy_only else self.pos - target
        ok = np.linalg.norm(d) < (p.settle_tol_m if tol is None else tol) and np.linalg.norm(self.vel) < p.settle_speed
        if ok and yaw is not None:
            ok = abs(wrap(yaw - float(self.veh.heading))) < math.radians(p.settle_yaw_deg)
        return bool(ok)

    # ----------------------------------------------------------------- guard
    def _guard(self):
        """Global safety checks.  Returns (handled, setpoint)."""
        v, p, ph, now = self.veh, self.p, self.phase, self.now
        if ph == 'arm':
            if v.nav_state == NAV_OFFBOARD:
                self._seen_offboard = True
            elif self._seen_offboard:
                self._release('left_offboard nav_state=%d' % v.nav_state)
                return True, None
            return False, None
        if ph not in FLIGHT_PHASES:
            return False, None
        if p.pilot_mode and v.armed and self.home is not None and self._in_control():
            up = float(self.home[2] - self.pos[2])
            M = [m_ for m_ in self._motion if now - m_[0] <= 0.5]   # climb rate from our own track (any source)
            v_up = float(M[0][1][2] - self.pos[2]) / max(now - M[0][0], 1e-3) if M and now - M[0][0] > 0.2 \
                else -float(self.vel[2])
            lim = self._intended_up() + p.climb_guard_m
            if up > lim and v_up > p.climb_guard_v:          # climbing away above anything we meant to fly at
                self._event('climb_guard', up_m=round(up, 3), limit_m=round(lim, 3), v_up=round(v_up, 3),
                            sub=self.sub)
                self._release('climb_guard up %.2f m > %.2f m, rising %.2f m/s' % (up, lim, v_up))
                return True, None
        pilot_landing = (p.pilot_mode and ph in ('land_target', 'land_home', 'staged_stop')
                         and self.sub in ('descend', 'pilot_hold'))
        if not v.armed:
            if pilot_landing:                     # the pilot (or PX4 auto-disarm) finished the landing
                if ph == 'land_target':
                    self._set_phase('disarm', 'pilot_landed')
                    return False, None
                if ph == 'staged_stop':
                    self._finish_staged_stop()
                    return True, None
                self._set_phase('done', 'pilot_landed_at_home')
                self._event('mission_complete', t_flight=round(now - self.t_arm, 2))
                return True, None
            if ph == 'land_target' and self.sub in ('descend', 'px4_land'):
                self._set_phase('disarm', 'px4_disarmed_after_touchdown')
                return False, None
            if ph == 'land_home' and self.sub == 'disarm':
                return False, None                # our own disarm went through: handler finishes
            if ph == 'staged_stop' and self.sub in ('descend', 'disarm', 'px4_land'):
                self._finish_staged_stop()
                return True, None
            if ph == 'land_home' and self.sub in ('descend', 'px4_land'):
                self._set_phase('done', 'px4_disarmed_at_home')
                self._event('mission_complete', t_flight=round(now - self.t_arm, 2))
                return True, None
            self._release('disarmed_in_flight')
            return True, None
        if not self._in_control():
            if pilot_landing:                     # the pilot took it to land: keep state, publish nothing
                self.sub = 'pilot_hold'
                return True, None
            if self._handoff_land and v.nav_state == NAV_AUTO_LAND:
                self.sub = 'px4_land'             # we asked (touchdown hand-off, or staged stop with stop_land_px4):
                                                  # PX4 finishes the landing and disarms itself
                return True, None
            self._release('left_offboard nav_state=%d offboard_active=%s' % (v.nav_state, v.offboard_active))
            return True, None
        if p.pilot_mode and self.sub == 'pilot_hold':
            return False, None                    # waiting for the pilot to land: no fence / time / pose aborts
        if not v.pose_ok:
            self._vf_pose_drop = True             # (v4K) -> VIO-fault handling once it is back
            if self._pose_bad_since is None:
                self._pose_bad_since = now
                self._event('pose_unhealthy')
                if self._fol is not None:
                    self._fol.v = 0.0
            if now - self._pose_bad_since > p.pose_bad_land_s:
                return True, self._abort('pose_unhealthy', brake=False)
            hold = self._last_sp if self._last_sp is not None else self.pos
            return True, self._hold_sp(hold, self.yaw_sp)
        if self._pose_bad_since is not None:
            self._event('pose_recovered', after_s=round(now - self._pose_bad_since, 2))
            self._pose_bad_since = None
        if self.t_arm is not None and now - self.t_arm > p.max_mission_s:
            return True, self._abort('mission_time_limit')
        if not self._inside(self.pos, self._trip_inset(), self.p.fence_inset_m):   # (ceiling: the airframe is flat)
            return True, self._abort('geofence', pos_start=self.sf.from_ned(self.pos))
        to = p.phase_timeout_s.get(ph)
        if to is not None and ph == 'staged_stop':
            to += p.stop_hover_s
        if to is not None and now - self.t_phase > to:
            return True, self._abort('timeout_%s_%s' % (ph, self.sub))
        return False, None

    def _abort(self, reason, brake=True, **kw):
        self.abort_reason = reason
        self._event('abort', reason=reason, **kw)
        spot = None                               # (before the phase change: it drops the follower = the flown path)
        if brake and self.veh.pose_ok and not self.p.pilot_mode and \
                (reason.endswith('_blocked') or reason.startswith('timeout')):
            spot = self._abort_land_spot()
        self._set_phase(ABORT_PHASE, reason)
        self.sub = 'brake' if (brake and self.veh.pose_ok) else 'land'
        self._hold = self.pos.copy()
        self._land_from, self._land_move_s = None, 0.0
        if spot is not None and self.sub == 'brake':
            self._land_from, self._hold = self.pos.copy(), spot
            self._land_move_s = float(np.linalg.norm((spot - self.pos)[:2])) / 0.2 + 2.0
        self._t_land_req = None
        return self._ph_abort_land()

    def _abort_land_spot(self):
        """Landing spot for a non-emergency abort: if the vehicle is within land_clear_m of a prior wall line or a
        measured board (their A-frame feet and cinder blocks stand on the floor beside them), the nearest point up to
        1 m back along the path already flown that is clear of both; None: land here."""
        p, f = self.p, self._fol
        if p.land_clear_m <= 0 or f is None or self.sf is None or not getattr(self, 'g', None):
            return None
        pos = self.pos
        B = self._boards_xy(pos[:2] - 1.6, pos[:2] + 1.6, float(pos[2]))
        B = np.zeros((0, 2)) if B is None else np.asarray(B, float).reshape(-1, 2)

        def clr(q):
            d = float(np.min(np.hypot(B[:, 0] - q[0], B[:, 1] - q[1]))) if len(B) else math.inf
            for a, b, wt in self.g.get('walls', ()):
                d = min(d, float(_seg_min_dist(a, np.asarray(b, float)[None, :], np.asarray(q, float)[None, :2])[0]))
            return d
        c0 = clr(pos)
        if c0 >= p.land_clear_m:
            return None
        s_v = f.project(pos)
        for back in np.arange(0.1, 1.0 + 1e-9, 0.1):
            q = f.ref(max(0.0, s_v - float(back)), 0.0)
            q = np.array([q[0], q[1], pos[2]])
            if not self._keep_ok(q, pos):         # (never back toward the barrier / the unseen rear of the start)
                break
            if clr(q) >= p.land_clear_m and self._valid(q):
                self._event('abort_land_moved', frm=pos.copy(), to=q.copy(), clear_before_m=round(c0, 3),
                            clear_m=round(clr(q), 3))
                return q
        return None

    def _intended_up(self):
        """Highest altitude above home any setpoint of this mission means to fly at: the prior's targets and cruise
        height, plus the over-the-wall height in the wall phases."""
        P = self.prior
        ups = [P.cruise_up_m, float(P.tunnel_entry[2]), float(P.window_center[2])]
        ups += [float(c[2]) for c in list(P.corridor) + list(P.chicane) + list(P.landing_route) +
                list(P.landing_route_gate_b) + list(P.landing_route_window)]
        if P.gate_b_center is not None:
            ups.append(float(P.gate_b_center[2]))
        if self.phase in ('climb_over_wall', 'return_home', 'land_home', 'rearm'):
            top = P.wall_top_up
            m = self.per.wall_top_d
            if m is not None and math.isfinite(float(m)):
                top = max(top, float(self.home[2] - float(m)))
            ups.append(top + self.p.wall_margin_m)
        return max(ups)

    def _release(self, reason):
        self.release_reason = reason
        self._event('release', reason=reason)
        self._set_phase('released', reason)

    # ---------------------------------------------------------------- phases
    def _ph_abort_land(self):
        v = self.veh
        if not v.armed:
            self._set_phase('aborted', self.abort_reason or '')
            return None
        if self.p.pilot_mode:
            if not self._in_control():            # the pilot has it: wait silently for the disarm
                self.sub = 'pilot_hold'
                return None
            if self.sub == 'brake' and self.now - self.t_phase < self.p.brake_s and v.pose_ok:
                return self._hold_sp(self._hold)
            if self.sub in ('brake', 'land'):
                self.sub = 'descend'
                self._desc = Descent(self._hold[:2], float(self.pos[2]), pose_only_gain=self.p.pilot_pos_gain)
                self._t_desc = self.now
            return self._pilot_landing_step(self._hold[:2])
        if v.nav_state == NAV_AUTO_LAND:          # PX4 is landing: stop streaming immediately
            self.sub = 'px4_land'
            return None
        if v.nav_state != NAV_OFFBOARD:
            self._release('left_offboard_while_aborting nav_state=%d' % v.nav_state)
            return None
        if self.sub == 'brake':
            mv = getattr(self, '_land_move_s', 0.0)
            if self.now - self.t_phase < self.p.brake_s + mv and v.pose_ok:
                fr = getattr(self, '_land_from', None)
                if fr is not None:                # (v4K) slide to the landing spot at <= 0.2 m/s
                    d = float(np.linalg.norm((self._hold - fr)[:2]))
                    k = min(1.0, 0.2 * (self.now - self.t_phase) / max(d, 1e-6))
                    if k < 1.0 or float(np.linalg.norm((self.pos - self._hold)[:2])) > 0.05:
                        return self._hold_sp(fr + k * (self._hold - fr))
                    self.sub = 'land'
                else:
                    return self._hold_sp(self._hold)
            if self.sub == 'brake':
                self.sub = 'land'
        if self.sub == 'px4_land':                # PX4 dropped AUTO_LAND back to Offboard: ask again
            self.sub = 'land'
        if self._t_land_req is None:
            self._t_land_req = self.now
        self._cmd(cmd_land())
        # fallback while PX4 has not switched yet: descend in place ourselves; disarm only with the
        # same gate as a normal landing (ToF ~0 AND PX4 landed, held) and only if LAND was ignored
        if self._desc is None:
            self._desc = Descent(self._hold[:2], float(self.pos[2]))
        sp, touched = self._desc.step(self.dt, self.now, self.pos, v.landed, self._tof(), self.p)
        if touched and self.now - self._t_land_req >= 2.0 * self.p.cmd_retry_s:
            self._cmd(cmd_disarm())
        return sp

    def _ph_wait(self):
        v, p = self.veh, self.p
        if self._stream_since is None:
            self._stream_since = self.now
        if p.pilot_mode:                          # the pilot armed + switched to Position: Offboard is ours now;
            if self.started and v.armed and self._in_control() and v.pose_ok:   # take off at once (10 s rule)
                self._record_home()
                self._set_phase('takeoff', 'pilot_offboard_active')
            return self._hold_sp(self.pos, float(v.heading))
        if v.armed:                               # never take over a vehicle someone else armed
            self._event_once('armed_before_start', 'armed_before_start')
            self._pose_ok_since = None
            return self._hold_sp(self.pos, float(v.heading))
        if v.pose_ok:
            if self._pose_ok_since is None:
                self._pose_ok_since = self.now
        else:
            self._pose_ok_since = None
        if (self.started and v.landed and self._pose_ok_since is not None
                and self.now - self._pose_ok_since >= p.wait_pose_ok_s
                and self.now - self._stream_since >= p.pre_stream_s):
            self._set_phase('arm', 'ready')
        return self._hold_sp(self.pos, float(v.heading))

    def _ph_arm(self):
        v = self.veh
        hold = self._hold_sp(self.pos, float(v.heading))
        if v.armed and v.nav_state == NAV_OFFBOARD:
            self._record_home()
            self._set_phase('takeoff', 'armed')
            return hold
        if v.armed:
            self._release('armed_outside_offboard nav_state=%d' % v.nav_state)
            return None
        if not v.pose_ok:
            self._pose_ok_since = None
            self._set_phase('wait', 'pose_unhealthy_before_arm')
            return hold
        if self.now - self.t_phase > self.p.phase_timeout_s.get('arm', 10.0):
            self._event('abort', reason='arm_timeout')
            self.abort_reason = 'arm_timeout'
            self._set_phase('aborted', 'arm_timeout')
            return None
        if v.nav_state != NAV_OFFBOARD:
            self._cmd(cmd_offboard())
        else:
            self._cmd(cmd_arm())
        return hold

    def _record_home(self):
        v, P = self.veh, self.prior
        self.home = self.pos.copy()
        self.home_yaw = float(v.heading)
        self.t_arm = self.now
        sf = self.sf = StartFrame(self.home, self.home_yaw)
        self.g = dict(
            corridor=[sf.to_ned(c) for c in P.corridor],
            tunnel_entry=sf.to_ned(P.tunnel_entry), tunnel_axis=sf.direction(P.tunnel_axis_deg),
            window_center=sf.to_ned(P.window_center), window_dir=sf.direction(P.window_through_deg),
            rock=sf.to_ned(P.landing_center),
            wall_from=sf.to_ned(P.wall_cross_from), wall_to=sf.to_ned(P.wall_cross_to),
            chicane=[sf.to_ned(c) for c in P.chicane],
            gate_center=None if P.gate_b_center is None else sf.to_ned(P.gate_b_center),
            gate_dir=sf.direction(P.gate_b_through_deg), gate_width=P.gate_b_width_m,
            walls=[(sf.to_ned((a[0], a[1], 0.0))[:2], sf.to_ned((b[0], b[1], 0.0))[:2], t) for a, b, t in P.walls],
            boulders=[(sf.to_ned((f, l, 0.0))[:2], h, r) for f, l, h, r in P.boulders])
        cp = self.p.checkpoint4 or P.checkpoint4 or ('gate_b' if P.gate_b_center is not None else 'window')
        if cp == 'gate_b' and P.gate_b_center is None:
            self._event('checkpoint4_fallback', wanted='gate_b', reason='no gate_b in the prior')
            cp = 'window'
        self.cp4 = cp
        route = (P.landing_route_gate_b if cp == 'gate_b' else P.landing_route_window) or P.landing_route
        self.g['landing_route'] = [sf.to_ned(c) for c in route]
        self._event('checkpoint4', choice=cp, chicane=len(P.chicane), landing_route=len(route))
        self._event('home', home_ned=self.home, yaw_deg=round(math.degrees(self.home_yaw), 2))
        if self.p.pilot_mode:                     # PX4 local vs cuVSLAM odom (skybeta-05: 0.46 m after a stack restart)
            od = getattr(self.veh, 'odom_d', None)
            self._event('pilot_z_offset', px4_minus_odom_m=None if od is None else round(float(self.pos[2] - od), 3),
                        px4_minus_clamp_origin_m=round(float(self.pos[2] - self.veh.clamp_origin_d), 3),
                        compensated=False)
        named = [('corridor[%d]' % i, c) for i, c in enumerate(self.g['corridor'])] + [
            ('tunnel_entry', self.g['tunnel_entry']), ('window_center', self.g['window_center']),
            ('landing', self.g['rock']), ('wall_from', self.g['wall_from']), ('wall_to', self.g['wall_to'])]
        for name, q in named:
            if not self._clear_of_fence(q):
                self._event('prior_in_fence_margin', point=name, start_frame=sf.from_ned(q))
        for i, q in enumerate(self.g['corridor']):          # coarse goals: pull them off the cage frame
            if not self._clear_of_fence(q):
                self.g['corridor'][i] = self._pull_in(q)
                self._event('corridor_goal_pulled_in', index=i, start_frame=sf.from_ned(self.g['corridor'][i]))

    def _ph_takeoff(self):
        p = self.p
        if self._fol is None:
            tgt = np.array([self.home[0], self.home[1], self._z_up(self.prior.cruise_up_m)])
            self._new_follower([tgt], p.v_up, yaw=self.home_yaw)
        r = self._fstep()
        if self.pos[2] <= self._fol.goal[2] + p.takeoff_tol_m:
            self._set_phase('corridor', 'cruise_alt')
        return r.sp

    def _standoff(self, kind):
        """Hover/detect point before a stage.  Shrinks (5 cm steps, not below standoff_min_m) until it lies
        inside the setpoint fence box, so the fence clip never moves it toward the stage (which would make
        every frame fail the detect_ahead_m check).  Computed once; the same point is the zone reference."""
        if kind in self._standoffs:
            return self._standoffs[kind].copy()
        g, p = self.g, self.p
        if kind == 'tunnel':
            c, d, s_nom = g['tunnel_entry'], g['tunnel_axis'], p.tunnel_standoff_m
        else:
            c, d, s_nom = g['window_center'], g['window_dir'], p.window_standoff_m
        z = self._clamp_alt(c[2])
        s = s_nom
        while True:
            q = c - d * s
            q[2] = z
            if self._clear_of_fence(q) or s <= min(s_nom, p.standoff_min_m) + 1e-9:
                break
            s = max(min(s_nom, p.standoff_min_m), s - 0.05)
        if not self._clear_of_fence(q):
            q = self._pull_in(q)                             # slide sideways into the clear box
            self._event('standoff_clipped', stage=kind, standoff_m=round(float(np.linalg.norm((c - q)[:2])), 3),
                        nominal_m=s_nom)
        elif s < s_nom - 1e-9:
            self._event('standoff_shrunk', stage=kind, standoff_m=round(s, 3), nominal_m=s_nom)
        self._standoffs[kind] = q
        return q.copy()

    def _insert_uturns(self, pts):
        """Serpentine U-turns: at a vertex V where the path reverses (horizontal turn >= uturn_min_deg) and the next
        leg is offset sideways from the incoming lane by >= uturn_min_shift_m, insert V + that sideways offset, so the
        path runs to the lane end, crosses perpendicular to the lane (behind / in front of the board end), and only
        then runs down the next lane.  Each half of the U is then an ordinary 90 deg corner: yaw ahead, <= 0.5 m/s,
        VIO wall standoff.  Explicit two-corner U-turns in the prior (turns < uturn_min_deg) are left as they are."""
        p = self.p
        out = [np.asarray(pts[0], float)]
        for i in range(1, len(pts) - 1):
            v, nxt = np.asarray(pts[i], float), np.asarray(pts[i + 1], float)
            out.append(v)
            u = (v - out[-2])[:2]
            w = (nxt - v)[:2]
            nu, nw = float(np.linalg.norm(u)), float(np.linalg.norm(w))
            if nu < 0.3 or nw < 0.3:
                continue
            u = u / nu
            turn = math.degrees(math.acos(float(np.clip(u @ (w / nw), -1.0, 1.0))))
            side = w - (w @ u) * u                        # sideways part of the next leg w.r.t. the incoming lane
            if turn >= p.uturn_min_deg and float(np.linalg.norm(side)) >= p.uturn_min_shift_m:
                q = v.copy()
                q[:2] = v[:2] + side
                out.append(q)
                self._event('uturn_waypoint_inserted', at=self.sf.from_ned(q) if self.sf else q,
                            turn_deg=round(turn, 1), shift_m=round(float(np.linalg.norm(side)), 2))
        out.append(np.asarray(pts[-1], float))
        return out

    def _ph_corridor(self):
        p = self.p
        if self._fol is None:
            goals = list(self.g['corridor']) + [self._standoff('tunnel')]
            goals = self._insert_uturns([self.pos.copy()] + goals)[1:]
            self._new_follower(goals, p.v_cruise, avoid='lateral', yaw_end=heading_of(self.g['tunnel_axis']),
                               side_max=p.corridor_side_max_m, gap_snap=p.wall_relative)
        r = self._fstep()
        if p.standoff_wall_clear_m > 0 and self.now - getattr(self, '_t_so_refine', -math.inf) >= 0.5:
            self._t_so_refine = self.now
            self._refine_standoff('tunnel')
        if r.status == 'stuck':
            return self._abort('corridor_blocked')
        if r.status == 'arrived':
            self._set_phase('tunnel_align', 'at_tunnel_standoff')
        return r.sp

    # ---- tunnel / window: approach -> hover+detect (N stable) -> align -> traverse
    def _feed(self, kind, require_hover=True, face_yaw=None, zone=None):
        """Consume the newest fresh detection.  Returns None | 'latched' | 'refuse'."""
        p = self.p
        est, stamp = getattr(self.per, kind), getattr(self.per, kind + '_t')
        if est is None or stamp is None:
            return None
        last = self._stamps.get(kind)
        if last is not None and stamp <= last:
            return None
        self._stamps[kind] = float(stamp)
        if require_hover and kind in ('tunnel', 'window'):
            self._adapt_latch(kind, float(stamp))
        age = self.now - stamp
        if age > p.detect_max_age_s:
            if not (require_hover and age <= p.detect_max_latency_s and self._hovered_since(float(stamp), face_yaw, zone)):
                self._event_once(('stale', kind, self.phase), 'detection_too_old', detector=kind, age_s=round(age, 2))
                return None
        if require_hover:
            if np.linalg.norm(self.vel) > p.detect_speed_max:
                return None
            if zone is not None:
                z_pt, z_dir = zone
                if np.linalg.norm(self.pos - z_pt) > p.detect_zone_m or (self.pos - z_pt) @ z_dir > p.detect_ahead_m:
                    return None
            if face_yaw is not None and abs(wrap(face_yaw - float(self.veh.heading))) > math.radians(p.detect_yaw_tol_deg):
                return None
        conf = getattr(est, 'confidence', 1.0)
        min_conf = {'tunnel': p.tunnel_min_conf, 'landing': p.landing_min_conf}.get(kind, p.detect_min_conf)
        if not getattr(est, 'found', False) or conf is None or float(conf) < min_conf:
            self._latch.miss()
            why = getattr(est, 'reason', '') or ('low_confidence' if getattr(est, 'found', False) else 'not_found')
            self._feed_outcome = ('miss', str(why))
            return None
        rejected_before = self._latch.rejected
        try:
            if kind == 'tunnel':
                stable = self._latch.feed(est.entry_ned, est.axis_ned, None, {'radius_m': est.radius_m})
            elif kind == 'window':
                stable = self._latch.feed(est.center_ned, est.normal_ned, bool(est.clearance_ok),
                                          {'width_m': est.width_m, 'height_m': est.height_m})
            else:
                stable = self._latch.feed(est.center_ned, None, None, {'top_d': est.top_d, 'extent_m': est.extent_m})
        except (AttributeError, TypeError, ValueError):
            self._latch.miss()
            self._event_once(('bad_estimate', kind), 'bad_estimate', detector=kind)
            self._feed_outcome = ('miss', 'bad_estimate')
            return None
        self._feed_outcome = ('miss', 'prior_gate') if self._latch.rejected > rejected_before else ('hit', '')
        if self._feed_outcome[0] == 'hit':
            self._t_hit = self.now
        if not stable:
            return None
        if kind == 'window' and self._latch.key is False:
            return 'refuse'
        return 'latched'

    def _adapt_latch(self, kind, stamp):
        """Detector rate -> frames to latch: ceil(detect_span_s / period) + 1, within detect_n_min..detect_n
        (round B: a frame every ~1.9 s -> 3 frames, ~4 s, instead of 5 frames, ~9.5 s)."""
        p = self.p
        st = self._det_stamps.setdefault(kind, [])
        st.append(stamp)
        del st[:-6]
        if len(st) < 3 or self._latch is None:
            return
        per = float(np.median(np.diff(st)))
        if not math.isfinite(per) or per <= 1e-3:
            return
        n = int(min(p.detect_n, max(p.detect_n_min, math.ceil(p.detect_span_s / per - 1e-9) + 1)))
        if n != self._latch.n:
            self._latch.set_n(n)
            self._event('detect_frames', detector=kind, n=n, period_s=round(per, 2))

    def _count_frame(self):
        """Update the consecutive-miss run from the last _feed outcome (only call while hovering)."""
        out, self._feed_outcome = self._feed_outcome, None
        if out is None:
            return
        if out[0] == 'hit':
            self._miss_run = 0
        else:
            self._miss_run += 1
            self._miss_reasons[out[1]] = self._miss_reasons.get(out[1], 0) + 1

    def _board_estimate(self, prior_c, prior_d):
        """Window board centre (median) and facing direction (PCA normal) from map points around the prior."""
        p, om = self.p, self.per.obstacles
        pts = getattr(om, 'points', None)
        if not (isinstance(pts, np.ndarray) and pts.ndim == 2 and pts.shape[1] == 3 and len(pts)):
            return None
        d = np.array([prior_d[0], prior_d[1], 0.0]) / max(1e-9, math.hypot(prior_d[0], prior_d[1]))
        n_lat = np.array([-d[1], d[0], 0.0])
        rel = pts - prior_c
        sel = ((np.abs(rel @ d) <= p.board_search_along_m) & (np.abs(rel @ n_lat) <= p.board_search_lat_m)
               & (rel[:, 2] >= -0.7) & (rel[:, 2] <= 0.9))
        P = pts[sel]
        if len(P) < p.board_min_points:
            return None
        c = np.median(P, axis=0)
        # along the board: the median is pulled toward the longer wall section (the hole has no points) -- use the
        # gap the hole leaves at hole height, else keep the prior's position along the board
        relc = P - prior_c
        dep = float(np.median(relc @ d))
        band = relc[np.abs(relc[:, 2]) < 0.2] @ n_lat
        along = float((c - prior_c) @ n_lat)       # (no hole found: the median, as before)
        if len(band) >= 4:
            a_s = np.sort(band)
            gaps = np.diff(a_s)
            mids = 0.5 * (a_s[1:] + a_s[:-1])
            okg = (gaps >= 0.4) & (gaps <= 1.0) & (np.abs(mids) <= min(p.board_search_lat_m, 0.8)) & \
                (np.abs(a_s[:-1]) <= 1.2) & (np.abs(a_s[1:]) <= 1.2)   # a hole-sized gap, board on both sides
            if okg.any():
                i = int(np.argmax(np.where(okg, gaps, -1.0)))
                along = float(mids[i])
        c = np.asarray(prior_c, float) + d * dep + n_lat * along + np.array([0.0, 0.0, c[2] - prior_c[2]])
        w, V = np.linalg.eigh(np.cov((P - P.mean(axis=0)).T))
        nrm = V[:, 0]
        nh = float(math.hypot(nrm[0], nrm[1]))
        out_d = d
        if nh > 0.7:
            nd = np.array([nrm[0], nrm[1], 0.0]) / nh
            if nd @ d < 0.0:
                nd = -nd
            if math.degrees(math.acos(min(1.0, float(nd @ d)))) <= 30.0:
                out_d = nd
        return c, out_d, int(len(P))

    def _reacquire_window(self, prior_c, prior_d):
        """Back off and re-centre on the board (map estimate, else the prior), facing it."""
        p = self.p
        k = self._reacq.get('window', 0) + 1
        self._reacq['window'] = k
        est = self._board_estimate(prior_c, prior_d)
        c, d = (est[0], est[1]) if est is not None else (prior_c, np.array([prior_d[0], prior_d[1], 0.0]))
        z = self._standoff('window')[2]
        s = min(p.window_standoff_max_m, p.window_standoff_m + k * p.reacquire_step_m)
        q, ang, s0 = None, 0.0, s
        for ang in (0.0, 6.0, -6.0, 12.0, -12.0):   # on the axis first (any standoff); a net close behind: up to
            a = math.radians(ang)                     # 12 deg off the axis (still "enclosed")
            dd = np.array([d[0] * math.cos(a) - d[1] * math.sin(a), d[0] * math.sin(a) + d[1] * math.cos(a)])
            s = s0
            while s >= p.standoff_min_m - 1e-9:      # fits the fence AND the move there is clear
                c_q = np.array([c[0] - dd[0] * s, c[1] - dd[1] * s, z])
                if self._clear_of_fence(c_q) and self._probe([self.pos, c_q], strict=True) == math.inf:
                    q = c_q
                    break
                s -= 0.05
            if q is not None:
                break
        self._event('reacquire', stage='window', attempt=k, standoff_m=round(s, 2) if q is not None else None,
                    source='map' if est else 'prior', board_points=est[2] if est else 0,
                    reasons=dict(self._miss_reasons), moved=q is not None, off_axis_deg=ang if q is not None else None)
        self._miss_run, self._miss_reasons = 0, {}
        if q is None:                             # nowhere clear: keep detecting from here
            return
        self._standoffs['window'] = q
        self._stage_dir['window'] = d
        self._latch.miss()
        self._new_follower([q], p.v_align, avoid='stop', yaw_end=heading_of(c - q))   # face the board centre
        self.sub = 'approach'

    def _align_phase(self, kind):
        p, g, P = self.p, self.g, self.prior
        if kind == 'tunnel':
            prior_c, prior_d, tol, align_m = g['tunnel_entry'], g['tunnel_axis'], P.tunnel_tol_m, p.tunnel_align_m
        else:
            prior_c, prior_d, tol, align_m = g['window_center'], g['window_dir'], P.window_tol_m, p.window_align_m
        stage_d = self._stage_dir.get(kind, prior_d)
        face = heading_of(stage_d)
        self._hints['detector'] = kind
        if self.sub == '':
            self._latch = Latch(p.detect_n, p.detect_pos_tol_m, p.detect_ang_tol_deg, prior=prior_c,
                                gate=tol + p.detect_prior_gate_m, prior_dir=prior_d, prior_ang_deg=p.detect_prior_ang_deg)
            self._new_follower([self._standoff(kind)], p.v_cruise, avoid='lateral', yaw_end=face)
            self.sub = 'approach'
        if self.sub in ('approach', 'detect'):
            res = self._feed(kind, require_hover=True, face_yaw=face, zone=(self._standoff(kind), stage_d))
            if self.sub == 'detect':
                self._count_frame()
                if (kind == 'window' and self._miss_run >= p.reacquire_n
                        and self._reacq.get('window', 0) < p.reacquire_max):
                    self._reacquire_window(prior_c, prior_d)
                    if self.sub == 'approach':
                        return self._fstep().sp
            if res == 'refuse':
                pos_, d_, ex_ = self._latch.value()
                return self._abort('window_clearance_refused', center=pos_, width_m=ex_.get('width_m'),
                                   height_m=ex_.get('height_m'))
            if res == 'latched':
                out = self._start_align(kind, align_m)
                if out is not None:
                    return out
            elif (kind == 'tunnel' and self.sub == 'detect' and p.tunnel_depth_fallback_s > 0 and
                  self.now - getattr(self, '_detect_t0', self.now) >= p.tunnel_depth_fallback_s and
                  (self.now - max(getattr(self, '_detect_t0', self.now), getattr(self, '_t_hit', -math.inf))
                   >= p.tunnel_depth_fallback_s or
                   self.now - getattr(self, '_detect_t0', self.now) >= 2.5 * p.tunnel_depth_fallback_s) and
                  self.now - getattr(self, '_t_depth_try', -math.inf) >= 0.5):
                # (the detector found nothing for that long -- or it sees something but never latches: 2.5x)
                self._t_depth_try = self.now      # no hoop latch: the opening from depth
                est = self._tunnel_from_depth()
                if est is not None:
                    c, r_, info = est
                    self._latch = _FixedLatch(c, g['tunnel_axis'], {'radius_m': r_})
                    self._tunnel_mode = 'depth_fallback'
                    self._event('tunnel_mode', mode='depth_fallback', center=c, radius_m=round(r_, 3), **info)
                    out = self._start_align(kind, align_m)
                    if out is not None:
                        return out
        if self.sub == 'approach':
            r = self._fstep()
            if r.status == 'stuck':
                return self._abort('%s_approach_blocked' % kind)
            if r.status == 'arrived':
                self.sub = 'detect'
                self._detect_t0 = self.now
                self._hold = self._fol.end_ref.copy()
                if np.linalg.norm(self._fol.goal - self._standoff(kind)) > 0.05:   # clipped by the VIO standoff:
                    self._standoffs[kind] = self._fol.goal.copy()                  # detect from where we hover
                self._miss_run, self._miss_reasons, self._feed_outcome = 0, {}, None
            return r.sp
        if self.sub == 'detect':
            return self._hold_sp(self._hold, face)
        # align
        r = self._fstep()
        if r.status == 'stuck':
            return self._abort('%s_align_blocked' % kind)
        if self._commit_ready(kind):
            self._set_phase(kind + '_traverse', 'aligned')
        return r.sp

    def _axis_errors(self, kind):
        """(along, horizontal cross-axis, vertical, perpendicular velocity) w.r.t. the latched axis line."""
        L = self._latched[kind]
        c, ax = L['center'], L['axis']
        e = self.pos - c
        along = float(e @ ax)
        perp = e - along * ax
        v_perp = self.vel - float(self.vel @ ax) * ax
        return along, perp, float(np.linalg.norm(perp[:2])), float(abs(perp[2])), float(np.linalg.norm(v_perp))

    def _commit_ready(self, kind):
        """Pre-traverse stability gate: on the axis (cross-axis and vertical) and still, held commit_hold_s."""
        p, L = self.p, self._latched[kind]
        along, _, lat, vert, vperp = self._axis_errors(kind)
        k = p.commit_relax_factor if self.now - self._t_align > p.commit_relax_after_s else 1.0
        yaw_ok = abs(wrap(L['yaw'] - float(self.veh.heading))) < math.radians(p.settle_yaw_deg)
        ok = (lat < p.commit_lat_tol_m * k and vert < p.commit_vert_tol_m * k and vperp < p.commit_lat_vel * k
              and abs(along + L['align_m']) < p.commit_along_tol_m and yaw_ok)
        if not ok:
            self._commit_since = None
            return False
        if self._commit_since is None:
            self._commit_since = self.now
        if self.now - self._commit_since < p.commit_hold_s:
            return False
        self._event('commit', stage=kind, lat_m=round(lat, 3), vert_m=round(vert, 3), vperp=round(vperp, 3),
                    relaxed=k > 1.0, align_s=round(self.now - self._t_align, 2))
        return True

    def _start_align(self, kind, align_m):
        p = self.p
        c, d, ex = self._latch.value()
        axis = np.array([d[0], d[1], 0.0])
        if kind == 'tunnel':
            radius = ex.get('radius_m')
            if radius is not None and radius < p.tunnel_min_radius_m:
                return self._abort('tunnel_too_small', radius_m=radius)
            z = self._clamp_alt(c[2])
            slack = (radius if radius is not None else 0.4) - p.half_span_m - 0.05
            if abs(z - c[2]) > max(0.05, slack):
                return self._abort('tunnel_out_of_altitude_band', entry=c)
        else:
            z = self._clamp_alt(c[2])
            if abs(z - c[2]) > 0.05:
                return self._abort('window_out_of_altitude_band', center=c)
        center = np.array([c[0], c[1], z])
        align = center - axis * align_m
        yaw = heading_of(axis)
        self._latched[kind] = dict(center=center, axis=axis, align=align, yaw=yaw, extra=ex)
        here = float((center - self.pos) @ axis)            # how far in front of the opening we hover now
        if kind == 'tunnel':
            L, r_ex = self.prior.tunnel_length_m, (radius if radius is not None else 0.4) - p.tunnel_exclude_margin_m
            front = max(p.tunnel_exclude_front_m, here + 0.15) if p.exclude_to_standoff else p.tunnel_exclude_front_m
            if r_ex > 0.05:
                self._exclude[kind] = [(center + axis * (L - front) / 2.0, axis, r_ex, (L + front) / 2.0)]
        else:
            w, h = ex.get('width_m'), ex.get('height_m')
            zones = []
            if w and h and min(w, h) / 2.0 - p.window_exclude_margin_m > 0.05:
                back, front = p.window_exclude_half_len_m, p.window_exclude_front_m
                zones.append((center + axis * (back - front) / 2.0, axis,
                              min(w, h) / 2.0 - p.window_exclude_margin_m, (back + front) / 2.0))
            if p.exclude_to_standoff and here + 0.15 > p.window_exclude_front_m:
                a0, a1 = 0.15, here + 0.15                   # open space in front of the board, up to the hover point
                zones.append((center - axis * (a0 + a1) / 2.0, axis, p.window_exclude_approach_r_m, (a1 - a0) / 2.0))
            if zones:
                self._exclude[kind] = zones
        self._event('latched', detector=kind, center=center, axis_deg=round(math.degrees(yaw), 1),
                    **{k: (round(v, 3) if v is not None else None) for k, v in ex.items()})
        self._latched[kind]['align_m'] = align_m
        self._new_follower([align], p.v_align, avoid='stop', yaw=yaw, tol=p.settle_tol_m)
        self.sub = 'align'
        self._t_align, self._commit_since = self.now, None
        return None

    def _traverse_phase(self, kind, length, exit_m, v, nxt):
        p, L = self.p, self._latched[kind]
        c, ax = L['center'], L['axis']
        depth = kind == 'tunnel' and getattr(self, '_tunnel_mode', '') == 'depth_fallback'
        if self._fol is None:
            end = c + ax * (length + exit_m)
            start = c + ax * float((self.pos - c) @ ax)       # straight line ON the latched axis
            if depth:                             # no detector axis: a narrow wall-relative corridor (side walls /
                r_ = float(L['extra'].get('radius_m') or 0.4)   # arch from the map; the middle vertex is centred)
                self._new_follower([c, c + ax * (0.5 * length), end], min(v, p.v_align), avoid='lateral',
                                   yaw=L['yaw'], run_out=p.run_out_m, start=start,
                                   side_max=max(0.0, r_ - p.half_span_m - 0.05), gap_snap=p.wall_relative)
            else:
                self._new_follower([c, end], v, avoid=None, yaw=L['yaw'], run_out=p.run_out_m, start=start)
            self._plane_logged = False
        r = self._fstep()
        along, perp, lat, vert, _ = self._axis_errors(kind)
        if not self._plane_logged and along >= 0.0:
            self._plane_logged = True
            self._event('plane_crossing', stage=kind, lat_m=round(lat, 3), vert_m=round(vert, 3))
        pos_sp, vel_sp, acc_sp, yaw = r.sp
        corr = -p.traverse_lat_gain * perp if not depth else np.zeros(3)   # (depth: the walls centre us)
        n = float(np.linalg.norm(corr))
        if n > p.traverse_lat_vmax:
            corr *= p.traverse_lat_vmax / n
        if r.status == 'stuck':
            return self._abort('%s_traverse_blocked' % kind)
        if r.status == 'passed':
            self._set_phase(nxt, '%s_exit' % kind)
        if p.pilot_mode:                          # pose only: express the correction as a position offset
            return (pos_sp + corr / p.pilot_pos_gain, vel_sp, acc_sp, yaw)
        return (pos_sp, vel_sp + corr, acc_sp, yaw)

    def _refine_standoff(self, kind):
        """The stage standoff moves toward the stage (5 cm steps, not below standoff_wall_min_m) until it is
        >= standoff_wall_clear_m from MEASURED walls (official run_005/009: hovering at the tunnel standoff 0.4 m from
        the north wall, then the 180 deg turn there -> VIO drift / geofence).  Moves the running follower's end too."""
        p, g = self.p, self.g
        if kind not in self._standoffs:
            return
        q0 = self._standoffs[kind]
        c, d = (g['tunnel_entry'], g['tunnel_axis']) if kind == 'tunnel' else (g['window_center'], g['window_dir'])
        s_cur = float(np.linalg.norm((c - q0)[:2]))
        lo = min(s_cur, p.standoff_wall_min_m)
        ds = np.arange(s_cur, lo - 1e-9, -0.05)
        if len(ds) < 2:
            return
        Q = c[None, :] - ds[:, None] * d[None, :]
        Q[:, 2] = q0[2]
        cl = self._clear_xy(Q, float(q0[2]))
        if cl is None or not len(cl) or cl[0] >= p.standoff_wall_clear_m:
            return
        okk = np.nonzero((cl >= p.standoff_wall_clear_m) & np.array([self._clear_of_fence(q) for q in Q]))[0]
        if not len(okk):
            return                                # nothing clear enough on the axis: leave it (VIO clip handles it)
        j = int(okk[0])
        if ds[j] > s_cur - 0.04:
            return
        q = Q[j].copy()
        B = self._boards_xy(np.minimum(q0[:2], q[:2]) - 0.6, np.maximum(q0[:2], q[:2]) + 0.6, float(q0[2]))
        if B is not None and len(B) and float(_seg_min_dist(q0, q[None, :2], B)[0]) < p.half_span_m:
            return                                # never move it through / past a wall
        self._standoffs[kind] = q
        self._event('standoff_wall_shrunk', stage=kind, standoff_m=round(float(ds[j]), 3),
                    clearance_m=round(float(cl[j]), 3))
        f = self._fol
        if f is not None and self.phase == 'corridor' and f.n_goal >= 2 and f.S_goal - f.s > 0.5 and f._clip is None:
            k = f.n_goal - 1
            f.P[k], f.P0[k] = q.copy(), q.copy()
            s_keep = f.s
            f._set_path(f.P, f.n_goal)
            f.s = min(s_keep, f.S)

    def _tunnel_from_depth(self):
        """(entry centre NED, free radius, info) of the tunnel opening measured in the obstacle map, or None.
        Cross-section = map points 0..tunnel_depth_slab_m behind the prior entry plane, within prior tol + 0.9 m of
        the prior axis; candidates around the prior centre score min(distance to those points, height above the
        floor); the best one is the opening centre.  Both side walls must be measured; with no arch measured the
        height stays at the prior's (never climbs out of an open top)."""
        p, g, P = self.p, self.g, self.prior
        om = self.per.obstacles
        pts = getattr(om, 'points', None) if om is not None else None
        if not (isinstance(pts, np.ndarray) and pts.ndim == 2 and len(pts)):
            return None
        c0, u = g['tunnel_entry'], np.asarray(g['tunnel_axis'], float)
        u = np.array([u[0], u[1], 0.0]) / max(1e-9, math.hypot(u[0], u[1]))
        n = np.array([-u[1], u[0], 0.0])
        floor_z = float(self.home[2])
        rel = pts - c0
        a = rel[:, 0] * u[0] + rel[:, 1] * u[1]
        b = rel[:, 0] * n[0] + rel[:, 1] * n[1]
        h = floor_z - pts[:, 2]
        wmax = P.tunnel_tol_m + 0.9
        sel = (a >= 0.0) & (a <= min(p.tunnel_depth_slab_m, P.tunnel_length_m)) & (np.abs(b) <= wmax) & (h > 0.05) & \
            (h < 2.6)
        B, H = b[sel], h[sel]
        if len(B) < 2 * p.tunnel_depth_min_pts:
            return None
        up0 = float(floor_z - c0[2])
        bb, hh = np.meshgrid(np.arange(-P.tunnel_tol_m - 0.3, P.tunnel_tol_m + 0.3 + 1e-9, 0.05),
                             np.arange(p.min_alt_m, 2.0 + 1e-9, 0.05))
        bb, hh = bb.ravel(), hh.ravel()
        d = np.full(len(bb), np.inf)
        for i in range(0, len(B), 4000):
            db = bb[:, None] - B[None, i:i + 4000]
            dh = hh[:, None] - H[None, i:i + 4000]
            d = np.minimum(d, np.sqrt(np.min(db * db + dh * dh, axis=1)))
        clr = np.minimum(d, hh)
        top = np.array([bool(np.any((np.abs(B - x) < 0.3) & (H > y))) for x, y in zip(bb, hh)])
        clr[~top & (hh > up0 + 1e-6)] = -np.inf    # no arch measured above: not higher than the prior
        score = clr - 0.1 * np.abs(hh - up0) - 0.05 * np.abs(bb)
        j = int(np.argmax(score))
        bc, hc, r = float(bb[j]), float(hh[j]), float(clr[j])
        band = np.abs(H - hc) < 0.4
        nl, nr = int(np.count_nonzero(band & (B < bc - 0.2))), int(np.count_nonzero(band & (B > bc + 0.2)))
        if nl < p.tunnel_depth_min_pts or nr < p.tunnel_depth_min_pts or not math.isfinite(r):
            self._event_once(('tunnel_depth_wait',), 'tunnel_depth_unmeasured', left_pts=nl, right_pts=nr)
            return None
        if r < p.half_span_m + 0.05:
            self._event_once(('tunnel_depth_narrow',), 'tunnel_depth_too_narrow', radius_m=round(r, 3))
            return None
        c = c0 + n * bc
        c = np.array([c[0], c[1], floor_z - hc])
        wl = B[band & (B < bc)]
        wr = B[band & (B > bc)]
        info = dict(width_m=round(float(wr.min() - wl.max()), 3) if len(wl) and len(wr) else None,
                    lateral_m=round(bc, 3), up_m=round(hc, 3), arch=bool(top[j]), points=int(len(B)))
        return c, r, info

    def _ph_tunnel_align(self):
        return self._align_phase('tunnel')

    def _ph_tunnel_traverse(self):
        return self._traverse_phase('tunnel', self.prior.tunnel_length_m, self.p.tunnel_exit_m, self.p.v_tunnel,
                                    'chicane' if self.g['chicane'] else self._cp4_phase())

    def _gate_standoff_nominal(self):
        P = self.prior
        return float(P.gate_b_standoff_m if P.gate_b_standoff_m is not None else self.p.gate_standoff_m)

    def _gate_line_up(self):
        """(Gate B line-up point, standoff): on the gate axis, the nominal standoff shrunk (5 cm steps, not below
        gate_standoff_min_m) until the point clears mapped boards by lane_clear_m (C1 runs 0.37 m from the
        nominal one in the venue map); none clears -> the clearest."""
        p, g = self.p, self.g
        c, u = g['gate_center'], g['gate_dir']
        d0 = self._gate_standoff_nominal()
        ds = np.arange(d0, min(d0, p.gate_standoff_min_m) - 1e-9, -0.05)
        Q = c[None, :] - ds[:, None] * u[None, :]
        cl = self._clear_xy(Q, float(c[2]))
        if cl is None or not len(cl):
            return Q[0].copy(), float(ds[0])
        okk = np.nonzero(cl >= p.lane_clear_m)[0]
        j = int(okk[0]) if len(okk) else int(np.argmax(cl))
        return Q[j].copy(), float(ds[j])

    def _cp4_line_up(self):
        """Where checkpoint 4 starts: Gate B line-up point, or the window standoff."""
        return self._gate_line_up()[0] if self.cp4 == 'gate_b' else self._standoff('window')

    def _cp4_phase(self):
        return 'gate_pass' if self.cp4 == 'gate_b' else 'window_align'

    def _after_cp4(self):
        return 'landing_route' if self.g['landing_route'] else 'land_target'

    # ---- optional stages of the official course map (all prior-driven)
    def _ph_chicane(self):
        """Tunnel exit -> the prior's chicane waypoints (around C1) -> checkpoint 4 (window or Gate B)."""
        p, g = self.p, self.g
        if self._fol is None:
            nxt = g['gate_dir'] if self.cp4 == 'gate_b' else g['window_dir']
            self._new_follower(list(g['chicane']), p.v_cruise, avoid='lateral', yaw_end=heading_of(nxt),
                               side_max=p.corridor_side_max_m, gap_snap=True, next_after=self._cp4_line_up())
            self._t_lineup = self.now
        elif self.now - getattr(self, '_t_lineup', -math.inf) >= 0.5:
            self._t_lineup = self.now             # the line-up point follows the map (Gate B standoff vs C1)
            q = self._cp4_line_up()
            self._fol.next_after = q
            self._fol_kw['next_after'] = q
        r = self._fstep()
        if r.status == 'stuck':
            return self._abort('chicane_blocked')
        if r.status == 'arrived':
            self._set_phase(self._cp4_phase(), 'chicane_done')
        return r.sp

    def _ph_gate_pass(self):
        """Gate B: an open gap -- no detector.  A centred line (standoff -> centre -> exit) through it; the centre is
        re-centred on the measured gap edges like any corridor vertex, side steps stay inside the gap."""
        p, g = self.p, self.g
        c, u = g['gate_center'], g['gate_dir']
        if self._fol is None:
            P = self.prior
            d0 = self._gate_standoff_nominal()
            self._gate_e0, d = self._gate_line_up()
            if d < d0 - 1e-6:
                self._event('gate_standoff_shrunk', standoff_m=round(d, 3), nominal_m=round(d0, 3))
            self._gate_e1 = c + u * float(P.gate_b_exit_m if P.gate_b_exit_m is not None else p.gate_exit_m)
            self._gate_crossed = False
            self.sub = 'approach'                 # to the line-up point (normal side steps), then the gap line
            self._new_follower([self._gate_e0], p.v_cruise, avoid='lateral', yaw_end=heading_of(u),
                               side_max=p.corridor_side_max_m)
        if self.sub == 'approach':
            r = self._fstep()
            if r.status == 'stuck':
                return self._abort('gate_approach_blocked')
            if r.status != 'arrived':
                return r.sp
            side = max(0.0, 0.5 * g['gate_width'] - p.prop_radius_m - p.gate_side_margin_m)
            self._new_follower([c.copy(), self._gate_e1], p.v_gate, avoid='lateral', yaw=heading_of(u),
                               side_max=side)
            self.sub = 'line'
        if not self._gate_crossed and float((self.pos - c)[:2] @ u[:2]) >= 0.0:
            self._gate_crossed = True
            lat = np.array([-u[1], u[0], 0.0])
            cen = self._fol.P[1] if len(self._fol.P) > 2 else c
            self._event('gate_passed', lat_m=round(float((self.pos - c) @ lat), 3),
                        lat_from_centred_m=round(float((self.pos - cen) @ lat), 3), up_m=round(float(c[2] - self.pos[2]), 3))
        r = self._fstep()
        if r.status == 'stuck':
            return self._abort('gate_blocked')
        if r.status == 'arrived':
            if not self._gate_crossed:
                return self._abort('gate_not_crossed')
            self._set_phase(self._after_cp4(), 'gate_b_passed')
        return r.sp

    def _ph_landing_route(self):
        """Checkpoint 4 -> the prior's landing-route waypoints (e.g. around W4) -> landing observe pose."""
        p, g = self.p, self.g
        if self._fol is None:
            last = g['landing_route'][-1]
            self._new_follower(list(g['landing_route']), p.v_cruise, avoid='lateral',
                               yaw_end=heading_of(g['rock'] - last), side_max=p.corridor_side_max_m,
                               gap_snap=p.wall_relative)
        r = self._fstep()
        if r.status == 'stuck':
            return self._abort('landing_route_blocked')
        if r.status == 'arrived':
            self._set_phase('land_target', 'landing_route_done')
        return r.sp

    def _ph_window_align(self):
        return self._align_phase('window')

    def _ph_window_traverse(self):
        return self._traverse_phase('window', 0.0, self.p.window_exit_m, self.p.v_window, self._after_cp4())

    # ---- landing target (blue H pad on the floor): approach + observe (horizontal camera) -> over -> descend
    def _ph_land_target(self):
        p, g = self.p, self.g
        rock = g['rock']
        self._hints['detector'] = 'landing'
        if self.sub == '':
            self._latch = Latch(p.detect_n, p.detect_pos_tol_m, p.detect_ang_tol_deg, prior=rock,
                                gate=max(self.prior.landing_tol_m + p.detect_prior_gate_m, p.landing_gate_m),
                                window=p.landing_window)
            d = rock[:2] - self.pos[:2]
            dist = float(np.linalg.norm(d))
            u = d / dist if dist > 1e-3 else np.array([math.cos(self.veh.heading), math.sin(self.veh.heading)])
            self._obs_u = u
            self._obs_dist = self._observe_dist()
            z = self._clamp_alt(rock[2] - p.observe_above_top_m)
            pick = self._pick_observe(rock, z, self._observe_min_dist(z), self._obs_dist)
            if pick is not None:
                obs, self._obs_dist, why = pick
                du = rock[:2] - obs[:2]
                self._obs_u = du / max(float(np.linalg.norm(du)), 1e-6)
            else:                                     # nothing checked clear: the old approach-line point
                obs_xy = rock[:2] - u * self._obs_dist if dist > self._obs_dist else self.pos[:2]
                obs = np.array([obs_xy[0], obs_xy[1], z])
                obs = obs if self._clear_of_fence(obs) else self._pull_in(obs)
                why = 'approach_line_fallback'
            self._face = math.atan2(self._obs_u[1], self._obs_u[0])
            self._event('observe_pose', pos=obs, observe_dist_m=round(self._obs_dist, 2), reason=why,
                        min_dist_m=round(self._observe_min_dist(z), 2))
            self._obs_target, self._obs_bad, self._obs_chk_t, self._obs_z = obs.copy(), [], self.now, z
            self._go_observe(obs)
            self.sub = 'approach'
        if self.sub == 'approach' and self.now - self._obs_chk_t >= 0.5 and self._latch is not None and \
                self._latched.get('landing') is None:
            self._obs_chk_t = self.now            # re-check the pose against the LIVE map; fall back to another one
            if not self._observe_ok(self._obs_target, rock):
                self._obs_bad.append(self._obs_target.copy())
                self._event('observe_pose_invalid', pos=self._obs_target)
                z = self._obs_z
                pick = self._pick_observe(rock, z, self._observe_min_dist(z), self._observe_dist(), avoid=self._obs_bad)
                if pick is None or len(self._obs_bad) > 4:   # nothing else checks out: observe from here
                    self._event('observe_pose_none', pos=self.pos)
                    self._fol = None
                    self.sub = 'observe'
                    self._hold = self.pos.copy()
                    self._miss_run, self._miss_reasons, self._feed_outcome = 0, {}, None
                    return self._hold_sp(self._hold, self._face)
                if pick is not None:
                    obs, self._obs_dist, why = pick
                    du = rock[:2] - obs[:2]
                    self._obs_u = du / max(float(np.linalg.norm(du)), 1e-6)
                    self._face = math.atan2(self._obs_u[1], self._obs_u[0])
                    self._obs_target = obs.copy()
                    self._event('observe_pose', pos=obs, observe_dist_m=round(self._obs_dist, 2), reason='fallback',
                                min_dist_m=round(self._observe_min_dist(z), 2))
                    self._go_observe(obs)
        if self.sub in ('approach', 'observe'):
            res = self._feed('landing', require_hover=False)
            if self.sub == 'observe' and res != 'latched':
                self._count_frame()
                if self._miss_run >= p.reacquire_n and self._reacq.get('landing', 0) < p.reacquire_max:
                    return self._reacquire_rock(rock)
            if res == 'latched':
                c, _, ex = self._latch.value()
                top = ex.get('top_d')
                top = float(c[2]) if top is None else float(top)
                self._latched['landing'] = dict(center=np.array([c[0], c[1], top]), extent_m=ex.get('extent_m'))
                self._event('latched', detector='landing', center=self._latched['landing']['center'],
                            extent_m=ex.get('extent_m'))
                over = np.array([c[0], c[1], min(self._clamp_alt(top - p.over_up_m), top - p.over_up_m)])
                self._new_follower([over], p.v_cruise, avoid='lateral', yaw=None, yaw_end=self._face)
                self.sub = 'over'
        if self.sub == 'approach':
            r = self._fstep()
            if r.status == 'stuck':
                return self._abort('landing_approach_blocked')
            if r.status == 'arrived':
                self.sub = 'observe'
                self._hold = self._fol.end_ref.copy()
                self._miss_run, self._miss_reasons, self._feed_outcome = 0, {}, None
            return r.sp
        if self.sub == 'observe':
            return self._hold_sp(self._hold, self._face)
        if self.sub == 'over':
            r = self._fstep()
            if r.status == 'stuck':
                return self._abort('landing_over_blocked')
            tgt = self._fol.goal
            if self._settled(tgt, xy_only=True, tol=p.land_settle_tol_m):
                L = self._latched['landing']
                self._desc = Descent(tgt[:2], float(self.pos[2]), surface_z=float(L['center'][2]),
                                     pose_only_gain=p.pilot_pos_gain if p.pilot_mode else None)
                self._t_desc = self.now
                self.sub = 'descend'
            return r.sp
        # descend
        if p.pilot_mode:
            return self._pilot_landing_step(self._desc.xy)
        if self.now - self._t_desc > p.descend_timeout_s:
            return self._abort('no_touchdown_confirmation_on_target', tof_m=self._tof(), landed=bool(self.veh.landed))
        sp, touched = self._desc.step(self.dt, self.now, self.pos, self.veh.landed, self._tof(), p)
        if touched:
            self._set_phase('disarm', 'touchdown_on_target')
        else:
            self._maybe_handoff()
        return sp

    def _observe_dist(self):
        p = self.p
        geo = p.observe_above_top_m / math.tan(math.radians(p.observe_max_depression_deg)) + self.prior.landing_tol_m
        return min(p.observe_dist_max_m, max(p.observe_dist_m, geo, self._observe_min_dist()))

    def _observe_min_dist(self, z=None):
        """Horizontal camera: the whole pad (its near edge) is inside the image from >= this far (scales with the
        camera height above the pad top; official-compat pad_observe.py: 1.46 m at 0.55 m)."""
        p = self.p
        h = p.observe_above_top_m if z is None else max(0.05, float(self.g['rock'][2]) - float(z))
        return (h / math.tan(math.radians(p.cam_half_vfov_down_deg)) + p.pad_half_m + p.observe_view_margin_m
                + p.cam_forward_m)                # base_link distance (the camera sits cam_forward_m closer)

    def _wall_side_ok(self, q):
        """q stays on the landing-pad side of the wall prior (never plan an observe pose behind the wall)."""
        g = self.g
        a, b = g.get('wall_from'), g.get('wall_to')
        if a is None or b is None:
            return True
        n = (b - a)[:2]
        ln = float(np.linalg.norm(n))
        mid = 0.5 * (a + b)[:2]                   # the wall stands between cross_from and cross_to
        return ln < 1e-6 or float((np.asarray(q, float)[:2] - mid) @ n) / ln <= -self.p.wall_side_margin_m

    def _observe_ok(self, c, rock):
        """Live-map check of an observe pose: inside the fence, pad side of the wall, the point itself clear (strict
        radius), line of sight camera -> pad centre clear."""
        p = self.p
        if not self._clear_of_fence(c) or not self._wall_side_ok(c):
            return False
        if self._probe([c, c + np.array([0.0, 0.0, -0.02])], strict=True) != math.inf:
            return False
        e = (rock - c)[:2]
        n = float(np.linalg.norm(e))
        e = e / n if n > 1e-6 else np.array([1.0, 0.0])
        cam = c + p.cam_forward_m * np.array([e[0], e[1], 0.0])
        return self._probe([cam, np.array([rock[0], rock[1], rock[2] - 0.05])], radius=0.05) == math.inf

    def _pick_observe(self, rock, z, d_lo, d_pref, exclude_near=None, avoid=()):
        """Observe pose around the prior pad centre: >= d_lo (whole pad in view), inside the fence, clear of the map,
        a clear straight path from here, on the pad side of the wall, line of sight to the pad; facing the pad.
        Prefers d_pref and a short move.  Returns (pos, dist, reason) or None."""
        p = self.p
        here = self.pos
        dist_now = float(np.linalg.norm(rock[:2] - here[:2]))
        dists = sorted({round(x, 3) for x in (d_pref, d_pref + 0.5, d_lo, 0.5 * (d_lo + d_pref)) if x >= d_lo - 1e-6
                        and x <= p.observe_dist_max_m + 0.5})
        best = None
        if d_lo - 1e-6 <= dist_now <= d_pref + 0.5 and exclude_near is None and self._clear_of_fence(here):
            cam = here + p.cam_forward_m * np.array([*((rock - here)[:2] / max(dist_now, 1e-6)), 0.0])
            if self._probe([cam, np.array([rock[0], rock[1], rock[2] - 0.05])], radius=0.05) == math.inf:
                best = (abs(dist_now - d_pref), np.array([here[0], here[1], z]), dist_now)   # observe from here
        n_b = max(4, int(round(360.0 / p.observe_bearing_step_deg)))
        for k in range(n_b):
            a = 2.0 * math.pi * k / n_b
            e = np.array([math.cos(a), math.sin(a)])
            for d in dists:
                c = np.array([rock[0] + d * e[0], rock[1] + d * e[1], z])
                if exclude_near is not None and float(np.linalg.norm(c[:2] - exclude_near[:2])) < 0.2:
                    continue
                if any(float(np.linalg.norm(c[:2] - a[:2])) < 0.35 for a in avoid):
                    continue                      # a pose that already failed the live re-check
                if not self._clear_of_fence(c) or not self._wall_side_ok(c):
                    continue
                move = float(np.linalg.norm(c - here))
                if move > 0.05:                       # the follower's chain: hard radius leaving the vehicle, then soft
                    mid = here + (c - here) * min(1.0, 0.3 / move)
                    if self._probe([here, mid, c]) != math.inf:
                        continue
                if self._probe([c, c + np.array([0.0, 0.0, -0.02])], strict=True) != math.inf:
                    continue
                cam = c - p.cam_forward_m * np.array([e[0], e[1], 0.0])
                if self._probe([cam, np.array([rock[0], rock[1], rock[2] - 0.05])], radius=0.05) != math.inf:
                    continue
                t = (c - here)[:2]
                tn = float(np.linalg.norm(t))
                back = max(0.0, float(t @ e) / tn) if tn > 1e-6 else 0.0   # moving away from the pad (unseen)
                cost = move * (1.0 + 0.5 * back) + abs(d - d_pref)
                samples = [here + (c - here) * u for u in np.linspace(0.25, 1.0, max(2, int(move / 0.2)))]
                if not all(self._seen(x) for x in samples):
                    cost += 2.0                   # not in the recent camera view: the live map cannot vouch for it
                if best is None or cost < best[0]:
                    best = (cost, c, d)
        if best is None:
            return None
        reason = 'here' if float(np.linalg.norm(best[1][:2] - here[:2])) < 0.1 else (
            'back_off' if best[2] > dist_now else 'approach')
        return best[1], float(best[2]), reason

    def _go_observe(self, obs):
        """Fly to the observe pose.  Moving away from / around the pad: keep the camera ON the pad (fixed yaw, slow,
        stop-only) -- never swing toward a nearby wall to look where we go."""
        p = self.p
        t = (obs - self.pos)[:2]
        tn = float(np.linalg.norm(t))
        fwd = np.array([math.cos(self._face), math.sin(self._face)])
        if tn > 1e-3 and float(t @ fwd) / tn < 0.5:
            self._new_follower([obs], p.v_backoff, avoid='lateral', yaw=self._face)   # may start near a wall: step off
            self._fol.vio_check = False           # (a slow, short move to a checked pose: never clipped short)
        else:
            self._new_follower([obs], p.v_cruise, avoid='lateral', yaw_end=self._face)

    def _reacquire_rock(self, rock):
        """The rock top keeps failing (no_flat_top / top_not_visible / out of view): back off along the approach
        line (smaller depression angle) and make sure we are at least observe_above_top_m above the top."""
        p = self.p
        k = self._reacq.get('landing', 0) + 1
        self._reacq['landing'] = k
        u = self._obs_u
        z = min(self.pos[2], self._clamp_alt(rock[2] - p.observe_above_top_m))
        want = min(p.observe_dist_max_m, self._obs_dist + p.reacquire_step_m)
        q = None
        d_lo = max(self._observe_min_dist(z), min(want, float(np.linalg.norm(rock[:2] - self.pos[:2])) + 0.2))
        pick = self._pick_observe(rock, z, d_lo, want, exclude_near=self.pos)
        if pick is not None:
            q, self._obs_dist, _ = pick
        else:
            for dist in np.arange(want, self._obs_dist + 0.05, -0.1):     # the back-off itself must be clear
                c = np.array([rock[0] - u[0] * dist, rock[1] - u[1] * dist, z])
                if self._clear_of_fence(c) and self._wall_side_ok(c) and \
                        self._probe([self.pos, c], strict=True) == math.inf:
                    q, self._obs_dist = c, float(dist)
                    break
        self._event('reacquire', stage='landing', attempt=k, observe_dist_m=round(self._obs_dist, 2),
                    reasons=dict(self._miss_reasons), moved=q is not None)
        self._miss_run, self._miss_reasons = 0, {}
        if q is None:                             # nowhere clear to back off to: keep observing here
            return self._hold_sp(self._hold, self._face)
        du = rock[:2] - q[:2]
        self._obs_u = du / max(float(np.linalg.norm(du)), 1e-6)
        self._face = math.atan2(self._obs_u[1], self._obs_u[0])
        self._new_follower([q], p.v_backoff, avoid='lateral', yaw=self._face)   # camera stays on the pad: back off
        self._fol.vio_check = False
        self.sub = 'approach'
        return self._fstep().sp

    def _ph_disarm(self):
        v = self.veh
        if self.p.pilot_mode:                     # the pilot / PX4 disarmed us (no command, no timeout)
            if v.armed:
                return self._hold_sp(self.pos) if self._in_control() else None
            if self._t_disarmed is None:
                self._t_disarmed = self.now
                self._event('disarmed_on_target', pos=self.pos, tof_m=self._tof())
            if self.now - self._t_disarmed >= self.p.disarmed_dwell_s:
                self._set_phase('rearm', 'dwell_done')
            return None
        if v.armed:
            if v.nav_state != NAV_OFFBOARD:
                self._release('left_offboard nav_state=%d' % v.nav_state)
                return None
            if self.now - self.t_phase > self.p.phase_timeout_s.get('disarm', 6.0):
                return self._abort('disarm_refused')
            self._cmd(cmd_disarm())
            return self._press_sp()
        if self._t_disarmed is None:
            self._t_disarmed = self.now
            self._nav_at_disarm = v.nav_state
            self._event('disarmed_on_target', pos=self.pos, tof_m=self._tof())
        elif v.nav_state not in (self._nav_at_disarm, NAV_OFFBOARD):
            self._release('mode_changed_while_disarmed nav_state=%d' % v.nav_state)
            return None
        if self.now - self._t_disarmed >= self.p.disarmed_dwell_s:
            self._set_phase('rearm', 'dwell_done')
        return self._hold_sp(self.pos)

    def _ph_rearm(self):
        v = self.veh
        hold = self._hold_sp(self.pos)
        if self.p.pilot_mode:                     # wait for the pilot's re-arm + the next Offboard session
            if not v.armed:                       # disarmed on the ground is safe: wait as long as it takes
                self._rearm_armed_since = None
                return None
            if self._in_control() and v.pose_ok:
                self._t_disarmed = None
                self._set_phase('climb_over_wall', 'pilot_offboard_active')
                return hold
            if self._rearm_armed_since is None:
                self._rearm_armed_since = self.now
            elif self.now - self._rearm_armed_since > self.p.pilot_rearm_offboard_wait_s:
                self._release('armed_without_offboard_after_rearm')   # 'staying out of Offboard': pilot owns it
            return None
        if v.armed and v.nav_state == NAV_OFFBOARD:
            self._t_disarmed = None
            self._set_phase('climb_over_wall', 'rearmed')
            return hold
        if v.armed:
            self._release('armed_outside_offboard nav_state=%d' % v.nav_state)
            return None
        if v.nav_state not in (self._nav_at_disarm, NAV_OFFBOARD):
            self._release('mode_changed_while_disarmed nav_state=%d' % v.nav_state)
            return None
        if self.now - self.t_phase > self.p.phase_timeout_s.get('rearm', 10.0):
            self.abort_reason = 'rearm_timeout'
            self._event('abort', reason='rearm_timeout')
            self._set_phase('aborted', 'rearm_timeout')
            return None
        if not v.pose_ok:
            self._event_once('rearm_wait_pose', 'rearm_waiting_for_pose')
            return hold
        if v.nav_state != NAV_OFFBOARD:
            self._cmd(cmd_offboard())
        else:
            self._cmd(cmd_arm())
        return hold

    # ---- wall: liftoff -> fly over at (wall top + margin) -> return home
    def _wall_z_over(self):
        top_up = self.prior.wall_top_up
        m, t = self.per.wall_top_d, self.per.wall_t
        if m is not None and math.isfinite(float(m)) and (t is None or self.now - t <= self.p.wall_measure_max_age_s):
            top_up = max(top_up, float(self.home[2] - float(m)))
        z = self._z_up(top_up + self.p.wall_margin_m)
        return None if z < self._ceiling_z() - 1e-6 else z

    def _start_cross(self):
        g, p = self.g, self.p
        z = self._z_over
        a = np.array([g['wall_from'][0], g['wall_from'][1], z])
        b = np.array([g['wall_to'][0], g['wall_to'][1], z])
        goals = [a, b]
        if self.pos[2] > z + 0.15:                # below the over-wall height: straight up first, then cross
            goals = [np.array([self.pos[0], self.pos[1], z])] + goals
        self._new_follower(goals, p.v_cruise, avoid='up', run_out=0.5)
        self._cross_a_idx = len(goals) - 1        # path vertex of cross_from (P[0] = start)

    def _ph_climb_over_wall(self):
        p, g = self.p, self.g
        if self.sub == '':
            z = self._wall_z_over()
            if z is None:
                return self._abort('wall_above_fence')
            self._z_over = z
            lift = self.pos + np.array([0.0, 0.0, -p.liftoff_m])
            self._new_follower([lift], p.v_up, yaw=heading_of(g['wall_to'] - g['wall_from']))
            self.sub = 'liftoff'
        if self.sub == 'liftoff':
            r = self._fstep()
            if self.pos[2] > self._fol.goal[2] + 0.1:
                return r.sp
            self.sub = 'cross'
            self._start_cross()
        z = self._wall_z_over()
        if z is None:
            return self._abort('wall_above_fence')
        if z < self._z_over - 0.1 and self._fol.project(self.pos) < self._fol.cum[self._cross_a_idx]:
            self._event('wall_higher', z_over=round(z, 3), was=round(self._z_over, 3))
            self._z_over = z
            self._start_cross()
        r = self._fstep()
        if r.status == 'stuck':
            return self._abort('wall_blocked')
        if r.status in ('passed', 'arrived'):
            self._set_phase('return_home', 'over_wall')
        return r.sp

    def _ph_return_home(self):
        p = self.p
        if self._fol is None:
            z = self._clamp_alt(self._z_up(p.return_up_m))
            goals = []
            if p.return_high:
                if self._z_over is not None:
                    z = max(self._ceiling_z(), min(z, self._z_over))    # NED: smaller z = higher
                if self.pos[2] > z + 0.1:                               # below the safe height: climb first
                    goals.append(np.array([self.pos[0], self.pos[1], z]))
            goals.append(np.array([self.home[0], self.home[1], z]))
            self._new_follower(goals, p.v_cruise, avoid='lateral')
        r = self._fstep()
        if r.status == 'stuck':
            return self._abort('return_blocked')
        if r.status == 'arrived':
            self._set_phase('land_home', 'over_home')
        return r.sp

    def _ph_land_home(self):
        p = self.p
        if self.sub == '':
            self._new_follower([np.array([self.home[0], self.home[1], self.pos[2]])], p.v_align, tol=p.settle_tol_m)
            self.sub = 'over'
        if self.sub == 'over':
            r = self._fstep()
            if self._settled(self._fol.goal, xy_only=True):
                self._desc = Descent(self.home[:2], float(self.pos[2]), surface_z=float(self.home[2]),
                                     pose_only_gain=p.pilot_pos_gain if p.pilot_mode else None)
                self._t_desc = self.now
                self.sub = 'descend'
            return r.sp
        if p.pilot_mode and self.sub in ('descend', 'pilot_hold'):
            return self._pilot_landing_step(self.home[:2])
        if self.sub == 'descend':
            if self.now - self._t_desc > p.descend_timeout_s:
                return self._abort('no_touchdown_confirmation_at_home', tof_m=self._tof(), landed=bool(self.veh.landed))
            sp, touched = self._desc.step(self.dt, self.now, self.pos, self.veh.landed, self._tof(), p)
            if not touched:
                self._maybe_handoff()
                return sp
            self.sub = 'disarm'
            self._event('touchdown_home', pos=self.pos, tof_m=self._tof())
        if self.veh.armed:
            self._cmd(cmd_disarm())
            return self._press_sp()
        self._set_phase('done', 'disarmed_at_home')
        self._event('mission_complete', t_flight=round(self.now - self.t_arm, 2))
        return None
