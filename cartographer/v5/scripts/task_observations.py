"""Exposure-time local observations; proposals never certify free space.

All positions use the caller's measured, reset-scoped local ENU frame. The
camera transform maps LEFT OPTICAL coordinates to that frame. No global pose,
course coordinates, dimensions, CAD, truth, or inferred depth are accepted.
"""
from dataclasses import dataclass, field
import math
from itertools import islice
import numpy as np


def plain(value):
    """Detach NumPy scalars/arrays into JSON-safe evidence values."""
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value


def vector(value, size=3):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError('Expected a finite vector')
    return a


def unit(value):
    a = vector(value)
    if np.linalg.norm(a) < 1e-6:
        raise ValueError('Zero normal')
    return a / np.linalg.norm(a)


@dataclass(frozen=True)
class Exposure:
    stamp: float
    image_stamp: float
    depth_stamp: float
    pose_stamp: float
    local_from_camera: np.ndarray
    epoch: str
    pose_error_m: float = .03
    tracking_valid: bool = True
    calibration_id: str = ''
    uncertainty_limit_m: float = .20
    position_error_xyz_m: object = None

    def validate(self, now, max_age=.6, max_skew=.025):
        stamps = np.array([self.stamp, self.image_stamp, self.depth_stamp, self.pose_stamp, now])
        T = np.asarray(self.local_from_camera, dtype=float)
        if not self.tracking_valid or not self.epoch or not self.calibration_id:
            raise ValueError('Tracking epoch and calibration required')
        if not np.isfinite(stamps).all() or not 0 <= now-self.stamp <= max_age:
            raise ValueError('Stale or future exposure')
        if np.max(np.abs(stamps[:4]-self.stamp)) > max_skew:
            raise ValueError('Image/depth/pose exposure skew')
        if T.shape != (4,4) or not np.isfinite(T).all() or not np.allclose(T[3], [0,0,0,1]):
            raise ValueError('Invalid rigid transform')
        R = T[:3,:3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or abs(np.linalg.det(R)-1) > 1e-5:
            raise ValueError('Transform is not a proper rotation')
        if not math.isfinite(self.uncertainty_limit_m) or not .01<=self.uncertainty_limit_m<=.5 or not math.isfinite(self.pose_error_m) or not 0 <= self.pose_error_m <= self.uncertainty_limit_m:
            raise ValueError('Pose uncertainty exceeds association budget')
        if self.position_error_xyz_m is not None:
            axes=vector(self.position_error_xyz_m)
            if (axes<0).any() or (axes>self.pose_error_m+1e-12).any():raise ValueError('Invalid per-axis position uncertainty')
        return T


def opening_interior(candidate):
    """Painted means consistent same-plane samples, never a semantic certainty."""
    evidence = candidate.get('center_evidence', {})
    if evidence.get('same_surface_pixels', 0) >= 3 and evidence.get('blocking_pixels', 0) == evidence.get('same_surface_pixels', 0):
        return 'painted'
    if evidence.get('blocking_pixels', 0) >= 3:
        return 'blocked'
    if evidence.get('background_observed', False):
        return 'free-background'
    return 'unknown'


@dataclass
class TargetTrack:
    id: str
    role: str
    kind: str
    epoch: str
    center: np.ndarray
    normal: np.ndarray
    dimensions: np.ndarray
    error_m: float
    stamp: float
    first_stamp: float
    camera_origins: list = field(default_factory=list)
    observation_count: int = 1
    consistent: bool = False
    interior: str = 'unknown'
    geometry_verified: bool = False
    passage_clearance_verified: bool = False
    boundary: object = None
    history: list = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def as_dict(self):
        return plain({**{k:v for k,v in vars(self).items() if k not in ('center','normal','dimensions','camera_origins','boundary')},
                'center': self.center.tolist(), 'normal': self.normal.tolist(),
                'dimensions': self.dimensions.tolist(),
                'camera_origins': [p.tolist() for p in self.camera_origins],
                'boundary': None if self.boundary is None else self.boundary.tolist()})


class TargetTracker:
    """Bounded nearest-plane association with independent measured viewpoints.

    Three frames AND a camera-translation baseline are necessary for consistent.
    Fresh contradictory interior observations immediately veto use. No averaging
    makes unknown pixels free; only the newest frame's evidence is used.
    """
    def __init__(self, max_tracks=24, ttl_s=2., min_baseline_m=.08):
        self.max_tracks, self.ttl_s, self.min_baseline_m = max_tracks, ttl_s, min_baseline_m
        self.tracks = {}; self.next_id = 1; self.epoch = None

    def update(self, candidates, exposure, role, now):
        T = exposure.validate(now)
        if self.epoch != exposure.epoch:
            self.tracks.clear(); self.epoch = exposure.epoch
        self.tracks = {key:t for key,t in self.tracks.items() if now-t.stamp <= self.ttl_s}
        used = set(); result = []
        for c in islice(candidates,12):
            center_c, normal_c = c.get('center_camera_m'), c.get('normal_camera')
            if center_c is None or normal_c is None or c.get('rim_measured_support', 0) < 24:
                continue
            center = T[:3,:3] @ vector(center_c) + T[:3,3]
            normal = unit(T[:3,:3] @ unit(normal_c))
            dims = vector(c['clearance_lower_m'], 2)
            if np.any(dims <= 0):
                continue
            pose_radius=exposure.pose_error_m if exposure.position_error_xyz_m is None else float(np.linalg.norm(vector(exposure.position_error_xyz_m)))
            error = pose_radius + max(.015, float(c.get('geometry_margin_m', .03)))
            matches = []
            for t in self.tracks.values():
                if t.id in used or t.role != role or t.kind != c['kind'] or exposure.stamp <= t.stamp:
                    continue
                separation = np.linalg.norm(center-t.center)
                # Association tolerance is a measurement policy, not field size.
                if separation < .12 + error + t.error_m and abs(normal @ t.normal) > math.cos(math.radians(12)) and np.max(np.abs(dims-t.dimensions)) < .12 + error+t.error_m:
                    matches.append((separation, t))
            interior = opening_interior(c)
            boundary_c = c.get('rim_boundary_camera_m', c.get('rim_corners_camera_m'))
            boundary = None if boundary_c is None else np.asarray(boundary_c) @ T[:3,:3].T + T[:3,3]
            if matches:
                t = min(matches, key=lambda p:p[0])[1]
                if normal @ t.normal < 0: normal = -normal
                t.center = .5*(t.center+center); t.normal = unit(t.normal+normal)
                t.dimensions = np.minimum(t.dimensions, dims); t.error_m = max(t.error_m,error)
                t.stamp = exposure.stamp; t.observation_count += 1
                t.camera_origins.append(T[:3,3].copy()); t.camera_origins = t.camera_origins[-16:]
                t.boundary = boundary; t.interior = interior
                baseline = max(np.linalg.norm(p-q) for p in t.camera_origins for q in t.camera_origins)
                t.consistent = t.observation_count >= 3 and baseline >= self.min_baseline_m
            else:
                # Reject duplicate or out-of-order detections rather than create
                # a second identity for the same target/exposure.
                if any(t.role == role and np.linalg.norm(t.center-center) < .12+error+t.error_m and exposure.stamp <= t.stamp for t in self.tracks.values()):
                    continue
                if len(self.tracks) >= self.max_tracks:
                    oldest = min(self.tracks, key=lambda k:self.tracks[k].stamp); del self.tracks[oldest]
                key = f'{exposure.epoch}:{self.next_id}'; self.next_id += 1
                t = TargetTrack(key,role,c['kind'],exposure.epoch,center,normal,dims,error,
                                exposure.stamp,exposure.stamp,[T[:3,3].copy()],interior=interior,boundary=boundary)
                self.tracks[key] = t
            t.history.append({'stamp': exposure.stamp, 'interior': interior, 'calibration_id': exposure.calibration_id,
                              'pose_error_m':exposure.pose_error_m,'uncertainty_limit_m':exposure.uncertainty_limit_m})
            t.history = t.history[-16:]
            t.details = c.get('details', {})
            t.geometry_verified = t.consistent
            t.passage_clearance_verified = False
            used.add(t.id); result.append(t)
        return result


def passage_views(track, body_radius_m, body_half_height_m, observation_standoff_m=.3):
    """Front/back body-center proposals. Both and every segment need map checks.

    body support along the measured plane normal accounts for wall tilt. The
    normal points toward the original observing camera. No course length enters.
    """
    n = unit(track.normal)
    support = body_radius_m*np.linalg.norm(n[:2]) + body_half_height_m*abs(n[2])
    offset = support + track.error_m + observation_standoff_m
    return {'before': (track.center+n*offset).tolist(),
            'after': (track.center-n*offset).tolist(), 'body_support_m': float(support),
            'offset_m': float(offset), 'verified': False, 'target_id':track.id}


@dataclass(frozen=True)
class ForwardEvidence:
    stamp: float
    depth_stamp: float
    pose_stamp: float
    epoch: str
    calibration_id: str
    camera_origin_local_m: tuple
    points_local_m: tuple
    quadrants: tuple
    source: str = 'direct measured stereo only'

    def as_dict(self):
        return {'stamp':self.stamp,'depth_stamp':self.depth_stamp,'pose_stamp':self.pose_stamp,
                'epoch':self.epoch,'calibration_id':self.calibration_id,
                'camera_origin_local_m':self.camera_origin_local_m,
                'direct_measured_points':len(self.points_local_m),'quadrants':self.quadrants,'source':self.source}


def measured_forward_evidence(measured_depth,K,exposure,now,measured_mask=None):
    """Fresh depth evidence after a crossing; never a free-space declaration.

    It does not require a rear camera or reacquisition of the crossed rim. The
    executive independently checks these measurements lie beyond that plane,
    a new map exists, and the complete prior VIO/path-certificate chain is valid.
    """
    T=exposure.validate(now);depth=np.asarray(measured_depth);K=vector(K,4)
    if depth.ndim!=2 or min(depth.shape)<32 or max(depth.shape)>1920 or min(K[:2])<=0:
        raise ValueError('Invalid measured depth grid/calibration')
    direct=np.isfinite(depth)&(depth>=.25)&(depth<=8.)
    if measured_mask is not None:
        if np.asarray(measured_mask).shape!=depth.shape:raise ValueError('Provenance mask mismatch')
        direct&=np.asarray(measured_mask,dtype=bool)
    yy,xx=np.nonzero(direct)
    if len(xx)<48:return None
    ids=np.linspace(0,len(xx)-1,min(len(xx),256)).astype(int);x,y=xx[ids],yy[ids]
    q=(x>=depth.shape[1]/2).astype(int)+2*(y>=depth.shape[0]/2).astype(int)
    if len(np.unique(q))<3:return None
    z=depth[y,x];points=np.column_stack(((x-K[2])/K[0]*z,(y-K[3])/K[1]*z,z))
    points=points@T[:3,:3].T+T[:3,3]
    return ForwardEvidence(exposure.stamp,exposure.depth_stamp,exposure.pose_stamp,exposure.epoch,
                           exposure.calibration_id,tuple(T[:3,3]),tuple(map(tuple,points)),tuple(map(int,q)))
