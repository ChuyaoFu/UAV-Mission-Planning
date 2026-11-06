"""Strict adapter to OnlineMap's complete-body measured-volume query.

Certificates are generated from an immutable OnlineMap snapshot. They authorize
one short commanded step; subsequent map updates cannot retroactively authorize
an already-flown step. Unknown/outside cells remain blocked. The sole exception
is OnlineMap's explicitly declared local launch envelope.
"""
from dataclasses import dataclass, field
import numpy as np
from itertools import islice
from task_observations import vector


@dataclass(frozen=True)
class PathCertificate:
    epoch: str
    issued: float
    valid_until: float
    map_stamp: float
    points: tuple
    margin_m: float
    purpose: str
    collision_margin_m: float
    _map: object = field(repr=False,compare=False)

    def as_dict(self):
        return {'epoch':self.epoch,'issued':self.issued,'valid_until':self.valid_until,
                'map_stamp':self.map_stamp,'points':[list(p) for p in self.points],
                'margin_m':self.margin_m,'tracking_tube_m':self.margin_m,
                'extra_collision_margin_m':self.collision_margin_m,'purpose':self.purpose,
                'full_body_verified':True,'unknown_is_blocked':True}

    def covers_motion(self,a,b,now,epoch):
        if epoch != self.epoch or not self.issued <= now <= self.valid_until: return False
        a,b=vector(a),vector(b)
        # Actual feedback must stay within the issued short segment's tube.
        start,end=np.asarray(self.points[0]),np.asarray(self.points[-1]); v=end-start
        def distance(p):
            t=np.clip(np.dot(p-start,v)/max(np.dot(v,v),1e-12),0.,1.)
            return np.linalg.norm(p-(start+t*v))
        if max(distance(a),distance(b))>self.margin_m: return False
        return self._map.segment(a,b,extra_margin=self.collision_margin_m)


class MeasuredMapGuard:
    def __init__(self,online_map,epoch,body_radius_m,body_half_height_m,max_age_s=.5,tracking_margin_m=.04,extra_collision_margin_m=0.):
        if not epoch or min(body_radius_m,body_half_height_m)<=0 or not 0<max_age_s<=1. or not 0<tracking_margin_m<=.15:
            raise ValueError('Invalid map guard limits')
        if not 0<=extra_collision_margin_m<=.15: raise ValueError('Invalid extra collision margin')
        if not np.allclose(online_map.extent,[body_radius_m,body_radius_m,body_half_height_m]):
            raise ValueError('Map and executive must use the same measured body envelope')
        self.map=online_map; self.epoch=epoch; self.max_age_s=max_age_s; self.margin=tracking_margin_m; self.extra=extra_collision_margin_m

    def certify(self,points,now,epoch,purpose):
        points=tuple(tuple(vector(p)) for p in islice(points,33))
        if len(points)<2 or len(points)>32 or epoch != self.epoch: return None
        if not np.isfinite(now) or not -1e-9 <= now-self.map.stamp <= self.max_age_s: return None
        lower=self.map.origin+self.map.extent+self.extra
        upper=self.map.origin+np.asarray(self.map.shape)*self.map.res-self.map.extent-self.extra
        if any(np.any(np.asarray(p)<lower) or np.any(np.asarray(p)>=upper) for p in points): return None
        if not all(self.map.segment(a,b,extra_margin=self.extra) for a,b in zip(points,points[1:])): return None
        return PathCertificate(epoch,now,min(now+.4,self.map.stamp+self.max_age_s),self.map.stamp,
                               points,self.margin,purpose,self.extra,self.map)
