"""Explicit estimator-proxy admission and conservative geometry accounting.

No covariance/gauge subtraction. Configured map bounds are declarations requiring
external qualification; they are not inferred from fusion health or a p95.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class UncertaintyPolicy:
    admission_limit_m: float = .20
    included_xy_margin_m: float = .08
    included_z_margin_m: float = .04
    map_pose_error_m: float = .20
    contact_local_limit_m: float = .04
    map_geometry_error_m: float = 0.
    tracking_allowance_m: float = .04

    def __post_init__(self):
        values=[self.admission_limit_m,self.included_xy_margin_m,self.included_z_margin_m,
                self.map_pose_error_m,self.contact_local_limit_m,self.map_geometry_error_m,self.tracking_allowance_m]
        if not np.isfinite(values).all() or not .01<=self.admission_limit_m<=.5:
            raise ValueError('Invalid explicit uncertainty admission domain')
        if min(values[1:])<0 or self.map_pose_error_m>.5 or self.contact_local_limit_m!=.04 or self.map_geometry_error_m>.02 or self.tracking_allowance_m>.15:
            raise ValueError('Invalid geometry accounting or changed local contact precision')

    def admits(self,error):
        return math.isfinite(error) and 0<=error<=self.admission_limit_m

    def require(self,error):
        if not self.admits(error):raise ValueError('Accepted body uncertainty exceeds task budget')
        return float(error)

    def extra(self,body_error_m,reference_error_m=None,include_map_geometry=True):
        """Full two-pose sum minus ONLY the existing explicit robot padding.

        The map/reference term is never assumed independent or zero. An explicit
        zero is permitted only as a separately qualified reference declaration.
        """
        body=self.require(body_error_m)
        reference=self.map_pose_error_m if reference_error_m is None else float(reference_error_m)
        if not math.isfinite(reference) or not 0<=reference<=.5:raise ValueError('Missing/invalid reference uncertainty')
        geometry=self.map_geometry_error_m if include_map_geometry else 0.
        return tuple(np.maximum(0.,body+reference+geometry+self.tracking_allowance_m-np.array([self.included_xy_margin_m]*2+[self.included_z_margin_m])).tolist())

    def evidence(self,body_error_m,reference_error_m=None,include_map_geometry=True):
        reference=self.map_pose_error_m if reference_error_m is None else reference_error_m
        return {'body_three_sigma_proxy_m':float(body_error_m),'reference_pose_error_m':float(reference),
                'included_margin_xyz_m':[self.included_xy_margin_m]*2+[self.included_z_margin_m],
                'extra_collision_margin_xyz_m':list(self.extra(body_error_m,reference,include_map_geometry)),
                'map_geometry_error_m':self.map_geometry_error_m if include_map_geometry else 0.,
                'tracking_allowance_m':self.tracking_allowance_m,
                'admission_limit_m':self.admission_limit_m,'gauge_covariance_subtracted':False,
                'relative_crosscov_available':False,'absolute_accuracy_qualified':False,
                'local_contact_limit_m':self.contact_local_limit_m}
