"""Reobserve own start surface from actual mechanical BODY anchor, no fake legacy cal."""
import math
import numpy as np
from task_perception import _fit_plane,OpeningLimits


def measured_home_candidates(depth,K,exposure,foot,depth_error):
    T=exposure.local_from_camera;depth=np.asarray(depth);K=np.asarray(K)
    valid=np.isfinite(depth)&(depth>=.25)&(depth<=8.);y,x=np.nonzero(valid)
    if len(x)<64:return []
    ids=np.linspace(0,len(x)-1,min(len(x),4096)).astype(int);x,y=x[ids],y[ids];z=depth[y,x]
    rays=np.c_[(x-K[2])/K[0],(y-K[3])/K[1],np.ones(len(x))]
    points=(rays*z[:,None])@T[:3,:3].T+T[:3,3]
    anchor=foot.home_body_anchor;home=np.asarray(anchor['position_local_m']);q=np.asarray(anchor['quaternion_xyzw'])
    from scipy.spatial.transform import Rotation
    contact_hint=home+Rotation.from_quat(q).apply([0,0,-foot.offset_m])
    enclosure=anchor['position_three_sigma_proxy_m']+exposure.pose_error_m+foot.error_m+depth_error
    near=(np.linalg.norm(points[:,:2]-home[:2],axis=1)<=1.2)&(abs(points[:,2]-contact_hint[2])<=enclosure)
    points=points[near]
    if len(points)<64:return []
    fit=_fit_plane(points[:400],OpeningLimits(min_rim_fraction=.8))
    if fit is None:return []
    n,d,inside,residual=fit
    if n[2]<0:n=-n;d=-d
    if n[2]<math.cos(math.radians(5)):return []
    center=home.copy();center[2]=-(d+n[:2]@home[:2])/n[2]
    if abs(center[2]-contact_hint[2])>enclosure:return []
    observed=points[:400][inside]
    if len(set(map(tuple,(observed[:,:2]>center[:2]).astype(int))))<4:return []
    return [dict(kind='landing_plane',frame='left_optical',center_camera_m=(T[:3,:3].T@(center-T[:3,3])).tolist(),
        normal_camera=(T[:3,:3].T@n).tolist(),clearance_lower_m=np.maximum(.01,np.ptp(observed[:,:2],axis=0)).tolist(),
        rim_measured_support=len(observed),geometry_margin_m=depth_error,
        details=dict(marker_kind='own_launch_plane',support_points_local_m=observed.tolist(),
            measured_home_body_anchor=anchor,mechanical_geometry_digest=foot.geometry_digest,
            ground_height_is_measured_not_field_prior=True),passage_clearance_verified=False)]
