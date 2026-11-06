"""Default-off SITL typed mechanical native contact, using measured geometry.

No course CAD/waypoints. Original covariance is retained, never conditioned or
subtracted. Historical support uses the sum of actual endpoint uncertainty
ellipsoids when no cross-covariance is available. Plane-model interpolation is
explicit; no map array, unknown voxel or raw depth image is filled.
"""
from dataclasses import dataclass,field
from collections import deque
import hashlib,math
import cv2
import numpy as np
from scipy.spatial import ConvexHull
from task_observations import vector,unit,plain
from task_contact import ContactLandingAdapter


def skew(v):
    x,y,z=v;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def sigma(P,j):return 3*math.sqrt(max(0.,float(j@P@j)))


@dataclass(frozen=True)
class PoseEnvelope:
    stamp_ns: int
    position: object
    rotation: object
    marginals: tuple
    interpolation_position_m: float=0.
    interpolation_angle_rad: float=0.

    def point_error(self,axis,point_local=None):
        axis=unit(axis);lever=np.zeros(3) if point_local is None else self.rotation.T@(vector(point_local)-self.position)
        j=np.r_[axis,-axis@self.rotation@skew(lever)]
        theta=max(3*math.sqrt(max(0.,np.linalg.eigvalsh(P[3:,3:]).max())) for P in self.marginals)+self.interpolation_angle_rad
        # Actual [p_WORLD, right BODY angle] covariance, including its cross terms.
        linear=max(sigma(P,j) for P in self.marginals)
        remainder=.5*np.linalg.norm(lever)*theta*theta
        return float(linear+remainder+self.interpolation_position_m+np.linalg.norm(lever)*self.interpolation_angle_rad)

    def normal_error(self,normal_local):
        b=self.rotation.T@unit(normal_local);J=skew(b)
        s=max(3*math.sqrt(max(0.,np.linalg.eigvalsh(J@P[3:,3:]@J.T).max())) for P in self.marginals)
        return float(2*math.asin(min(1.,s/2))+self.interpolation_angle_rad)

    def as_dict(self):
        return dict(stamp_ns=self.stamp_ns,position=np.asarray(self.position).tolist(),
            original_pose_covariances=[p.tolist() for p in self.marginals],
            interpolation_position_allowance_m=self.interpolation_position_m,
            interpolation_angle_allowance_rad=self.interpolation_angle_rad,
            covariance_subtracted=False,crosscov_available=False,temporal_crosscov_available=False,within_pose_cross_terms_used=True,chart='position WORLD, right BODY angle',
            accuracy_scope='actual posterior3sigma proxy with declared interpolation motion model; not hard hardware qualification')


def pose_envelope(history,stamp_ns):
    state=history.sample(stamp_ns)
    if state is None or history.fault:raise ValueError('Actual bracketed accepted BODY history required')
    if state.pose_covariance is not None:
        return PoseEnvelope(stamp_ns,state.position.copy(),state.rotation.as_matrix(),(np.asarray(state.pose_covariance).copy(),))
    items=list(history.items);left=[s for s in items if s.stamp_ns<stamp_ns];right=[s for s in items if s.stamp_ns>stamp_ns]
    if not left or not right:raise ValueError('Missing actual covariance brackets')
    a,b=left[-1],right[0];dt=(b.stamp_ns-a.stamp_ns)*1e-9
    if dt>.1 or a.pose_covariance is None or b.pose_covariance is None:raise ValueError('Covariance/interpolation continuity lost')
    # Same explicit2m/s2,4rad/s2 motion-model family as home-relative twist adapter.
    return PoseEnvelope(stamp_ns,state.position.copy(),state.rotation.as_matrix(),
        (np.asarray(a.pose_covariance).copy(),np.asarray(b.pose_covariance).copy()),2*dt*dt/8,4*dt*dt/8)


@dataclass(frozen=True)
class MechanicalStartup:
    offset_m: float
    error_m: float
    epoch: str
    calibration_id: str
    geometry_digest: str
    home_body_anchor: dict
    sample_count: int
    first_stamp: float
    last_stamp: float
    source: str='ROBOT mechanical dimensions plus actual grounded native/BODY/ToF dwell'

    @classmethod
    def from_observer(cls,report,epoch,calibration_id,now):
        if not report or not report['observed_dwell_ready'] or not report['known_home_body_anchor']:raise ValueError('Actual mechanical startup dwell/home BODY anchor required')
        anchor=report['home_body_anchor']
        if anchor['sensor_epoch']!=epoch or anchor['calibration_id']!=calibration_id:raise ValueError('Startup anchor epoch/calibration mismatch')
        if report['sample_count']<20 or report['span_s']<1. or not 0<report['error_m']<=.025:raise ValueError('Mechanical startup bound/dwell invalid')
        last=now-report['latest_age_s'];return cls(report['offset_m'],report['error_m'],epoch,calibration_id,
            report['geometry_digest'],anchor,report['sample_count'],last-report['span_s'],last)

    def as_dict(self):return plain(dict(vars(self)))


def bounded_2d_operator(B,errors):
    """Exact2D zonotope vertex norm for a bounded error box; no iid/sqrtN."""
    g=np.asarray(B).T*np.asarray(errors)[:,None]
    angles=np.unique(np.mod(np.arctan2(g[:,1],g[:,0])+np.pi/2,np.pi))
    mids=(angles+np.r_[angles[1:],angles[0]+np.pi])/2
    directions=np.c_[np.cos(mids),np.sin(mids)]
    vertices=np.sign(directions@g.T)@g
    return float(np.linalg.norm(vertices,axis=1).max())


def convex_xy(points):
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or not np.isfinite(p).all():raise ValueError('Measured planar polygon required')
    return p[ConvexHull(p).vertices]


def polygon_margin(poly,xy):
    h=ConvexHull(np.asarray(poly,float));return float(np.min(-(h.equations[:,:2]@np.asarray(xy)+h.equations[:,2])))


@dataclass(frozen=True)
class RetainedSupport:
    target_id: str
    role: str
    marker_kind: str
    stamp: float
    epoch: str
    calibration_id: str
    center_local_m: tuple
    normal_local: tuple
    landing_boundary_xy: object
    support_boundary_xy: object
    plane_origin_xy: object
    coefficients: object
    pseudoinverse: object
    point_vertical_errors: object
    source_pose: PoseEnvelope
    source_hash: str
    direct_points: int
    unknown_fraction: float
    normal_fit_error_rad: float
    measured_residual_m: float
    boundary_error_m: float
    source: str='current raw direct stereo with explicit static planar-support model'

    @property
    def error_m(self):return max(self.height_error(p) for p in self.support_boundary_xy)
    @property
    def half_width_m(self):return polygon_margin(self.landing_boundary_xy,np.asarray(self.center_local_m)[:2])
    def height(self,x,y):return float(np.r_[np.array([x,y])-self.plane_origin_xy,1.]@self.coefficients)
    def height_error(self,xy):return float(np.abs(np.r_[np.asarray(xy)-self.plane_origin_xy,1.]@self.pseudoinverse)@self.point_vertical_errors)
    def contains(self,xlo,xhi,ylo,yhi):
        eq=ConvexHull(self.support_boundary_xy).equations
        corners=np.array([[x,y] for x in (xlo,xhi) for y in (ylo,yhi)])
        return bool(np.all(corners@eq[:,:2].T+eq[:,2]<=1e-9))
    def as_dict(self):
        return dict(target_id=self.target_id,role=self.role,marker_kind=self.marker_kind,stamp=self.stamp,
            epoch=self.epoch,calibration_id=self.calibration_id,center_local_m=self.center_local_m,normal_local=self.normal_local,
            landing_boundary_xy=np.asarray(self.landing_boundary_xy).tolist(),support_boundary_xy=np.asarray(self.support_boundary_xy).tolist(),
            direct_points=self.direct_points,unknown_fraction=self.unknown_fraction,source_hash=self.source_hash,
            plane_height_error_m=self.error_m,boundary_error_m=self.boundary_error_m,normal_fit_error_rad=self.normal_fit_error_rad,
            source_pose=self.source_pose.as_dict(),source=self.source,
            static_support_assumed=True,plane_model_interpolates_unmeasured_surface=True,
            unmeasured_voxels_written_free=False,raw_depth_inpainted=False,sensor_accuracy_qualified=False)


def image_closed_boundary(image,seed_uv,K,T,n,c,minimum_radius):
    """Observed enclosing convex outline, never a preset pad radius/coordinate."""
    image=np.asarray(image);candidates=[];h,w=image.shape
    for mode in (cv2.THRESH_BINARY,cv2.THRESH_BINARY_INV):
        _,binary=cv2.threshold(image,0,255,mode|cv2.THRESH_OTSU)
        contours,_=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:100]:
            if len(contour)<5 or cv2.contourArea(contour)<.001*image.size:continue
            x,y,ww,hh=cv2.boundingRect(contour)
            if x<3 or y<3 or x+ww>w-3 or y+hh>h-3 or cv2.pointPolygonTest(contour,tuple(map(float,seed_uv)),False)<0:continue
            ellipse=cv2.fitEllipse(contour);center,axes,angle=ellipse
            if min(axes)<12:continue
            # Inset a fitted convex ellipse until EVERY boundary point lies inside
            # the actual observed contour with >=2pixel raster clearance.
            a=np.linspace(0,2*np.pi,48,endpoint=False);R=cv2.getRotationMatrix2D((0,0),-angle,1)[:,:2]
            for shrink in (.98,.94,.88,.8):
                uv=np.column_stack((np.cos(a)*axes[0]/2,np.sin(a)*axes[1]/2))@R.T*shrink+center
                if min(cv2.pointPolygonTest(contour,tuple(map(float,p)),True) for p in uv)<2:continue
                rays=np.column_stack(((uv[:,0]-K[2])/K[0],(uv[:,1]-K[3])/K[1],np.ones(len(uv))))@T[:3,:3].T
                denom=rays@n;distance=((c-T[:3,3])@n)/denom
                if (distance<=0).any():continue
                boundary=T[:3,3]+rays*distance[:,None];xy=convex_xy(boundary[:,:2])
                if polygon_margin(xy,c[:2])>=minimum_radius:candidates.append((abs(cv2.contourArea(contour)),xy,uv))
                break
    return None if not candidates else max(candidates,key=lambda x:x[0])[1:]


def build_retained_support(image,depth,K,exposure,track,body_from_camera,source_pose,depth_error,rig_point_error,
                           body_radius,home_anchor=None):
    """Fit a bounded raw plane and observed landing boundary, independently of P.

    A continuous rigid static supporting-plane model is explicit. Sparse raw
    points constrain its complete error via |A+|e, not residual/sqrt(N). Every
    finite contradictory foreground hit vetoes. Unknown remains unknown in map.
    """
    T=exposure.local_from_camera;K=vector(K,4);depth=np.asarray(depth)
    if track.role not in ('dock','home') or not track.consistent or not track.geometry_verified: return None
    if not (0<depth_error<=.02 and 0<=rig_point_error<=.02):raise ValueError('Explicit original depth and complete rig-point bounds required')
    c=vector(track.center);n=unit(track.normal)
    if n[2]<0:n=-n
    if n[2]<math.cos(math.radians(12)):return None
    camera=T[:3,:3].T@(c-T[:3,3])
    if camera[2]<=.25:return None
    seed=np.array([camera[0]/camera[2]*K[0]+K[2],camera[1]/camera[2]*K[1]+K[3]])
    if track.role=='dock':
        outline=image_closed_boundary(image,seed,K,T,n,c,body_radius)
        if outline is None:return None
        landing,uv=outline
    else:
        # Home boundary is its measured support region, centered at actual own
        # BODY anchor; no fixed field coordinates or assumed ground height.
        landing=None;uv=None
    yy,xx=np.indices(depth.shape);finite=np.isfinite(depth)&(depth>=.25)&(depth<=8.)
    if uv is not None:
        region=np.zeros(depth.shape,np.uint8);cv2.fillConvexPoly(region,np.rint(uv).astype(np.int32),1);region=region!=0
    else:
        region=finite.copy()
    y,x=np.nonzero(region&finite)
    if len(x)<64:return None
    # Preserve spatially distributed direct points, bounded memory/CPU.
    ids=np.linspace(0,len(x)-1,min(512,len(x))).astype(int);x=x[ids];y=y[ids];z=depth[y,x]
    rays=np.column_stack(((x-K[2])/K[0],(y-K[3])/K[1],np.ones(len(x))))
    points=(rays*z[:,None])@T[:3,:3].T+T[:3,3]
    if track.role=='home':
        home=np.asarray(home_anchor['position_local_m']);hint=home.copy();hint[2]-=home_anchor['mechanical_offset_m']
        tolerance=exposure.pose_error_m+home_anchor['position_three_sigma_proxy_m']+depth_error+home_anchor['mechanical_error_m']
        keep=(np.linalg.norm(points[:,:2]-home[:2],axis=1)<max(1.,3*body_radius))&(abs(points[:,2]-hint[2])<=tolerance)
        points=points[keep];rays=rays[keep]
        if len(points)<64:return None
    origin=c[:2].copy();A=np.column_stack((points[:,:2]-origin,np.ones(len(points))));pinv=np.linalg.pinv(A)
    if np.linalg.matrix_rank(A)<3:return None
    coeff=pinv@points[:,2];normal=np.r_[-coeff[:2],1.];normal/=np.linalg.norm(normal)
    # A depth interval is optical-z, so convert each actual ray to3D. A complete
    # rig-point error includes mounting rotation lever at this admitted range.
    point_errors=np.linalg.norm(rays,axis=1)*depth_error+rig_point_error
    k=bounded_2d_operator(pinv[:2],point_errors)
    if k>=1:return None
    normal_error=math.asin(k)
    gradient=float(np.linalg.norm(coeff[:2]));gmax=(gradient+k*math.sqrt(gradient*gradient+1-k*k))/(1-k*k)
    errors=point_errors*math.sqrt(1+gmax*gmax)
    residual=abs(A@coeff-points[:,2])
    if (residual>errors).any():return None
    if math.acos(np.clip(normal[2],-1,1))+normal_error>math.radians(5):return None
    support=convex_xy(points[:,:2])
    if landing is None:landing=support;c=np.asarray(home_anchor['position_local_m']).copy()
    c=c.copy();c[2]=float(np.r_[c[:2]-origin,1.]@coeff)
    if polygon_margin(support,c[:2])<body_radius:return None
    # Check ALL finite raw returns in bounded chunks. No decimation may hide a
    # foreground contradiction. Missing returns remain an explicit plane-model
    # assumption and do not alter any occupancy voxel.
    fy,fx=np.nonzero(region&finite);hull=ConvexHull(landing)
    for offset in range(0,len(fx),120000):
        bx=fx[offset:offset+120000];by=fy[offset:offset+120000]
        rr=np.column_stack(((bx-K[2])/K[0],(by-K[3])/K[1],np.ones(len(bx))))
        pp=(rr*depth[by,bx,None])@T[:3,:3].T+T[:3,3]
        if track.role=='dock':inside=np.ones(len(pp),bool)
        else:
            rw=rr@T[:3,:3].T;denom=rw[:,2]-rw[:,:2]@coeff[:2]
            distance=(coeff[2]+(T[:3,3][:2]-origin)@coeff[:2]-T[2,3])/denom
            expected=T[:3,3]+rw*distance[:,None]
            inside=(distance>0)&np.all(expected[:,:2]@hull.equations[:,:2].T+hull.equations[:,2]<=1e-8,axis=1)
        allerr=(np.linalg.norm(rr,axis=1)*depth_error+rig_point_error)*math.sqrt(1+gmax*gmax)
        difference=abs(pp[:,2]-np.c_[pp[:,:2]-origin,np.ones(len(pp))]@coeff)
        if np.any(inside&(difference>allerr+float(np.max(residual)))):return None
    boundary_error=float(np.max(errors))
    if uv is not None:
        measured=[];edge_errors=[];edge_ids=[]
        for i,pixel in enumerate(uv):
            q=np.rint(pixel).astype(int);x0,x1=max(0,q[0]-2),min(depth.shape[1],q[0]+3);y0,y1=max(0,q[1]-2),min(depth.shape[0],q[1]+3)
            ey,ex=np.nonzero(finite[y0:y1,x0:x1])
            if not len(ex):continue
            ex=ex+x0;ey=ey+y0;j=int(np.argmin((ex-pixel[0])**2+(ey-pixel[1])**2));px,py=int(ex[j]),int(ey[j]);zz=depth[py,px]
            ray=np.array([(px-K[2])/K[0],(py-K[3])/K[1],1.]);point=T[:3,3]+T[:3,:3]@(ray*zz)
            error=np.linalg.norm(ray)*depth_error+rig_point_error+np.linalg.norm([px-pixel[0],py-pixel[1]])*zz/min(K[:2])
            if abs(point[2]-np.r_[point[:2]-origin,1.]@coeff)>error/math.cos(math.radians(5)):continue
            measured.append(point[:2]);edge_errors.append(error);edge_ids.append(i)
        if len(measured)<12:return None
        circular=np.r_[sorted(edge_ids),min(edge_ids)+len(uv)]
        if np.diff(circular).max()>len(uv)/4:return None
        landing=convex_xy(measured);boundary_error=float(max(edge_errors))
        if polygon_margin(landing,c[:2])<body_radius:return None
    rawhash=hashlib.sha256(np.asarray(depth,dtype='<f4').tobytes()+np.asarray(T,dtype='<f8').tobytes()).hexdigest()
    return RetainedSupport(track.id,track.role,track.details.get('marker_kind','own_launch_plane'),exposure.stamp,
        exposure.epoch,exposure.calibration_id,tuple(c),tuple(normal),landing,support,origin,coeff,pinv,errors,
        source_pose,rawhash,len(points),float(1-finite[region].mean()),normal_error,float(residual.max()),boundary_error)


@dataclass(frozen=True)
class NativeContactCertificate:
    epoch: str
    issued: float
    valid_until: float
    map_stamp: float
    mode: str
    points: tuple
    support: RetainedSupport
    calibration: MechanicalStartup
    lateral_uncertainty_m: float
    vertical_error_m: float
    grounded: bool
    observation: dict
    uncertainty: dict
    _adapter: object=field(repr=False,compare=False)
    _map: object=field(repr=False,compare=False)
    _extra: object=field(repr=False,compare=False)

    def as_dict(self):
        return dict(kind='typed_mechanical_native_'+self.mode,epoch=self.epoch,issued=self.issued,
            valid_until=self.valid_until,map_stamp=self.map_stamp,mode=self.mode,points=self.points,
            support=self.support.as_dict(),mechanical_calibration=self.calibration.as_dict(),
            lateral_uncertainty_m=self.lateral_uncertainty_m,vertical_error_m=self.vertical_error_m,
            native_grounded=self.grounded,observation=self.observation,uncertainty=self.uncertainty,
            map_changed=False,unknown_above_support_allowed=False,old_global4cm_gate_applied=False,
            native_LAND_continuous_monitor_required=self.mode=='descent')

    def covers_uncertainty(self,error_m):
        # Keep the full accepted proxy, not a substituted local foot error.
        return bool(not self.uncertainty or error_m<=self.uncertainty['body_three_sigma_proxy_m']+1e-12)

    def covers_motion(self,a,b,now,epoch):
        if epoch!=self.epoch or not self.issued-1e-9<=now<=self.valid_until:return False
        a,b=vector(a),vector(b);start,goal=np.array(self.points[0]),np.array(self.points[-1])
        for p in (a,b):
            if polygon_margin(self.support.landing_boundary_xy,p[:2])<self._map.extent[0]+self.lateral_uncertainty_m:return False
            if p[2]>max(start[2],goal[2])+self.vertical_error_m or p[2]<min(start[2],goal[2])-self.vertical_error_m:return False
        if self.mode=='descent' and b[2]>a[2]+self.vertical_error_m:return False
        if self.mode=='liftoff' and b[2]<a[2]-self.vertical_error_m:return False
        return self._adapter._volume(a,b,self.support,self.calibration,self._map,self._extra) is not None


@dataclass(frozen=True)
class NativeContactContext:
    adapter: object
    support: RetainedSupport
    calibration: MechanicalStartup
    range_pose: object
    body_history: object
    native: object


class NativeMechanicalContact:
    """Complete typed certifier. Static plane history survives camera occlusion."""
    def __init__(self,mount,axis,calibration_id,max_support_age_s=120.,max_descent_speed_mps=.25):
        self.mount=vector(mount);self.axis=unit(axis);self.calibration_id=calibration_id
        if not 1<=max_support_age_s<=180 or not .05<=max_descent_speed_mps<=.4:raise ValueError('Bounded static support/native descent domain required')
        self.max_support_age=max_support_age_s;self.max_descent_speed=max_descent_speed_mps
        self._volume_adapter=ContactLandingAdapter(mount,axis,calibration_id)
        self.last_observation=None;self.last_rejection=None

    def _volume(self,a,b,support,cal,m,extra):
        # Conservative segment AABB, vectorized. Same full inflated upper/side
        # envelope, nominal supporting-plane lower clip, unknown veto and exact
        # occupied-support intersection; no map or raw data is modified.
        a,b=vector(a),vector(b);extra=np.broadcast_to(np.asarray(extra,float),(3,))
        if not 0<m.res<=.2:return None
        radius=m.extent[:2]+extra[:2];xylo=np.minimum(a[:2],b[:2])-radius;xyhi=np.maximum(a[:2],b[:2])+radius
        corners=np.array([[x,y] for x in (xylo[0],xyhi[0]) for y in (xylo[1],xyhi[1])])
        lower_z=max(min(a[2],b[2])-cal.offset_m-cal.error_m-extra[2],min(support.height(*q) for q in corners)+1e-9)
        lo=m.idx(np.r_[xylo,lower_z]);hi=m.idx(np.r_[xyhi,max(a[2],b[2])+m.extent[2]+extra[2]])
        if not m.inside(lo) or not m.inside(hi) or np.prod(hi-lo+1)>32768:return None
        celllo=m.origin[:2]+lo[:2]*m.res;cellhi=m.origin[:2]+(hi[:2]+1)*m.res
        if not support.contains(*[celllo[0],cellhi[0],celllo[1],cellhi[1]]):return None
        indices=np.indices(tuple(hi-lo+1)).reshape(3,-1).T+lo
        centers=m.origin+indices*m.res;tops=centers[:,2]+m.res
        eq=np.asarray(support.coefficients);origin=np.asarray(support.plane_origin_xy)
        xy=centers[:,:2]
        cxy=np.stack((xy,xy+[m.res,0],xy+[0,m.res],xy+m.res),axis=1)
        design=np.concatenate((cxy-origin,np.ones((*cxy.shape[:2],1))),axis=2)
        heights=design@eq
        relevant=tops>heights.min(axis=1)+1e-9
        indices=indices[relevant];centers=centers[relevant];tops=tops[relevant];heights=heights[relevant];design=design[relevant]
        states=m.state[tuple(indices.T)]
        if np.any(~np.isin(states,[0,-2,1])):return None
        occupied=states==1
        if not occupied.any():return 0
        heights=heights[occupied];design=design[occupied];centers=centers[occupied];tops=tops[occupied]
        errors=(np.abs(design@support.pseudoinverse)@support.point_vertical_errors).max(axis=1)
        if np.any(centers[:,2]>heights.max(axis=1)+errors) or np.any(tops<heights.min(axis=1)-errors):return None
        return int(occupied.sum())

    def certify(self,guard,support,calibration,sample,now,liftoff_height_m=None,*,body_history=None,native=None):
        self.last_observation=None
        try:
            s=sample;cal=calibration
            if not isinstance(cal,MechanicalStartup) or not isinstance(support,RetainedSupport):raise ValueError('Exact typed mechanical/retained support required')
            if cal.epoch!=s.epoch or s.epoch!=support.epoch or guard.epoch!=s.epoch or cal.calibration_id!=self.calibration_id or support.calibration_id!=self.calibration_id:raise ValueError('Contact epoch/robot/source changed')
            if not 0<=now-s.stamp<=.15 or not s.range_valid or not s.tracking_valid or not 0<=now-support.stamp<=self.max_support_age:raise ValueError('Current ToF/BODY or original support history stale')
            if not -1e-9<=now-guard.map.stamp<=guard.max_age_s:raise ValueError('Current immutable map stale')
            current=pose_envelope(body_history,int(round(s.stamp*1e9)));source=support.source_pose
            actual_proxy=max(3*math.sqrt(max(0.,np.diag(P)[:3].max())) for P in current.marginals)+current.interpolation_position_m
            if guard.uncertainty and actual_proxy>guard.uncertainty['body_three_sigma_proxy_m']+1e-9:raise ValueError('Actual current BODY covariance exceeds the retained global map envelope')
            flags=None if native is None else native.flags(now,2)
            if flags is None or flags!=(s.armed,s.landed):raise ValueError('Actual native status/landed does not match current BODY/range feedback')
            p=vector(s.position_local_m);R=np.asarray(s.local_from_body_rotation);n=unit(support.normal_local)
            pose_up_error=current.normal_error([0,0,1]);plane_error=support.normal_fit_error_rad+source.normal_error(n)
            angle=math.acos(float(np.clip((R.T@n)[2],-1,1)))+pose_up_error+plane_error
            if angle>math.radians(5):raise ValueError('Actual complete mechanical plane/BODY tilt enclosure exceeds5deg')
            v=vector(s.velocity_local_mps);w=vector(s.angular_velocity_body_radps)
            rising=liftoff_height_m is not None
            if rising and (not s.armed or not .01<liftoff_height_m<=.5):raise ValueError('Actual native arm and bounded liftoff required')
            if np.linalg.norm(v[:2])>.08 or np.linalg.norm(w)>.2 or not (-.04<=v[2]<=self.max_descent_speed if rising else -self.max_descent_speed<=v[2]<=.04):raise ValueError('Native contact measured lateral/angular/vertical speed exceeds domain')
            # Source/current marginals sum; no guessed cancellation or independence.
            lateral=max(max(source.point_error(a,[*xy,support.height(*xy)]) for xy in support.landing_boundary_xy)+current.point_error(a) for a in ([1,0,0],[0,1,0]))*math.sqrt(2)
            # Future translation reserve was charged to immutable map geometry
            # before control. Charge it also to the separate support boundary
            # and vertical contact tube; never substitute it for actual P.
            future=max(0.,guard.uncertainty.get('body_three_sigma_proxy_m',actual_proxy)-actual_proxy)
            lateral+=future*math.sqrt(2)
            # Include semantic plane-boundary sensor enclosure and existing tracking
            # envelope, then derive remaining room from the OBSERVED boundary.
            boundary_error=support.boundary_error_m # map registration margin is checked separately by the full map query
            required=float(guard.map.extent[0])+lateral+boundary_error
            margin=polygon_margin(support.landing_boundary_xy,p[:2])-required
            if margin<0:raise ValueError('Measured landing boundary cannot contain complete body and actual uncertainty')
            if polygon_margin(support.support_boundary_xy,p[:2])<required:raise ValueError('Foot/body leaves observed planar support extent')
            z_error=source.point_error([0,0,1],support.center_local_m)+current.point_error([0,0,1])+support.height_error(p[:2])
            z_error+=future
            hit=p+R@(self.mount+self.axis*s.range_m);residual=float(n@(hit-np.asarray(support.center_local_m)))
            # The original declared ToF20+5mm inputs are supplied on context by
            # the adapter. They are retained as explicit error-model assumptions.
            tof_error=float(self.tof_error_m+self.tof_extrinsic_error_m)
            if abs(residual)>z_error+tof_error:raise ValueError('Current ToF disagrees with retained supporting plane and actual pose enclosure')
            clearance=-float((R@(self.mount+self.axis*s.range_m))[2])-cal.offset_m*R[2,2]
            clearance_error=tof_error+cal.error_m
            plane_clearance=float(p[2]-support.height(p[0],p[1])-cal.offset_m)
            if abs(plane_clearance-clearance)>z_error+clearance_error:raise ValueError('Retained plane and ToF/body contact histories disagree')
            if clearance<-(clearance_error+z_error) or clearance>max(.8,liftoff_height_m or 0)+clearance_error:raise ValueError('Actual foot clearance outside retained contact phase')
            goal=np.array([p[0],p[1],support.height(p[0],p[1])+cal.offset_m+(liftoff_height_m or 0.)])
            if polygon_margin(support.landing_boundary_xy,goal[:2])<required:raise ValueError('Desired contact endpoint does not fit measured boundary')
            if self._volume(p,goal,support,cal,guard.map,guard.extra) is None:raise ValueError('Unknown/occupied non-support upper or side body sweep')
            flags=None if native is None else native.flags(now,2);land=None if native is None else native.get('land',now,1.25)
            grounded=bool(flags is not None and flags[1] and land is not None and getattr(land,'at_rest',False) and
                np.linalg.norm(v)<=.06 and abs(clearance)<=clearance_error+z_error)
            observation=dict(stamp=s.stamp,epoch=s.epoch,target_id=support.target_id,
                foot_clearance_m=clearance,foot_clearance_error_m=clearance_error,
                retained_plane_clearance_m=plane_clearance,vertical_registration_error_m=z_error,
                tof_plane_residual_m=residual,remaining_boundary_margin_m=margin,
                actual_pose_lateral_error_sum_m=lateral,source_stamp=support.stamp,
                source_region_sha256=support.source_hash,current_covariance=current.as_dict(),
                actual_native_grounded=grounded,native_at_rest=False if land is None else bool(getattr(land,'at_rest',False)),
                native_land_source_us=None if native is None or 'land' not in native.records else int(native.records['land'][0]),
                native_status_source_us=None if native is None or 'status' not in native.records else int(native.records['status'][0]),
                current_camera_visibility_required=False,plane_model_assumption='static rigid continuous supporting surface',
                sensor_accuracy_qualified=False,global_covariance_subtracted=False)
            self.last_observation=observation;self.last_rejection=None
            forecast=guard.uncertainty.get('causal_forward_reserve')
            deadline=now+.25 if forecast is None else min(now+.25,forecast['valid_until'])
            if deadline<now:raise ValueError('Prior causal position forecast expired before native contact certificate')
            return NativeContactCertificate(s.epoch,now,min(deadline,guard.map.stamp+guard.max_age_s),guard.map.stamp,
                'liftoff' if rising else 'descent',(tuple(p),tuple(goal)),support,cal,lateral+support.boundary_error_m,
                z_error+clearance_error,grounded,observation,dict(guard.uncertainty),self,guard.map,guard.extra)
        except (ValueError,TypeError,AttributeError,OverflowError) as e:
            self.last_rejection=str(e);return None

    def configure_tof(self,range_error_m,extrinsic_error_m):
        if not 0<range_error_m<=.02 or not 0<=extrinsic_error_m<=.02:raise ValueError('Original ToF input domain required')
        self.tof_error_m=range_error_m;self.tof_extrinsic_error_m=extrinsic_error_m;return self


def observed_vertical_frontier(m,position,goal=None,extra=0.,max_step=.3):
    """Uncapped observed-free3D frontier. Never infer a ceiling or field height."""
    p=vector(position);component=m.component(p)
    if not component:return None
    indices=np.argwhere((m.components==component)&m.safe);points=m.xyz(indices)
    vertical,_=m.vertical_viewpoint_candidates(p)
    points=np.concatenate([points,vertical])
    delta=points-p;distance=np.linalg.norm(delta,axis=1)
    keep=(distance>m.res*.8)&(distance<1.4)&(abs(delta[:,2])>=m.res*.8)
    if goal is not None:keep&=abs(points[:,2]-goal[2])<abs(p[2]-goal[2])
    else:keep&=delta[:,2]>0
    points=points[keep];distance=distance[keep]
    if not len(points):return None
    score=(np.linalg.norm(p-vector(goal))-np.linalg.norm(points-vector(goal),axis=1) if goal is not None else points[:,2]-p[2])-.05*distance
    for i in np.argsort(score)[::-1][:64]:
        q=p+(points[i]-p)*min(1.,max_step/distance[i])
        if m.segment(p,q,extra_margin=extra):
            return q,dict(source='observed3D full-body free sweep only; no fixed wall/field height',
                observed_free_altitude_maneuver=True,vertical_delta_m=float(q[2]-p[2]),measured_frontier_local_m=points[i].tolist())
    return None
