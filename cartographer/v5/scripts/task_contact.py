"""Local measured-contact descent certificates for a mature native LAND mode.

The global map is NEVER changed. Its already-padded horizontal/upper envelope
is retained. Only the lower artificial inflation and occupied voxels proven to
represent the measured support plane are handled as CONTACT, not as free space.
Unknown voxels in the queried above-support body volume still veto descent.
Dense direct-stereo support must cover every native pixel of the enlarged patch;
missing pixels, off-plane hits, edges and holes refuse the proof.
"""
from dataclasses import dataclass,field
import math
import cv2
import numpy as np
from task_observations import vector,unit
from task_landing import rotation,surface_relative_observation


@dataclass(frozen=True)
class SupportPatch:
    target_id: str
    stamp: float
    epoch: str
    calibration_id: str
    center_local_m: tuple
    normal_local: tuple
    half_width_m: float
    error_m: float
    direct_pixels: int
    max_plane_residual_m: float
    max_pixel_spacing_m: float
    marker_kind: str
    measurement_stride_px: float = 1.
    source: str = 'dense native-grid direct stereo measured support plane'

    def height(self,x,y):
        c=self.center_local_m;n=self.normal_local
        return c[2]-(n[0]*(x-c[0])+n[1]*(y-c[1]))/n[2]

    def contains(self,xlo,xhi,ylo,yhi):
        c=self.center_local_m;h=self.half_width_m
        return xlo>=c[0]-h and xhi<=c[0]+h and ylo>=c[1]-h and yhi<=c[1]+h

    def as_dict(self):return dict(vars(self))


def build_support_patch(measured_depth,K,exposure,track,now,required_half_width_m,
                        calibrated_depth_error_m,measured_mask=None,max_pixel_spacing_m=.01,measurement_stride_px=1.):
    """Return None on any support uncertainty; no hull/interior hole filling.

    Call while hovering before LAND, with a stable measured horizontal-plane
    track (dock or own observed home). Required coverage includes whole map cells
    under the padded body, not just marker corners or a central ToF ray.
    """
    T=exposure.validate(now);depth=np.asarray(measured_depth);K=vector(K,4)
    if not math.isfinite(measurement_stride_px) or not 1<=measurement_stride_px<=8:raise ValueError('Invalid independent stereo measurement spacing')
    if depth.ndim!=2 or min(depth.shape)<32 or max(depth.shape)>1920 or min(K[:2])<=0:
        raise ValueError('Invalid direct depth grid or calibration')
    if not 0<calibrated_depth_error_m<=.02 or not 0<required_half_width_m<=1.5:
        raise ValueError('Measured stereo error bound and bounded robot support patch required')
    if track.epoch!=exposure.epoch or not track.consistent or not track.geometry_verified or not 0<=now-track.stamp<=.6:
        return None
    if track.role not in ('dock','home') or track.kind!='landing_plane' or track.details.get('marker_kind') not in ('h','cross','closed_mark','own_launch_plane'):
        return None
    if track.observation_count<3 or len({h.get('stamp') for h in track.history if h.get('calibration_id')})<3:return None
    n=unit(track.normal);c=vector(track.center)
    if n[2]<0:n=-n
    if n[2]<math.cos(math.radians(5)) or exposure.pose_error_m>.04:return None
    # Caller supplies the already-padded body footprint and registration
    # budget. Add only two ray pixels for raster boundaries, not another
    # hidden copy of the navigator's pose/tracking padding.
    half=required_half_width_m+2*max_pixel_spacing_m
    xy=np.array([[c[0]-half,c[1]-half],[c[0]+half,c[1]-half],
                 [c[0]+half,c[1]+half],[c[0]-half,c[1]+half]])
    z=c[2]-(xy-c[:2])@n[:2]/n[2];corners=np.column_stack((xy,z))
    camera=(corners-T[:3,3])@T[:3,:3]
    if (camera[:,2]<=.25).any():return None
    uv=np.column_stack((camera[:,0]/camera[:,2]*K[0]+K[2],camera[:,1]/camera[:,2]*K[1]+K[3]))
    h,w=depth.shape
    if (uv[:,0]<2).any() or (uv[:,0]>w-3).any() or (uv[:,1]<2).any() or (uv[:,1]>h-3).any():return None
    mask=np.zeros(depth.shape,np.uint8);cv2.fillConvexPoly(mask,np.rint(uv).astype(np.int32),1)
    yy,xx=np.nonzero(mask)
    if not 64<=len(xx)<=250000:return None
    z=depth[yy,xx]
    direct=np.isfinite(z)&(z>=.25)&(z<=8.)
    if measured_mask is not None:
        if np.asarray(measured_mask).shape!=depth.shape:raise ValueError('Provenance mask mismatch')
        direct&=np.asarray(measured_mask,dtype=bool)[yy,xx]
    if not direct.all():return None
    rays=np.column_stack(((xx-K[2])/K[0],(yy-K[3])/K[1],np.ones(len(xx))))
    points=rays*z[:,None];local=points@T[:3,:3].T+T[:3,3]
    residual=np.abs((local-c)@n)
    if residual.max()>calibrated_depth_error_m:return None
    # Perspective/tilt projected footprint of one native ray; conservative
    # factor covers foreshortening. No upsampled/inferred pixels count.
    spacing=float(measurement_stride_px*np.max(z)/min(K[:2])/max(.1,abs(n@(T[:3,:3]@[0,0,1]))))
    if spacing>max_pixel_spacing_m:return None
    return SupportPatch(track.id,exposure.stamp,exposure.epoch,exposure.calibration_id,
                        tuple(c),tuple(n),required_half_width_m,
                        exposure.pose_error_m+calibrated_depth_error_m,len(xx),float(residual.max()),spacing,
                        track.details['marker_kind'],measurement_stride_px)


@dataclass(frozen=True)
class ContactDescentCertificate:
    epoch: str
    issued: float
    valid_until: float
    map_stamp: float
    points: tuple
    support: SupportPatch
    exempted_support_voxels: int
    calibration: object
    mode: str
    extra_collision_margin_m: float
    _adapter: object=field(repr=False,compare=False)
    _map: object=field(repr=False,compare=False)

    def as_dict(self):
        return {'kind':'measured_contact_'+self.mode,'epoch':self.epoch,'issued':self.issued,
                'valid_until':self.valid_until,'map_stamp':self.map_stamp,'points':self.points,
                'support':self.support.as_dict(),'foot_calibration':self.calibration.as_dict(),
                'contact_support_voxels':self.exempted_support_voxels,
                'extra_collision_margin_m':self.extra_collision_margin_m,
                'global_map_changed':False,'unknown_above_support_allowed':False,
                'free_upper_volume_and_measured_contact_verified':True,
                'full_body_free':False,'requires_native_land_and_continuous_monitoring':self.mode=='descent',
                'requires_monitored_liftoff':self.mode=='liftoff'}

    def covers_motion(self,a,b,now,epoch):
        if epoch!=self.epoch or not self.issued<=now<=self.valid_until:return False
        a,b=vector(a),vector(b);start=np.asarray(self.points[0]);goal=np.asarray(self.points[-1])
        for p in (a,b):
            if np.linalg.norm(p[:2]-start[:2])>self._adapter.max_lateral_error:return False
            if p[2]>max(start[2],goal[2])+.015 or p[2]<min(start[2],goal[2])-self.calibration.error_m-.015:return False
        if self.mode=='descent' and b[2]>a[2]+.015:return False
        if self.mode=='liftoff' and b[2]<a[2]-.015:return False
        return self._adapter._volume(a,b,self.support,self.calibration,self._map,self.extra_collision_margin_m) is not None


class ContactLandingAdapter:
    """One static measured landing patch; refresh at every accepted pose/range.

    Native PX4 LAND owns the descent. The caller must abort/brake native LAND on
    a rejection, stop offboard position commands while LAND owns the vehicle,
    retain VIO coordinates across disarm, and never apply ToF global-z fixes.
    """
    def __init__(self,tof_in_body_m,tof_axis_body,robot_calibration_id,
                 max_lateral_error_m=.04,max_descent_speed_mps=.25,max_support_age_s=10.):
        self.mount=vector(tof_in_body_m);self.axis=unit(tof_axis_body)
        if not robot_calibration_id or not 0<max_lateral_error_m<=.08 or not 0<max_descent_speed_mps<=.4 or not 0<max_support_age_s<=10:
            raise ValueError('Bounded native-LAND monitoring parameters required')
        self.robot_calibration_id=robot_calibration_id;self.max_lateral_error=max_lateral_error_m
        self.max_descent_speed=max_descent_speed_mps;self.max_support_age=max_support_age_s
        self.last_rejection=None;self.last_observation=None

    def _reject(self,reason):self.last_rejection=reason;return None

    def _volume(self,a,b,support,cal,m,extra=0.):
        """Query coarse cells without modifying any array or filling unknown.

        Lower inflation is clipped to the measured support/contact band. Every
        queried unknown/out-of-map cell vetoes. Occupied cells are exempt only
        when that exact cell intersects the densely verified plane; all its XY
        extent must be inside the measured patch. Upper/side inflation remains.
        """
        radius=float(m.extent[0])+extra;top=float(m.extent[2])+extra;res=float(m.res)
        if res<=0 or res>.2:return None
        exempt=set()
        distance=np.linalg.norm(b-a);samples=max(2,int(math.ceil(distance/(res*.25)))+1)
        if samples>80:return None
        for p in np.linspace(a,b,samples):
            lo_xy=p[:2]-radius;hi_xy=p[:2]+radius
            if not support.contains(lo_xy[0]-res,hi_xy[0]+res,lo_xy[1]-res,hi_xy[1]+res):return None
            lower=vector([lo_xy[0],lo_xy[1],p[2]-cal.offset_m-cal.error_m])
            upper=vector([hi_xy[0],hi_xy[1],p[2]+top])
            # The contact surface is not a free voxel: remove only the
            # artificial below-floor padding from this local query.
            lower[2]=max(lower[2],support.height(p[0],p[1])+support.error_m)
            if lower[2]>upper[2]:return None
            lo=m.idx(lower);hi=m.idx(upper)
            if not m.inside(lo) or not m.inside(hi):return None
            if np.prod(hi-lo+1)>32768:return None
            for i in range(lo[0],hi[0]+1):
                for j in range(lo[1],hi[1]+1):
                    x0,y0=(m.origin+np.array([i,j,0])*res)[:2];x1,y1=x0+res,y0+res
                    if not support.contains(x0,x1,y0,y1):return None
                    heights=[support.height(x,y) for x in (x0,x1) for y in (y0,y1)]
                    plane_low=min(heights)-support.error_m;plane_high=max(heights)+support.error_m
                    for k in range(lo[2],hi[2]+1):
                        state=int(m.state[i,j,k])
                        if state in (0,-2):continue
                        if state!=1:return None
                        z0=m.origin[2]+k*res;z1=z0+res
                        if not (z0<=plane_high and z1>=plane_low):return None
                        exempt.add((i,j,k))
        return len(exempt)

    def certify(self,guard,support,calibration,sample,now,liftoff_height_m=None):
        s=sample;cal=calibration;self.last_observation=None
        if support is None or cal is None:return self._reject('measured support and foot calibration required')
        if support.source!='dense native-grid direct stereo measured support plane' or support.direct_pixels<64 or not 0<support.max_pixel_spacing_m<=.01 or not 0<=support.max_plane_residual_m<=.02 or not support.calibration_id:
            return self._reject('support is not a high-resolution direct measured plane proof')
        if unit(support.normal_local)[2]<math.cos(math.radians(5)):
            return self._reject('contact support plane is too tilted')
        if cal.robot_calibration_id!=self.robot_calibration_id or not 0<cal.error_m<=.025 or not .01<=cal.offset_m<=.6:
            return self._reject('robot/ToF foot calibration mismatch or uncertainty')
        if s.epoch!=support.epoch or s.epoch!=cal.epoch or s.epoch!=guard.epoch:
            return self._reject('VIO gauge must remain unchanged across landing/disarm')
        if not 0<=now-support.stamp<=self.max_support_age or not 0<support.error_m<=.06:
            return self._reject('support evidence stale or low confidence')
        if not -1e-9<=now-guard.map.stamp<=guard.max_age_s:return self._reject('current map is stale')
        try:
            p=vector(s.position_local_m);R=rotation(s.local_from_body_rotation);v=vector(s.velocity_local_mps)
            rising=liftoff_height_m is not None
            if rising and (not 0<liftoff_height_m<=.5 or not s.armed):return self._reject('confirmed arming and bounded liftoff height required')
            velocity_ok=(-.04<=v[2]<=self.max_descent_speed) if rising else (-self.max_descent_speed<=v[2]<=.04)
            if R[2,2]<math.cos(math.radians(3)) or np.linalg.norm(vector(s.angular_velocity_body_radps))>.2 or np.linalg.norm(v[:2])>.08 or not velocity_ok:
                return self._reject('native LAND attitude or lateral/vertical velocity limit')
            if np.linalg.norm(p[:2]-np.asarray(support.center_local_m)[:2])>self.max_lateral_error:
                return self._reject('native LAND lateral drift beyond measured support center')
            obs=surface_relative_observation(s,now,self.mount,self.axis,cal,support.center_local_m,support.normal_local,support.target_id)
            if abs(obs['plane_residual_m'])>support.error_m+cal.error_m:
                return self._reject('ToF hit does not agree with this observed support plane')
            if not -.015-cal.error_m<=obs['foot_clearance_m']<=.5:
                return self._reject('terminal descent clearance outside calibrated contact interval')
            if not s.armed and not s.landed:return self._reject('airborne without arming confirmation')
            # Ground contact and initial liftoff share the same local measured
            # support constraint. Upper/side volume always uses the normal map.
            goal=np.array([p[0],p[1],support.height(p[0],p[1])+cal.offset_m+(liftoff_height_m if rising else 0.)])
            if p[2]<support.height(p[0],p[1])+cal.offset_m-cal.error_m-.015:return self._reject('body descended below calibrated support contact')
            count=self._volume(p,goal,support,cal,guard.map,guard.extra)
            if count is None:return self._reject('unknown/occupied non-support body volume or unsupported edge')
            self.last_rejection=None;self.last_observation=obs
            return ContactDescentCertificate(s.epoch,now,min(now+.25,guard.map.stamp+guard.max_age_s),guard.map.stamp,
                                             (tuple(p),tuple(goal)),support,count,cal,'liftoff' if rising else 'descent',guard.extra,self,guard.map)
        except (ValueError,TypeError,OverflowError):return self._reject('invalid measured terminal descent packet')


@dataclass(frozen=True)
class LandingContext:
    adapter: ContactLandingAdapter
    support: SupportPatch
    calibration: object
    range_pose: object
